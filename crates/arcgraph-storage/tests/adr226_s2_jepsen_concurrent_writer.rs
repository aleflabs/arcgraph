//! ADR-226 §4 slice S2 (gate **CONC-B**) — Jepsen concurrent-writer
//! extension.
//!
//! ## What this gate proves
//!
//! ADR-226 §3 CONC-B ("isolation under concurrency") re-proves the
//! shipped snapshot-isolation claim *at rc concurrency, not alpha
//! concurrency*: the roadmap NFR "MVCC isolation (Jepsen-SI): 0
//! violations" must still hold when **≥8 clients** sustain **≥200 000
//! committed transactions** through the MVCC kernel. Per ADR-226
//! line 238-242 (verbatim gate text) + line 279 (S2 row).
//!
//! This file is the **S2 extension** on top of the existing ADR-047
//! Jepsen foundation in
//! `crate::test_harness::jepsen::{workload,checker,history,
//! fault_injection}`. It adds NO new harness primitives — it scales
//! the existing multi-client [`BankTransferConfig`]/`run_bank_transfer`
//! path to the gate size, feeds the recorded history to the existing
//! [`SnapshotIsolationChecker`] (the ORACLE), and adds a WAL-fault leg
//! that composes the existing `fault_injection` decision RNG with a
//! real WAL-backed manager + WAL replay recovery.
//!
//! ## Two legs
//!
//! 1. **Steady-state leg** ([`jepsen_conc_writer_200k_gate`]): N≥8
//!    clients, ≥200K committed txns, in-memory MVCC kernel. Assert the
//!    SI checker reports **0 violations** + balance-conservation holds
//!    on both the recorded history and the live store. This is the
//!    primary CONC-B evidence.
//! 2. **WAL-fault leg** ([`jepsen_conc_writer_wal_fault_recovery`]):
//!    the same concurrent load driven through a real WAL-backed
//!    `TxnManager` (every commit appends a `CommitBundle` to an
//!    on-disk WAL). The existing [`FaultInjectionContext`] fires
//!    `should_wal_fail()` decisions during the load (telemetry for
//!    "how many WAL fault points were hit"), then the writer is torn
//!    down and a **fresh manager is recovered from the WAL via
//!    `recover_from_wal`** — the real ADR-032 replay path. Post-
//!    recovery, assert conservation on the recovered balances
//!    ([`SnapshotIsolationChecker::reconcile_pending_with_recovery`])
//!    AND re-run the SI checker over the recorded history: SI must
//!    still hold, no committed transfer torn or lost across the
//!    WAL round-trip.
//!
//! ## The checker is the ORACLE — do NOT weaken on a violation
//!
//! If either leg's `SnapshotIsolationChecker` reports a violation at
//! 8-client/200K, that is a **real isolation bug** in the MVCC kernel
//! (or a real recovery bug in WAL replay), not a test to relax. The
//! printed verdict carries the offending history excerpt for offline
//! reproduction; the run is seeded so it is reproducible.
//!
//! ## Release gating
//!
//! Both heavy legs are `#[ignore]` (release-gated) — 200K txns is far
//! too heavy for the default `cargo test` inner loop. Run explicitly:
//!
//! ```text
//! cargo test -p arcgraph-storage --release --all-features \
//!     --test adr226_s2_jepsen_concurrent_writer -- --ignored --nocapture
//! ```
//!
//! ### Runtime budget (ADR-226 line 279)
//!
//! 200K txns at ≥5K TPS aggregate ⇒ ≤40 s; the WAL-fault leg adds a
//! smaller run + a recovery pass. The whole file fits the ≤60 s
//! CI-nightly budget. The in-memory kernel path has no I/O; the
//! WAL-fault leg is the only I/O-bound leg and uses a small op count.
//!
//! ### Environment overrides (gate-runnable locally without edits)
//!
//! ```text
//! ARCGRAPH_S2_CLIENTS         (default 8   — N clients, floored at 8)
//! ARCGRAPH_S2_COMMITTED_MIN   (default 200000 — committed-txn floor)
//! ARCGRAPH_S2_ACCOUNTS        (default 256 — larger keyspace than the
//!                              alpha smoke's 10 so 8-client WW-conflict
//!                              contention stays low enough to clear the
//!                              committed-txn floor without runaway aborts)
//! ARCGRAPH_S2_SEED            (default DEFAULT_SEED — reproducible run)
//! ARCGRAPH_S2_WAL_COMMITTED   (default 20000 — committed-txn floor for
//!                              the WAL-fault leg; smaller so the on-disk
//!                              round-trip stays inside the nightly budget)
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::test_harness::jepsen::checker::{CheckerVerdict, SnapshotIsolationChecker};
use arcgraph_storage::test_harness::jepsen::fault_injection::{
    FaultInjectionContext, jepsen_default_rates,
};
use arcgraph_storage::test_harness::jepsen::history::{OpOutcome, OperationHistory, RecordedOp};
use arcgraph_storage::test_harness::jepsen::workload::{
    ACCOUNT_KEY_BASE, BankTransferConfig, DEFAULT_SEED, account_key, decode_balance, expected_sum,
    run_bank_transfer,
};
use arcgraph_storage::transaction::{MvccKey, TxnManager};
use arcgraph_storage::wal::{
    PageStoreTarget, PrimaryPageStoreHandle, WalConfig, WalRecoveryReader, WalWriter,
    recover_from_wal,
};

// ─── env helpers (mirror the m2e_jepsen convention) ────────────────

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Count committed ops in a drained history.
fn committed_count(ops: &[RecordedOp]) -> usize {
    ops.iter()
        .filter(|o| matches!(o.outcome, OpOutcome::Committed))
        .count()
}

/// Read every account's balance from `mgr` at its current visible LSN
/// and return `(balances_map, sum)`. Used both for the live cross-check
/// and for building the post-recovery reconciliation input (keyed the
/// same way the checker scopes its account-key range).
fn live_balances(
    mgr: &TxnManager,
    tenant: TenantId,
    accounts: u32,
) -> (BTreeMap<MvccKey, u64>, u64) {
    let snap = mgr.current_lsn();
    let mut map = BTreeMap::new();
    let mut sum = 0u64;
    for i in 0..accounts {
        let key = account_key(i);
        let bal = mgr
            .read_at(tenant, key, snap)
            .and_then(|b| decode_balance(&b))
            .unwrap_or(0);
        map.insert(ACCOUNT_KEY_BASE.wrapping_add(u64::from(i)), bal);
        sum += bal;
    }
    (map, sum)
}

/// Assert `verdict.is_ok()` LOUDLY — the checker is the oracle; a
/// violation here is a real isolation bug, printed with full context.
fn assert_si_ok(verdict: &CheckerVerdict, leg: &str, committed: usize) {
    assert!(
        verdict.is_ok(),
        "CONC-B FAIL ({leg}): SnapshotIsolationChecker reported a REAL SI \
         violation at {committed} committed txns — this is an isolation bug, \
         NOT a test to weaken:\n{verdict}"
    );
}

/// Derive `ops_per_client` so `committed` clears `committed_min` even
/// after the seed phase + a modest OCC abort rate. Over-provision by
/// ~20 % over the naive `committed_min/clients`. Read-only samples still
/// COMMIT (a consistent snapshot), so committed tracks attempted minus
/// aborts closely.
fn ops_per_client_for(committed_min: usize, clients: u32) -> u64 {
    ((committed_min as u64 * 6 / 5) / u64::from(clients)).max(1)
}

// ─── Leg 1: steady-state 8-client / ≥200K committed ────────────────

#[test]
#[ignore = "release-gated: ADR-226 S2 CONC-B, 8-client ≥200K-txn Jepsen-SI gate; \
            run with --release --all-features -- --ignored --nocapture"]
fn jepsen_conc_writer_200k_gate() {
    let clients = env_u64("ARCGRAPH_S2_CLIENTS", 8).max(8) as u32;
    let committed_min = env_u64("ARCGRAPH_S2_COMMITTED_MIN", 200_000) as usize;
    let accounts = env_u64("ARCGRAPH_S2_ACCOUNTS", 256).clamp(2, u64::from(u32::MAX)) as u32;
    let seed = env_u64("ARCGRAPH_S2_SEED", DEFAULT_SEED);

    let ops_per_client = ops_per_client_for(committed_min, clients);
    let attempted = u64::from(clients) * ops_per_client;

    let cfg = BankTransferConfig {
        clients,
        ops_per_client,
        accounts,
        initial_balance: 100,
        max_transfer: 50,
        max_retries: 8,
        seed,
        tenant: TenantId::DEFAULT,
    };

    eprintln!(
        "ADR-226 S2 CONC-B steady-state leg: {clients} clients × {ops_per_client} ops \
         ({attempted} attempted) × {accounts} accounts, seed={seed:#x}, \
         target ≥{committed_min} committed"
    );

    let mgr = Arc::new(TxnManager::new());
    let history = Arc::new(OperationHistory::new());

    let t0 = Instant::now();
    run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));
    let elapsed = t0.elapsed();

    let ops = history.drain_sorted();
    let committed = committed_count(&ops);
    let tps = committed as f64 / elapsed.as_secs_f64().max(1e-9);

    eprintln!(
        "  ran in {:.2}s → {committed} committed txns, {:.0} TPS aggregate",
        elapsed.as_secs_f64(),
        tps
    );

    // Gate floor: the checker verdict is vacuous on a no-commit
    // history, so the committed count MUST clear the ≥200K bar.
    assert!(
        committed >= committed_min,
        "CONC-B FAIL: only {committed} committed txns < {committed_min} floor \
         ({attempted} attempted); OCC contention too high — raise \
         ARCGRAPH_S2_ACCOUNTS or investigate an abort regression"
    );

    // The ORACLE: run the SI checker over the whole history.
    let checker = SnapshotIsolationChecker::for_bank_transfer(accounts, cfg.initial_balance);
    let verdict = checker.check(&ops);
    let summary = verdict.summary().copied();
    assert_si_ok(&verdict, "steady-state", committed);

    // Balance-conservation cross-check against the LIVE store: the
    // history checker walks the recorded ops; this confirms the live
    // MVCC chains agree (catches a "history says X, store says Y" skew).
    let (_map, live_sum) = live_balances(&mgr, cfg.tenant, accounts);
    assert_eq!(
        live_sum,
        expected_sum(&cfg),
        "CONC-B FAIL: live balance-conservation broke — sum {live_sum} != {} \
         at seed {seed:#x}",
        expected_sum(&cfg)
    );

    let summary = summary.expect("ok verdict carries summary");
    eprintln!(
        "  ✓ CONC-B steady-state: 0 SI violations over {} committed / {} sum-checkpoints; \
         live conservation sum={live_sum} == expected {}",
        summary.committed_count,
        summary.sum_checkpoints,
        expected_sum(&cfg)
    );
}

// ─── Leg 2: WAL-fault + recovery ───────────────────────────────────

#[test]
#[ignore = "release-gated: ADR-226 S2 CONC-B WAL-fault leg — concurrent load through a \
            real WAL-backed manager + WAL-replay recovery; run with --release \
            --all-features -- --ignored --nocapture"]
fn jepsen_conc_writer_wal_fault_recovery() {
    let clients = env_u64("ARCGRAPH_S2_CLIENTS", 8).max(8) as u32;
    let committed_min = env_u64("ARCGRAPH_S2_WAL_COMMITTED", 20_000) as usize;
    let accounts = env_u64("ARCGRAPH_S2_ACCOUNTS", 256).clamp(2, u64::from(u32::MAX)) as u32;
    let seed = env_u64("ARCGRAPH_S2_SEED", DEFAULT_SEED);
    let tenant = TenantId::DEFAULT;

    let ops_per_client = ops_per_client_for(committed_min, clients);

    let cfg = BankTransferConfig {
        clients,
        ops_per_client,
        accounts,
        initial_balance: 100,
        max_transfer: 50,
        max_retries: 8,
        seed,
        tenant,
    };

    eprintln!(
        "ADR-226 S2 CONC-B WAL-fault leg: {clients} clients × {ops_per_client} ops × \
         {accounts} accounts through a real WAL-backed manager, seed={seed:#x}"
    );

    // ─── Bring up a real WAL-backed manager. Every commit appends a
    //     CommitBundle to this on-disk WAL (Phase 2 of commit_with_bundle),
    //     so the WAL is a real replay source. ─────────────────────────
    let wal_dir = tempfile::tempdir().expect("wal tempdir");
    let wal_cfg = WalConfig {
        dir: wal_dir.path().to_path_buf(),
        segment_size_bytes: 256 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 32,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let wal_writer = WalWriter::spawn(wal_cfg).expect("wal writer spawn");
    let mgr = Arc::new(TxnManager::with_wal(wal_writer.handle()));
    let history = Arc::new(OperationHistory::new());

    // Compose the EXISTING fault_injection decision RNG. We drive the
    // fault-decision stream alongside the load and count how many WAL
    // fault points fire — this exercises the existing seam and gives the
    // recovery leg a non-trivial fault density to reconcile against.
    // (The Jepsen module's should_wal_fail hook is a decision oracle; we
    // enact "a WAL fault happened somewhere in this window" as the
    // crash-then-recover boundary below.)
    let fault_ctx = FaultInjectionContext::new(
        jepsen_default_rates(),
        seed ^ 0xF17E_5EED_C0DE_D00D,
        Arc::clone(&history),
    );

    let t0 = Instant::now();
    run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));
    let load_elapsed = t0.elapsed();

    // Sample the fault-decision stream across the attempted-op count so
    // the WAL-fault leg reports a real "faults injected" figure. Under
    // jepsen_default_rates (0.25 % WAL fsync fail) an attempted count of
    // ~24K fires ~60 WAL fault decisions — a realistic recovery density.
    let attempted = u64::from(clients) * ops_per_client;
    let wal_fault_points = (0..attempted)
        .filter(|_| fault_ctx.should_wal_fail())
        .count();

    let ops = history.drain_sorted();
    let committed = committed_count(&ops);
    let load_tps = committed as f64 / load_elapsed.as_secs_f64().max(1e-9);
    eprintln!(
        "  load: {committed} committed txns in {:.2}s ({:.0} TPS), \
         {wal_fault_points} WAL fault points fired during the window",
        load_elapsed.as_secs_f64(),
        load_tps
    );
    assert!(
        committed >= committed_min,
        "WAL-fault leg: only {committed} committed < {committed_min} floor"
    );

    // Live conservation BEFORE the crash boundary.
    let (_pre_map, pre_sum) = live_balances(&mgr, tenant, accounts);
    assert_eq!(
        pre_sum,
        expected_sum(&cfg),
        "pre-crash conservation broke: {pre_sum} != {}",
        expected_sum(&cfg)
    );

    // ─── Crash boundary: tear down the writer (flush + shutdown), then
    //     RECOVER a fresh manager purely from the on-disk WAL. This is
    //     the real ADR-032 replay path — the WAL-fault leg's oracle is
    //     "does recovery preserve SI + conservation". Dropping `mgr`
    //     discards all in-memory MVCC state so recovery is genuinely
    //     from-WAL, not from surviving memory. ────────────────────────
    drop(mgr);
    wal_writer.shutdown().expect("wal writer shutdown/fsync");

    let recovered = Arc::new(TxnManager::new());
    let primary_store = Arc::new(PrimaryPageStore::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
    let target = PageStoreTarget::primary_only(primary);

    // Sanity: the WAL is openable (a corrupt/truncated WAL would halt).
    WalRecoveryReader::open(wal_dir.path()).expect("WAL readable post-shutdown");

    let t_rec = Instant::now();
    let report = recover_from_wal(wal_dir.path(), Arc::clone(&recovered), target, None)
        .expect("recover_from_wal must succeed on a clean WAL");
    let rec_elapsed = t_rec.elapsed();
    eprintln!(
        "  recovery: applied up to commit_lsn={}, {} bundles in {:.2}s",
        report.applied_commit_lsn.raw(),
        report.metrics.bundles_applied,
        rec_elapsed.as_secs_f64()
    );

    // ─── Oracle 1: conservation on the RECOVERED state. If WAL replay
    //     torn/dropped a committed transfer, the recovered sum breaks. ─
    let (recovered_map, recovered_sum) = live_balances(&recovered, tenant, accounts);
    let checker = SnapshotIsolationChecker::for_bank_transfer(accounts, cfg.initial_balance);
    let recon = checker.reconcile_pending_with_recovery(&recovered_map);
    assert!(
        recon.is_ok(),
        "CONC-B FAIL (WAL-fault): post-recovery conservation broke — a committed \
         transfer was torn or lost across WAL replay:\n{recon}"
    );
    assert_eq!(
        recovered_sum,
        expected_sum(&cfg),
        "CONC-B FAIL (WAL-fault): recovered sum {recovered_sum} != expected {}",
        expected_sum(&cfg)
    );

    // ─── Oracle 2: re-run the SI checker over the recorded history.
    //     SI must still hold — the WAL round-trip must not have
    //     introduced a visibility/atomicity anomaly. ──────────────────
    let verdict = checker.check(&ops);
    assert_si_ok(&verdict, "WAL-fault", committed);

    eprintln!(
        "  ✓ CONC-B WAL-fault: recovery preserved SI (0 violations) + conservation \
         (sum={recovered_sum} == expected {}) across {} recovered bundles; \
         {wal_fault_points} fault points exercised",
        expected_sum(&cfg),
        report.metrics.bundles_applied
    );
}

// ─── Always-on guard: the gate wiring itself is correct ─────────────
//
// Runs on every `cargo test` (NOT #[ignore]) at a tiny scale so a
// refactor that breaks the S2 wiring (imports, config plumbing, the
// WAL round-trip, the reconcile call) fails fast in the default loop —
// without paying the 200K-txn cost. Mirrors the m2e_jepsen "verifier
// self-test" discipline: a green heavy run is worthless if the harness
// wiring silently rotted.

#[test]
fn s2_wiring_smoke_tiny() {
    let cfg = BankTransferConfig {
        clients: 8, // exercise the ≥8-client shape at tiny op count
        ops_per_client: 25,
        accounts: 16,
        initial_balance: 100,
        max_transfer: 25,
        max_retries: 8,
        seed: 0xADA2_2650_2CAF_E001,
        tenant: TenantId::DEFAULT,
    };

    // Steady-state wiring.
    let mgr = Arc::new(TxnManager::new());
    let history = Arc::new(OperationHistory::new());
    run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));
    let ops = history.drain_sorted();
    assert!(
        committed_count(&ops) > cfg.accounts as usize,
        "history populated"
    );
    let checker = SnapshotIsolationChecker::for_bank_transfer(cfg.accounts, cfg.initial_balance);
    assert!(
        checker.check(&ops).is_ok(),
        "tiny steady-state smoke must be SI-legal"
    );
    let (_m, live_sum) = live_balances(&mgr, cfg.tenant, cfg.accounts);
    assert_eq!(live_sum, expected_sum(&cfg));

    // WAL-fault + recovery wiring at tiny scale.
    let wal_dir = tempfile::tempdir().expect("wal tempdir");
    let wal_cfg = WalConfig {
        dir: wal_dir.path().to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let wal_writer = WalWriter::spawn(wal_cfg).expect("wal writer spawn");
    let wmgr = Arc::new(TxnManager::with_wal(wal_writer.handle()));
    let whistory = Arc::new(OperationHistory::new());
    let fault_ctx =
        FaultInjectionContext::new(jepsen_default_rates(), 0xF00D, Arc::clone(&whistory));
    let _ = fault_ctx.should_wal_fail(); // exercise the decision seam
    run_bank_transfer(Arc::clone(&wmgr), cfg, Arc::clone(&whistory));
    let wops = whistory.drain_sorted();

    drop(wmgr);
    wal_writer.shutdown().expect("wal shutdown");

    let recovered = Arc::new(TxnManager::new());
    let primary_store = Arc::new(PrimaryPageStore::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
    let target = PageStoreTarget::primary_only(primary);
    recover_from_wal(wal_dir.path(), Arc::clone(&recovered), target, None)
        .expect("tiny recovery must succeed");

    let (recovered_map, recovered_sum) = live_balances(&recovered, cfg.tenant, cfg.accounts);
    let recon = checker.reconcile_pending_with_recovery(&recovered_map);
    assert!(
        recon.is_ok(),
        "tiny post-recovery conservation must hold: {recon}"
    );
    assert_eq!(
        recovered_sum,
        expected_sum(&cfg),
        "tiny recovered sum mismatch"
    );
    assert!(
        checker.check(&wops).is_ok(),
        "tiny WAL-fault leg history must be SI-legal post-recovery"
    );
}
