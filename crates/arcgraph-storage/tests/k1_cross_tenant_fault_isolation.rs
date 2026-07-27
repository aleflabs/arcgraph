//! K-1b cross-tenant fault isolation pin (issue #214).
//!
//! ## Scope: in-process WAL-teardown isolation only
//!
//! This test exercises cross-tenant fault isolation under
//! **in-process WAL teardown + restart** (the multi-tenant runner's
//! `restart_wal` closure tears down the in-memory `K1Stack` and
//! recovers from disk into a fresh stack). True OS-level
//! multi-process SIGKILL × multi-tenant is **explicitly out of scope
//! at K-1b** and lands at K-1c — see `docs/roadmap.md:305-330`.
//! The K-1a `tests/k1_subprocess_smoke.rs` covers single-tenant
//! SIGKILL; multi-tenant SIGKILL is the K-1c bottleneck.
//!
//! Per ADR-038 amendment-03 §"Slice K", K-1b extends the K-1a
//! single-tenant harness scaffold (PR #176) with:
//!
//!  - **Multi-tenant workload generator**
//!    ([`arcgraph_storage::test_harness::k1::multi_tenant`]) that
//!    interleaves N tenants' commits with per-tenant
//!    [`InjectionConfig`].
//!  - **Per-tenant pre-crash ledger separation**
//!    ([`PreCrashLedger::create_in_dir`]) that writes one CSV per
//!    tenant — preventing fault-induced cross-pollution from
//!    masquerading as a recovery oracle pass.
//!  - **Cross-tenant invariant in the recovery oracle**
//!    ([`verify_cross_tenant_invariants`]) that pins: NO row whose
//!    bytes were committed under tenant A appears in tenant B's
//!    recovered state UNLESS B independently committed the same
//!    bytes.
//!
//! ## What this test verifies
//!
//! Three tenants (A=1, B=1001, C=1002) commit interleaved rows under
//! T1 strict durability:
//!
//!  - Tenant A carries an ELEVATED WAL fault rate (`wal_failure_rate
//!    = 0.20`) — enough to fire repeatedly across the workload but
//!    not so high that tenant A makes no progress.
//!  - Tenants B + C carry [`InjectionConfig::no_op`] — no faults
//!    fire on their ops.
//!  - Disjoint label spaces prevent coincidental byte-level matches:
//!    tenant A's labels live in `100_000..200_000`; tenant B's in
//!    `200_000..300_000`; tenant C's in `300_000..400_000`. Any
//!    cross-tenant byte appearance is real contamination, not
//!    coincidence.
//!
//! After the workload completes, the test recovers the WAL stack
//! from disk + runs [`verify_cross_tenant_invariants`] across all
//! three tenants. The hard contract:
//!
//!  - Tenants B + C: 100 % T1-strict recovery + 1:1 unique:total
//!    invariant + 0 ghost / unknown keys.
//!  - Tenant A: T1-strict acked commits all recovered (commits that
//!    failed pre-crash are not in the ledger so the oracle never
//!    expects them).
//!  - Cross-tenant: ZERO `CrossTenantContamination` violations.
//!
//! Determinism: the test runs the workload across 5 distinct seeds
//! to exercise the `(workload_seed, injection_seed_base)` determinism
//! contract under multi-tenant fault injection.
//!
//! ## What this test is NOT
//!
//! - It is NOT a SIGKILL subprocess workload (K-1b uses in-thread
//!   WAL teardown + restart cycles; the K-1a `k1_subprocess_smoke`
//!   covers SIGKILL recovery for a single tenant).
//! - It is NOT a 1-hour campaign (that's K-1c).
//! - It does NOT exercise snapshot install crash points (K-1c+d).
//!
//! Run:
//!
//! ```ignore
//! cargo test -p arcgraph-storage --release \
//!   --test k1_cross_tenant_fault_isolation
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
use arcgraph_storage::test_harness::k1::injection::InjectionConfig;
use arcgraph_storage::test_harness::k1::multi_tenant::{
    CommitOutcome, Interleave, MultiTenantWorkloadConfig, run_multi_tenant_workload,
};
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedState, CommittedStatsRebuild, CrossTenantOracleInput, OracleConfig, OracleViolation,
    RecoveredState, snapshot_catalog_stats, verify_cross_tenant_invariants,
};
use arcgraph_storage::test_harness::k1::subprocess::{LedgerRecord, PreCrashLedger};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use std::collections::{HashMap, HashSet};
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

fn recover_stack(dir: &std::path::Path) -> K1Stack {
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

    // M4-41 cold-start MVCC stats rebuild (per ADR-038 amendment-06
    // §D-25.1). The K-1b cross-tenant variant is the strictest
    // oracle for the cross-tenant contamination boundary — the
    // rebuild path's per-tenant fault isolation (per amendment-06
    // §2.5.1 / §D-25.4 Q3) is exercised when a tenant rebuild
    // partial-fails: the failure is logged + marked recovery_failed,
    // but other tenants' rebuilds proceed unaffected.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        eprintln!(
            "k1_cross_tenant_fault_isolation M4-41 rebuild: {} tenant(s) marked \
             recovery_failed: {:?}",
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
// Per-tenant counter / label-space helpers
// ─────────────────────────────────────────────────────────────────

/// Disjoint label-space allocator. Tenant A's labels are
/// `100_000..200_000`; tenant B's `200_000..300_000`; tenant C's
/// `300_000..400_000`. Disjoint byte spaces are critical for the
/// cross-tenant oracle: if tenant B's recovered state surfaces a
/// label in tenant A's range, that's unambiguous contamination
/// (false-positive impossible by construction).
fn label_offset_for(tenant: TenantId) -> u32 {
    match tenant.raw() {
        1 => 100_000,
        1001 => 200_000,
        1002 => 300_000,
        _ => panic!("tenant {} not in K-1b test set", tenant.raw()),
    }
}

/// Build a [`CommittedState`] from a per-tenant ledger slice.
fn build_committed_state(rows: &[LedgerRecord]) -> CommittedState {
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

/// Read back tenant `t`'s recovered state from the post-recovery
/// stack. Reads every key in `t`'s pre-crash any_history (so the
/// recovered map contains exactly the keys the oracle compares
/// against).
fn build_recovered_state(
    stack: &K1Stack,
    tenant: TenantId,
    pre: &CommittedState,
    labels: &[LabelId],
) -> RecoveredState {
    let mut rec = RecoveredState::default();
    for (t, id) in pre.any_history.keys() {
        debug_assert_eq!(*t, tenant, "pre-crash state must be per-tenant scoped");
        let tx = stack.mgr.begin(*t);
        match read_node_with_store(&stack.store, &tx, *id) {
            Ok(Some(rec_node)) => {
                rec.bytes_by_key.insert(
                    (*t, *id),
                    (
                        rec_node.label_id,
                        rec_node.inline_u32a,
                        rec_node.inline_u32b,
                    ),
                );
            }
            Ok(None) => {}
            Err(e) => panic!(
                "k1_cross_tenant_fault_isolation read_node_with_store error \
                 for tenant {} node {}: {e:?}",
                t.raw(),
                id.raw()
            ),
        }
    }
    if let Some(stats) = stack.store.catalog_stats(tenant) {
        let snap = snapshot_catalog_stats(&stats, labels, &[]);
        rec.stats_by_tenant.insert(tenant, snap);
    }
    let _ = stack.catalog;
    rec
}

// ─────────────────────────────────────────────────────────────────
// One-seed run
// ─────────────────────────────────────────────────────────────────

/// Run the K-1b cross-tenant workload + recover + oracle for a
/// single seed. Asserts the cross-tenant invariant holds.
fn run_one_seed(seed: u64, commits_per_tenant: u64) -> CrossTenantSeedReport {
    let workspace = TempDir::new().expect("tempdir");
    let wal_dir = workspace.path().join("wal");
    let ledger_dir = workspace.path().join("pre_crash_ledger");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let stack_holder = Arc::new(Mutex::new(Some(K1Stack::build(&wal_dir))));
    let ledger = PreCrashLedger::create_in_dir(&ledger_dir).expect("ledger create_in_dir");

    let tenant_a = TenantId::new(1);
    let tenant_b = TenantId::new(1001);
    let tenant_c = TenantId::new(1002);
    let tenants = vec![tenant_a, tenant_b, tenant_c];

    // Tenant A: elevated WAL fault rate. Tenants B + C: no_op.
    // 0.20 = 1 in 5 ops fires a fault. With per-tenant target=N,
    // tenant A sees ~N/5 faults across the workload.
    let cfg = MultiTenantWorkloadConfig::baseline(
        tenants.clone(),
        commits_per_tenant,
        seed,
        seed.wrapping_mul(0x5EED_C0DE_DEAD_BEEF),
    )
    .with_injection(
        tenant_a,
        InjectionConfig {
            wal_failure_rate: 0.20,
            ..InjectionConfig::no_op()
        },
    )
    .with_interleave(Interleave::RoundRobin);

    // Per-tenant counters keep label / a / b deterministic so
    // determinism + disjoint-byte-space invariants both hold.
    let counters: Mutex<HashMap<TenantId, u32>> = Mutex::new(HashMap::new());

    let commit_op = |tenant: TenantId| -> Option<CommitOutcome> {
        let stack_guard = stack_holder.lock().unwrap();
        let stack = stack_guard.as_ref()?;
        let mut tx = stack.mgr.begin(tenant);
        let mut counters_g = counters.lock().unwrap();
        let local = counters_g.entry(tenant).or_insert(0);
        *local = local.wrapping_add(1);
        let local_val = *local;
        drop(counters_g);
        // Disjoint label space per tenant.
        let label = label_offset_for(tenant) + local_val;
        // a/b are derived from the local counter — deterministic +
        // distinct between tenants because the label-space
        // disjointness already ensures the bytes are distinct, but
        // we make a/b also tenant-tagged for redundant evidence.
        let a = local_val.wrapping_mul(7);
        let b = local_val.wrapping_mul(13);
        let id = match create_node(
            &stack.store,
            &mut tx,
            tenant,
            LabelId::new(label),
            &PropertyData::InlineU32Pair(a, b),
        ) {
            Ok(id) => id,
            Err(_) => return None,
        };
        let tier = stack.catalog.durability_tier(tenant);
        match commit(tx, &stack.store) {
            Ok(_) => Some(CommitOutcome {
                node_id_raw: id.raw(),
                label,
                a,
                b,
                // K-1b workload commits T1 Strict by default (tier
                // tag from catalog::durability_tier). The oracle
                // expects tier=1 for Strict per oracle::tier_to_u8.
                tier: match tier {
                    arcgraph_core::DurabilityTier::Strict => 1,
                    arcgraph_core::DurabilityTier::Periodic { .. } => 3,
                },
            }),
            Err(_) => None,
        }
    };

    let restart_wal = || {
        // Mirror the K-1a 30 s smoke pattern: tear down the WAL +
        // recover into a fresh K1Stack rooted at the same WAL dir.
        let mut guard = stack_holder.lock().unwrap();
        if let Some(prior) = guard.take() {
            prior.shutdown();
            let recovered = recover_stack(&wal_dir);
            *guard = Some(recovered);
        }
    };

    let report = run_multi_tenant_workload(&cfg, &ledger, commit_op, restart_wal);

    // Final shutdown + recover for the oracle pass.
    let final_stack = stack_holder.lock().unwrap().take();
    if let Some(s) = final_stack {
        s.shutdown();
    }
    let recovered = recover_stack(&wal_dir);

    // Read back per-tenant ledgers — each tenant's CSV is physically
    // separate so a torn-tail in tenant A cannot leak into tenant B.
    let per_tenant_rows: HashMap<TenantId, Vec<LedgerRecord>> = tenants
        .iter()
        .map(|t| {
            let rows = PreCrashLedger::read_for(&ledger_dir, t.raw())
                .expect("read tenant CSV from per-tenant directory");
            (*t, rows)
        })
        .collect();

    // Build per-tenant pre-crash + recovered states.
    let mut pre_crash_per_tenant: HashMap<TenantId, CommittedState> = HashMap::new();
    let mut recovered_per_tenant: HashMap<TenantId, RecoveredState> = HashMap::new();
    let mut all_labels: Vec<LabelId> = Vec::new();
    for tenant in &tenants {
        let rows = &per_tenant_rows[tenant];
        let pre = build_committed_state(rows);
        for stats in pre.stats_by_tenant.values() {
            for label in stats.label_counts.keys() {
                all_labels.push(*label);
            }
        }
        let rec = build_recovered_state(&recovered, *tenant, &pre, &all_labels);
        pre_crash_per_tenant.insert(*tenant, pre);
        recovered_per_tenant.insert(*tenant, rec);
    }
    all_labels.sort_by_key(|l| l.raw());
    all_labels.dedup();

    let oracle_input = CrossTenantOracleInput {
        pre_crash_per_tenant,
        recovered_per_tenant,
    };

    // K-1b oracle config: M4-41 cold-start MVCC stats rebuild (per
    // ADR-038 amendment-06 §3 R1 acceptance criteria items 2 + 3)
    // ratified the persistence shape; this PR implements it and flips
    // `stats_inconsistency_fatal` from `false` → `true`. The
    // `fail_fast: false` knob remains as it was in PR #219 — the
    // K-1b cross-tenant oracle must accumulate ALL cross-tenant
    // violations + per-tenant violations into the report so the
    // contamination signal doesn't hide behind a stats-drift
    // short-circuit. The strict-mode stats comparison + the
    // collect-all violation path are both load-bearing for this
    // smoke's K-1b cross-tenant fault-isolation contract.
    //
    // The four R1 sites this implementation flips together (per
    // amendment-06 §3 R1 enumeration):
    // - `tests/k1_smoke_30s.rs` (K-1a default in-thread)
    // - `tests/k1_extended_smoke_5min.rs` (K-1a 5-min extended)
    // - `tests/k1_subprocess_smoke.rs` (K-1a SIGKILL subprocess)
    // - this site (K-1b cross-tenant fault-isolation, PR #219)
    let oracle_cfg = OracleConfig {
        fail_fast: false,
        ..OracleConfig::default()
    };
    assert!(
        oracle_cfg.stats_inconsistency_fatal,
        "M4-41 R1 acceptance criterion (per ADR-038 amendment-06 §3 item 3): \
         this K-1b cross-tenant smoke MUST run with stats_inconsistency_fatal=true; \
         the cold-start rebuild path makes the strict oracle non-vacuous AT THE \
         cross-tenant contamination boundary."
    );
    let oracle_report = verify_cross_tenant_invariants(&oracle_input, &oracle_cfg)
        .expect("verify_cross_tenant_invariants should not Err under fail_fast=false");

    // ── Hard contract: ZERO CrossTenantContamination ──
    let contamination_count = oracle_report
        .cross_tenant_violations
        .iter()
        .filter(|v| matches!(v, OracleViolation::CrossTenantContamination { .. }))
        .count();
    assert_eq!(
        contamination_count, 0,
        "seed {seed:#x}: cross-tenant contamination MUST NOT fire; \
         got {contamination_count} violations: {:?}",
        oracle_report.cross_tenant_violations
    );

    // ── Hard contract: tenants B + C have NO faults injected ──
    let t_b_report = report.per_tenant.get(&tenant_b).unwrap();
    let t_c_report = report.per_tenant.get(&tenant_c).unwrap();
    assert_eq!(
        t_b_report.total_faults(),
        0,
        "seed {seed:#x}: tenant B must have 0 faults (rate=0); got {}",
        t_b_report.total_faults()
    );
    assert_eq!(
        t_c_report.total_faults(),
        0,
        "seed {seed:#x}: tenant C must have 0 faults (rate=0); got {}",
        t_c_report.total_faults()
    );

    // ── HARD CONTRACT (codex review of PR #219, HIGH-1) ──
    //
    // Closes the K-1a BLOCKER-1 oracle-relaxation pattern recurrence
    // flagged by the K-1a retro (memory:
    // `feedback_review_oracle_relaxations.md` — "test suite green ≠
    // test correctness; relaxed oracles can mask the bugs they're
    // supposed to catch").
    //
    // Pre-fix this test pinned `report.t1_satisfied == report.t1_keys`,
    // which is VACUOUS when recovery loses keys: in
    // `oracle.rs::verify_post_recovery_invariants`, both `t1_keys`
    // (line 416) and `t1_satisfied` (line 430) are incremented ONLY in
    // the `(observed=Some, latest_t1=Some)` arm. A total-recovery-loss
    // case lands in the `(observed=None, latest_t1=Some)` arm at line
    // 454 — that arm pushes `T1Missing` into `report.violations` (under
    // `fail_fast=false`) but increments NEITHER counter. The vacuous
    // `0 == 0` therefore passes green even when every key is lost.
    //
    // The non-vacuous pin is `violations.is_empty()`. With
    // `OracleConfig::fail_fast=false` (set above), every per-tenant
    // violation (`T1Missing`, `T1StrictDrift`, `GhostBytes`,
    // `UnknownKey`, `T3RpoLossExceeded`) accumulates into
    // `report.violations`; an empty vec therefore certifies the K-1a
    // recovery contract held for that tenant.
    //
    // Per-tenant contract:
    //
    //  - Tenants B + C: `wal_failure_rate=0.0` → ZERO faults → the
    //    pre-crash ledger reflects EVERY ack and recovery MUST
    //    reproduce every row (T1 strict). No recovery loss, no
    //    drift, no ghosts.
    //
    //  - Tenant A: `wal_failure_rate=0.20`, BUT the multi-tenant
    //    workload's `commit_skipped_due_to_fault` path
    //    (`multi_tenant.rs:368-371`) `continue`s BEFORE invoking
    //    `commit_op` for any op whose roll fired a WAL/process-crash
    //    fault — and the ledger.record() call is downstream of
    //    `commit_op` returning `Some`. Therefore tenant A's pre-crash
    //    `any_history` contains ONLY rows that survived the fault roll
    //    AND committed AND ledger-recorded; recovery MUST reproduce
    //    every such row. Faults disrupt the WAL but do not corrupt
    //    A's pre-crash ledger.
    //
    // Workload-progress sanity (`unique_keys > 0`) is retained so a
    // workload regression that produces ZERO commits doesn't slip
    // through `violations.is_empty()` (vacuous-on-empty-state is the
    // exact failure mode this fix closes).
    let report_a = oracle_report.per_tenant.get(&tenant_a).unwrap();
    let report_b = oracle_report.per_tenant.get(&tenant_b).unwrap();
    let report_c = oracle_report.per_tenant.get(&tenant_c).unwrap();
    assert!(
        report_a.unique_keys > 0,
        "seed {seed:#x}: tenant A should have committed at least 1 row \
         (faults are pre-commit; A's ledger should still have non-faulty acks)"
    );
    assert!(
        report_b.unique_keys > 0,
        "seed {seed:#x}: tenant B should have committed at least 1 row"
    );
    assert!(
        report_c.unique_keys > 0,
        "seed {seed:#x}: tenant C should have committed at least 1 row"
    );
    assert!(
        report_a.violations.is_empty(),
        "seed {seed:#x}: tenant A post-recovery: every acked T1 commit MUST recover \
         (faults skip commit_op pre-ledger-write, so A's pre-crash ledger excludes \
         faulty rows; non-faulty acks MUST round-trip through WAL recovery). \
         Got {} violations: {:#?}",
        report_a.violations.len(),
        report_a.violations,
    );
    assert!(
        report_b.violations.is_empty(),
        "seed {seed:#x}: tenant B post-recovery (wal_failure_rate=0): NO faults \
         injected for B, so EVERY pre-crash row MUST recover under T1 strict. \
         Got {} violations: {:#?}",
        report_b.violations.len(),
        report_b.violations,
    );
    assert!(
        report_c.violations.is_empty(),
        "seed {seed:#x}: tenant C post-recovery (wal_failure_rate=0): NO faults \
         injected for C, so EVERY pre-crash row MUST recover under T1 strict. \
         Got {} violations: {:#?}",
        report_c.violations.len(),
        report_c.violations,
    );

    // ── Cross-check: t1_satisfied/t1_keys are now redundant with the
    // HARD CONTRACT above. Retained as additional pins; they're
    // strictly weaker (vacuous-on-loss) so they cannot fail in
    // isolation, but they document the expected counter shape and
    // would catch any future oracle refactor that broke the
    // `(Some, Some)` arm increment logic without adding a violation.
    assert_eq!(
        report_a.t1_satisfied, report_a.t1_keys,
        "seed {seed:#x}: tenant A's acked T1 commits must all recover"
    );
    assert_eq!(
        report_b.t1_satisfied, report_b.t1_keys,
        "seed {seed:#x}: tenant B (no faults) must have 100 % T1-strict satisfaction"
    );
    assert_eq!(
        report_c.t1_satisfied, report_c.t1_keys,
        "seed {seed:#x}: tenant C (no faults) must have 100 % T1-strict satisfaction"
    );

    // ── Sanity: tenant A actually fired SOME faults ──
    // With 0.20 rate over `commits_per_tenant` ops, expected faults
    // ≈ 0.20 × N. Allow a generous floor so flakiness doesn't gate
    // CI: floor = max(1, expected/4).
    let t_a_report = report.per_tenant.get(&tenant_a).unwrap();
    let expected_faults_a = (commits_per_tenant as f64) * 0.20;
    let fault_floor = ((expected_faults_a / 4.0).floor() as u64).max(1);
    assert!(
        t_a_report.total_faults() >= fault_floor,
        "seed {seed:#x}: tenant A fault count {} below floor {fault_floor} \
         (expected ≈ {expected_faults_a:.1}); injection rate too low or RNG drift",
        t_a_report.total_faults()
    );

    let seed_report = CrossTenantSeedReport {
        seed,
        commits_acked_a: t_a_report.commits_acked,
        commits_acked_b: t_b_report.commits_acked,
        commits_acked_c: t_c_report.commits_acked,
        faults_a: t_a_report.total_faults(),
        wal_restarts: report.wal_restarts,
        cross_tenant_checks: oracle_report.cross_tenant_checks,
    };

    recovered.shutdown();
    seed_report
}

#[derive(Debug)]
struct CrossTenantSeedReport {
    seed: u64,
    commits_acked_a: u64,
    commits_acked_b: u64,
    commits_acked_c: u64,
    faults_a: u64,
    wal_restarts: u64,
    cross_tenant_checks: u64,
}

// ─────────────────────────────────────────────────────────────────
// The test
// ─────────────────────────────────────────────────────────────────

/// K-1b: cross-tenant fault isolation across multiple seeds.
///
/// Per the issue #214 spec + the K-1b spawn prompt, run the workload
/// across N distinct seeds to exercise the determinism contract under
/// multi-tenant fault injection. Each seed:
///
///  1. Builds 3 tenants (A, B, C) with disjoint label spaces.
///  2. Runs the multi-tenant workload with elevated WAL fault rate
///     ONLY on tenant A; tenants B + C have no_op.
///  3. Recovers the WAL stack from disk.
///  4. Asserts the cross-tenant invariant holds:
///     - 0 `CrossTenantContamination` violations.
///     - Tenants B + C have 0 faults injected + 100 % T1 recovery.
///     - Tenant A's acked commits all recover.
#[test]
fn k1b_in_process_wal_teardown_cross_tenant_fault_isolation_across_seeds() {
    // 5 seeds — enough to exercise the determinism contract without
    // dominating CI wall time. Each seed spawns ~30 ops × 3 tenants
    // = 90 total commits; with 0.20 fault rate on tenant A, ~6 WAL
    // restarts per seed. Total wall time ≈ 5 s across all seeds.
    let seeds: [u64; 5] = [
        0xA11C_E001_0000_0001,
        0xA11C_E002_0000_0002,
        0xA11C_E003_0000_0003,
        0xA11C_E004_0000_0004,
        0xA11C_E005_0000_0005,
    ];
    let commits_per_tenant: u64 = 30;

    let mut reports = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let report = run_one_seed(seed, commits_per_tenant);
        eprintln!(
            "k1b_cross_tenant_seed {:#x}: \
             A=(commits={}, faults={}) B=commits={} C=commits={} restarts={} checks={}",
            report.seed,
            report.commits_acked_a,
            report.faults_a,
            report.commits_acked_b,
            report.commits_acked_c,
            report.wal_restarts,
            report.cross_tenant_checks,
        );
        reports.push(report);
    }

    // Sanity: across the 5 seeds, fault counts vary (proves seed
    // partitioning works; if every seed fired the same fault count
    // the determinism contract or per-seed rng partitioning is
    // broken).
    let unique_fault_counts: HashSet<u64> = reports.iter().map(|r| r.faults_a).collect();
    assert!(
        unique_fault_counts.len() >= 2,
        "across 5 distinct seeds expected ≥ 2 distinct fault counts; \
         got {} unique values: {:?} \
         (per-seed rng partitioning may be broken)",
        unique_fault_counts.len(),
        unique_fault_counts
    );
}

/// K-1b determinism: same seed → same fault sequence + same per-tenant
/// commit counts. Mirror of K-1a's `interleaved_per_op_decision_
/// sequence_deterministic` lifted to multi-tenant.
#[test]
fn k1b_workload_is_deterministic_per_seed() {
    let seed = 0xDECA_FBAD_0000_0042;
    let commits_per_tenant = 20;
    let a = run_one_seed(seed, commits_per_tenant);
    let b = run_one_seed(seed, commits_per_tenant);
    assert_eq!(
        a.faults_a, b.faults_a,
        "same seed must produce same tenant-A fault count: \
         a={} b={}",
        a.faults_a, b.faults_a
    );
    assert_eq!(
        a.wal_restarts, b.wal_restarts,
        "same seed must produce same WAL restart count"
    );
    assert_eq!(
        a.commits_acked_b, b.commits_acked_b,
        "same seed must produce same tenant-B commit count"
    );
    assert_eq!(
        a.commits_acked_c, b.commits_acked_c,
        "same seed must produce same tenant-C commit count"
    );
}
