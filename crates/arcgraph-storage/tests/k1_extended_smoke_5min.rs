//! K-1 extended smoke (5 min) — gated by `K1_EXTENDED_SMOKE=1`.
//!
//! Per ADR-038 amendment-03 §"Slice K" + spec D6 / Part 5, the
//! 5-minute extended smoke is the next-tier campaign that:
//!
//!  - Runs the same per-op rate-based injection harness as the 30 s
//!    smoke, scaled to 5 minutes.
//!  - Exercises the per-op injection rates closer to their
//!    asymptotic regime (with ~6000 rolls at 0.5 % we expect ~30
//!    WAL faults — comfortably within the law-of-large-numbers
//!    convergence band).
//!  - Stays opt-in: gated by the `K1_EXTENDED_SMOKE` env var so CI
//!    does not pay the 5 min wall-clock per push.
//!  - Lives outside the K-3 multi-hour campaign budget. K-3
//!    extends beyond this with 10 K-cycle long-running campaigns +
//!    encoding-mismatch I-V coverage (per spec D1 forward-references).
//!
//! ## Why a separate test file (vs. an env-driven knob on the 30 s
//! smoke)?
//!
//! The 30 s smoke is a CI-gated test; its assertions are tuned to
//! the 30 s wall clock (e.g., the "≥ 1 WAL fault" floor). The
//! extended smoke uses a tighter tolerance (≥ 10 WAL faults at
//! 5 min) that would generate spurious failures if dialed into the
//! 30 s smoke. Keeping the test files separate keeps the
//! assertions tight without relying on env-var gating to switch
//! invariants.
//!
//! ## Run
//!
//! ```ignore
//! K1_EXTENDED_SMOKE=1 cargo test -p arcgraph-storage \
//!   --test k1_extended_smoke_5min -- --ignored --nocapture
//! ```
//!
//! Without the env var the test is `#[ignore]`-marked AND no-ops
//! when invoked — the env-var gate is belt-and-braces against an
//! engineer accidentally running `cargo test --ignored` and
//! waiting 5 minutes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use arcgraph_core::{DurabilityTier, LabelId, NodeId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::test_harness::k1::injection::{
    InjectionConfig, InjectionDecisionRng, InjectionKind, InjectionTally, maybe_inject_wal_failure,
};
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedState, CommittedStatsRebuild, OracleConfig, RecoveredState, snapshot_catalog_stats,
    verify_post_recovery_invariants,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;

const EXTENDED_SMOKE_ENV: &str = "K1_EXTENDED_SMOKE";

fn extended_smoke_enabled() -> bool {
    std::env::var(EXTENDED_SMOKE_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn extended_smoke_duration_secs() -> u64 {
    std::env::var("K1_EXTENDED_SMOKE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300)
}

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
    // §D-25.1). See `tests/k1_smoke_30s.rs::recover_stack` for the
    // full justification — synchronously rebuilds per-tenant
    // CatalogStats from the recovered MVCC state at the recovered
    // LSN, so the strict-mode oracle in the smoke proper has populated
    // stats to compare against.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        eprintln!(
            "k1_extended_smoke_5min M4-41 rebuild: {} tenant(s) marked recovery_failed: {:?}",
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

type LedgerRow = (TenantId, NodeId, u32, u32, u32, DurabilityTier);

#[derive(Default)]
struct InProcessLedger {
    rows: Mutex<Vec<LedgerRow>>,
}

impl InProcessLedger {
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        tenant: TenantId,
        id: NodeId,
        label: u32,
        a: u32,
        b: u32,
        tier: DurabilityTier,
    ) {
        self.rows
            .lock()
            .unwrap()
            .push((tenant, id, label, a, b, tier));
    }

    fn snapshot(&self) -> Vec<LedgerRow> {
        self.rows.lock().unwrap().clone()
    }
}

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
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn build_committed_state(rows: &[LedgerRow]) -> CommittedState {
    let mut s = CommittedState::default();
    let mut stats: HashMap<TenantId, CommittedStatsRebuild> = HashMap::new();
    for (tenant, id, label, a, b, tier) in rows {
        let key = (*tenant, *id);
        let bytes = (*label, *a, *b);
        s.any_history
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(bytes);
        if matches!(tier, DurabilityTier::Strict) {
            s.latest_t1.insert(key, bytes);
        }
        let st = stats.entry(*tenant).or_default();
        *st.label_counts.entry(LabelId::new(*label)).or_insert(0) += 1;
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
            Err(e) => panic!("k1_extended_smoke read_node_with_store error: {e:?}"),
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

fn do_commit(
    stack: &K1Stack,
    ledger: &InProcessLedger,
    tenant: TenantId,
    label: u32,
    a: u32,
    b: u32,
) -> Result<NodeId, ()> {
    let mut tx = stack.mgr.begin(tenant);
    let id = match create_node(
        &stack.store,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(a, b),
    ) {
        Ok(id) => id,
        Err(_) => return Err(()),
    };
    let tier = stack.catalog.durability_tier(tenant);
    match commit(tx, &stack.store) {
        Ok(_) => {
            ledger.record(tenant, id, label, a, b, tier);
            Ok(id)
        }
        Err(_) => Err(()),
    }
}

#[test]
#[ignore = "K-1 extended smoke (5 min). Gated by K1_EXTENDED_SMOKE=1; see module doc."]
fn k1_extended_smoke_5min() {
    if !extended_smoke_enabled() {
        eprintln!(
            "k1_extended_smoke_5min: skipped (set {}=1 to enable)",
            EXTENDED_SMOKE_ENV
        );
        return;
    }
    let duration = Duration::from_secs(extended_smoke_duration_secs());

    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let stack_holder = Arc::new(Mutex::new(Some(K1Stack::build(&wal_dir))));
    let ledger = Arc::new(InProcessLedger::default());

    let stop = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));
    let injection_attempts = Arc::new(AtomicU64::new(0));
    let injection_tally = Arc::new(InjectionTally::new());

    // Tighter rates than the 30 s smoke — 5 minutes lets us reach
    // closer to spec D2 defaults without dominating wall time.
    let injection_config = InjectionConfig {
        wal_failure_rate: 0.01,
        snapshot_failure_rate: 0.0,
        process_crash_rate: 0.0,
        background_fsync_failure_rate: 0.0,
        wal_partial_write_rate: 0.0,
    };
    let injection_rng = Arc::new(InjectionDecisionRng::new(0xEC7E_9DED_5777_E000));

    let worker_a = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&stack_holder);
        let ledger = Arc::clone(&ledger);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xABCD_1234);
            let mut user_label: u32 = 100_000;
            while !stop.load(Ordering::Relaxed) {
                user_label = user_label.wrapping_add(1);
                let a = rng.next_u32();
                let b = rng.next_u32();
                if let Some(stack) = stack_holder.lock().unwrap().as_ref() {
                    let _ = do_commit(stack, &ledger, TenantId::DEFAULT, user_label, a, b);
                    total_ops.fetch_add(1, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(8));
            }
        })
    };

    let worker_b = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&stack_holder);
        let ledger = Arc::clone(&ledger);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xFACE_0F11);
            let tenant = TenantId::new(1001);
            let mut user_label: u32 = 200_000;
            while !stop.load(Ordering::Relaxed) {
                user_label = user_label.wrapping_add(1);
                let a = rng.next_u32();
                let b = rng.next_u32();
                if let Some(stack) = stack_holder.lock().unwrap().as_ref() {
                    let _ = do_commit(stack, &ledger, tenant, user_label, a, b);
                    total_ops.fetch_add(1, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(7));
            }
        })
    };

    let injector = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&stack_holder);
        let cfg = injection_config;
        let rng = Arc::clone(&injection_rng);
        let attempts = Arc::clone(&injection_attempts);
        let tally = Arc::clone(&injection_tally);
        let wal_dir = wal_dir.clone();
        thread::spawn(move || {
            let mut op_count: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
                op_count = op_count.saturating_add(1);
                attempts.fetch_add(1, Ordering::Relaxed);
                if let Some(kind) = maybe_inject_wal_failure(&cfg, &rng, op_count) {
                    tally.record(kind);
                    match kind {
                        InjectionKind::WalFsyncFail | InjectionKind::WalPartialWrite => {
                            let mut guard = stack_holder.lock().unwrap();
                            if let Some(prior) = guard.take() {
                                prior.shutdown();
                                let recovered = recover_stack(&wal_dir);
                                *guard = Some(recovered);
                            }
                        }
                        _ => {}
                    }
                }
            }
        })
    };

    thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);
    let _ = worker_a.join();
    let _ = worker_b.join();
    let _ = injector.join();

    let final_stack = stack_holder.lock().unwrap().take();
    if let Some(s) = final_stack {
        s.shutdown();
    }
    let recovered = recover_stack(&wal_dir);

    let rows = ledger.snapshot();
    let pre = build_committed_state(&rows);
    let mut all_labels: Vec<LabelId> = pre
        .stats_by_tenant
        .values()
        .flat_map(|s| s.label_counts.keys().copied())
        .collect();
    all_labels.sort_by_key(|l| l.raw());
    all_labels.dedup();
    let rec = build_recovered_state(&recovered, &pre, &all_labels);

    // M4-41 cold-start MVCC stats rebuild (per ADR-038 amendment-06
    // §3 R1 acceptance criteria items 2 + 3): see
    // `tests/k1_smoke_30s.rs` for full justification. Strict-mode
    // oracle comparison verifies the cold-start rebuild's correctness
    // end-to-end over the 5-minute extended window — the longer
    // window exercises more commit volume + more recovery cycles, so
    // any rebuild regression has more surface area to surface.
    let oracle_cfg = OracleConfig::default();
    assert!(
        oracle_cfg.stats_inconsistency_fatal,
        "M4-41 R1 acceptance criterion (per ADR-038 amendment-06 §3 item 3): \
         this K-1 extended smoke MUST run with stats_inconsistency_fatal=true; \
         the cold-start rebuild path makes the strict oracle non-vacuous."
    );
    let report = verify_post_recovery_invariants(&pre, &rec, &oracle_cfg)
        .expect("k1 extended smoke oracle violation");

    let wal_faults = injection_tally.count(InjectionKind::WalFsyncFail)
        + injection_tally.count(InjectionKind::WalPartialWrite);

    eprintln!(
        "k1_extended_smoke_5min: duration={}s ops={} unique_keys={} t1_keys={} \
         t1_satisfied={} historical_match={} t3_rpo_lost={} \
         injection_attempts={} wal_faults={}",
        extended_smoke_duration_secs(),
        total_ops.load(Ordering::Relaxed),
        report.unique_keys,
        report.t1_keys,
        report.t1_satisfied,
        report.historical_match,
        report.t3_rpo_lost,
        injection_attempts.load(Ordering::Relaxed),
        wal_faults,
    );

    // M4-41 cold-start rebuild closure: with `stats_inconsistency_fatal
    // = true` (per amendment-06 §3 R1 above), any per-tenant drift
    // would have caused `verify_post_recovery_invariants` to return
    // Err(StatsInconsistent) — short-circuiting via fail_fast. Reaching
    // this point means every tenant's post-rebuild stats matched the
    // rebuilt-from-ledger expectation over the full 5-minute window.
    // The K-1a honest-deferral eprintln is no longer needed; it has
    // been promoted to a hard CI assertion via the strict oracle.
    // PR #176 K-1a BLOCKER-1 deferral closes here at the extended
    // smoke too; mirrors the matching closure in k1_smoke_30s.

    // Tighter contract checks at 5 min:
    assert!(report.unique_keys > 0, "extended smoke: 0 commits");
    assert_eq!(report.t1_keys, report.unique_keys);
    assert_eq!(report.t1_satisfied, report.t1_keys);
    // Expected fault count: rolls happen every 50 ms ⇒ rolls/sec = 20.
    // At 1 % rate over `duration` seconds: expected ≈ 0.20 × duration.
    // Floor = max(1, expected / 4) to allow ±4× tolerance for RNG
    // and scheduler jitter; the full 5 min run yields ~60 expected
    // faults so floor=15.
    let dur_secs = extended_smoke_duration_secs() as f64;
    let expected_faults = 0.20 * dur_secs;
    let fault_floor = ((expected_faults / 4.0).floor() as u64).max(1);
    assert!(
        wal_faults >= fault_floor,
        "extended smoke: only {wal_faults} WAL faults at 1 % over {dur_secs}s \
         (expected ≈ {expected_faults:.1}; floor {fault_floor}) — \
         injection rate too low or RNG drift"
    );

    recovered.shutdown();
}
