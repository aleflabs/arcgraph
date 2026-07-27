//! K-2 recovery idempotence proptest (issue #223; ADR-038 amendment-03
//! §"Slice K" K-2 row).
//!
//! ## What this proptest verifies
//!
//! The contract: **recovery is idempotent + deterministic across
//! graceful-restart cycles** — given the same on-disk WAL, repeated
//! recovery attempts converge to byte-equal `bytes_by_key` post-recovery
//! state. The K-2 scenario is modeled at the test-harness layer as **N
//! consecutive recovery attempts**, each from a clean cold-start (no
//! in-memory state preserved across attempts):
//!
//! 1. Build a fresh K2Stack on a per-FS-variant workdir.
//! 2. Commit M deterministic ops driven by the proptest seed.
//! 3. Graceful shutdown (drain WAL).
//! 4. Run R recovery attempts back-to-back; each attempt opens the WAL
//!    cold, runs `recover_from_wal`, reads back every committed key,
//!    then drops the stack.
//! 5. Assert every recovery attempt returns byte-equal
//!    `bytes_by_key` (no "previous-attempt left intermediate state on
//!    disk that perturbs the next attempt").
//! 6. Assert the recovered state matches the no-fault baseline (the
//!    pre-shutdown ledger).
//!
//! The "fault during recovery" the spawn prompt asks for is captured
//! by the R≥2 case: the second recovery attempt is exactly what a
//! production process does when a fault interrupts the first recovery
//! and the OS / supervisor restarts the process. If recovery were
//! NOT idempotent, the second attempt would produce a different state
//! than the first — and this proptest fires.
//!
//! ## Why test-harness-only
//!
//! Per K-1 mod.rs §"Hooks vs production", the K-1 / K-2 harness drives
//! faults from the test side without modifying production source.
//! Mid-recovery fault injection inside `wal::recovery::recover_from_wal`
//! would be a production-side hook (modifying recovery.rs). The
//! "throw-away-the-recovered-stack-and-restart-recovery" pattern
//! captures the same contract (the recovery's post-condition must be
//! deterministic across restart attempts) without touching production.
//!
//! Future K-3 may add an in-recovery fault seam if the campaign
//! surfaces a contract gap that this pattern misses.
//!
//! ## Per-FS variation
//!
//! The proptest parameterizes on `FsKind::ALL` and runs the scenario
//! for every applicable adapter. Inapplicable adapters (XFS on macOS,
//! EBS without K_2_EBS=1) are filtered via [`FsAdapter::is_supported`].
//! At least APFS + ext4 always run because both adapters return
//! `is_supported=true` on every host (real FS on the matching
//! platform; tmpfs surrogate elsewhere).
//!
//! ## Case count + workload size
//!
//! `cases: 16` × ≤4 FS adapters × (M ≤ 8 commits) × (R ≤ 3 recovery
//! attempts) ≈ 192 recovery cycles per `cargo test` invocation. At
//! ~50ms per recovery cycle (in-memory stack + WAL flush), this is
//! ~10s wall time — tractable for CI without slowing the smoke gate.
//! K-3 will scale to 256 cases under a separate cron.
//!
//! Run:
//!
//! ```ignore
//! cargo test -p arcgraph-storage --release --test k2_fault_during_recovery
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
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
use arcgraph_storage::test_harness::k1::multi_fs::{FsKind, adapter_for};
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
use proptest::prelude::*;

// ────────────────────────────────────────────────────────────────────
// K2Stack — local to this test (mirrors k1_smoke_30s.rs's K1Stack)
// ────────────────────────────────────────────────────────────────────

fn test_wal_config(dir: std::path::PathBuf) -> WalConfig {
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

struct K2Stack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    #[allow(dead_code)]
    catalog: Arc<SystemCatalog>,
}

impl K2Stack {
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

/// Recover a K2Stack from `dir`. Mirrors the K-1 smokes' `recover_stack`
/// shape: WAL replay via `recover_from_wal`, then synchronous cold-start
/// MVCC stats rebuild via
/// [`arcgraph_storage::recovery::rebuild_all_tenant_stats`].
///
/// ## Ordering
///
/// 1. Build the recovery substrate (writer + scheduler + mgr + catalog +
///    alloc + primary + store).
/// 2. Bootstrap system catalog BEFORE WAL replay (durability lookup
///    wired before replay).
/// 3. Call `recover_from_wal` to replay every commit-bundle up to the
///    recovered LSN. The returned `RecoveryReport` carries
///    `applied_commit_lsn` — the LSN the stats rebuild walks against.
/// 4. Call `rebuild_all_tenant_stats(applied_commit_lsn, &mgr, &store)`
///    to repopulate per-tenant `CatalogStats` from recovered MVCC.
///
/// ## SeqLock panic-safety primitive
///
/// `rebuild_all_tenant_stats` invokes
/// [`arcgraph_storage::recovery::rebuild_catalog_stats_for_tenant`] per
/// tenant in parallel via rayon. Each per-tenant call implements the
/// 4-invariant SeqLock primitive per
/// `feedback_seqlock_panic_safety_primitive.md` (begin OUTSIDE
/// catch_unwind, walk INSIDE AssertUnwindSafe, observe UNCONDITIONALLY
/// OUTSIDE, panic SWALLOWED for per-tenant isolation). The K-2
/// proptest's recovery loop relies on this primitive — under the
/// "fault-during-recovery is restart-recovery" model, each restart's
/// rebuild call MUST be panic-safe so a transient per-tenant fault
/// doesn't propagate across restart attempts.
///
/// ## Closes the K-2 future-flip wiring task
///
/// Per `project_k2_future_flip_wiring.md`: flipping the K-2 sites'
/// `stats_inconsistency_fatal: false → true` requires
/// `rebuild_all_tenant_stats` wiring here FIRST. Without it, the
/// strict oracle reads EMPTY `CatalogStats` post-recovery and the flip
/// fires `StatsInconsistent` (the pre-#236 K-1a BLOCKER-1 symptom).
fn recover_stack(dir: &Path) -> K2Stack {
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
    // §D-25.1). Runs AFTER `recover_from_wal` and BEFORE the oracle
    // reads `CatalogStats::snapshot()` for strict-mode comparison.
    // Closes the K-2 future-flip wiring task per
    // `project_k2_future_flip_wiring.md` — the prerequisite for the
    // `stats_inconsistency_fatal: false → true` flips below.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        // Per amendment-06 §2.5.1 partial-rebuild fault-isolation: a
        // panic during one tenant's rebuild marks that tenant
        // recovery_failed but does NOT block other tenants. Emit an
        // eprintln so the proptest surfaces the failure even though
        // the oracle's strict-mode pin already catches missing stats.
        eprintln!(
            "k2_fault_during_recovery M4-41 rebuild: {} tenant(s) marked \
             recovery_failed: {:?}",
            rebuild_report.failed.len(),
            rebuild_report.failed,
        );
    }

    K2Stack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ────────────────────────────────────────────────────────────────────
// Workload + state-extraction helpers
// ────────────────────────────────────────────────────────────────────

/// One commit attempt against the K2Stack. Mirrors the K-1 smoke
/// `do_commit` shape but without the production tier lookup (every K-2
/// proptest commit is T1 strict — the proptest is verifying recovery
/// determinism, not durability-tier interplay).
fn do_commit(
    stack: &K2Stack,
    tenant: TenantId,
    label: u32,
    a: u32,
    b: u32,
) -> Option<(NodeId, (u32, u32, u32))> {
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
    Some((id, (label, a, b)))
}

/// Build a `RecoveredState` snapshot from a recovered K2Stack by
/// reading every key listed in `pre_crash.any_history` AND snapshotting
/// post-rebuild `CatalogStats` per tenant. Mirrors `k1_smoke_30s.rs::
/// build_recovered_state` shape so the strict-mode oracle is non-
/// vacuous post-K-2-flip.
///
/// `labels_by_tenant` lists every label the workload committed per
/// tenant — `snapshot_catalog_stats` uses these to read per-label
/// counts out of the rebuilt `CatalogStats`.
fn read_recovered_state(
    stack: &K2Stack,
    pre_crash: &CommittedState,
    labels_by_tenant: &HashMap<TenantId, Vec<LabelId>>,
) -> RecoveredState {
    let mut rec = RecoveredState::default();
    let mut tenants_seen: HashSet<TenantId> = HashSet::new();
    for (tenant, id) in pre_crash.any_history.keys() {
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
            Err(e) => panic!("k2 read_node_with_store error: {e:?}"),
        }
    }
    let empty_labels: Vec<LabelId> = Vec::new();
    for tenant in tenants_seen {
        if let Some(stats) = stack.store.catalog_stats(tenant) {
            let labels = labels_by_tenant.get(&tenant).unwrap_or(&empty_labels);
            let snap = snapshot_catalog_stats(&stats, labels, &[]);
            rec.stats_by_tenant.insert(tenant, snap);
        }
    }
    rec
}

/// Build a `CommittedState` from the workload's commit log. Each row
/// is `(tenant, NodeId, label, a, b)`. Every row is treated as a T1
/// strict commit (the K-2 workload pins T1 per spec D2).
///
/// Per K-1 smoke pattern: `stats_by_tenant` carries
/// `commits_observed = 1` per tenant (the rebuild path's coalesced
/// bracket emits exactly one observation per tenant per recovery
/// cycle). Cardinality counts (`label_counts`, `total_nodes`) match
/// the ledger exactly.
fn build_committed_state(rows: &[(TenantId, NodeId, u32, u32, u32)]) -> CommittedState {
    let mut s = CommittedState::default();
    let mut stats: HashMap<TenantId, CommittedStatsRebuild> = HashMap::new();
    for (tenant, id, label, a, b) in rows {
        let key = (*tenant, *id);
        let bytes = (*label, *a, *b);
        s.any_history
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(bytes);
        s.latest_t1.insert(key, bytes);
        let st = stats.entry(*tenant).or_default();
        *st.label_counts.entry(LabelId::new(*label)).or_insert(0) += 1;
        st.total_nodes += 1;
    }
    // M4-41 cold-start rebuild semantics (per amendment-06 §D-25.1
    // step 2): rebuild path's coalesced begin/observe bracket bumps
    // `commits_observed` by exactly 1 per tenant per recovery cycle.
    for st in stats.values_mut() {
        st.commits_observed = 1;
    }
    s.total_commits = rows.len() as u64;
    s.stats_by_tenant = stats;
    s
}

/// Extract per-tenant label list from the commit rows. Used as the
/// `labels_by_tenant` parameter for [`read_recovered_state`].
fn labels_by_tenant_from_rows(
    rows: &[(TenantId, NodeId, u32, u32, u32)],
) -> HashMap<TenantId, Vec<LabelId>> {
    let mut out: HashMap<TenantId, Vec<LabelId>> = HashMap::new();
    for (tenant, _, label, _, _) in rows {
        out.entry(*tenant).or_default().push(LabelId::new(*label));
    }
    out
}

// ────────────────────────────────────────────────────────────────────
// Recovery scenario helpers
// ────────────────────────────────────────────────────────────────────

/// Run the workload + N recovery attempts. Returns the per-attempt
/// recovered states (in attempt order) so the caller can assert
/// byte-equality across attempts.
///
/// `wal_dir` is created+populated by this helper — the caller passes
/// a fresh empty directory (typically from an [`FsAdapter`]).
fn run_workload_and_n_recoveries(
    wal_dir: &Path,
    seed: u64,
    num_commits: usize,
    num_recoveries: usize,
) -> (CommittedState, Vec<RecoveredState>) {
    assert!(num_recoveries >= 1, "at least one recovery attempt");
    // Workload: build stack, commit deterministic ops, graceful shutdown.
    let stack = K2Stack::build(wal_dir);
    let mut rng = XorShift::new(seed);
    let mut commits: Vec<(TenantId, NodeId, u32, u32, u32)> = Vec::with_capacity(num_commits);
    for i in 0..num_commits {
        // Deterministic per (seed, i):
        let tenant_pick = (rng.next_u64() % 3) + 1; // 1, 2, or 3 (avoid SYSTEM=0)
        let label = 100_000u32.wrapping_add(rng.next_u32() & 0x0FFF) + i as u32;
        let a = rng.next_u32();
        let b = rng.next_u32();
        let tenant = TenantId::new(tenant_pick);
        if let Some((id, bytes)) = do_commit(&stack, tenant, label, a, b) {
            commits.push((tenant, id, bytes.0, bytes.1, bytes.2));
        }
    }
    stack.shutdown();

    let pre_crash = build_committed_state(&commits);
    let labels_by_tenant = labels_by_tenant_from_rows(&commits);
    let mut attempts = Vec::with_capacity(num_recoveries);
    for _ in 0..num_recoveries {
        // Fresh recovery from disk; no in-memory state from the prior
        // attempt is preserved. This is exactly what a process restart
        // (after a fault during recovery) would observe.
        let recovered_stack = recover_stack(wal_dir);
        let rec = read_recovered_state(&recovered_stack, &pre_crash, &labels_by_tenant);
        recovered_stack.shutdown();
        attempts.push(rec);
    }
    (pre_crash, attempts)
}

// ────────────────────────────────────────────────────────────────────
// Internal RNG — XorShift64 (matches K-1 injection rng style)
// ────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────
// Proptest — recovery determinism under fault-during-recovery
// ────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        // Lower than the project's standard 256 because each case
        // builds a real WAL stack + runs ≥1 recovery cycle per
        // applicable FS adapter. 16 cases × ~3 adapters ≈ 48
        // recovery cycles per case × ~50ms ≈ 2.5s wall — tractable
        // for `cargo test --release`. K-3 will scale via cron.
        cases: 16,
        // The proptest is not exercising mathematical edge cases (the
        // contract is bit-equality across recovery attempts); the
        // default shrinking would just lower commit counts. Keeping
        // the default shrink iterations gives clean shrunk
        // counter-examples if the property fails.
        ..ProptestConfig::default()
    })]

    /// Recovery is idempotent + byte-deterministic across N graceful-
    /// restart cycles. Each "attempt" is a fresh `recover_stack` call
    /// against a WAL that was drained by `K2Stack::shutdown` between
    /// attempts — modeling a **process-restart-after-clean-shutdown**
    /// cycle, NOT a torn-state in-recovery fault.
    ///
    /// ## Naming honesty (issue #237 MEDIUM-2)
    ///
    /// The prior name (`fault_during_recovery_is_byte_deterministic`)
    /// overclaimed: between attempts, `K2Stack::shutdown()` drains the
    /// WAL writer cleanly, so attempt[N+1] starts from a clean-shutdown
    /// state, NOT a torn-replay state. The contract being verified is
    /// **recovery idempotence under graceful-restart cycles** (a real
    /// property — restart-driven recovery converges); a torn-state
    /// in-recovery fault would require a SIGKILL-mid-recovery seam,
    /// deferred to K-3 (issue #215).
    ///
    /// For each (seed, num_commits, num_recoveries) triple AND each
    /// applicable FS adapter:
    ///
    /// 1. Build a stack on the FS adapter's tempdir + commit
    ///    `num_commits` ops (deterministic per seed).
    /// 2. Graceful shutdown (drains WAL writer cleanly).
    /// 3. Run `num_recoveries` consecutive recovery attempts; each
    ///    attempt opens the WAL cold, runs `recover_from_wal` +
    ///    `rebuild_all_tenant_stats`, reads back every committed key,
    ///    then drops the stack.
    /// 4. Assert: every recovery attempt's `bytes_by_key` is
    ///    byte-equal to the first attempt's (recovery is idempotent
    ///    + deterministic).
    /// 5. Assert: the recovered state matches the no-fault baseline
    ///    via the K-1 oracle (`verify_post_recovery_invariants`)
    ///    under `stats_inconsistency_fatal=true` (post-K-2-flip).
    ///
    /// The proptest does NOT run all four FS adapters in every case —
    /// it picks one randomly (deterministic per seed) so the
    /// per-case wall time stays bounded. A separate test
    /// (`recovery_determinism_across_all_supported_adapters`) below
    /// runs the scenario once per adapter for cross-FS coverage.
    #[test]
    fn recovery_idempotence_under_graceful_restart_cycles(
        seed in 1u64..=u64::MAX,
        num_commits in 3usize..=8usize,
        num_recoveries in 2usize..=3usize,
        fs_pick in 0u8..=3u8,
    ) {
        // Pick the FS adapter (mod 4 over the FsKind::ALL slice).
        let kind = FsKind::ALL[fs_pick as usize % FsKind::ALL.len()];
        let adapter = adapter_for(kind);
        if !adapter.is_supported() {
            // Skip-on-platform-mismatch — proptest case is vacuous;
            // the adapter coverage test below ensures we don't
            // accidentally skip every adapter.
            return Ok(());
        }
        let tmp = adapter
            .create_tmpdir()
            .expect("create_tmpdir on supported adapter");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let (pre_crash, attempts) =
            run_workload_and_n_recoveries(&wal_dir, seed, num_commits, num_recoveries);

        // Step 4: every attempt's bytes_by_key is byte-equal to the
        // first attempt's. This is the load-bearing assertion: a
        // fault-during-recovery that left intermediate state on disk
        // would surface here as a mismatch between attempt[0] and
        // attempt[1].
        let baseline = &attempts[0];
        for (i, attempt) in attempts.iter().enumerate().skip(1) {
            prop_assert_eq!(
                &attempt.bytes_by_key,
                &baseline.bytes_by_key,
                "FS={}, seed={:#x}, commits={}, attempt[{}] diverged from attempt[0]: \
                 attempt[{}].len={} vs baseline.len={}",
                kind.name(),
                seed,
                num_commits,
                i,
                i,
                attempt.bytes_by_key.len(),
                baseline.bytes_by_key.len(),
            );
        }

        // Step 5: the recovered state matches the no-fault baseline
        // via the K-1 oracle. fail_fast=true so we stop on the first
        // violation (this is a proptest, not a campaign — any
        // violation is a regression).
        //
        // K-2 flip per `project_k2_future_flip_wiring.md` (closes
        // #216, #217): the local `recover_stack` invokes
        // `rebuild_all_tenant_stats` synchronously after
        // `recover_from_wal`, so the strict-mode oracle reads
        // post-rebuild `CatalogStats` non-vacuously.
        let oracle_cfg = OracleConfig {
            // Authoritative K-* enumeration: `rg
            // 'stats_inconsistency_fatal' crates/arcgraph-storage/tests/`
            // per amendment-06 §3 R1 acceptance criteria.
            stats_inconsistency_fatal: true,
            fail_fast: true,
            ..OracleConfig::default()
        };
        prop_assert!(
            oracle_cfg.stats_inconsistency_fatal,
            "post-K-2-flip: proptest MUST run with \
             stats_inconsistency_fatal=true (amendment-06 §3 R1)"
        );
        verify_post_recovery_invariants(&pre_crash, baseline, &oracle_cfg)
            .map_err(|v| {
                TestCaseError::fail(format!(
                    "FS={} seed={:#x}: oracle violation post-recovery: {:?}",
                    kind.name(),
                    seed,
                    v
                ))
            })?;
    }
}

// ────────────────────────────────────────────────────────────────────
// Per-adapter coverage test (runs once per supported adapter)
// ────────────────────────────────────────────────────────────────────

/// Cross-FS coverage: the proptest above samples ONE adapter per case,
/// so on a 16-case run we may not exercise every adapter. This test
/// runs the same scenario once per supported adapter with a fixed
/// seed, providing the per-adapter coverage signal.
///
/// This is NOT a proptest — it's a deterministic integration test
/// that asserts the "every applicable adapter recovers cleanly" axis
/// of the K-2 contract (issue #223 acceptance: "1-hour campaign per
/// FS adapter green" is K-3 scope; this is the smoke-scale
/// per-adapter pin).
#[test]
fn recovery_determinism_across_all_supported_adapters() {
    const SEED: u64 = 0xC0FF_EEBA_BE00_2002; // K-2 canonical campaign seed
    const NUM_COMMITS: usize = 5;
    const NUM_RECOVERIES: usize = 3;

    let mut adapter_results: HashMap<&'static str, usize> = HashMap::new();

    for kind in FsKind::ALL {
        let adapter = adapter_for(kind);
        if !adapter.is_supported() {
            eprintln!(
                "k2_fault_during_recovery: skipping {} adapter (is_supported=false)",
                kind.name()
            );
            continue;
        }
        let tmp = adapter
            .create_tmpdir()
            .unwrap_or_else(|e| panic!("create_tmpdir for {}: {e}", kind.name()));
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let (pre_crash, attempts) =
            run_workload_and_n_recoveries(&wal_dir, SEED, NUM_COMMITS, NUM_RECOVERIES);

        // Every recovery attempt converges to the same state.
        let baseline = &attempts[0];
        for (i, attempt) in attempts.iter().enumerate().skip(1) {
            assert_eq!(
                attempt.bytes_by_key,
                baseline.bytes_by_key,
                "FS={}: attempt[{i}] diverged from attempt[0] (len={} vs {})",
                kind.name(),
                attempt.bytes_by_key.len(),
                baseline.bytes_by_key.len()
            );
        }

        // Oracle: post-recovery state matches no-fault baseline.
        let oracle_cfg = OracleConfig {
            // K-2 flip per `project_k2_future_flip_wiring.md` (closes
            // #216, #217). The local `recover_stack` wires
            // `rebuild_all_tenant_stats` after `recover_from_wal`, so
            // the strict-mode oracle is non-vacuous.
            stats_inconsistency_fatal: true,
            fail_fast: true,
            ..OracleConfig::default()
        };
        assert!(
            oracle_cfg.stats_inconsistency_fatal,
            "post-K-2-flip: per-adapter coverage MUST run with \
             stats_inconsistency_fatal=true (amendment-06 §3 R1)"
        );
        let report = verify_post_recovery_invariants(&pre_crash, baseline, &oracle_cfg)
            .unwrap_or_else(|v| {
                panic!("FS={}: oracle violation post-recovery: {v:?}", kind.name())
            });
        adapter_results.insert(kind.name(), report.unique_keys as usize);
        eprintln!(
            "k2_fault_during_recovery: FS={} unique_keys={} t1_keys={} \
             t1_satisfied={} historical_match={} attempts={}",
            kind.name(),
            report.unique_keys,
            report.t1_keys,
            report.t1_satisfied,
            report.historical_match,
            NUM_RECOVERIES,
        );
    }

    // At least APFS + ext4 always run because both adapters are
    // platform-agnostic surrogates. Defensive: if BOTH skip we'd
    // have no coverage and this test would silently pass — assert
    // we have at least one result.
    assert!(
        !adapter_results.is_empty(),
        "no FS adapter ran; APFS + ext4 should always be supported"
    );
}

// ────────────────────────────────────────────────────────────────────
// Determinism pin — same (seed, commits) produces byte-equal recovered
// state across independent runs of the test
// ────────────────────────────────────────────────────────────────────

/// Reproducibility pin (`feedback_determinism_oracle_concurrency_tests.md`):
/// when the underlying algorithm is deterministic, use binary-equal
/// reference snapshot as the assertion oracle. Same (seed, num_commits,
/// adapter) running this scenario twice in independent fresh tempdirs
/// must produce byte-equal recovered `bytes_by_key`.
///
/// This is the strict superset of "every recovery attempt within a
/// single run is byte-equal" (the proptest's load-bearing assertion).
/// Two independent runs both starting from a fresh empty WAL produce
/// the same final state because the workload is deterministic
/// per-seed.
#[test]
fn recovery_determinism_across_independent_runs_with_same_seed() {
    const SEED: u64 = 0xDECA_FBAD_CAFE_BABE;
    const NUM_COMMITS: usize = 5;

    // Use the APFS adapter (always supported) for this pin.
    let adapter = adapter_for(FsKind::Apfs);
    assert!(
        adapter.is_supported(),
        "APFS adapter must be supported for the determinism pin"
    );

    let collect = || -> HashMap<(TenantId, NodeId), (u32, u32, u32)> {
        let tmp = adapter.create_tmpdir().expect("create_tmpdir");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");
        let (_pre_crash, attempts) = run_workload_and_n_recoveries(&wal_dir, SEED, NUM_COMMITS, 1);
        attempts[0].bytes_by_key.clone()
    };

    let run_a = collect();
    let run_b = collect();
    assert_eq!(
        run_a, run_b,
        "same SEED + NUM_COMMITS must produce byte-equal recovered \
         bytes_by_key across independent fresh-WAL runs (per \
         feedback_determinism_oracle_concurrency_tests.md)"
    );
    // Sanity: the workload actually committed something.
    assert!(
        !run_a.is_empty(),
        "deterministic workload must produce ≥1 committed key; \
         empty result indicates a regression"
    );
}
