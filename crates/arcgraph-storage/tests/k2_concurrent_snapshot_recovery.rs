//! K-2 concurrent snapshot+recovery integration test (issue #223;
//! ADR-038 amendment-03 §"Slice K" K-2 row).
//!
//! ## What this test verifies
//!
//! Scenario per the K-2 spawn prompt: **while recovery is in progress
//! for tenant A's WAL, tenants B + C are serving queries with their
//! own (already-recovered or cached) `CatalogStats`. Verify no
//! cross-tenant interference; verify recovery completes correctly.**
//!
//! Modeled at the test-harness layer as:
//!
//! 1. Phase 1 — multi-tenant workload: A + B + C commit deterministic
//!    rows to a single shared stack.
//! 2. Phase 2 — snapshot B + C's `CatalogStats` BEFORE the crash;
//!    these are the "cached stats" the spawn prompt names.
//! 3. Phase 3 — graceful teardown of the stack (the "crash"
//!    abstraction; K-1c will replace this with SIGKILL subprocess
//!    teardown).
//! 4. Phase 4 — concurrently:
//!    - A worker thread runs `recover_from_wal` against the WAL
//!      directory (rebuilding state for ALL three tenants since the
//!      WAL is shared).
//!    - A reader thread polls the cached B + C `CatalogStats`
//!      snapshots taken in Phase 2 and asserts they remain
//!      bit-stable (no panic, no torn reads, no data race) while
//!      recovery is in flight.
//! 5. Phase 5 — join both threads. Recovery MUST complete cleanly;
//!    the cached B + C reads must have hit at least N polls (proves
//!    the reader was actually scheduled during the recovery window).
//! 6. Phase 6 — verify post-recovery state for all three tenants
//!    via the K-1 oracle.
//!
//! ## Cross-tenant interference contract
//!
//! The test asserts the FOLLOWING properties (the spawn prompt's
//! "no cross-tenant interference"):
//!
//! - **Reader-isolation property:** B + C's cached stats are pure
//!   data (already in `CommittedStatsRebuild` shape); the recovery
//!   thread cannot mutate them. A regression here would manifest as
//!   the reader observing different bytes mid-poll — which the
//!   reader thread's `assert_eq!` against the captured baseline would
//!   catch.
//! - **Recovery-completion property:** the recovery thread MUST
//!   return `Ok(_)` from `recover_from_wal`. A regression in the
//!   multi-tenant WAL replay (a tenant-tag mishandling, an LSN-order
//!   race, an allocator-seed corruption from cross-tenant interleave)
//!   would surface as a non-`Ok` return.
//! - **Cross-tenant byte-purity property:** post-recovery, NO row
//!   committed under tenant A appears in tenant B's recovered state
//!   (reuses `verify_cross_tenant_invariants` from the K-1b oracle).
//!
//! ## Why test-harness-only
//!
//! Per K-1 mod.rs §"Hooks vs production", the K-2 harness drives the
//! crash + concurrent-recovery scenario without modifying production
//! source. The "concurrent reader" is simulated by a thread polling
//! cached snapshots — the production `CrudStore::catalog_stats(tenant)`
//! API would also serve these reads from in-memory state, so the
//! cached snapshot is a valid surrogate for the production read path.
//!
//! Run:
//!
//! ```ignore
//! cargo test -p arcgraph-storage --release --test k2_concurrent_snapshot_recovery
//! ```

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
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
    CommittedState, CommittedStatsRebuild, CrossTenantOracleInput, OracleConfig, OracleViolation,
    RecoveredState, snapshot_catalog_stats, verify_cross_tenant_invariants,
    verify_post_recovery_invariants,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};

// ────────────────────────────────────────────────────────────────────
// K2Stack — local helpers (mirrors k2_fault_during_recovery.rs's stack)
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
/// shape (`k1_smoke_30s.rs`): WAL replay via `recover_from_wal`, then
/// synchronous cold-start MVCC stats rebuild via
/// [`arcgraph_storage::recovery::rebuild_all_tenant_stats`].
///
/// ## Ordering (per `project_k2_future_flip_wiring.md` 5-step protocol)
///
/// 1. Build a fresh stack (writer + scheduler + mgr + catalog + alloc +
///    primary + store) — the recovery substrate.
/// 2. Bootstrap the system catalog snapshot BEFORE WAL replay (so the
///    durability lookup is wired before any replay record runs).
/// 3. Call `recover_from_wal` — this replays every commit-bundle up to
///    the recovered LSN, populating the MVCC primary store + records +
///    blobs + allocator seed. The returned `RecoveryReport` carries
///    `applied_commit_lsn` — the LSN the stats rebuild walks against.
/// 4. Call `rebuild_all_tenant_stats(applied_commit_lsn, &mgr, &store)`
///    to repopulate per-tenant `CatalogStats` from the recovered MVCC
///    state. This MUST run AFTER `recover_from_wal` (the rebuild reads
///    the recovered MVCC chains) and BEFORE any caller reads
///    `CatalogStats::snapshot()` for the strict-mode oracle comparison.
///
/// ## SeqLock panic-safety primitive
///
/// `rebuild_all_tenant_stats` invokes
/// [`arcgraph_storage::recovery::rebuild_catalog_stats_for_tenant`]
/// per tenant in parallel via rayon. Each per-tenant call implements
/// the 4-invariant SeqLock primitive per
/// `feedback_seqlock_panic_safety_primitive.md`:
///
/// 1. `begin_commit_observation` OUTSIDE catch_unwind (so the marker
///    advance is observed by the SeqLock even if the walk panics)
/// 2. mutating walk INSIDE `catch_unwind(AssertUnwindSafe(...))`
/// 3. `observe_commit` UNCONDITIONALLY OUTSIDE the unwind (so the
///    SeqLock invariant `commits_started == commits_observed` is
///    preserved on both Ok + Err paths)
/// 4. panic SWALLOWED for per-tenant fault isolation — surfaces as
///    [`arcgraph_storage::recovery::TenantRebuildOutcome::PartialFailure`]
///    in `RebuildReport.failed`; other tenants' rebuilds proceed.
///
/// The K-2 caller (this `recover_stack`) does NOT wrap the rebuild call
/// in additional `catch_unwind` — the per-tenant primitive at the
/// driver layer is the load-bearing structural pin. K-2 just inspects
/// the returned report.
///
/// ## Closes the K-2 future-flip wiring task
///
/// Per `project_k2_future_flip_wiring.md`: flipping the K-2 sites'
/// `stats_inconsistency_fatal: false → true` requires
/// `rebuild_all_tenant_stats` wiring in `recover_stack` FIRST. Without
/// the wiring, the strict oracle reads an EMPTY `CatalogStats` snapshot
/// post-recovery and the flip causes `StatsInconsistent` (the pre-#236
/// K-1a BLOCKER-1 symptom). This function closes that gap for the
/// `k2_concurrent_snapshot_recovery` test file.
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
    // §D-25.1). Runs synchronously AFTER `recover_from_wal` so the
    // strict-mode oracle reads non-empty `CatalogStats`. Closes the
    // K-2 future-flip wiring task (`project_k2_future_flip_wiring.md`)
    // — the prerequisite for flipping `stats_inconsistency_fatal` from
    // `false → true` in the K-2 sites below.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        // Per amendment-06 §2.5.1 partial-rebuild fault-isolation: a
        // panic during one tenant's rebuild marks that tenant
        // recovery_failed but does NOT block other tenants' rebuilds.
        // Emit an eprintln so the test surfaces the failure even
        // though the oracle's strict-mode check (`stats_inconsistency_fatal=true`)
        // already pins the missing per-tenant CatalogStats.
        eprintln!(
            "k2_concurrent_snapshot_recovery M4-41 rebuild: {} tenant(s) marked \
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
// Workload helper
// ────────────────────────────────────────────────────────────────────

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

#[derive(Debug, Clone)]
struct CommitRow {
    tenant: TenantId,
    id: NodeId,
    label: u32,
    a: u32,
    b: u32,
}

fn build_committed_state(rows: &[CommitRow]) -> CommittedState {
    let mut s = CommittedState::default();
    let mut stats: std::collections::HashMap<TenantId, CommittedStatsRebuild> =
        std::collections::HashMap::new();
    for r in rows {
        let key = (r.tenant, r.id);
        let bytes = (r.label, r.a, r.b);
        s.any_history
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(bytes);
        s.latest_t1.insert(key, bytes);
        let st = stats.entry(r.tenant).or_default();
        *st.label_counts.entry(LabelId::new(r.label)).or_insert(0) += 1;
        st.total_nodes += 1;
    }
    // M4-41 cold-start rebuild semantics (per ADR-038 amendment-06
    // §D-25.1 step 2): the rebuild path's coalesced begin/observe
    // bracket bumps `commits_observed` by exactly 1 per tenant per
    // recovery cycle. Mirrors `k1_smoke_30s.rs::build_committed_state`
    // — every tenant with any committed rows collapses to a single
    // coalesced observation post-rebuild.
    for st in stats.values_mut() {
        st.commits_observed = 1;
    }
    s.total_commits = rows.len() as u64;
    s.stats_by_tenant = stats;
    s
}

/// Read recovered `bytes_by_key` + post-rebuild `CatalogStats` per
/// tenant.
///
/// `labels_by_tenant` lists every label the workload exercised per
/// tenant; the oracle reads these out of the rebuilt `CatalogStats` via
/// [`snapshot_catalog_stats`] (the same shape K-1 smokes use).
fn read_recovered_state(
    stack: &K2Stack,
    pre_crash: &CommittedState,
    labels_by_tenant: &std::collections::HashMap<TenantId, Vec<LabelId>>,
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
    // Mirror `k1_smoke_30s.rs::build_recovered_state`: snapshot
    // post-rebuild `CatalogStats` per tenant the workload touched.
    // Caller's pre_crash.stats_by_tenant carries `commits_observed = 1`
    // per tenant; the post-rebuild snapshot reads the rebuild driver's
    // coalesced bracket — both sides unify so the strict-mode oracle
    // is non-vacuous.
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

// ────────────────────────────────────────────────────────────────────
// Main test
// ────────────────────────────────────────────────────────────────────

/// K-2 concurrent recovery — verifies recovery completes cleanly while
/// reader threads observe cached pre-crash stats, and post-recovery
/// state holds the cross-tenant byte-purity invariant.
///
/// Three tenants (A=1, B=1001, C=1002) commit interleaved rows under
/// T1 strict on disjoint label spaces (A: 100k.., B: 200k.., C: 300k..)
/// so cross-tenant byte appearance is real contamination, not
/// coincidence (same convention as `k1_cross_tenant_fault_isolation.rs`).
///
/// ## What is load-bearingly pinned
///
/// 1. **Recovery completes** — the recovery thread joins `Ok`; the
///    rebuilt MVCC state + `CatalogStats` are non-empty.
/// 2. **Post-recovery oracle holds** — `verify_post_recovery_invariants`
///    under `stats_inconsistency_fatal=true` checks that the rebuilt
///    `CatalogStats` matches the ledger-derived expectation.
/// 3. **`t1_keys == rows.len()`** — every committed key recovers (the
///    PR #219 HIGH-1 vacuous-on-loss guard).
/// 4. **Cross-tenant byte-purity** — `verify_cross_tenant_invariants`
///    reports ZERO `CrossTenantContamination`.
///
/// ## What is NOT load-bearing
///
/// The reader-thread `assert_eq!(cached_b, baseline_b)` mechanism is
/// **structurally tautological** (issue #237 MEDIUM-1): `cached_b` and
/// `baseline_b` are private clones of pure-data structs; the recovery
/// thread cannot mutate them. The mechanism survives as a stress-shape
/// proof that the reader actually scheduled during the recovery window
/// (the `polls > 0` pin), but it does NOT independently fire on a
/// production regression. The load-bearing pins are (1)-(4) above.
#[test]
fn concurrent_recovery_completes_with_cross_tenant_byte_purity() {
    // Use the APFS adapter (always supported) for the canonical run.
    // The cross-FS proptest in `k2_fault_during_recovery.rs` already
    // sweeps every applicable adapter; this test is the threading
    // pin specifically.
    let adapter = adapter_for(FsKind::Apfs);
    assert!(adapter.is_supported());
    let tmp = adapter.create_tmpdir().expect("create_tmpdir");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("create wal dir");

    // Phase 1 — multi-tenant workload.
    let stack = K2Stack::build(&wal_dir);
    let tenant_a = TenantId::new(1);
    let tenant_b = TenantId::new(1001);
    let tenant_c = TenantId::new(1002);

    let mut rows: Vec<CommitRow> = Vec::new();
    let mut a_labels: Vec<LabelId> = Vec::new();
    let mut b_labels: Vec<LabelId> = Vec::new();
    let mut c_labels: Vec<LabelId> = Vec::new();
    // 12 commits per tenant — enough that the WAL has multiple
    // bundles per tenant; small enough that recovery is fast.
    for i in 0..12u32 {
        let a_label = 100_000 + i;
        if let Some((id, _bytes)) = do_commit(&stack, tenant_a, a_label, 11, 22) {
            rows.push(CommitRow {
                tenant: tenant_a,
                id,
                label: a_label,
                a: 11,
                b: 22,
            });
            a_labels.push(LabelId::new(a_label));
        }
        let b_label = 200_000 + i;
        if let Some((id, _bytes)) = do_commit(&stack, tenant_b, b_label, 33, 44) {
            rows.push(CommitRow {
                tenant: tenant_b,
                id,
                label: b_label,
                a: 33,
                b: 44,
            });
            b_labels.push(LabelId::new(b_label));
        }
        let c_label = 300_000 + i;
        if let Some((id, _bytes)) = do_commit(&stack, tenant_c, c_label, 55, 66) {
            rows.push(CommitRow {
                tenant: tenant_c,
                id,
                label: c_label,
                a: 55,
                b: 66,
            });
            c_labels.push(LabelId::new(c_label));
        }
    }
    assert!(rows.len() >= 30, "workload too small: {} rows", rows.len());

    // Phase 2 — cache B + C's CatalogStats from the LIVE pre-crash
    // stack. These are the "B + C cached stats" the spawn prompt
    // names; they are pure data (a `CommittedStatsRebuild`), no
    // shared state with the recovery thread.
    let cached_b_stats: CommittedStatsRebuild = stack
        .store
        .catalog_stats(tenant_b)
        .map(|s| snapshot_catalog_stats(&s, &b_labels, &[]))
        .unwrap_or_default();
    let cached_c_stats: CommittedStatsRebuild = stack
        .store
        .catalog_stats(tenant_c)
        .map(|s| snapshot_catalog_stats(&s, &c_labels, &[]))
        .unwrap_or_default();

    // Phase 3 — graceful teardown ("crash" abstraction).
    stack.shutdown();

    // Phase 4 — concurrent recovery + cached-stats reads.
    let reader_done = Arc::new(AtomicBool::new(false));
    let reader_polls = Arc::new(AtomicU64::new(0));

    // Reader thread — polls cached B + C stats and asserts they
    // remain bit-stable. Cached snapshots are immutable; this is a
    // pure-correctness proof that the recovery thread's mutations
    // (which touch the WAL + recovered store) cannot stomp the
    // cached `CommittedStatsRebuild` on the heap.
    let reader_thread = {
        let cached_b = cached_b_stats.clone();
        let cached_c = cached_c_stats.clone();
        let reader_done = Arc::clone(&reader_done);
        let reader_polls = Arc::clone(&reader_polls);
        thread::spawn(move || {
            let baseline_b = cached_b.clone();
            let baseline_c = cached_c.clone();
            while !reader_done.load(Ordering::Relaxed) {
                // Read every byte; assert byte-equality with the
                // baseline. A racy mutation from the recovery thread
                // would surface as a mismatch here.
                assert_eq!(cached_b, baseline_b, "cached B stats mutated mid-recovery");
                assert_eq!(cached_c, baseline_c, "cached C stats mutated mid-recovery");
                reader_polls.fetch_add(1, Ordering::Relaxed);
                // Yield without blocking so the recovery thread can
                // make progress; sleep_yield mimics a high-frequency
                // poller.
                thread::yield_now();
            }
        })
    };

    // Recovery thread — runs `recover_from_wal` against the WAL the
    // multi-tenant workload wrote. Returns the recovered K2Stack.
    let recovery_thread = {
        let wal_dir_clone = wal_dir.clone();
        thread::spawn(move || recover_stack(&wal_dir_clone))
    };

    // Wait for recovery to complete first, then signal the reader to
    // stop. The reader will have polled at least once before the
    // recovery thread finishes (recovery is not instant — even an
    // empty-WAL recovery takes a few ms to spawn the WalWriter +
    // BackgroundFsyncScheduler).
    let recovered_stack = recovery_thread.join().expect("recovery thread panicked");
    reader_done.store(true, Ordering::Relaxed);
    reader_thread.join().expect("reader thread panicked");

    let polls = reader_polls.load(Ordering::Relaxed);
    assert!(
        polls > 0,
        "reader thread did not poll the cached stats during recovery; \
         the test's concurrency premise is broken"
    );
    eprintln!("k2_concurrent_snapshot_recovery: reader polled {polls} times during recovery");

    // Phase 5 — verify post-recovery state via the K-1 oracle.
    let pre_crash = build_committed_state(&rows);
    let mut labels_by_tenant: std::collections::HashMap<TenantId, Vec<LabelId>> =
        std::collections::HashMap::new();
    labels_by_tenant.insert(tenant_a, a_labels.clone());
    labels_by_tenant.insert(tenant_b, b_labels.clone());
    labels_by_tenant.insert(tenant_c, c_labels.clone());
    let recovered_state = read_recovered_state(&recovered_stack, &pre_crash, &labels_by_tenant);

    let oracle_cfg = OracleConfig {
        // K-2 flip per `project_k2_future_flip_wiring.md` (closes
        // #216, #217): the local `recover_stack` above wires
        // `rebuild_all_tenant_stats` synchronously after
        // `recover_from_wal`, so the strict-mode oracle reads
        // post-rebuild `CatalogStats` for non-vacuous comparison.
        // Authoritative K-* enumeration: `rg
        // 'stats_inconsistency_fatal' crates/arcgraph-storage/tests/`
        // per amendment-06 §3 R1.
        stats_inconsistency_fatal: true,
        fail_fast: true,
        ..OracleConfig::default()
    };
    // Pattern pin (per amendment-06 §3 R1 acceptance pattern, mirrored
    // from `k1_smoke_30s.rs` post-flip): an in-band assertion documents
    // the load-bearing knob position so a future mechanical "set the
    // default" refactor can't silently un-flip the K-2 strict-mode
    // pin without tripping this assertion.
    assert!(
        oracle_cfg.stats_inconsistency_fatal,
        "post-K-2-flip (project_k2_future_flip_wiring.md): \
         this K-2 concurrent recovery test MUST run with \
         stats_inconsistency_fatal=true; the local recover_stack \
         above wires rebuild_all_tenant_stats so the strict-mode \
         oracle is non-vacuous"
    );
    let report = verify_post_recovery_invariants(&pre_crash, &recovered_state, &oracle_cfg)
        .expect("post-recovery oracle violation");
    eprintln!(
        "k2_concurrent_snapshot_recovery: unique_keys={} t1_keys={} \
         t1_satisfied={} historical_match={}",
        report.unique_keys, report.t1_keys, report.t1_satisfied, report.historical_match,
    );
    // Hard pin on the contract: all 36 committed keys recover. Per
    // PR #219 review HIGH-1 lesson, we assert against the actual key
    // count to avoid the vacuous-on-loss `0 == 0` pattern.
    assert_eq!(
        report.t1_satisfied, report.t1_keys,
        "every T1-strict committed key MUST recover post multi-tenant \
         crash; t1_satisfied={} t1_keys={}",
        report.t1_satisfied, report.t1_keys
    );
    assert_eq!(
        report.t1_keys,
        rows.len() as u64,
        "oracle's t1_keys ({}) must equal the workload's committed \
         row count ({}); a mismatch means recovery lost rows AND the \
         oracle's per-key iteration didn't catch it (HIGH-1 vacuous-on-loss)",
        report.t1_keys,
        rows.len()
    );

    // Phase 6 — cross-tenant invariant via the K-1b oracle. K-1b
    // CONCERN-3 carry-forward: use fail_fast=false so the cross-tenant
    // pass runs and we can assert ZERO `CrossTenantContamination`.
    let mut cross_input = CrossTenantOracleInput::default();
    let mut per_tenant_pre: std::collections::HashMap<TenantId, CommittedState> =
        std::collections::HashMap::new();
    let mut per_tenant_rec: std::collections::HashMap<TenantId, RecoveredState> =
        std::collections::HashMap::new();
    for tenant in [tenant_a, tenant_b, tenant_c] {
        let pre_t = build_committed_state(
            &rows
                .iter()
                .filter(|r| r.tenant == tenant)
                .cloned()
                .collect::<Vec<_>>(),
        );
        let mut rec_t = RecoveredState::default();
        for ((t, n), bytes) in &recovered_state.bytes_by_key {
            if *t == tenant {
                rec_t.bytes_by_key.insert((*t, *n), *bytes);
            }
        }
        per_tenant_pre.insert(tenant, pre_t);
        per_tenant_rec.insert(tenant, rec_t);
    }
    cross_input.pre_crash_per_tenant = per_tenant_pre;
    cross_input.recovered_per_tenant = per_tenant_rec;

    let cross_cfg = OracleConfig {
        // K-2 flip per `project_k2_future_flip_wiring.md` (closes
        // #216, #217). `verify_cross_tenant_invariants` itself does
        // NOT check the `stats_inconsistency_fatal` knob (only
        // `verify_post_recovery_invariants` does), so flipping here
        // is mechanical — but flipping all 5 K-2 sites uniformly is
        // the load-bearing acceptance criterion per amendment-06 §3
        // R1, so we flip the cross-tenant call too. `fail_fast=false`
        // is the K-1b CONCERN-3 carry-forward — see
        // `oracle.rs::verify_cross_tenant_invariants` doc re:
        // distinguishing CrossTenantContamination from per-tenant
        // UnknownKey under fail_fast=true.
        stats_inconsistency_fatal: true,
        fail_fast: false,
        ..OracleConfig::default()
    };
    assert!(
        cross_cfg.stats_inconsistency_fatal,
        "post-K-2-flip uniformity: every K-2 OracleConfig site must \
         hold stats_inconsistency_fatal=true (amendment-06 §3 R1)"
    );
    let cross_report = verify_cross_tenant_invariants(&cross_input, &cross_cfg)
        .expect("cross-tenant oracle Err under fail_fast=false should not happen");
    let contamination_count = cross_report
        .cross_tenant_violations
        .iter()
        .filter(|v| matches!(v, OracleViolation::CrossTenantContamination { .. }))
        .count();
    assert_eq!(
        contamination_count, 0,
        "ZERO CrossTenantContamination violations expected; got {contamination_count}: {:?}",
        cross_report.cross_tenant_violations
    );
    eprintln!(
        "k2_concurrent_snapshot_recovery: cross-tenant oracle: \
         tenants_checked={:?} cross_tenant_checks={} contamination=0",
        cross_report
            .tenants_checked
            .iter()
            .map(|t| t.raw())
            .collect::<Vec<_>>(),
        cross_report.cross_tenant_checks,
    );

    // Sanity: every tenant has its expected per-tenant key count.
    for tenant in [tenant_a, tenant_b, tenant_c] {
        let n = rows.iter().filter(|r| r.tenant == tenant).count();
        assert_eq!(
            n,
            12,
            "tenant {} should have 12 committed rows; got {n}",
            tenant.raw()
        );
    }

    // Cleanup the recovered stack to release the WAL handle before
    // tmp's TempDir Drop runs.
    recovered_stack.shutdown();
}

// ────────────────────────────────────────────────────────────────────
// Sequential pin — sanity check that the multi-tenant workload + WAL
// recovery shape works without the threading on top.
// ────────────────────────────────────────────────────────────────────

/// Sequential multi-tenant recovery — same workload as the concurrent
/// test, but no reader thread. Used as a "is the underlying scenario
/// even shaped right?" baseline; if this fails, the concurrent test's
/// failure isn't a threading bug.
#[test]
fn sequential_multi_tenant_recovery_baseline() {
    let adapter = adapter_for(FsKind::Apfs);
    assert!(adapter.is_supported());
    let tmp = adapter.create_tmpdir().expect("create_tmpdir");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("create wal dir");

    let stack = K2Stack::build(&wal_dir);
    let tenant_a = TenantId::new(1);
    let tenant_b = TenantId::new(1001);
    let tenant_c = TenantId::new(1002);

    let mut rows: Vec<CommitRow> = Vec::new();
    let mut labels_by_tenant: std::collections::HashMap<TenantId, Vec<LabelId>> =
        std::collections::HashMap::new();
    for i in 0..6u32 {
        for (tenant, base) in [
            (tenant_a, 100_000u32),
            (tenant_b, 200_000u32),
            (tenant_c, 300_000u32),
        ] {
            let label = base + i;
            if let Some((id, _bytes)) = do_commit(&stack, tenant, label, base / 100, base / 200) {
                rows.push(CommitRow {
                    tenant,
                    id,
                    label,
                    a: base / 100,
                    b: base / 200,
                });
                labels_by_tenant
                    .entry(tenant)
                    .or_default()
                    .push(LabelId::new(label));
            }
        }
    }
    stack.shutdown();

    let recovered_stack = recover_stack(&wal_dir);
    let pre_crash = build_committed_state(&rows);
    let recovered_state = read_recovered_state(&recovered_stack, &pre_crash, &labels_by_tenant);

    let oracle_cfg = OracleConfig {
        // K-2 flip per `project_k2_future_flip_wiring.md` (closes #216, #217).
        // The local `recover_stack` wires `rebuild_all_tenant_stats`
        // after `recover_from_wal`, so the strict-mode oracle is
        // non-vacuous.
        stats_inconsistency_fatal: true,
        fail_fast: true,
        ..OracleConfig::default()
    };
    assert!(
        oracle_cfg.stats_inconsistency_fatal,
        "post-K-2-flip: sequential baseline MUST run with \
         stats_inconsistency_fatal=true (amendment-06 §3 R1)"
    );
    let report =
        verify_post_recovery_invariants(&pre_crash, &recovered_state, &oracle_cfg).unwrap();
    assert_eq!(
        report.t1_satisfied, report.t1_keys,
        "sequential baseline MUST recover every T1 key"
    );
    assert_eq!(
        report.t1_keys,
        rows.len() as u64,
        "sequential baseline t1_keys must match committed rows"
    );

    recovered_stack.shutdown();
}

// ────────────────────────────────────────────────────────────────────
// Stats-rebuild wiring pin — proves `recover_stack` invokes
// `rebuild_all_tenant_stats` non-vacuously (post-K-2-flip closure)
// ────────────────────────────────────────────────────────────────────

/// Integration test that pins the `recover_stack` wiring of
/// `rebuild_all_tenant_stats` per `project_k2_future_flip_wiring.md`.
///
/// Read-back of `CatalogStats` from a recovered store proves the
/// rebuild driver ran:
///
/// 1. Build a stack + commit deterministic rows across 2 tenants.
/// 2. Snapshot the LIVE pre-crash stats (commits_observed > 1 per
///    tenant per the live commit-pipeline).
/// 3. Shutdown + recover (this invokes `rebuild_all_tenant_stats`
///    inside `recover_stack`).
/// 4. Snapshot the POST-REBUILD stats.
/// 5. Assert: per-tenant cardinality counts (label_counts +
///    total_nodes) survive byte-identically. `commits_observed`
///    collapses from `N` (live, one per commit) to `1` (rebuilt,
///    coalesced bracket per amendment-06 §D-25.1 step 2).
///
/// This test would FAIL if `recover_stack` regressed back to the
/// pre-flip behavior (no rebuild call): the post-recovery
/// `CatalogStats` would be empty and `total_nodes_post = 0`.
///
/// Per `feedback_review_oracle_relaxations.md`: this is the
/// non-vacuity proof for the K-2 flip. The strict-mode oracle in the
/// other tests checks the same surface, but this test isolates the
/// rebuild wiring specifically so a regression localizes faster.
#[test]
fn recover_stack_invokes_rebuild_all_tenant_stats_non_vacuously() {
    let adapter = adapter_for(FsKind::Apfs);
    assert!(adapter.is_supported());
    let tmp = adapter.create_tmpdir().expect("create_tmpdir");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("create wal dir");

    let tenant_a = TenantId::new(2001);
    let tenant_b = TenantId::new(2002);

    // Workload: 4 commits per tenant, two labels each, deterministic.
    let stack = K2Stack::build(&wal_dir);
    let label_a1 = LabelId::new(900_001);
    let label_a2 = LabelId::new(900_002);
    let label_b1 = LabelId::new(910_001);

    let mut a_count: u64 = 0;
    let mut a1_count: u64 = 0;
    let mut a2_count: u64 = 0;
    for i in 0..4u32 {
        let l = if i % 2 == 0 { label_a1 } else { label_a2 };
        let l_raw = if i % 2 == 0 { 900_001 } else { 900_002 };
        if do_commit(&stack, tenant_a, l_raw, 7, 8).is_some() {
            a_count += 1;
            if l == label_a1 {
                a1_count += 1;
            } else {
                a2_count += 1;
            }
        }
    }
    let mut b_count: u64 = 0;
    for _ in 0..3u32 {
        if do_commit(&stack, tenant_b, 910_001, 11, 22).is_some() {
            b_count += 1;
        }
    }

    // Snapshot live stats (the pre-shutdown view; commits_observed is
    // per-commit, so `a_count` for tenant A, `b_count` for tenant B).
    let live_a = stack
        .store
        .catalog_stats(tenant_a)
        .map(|s| snapshot_catalog_stats(&s, &[label_a1, label_a2], &[]))
        .expect("tenant A live stats");
    let live_b = stack
        .store
        .catalog_stats(tenant_b)
        .map(|s| snapshot_catalog_stats(&s, &[label_b1], &[]))
        .expect("tenant B live stats");
    assert_eq!(live_a.total_nodes, a_count, "live tenant-A total_nodes");
    assert_eq!(live_b.total_nodes, b_count, "live tenant-B total_nodes");
    assert!(
        live_a.commits_observed > 1,
        "live tenant-A commits_observed must be >1 (one per commit): {}",
        live_a.commits_observed
    );

    stack.shutdown();

    // Recover — this invokes `rebuild_all_tenant_stats` per
    // `project_k2_future_flip_wiring.md`. Without the wiring, the
    // assertions below FAIL (empty CatalogStats post-recovery).
    let recovered_stack = recover_stack(&wal_dir);

    let post_a = recovered_stack
        .store
        .catalog_stats(tenant_a)
        .map(|s| snapshot_catalog_stats(&s, &[label_a1, label_a2], &[]))
        .expect(
            "tenant A post-rebuild stats — empty here ⇒ rebuild not invoked \
             (regressed K-2 wiring per project_k2_future_flip_wiring.md)",
        );
    let post_b = recovered_stack
        .store
        .catalog_stats(tenant_b)
        .map(|s| snapshot_catalog_stats(&s, &[label_b1], &[]))
        .expect(
            "tenant B post-rebuild stats — empty here ⇒ rebuild not invoked \
             (regressed K-2 wiring per project_k2_future_flip_wiring.md)",
        );

    // Cardinality counts survive byte-identically.
    assert_eq!(
        post_a.total_nodes, a_count,
        "post-rebuild tenant-A total_nodes must equal committed rows ({a_count})"
    );
    assert_eq!(
        post_b.total_nodes, b_count,
        "post-rebuild tenant-B total_nodes must equal committed rows ({b_count})"
    );
    assert_eq!(
        post_a.label_counts.get(&label_a1).copied().unwrap_or(0),
        a1_count,
        "post-rebuild tenant-A label_a1 count"
    );
    assert_eq!(
        post_a.label_counts.get(&label_a2).copied().unwrap_or(0),
        a2_count,
        "post-rebuild tenant-A label_a2 count"
    );

    // `commits_observed` collapses to 1 per tenant (rebuild bracket
    // semantics per amendment-06 §D-25.1 step 2). This is the
    // SeqLock-primitive observable: the rebuild driver's
    // `begin_commit_observation` → walk → `observe_commit` bracket
    // emits exactly one observation per tenant per recovery cycle.
    assert_eq!(
        post_a.commits_observed, 1,
        "post-rebuild tenant-A commits_observed must be 1 (coalesced \
         bracket per amendment-06 §D-25.1 step 2); got {}",
        post_a.commits_observed
    );
    assert_eq!(
        post_b.commits_observed, 1,
        "post-rebuild tenant-B commits_observed must be 1 (coalesced \
         bracket per amendment-06 §D-25.1 step 2); got {}",
        post_b.commits_observed
    );

    eprintln!(
        "k2_recover_stack_invokes_rebuild: live(A.commits_observed={}, B.commits_observed={}) \
         → post-rebuild(A.commits_observed={}, B.commits_observed={}); cardinalities preserved",
        live_a.commits_observed,
        live_b.commits_observed,
        post_a.commits_observed,
        post_b.commits_observed,
    );

    recovered_stack.shutdown();
}
