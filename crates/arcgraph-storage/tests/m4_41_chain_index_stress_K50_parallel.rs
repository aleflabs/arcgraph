//! Issue #247 — parallel cold-start rebuild acceptance gate
//! (per ADR-038 amendment-06 §D-25.2 multi-tenant scaling paragraph).
//!
//! # Acceptance gate shape — amendment-06 §D-25.2 watermark
//!
//! ADR-038 amendment-06 §D-25.2 budgets the cold-start rebuild path
//! at the **per-tenant** 10M-node trigger watermark with the v1.0-alpha
//! 5 s p99 process-restart budget shared across the primary store
//! plus all derivative substrates (vector arena, BM25, community,
//! `CatalogStats`). The amendment's §D-25.2 multi-tenant scaling
//! paragraph specifies the parallelisation contract verbatim:
//!
//! > Per-tenant rebuild is independent (per Q1/Q2/Q3 local-only
//! > checklist); a 50-tenant deployment near the watermark is
//! > parallelizable (one rebuild thread per tenant; bounded by
//! > `min(num_tenants, num_cpus)`). The watermark is per-tenant
//! > precisely to keep the per-tenant scan-time predictable under
//! > multi-tenant fanout.
//!
//! This test is the v1.0-alpha pin of that contract: K=50 tenants
//! × N=200K records (200K per tenant — sub-watermark per §D-25.2's
//! per-tenant 10M trigger; 10M aggregate is a memory-fit choice
//! for dev hardware, NOT the watermark)
//! cold-start rebuild via the parallel
//! [`rebuild_all_tenant_stats`] driver. The test-internal acceptance
//! threshold is **2.5 s** (strictly tighter than the amendment's 5 s
//! p99 budget) — see [`P99_BUDGET`] for the load-bearing rationale
//! (post PR #253 round-2 reviewer M-1: at the 5 s amendment budget
//! the unfixed serial path also passes on uncongested commodity dev
//! hardware ~4.4-4.8 s, so a slack threshold does not pin
//! parallelism). Issue #243's chain-index gate covered the SHAPE
//! (`O(K × N_per_tenant)` aggregate work post-#238); this gate adds
//! the WALL-CLOCK budget (parallelism brings the wall-time down by
//! `~min(K, num_cpus)×`).
//!
//! # Phase 4.3 reverse-test discipline (orchestrator-driven)
//!
//! Reverting the parallel driver to a serial `for tenant in tenants`
//! loop on the SAME shape produces a wall-time of ~4.4-4.8 s
//! max_of_10 on commodity dev hardware (Apple M3 / 12-core class) —
//! within the amendment's 5 s budget on uncongested hosts (so a
//! 5 s test threshold does not catch the regression on clean CI
//! hardware), but well above the test's tightened 2.5 s threshold.
//! Under PR #243's 10-sibling-cargo-load capture the serial path
//! climbed to 21 s (4.2× over the amendment budget); the tightened
//! 2.5 s gate catches the regression on any commodity host
//! independent of sibling load.
//! The reverse-test demonstrates parallelism is the load-bearing
//! primitive; the per-tenant chain index from PR #243 is a
//! necessary-but-insufficient prerequisite.
//!
//! # Memory budget
//!
//! Each chain entry holds a `Bytes` handle (24 B pointer) into a
//! shared static 64-byte `NodeRecord` value, plus a Version metadata
//! struct (~80 B). Per-tenant chain index DashSet entries (~32 B per
//! key) plus DashMap shard overhead (~10 B per key amortised) bring
//! per-record cost to ~150 B. Aggregate at 10M records: ~1.5 GB peak
//! RSS — fits a 16 GB dev machine with parallel cargo agents (will
//! hit memory pressure / swap on a loaded host; acceptance is host-
//! load-dependent).
//!
//! # Why max_of_10 vs. p99
//!
//! At RUNS=10 the empirical p99 is undefined; max-of-10 is the
//! conservative proxy under the assumption that run-to-run noise is
//! not heavy-tailed enough to invalidate the worst observed sample.
//! Mirrors the discipline established by issue #238's chain-index
//! gate (`tests/m4_41_chain_index_stress.rs::percentile_distribution`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, NodeRecord, TenantId};
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::recovery::stats_rebuild::rebuild_all_tenant_stats;
use arcgraph_storage::transaction::{ReplayApplyOutcome, TxnManager};
use bytes::Bytes;

const K_TENANTS: u64 = 50;
const N_PER_TENANT: u64 = 200_000;
const RUNS: usize = 10;
/// Test-internal acceptance threshold (tightened from the amendment's
/// 5 s p99 process-restart budget per PR #253 round-2 reviewer M-1).
///
/// **The amendment's budget is unchanged at 5 s** (ADR-038 amendment-06
/// §D-25.2). This 2.5 s gate is a strictly tighter test-internal pin
/// chosen to make the test load-bearing on the parallelism property.
/// Empirical data motivating the choice:
///
/// | Driver | min | p50 | max_of_10 |
/// |---|---|---|---|
/// | Parallel (post-#247) | ~0.91 s | ~1.02 s | ~1.36 s |
/// | Serial (pre-#247 reverse-test) | ~4.16 s | ~4.62 s | ~4.80 s |
///
/// At the amendment's 5 s budget the unfixed serial path also passes
/// on uncongested commodity dev hardware (~4.4-4.8 s, 4-12 % headroom);
/// a future refactor that silently reverts `.into_par_iter()` →
/// `.into_iter()` would NOT be caught. A 2.5 s threshold cleanly
/// distinguishes parallel-PASS from serial-FAIL on any hardware where
/// the speedup is ≥ 1.6× — converting the test to a structural pin on
/// parallelism rather than a slack absolute-budget pin. Per
/// `feedback_review_oracle_relaxations.md`: test-suite green ≠ test
/// correctness; a load-bearing oracle must distinguish the
/// before-state from the after-state under representative
/// reverse-test mutation.
const P99_BUDGET: Duration = Duration::from_millis(2_500);

/// A canonical valid `NodeRecord` encoding shared across every record
/// in the populated dataset. The rebuild's per-record decode produces
/// the same `LabelId(1)` for all 10M records — uniform load-bearing
/// cost on the SeqLock-protected counter increments. The shared
/// static lets every chain version reference the SAME 64-byte buffer
/// via `Bytes::from_static`, keeping setup memory bounded by per-key
/// metadata rather than per-record payload duplication.
fn canonical_node_bytes() -> &'static [u8; NodeRecord::SIZE] {
    use std::sync::OnceLock;
    static BYTES: OnceLock<[u8; NodeRecord::SIZE]> = OnceLock::new();
    BYTES.get_or_init(|| {
        let rec = NodeRecord::new(arcgraph_core::NodeId::new(1), LabelId::new(1), Lsn::new(1));
        rec.to_bytes()
    })
}

fn populate_tenant(mgr: &TxnManager, tenant_raw: u64, n: u64) {
    let t = TenantId::new(tenant_raw);
    let value = Bytes::from_static(canonical_node_bytes());
    let lsn_base = tenant_raw.saturating_mul(n);
    for k in 0..n {
        let commit_lsn = Lsn::new(lsn_base.saturating_add(k).saturating_add(1));
        let outcome = mgr.apply_replay_mvcc_write(commit_lsn, t, k, Some(value.clone()));
        debug_assert_eq!(outcome, ReplayApplyOutcome::Applied);
        let _ = outcome;
    }
}

fn build_seeded_manager(n_per_tenant: u64) -> Arc<TxnManager> {
    let mgr = Arc::new(TxnManager::new());
    let setup_start = Instant::now();
    let handles: Vec<_> = (0..K_TENANTS)
        .map(|raw| {
            let mgr = Arc::clone(&mgr);
            std::thread::spawn(move || populate_tenant(&mgr, raw, n_per_tenant))
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let max_lsn = Lsn::new(K_TENANTS.saturating_mul(n_per_tenant));
    mgr.seed_after_replay(max_lsn);
    eprintln!(
        "issue-247 K={} N={} setup: {} chains in {:?}",
        K_TENANTS,
        n_per_tenant,
        K_TENANTS.saturating_mul(n_per_tenant),
        setup_start.elapsed()
    );
    mgr
}

/// Distribution stats over a 10-run wall-time sample. Mirrors
/// `tests/m4_41_chain_index_stress.rs::percentile_distribution`
/// (issue #238 round-2 MED-2 closure: max_of_10 vs. p99 honesty at
/// RUNS=10).
struct DistributionStats {
    min: Duration,
    p50: Duration,
    max_of_10: Duration,
}

fn percentile_distribution(label: &str, samples: &[Duration]) -> DistributionStats {
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();
    let min = sorted[0];
    let p50 = sorted[sorted.len() / 2];
    let max_of_10 = sorted[sorted.len() - 1];
    eprintln!(
        "issue-247 {label} distribution ({} runs):\n  \
         min: {min:?}\n  \
         p50: {p50:?}\n  \
         max_of_10: {max_of_10:?}",
        samples.len()
    );
    DistributionStats {
        min,
        p50,
        max_of_10,
    }
}

/// K=50 N=200K (200K per tenant — sub-watermark per amendment-06
/// §D-25.2's per-tenant 10M trigger; 10M aggregate, NOT the
/// per-tenant watermark) parallel cold-start rebuild gate.
///
/// 10 release-build runs through the **production**
/// [`rebuild_all_tenant_stats`] driver (issue #247 parallel rayon
/// pool over per-tenant `rebuild_catalog_stats_for_tenant`); asserts
/// `max_of_10 ≤ 2.5 s` (test-internal threshold, strictly tighter
/// than the amendment's 5 s p99 process-restart budget — see
/// [`P99_BUDGET`] for the load-bearing rationale). The conservative
/// max-of-10 proxy stands in for an empirical p99 at RUNS=10 (per
/// `percentile_distribution` rustdoc).
///
/// **What this test pins.** The absolute wall-clock budget at the
/// amendment-06 §D-25.2 watermark under parallel rebuild AND that
/// parallelism is the load-bearing primitive (the tightened 2.5 s
/// threshold cleanly distinguishes parallel-PASS from serial-FAIL
/// on any hardware where the speedup is ≥ 1.6×, per reviewer M-1).
/// Closes the v1.0-alpha gap left by issue #238's PR #243 round-2
/// reframe (Tier-2 advisory log; budget exceeded under serial driver
/// on loaded host). Issue #251 tracked the clean-host re-verification
/// follow-up; this gate's pass on commodity dev hardware (with
/// parallelism reducing the wall-time by `~num_cpus×`) closes both
/// follow-ups together.
///
/// **Phase 4.3 reverse-test (orchestrator-driven).** Reverting
/// `rebuild_all_tenant_stats` to the pre-#247 serial loop (e.g.,
/// `into_iter()` instead of `into_par_iter()`) on the same host
/// produces `max_of_10 ≈ 4.4-4.8 s` on commodity dev hardware
/// (Apple M3 / 12-core class) — under the amendment's 5 s budget on
/// uncongested hosts but well above the test's tightened 2.5 s
/// threshold, so the gate FAILs as designed. Under PR #243's
/// 10-sibling-cargo-load capture: 21 s.
///
/// **What this test does NOT pin.** Stats correctness (the canonical
/// shared-bytes setup means every record decodes to the same
/// `LabelId(1)`, so per-tenant `total_node_count` = N=200K and
/// `label_cardinality(1)` = N=200K — verified as a smoke check, but
/// the load-bearing cross-key correctness invariants are pinned by
/// the in-module unit tests in `src/recovery/stats_rebuild.rs`). This
/// gate is wall-clock only; correctness is a sanity assert.
#[test]
#[ignore = "stress test: requires release build, ~1.5 GB peak RSS, ~10-30 s wall-clock"]
#[allow(non_snake_case)]
fn rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99() {
    let mgr = build_seeded_manager(N_PER_TENANT);
    let expected_per_tenant_nodes = N_PER_TENANT;
    let expected_total_nodes = K_TENANTS.saturating_mul(N_PER_TENANT);

    let mut runs: Vec<Duration> = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        // Fresh CrudStore per run so the rebuild starts from a clean
        // CatalogStats state. Mirrors a process-restart scenario
        // exactly: each restart constructs a fresh `CrudStore`, then
        // rebuilds from MVCC.
        let store = Arc::new(CrudStore::new());
        let recovered_lsn = mgr.current_lsn();

        let start = Instant::now();
        let report = rebuild_all_tenant_stats(recovered_lsn, &mgr, &store);
        let elapsed = start.elapsed();

        // Sanity: all 50 tenants succeeded; aggregate matches expected.
        assert_eq!(report.successful.len() as u64, K_TENANTS);
        assert!(report.failed.is_empty());
        assert_eq!(report.total_nodes_walked(), expected_total_nodes);
        // Per-tenant CatalogStats spot-check on tenant 0 for the
        // first run (cheap: a single DashMap lookup). The cross-key
        // invariant `sum(label_cards) == total_nodes` is the load-
        // bearing assertion; reuse the unit-test pattern.
        if run == 0 {
            let stats0 = store.catalog_stats(TenantId::new(0)).unwrap();
            let snap = stats0.snapshot();
            assert_eq!(snap.total_nodes(), Some(expected_per_tenant_nodes));
            assert_eq!(
                snap.label_card(LabelId::new(1)),
                Some(expected_per_tenant_nodes)
            );
        }

        eprintln!("issue-247 [parallel K=50 N=200K] run {run}: {elapsed:?}");
        runs.push(elapsed);
    }

    let stats = percentile_distribution("parallel K=50 N=200K", &runs);

    // Hard assertion (PR #243 round-2 HIGH-1 (b) closure direction +
    // PR #253 round-2 reviewer M-1 closure): the parallel driver MUST
    // hold a 2.5 s test-internal threshold (strictly tighter than the
    // amendment's 5 s p99 budget so a silent revert to `.into_iter()`
    // is caught on uncongested commodity hardware — see [`P99_BUDGET`]
    // for the load-bearing rationale). The conservative max_of_10
    // proxy at RUNS=10 stands in for the empirical p99.
    assert!(
        stats.max_of_10 <= P99_BUDGET,
        "issue-247 parallel K=50 N=200K: 2.5 s parallelism gate NOT held: \
         max_of_10 = {:?} > budget = {P99_BUDGET:?}; \
         min = {:?}, p50 = {:?}; full distribution = {runs:?}. \
         If this triggers under host load, rerun on an uncongested host \
         before treating it as a regression. \
         If this triggers without host load, the parallel driver may \
         have silently regressed to serial — see [`P99_BUDGET`] rustdoc.",
        stats.max_of_10,
        stats.min,
        stats.p50,
    );
}

/// Determinism pin: two consecutive parallel rebuilds against the
/// same fully-populated `TxnManager` produce a `RebuildReport` with
/// **identical `(TenantId, …)` ordering** in both `successful` and
/// `failed` lists.
///
/// This locks in the `outcomes.sort_by_key(|t| t.raw())` step in
/// [`rebuild_all_tenant_stats`]. Rayon's `collect` already preserves
/// source order (the input Vec is sorted by `tenants_with_chains()`),
/// so this is a regression pin against a future driver swap that
/// would lose source-order preservation (e.g., `for_each_with` +
/// shared `Mutex<Vec>`). Mirrors the determinism oracle pattern from
/// `feedback_determinism_oracle_concurrency_tests.md`.
///
/// **Why K=50 N=20K (1M aggregate) instead of the budget-gate shape
/// K=50 N=200K (10M aggregate; sub-watermark per-tenant):** the
/// determinism invariant is
/// structural (sort by raw `TenantId` after `collect`), not
/// scale-dependent — any K ≥ 2 with a stable input order exercises
/// the same sort-discipline assertion. The smaller shape keeps the
/// determinism unit-test runtime bounded (~150 MB peak RSS, sub-
/// second wall-time per call) so it can ride the routine
/// `--ignored` stress tier. The budget-gate shape (10M aggregate;
/// sub-watermark per-tenant) is pinned by the acceptance gate above.
#[test]
#[ignore = "stress test: requires release build, ~150 MB peak RSS"]
#[allow(non_snake_case)]
fn rebuild_all_tenant_stats_parallel_report_is_deterministic_across_runs() {
    // Smaller shape for the determinism check: K=50 N=20K = 1M
    // aggregate is sufficient for the ordering invariant; the gate
    // is structural (sort discipline), not budget.
    const N_SMALL: u64 = 20_000;
    let mgr = build_seeded_manager(N_SMALL);
    let recovered_lsn = mgr.current_lsn();

    let store_a = Arc::new(CrudStore::new());
    let store_b = Arc::new(CrudStore::new());

    let report_a = rebuild_all_tenant_stats(recovered_lsn, &mgr, &store_a);
    let report_b = rebuild_all_tenant_stats(recovered_lsn, &mgr, &store_b);

    // Tenant ordering identical in both reports.
    let order_a: Vec<u64> = report_a
        .successful
        .iter()
        .map(|(t, _, _)| t.raw())
        .collect();
    let order_b: Vec<u64> = report_b
        .successful
        .iter()
        .map(|(t, _, _)| t.raw())
        .collect();
    assert_eq!(
        order_a, order_b,
        "successful list ordering must be deterministic"
    );
    assert_eq!(report_a.failed.len(), 0);
    assert_eq!(report_b.failed.len(), 0);

    // Sorted by raw TenantId ascending.
    let sorted = {
        let mut s = order_a.clone();
        s.sort();
        s
    };
    assert_eq!(
        order_a, sorted,
        "successful list must be sorted by raw TenantId"
    );

    // Per-tenant counts identical.
    for ((t_a, n_a, r_a), (t_b, n_b, r_b)) in
        report_a.successful.iter().zip(report_b.successful.iter())
    {
        assert_eq!(t_a, t_b);
        assert_eq!(n_a, n_b);
        assert_eq!(r_a, r_b);
    }
}
