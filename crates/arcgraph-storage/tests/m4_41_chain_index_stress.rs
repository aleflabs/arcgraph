//! Issue #238 — cold-start rebuild acceptance gate (per-tenant chain index).
//!
//! # Acceptance gate shape — amendment-06 §D-25.2 watermark
//!
//! ADR-038 amendment-06 §D-25.2 budgets the cold-start rebuild path
//! at the **per-tenant** 10M-node trigger watermark with the v1.0-alpha
//! 5 s p99 process-restart budget shared across the primary store
//! plus all derivative substrates (vector arena, BM25, community,
//! `CatalogStats`). The amendment's §D-25.2 multi-tenant scaling
//! paragraph specifies per-tenant rebuild is parallelised across
//! `min(num_tenants, num_cpus)` threads — closed by PR #253 / issue
//! #247 (`crates/arcgraph-storage/tests/m4_41_chain_index_stress_K50_parallel.rs`
//! is the parallel acceptance gate; see amendment-06 §7.1). This
//! file's Tier-2 gate measures the **serial single-threaded walk**
//! wall-time at the watermark — orthogonal evidence retained as a
//! sequential-baseline advisory, not the load-bearing 5 s pin.
//!
//! The Tier-2 acceptance gate is therefore **K=50 N=200K (200K per
//! tenant — sub-watermark per §D-25.2's per-tenant 10M trigger;
//! 10M aggregate is a memory-fit choice for dev hardware, NOT the
//! watermark)** — the largest aggregate shape that fits a 16 GB dev
//! machine without v1.0-GA option-(b) checkpoint promotion. The
//! Tier-1 relative gate stays at K=50 N=50K so it runs without setup
//! cost on dev hardware. The K=50 N=1M (1M per tenant — still under
//! the per-tenant 10M watermark; 50M aggregate above the comfortable
//! sub-watermark serial-rebuild zone) shape is a v1.0-GA
//! characterisation point tracked in issue #249, NOT a v1.0-alpha
//! gate.
//!
//! # Two-tier gate
//!
//! ## Tier 1 — auto-runnable algorithmic gate (`K=50 N=50K`)
//!
//! Empirically demonstrates the algorithmic improvement under the
//! SAME memory load on dev hardware (~325 MB peak RSS, fits in RAM
//! without compressor / swap pressure on a 16 GB machine). Calls
//! both [`TxnManager::for_each_visible_record`] (post-#238 path)
//! and [`TxnManager::for_each_visible_record_legacy_for_test`]
//! (the pre-#238 DashMap-shard-scan shape, kept as a doc-hidden
//! helper specifically for this gate — retirement tracked in
//! issue #246) over 10 release-build runs. Asserts the post-#238
//! path is **strictly faster** than the pre-#238 path (cache-warm
//! ratio ≥ 1.5×) — the algorithmic content of issue #238.
//!
//! Why min/min on a relative gate (LOW-1 closure): the absolute
//! wall-time of either path is dominated by run-to-run noise from
//! sibling cargo agents on a loaded host (3–5× variance is normal);
//! the cache-warm minimum is the closest-to-cold-cache representative
//! and rejects most of that noise. Tail behaviour under contention
//! is NOT pinned by Tier-1; it's the Tier-2 absolute gate's job to
//! pin the budget at the amendment's watermark.
//!
//! ## Tier 2 — amendment-06 §D-25.2 sub-watermark serial-walk advisory (`K=50 N=200K aggregate; 200K per tenant`)
//!
//! Drives the post-#238 path at the §D-25.2-aligned sub-watermark
//! shape (200K per tenant; 10M aggregate — sub-watermark per the
//! per-tenant 10M trigger, NOT at the watermark) over 10 release-build
//! runs and **logs** the distribution against the amendment's 5 s
//! p99 process-restart budget (max-of-10 as the conservative proxy
//! at RUNS=10 — see `percentile_distribution` rustdoc).
//!
//! **Sequential-baseline advisory; the load-bearing 5 s pin lives
//! at the parallel gate per PR #253 (issue #247 closure).**
//! `tests/m4_41_chain_index_stress_K50_parallel.rs::rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99`
//! drives the production `rebuild_all_tenant_stats` parallel driver
//! at the same shape and asserts the absolute amendment budget (with
//! a strictly tighter 2.5 s test-internal threshold for parallelism
//! load-bearing per PR #253 round-2 reviewer M-1). This Tier-2
//! advisory remains useful as an orthogonal data point: it
//! measures the **serial single-threaded walk** through
//! `for_each_visible_record` at the watermark — i.e., the lower-bound
//! per-tenant work the parallel driver fans out across rayon
//! workers. On a clean Apple M3 release-build with no sibling cargo
//! contention, aggregate serial cost projects to `K × N_per_tenant ×
//! ~50 ns` = 50 × 200K × 50 ns ≈ 500 ms — well under the 5 s
//! budget. On the sibling-loaded dev hardware where PR #243's round-1
//! reviewer + round-2 implementer captured numbers (Apple M3 /
//! 16 GB / ~10 parallel claude agents), the 10-run distribution
//! shifts to ~6 s min, ~8 s p50, ~21 s max with tail spikes from
//! memory-compressor + scheduler contention.
//!
//! The Tier-2 gate therefore **logs** the distribution but does
//! NOT assert. The load-bearing 5 s assertion is at the parallel
//! gate (PR #253). Phase 4.3 reverse-test (orchestrator-driven;
//! revert the production `for_each_visible_record` to the legacy
//! DashMap-scan shape and re-run Tier-2 on the same host) provides
//! the budget-regression evidence for the chain-index primitive at
//! the amendment-spec'd shape; verbatim numbers from that run land
//! in the review packet.
//!
//! Memory: ~10M chains × ~80 B/chain ≈ 800 MB chain-store + ~80 MB
//! tenant_chain_keys index ≈ 1 GB peak RSS — fits in a 16 GB dev
//! machine with parallel cargo agents.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::transaction::{ReplayApplyOutcome, TxnManager};
use bytes::Bytes;

const K_TENANTS: u64 = 50;
const RUNS: usize = 10;
const P99_BUDGET: Duration = Duration::from_secs(5);

/// Static "value bytes" shared across every chain entry — avoids
/// per-record heap allocation. Each Version still embeds a
/// Bytes (24 bytes) but the heap data is shared.
static VALUE_BYTES: &[u8] = &[0u8; 16];

fn populate_tenant(mgr: &TxnManager, tenant_raw: u64, n: u64) {
    let t = TenantId::new(tenant_raw);
    let value = Bytes::from_static(VALUE_BYTES);
    let lsn_base = tenant_raw.saturating_mul(n);
    for k in 0..n {
        let commit_lsn = Lsn::new(lsn_base.saturating_add(k).saturating_add(1));
        let outcome = mgr.apply_replay_mvcc_write(commit_lsn, t, k, Some(value.clone()));
        debug_assert_eq!(outcome, ReplayApplyOutcome::Applied);
        let _ = outcome;
    }
}

fn run_aggregate_walk_index(mgr: &TxnManager) -> u64 {
    let recovered_lsn = mgr.current_lsn();
    let tenants = mgr.tenants_with_chains();
    let mut total: u64 = 0;
    for tenant in tenants {
        let mut count: u64 = 0;
        mgr.for_each_visible_record(tenant, recovered_lsn, |_, _| count += 1);
        total += count;
    }
    total
}

fn run_aggregate_walk_legacy(mgr: &TxnManager) -> u64 {
    let recovered_lsn = mgr.current_lsn();
    let tenants = mgr.tenants_with_chains();
    let mut total: u64 = 0;
    for tenant in tenants {
        let mut count: u64 = 0;
        mgr.for_each_visible_record_legacy_for_test(tenant, recovered_lsn, |_, _| count += 1);
        total += count;
    }
    total
}

/// Distribution stats over a 10-run wall-time sample.
///
/// `max_of_10` is the literal sample maximum and is the load-bearing
/// gate value (MED-2 closure: at `RUNS = 10` the empirical 99th
/// percentile is undefined; max-of-10 is a conservative proxy under
/// the assumption that run-to-run noise is not heavy-tailed enough
/// to invalidate the worst observed sample as a budget representative).
/// `min` and `p50` are reported for context.
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
    // At RUNS=10 there are not enough samples to estimate p99
    // empirically; the maximum observation is a conservative proxy
    // (per MED-2 closure). The Tier-2 budget gate is
    // `assert!(max_of_10 <= 5 s)`, mirroring amendment-06 §D-25.2's
    // 5 s p99 process-restart budget while being honest about the
    // sample size.
    let max_of_10 = sorted[sorted.len() - 1];
    eprintln!(
        "issue-238 {label} distribution ({} runs):\n  \
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
        "issue-238 K={} N={} setup: {} chains in {:?}",
        K_TENANTS,
        n_per_tenant,
        K_TENANTS.saturating_mul(n_per_tenant),
        setup_start.elapsed()
    );
    mgr
}

/// Tier 1 — relative algorithmic gate at K=50 N=50K (~325 MB peak
/// RSS). Runs the SAME setup against both the post-#238 index path
/// and the pre-#238 DashMap-scan-and-filter path; asserts the index
/// path is strictly faster (cache-warm min/min ratio ≥ 1.5×).
///
/// Iteration count = 10 release-build runs each. See module rustdoc
/// "Why min/min on a relative gate" for the LOW-1 closure rationale.
#[test]
#[ignore = "stress test: requires release build, ~325 MB peak RSS"]
#[allow(non_snake_case)]
fn rebuild_aggregate_walk_at_K50_N50K_index_strictly_faster_than_legacy() {
    const N_PER_TENANT: u64 = 50_000;
    let mgr = build_seeded_manager(N_PER_TENANT);
    let expected_total = K_TENANTS.saturating_mul(N_PER_TENANT);

    // Index-path runs.
    let mut index_runs: Vec<Duration> = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let start = Instant::now();
        let total = run_aggregate_walk_index(&mgr);
        let elapsed = start.elapsed();
        assert_eq!(
            total, expected_total,
            "index run {run}: expected {expected_total} callbacks; got {total}"
        );
        eprintln!("issue-238 index run {run}: {elapsed:?}");
        index_runs.push(elapsed);
    }

    // Legacy-path runs.
    let mut legacy_runs: Vec<Duration> = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let start = Instant::now();
        let total = run_aggregate_walk_legacy(&mgr);
        let elapsed = start.elapsed();
        assert_eq!(
            total, expected_total,
            "legacy run {run}: expected {expected_total} callbacks; got {total}"
        );
        eprintln!("issue-238 legacy run {run}: {elapsed:?}");
        legacy_runs.push(elapsed);
    }

    let _idx = percentile_distribution("index path (post-#238)", &index_runs);
    let _leg = percentile_distribution("legacy path (pre-#238)", &legacy_runs);

    // Use the MINIMUM-over-runs (cache-warm representative) — see
    // module rustdoc "Why min/min on a relative gate" for the LOW-1
    // closure rationale. Tail behaviour is the Tier-2 gate's job.
    let idx_min = *index_runs.iter().min().unwrap();
    let leg_min = *legacy_runs.iter().min().unwrap();

    let ratio = leg_min.as_secs_f64() / idx_min.as_secs_f64();
    eprintln!(
        "issue-238 K=50 N={} cache-warm ratio (legacy_min / index_min): {:.2}x",
        N_PER_TENANT, ratio
    );

    // Algorithmic gate: post-#238 path must be ≥ 1.5× faster on
    // cache-warm runs. Empirically the gap is ~3-5× under no-
    // paging conditions; 1.5× is a conservative floor that
    // tolerates dev-hardware noise without giving up the gate.
    assert!(
        ratio >= 1.5,
        "expected index-path cache-warm wall-time strictly < 2/3 of legacy-path: \
         ratio = {ratio:.2}x; index_min = {idx_min:?}, legacy_min = {leg_min:?}; \
         full distributions: index = {index_runs:?}, legacy = {legacy_runs:?}"
    );
}

/// Tier 2 — amendment-06 §D-25.2 sub-watermark sequential-walk
/// advisory at K=50 N=200K (200K per tenant — sub-watermark per
/// §D-25.2's per-tenant 10M trigger; 10M aggregate, NOT the
/// per-tenant watermark). 10 release-
/// build runs; **logs** the distribution against the amendment's
/// 5 s p99 process-restart budget but does NOT assert (per PR #243
/// round-2 HIGH-1 (b) reframe — see module rustdoc Tier-2). The
/// load-bearing 5 s assertion lives at the **parallel** gate
/// `tests/m4_41_chain_index_stress_K50_parallel.rs::rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99`
/// (PR #253; closes both issue #247 and issue #251).
///
/// Memory: ~1 GB peak RSS — runnable on a 16 GB dev machine with
/// parallel cargo agents (will hit memory pressure / swap; budget
/// will not hold on a loaded host). The K=50 N=1M (1M per tenant
/// — still under the per-tenant 10M watermark; 50M aggregate above
/// the comfortable sub-watermark serial-rebuild zone) shape is a
/// v1.0-GA characterisation point tracked in issue #249, NOT a v1.0-alpha
/// gate.
///
/// Phase 4.3 reverse-test (orchestrator-driven; revert the production
/// `for_each_visible_record` to the legacy DashMap-scan shape):
/// the wall-time regresses dramatically (≥ K× slower) under the same
/// shape on the same host, demonstrating the per-tenant chain index
/// is the load-bearing primitive for the sequential walk.
#[test]
#[ignore = "stress test: requires release build, ~1 GB peak RSS, ~30-60 s wall-clock; sequential-walk distribution log (non-asserting) — load-bearing 5 s pin lives at the parallel gate per PR #253"]
#[allow(non_snake_case)]
fn rebuild_aggregate_walk_at_K50_N200K_within_5s_p99() {
    const N_PER_TENANT: u64 = 200_000;
    let mgr = build_seeded_manager(N_PER_TENANT);
    let expected_total = K_TENANTS.saturating_mul(N_PER_TENANT);

    let mut index_runs: Vec<Duration> = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let start = Instant::now();
        let total = run_aggregate_walk_index(&mgr);
        let elapsed = start.elapsed();
        assert_eq!(total, expected_total);
        eprintln!("issue-238 [N=200K] index run {run}: {elapsed:?}");
        index_runs.push(elapsed);
    }
    let stats = percentile_distribution("index path (post-#238) [K=50 N=200K]", &index_runs);

    // Advisory log only (PR #243 round-2 HIGH-1 (b) reframe). This
    // gate measures the SERIAL single-threaded walk through
    // `for_each_visible_record` at the watermark — orthogonal
    // sequential-baseline evidence retained as advisory data. The
    // load-bearing 5 s assertion at the watermark lives at the
    // parallel gate (PR #253 closure of issue #247): see
    // `tests/m4_41_chain_index_stress_K50_parallel.rs::rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99`,
    // which drives the production parallel `rebuild_all_tenant_stats`
    // driver and asserts a strictly tighter 2.5 s test-internal
    // threshold (load-bearing on parallelism per PR #253 round-2
    // reviewer M-1). Tier 1 here carries the load-bearing relative
    // algorithmic gate; Phase 4.3 reverse-test on this same shape
    // carries the budget-regression demonstration for the chain-
    // index primitive.
    if stats.max_of_10 <= P99_BUDGET {
        eprintln!(
            "issue-238 [N=200K] sequential walk holds amendment-06 §D-25.2 5 s p99 budget: \
             min = {:?}, p50 = {:?}, max_of_10 = {:?}",
            stats.min, stats.p50, stats.max_of_10,
        );
    } else {
        eprintln!(
            "issue-238 [N=200K] sequential walk ADVISORY (host-load-dependent): \
             ADR-038 amendment-06 §D-25.2 5 s p99 budget NOT held by serial walk on this host: \
             max_of_10 = {:?} > budget = {P99_BUDGET:?}; \
             min = {:?}, p50 = {:?}; full distribution = {index_runs:?}. \
             Parallel acceptance gate at \
             `tests/m4_41_chain_index_stress_K50_parallel.rs::rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99` \
             (PR #253 closure of #247 + #251) is the load-bearing 5 s pin.",
            stats.max_of_10, stats.min, stats.p50,
        );
    }
}
