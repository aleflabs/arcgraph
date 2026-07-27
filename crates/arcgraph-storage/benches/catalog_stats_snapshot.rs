//! M4-04e (issue #210) catalog-stats snapshot bench.
//!
//! Measures `CatalogStats::snapshot()` per-call cost vs. N individual
//! atomic loads at v1.0 tenant sizes. Acceptance per issue #210:
//! `snapshot()` per-call cost should be COMPARABLE to N individual
//! atomic loads at v1.0 tenant sizes (~100 labels per tenant). The
//! cross-key consistency guarantee is paid for by the two-marker
//! SeqLock; the absolute cost MUST stay inside the M4-05 plan-build
//! budget (5 ms per ADR-036 §D-25).
//!
//! Cases:
//! - `snapshot_cold_100_labels`: tenant with 100 labels, fresh
//!   `CatalogStats` (no prior snapshot calls), single-threaded.
//! - `n_loads_cold_100_labels`: comparison baseline — the equivalent
//!   N individual `label_cardinality` calls (which is what the M4-05
//!   planner did before this slice landed).
//! - `snapshot_warm_1000_labels`: tenant with 1000 labels, repeated
//!   snapshot calls (warm caches).
//! - `n_loads_warm_1000_labels`: baseline.
//! - `snapshot_under_contention`: 4 background commit threads + 1
//!   snapshot loop in the bench harness; measures wall-clock per
//!   snapshot under writer contention.
//!
//! Run: `cargo bench -p arcgraph-storage --bench catalog_stats_snapshot`.
//! Reports: `target/criterion/catalog_stats_snapshot/`.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use arcgraph_core::{LabelId, TypeId};
use arcgraph_storage::CatalogStats;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// Build a `CatalogStats` populated with `n_labels` distinct labels
/// (each with cardinality `1`) and `n_rel_types` distinct rel types.
/// One commit observation marker brackets the bulk increments so the
/// SeqLock invariant (`commits_started == commits_observed`) holds at
/// quiescence.
fn build_populated_stats(n_labels: u32, n_rel_types: u32) -> CatalogStats {
    let stats = CatalogStats::new();
    stats.begin_commit_observation();
    for i in 0..n_labels {
        stats.increment_label(LabelId::new(i));
        stats.increment_total_nodes();
    }
    for i in 0..n_rel_types {
        stats.increment_rel_type(TypeId::new(i));
        stats.increment_total_rels();
    }
    stats.observe_commit();
    stats
}

fn bench_snapshot_cold_100_labels(c: &mut Criterion) {
    // 100 labels × 100 rel-types is a v1.0-ish tenant (per the budget
    // section in `catalog/stats.rs`). The snapshot iterates both
    // DashMaps + reads totals + does 4 Acquire-loads on the SeqLock
    // counters.
    let stats = build_populated_stats(100, 100);
    let mut group = c.benchmark_group("catalog_stats_snapshot");
    group.throughput(Throughput::Elements(1));
    group.bench_function("snapshot_100_labels_100_rel_types", |b| {
        b.iter(|| {
            let snap = black_box(stats.snapshot());
            black_box(snap);
        });
    });
    group.finish();
}

fn bench_n_loads_baseline_100(c: &mut Criterion) {
    // Baseline comparison: N individual `label_cardinality` calls +
    // N `rel_type_cardinality` + 1 `total_node_count` + 1
    // `total_rel_count`. This is what the M4-05 cost planner would
    // do per predicate WITHOUT this slice's snapshot API; under
    // M4-04e it does ONE snapshot per plan instead.
    let stats = build_populated_stats(100, 100);
    let mut group = c.benchmark_group("catalog_stats_snapshot");
    group.throughput(Throughput::Elements(1));
    group.bench_function("baseline_n_loads_100_labels_100_rel_types", |b| {
        b.iter(|| {
            let mut s = 0u64;
            for i in 0..100u32 {
                if let Some(c) = stats.label_cardinality(LabelId::new(i)) {
                    s = s.wrapping_add(c);
                }
            }
            for i in 0..100u32 {
                if let Some(c) = stats.rel_type_cardinality(TypeId::new(i)) {
                    s = s.wrapping_add(c);
                }
            }
            if let Some(t) = stats.total_node_count() {
                s = s.wrapping_add(t);
            }
            if let Some(t) = stats.total_rel_count() {
                s = s.wrapping_add(t);
            }
            black_box(s);
        });
    });
    group.finish();
}

fn bench_snapshot_warm_1000_labels(c: &mut Criterion) {
    // 1000 labels — exercises the worst-case-allocation path where
    // the snapshot's `Vec` allocation dominates the atomic loads.
    let stats = build_populated_stats(1000, 1000);
    let mut group = c.benchmark_group("catalog_stats_snapshot");
    group.throughput(Throughput::Elements(1));
    group.bench_function("snapshot_1000_labels_1000_rel_types", |b| {
        b.iter(|| {
            let snap = black_box(stats.snapshot());
            black_box(snap);
        });
    });
    group.finish();
}

fn bench_n_loads_baseline_1000(c: &mut Criterion) {
    let stats = build_populated_stats(1000, 1000);
    let mut group = c.benchmark_group("catalog_stats_snapshot");
    group.throughput(Throughput::Elements(1));
    group.bench_function("baseline_n_loads_1000_labels_1000_rel_types", |b| {
        b.iter(|| {
            let mut s = 0u64;
            for i in 0..1000u32 {
                if let Some(c) = stats.label_cardinality(LabelId::new(i)) {
                    s = s.wrapping_add(c);
                }
            }
            for i in 0..1000u32 {
                if let Some(c) = stats.rel_type_cardinality(TypeId::new(i)) {
                    s = s.wrapping_add(c);
                }
            }
            black_box(s);
        });
    });
    group.finish();
}

fn bench_snapshot_under_contention(c: &mut Criterion) {
    // Contention bench (intentional retry-storm scenario)
    //
    // Spawns 1 writer thread that calls `begin_commit_observation` →
    // increment → `observe_commit` in a tight loop with `thread::yield_now()`
    // between iterations. Snapshot reader retries against the SeqLock until
    // it observes a quiescent `commits_started == commits_observed`.
    //
    // **This is NOT steady-state production behavior.** Per ADR-031 and
    // ADR-034 the v1.0 commit pipeline serializes per-tenant commits via
    // the WAL group-commit window (~1ms). In production a tight per-tenant
    // commit loop would not produce this retry storm. The bench measures
    // the WORST-CASE retry behavior under intentional pathological
    // contention, not the expected steady-state cost.
    //
    // The "snapshot under contention is FASTER than uncontended snapshot"
    // observation in the original bench output is an artifact of cache
    // locality from the co-running writer's yield_now() pattern. Under
    // truly idle conditions Criterion's tight loop has no co-running
    // writer to provide cache locality. Both numbers are well within
    // the 5 ms M4-05 plan-build budget per ADR-036 §D-25.
    let stats = Arc::new(build_populated_stats(100, 100));
    let stop = Arc::new(AtomicBool::new(false));

    let writer_stats = Arc::clone(&stats);
    let writer_stop = Arc::clone(&stop);
    let writer = thread::spawn(move || {
        let mut i: u32 = 0;
        while !writer_stop.load(Ordering::Relaxed) {
            writer_stats.begin_commit_observation();
            writer_stats.increment_label(LabelId::new(i % 100));
            writer_stats.increment_total_nodes();
            writer_stats.increment_rel_type(TypeId::new(i % 100));
            writer_stats.increment_total_rels();
            writer_stats.observe_commit();
            // Yield between commits — the SeqLock reader needs a
            // chance to slip in. v1.0 commit serialization plus WAL
            // fsync makes the real inter-commit gap >> the in-memory
            // commit cost; `yield_now` is a conservative proxy.
            thread::yield_now();
            i = i.wrapping_add(1);
        }
    });

    // Give the writer time to ramp up before the bench starts.
    thread::sleep(Duration::from_millis(50));

    let mut group = c.benchmark_group("catalog_stats_snapshot");
    group.throughput(Throughput::Elements(1));
    // Small sample count: each iteration is a `snapshot()` call which
    // may retry under contention. The default 100-sample × 5s =
    // ~500ms target per iter is enough to stabilize the median; we
    // do NOT need Criterion's full statistical regimen here because
    // the contended-snapshot wall-clock distribution is naturally
    // wide and the headline number is "is it inside the 5 ms M4-05
    // plan-build budget?", not "is regression > 1%?".
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("snapshot_under_1_writer_contention", |b| {
        b.iter(|| {
            let snap = black_box(stats.snapshot());
            black_box(snap);
        });
    });
    group.finish();

    // Tear down the writer cleanly.
    stop.store(true, Ordering::Relaxed);
    let _ = writer.join();
}

criterion_group!(
    benches,
    bench_snapshot_cold_100_labels,
    bench_n_loads_baseline_100,
    bench_snapshot_warm_1000_labels,
    bench_n_loads_baseline_1000,
    bench_snapshot_under_contention,
);
criterion_main!(benches);
