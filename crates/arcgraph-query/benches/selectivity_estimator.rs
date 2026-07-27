//! Criterion benchmark for `SelectivityEstimator` per-call latency.
//!
//! Per codex M4-2x retro Sin #1 + testing strategy ("Every public
//! performance-sensitive API has a benchmark"): the M4-05 cost-based
//! planner is the load-bearing consumer; per-call budget is **p99 ≤
//! 50ns** at a 1M-row tenant (per `selectivity.rs` BUDGET header).
//!
//! This bench establishes the v1.0 baseline so future stats-redesign
//! regressions >2× are caught at the bench boundary, not at M4-05
//! integration time. Each bench fn measures one estimator call against
//! a `StubCatalogProvider` populated with realistic ~1M-row stats and
//! 100 distinct labels / 50 distinct rel-types — close enough to the
//! LDBC SNB SF-1 Person-graph shape to make the numbers meaningful
//! without paying for a real catalog wire-up.
//!
//! Run: `cargo bench -p arcgraph-query --bench selectivity_estimator`.

use arcgraph_core::{LabelId, TypeId};
use arcgraph_query::semantic::bound_ast::BindingId;
use arcgraph_query::semantic::{SelectivityEstimator, StubCatalogProvider};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Build a representative ~1M-row stub catalog with 100 labels and
/// 50 rel-types. Selectivities range across the realistic spectrum
/// (rare label = 1k cardinality, common label = 250k cardinality).
fn build_stub() -> StubCatalogProvider {
    let mut stub = StubCatalogProvider::new()
        .with_total_node_count(1_000_000)
        .with_total_rel_count(5_000_000);
    for i in 0..100u32 {
        // Spread label cardinalities geometrically so a typical
        // estimate_label call exercises the divide path with varied
        // operand magnitudes (catches naïve compiler-folding bench
        // false-positives).
        let card = 1_000 + u64::from(i) * 2_500;
        stub = stub.with_label_cardinality(LabelId::new(i + 1), card);
    }
    for i in 0..50u32 {
        let card = 5_000 + u64::from(i) * 5_000;
        stub = stub.with_rel_type_cardinality(TypeId::new(i + 1), card);
    }
    stub
}

fn bench_estimate_eq(c: &mut Criterion) {
    let cat = build_stub();
    let est = SelectivityEstimator::new(&cat);
    c.bench_function("SelectivityEstimator::estimate_eq", |b| {
        b.iter(|| {
            black_box(est.estimate_eq(black_box(BindingId::new(0)), black_box(None)));
        });
    });
}

fn bench_estimate_lt(c: &mut Criterion) {
    let cat = build_stub();
    let est = SelectivityEstimator::new(&cat);
    c.bench_function("SelectivityEstimator::estimate_lt", |b| {
        b.iter(|| {
            black_box(est.estimate_lt(black_box(BindingId::new(0)), black_box(None)));
        });
    });
}

fn bench_estimate_in(c: &mut Criterion) {
    let cat = build_stub();
    let est = SelectivityEstimator::new(&cat);
    c.bench_function("SelectivityEstimator::estimate_in", |b| {
        b.iter(|| {
            black_box(est.estimate_in(black_box(BindingId::new(0)), black_box(None), black_box(8)));
        });
    });
}

fn bench_estimate_label(c: &mut Criterion) {
    let cat = build_stub();
    let est = SelectivityEstimator::new(&cat);
    // Cycle through the 100 known labels so the divide operands vary
    // call-to-call; defeats trivial constant-folding the compiler
    // might attempt on a single-label hot path.
    let labels: Vec<LabelId> = (1..=100u32).map(LabelId::new).collect();
    let mut idx = 0usize;
    c.bench_function("SelectivityEstimator::estimate_label", |b| {
        b.iter(|| {
            let label = labels[idx % labels.len()];
            idx = idx.wrapping_add(1);
            black_box(est.estimate_label(black_box(label)));
        });
    });
}

fn bench_estimate_rel_type(c: &mut Criterion) {
    let cat = build_stub();
    let est = SelectivityEstimator::new(&cat);
    let types: Vec<TypeId> = (1..=50u32).map(TypeId::new).collect();
    let mut idx = 0usize;
    c.bench_function("SelectivityEstimator::estimate_rel_type", |b| {
        b.iter(|| {
            let rel_type = types[idx % types.len()];
            idx = idx.wrapping_add(1);
            black_box(est.estimate_rel_type(black_box(rel_type)));
        });
    });
}

criterion_group!(
    benches,
    bench_estimate_eq,
    bench_estimate_lt,
    bench_estimate_in,
    bench_estimate_label,
    bench_estimate_rel_type,
);
criterion_main!(benches);
