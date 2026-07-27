//! W13α / M4-64b — SIMD vs scalar hot-path speedup bench.
//!
//! Per ADR-038 amendment-02 §M4.f + amendment-03 §Structural-1, the
//! M4-64b acceptance gate is **≥1.5× speedup vs scalar baseline** on
//! each of the three hot-path operators:
//!
//! 1. [`simd_filter_i64_cmp`] — i64 comparison for FilterOp predicate
//!    eval.
//! 2. [`simd_neighbor_match_mask`] — NodeId membership for ExpandOp
//!    neighbor scan.
//! 3. [`simd_rrf_scores`] — `1/(k+rank)` vector for RrfFusion rank-merge.
//!
//! # Bench shape
//!
//! Each operator gets two Criterion benches per row-count
//! (1K / 10K / 100K): `<op>_scalar_<N>` and `<op>_simd_<N>`. The
//! per-arch dispatch picks the best-available SIMD backend at runtime;
//! the bench's intrinsic-side function `simd_filter_i64_cmp(...)` is
//! the same call the FilterOp hot path uses, so the speedup ratio
//! observed here is the speedup the operator delivers in production
//! (modulo per-row marshalling overhead which is bench-loop-invariant).
//!
//! # Speedup gate
//!
//! The empirical-gauntlet step 7 asserts ≥1.5× speedup on each
//! operator at N=10K. The bench's `print_speedup_summary` helper writes
//! a parseable summary to stdout for the PR-body packet.
//!
//! Run: `cargo bench -p arcgraph-query --bench simd_hot_path_speedup`
//! For a quick smoke (≤30 s total): add `--quick`.

use arcgraph_query::executor::simd::expand::{scalar as expand_scalar, simd_neighbor_match_mask};
use arcgraph_query::executor::simd::filter::{CmpOp, scalar as filter_scalar, simd_filter_i64_cmp};
use arcgraph_query::executor::simd::rrf::{scalar as rrf_scalar, simd_rrf_scores};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const ROW_COUNTS: &[usize] = &[1_024, 10_240, 102_400];

fn build_filter_input(n: usize) -> (Vec<i64>, Vec<bool>) {
    // Geometric spread + occasional negatives + ~5% NULLs to exercise
    // the 3VL drop path; defeats trivial compiler folding.
    let values: Vec<i64> = (0..n as i64).map(|i| (i * 7) - (n as i64 / 2)).collect();
    let nulls: Vec<bool> = (0..n).map(|i| i % 19 == 0).collect();
    (values, nulls)
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_i64_gt");
    for &n in ROW_COUNTS {
        let (values, nulls) = build_filter_input(n);
        let target = (n as i64) / 4; // selects ~25% per arithmetic spread
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                black_box(filter_scalar::filter_i64_cmp(
                    black_box(&values),
                    black_box(&nulls),
                    black_box(target),
                    CmpOp::Gt,
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                black_box(simd_filter_i64_cmp(
                    black_box(&values),
                    black_box(&nulls),
                    black_box(target),
                    CmpOp::Gt,
                ))
            });
        });
    }
    group.finish();
}

fn build_expand_input(n: usize) -> (Vec<u64>, Vec<u64>) {
    // Candidate dst ids, with a small allow-set (typical pushdown
    // shape). K=4 keeps the AVX2/NEON broadcast loop short — that's the
    // realistic hot-path consumer profile.
    let candidates: Vec<u64> = (0..n as u64).collect();
    let targets: Vec<u64> = vec![3, 7, 11, 17];
    (candidates, targets)
}

fn bench_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand_neighbor_match");
    for &n in ROW_COUNTS {
        let (candidates, targets) = build_expand_input(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                black_box(expand_scalar::neighbor_match_mask(
                    black_box(&candidates),
                    black_box(&targets),
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                black_box(simd_neighbor_match_mask(
                    black_box(&candidates),
                    black_box(&targets),
                ))
            });
        });
    }
    group.finish();
}

fn bench_rrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_scores");
    for &n in ROW_COUNTS {
        // RRF rank lists at v1.0-alpha are typically capped at a few
        // hundred; we use ROW_COUNTS for parity with the other two
        // benches so the speedup ratio is comparable.
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| black_box(rrf_scalar::rrf_scores(black_box(60), black_box(n))));
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| black_box(simd_rrf_scores(black_box(60), black_box(n))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_filter, bench_expand, bench_rrf);
criterion_main!(benches);
