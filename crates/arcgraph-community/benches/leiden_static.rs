//! Criterion bench for the static GVE-Leiden algorithm per
//! ADR-040 §D-1 + §D-9.
//!
//! Three sub-benches:
//!
//! 1. `zachary` — 34-node, 78-edge canonical Leiden benchmark
//!    (Zachary 1977). Single sub-bench measures end-to-end
//!    runtime per pass.
//! 2. `sbm_1k` — Synthetic SBM with `n=1000, k=20, p_in=0.05,
//!    p_out=0.001`. Median ~5 K-edge graph; reports throughput.
//! 3. `sbm_10k` — `n=10000, k=100, p_in=0.005, p_out=0.0001`.
//!    Median ~250 K edges; reports runtime.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p arcgraph-community --bench leiden_static -- --quick
//! ```
//!
//! ## Why no SBM-100K here
//!
//! This local benchmark validates algorithmic behavior across
//! deterministic fixture scales; it does not make a 100 M-edge
//! performance claim.

use std::hint::black_box;

use arcgraph_community::{Graph, GveLeiden, LeidenParams};
use criterion::{Criterion, criterion_group, criterion_main};

const ZACHARY_EDGES: &[(u32, u32)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

fn zachary() -> Graph {
    let edges: Vec<(u32, u32, f32)> = ZACHARY_EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    Graph::from_edges_undirected(34, &edges)
}

fn sbm(n: u32, k: u32, p_in: f64, p_out: f64, seed: u64) -> Graph {
    assert!(n % k == 0);
    let block_size = n / k;
    let block_of = |v: u32| v / block_size;
    let mut state = seed;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if block_of(u) == block_of(v) {
                p_in
            } else {
                p_out
            };
            if next_unit() < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    Graph::from_edges_undirected(n, &edges)
}

fn bench_zachary(c: &mut Criterion) {
    let g = zachary();
    let mut group = c.benchmark_group("leiden_static");
    group.bench_function("zachary_34n_78e", |b| {
        b.iter(|| {
            let r = GveLeiden::run(black_box(&g), LeidenParams::default());
            black_box(r);
        });
    });
    group.finish();
}

fn bench_sbm_1k(c: &mut Criterion) {
    // n=1000, k=20 → 50 per block. p_in=0.05 → ~1225 intra-edges
    // per block × 20 = 24 500. p_out=0.001 → ~475 inter-edges.
    // Total ≈ 25 K edges.
    let g = sbm(1_000, 20, 0.05, 0.001, 0xC0FFEE);
    let mut group = c.benchmark_group("leiden_static");
    group.sample_size(20);
    group.bench_function("sbm_1k_20clusters", |b| {
        b.iter(|| {
            let r = GveLeiden::run(black_box(&g), LeidenParams::default());
            black_box(r);
        });
    });
    group.finish();
}

fn bench_sbm_10k(c: &mut Criterion) {
    // n=10000, k=100 → 100 per block. p_in=0.05 → ~4950
    // intra-edges per block × 100 = 495 000. p_out=0.0001 →
    // ~5000 inter-edges. Total ≈ 500 K edges.
    let g = sbm(10_000, 100, 0.05, 0.0001, 0xC0FFEE);
    let mut group = c.benchmark_group("leiden_static");
    group.sample_size(10);
    group.bench_function("sbm_10k_100clusters", |b| {
        b.iter(|| {
            let r = GveLeiden::run(black_box(&g), LeidenParams::default());
            black_box(r);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_zachary, bench_sbm_1k, bench_sbm_10k);
criterion_main!(benches);
