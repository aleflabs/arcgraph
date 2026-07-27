//! Slice B distance-kernel uplift bench.
//!
//! Compares the [`simsimd`]-backed [`L2F32`] kernel against a
//! hand-written LLVM-autovectorized scalar baseline at the
//! production working dimension `dim=768` (per ADR-035 §1.2).
//!
//! ## Acceptance criterion (ADR-035 D-2 / Slice B AC)
//!
//! P50 simsimd time MUST be < 0.4× scalar P50 (i.e., ≥ 2.5×
//! speedup). The PR #95 F3 finding targets 3–8× on AVX-2 / AVX-512
//! hardware; recall@10 unaffected — this is a free distance-kernel
//! lunch.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p arcgraph-vector --bench distance -- --quick
//! ```
//!
//! `--quick` runs in ~5 s with looser confidence intervals; the
//! full sweep (`cargo bench -p arcgraph-vector --bench distance`)
//! takes ~60 s and produces HTML reports under
//! `target/criterion/`. The CI gate is the `--quick` variant; the
//! full sweep is owner-driven during release validation.

use std::hint::black_box;

use arcgraph_vector::{
    DistanceKernel,
    distance::{CosineF32, IpF32, L2F32, L2RaBitQSym, L2Sq8},
    quantizer::{self, RaBitQTrainer, Sq8Trainer},
};
use criterion::{Criterion, criterion_group, criterion_main};

const DIM: usize = 768;

/// Scalar L2 baseline. Idiomatic Rust loop over `(a[i] − b[i])²`;
/// LLVM autovectorizes this on `-O2 / opt-level=3` so it is the
/// fair "no-simd" floor — not a bad-faith strawman.
fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Scalar inner product baseline.
fn scalar_ip(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Scalar cosine *distance* baseline (`1 - cos(θ)`). Matches the
/// simsimd return convention so the apples-to-apples bench
/// remains honest.
fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f32::MIN_POSITIVE);
    1.0_f32 - (dot / denom)
}

fn make_input(seed_a: f32, seed_b: f32) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..DIM)
        .map(|i| (i as f32 * 0.013 + seed_a).sin())
        .collect();
    let b: Vec<f32> = (0..DIM)
        .map(|i| (i as f32 * 0.011 + seed_b).cos())
        .collect();
    (a, b)
}

fn training_sample() -> Vec<Vec<f32>> {
    (0..(DIM + 16))
        .map(|j| {
            (0..DIM)
                .map(|i| ((i as f32 * 0.017) + (j as f32 * 0.031)).sin())
                .collect()
        })
        .collect()
}

fn bench_l2_f32_768(c: &mut Criterion) {
    let (a, b) = make_input(0.5, 0.7);
    let a_bytes: &[u8] = bytemuck::cast_slice(&a);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b);
    let kernel = L2F32;

    let mut group = c.benchmark_group("l2_f32_dim768");
    group.bench_function("simsimd", |bch| {
        bch.iter(|| kernel.distance(black_box(a_bytes), black_box(b_bytes)))
    });
    group.bench_function("scalar", |bch| {
        bch.iter(|| scalar_l2(black_box(&a), black_box(&b)))
    });
    group.finish();
}

fn bench_ip_f32_768(c: &mut Criterion) {
    let (a, b) = make_input(0.3, 0.9);
    let a_bytes: &[u8] = bytemuck::cast_slice(&a);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b);
    let kernel = IpF32;

    let mut group = c.benchmark_group("ip_f32_dim768");
    group.bench_function("simsimd", |bch| {
        bch.iter(|| kernel.distance(black_box(a_bytes), black_box(b_bytes)))
    });
    group.bench_function("scalar", |bch| {
        bch.iter(|| scalar_ip(black_box(&a), black_box(&b)))
    });
    group.finish();
}

fn bench_cosine_f32_768(c: &mut Criterion) {
    let (a, b) = make_input(0.1, 0.4);
    let a_bytes: &[u8] = bytemuck::cast_slice(&a);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b);
    let kernel = CosineF32;

    let mut group = c.benchmark_group("cosine_f32_dim768");
    group.bench_function("simsimd", |bch| {
        bch.iter(|| kernel.distance(black_box(a_bytes), black_box(b_bytes)))
    });
    group.bench_function("scalar", |bch| {
        bch.iter(|| scalar_cosine(black_box(&a), black_box(&b)))
    });
    group.finish();
}

fn bench_l2_sq8_768(c: &mut Criterion) {
    let samples = training_sample();
    let refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    let cb = Sq8Trainer.train(&refs).expect("train sq8");
    let (a, b) = make_input(0.2, 0.8);
    let a_i8 = cb.encode(&a).expect("encode a");
    let b_i8 = cb.encode(&b).expect("encode b");
    let a_bytes: Vec<u8> = a_i8.iter().map(|&x| x as u8).collect();
    let b_bytes: Vec<u8> = b_i8.iter().map(|&x| x as u8).collect();
    let kernel = L2Sq8;

    c.bench_function("l2_sq8_dim768", |bch| {
        bch.iter(|| kernel.distance(black_box(&a_bytes), black_box(&b_bytes)))
    });
}

fn bench_rabitq_estimate_l2_sq_768(c: &mut Criterion) {
    let samples = training_sample();
    let refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    let cb = RaBitQTrainer
        .train(&refs, 0x7580_2090_0000_0002)
        .expect("train rabitq");
    let (a, q) = make_input(0.2, 0.8);
    let payload = cb.encode_aligned(&a).expect("encode");
    let prepared = cb.prepare_query(&q).expect("prepare");

    c.bench_function("rabitq_estimate_l2_sq_dim768", |bch| {
        bch.iter(|| quantizer::estimate_l2_sq(black_box(&prepared), black_box(&payload)))
    });
}

fn bench_rabitq_prepare_query_768(c: &mut Criterion) {
    let samples = training_sample();
    let refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    let cb = RaBitQTrainer
        .train(&refs, 0x7580_2090_0000_0002)
        .expect("train rabitq");
    let (_, q) = make_input(0.2, 0.8);

    c.bench_function("rabitq_prepare_query_dim768", |bch| {
        bch.iter(|| cb.prepare_query(black_box(&q)).expect("prepare"))
    });
}

fn bench_l2_rabitq_sym_768(c: &mut Criterion) {
    let samples = training_sample();
    let refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    let cb = RaBitQTrainer
        .train(&refs, 0x7580_2090_0000_0002)
        .expect("train rabitq");
    let (a, b) = make_input(0.2, 0.8);
    let a_payload = cb.encode_aligned(&a).expect("encode a");
    let b_payload = cb.encode_aligned(&b).expect("encode b");
    let kernel = L2RaBitQSym::new(DIM);

    c.bench_function("l2_rabitq_sym_dim768", |bch| {
        bch.iter(|| kernel.distance(black_box(&a_payload), black_box(&b_payload)))
    });
}

criterion_group!(
    benches,
    bench_l2_f32_768,
    bench_ip_f32_768,
    bench_cosine_f32_768,
    bench_l2_sq8_768,
    bench_rabitq_estimate_l2_sq_768,
    bench_rabitq_prepare_query_768,
    bench_l2_rabitq_sym_768,
);
criterion_main!(benches);
