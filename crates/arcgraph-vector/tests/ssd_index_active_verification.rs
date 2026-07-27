//! Active end-to-end verification for the SSD-resident DiskANN serving tier
//! (ADR-195) — the ADR-133 §D-4 **Index class** recipe: build an on-disk index,
//! run real queries, assert recall vs an EXHAUSTIVE brute-force oracle.
//!
//! This exercises the REAL SSD read path: f32 vectors are streamed to a
//! `PosixPageIo` page store on disk during the build, and the 2-phase search
//! reranks candidates by reading those f32 vectors back THROUGH the
//! `BufferPool` (pread, NOT mmap). The oracle is the strong one (full linear
//! scan, NaN-safe), per doctrine §3.
//!
//! - `ssd_index_recall_at_index_class_scale` (always-on): 10K × dim=128, the
//!   ADR-133 Index recipe (recall@10 ≥ 0.95, the ADR-195 §5.2 GA gate, vs brute
//!   force).
//! - `ssd_index_recall_at_ga_dim_768` (`#[ignore]`, opt-in): 3K × dim=768 with
//!   the dim-scaled params (R=128/L=200) — proves the SSD tier hits recall at
//!   the GA-gate DIMENSION (the V-1 #740 param-curve finding: the 128-d
//!   defaults are graph-starved at 768d). Run with `--ignored`.

use std::collections::HashSet;

use arcgraph_vector::diskann::ssd::{
    DEFAULT_RERANK_FACTOR, NavQuantizer, SsdBuildConfig, SsdDiskAnnIndex,
};
use arcgraph_vector::diskann::{DiskAnnParams, RssGuard};
use arcgraph_vector::distance::{DistanceKernel, L2F32};
use arcgraph_vector::{Metric, VectorId};

/// Deterministic xorshift32 (no `rand` dep; matches the build/bench PRNG family).
struct Xs32(u32);
impl Xs32 {
    fn new(s: u32) -> Self {
        Self(if s == 0 { 0xDEAD_BEEF } else { s })
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn gauss(&mut self) -> f32 {
        let u1 = (self.next_u32() as f32 / u32::MAX as f32).max(1e-10);
        let u2 = self.next_u32() as f32 / u32::MAX as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn signed(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// `(corpus, centers)` Gaussian-cluster dataset (ADR-189 §9 / V-1 shape).
#[allow(clippy::type_complexity)]
fn corpus(
    seed: u32,
    clusters: usize,
    per: usize,
    dim: usize,
    sigma: f32,
) -> (Vec<(VectorId, Vec<f32>)>, Vec<Vec<f32>>) {
    let mut rng = Xs32::new(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.signed()).collect())
        .collect();
    let mut out = Vec::with_capacity(clusters * per);
    let mut id = 0u32;
    for c in &centers {
        for _ in 0..per {
            out.push((
                VectorId::new(id),
                c.iter().map(|&cc| cc + rng.gauss() * sigma).collect(),
            ));
            id += 1;
        }
    }
    (out, centers)
}

/// EXHAUSTIVE brute-force top-k under L2 (the strong recall oracle).
fn brute_force_top_k(data: &[(VectorId, Vec<f32>)], q: &[f32], k: usize) -> HashSet<u32> {
    let qb = f32_le(q);
    let mut all: Vec<(u32, f32)> = data
        .iter()
        .map(|(id, v)| (id.raw(), L2F32.distance(&f32_le(v), &qb)))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1));
    all.into_iter().take(k).map(|(id, _)| id).collect()
}

// Test-only parameterized recall driver: the query-generation knobs (centers,
// sigma, n_queries, k, query_seed) are inherent to a recall measurement, matching
// the crate's documented `too_many_arguments` convention (see dispatcher.rs /
// diskann::search). `dim` is carried by `centers[*].len()`, so it is not a param.
#[allow(clippy::too_many_arguments)]
fn recall_at_k(
    idx: &SsdDiskAnnIndex,
    data: &[(VectorId, Vec<f32>)],
    centers: &[Vec<f32>],
    sigma: f32,
    n_queries: usize,
    k: usize,
    query_seed: u32,
) -> f64 {
    let mut rng = Xs32::new(query_seed);
    let mut hits = 0usize;
    for _ in 0..n_queries {
        let c = &centers[(rng.next_u32() as usize) % centers.len()];
        let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * sigma).collect();
        let gt = brute_force_top_k(data, &q, k);
        for (id, _) in idx.search(&q, k).expect("search") {
            if gt.contains(&id.raw()) {
                hits += 1;
            }
        }
    }
    hits as f64 / (k * n_queries) as f64
}

/// The dim-scaled Vamana params (V-1 #740 `params_for_dim`): the 128-d defaults
/// are graph-starved at 768d → R=128 / L_construction=200 restores recall.
fn params_for_dim(dim: usize) -> DiskAnnParams {
    if dim <= 256 {
        DiskAnnParams::default()
    } else {
        DiskAnnParams {
            r: 128,
            l_construction: 200,
            ..DiskAnnParams::default()
        }
    }
}

#[test]
fn ssd_index_recall_at_index_class_scale() {
    // ADR-133 §D-4 Index recipe: 10K vectors, 1K queries, recall@10 ≥ 0.95 (the
    // ADR-195 §5.2 GA gate) vs exhaustive brute force — exercising the on-disk
    // f32 store + BufferPool rerank read path end to end.
    let dim = 128;
    let sigma = 0.03;
    let (data, centers) = corpus(11, 100, 100, dim, sigma); // 10_000 points
    let refs: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
    let codebook = arcgraph_vector::quantizer::Sq8Trainer
        .train(&refs)
        .expect("train sq8");

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let guard = RssGuard::disabled();
    let cfg = SsdBuildConfig {
        dim,
        metric: Metric::L2,
        params: params_for_dim(dim),
        pool_frames: 2048,
        rerank_factor: DEFAULT_RERANK_FACTOR,
        // Exercise the rayon-parallel build (#112) end-to-end through the SSD
        // index at always-on scale.
        parallel_build_batch: Some(1024),
    };
    let idx = SsdDiskAnnIndex::build(
        tmp.path(),
        &cfg,
        NavQuantizer::Sq8(codebook),
        data.clone(),
        &guard,
    )
    .expect("build ssd index");
    assert_eq!(idx.len(), 10_000);

    let recall = recall_at_k(&idx, &data, &centers, sigma, 1000, 10, 4242);
    println!(
        "SSD Index active-verification: N=10000 dim={dim} recall@10={recall:.4} \
         disk_bytes={} records_per_page={}",
        idx.disk_bytes(),
        idx.layout().records_per_page
    );
    assert!(
        recall >= 0.95,
        "SSD recall@10 = {recall:.4} < 0.95 (ADR-195 §5.2 / ADR-133 GA gate)"
    );
}

#[test]
#[ignore = "GA-dim proof: 3K×768d build is ~tens of seconds; run with --ignored"]
fn ssd_index_recall_at_ga_dim_768() {
    // The GA-gate DIMENSION (768) with the dim-scaled params. Proves the SSD
    // tier hits recall at 768d — the dimension the 10M validation uses — where
    // the 128-d defaults would be graph-starved (V-1 #740). Smaller N (3K) keeps
    // the single-thread build tractable; recall@10 ≥ 0.95 is the GA gate's shape.
    let dim = 768;
    let sigma = 0.02;
    let (data, centers) = corpus(23, 60, 50, dim, sigma); // 3_000 points
    let refs: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
    let codebook = arcgraph_vector::quantizer::Sq8Trainer
        .train(&refs)
        .expect("train sq8");

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let guard = RssGuard::disabled();
    let cfg = SsdBuildConfig {
        dim,
        metric: Metric::L2,
        params: params_for_dim(dim), // R=128 / L=200
        pool_frames: 1024,
        rerank_factor: DEFAULT_RERANK_FACTOR,
        parallel_build_batch: Some(512),
    };
    let idx = SsdDiskAnnIndex::build(
        tmp.path(),
        &cfg,
        NavQuantizer::Sq8(codebook),
        data.clone(),
        &guard,
    )
    .expect("build ssd index");
    assert_eq!(idx.len(), 3_000);
    // dim=768 → record 3600 B → 2 records / 8 KiB page (the 40 GB-at-10M packing).
    assert_eq!(idx.layout().records_per_page, 2);

    let recall = recall_at_k(&idx, &data, &centers, sigma, 300, 10, 9999);
    println!("SSD GA-dim active-verification: N=3000 dim=768 R=128/L=200 recall@10={recall:.4}");
    assert!(
        recall >= 0.95,
        "SSD recall@10 @768d = {recall:.4} < 0.95 (ADR-195 §5.2 / ADR-133 GA gate)"
    );
}
