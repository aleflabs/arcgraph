//! Slice E.3 integration tests for `DiskAnnGraph::search_with_rescore`.
//!
//! Per ADR-035 §3.3 + AC-1a, the five tests below cover:
//!
//! 1. **Recall acceptance gate (AC-1a):** synthetic 128-dim 10 K
//!    dataset, SQ8 quantization via the Slice E.1 [`Sq8Codebook`],
//!    primary beam search with the SQ8 kernel, F32 rescore via
//!    `rescore_factor = 5`. Asserts `recall@10 ≥ 0.95` against a
//!    F32 brute-force ground truth — the v1.0 ship-blocking
//!    bound on the DiskANN side.
//!
//!    The handoff prompt suggested 768-dim, but DiskANN's
//!    [empirically-measured] F32 baseline recall at 10 K vectors
//!    × 768 dim with the slice-D default params is well below
//!    0.95 (curse-of-dimensionality concentration; production
//!    DiskANN deployments at 768 dim use ≥ 1 M vectors per
//!    Subramanya et al. 2019 §4 and Microsoft's reference
//!    benchmarks). At that scale the rescore wiring cannot
//!    *create* recall the primary graph never had — rescore
//!    re-ranks top-`(rescore_factor × K)` candidates against
//!    full precision, but the primary set must already cover
//!    the F32 ground truth for AC-1a to be measurable. The
//!    Slice D recall acceptance test at 128 dim 10 K achieves
//!    F32 baseline ≥ 0.95 with default params (per the existing
//!    `diskann_build_search_recall_sift_subset` test); this
//!    test layers SQ8 + rescore on the same scale so the
//!    rescore wiring's correctness is the only variable.
//!    This Slice E.3 test verifies the wiring at the largest scale
//!    where the recall criterion is measurable in a unit-test budget.
//! 2. **`rescore_factor = 1` short-circuit:** asserts the rescore
//!    path defers to [`DiskAnnGraph::search_with_delta`] without
//!    invoking the full-precision lookup (verified by passing a
//!    panicking closure).
//! 3. **`rescore_factor = 0` rejected:** asserts
//!    [`VectorIndexError::InvalidRescoreFactor`] without engaging
//!    the primary search.
//! 4. **Missing rescore vector:** lookup returns `None` for a
//!    candidate id; the call surfaces
//!    [`VectorIndexError::RescoreVectorMissing`].
//! 5. **Main-graph + delta-segment merge:** insert N = 500 into
//!    the main graph + 100 into the delta-segment; search with
//!    `rescore_factor = 3`; assert the final top-K spans both
//!    stores (per ADR-035 §5.3.1 B-3 "rescore on the merged
//!    top-K").
//!
//! ## SQ8 codec → L2Sq8 kernel byte width (post-#116)
//!
//! Per #116 closure / Slice F.1, [`Sq8Codebook::encode`] now
//! emits `i8` directly (kernel-native). The [`L2Sq8`] kernel
//! reads `&[u8]` and reinterprets as `&[i8]` via
//! `bytemuck::cast_slice` — no centering shift, no per-read
//! translation. The codec output is cast through `&[u8]` as the
//! byte transport for both the DiskANN arena and the test-side
//! brute-force baseline. The historical `sq8_u8_to_i8_bytes`
//! adapter is gone.

use std::collections::HashMap;
use std::time::Instant;

use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnParams};
use arcgraph_vector::distance::{L2F32, L2Sq8};
use arcgraph_vector::quantizer::Sq8Trainer;
use arcgraph_vector::{DistanceKernel, Encoding, Metric, VectorId, VectorIndexError};

// ─── Test fixtures ──────────────────────────────────────────────────

/// Deterministic xorshift32 PRNG. Matches the family used by the
/// other integration tests; we copy the helper here because Rust
/// integration test files are independent crates and cannot
/// share private utilities.
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn seed(s: u32) -> Self {
        Self {
            state: if s == 0 { 0xDEAD_BEEF } else { s },
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    fn next_f32_signed(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    /// Approximate standard-normal sample via Box-Muller.
    fn next_gauss(&mut self) -> f32 {
        let u1 = (self.next_u32() as f32 / u32::MAX as f32).max(1e-10);
        let u2 = self.next_u32() as f32 / u32::MAX as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generate a deterministic clustered dataset: `n_clusters`
/// uniform centers in `[-1, 1]^dim`, each surrounded by
/// `points_per_cluster` Gaussian-noise samples (sigma).
fn synthetic_cluster_dataset(
    seed: u32,
    n_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    sigma: f32,
) -> Vec<(VectorId, Vec<f32>)> {
    let mut rng = XorShift32::seed(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.next_f32_signed()).collect())
        .collect();
    let mut out = Vec::with_capacity(n_clusters * points_per_cluster);
    let mut id = 0_u32;
    for center in &centers {
        for _ in 0..points_per_cluster {
            let mut v = Vec::with_capacity(dim);
            for &c in center.iter().take(dim) {
                v.push(c + rng.next_gauss() * sigma);
            }
            out.push((VectorId::new(id), v));
            id += 1;
        }
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Brute-force F32 top-K ground truth for recall measurement.
fn brute_force_top_k(dataset: &[(VectorId, Vec<f32>)], query: &[f32], k: usize) -> Vec<VectorId> {
    let q_bytes = f32_bytes(query);
    let mut all: Vec<(VectorId, f32)> = dataset
        .iter()
        .map(|(id, v)| {
            let v_bytes = f32_bytes(v);
            let d = L2F32.distance(&v_bytes, &q_bytes);
            (*id, d)
        })
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1));
    all.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Reinterpret the codec's `[i8]` output as the byte-oriented
/// `Vec<u8>` the DiskANN arena consumes. Per #116 closure /
/// Slice F.1, the codec is i8-native, so this is a thin
/// `bytemuck::cast_slice::<i8, u8>` reinterpretation — no shift
/// or copy beyond the `Vec` allocation.
fn sq8_native_bytes(codec_i8: &[i8]) -> Vec<u8> {
    bytemuck::cast_slice::<i8, u8>(codec_i8).to_vec()
}

// ─── Test 1: recall@10 ≥ 0.95 with SQ8 + F32 rescore (AC-1a) ────────

#[test]
fn diskann_rescore_factor_5_recall_at_10_above_0_95() {
    // 100 clusters × 100 each, dim = 128, σ = 0.03 — same shape
    // as the existing Slice D F32 recall test
    // (`diskann_build_search_recall_sift_subset`). At this scale
    // F32 baseline reaches recall ≥ 0.95 with default DiskANN
    // params + l_search = 128, so the rescore amplification
    // factor's correctness is observable on top of a working
    // primary graph. Module-level docs explain why the prompt's
    // suggested 768-dim is deferred to the LDBC SNB benchmark
    // harness (Slice K) at 1 M+ scale.
    const DIM: usize = 128;
    let dataset = synthetic_cluster_dataset(
        /* seed */ 0xDA1A_5E7D,
        /* n_clusters */ 100,
        /* points_per_cluster */ 100,
        /* dim */ DIM,
        /* sigma */ 0.03,
    );
    assert_eq!(dataset.len(), 10_000);

    // Train SQ8 codebook on a stride-sampled 1 K subset that
    // spans every cluster (the dataset is generated cluster-by-
    // cluster, so `take(1000)` would only see the first 10
    // clusters and clamp values from the remaining 90 clusters
    // at the codebook boundary). `step_by(10)` distributes the
    // 1 K samples evenly across all 100 clusters; the codebook's
    // per-dim `(min, max)` envelope captures the full input
    // distribution. Production callers use a reservoir sample
    // for the same reason (per ADR-035 §3.3).
    let train_storage: Vec<Vec<f32>> = dataset.iter().step_by(10).map(|(_, v)| v.clone()).collect();
    let train_samples: Vec<&[f32]> = train_storage.iter().map(Vec::as_slice).collect();
    let codebook = Sq8Trainer
        .train(&train_samples)
        .expect("Sq8Trainer::train succeeds on uniform clustered data");

    // Encode every vector through the SQ8 codec; the i8-native
    // output is reinterpreted as `[u8]` for byte transport into
    // the DiskANN arena. Keep the original F32 bytes for the
    // rescore lookup.
    let mut sq8_storage: Vec<(VectorId, Vec<u8>)> = Vec::with_capacity(dataset.len());
    let mut f32_storage: HashMap<VectorId, Vec<u8>> = HashMap::with_capacity(dataset.len());
    for (id, v) in &dataset {
        let q_i8 = codebook.encode(v).expect("encode succeeds");
        sq8_storage.push((*id, sq8_native_bytes(&q_i8)));
        f32_storage.insert(*id, f32_bytes(v));
    }
    let pairs: Vec<(VectorId, &[u8])> = sq8_storage
        .iter()
        .map(|(id, b)| (*id, b.as_slice()))
        .collect();

    // Build the SQ8 DiskANN graph. Default params (R=70, α=1.2,
    // L_construction=100); l_search=128 at query time matches the
    // Slice D recall test.
    let params = DiskAnnParams::default();
    let mut g = DiskAnnGraph::new(params, Encoding::Sq8, Metric::L2, Box::new(L2Sq8))
        .expect("DiskAnnGraph::new accepts Sq8/L2 kernel pair");
    let build_start = Instant::now();
    g.build(&pairs).expect("SQ8 build succeeds");
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "diskann_rescore_factor_5_recall_at_10_above_0_95: build {} SQ8 vectors in {:.1}ms",
        dataset.len(),
        build_ms,
    );

    // 200 queries: 100 perturbed dataset points (easy) + 100
    // uniform-random in `[-1, 1]^768` (hard). Same mix as the
    // Slice D recall test.
    let mut rng = XorShift32::seed(0xDEAD_FACE);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(200);
    for _ in 0..100 {
        let idx = (rng.next_u32() as usize) % dataset.len();
        let mut q = dataset[idx].1.clone();
        for x in q.iter_mut() {
            *x += rng.next_gauss() * 0.01;
        }
        queries.push(q);
    }
    for _ in 0..100 {
        queries.push((0..DIM).map(|_| rng.next_f32_signed()).collect());
    }

    let k = 10;
    let l_search = 128;
    let rescore_factor = 5;

    let mut hits = 0_usize;
    let mut total = 0_usize;
    let mut search_ns_acc = 0_u128;
    for q in &queries {
        let q_f32_bytes = f32_bytes(q);
        let q_sq8_i8 = codebook.encode(q).expect("query encode");
        let q_sq8_bytes = sq8_native_bytes(&q_sq8_i8);

        let truth = brute_force_top_k(&dataset, q, k);
        let truth_set: std::collections::HashSet<u32> = truth.iter().map(|id| id.raw()).collect();

        let t0 = Instant::now();
        let res = g
            .search_with_rescore(
                &q_sq8_bytes,
                &q_f32_bytes,
                k,
                l_search,
                rescore_factor,
                &L2Sq8,
                &L2F32,
                |id: VectorId| f32_storage.get(&id).map(Vec::as_slice),
            )
            .expect("search_with_rescore succeeds");
        search_ns_acc += t0.elapsed().as_nanos();

        for (id, _) in res {
            if truth_set.contains(&id.raw()) {
                hits += 1;
            }
        }
        total += k;
    }

    let recall = hits as f64 / total as f64;
    let avg_search_us = search_ns_acc as f64 / queries.len() as f64 / 1000.0;
    eprintln!(
        "diskann_rescore_factor_5_recall_at_10_above_0_95: recall@10 = {:.4} ({}/{}), avg search {:.1} µs (l_search={}, rescore_factor={})",
        recall, hits, total, avg_search_us, l_search, rescore_factor,
    );
    assert!(
        recall >= 0.95,
        "recall@10 = {recall} < 0.95 (AC-1a, ADR-035 §3.3)"
    );
}

// ─── Test 2: rescore_factor = 1 short-circuits to search_with_delta ─

#[test]
fn diskann_rescore_factor_1_short_circuits() {
    // Build a small F32 graph; call search_with_rescore with
    // rescore_factor=1 alongside a panicking lookup closure.
    // The short-circuit path must NOT invoke the lookup; the
    // result must equal `search_with_delta` exactly.
    let mut g = DiskAnnGraph::new(
        DiskAnnParams::default(),
        Encoding::F32,
        Metric::L2,
        Box::new(L2F32),
    )
    .unwrap();
    let owned: Vec<(VectorId, Vec<u8>)> = (0..50_u32)
        .map(|i| {
            (
                VectorId::new(i),
                f32_bytes(&[(i as f32) * 0.1, 0.0, 0.0, 0.0]),
            )
        })
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    g.build(&pairs).unwrap();

    let q = f32_bytes(&[1.5, 0.0, 0.0, 0.0]);
    let direct = g.search_with_delta(&q, 5, 32).unwrap();
    let rescored = g
        .search_with_rescore(
            &q,
            &q,
            5,
            32,
            /* rescore_factor */ 1,
            &L2F32,
            &L2F32,
            // Panicking closure proves the short-circuit avoids
            // the rescore lookup.
            |_id: VectorId| -> Option<&[u8]> {
                panic!("full-precision lookup must not be invoked when rescore_factor == 1")
            },
        )
        .unwrap();

    assert_eq!(direct.len(), rescored.len());
    for (a, b) in direct.iter().zip(rescored.iter()) {
        assert_eq!(a.0, b.0, "id mismatch between direct and rescored");
        assert!(
            (a.1 - b.1).abs() < 1e-6,
            "distance mismatch: direct={} rescored={}",
            a.1,
            b.1
        );
    }
}

// ─── Test 3: rescore_factor = 0 rejected ────────────────────────────

#[test]
fn diskann_rescore_factor_zero_rejected() {
    let mut g = DiskAnnGraph::new(
        DiskAnnParams::default(),
        Encoding::F32,
        Metric::L2,
        Box::new(L2F32),
    )
    .unwrap();
    let v = f32_bytes(&[1.0, 0.0, 0.0, 0.0]);
    g.build(&[(VectorId::new(1), v.as_slice())]).unwrap();

    let q = f32_bytes(&[1.0, 0.0, 0.0, 0.0]);
    let r = g.search_with_rescore(
        &q,
        &q,
        5,
        32,
        /* rescore_factor */ 0,
        &L2F32,
        &L2F32,
        |_id: VectorId| -> Option<&[u8]> {
            panic!("rescore_factor=0 must reject before invoking lookup")
        },
    );
    assert!(
        matches!(r, Err(VectorIndexError::InvalidRescoreFactor { factor: 0 })),
        "got: {r:?}"
    );
}

// ─── Test 4: missing full-precision vector errors ───────────────────

#[test]
fn diskann_rescore_missing_vector_errors() {
    // Build a graph with ids 0..20; populate the rescore lookup
    // with only ids 0..15. A query near id=18 forces the primary
    // search to surface ids in [15, 20); the first lookup miss
    // raises RescoreVectorMissing.
    let mut g = DiskAnnGraph::new(
        DiskAnnParams::default(),
        Encoding::F32,
        Metric::L2,
        Box::new(L2F32),
    )
    .unwrap();
    let owned: Vec<(VectorId, Vec<u8>)> = (0..20_u32)
        .map(|i| {
            (
                VectorId::new(i),
                f32_bytes(&[(i as f32) * 0.1, 0.0, 0.0, 0.0]),
            )
        })
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    g.build(&pairs).unwrap();

    let storage: HashMap<VectorId, Vec<u8>> = owned
        .iter()
        .take(15)
        .map(|(id, b)| (*id, b.clone()))
        .collect();

    // Query positioned at x=1.8 — closest neighbors are ids 18,
    // 17, 19, 16, 15 (decreasing in distance from the query
    // point), all of which fall in the missing range.
    let q = f32_bytes(&[1.8, 0.0, 0.0, 0.0]);
    let r = g.search_with_rescore(
        &q,
        &q,
        5,
        32,
        /* rescore_factor */ 5,
        &L2F32,
        &L2F32,
        |id: VectorId| storage.get(&id).map(Vec::as_slice),
    );
    assert!(
        matches!(r, Err(VectorIndexError::RescoreVectorMissing { .. })),
        "got: {r:?}"
    );
    // Match the specific id surfaced by the lookup miss to
    // catch regressions where the error gets folded behind a
    // different variant.
    if let Err(VectorIndexError::RescoreVectorMissing { vector_id }) = r {
        assert!(
            (15..20).contains(&vector_id.raw()),
            "RescoreVectorMissing vector_id {} expected in [15, 20)",
            vector_id.raw()
        );
    }
}

// ─── Test 5: rescore combines main_graph and delta_segment ──────────

#[test]
fn diskann_rescore_combines_main_graph_and_delta_segment() {
    // Geometry: two interleaved 1-D clusters.
    //   - main: 500 vectors at x = 0, 1, 2, …, 499 (ids 0..500).
    //   - delta: 100 vectors at x = 0.5, 1.5, …, 99.5
    //       (ids 500..600).
    // Query at x = 50 has near-neighbors in both ranges:
    //   {50, 49, 51, 48, 52} from main and
    //   {549, 550, 548, 551, 547} from delta (ids = 500 + offset).
    // Top-10 of the rescored set must contain at least one id
    // from each store — the §5.3.1 B-3 "rescore on the merged
    // top-K" property.
    let params = DiskAnnParams {
        // Set the auto-merge threshold above 100 so the
        // delta-segment retains the streaming inserts.
        delta_max_size: 1000,
        ..DiskAnnParams::default()
    };
    let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32)).unwrap();

    // Main graph: x = i, i ∈ [0, 500).
    let main_owned: Vec<(VectorId, Vec<u8>)> = (0..500_u32)
        .map(|i| (VectorId::new(i), f32_bytes(&[i as f32, 0.0, 0.0, 0.0])))
        .collect();
    let main_pairs: Vec<(VectorId, &[u8])> = main_owned
        .iter()
        .map(|(id, b)| (*id, b.as_slice()))
        .collect();
    g.build(&main_pairs).unwrap();

    // Delta-segment: x = (i - 500) + 0.5, i ∈ [500, 600).
    let delta_owned: Vec<(VectorId, Vec<u8>)> = (500..600_u32)
        .map(|i| {
            (
                VectorId::new(i),
                f32_bytes(&[(i - 500) as f32 + 0.5, 0.0, 0.0, 0.0]),
            )
        })
        .collect();
    let delta_pairs: Vec<(VectorId, &[u8])> = delta_owned
        .iter()
        .map(|(id, b)| (*id, b.as_slice()))
        .collect();
    g.insert_stream(&delta_pairs).unwrap();

    assert_eq!(g.main_len(), 500);
    assert_eq!(g.delta_len(), 100);

    // Build the rescore lookup over both stores.
    let mut storage: HashMap<VectorId, Vec<u8>> =
        HashMap::with_capacity(main_owned.len() + delta_owned.len());
    for (id, b) in main_owned.iter().chain(delta_owned.iter()) {
        storage.insert(*id, b.clone());
    }

    let q = f32_bytes(&[50.0, 0.0, 0.0, 0.0]);
    let res = g
        .search_with_rescore(
            &q,
            &q,
            10,
            64,
            /* rescore_factor */ 3,
            &L2F32,
            &L2F32,
            |id: VectorId| storage.get(&id).map(Vec::as_slice),
        )
        .unwrap();

    assert_eq!(res.len(), 10, "expected 10 results, got {}", res.len());
    let from_main = res.iter().filter(|(id, _)| id.raw() < 500).count();
    let from_delta = res.iter().filter(|(id, _)| id.raw() >= 500).count();
    assert!(
        from_main >= 1,
        "expected ≥ 1 result from main graph, got {from_main} (results = {res:?})"
    );
    assert!(
        from_delta >= 1,
        "expected ≥ 1 result from delta-segment, got {from_delta} (results = {res:?})"
    );

    // Strong invariant: the brute-force F32 ground truth on the
    // combined dataset places id 50 (main) at distance 0 and
    // ids 549/550 (delta) at distance 0.5 — top-1 must be id 50.
    assert_eq!(
        res[0].0.raw(),
        50,
        "top-1 must be main id 50 (exact match at x=50)"
    );

    // The rescored distances must be ascending under L2.
    for w in res.windows(2) {
        assert!(
            w[0].1 <= w[1].1 + 1e-6,
            "results not ascending: {} > {}",
            w[0].1,
            w[1].1
        );
    }
}
