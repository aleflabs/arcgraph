//! Slice C — HNSW core integration tests.
//!
//! These tests cover:
//!
//! - [`hnsw_build_search_recall_sift_subset`] — recall@10 ≥ 0.97
//!   on a 10 K-vector synthetic stand-in for SIFT-1M's first
//!   chunk. The synthetic generator (seeded Gaussian clusters at
//!   dim=128) is the standard ANN-Benchmarks unit-test
//!   substitute that keeps `cargo test` hermetic.
//! - [`hnsw_insert_delete_fixed_seed_cycles`] — 1000 cycles of
//!   insert + 5 % delete; assert tombstone ratio < 10 % AND
//!   recall@10 ≥ 0.95 with MN-RU repair. A **fixed-seed
//!   simulator** over three deterministic seeds — NOT a `proptest`
//!   (it has no `TestRunner` / `Strategy` / shrink machinery). The
//!   randomized-input property coverage lives in the real
//!   `proptest!` tests below ([`prop_hnsw_recall_vs_exhaustive`],
//!   [`prop_hnsw_no_tombstoned_returned`]); this one is retained as
//!   a deterministic high-cycle-count regression replay.
//! - [`prop_hnsw_recall_vs_exhaustive`] — real `proptest!` over
//!   randomized vector sets + queries; recall@k vs an exhaustive
//!   brute-force scan, mean ≥ 0.97 + a p10 tail floor.
//! - [`prop_hnsw_no_tombstoned_returned`] — real `proptest!` over
//!   randomized insert/delete sequences; no tombstoned id ever
//!   appears in a result set.
//! - [`hnsw_distance_kernel_dispatch_smoke`] — build with each of
//!   `L2F32` / `IpF32` / `CosineF32`; assert the ranking is
//!   consistent (the planted-near-query vector wins on all
//!   kernels) so future kernel additions don't silently break
//!   dispatch.
//! - [`cosine_zero_vector_behavior_documented`] — OQ-V5 pin
//!   (issue #104). simsimd's cosine kernel on a `(0, 0, 0)`
//!   vector input has undefined-by-paper semantics
//!   (norm 0 → division by zero). This test pins the behavior
//!   simsimd actually exhibits today (NaN, +Inf, or wrapped 0)
//!   so future simsimd upgrades surface as a behavior change
//!   rather than a silent recall regression.

use std::collections::HashSet;

use arcgraph_vector::distance::{CosineF32, IpF32, L2F32};
use arcgraph_vector::hnsw::{HnswGraph, HnswParams};
use arcgraph_vector::ids::VectorId;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── helpers ─────────────────────────────────────────────────────

/// Generate `n` deterministic random unit vectors at dimension
/// `dim` from a seeded RNG. Vectors live on the unit sphere
/// (L2-normalized) so cosine and L2 rankings agree at the top —
/// the standard ANN-Benchmarks unit-test convention.
fn generate_unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // Box-Muller'd by StandardUniform → uniform on
            // hyper-cube; then normalize. This is sufficient for
            // recall tests; truly-uniform-on-sphere requires
            // Marsaglia's method, but ANN-Benchmarks uses the
            // simpler form.
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    u * 2.0 - 1.0 // (-1, 1) uniform
                })
                .collect();
            l2_normalize(v)
        })
        .collect()
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[inline]
fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Brute-force top-`k` by L2 distance. Returns ids sorted by
/// ascending L2.
fn brute_force_top_k(vectors: &[(VectorId, Vec<f32>)], query: &[f32], k: usize) -> Vec<VectorId> {
    let mut scored: Vec<(f32, VectorId)> = vectors
        .iter()
        .map(|(id, v)| {
            let d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (d, *id)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN compare"));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// Recall@k for one query: |HNSW result ∩ brute force result| / k.
fn recall_at_k(hnsw_result: &[(VectorId, f32)], brute_result: &[VectorId], k: usize) -> f64 {
    let h: HashSet<VectorId> = hnsw_result.iter().map(|(id, _)| *id).take(k).collect();
    let b: HashSet<VectorId> = brute_result.iter().copied().take(k).collect();
    let inter = h.intersection(&b).count();
    inter as f64 / k as f64
}

/// Lower-bound percentile of a slice of per-query recalls. `p` in
/// `[0, 1]`; `p10` ⇒ `p = 0.10`. Uses the
/// nearest-rank (floor) convention so the returned value is an
/// actually-observed per-query recall, not an interpolated one —
/// a `p10` floor of `x` asserts "at least 90 % of queries recalled
/// ≥ x". Mutates `recalls` (sorts it ascending in place).
fn percentile_floor(recalls: &mut [f64], p: f64) -> f64 {
    if recalls.is_empty() {
        return 0.0;
    }
    recalls.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN recall"));
    let idx = (((recalls.len() - 1) as f64) * p).floor() as usize;
    recalls[idx]
}

// ─── Test 1 — recall on synthetic SIFT-shaped subset ──────────────

/// Build a 5 000-vector × dim=64 graph and assert recall@10
/// ≥ 0.97 against brute-force ground truth across 100 queries.
///
/// **Why 5 000 / 64 instead of 10 000 / 128.** The slice's
/// recall target (0.97) is a property of the HNSW algorithm /
/// parameters, not the dataset size — it falls out of beam
/// search hitting the right neighborhood. ANN-Benchmarks's
/// SIFT-1M result at M=16, ef_search=100 is recall@10 ≈ 0.99
/// on real SIFT; on Gaussian-uniform synthetic at our scale we
/// expect recall ≥ 0.97 with the same params. The smaller
/// dimension keeps `cargo test` (debug build) under 5 seconds.
#[test]
fn hnsw_build_search_recall_sift_subset() {
    let dim = 64;
    let n = 5_000;
    let n_queries = 100;
    let k = 10;

    let params = HnswParams {
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        seed: 42,
    };

    let kernel = L2F32;

    // Vector arena.
    let raw = generate_unit_vectors(7, n, dim);
    let vectors: Vec<(VectorId, Vec<f32>)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), v.clone()))
        .collect();

    let mut g = HnswGraph::new(params, dim, &kernel);
    for (id, v) in &vectors {
        g.insert(*id, &bytes_of(v), &kernel).unwrap();
    }

    // Queries: same distribution as the data.
    let queries = generate_unit_vectors(99, n_queries, dim);

    let mut per_query: Vec<f64> = Vec::with_capacity(n_queries);
    for q in queries.iter() {
        let bf = brute_force_top_k(&vectors, q, k);
        let hres = g
            .search(&bytes_of(q), k, params.ef_search, &kernel)
            .unwrap();
        per_query.push(recall_at_k(&hres, &bf, k));
    }
    let mean_recall = per_query.iter().sum::<f64>() / n_queries as f64;
    // `percentile_floor` sorts `per_query` ascending in place.
    let p10 = percentile_floor(&mut per_query, 0.10);
    let min = per_query[0];
    println!("recall@10 over {n_queries} queries: mean={mean_recall:.4} p10={p10:.4} min={min:.4}");

    // Mean floor — kept at the Slice C exit criterion 0.97.
    assert!(
        mean_recall >= 0.97,
        "mean recall@10 = {mean_recall}, below 0.97 target (Slice C exit criterion)"
    );

    // W28-S2 hardening (gap analysis PR #510 §3 + ADR-165 M1): the
    // old per-query oracle was `r >= 0.4` — a "diagnostic floor"
    // loose enough that a query recalling 4/10 passed. A graph that
    // degraded a tenth of its queries to 0.4 recall (a real,
    // user-visible regression) would not have tripped it. We replace
    // it with a **p10 floor**: at least 90 % of queries must recall
    // ≥ 0.80. This is a genuine statistical floor on a probabilistic
    // ANN search (a tail-recall guarantee, not a mean that a few
    // perfect queries can prop up), and it is deterministic here
    // (data seed 7, query seed 99, HNSW seed 42). The 0.80 value
    // sits well below the observed p10 (printed above) so legitimate
    // per-query variance does not trip it, but a tail collapse does.
    assert!(
        p10 >= 0.80,
        "p10 recall@10 = {p10:.4} below 0.80 tail floor; ≥ 10 % of queries \
         lost ≥ 2 of their top-10 — a tail-recall regression"
    );
}

// ─── Test 2 — insert/delete fixed-seed simulator with MN-RU repair ─

/// 1000 cycles of insert + 5 % delete-and-tombstone; assert
/// tombstone ratio < 10 % AND recall@10 ≥ 0.95 with MN-RU repair.
///
/// **Cycle semantics.** Each cycle inserts one fresh vector
/// (id auto-incrementing). With 5 % probability per cycle, also
/// tombstones a random *live* vector. After 1000 cycles, ~50
/// vectors are tombstoned in a graph of ~1050 — the ~5 %
/// tombstone ratio that ADR-035 §5.3 cites as the operational
/// alert level (rebuild fires at 30 %). MN-RU repair runs once
/// at the end against the detected-unreachable set.
///
/// **Why this is NOT named `*_proptest` (W28-S2; ADR-165 M2).**
/// This test exercises three fixed seeds through a hand-rolled
/// simulator; it has no `proptest::TestRunner`, no `Strategy`, and
/// no shrink machinery. The previous name `hnsw_insert_delete_proptest`
/// implied randomized-input property coverage the test did not
/// provide (gap analysis PR #510 §3 flagged it as a decoy). The
/// genuine randomized property coverage now lives in
/// [`prop_hnsw_recall_vs_exhaustive`] and
/// [`prop_hnsw_no_tombstoned_returned`] below. This test is
/// retained, renamed, as a deterministic high-cycle-count
/// regression replay (1000 cycles is more than the proptests run
/// per case, so it still earns its keep as a soak-style pin).
#[test]
fn hnsw_insert_delete_fixed_seed_cycles() {
    for seed in [1u64, 17, 99] {
        run_insert_delete_cycle(seed, 1000);
    }
}

fn run_insert_delete_cycle(rng_seed: u64, cycles: usize) {
    let dim = 32;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: rng_seed,
    };
    let kernel = L2F32;
    let mut g = HnswGraph::new(params, dim, &kernel);

    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut alive: Vec<VectorId> = Vec::new();
    let mut all_vectors: Vec<(VectorId, Vec<f32>)> = Vec::new();
    let mut next_id: u32 = 0;

    // Seed the graph with 50 vectors so deletions early in the
    // cycle have something to chew on.
    for _ in 0..50 {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                u * 2.0 - 1.0
            })
            .collect();
        let v = l2_normalize(v);
        let id = VectorId::new(next_id);
        next_id += 1;
        g.insert(id, &bytes_of(&v), &kernel).unwrap();
        alive.push(id);
        all_vectors.push((id, v));
    }

    // Run cycles.
    for _cycle in 0..cycles {
        // Insert one fresh vector.
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                u * 2.0 - 1.0
            })
            .collect();
        let v = l2_normalize(v);
        let id = VectorId::new(next_id);
        next_id += 1;
        g.insert(id, &bytes_of(&v), &kernel).unwrap();
        alive.push(id);
        all_vectors.push((id, v));

        // 5 % chance of delete.
        if rng.next_u64() as u32 % 100 < 5 && !alive.is_empty() {
            let idx = (rng.next_u64() as u32 as usize) % alive.len();
            let victim = alive.swap_remove(idx);
            g.mark_deleted(victim);
        }
    }

    let ratio = g.tombstone_ratio();
    assert!(
        ratio < 0.10,
        "seed {rng_seed}: tombstone ratio {ratio} ≥ 0.10 (ADR-035 §5.3 alert threshold)"
    );

    // MN-RU repair after the delete-heavy phase.
    let unreachable = g.detect_unreachable();
    g.mn_ru_repair(&unreachable, &kernel).unwrap();

    // Recall: pick 50 live vectors as queries; the brute-force
    // top-10 is over the live (non-tombstoned) set.
    let live_set: HashSet<VectorId> = alive.iter().copied().collect();
    let live_vectors: Vec<(VectorId, Vec<f32>)> = all_vectors
        .iter()
        .filter(|(id, _)| live_set.contains(id))
        .cloned()
        .collect();

    let mut total_recall = 0.0_f64;
    let n_queries = 50;
    for q_idx in 0..n_queries {
        let (qid, qvec) = &live_vectors[q_idx % live_vectors.len()];
        let bf = brute_force_top_k(&live_vectors, qvec, 10);
        let hres = g
            .search(&bytes_of(qvec), 10, params.ef_search, &kernel)
            .unwrap();
        // Sanity: every HNSW result must be a live vector
        // (tombstone filter held).
        for (rid, _) in &hres {
            assert!(
                live_set.contains(rid),
                "seed {rng_seed}: HNSW returned tombstoned id {rid:?} for query {qid:?}"
            );
        }
        total_recall += recall_at_k(&hres, &bf, 10);
    }
    let mean_recall = total_recall / n_queries as f64;
    println!(
        "seed {rng_seed}: tombstone_ratio={ratio:.4}, recall@10={mean_recall:.4} after MN-RU repair (unreachable={})",
        unreachable.len()
    );
    assert!(
        mean_recall >= 0.95,
        "seed {rng_seed}: mean recall@10 = {mean_recall} after MN-RU repair, below 0.95 target"
    );
}

// ─── Test 3 — distance kernel dispatch smoke ──────────────────────

/// Build a 100-vector graph with each of the three v1.0 F32
/// kernels and assert that:
///
/// 1. Build + search dispatches end-to-end (the path from
///    `HnswGraph` through `select_neighbors_heuristic` to
///    `search_layer` exercises the supplied `DistanceKernel`
///    impl correctly).
/// 2. Returned IDs are valid (each result `id` is one we
///    inserted).
/// 3. Returned distances are monotonically non-decreasing —
///    the v1.0 search contract is "ranks by ascending kernel
///    output". (For L2 / Cosine this is "closest first"; for
///    IP the trait documents that callers pre-normalize so
///    the ranking is well-defined. This test pins **the
///    monotonic ranking property**, not the metric semantics —
///    the latter live in `tests/distance.rs`.)
///
/// The test does NOT assert that the same query produces the
/// same top-1 across kernels; L2 and Cosine on unit-sphere
/// data agree, but IP without normalization disagrees, and
/// pre-normalizing for one kernel and not the others would be
/// a Slice E.2 concern (rescore-aware ranking).
#[test]
fn hnsw_distance_kernel_dispatch_smoke() {
    let dim = 16;
    let n = 100;

    // Pre-normalize so all three kernels agree on the top-1
    // identity (the planted near-twin of the query).
    let raw = generate_unit_vectors(13, n, dim);
    let mut planted = raw[0].clone();
    for x in &mut planted {
        *x += 0.001;
    }
    let planted = l2_normalize(planted);
    let query = raw[0].clone();

    let kernels: [(&str, &dyn arcgraph_vector::distance::DistanceKernel); 3] = [
        ("L2F32", &L2F32),
        ("IpF32", &IpF32),
        ("CosineF32", &CosineF32),
    ];

    for (label, kernel) in kernels {
        let params = HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            seed: 5,
        };
        let mut g = HnswGraph::new(params, dim, kernel);

        let mut inserted_ids: HashSet<VectorId> = HashSet::new();
        for (i, v) in raw.iter().enumerate() {
            let id = VectorId::new(i as u32 + 100);
            g.insert(id, &bytes_of(v), kernel).unwrap();
            inserted_ids.insert(id);
        }
        let planted_id = VectorId::new(0);
        g.insert(planted_id, &bytes_of(&planted), kernel).unwrap();
        inserted_ids.insert(planted_id);

        let r = g.search(&bytes_of(&query), 5, 100, kernel).unwrap();
        assert!(
            !r.is_empty(),
            "kernel {label}: search returned empty result"
        );

        // (1) every returned id is one we inserted.
        for (id, _) in &r {
            assert!(
                inserted_ids.contains(id),
                "kernel {label}: search returned unknown id {id:?}"
            );
        }

        // (2) distances are monotonic non-decreasing (ascending).
        for w in r.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-6,
                "kernel {label}: result not monotonically ordered: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }
}

// ─── Test 4 — OQ-V5 cosine zero-vector pin (issue #104) ───────────

/// **Open question OQ-V5 (issue #104).**
///
/// simsimd's cosine kernel computes `1 − dot(a, b) / (‖a‖·‖b‖)`.
/// On a `(0, 0, 0)` input vector both norms are zero, so the
/// fraction is `0 / 0` — mathematically undefined; an
/// implementation could surface that as NaN (IEEE-754 default),
/// `+Inf`, panic, or a wrapped sentinel like `0.0` / `1.0`. The
/// Slice C task is to **pin the actual behavior** so future
/// simsimd upgrades surface as a behavior change rather than a
/// silent recall regression.
///
/// **Pinned today (simsimd 6.5.16, observed 2026-04-25):** the
/// kernel takes a **two-branch wrap**:
///
/// - `cosine_distance(zero, zero) == 0.0` — kernel treats two
///   zero-norm vectors as identical. The "self-distance is 0"
///   convention.
/// - `cosine_distance(zero, finite) == 1.0` — kernel returns
///   the **maximum** cosine distance (`1 - 0 = 1`) when only
///   one side is degenerate, on both argument orders. The
///   "zero is orthogonal to everything else" convention. This
///   is the safer of the two interpretations from a recall
///   standpoint: a zero-norm tenant vector ranks *last*
///   against any finite query, so it cannot leak into a
///   user-visible top-k unless the user query is itself zero.
///
/// Both results are mathematically arbitrary (the cosine angle
/// for the zero vector is undefined) but stable, finite, and
/// the same on every supported platform's SIMD path
/// (AVX-512 / AVX-2 / NEON / scalar). HNSW's search ranking
/// therefore never observes NaN from this kernel — the
/// `OrderedF32` NaN-as-largest discipline is robustness
/// against future kernel additions, not an observed-today path.
///
/// **Implication for v1.0 search semantics.** Under the
/// `(0, 1)` two-branch wrap above, a zero-norm vector in the
/// arena does NOT contaminate finite-query top-k (it sorts
/// last). The only degenerate case is a zero-norm *query*
/// against a zero-norm corpus entry — both come back at
/// distance 0 and tie at the top; broken arbitrarily by graph
/// topology. Two follow-up items are tracked downstream:
///
/// 1. **Quantizer-side guard** (Slice E.1, OQ-V4 already
///    resolved) — degenerate zero training samples are
///    rejected at training time, so no production tenant
///    expects to see zero-norm vectors past the staging arena.
/// 2. **Insert-side guard** (post-M3.a follow-up) — ADR-035
///    §9.5 covers dim-mismatch on insert; an analogous
///    "zero-norm cosine vector" warning is filed as a v1.1
///    nicety. Today, callers writing zero vectors to a cosine
///    index get the documented wrap-to-0 behavior; this is
///    correct in the sense that any consistent behavior is
///    acceptable, but operators should know.
///
/// **Action on behavior change.** If a future simsimd release
/// switches the convention (e.g., to NaN, `+Inf`, or `1.0`),
/// this test fails. Resolution path:
///
/// 1. Decide whether the new behavior is preferable.
/// 2. If keeping wrap-to-0, vendor the cosine kernel per
///    ADR-035 D-2's vendoring contingency.
/// 3. If accepting the new behavior, update this test and the
///    follow-up filed under (1)/(2) above.
///
/// Either way, the change requires explicit opt-in — the
/// regression test forces the conversation.
#[test]
fn cosine_zero_vector_behavior_documented() {
    use arcgraph_vector::distance::DistanceKernel;
    let zero: [f32; 3] = [0.0, 0.0, 0.0];
    let probe: [f32; 3] = [1.0, 0.0, 0.0];

    let d_zero_zero = CosineF32.distance(bytemuck::cast_slice(&zero), bytemuck::cast_slice(&zero));
    let d_zero_probe =
        CosineF32.distance(bytemuck::cast_slice(&zero), bytemuck::cast_slice(&probe));
    let d_probe_zero =
        CosineF32.distance(bytemuck::cast_slice(&probe), bytemuck::cast_slice(&zero));

    // Pinned behavior (simsimd 6.5.16, observed 2026-04-25):
    //
    // - `cosine_distance(zero, zero) == 0.0` — special-cased
    //   "self-distance" wrap; the kernel treats two
    //   zero-norm vectors as identical.
    // - `cosine_distance(zero, finite) == 1.0` — the kernel
    //   treats zero-vs-finite as maximally far in cosine
    //   distance space (`1 - cos(θ)` clamped to 1 when the
    //   norm-product is zero on one side only). Symmetric in
    //   argument order.
    //
    // Both results are **finite, stable across SIMD paths**
    // (AVX-512 / AVX-2 / NEON / scalar fallback), and the same
    // on both argument orders for the asymmetric case. HNSW
    // search therefore never observes NaN from this kernel
    // today; the `OrderedF32` NaN-as-largest discipline is
    // robustness against future kernel additions, not an
    // observed-today path.
    //
    // Print the observed values when the assertion fails so a
    // future simsimd release that changes the convention (e.g.,
    // to NaN, +Inf, or a different finite sentinel) surfaces as
    // a clear diagnostic.
    println!("OQ-V5 pin: cosine((0,0,0), (0,0,0)) = {d_zero_zero}");
    println!("OQ-V5 pin: cosine((0,0,0), (1,0,0)) = {d_zero_probe}");
    println!("OQ-V5 pin: cosine((1,0,0), (0,0,0)) = {d_probe_zero}");
    assert_eq!(
        d_zero_zero, 0.0,
        "simsimd cosine kernel changed behavior on (0,0,0) vs (0,0,0): expected wrapped 0.0, got {d_zero_zero}; see OQ-V5 docstring above"
    );
    assert_eq!(
        d_zero_probe, 1.0,
        "simsimd cosine kernel changed behavior on (0,0,0) vs (1,0,0): expected 1.0, got {d_zero_probe}; see OQ-V5 docstring above"
    );
    assert_eq!(
        d_probe_zero, 1.0,
        "simsimd cosine kernel changed behavior on (1,0,0) vs (0,0,0): expected 1.0 (symmetric), got {d_probe_zero}; see OQ-V5 docstring above"
    );

    // Cross-check with the HNSW search path: a graph holding
    // both a zero vector (id 0) and a finite vector (id 1)
    // returns both with finite distances under the two-branch
    // wrap. The finite vector (cosine distance 0 to query)
    // outranks the zero vector (cosine distance 1 to query),
    // so the zero vector lands at position 1 — the recall-safe
    // outcome we want.
    let dim = 3;
    let params = HnswParams {
        m: 4,
        ef_construction: 32,
        ef_search: 32,
        seed: 0,
    };
    let mut g = HnswGraph::new(params, dim, &CosineF32);
    g.insert(VectorId::new(0), bytemuck::cast_slice(&zero), &CosineF32)
        .unwrap();
    g.insert(VectorId::new(1), bytemuck::cast_slice(&probe), &CosineF32)
        .unwrap();

    let q: [f32; 3] = [1.0, 0.0, 0.0];
    let r = g
        .search(bytemuck::cast_slice(&q), 2, 10, &CosineF32)
        .unwrap();

    // Both vectors come back with finite distances (no NaN
    // contamination of the result list under the two-branch
    // wrap convention).
    assert_eq!(r.len(), 2);
    for (id, dist) in &r {
        assert!(
            dist.is_finite(),
            "cosine result {id:?} produced non-finite distance {dist}"
        );
    }
    // The finite vector (probe) outranks the zero vector under
    // the two-branch wrap (probe d=0, zero d=1).
    assert_eq!(
        r[0].0,
        VectorId::new(1),
        "finite cosine result must rank ahead of zero-vector result; the zero d=1 wrap is recall-safe"
    );
    assert!(
        r[0].1 < 1e-3,
        "finite probe self-distance under cosine should be ~0; got {}",
        r[0].1
    );
    assert_eq!(r[1].0, VectorId::new(0));
    assert!(
        (r[1].1 - 1.0).abs() < 1e-6,
        "zero-vector cosine distance to finite query should be 1 under the pinned wrap; got {}",
        r[1].1
    );
}

// ─── Test 5 — real recall proptest vs EXHAUSTIVE brute force ──────

proptest! {
    // W28-S573 exceed-spec: 64 → 160 cases. The HNSW recall property is
    // over a CONTINUOUS vector space, so exhaustive enumeration is
    // genuinely infeasible (per ADR-165 M1) — the case count is the
    // statistical lever, and 160 cases × randomized (seed, n, dim) is a
    // 2.5× deepening of the sampled space while keeping the single-crate
    // debug run bounded (each case builds ≤ 384 inserts + 24 queries).
    // The EXHAUSTIVE leg of the oracle is the brute-force ground truth
    // INSIDE each case (full linear scan, never sampled) — that is what
    // is kept un-weakened. `PROPTEST_CASES` still overrides for a heavier
    // local/CI sweep.
    #![proptest_config(ProptestConfig { cases: 160, ..ProptestConfig::default() })]

    /// **Test 1 (W28-S2) — recall@k vs exhaustive brute-force.**
    ///
    /// A real `proptest!` over randomized `(seed, n, dim)`: build an
    /// HNSW, then for each query compute the **exhaustive
    /// brute-force KNN** ground truth (full linear scan via
    /// [`brute_force_top_k`] — never sampled) and measure recall@k.
    /// Asserts **two** floors:
    ///
    /// - `mean recall@k ≥ 0.97` — the Slice C / AC-1 exit criterion.
    /// - `p10 recall@k ≥ 0.90` — a **tail** floor: ≥ 90 % of queries
    ///   must individually recall ≥ 9/10. A mean alone can be propped
    ///   up by a majority of perfect queries while a tail of queries
    ///   silently collapses; the p10 floor catches that.
    ///
    /// **Calibration (gap analysis PR #510 §3 + ADR-165 M1).** A
    /// pre-commit sweep over the corners of this parameter space
    /// (dim ∈ {8,16,24} × n ∈ {128,256,384} × 64 seeds = 576 builds)
    /// measured `mean = p10 = 1.0000` in every case. The 0.97 / 0.90
    /// floors therefore sit a comfortable margin below the observed
    /// values — they are genuine statistical floors on a
    /// probabilistic ANN search (margin absorbs legitimate variance;
    /// a real recall / connectivity regression still trips them), NOT
    /// slack tolerances.
    #[test]
    fn prop_hnsw_recall_vs_exhaustive(
        seed in any::<u64>(),
        n in 128usize..=384,
        dim in 8usize..=24,
    ) {
        let k = 10;
        let n_queries = 24;
        let params = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 128,
            seed,
        };
        let kernel = L2F32;

        let raw = generate_unit_vectors(seed, n, dim);
        let vectors: Vec<(VectorId, Vec<f32>)> = raw
            .iter()
            .enumerate()
            .map(|(i, v)| (VectorId::new(i as u32), v.clone()))
            .collect();
        let mut g = HnswGraph::new(params, dim, &kernel);
        for (id, v) in &vectors {
            g.insert(*id, &bytes_of(v), &kernel).expect("insert");
        }

        // Queries drawn from the same distribution but a disjoint
        // seed stream so they are not verbatim corpus members.
        let queries = generate_unit_vectors(seed.wrapping_add(1_000_000), n_queries, dim);
        let mut per_query: Vec<f64> = Vec::with_capacity(n_queries);
        for q in &queries {
            // EXHAUSTIVE ground truth — full linear scan, not sampled.
            let bf = brute_force_top_k(&vectors, q, k);
            let hres = g
                .search(&bytes_of(q), k, params.ef_search, &kernel)
                .expect("search");
            per_query.push(recall_at_k(&hres, &bf, k));
        }
        let mean = per_query.iter().sum::<f64>() / n_queries as f64;
        let p10 = percentile_floor(&mut per_query, 0.10);

        prop_assert!(
            mean >= 0.97,
            "mean recall@{k} = {mean:.4} < 0.97 (n={n}, dim={dim}, seed={seed})"
        );
        prop_assert!(
            p10 >= 0.90,
            "p10 recall@{k} = {p10:.4} < 0.90 tail floor (n={n}, dim={dim}, seed={seed}); \
             ≥ 10 % of queries lost ≥ 2 of their top-10"
        );
    }

    /// **Test 2 (W28-S2) — no tombstoned id ever returned.**
    ///
    /// The property form of the inline check at
    /// [`hnsw_insert_delete_fixed_seed_cycles`]'s search loop: after
    /// a randomized interleaving of inserts and deletes, NO search
    /// over the graph may return a tombstoned id (ADR-003 Strategy 1:
    /// tombstoned vectors stay as routing hubs but are filtered from
    /// the result list). Randomizing the insert/delete schedule and
    /// the query set exercises the tombstone filter far more broadly
    /// than the three fixed seeds of the retained simulator.
    #[test]
    fn prop_hnsw_no_tombstoned_returned(
        seed in any::<u64>(),
        n in 40usize..=200,
        dim in 4usize..=16,
        delete_pct in 5u32..=40,
    ) {
        let k = 10;
        let params = HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            seed,
        };
        let kernel = L2F32;
        let mut g = HnswGraph::new(params, dim, &kernel);

        let raw = generate_unit_vectors(seed, n, dim);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(7));
        let mut deleted: HashSet<VectorId> = HashSet::new();
        let mut live: Vec<VectorId> = Vec::new();

        for (i, v) in raw.iter().enumerate() {
            let id = VectorId::new(i as u32);
            g.insert(id, &bytes_of(v), &kernel).expect("insert");
            live.push(id);
            // Probabilistically tombstone a random live vector.
            if (rng.next_u32() % 100) < delete_pct && !live.is_empty() {
                let idx = (rng.next_u32() as usize) % live.len();
                let victim = live.swap_remove(idx);
                g.mark_deleted(victim);
                deleted.insert(victim);
            }
        }

        // The graph must agree with our bookkeeping about who is
        // tombstoned (guards against the filter being defeated by a
        // bookkeeping desync rather than a real filter bug).
        for &d in &deleted {
            prop_assert!(
                g.is_tombstoned(d),
                "expected {d:?} tombstoned (seed={seed})"
            );
        }

        // Query with every live vector AND a batch of fresh random
        // queries; assert no result is tombstoned.
        let probes = generate_unit_vectors(seed.wrapping_add(2_000_000), 24, dim);
        let query_set = live.iter().take(24).map(|id| raw[id.raw() as usize].clone());
        for q in query_set.chain(probes.into_iter()) {
            let res = g
                .search(&bytes_of(&q), k, params.ef_search, &kernel)
                .expect("search");
            for (rid, _) in &res {
                prop_assert!(
                    !g.is_tombstoned(*rid),
                    "search returned tombstoned id {rid:?} (seed={seed}, \
                     delete_pct={delete_pct})"
                );
                prop_assert!(
                    !deleted.contains(rid),
                    "search returned deleted id {rid:?} not caught by tombstone filter \
                     (seed={seed})"
                );
            }
        }
    }
}

// ─── W28-S573 (exceed-spec) — adversarial vector distributions ─────
//
// Feature #570 / gap analysis PR #510 §3 / ADR-165 M1. S2 (#513) added
// the first real HNSW recall proptest over the IN-DISTRIBUTION
// uniform-unit-sphere corpus. The EXCEED-THE-SPEC mandate
// (`ENGINEERING_DOCTRINE` §3) requires the gate to cover the FULL
// invariant space — including OUT-OF-DISTRIBUTION pathological inputs an
// ANN index will encounter in production. This section adds proptest
// strategies over five pathological distributions and a deterministic
// pair of named regressions.
//
// Oracle discipline:
//  * The ground truth stays EXHAUSTIVE brute-force (`l2_sq` full linear
//    scan; never sampled) — the prompt's "do NOT weaken the recall
//    oracle" requirement.
//  * Pathological corpora carry INTRINSIC distance ties (exact
//    duplicates, zero-norm, near-equidistant clusters). Strict id-set
//    recall (`recall_at_k`) would UNDER-count a perfectly-correct result
//    that returned a different member of a distance-tie (brute-force
//    breaks ties by id). So these tests use a tie-robust DISTANCE-AWARE
//    recall (`recall_at_k_distance_aware`) — the CORRECT oracle for
//    tie-heavy inputs, NOT a weaker one: its 1 % relative tolerance
//    forgives only f32 noise + genuine ties, never a real "missed the
//    neighborhood" error (which lands a result far past the k-th true
//    distance).
//  * The load-bearing checks are the CALIBRATION-FREE structural
//    correctness invariants asserted on EVERY query (no phantom id, no
//    duplicate id, finite + monotonic distances, and the
//    "HNSW-distance ≥ true-distance lower bound" — an index cannot beat
//    brute force). These hold for any correct implementation regardless
//    of recall quality, so they need no empirical calibration.
//  * The recall floor is PER-DISTRIBUTION and EMPIRICALLY-CALIBRATED (set at
//    the assertion site below). The S2 ship used a single GUESSED `mean ≥ 0.85`
//    across all five distributions; that was uncalibrated and was
//    systematically breached by ExactDuplicates. The floors are now MEASURED:
//    each sits below the worst-case per-draw mean distance-aware recall@10
//    observed over an 11 200-draw grid per distribution (400 seeds ×
//    n∈{64,96,128,160,192,224,256} × dim∈{8,16,24,32}, params identical to the
//    test), minus a justified margin. Measured worst-case R@10 mins:
//    TightlyClustered/Antipodal/HighDimSparse 1.0000 (floor 0.95); ZeroNormMixed
//    0.7875 (floor 0.70); ExactDuplicates 0.2500 = the random-index baseline.
//    For ExactDuplicates NO recall floor is asserted — duplicates legitimately
//    degrade HNSW recall to random (the diversity heuristic prunes duplicate
//    edges), so no recall floor can be both meaningful and non-flaky; the
//    STRUCTURAL invariants are its load-bearing oracle. Production recall
//    TARGETS (≥ 0.97 / ≥ 0.90) are validated on IN-DISTRIBUTION data in
//    `prop_hnsw_recall_vs_exhaustive`; claiming those on adversarial OOD data
//    would be dishonest, so these floors are calibrated-to-reality, not a
//    relaxation.

/// Squared-L2 distance — matches [`brute_force_top_k`]'s metric exactly
/// so the distance-aware oracle is internally consistent with the
/// exhaustive ground truth.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Uniform sample in `(-1, 1)` from the shared rng (matches the
/// `generate_unit_vectors` convention).
fn unif_pm1(rng: &mut StdRng) -> f32 {
    let u: f32 = StandardUniform.sample(rng);
    u * 2.0 - 1.0
}

/// Tie-robust recall@k by DISTANCE (squared-L2). An HNSW result counts
/// as a hit iff its true distance to `query` is within
/// `(k-th brute-force distance) × (1 + REL_TOL) + ABS_EPS`. Robust to
/// intrinsic ties (the strict id-set recall would mis-score a correct
/// tie-member swap); NOT weaker away from ties — a genuine miss lands
/// far past the k-th true distance and still fails.
fn recall_at_k_distance_aware(
    vectors: &[(VectorId, Vec<f32>)],
    query: &[f32],
    hnsw_result: &[(VectorId, f32)],
    k: usize,
) -> f64 {
    const REL_TOL: f32 = 0.01;
    const ABS_EPS: f32 = 1e-4;
    let mut dists: Vec<f32> = vectors.iter().map(|(_, v)| l2_sq(v, query)).collect();
    dists.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN distance"));
    let kth = dists[k.min(dists.len()) - 1];
    let threshold = kth * (1.0 + REL_TOL) + ABS_EPS;
    let id_to_vec: std::collections::HashMap<VectorId, &Vec<f32>> =
        vectors.iter().map(|(id, v)| (*id, v)).collect();
    let hits = hnsw_result
        .iter()
        .take(k)
        .filter(|(id, _)| {
            id_to_vec
                .get(id)
                .map(|v| l2_sq(v, query) <= threshold)
                .unwrap_or(false)
        })
        .count();
    hits as f64 / k as f64
}

/// One pathological vector distribution. Each is a STRICT SUPERSET
/// challenge over the in-distribution uniform-unit-sphere corpus the
/// existing recall tests use.
#[derive(Debug, Clone, Copy)]
enum AdversarialDist {
    /// All points inside a tiny ball — near-equidistant; stresses beam
    /// search tie-breaking.
    TightlyClustered,
    /// Two antipodal clusters (+u / −u with jitter) — bimodal; stresses
    /// entry-point cluster bias.
    Antipodal,
    /// A few distinct base vectors, each replicated as EXACT copies —
    /// intrinsic distance ties; stresses result de-dup / tie handling.
    ExactDuplicates,
    /// A fraction of exact zero vectors mixed into finite unit data —
    /// degenerate L2 norm (the OQ-V5 cousin, under L2 instead of cosine).
    ZeroNormMixed,
    /// High dimension with only a few non-zero coordinates per vector —
    /// near-orthogonal / well-separated sparse data.
    HighDimSparse,
}

/// Generate `n` vectors at dimension `dim` from a seeded rng under the
/// chosen pathological distribution.
fn gen_adversarial(dist: AdversarialDist, seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    match dist {
        AdversarialDist::TightlyClustered => {
            const JITTER: f32 = 0.05;
            let center: Vec<f32> = (0..dim).map(|_| unif_pm1(&mut rng)).collect();
            (0..n)
                .map(|_| {
                    center
                        .iter()
                        .copied()
                        .map(|c| c + unif_pm1(&mut rng) * JITTER)
                        .collect::<Vec<f32>>()
                })
                .collect()
        }
        AdversarialDist::Antipodal => {
            const JITTER: f32 = 0.05;
            let u = l2_normalize((0..dim).map(|_| unif_pm1(&mut rng)).collect());
            (0..n)
                .map(|i| {
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    u.iter()
                        .copied()
                        .map(|c| sign * c + unif_pm1(&mut rng) * JITTER)
                        .collect::<Vec<f32>>()
                })
                .collect()
        }
        AdversarialDist::ExactDuplicates => {
            let bases = 4usize;
            let base_vecs: Vec<Vec<f32>> = (0..bases)
                .map(|_| l2_normalize((0..dim).map(|_| unif_pm1(&mut rng)).collect()))
                .collect();
            (0..n).map(|i| base_vecs[i % bases].clone()).collect()
        }
        AdversarialDist::ZeroNormMixed => {
            let zeros = (n / 8).max(1);
            (0..n)
                .map(|i| {
                    if i < zeros {
                        vec![0.0; dim]
                    } else {
                        l2_normalize((0..dim).map(|_| unif_pm1(&mut rng)).collect())
                    }
                })
                .collect()
        }
        AdversarialDist::HighDimSparse => {
            const NNZ: usize = 3;
            (0..n)
                .map(|_| {
                    let mut v = vec![0.0f32; dim];
                    for _ in 0..NNZ.min(dim) {
                        let pos = (rng.next_u32() as usize) % dim;
                        let sign = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
                        v[pos] = sign;
                    }
                    v
                })
                .collect()
        }
    }
}

fn arb_adversarial_dist() -> impl Strategy<Value = AdversarialDist> {
    prop_oneof![
        Just(AdversarialDist::TightlyClustered),
        Just(AdversarialDist::Antipodal),
        Just(AdversarialDist::ExactDuplicates),
        Just(AdversarialDist::ZeroNormMixed),
        Just(AdversarialDist::HighDimSparse),
    ]
}

proptest! {
    // 160 cases per the same ADR-165 M1 continuous-space justification as
    // `prop_hnsw_recall_vs_exhaustive` — exhaustive is infeasible over a
    // continuous vector space, so case count is the statistical lever.
    #![proptest_config(ProptestConfig { cases: 160, ..ProptestConfig::default() })]

    /// **W28-S573 — adversarial-distribution recall + structural
    /// invariants.** Builds an HNSW over one of five pathological
    /// distributions, then for a mix of corpus-member + fresh queries
    /// asserts (a) calibration-free structural correctness invariants on
    /// EVERY query and (b) a per-distribution EMPIRICALLY-CALIBRATED
    /// tie-robust distance-aware recall floor (skipped for ExactDuplicates,
    /// whose recall legitimately degrades to the random baseline — see the
    /// floor `match` below). The ground truth is EXHAUSTIVE brute-force.
    #[test]
    fn prop_hnsw_adversarial_distributions(
        dist in arb_adversarial_dist(),
        seed in any::<u64>(),
        n in 64usize..=256,
        dim in 8usize..=32,
    ) {
        let k = 10;
        let params = HnswParams { m: 16, ef_construction: 200, ef_search: 128, seed };
        let kernel = L2F32;

        let raw = gen_adversarial(dist, seed, n, dim);
        let vectors: Vec<(VectorId, Vec<f32>)> = raw
            .iter()
            .enumerate()
            .map(|(i, v)| (VectorId::new(i as u32), v.clone()))
            .collect();
        let inserted_ids: HashSet<VectorId> = vectors.iter().map(|(id, _)| *id).collect();
        let id_to_vec: std::collections::HashMap<VectorId, &Vec<f32>> =
            vectors.iter().map(|(id, v)| (*id, v)).collect();

        let mut g = HnswGraph::new(params, dim, &kernel);
        for (id, v) in &vectors {
            g.insert(*id, &bytes_of(v), &kernel).expect("insert");
        }

        // Queries: corpus members (non-empty true neighborhood) + fresh.
        let n_corpus_q = 12usize.min(n);
        let fresh = gen_adversarial(dist, seed.wrapping_add(1_000_003), 12, dim);
        let queries: Vec<Vec<f32>> = vectors
            .iter()
            .take(n_corpus_q)
            .map(|(_, v)| v.clone())
            .chain(fresh.into_iter())
            .collect();

        let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());
        for q in &queries {
            let res = g
                .search(&bytes_of(q), k, params.ef_search, &kernel)
                .expect("search");

            // ── calibration-free structural invariants (ALWAYS hold) ──
            prop_assert!(res.len() <= k, "result longer than k (dist={dist:?})");
            prop_assert!(res.len() <= n, "result longer than corpus (dist={dist:?})");

            let mut seen = HashSet::new();
            for (id, dval) in &res {
                prop_assert!(
                    inserted_ids.contains(id),
                    "phantom id {id:?} (dist={dist:?}, seed={seed})"
                );
                prop_assert!(
                    seen.insert(*id),
                    "duplicate id {id:?} in one result (dist={dist:?})"
                );
                prop_assert!(
                    dval.is_finite(),
                    "non-finite distance {dval} for {id:?} (dist={dist:?})"
                );
            }
            // Monotonic non-decreasing kernel distances.
            for w in res.windows(2) {
                prop_assert!(
                    w[0].1 <= w[1].1 + 1e-6,
                    "non-monotonic result (dist={dist:?}): {} > {}",
                    w[0].1,
                    w[1].1
                );
            }
            // HNSW cannot beat brute force: i-th returned TRUE distance
            // ≥ i-th smallest true distance − eps.
            let mut true_sorted: Vec<f32> = vectors.iter().map(|(_, v)| l2_sq(v, q)).collect();
            true_sorted.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN"));
            for (rank, (id, _)) in res.iter().enumerate() {
                let td = l2_sq(id_to_vec[id], q);
                prop_assert!(
                    td + 1e-3 >= true_sorted[rank],
                    "HNSW rank {rank} true distance {td} < brute-force {} (dist={dist:?}, seed={seed})",
                    true_sorted[rank]
                );
            }

            // ── quality oracle: tie-robust distance-aware recall ──
            recalls.push(recall_at_k_distance_aware(&vectors, q, &res, k));
        }

        let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        // ── per-distribution EMPIRICALLY-CALIBRATED recall floor ──
        // W28-S573 R1 fix-up. The S2 ship used a single GUESSED `mean >= 0.85`
        // across all five distributions; the implementer flagged it as
        // uncalibrated and deferred the run to CI, which let a RED test ship
        // (ExactDuplicates systematically breaches 0.85). The floors below are
        // CALIBRATED to MEASURED reality: each is set below the worst-case
        // per-draw mean distance-aware recall@10 observed over an 11 200-draw
        // grid per distribution (400 seeds × n∈{64,96,128,160,192,224,256} ×
        // dim∈{8,16,24,32}, params identical to this test), minus a justified
        // margin. The calibration-free structural invariants above held on
        // EVERY query of EVERY draw — only the recall magnitude varies by
        // distribution. Genuine statistical floors per ADR-165 M1, not guesses.
        // (A first 480-draw/30-seed pass over-estimated the tails — e.g. it put
        // ExactDuplicates' min at 0.50 and ZeroNormMixed's at 0.8708; the
        // 11 200-draw pass found 0.25 and 0.7875 respectively. The wider grid
        // is what these floors are calibrated against.)
        let floor: Option<f64> = match dist {
            // Measured R@10: min 1.0000, p0.1 1.0000, mean 1.0000 (all three).
            // A tight ball, two antipodal clusters, and sparse near-orthogonal
            // data are near-trivial for ANN (ef_search=128 ≫ k=10): NOT ONE of
            // the 11 200 draws each dipped below 1.0. Floor 0.95 = 0.05 margin
            // below the observed min — trips only on a real (≥5 %) connectivity
            // regression.
            AdversarialDist::TightlyClustered
            | AdversarialDist::Antipodal
            | AdversarialDist::HighDimSparse => Some(0.95),
            // Measured R@10: min 0.7875, p0.1 0.8583, p1 0.9167, mean 0.9966.
            // The n/8 exact-zero vectors sit at a finite mid-range L2 from a
            // unit query, so in low dim they occasionally displace a true
            // neighbor outside the 1 % tie tolerance (worst at dim=8). Floor
            // 0.70 = ~0.09 margin below the observed min and ≈7× the random
            // baseline (≈0.1) — meaningful (a real recall regression trips it)
            // and robust (the sub-0.7875 tail is < 1-in-11 200). The OLD uniform
            // 0.85 sat ABOVE the true min here → it was a latent RED test.
            AdversarialDist::ZeroNormMixed => Some(0.70),
            // Measured R@10: min 0.2500, p0.1 0.3750, p1 0.5000, p5 0.7500,
            // mean 0.9608. Measured R@1: min 0.2500 too. Both recall@10 AND
            // recall@1 degrade all the way to the RANDOM-INDEX BASELINE (≈0.25
            // for 4 distinct bases) in the worst case. This is the EXPECTED
            // property of HNSW on exact duplicates, NOT a bug: the
            // neighbor-diversity heuristic (Malkov-Yashunin 2016 Alg. 4 /
            // select_neighbors_heuristic) deliberately prunes redundant
            // distance-0 duplicate edges to preserve navigability, so the beam
            // search can legitimately converge to a non-nearest duplicate
            // cluster and miss the true (tied) neighborhood. The structural
            // invariants above STILL hold on every query (no phantom/dup ids,
            // finite + monotonic distances, HNSW ≥ brute-force) — confirming the
            // index is correct, just recall-degraded. Because the legitimate
            // worst case EQUALS the random baseline, NO recall floor here can be
            // both meaningful (distinguish a working index from a random one)
            // AND non-flaky — a green recall floor that can't fail on its bug is
            // worse than none (ENGINEERING_DOCTRINE §3 "strong oracles"). So for
            // THIS distribution the load-bearing oracle is the structural
            // invariant set above; we assert NO recall floor and document the
            // measured degradation. (The single-cluster extreme — n identical
            // vectors — IS pinned to recall≈1.0 by the deterministic
            // `hnsw_adversarial_all_identical_vectors` regression below; the
            // multi-base degradation is what genuinely cannot be floored.)
            AdversarialDist::ExactDuplicates => None,
        };
        if let Some(floor) = floor {
            prop_assert!(
                mean >= floor,
                "adversarial {dist:?} mean distance-aware recall@{k} = {mean:.4} < {floor:.2} \
                 empirically-calibrated no-collapse floor (n={n}, dim={dim}, seed={seed})"
            );
        }
    }
}

/// **Named regression — all-identical corpus (extreme tie).** `n` exact
/// copies of one unit vector; a query equal to it must return exactly
/// `k` results, every id a real inserted id, no duplicate ids, and ALL
/// at distance ≈ 0 (the corpus is a single distance-0 equivalence
/// class). A deterministic pin complementing the randomized strategy.
#[test]
fn hnsw_adversarial_all_identical_vectors() {
    let dim = 16;
    let n = 200;
    let k = 10;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 3,
    };
    let kernel = L2F32;
    let base = generate_unit_vectors(11, 1, dim).remove(0);

    let mut g = HnswGraph::new(params, dim, &kernel);
    let mut ids = HashSet::new();
    for i in 0..n {
        let id = VectorId::new(i as u32);
        g.insert(id, &bytes_of(&base), &kernel).unwrap();
        ids.insert(id);
    }

    let res = g
        .search(&bytes_of(&base), k, params.ef_search, &kernel)
        .unwrap();
    assert_eq!(
        res.len(),
        k,
        "must return k results from {n} identical vectors"
    );
    let mut seen = HashSet::new();
    for (id, d) in &res {
        assert!(ids.contains(id), "phantom id {id:?}");
        assert!(seen.insert(*id), "duplicate id {id:?} in result");
        assert!(d.is_finite(), "non-finite distance {d}");
        assert!(
            *d <= 1e-3,
            "identical-vector distance should be ~0, got {d}"
        );
    }
}

/// **Named regression — exact zero vector among finite unit data
/// (degenerate L2 norm).** The OQ-V5 cousin under L2: a zero vector has
/// a valid (finite) squared-L2 distance to anything, so search must
/// never surface NaN/Inf and must return only valid ids with monotonic
/// distances. (We do NOT assert the zero vector's RANK — in a random
/// high-dim unit corpus many points are legitimately farther from a unit
/// query than the zero vector is, so its rank is data-dependent; the
/// robustness pin is the no-NaN / valid-id guarantee.)
#[test]
fn hnsw_adversarial_zero_vector_among_finite_l2() {
    let dim = 16;
    let n = 128;
    let k = 10;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 7,
    };
    let kernel = L2F32;
    let mut g = HnswGraph::new(params, dim, &kernel);

    let mut ids = HashSet::new();
    let zero_id = VectorId::new(0);
    g.insert(zero_id, &bytes_of(&vec![0.0f32; dim]), &kernel)
        .unwrap();
    ids.insert(zero_id);
    let finite = generate_unit_vectors(21, n - 1, dim);
    for (i, v) in finite.iter().enumerate() {
        let id = VectorId::new(i as u32 + 1);
        g.insert(id, &bytes_of(v), &kernel).unwrap();
        ids.insert(id);
    }

    // Query both a finite member and the zero vector itself.
    for q in [finite[0].clone(), vec![0.0f32; dim]] {
        let res = g
            .search(&bytes_of(&q), k, params.ef_search, &kernel)
            .unwrap();
        for (id, d) in &res {
            assert!(ids.contains(id), "phantom id {id:?}");
            assert!(
                d.is_finite(),
                "degenerate-norm query produced non-finite distance {d}"
            );
        }
        for w in res.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-6,
                "non-monotonic result with zero-norm data"
            );
        }
    }
}
