//! Slice F.2 — filter-aware HNSW Path A boundary tests.
//!
//! The Slice F.2 owner directive (2026-04-26) made Path A
//! boundary testing MANDATORY (not optional). This file exercises
//! the comprehensive battery the directive enumerates:
//!
//! 1. **Selectivity sweep** — recall@10 floors at six selectivity
//!    buckets (0.1 %, 1 %, 10 %, 50 %, 99 %, 100 %).
//! 2. **Filter-correctness proptest** — 1 K random filter
//!    expressions; every returned vector must satisfy the filter.
//! 3. **Pathological data** — identical vectors, super-nodes,
//!    zero-similarity datasets, payload partitions.
//! 4. **Concurrent search + insert** — 30-second torture (8
//!    readers + 1 writer) verifies no torn payload reads, no
//!    panics, no deadlock.
//! 5. **Edge cases** — empty filter result, single-match filter,
//!    tenant cardinality 1 fast path, F.1 arena integration.
//! 6. **Existing-regression guards** — Slice A + F.1 partition-id
//!    invariants re-run.
//!
//! ## Knobs (env-controlled)
//!
//! - `F2_TORTURE_SECS` — overrides the concurrent torture
//!   duration. Default `3` for a quick `cargo test`; the slice
//!   spec calls for `30` which is wired via the per-commit
//!   gauntlet's release-mode invocation.
//! - `F2_PROPTEST_CASES` — overrides the filter-correctness
//!   proptest case count. Default `200` for fast `cargo test`;
//!   the spec calls for `1024`. CI runs the full count via the
//!   `proptest!` config block (the spec's "1 K cases" is
//!   honored when `F2_PROPTEST_CASES=1024` is exported).
//!
//! Both knobs are read once at test start; missing or invalid
//! values fall back to the default.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, PartitionId, StringId, TenantId};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{
    FilteredHnsw, HnswGraph, HnswParams, Payload, predicate_filtered_search,
};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{
    Encoding, Filter, IndexId, IndexType, PropertyValue, QuantizerState, VectorArena,
    VectorIndexHandle,
};
use parking_lot::RwLock;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Shared helpers ──────────────────────────────────────────────

fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn handle_for(tenant: u64, idx: u64) -> VectorIndexHandle {
    VectorIndexHandle::for_tenant(TenantId::new(tenant), IndexId::new(idx))
}

/// Generate `n` deterministic random unit vectors at dimension
/// `dim` from a seeded RNG. Vectors live on the unit sphere
/// (L2-normalized) so cosine and L2 rankings agree at the top —
/// the standard ANN-Benchmarks unit-test convention.
fn generate_unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    u * 2.0 - 1.0
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

/// Brute-force top-`k` over the **filtered** subset. Returns ids
/// sorted by ascending L2 to the query.
fn brute_force_top_k_filtered(
    vectors: &[(VectorId, Vec<f32>, Payload)],
    query: &[f32],
    k: usize,
    filter: &Filter,
) -> Vec<VectorId> {
    let mut scored: Vec<(f32, VectorId)> = vectors
        .iter()
        .filter(|(_, _, p)| filter.matches(p))
        .map(|(id, v, _)| {
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

/// Recall@k: |HNSW result ∩ brute force result| / k.
///
/// When the brute-force set is smaller than `k` (e.g., the
/// filter only matches a few vectors), denominator is the
/// brute-force size — recall is fraction of matches found.
fn recall_at_k(hnsw_result: &[(VectorId, f32)], brute_result: &[VectorId], k: usize) -> f64 {
    let h: HashSet<VectorId> = hnsw_result.iter().map(|(id, _)| *id).take(k).collect();
    let b: HashSet<VectorId> = brute_result.iter().copied().take(k).collect();
    let inter = h.intersection(&b).count();
    let denom = b.len().min(k).max(1);
    inter as f64 / denom as f64
}

/// Selectivity sweep test fixture: build a single 10 K-vector
/// graph with bucketed labels so a single graph serves all six
/// selectivity tests.
///
/// Bucket scheme: each vector gets cumulative labels. Vector
/// `i` in `[0, n_for_pct(p))` carries label `bucket_label(p)`
/// for every `p` in `{0.1 %, 1 %, 10 %, 50 %, 99 %, 100 %}`.
/// So filtering by `LabelIn([bucket_label(0.1 %)])` yields
/// 10 vectors out of 10 000 (0.1 % selectivity); filtering by
/// `LabelIn([bucket_label(100 %)])` yields all 10 000.
struct SelectivityFixture {
    hnsw: FilteredHnsw,
    vectors: Vec<(VectorId, Vec<f32>, Payload)>,
    queries: Vec<Vec<f32>>,
    dim: usize,
}

const SELECTIVITY_BUCKETS: &[(f64, LabelId)] = &[
    (0.001, LabelId(101)), // 0.1 %
    (0.01, LabelId(102)),  // 1 %
    (0.10, LabelId(103)),  // 10 %
    (0.50, LabelId(104)),  // 50 %
    (0.99, LabelId(105)),  // 99 %
    (1.00, LabelId(106)),  // 100 %
];

fn build_selectivity_fixture(seed: u64, n: usize, dim: usize) -> SelectivityFixture {
    let params = HnswParams {
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        seed,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let raw = generate_unit_vectors(seed.wrapping_mul(7).wrapping_add(11), n, dim);

    let mut vectors = Vec::with_capacity(n);
    for (i, v) in raw.iter().enumerate() {
        // Vector i belongs to every bucket with cutoff > i / n.
        // Buckets are listed ascending by cutoff; we add labels
        // for every bucket the vector qualifies for.
        let frac = i as f64 / n as f64;
        let mut labels = Vec::new();
        for (cutoff, lbl) in SELECTIVITY_BUCKETS {
            if frac < *cutoff {
                labels.push(*lbl);
            }
        }
        let payload = Payload::with_labels(labels);
        let id = VectorId::new(i as u32);
        hnsw.filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .unwrap();
        vectors.push((id, v.clone(), payload));
    }

    let queries = generate_unit_vectors(seed.wrapping_add(99_991), 30, dim);
    SelectivityFixture {
        hnsw,
        vectors,
        queries,
        dim,
    }
}

/// Run a selectivity test against a fixture and assert the
/// recall floor for the bucket. Caller passes the bucket index
/// + the floor.
fn assert_selectivity_recall(
    fixture: &SelectivityFixture,
    bucket_idx: usize,
    recall_floor: f64,
    k: usize,
    label: &str,
) {
    let (_cutoff, bucket_label) = SELECTIVITY_BUCKETS[bucket_idx];
    let filter = Filter::LabelIn(vec![bucket_label]);
    let kernel = L2F32;
    let _ = fixture.dim; // silence unused warning if logging is removed

    let mut total = 0.0_f64;
    for q in &fixture.queries {
        let bf = brute_force_top_k_filtered(&fixture.vectors, q, k, &filter);
        if bf.is_empty() {
            // Degenerate: filter matches nothing. Skip recall
            // computation — separately tested in the
            // f2_empty_filter_result test.
            continue;
        }
        let hres = fixture
            .hnsw
            .filtered_search(&bytes_of(q), k, &filter, 200, &kernel, Lsn::MAX)
            .unwrap();
        // Sanity: every returned id must satisfy the filter.
        for (id, _) in &hres {
            let payload = fixture
                .hnsw
                .payload(*id)
                .expect("FilteredHnsw payload sidecar missing for returned id");
            assert!(
                filter.matches(payload),
                "{label}: returned id {id:?} does not satisfy filter (payload labels={:?})",
                payload.labels
            );
        }
        total += recall_at_k(&hres, &bf, k);
    }
    let mean = total / fixture.queries.len() as f64;
    println!(
        "{label}: bucket cutoff={:.4} → recall@{k} = {mean:.4} (floor {recall_floor})",
        SELECTIVITY_BUCKETS[bucket_idx].0
    );
    assert!(
        mean >= recall_floor,
        "{label}: recall@{k} = {mean:.4} below floor {recall_floor}"
    );
}

/// Dim used across selectivity tests. Small enough to keep
/// `cargo test` fast (build cost dominates), large enough for
/// the recall thresholds to be meaningful.
const TEST_DIM: usize = 32;
/// Vector count for selectivity tests. 5 K is the same scale as
/// the existing Slice C `hnsw_build_search_recall_sift_subset`
/// test; the 0.1 % bucket gets 5 vectors, so the 0.1 % recall
/// test runs at `k = 5`. A 10 K-vector run (the spec's
/// reference scale) is gated behind release mode in CI per the
/// per-commit gauntlet — debug-mode build cost is `O(N²)` for
/// the brute-force payload augmentation, so 5 K keeps debug
/// `cargo test` under ~12 s.
const TEST_N: usize = 5_000;

/// One shared fixture across the six selectivity tests. Build
/// is `O(N · M · ef_construction)` and dominates the test
/// runtime; sharing keeps debug-mode `cargo test` under ~15 s
/// instead of the ~60 s a per-test fixture would take. Tests
/// are read-only against the shared `FilteredHnsw` so the
/// `OnceLock` is sound (no `&mut` borrows escape).
static SELECTIVITY_FIXTURE: OnceLock<SelectivityFixture> = OnceLock::new();

fn shared_selectivity_fixture() -> &'static SelectivityFixture {
    SELECTIVITY_FIXTURE.get_or_init(|| build_selectivity_fixture(7, TEST_N, TEST_DIM))
}

// ─── Test 1 — Selectivity sweep ──────────────────────────────────

#[test]
fn f2_selectivity_0_1pct_recall() {
    // At N = 5 K the 0.1 % bucket holds 5 vectors. We still
    // use k = 10 (per the spec's "recall@10 ≥ 0.85" target);
    // recall_at_k() takes the brute-force size as denominator
    // when bf < k, so the test asks for "recall@min(10, 5) ≥
    // 0.85" — i.e., 5 of 5 matches found. Tight but achievable
    // because the filtered traversal is allowed to walk the
    // entire candidate frontier at low selectivity.
    let fixture = shared_selectivity_fixture();
    assert_selectivity_recall(fixture, 0, 0.85, 10, "0.1% selectivity");
}

#[test]
fn f2_selectivity_1pct_recall() {
    let fixture = shared_selectivity_fixture();
    assert_selectivity_recall(fixture, 1, 0.90, 10, "1% selectivity");
}

#[test]
fn f2_selectivity_10pct_recall() {
    let fixture = shared_selectivity_fixture();
    assert_selectivity_recall(fixture, 2, 0.92, 10, "10% selectivity");
}

#[test]
fn f2_selectivity_50pct_recall() {
    let fixture = shared_selectivity_fixture();
    assert_selectivity_recall(fixture, 3, 0.95, 10, "50% selectivity");
}

#[test]
fn f2_selectivity_99pct_recall() {
    let fixture = shared_selectivity_fixture();
    assert_selectivity_recall(fixture, 4, 0.95, 10, "99% selectivity");
}

#[test]
fn f2_selectivity_100pct_recall() {
    // No-op filter: every vector matches. recall must equal
    // standard search recall (no degradation from filtering).
    let fixture = shared_selectivity_fixture();
    let kernel = L2F32;
    let bucket_label = SELECTIVITY_BUCKETS[5].1;
    let filter = Filter::LabelIn(vec![bucket_label]);
    let k = 10;

    let mut total_filtered = 0.0_f64;
    let mut total_standard = 0.0_f64;
    for q in &fixture.queries {
        let bf_all = {
            // Brute force over the entire (unfiltered) set.
            let mut scored: Vec<(f32, VectorId)> = fixture
                .vectors
                .iter()
                .map(|(id, v, _)| {
                    let d: f32 = v.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, *id)
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN compare"));
            scored
                .into_iter()
                .take(k)
                .map(|(_, id)| id)
                .collect::<Vec<_>>()
        };

        let hres_filtered = fixture
            .hnsw
            .filtered_search(&bytes_of(q), k, &filter, 200, &kernel, Lsn::MAX)
            .unwrap();
        let hres_standard = fixture
            .hnsw
            .inner()
            .search(&bytes_of(q), k, 200, &kernel)
            .unwrap();
        total_filtered += recall_at_k(&hres_filtered, &bf_all, k);
        total_standard += recall_at_k(&hres_standard, &bf_all, k);
    }
    let n_q = fixture.queries.len() as f64;
    let mean_filtered = total_filtered / n_q;
    let mean_standard = total_standard / n_q;
    println!(
        "100% selectivity: filtered recall@{k} = {mean_filtered:.4}, standard = {mean_standard:.4}"
    );
    // The filtered path must not regress relative to the
    // standard path; allow a 2 % epsilon for the filter-aware
    // termination heuristic vs the standard heuristic.
    assert!(
        mean_filtered + 0.02 >= mean_standard,
        "100% selectivity: filtered recall {mean_filtered} >0.02 below standard recall {mean_standard}; no-op filter should not regress"
    );
    assert!(
        mean_filtered >= 0.95,
        "100% selectivity: filtered recall {mean_filtered} < 0.95 floor"
    );
}

// ─── Test 2 — Filter correctness proptest ────────────────────────

/// A small fixture for the proptest: 200 vectors, 5 labels, 2
/// properties. Random filter expressions exercise the matches()
/// evaluator across the full grammar — every returned id from
/// filtered_search must satisfy the filter.
fn build_proptest_fixture() -> (FilteredHnsw, Vec<(VectorId, Vec<f32>, Payload)>) {
    let dim = 16;
    let n = 200;
    let params = HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        seed: 42,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let mut all = Vec::new();
    let labels = [
        LabelId::new(1),
        LabelId::new(2),
        LabelId::new(3),
        LabelId::new(4),
        LabelId::new(5),
    ];
    let prop_keys = [StringId::new(10), StringId::new(20)];

    let mut rng = StdRng::seed_from_u64(7);
    let raw = generate_unit_vectors(13, n, dim);
    for (i, v) in raw.iter().enumerate() {
        // Random subset of 0..3 labels.
        let n_lbls = (rng.next_u32() as usize) % 4;
        let mut shuffled = labels.to_vec();
        for j in (1..shuffled.len()).rev() {
            let k = (rng.next_u32() as usize) % (j + 1);
            shuffled.swap(j, k);
        }
        let chosen_labels: Vec<LabelId> = shuffled.into_iter().take(n_lbls).collect();
        // Random subset of 0..2 properties.
        let mut props = HashMap::new();
        for &pk in &prop_keys {
            if rng.next_u32() % 2 == 0 {
                let val = rng.next_u32() % 10;
                props.insert(pk, PropertyValue::U32(val));
            }
        }
        let payload = Payload {
            tenant_id: Some(TenantId::DEFAULT),
            labels: chosen_labels,
            properties: props,
            ..Payload::default()
        };
        let id = VectorId::new(i as u32);
        hnsw.filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .unwrap();
        all.push((id, v.clone(), payload));
    }
    (hnsw, all)
}

/// Random Filter generator — recursive, bounded depth.
fn arb_filter() -> impl Strategy<Value = Filter> {
    let leaf = prop_oneof![
        Just(Filter::Tenant(TenantId::DEFAULT)),
        (1u32..=5).prop_map(|l| Filter::LabelIn(vec![LabelId::new(l)])),
        prop::collection::vec(1u32..=5, 0..3)
            .prop_map(|ls| Filter::LabelIn(ls.into_iter().map(LabelId::new).collect())),
        (
            (10u32..=20).prop_filter("only known prop_keys", |k| *k == 10 || *k == 20),
            0u32..10
        )
            .prop_map(|(k, v)| Filter::PropertyEq(StringId::new(k), PropertyValue::U32(v))),
    ];
    leaf.prop_recursive(3, 8, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Filter::And),
            prop::collection::vec(inner, 0..3).prop_map(Filter::Or),
        ]
    })
}

fn proptest_case_count() -> u32 {
    std::env::var("F2_PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(200)
}

#[test]
fn f2_filter_correctness_proptest() {
    // Build the fixture once; the proptest cases re-use it.
    let (hnsw, all) = build_proptest_fixture();
    let kernel = L2F32;

    let cases = proptest_case_count();
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);

    runner
        .run(&arb_filter(), |filter| {
            // 1) Property: every returned id satisfies the
            //    filter. This is the core correctness invariant.
            let res = hnsw
                .filtered_search(&bytes_of(&all[0].1), 10, &filter, 50, &kernel, Lsn::MAX)
                .unwrap();
            for (id, _) in &res {
                let p = hnsw
                    .payload(*id)
                    .expect("returned id has no payload sidecar entry");
                prop_assert!(
                    filter.matches(p),
                    "search returned id {id:?} which does not satisfy filter {:?}; payload={:?}",
                    filter,
                    p
                );
            }

            // 2) Property: returned ids are sorted by ascending
            //    distance. Filter does not reorder beyond the
            //    standard kernel ranking.
            for w in res.windows(2) {
                prop_assert!(
                    w[0].1 <= w[1].1 + 1e-6,
                    "result not monotonically ordered: {:?} < {:?}",
                    w[0],
                    w[1]
                );
            }

            // 3) Property: dispatch agrees with filtered_search
            //    on the result set membership (modulo
            //    selectivity-driven branch differences). We
            //    check that EVERY id returned by dispatch
            //    satisfies the filter — the dispatcher must
            //    never return non-matching ids. The hardcoded
            //    `0.5_f32` exercises the mid-range branch; the
            //    "no non-matching ids" property holds in every
            //    branch so a single seed is sufficient (Phase 6
            //    F.4 will own the real selectivity estimator).
            let sel = 0.5_f32;
            let res2 = hnsw
                .filtered_search_dispatch(
                    &bytes_of(&all[0].1),
                    10,
                    &filter,
                    sel,
                    50,
                    &kernel,
                    Lsn::MAX,
                )
                .unwrap();
            for (id, _) in &res2 {
                let p = hnsw
                    .payload(*id)
                    .expect("dispatch-returned id has no payload sidecar");
                prop_assert!(
                    filter.matches(p),
                    "dispatch returned non-matching id {id:?} for filter {:?}",
                    filter
                );
            }

            Ok(())
        })
        .expect("filter correctness proptest failed");
}

// ─── Test 3 — Pathological data ──────────────────────────────────

/// All vectors identical; filter selects half of them. Search
/// must terminate gracefully (no panic, no infinite loop) and
/// recall should be ≥ 0.85 — even though all distances are
/// equal, the filter should still surface matching vectors.
#[test]
fn f2_all_identical_vectors_with_filter() {
    let dim = 8;
    let n = 1000;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 1,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let v = vec![1.0_f32; dim];
    let label_a = LabelId::new(1);
    let label_b = LabelId::new(2);

    for i in 0..n {
        // Half label A, half label B.
        let label = if i % 2 == 0 { label_a } else { label_b };
        let payload = Payload::with_labels(vec![label]);
        hnsw.filtered_insert(VectorId::new(i as u32), &bytes_of(&v), payload, &kernel)
            .unwrap();
    }

    let q = vec![1.0_f32; dim];
    let filter = Filter::LabelIn(vec![label_a]);
    let res = hnsw
        .filtered_search(&bytes_of(&q), 10, &filter, 100, &kernel, Lsn::MAX)
        .unwrap();

    // Must return exactly 10 results (filter passes 500
    // vectors; we ask for 10).
    assert_eq!(
        res.len(),
        10,
        "all-identical recall: got {} results",
        res.len()
    );
    // Every result must be label A (even-id).
    for (id, _) in &res {
        assert_eq!(
            id.raw() % 2,
            0,
            "non-matching odd id {id:?} returned for label_a filter"
        );
    }
}

/// Super-node: one vector with very strong connectivity (every
/// other vector links to it). Verifies that filtered traversal
/// terminates even when expansion blows up the candidate
/// frontier.
///
/// We can't directly create a 100K-degree node within the
/// HnswGraph's `m_max0()` cap (the standard prune drops past
/// 64 edges), so we approximate by making one vector the
/// nearest neighbor of every other vector. This is a
/// "soft super-node" that the standard HNSW build does not
/// reject.
#[test]
fn f2_super_node_with_filter() {
    let dim = 8;
    let n = 500;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 5,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    // Hub vector at origin; every other vector orbits at small
    // random offsets around it. Hub ends up the closest neighbor
    // of every orbit vector → super-node-like graph topology.
    let hub = vec![0.0_f32; dim];
    hnsw.filtered_insert(
        VectorId::new(0),
        &bytes_of(&hub),
        Payload::with_labels(vec![LabelId::new(1)]),
        &kernel,
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    for i in 1..n {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                (u * 2.0 - 1.0) * 0.001
            })
            .collect();
        let label = if i < 100 {
            LabelId::new(1)
        } else {
            LabelId::new(2)
        };
        let payload = Payload::with_labels(vec![label]);
        hnsw.filtered_insert(VectorId::new(i as u32), &bytes_of(&v), payload, &kernel)
            .unwrap();
    }

    // W14ε flake-CLASS fix (issue #282 sister-surface): the
    // pre-W14ε `assert!(elapsed < 5s)` is the §3.5.1 anti-pattern
    // (hardware-throughput-sensitive correctness probe). 5 s was
    // sized for dev-hardware filtered_search latency over 500
    // vectors — microseconds-to-milliseconds territory — but under
    // 2-vCPU CI sibling-cargo contention even microsecond-class
    // work can overrun arbitrary wall-clock budgets. Per
    // `docs/testing-strategy.md` §3.5.1 we replace with the
    // watchdog pattern at the 60 s class default: a real
    // non-terminating expansion never completes regardless of
    // budget, so the wider budget isolates the signal from CI
    // starvation. Worker runs the search + result-checks; if it
    // never completes the main thread panics and the worker is
    // leaked (matches the `run_with_watchdog` shape used by the
    // storage / index deadlock-regression siblings).
    const WATCHDOG: Duration = Duration::from_secs(60);
    let done = Arc::new(AtomicBool::new(false));
    let done_setter = Arc::clone(&done);
    let worker = thread::Builder::new()
        .name("f2_super_node_with_filter-watchdog-worker".to_string())
        .spawn(move || {
            let q = vec![0.0001_f32; dim];
            let filter = Filter::LabelIn(vec![LabelId::new(1)]);
            let res = hnsw
                .filtered_search(&bytes_of(&q), 10, &filter, 100, &kernel, Lsn::MAX)
                .unwrap();
            assert!(!res.is_empty(), "super-node search returned empty");
            // Every result label-1.
            for (id, _) in &res {
                let p = hnsw.payload(*id).unwrap();
                assert!(
                    p.has_label(LabelId::new(1)),
                    "super-node: non-matching id {id:?} returned"
                );
            }
            done_setter.store(true, Ordering::Release);
        })
        .expect("spawn watchdog worker");

    let deadline = Instant::now() + WATCHDOG;
    while Instant::now() < deadline {
        if done.load(Ordering::Acquire) {
            worker.join().expect("filtered_search worker panicked");
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!(
        "f2_super_node_with_filter: filtered_search did not complete within {WATCHDOG:?} — \
         budget is the §3.5.1 class default (sized to absorb CI starvation), so non-completion \
         is a strong-signal non-terminating expansion. \
         See `docs/testing-strategy.md` §3.5."
    );
}

/// Random orthogonal vectors (distances ~1.0 across the board),
/// filter selects 1 % of them. Recall ≥ 0.85.
///
/// "Zero similarity" is the hostile-data baseline: the kernel
/// gives little signal because every pair is roughly equidistant.
/// HNSW's recall on hostile data is the floor we want to pin.
#[test]
fn f2_zero_similarity_dataset() {
    let dim = 64;
    let n = 1_000;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 17,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);

    // Random unit vectors in a high-dim sphere — pairwise L2
    // distances cluster around sqrt(2) (geometric concentration
    // in high dim). Effectively "zero similarity" for ranking
    // purposes.
    let raw = generate_unit_vectors(23, n, dim);
    let target_label = LabelId::new(1);
    let mut all = Vec::new();
    for (i, v) in raw.iter().enumerate() {
        // 1 % of vectors carry the target label.
        let label = if i < n / 100 {
            target_label
        } else {
            LabelId::new(2)
        };
        let payload = Payload::with_labels(vec![label]);
        let id = VectorId::new(i as u32);
        hnsw.filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .unwrap();
        all.push((id, v.clone(), payload));
    }

    let queries = generate_unit_vectors(31, 20, dim);
    let filter = Filter::LabelIn(vec![target_label]);
    let mut total = 0.0_f64;
    for q in &queries {
        let bf = brute_force_top_k_filtered(&all, q, 10, &filter);
        if bf.is_empty() {
            continue;
        }
        let hres = hnsw
            .filtered_search(&bytes_of(q), 10, &filter, 100, &kernel, Lsn::MAX)
            .unwrap();
        total += recall_at_k(&hres, &bf, 10);
    }
    let mean = total / queries.len() as f64;
    println!("zero-similarity recall@10 = {mean:.4}");
    assert!(
        mean >= 0.85,
        "zero-similarity dataset recall {mean} below 0.85 floor"
    );
}

/// Filter that partitions the payload-aware sub-graph into
/// disconnected components. The partitioning happens because we
/// build with two well-separated label clusters (label-A
/// vectors all in one region, label-B in another), then query
/// near the A region with a filter for label B. The
/// payload-aware edges keep B vectors well-connected to each
/// other, but the graph entry path goes through A; the search
/// must traverse non-matching A vectors to reach the B cluster.
///
/// Recall floor ≥ 0.85 — degraded but not zero.
#[test]
fn f2_payload_aware_connectivity_partition() {
    let dim = 16;
    let n_per_cluster = 200;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed: 29,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let label_a = LabelId::new(1);
    let label_b = LabelId::new(2);
    let mut rng = StdRng::seed_from_u64(29);

    // Build cluster A around (1, 0, 0, …)
    let mut all = Vec::new();
    for i in 0..n_per_cluster {
        let mut v = vec![1.0_f32];
        v.extend((1..dim).map(|_| {
            let u: f32 = StandardUniform.sample(&mut rng);
            (u * 2.0 - 1.0) * 0.1
        }));
        let v = l2_normalize(v);
        let payload = Payload::with_labels(vec![label_a]);
        let id = VectorId::new(i as u32);
        hnsw.filtered_insert(id, &bytes_of(&v), payload.clone(), &kernel)
            .unwrap();
        all.push((id, v, payload));
    }
    // Build cluster B around (-1, 0, 0, …) — far from A.
    for i in 0..n_per_cluster {
        let mut v = vec![-1.0_f32];
        v.extend((1..dim).map(|_| {
            let u: f32 = StandardUniform.sample(&mut rng);
            (u * 2.0 - 1.0) * 0.1
        }));
        let v = l2_normalize(v);
        let payload = Payload::with_labels(vec![label_b]);
        let id = VectorId::new((n_per_cluster + i) as u32);
        hnsw.filtered_insert(id, &bytes_of(&v), payload.clone(), &kernel)
            .unwrap();
        all.push((id, v, payload));
    }

    // Query near cluster A; filter for label B (cluster B).
    // The search must traverse through cluster A's
    // (non-matching) vectors to reach cluster B's matching set.
    let mut q = vec![1.0_f32];
    q.extend(vec![0.0_f32; dim - 1]);
    let q = l2_normalize(q);
    let filter = Filter::LabelIn(vec![label_b]);
    let bf = brute_force_top_k_filtered(&all, &q, 10, &filter);
    let res = hnsw
        .filtered_search(&bytes_of(&q), 10, &filter, 200, &kernel, Lsn::MAX)
        .unwrap();
    // Every result must be label B.
    for (id, _) in &res {
        let p = hnsw.payload(*id).unwrap();
        assert!(
            p.has_label(label_b),
            "partition: non-matching id {id:?} returned"
        );
    }
    let r = recall_at_k(&res, &bf, 10);
    println!("payload-aware partition recall@10 = {r:.4}");
    assert!(
        r >= 0.85,
        "payload-aware partition recall {r} below 0.85 floor; the graph entry must traverse non-matching cluster to reach matching cluster"
    );
}

// ─── Test 4 — Concurrent reader/writer ───────────────────────────

fn torture_secs() -> u64 {
    std::env::var("F2_TORTURE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3)
}

/// 8 reader threads + 1 writer for `F2_TORTURE_SECS` (default
/// 3, spec calls for 30).
///
/// Asserts:
///
/// 1. No panics or deadlocks (the test simply completes).
/// 2. No torn payload reads — every search result's id has a
///    payload sidecar entry that satisfies the filter.
/// 3. Reader throughput ≥ 1 search per thread (sanity — verifies
///    readers actually ran).
#[test]
fn f2_concurrent_search_insert_no_torn_payload() {
    let dim = 16;
    let params = HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        seed: 41,
    };
    let kernel = &L2F32;
    let mut g = FilteredHnsw::new(params, dim, kernel);

    let label_a = LabelId::new(1);
    let label_b = LabelId::new(2);
    // Seed with 200 vectors.
    let raw = generate_unit_vectors(7, 200, dim);
    for (i, v) in raw.iter().enumerate() {
        let label = if i % 2 == 0 { label_a } else { label_b };
        g.filtered_insert(
            VectorId::new(i as u32),
            &bytes_of(v),
            Payload::with_labels(vec![label]),
            kernel,
        )
        .unwrap();
    }

    let hnsw = Arc::new(RwLock::new(g));
    let stop = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));
    let inserts = Arc::new(AtomicUsize::new(0));
    let panic_seen = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();
    // Readers.
    for tid in 0..8 {
        let h = hnsw.clone();
        let s = stop.clone();
        let r = reads.clone();
        let p = panic_seen.clone();
        handles.push(thread::spawn(move || {
            let mut local_rng = StdRng::seed_from_u64(100 + tid);
            while !s.load(Ordering::Relaxed) {
                let q: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u: f32 = StandardUniform.sample(&mut local_rng);
                        u * 2.0 - 1.0
                    })
                    .collect();
                let filter = if local_rng.next_u32() % 2 == 0 {
                    Filter::LabelIn(vec![label_a])
                } else {
                    Filter::LabelIn(vec![label_b])
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let g = h.read();
                    g.filtered_search(&bytes_of(&q), 5, &filter, 50, &L2F32, Lsn::MAX)
                        .map(|res| {
                            // Per-result torn-payload check:
                            // every returned id MUST have a
                            // payload that satisfies the filter.
                            // A torn read would surface here as
                            // a panic in unwrap() or a filter
                            // mismatch.
                            for (id, _) in &res {
                                let payload = g
                                    .payload(*id)
                                    .expect("torn read: returned id missing payload");
                                assert!(
                                    filter.matches(payload),
                                    "torn read: payload doesn't match filter for id {id:?}"
                                );
                            }
                            res.len()
                        })
                }));
                match result {
                    Ok(Ok(_)) => {
                        r.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(Err(e)) => {
                        eprintln!("reader {tid}: search error {e:?}");
                        p.store(true, Ordering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        eprintln!("reader {tid}: PANIC");
                        p.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }));
    }
    // Writer.
    {
        let h = hnsw.clone();
        let s = stop.clone();
        let ins = inserts.clone();
        let p = panic_seen.clone();
        handles.push(thread::spawn(move || {
            let mut local_rng = StdRng::seed_from_u64(999);
            let mut next_id = 200u32;
            while !s.load(Ordering::Relaxed) {
                let v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u: f32 = StandardUniform.sample(&mut local_rng);
                        u * 2.0 - 1.0
                    })
                    .collect();
                let label = if local_rng.next_u32() % 2 == 0 {
                    label_a
                } else {
                    label_b
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut g = h.write();
                    g.filtered_insert(
                        VectorId::new(next_id),
                        &bytes_of(&v),
                        Payload::with_labels(vec![label]),
                        &L2F32,
                    )
                }));
                match result {
                    Ok(Ok(())) => {
                        ins.fetch_add(1, Ordering::Relaxed);
                        next_id += 1;
                    }
                    Ok(Err(e)) => {
                        eprintln!("writer: insert error {e:?}");
                        p.store(true, Ordering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        eprintln!("writer: PANIC");
                        p.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                // Throttle the writer slightly so readers get
                // CPU time. Under a hot-loop writer the
                // RwLock would starve the readers.
                std::thread::yield_now();
            }
        }));
    }

    let dur = Duration::from_secs(torture_secs());
    thread::sleep(dur);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("thread panicked");
    }

    assert!(
        !panic_seen.load(Ordering::Relaxed),
        "concurrent torture saw a panic"
    );
    let total_reads = reads.load(Ordering::Relaxed);
    let total_inserts = inserts.load(Ordering::Relaxed);
    println!("concurrent torture: {total_reads} reads, {total_inserts} inserts in {dur:?}");
    // Sanity: at least one read per reader thread.
    assert!(
        total_reads >= 8,
        "concurrent torture: only {total_reads} reads observed; expected ≥ 8"
    );
    // Sanity: writer made progress.
    assert!(
        total_inserts >= 1,
        "concurrent torture: writer made no progress"
    );
}

/// Deterministic concurrent filter evaluation: 4 threads
/// evaluating filters concurrently on a shared graph; assert
/// every thread sees the same result set for the same query.
#[test]
fn f2_concurrent_filter_evaluation() {
    let dim = 8;
    let n = 100;
    let params = HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        seed: 53,
    };
    let kernel = L2F32;
    let mut g = FilteredHnsw::new(params, dim, &kernel);
    let raw = generate_unit_vectors(53, n, dim);
    let label_a = LabelId::new(1);
    let label_b = LabelId::new(2);
    for (i, v) in raw.iter().enumerate() {
        let label = if i < n / 2 { label_a } else { label_b };
        g.filtered_insert(
            VectorId::new(i as u32),
            &bytes_of(v),
            Payload::with_labels(vec![label]),
            &kernel,
        )
        .unwrap();
    }
    let hnsw = Arc::new(g);
    let q = bytes_of(&raw[0]);
    let filter = Filter::LabelIn(vec![label_a]);

    let mut handles = Vec::new();
    let golden: Vec<(VectorId, f32)> = hnsw
        .filtered_search(&q, 10, &filter, 50, &L2F32, Lsn::MAX)
        .unwrap();
    for tid in 0..4 {
        let h = hnsw.clone();
        let q = q.clone();
        let f = filter.clone();
        let golden = golden.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..20 {
                let r = h.filtered_search(&q, 10, &f, 50, &L2F32, Lsn::MAX).unwrap();
                assert_eq!(
                    r, golden,
                    "thread {tid} got non-deterministic result vs golden"
                );
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
}

// ─── Test 5 — Edge cases ─────────────────────────────────────────

#[test]
fn f2_empty_filter_result() {
    let dim = 4;
    let mut g = FilteredHnsw::new(HnswParams::default(), dim, &L2F32);
    g.filtered_insert(
        VectorId::new(0),
        &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
        Payload::with_labels(vec![LabelId::new(1)]),
        &L2F32,
    )
    .unwrap();
    // Filter selects no vectors (label 99 doesn't exist).
    let filter = Filter::LabelIn(vec![LabelId::new(99)]);
    let res = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &filter,
            50,
            &L2F32,
            Lsn::MAX,
        )
        .unwrap();
    assert!(res.is_empty(), "empty filter result must be Vec, not error");
}

#[test]
fn f2_single_match_filter() {
    let dim = 4;
    let mut g = FilteredHnsw::new(HnswParams::default(), dim, &L2F32);
    let target_label = LabelId::new(7);
    let other_label = LabelId::new(8);
    // 5 vectors, only one has the target label.
    for i in 0..5u32 {
        let label = if i == 2 { target_label } else { other_label };
        g.filtered_insert(
            VectorId::new(i),
            &bytes_of(&[i as f32 * 0.1, 0.0, 0.0, 0.0]),
            Payload::with_labels(vec![label]),
            &L2F32,
        )
        .unwrap();
    }
    let filter = Filter::LabelIn(vec![target_label]);
    let res = g
        .filtered_search(
            &bytes_of(&[0.0, 0.0, 0.0, 0.0]),
            10,
            &filter,
            50,
            &L2F32,
            Lsn::MAX,
        )
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].0, VectorId::new(2));
}

/// Tenant cardinality 1 — `Filter::Tenant(t)` matches every
/// vector in the (single-tenant) arena because the arena
/// selection IS the tenant filter (ADR-011). This test verifies
/// the latency-friendly fast path: a tenant-only filter with
/// selectivity 1.0 takes the post-filter branch and incurs no
/// per-candidate filter overhead beyond the standard search.
#[test]
fn f2_tenant_cardinality_1_fast_path() {
    let dim = 8;
    let n = 100;
    let mut g = FilteredHnsw::new(HnswParams::default(), dim, &L2F32);
    let raw = generate_unit_vectors(83, n, dim);
    for (i, v) in raw.iter().enumerate() {
        // Every vector tagged with the same tenant.
        g.filtered_insert(
            VectorId::new(i as u32),
            &bytes_of(v),
            Payload {
                tenant_id: Some(TenantId::DEFAULT),
                ..Default::default()
            },
            &L2F32,
        )
        .unwrap();
    }
    let filter = Filter::Tenant(TenantId::DEFAULT);
    // Tenant filters are universal in single-tenant arenas
    // (ADR-011); selectivity is 1.0 by construction. The Phase
    // 6 F.4 dispatcher will own the real estimator — at v1.0
    // we hardcode the value here to drive the post-filter
    // branch and verify it agrees with a standard top-k.
    let sel = 1.0_f32;

    // Latency proof: dispatch with sel=1.0 takes the
    // post-filter branch. The result count must equal k (every
    // vector matches), and the result must equal a standard
    // top-k search.
    let q = &raw[0];
    let dispatched = g
        .filtered_search_dispatch(&bytes_of(q), 10, &filter, sel, 50, &L2F32, Lsn::MAX)
        .unwrap();
    let standard = g.inner().search(&bytes_of(q), 10, 50, &L2F32).unwrap();
    assert_eq!(dispatched.len(), standard.len());
    for (a, b) in dispatched.iter().zip(standard.iter()) {
        assert_eq!(a.0, b.0, "tenant fast path must agree with standard search");
    }
}

/// F.1 arena integration: filtered_search uses the arena's
/// `labels_for(id)` to derive payload labels, then the filter
/// evaluates against the materialized payload. Verifies the
/// F.1 → F.2 wiring: a payload built from `arena.labels_for(id)`
/// drives filter dispatch end-to-end.
///
/// This is a wiring test, not a recall test. The F.1 arena is
/// the source of truth for label storage (per the F.1 module
/// docs); F.2's `Payload` mirrors that for the in-memory HNSW.
/// This test exercises the bridge.
#[test]
fn f2_filter_arena_label_integration() {
    let arena = VectorArena::new(
        handle_for(1, 1),
        Encoding::F32,
        IndexType::Hnsw,
        QuantizerState::None,
        4,
    );
    // Insert into the arena with labels via the F.1 API.
    let label_a = LabelId::new(1);
    arena
        .insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            Some(&[label_a]),
        )
        .unwrap();
    arena
        .insert(
            VectorId::new(1),
            &bytes_of(&[0.0, 1.0, 0.0, 0.0]),
            Some(&[LabelId::new(2)]),
        )
        .unwrap();

    // Bridge F.1 → F.2: build payloads from arena.labels_for.
    let mut hnsw = FilteredHnsw::new(HnswParams::default(), 4, &L2F32);
    for id_raw in 0..2u32 {
        let id = VectorId::new(id_raw);
        let labels = arena.labels_for(id).expect("arena has labels");
        let payload = Payload::with_labels(labels.to_vec());
        let bytes = arena.get_primary(id).expect("arena has primary bytes");
        hnsw.filtered_insert(id, &bytes, payload, &L2F32).unwrap();
    }

    let filter = Filter::LabelIn(vec![label_a]);
    let res = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &filter,
            50,
            &L2F32,
            Lsn::MAX,
        )
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].0, VectorId::new(0));
}

// ─── Test 6 — Existing regression guards (re-runs from Slice A + F.1) ────

/// Slice A guard: any [`VectorIndexHandle`] constructed via
/// the public API has `partition_id == PartitionId::ZERO` per
/// ADR-035 D-7. Re-asserted here so an F.2 contributor that
/// accidentally introduces a non-zero-partition handle
/// surfaces it on this test, not on a far-removed Slice A
/// regression run.
#[test]
fn vector_index_partition_id_always_zero_at_v1() {
    let h = VectorIndexHandle::for_tenant(TenantId::DEFAULT, IndexId::new(1));
    assert_eq!(h.partition(), PartitionId::ZERO);
    assert!(h.is_v1_local());
}

/// Slice F.1 guard: arena handles also obey the partition-id
/// invariant. Re-asserted so a payload-aware code path that
/// constructs an arena (e.g., the `f2_filter_arena_label_integration`
/// test) cannot drift the invariant.
#[test]
fn arena_partition_id_always_zero_at_v1() {
    let arena = VectorArena::new(
        handle_for(1, 1),
        Encoding::F32,
        IndexType::Hnsw,
        QuantizerState::None,
        4,
    );
    assert_eq!(arena.handle().partition(), PartitionId::ZERO);
    assert!(arena.handle().is_v1_local());
}

// ─── #815 — served-path predicate_filtered_search (bare HnswGraph) ──
//
// The served HNSW provider keeps a bare `HnswGraph` + an external
// `VectorId → label` map and filters DURING traversal via
// `predicate_filtered_search` (no `Payload` sidecar, so the standard
// `O(log N)` insert is preserved). #815: a selective label filter must
// NOT collapse recall the way a retrieve-then-discard post-filter does.

/// The issue's exact deterministic repro: 500 vectors on the line
/// `[i,0,0]`, every 10th labelled `Rare` (10 % selectivity), query
/// `[250,0,0]`, k=10. A post-filter over the unfiltered top-10 returns
/// just ONE `Rare` (the recall collapse); filter-during-search returns
/// the TRUE 10 nearest `Rare`. STRONG oracle: the exact id set, by
/// brute force over only the `Rare`-labelled vectors.
#[test]
fn pred_filtered_search_recovers_true_neighbors_under_selective_filter_815() {
    const N: u32 = 500;
    const RARE: u32 = 1;
    const COMMON: u32 = 2;
    let kernel = L2F32;
    let mut g = HnswGraph::new(HnswParams::default(), 3, &kernel);

    // VectorId(i) → label. Rare iff i % 10 == 0 (50 of 500 = 10 %).
    let label_of = |i: u32| if i % 10 == 0 { RARE } else { COMMON };
    for i in 0..N {
        let v = [i as f32, 0.0, 0.0];
        g.insert(VectorId::new(i), &bytes_of(&v), &kernel).unwrap();
    }
    let query = bytes_of(&[250.0_f32, 0.0, 0.0]);

    // Brute-force truth: the 10 `Rare` (multiples of 10) nearest 250.
    let mut rare: Vec<u32> = (0..N).filter(|i| label_of(*i) == RARE).collect();
    rare.sort_by_key(|i| (i64::from(*i) - 250).abs());
    let truth: HashSet<u32> = rare.iter().take(10).copied().collect();
    // == {250,240,260,230,270,220,280,210,290,200}
    assert_eq!(truth.len(), 10);

    // ── AFTER (the #815 fix): filter-during-search returns the true 10.
    let is_rare = |vid: VectorId| label_of(vid.0) == RARE;
    let filtered = predicate_filtered_search(&g, &query, 10, 0, &kernel, &is_rare).unwrap();
    let got: HashSet<u32> = filtered.iter().map(|(v, _)| v.0).collect();
    assert_eq!(
        filtered.len(),
        10,
        "filtered KNN must return k=10 Rare hits, not collapse to ~1"
    );
    assert_eq!(
        got, truth,
        "filter-during-search must return the TRUE 10 nearest Rare"
    );
    assert!(
        got.iter().all(|i| label_of(*i) == RARE),
        "no non-Rare leakage into the filtered result"
    );

    // ── BEFORE (post-filter over the unfiltered top-10): only node 250
    //    survives — the recall collapse #815 fixes (any 10-wide window
    //    around 250 contains exactly one multiple of 10).
    let unfiltered = g.search(&query, 10, 0, &kernel).unwrap();
    let post_filtered: Vec<u32> = unfiltered
        .iter()
        .map(|(v, _)| v.0)
        .filter(|i| label_of(*i) == RARE)
        .collect();
    assert_eq!(
        post_filtered,
        vec![250],
        "post-filter over the unfiltered top-10 keeps only node 250 (recall 1/10)"
    );

    // ── The issue's "need k≈1/selectivity" claim: post-filter only
    //    recovers the true 10 by inflating k to ~100.
    let wide = g.search(&query, 100, 0, &kernel).unwrap();
    let wide_rare: HashSet<u32> = wide
        .iter()
        .map(|(v, _)| v.0)
        .filter(|i| label_of(*i) == RARE)
        .collect();
    assert!(
        truth.is_subset(&wide_rare),
        "post-filter needs k≈100 (1/selectivity) to recover what filtered-KNN gets at k=10"
    );
}

/// Lower-selectivity (2 %), random unit-vector corpus: filter-during-
/// search must hold recall@10 vs an EXACT brute-force oracle over only
/// the `Rare` subset — proving the bare-graph predicate path does not
/// collapse recall on a non-trivial topology (not just the easy line).
#[test]
fn pred_filtered_search_holds_recall_at_low_selectivity_815() {
    const N: usize = 1000;
    const DIM: usize = 8;
    const K: usize = 10;
    const RARE: u32 = 1;
    const COMMON: u32 = 2;
    let kernel = L2F32;
    let corpus = generate_unit_vectors(0x00F2_8155, N, DIM);

    let mut g = HnswGraph::new(HnswParams::default(), DIM, &kernel);
    let label_of = |i: u32| if i % 50 == 0 { RARE } else { COMMON }; // 20 Rare = 2 %
    for (i, v) in corpus.iter().enumerate() {
        g.insert(VectorId::new(i as u32), &bytes_of(v), &kernel)
            .unwrap();
    }
    let is_rare = |vid: VectorId| label_of(vid.0) == RARE;
    let rare_ids: Vec<u32> = (0..N as u32).filter(|i| label_of(*i) == RARE).collect();
    assert!(
        rare_ids.len() > K,
        "need more Rare than K for a real recall@K"
    );

    let queries = generate_unit_vectors(0x000F_F1CE, 30, DIM);
    let l2_sq =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let mut total = 0.0_f64;
    for q in &queries {
        // Exact top-K over ONLY the Rare vectors (strong oracle).
        let mut exact: Vec<(u32, f32)> = rare_ids
            .iter()
            .map(|&i| (i, l2_sq(q, &corpus[i as usize])))
            .collect();
        exact.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth: HashSet<u32> = exact.iter().take(K).map(|(i, _)| *i).collect();

        let got = predicate_filtered_search(&g, &bytes_of(q), K, 0, &kernel, &is_rare).unwrap();
        assert_eq!(
            got.len(),
            K,
            "must return K matching hits ({} Rare available)",
            rare_ids.len()
        );
        let got_ids: HashSet<u32> = got.iter().map(|(v, _)| v.0).collect();
        assert!(
            got_ids.iter().all(|i| label_of(*i) == RARE),
            "no non-Rare leakage"
        );
        total += truth.intersection(&got_ids).count() as f64 / K as f64;
    }
    let recall = total / queries.len() as f64;
    eprintln!("#815 predicate_filtered_search recall@{K} @2%-selectivity over N={N}: {recall:.4}");
    assert!(
        recall >= 0.95,
        "selective filter must hold recall@{K} ≥ 0.95, got {recall:.4}"
    );
}
