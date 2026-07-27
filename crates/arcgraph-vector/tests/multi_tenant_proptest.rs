//! Multi-tenant proptest + asymmetric-empty regression — Slice F.5 (M3.a Phase 6).
//!
//! Per ADR-035 amendment-04 + the F.5 slice prompt. Stress-tests
//! the F.4 selectivity dispatcher under controlled multi-tenant
//! conditions using mock backends; closes Reviewer Finding 1
//! (asymmetric-empty) from PR #134 with two deterministic
//! regressions pinning the amendment-04 D-4 catalog-sync invariant.
//!
//! ## Properties pinned (proptest, 256 default cases, 1024 in CI)
//!
//! 1. Tenant isolation (2-tenant) — dispatch returns only the
//!    queried tenant's vector ids.
//! 2. Variant routing determinism — same inputs → same routing
//!    every time.
//! 3. Escalation correctness — primary `UnsupportedFilter` →
//!    fallback's result.
//! 4. Result-count bounded + filter satisfaction — `result.len() ≤ k`,
//!    every returned id satisfies the filter.
//! 5. Empty-result correctness — non-matching filter → `Ok(Vec::new())`.
//! 6. No-backends edge case — empty `BackendSet` →
//!    `Err(UnsupportedFilter)`.
//! 7. Fault-injection escalation matrix — every (primary, fallback)
//!    response combo → contracted dispatcher result.
//!
//! ## Asymmetric-empty regressions (deterministic)
//!
//! - `dispatcher_asymmetric_empty_violates_catalog_contract` —
//!   forward asymmetry (primary empty, fallback non-empty).
//! - `dispatcher_asymmetric_empty_reverse_after_unsupported_escalation` —
//!   reverse asymmetry (primary rejects, fallback empty).
//!
//! Both pin the amendment-04 D-4 catalog-sync invariant.
//!
//! ## Knobs
//!
//! - `IV_PROPTEST_CASES` — overrides proptest case count (default
//!   256; CI exports 1024).
//!
//! Run:
//!   cargo test -p arcgraph-vector --test multi_tenant_proptest
//!   IV_PROPTEST_CASES=1024 cargo test -p arcgraph-vector --release \
//!       --test multi_tenant_proptest

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::{LabelId, Lsn, StringId, TenantId};
use arcgraph_vector::distance::{DistanceKernel, L2F32};
use arcgraph_vector::hnsw::Payload;
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{
    BackendKind, BackendSet, Filter, FilteredVectorIndex, PropertyValue, VectorIndexError,
};
use parking_lot::Mutex;
use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Knobs ───────────────────────────────────────────────────────

/// Per-file proptest case count (default 256, CI overrides via
/// `IV_PROPTEST_CASES=1024`). Mirrors the helper in
/// `tests/iv_invariants.rs` so the gauntlet env var is uniform.
fn proptest_case_count(default: u32) -> u32 {
    std::env::var("IV_PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default)
}

// ─── Bytes helper ────────────────────────────────────────────────

fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

// ─── MockBackend ────────────────────────────────────────────────
//
// Strictly more flexible than the unit-test mock in
// `src/dispatcher.rs`: holds a real dataset, supports brute-force
// filter+top-k against its own vectors, AND supports per-call
// fault-injection via a response override. This lets a single
// proptest configure "this backend rejects PropertyEq" while the
// rest pass through with realistic results.

/// Test-only mock backend that produces realistic search results
/// over its own dataset and supports per-call fault-injection.
struct MockBackend {
    kind: BackendKind,
    /// Owned dataset: `(id, vector, payload)` triples. Brute-force
    /// search filters this set via `Filter::matches`, then top-k
    /// by L2 distance.
    vectors: Vec<(VectorId, Vec<f32>, Payload)>,
    /// Vector dimensionality (in elements, not bytes). Used to
    /// validate the query length on every call so the
    /// `DimensionMismatch` error path is exercisable from the
    /// proptest.
    dim: usize,
    /// If `Some`, override the next `filtered_search` response
    /// instead of running brute-force. Used by the fault-injection
    /// tests; `None` means "do brute-force search".
    response_override: Mutex<Option<MockResponse>>,
    /// Counts every `filtered_search` invocation, regardless of
    /// brute-force vs override path.
    calls: AtomicUsize,
}

/// Synthetic response shape for fault-injection. A `MockBackend`
/// in `response_override` mode returns the configured shape on its
/// next `filtered_search` call.
#[derive(Clone)]
enum MockResponse {
    Ok(Vec<(VectorId, f32)>),
    UnsupportedFilter,
    DimensionMismatch,
}

impl MockBackend {
    /// Construct a brute-force backend with the given dataset.
    fn new_brute(
        kind: BackendKind,
        vectors: Vec<(VectorId, Vec<f32>, Payload)>,
        dim: usize,
    ) -> Self {
        Self {
            kind,
            vectors,
            dim,
            response_override: Mutex::new(None),
            calls: AtomicUsize::new(0),
        }
    }

    /// Construct a backend that always returns
    /// `Err(UnsupportedFilter)`. The dataset is empty (so the
    /// dispatcher's empty-graph short-circuit is NOT triggered we
    /// bias `len()` to 1 below).
    fn new_rejecting(kind: BackendKind, dim: usize) -> Self {
        let me = Self {
            kind,
            // Non-empty so the dispatcher's empty-graph short-
            // circuit does not fire; the override path is what
            // we want to exercise.
            vectors: vec![],
            dim,
            response_override: Mutex::new(Some(MockResponse::UnsupportedFilter)),
            calls: AtomicUsize::new(0),
        };
        // The override is sticky (set_response_override below
        // does the same). We tag len_override so `len() > 0`
        // even though `vectors.is_empty()`.
        me.set_len_marker();
        me
    }

    /// Construct an empty backend: zero vectors, no override.
    /// `is_empty()` → true; the dispatcher's empty-graph short-
    /// circuit fires before `filtered_search` is ever called.
    fn empty(kind: BackendKind, dim: usize) -> Self {
        Self {
            kind,
            vectors: vec![],
            dim,
            response_override: Mutex::new(None),
            calls: AtomicUsize::new(0),
        }
    }

    /// Workaround for the rejecting backend: we want
    /// `is_empty() == false` so the empty-graph short-circuit does
    /// NOT fire, but we hold no real vectors. We treat
    /// `response_override = Some(_)` as a "len-override" marker —
    /// see `len()` below.
    fn set_len_marker(&self) {
        // No-op tag; the marker IS the override being Some.
    }

    /// Atomic call counter — number of times `filtered_search` was
    /// invoked on this backend.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Swap in a new response override for the next call. The
    /// override is consumed on each call (`Option::take`-style),
    /// so a fresh override must be set before every dispatch.
    fn set_response_override(&self, response: MockResponse) {
        *self.response_override.lock() = Some(response);
    }

    /// Brute-force top-k filtered search against the owned dataset.
    /// Used by all proptest "real" backends (both slots in
    /// properties 1, 2, 4, 5; one slot in property 3 / 7).
    fn brute_force(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<(VectorId, f32)> {
        let mut scored: Vec<(VectorId, f32)> = self
            .vectors
            .iter()
            .filter(|(_, _, p)| filter.matches(p))
            .map(|(id, v, _)| {
                let d: f32 = v
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (*id, d)
            })
            .collect();
        // Stable sort: primary by distance ascending, secondary
        // by id ascending (for tie-breaking determinism).
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .expect("non-NaN compare")
                .then_with(|| a.0.raw().cmp(&b.0.raw()))
        });
        scored.truncate(k);
        scored
    }
}

impl FilteredVectorIndex for MockBackend {
    fn kind(&self) -> BackendKind {
        self.kind
    }

    fn len(&self) -> usize {
        // The rejecting backend constructor parks an
        // UnsupportedFilter override in `response_override` AND
        // holds zero real vectors; we want `len() > 0` so the
        // dispatcher invokes `filtered_search` (which then returns
        // UnsupportedFilter) instead of short-circuiting via
        // is_empty(). Treat "override set" as "len = 1" for the
        // purposes of the emptiness check.
        if self.vectors.is_empty() && self.response_override.lock().is_some() {
            return 1;
        }
        self.vectors.len()
    }

    fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        _ef: usize,
        _kernel: &dyn DistanceKernel,
        _read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        // Validate query length first so the proptest can
        // exercise the DimensionMismatch propagation path even
        // for "real" (non-overridden) backends.
        let expected_bytes = self.dim * std::mem::size_of::<f32>();
        if query.len() != expected_bytes {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim,
                got: query.len() / std::mem::size_of::<f32>(),
            });
        }

        // Take the override if present; otherwise fall through
        // to brute-force search.
        let taken = self.response_override.lock().take();
        if let Some(resp) = taken {
            return match resp {
                MockResponse::Ok(hits) => Ok(hits),
                MockResponse::UnsupportedFilter => {
                    // Re-arm the override so subsequent calls in
                    // the same proptest case (e.g., the
                    // determinism property's two-call snapshot)
                    // continue to reject.
                    *self.response_override.lock() = Some(MockResponse::UnsupportedFilter);
                    Err(VectorIndexError::UnsupportedFilter {
                        reason: format!("mock {} rejected filter", self.kind.as_str()),
                    })
                }
                MockResponse::DimensionMismatch => {
                    *self.response_override.lock() = Some(MockResponse::DimensionMismatch);
                    Err(VectorIndexError::DimensionMismatch {
                        expected: self.dim,
                        got: 0,
                    })
                }
            };
        }

        // Brute-force path: cast the query bytes back to f32 and
        // run filtered top-k against the dataset.
        let q: &[f32] = bytemuck::cast_slice(query);
        Ok(self.brute_force(q, k, filter))
    }
}

// ─── Strategies ─────────────────────────────────────────────────

/// Random small filter (depth ≤ 2) covering every variant. Keeps
/// both leaves and compounds in the strategy so the proptest hits
/// the full variant routing table from the F.4 dispatcher's D-2.
fn arb_filter() -> impl Strategy<Value = Filter> {
    let leaf = prop_oneof![
        Just(Filter::Any),
        prop_oneof![
            Just(Filter::Tenant(TenantId::new(1))),
            Just(Filter::Tenant(TenantId::new(2))),
        ],
        (1u32..=5).prop_map(|l| Filter::LabelEq(LabelId::new(l))),
        prop::collection::vec(1u32..=5, 1..3)
            .prop_map(|ls| Filter::LabelIn(ls.into_iter().map(LabelId::new).collect())),
        (10u32..=11, 0u32..5)
            .prop_map(|(k, v)| Filter::PropertyEq(StringId::new(k), PropertyValue::U32(v))),
    ];
    // depth ≤ 2: leaf | And(leaf*) | Or(leaf*).
    leaf.prop_recursive(2, 6, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Filter::And),
            prop::collection::vec(inner, 0..3).prop_map(Filter::Or),
        ]
    })
}

/// Same as `arb_filter` but biased toward the variants the
/// dispatcher's variant table treats specially (Any, LabelEq for
/// DiskAnnPreferred; the rest for HnswOnly). Used by the
/// determinism + filter-satisfaction properties.
fn arb_filter_balanced() -> impl Strategy<Value = Filter> {
    arb_filter()
}

// ─── Fixture builders ───────────────────────────────────────────

/// Build a deterministic 2-tenant dataset.
///
/// Tenant A holds ids `[0, n_a)`; tenant B holds ids `[1000, 1000
/// + n_b)`. Each vector's payload carries its tenant_id, a label
/// drawn from `{1..=5}` cyclically, and one of two property keys
/// (`StringId(10)`, `StringId(11)`) when the per-vector RNG decides
/// to attach one. The disjoint id ranges mirror the I-V2 pattern
/// in `tests/iv_invariants.rs`.
fn build_tenant_dataset(
    seed: u64,
    n: usize,
    tenant_id: TenantId,
    id_base: u32,
    dim: usize,
) -> Vec<(VectorId, Vec<f32>, Payload)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                u * 2.0 - 1.0
            })
            .collect();
        let mut props = HashMap::new();
        if rng.next_u32() % 2 == 0 {
            let key = if rng.next_u32() % 2 == 0 {
                StringId::new(10)
            } else {
                StringId::new(11)
            };
            props.insert(key, PropertyValue::U32(rng.next_u32() % 5));
        }
        let payload = Payload {
            tenant_id: Some(tenant_id),
            labels: vec![LabelId::new(((i % 5) as u32) + 1)],
            properties: props,
            ..Payload::default()
        };
        out.push((VectorId::new(id_base + i as u32), v, payload));
    }
    out
}

/// A deterministic single-tenant fixture for the routing /
/// determinism / filter-satisfaction properties.
fn build_mixed_fixture(seed: u64, n: usize, dim: usize) -> Vec<(VectorId, Vec<f32>, Payload)> {
    build_tenant_dataset(seed, n, TenantId::new(1), 0, dim)
}

// ─── Property 1 — Tenant isolation (2-tenant) ───────────────────

#[test]
fn property_1_tenant_isolation_2_tenant() {
    // Per-tenant arenas hold disjoint id ranges (cf. iv2). The
    // dispatcher does NOT enforce tenant isolation directly —
    // arena selection upstream does. This proptest validates that
    // routing each tenant's query to its OWN BackendSet returns
    // ONLY ids in that tenant's range, regardless of the random
    // filter shape. Cross-tenant leakage (an id from the wrong
    // range) is a P0 violation.
    let dim = 8;
    let n_a = 60;
    let n_b = 60;
    let kernel = L2F32;
    let data_a = build_tenant_dataset(0xF500_C001, n_a, TenantId::new(1), 0, dim);
    let data_b = build_tenant_dataset(0xF500_C002, n_b, TenantId::new(2), 1000, dim);

    // Each tenant gets a (HNSW + DiskANN) BackendSet — both slots
    // populated, both holding ONLY that tenant's data. The
    // dispatcher routes per the variant table; tenant B's data
    // is unreachable from tenant A's BackendSet by construction.
    let hnsw_a = MockBackend::new_brute(BackendKind::Hnsw, data_a.clone(), dim);
    let diskann_a = MockBackend::new_brute(BackendKind::DiskAnn, data_a.clone(), dim);
    let hnsw_b = MockBackend::new_brute(BackendKind::Hnsw, data_b.clone(), dim);
    let diskann_b = MockBackend::new_brute(BackendKind::DiskAnn, data_b.clone(), dim);

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(
            &(any::<bool>(), 0usize..n_a.min(n_b), 1usize..6, arb_filter()),
            |(query_a, q_idx, k, filter)| {
                let (qvec, expected_range) = if query_a {
                    (&data_a[q_idx].1, 0u32..(n_a as u32))
                } else {
                    (&data_b[q_idx].1, 1000u32..(1000 + n_b as u32))
                };
                let q_bytes = bytes_of(qvec);

                let set = if query_a {
                    BackendSet::new()
                        .with_hnsw(&hnsw_a)
                        .with_diskann(&diskann_a)
                } else {
                    BackendSet::new()
                        .with_hnsw(&hnsw_b)
                        .with_diskann(&diskann_b)
                };
                let res = set
                    .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                    .expect("dispatch");

                for (id, _) in &res {
                    prop_assert!(
                        expected_range.contains(&id.raw()),
                        "F.5 P1 tenant leakage: query for tenant in range \
                         {expected_range:?} returned id {id:?} (filter \
                         {filter:?})"
                    );
                }
                prop_assert!(res.len() <= k, "F.5 P1 |result| > k");
                Ok(())
            },
        )
        .expect("F.5 P1 tenant isolation proptest");
}

// ─── Property 2 — Variant routing determinism ───────────────────

#[test]
fn property_2_variant_routing_determinism() {
    // For any (filter, BackendSet) pair, two consecutive
    // dispatch_filtered_search calls return identical results AND
    // select the same backend. The dispatcher is documented as a
    // pure function over its inputs; this property pins the
    // documented contract.
    let dim = 8;
    let n = 50;
    let kernel = L2F32;
    let data = build_mixed_fixture(0xF500_D001, n, dim);
    let hnsw = MockBackend::new_brute(BackendKind::Hnsw, data.clone(), dim);
    let diskann = MockBackend::new_brute(BackendKind::DiskAnn, data.clone(), dim);

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(
            &(0usize..n, 1usize..10, arb_filter_balanced()),
            |(q_idx, k, filter)| {
                let q_bytes = bytes_of(&data[q_idx].1);
                let set = BackendSet::new().with_hnsw(&hnsw).with_diskann(&diskann);

                let h_before_1 = hnsw.calls();
                let d_before_1 = diskann.calls();
                let res_1 = set
                    .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                    .expect("dispatch run 1");
                let h_after_1 = hnsw.calls();
                let d_after_1 = diskann.calls();

                let res_2 = set
                    .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                    .expect("dispatch run 2");
                let h_after_2 = hnsw.calls();
                let d_after_2 = diskann.calls();

                let h_delta_1 = h_after_1 - h_before_1;
                let d_delta_1 = d_after_1 - d_before_1;
                let h_delta_2 = h_after_2 - h_after_1;
                let d_delta_2 = d_after_2 - d_after_1;

                prop_assert_eq!(
                    &res_1,
                    &res_2,
                    "F.5 P2 dispatcher non-determinism: two runs of the \
                     same (filter, query, k) returned different results"
                );
                prop_assert_eq!(
                    h_delta_1,
                    h_delta_2,
                    "F.5 P2 routing non-determinism: HNSW called {} times in \
                     run 1, {} times in run 2",
                    h_delta_1,
                    h_delta_2
                );
                prop_assert_eq!(
                    d_delta_1,
                    d_delta_2,
                    "F.5 P2 routing non-determinism: DiskANN called {} times \
                     in run 1, {} times in run 2",
                    d_delta_1,
                    d_delta_2
                );
                Ok(())
            },
        )
        .expect("F.5 P2 routing determinism proptest");
}

// ─── Property 3 — Escalation correctness ────────────────────────

#[test]
fn property_3_escalation_correctness() {
    // Build (real HNSW + rejecting DiskANN) for a Filter::LabelEq
    // query (DiskANN-preferred per the variant table). The
    // dispatcher tries DiskANN first → UnsupportedFilter → escalates
    // to HNSW. Compare the escalated result against directly
    // calling HNSW's filtered_search; since both go through
    // brute-force, the results should be byte-identical.
    let dim = 8;
    let n = 50;
    let kernel = L2F32;
    let data = build_mixed_fixture(0xF500_E001, n, dim);
    let hnsw = MockBackend::new_brute(BackendKind::Hnsw, data.clone(), dim);
    let diskann_rej = MockBackend::new_rejecting(BackendKind::DiskAnn, dim);

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(&(0usize..n, 1u32..=5, 1usize..10), |(q_idx, label, k)| {
            let q_bytes = bytes_of(&data[q_idx].1);
            let filter = Filter::LabelEq(LabelId::new(label));

            let set = BackendSet::new()
                .with_hnsw(&hnsw)
                .with_diskann(&diskann_rej);

            let escalated = set
                .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                .expect("dispatch");

            // Direct call against HNSW backend (the fallback).
            // Must match the escalated dispatcher result byte-
            // identically — both paths run brute-force on the
            // same dataset.
            let direct = hnsw
                .filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                .expect("direct hnsw");

            prop_assert_eq!(
                &escalated,
                &direct,
                "F.5 P3 escalation result mismatch: escalated dispatcher \
                     result differs from direct fallback call (filter {:?})",
                filter
            );
            Ok(())
        })
        .expect("F.5 P3 escalation correctness proptest");
}

// ─── Property 4 — Result-count bounded + filter satisfaction ────

#[test]
fn property_4_result_count_bounded_and_filter_satisfaction() {
    // For any random filter and k, the dispatcher returns ≤ k
    // results AND every returned id satisfies Filter::matches
    // against its payload.
    let dim = 8;
    let n = 60;
    let kernel = L2F32;
    let data = build_mixed_fixture(0xF500_F001, n, dim);
    // Both backends present; both do real brute-force search.
    let hnsw = MockBackend::new_brute(BackendKind::Hnsw, data.clone(), dim);
    let diskann = MockBackend::new_brute(BackendKind::DiskAnn, data.clone(), dim);
    // Index payloads by id for the satisfaction check.
    let payload_by_id: HashMap<VectorId, &Payload> =
        data.iter().map(|(id, _, p)| (*id, p)).collect();

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(
            &(0usize..n, 1usize..=20, arb_filter()),
            |(q_idx, k, filter)| {
                let q_bytes = bytes_of(&data[q_idx].1);
                let set = BackendSet::new().with_hnsw(&hnsw).with_diskann(&diskann);
                let res = set
                    .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                    .expect("dispatch");

                prop_assert!(
                    res.len() <= k,
                    "F.5 P4 |result| > k: got {} results for k={}",
                    res.len(),
                    k
                );
                for (id, _) in &res {
                    let p = payload_by_id.get(id).expect("id from dataset");
                    prop_assert!(
                        filter.matches(p),
                        "F.5 P4 filter violation: id {id:?} returned but does \
                         not satisfy filter {filter:?} (payload tenant={:?}, \
                         labels={:?}, properties={:?})",
                        p.tenant_id,
                        p.labels,
                        p.properties
                    );
                }
                Ok(())
            },
        )
        .expect("F.5 P4 result-bounded + filter-satisfaction proptest");
}

// ─── Property 5 — Empty-result correctness ──────────────────────

#[test]
fn property_5_empty_result_correctness() {
    // For any filter that matches zero vectors in the dataset, the
    // dispatcher returns Ok(Vec::new()). We construct filters
    // guaranteed to match nothing against the fixture, then verify
    // brute-force ground-truth = 0 (precondition) before
    // dispatching.
    let dim = 8;
    let n = 60;
    let kernel = L2F32;
    let data = build_mixed_fixture(0xF500_5001, n, dim);
    let hnsw = MockBackend::new_brute(BackendKind::Hnsw, data.clone(), dim);
    let diskann = MockBackend::new_brute(BackendKind::DiskAnn, data.clone(), dim);

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    // Strategy: pick a filter shape guaranteed to match nothing,
    // varying both shape and the absent value.
    // - LabelEq(absent_label) — labels in fixture are 1..=5; pick from 100..200.
    // - PropertyEq(absent_key, _) — keys in fixture are {10, 11}; pick from 50..60.
    // - Tenant(absent_tenant) — fixture is tenant 1; pick from 999..1099.
    let nonmatching_strategy = prop_oneof![
        (100u32..200u32).prop_map(|l| Filter::LabelEq(LabelId::new(l))),
        (50u32..60u32, 0u32..5)
            .prop_map(|(k, v)| Filter::PropertyEq(StringId::new(k), PropertyValue::U32(v))),
        (999u64..1099u64).prop_map(|t| Filter::Tenant(TenantId::new(t))),
    ];

    runner
        .run(
            &(0usize..n, 1usize..=10, nonmatching_strategy),
            |(q_idx, k, filter)| {
                // Precondition: filter matches zero vectors in the
                // dataset. If not, skip (a misconfigured strategy
                // would invalidate the property).
                let ground_truth_count = data.iter().filter(|(_, _, p)| filter.matches(p)).count();
                prop_assume!(ground_truth_count == 0);

                let q_bytes = bytes_of(&data[q_idx].1);
                let set = BackendSet::new().with_hnsw(&hnsw).with_diskann(&diskann);
                let res = set
                    .dispatch_filtered_search(&q_bytes, k, &filter, 32, &kernel, Lsn::MAX)
                    .expect("dispatch");

                prop_assert!(
                    res.is_empty(),
                    "F.5 P5 empty-result violation: filter {filter:?} matches \
                     zero vectors but dispatcher returned {} hits",
                    res.len()
                );
                Ok(())
            },
        )
        .expect("F.5 P5 empty-result correctness proptest");
}

// ─── Property 6 — No-backends edge case ─────────────────────────

#[test]
fn property_6_no_backends_returns_unsupported_filter() {
    // For every random Filter variant, an empty BackendSet returns
    // Err(UnsupportedFilter). This matches the F.4 implementation
    // (dispatcher.rs lines 395–403): with no backends, the
    // dispatcher surfaces UnsupportedFilter regardless of the
    // filter variant — including Filter::Any, which has no
    // special-case path through the no-backends branch.
    let kernel = L2F32;
    let dim = 8;
    let q = vec![0.0_f32; dim];
    let q_bytes = bytes_of(&q);

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(&arb_filter(), |filter| {
            let set = BackendSet::new();
            let res = set.dispatch_filtered_search(&q_bytes, 5, &filter, 10, &kernel, Lsn::MAX);
            prop_assert!(
                matches!(res, Err(VectorIndexError::UnsupportedFilter { .. })),
                "F.5 P6 no-backends violation: filter {filter:?} on empty \
                 BackendSet should yield UnsupportedFilter; got {res:?}"
            );
            Ok(())
        })
        .expect("F.5 P6 no-backends proptest");
}

// ─── Property 7 — Fault-injection escalation matrix ─────────────
//
// Pin the (primary_response, fallback_response) → dispatcher_result
// matrix from amendment-04 D-3:
//
// | primary returns               | fallback returns        | Expected dispatcher result |
// |-------------------------------|-------------------------|----------------------------|
// | Ok(hits)                      | (anything)              | Ok(hits) (fallback never)  |
// | Err(UnsupportedFilter)        | Ok(hits)                | Ok(hits) from fallback     |
// | Err(UnsupportedFilter)        | Err(UnsupportedFilter)  | Err(UnsupportedFilter)     |
// | Err(UnsupportedFilter)        | Err(DimensionMismatch)  | Err(DimensionMismatch)     |
// | Err(DimensionMismatch)        | (anything)              | Err(DimensionMismatch)     |

#[derive(Debug, Clone, Copy)]
enum MatrixRow {
    OkPrimary,
    PrimaryUnsupportedFallbackOk,
    BothUnsupported,
    PrimaryUnsupportedFallbackDim,
    PrimaryDim,
}

fn arb_matrix_row() -> impl Strategy<Value = MatrixRow> {
    prop_oneof![
        Just(MatrixRow::OkPrimary),
        Just(MatrixRow::PrimaryUnsupportedFallbackOk),
        Just(MatrixRow::BothUnsupported),
        Just(MatrixRow::PrimaryUnsupportedFallbackDim),
        Just(MatrixRow::PrimaryDim),
    ]
}

#[test]
fn property_7_fault_injection_escalation_matrix() {
    let dim = 8;
    let kernel = L2F32;
    let q = vec![0.5_f32; dim];
    let q_bytes = bytes_of(&q);

    // For Filter::LabelEq the variant table puts DiskANN as primary
    // and HNSW as fallback. Both slots get fault-injection backends
    // so we can drive the full 5-row matrix. Each holds a single
    // dummy vector — len() > 0 so the empty-graph short-circuit
    // does NOT preempt the override path; the actual response is
    // always supplied by the override.
    let primary = MockBackend::new_brute(
        BackendKind::DiskAnn,
        vec![(VectorId::new(0), vec![0.0; dim], Payload::default())],
        dim,
    );
    let fallback = MockBackend::new_brute(
        BackendKind::Hnsw,
        vec![(VectorId::new(0), vec![0.0; dim], Payload::default())],
        dim,
    );

    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(&(arb_matrix_row(), 0usize..5, 1u32..=5), |(row, k_synth, label)| {
            // Synthesize an Ok(hits) shape of variable size so the
            // proptest randomizes the result body too.
            let synth_hits: Vec<(VectorId, f32)> = (0..k_synth)
                .map(|i| (VectorId::new(100 + i as u32), i as f32 * 0.1))
                .collect();
            let alt_hits: Vec<(VectorId, f32)> = (0..k_synth)
                .map(|i| (VectorId::new(900 + i as u32), 0.5 + i as f32 * 0.1))
                .collect();

            // Configure both backends per the matrix row.
            match row {
                MatrixRow::OkPrimary => {
                    primary.set_response_override(MockResponse::Ok(synth_hits.clone()));
                    // Fallback set to UnsupportedFilter to prove
                    // it's never consulted on an Ok primary.
                    fallback.set_response_override(MockResponse::UnsupportedFilter);
                }
                MatrixRow::PrimaryUnsupportedFallbackOk => {
                    primary.set_response_override(MockResponse::UnsupportedFilter);
                    fallback.set_response_override(MockResponse::Ok(synth_hits.clone()));
                }
                MatrixRow::BothUnsupported => {
                    primary.set_response_override(MockResponse::UnsupportedFilter);
                    fallback.set_response_override(MockResponse::UnsupportedFilter);
                }
                MatrixRow::PrimaryUnsupportedFallbackDim => {
                    primary.set_response_override(MockResponse::UnsupportedFilter);
                    fallback.set_response_override(MockResponse::DimensionMismatch);
                }
                MatrixRow::PrimaryDim => {
                    primary.set_response_override(MockResponse::DimensionMismatch);
                    // Set fallback to a distinct Ok(alt_hits) so a
                    // contract-violating fallback dispatch would
                    // surface the WRONG ids in the result.
                    fallback.set_response_override(MockResponse::Ok(alt_hits.clone()));
                }
            }

            let p_before = primary.calls();
            let f_before = fallback.calls();
            let set = BackendSet::new().with_hnsw(&fallback).with_diskann(&primary);
            let res = set.dispatch_filtered_search(
                &q_bytes,
                10,
                &Filter::LabelEq(LabelId::new(label)),
                32,
                &kernel, Lsn::MAX,
            );
            let p_calls = primary.calls() - p_before;
            let f_calls = fallback.calls() - f_before;

            // Per amendment-04 D-3, every row pins a contracted
            // (result, primary_calls, fallback_calls) triple.
            match row {
                MatrixRow::OkPrimary => {
                    prop_assert!(
                        matches!(&res, Ok(hits) if hits == &synth_hits),
                        "F.5 P7 OkPrimary: expected Ok(synth_hits); got {res:?}"
                    );
                    prop_assert_eq!(p_calls, 1, "F.5 P7 OkPrimary: primary called");
                    prop_assert_eq!(f_calls, 0, "F.5 P7 OkPrimary: fallback NOT called");
                }
                MatrixRow::PrimaryUnsupportedFallbackOk => {
                    prop_assert!(
                        matches!(&res, Ok(hits) if hits == &synth_hits),
                        "F.5 P7 PrimaryUnsupportedFallbackOk: expected fallback's Ok; got {res:?}"
                    );
                    prop_assert_eq!(p_calls, 1, "F.5 P7: primary called");
                    prop_assert_eq!(f_calls, 1, "F.5 P7: fallback called");
                }
                MatrixRow::BothUnsupported => {
                    let ok = matches!(&res, Err(VectorIndexError::UnsupportedFilter { reason })
                        if reason.contains("primary") && reason.contains("fallback"));
                    prop_assert!(
                        ok,
                        "F.5 P7 BothUnsupported: expected combined-reason \
                         UnsupportedFilter; got {res:?}"
                    );
                    prop_assert_eq!(p_calls, 1, "F.5 P7: primary called");
                    prop_assert_eq!(f_calls, 1, "F.5 P7: fallback called");
                }
                MatrixRow::PrimaryUnsupportedFallbackDim => {
                    prop_assert!(
                        matches!(&res, Err(VectorIndexError::DimensionMismatch { .. })),
                        "F.5 P7 PrimaryUnsupportedFallbackDim: expected \
                         DimensionMismatch (real error wins); got {res:?}"
                    );
                    prop_assert_eq!(p_calls, 1, "F.5 P7: primary called");
                    prop_assert_eq!(f_calls, 1, "F.5 P7: fallback called");
                }
                MatrixRow::PrimaryDim => {
                    prop_assert!(
                        matches!(&res, Err(VectorIndexError::DimensionMismatch { .. })),
                        "F.5 P7 PrimaryDim: expected DimensionMismatch; got {res:?}"
                    );
                    prop_assert_eq!(p_calls, 1, "F.5 P7: primary called");
                    prop_assert_eq!(
                        f_calls, 0,
                        "F.5 P7: fallback NOT called — DimensionMismatch is \
                         not a routing escalation"
                    );
                }
            }
            Ok(())
        })
        .expect("F.5 P7 fault-injection escalation matrix proptest");
}

// ─── Asymmetric-empty regressions (deterministic) ───────────────
//
// Closes Reviewer Finding 1 from PR #134. Per amendment-04 D-4
// (catalog-sync invariant clause), the production catalog
// guarantees that primary.is_empty() ⇒ fallback.is_empty(). These
// two tests EXPLICITLY construct the invariant-violating "drift"
// states to pin the dispatcher's behavior so a future contract
// change surfaces predictably.

/// Forward asymmetric-empty: empty primary, non-empty fallback.
///
/// Setup:
///   - primary slot = empty DiskANN backend (`is_empty() == true`).
///   - fallback slot = populated HNSW backend.
///   - filter = Filter::LabelEq (DiskANN-preferred per the variant
///     table).
///
/// Expected per amendment-04 D-4 + dispatcher.rs (release mode):
///   - `dispatch_filtered_search` returns `Ok(Vec::new())` because
///     the primary's `is_empty()` short-circuit fires.
///   - Neither backend's `filtered_search` is invoked.
///
/// Codex retro V2 — defense-in-depth observability (debug mode):
///   - The dispatcher's empty-primary short-circuit emits a
///     `tracing::error!(catalog_invariant = "primary_empty_fallback_nonempty", …)`
///     unconditionally, plus a `debug_assert!(fb.is_empty(), …)` that
///     fires only when `debug_assertions` is on. The release-mode
///     production semantics are unchanged; the debug-mode assertion
///     surfaces catalog drift in test runs without altering the
///     pinned contract on the production hot path.
///   - In debug builds (i.e., `cargo test` default), this test
///     captures the resulting panic via `catch_unwind` and asserts
///     the message names amendment-04 D-4. Neither backend is
///     invoked because the panic fires before the
///     `filtered_search` call site.
///
/// Catalog-drift context:
///   - This is the catalog-drift scenario the reviewer (Finding 1)
///     flagged on PR #134.
///   - Per amendment-04 D-4, this state SHOULD NOT exist in
///     production — the catalog enforces
///     `primary.is_empty() ⇒ fallback.is_empty()` so an empty
///     primary implies an empty fallback.
///   - The test EXPLICITLY constructs the invariant-violating
///     state to pin the dispatcher's contracted behavior so future
///     catalog drift surfaces predictably.
///   - If a future change to the dispatcher makes it consult the
///     fallback in this scenario, this test fails — the contract
///     change must be reviewed and the amendment-04 D-4 clause
///     updated in lockstep.
///   - If this test fails BECAUSE production catalog drift is
///     allowing primary.is_empty() with fallback non-empty, that's
///     a CATALOG bug (investigate the catalog layer), not a
///     dispatcher bug.
///
/// **Name is load-bearing**: amendment-04 D-4 references this exact
/// test name verbatim. Do not rename.
#[test]
fn dispatcher_asymmetric_empty_violates_catalog_contract() {
    let dim = 8;
    let kernel = L2F32;
    let primary = MockBackend::empty(BackendKind::DiskAnn, dim);

    // Populated fallback — the catalog-drift scenario.
    let v = vec![0.1_f32; dim];
    let payload = Payload {
        tenant_id: Some(TenantId::DEFAULT),
        labels: vec![LabelId::new(1)],
        properties: HashMap::new(),
        ..Payload::default()
    };
    let fallback_data = vec![
        (VectorId::new(0), v.clone(), payload.clone()),
        (VectorId::new(1), v.clone(), payload),
    ];
    let fallback = MockBackend::new_brute(BackendKind::Hnsw, fallback_data, dim);

    assert!(primary.is_empty(), "F.5 setup: primary must be empty");
    assert!(
        !fallback.is_empty(),
        "F.5 setup: fallback must be non-empty"
    );

    let set = BackendSet::new()
        .with_hnsw(&fallback)
        .with_diskann(&primary);
    let q = bytes_of(&v);

    if cfg!(debug_assertions) {
        // Codex V2 — debug-mode defense-in-depth path. The
        // dispatcher's `debug_assert!` fires before the short-
        // circuit returns. Suppress the panic-handler stderr
        // noise during the catch_unwind so the test output stays
        // clean, then assert the panic payload names amendment-04
        // D-4.
        let prior_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set.dispatch_filtered_search(
                &q,
                5,
                &Filter::LabelEq(LabelId::new(1)),
                32,
                &kernel,
                Lsn::MAX,
            )
        }));
        std::panic::set_hook(prior_hook);

        match result {
            Ok(_) => panic!(
                "V2 codex retro: debug_assert! on amendment-04 D-4 drift \
                 must fire in debug builds"
            ),
            Err(payload) => {
                let msg: String = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| {
                        payload
                            .downcast_ref::<&'static str>()
                            .map(|s| (*s).to_owned())
                    })
                    .unwrap_or_default();
                assert!(
                    msg.contains("amendment-04 D-4 violated"),
                    "V2 codex retro: expected debug_assert! message naming \
                     amendment-04 D-4; got: {msg}"
                );
            }
        }
        // The debug_assert! fires before either backend's
        // filtered_search would have been invoked.
        assert_eq!(primary.calls(), 0, "primary must not be invoked");
        assert_eq!(fallback.calls(), 0, "fallback must not be invoked");
    } else {
        // Release-mode pinned contract per amendment-04 D-4: the
        // empty-primary short-circuit returns Ok(vec![]) without
        // consulting either backend. Production semantics are
        // unchanged from PR #134.
        let res = set
            .dispatch_filtered_search(
                &q,
                5,
                &Filter::LabelEq(LabelId::new(1)),
                32,
                &kernel,
                Lsn::MAX,
            )
            .expect("dispatch must succeed via short-circuit");

        assert!(
            res.is_empty(),
            "F.5 D-4 contract: primary.is_empty() short-circuits to Ok(vec![]); \
             got {res:?}"
        );
        assert_eq!(
            primary.calls(),
            0,
            "F.5 D-4 contract: empty primary's filtered_search must NOT be \
             invoked; called {} times",
            primary.calls()
        );
        assert_eq!(
            fallback.calls(),
            0,
            "F.5 D-4 contract: empty-primary short-circuit fires BEFORE the \
             fallback is consulted; fallback called {} times",
            fallback.calls()
        );
    }
}

/// Reverse asymmetric-empty: rejecting primary, empty fallback.
///
/// Setup:
///   - primary slot = rejecting DiskANN backend (always returns
///     `Err(UnsupportedFilter)`).
///   - fallback slot = empty HNSW backend (`is_empty() == true`).
///   - filter = Filter::LabelEq (DiskANN-preferred).
///
/// Expected per dispatcher.rs:437–439:
///   - Dispatcher invokes primary; primary returns
///     `Err(UnsupportedFilter)` → escalation path entered.
///   - Dispatcher checks `fallback.is_empty()`, sees `true`,
///     short-circuits to `Ok(Vec::new())` WITHOUT calling fallback.
///   - `primary.calls() == 1`, `fallback.calls() == 0`.
///
/// Catalog-drift context:
///   - This is the "reverse asymmetry" mirror of Test A: catalog
///     drift the other direction (primary present-but-rejecting,
///     fallback empty).
///   - Per amendment-04 D-4, this state also SHOULD NOT exist in
///     production — an empty fallback with a present, rejecting
///     primary indicates the catalog has not kept the two backend
///     slots in sync.
///   - The test pins the contract that the secondary empty-check
///     fires correctly; same operational meaning as Test A.
#[test]
fn dispatcher_asymmetric_empty_reverse_after_unsupported_escalation() {
    let dim = 8;
    let kernel = L2F32;
    let primary = MockBackend::new_rejecting(BackendKind::DiskAnn, dim);
    let fallback = MockBackend::empty(BackendKind::Hnsw, dim);

    assert!(
        !primary.is_empty(),
        "F.5 setup: rejecting primary must report non-empty so the \
         dispatcher invokes its filtered_search"
    );
    assert!(fallback.is_empty(), "F.5 setup: fallback must be empty");

    let set = BackendSet::new()
        .with_hnsw(&fallback)
        .with_diskann(&primary);
    let q_bytes = bytes_of(&vec![0.0_f32; dim]);
    let res = set
        .dispatch_filtered_search(
            &q_bytes,
            5,
            &Filter::LabelEq(LabelId::new(1)),
            32,
            &kernel,
            Lsn::MAX,
        )
        .expect("dispatch must succeed via secondary short-circuit");

    assert!(
        res.is_empty(),
        "F.5 D-4 contract: empty-fallback secondary short-circuit returns \
         Ok(vec![]); got {res:?}"
    );
    assert_eq!(
        primary.calls(),
        1,
        "F.5 D-4 contract: primary must be invoked exactly once before \
         escalation; called {} times",
        primary.calls()
    );
    assert_eq!(
        fallback.calls(),
        0,
        "F.5 D-4 contract: empty-fallback short-circuit fires BEFORE the \
         fallback's filtered_search is invoked; called {} times",
        fallback.calls()
    );
}
