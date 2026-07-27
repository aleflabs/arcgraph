//! Path A boundary pins for the F.4 selectivity dispatcher.
//!
//! Per ADR-035 amendment-04 + Phase 6 F.4 slice prompt
//! ("12+ pins: per-variant routing, escalation, tenant isolation,
//! LabelEq fast-path, recall floor, determinism, empty-result,
//! compound routing").
//!
//! ## Pinned invariants
//!
//! 1. **Per-variant routing (7 pins)** — for each canonical
//!    [`Filter`] variant, the dispatcher routes to the backend
//!    named by the amendment-04 D-2 routing table. The integration
//!    test uses a wrapper backend (`InstrumentedBackend`) that
//!    delegates to a real F.2 / F.3 instance while counting
//!    `filtered_search` invocations — both correctness (the
//!    returned hits match what the chosen backend produces
//!    standalone) AND routing (which backend was called) are
//!    pinned.
//! 2. **Escalation pin** — when the primary returns
//!    [`VectorIndexError::UnsupportedFilter`], the dispatcher
//!    routes to the fallback if present. Built around the v1.0
//!    case where DiskANN rejects an unsupported variant; the
//!    routing table normally bypasses DiskANN for those variants
//!    (HnswOnly preference) so the fallback path is exercised
//!    via a custom `RejectingBackend` wrapper that forces
//!    DiskANN-like rejection on a `LabelEq` query.
//! 3. **Tenant isolation pin** — cross-cuts I-V2; the dispatcher
//!    respects HNSW's payload tenant predicate.
//! 4. **LabelEq fast-path pin** — dispatcher result on
//!    [`Filter::LabelEq`] matches `DiskAnnGraph::filtered_search`
//!    standalone, byte-identically (both id sequence and float
//!    distances).
//! 5. **Recall floor pin** — for every supported [`Filter`]
//!    variant, dispatcher recall@10 vs brute-force ground truth
//!    is ≥ 0.85 (cross-cuts I-V4 at the dispatcher layer).
//! 6. **Determinism pin** — three identical dispatch calls
//!    produce identical outputs.
//! 7. **Empty result pin** — filters matching zero vectors
//!    return `Ok(Vec::new())`, not an error.
//! 8. **Compound filter routing pin** — `And` / `Or` filters
//!    route to HNSW only (DiskANN's call counter stays at 0).
//! 9. **No-backends pin** — empty [`BackendSet`] surfaces
//!    [`VectorIndexError::UnsupportedFilter`].
//! 10. **Empty-graph short-circuit pin** — dispatching against
//!     an empty primary backend short-circuits to `Ok(Vec::new())`
//!     without invoking the backend.
//!
//! These pins guard the contract that F.5 multi-tenant proptest,
//! G.4 / G.5 storage integration, M4 query-layer integration, and
//! M5 MCP tool layer all rest on. Failures here block any future
//! consumer of the dispatcher.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::{LabelId, Lsn, StringId, TenantId};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::{DistanceKernel, L2F32};
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{
    BackendKind, BackendSet, DispatchPreference, Encoding, Filter, FilteredVectorIndex, Metric,
    PropertyValue, VectorIndexError, dispatch_preference,
};

use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;

// ─── shared helpers ──────────────────────────────────────────────

fn fxd(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
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

fn unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
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

/// Build a DiskANN graph with the F.3 filtered-build path.
fn build_diskann(
    raw: &[Vec<f32>],
    labels: &[Option<DiskAnnLabelId>],
    params: DiskAnnParams,
) -> DiskAnnGraph {
    let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32))
        .expect("DiskAnnGraph::new");
    let owned: Vec<(VectorId, Vec<u8>)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), fxd(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    g.build_filtered(&pairs, labels, &L2F32)
        .expect("build_filtered");
    g
}

// ─── instrumented backend wrappers ───────────────────────────────
//
// The integration tests need to verify "which backend the
// dispatcher invoked" — not just the result. The unit tests in
// `src/dispatcher.rs` use pure-mock backends; the integration
// tests use REAL F.2 / F.3 backends wrapped in an instrumented
// wrapper that increments an atomic counter on every
// `filtered_search` call. This validates that the routing table
// + the real backends compose correctly under the dispatcher
// contract.

/// Wrapper that delegates to an inner backend while counting
/// invocations. The kind is carried separately so we can confirm
/// at assertion time that the dispatcher hit the expected slot.
struct InstrumentedBackend<'a> {
    inner: &'a dyn FilteredVectorIndex,
    kind: BackendKind,
    calls: AtomicUsize,
}

impl<'a> InstrumentedBackend<'a> {
    fn new(inner: &'a dyn FilteredVectorIndex, kind: BackendKind) -> Self {
        Self {
            inner,
            kind,
            calls: AtomicUsize::new(0),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl FilteredVectorIndex for InstrumentedBackend<'_> {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .filtered_search(query, k, filter, ef, kernel, read_lsn)
    }
}

/// Wrapper that ALWAYS rejects the filter with
/// [`VectorIndexError::UnsupportedFilter`]. Used to drive the
/// escalation pin without modifying real F.3 behavior.
struct RejectingBackend<'a> {
    inner: &'a dyn FilteredVectorIndex,
    kind: BackendKind,
    calls: AtomicUsize,
}

impl<'a> RejectingBackend<'a> {
    fn new(inner: &'a dyn FilteredVectorIndex, kind: BackendKind) -> Self {
        Self {
            inner,
            kind,
            calls: AtomicUsize::new(0),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl FilteredVectorIndex for RejectingBackend<'_> {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn filtered_search(
        &self,
        _query: &[u8],
        _k: usize,
        _filter: &Filter,
        _ef: usize,
        _kernel: &dyn DistanceKernel,
        _read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(VectorIndexError::UnsupportedFilter {
            reason: format!(
                "rejecting backend {} simulated rejection",
                self.kind.as_str()
            ),
        })
    }
}

/// Brute-force filtered top-k via HNSW's payload predicate (the
/// canonical Filter::matches semantics). Used as the recall-floor
/// ground truth. Returns just the sorted ids.
fn brute_force_top_k_filtered(
    raw: &[Vec<f32>],
    payloads: &[Payload],
    query: &[f32],
    k: usize,
    filter: &Filter,
) -> Vec<VectorId> {
    let mut scored: Vec<(VectorId, f32)> = raw
        .iter()
        .enumerate()
        .filter(|(i, _)| filter.matches(&payloads[*i]))
        .map(|(i, v)| {
            // L2 distance squared (consistent with L2F32's
            // direction).
            let d = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            (VectorId::new(i as u32), d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.raw().cmp(&b.0.raw())));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

// Standard test fixture: 80 normalized 8-dim vectors with
// label distribution 0=40, 1=20, 2=20, plus tenant assignments
// 0..40 = tenant 1, 40..80 = tenant 2.
struct Fixture {
    raw: Vec<Vec<f32>>,
    labels: Vec<Option<DiskAnnLabelId>>,
    payloads: Vec<Payload>, // for HNSW (carries tenant + labels)
    diskann: DiskAnnGraph,
    hnsw: FilteredHnsw,
    dim: usize,
}

impl Fixture {
    fn build() -> Self {
        let n = 80;
        let dim = 8;
        let raw = unit_vectors(0xF400_C001, n, dim);

        let labels: Vec<Option<DiskAnnLabelId>> = (0..n)
            .map(|i| {
                if i < 40 {
                    Some(0_u32)
                } else if i < 60 {
                    Some(1_u32)
                } else {
                    Some(2_u32)
                }
            })
            .collect();
        let tenants: Vec<TenantId> = (0..n)
            .map(|i| TenantId::new(if i < 40 { 1 } else { 2 }))
            .collect();
        let payloads: Vec<Payload> = (0..n)
            .map(|i| Payload {
                tenant_id: Some(tenants[i]),
                labels: vec![LabelId::new(labels[i].unwrap())],
                ..Default::default()
            })
            .collect();

        let diskann_params = DiskAnnParams {
            r: 16,
            alpha: 1.2,
            l_construction: 48,
            l_search_default: 64,
            ..DiskAnnParams::default()
        };
        let diskann = build_diskann(&raw, &labels, diskann_params);

        let hnsw_params = HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            seed: 99,
        };
        // Build HNSW from the payloads (which carry tenant + label).
        let mut hnsw = FilteredHnsw::new(hnsw_params, dim, &L2F32);
        for (i, v) in raw.iter().enumerate() {
            hnsw.filtered_insert(
                VectorId::new(i as u32),
                &fxd(v),
                payloads[i].clone(),
                &L2F32,
            )
            .expect("filtered_insert");
        }

        Self {
            raw,
            labels,
            payloads,
            diskann,
            hnsw,
            dim,
        }
    }

    fn query(&self, idx: usize) -> Vec<u8> {
        fxd(&self.raw[idx])
    }
}

// ─── 1. Per-variant routing pins (7 tests) ───────────────────────

#[test]
fn dispatcher_routes_filter_any_to_diskann() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let r = set
        .dispatch_filtered_search(&f.query(0), 5, &Filter::Any, 64, &L2F32, Lsn::MAX)
        .expect("dispatch");
    assert_eq!(r.len(), 5);
    assert_eq!(d.calls(), 1, "Filter::Any routes DiskANN");
    assert_eq!(h.calls(), 0, "HNSW must not be called for Any");
}

#[test]
fn dispatcher_routes_filter_label_eq_to_diskann() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::LabelEq(LabelId::new(0)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(!r.is_empty());
    for (id, _) in &r {
        assert!(
            id.raw() < 40,
            "Filter::LabelEq(0) returned non-label-0 id {id:?}"
        );
    }
    assert_eq!(d.calls(), 1, "Filter::LabelEq routes DiskANN");
    assert_eq!(h.calls(), 0, "HNSW must not be called for LabelEq");
}

#[test]
fn dispatcher_routes_filter_tenant_to_hnsw_only() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::Tenant(TenantId::new(1)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(!r.is_empty());
    assert_eq!(h.calls(), 1, "HNSW called for Tenant");
    assert_eq!(d.calls(), 0, "DiskANN must NOT be called for Tenant");
}

#[test]
fn dispatcher_routes_filter_label_in_to_hnsw_only() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::LabelIn(vec![LabelId::new(0), LabelId::new(1)]),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(!r.is_empty());
    assert_eq!(h.calls(), 1);
    assert_eq!(d.calls(), 0, "DiskANN must NOT be called for LabelIn");
}

#[test]
fn dispatcher_routes_filter_property_eq_to_hnsw_only() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    // The fixture's payloads don't carry any properties, so the
    // filter matches nothing — but the routing still goes to HNSW
    // and returns Ok(empty), not an error.
    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::PropertyEq(StringId::new(0), PropertyValue::U32(42)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(r.is_empty(), "no payloads carry properties; expect empty");
    assert_eq!(h.calls(), 1);
    assert_eq!(d.calls(), 0, "DiskANN must NOT be called for PropertyEq");
}

#[test]
fn dispatcher_routes_filter_and_to_hnsw_only() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let and = Filter::And(vec![
        Filter::Tenant(TenantId::new(1)),
        Filter::LabelIn(vec![LabelId::new(0)]),
    ]);
    let r = set
        .dispatch_filtered_search(&f.query(0), 5, &and, 64, &L2F32, Lsn::MAX)
        .expect("dispatch");
    assert!(!r.is_empty());
    for (id, _) in &r {
        assert!(id.raw() < 40, "And-filter leaked non-tenant-1 id {id:?}");
    }
    assert_eq!(h.calls(), 1);
    assert_eq!(d.calls(), 0, "DiskANN must NOT be called for And");
}

#[test]
fn dispatcher_routes_filter_or_to_hnsw_only() {
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let or = Filter::Or(vec![
        Filter::LabelIn(vec![LabelId::new(0)]),
        Filter::LabelIn(vec![LabelId::new(2)]),
    ]);
    let r = set
        .dispatch_filtered_search(&f.query(0), 5, &or, 64, &L2F32, Lsn::MAX)
        .expect("dispatch");
    assert!(!r.is_empty());
    for (id, _) in &r {
        let lbl = f.labels[id.raw() as usize].unwrap();
        assert!(
            lbl == 0 || lbl == 2,
            "Or-filter leaked id {id:?} with label {lbl} not in {{0, 2}}"
        );
    }
    assert_eq!(h.calls(), 1);
    assert_eq!(d.calls(), 0, "DiskANN must NOT be called for Or");
}

// ─── 2. Escalation pin ───────────────────────────────────────────

#[test]
fn dispatcher_escalates_unsupported_filter_to_fallback() {
    // Wrap DiskANN in a RejectingBackend so the primary always
    // says UnsupportedFilter; HNSW (the fallback per the variant
    // table for LabelEq) handles the request. Validates the
    // amendment-04 D-3 escalation contract using a real HNSW
    // backend on the fallback path.
    let f = Fixture::build();
    let rejecting_d = RejectingBackend::new(&f.diskann, BackendKind::DiskAnn);
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&rejecting_d);

    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::LabelEq(LabelId::new(0)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(!r.is_empty());
    for (id, _) in &r {
        assert!(
            id.raw() < 40,
            "Escalated dispatch leaked non-label-0 id {id:?}"
        );
    }
    assert_eq!(rejecting_d.calls(), 1, "DiskANN tried first (rejected)");
    assert_eq!(h.calls(), 1, "HNSW called via escalation");
}

// ─── 3. Tenant isolation pin (cross-cuts I-V2) ───────────────────

#[test]
fn dispatcher_respects_tenant_isolation_via_hnsw() {
    // I-V2 sub-property at the dispatcher layer: a tenant filter
    // dispatched through F.4 returns only the matching tenant's
    // vectors. The dispatcher does not enforce isolation
    // directly; it routes to HNSW which evaluates the predicate.
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let r_t1 = set
        .dispatch_filtered_search(
            &f.query(0),
            10,
            &Filter::Tenant(TenantId::new(1)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch t1");
    for (id, _) in &r_t1 {
        assert!(
            id.raw() < 40,
            "Tenant 1 dispatch leaked tenant 2 vector {id:?}"
        );
    }
    assert!(!r_t1.is_empty());

    let r_t2 = set
        .dispatch_filtered_search(
            &f.query(60),
            10,
            &Filter::Tenant(TenantId::new(2)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch t2");
    for (id, _) in &r_t2 {
        assert!(
            id.raw() >= 40,
            "Tenant 2 dispatch leaked tenant 1 vector {id:?}"
        );
    }
    assert!(!r_t2.is_empty());
}

// ─── 4. LabelEq fast-path byte-identity pin ──────────────────────

#[test]
fn dispatcher_label_eq_results_match_diskann_standalone_byte_identical() {
    // The dispatcher must not drop, reorder, or transform results
    // from the chosen backend. For Filter::LabelEq, dispatch
    // routes to DiskANN — the result must be byte-identical
    // (id sequence + float distances) to calling
    // DiskAnnGraph::filtered_search directly.
    let f = Fixture::build();
    let set = BackendSet::new()
        .with_hnsw(&f.hnsw)
        .with_diskann(&f.diskann);

    let q = f.query(5);
    let filter = Filter::LabelEq(LabelId::new(0));

    let via_dispatch = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .expect("dispatch");
    let standalone = f
        .diskann
        .filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .expect("standalone");

    assert_eq!(
        via_dispatch.len(),
        standalone.len(),
        "Dispatch result count differs from standalone DiskANN"
    );
    for (a, b) in via_dispatch.iter().zip(standalone.iter()) {
        assert_eq!(a.0, b.0, "Dispatch id sequence diverges from standalone");
        assert!(
            a.1.to_bits() == b.1.to_bits(),
            "Dispatch distance bits diverge from standalone for id {:?}: {} vs {}",
            a.0,
            a.1,
            b.1
        );
    }
}

// ─── 5. Recall floor pin (cross-cuts I-V4) ───────────────────────

#[test]
fn dispatcher_recall_floor_at_least_0_85_across_supported_variants() {
    // For every supported Filter variant against the fixture,
    // the dispatcher's recall@10 vs brute-force ground truth is
    // ≥ 0.85. Cross-cuts the AC-5/AC-6 floor at the dispatcher
    // layer so a routing-induced regression in either backend
    // surfaces here.
    let f = Fixture::build();
    let set = BackendSet::new()
        .with_hnsw(&f.hnsw)
        .with_diskann(&f.diskann);

    let queries = unit_vectors(0xF400_F101, 10, f.dim);
    let k = 10;

    // Build the variant set: each variant tested against a
    // selectivity bucket that produces enough matches for a
    // recall measurement.
    let variants: Vec<(&'static str, Filter)> = vec![
        ("Any", Filter::Any),
        ("LabelEq(0)", Filter::LabelEq(LabelId::new(0))),
        // LabelIn covering labels 0+1 → 60/80 = 0.75 selectivity
        (
            "LabelIn(0,1)",
            Filter::LabelIn(vec![LabelId::new(0), LabelId::new(1)]),
        ),
        // Tenant 1 → 40/80 = 0.5 selectivity
        ("Tenant(1)", Filter::Tenant(TenantId::new(1))),
        // And: tenant 1 + label 0 → 40/80 = 0.5 (every tenant 1
        // vector has label 0 by fixture construction)
        (
            "And(Tenant(1), LabelEq(0))",
            Filter::And(vec![
                Filter::Tenant(TenantId::new(1)),
                Filter::LabelEq(LabelId::new(0)),
            ]),
        ),
        // Or: label 1 OR label 2 → 40/80 = 0.5
        (
            "Or(LabelIn(1), LabelIn(2))",
            Filter::Or(vec![
                Filter::LabelIn(vec![LabelId::new(1)]),
                Filter::LabelIn(vec![LabelId::new(2)]),
            ]),
        ),
    ];

    for (name, filter) in &variants {
        let mut total_recall = 0.0_f64;
        let mut q_count = 0_usize;
        for q in &queries {
            let bf = brute_force_top_k_filtered(&f.raw, &f.payloads, q, k, filter);
            if bf.is_empty() {
                continue;
            }
            let res = set
                .dispatch_filtered_search(&fxd(q), k, filter, 100, &L2F32, Lsn::MAX)
                .unwrap_or_else(|e| panic!("dispatch failed for {name}: {e}"));
            // Pin: every returned id satisfies the filter.
            for (id, _) in &res {
                let p = f
                    .hnsw
                    .payload(*id)
                    .or_else(|| f.payloads.get(id.raw() as usize))
                    .expect("payload must exist");
                assert!(
                    filter.matches(p),
                    "Dispatcher returned id {id:?} that fails filter {name}"
                );
            }
            let h: HashSet<VectorId> = res.iter().map(|(id, _)| *id).take(k).collect();
            let b: HashSet<VectorId> = bf.iter().copied().take(k).collect();
            let inter = h.intersection(&b).count();
            let denom = b.len().min(k).max(1);
            total_recall += inter as f64 / denom as f64;
            q_count += 1;
        }
        if q_count == 0 {
            continue;
        }
        let mean = total_recall / q_count as f64;
        assert!(
            mean >= 0.85,
            "Dispatcher recall@{k}={mean:.3} for filter `{name}` < 0.85 floor"
        );
    }
}

// ─── 6. Determinism pin ──────────────────────────────────────────

#[test]
fn dispatcher_is_deterministic_across_repeated_calls() {
    // Three identical dispatch calls produce identical results.
    // Validates that the dispatcher introduces no non-determinism
    // (the underlying backends own their own determinism).
    let f = Fixture::build();
    let set = BackendSet::new()
        .with_hnsw(&f.hnsw)
        .with_diskann(&f.diskann);

    let q = f.query(0);
    let filter = Filter::LabelEq(LabelId::new(0));

    let r1 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();
    let r2 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();
    let r3 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();

    assert_eq!(r1.len(), r2.len());
    assert_eq!(r1.len(), r3.len());
    for ((a, b), c) in r1.iter().zip(r2.iter()).zip(r3.iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.0, c.0);
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        assert_eq!(a.1.to_bits(), c.1.to_bits());
    }
}

#[test]
fn dispatcher_is_deterministic_for_compound_filter() {
    // Same as above but for a compound filter that goes through
    // HNSW. Validates determinism on the HnswOnly path too.
    let f = Fixture::build();
    let set = BackendSet::new()
        .with_hnsw(&f.hnsw)
        .with_diskann(&f.diskann);

    let q = f.query(10);
    let filter = Filter::And(vec![
        Filter::Tenant(TenantId::new(1)),
        Filter::LabelEq(LabelId::new(0)),
    ]);

    let r1 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();
    let r2 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();
    let r3 = set
        .dispatch_filtered_search(&q, 7, &filter, 64, &L2F32, Lsn::MAX)
        .unwrap();

    assert_eq!(r1, r2);
    assert_eq!(r1, r3);
}

// ─── 7. Empty-result pin ─────────────────────────────────────────

#[test]
fn dispatcher_returns_empty_vec_when_filter_matches_nothing() {
    // A filter that matches zero vectors must return
    // Ok(Vec::new()), not an error. Validated through both an
    // HnswOnly path (PropertyEq) and a DiskAnnPreferred path
    // (LabelEq on a non-existent label).
    let f = Fixture::build();
    let set = BackendSet::new()
        .with_hnsw(&f.hnsw)
        .with_diskann(&f.diskann);

    // PropertyEq — fixture payloads carry no properties.
    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::PropertyEq(StringId::new(0), PropertyValue::U32(42)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(
        r.is_empty(),
        "PropertyEq with no matching props must be empty"
    );

    // LabelEq on an unused label.
    let r = set
        .dispatch_filtered_search(
            &f.query(0),
            5,
            &Filter::LabelEq(LabelId::new(999)),
            64,
            &L2F32,
            Lsn::MAX,
        )
        .expect("dispatch");
    assert!(r.is_empty(), "LabelEq on absent label must be empty");
}

// ─── 8. Compound filter routing pin ──────────────────────────────

#[test]
fn dispatcher_compound_filters_never_invoke_diskann() {
    // Restates the per-variant pins for And/Or as a single
    // pin guarding the "DiskANN never sees compound filters at
    // v1.0" contract — load-bearing for the F.5 / G.4 follow-up
    // that lifts compound-filter support to DiskANN (an
    // amendment-05 will need to flip this assertion).
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&f.diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    let compounds = vec![
        Filter::And(vec![Filter::Any]),
        Filter::And(vec![]),
        Filter::Or(vec![Filter::Any]),
        Filter::Or(vec![]),
        Filter::And(vec![
            Filter::LabelEq(LabelId::new(0)),
            Filter::Tenant(TenantId::new(1)),
        ]),
        Filter::Or(vec![
            Filter::LabelIn(vec![LabelId::new(0)]),
            Filter::LabelIn(vec![LabelId::new(1)]),
        ]),
    ];

    for f_compound in &compounds {
        // Empty Or matches nothing — that's fine, dispatcher still
        // routes to HNSW; HNSW returns empty.
        let _ = set
            .dispatch_filtered_search(&f.query(0), 3, f_compound, 64, &L2F32, Lsn::MAX)
            .expect("dispatch");
    }

    assert_eq!(
        d.calls(),
        0,
        "DiskANN must NOT be called for any compound filter at v1.0"
    );
    assert_eq!(
        h.calls(),
        compounds.len(),
        "Every compound filter should hit HNSW exactly once"
    );
}

// ─── 9. No-backends pin ──────────────────────────────────────────

#[test]
fn dispatcher_with_no_backends_surfaces_unsupported_filter() {
    let set = BackendSet::new();
    let r = set.dispatch_filtered_search(&[0u8; 32], 5, &Filter::Any, 32, &L2F32, Lsn::MAX);
    match r {
        Err(VectorIndexError::UnsupportedFilter { reason }) => {
            assert!(
                reason.contains("no backend"),
                "Empty BackendSet error should name the misconfiguration; got {reason:?}"
            );
        }
        other => panic!("Empty BackendSet must return UnsupportedFilter; got {other:?}"),
    }
}

// ─── 10. Empty-graph short-circuit pin ───────────────────────────

#[test]
fn dispatcher_empty_primary_short_circuits_without_backend_call() {
    // Build a DiskANN with zero vectors → primary is_empty.
    // The dispatcher must short-circuit to Ok(Vec::new()) without
    // calling the backend's filtered_search.
    //
    // Codex retro V2: the asymmetric drift case (empty primary +
    // populated fallback) now emits an operator-visible
    // `tracing::error!` and a debug-mode-only `debug_assert!` per
    // amendment-04 D-4 defense-in-depth. The release-mode pinned
    // contract (`Ok(vec![])` + 0 calls) is preserved; the debug
    // build path is exercised via `catch_unwind` so the test stays
    // green in `cargo test` (debug) and `cargo test --release`.
    let empty_diskann = build_diskann(&[], &[], DiskAnnParams::default());
    let f = Fixture::build();
    let h = InstrumentedBackend::new(&f.hnsw, BackendKind::Hnsw);
    let d = InstrumentedBackend::new(&empty_diskann, BackendKind::DiskAnn);
    let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

    if cfg!(debug_assertions) {
        let prior_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set.dispatch_filtered_search(&[0u8; 32], 5, &Filter::Any, 32, &L2F32, Lsn::MAX)
        }));
        std::panic::set_hook(prior_hook);

        match result {
            Ok(_) => panic!(
                "V2 codex retro: debug_assert! must fire on amendment-04 \
                 D-4 drift in debug builds"
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
                    "expected V2 debug_assert!; got: {msg}"
                );
            }
        }
        assert_eq!(d.calls(), 0, "Empty DiskANN must not be called");
        assert_eq!(h.calls(), 0, "HNSW must not be called either");
    } else {
        let r = set
            .dispatch_filtered_search(&[0u8; 32], 5, &Filter::Any, 32, &L2F32, Lsn::MAX)
            .expect("dispatch");
        assert!(r.is_empty());
        assert_eq!(d.calls(), 0, "Empty DiskANN must not be called");
        assert_eq!(h.calls(), 0, "HNSW must not be called either");
    }
}

// ─── 11. Routing preference exposed correctly ───────────────────

#[test]
fn dispatch_preference_function_matches_dispatcher_routing() {
    // The public `dispatch_preference` helper must agree with
    // the dispatcher's internal routing for every variant —
    // F.5 multi-tenant proptest uses this helper to assert
    // routing without calling the dispatcher itself.
    assert_eq!(
        dispatch_preference(&Filter::Any),
        DispatchPreference::DiskAnnPreferred
    );
    assert_eq!(
        dispatch_preference(&Filter::LabelEq(LabelId::new(0))),
        DispatchPreference::DiskAnnPreferred
    );
    for f in [
        Filter::Tenant(TenantId::new(1)),
        Filter::LabelIn(vec![LabelId::new(0)]),
        Filter::PropertyEq(StringId::new(0), PropertyValue::U32(0)),
        Filter::And(vec![]),
        Filter::Or(vec![]),
    ] {
        assert_eq!(
            dispatch_preference(&f),
            DispatchPreference::HnswOnly,
            "filter {f:?} should be HnswOnly"
        );
    }
}

// ─── 12. Backend kind identity ───────────────────────────────────

#[test]
fn real_backends_advertise_correct_kind() {
    let f = Fixture::build();
    assert_eq!(
        <FilteredHnsw as FilteredVectorIndex>::kind(&f.hnsw),
        BackendKind::Hnsw
    );
    assert_eq!(
        <DiskAnnGraph as FilteredVectorIndex>::kind(&f.diskann),
        BackendKind::DiskAnn
    );
}
