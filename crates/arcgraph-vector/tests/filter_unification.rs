//! Cross-backend Filter unification correctness pins.
//!
//! Per ADR-035 amendment-03 (issue #127). This file pins the
//! invariants that the canonical [`arcgraph_vector::Filter`]
//! enum imposes across the F.2 (HNSW) and F.3 (DiskANN)
//! backends.
//!
//! ## Pinned invariants
//!
//! Equivalence under [`Filter::LabelEq`] — same dataset + same
//! query + same single-label filter yields a high jaccard
//! overlap between the two backends' top-10 results. The exact
//! ranking can differ (HNSW's stochastic graph and Vamana's
//! deterministic α-prune produce different topologies) but the
//! matching set must be substantially the same.
//!
//! DiskANN v1.0 capability gate — every variant outside
//! [`Filter::Any`] / [`Filter::LabelEq`] surfaces as
//! [`VectorIndexError::UnsupportedFilter`]. The Phase 6 F.4
//! dispatcher inspects this error and routes such filters to
//! HNSW.
//!
//! HNSW universal coverage — every canonical variant evaluates
//! correctly against the F.2 payload sidecar.
//!
//! Tenant isolation regression guard for I-V2 — a tenant filter
//! on HNSW returns only the matching tenant's vectors, even
//! when both tenants live in the same arena.
//!
//! These pins guard the contract that Phase 6 F.4 builds on:
//! the dispatcher's input is `&Filter`; cross-backend dispatch
//! is sound iff the backends agree on the variant semantics
//! they BOTH support, and reject (rather than silently mis-
//! handle) variants they don't.

use std::collections::HashSet;

use arcgraph_core::{LabelId, Lsn, StringId, TenantId};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{Encoding, Filter, Metric, PropertyValue, VectorIndexError};

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

/// Deterministic seeded set of `n` unit vectors at dimension
/// `dim`. Same seed → same dataset across backends.
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

/// Jaccard overlap between two id sets — `|A ∩ B| / |A ∪ B|`.
/// `1.0` iff the sets are equal; `0.0` iff disjoint.
fn jaccard(a: &[VectorId], b: &[VectorId]) -> f64 {
    let sa: HashSet<VectorId> = a.iter().copied().collect();
    let sb: HashSet<VectorId> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count() as f64;
    let uni = sa.union(&sb).count() as f64;
    if uni == 0.0 { 1.0 } else { inter / uni }
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

/// Build an HNSW with the F.2 filtered-insert path. Each
/// vector's payload carries the same `Option<DiskAnnLabelId>` as
/// the DiskANN sibling so cross-backend results compare on the
/// same payload schema.
fn build_hnsw_with_labels(
    raw: &[Vec<f32>],
    labels: &[Option<DiskAnnLabelId>],
    dim: usize,
    params: HnswParams,
) -> FilteredHnsw {
    let mut g = FilteredHnsw::new(params, dim, &L2F32);
    for (i, v) in raw.iter().enumerate() {
        let payload = match labels[i] {
            Some(l) => Payload::with_labels(vec![LabelId::new(l)]),
            None => Payload::with_labels(vec![]),
        };
        g.filtered_insert(VectorId::new(i as u32), &fxd(v), payload, &L2F32)
            .expect("filtered_insert");
    }
    g
}

// ─── 1. Equivalence under Filter::LabelEq ───────────────────────

#[test]
fn filter_label_eq_hnsw_diskann_same_results() {
    // Pin: same dataset + same Filter::LabelEq(l) + high beam
    // width yields nearly-identical top-10 result sets across
    // both backends. The matching subset is dense (label-0
    // covers half the dataset) and well-separated; with l_search
    // / ef = 96 the recall@10 against brute-force is ≥ 0.95 on
    // each backend, and the cross-backend jaccard is ≥ 0.6
    // (allowing for ranking-tie noise on the matching tail).

    let n = 100;
    let dim = 8;
    let raw = unit_vectors(0xC127_B001, n, dim);
    // 50 label-0, 50 label-1.
    let labels: Vec<Option<DiskAnnLabelId>> = (0..n)
        .map(|i| Some(if i < 50 { 0_u32 } else { 1_u32 }))
        .collect();

    let diskann_params = DiskAnnParams {
        r: 16,
        alpha: 1.2,
        l_construction: 48,
        l_search_default: 96,
        ..DiskAnnParams::default()
    };
    let diskann = build_diskann(&raw, &labels, diskann_params);
    let hnsw = build_hnsw_with_labels(&raw, &labels, dim, HnswParams::default());

    // Query is the first label-0 vector; both backends search
    // for the same canonical Filter::LabelEq(0).
    let q = fxd(&raw[0]);
    let filter = Filter::label_eq(0_u32);
    let k = 10;

    let diskann_hits: Vec<VectorId> = diskann
        .filtered_search(&q, k, &filter, 96, &L2F32, Lsn::MAX)
        .expect("diskann filtered_search")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let hnsw_hits: Vec<VectorId> = hnsw
        .filtered_search(&q, k, &filter, 96, &L2F32, Lsn::MAX)
        .expect("hnsw filtered_search")
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    // Both must obey the filter — only label-0 ids (idx < 50).
    for id in &diskann_hits {
        assert!(
            (id.raw() as usize) < 50,
            "DiskANN returned non-label-0 id {id:?}"
        );
    }
    for id in &hnsw_hits {
        assert!(
            (id.raw() as usize) < 50,
            "HNSW returned non-label-0 id {id:?}"
        );
    }

    // Both must return exactly k results (label-0 has 50
    // members; recall@10 has plenty of room).
    assert_eq!(diskann_hits.len(), k, "DiskANN top-{k} short");
    assert_eq!(hnsw_hits.len(), k, "HNSW top-{k} short");

    // Cross-backend jaccard: ≥ 0.6 at k=10 means at least 7
    // of 10 ids overlap. ANN backends with different graph
    // topologies routinely tie-break differently on the recall
    // tail; 0.6 is the empirical floor on a synthetic
    // well-separated dataset of this size.
    let overlap = jaccard(&diskann_hits, &hnsw_hits);
    assert!(
        overlap >= 0.6,
        "Cross-backend Filter::LabelEq jaccard {overlap:.3} < 0.6; \
        DiskANN={diskann_hits:?}, HNSW={hnsw_hits:?}"
    );
}

// ─── 2. DiskANN v1.0 capability gate ───────────────────────────

#[test]
fn filter_label_in_hnsw_works_diskann_unsupported() {
    let n = 30;
    let dim = 4;
    let raw = unit_vectors(0xC127_B002, n, dim);
    let labels: Vec<Option<DiskAnnLabelId>> = (0..n).map(|i| Some((i % 3) as u32)).collect();

    let diskann_params = DiskAnnParams {
        r: 8,
        alpha: 1.2,
        l_construction: 16,
        l_search_default: 32,
        ..DiskAnnParams::default()
    };
    let diskann = build_diskann(&raw, &labels, diskann_params);
    let hnsw = build_hnsw_with_labels(&raw, &labels, dim, HnswParams::default());

    let q = fxd(&raw[0]);
    let filter = Filter::label_in([0_u32, 1]);

    // F.3 DiskANN: rejects with UnsupportedFilter.
    let r = diskann.filtered_search(&q, 5, &filter, 32, &L2F32, Lsn::MAX);
    assert!(
        matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
        "DiskANN must reject Filter::LabelIn with UnsupportedFilter; got {r:?}"
    );

    // F.2 HNSW: returns results that satisfy the filter.
    let hnsw_hits = hnsw
        .filtered_search(&q, 5, &filter, 32, &L2F32, Lsn::MAX)
        .expect("hnsw filtered_search");
    assert!(!hnsw_hits.is_empty(), "HNSW LabelIn returned empty");
    for (id, _) in &hnsw_hits {
        let lbl = (id.raw() as usize) % 3;
        assert!(
            lbl == 0 || lbl == 1,
            "HNSW LabelIn returned id {id:?} with label {lbl} not in {{0, 1}}"
        );
    }
}

#[test]
fn filter_compound_and_or_hnsw_works_diskann_unsupported() {
    let n = 24;
    let dim = 4;
    let raw = unit_vectors(0xC127_B003, n, dim);
    let labels: Vec<Option<DiskAnnLabelId>> = (0..n).map(|i| Some((i % 4) as u32)).collect();

    let diskann_params = DiskAnnParams {
        r: 8,
        alpha: 1.2,
        l_construction: 16,
        l_search_default: 32,
        ..DiskAnnParams::default()
    };
    let diskann = build_diskann(&raw, &labels, diskann_params);
    let hnsw = build_hnsw_with_labels(&raw, &labels, dim, HnswParams::default());

    let q = fxd(&raw[0]);

    // Filter::And — DiskANN rejects.
    let and = Filter::And(vec![
        Filter::LabelIn(vec![LabelId::new(0), LabelId::new(1)]),
        Filter::Tenant(TenantId::DEFAULT),
    ]);
    let r = diskann.filtered_search(&q, 5, &and, 32, &L2F32, Lsn::MAX);
    assert!(
        matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
        "DiskANN must reject Filter::And; got {r:?}"
    );
    let hnsw_and = hnsw
        .filtered_search(&q, 5, &and, 32, &L2F32, Lsn::MAX)
        .expect("hnsw And");
    assert!(
        !hnsw_and.is_empty(),
        "HNSW And returned empty (label 0 or 1 with default tenant should match)"
    );

    // Filter::Or — DiskANN rejects.
    let or = Filter::Or(vec![
        Filter::LabelIn(vec![LabelId::new(0)]),
        Filter::LabelIn(vec![LabelId::new(2)]),
    ]);
    let r = diskann.filtered_search(&q, 5, &or, 32, &L2F32, Lsn::MAX);
    assert!(
        matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
        "DiskANN must reject Filter::Or; got {r:?}"
    );
    let hnsw_or = hnsw
        .filtered_search(&q, 5, &or, 32, &L2F32, Lsn::MAX)
        .expect("hnsw Or");
    assert!(!hnsw_or.is_empty(), "HNSW Or returned empty");
    for (id, _) in &hnsw_or {
        let lbl = (id.raw() as usize) % 4;
        assert!(
            lbl == 0 || lbl == 2,
            "HNSW Or returned id {id:?} with label {lbl} not in {{0, 2}}"
        );
    }

    // Filter::PropertyEq — DiskANN rejects.
    let prop = Filter::PropertyEq(StringId::new(0), PropertyValue::U32(0));
    let r = diskann.filtered_search(&q, 5, &prop, 32, &L2F32, Lsn::MAX);
    assert!(
        matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
        "DiskANN must reject Filter::PropertyEq; got {r:?}"
    );

    // Filter::Tenant — DiskANN rejects.
    let tenant = Filter::Tenant(TenantId::new(7));
    let r = diskann.filtered_search(&q, 5, &tenant, 32, &L2F32, Lsn::MAX);
    assert!(
        matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
        "DiskANN must reject Filter::Tenant; got {r:?}"
    );
}

// ─── 3. Filter::Any — both backends accept ─────────────────────

#[test]
fn filter_any_both_backends_return_unfiltered_results() {
    let n = 20;
    let dim = 4;
    let raw = unit_vectors(0xC127_B004, n, dim);
    // Mixed labels — Any must return regardless.
    let labels: Vec<Option<DiskAnnLabelId>> = (0..n).map(|i| Some((i % 5) as u32)).collect();

    let diskann_params = DiskAnnParams {
        r: 8,
        alpha: 1.2,
        l_construction: 16,
        l_search_default: 32,
        ..DiskAnnParams::default()
    };
    let diskann = build_diskann(&raw, &labels, diskann_params);
    let hnsw = build_hnsw_with_labels(&raw, &labels, dim, HnswParams::default());

    let q = fxd(&raw[0]);
    let filter = Filter::any();
    let k = 5;

    let diskann_hits = diskann
        .filtered_search(&q, k, &filter, 32, &L2F32, Lsn::MAX)
        .expect("diskann Any");
    let hnsw_hits = hnsw
        .filtered_search(&q, k, &filter, 32, &L2F32, Lsn::MAX)
        .expect("hnsw Any");

    assert_eq!(diskann_hits.len(), k, "DiskANN Any short");
    assert_eq!(hnsw_hits.len(), k, "HNSW Any short");
    // First hit on both should be the query vector (id 0)
    // since the dataset contains the query exactly.
    assert_eq!(
        diskann_hits[0].0.raw(),
        0,
        "DiskANN Any nearest-neighbor not the query itself"
    );
    assert_eq!(
        hnsw_hits[0].0.raw(),
        0,
        "HNSW Any nearest-neighbor not the query itself"
    );
}

// ─── 4. Tenant isolation — regression guard for I-V2 ────────────

#[test]
fn filter_tenant_isolation_hnsw_respects_tenant_boundary() {
    // I-V2 sub-property: a Filter::Tenant(t) on HNSW returns
    // only vectors whose payload tenant matches `t`, even when
    // both tenants live in the same arena. (Cross-tenant arenas
    // are forbidden in production per ADR-011 §6.1, but this
    // pin guards the in-arena evaluator's correctness — the
    // F.4 dispatcher relies on it.)
    let dim = 4;
    let mut hnsw = FilteredHnsw::new(HnswParams::default(), dim, &L2F32);

    let t1 = TenantId::new(1);
    let t2 = TenantId::new(2);

    // Tenant 1 vectors: ids 0..5.
    for i in 0..5_u32 {
        let payload = Payload {
            tenant_id: Some(t1),
            labels: vec![LabelId::new(0)],
            ..Default::default()
        };
        hnsw.filtered_insert(
            VectorId::new(i),
            &fxd(&[i as f32 * 0.01, 0.0, 0.0, 0.0]),
            payload,
            &L2F32,
        )
        .unwrap();
    }
    // Tenant 2 vectors: ids 10..15.
    for i in 10..15_u32 {
        let payload = Payload {
            tenant_id: Some(t2),
            labels: vec![LabelId::new(0)],
            ..Default::default()
        };
        hnsw.filtered_insert(
            VectorId::new(i),
            &fxd(&[i as f32 * 0.01, 0.0, 0.0, 0.0]),
            payload,
            &L2F32,
        )
        .unwrap();
    }

    // Query at origin; without a filter we'd see both tenants.
    let q = fxd(&[0.0, 0.0, 0.0, 0.0]);

    let r_t1 = hnsw
        .filtered_search(&q, 10, &Filter::tenant(t1), 32, &L2F32, Lsn::MAX)
        .expect("hnsw tenant filter");
    for (id, _) in &r_t1 {
        assert!(
            id.raw() < 5,
            "Filter::tenant(t1) leaked tenant 2 vector {id:?}"
        );
    }
    assert!(!r_t1.is_empty(), "tenant 1 filter returned empty");

    let r_t2 = hnsw
        .filtered_search(&q, 10, &Filter::tenant(t2), 32, &L2F32, Lsn::MAX)
        .expect("hnsw tenant filter");
    for (id, _) in &r_t2 {
        assert!(
            id.raw() >= 10 && id.raw() < 15,
            "Filter::tenant(t2) leaked tenant 1 vector {id:?}"
        );
    }
    assert!(!r_t2.is_empty(), "tenant 2 filter returned empty");
}

// ─── 5. Constructor parity — `u32` and `LabelId` interchangeable ──

#[test]
fn filter_label_eq_u32_and_label_id_constructors_equivalent() {
    // Pin: callers can pass either `u32` or `LabelId` to
    // `Filter::label_eq` and observe identical behavior. This
    // keeps the F.3 callsite ergonomics from breaking under
    // amendment-03 while making `LabelId` the canonical type.
    let from_u32 = Filter::label_eq(7_u32);
    let from_label = Filter::label_eq(LabelId::new(7));
    assert_eq!(from_u32, from_label);
    assert_eq!(from_u32.required_label(), Some(LabelId::new(7)));
}
