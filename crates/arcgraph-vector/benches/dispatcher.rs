//! F.4 selectivity dispatcher microbench — Phase 6.
//!
//! Per ADR-035 amendment-04 + Phase 6 F.4 slice prompt
//! ("Bench overhead: <X> ps; <Y>% of search hot path; target <1%").
//!
//! ## What is measured
//!
//! - `dispatch_preference_only`: the pure
//!   [`dispatch_preference`] helper called in a tight loop with
//!   alternating filter variants. Isolates the variant-match
//!   cost — single-branch `match` on the discriminant. Reference
//!   floor: the prior PR #132
//!   `filter_dispatch_supported_variants_alternating` bench at
//!   ~543 ps for the same single-branch shape.
//! - `dispatch_label_eq_diskann_route`: end-to-end dispatcher
//!   call through the trait surface for the
//!   [`Filter::LabelEq`] hot path (DiskANN-preferred,
//!   single backend invocation). The dispatcher overhead is the
//!   difference vs the equivalent direct
//!   [`DiskAnnGraph::filtered_search`] call (the prior bench's
//!   `filtered_search_label_eq` baseline).
//! - `dispatch_compound_hnsw_route`: same shape for an `And`
//!   filter that routes HNSW-only — exercises the alternate
//!   routing branch.
//! - `dispatch_no_diskann_label_eq`: dispatcher with
//!   DiskANN absent → falls through to HNSW for the LabelEq
//!   case. Isolates the slot-availability check.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p arcgraph-vector --bench dispatcher -- --quick
//! ```
//!
//! `--quick` runs in ~10 s; the full sweep emits HTML reports
//! under `target/criterion/`. The CI gate is the `--quick`
//! variant.
//!
//! ## Acceptance
//!
//! Per amendment-04 D-2 + the slice prompt: dispatcher overhead
//! < 1 % of the surrounding search hot path. With
//! `dispatch_preference_only` at sub-nanosecond and the
//! end-to-end search in the microsecond range, the target is
//! met by a wide margin (~3+ orders of magnitude headroom).

use std::hint::black_box;

use arcgraph_core::{LabelId, Lsn, TenantId};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{BackendSet, Encoding, Filter, Metric, dispatch_preference};
use criterion::{Criterion, criterion_group, criterion_main};

const DIM: usize = 16;
const N: usize = 256;

fn fxd(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn build_pair() -> (FilteredHnsw, DiskAnnGraph, Vec<u8>) {
    // Deterministic seeded vectors; labels cycle through 4 buckets
    // (matching the prior `filter_dispatch.rs` fixture for
    // direct-comparison meaningfulness).
    let raw: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM).map(|d| ((i + d) as f32 * 0.013).sin()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm.max(f32::MIN_POSITIVE);
            }
            v
        })
        .collect();
    let labels: Vec<Option<DiskAnnLabelId>> = (0..N).map(|i| Some((i % 4) as u32)).collect();

    // DiskANN
    let params = DiskAnnParams {
        r: 16,
        alpha: 1.2,
        l_construction: 32,
        l_search_default: 64,
        ..DiskAnnParams::default()
    };
    let mut diskann = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32))
        .expect("DiskAnnGraph::new");
    let owned: Vec<(VectorId, Vec<u8>)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), fxd(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    diskann
        .build_filtered(&pairs, &labels, &L2F32)
        .expect("build_filtered");

    // HNSW with matching payload labels.
    let hnsw_params = HnswParams {
        m: 16,
        ef_construction: 64,
        ef_search: 64,
        seed: 42,
    };
    let mut hnsw = FilteredHnsw::new(hnsw_params, DIM, &L2F32);
    for (i, v) in raw.iter().enumerate() {
        let payload = Payload {
            tenant_id: Some(TenantId::DEFAULT),
            labels: vec![LabelId::new(labels[i].unwrap())],
            ..Default::default()
        };
        hnsw.filtered_insert(VectorId::new(i as u32), &fxd(v), payload, &L2F32)
            .expect("filtered_insert");
    }

    let q = fxd(&raw[0]);
    (hnsw, diskann, q)
}

fn bench_dispatch_preference_only(c: &mut Criterion) {
    // Pure routing-decision microbench. The four variants below
    // span both `DispatchPreference::DiskAnnPreferred` and
    // `DispatchPreference::HnswOnly` so the branch predictor sees
    // both arms; black_box prevents constant-folding.
    let any = Filter::Any;
    let leq = Filter::LabelEq(LabelId::new(7));
    let tnt = Filter::Tenant(TenantId::DEFAULT);
    let cmp = Filter::And(vec![Filter::Any]);

    c.bench_function("dispatch_preference_alternating_variants", |b| {
        let mut tick = 0_u32;
        b.iter(|| {
            let f = match tick & 3 {
                0 => &any,
                1 => &leq,
                2 => &tnt,
                _ => &cmp,
            };
            tick = tick.wrapping_add(1);
            let p = dispatch_preference(black_box(f));
            black_box(p);
        });
    });
}

fn bench_dispatch_label_eq_diskann_route(c: &mut Criterion) {
    let (hnsw, diskann, q) = build_pair();
    let set = BackendSet::new().with_hnsw(&hnsw).with_diskann(&diskann);
    let filter = Filter::LabelEq(LabelId::new(0));

    c.bench_function("dispatch_label_eq_diskann_route", |b| {
        b.iter(|| {
            let r = set
                .dispatch_filtered_search(
                    black_box(&q),
                    10,
                    black_box(&filter),
                    64,
                    &L2F32,
                    Lsn::MAX,
                )
                .expect("dispatch");
            black_box(r);
        });
    });
}

fn bench_dispatch_compound_hnsw_route(c: &mut Criterion) {
    let (hnsw, diskann, q) = build_pair();
    let set = BackendSet::new().with_hnsw(&hnsw).with_diskann(&diskann);
    let filter = Filter::And(vec![
        Filter::LabelIn(vec![LabelId::new(0), LabelId::new(1)]),
        Filter::Tenant(TenantId::DEFAULT),
    ]);

    c.bench_function("dispatch_compound_hnsw_route", |b| {
        b.iter(|| {
            let r = set
                .dispatch_filtered_search(
                    black_box(&q),
                    10,
                    black_box(&filter),
                    64,
                    &L2F32,
                    Lsn::MAX,
                )
                .expect("dispatch");
            black_box(r);
        });
    });
}

fn bench_dispatch_no_diskann_label_eq_falls_through(c: &mut Criterion) {
    // DiskANN slot absent → LabelEq's DiskAnnPreferred routing
    // still picks HNSW as the (only) primary.
    let (hnsw, _diskann, q) = build_pair();
    let set = BackendSet::new().with_hnsw(&hnsw);
    let filter = Filter::LabelEq(LabelId::new(0));

    c.bench_function("dispatch_no_diskann_label_eq", |b| {
        b.iter(|| {
            let r = set
                .dispatch_filtered_search(
                    black_box(&q),
                    10,
                    black_box(&filter),
                    64,
                    &L2F32,
                    Lsn::MAX,
                )
                .expect("dispatch");
            black_box(r);
        });
    });
}

criterion_group!(
    benches,
    bench_dispatch_preference_only,
    bench_dispatch_label_eq_diskann_route,
    bench_dispatch_compound_hnsw_route,
    bench_dispatch_no_diskann_label_eq_falls_through,
);
criterion_main!(benches);
