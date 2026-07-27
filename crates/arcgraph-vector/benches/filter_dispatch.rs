//! Filter dispatch microbench — Phase 6 prerequisite.
//!
//! Per ADR-035 amendment-03 (issue #127). The canonical
//! [`Filter`] enum unifies the F.2 (HNSW) and F.3 (DiskANN)
//! filter types into a 7-variant enum. F.3's
//! [`DiskAnnGraph::filtered_search`] dispatches on the variant
//! at the public boundary (`Any` / `LabelEq` → fast path,
//! everything else → `UnsupportedFilter`). This bench validates
//! that the dispatch overhead is < 1 % of the surrounding
//! filtered-search cost — the sanity-check the amendment-03
//! decision rests on.
//!
//! ## What is measured
//!
//! - `filtered_search_label_eq`: end-to-end filtered search
//!   under [`Filter::LabelEq`] — the F.3 hot path that
//!   dispatchers will hit most often. The dispatch (canonical
//!   Filter → `Option<DiskAnnLabelId>`) is one of many hops in
//!   this measurement; if the variant-match cost regressed
//!   meaningfully we'd see total throughput shift.
//! - `filtered_search_any`: same path under [`Filter::Any`] —
//!   exercises the alternate fast-path branch in
//!   [`diskann_required_label`].
//! - `filter_dispatch_only`: the dispatch helper alone, called
//!   in a tight loop with both supported variants. Isolates the
//!   variant-match cost to confirm it's near-zero (single
//!   branch on the discriminant).
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p arcgraph-vector --bench filter_dispatch -- --quick
//! ```
//!
//! `--quick` runs in ~5 s; the full sweep (~60 s) emits HTML
//! reports under `target/criterion/`. The CI gate is the
//! `--quick` variant.

use std::hint::black_box;

use arcgraph_core::{LabelId, Lsn};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{Encoding, Filter, Metric};
use criterion::{Criterion, criterion_group, criterion_main};

const DIM: usize = 16;
const N: usize = 256;

fn fxd(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn build_graph() -> (DiskAnnGraph, Vec<u8>) {
    // Deterministic seeded vectors; labels cycle through 4
    // buckets so each carries `N/4 = 64` members — well above
    // the per-label entry-point cache's smoke-test threshold.
    let raw: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM).map(|d| ((i + d) as f32 * 0.013).sin()).collect();
            // L2 normalize.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm.max(f32::MIN_POSITIVE);
            }
            v
        })
        .collect();
    let labels: Vec<Option<DiskAnnLabelId>> = (0..N).map(|i| Some((i % 4) as u32)).collect();

    let params = DiskAnnParams {
        r: 16,
        alpha: 1.2,
        l_construction: 32,
        l_search_default: 64,
        ..DiskAnnParams::default()
    };
    let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32))
        .expect("DiskAnnGraph::new");
    let owned: Vec<(VectorId, Vec<u8>)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), fxd(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    g.build_filtered(&pairs, &labels, &L2F32)
        .expect("build_filtered");

    let q = fxd(&raw[0]);
    (g, q)
}

fn bench_filtered_search_label_eq(c: &mut Criterion) {
    let (g, q) = build_graph();
    let filter = Filter::label_eq(0_u32);

    c.bench_function("filtered_search_label_eq", |b| {
        b.iter(|| {
            let r = g
                .filtered_search(black_box(&q), 10, black_box(&filter), 64, &L2F32, Lsn::MAX)
                .expect("search");
            black_box(r);
        });
    });
}

fn bench_filtered_search_any(c: &mut Criterion) {
    let (g, q) = build_graph();
    let filter = Filter::any();

    c.bench_function("filtered_search_any", |b| {
        b.iter(|| {
            let r = g
                .filtered_search(black_box(&q), 10, black_box(&filter), 64, &L2F32, Lsn::MAX)
                .expect("search");
            black_box(r);
        });
    });
}

fn bench_filter_dispatch_only(c: &mut Criterion) {
    // Isolated micro-bench: just the canonical Filter → DiskANN
    // capability gate. The supported variants alternate so the
    // branch predictor can't lock onto one arm.
    let any = Filter::any();
    let leq = Filter::LabelEq(LabelId::new(7));

    c.bench_function("filter_dispatch_supported_variants_alternating", |b| {
        let mut tick = 0_u32;
        b.iter(|| {
            // Pick variant based on a tick so the branch
            // predictor sees both arms; black_box prevents the
            // optimizer from constant-folding away the work.
            let f = if tick & 1 == 0 { &any } else { &leq };
            tick = tick.wrapping_add(1);
            // The required_label() helper is what F.3 calls at
            // the public boundary post-amendment-03.
            let r = black_box(f).required_label();
            black_box(r);
        });
    });
}

criterion_group!(
    benches,
    bench_filtered_search_label_eq,
    bench_filtered_search_any,
    bench_filter_dispatch_only,
);
criterion_main!(benches);
