//! RaBitQ asymmetric estimator kernel bench.
//!
//! The ANN in-RAM beam traversal hot path repeatedly evaluates
//! `estimate_ip_unit(query, payload)`. At `dim=768`, the legacy scalar loop
//! performed 768 bit-unpack + branch + f64-widen iterations per payload. The
//! production estimator now prepares ADR-223 FastScan B=4 query planes for the
//! SSD nav path, uses runtime AVX2+FMA dispatch when FastScan metadata is not
//! present, and falls back to the scalar path on older CPUs.
//!
//! Run:
//!
//! ```bash
//! cargo bench -p arcgraph-vector --bench rabitq_kernel -- --quick
//! ```

use std::hint::black_box;

use arcgraph_vector::{
    Encoding,
    quantizer::{RABITQ_FASTSCAN_QUERY_BITS, RaBitQQuery, estimate_ip_unit},
};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

const DIM: usize = 768;

fn scalar_estimate_ip_unit(query: &RaBitQQuery, payload: &[u8]) -> f32 {
    let dim = query.y_q.len();
    let code_bytes = dim.div_ceil(8);
    let codes = &payload[..code_bytes];
    let f_o = f32::from_le_bytes(
        payload[code_bytes..code_bytes + 4]
            .try_into()
            .expect("payload has f_o"),
    );
    if f_o == 0.0 {
        return 0.0;
    }

    let mut s_dot = 0.0_f64;
    for d in 0..dim {
        let bit = (codes[d / 8] >> (d % 8)) & 1;
        let y = f64::from(query.y_q[d]);
        s_dot += if bit == 1 { y } else { -y };
    }
    ((s_dot / (dim as f64).sqrt()) / f64::from(f_o)) as f32
}

fn input() -> (RaBitQQuery, Vec<u8>) {
    let mut rng = StdRng::seed_from_u64(0x7580_2090_3000_0000);
    let mut payload = vec![0u8; Encoding::RaBitQ.bytes_per_vector_unaligned(DIM)];
    let code_bytes = DIM.div_ceil(8);
    rng.fill_bytes(&mut payload[..code_bytes]);
    payload[code_bytes..code_bytes + 4].copy_from_slice(&0.73_f32.to_le_bytes());
    payload[code_bytes + 4..code_bytes + 8].copy_from_slice(&1.91_f32.to_le_bytes());

    let y_q = (0..DIM)
        .map(|i| ((i as f32 * 0.013).sin() * 0.75) + rng.random_range(-0.25_f32..0.25_f32))
        .collect();
    (RaBitQQuery::new(y_q, 1.37), payload)
}

fn bench_estimate_ip_unit_768(c: &mut Criterion) {
    let (query, payload) = input();
    let fastscan_query = query.clone().with_fastscan(RABITQ_FASTSCAN_QUERY_BITS);
    let scalar = scalar_estimate_ip_unit(&query, &payload);
    let runtime = estimate_ip_unit(&query, &payload);
    let fastscan = estimate_ip_unit(&fastscan_query, &payload);
    assert!(
        (runtime - scalar).abs() <= 2.0e-7,
        "runtime={runtime} scalar={scalar}"
    );
    assert!(
        (fastscan - scalar).abs() / scalar.abs().max(1.0e-6) <= 0.20,
        "fastscan={fastscan} scalar={scalar}"
    );

    let mut group = c.benchmark_group("rabitq_estimate_ip_unit_dim768");
    group.bench_function("scalar", |bch| {
        bch.iter(|| scalar_estimate_ip_unit(black_box(&query), black_box(&payload)))
    });
    group.bench_function("runtime", |bch| {
        bch.iter(|| estimate_ip_unit(black_box(&query), black_box(&payload)))
    });
    group.bench_function("fastscan_b4", |bch| {
        bch.iter(|| estimate_ip_unit(black_box(&fastscan_query), black_box(&payload)))
    });
    group.finish();
}

criterion_group!(benches, bench_estimate_ip_unit_768);
criterion_main!(benches);
