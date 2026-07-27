//! ADR-209 slice 1 RaBitQ codec acceptance tests.

use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::quantizer::{
    RaBitQCodebook, RaBitQParams, RaBitQTrainer, Sq8Trainer, auto_quantizer_for_collection,
    binary_encode,
};
use arcgraph_vector::{Encoding, Metric, QuantizerState, VectorIndexError};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestRunner};

const TRAIN_SEED: u64 = 0x7580_0209;

struct Xs64(u64);

impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / ((1u32 << 24) as f32);
        lo + (hi - lo) * unit
    }

    fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform(f32::MIN_POSITIVE, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = Xs64::new(seed);
    (0..n)
        .map(|_| normalize((0..dim).map(|_| rng.gaussian()).collect()))
        .collect()
}

fn clustered_fixture(
    seed: u64,
    n: usize,
    q: usize,
    dim: usize,
    sigma: f32,
    clusters: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Xs64::new(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect())
        .collect();
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let c = &centers[i % centers.len()];
            (0..dim).map(|d| c[d] + sigma * rng.gaussian()).collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..q)
        .map(|i| {
            let c = &centers[(i * 17) % centers.len()];
            (0..dim).map(|d| c[d] + sigma * rng.gaussian()).collect()
        })
        .collect();
    (data, queries)
}

fn train_rabitq(data: &[Vec<f32>]) -> RaBitQCodebook {
    let samples: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    RaBitQTrainer.train(&samples, TRAIN_SEED).unwrap()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn recall_at(found: &[usize], truth: &[usize], k: usize) -> f32 {
    let hits = found
        .iter()
        .take(k)
        .filter(|id| truth.iter().take(k).any(|t| t == *id))
        .count();
    hits as f32 / k as f32
}

#[test]
fn v1_recall_byte_acceptance_seeded_fixture() {
    // Normal CI uses a bounded fixture; set ARCGRAPH_RABITQ_FULL_V1=1
    // for the ADR-sized 8192 x 768 report run.
    //
    // ADR-209 §2 / the RaBitQ theorem calibrate a 1-bit estimator
    // at tight Theta(1/sqrt(D)) error. A full-V1 fixture whose
    // true top-100 live inside one 128-member cluster asks phase 1
    // to resolve within-cluster unit-IP gaps roughly 30x below that
    // scale, so no 1-bit code can certify it. The full fixture below
    // instead keeps the same gate shape and tests cross-cluster
    // ordering, where 1-bit codes are information-bearing in slice 1.
    //
    // SQ8 phase 1 here is conservative decode-then-f32-L2, not the
    // i8 SIMD kernel. slice-2/3 (D-5): within-cluster recall/byte
    // certification on the 1M SQ8-parity and 10M §B read gates.
    // TODO(#758): certify within-cluster recall/byte at slice 2/3.
    let full = std::env::var("ARCGRAPH_RABITQ_FULL_V1").ok().as_deref() == Some("1");
    let (n, q, dim, sigma, clusters) = if full {
        (8192, 100, 768, 0.05, 512)
    } else {
        (512, 20, 768, 0.005, 64)
    };
    let (data, queries) = clustered_fixture(0xA209_7581, n, q, dim, sigma, clusters);
    let samples: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();

    let sq8 = Sq8Trainer.train(&samples).unwrap();
    let sq8_codes: Vec<Vec<i8>> = data.iter().map(|v| sq8.encode(v).unwrap()).collect();
    let sq8_decoded: Vec<Vec<f32>> = sq8_codes.iter().map(|v| sq8.decode(v).unwrap()).collect();

    let bin_codes: Vec<Vec<u8>> = data.iter().map(|v| binary_encode(v)).collect();

    let rabitq = train_rabitq(&data);
    let rabitq_codes: Vec<Vec<u8>> = data.iter().map(|v| rabitq.encode(v).unwrap()).collect();

    let mut sq8_p1 = 0.0;
    let mut sq8_end = 0.0;
    let mut bin_p1 = 0.0;
    let mut bin_end = 0.0;
    let mut rab_p1 = 0.0;
    let mut rab_end = 0.0;

    for query in &queries {
        let mut truth: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth_ids: Vec<usize> = truth.iter().map(|(i, _)| *i).collect();

        let mut sq8_rank: Vec<(usize, f32)> = sq8_decoded
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        sq8_rank.sort_by(|a, b| a.1.total_cmp(&b.1));
        let sq8_ids: Vec<usize> = sq8_rank.iter().map(|(i, _)| *i).collect();
        sq8_p1 += recall_at(&sq8_ids, &truth_ids, 100.min(n));
        let rerank_k = if full { 20 } else { 100 };
        sq8_end += rerank_recall(&sq8_ids, &data, query, &truth_ids, rerank_k);

        let q_bin = binary_encode(query);
        let mut bin_rank: Vec<(usize, u32)> = bin_codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let dist = code
                    .iter()
                    .zip(&q_bin)
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                (i, dist)
            })
            .collect();
        bin_rank.sort_by_key(|(_, d)| *d);
        let bin_ids: Vec<usize> = bin_rank.iter().map(|(i, _)| *i).collect();
        bin_p1 += recall_at(&bin_ids, &truth_ids, 100.min(n));
        bin_end += rerank_recall(&bin_ids, &data, query, &truth_ids, rerank_k);

        let prepared = rabitq.prepare_query(query).unwrap();
        let mut rab_rank: Vec<(usize, f32)> = rabitq_codes
            .iter()
            .enumerate()
            .map(|(i, payload)| (i, rabitq.estimate_l2_sq(&prepared, payload)))
            .collect();
        rab_rank.sort_by(|a, b| a.1.total_cmp(&b.1));
        let rab_ids: Vec<usize> = rab_rank.iter().map(|(i, _)| *i).collect();
        rab_p1 += recall_at(&rab_ids, &truth_ids, 100.min(n));
        rab_end += rerank_recall(&rab_ids, &data, query, &truth_ids, rerank_k);
    }

    let denom = q as f32;
    let sq8_p1 = sq8_p1 / denom;
    let sq8_end = sq8_end / denom;
    let bin_p1 = bin_p1 / denom;
    let bin_end = bin_end / denom;
    let rab_p1 = rab_p1 / denom;
    let rab_end = rab_end / denom;
    println!("codec,bytes_per_vec,phase1_recall,end_recall");
    println!("SQ8,{dim},{sq8_p1:.4},{sq8_end:.4}");
    println!("Binary,{},{bin_p1:.4},{bin_end:.4}", dim.div_ceil(8));
    println!(
        "RaBitQ,{},{rab_p1:.4},{rab_end:.4}",
        Encoding::RaBitQ.bytes_per_vector_unaligned(dim)
    );
    assert!(rab_p1 >= bin_p1 + 0.03);
    assert!(rab_end >= sq8_end - 0.02);
}

fn rerank_recall(
    ids: &[usize],
    data: &[Vec<f32>],
    query: &[f32],
    truth: &[usize],
    rerank_k: usize,
) -> f32 {
    let mut top: Vec<(usize, f32)> = ids
        .iter()
        .take(rerank_k.min(ids.len()))
        .map(|&i| (i, l2(query, &data[i])))
        .collect();
    top.sort_by(|a, b| a.1.total_cmp(&b.1));
    let reranked: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    recall_at(&reranked, truth, 10.min(truth.len()))
}

#[test]
fn v2_v3_unbiasedness_and_error_bound_calibration() {
    const DIM: usize = 256;
    const M: usize = 1024;
    let train = unit_vectors(11, 128, DIM);
    let cb = train_rabitq(&train);
    let objects = unit_vectors(13, M, DIM);
    let queries = unit_vectors(17, M, DIM);

    let mut errors = Vec::with_capacity(M);
    for (o, q) in objects.iter().zip(&queries) {
        let payload = cb.encode(o).unwrap();
        let prepared = cb.prepare_query(q).unwrap();
        let est = cb.estimate_ip_unit(&prepared, &payload);
        let exact = centered_unit_dot(cb.params(), o, q);
        errors.push(est - exact);
    }

    let mean = errors.iter().copied().sum::<f32>() / M as f32;
    let std = (errors
        .iter()
        .map(|e| {
            let d = e - mean;
            d * d
        })
        .sum::<f32>()
        / M as f32)
        .sqrt();
    let bound6 = 6.0 / (DIM as f32).sqrt();
    let bound12 = 12.0 / (DIM as f32).sqrt();
    let beyond6 = errors.iter().filter(|e| e.abs() > bound6).count();
    let beyond12 = errors.iter().filter(|e| e.abs() > bound12).count();
    println!(
        "DIM={DIM} M={M} mean={mean:.6} std={std:.6} beyond6={} ({:.4}) beyond12={}",
        beyond6,
        beyond6 as f32 / M as f32,
        beyond12
    );
    assert!(mean.abs() <= 5.0 * std / (M as f32).sqrt());
    assert!((beyond6 as f32 / M as f32) <= 0.01);
    assert_eq!(beyond12, 0);
}

#[test]
fn v4_exactness_identities() {
    let data = unit_vectors(101, 128, 128);
    let cb = train_rabitq(&data);
    let o = &data[0];
    let payload = cb.encode(o).unwrap();
    let prepared = cb.prepare_query(o).unwrap();
    let bound = 6.0 / (cb.dim() as f32).sqrt();
    let ip = cb.estimate_ip_unit(&prepared, &payload);
    assert!((ip - 1.0).abs() <= bound);
    assert!(cb.estimate_l2_sq(&prepared, &payload) <= 2.0 * prepared.n_q * prepared.n_q * bound);

    let c = cb.params().centroid.clone();
    let c_payload = cb.encode(&c).unwrap();
    let q = cb.prepare_query(o).unwrap();
    assert_eq!(cb.estimate_l2_sq(&q, &c_payload), q.n_q * q.n_q);

    let c_query = cb.prepare_query(&c).unwrap();
    let parsed = cb.parse_payload(&payload).unwrap();
    assert_eq!(
        cb.estimate_l2_sq(&c_query, &payload),
        parsed.n_o * parsed.n_o
    );
}

#[test]
fn v5_trainer_orthonormality_rejections_and_determinism() {
    for dim in [8, 64, 128] {
        let data = unit_vectors(dim as u64, dim + 8, dim);
        let cb = train_rabitq(&data);
        let max_err = max_orthonormal_error(cb.params());
        assert!(max_err < 1e-4, "dim={dim} max_err={max_err}");
    }

    let empty: Vec<&[f32]> = Vec::new();
    assert!(RaBitQTrainer.train(&empty, 1).is_err());
    let bad = [vec![1.0, 2.0], vec![1.0]];
    let refs: Vec<&[f32]> = bad.iter().map(Vec::as_slice).collect();
    assert!(RaBitQTrainer.train(&refs, 1).is_err());
    let bad = [vec![1.0, f32::NAN]];
    let refs: Vec<&[f32]> = bad.iter().map(Vec::as_slice).collect();
    assert!(RaBitQTrainer.train(&refs, 1).is_err());

    let data = unit_vectors(919, 32, 32);
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let a = RaBitQTrainer.train(&refs, 42).unwrap();
    let b = RaBitQTrainer.train(&refs, 42).unwrap();
    assert_eq!(a.params(), b.params());
    assert_eq!(a.encode(&data[0]).unwrap(), b.encode(&data[0]).unwrap());
}

#[test]
fn v6_codec_invariants_proptest() {
    let mut runner = TestRunner::new(ProptestConfig {
        cases: 96,
        ..ProptestConfig::default()
    });
    runner
        .run(
            &(
                1usize..=128,
                proptest::collection::vec(-10.0f32..10.0, 1..=128),
            ),
            |(dim, mut v)| {
                v.resize(dim, 0.25);
                let train = [v.clone(), vec![0.0; dim], vec![1.0; dim]];
                let refs: Vec<&[f32]> = train.iter().map(Vec::as_slice).collect();
                let cb = RaBitQTrainer.train(&refs, 3).unwrap();
                let payload = cb.encode(&v).unwrap();
                prop_assert_eq!(
                    payload.len(),
                    Encoding::RaBitQ.bytes_per_vector_unaligned(dim)
                );
                let aligned = cb.encode_aligned(&v).unwrap();
                prop_assert_eq!(
                    aligned.len(),
                    Encoding::RaBitQ.bytes_per_vector_aligned(dim)
                );
                prop_assert!(aligned[payload.len()..].iter().all(|b| *b == 0));
                let parsed = cb.parse_payload(&payload).unwrap();
                prop_assert_eq!(parsed.codes.len(), dim.div_ceil(8));
                prop_assert!(parsed.f_o.is_finite());
                prop_assert!(parsed.n_o.is_finite());
                let encode_mismatch = matches!(
                    cb.encode(&v[..dim - 1]).unwrap_err(),
                    VectorIndexError::DimensionMismatch { .. }
                );
                prop_assert!(encode_mismatch);
                let query_mismatch = matches!(
                    cb.prepare_query(&v[..dim - 1]).unwrap_err(),
                    VectorIndexError::DimensionMismatch { .. }
                );
                prop_assert!(query_mismatch);
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn v7_serde_backcompat_and_wire_name() {
    let data = unit_vectors(4, 8, 8);
    let cb = train_rabitq(&data);
    let state = QuantizerState::RaBitQ {
        params: cb.into_params(),
    };
    let json = serde_json::to_string(&state).unwrap();
    let round: QuantizerState = serde_json::from_str(&json).unwrap();
    assert_eq!(state, round);
    assert!(serde_json::from_str::<QuantizerState>(r#"{"kind":"none"}"#).is_ok());
    assert!(
        serde_json::from_str::<QuantizerState>(
            r#"{"kind":"sq8","params":{"scale":[1.0],"bias":[0.0]}}"#
        )
        .is_ok()
    );
    assert!(serde_json::from_str::<QuantizerState>(r#"{"kind":"binary"}"#).is_ok());
    assert_eq!(
        serde_json::to_string(&Encoding::RaBitQ).unwrap(),
        r#""rabitq""#
    );
    assert_eq!(
        serde_json::from_str::<Encoding>(r#""rabitq""#).unwrap(),
        Encoding::RaBitQ
    );
}

#[test]
fn v8_rejection_arms_and_dispatch_stays_sq8() {
    let err = DiskAnnGraph::new(
        DiskAnnParams::default(),
        Encoding::RaBitQ,
        Metric::L2,
        Box::new(L2F32),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        VectorIndexError::UnsupportedFlags {
            encoding: Encoding::RaBitQ,
            metric: Metric::L2
        }
    ));
    assert!(Encoding::RaBitQ.requires_training());
    for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
        assert!(metric.is_valid_for(Encoding::RaBitQ));
    }
    assert!(!Metric::Hamming.is_valid_for(Encoding::RaBitQ));
    for n in [0, 1, 9_999_999, 10_000_000, 50_000_000] {
        assert_ne!(auto_quantizer_for_collection(n), Some(Encoding::RaBitQ));
    }
}

#[test]
#[ignore = "ADR-209 V9 neuter evidence; run manually for PR evidence"]
fn v9_neuter_evidence_identity_and_rotation_load_bearing() {
    let data = unit_vectors(202, 128, 128);
    let cb = train_rabitq(&data);
    let payload = cb.encode(&data[0]).unwrap();
    let prepared = cb.prepare_query(&data[0]).unwrap();
    let parsed = cb.parse_payload(&payload).unwrap();
    let mut neutered = payload.clone();
    let code_bytes = cb.dim().div_ceil(8);
    neutered[code_bytes..code_bytes + 4].copy_from_slice(&1.0f32.to_le_bytes());
    println!(
        "V9 leg1 identity: correct_ip={:.6} f_o={} neutered_ip={:.6}",
        cb.estimate_ip_unit(&prepared, &payload),
        parsed.f_o,
        cb.estimate_ip_unit(&prepared, &neutered)
    );
    assert!(cb.estimate_ip_unit(&prepared, &payload) > 0.9);
    assert!(cb.estimate_ip_unit(&prepared, &neutered) < 0.9);

    let (mut scaled, mut queries) = clustered_fixture(0xBAD5_CAFE, 512, 40, 768, 0.02, 64);
    for v in &mut scaled {
        for x in v.iter_mut().take(16) {
            *x *= 30.0;
        }
    }
    for q in &mut queries {
        for x in q.iter_mut().take(16) {
            *x *= 30.0;
        }
    }
    let trained = train_rabitq(&scaled);
    let identity = RaBitQCodebook::from_params(
        RaBitQParams::try_new(768, trained.params().centroid.clone(), identity(768)).unwrap(),
    );
    let trained_recall = phase1_recall(&trained, &scaled, &queries);
    let identity_recall = phase1_recall(&identity, &scaled, &queries);
    let trained_f_o = average_f_o(&trained, &scaled);
    let identity_f_o = average_f_o(&identity, &scaled);
    println!(
        "V9 leg2 rotation: trained_phase1={trained_recall:.4} identity_phase1={identity_recall:.4} trained_f_o={trained_f_o:.4} identity_f_o={identity_f_o:.4}"
    );
    assert!(
        trained_recall > identity_recall + 0.03,
        "rotation must help on directional strong-skew fixture: trained={trained_recall:.4} identity={identity_recall:.4}"
    );
    assert!(
        trained_f_o > identity_f_o + 0.2,
        "rotation must isotropize axis-concentrated energy: trained_f_o={trained_f_o:.4} identity_f_o={identity_f_o:.4}"
    );
}

fn centered_unit_dot(params: &RaBitQParams, a: &[f32], b: &[f32]) -> f32 {
    let ar: Vec<f64> = a
        .iter()
        .zip(&params.centroid)
        .map(|(&x, &c)| f64::from(x) - f64::from(c))
        .collect();
    let br: Vec<f64> = b
        .iter()
        .zip(&params.centroid)
        .map(|(&x, &c)| f64::from(x) - f64::from(c))
        .collect();
    let an = ar.iter().map(|x| x * x).sum::<f64>().sqrt();
    let bn = br.iter().map(|x| x * x).sum::<f64>().sqrt();
    if an == 0.0 || bn == 0.0 {
        0.0
    } else {
        (ar.iter().zip(&br).map(|(x, y)| x * y).sum::<f64>() / (an * bn)) as f32
    }
}

fn max_orthonormal_error(params: &RaBitQParams) -> f64 {
    let dim = params.dim();
    let mut max_err = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            let mut dot = 0.0;
            for k in 0..dim {
                dot += f64::from(params.rotation[k * dim + i])
                    * f64::from(params.rotation[k * dim + j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            let err = (dot - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

fn phase1_recall(cb: &RaBitQCodebook, data: &[Vec<f32>], queries: &[Vec<f32>]) -> f32 {
    let payloads: Vec<Vec<u8>> = data.iter().map(|v| cb.encode(v).unwrap()).collect();
    let mut total = 0.0;
    for q in queries {
        let mut truth: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(q, v)))
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth_ids: Vec<usize> = truth.iter().map(|(i, _)| *i).collect();
        let prepared = cb.prepare_query(q).unwrap();
        let mut rank: Vec<(usize, f32)> = payloads
            .iter()
            .enumerate()
            .map(|(i, payload)| (i, cb.estimate_l2_sq(&prepared, payload)))
            .collect();
        rank.sort_by(|a, b| a.1.total_cmp(&b.1));
        let ids: Vec<usize> = rank.iter().map(|(i, _)| *i).collect();
        total += recall_at(&ids, &truth_ids, 100);
    }
    total / queries.len() as f32
}

fn average_f_o(cb: &RaBitQCodebook, data: &[Vec<f32>]) -> f32 {
    data.iter()
        .map(|v| {
            let payload = cb.encode(v).unwrap();
            cb.parse_payload(&payload).unwrap().f_o
        })
        .sum::<f32>()
        / data.len() as f32
}

fn identity(dim: usize) -> Vec<f32> {
    let mut p = vec![0.0; dim * dim];
    for d in 0..dim {
        p[d * dim + d] = 1.0;
    }
    p
}
