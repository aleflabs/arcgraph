//! Slice F.1 — `VectorArena` + `VectorArenaRegistry` integration tests.
//!
//! Per the M3.a Slice F.1 spec, these six tests exercise the
//! per-tenant arena's contract from end to end.
//!
//! `arena_per_tenant_isolation` — two arenas keyed by different
//! `TenantId`s never see each other's vector ids. Tenant A's
//! `get_primary` for tenant B's ids returns `None` and vice
//! versa after both arenas insert disjoint ranges.
//!
//! `arena_round_trip_sq8` — F32 input → arena → primary SQ8 (i8
//! byte transport, kernel-native per #116) + F32 rescore.
//! Recall@10 ≥ 0.95 on a 5 K corpus via
//! `HnswGraph::search_with_rescore` driven by the arena's
//! primary + rescore views.
//!
//! `arena_round_trip_binary` — F32 input → arena → 128-byte
//! cache-line-aligned packed binary (per ADR-035 §S-1 fold-in) +
//! F32 rescore. Length sanity checks plus Hamming distance
//! kernel correctness on the stored bytes.
//!
//! `arena_round_trip_f32_no_quantizer` — `Encoding::F32` +
//! `QuantizerState::None`: rescore arena is `None`;
//! `get_primary` returns the verbatim F32 byte slice; no
//! rescore allocated.
//!
//! `arena_label_index_per_vector` — labels passed at insert time
//! round-trip through `labels_for`. Pre-Filtered-DiskANN
//! groundwork (Slice F.3 plugs in the pruning semantics).
//!
//! `arena_partition_id_always_zero_at_v1` — every handle
//! constructed via the registry carries
//! `partition_id == PartitionId::ZERO`, mirroring Slice A's
//! `vector_index_partition_id_always_zero_at_v1` regression
//! guard at the arena layer (per ADR-035 D-7).

use std::collections::HashSet;

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_vector::distance::{HammingBinary, L2F32, L2Sq8};
use arcgraph_vector::hnsw::{HnswGraph, HnswParams};
use arcgraph_vector::quantizer::{Sq8Trainer, binary_encode_aligned};
use arcgraph_vector::{
    Encoding, IndexId, IndexType, QuantizerState, VectorArena, VectorArenaRegistry, VectorId,
    VectorIndexHandle, distance::DistanceKernel,
};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;

// ─── helpers ─────────────────────────────────────────────────────

fn handle(tenant: u64, idx: u64) -> VectorIndexHandle {
    VectorIndexHandle::for_tenant(TenantId::new(tenant), IndexId::new(idx))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
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

/// 50-cluster, noise_radius=0.2 unit-normalized embeddings —
/// mirrors the AC-1a corpus shape from `tests/hnsw_rescore.rs`
/// so the recall target translates directly.
fn clustered_unit_vectors(
    seed: u64,
    n: usize,
    dim: usize,
    n_clusters: usize,
    noise_radius: f32,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    u * 2.0 - 1.0
                })
                .collect();
            l2_normalize(v)
        })
        .collect();
    (0..n)
        .map(|i| {
            let center = &centers[i % n_clusters];
            let noisy: Vec<f32> = center
                .iter()
                .map(|c| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    c + (u * 2.0 - 1.0) * noise_radius
                })
                .collect();
            l2_normalize(noisy)
        })
        .collect()
}

fn brute_force_top_k_f32(
    corpus: &[(VectorId, Vec<f32>)],
    query: &[f32],
    k: usize,
) -> Vec<VectorId> {
    let mut scored: Vec<(f32, VectorId)> = corpus
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

fn recall_at_k(actual: &[(VectorId, f32)], expected: &[VectorId], k: usize) -> f64 {
    let a: HashSet<VectorId> = actual.iter().map(|(id, _)| *id).take(k).collect();
    let e: HashSet<VectorId> = expected.iter().copied().take(k).collect();
    a.intersection(&e).count() as f64 / k as f64
}

// ─── Test 1 — per-tenant isolation ───────────────────────────────

#[test]
fn arena_per_tenant_isolation() {
    const N: usize = 100;
    let dim = 4;

    let registry = VectorArenaRegistry::new();
    let h_a = handle(/* tenant */ 1, /* idx */ 7);
    let h_b = handle(/* tenant */ 2, /* idx */ 7);

    let arena_a = registry.create_arena(
        h_a,
        Encoding::F32,
        IndexType::Hnsw,
        QuantizerState::None,
        dim,
    );
    let arena_b = registry.create_arena(
        h_b,
        Encoding::F32,
        IndexType::Hnsw,
        QuantizerState::None,
        dim,
    );

    // Insert disjoint ID ranges into each arena. Tenant A: ids 0..N.
    // Tenant B: ids 1000..1000+N. (Disjoint ranges make the
    // assertion crisp — no shared id can mask a missed isolation
    // bug.)
    for i in 0..N {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        arena_a
            .insert(VectorId::new(i as u32), &f32_bytes(&v), None)
            .unwrap();
        let v_b = vec![(i + 1000) as f32, 0.0, 0.0, 0.0];
        arena_b
            .insert(VectorId::new((i + 1000) as u32), &f32_bytes(&v_b), None)
            .unwrap();
    }

    assert_eq!(arena_a.vectors_count(), N);
    assert_eq!(arena_b.vectors_count(), N);

    // Tenant A must not see tenant B's ids, and vice versa.
    for i in 0..N {
        let id_b = VectorId::new((i + 1000) as u32);
        assert!(
            arena_a.get_primary(id_b).is_none(),
            "tenant A's arena should not hold tenant B's id {}",
            id_b.raw()
        );
        let id_a = VectorId::new(i as u32);
        assert!(
            arena_b.get_primary(id_a).is_none(),
            "tenant B's arena should not hold tenant A's id {}",
            id_a.raw()
        );
    }

    // Sanity: each arena holds its own ids.
    for i in 0..N {
        let id_a = VectorId::new(i as u32);
        assert!(arena_a.get_primary(id_a).is_some());
        let id_b = VectorId::new((i + 1000) as u32);
        assert!(arena_b.get_primary(id_b).is_some());
    }

    // Registry routes each handle to the right arena.
    let looked_a = registry.for_tenant_index(h_a).unwrap();
    let looked_b = registry.for_tenant_index(h_b).unwrap();
    assert_eq!(looked_a.vectors_count(), N);
    assert_eq!(looked_b.vectors_count(), N);
    // Cross-handle lookup with a non-existent tenant is None.
    assert!(registry.for_tenant_index(handle(999, 7)).is_none());
}

// ─── Test 2 — SQ8 round-trip via arena + HNSW + rescore ──────────

/// Insert 5K F32 vectors into an arena configured for `Sq8`,
/// then drive a `HnswGraph::search_with_rescore` using the
/// arena's primary i8 bytes + rescore F32 bytes. Assert
/// recall@10 ≥ 0.95.
///
/// This is the production-shaped end-to-end test for Slice F.1's
/// principal claim: the arena owns the encoding step, and the
/// rescore-side full-precision view is reachable through the
/// arena's `get_rescore` callback. The HNSW graph remains
/// entirely byte-oriented; all SQ8 ↔ F32 plumbing lives in the
/// arena, not at the call site.
#[test]
fn arena_round_trip_sq8() {
    const DIM: usize = 768;
    const N: usize = 5_000;
    const K: usize = 10;
    const RESCORE_FACTOR: usize = 5;
    const N_QUERIES: usize = 20;

    let f32_corpus = clustered_unit_vectors(42, N, DIM, 50, 0.2);

    // Train SQ8 codebook (5K samples — full corpus is the
    // training set here, well under ADR-035 §3.3's 1M cap).
    let samples: Vec<&[f32]> = f32_corpus.iter().map(Vec::as_slice).collect();
    let codebook = Sq8Trainer.train(&samples).expect("SQ8 train succeeds");
    let params = codebook.into_params();

    // Construct a SQ8 arena.
    let arena = VectorArena::new(
        handle(1, 1),
        Encoding::Sq8,
        IndexType::Hnsw,
        QuantizerState::Sq8 {
            params: params.clone(),
        },
        DIM,
    );
    assert!(arena.has_rescore());

    // Insert via the arena. The arena owns the f32 → i8 step.
    for (i, v) in f32_corpus.iter().enumerate() {
        arena
            .insert(VectorId::new(i as u32), &f32_bytes(v), None)
            .unwrap();
    }
    assert_eq!(arena.vectors_count(), N);

    // Build the HNSW graph against the arena's primary bytes.
    let hnsw_params = HnswParams {
        m: 24,
        ef_construction: 250,
        ef_search: 400,
        seed: 42,
    };
    let primary = L2Sq8;
    let full_precision = L2F32;
    let mut graph = HnswGraph::new(hnsw_params, DIM, &primary);
    for i in 0..N {
        let id = VectorId::new(i as u32);
        let bytes = arena.get_primary(id).expect("arena holds id");
        // Sanity: primary holds 1 byte per dim (i8 encoded as u8
        // for byte transport), not 4 bytes per dim (which would
        // be F32).
        assert_eq!(bytes.len(), DIM, "SQ8 primary byte length should equal DIM");
        graph.insert(id, &bytes, &primary).unwrap();
    }
    assert_eq!(graph.len(), N);

    // Rescore arena round-trip: every primary id has a rescore
    // entry with F32 bytes.
    for i in 0..N {
        let id = VectorId::new(i as u32);
        let rescore = arena.get_rescore(id).expect("rescore holds id");
        assert_eq!(
            rescore.len(),
            DIM * std::mem::size_of::<f32>(),
            "rescore byte length should equal DIM * 4 (F32)"
        );
    }

    // Drive recall via search_with_rescore using the arena as
    // both the primary store and the rescore source. The lookup
    // closure that `search_with_rescore` consumes must return
    // `&[u8]` references that outlive each callback invocation;
    // DashMap guards have a per-call lifetime, so we materialize
    // a stable owned-bytes shadow of the rescore arena once and
    // index it by `VectorId.raw()`. This is a test-side
    // convenience: production rescore wiring (Slice E.2 / E.3
    // already lands in the search path) holds the arena's
    // `Arc<VectorArena>` and threads guards through the search
    // pipeline; the F.1 test exercises the storage shape, not
    // the production lookup pipeline.
    let rescore_shadow: Vec<Vec<u8>> = (0..N)
        .map(|i| {
            arena
                .get_rescore(VectorId::new(i as u32))
                .map(|g| g.to_vec())
                .expect("SQ8 arena holds rescore for every inserted id")
        })
        .collect();
    let rescore_lookup = |id: VectorId| -> Option<&[u8]> {
        rescore_shadow.get(id.raw() as usize).map(Vec::as_slice)
    };

    let queries = clustered_unit_vectors(99, N_QUERIES, DIM, 50, 0.2);
    let baseline: Vec<(VectorId, Vec<f32>)> = f32_corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), v.clone()))
        .collect();

    let mut total_recall = 0.0_f64;
    for query_f32 in &queries {
        // Encode the query through the same codebook the arena
        // used at insert time. The arena does not yet expose a
        // "encode this query through my installed quantizer"
        // helper — that's a Slice F.5 surface; F.1 keeps the
        // codec at the caller and uses the arena solely for
        // storage + rescore lookup.
        let q_i8 = arcgraph_vector::quantizer::Sq8Codebook::from_params(params.clone())
            .encode(query_f32)
            .unwrap();
        let q_sq8: Vec<u8> = bytemuck::cast_slice::<i8, u8>(&q_i8).to_vec();
        let q_f32_bytes = f32_bytes(query_f32);

        let results = graph
            .search_with_rescore(
                &q_sq8,
                &q_f32_bytes,
                K,
                hnsw_params.ef_search,
                RESCORE_FACTOR,
                &primary,
                &full_precision,
                &rescore_lookup,
            )
            .expect("search_with_rescore succeeds");
        assert!(results.len() <= K);
        let bf = brute_force_top_k_f32(&baseline, query_f32, K);
        total_recall += recall_at_k(&results, &bf, K);
    }
    let mean_recall = total_recall / N_QUERIES as f64;
    println!("arena_round_trip_sq8: mean recall@{K} over {N_QUERIES} queries = {mean_recall:.4}");
    assert!(
        mean_recall >= 0.95,
        "arena round-trip recall {mean_recall} < 0.95 (AC-1a violated)"
    );
}

// ─── Test 3 — Binary round-trip via arena ─────────────────────────

/// Binary encoding: arena stores 128-byte cache-line-aligned
/// packed bytes per ADR-035 §S-1. The rescore arena holds the
/// raw F32 bytes (used by Slice E.3 binary rescore path —
/// captured here as a sanity check that the arena allocates
/// the rescore mirror).
///
/// Hamming kernel correctness: pick two vectors, hand-compute
/// their Hamming distance, compare to `HammingBinary::distance`
/// on the arena's stored bytes.
#[test]
fn arena_round_trip_binary() {
    const DIM: usize = 768;
    let aligned_len = Encoding::Binary.bytes_per_vector_aligned(DIM);
    assert_eq!(aligned_len, 128, "DIM=768 binary aligns to 128 bytes");

    let arena = VectorArena::new(
        handle(1, 1),
        Encoding::Binary,
        IndexType::Hnsw,
        QuantizerState::Binary,
        DIM,
    );
    assert!(arena.has_rescore());

    // Three planted vectors with known sign patterns so we can
    // hand-compute the expected Hamming distances:
    //   v0: alternating +1/-1 each dim
    //   v1: all +1 every dim
    //   v2: same as v0 except dim 0 flipped
    let v0: Vec<f32> = (0..DIM)
        .map(|d| if d % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let v1: Vec<f32> = vec![1.0; DIM];
    let mut v2 = v0.clone();
    v2[0] = -1.0; // single sign flip

    arena
        .insert(VectorId::new(0), &f32_bytes(&v0), None)
        .unwrap();
    arena
        .insert(VectorId::new(1), &f32_bytes(&v1), None)
        .unwrap();
    arena
        .insert(VectorId::new(2), &f32_bytes(&v2), None)
        .unwrap();
    assert_eq!(arena.vectors_count(), 3);

    // Primary: aligned packed bytes.
    let g0 = arena.get_primary(VectorId::new(0)).unwrap();
    assert_eq!(g0.len(), aligned_len);
    drop(g0);
    let g1 = arena.get_primary(VectorId::new(1)).unwrap();
    assert_eq!(g1.len(), aligned_len);
    drop(g1);
    let g2 = arena.get_primary(VectorId::new(2)).unwrap();
    assert_eq!(g2.len(), aligned_len);
    drop(g2);

    // Compare against direct binary_encode_aligned for a sanity
    // ground truth.
    let expected_v0 = binary_encode_aligned(&v0);
    let g0 = arena.get_primary(VectorId::new(0)).unwrap();
    assert_eq!(&*g0, expected_v0.as_slice());
    drop(g0);

    // Rescore: F32 bytes round-trip exactly.
    let r0 = arena.get_rescore(VectorId::new(0)).unwrap();
    assert_eq!(&*r0, f32_bytes(&v0).as_slice());
    drop(r0);

    // Hamming distance kernel sanity. v0 vs v1: every dim where
    // v0 has -1 differs from v1's +1 → DIM/2 bit-flips for the
    // even-index sign convention. v0 vs v2: exactly one bit
    // flip (the dim 0 sign).
    let primary_v0 = arena.get_primary(VectorId::new(0)).unwrap();
    let primary_v1 = arena.get_primary(VectorId::new(1)).unwrap();
    let primary_v2 = arena.get_primary(VectorId::new(2)).unwrap();
    let kernel = HammingBinary;
    let d_v0_v1 = kernel.distance(&primary_v0, &primary_v1) as usize;
    let d_v0_v2 = kernel.distance(&primary_v0, &primary_v2) as usize;
    assert_eq!(
        d_v0_v1,
        DIM / 2,
        "v0 vs v1 Hamming should be DIM/2 = {} (got {d_v0_v1})",
        DIM / 2
    );
    assert_eq!(
        d_v0_v2, 1,
        "v0 vs v2 Hamming should be 1 (single flip; got {d_v0_v2})"
    );
}

// ─── Test 4 — F32 + no quantizer (no rescore arena) ───────────────

#[test]
fn arena_round_trip_f32_no_quantizer() {
    const DIM: usize = 16;
    let arena = VectorArena::new(
        handle(1, 1),
        Encoding::F32,
        IndexType::Hnsw,
        QuantizerState::None,
        DIM,
    );
    assert!(!arena.has_rescore(), "F32+None must not allocate rescore");

    let v: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.1).collect();
    arena
        .insert(VectorId::new(0), &f32_bytes(&v), None)
        .unwrap();
    let bytes = arena.get_primary(VectorId::new(0)).unwrap();
    assert_eq!(
        bytes.len(),
        DIM * std::mem::size_of::<f32>(),
        "F32 primary holds 4 bytes per dim"
    );
    assert_eq!(&*bytes, f32_bytes(&v).as_slice());
    drop(bytes);

    // Rescore returns None for every id — the rescore arena is
    // not allocated.
    assert!(arena.get_rescore(VectorId::new(0)).is_none());
}

// ─── Test 5 — per-vector label index (Filtered-DiskANN groundwork) ──

#[test]
fn arena_label_index_per_vector() {
    let arena = VectorArena::new(
        handle(1, 1),
        Encoding::F32,
        IndexType::DiskAnn,
        QuantizerState::None,
        4,
    );
    let v = f32_bytes(&[0.0, 1.0, 2.0, 3.0]);

    // Vector with 2 labels.
    let labels_a = [LabelId::new(7), LabelId::new(42)];
    arena.insert(VectorId::new(0), &v, Some(&labels_a)).unwrap();

    // Vector with 0 labels (explicit empty list — distinct from
    // "no labels recorded").
    arena.insert(VectorId::new(1), &v, Some(&[])).unwrap();

    // Vector with no labels recorded.
    arena.insert(VectorId::new(2), &v, None).unwrap();

    // Vector with a single label.
    let labels_d = [LabelId::new(99)];
    arena.insert(VectorId::new(3), &v, Some(&labels_d)).unwrap();

    // Round-trip checks.
    let g = arena.labels_for(VectorId::new(0)).unwrap();
    assert_eq!(&*g, labels_a.as_slice());
    drop(g);

    let g = arena.labels_for(VectorId::new(1)).unwrap();
    assert!(
        g.is_empty(),
        "explicit empty labels should round-trip empty"
    );
    drop(g);

    // The "no labels recorded" vector returns None — distinct
    // from "empty labels" via the Option layer. Filtered-DiskANN
    // (Slice F.3) treats absence as "this vector has no
    // payload to filter on".
    assert!(arena.labels_for(VectorId::new(2)).is_none());

    let g = arena.labels_for(VectorId::new(3)).unwrap();
    assert_eq!(&*g, labels_d.as_slice());
}

// ─── Test 6 — partition_id == ZERO for every registry-built handle ──

#[test]
fn arena_partition_id_always_zero_at_v1() {
    let registry = VectorArenaRegistry::new();

    // Build arenas for a representative spread of (tenant, index)
    // pairs. Every handle constructed via the registry must
    // satisfy `partition_id == PartitionId::ZERO` — the v1.0
    // local-only invariant invariant per ADR-035 D-7.
    for (tenant, idx) in [(0, 0), (1, 1), (1, 7), (42, 99), (100, 1_000_000_000)] {
        let h = handle(tenant, idx);
        let arena =
            registry.create_arena(h, Encoding::F32, IndexType::Hnsw, QuantizerState::None, 4);
        assert_eq!(
            arena.handle().partition(),
            PartitionId::ZERO,
            "arena handle for tenant={tenant} idx={idx} must have PartitionId::ZERO"
        );
        assert!(
            arena.handle().is_v1_local(),
            "arena handle for tenant={tenant} idx={idx} must satisfy is_v1_local"
        );
    }

    // Re-pull from the registry to confirm the stored handles
    // round-trip the v1.0 invariant.
    for (tenant, idx) in [(0, 0), (1, 1), (1, 7), (42, 99), (100, 1_000_000_000)] {
        let h = handle(tenant, idx);
        let arena = registry
            .for_tenant_index(h)
            .expect("registry holds the arena we just created");
        assert_eq!(arena.handle().partition(), PartitionId::ZERO);
    }
}
