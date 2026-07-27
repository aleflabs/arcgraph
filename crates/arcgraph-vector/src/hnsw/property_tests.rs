//! HNSW structural property tests (W28-S2).
//!
//! These tests assert **structural invariants of the built graph**
//! — out-degree caps and build determinism — that require access to
//! the `pub(crate)` adjacency / vector maps on [`HnswGraph`], so
//! they live in-crate (`#[cfg(test)]`) rather than in
//! `tests/hnsw.rs` (which is restricted to the public
//! insert/search/delete surface).
//!
//! ## Why these exist (gap analysis PR #510 §3 + testing strategy)
//!
//! testing strategy mandates `proptest` coverage for "every data
//! structure with invariants … HNSW". Before W28-S2 the only
//! degree-related check was [`super::insert::tests`]'s
//! `select_neighbors_heuristic_caps_at_m`, which feeds the pruning
//! heuristic a **hand-built candidate list** — it never asserts the
//! cap holds after a sequence of *real* inserts (where backward
//! edges accumulate and re-prune). And HNSW had **no**
//! build-determinism test at all (only DiskANN pinned byte-identity,
//! `tests/dispatcher.rs`). This module closes both gaps:
//!
//! - [`prop_hnsw_degree_cap`] — after randomized real inserts,
//!   every node's per-layer out-degree respects `m_max0` (layer 0)
//!   / `m_max` (upper layers) per Malkov & Yashunin TPAMI 2018 §4
//!   Algorithm 1 line 17.
//! - [`hnsw_build_determinism`] — building twice from the same
//!   `(seed, insert-sequence)` yields a **binary-equal** node/edge
//!   set (the strongest available oracle per the
//!   determinism-oracle discipline: a structural snapshot equality,
//!   not a weaker dedupe/consistency check).

use std::collections::BTreeMap;

use proptest::prelude::*;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;

use super::graph::{HnswGraph, HnswParams};
use crate::distance::L2F32;
use crate::ids::VectorId;

/// L2-normalize in place (unit-sphere data, as the recall tests use).
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Build an HNSW graph deterministically from `(seed, n, dim)`.
///
/// Both the vector data (RNG seeded from `seed ^ DATA_SALT`) and the
/// graph topology (level-assignment RNG seeded from `params.seed =
/// seed`) are functions of `seed` alone, so two calls with identical
/// arguments build identical graphs — the property
/// [`hnsw_build_determinism`] relies on. Ids are `0..n` (unique, so
/// the `insert`-as-replace / `detach_node` path — the only place the
/// build consults `HashMap` iteration order — is never taken).
fn build_graph(seed: u64, n: usize, dim: usize) -> HnswGraph {
    const DATA_SALT: u64 = 0x5EED_0DA7;
    let params = HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        seed,
    };
    let kernel = L2F32;
    let mut g = HnswGraph::new(params, dim, &kernel);
    let mut rng = StdRng::seed_from_u64(seed ^ DATA_SALT);
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                u * 2.0 - 1.0
            })
            .collect();
        let bytes = bytemuck::cast_slice(&l2_normalize(v)).to_vec();
        g.insert(VectorId::new(i as u32), &bytes, &kernel)
            .expect("insert");
    }
    g
}

/// Canonical, order-independent snapshot of the entire graph state:
/// entry point, max level, vector bytes, and per-node per-layer
/// adjacency. `BTreeMap` canonicalizes the `HashMap` key order; the
/// inner adjacency `Vec<Vec<VectorId>>` is compared **in order**
/// (the build is deterministic, so neighbor ordering is itself part
/// of the invariant — sorting it would weaken the oracle).
type GraphSnapshot = (
    Option<VectorId>,
    usize,
    BTreeMap<VectorId, Vec<u8>>,
    BTreeMap<VectorId, Vec<Vec<VectorId>>>,
);

fn snapshot(g: &HnswGraph) -> GraphSnapshot {
    let vectors: BTreeMap<VectorId, Vec<u8>> =
        g.vectors.iter().map(|(k, v)| (*k, v.clone())).collect();
    let nodes: BTreeMap<VectorId, Vec<Vec<VectorId>>> = g
        .nodes
        .iter()
        .map(|(k, node)| (*k, node.neighbors.clone()))
        .collect();
    (g.entry_point, g.max_level, vectors, nodes)
}

proptest! {
    // W28-S573 exceed-spec: 48 → 128 cases. The degree-cap invariant is
    // a HARD structural bound (Malkov & Yashunin §4 Alg 1 line 17) that
    // must hold for EVERY node at EVERY layer after any real insert
    // sequence; over a continuous vector space exhaustive enumeration is
    // infeasible (ADR-165 M1), so the case count is the statistical lever
    // — 128 cases × randomized (seed, n, dim) is a ~2.7× deepening that
    // keeps the single-crate run bounded (each builds ≤ 200 inserts).
    // `PROPTEST_CASES` still overrides for a heavier sweep.
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// **Test 3 — degree cap after real inserts.**
    ///
    /// Per Malkov & Yashunin §4 Algorithm 1 the bidirectional-edge
    /// insert prunes every adjacency list back to `m_max` (upper
    /// layers) / `m_max0 = 2·M` (layer 0). This proptest builds a
    /// graph from a *random real insert sequence* and asserts the
    /// cap holds for **every node at every layer** — the
    /// post-real-insert check that `select_neighbors_heuristic_caps_at_m`
    /// (hand-built candidate list) does not provide.
    #[test]
    fn prop_hnsw_degree_cap(
        seed in any::<u64>(),
        n in 32usize..=200,
        dim in 4usize..=24,
    ) {
        let g = build_graph(seed, n, dim);
        let m_max = g.params.m_max();
        let m_max0 = g.params.m_max0();

        for (id, node) in g.nodes.iter() {
            for (layer, adj) in node.neighbors.iter().enumerate() {
                let cap = if layer == 0 { m_max0 } else { m_max };
                prop_assert!(
                    adj.len() <= cap,
                    "node {id:?} layer {layer} out-degree {} exceeds cap {cap} \
                     (M={m_max}, M0={m_max0})",
                    adj.len()
                );
            }
        }

        // Sanity: a unique-id insert sequence keeps exactly `n`
        // vectors and (for n ≥ 1) a live entry point.
        prop_assert_eq!(g.vectors.len(), n);
        prop_assert!(g.entry_point.is_some());
    }
}

/// **Test 4 — build determinism (binary-equal reference oracle).**
///
/// Building the HNSW twice from the same `(seed, insert-sequence)`
/// must yield structurally identical graphs: same entry point, same
/// max level, same vector bytes, and a **byte-for-byte identical
/// node/edge set**. This is the binary-equal reference oracle the
/// determinism-oracle discipline prefers — strictly stronger than a
/// "the two runs agree on recall / dedupe consistency" check.
///
/// HNSW had no build-determinism pin before W28-S2 (only DiskANN
/// pinned byte-identity at the dispatcher layer). The property is
/// load-bearing: persistence / snapshot replay (Slice G) and any
/// future reproducibility claim depend on `(seed, sequence)` ⇒
/// graph being a function, not a relation.
#[test]
fn hnsw_build_determinism() {
    // A spread of (seed, n, dim) configs — small/large dim, small/
    // large n — so the determinism claim is not pinned to one shape.
    for &(seed, n, dim) in &[
        (1u64, 120usize, 16usize),
        (42, 200, 8),
        (7, 64, 24),
        (99, 150, 12),
    ] {
        let g1 = build_graph(seed, n, dim);
        let g2 = build_graph(seed, n, dim);
        let s1 = snapshot(&g1);
        let s2 = snapshot(&g2);
        assert_eq!(
            s1.0, s2.0,
            "entry_point diverges across identical builds (seed={seed}, n={n}, dim={dim})"
        );
        assert_eq!(
            s1.1, s2.1,
            "max_level diverges across identical builds (seed={seed}, n={n}, dim={dim})"
        );
        assert_eq!(
            s1.2, s2.2,
            "vector bytes diverge across identical builds (seed={seed}, n={n}, dim={dim})"
        );
        assert_eq!(
            s1.3, s2.3,
            "node/edge set diverges across identical builds (seed={seed}, n={n}, dim={dim}) \
             — HNSW build is non-deterministic"
        );
    }
}
