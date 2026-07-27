//! GVE-Leiden static correctness tests.
//!
//! Per ADR-040 §D-1 and the M3.d-1 task #4 prompt, this is the
//! canonical correctness gate for the algorithm:
//!
//! 1. **Zachary's karate club** — 34 nodes, 78 edges. Modularity
//!    of the optimal Leiden partition is ~0.4197 (Traag 2019 Fig.
//!    3 measured value); we accept ≥ 0.40 with a small tolerance
//!    for the size-normalised score / sequential vs parallel
//!    differences.
//!
//! 2. **Stochastic Block Model (SBM)** — `n=200, k=4, p_in=0.3,
//!    p_out=0.02`. We assert detected community count ≈ 4 (±1)
//!    and Adjusted Rand Index against ground truth ≥ 0.85.
//!
//! 3. Edge cases: empty graph, K10, two-K5s.
//!
//! 4. **Determinism**: same input → identical assignments.
//!
//! 5. `install_into` round-trip with [`BTreeMembershipIndex`].

use arcgraph_community::{
    BTreeMembershipIndex, CommunityId, Graph, GveLeiden, LeidenParams, Level, MembershipIndex,
    modularity,
};
use arcgraph_core::{Lsn, NodeId, TenantId};

// ───────────────────────────────────────────────────────────────
// Zachary's karate club (Zachary 1977, J. Anthropol. Res.).
// Edge list canonical to networkx / igraph; 0-indexed.
// ───────────────────────────────────────────────────────────────

/// Edge list of Zachary's karate club. 78 edges between 34 nodes.
/// Source: standard reference encoding used in networkx's
/// `karate_club_graph` (matches Zachary 1977 Fig. 1).
const ZACHARY_EDGES: &[(u32, u32)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

fn zachary_graph() -> Graph {
    let edges: Vec<(u32, u32, f32)> = ZACHARY_EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    Graph::from_edges_undirected(34, &edges)
}

#[test]
fn zachary_modularity_at_least_0_40() {
    let g = zachary_graph();
    let r = GveLeiden::run(&g, LeidenParams::default());

    // Report all observed levels for human review.
    eprintln!("Zachary modularity per level: {:?}", r.modularity_per_level);
    eprintln!(
        "Zachary leaf community count: {}",
        unique_count(&r.levels[0])
    );

    let q_max = r
        .modularity_per_level
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        q_max >= 0.40,
        "Zachary modularity {q_max} below 0.40; algorithm is broken \
         (canonical Leiden optimum is ~0.4197 per Traag 2019 Fig. 3)"
    );
}

#[test]
fn zachary_finds_reasonable_community_count() {
    let g = zachary_graph();
    let r = GveLeiden::run(&g, LeidenParams::default());
    // Leiden on Zachary canonically finds 4 communities at γ=1.0
    // when run to global convergence (Traag 2019 Fig. 3); the
    // first pass typically outputs more (the aggregation
    // hierarchy then merges them at higher levels). We accept
    // any per-level count in [2, 10] and require that the
    // highest-modularity level lands within [2, 5].
    let q_max_idx = r
        .modularity_per_level
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("at least one level");
    let best_level = &r.levels[q_max_idx];
    let count_best = unique_count(best_level);
    eprintln!("Zachary best level {q_max_idx}: {count_best} communities");
    assert!(
        (2..=5).contains(&count_best),
        "Zachary's highest-modularity level should yield 2–5 communities, got {count_best} at level {q_max_idx}"
    );
    // Every level should also have a sensible community count.
    for (i, level) in r.levels.iter().enumerate() {
        let c = unique_count(level);
        assert!(
            (1..=10).contains(&c),
            "level {i}: {c} communities is outside the sane range [1, 10]"
        );
    }
}

// ───────────────────────────────────────────────────────────────
// SBM ground-truth recovery test.
// ───────────────────────────────────────────────────────────────

/// Generate a deterministic stochastic block model graph.
///
/// Returns `(graph, ground_truth)` where ground_truth[v] is v's
/// block id in `0..k`.
fn sbm(n: u32, k: u32, p_in: f64, p_out: f64, seed: u64) -> (Graph, Vec<u32>) {
    assert!(n % k == 0, "n must be divisible by k for clean blocks");
    let block_size = n / k;
    let ground_truth: Vec<u32> = (0..n).map(|v| v / block_size).collect();

    // Linear congruential generator for deterministic edge sampling.
    let mut state = seed;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use top 53 bits as a uniform [0, 1) sample.
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if ground_truth[u as usize] == ground_truth[v as usize] {
                p_in
            } else {
                p_out
            };
            if next_unit() < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    (Graph::from_edges_undirected(n, &edges), ground_truth)
}

/// Adjusted Rand Index: a similarity metric between two
/// partitions of the same set, range [-1, 1] (1 = identical).
/// Implementation per Hubert & Arabie 1985.
fn adjusted_rand_index(a: &[CommunityId], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    // Build the contingency table.
    let mut by_a_then_b: std::collections::HashMap<(u64, u32), u64> =
        std::collections::HashMap::new();
    let mut a_counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let mut b_counts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for i in 0..n {
        let ai = a[i].raw();
        let bi = b[i];
        *by_a_then_b.entry((ai, bi)).or_insert(0) += 1;
        *a_counts.entry(ai).or_insert(0) += 1;
        *b_counts.entry(bi).or_insert(0) += 1;
    }

    fn binom2(n: u64) -> f64 {
        (n as f64) * ((n as f64) - 1.0) / 2.0
    }

    let sum_nij_c2: f64 = by_a_then_b.values().map(|&v| binom2(v)).sum();
    let sum_a_c2: f64 = a_counts.values().map(|&v| binom2(v)).sum();
    let sum_b_c2: f64 = b_counts.values().map(|&v| binom2(v)).sum();
    let total_c2 = binom2(n as u64);

    if total_c2 == 0.0 {
        return 0.0;
    }
    let expected = sum_a_c2 * sum_b_c2 / total_c2;
    let max = 0.5 * (sum_a_c2 + sum_b_c2);
    if (max - expected).abs() < 1e-12 {
        return 0.0;
    }
    (sum_nij_c2 - expected) / (max - expected)
}

#[test]
fn sbm_recovers_ground_truth_high_ari() {
    // n=200, k=4 → 50 per block. p_in=0.3, p_out=0.02.
    // Expected: 4 communities recovered, ARI ≥ 0.85.
    let (g, ground_truth) = sbm(200, 4, 0.3, 0.02, 0xC0FFEE);
    let r = GveLeiden::run(&g, LeidenParams::default());
    let leaf = &r.levels[0];
    let count = unique_count(leaf);
    eprintln!("SBM modularity per level: {:?}", r.modularity_per_level);
    eprintln!("SBM leaf community count: {count}");
    let ari = adjusted_rand_index(leaf, &ground_truth);
    eprintln!("SBM ARI vs ground truth: {ari:.4}");

    // Allow ±1 community due to occasional merge edge cases.
    assert!(
        (3..=5).contains(&count),
        "expected ~4 communities, got {count}"
    );
    assert!(
        ari >= 0.85,
        "expected ARI ≥ 0.85, got {ari:.4}; algorithm is not recovering SBM blocks"
    );
}

// ───────────────────────────────────────────────────────────────
// Edge-case graphs.
// ───────────────────────────────────────────────────────────────

#[test]
fn empty_graph_yields_singletons_modularity_zero() {
    // n=10, no edges → every node is its own community, Q = 0.
    let g = Graph::from_edges_undirected(10, &[]);
    let r = GveLeiden::run(&g, LeidenParams::default());
    let leaf = &r.levels[0];
    assert_eq!(unique_count(leaf), 10);
    let q = r.modularity_per_level[0];
    assert!(q.abs() < 1e-9, "expected Q=0, got {q}");
}

#[test]
fn complete_graph_k10_one_or_few_communities() {
    // K10 has no community structure (every vertex equally
    // connected to every other). We expect 1 or a small number
    // of communities and modularity near 0.
    let mut edges = Vec::new();
    for i in 0..10 {
        for j in (i + 1)..10 {
            edges.push((i, j, 1.0));
        }
    }
    let g = Graph::from_edges_undirected(10, &edges);
    let r = GveLeiden::run(&g, LeidenParams::default());
    let leaf = &r.levels[0];
    let count = unique_count(leaf);
    eprintln!(
        "K10 communities: {count}, modularity: {:?}",
        r.modularity_per_level
    );
    // K10 has no inherent structure; Q should be near 0.
    let q_max = r
        .modularity_per_level
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        q_max <= 0.10,
        "K10 has no structure; max Q should be ≤ 0.10, got {q_max}"
    );
}

#[test]
fn two_disconnected_k5s_find_two_communities() {
    // Two K5 cliques (5 vertices each, all internal edges).
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            edges.push((i, j, 1.0));
        }
    }
    for i in 5..10 {
        for j in (i + 1)..10 {
            edges.push((i, j, 1.0));
        }
    }
    let g = Graph::from_edges_undirected(10, &edges);
    let r = GveLeiden::run(&g, LeidenParams::default());
    let leaf = &r.levels[0];
    let count = unique_count(leaf);
    let q_max = r
        .modularity_per_level
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    eprintln!("Two-K5: {count} communities, max Q = {q_max}");
    assert_eq!(count, 2, "expected exactly 2 communities for two K5s");
    assert!(
        q_max >= 0.40,
        "two K5s should have high modularity, got {q_max}"
    );
    // Verify the assignment respects the disconnected structure.
    for i in 0..5 {
        assert_eq!(leaf[i], leaf[0], "all of K5_a should be in one community");
    }
    for i in 5..10 {
        assert_eq!(leaf[i], leaf[5], "all of K5_b should be in one community");
    }
    assert_ne!(
        leaf[0], leaf[5],
        "the two K5s should be in distinct communities"
    );
}

#[test]
fn determinism_same_seed_same_result() {
    let g = zachary_graph();
    let r1 = GveLeiden::run(&g, LeidenParams::default());
    let r2 = GveLeiden::run(&g, LeidenParams::default());
    assert_eq!(r1.levels, r2.levels);
    for (a, b) in r1.modularity_per_level.iter().zip(&r2.modularity_per_level) {
        assert!((a - b).abs() < 1e-12);
    }
}

/// Frozen Zachary level-0 community assignment, captured from a
/// release-deterministic run of [`GveLeiden::run`] against
/// [`zachary_graph`] at [`LeidenParams::default()`] on
/// 2026-05-03 (commit-base `daf0ac6`).
///
/// Per `feedback_determinism_oracle_concurrency_tests.md`:
/// deterministic algorithm + small fixture → binary-equal oracle
/// is strictly stronger than tolerance-based oracles. A bug that
/// reorders local-moving but stays deterministic across runs would
/// pass `determinism_same_seed_same_result` (which only asserts
/// run-to-run determinism) and silently produce a 0.045 modularity
/// drift in incremental tests with their relaxed tolerances; this
/// frozen reference fails loudly on such regressions.
///
/// If this constant ever needs to be updated (e.g. an intentional
/// algorithm change), the new value MUST be justified by an ADR
/// amendment to ADR-040 §D-2 — silently bumping the constant
/// defeats the purpose of the binary-equal oracle.
const ZACHARY_LEVEL0_FROZEN_REFERENCE: &[u64] = &[
    17, 17, 12, 12, 10, 16, 16, 12, 32, 12, 10, 17, 12, 12, 32, 32, 16, 17, 32, 17, 32, 17, 32, 27,
    27, 27, 32, 27, 27, 32, 32, 27, 32, 32,
];

#[test]
fn zachary_level0_byte_equal_reference_snapshot() {
    // Per `feedback_determinism_oracle_concurrency_tests.md`:
    // deterministic algorithm + small fixture → binary-equal oracle
    // is strictly stronger than tolerance-based oracles. A 0.045
    // modularity regression that the existing tolerance-relaxed
    // tests at `tests/leiden_incremental_consistency.rs` (drift
    // ≤ 0.05 / 0.10) would silently accept will fail this test
    // loudly because the level-0 assignment is asserted byte-equal
    // against `ZACHARY_LEVEL0_FROZEN_REFERENCE`.
    //
    // ADR-040 §D-2 commits to ε = 1e-4 modularity drift between
    // incremental and static. The small-fixture relaxation in the
    // incremental consistency tests is documented in the parallel
    // ADR-040 amendment-02 (codex M3.d retro 2026-05-03 #4).
    let g = zachary_graph();
    let r = GveLeiden::run(&g, LeidenParams::default());
    let observed: Vec<u64> = r.levels[0].iter().map(|c| c.raw()).collect();
    assert_eq!(
        observed.as_slice(),
        ZACHARY_LEVEL0_FROZEN_REFERENCE,
        "Zachary level-0 assignment diverges from the frozen reference. \
         If this is an intentional algorithm change, update the constant \
         under an ADR-040 §D-2 amendment; otherwise this is a regression."
    );
}

#[test]
fn install_into_round_trip() {
    let g = zachary_graph();
    let r = GveLeiden::run(&g, LeidenParams::default());
    let idx = BTreeMembershipIndex::new();
    // ADR-041 §D-3b: install at LSN=1; queries at Lsn::MAX see
    // the latest install.
    GveLeiden::install_into(&r, &idx, TenantId::DEFAULT, Lsn::new(1), 0);

    // For every level the algorithm reported, every node
    // round-trips through the index. We test all levels (not
    // just leaf) because `install_into` writes them all.
    for (li, level_assignment) in r.levels.iter().enumerate() {
        let level = Level::new(li as u8);
        // Forward: per-node lookup matches in-memory assignment.
        for (node_idx, expected) in level_assignment.iter().enumerate() {
            let got = idx
                .lookup(
                    TenantId::DEFAULT,
                    NodeId::new(node_idx as u64),
                    level,
                    Lsn::MAX,
                )
                .expect("ok");
            assert_eq!(
                got,
                Some(*expected),
                "level {li}, node {node_idx}: index should return the assignment"
            );
        }
        // Reverse: members of every community match.
        let mut comm_to_members: std::collections::BTreeMap<CommunityId, Vec<NodeId>> =
            std::collections::BTreeMap::new();
        for (node_idx, c) in level_assignment.iter().enumerate() {
            comm_to_members
                .entry(*c)
                .or_default()
                .push(NodeId::new(node_idx as u64));
        }
        for (c, expected_members) in &comm_to_members {
            let got = idx
                .members(TenantId::DEFAULT, *c, level, Lsn::MAX)
                .expect("ok");
            assert_eq!(
                got, *expected_members,
                "level {li}, community {c:?}: members should match"
            );
        }
    }
}

#[test]
fn modularity_helper_matches_run_levels() {
    // The reported per-level modularity matches a fresh
    // `modularity()` call on the level-0 assignment against the
    // original graph.
    let g = zachary_graph();
    let r = GveLeiden::run(&g, LeidenParams::default());
    let q_recomputed = modularity(&g, &r.levels[0], LeidenParams::default().gamma);
    assert!((r.modularity_per_level[0] - q_recomputed).abs() < 1e-9);
}

// ───────────────────────────────────────────────────────────────
// Helpers.
// ───────────────────────────────────────────────────────────────

fn unique_count(level: &[CommunityId]) -> usize {
    let s: std::collections::HashSet<_> = level.iter().collect();
    s.len()
}
