//! Cross-cutting integration tests for DF Leiden incremental
//! community detection (M3.d-2).
//!
//! These tests pin the **modularity ε bound** between the
//! incremental algorithm and a fresh static GVE-Leiden recompute on
//! nontrivial graphs (Zachary's karate club + SBM(n=200, k=4)).
//! Per ADR-040 §D-2 the incremental algorithm is allowed to drift
//! within a small ε of the static optimum across a refresh cycle;
//! the daily refresh scheduler (ADR-040 §D-7) is the canonical reset
//! that resets the drift to zero. These tests pin that drift is
//! BOUNDED, not eliminated, on representative fixtures.
//!
//! Cross-references:
//! - Sahu *Dynamic Community Detection with Leiden* (arxiv 2024,
//!   2405.11658) §6 specifies the algorithm and the ε bound.
//! - ADR-040 §D-2 commits ArcGraph to the published ε bound.
//!
//! ε tolerances used here are intentionally conservative for small
//! synthetic fixtures — Sahu §6's tighter 1e-4 bound holds on
//! production-scale graphs but not necessarily on n=200 SBMs with
//! synthetic edge batches.

use arcgraph_community::{
    CommunityId, EdgeUpdate, Graph, GveLeiden, IncrementalResult, LeidenIncremental, LeidenParams,
    modularity,
};

// ───────────────────────────────────────────────────────────────
// Fixtures — Zachary's karate club + SBM helper.
//
// We deliberately copy these inline rather than sharing a
// `tests/common.rs` module: per the M3.d-2 prompt, sharing
// fixtures across integration test files is a v1.1 cleanup, not
// a M3.d-2 task. Keep this file self-contained.
// ───────────────────────────────────────────────────────────────

/// Edge list of Zachary's karate club. 78 edges between 34 nodes.
/// Source: standard reference encoding used by networkx's
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

/// Construct Zachary's karate club graph (34 nodes, 78 unweighted
/// undirected edges).
fn zachary_graph() -> Graph {
    let edges: Vec<(u32, u32, f32)> = ZACHARY_EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    Graph::from_edges_undirected(34, &edges)
}

/// Zachary's edge list as a HashSet of canonical (min, max)
/// pairs — used to find non-existing edges for perturbation.
fn zachary_edge_set() -> std::collections::HashSet<(u32, u32)> {
    ZACHARY_EDGES
        .iter()
        .map(|&(u, v)| if u <= v { (u, v) } else { (v, u) })
        .collect()
}

/// Linear congruential generator step. Same constants as the
/// `ldbc_community_detection.rs` bench; reproduces deterministic
/// pseudo-random sequences across test runs.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Uniform [0, 1) sample using top 53 bits of an LCG step.
fn lcg_unit(state: &mut u64) -> f64 {
    let raw = lcg_next(state);
    ((raw >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Generate a deterministic stochastic block model graph.
///
/// Returns `(graph, ground_truth)` where ground_truth[v] is v's
/// block id in `0..k`. Same shape as the `leiden_static_correctness`
/// fixture (kept inline per the M3.d-2 split decision).
fn sbm(n: u32, k: u32, p_in: f64, p_out: f64, seed: u64) -> (Graph, Vec<u32>) {
    assert!(n % k == 0, "n must be divisible by k for clean blocks");
    let block_size = n / k;
    let ground_truth: Vec<u32> = (0..n).map(|v| v / block_size).collect();

    let mut state = seed;
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if ground_truth[u as usize] == ground_truth[v as usize] {
                p_in
            } else {
                p_out
            };
            if lcg_unit(&mut state) < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    (Graph::from_edges_undirected(n, &edges), ground_truth)
}

// ───────────────────────────────────────────────────────────────
// Helpers.
// ───────────────────────────────────────────────────────────────

/// Convert a raw `u32` Level-0 assignment into the
/// `[CommunityId]` representation that [`modularity`] consumes.
fn raw_to_cid(raw: &[u32]) -> Vec<CommunityId> {
    raw.iter()
        .map(|&c| CommunityId::new(u64::from(c)))
        .collect()
}

/// Project a [`GveLeiden::run`] result's Level-0 assignment to
/// the raw `u32` representation that [`LeidenIncremental::apply_batch`]
/// expects as `c_prev`.
fn level0_as_raw(static_result_levels0: &[CommunityId]) -> Vec<u32> {
    static_result_levels0
        .iter()
        .map(|c| c.raw() as u32)
        .collect()
}

/// Evaluate `(graph, raw_assignment, gamma)` modularity by
/// projecting through `[CommunityId]`.
fn q_of(graph: &Graph, raw: &[u32], gamma: f64) -> f64 {
    modularity(graph, &raw_to_cid(raw), gamma)
}

// ───────────────────────────────────────────────────────────────
// Test cases.
// ───────────────────────────────────────────────────────────────

/// ε bound on Zachary, single non-existing inter-community edge
/// inserted. Per Sahu §6 the incremental modularity must stay
/// within a small ε of a fresh static recompute.
#[test]
fn epsilon_bound_zachary_single_edge_perturbation() {
    let g_old = zachary_graph();
    let params = LeidenParams::default();
    let prior = GveLeiden::run(&g_old, params);
    let prior_l0 = &prior.levels[0];

    // Find a non-existing edge whose endpoints are in different
    // communities per `prior`. Iterate ascending pairs and pick
    // the first one that satisfies both predicates so the test
    // is deterministic.
    let edge_set = zachary_edge_set();
    let mut candidate: Option<(u32, u32)> = None;
    'outer: for u in 0..34u32 {
        for v in (u + 1)..34u32 {
            if edge_set.contains(&(u, v)) {
                continue;
            }
            if prior_l0[u as usize] != prior_l0[v as usize] {
                candidate = Some((u, v));
                break 'outer;
            }
        }
    }
    let (u, v) = candidate.expect("Zachary has at least one non-edge across community boundary");
    eprintln!("perturbation edge: ({u}, {v})");

    // Build `g_new` with the edge appended.
    let mut new_edges: Vec<(u32, u32, f32)> =
        ZACHARY_EDGES.iter().map(|&(a, b)| (a, b, 1.0)).collect();
    new_edges.push((u, v, 1.0));
    let g_new = Graph::from_edges_undirected(34, &new_edges);

    // Apply incremental against the prior level-0 assignment.
    let prior_raw = level0_as_raw(prior_l0);
    let inc =
        LeidenIncremental::apply_batch(&g_new, &prior_raw, &[EdgeUpdate::Insert { u, v }], &params);
    let q_inc = q_of(&g_new, &inc.assignment, params.gamma);

    // Static recompute on `g_new`.
    let static_after = GveLeiden::run(&g_new, params);
    let q_static = modularity(&g_new, &static_after.levels[0], params.gamma);

    let drift = (q_static - q_inc).abs();
    eprintln!("Zachary single-edge: q_inc={q_inc:.6}, q_static={q_static:.6}, drift={drift:.6}",);
    assert!(
        drift <= 0.05,
        "Zachary single-edge drift {drift:.6} exceeds ε=0.05; q_inc={q_inc}, q_static={q_static}",
    );
}

/// ε bound on a 4-block SBM with a 50-edge mixed batch
/// (insertions + deletions). Per Sahu §6 + ADR-040 §D-2.
///
/// DETERMINISM (issue #505): this test was previously `#[ignore]`-
/// quarantined because the `drift <= 0.05` assertion flaked on Linux
/// CI (failed run 26619310555, then passed on an immediate re-run of
/// the *same commit* — the signature of process-random state, not
/// cross-platform FP divergence). The community engine uses no RNG;
/// the nondeterminism was `HashMap` iteration order in
/// `leiden_static::aggregate`, whose `pair_weights` map built the
/// super-graph edge list in a process-random `.into_iter()` order.
/// That changed neighbour order at deeper aggregation levels, which
/// changed non-associative FP summation order, which flipped the
/// `dq > best_dq + 1e-12` move tie at ULP boundaries → a different
/// agglomeration and a different `q_static`. The fix (leiden_static.rs:
/// `pair_weights` + `modularity`'s `comm_volume` → `BTreeMap`) makes
/// `GveLeiden::run` reproducible across runs and processes.
///
/// Per `feedback_determinism_oracle_concurrency_tests.md` we add a
/// binary-equal reference-snapshot oracle: a second static recompute
/// on the same graph must produce a byte-identical level hierarchy.
/// That is strictly stronger than the ε tolerance and would have
/// caught the original nondeterminism in-process (two recomputes use
/// distinct `RandomState` seeds via the per-thread counter, so the
/// pre-fix code could already diverge within a single run). The
/// ε=0.05 drift assertion is RETAINED because it pins a genuinely
/// different, non-zero property — incremental-vs-static approximation
/// quality — that a determinism oracle cannot replace; it is now
/// reproducible rather than flaky.
#[test]
fn epsilon_bound_sbm_4block_50edge_batch() {
    let (g, _ground_truth) = sbm(200, 4, 0.3, 0.02, 0xC0FFEE);
    let params = LeidenParams::default();
    let prior = GveLeiden::run(&g, params);
    let prior_l0 = &prior.levels[0];
    let prior_raw = level0_as_raw(prior_l0);

    // Build an existing-edge set so deletions can target real edges.
    //
    // DETERMINISM (issue #505): `BTreeSet`, not `HashSet`. The
    // engine-side `BTreeMap` fix makes `GveLeiden::run` reproducible,
    // but the *fixture* would still vary across processes if this were
    // a `HashSet`: `existing_vec` (the deletion pool) and
    // `working_edges` (which sets `g_after`'s neighbour order) are both
    // built from this set's iteration order, so a process-random order
    // would delete different edges AND give `g_after` a different
    // neighbour order — re-introducing the exact cross-run variance we
    // are eliminating. `BTreeSet` iterates in sorted `(u, v)` order, so
    // `g_after` is byte-identical across every run.
    let n = g.n();
    let mut existing: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for u in 0..n {
        for (w, _) in g.neighbors(u) {
            if u < w {
                existing.insert((u, w));
            }
        }
    }

    // Generate 50 edge updates: alternating insertions (deterministic
    // LCG-sampled non-edges) and deletions (deterministic
    // sorted-pool picks). The mix exercises both paths.
    let mut state: u64 = 0xBEEFFACE;
    let mut updates: Vec<EdgeUpdate> = Vec::with_capacity(50);
    let existing_vec: Vec<(u32, u32)> = existing.iter().copied().collect();

    let mut working_edges: Vec<(u32, u32, f32)> = Vec::new();
    for &e in &existing {
        working_edges.push((e.0, e.1, 1.0));
    }
    let mut deleted: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut inserted: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

    for i in 0..50 {
        if i % 2 == 0 {
            // Insertion: sample two distinct vertices until we get
            // a non-existing, non-already-inserted pair.
            for _retry in 0..1000 {
                let a = (lcg_next(&mut state) % u64::from(n)) as u32;
                let b = (lcg_next(&mut state) % u64::from(n)) as u32;
                if a == b {
                    continue;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if existing.contains(&(lo, hi)) || inserted.contains(&(lo, hi)) {
                    continue;
                }
                inserted.insert((lo, hi));
                working_edges.push((lo, hi, 1.0));
                updates.push(EdgeUpdate::Insert { u: lo, v: hi });
                break;
            }
        } else {
            // Deletion: pick a random existing edge that hasn't
            // been deleted yet.
            for _retry in 0..1000 {
                let idx = (lcg_next(&mut state) as usize) % existing_vec.len();
                let pair = existing_vec[idx];
                if deleted.contains(&pair) {
                    continue;
                }
                deleted.insert(pair);
                updates.push(EdgeUpdate::Delete {
                    u: pair.0,
                    v: pair.1,
                });
                break;
            }
        }
    }
    assert!(
        updates.len() >= 40,
        "expected ~50 updates after retry budget, got {}",
        updates.len(),
    );

    // Materialise `g_after` = working_edges minus `deleted`.
    let final_edges: Vec<(u32, u32, f32)> = working_edges
        .into_iter()
        .filter(|&(a, b, _)| !deleted.contains(&(a, b)))
        .collect();
    let g_after = Graph::from_edges_undirected(n, &final_edges);

    // Run incremental.
    let inc = LeidenIncremental::apply_batch(&g_after, &prior_raw, &updates, &params);
    let q_inc = q_of(&g_after, &inc.assignment, params.gamma);

    // Run fresh static.
    let static_after = GveLeiden::run(&g_after, params);
    let q_static = modularity(&g_after, &static_after.levels[0], params.gamma);

    // Binary-equal determinism oracle (issue #505): a second static
    // recompute on the same graph must produce a byte-identical level
    // hierarchy. Before the `BTreeMap` fix in `leiden_static.rs` this
    // could differ run-to-run (each `GveLeiden::run` allocates fresh
    // `HashMap`s with distinct `RandomState` seeds via the per-thread
    // counter, so even two in-process recomputes could pick different
    // aggregation orders). It now holds unconditionally, and is
    // strictly stronger than the ε tolerance below
    // (feedback_determinism_oracle_concurrency_tests.md).
    let static_again = GveLeiden::run(&g_after, params);
    assert_eq!(
        static_after.levels, static_again.levels,
        "GveLeiden::run must produce a byte-identical hierarchy across \
         recomputes (issue #505 determinism oracle)",
    );

    let drift = (q_static - q_inc).abs();
    eprintln!(
        "SBM 50-batch: q_inc={q_inc:.6}, q_static={q_static:.6}, drift={drift:.6}, \
         iterations={}, vertices_moved={}",
        inc.iterations, inc.vertices_moved,
    );
    assert!(
        drift <= 0.05,
        "SBM 50-batch drift {drift:.6} exceeds ε=0.05",
    );
    assert!(
        inc.iterations >= 1,
        "expected at least one iteration for a 50-edge batch, got {}",
        inc.iterations,
    );
    assert!(
        inc.vertices_moved >= 1,
        "expected at least one move for a 50-edge batch, got {}",
        inc.vertices_moved,
    );
}

/// 10-batch cumulative drift test. Each batch is ~10 edges and the
/// running incremental assignment is fed forward through all 10.
/// Per ADR-040 §D-7 drift accumulates across a refresh cycle but
/// remains BOUNDED — this test pins a 10× looser tolerance than the
/// per-batch ε.
#[test]
fn cumulative_stream_stays_within_epsilon() {
    let (g, _gt) = sbm(200, 4, 0.3, 0.02, 0xCAFEBABE);
    let params = LeidenParams::default();
    let c0 = GveLeiden::run(&g, params);
    let mut running_raw = level0_as_raw(&c0.levels[0]);

    // Track cumulative working edge set.
    //
    // DETERMINISM (issue #505): `BTreeSet`, not `HashSet`, for the same
    // reason as `epsilon_bound_sbm_4block_50edge_batch` above — both
    // `working` (which sets each `g_step`'s neighbour order) and the
    // deletion `pool` below are built from this set's iteration order,
    // so a process-random order would re-introduce cross-run variance.
    // Sorted iteration keeps every `g_step` byte-identical across runs.
    let mut existing: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for u in 0..g.n() {
        for (w, _) in g.neighbors(u) {
            if u < w {
                existing.insert((u, w));
            }
        }
    }
    let mut working: Vec<(u32, u32, f32)> = existing.iter().map(|&(a, b)| (a, b, 1.0)).collect();
    let mut deleted: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

    let mut state: u64 = 0xDEADBEEF_CAFEBABE;
    let n = g.n();

    for batch_idx in 0..10 {
        let mut updates: Vec<EdgeUpdate> = Vec::with_capacity(10);
        let mut local_inserted: Vec<(u32, u32)> = Vec::new();
        let mut local_deleted: Vec<(u32, u32)> = Vec::new();
        for j in 0..10 {
            if (batch_idx + j) % 2 == 0 {
                // Insertion.
                for _retry in 0..500 {
                    let a = (lcg_next(&mut state) % u64::from(n)) as u32;
                    let b = (lcg_next(&mut state) % u64::from(n)) as u32;
                    if a == b {
                        continue;
                    }
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    if existing.contains(&(lo, hi)) {
                        continue;
                    }
                    existing.insert((lo, hi));
                    local_inserted.push((lo, hi));
                    updates.push(EdgeUpdate::Insert { u: lo, v: hi });
                    break;
                }
            } else {
                // Deletion: pick the first non-deleted existing edge
                // hashed by state.
                let pool: Vec<(u32, u32)> = existing
                    .iter()
                    .filter(|p| !deleted.contains(*p))
                    .copied()
                    .collect();
                if pool.is_empty() {
                    continue;
                }
                let idx = (lcg_next(&mut state) as usize) % pool.len();
                let pair = pool[idx];
                deleted.insert(pair);
                local_deleted.push(pair);
                updates.push(EdgeUpdate::Delete {
                    u: pair.0,
                    v: pair.1,
                });
            }
        }

        // Update the working edge list by adding insertions; mark
        // deletions in `deleted` (skipped at materialisation).
        for &(a, b) in &local_inserted {
            working.push((a, b, 1.0));
        }

        // Materialise the cumulative graph after this batch.
        let cumulative_edges: Vec<(u32, u32, f32)> = working
            .iter()
            .copied()
            .filter(|&(a, b, _)| !deleted.contains(&(a, b)))
            .collect();
        let g_step = Graph::from_edges_undirected(n, &cumulative_edges);

        let inc = LeidenIncremental::apply_batch(&g_step, &running_raw, &updates, &params);
        running_raw = inc.assignment;
    }

    // Final cumulative graph.
    let final_edges: Vec<(u32, u32, f32)> = working
        .iter()
        .copied()
        .filter(|&(a, b, _)| !deleted.contains(&(a, b)))
        .collect();
    let g_final = Graph::from_edges_undirected(n, &final_edges);

    let q_running = q_of(&g_final, &running_raw, params.gamma);
    let static_final = GveLeiden::run(&g_final, params);
    let q_static = modularity(&g_final, &static_final.levels[0], params.gamma);

    let drift = (q_static - q_running).abs();
    eprintln!(
        "Cumulative 10-batch stream: q_running={q_running:.6}, q_static={q_static:.6}, drift={drift:.6}",
    );
    assert!(
        drift <= 0.10,
        "cumulative drift {drift:.6} exceeds 0.10 (10× per-batch ε)",
    );
}

/// Deletion-only batch on intra-community edges. Pins that the
/// algorithm doesn't panic, returns sensible counters, and stays
/// within ε of a static recompute.
#[test]
fn deletion_only_batch_is_handled() {
    let (g, _gt) = sbm(100, 4, 0.3, 0.02, 0xABCD_1234);
    let params = LeidenParams::default();
    let prior = GveLeiden::run(&g, params);
    let prior_l0 = &prior.levels[0];
    let prior_raw = level0_as_raw(prior_l0);

    // Pick 10 intra-community edges.
    let mut intra_edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..g.n() {
        for (w, _) in g.neighbors(u) {
            if u >= w {
                continue;
            }
            if prior_l0[u as usize] == prior_l0[w as usize] {
                intra_edges.push((u, w));
                if intra_edges.len() >= 10 {
                    break;
                }
            }
        }
        if intra_edges.len() >= 10 {
            break;
        }
    }
    assert!(
        intra_edges.len() >= 10,
        "fixture should yield at least 10 intra-community edges, got {}",
        intra_edges.len(),
    );

    let updates: Vec<EdgeUpdate> = intra_edges
        .iter()
        .map(|&(u, v)| EdgeUpdate::Delete { u, v })
        .collect();

    // Materialise `g_after` minus those edges.
    let to_delete: std::collections::HashSet<(u32, u32)> = intra_edges.iter().copied().collect();
    let mut all_edges: Vec<(u32, u32, f32)> = Vec::new();
    for u in 0..g.n() {
        for (w, weight) in g.neighbors(u) {
            if u < w {
                all_edges.push((u, w, weight));
            }
        }
    }
    let final_edges: Vec<(u32, u32, f32)> = all_edges
        .into_iter()
        .filter(|&(a, b, _)| !to_delete.contains(&(a, b)))
        .collect();
    let g_after = Graph::from_edges_undirected(g.n(), &final_edges);

    let inc = LeidenIncremental::apply_batch(&g_after, &prior_raw, &updates, &params);
    assert!(
        inc.iterations >= 1,
        "expected at least one iteration on a 10-deletion batch, got {}",
        inc.iterations,
    );
    // frontier_visits ≥ |affected vertex set| (Sahu §6).
    assert!(
        inc.frontier_visits >= 2,
        "frontier_visits {} should cover the affected set",
        inc.frontier_visits,
    );

    let q_inc = q_of(&g_after, &inc.assignment, params.gamma);
    let static_after = GveLeiden::run(&g_after, params);
    let q_static = modularity(&g_after, &static_after.levels[0], params.gamma);
    let drift = (q_static - q_inc).abs();
    eprintln!("Deletion-only batch: q_inc={q_inc:.6}, q_static={q_static:.6}, drift={drift:.6}",);
    assert!(drift <= 0.05, "deletion-only drift {drift:.6} > ε=0.05");
}

/// Empty batch is a strict no-op: byte-identical assignment to
/// the prior, zero counters across the board.
#[test]
fn empty_batch_is_strict_noop() {
    let g = zachary_graph();
    let params = LeidenParams::default();
    let prior = GveLeiden::run(&g, params);
    let prior_raw = level0_as_raw(&prior.levels[0]);

    let result: IncrementalResult = LeidenIncremental::apply_batch(&g, &prior_raw, &[], &params);
    assert_eq!(
        result.assignment, prior_raw,
        "empty batch must return byte-identical assignment to c_prev",
    );
    assert_eq!(result.iterations, 0, "empty batch: iterations must be 0");
    assert_eq!(
        result.vertices_moved, 0,
        "empty batch: vertices_moved must be 0",
    );
    assert_eq!(
        result.frontier_visits, 0,
        "empty batch: frontier_visits must be 0",
    );
}

/// Determinism: three runs with identical args produce
/// byte-identical results across all four counters.
#[test]
fn determinism_repeated_runs_are_byte_identical() {
    let (g, _gt) = sbm(100, 4, 0.3, 0.02, 0x9999_AAAA);
    let params = LeidenParams::default();
    let prior = GveLeiden::run(&g, params);
    let prior_raw = level0_as_raw(&prior.levels[0]);

    // Generate a deterministic 20-edge batch.
    let mut state: u64 = 0x1234_5678_DEAD_BEEF;
    let mut existing: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for u in 0..g.n() {
        for (w, _) in g.neighbors(u) {
            if u < w {
                existing.insert((u, w));
            }
        }
    }
    let n = g.n();
    let mut updates: Vec<EdgeUpdate> = Vec::with_capacity(20);
    let mut chosen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    while updates.len() < 20 {
        let a = (lcg_next(&mut state) % u64::from(n)) as u32;
        let b = (lcg_next(&mut state) % u64::from(n)) as u32;
        if a == b {
            continue;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if chosen.contains(&(lo, hi)) {
            continue;
        }
        chosen.insert((lo, hi));
        if existing.contains(&(lo, hi)) {
            updates.push(EdgeUpdate::Delete { u: lo, v: hi });
        } else {
            updates.push(EdgeUpdate::Insert { u: lo, v: hi });
        }
    }

    // Materialise g_after.
    let mut working: Vec<(u32, u32, f32)> = existing.iter().map(|&(a, b)| (a, b, 1.0)).collect();
    let mut deleted: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for upd in &updates {
        match *upd {
            EdgeUpdate::Insert { u, v } => {
                working.push((u, v, 1.0));
            }
            EdgeUpdate::Delete { u, v } => {
                deleted.insert((u, v));
            }
        }
    }
    let final_edges: Vec<(u32, u32, f32)> = working
        .into_iter()
        .filter(|&(a, b, _)| !deleted.contains(&(a, b)))
        .collect();
    let g_after = Graph::from_edges_undirected(n, &final_edges);

    let r1 = LeidenIncremental::apply_batch(&g_after, &prior_raw, &updates, &params);
    let r2 = LeidenIncremental::apply_batch(&g_after, &prior_raw, &updates, &params);
    let r3 = LeidenIncremental::apply_batch(&g_after, &prior_raw, &updates, &params);

    assert_eq!(r1.assignment, r2.assignment);
    assert_eq!(r2.assignment, r3.assignment);
    assert_eq!(r1.iterations, r2.iterations);
    assert_eq!(r2.iterations, r3.iterations);
    assert_eq!(r1.vertices_moved, r2.vertices_moved);
    assert_eq!(r2.vertices_moved, r3.vertices_moved);
    assert_eq!(r1.frontier_visits, r2.frontier_visits);
    assert_eq!(r2.frontier_visits, r3.frontier_visits);
}
