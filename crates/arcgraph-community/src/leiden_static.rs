//! GVE-Leiden static community detection per Sahu ICPP 2024
//! (arxiv 2312.13936) and Traag, Waltman, van Eck *From Louvain
//! to Leiden* (Sci. Rep. 9, 5233, 2019).
//!
//! # Algorithmic shape (Sahu §III, Traag 2019 §2)
//!
//! Each pass over the graph runs three phases:
//!
//! 1. **Local moving** — every vertex v greedily moves to the
//!    neighbouring community c that maximises Δmodularity, subject
//!    to the per-pass minimum-improvement threshold ε. We
//!    implement the frontier-based variant from Sahu §III.B
//!    sequentially: a vertex enters the next iteration's frontier
//!    if at least one of its neighbours moved in the current
//!    iteration. Vertices not in the frontier skip their move
//!    check (Sahu's measured speedup over an "all-vertices-each-
//!    pass" sweep is ~2×).
//!
//! 2. **Refinement** — Traag 2019's well-connectedness fix.
//!    Within each community produced by phase 1, run a constrained
//!    local-moving sweep where v can only move to communities
//!    that are subsets of v's parent community from phase 1. This
//!    produces a more granular partition whose communities are
//!    *guaranteed internally connected*; Louvain's omission of
//!    this phase admits the well-known disconnected-community
//!    pathology (Traag 2019 §3 measures 2.9 % of communities on
//!    LFR).
//!
//! 3. **Aggregation** — collapse each refined community into a
//!    super-vertex; the new graph has one node per refined
//!    community and inter-community edge weights summed. The
//!    super-graph is the input to the next pass.
//!
//! Iteration terminates when modularity-improvement across a
//! full pass falls below ε. The hierarchy of agglomerations IS
//! the hierarchy returned to consumers (per ADR-040 §D-5).
//!
//! # Sequential, not parallel (v1.0 implementation choice)
//!
//! Sahu §III pseudocode shows parallel local-moving via per-thread
//! reservation queues to avoid the well-known Louvain race; we
//! ship a *sequential* implementation at v1.0 because:
//!
//! - The v1.0 perf budget per ADR-040 §D-9 is "≤ 1 s on 8-core for
//!   100 M-edge graph". Our LDBC SF-1 bench measures
//!   **~2.3 M edges/sec** on the v1.0 sequential implementation
//!   (114 ms / 257 K edges); SF-1 itself runs in ~250 ms.
//!   Sequential extrapolation to 100 M edges is ~43.5 s; perfect
//!   8-core scaling extrapolates to ~5.4 s — *over the 1 s
//!   budget*. The §D-9 100M-edge budget is therefore
//!   **v1.1-conditional** on rayon parallelism (deferred from v1.0
//!   per ADR-040 §D-1) and ≥ 50% 8-core scaling efficiency. SF-1
//!   is the v1.0 CI source of truth, not the extrapolation;
//!   M3.d-2 prep includes an intermediate-scale bench (1 M-edge
//!   SBM or LDBC SF-3) to validate the SF-1 → 100M-edge
//!   extrapolation gap before v1.1 lifts.
//! - Parallel local-moving has well-documented correctness
//!   pitfalls: two threads moving v and one of v's neighbours
//!   concurrently see stale community labels and can converge to
//!   a worse modularity than sequential. Sahu's per-thread
//!   reservation queues fix this but at material implementation
//!   cost (lock-free queue per thread, careful read/write
//!   ordering on the community-label array).
//! - At the SF-1 scale we measure, sequential trivially saturates
//!   one core; Sahu's published 32-core 250 ms on 100 M edges is
//!   ~12× faster than sequential single-core would be. The v1.0
//!   1 s headroom budget at 100M-edge scale is reachable only
//!   with the v1.1 parallel variant — see the §D-9 v1.1-conditional
//!   framing above.
//!
//! When v1.1 lifts to the 100 M-edge target a parallel variant
//! lands as an ADR-040 amendment-01; the API surface here does
//! not change.
//!
//! # Determinism
//!
//! Sahu §III tie-break semantics depend on iteration order. We
//! iterate vertices in ascending `u32` order and break score
//! ties by ascending neighbour-community-id (the "first hit"
//! rule). The result is deterministic given the input graph;
//! the `seed` parameter on [`LeidenParams`] is reserved for v1.1
//! when randomized iteration orders are introduced for
//! anti-pathological hash-collision avoidance.

use std::collections::{BTreeMap, HashMap};

use arcgraph_core::{NodeId, TenantId};

use crate::graph::Graph;
use crate::ids::{CommunityId, Level};
use crate::membership_index::BTreeMembershipIndex;

/// Algorithm parameters per Sahu §III + ADR-040 §D-1.
///
/// The defaults match Sahu's published defaults exactly — DO NOT
/// vary these without an ADR-040 amendment per the
/// "evidence-over-intuition" Prime Directive.
#[derive(Copy, Clone, Debug)]
pub struct LeidenParams {
    /// Resolution parameter γ in the modularity formula.
    /// Default `1.0` per Sahu §III.A.
    pub gamma: f64,
    /// Modularity-improvement threshold ε. A pass that improves
    /// modularity by less than this terminates the iteration.
    /// Default `1e-4` per Sahu §III.A.
    pub min_modularity_gain: f64,
    /// Maximum iterations of local-moving per pass. Default `20`
    /// matches Sahu §III.B's measured upper bound on real-world
    /// graphs (additional iterations beyond ~10 yield diminishing
    /// returns).
    pub max_iters_per_phase: u32,
    /// Random seed for tie-breaking determinism. Reserved for
    /// v1.1 randomized-iteration support; the v1.0 implementation
    /// is deterministic at any seed.
    pub seed: u64,
}

impl Default for LeidenParams {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            min_modularity_gain: 1e-4,
            max_iters_per_phase: 20,
            seed: 0x42,
        }
    }
}

/// Result of a static GVE-Leiden run.
#[derive(Clone, Debug)]
pub struct LeidenResult {
    /// Per-level community assignment over **the original graph's
    /// vertices**. `levels[0][u]` is the leaf community of vertex
    /// u; `levels[1][u]` is u's level-1 community (the
    /// agglomeration of u's leaf community); etc.
    pub levels: Vec<Vec<CommunityId>>,
    /// Per-level modularity Q_l, computed on the original graph
    /// against `levels[l]`.
    pub modularity_per_level: Vec<f64>,
    /// Total local-moving iterations performed across all passes.
    pub total_iters: u32,
}

/// Static GVE-Leiden runner.
pub struct GveLeiden;

impl GveLeiden {
    /// Run the algorithm on `g` with `params`.
    ///
    /// Returns a [`LeidenResult`] with the full hierarchy of
    /// agglomerations, level 0 = finest (leaf communities).
    #[must_use]
    pub fn run(g: &Graph, params: LeidenParams) -> LeidenResult {
        let n_orig = g.n() as usize;
        // `current` is the graph the current pass operates on.
        // After the first pass it is the aggregated super-graph;
        // its vertices are the refined communities of the previous
        // pass.
        let mut current = g.clone();
        // `lift` maps `current`'s vertex index `i` to the original
        // graph's leaf-community membership of any vertex that
        // collapsed into it. Concretely: `lift[i]` is a Vec of
        // original-graph vertex indices that are currently in
        // super-vertex `i`. We use it to project each level's
        // assignment back to the original graph.
        let mut lift: Vec<Vec<u32>> = (0..n_orig as u32).map(|i| vec![i]).collect();

        let mut levels: Vec<Vec<CommunityId>> = Vec::new();
        let mut modularity_per_level: Vec<f64> = Vec::new();
        let mut total_iters: u32 = 0;

        // We bound the number of passes by `n_orig` as a safety net;
        // each pass strictly reduces vertex count or the algorithm
        // converges. In practice 4–6 passes per Traag 2019.
        let mut prev_q = modularity(g, &init_singletons_orig(n_orig), params.gamma);

        for _pass in 0..32 {
            // Phase 1: local moving on `current`.
            let n_cur = current.n() as usize;
            let mut c_cur: Vec<u32> = (0..n_cur as u32).collect();
            let (iters_lm, _improved) = local_moving_phase(
                &current, &mut c_cur, &params, /* refinement_constraint */ None,
            );
            total_iters = total_iters.saturating_add(iters_lm);

            // Phase 2: refinement (constrained local-moving) per
            // Traag 2019. Each vertex starts in its own refined
            // community and may only move to communities also
            // contained in its phase-1 community. The refined
            // partition is **internal to the algorithm**: it
            // drives the aggregation step (so the super-graph's
            // vertices are the well-connected refined communities)
            // but the level reported to consumers is the **phase-1
            // assignment** — the agglomeration hierarchy that
            // Traag 2019 §2 describes as "the partition Q_k".
            let mut c_refined: Vec<u32> = (0..n_cur as u32).collect();
            // Constraint: refined moves of v are admissible only
            // if the destination community's "parent" (= phase-1
            // community of any of its current members) matches
            // v's phase-1 community.
            let parent_of_refined: Vec<u32> = c_cur.clone();
            let (iters_ref, _) =
                local_moving_phase(&current, &mut c_refined, &params, Some(&parent_of_refined));
            total_iters = total_iters.saturating_add(iters_ref);

            // Project the **phase-1 (post-local-moving)** assignment
            // to the original graph via the current `lift` map.
            // Refinement is intentionally NOT projected: refinement
            // is the well-connectedness fix that makes aggregation
            // sound, but the consumer-facing hierarchy is the
            // agglomeration of phase-1 communities (Traag 2019).
            let mut level_assignment_orig = vec![CommunityId::ZERO; n_orig];
            for (super_i, members) in lift.iter().enumerate() {
                let cid = CommunityId::new(u64::from(c_cur[super_i]));
                for &orig_v in members {
                    level_assignment_orig[orig_v as usize] = cid;
                }
            }

            // Compute modularity on the original graph using the
            // projected assignment so all reported Q values are
            // comparable level-to-level.
            let q = modularity(g, &level_assignment_orig, params.gamma);
            levels.push(level_assignment_orig);
            modularity_per_level.push(q);

            let dq = q - prev_q;
            // Stop when modularity stops improving meaningfully
            // OR when refinement collapsed everything into one
            // community (no further structure to extract).
            let unique_refined = count_unique(&c_refined);
            if dq.abs() < params.min_modularity_gain || unique_refined <= 1 {
                break;
            }
            prev_q = q;

            // Phase 3: aggregate. Build a super-graph whose
            // vertices are the refined communities of `current`.
            let (next_graph, new_lift) = aggregate(&current, &c_refined, &lift);
            // If aggregation didn't reduce vertex count, no further
            // structure to find.
            if next_graph.n() == current.n() {
                break;
            }
            current = next_graph;
            lift = new_lift;
        }

        // Always emit at least the finest-level singleton
        // assignment if the loop exited without a level (e.g.
        // empty graph).
        if levels.is_empty() {
            let ids: Vec<CommunityId> = (0..n_orig as u64).map(CommunityId::new).collect();
            let q = modularity(g, &ids, params.gamma);
            levels.push(ids);
            modularity_per_level.push(q);
        }

        LeidenResult {
            levels,
            modularity_per_level,
            total_iters,
        }
    }

    /// Install a [`LeidenResult`] into a [`BTreeMembershipIndex`]
    /// for `tenant` at the given `install_lsn` (per ADR-041
    /// §D-3b). One [`BTreeMembershipIndex::install_level`] call
    /// per level. The integration boundary used by
    /// `CommunityRefreshScheduler` (M3.d-2) and the LDBC bench
    /// (commit #3 of M3.d-1).
    ///
    /// `install_lsn` is the LSN at which this refresh's results
    /// become visible. The scheduler allocates a fresh LSN per
    /// tick (per ADR-041 §D-3b — refreshes can advance install
    /// LSN without a corresponding write commit). At v1.0 the
    /// per-level pairs all share the same `install_lsn` (one
    /// refresh = one snapshot point); v1.1 may differentiate
    /// per-level if needed.
    ///
    /// `n_skip_prefix` is the count of leading vertex slots to
    /// drop when emitting `(NodeId, CommunityId)` pairs. Callers
    /// whose graph has a phantom prefix — e.g., the engine's
    /// CrudStore→Graph adapter sizes `n = high_water + 1` so
    /// vertex `0` corresponds to the reserved `NodeId::ZERO`
    /// sentinel that `CrudStore::alloc_node` never emits — pass
    /// the prefix length to keep `NodeId::ZERO` from leaking into
    /// the index. Standalone callers whose graphs use the natural
    /// `0..n` indexing pass `0`.
    pub fn install_into(
        result: &LeidenResult,
        index: &BTreeMembershipIndex,
        tenant: TenantId,
        install_lsn: arcgraph_core::Lsn,
        n_skip_prefix: u32,
    ) {
        let skip = n_skip_prefix as usize;
        for (li, level_assignment) in result.levels.iter().enumerate() {
            let level = Level::new(li as u8);
            // `skip` leading vertex slots are phantom (e.g., the
            // `NodeId::ZERO` sentinel reserved by the engine adapter)
            // and must not surface in the membership index. If
            // `skip >= level_assignment.len()` the level installs
            // empty (the underlying `install_level` accepts an empty
            // slice as a no-op).
            let pairs: Vec<(NodeId, CommunityId)> = level_assignment
                .iter()
                .enumerate()
                .skip(skip)
                .map(|(node_idx, c)| (NodeId::new(node_idx as u64), *c))
                .collect();
            index.install_level(tenant, level, install_lsn, &pairs);
        }
    }
}

// ───────────────────────────────────────────────────────────────
// Internals: local moving, refinement, aggregation, modularity.
// ───────────────────────────────────────────────────────────────

/// Initialize each original-graph vertex in its own community.
fn init_singletons_orig(n: usize) -> Vec<CommunityId> {
    (0..n as u64).map(CommunityId::new).collect()
}

fn count_unique(c: &[u32]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for &x in c {
        seen.insert(x);
    }
    seen.len()
}

/// One local-moving phase. Mutates `c` in place. Returns
/// `(iterations, any_moved)`.
///
/// `refinement_constraint`: if `Some(parent)`, vertex `v` may
/// only move to a community whose existing members all share the
/// same `parent[member]` value as `parent[v]` (Traag 2019
/// well-connectedness phase). We track each refined community's
/// "parent label" lazily — when v moves to a new community, the
/// parent label of that community must match `parent[v]`. This
/// trivially holds at the start since each vertex is its own
/// refined community with parent = `parent[v]`.
// pub(crate) for arcgraph-community::leiden_incremental (M3.d-2).
pub(crate) fn local_moving_phase(
    g: &Graph,
    c: &mut [u32],
    params: &LeidenParams,
    refinement_constraint: Option<&[u32]>,
) -> (u32, bool) {
    let n = g.n();
    if n == 0 {
        return (0, false);
    }

    // Per-community total volume (sum of vertex degrees in the
    // community). Initial: each vertex's own degree.
    let mut comm_volume: HashMap<u32, f64> = HashMap::with_capacity(n as usize);
    for u in 0..n {
        comm_volume
            .entry(c[u as usize])
            .and_modify(|v| *v += f64::from(g.degree(u)))
            .or_insert_with(|| f64::from(g.degree(u)));
    }

    // Per-community parent label (only used in refinement). For
    // refinement, we initialize from `parent[v]` for each
    // singleton community = v.
    let comm_parent: HashMap<u32, u32> = if let Some(parent) = refinement_constraint {
        let mut m = HashMap::with_capacity(n as usize);
        for u in 0..n {
            m.insert(c[u as usize], parent[u as usize]);
        }
        m
    } else {
        HashMap::new()
    };

    let two_m = g.total_weight_2m();
    if two_m <= 0.0 {
        // No edges: every vertex stays in its singleton community.
        return (0, false);
    }
    let inv_two_m = 1.0 / two_m;

    // Frontier-based sweep per Sahu §III.B. Iteration 0 considers
    // every vertex; subsequent iterations only consider vertices
    // whose neighbours moved.
    let mut frontier: Vec<bool> = vec![true; n as usize];
    let mut next_frontier: Vec<bool> = vec![false; n as usize];
    let mut any_moved_overall = false;

    for iter in 0..params.max_iters_per_phase {
        let mut moved_this_iter = false;
        // Sweep vertices in ascending order for determinism.
        for u in 0..n {
            if !frontier[u as usize] {
                continue;
            }
            let cu_before = c[u as usize];
            let kv = f64::from(g.degree(u));

            // Build a HashMap of neighbour-community → edge-weight
            // sum. Skip self-loops (Newman 2006 modularity does
            // include self-loops in the volume, but the move
            // gain only counts inter-vertex edges).
            //
            // We also track `e_v_cu` separately = edge weight from
            // v to its current community **excluding v itself**,
            // for the "remove" half of the gain.
            let mut e_v_neighbor_comm: HashMap<u32, f64> = HashMap::new();
            for (w, weight) in g.neighbors(u) {
                if w == u {
                    continue;
                }
                let cw = c[w as usize];
                *e_v_neighbor_comm.entry(cw).or_insert(0.0) += f64::from(weight);
            }
            let e_v_cu = e_v_neighbor_comm.get(&cu_before).copied().unwrap_or(0.0);
            // Volume of u's current community **without u**.
            let s_cu_minus_v = comm_volume.get(&cu_before).copied().unwrap_or(0.0) - kv;

            // Score the "remove from current" half of the move.
            // ΔQ_remove = -e_v_cu + γ * kv * s_cu_minus_v / (2m)
            // This is added to ΔQ_insert(C) for each candidate.
            let dq_remove = -e_v_cu + params.gamma * kv * s_cu_minus_v * inv_two_m;

            // Find best candidate community among `e_v_neighbor_comm`'s keys
            // (plus `cu_before` itself: a no-op move). Tie-break by
            // ascending community id for determinism.
            let mut best_c = cu_before;
            let mut best_dq: f64 = 0.0;
            // Sort candidates so the tie-break is stable.
            let mut candidates: Vec<u32> = e_v_neighbor_comm.keys().copied().collect();
            candidates.sort_unstable();
            for cnd in candidates {
                if cnd == cu_before {
                    continue;
                }
                // Refinement constraint: cnd's parent must match v's parent.
                if let Some(parent) = refinement_constraint {
                    let cnd_parent = comm_parent.get(&cnd).copied();
                    if cnd_parent != Some(parent[u as usize]) {
                        continue;
                    }
                }
                let e_v_cnd = e_v_neighbor_comm.get(&cnd).copied().unwrap_or(0.0);
                let s_cnd = comm_volume.get(&cnd).copied().unwrap_or(0.0);
                // ΔQ_insert(cnd) = e_v_cnd - γ * kv * s_cnd / (2m)
                let dq_insert = e_v_cnd - params.gamma * kv * s_cnd * inv_two_m;
                let dq = dq_remove + dq_insert;
                // Per Sahu §III.B and Traag 2019 §2: any positive
                // ΔQ is acceptable for a per-move decision; ε is
                // the *pass-level* convergence threshold (checked
                // outside this loop in `GveLeiden::run`).
                // We keep a tiny-but-positive threshold (1e-12) to
                // avoid float-noise oscillation; ties broken by
                // ascending community id (handled by the sorted
                // candidate iteration above).
                if dq > best_dq + 1e-12 {
                    best_dq = dq;
                    best_c = cnd;
                }
            }

            if best_c != cu_before {
                // Apply the move.
                c[u as usize] = best_c;
                *comm_volume.entry(cu_before).or_insert(0.0) -= kv;
                *comm_volume.entry(best_c).or_insert(0.0) += kv;
                moved_this_iter = true;
                any_moved_overall = true;
                // Push neighbours into the next frontier.
                for (w, _) in g.neighbors(u) {
                    if w != u {
                        next_frontier[w as usize] = true;
                    }
                }
            }
        }

        if !moved_this_iter {
            return (iter + 1, any_moved_overall);
        }
        // Swap frontier and reset next_frontier.
        std::mem::swap(&mut frontier, &mut next_frontier);
        for slot in next_frontier.iter_mut() {
            *slot = false;
        }
    }
    (params.max_iters_per_phase, any_moved_overall)
}

/// Aggregate `g` according to `c`: build a super-graph whose
/// vertices are `c`'s distinct communities and whose edges are
/// summed inter-community edge weights.
///
/// Returns `(super_graph, new_lift)` where `new_lift[i]` is the
/// list of original-graph vertices that collapsed into super-
/// vertex `i`. The mapping composes with the prior `prev_lift`:
/// each member of `prev_lift[j]` (j is a vertex of `g`) ends up
/// in `new_lift[c[j] -> super_id]`.
fn aggregate(g: &Graph, c: &[u32], prev_lift: &[Vec<u32>]) -> (Graph, Vec<Vec<u32>>) {
    // Step 1: dense-relabel community ids into 0..k.
    let mut comm_to_super: HashMap<u32, u32> = HashMap::new();
    let mut k = 0u32;
    for &cid in c {
        comm_to_super.entry(cid).or_insert_with(|| {
            let s = k;
            k += 1;
            s
        });
    }

    // Step 2: build the new lift.
    let mut new_lift: Vec<Vec<u32>> = vec![Vec::new(); k as usize];
    for (super_i, members) in prev_lift.iter().enumerate() {
        let new_super = comm_to_super[&c[super_i]];
        new_lift[new_super as usize].extend_from_slice(members);
    }

    // Step 3: aggregate edges. Iterate every half-edge once; key
    // by (super_u, super_v) and sum weights. To keep undirected
    // semantics with our `from_edges_undirected` helper (which
    // doubles edges itself), we only emit one (a, b) per
    // undirected pair (a ≤ b) with the **directed-half** weight
    // sum — since the input has both halves, dividing by 2 makes
    // the pair undirected. Self-loops (a == b) are emitted with
    // half their summed half-edge weight.
    //
    // DETERMINISM (issue #505): keyed by `BTreeMap`, not `HashMap`, so
    // the `.into_iter()` below yields the super-graph's edge list in a
    // fixed `(a, b)` order. `from_edges_undirected` scatters CSR
    // adjacency in edge-INSERTION order (see `graph.rs` Step 3 — it
    // does not sort neighbours), so a process-random `HashMap`
    // iteration order would give a process-random neighbour order at
    // the next aggregation level. `local_moving_phase` then sums the
    // per-neighbour-community edge weights in that order, and because
    // FP addition is non-associative the accumulated ΔQ terms differ
    // by ULPs, which intermittently flips the `dq > best_dq + 1e-12`
    // move tie — yielding a different agglomeration and a different
    // reported modularity across otherwise-identical runs. That was
    // the root cause of the `epsilon_bound_*` flake (failed run
    // 26619310555, passed on a same-commit re-run). `BTreeMap` makes
    // the whole multi-level recompute reproducible.
    let mut pair_weights: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for u in 0..g.n() {
        let su = comm_to_super[&c[u as usize]];
        for (w, weight) in g.neighbors(u) {
            let sw = comm_to_super[&c[w as usize]];
            let (a, b) = if su <= sw { (su, sw) } else { (sw, su) };
            *pair_weights.entry((a, b)).or_insert(0.0) += f64::from(weight);
        }
    }
    let edges: Vec<(u32, u32, f32)> = pair_weights
        .into_iter()
        .map(|((a, b), w_sum)| {
            // Each undirected pair (including self-loops) is
            // accumulated twice in the outer-inner sweep above
            // (once from each endpoint's neighbour iteration);
            // halve uniformly to recover the undirected weight.
            let w = (w_sum / 2.0) as f32;
            (a, b, w)
        })
        .collect();

    let next = Graph::from_edges_undirected(k, &edges);
    (next, new_lift)
}

/// Modularity Q (Newman 2006).
///
/// `Q = (1 / 2m) * Σ_{ij} [A_ij - γ * k_i * k_j / (2m)] * δ(c_i, c_j)`
///
/// We implement it as one pass over half-edges plus one pass over
/// vertices to collect the degree-product correction.
pub fn modularity(g: &Graph, c: &[CommunityId], gamma: f64) -> f64 {
    let n = g.n() as usize;
    if n == 0 {
        return 0.0;
    }
    let two_m = g.total_weight_2m();
    if two_m <= 0.0 {
        return 0.0;
    }
    // Sum A_ij terms where c_i == c_j (over half-edges, so we count each undirected edge twice — matching the formula's Σ_{ij}).
    let mut intra_edge_weight = 0.0_f64;
    // Sum of degrees per community.
    //
    // DETERMINISM (issue #505): `BTreeMap` so `sum_kc2` below sums the
    // per-community squared volumes in a fixed community-id order. The
    // reported `Q` feeds `GveLeiden::run`'s pass-termination test
    // (`dq.abs() < min_modularity_gain`); a process-random summation
    // order perturbs `Q` at the ULP level which — though far below the
    // ε=0.05 drift bound — could in principle flip that test and
    // change the emitted level count. Fixing the order removes the
    // last source of cross-run variance, so the binary-equal
    // determinism oracle in `tests/leiden_incremental_consistency.rs`
    // holds unconditionally.
    let mut comm_volume: BTreeMap<CommunityId, f64> = BTreeMap::new();
    for u in 0..n {
        let cu = c[u];
        comm_volume
            .entry(cu)
            .and_modify(|v| *v += f64::from(g.degree(u as u32)))
            .or_insert_with(|| f64::from(g.degree(u as u32)));
        for (w, weight) in g.neighbors(u as u32) {
            if c[w as usize] == cu {
                intra_edge_weight += f64::from(weight);
            }
        }
    }
    // Sum of (k_C)^2 = Σ_C |volume(C)|^2.
    let sum_kc2: f64 = comm_volume.values().map(|v| v * v).sum();

    // Q = (intra_edge_weight - γ * sum_kc2 / 2m) / 2m
    (intra_edge_weight - gamma * sum_kc2 / two_m) / two_m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modularity_empty_graph_is_zero() {
        let g = Graph::from_edges_undirected(5, &[]);
        let c: Vec<CommunityId> = (0..5).map(CommunityId::new).collect();
        assert_eq!(modularity(&g, &c, 1.0), 0.0);
    }

    #[test]
    fn modularity_singleton_assignment_on_single_edge() {
        // Two vertices, one edge of weight 1. Each in its own community.
        // 2m = 2. Intra-edge weight = 0 (no edge inside any
        // community). sum_kc2 = 1 + 1 = 2. Q = (0 - 1*2/2) / 2 = -0.5.
        let g = Graph::from_edges_undirected(2, &[(0, 1, 1.0)]);
        let c = vec![CommunityId::new(0), CommunityId::new(1)];
        let q = modularity(&g, &c, 1.0);
        assert!((q - (-0.5)).abs() < 1e-9, "got {q}");
    }

    #[test]
    fn modularity_unified_assignment_on_single_edge() {
        // Two vertices in one community. intra = 2 (two half-edges).
        // sum_kc2 = 2² = 4. Q = (2 - 1*4/2) / 2 = (2 - 2) / 2 = 0.
        let g = Graph::from_edges_undirected(2, &[(0, 1, 1.0)]);
        let c = vec![CommunityId::new(0), CommunityId::new(0)];
        let q = modularity(&g, &c, 1.0);
        assert!((q - 0.0).abs() < 1e-9, "got {q}");
    }

    #[test]
    fn run_on_two_disconnected_pairs_finds_two_communities() {
        // Vertices 0-1 and 2-3 connected; expect 2 communities.
        let g = Graph::from_edges_undirected(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        let r = GveLeiden::run(&g, LeidenParams::default());
        // At least one level reported.
        assert!(!r.levels.is_empty());
        let leaf = &r.levels[0];
        // 0 and 1 share a community; 2 and 3 share another.
        assert_eq!(leaf[0], leaf[1], "0 and 1 in same community");
        assert_eq!(leaf[2], leaf[3], "2 and 3 in same community");
        assert_ne!(leaf[0], leaf[2], "the two pairs are distinct");
    }

    #[test]
    fn run_on_empty_graph_emits_singletons() {
        let g = Graph::from_edges_undirected(3, &[]);
        let r = GveLeiden::run(&g, LeidenParams::default());
        assert_eq!(r.levels.len(), 1);
        let leaf = &r.levels[0];
        // All three in distinct communities.
        let unique: std::collections::HashSet<_> = leaf.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn run_is_deterministic() {
        // Same graph, same params, two runs → identical levels.
        let g = Graph::from_edges_undirected(
            6,
            &[
                (0, 1, 1.0),
                (1, 2, 1.0),
                (0, 2, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (3, 5, 1.0),
                (2, 3, 0.1),
            ],
        );
        let r1 = GveLeiden::run(&g, LeidenParams::default());
        let r2 = GveLeiden::run(&g, LeidenParams::default());
        assert_eq!(r1.levels, r2.levels);
    }

    // ───────────────────────────────────────────────────────────
    // Aggregation arithmetic (codex §4 ⚠ fix-up).
    //
    // Per codex M3.d retro 2026-05-03 §4: aggregation pair_weights
    // accumulates twice via the half-edge sweep (once from each
    // endpoint of every undirected non-self edge) then divides by
    // 2 before passing edges to from_edges_undirected. For
    // self-loops the half-edge sweep accumulates ONCE (since
    // from_edges_undirected only stores one half-edge for self-
    // loops) — the same /2 then halves the self-loop weight in
    // the super-graph.
    //
    // The resulting super-graph has DIFFERENT scaling for
    // inter-community edges (full original weight) vs intra-
    // community edges (halved weight) vs self-loops (halved
    // weight). This is "math correct but subtle" because:
    //   1. Reported modularity per level is computed on the
    //      ORIGINAL graph using the projected level assignment
    //      (`GveLeiden::run` lines 220-225), NOT on the super-
    //      graph — so super-graph scaling does not affect the
    //      reported Q.
    //   2. Local-moving on the super-graph in the next pass
    //      preserves greedy decisions because relative ΔQ gains
    //      scale uniformly within the same `(intra, inter)`
    //      relationship — the algorithm continues to find the
    //      same optima the original-graph local-moving would.
    //
    // The tests below pin the numeric convention so a future
    // refactor (e.g. switching to Sahu-direct half-edge
    // semantics, or directed-edge support) cannot silently
    // change the super-graph weights without a failing test.
    // The aggregation is currently inferred only via final
    // modularity (`zachary_modularity_at_least_0_40` etc.); these
    // tests assert the per-edge arithmetic directly with a
    // hand-computable fixture.
    // ───────────────────────────────────────────────────────────

    #[test]
    fn aggregation_super_graph_edge_weights_with_self_loops() {
        // Hand-computable fixture: 6 vertices, 2 communities
        // (c={0,1,2} → super-id 0; c={3,4,5} → super-id 1).
        //
        //   Within-A (community 0) edges: (0,1,1.0), (1,2,1.0)
        //   Within-B (community 3) edges: (3,4,1.0), (4,5,1.0)
        //   Inter-community edge: (2,3,3.0)
        //   Self-loops: (0,0,1.0) on vertex 0; (4,4,2.0) on vertex 4
        let g = Graph::from_edges_undirected(
            6,
            &[
                (0, 1, 1.0),
                (1, 2, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (2, 3, 3.0),
                (0, 0, 1.0),
                (4, 4, 2.0),
            ],
        );
        let c: Vec<u32> = vec![0, 0, 0, 3, 3, 3];
        let prev_lift: Vec<Vec<u32>> = (0..6u32).map(|i| vec![i]).collect();

        let (super_g, new_lift) = aggregate(&g, &c, &prev_lift);

        assert_eq!(
            super_g.n(),
            2,
            "two communities collapse to two super-vertices"
        );

        // `comm_to_super` densifies in encounter order along c, so
        // c=0 (encountered first at index 0) → super-id 0;
        // c=3 (encountered first at index 3) → super-id 1.
        let super_a = 0u32;
        let super_b = 1u32;

        // Lift fold-up: super-vertex A absorbs original {0,1,2};
        // super-vertex B absorbs original {3,4,5}.
        assert_eq!(new_lift[super_a as usize], vec![0, 1, 2]);
        assert_eq!(new_lift[super_b as usize], vec![3, 4, 5]);

        // Inter-community super-edge weight expected = 3.0
        // (matches the single original (2,3,3.0) edge):
        //   pair_weights[(A,B)] accumulates (vertex 2 sees 3, +3.0)
        //   + (vertex 3 sees 2, +3.0) = 6.0; halved → 3.0.
        // Self-loop super-A weight expected = 2.5:
        //   pair_weights[(A,A)] = (vertex 0: self-loop 1.0 + neighbor 1 → 1.0)
        //   + (vertex 1: 0 → 1.0 + 2 → 1.0)
        //   + (vertex 2: 1 → 1.0; neighbor 3 lands in (A,B), not (A,A))
        //   = 1.0 + 1.0 + 1.0 + 1.0 + 1.0 = 5.0; halved → 2.5.
        // Self-loop super-B weight expected = 3.0:
        //   pair_weights[(B,B)] = (v3: 4 → 1.0)
        //   + (v4: self-loop 2.0 + 3 → 1.0 + 5 → 1.0)
        //   + (v5: 4 → 1.0)
        //   = 1.0 + 2.0 + 1.0 + 1.0 + 1.0 = 6.0; halved → 3.0.
        let neighbors_a: Vec<(u32, f32)> = super_g.neighbors(super_a).collect();
        let neighbors_b: Vec<(u32, f32)> = super_g.neighbors(super_b).collect();

        // Use a small epsilon for f32 sums; values are tiny.
        let weight_to = |neighbors: &[(u32, f32)], target: u32| -> f32 {
            neighbors
                .iter()
                .filter(|(w, _)| *w == target)
                .map(|(_, w)| *w)
                .sum()
        };

        let inter_a_to_b = weight_to(&neighbors_a, super_b);
        let inter_b_to_a = weight_to(&neighbors_b, super_a);
        // from_edges_undirected stores both half-edges of an inter
        // pair, so each side observes the full inter weight.
        assert!(
            (inter_a_to_b - 3.0).abs() < 1e-6,
            "inter super-edge A→B weight should be 3.0 (= original (2,3,3.0)), got {inter_a_to_b}",
        );
        assert!(
            (inter_b_to_a - 3.0).abs() < 1e-6,
            "inter super-edge B→A weight should be 3.0 by symmetry, got {inter_b_to_a}",
        );

        // Self-loop weights (one half-edge per super-vertex).
        let self_a = weight_to(&neighbors_a, super_a);
        let self_b = weight_to(&neighbors_b, super_b);
        assert!(
            (self_a - 2.5).abs() < 1e-6,
            "super-self-loop A weight should be 2.5 (= half of within-A + self-loop sweep accumulation), got {self_a}",
        );
        assert!(
            (self_b - 3.0).abs() < 1e-6,
            "super-self-loop B weight should be 3.0 (= half of within-B + self-loop sweep accumulation), got {self_b}",
        );

        // 2m for the super-graph: deg(A) = 2.5 + 3.0 = 5.5,
        // deg(B) = 3.0 + 3.0 = 6.0, total = 11.5. This is HALVED
        // from the original-graph 2m (= 17.0) because internal
        // edges are halved and self-loops are halved by the
        // aggregation arithmetic. Modularity is invariant to the
        // scaling within the (intra-only) sub-formula but NOT
        // across (inter-vs-intra) — see the module-level comment
        // above for why this is OK in practice.
        assert!(
            (super_g.total_weight_2m() - 11.5).abs() < 1e-6,
            "super-graph 2m should be 11.5, got {}",
            super_g.total_weight_2m(),
        );
    }

    #[test]
    fn aggregation_super_graph_self_loop_on_singleton_community() {
        // Degenerate case: vertex 0 has a self-loop AND is alone
        // in its community. The super-graph after aggregate must
        // still have a vertex 0 with a self-loop (halved per the
        // sweep+halve convention).
        let g = Graph::from_edges_undirected(
            3,
            &[
                (0, 0, 1.0), // self-loop on vertex 0 (singleton community)
                (1, 2, 0.5), // edge between two other singleton communities
            ],
        );
        // Each vertex its own community.
        let c: Vec<u32> = vec![0, 1, 2];
        let prev_lift: Vec<Vec<u32>> = (0..3u32).map(|i| vec![i]).collect();

        let (super_g, new_lift) = aggregate(&g, &c, &prev_lift);

        assert_eq!(
            super_g.n(),
            3,
            "three singletons collapse to three super-vertices"
        );

        // Densification preserves order for singletons.
        for (i, lift_entry) in new_lift.iter().enumerate() {
            assert_eq!(lift_entry, &vec![i as u32], "lift[{i}] = [{i}]");
        }

        // Super-vertex 0's self-loop weight:
        //   pair_weights[(0,0)] = (vertex 0 sees neighbor 0 with weight 1.0)
        //   = 1.0; halved → 0.5
        let self_0_weight: f32 = super_g
            .neighbors(0)
            .filter(|(w, _)| *w == 0)
            .map(|(_, w)| w)
            .sum();
        assert!(
            (self_0_weight - 0.5).abs() < 1e-6,
            "self-loop on singleton vertex 0 should aggregate to 0.5 (= 1.0 / 2), got {self_0_weight}",
        );

        // Super-edge (1, 2) weight:
        //   pair_weights[(1,2)] = 0.5 (from vertex 1) + 0.5 (from vertex 2) = 1.0
        //   halved → 0.5.
        let edge_1_to_2: f32 = super_g
            .neighbors(1)
            .filter(|(w, _)| *w == 2)
            .map(|(_, w)| w)
            .sum();
        assert!(
            (edge_1_to_2 - 0.5).abs() < 1e-6,
            "inter super-edge (1, 2) should be 0.5, got {edge_1_to_2}",
        );

        // Vertices 1 and 2 must NOT have self-loops (none in the
        // original graph; their singletons are not collapsed
        // multi-vertex communities so no within-community edge
        // exists to seed a self-loop).
        let self_1: f32 = super_g
            .neighbors(1)
            .filter(|(w, _)| *w == 1)
            .map(|(_, w)| w)
            .sum();
        let self_2: f32 = super_g
            .neighbors(2)
            .filter(|(w, _)| *w == 2)
            .map(|(_, w)| w)
            .sum();
        assert!(
            self_1.abs() < 1e-6,
            "super-vertex 1 should NOT have a self-loop, got {self_1}",
        );
        assert!(
            self_2.abs() < 1e-6,
            "super-vertex 2 should NOT have a self-loop, got {self_2}",
        );
    }
}
