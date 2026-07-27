//! Bounded k-hop expansion (`GraphTraversalHandle::k_hop`).
//!
//! Two sampling strategies per the roadmap V11-S-04 row:
//!
//! - [`SamplingStrategy::Bfs`] — exhaustive breadth-first expansion. With
//!   `limit = usize::MAX` + `cost_budget = Some(b)` this reproduces
//!   PRIM-1's `collect_reachable` observable contract exactly (ADR-205
//!   §D-5 parity oracle): cost charged anchor → first-seen-edge →
//!   first-seen-node, budget checked INSIDE the loop (H-1), structured
//!   trip error carrying the consumed total including the tripping item.
//! - [`SamplingStrategy::ReservoirVitterL`] — level-synchronous per-hop
//!   weighted reservoir (J6 §5 line 234 degree-aware sampling): every
//!   newly-discovered candidate is offered with weight `1/deg_f(u)` of its
//!   *expanding* node `u` (free: the length of `u`'s neighbor batch), so a
//!   10⁶-degree hub's candidates are down-weighted 10⁻⁶ each instead of
//!   flooding the context distribution. Roadmap-bound API token; algorithm
//!   attribution in [`crate::rng`] (Vitter 1985 / Li 1994 / Efraimidis–
//!   Spirakis 2006 A-Res).
//!
//! Budget (PD#5): BFS visits `O(Σ deg_f(expanded))` neighbor items with
//! `O(1)` amortized bookkeeping each (hash probes + saturating adds); k=3,
//! LIMIT=1000, d̄≤16 ⇒ ≤ ~5×10⁴ items ⇒ µs–low-ms in-crate (the adapter's
//! `neighbors` calls dominate end-to-end). Reservoir adds one RNG draw per
//! new candidate + `O(log s)` heap work per accepted candidate. Memory is
//! result-proportional + one frontier.

use std::collections::{HashSet, VecDeque};

use arcgraph_core::{NodeId, RelId};

use crate::error::TraversalError;
use crate::rng::{SplitMix64, WeightedReservoir};
use crate::source::{EdgeSource, Neighbor, TraversalDirection};

/// How `k_hop` explores each hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Exhaustive breadth-first expansion (the PRIM-1 v1.0-α behavior).
    Bfs,
    /// Degree-aware per-hop weighted reservoir sampling. The token is the
    /// roadmap-bound name; see the module docs for the (corrected)
    /// algorithm attribution.
    ReservoirVitterL,
}

/// `k_hop` request. Construct with [`KHopRequest::new`] then override
/// fields; every override is additive so call sites stay stable as axes
/// grow.
#[derive(Debug, Clone)]
pub struct KHopRequest {
    /// Maximum hop depth (`k`). Nodes at depth `k` are retained but not
    /// expanded (the `collect_reachable` depth-budget rule).
    pub k: u32,
    /// Edge orientation to follow.
    pub direction: TraversalDirection,
    /// Edge predicate pushed into the source.
    pub filter: crate::source::EdgeFilter,
    /// Maximum nodes retained INCLUDING the anchor (LIMIT pushdown: the
    /// traversal stops at the first trip; it never scans-then-truncates).
    /// `usize::MAX` = unbounded. `0` is rejected (`InvalidRequest`).
    pub limit: usize,
    /// H-1 cost budget over adapter-defined units (bytes for PRIM-1);
    /// `None` = unbounded. The trip is [`TraversalError::CostBudgetExceeded`]
    /// mid-loop, never a silently truncated result.
    pub cost_budget: Option<u64>,
    /// Supernode-firewall hook (ADR-025 §5 / V11-S-06): caps how many NEW
    /// nodes any single hop may retain. Trips degrade the result
    /// (reported as [`Truncation::FrontierCapped`]) instead of erroring —
    /// the firewall bounds blowup; the query proceeds.
    pub per_hop_frontier_cap: Option<usize>,
    /// Exploration strategy.
    pub sampling: SamplingStrategy,
    /// Determinism seed (reservoir mode). Same `(seed, request, source
    /// order)` ⇒ identical result.
    pub seed: u64,
}

impl KHopRequest {
    /// BFS request with no limits beyond depth `k` — the PRIM-1 shape.
    #[must_use]
    pub fn new(k: u32, direction: TraversalDirection) -> Self {
        Self {
            k,
            direction,
            filter: crate::source::EdgeFilter::any(),
            limit: usize::MAX,
            cost_budget: None,
            per_hop_frontier_cap: None,
            sampling: SamplingStrategy::Bfs,
            seed: 0,
        }
    }
}

/// One retained node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KHopNode<N> {
    /// Node identity.
    pub node: NodeId,
    /// BFS depth at first retention (anchor = 0). In reservoir mode this
    /// is the hop the node was sampled in (still its minimal depth among
    /// *sampled* parents).
    pub depth: u32,
    /// Adapter payload (e.g. `NodeView`).
    pub data: N,
}

/// One retained edge (first-sighting per `RelId`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KHopEdge<E> {
    /// Relationship identity (the dedupe axis).
    pub rel_id: RelId,
    /// Source endpoint as traversed.
    pub src: NodeId,
    /// Destination endpoint as traversed.
    pub dst: NodeId,
    /// Adapter payload (e.g. `RelView`).
    pub data: E,
}

/// Why the result is smaller than exhaustive expansion would have been.
///
/// `#[non_exhaustive]`: downstream consumers (the V11-S-06 firewall,
/// EXPLAIN surfaces) must tolerate future reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Truncation {
    /// Exhaustive within the requested `k`/filter — nothing dropped.
    #[default]
    None,
    /// The retained-node `limit` bound the traversal (LIMIT pushdown
    /// stop, or the reservoir capacity was bound by the remaining limit).
    LimitReached,
    /// The `per_hop_frontier_cap` bound at least one hop (the ADR-025 §5
    /// supernode-firewall degradation signal V11-S-06 consumes).
    FrontierCapped,
}

/// `k_hop` output.
#[derive(Debug, Clone)]
pub struct KHopResult<N, E> {
    /// Retained nodes (anchor first; then retention order, which is
    /// deterministic for a deterministic source).
    pub nodes: Vec<KHopNode<N>>,
    /// First-seen edges among expanded nodes (includes cycle-closing
    /// edges between retained nodes). Every edge's endpoints are present
    /// in `nodes`.
    pub edges: Vec<KHopEdge<E>>,
    /// Strongest binding constraint that shrank the result.
    pub truncation: Truncation,
    /// Total cost charged (adapter units) — equals PRIM-1's
    /// `bytes_consumed` accounting under the parity configuration.
    pub cost_consumed: u64,
}

/// Charge `add` against the running total, tripping STRICTLY ABOVE the
/// budget (`collect_reachable` parity: `bytes > byte_budget` trips; equal
/// passes).
fn charge<E>(cost: &mut u64, add: u64, budget: Option<u64>) -> Result<(), TraversalError<E>> {
    *cost = cost.saturating_add(add);
    if let Some(b) = budget {
        if *cost > b {
            return Err(TraversalError::CostBudgetExceeded {
                cost_budget: b,
                cost_consumed: *cost,
            });
        }
    }
    Ok(())
}

/// Sugar for `k_hop`'s return type (clippy type-complexity; the
/// projection keeps call sites readable without erasing the source's
/// associated types).
pub type KHopOutcome<S> = Result<
    KHopResult<<S as EdgeSource>::NodeData, <S as EdgeSource>::EdgeData>,
    TraversalError<<S as EdgeSource>::Error>,
>;

/// One reservoir entry: `(stream position, source, candidate)` — the position
/// restores deterministic retention order after the heap scrambles
/// winners.
type SampledCandidate<S> = (
    usize,
    NodeId,
    Neighbor<<S as EdgeSource>::NodeData, <S as EdgeSource>::EdgeData>,
);

/// Bounded k-hop expansion from `anchor`. See the module docs; prefer
/// calling through [`crate::GraphTraversalHandle::k_hop`].
pub fn k_hop<S: EdgeSource + ?Sized>(
    source: &S,
    anchor: NodeId,
    anchor_data: S::NodeData,
    anchor_cost: u64,
    req: &KHopRequest,
) -> KHopOutcome<S> {
    if req.limit == 0 {
        return Err(TraversalError::InvalidRequest {
            reason: "k_hop limit must retain at least the anchor (limit >= 1)",
        });
    }
    if req.per_hop_frontier_cap == Some(0) {
        return Err(TraversalError::InvalidRequest {
            reason: "per_hop_frontier_cap of 0 forbids all expansion; omit the cap or use k = 0",
        });
    }
    tracing::debug!(
        anchor = anchor.raw(),
        k = req.k,
        limit = req.limit,
        sampling = ?req.sampling,
        "k_hop start"
    );

    let mut out = KHopResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        truncation: Truncation::None,
        cost_consumed: 0,
    };

    // Seed: the anchor's cost counts too — an absurdly small budget can
    // trip on the anchor alone, which is the correct discipline (the
    // caller asked for less budget than a single node costs). Parity with
    // `collect_reachable`'s anchor seeding.
    charge(&mut out.cost_consumed, anchor_cost, req.cost_budget)?;
    out.nodes.push(KHopNode {
        node: anchor,
        depth: 0,
        data: anchor_data,
    });

    match req.sampling {
        SamplingStrategy::Bfs => bfs(source, anchor, req, &mut out)?,
        SamplingStrategy::ReservoirVitterL => reservoir(source, anchor, req, &mut out)?,
    }
    Ok(out)
}

/// Exhaustive BFS (the `collect_reachable` absorption).
fn bfs<S: EdgeSource + ?Sized>(
    source: &S,
    anchor: NodeId,
    req: &KHopRequest,
    out: &mut KHopResult<S::NodeData, S::EdgeData>,
) -> Result<(), TraversalError<S::Error>> {
    let mut seen_nodes: HashSet<NodeId> = HashSet::from([anchor]);
    let mut seen_edges: HashSet<RelId> = HashSet::new();
    let mut queue: VecDeque<(NodeId, u32)> = VecDeque::from([(anchor, 0)]);
    // Newly-retained count per depth, for the firewall cap. Index = depth.
    let mut per_hop_new: Vec<usize> = vec![0; req.k as usize + 1];

    while let Some((cur, cur_depth)) = queue.pop_front() {
        if cur_depth >= req.k {
            // Don't expand past the depth budget (nodes AT depth k are
            // retained but never expanded).
            continue;
        }
        let batch = source
            .neighbors(cur, req.direction, &req.filter)
            .map_err(TraversalError::Source)?;
        let dst_depth = cur_depth + 1;
        for nb in batch {
            let is_new = !seen_nodes.contains(&nb.dst);

            // Firewall cap: a NEW node at a capped-full hop is skipped
            // entirely (edge included — no dangling endpoints), reported
            // as a degradation, and the traversal continues.
            if is_new {
                if let Some(cap) = req.per_hop_frontier_cap {
                    if per_hop_new[dst_depth as usize] >= cap {
                        out.truncation = Truncation::FrontierCapped;
                        continue;
                    }
                }
                // LIMIT pushdown: stop the whole traversal at the first
                // would-exceed point. Edges recorded so far stay; no edge
                // to a never-retained node is emitted.
                if out.nodes.len() >= req.limit {
                    out.truncation = Truncation::LimitReached;
                    return Ok(());
                }
            }

            // H-1 charge order parity: edge (first sight per RelId) ...
            if seen_edges.insert(nb.rel_id) {
                charge(&mut out.cost_consumed, nb.edge_cost, req.cost_budget)?;
                out.edges.push(KHopEdge {
                    rel_id: nb.rel_id,
                    src: cur,
                    dst: nb.dst,
                    data: nb.edge_data,
                });
            }
            // ... then node (first sight per NodeId).
            if is_new {
                seen_nodes.insert(nb.dst);
                charge(&mut out.cost_consumed, nb.node_cost, req.cost_budget)?;
                per_hop_new[dst_depth as usize] += 1;
                out.nodes.push(KHopNode {
                    node: nb.dst,
                    depth: dst_depth,
                    data: nb.dst_data,
                });
                if dst_depth < req.k {
                    queue.push_back((nb.dst, dst_depth));
                }
            }
        }
    }
    Ok(())
}

/// Level-synchronous per-hop weighted reservoir (degree-aware, J6 §5).
fn reservoir<S: EdgeSource + ?Sized>(
    source: &S,
    anchor: NodeId,
    req: &KHopRequest,
    out: &mut KHopResult<S::NodeData, S::EdgeData>,
) -> Result<(), TraversalError<S::Error>> {
    let mut seen_nodes: HashSet<NodeId> = HashSet::from([anchor]);
    let mut seen_edges: HashSet<RelId> = HashSet::new();
    let mut frontier: Vec<NodeId> = vec![anchor];

    for depth in 1..=req.k {
        if frontier.is_empty() {
            break;
        }
        let remaining_limit = req.limit.saturating_sub(out.nodes.len());
        let cap = req.per_hop_frontier_cap.unwrap_or(usize::MAX);
        let capacity = cap.min(remaining_limit);
        // Per-hop deterministic RNG stream: decoupling the stream from
        // candidate COUNTS in earlier hops would require replaying them;
        // seeding per (seed, depth) keeps each hop reproducible on its
        // own inputs.
        let mut rng =
            SplitMix64::new(req.seed ^ (u64::from(depth)).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut sampler: WeightedReservoir<SampledCandidate<S>> =
            WeightedReservoir::new(capacity.max(1));
        let mut offered_new: usize = 0;
        let mut stream_pos: usize = 0;

        for &u in &frontier {
            let batch = source
                .neighbors(u, req.direction, &req.filter)
                .map_err(TraversalError::Source)?;
            // Degree-aware weight: 1/deg_f(u) for every candidate emitted
            // from u — free, since the batch length IS u's filtered
            // degree. A hub's candidates are individually down-weighted
            // by its fanout (J6 §5 line 234).
            let w = 1.0 / (batch.len().max(1) as f64);
            for nb in batch {
                if seen_nodes.contains(&nb.dst) {
                    // Cycle-closing edge between retained nodes: record
                    // eagerly (first sight), charged under H-1.
                    if seen_edges.insert(nb.rel_id) {
                        charge(&mut out.cost_consumed, nb.edge_cost, req.cost_budget)?;
                        out.edges.push(KHopEdge {
                            rel_id: nb.rel_id,
                            src: u,
                            dst: nb.dst,
                            data: nb.edge_data,
                        });
                    }
                    continue;
                }
                if capacity == 0 {
                    // No retention room this hop (limit exhausted): every
                    // new candidate is dropped by the binding limit.
                    out.truncation = Truncation::LimitReached;
                    continue;
                }
                offered_new += 1;
                sampler.offer((stream_pos, u, nb), w, &mut rng);
                stream_pos += 1;
            }
        }

        if capacity == 0 {
            // Nothing can be retained at this or any deeper hop.
            return Ok(());
        }

        let accepted = sampler.len();
        if offered_new > accepted {
            // Report the strongest binding constraint. (With neither cap
            // nor limit binding, capacity is usize::MAX and the reservoir
            // never drops, so this branch cannot fire spuriously.)
            out.truncation = if capacity == remaining_limit && remaining_limit < cap {
                Truncation::LimitReached
            } else {
                Truncation::FrontierCapped
            };
        }

        let mut winners = sampler.into_items();
        // Deterministic retention order = stream order.
        winners.sort_by_key(|(pos, _, _)| *pos);

        let mut next_frontier: Vec<NodeId> = Vec::with_capacity(winners.len());
        for (_, src, nb) in winners {
            // Same-dst candidates can win twice (two parents); first
            // stream position retains, later ones contribute their edge
            // only (it closes a cycle onto a just-retained node).
            let is_new = seen_nodes.insert(nb.dst);
            if seen_edges.insert(nb.rel_id) {
                charge(&mut out.cost_consumed, nb.edge_cost, req.cost_budget)?;
                out.edges.push(KHopEdge {
                    rel_id: nb.rel_id,
                    src,
                    dst: nb.dst,
                    data: nb.edge_data,
                });
            }
            if is_new {
                charge(&mut out.cost_consumed, nb.node_cost, req.cost_budget)?;
                out.nodes.push(KHopNode {
                    node: nb.dst,
                    depth,
                    data: nb.dst_data,
                });
                if depth < req.k {
                    next_frontier.push(nb.dst);
                }
            }
        }
        frontier = next_frontier;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEdgeSource;
    use crate::source::EdgeFilter;

    fn chain(n: u64) -> MemoryEdgeSource {
        // 1 -> 2 -> 3 -> ... -> n
        let mut g = MemoryEdgeSource::new();
        for i in 1..n {
            g.add_edge(NodeId::new(i), NodeId::new(i + 1), None);
        }
        g
    }

    fn req(k: u32) -> KHopRequest {
        KHopRequest::new(k, TraversalDirection::Outbound)
    }

    #[test]
    fn bfs_depths_match_hop_distance_on_a_chain() {
        let g = chain(10);
        let r = k_hop(&g, NodeId::new(1), (), 0, &req(3)).expect("traversal");
        let depths: Vec<(u64, u32)> = r.nodes.iter().map(|n| (n.node.raw(), n.depth)).collect();
        assert_eq!(depths, vec![(1, 0), (2, 1), (3, 2), (4, 3)]);
        assert_eq!(r.edges.len(), 3);
        assert_eq!(r.truncation, Truncation::None);
    }

    #[test]
    fn depth_k_nodes_are_retained_but_not_expanded() {
        let g = chain(10);
        let r = k_hop(&g, NodeId::new(1), (), 0, &req(1)).expect("traversal");
        assert_eq!(r.nodes.len(), 2, "anchor + exactly one hop");
    }

    #[test]
    fn anchor_cost_alone_can_trip_the_budget() {
        let g = chain(3);
        let mut rq = req(2);
        rq.cost_budget = Some(4);
        let err = k_hop(&g, NodeId::new(1), (), 5, &rq).expect_err("must trip");
        match err {
            TraversalError::CostBudgetExceeded {
                cost_budget,
                cost_consumed,
            } => {
                assert_eq!(cost_budget, 4);
                assert_eq!(cost_consumed, 5, "includes the tripping item");
            }
            other => panic!("expected budget trip, got {other:?}"),
        }
    }

    #[test]
    fn budget_trips_mid_loop_on_edge_then_node_charges() {
        // anchor cost 2; each edge cost 3; each node cost 7.
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.set_uniform_costs(3, 7);
        let mut rq = req(1);
        // 2 (anchor) + 3 (edge) = 5 <= 5 passes; +7 (node) = 12 > 5 trips.
        rq.cost_budget = Some(5);
        let err = k_hop(&g, NodeId::new(1), (), 2, &rq).expect_err("must trip");
        match err {
            TraversalError::CostBudgetExceeded { cost_consumed, .. } => {
                assert_eq!(cost_consumed, 12, "edge charged before node; node tripped");
            }
            other => panic!("expected budget trip, got {other:?}"),
        }
    }

    #[test]
    fn equal_to_budget_passes_strictly_above_trips() {
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.set_uniform_costs(3, 7);
        let mut rq = req(1);
        rq.cost_budget = Some(12); // 2 + 3 + 7 == 12 exactly: passes.
        let r = k_hop(&g, NodeId::new(1), (), 2, &rq).expect("equal-to-budget passes");
        assert_eq!(r.cost_consumed, 12);
    }

    #[test]
    fn parallel_edges_dedupe_by_rel_id_and_diamond_nodes_retain_once() {
        // 1 -> 2 (two parallel rels), 1 -> 3, 2 -> 4, 3 -> 4 (diamond).
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.add_edge(NodeId::new(1), NodeId::new(3), None);
        g.add_edge(NodeId::new(2), NodeId::new(4), None);
        g.add_edge(NodeId::new(3), NodeId::new(4), None);
        let r = k_hop(&g, NodeId::new(1), (), 0, &req(2)).expect("traversal");
        assert_eq!(r.nodes.len(), 4, "4 distinct nodes");
        assert_eq!(
            r.edges.len(),
            5,
            "all 5 rels first-seen (incl. the parallel + cycle-closing)"
        );
        let n4 = r.nodes.iter().find(|n| n.node.raw() == 4).expect("node 4");
        assert_eq!(n4.depth, 2, "min-hop retention");
    }

    #[test]
    fn limit_pushdown_stops_immediately() {
        let g = chain(100);
        let mut rq = req(50);
        rq.limit = 3;
        let r = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        assert_eq!(r.nodes.len(), 3);
        assert_eq!(r.truncation, Truncation::LimitReached);
    }

    #[test]
    fn frontier_cap_degrades_and_continues() {
        // Star: 1 -> {2..=11}; spokes 2 -> 12, 3 -> 13.
        let mut g = MemoryEdgeSource::new();
        for i in 2..=11 {
            g.add_edge(NodeId::new(1), NodeId::new(i), None);
        }
        g.add_edge(NodeId::new(2), NodeId::new(12), None);
        g.add_edge(NodeId::new(3), NodeId::new(13), None);
        let mut rq = req(2);
        rq.per_hop_frontier_cap = Some(4);
        let r = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        assert_eq!(r.truncation, Truncation::FrontierCapped);
        let hop1 = r.nodes.iter().filter(|n| n.depth == 1).count();
        assert_eq!(hop1, 4, "hop-1 capped at 4");
        // Source order retains 2,3,4,5 at hop 1 → both spokes reachable.
        let hop2 = r.nodes.iter().filter(|n| n.depth == 2).count();
        assert_eq!(hop2, 2, "traversal continued past the capped hop");
    }

    #[test]
    fn reservoir_is_deterministic_and_respects_per_hop_cap() {
        let mut g = MemoryEdgeSource::new();
        for i in 2..=101 {
            g.add_edge(NodeId::new(1), NodeId::new(i), None);
        }
        let mut rq = req(1);
        rq.sampling = SamplingStrategy::ReservoirVitterL;
        rq.per_hop_frontier_cap = Some(10);
        rq.seed = 99;
        let a = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        let b = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        let ids =
            |r: &KHopResult<(), ()>| -> Vec<u64> { r.nodes.iter().map(|n| n.node.raw()).collect() };
        assert_eq!(ids(&a), ids(&b), "same seed reproduces the sample");
        assert_eq!(a.nodes.len(), 11, "anchor + 10 sampled");
        assert_eq!(a.truncation, Truncation::FrontierCapped);

        rq.seed = 100;
        let c = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        assert_ne!(ids(&a), ids(&c), "different seed differs (w.h.p.)");
    }

    #[test]
    fn reservoir_downweights_hub_candidates() {
        // Anchor 1 -> hub 2 and -> leaf 3 (hop 1, both always taken: the
        // hop-1 frontier is just the anchor and capacity is unbound).
        // Hop 2: hub 2 has 100 children (each weight 1/100); leaf 3 has
        // one child 200 (weight 1). With capacity 10, child 200 must be
        // sampled essentially always; an unweighted sampler would drop it
        // ~90% of the time. Run across seeds to falsify unweighted.
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.add_edge(NodeId::new(1), NodeId::new(3), None);
        for i in 0..100u64 {
            g.add_edge(NodeId::new(2), NodeId::new(1000 + i), None);
        }
        g.add_edge(NodeId::new(3), NodeId::new(200), None);
        let mut kept = 0u32;
        const TRIALS: u64 = 200;
        for seed in 0..TRIALS {
            let mut rq = req(2);
            rq.sampling = SamplingStrategy::ReservoirVitterL;
            rq.per_hop_frontier_cap = Some(10);
            rq.seed = seed;
            let r = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
            if r.nodes.iter().any(|n| n.node.raw() == 200) {
                kept += 1;
            }
        }
        // Weighted: P(keep 200) ≈ 1 (weight 1 vs hub children at 0.01,
        // 10 slots). Unweighted: ≈ 10/101 ≈ 0.099 → ~20/200. The 150
        // floor cleanly separates the two hypotheses.
        assert!(
            kept > 150,
            "degree-aware sampling must keep the low-degree branch: kept {kept}/{TRIALS}"
        );
    }

    #[test]
    fn zero_limit_and_zero_cap_are_structured_errors() {
        let g = chain(3);
        let mut rq = req(1);
        rq.limit = 0;
        assert!(matches!(
            k_hop(&g, NodeId::new(1), (), 0, &rq),
            Err(TraversalError::InvalidRequest { .. })
        ));
        let mut rq2 = req(1);
        rq2.per_hop_frontier_cap = Some(0);
        assert!(matches!(
            k_hop(&g, NodeId::new(1), (), 0, &rq2),
            Err(TraversalError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn rel_type_filter_pushes_down() {
        use arcgraph_core::TypeId;
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), Some(TypeId::new(7)));
        g.add_edge(NodeId::new(1), NodeId::new(3), Some(TypeId::new(8)));
        let mut rq = req(1);
        rq.filter = EdgeFilter::rel_type(TypeId::new(7));
        let r = k_hop(&g, NodeId::new(1), (), 0, &rq).expect("traversal");
        let ids: Vec<u64> = r.nodes.iter().map(|n| n.node.raw()).collect();
        assert_eq!(ids, vec![1, 2]);
    }
}
