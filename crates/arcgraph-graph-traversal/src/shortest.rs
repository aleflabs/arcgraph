//! Unweighted shortest path: meet-in-the-middle bidirectional BFS
//! (ADR-036 §D-4; implementable since ADR-131 gave the substrate real
//! inbound rows).
//!
//! Budget (PD#5): each side explores O(b^(d/2)) nodes (b = branching,
//! d = path length) vs O(b^d) single-sided — the classic bidirectional
//! win. Meet detection is O(1) per discovered node (hash probe into the
//! other side's parent map). Memory: two parent maps + two frontiers.
//!
//! Correctness discipline: levels expand WHOLE (the smaller frontier
//! first), and all meets discovered during a level are collected before
//! choosing the minimum-total — avoiding the classic per-node
//! early-exit off-by-one. The property suite pins the oracle
//! `bidirectional length == unidirectional BFS distance` on random
//! graphs.

use std::collections::HashMap;

use arcgraph_core::{NodeId, RelId};

use crate::error::TraversalError;
use crate::source::{EdgeFilter, EdgeSource, TraversalDirection};

/// A reconstructed path. `nodes.len() == rels.len() + 1`; `nodes[0]` is
/// the requested `src` and `nodes.last()` the requested `dst`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathResult {
    /// Node sequence from `src` to `dst` inclusive.
    pub nodes: Vec<NodeId>,
    /// Relationship traversed between each consecutive node pair.
    pub rels: Vec<RelId>,
}

impl PathResult {
    /// Path length in hops.
    #[must_use]
    pub fn hops(&self) -> usize {
        self.rels.len()
    }
}

/// Parent link: how a node was first discovered on one search side.
type ParentMap = HashMap<NodeId, Option<(NodeId, RelId)>>;

/// Expand one whole BFS level on one side. Returns the next frontier;
/// records first-discovery parents; collects meets against the OTHER
/// side's parent map.
#[allow(clippy::too_many_arguments)] // 7 args reflect the level contract: source, frontier,
// orientation, filter, own/other parent maps, meet sink. Bundling into a struct adds
// indirection without simplifying the two call sites (fwd + bwd), which already pass each
// side's pieces directly — the `collect_reachable` 8-arg precedent.
fn expand_level<S: EdgeSource + ?Sized>(
    source: &S,
    frontier: &[NodeId],
    direction: TraversalDirection,
    filter: &EdgeFilter,
    parents: &mut ParentMap,
    other_parents: &ParentMap,
    meets: &mut Vec<NodeId>,
) -> Result<Vec<NodeId>, TraversalError<S::Error>> {
    let mut next = Vec::new();
    for &u in frontier {
        let batch = source
            .neighbors(u, direction, filter)
            .map_err(TraversalError::Source)?;
        for nb in batch {
            if parents.contains_key(&nb.dst) {
                continue; // already discovered on this side at <= depth
            }
            parents.insert(nb.dst, Some((u, nb.rel_id)));
            if other_parents.contains_key(&nb.dst) {
                meets.push(nb.dst);
            }
            next.push(nb.dst);
        }
    }
    Ok(next)
}

/// Walk a parent chain back to its root. Returns `(nodes, rels)` ordered
/// root → start-of-walk... reversed by the caller as needed.
fn chain_to_root(parents: &ParentMap, from: NodeId) -> (Vec<NodeId>, Vec<RelId>) {
    let mut nodes = vec![from];
    let mut rels = Vec::new();
    let mut cur = from;
    while let Some(Some((parent, rel))) = parents.get(&cur) {
        nodes.push(*parent);
        rels.push(*rel);
        cur = *parent;
    }
    (nodes, rels)
}

/// Depth of `node` on a side (root = 0). The parent maps don't store
/// depth; chain length is O(d) which is fine for reconstruction-sized d.
fn depth_of(parents: &ParentMap, node: NodeId) -> usize {
    let (nodes, _) = chain_to_root(parents, node);
    nodes.len() - 1
}

/// Meet-in-the-middle unweighted shortest path from `src` to `dst`
/// following `direction`-oriented edges (`Outbound` = a directed
/// `src → … → dst` path; `Undirected` = orientation-blind; `Inbound` = a
/// directed `dst → … → src` path expressed from `src`'s perspective).
///
/// Returns `None` when no path exists within `max_hops`.
pub fn bidirectional_shortest<S: EdgeSource + ?Sized>(
    source: &S,
    src: NodeId,
    dst: NodeId,
    max_hops: u32,
    direction: TraversalDirection,
    filter: &EdgeFilter,
) -> Result<Option<PathResult>, TraversalError<S::Error>> {
    if src == dst {
        return Ok(Some(PathResult {
            nodes: vec![src],
            rels: Vec::new(),
        }));
    }
    if max_hops == 0 {
        return Ok(None);
    }
    tracing::debug!(
        src = src.raw(),
        dst = dst.raw(),
        max_hops,
        "bidirectional_shortest"
    );

    let mut fwd_parents: ParentMap = HashMap::from([(src, None)]);
    let mut bwd_parents: ParentMap = HashMap::from([(dst, None)]);
    let mut fwd_frontier = vec![src];
    let mut bwd_frontier = vec![dst];
    let mut expanded_levels: u32 = 0;

    while !fwd_frontier.is_empty() && !bwd_frontier.is_empty() && expanded_levels < max_hops {
        let mut meets: Vec<NodeId> = Vec::new();
        // Expand the cheaper side (classic balancing heuristic).
        if fwd_frontier.len() <= bwd_frontier.len() {
            fwd_frontier = expand_level(
                source,
                &fwd_frontier,
                direction,
                filter,
                &mut fwd_parents,
                &bwd_parents,
                &mut meets,
            )?;
        } else {
            bwd_frontier = expand_level(
                source,
                &bwd_frontier,
                direction.inverse(),
                filter,
                &mut bwd_parents,
                &fwd_parents,
                &mut meets,
            )?;
        }
        expanded_levels += 1;

        if !meets.is_empty() {
            // Deterministic best meet: minimal total length, then NodeId.
            let best = meets
                .into_iter()
                .map(|m| {
                    let total = depth_of(&fwd_parents, m) + depth_of(&bwd_parents, m);
                    (total, m)
                })
                .min_by_key(|&(total, m)| (total, m.raw()))
                .expect("non-empty meets has a minimum");
            let (total, meet) = best;
            if total as u32 > max_hops {
                return Ok(None);
            }
            // Forward half: chain meet → src, reversed to src → meet.
            let (mut f_nodes, mut f_rels) = chain_to_root(&fwd_parents, meet);
            f_nodes.reverse();
            f_rels.reverse();
            // Backward half: chain meet → dst is ALREADY in walk order
            // (bwd parents point toward dst).
            let (b_nodes, b_rels) = chain_to_root(&bwd_parents, meet);
            // Stitch: f_nodes ends at meet; b_nodes starts at meet.
            let mut nodes = f_nodes;
            nodes.extend_from_slice(&b_nodes[1..]);
            let mut rels = f_rels;
            rels.extend_from_slice(&b_rels);
            debug_assert_eq!(nodes.len(), rels.len() + 1);
            debug_assert_eq!(rels.len(), total);
            return Ok(Some(PathResult { nodes, rels }));
        }
    }
    Ok(None)
}

/// Plain unidirectional BFS shortest path honoring exclusion sets — the
/// Yen subroutine (ADR-205 §D-4: no source mutation; bans are O(|banned|)
/// hash sets). Also the reference oracle for the bidirectional property
/// test (exposed `pub(crate)` for the test + yen modules; deliberately
/// not public API — `bidirectional_shortest` is the product surface).
#[allow(clippy::too_many_arguments)] // 8 args reflect the constrained-BFS contract (Yen
// threads ban sets per spur root); a request struct would be rebuilt per spur iteration
// for no call-site simplification — the `collect_reachable` 8-arg precedent.
pub(crate) fn constrained_shortest<S: EdgeSource + ?Sized>(
    source: &S,
    src: NodeId,
    dst: NodeId,
    max_hops: u32,
    direction: TraversalDirection,
    filter: &EdgeFilter,
    banned_nodes: &std::collections::HashSet<NodeId>,
    banned_edges: &std::collections::HashSet<RelId>,
) -> Result<Option<PathResult>, TraversalError<S::Error>> {
    if banned_nodes.contains(&src) || banned_nodes.contains(&dst) {
        return Ok(None);
    }
    if src == dst {
        return Ok(Some(PathResult {
            nodes: vec![src],
            rels: Vec::new(),
        }));
    }
    let mut parents: ParentMap = HashMap::from([(src, None)]);
    let mut frontier = vec![src];
    for _ in 0..max_hops {
        if frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for &u in &frontier {
            let batch = source
                .neighbors(u, direction, filter)
                .map_err(TraversalError::Source)?;
            for nb in batch {
                if banned_edges.contains(&nb.rel_id)
                    || banned_nodes.contains(&nb.dst)
                    || parents.contains_key(&nb.dst)
                {
                    continue;
                }
                parents.insert(nb.dst, Some((u, nb.rel_id)));
                if nb.dst == dst {
                    let (mut nodes, mut rels) = chain_to_root(&parents, dst);
                    nodes.reverse();
                    rels.reverse();
                    return Ok(Some(PathResult { nodes, rels }));
                }
                next.push(nb.dst);
            }
        }
        frontier = next;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEdgeSource;

    fn grid() -> MemoryEdgeSource {
        // 1 → 2 → 3
        // ↓         ↘
        // 4 → 5 → 6 → 7   (two routes 1..7: 1-2-3-7 and 1-4-5-6-7)
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.add_edge(NodeId::new(2), NodeId::new(3), None);
        g.add_edge(NodeId::new(3), NodeId::new(7), None);
        g.add_edge(NodeId::new(1), NodeId::new(4), None);
        g.add_edge(NodeId::new(4), NodeId::new(5), None);
        g.add_edge(NodeId::new(5), NodeId::new(6), None);
        g.add_edge(NodeId::new(6), NodeId::new(7), None);
        g
    }

    #[test]
    fn finds_the_shorter_of_two_routes() {
        let g = grid();
        let p = bidirectional_shortest(
            &g,
            NodeId::new(1),
            NodeId::new(7),
            10,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal")
        .expect("path exists");
        assert_eq!(p.hops(), 3);
        let ids: Vec<u64> = p.nodes.iter().map(|n| n.raw()).collect();
        assert_eq!(ids, vec![1, 2, 3, 7]);
    }

    #[test]
    fn respects_max_hops() {
        let g = grid();
        let p = bidirectional_shortest(
            &g,
            NodeId::new(1),
            NodeId::new(7),
            2,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert!(p.is_none(), "shortest route needs 3 hops; cap is 2");
    }

    #[test]
    fn directed_dst_to_src_is_none_but_undirected_connects() {
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        let directed = bidirectional_shortest(
            &g,
            NodeId::new(2),
            NodeId::new(1),
            5,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert!(directed.is_none(), "no directed 2→1 path");
        let undirected = bidirectional_shortest(
            &g,
            NodeId::new(2),
            NodeId::new(1),
            5,
            TraversalDirection::Undirected,
            &EdgeFilter::any(),
        )
        .expect("traversal")
        .expect("undirected path");
        assert_eq!(undirected.hops(), 1);
    }

    #[test]
    fn src_equals_dst_is_the_empty_path() {
        let g = grid();
        let p = bidirectional_shortest(
            &g,
            NodeId::new(1),
            NodeId::new(1),
            0,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal")
        .expect("trivial path");
        assert_eq!(p.hops(), 0);
        assert_eq!(p.nodes, vec![NodeId::new(1)]);
    }

    #[test]
    fn constrained_bans_reroute() {
        let g = grid();
        // Ban node 2 → forced onto the long route.
        let banned_nodes = std::collections::HashSet::from([NodeId::new(2)]);
        let p = constrained_shortest(
            &g,
            NodeId::new(1),
            NodeId::new(7),
            10,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
            &banned_nodes,
            &std::collections::HashSet::new(),
        )
        .expect("traversal")
        .expect("long route exists");
        assert_eq!(p.hops(), 4);
    }
}
