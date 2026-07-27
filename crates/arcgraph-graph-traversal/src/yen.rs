//! Yen's k shortest loopless paths (Yen 1971, *Finding the K Shortest
//! Loopless Paths in a Network*, Management Science 17(11)) over the
//! unweighted hop-count metric — the ADR-036 §D-4 "path-as-evidence"
//! primitive at v1.1.
//!
//! Budget (PD#5): at most `k · L` constrained-BFS runs (L = longest
//! accepted path length, bounded by `max_hops`); each run is a plain BFS
//! over the filtered graph with O(|banned|) hash-set overhead. Intended
//! for evidence-sized k (≤ ~10), not bulk analytics — the back-of-envelope
//! in ADR-205 §Back-of-envelope binds this.
//!
//! Determinism: candidate paths order by `(hops, node-sequence
//! lexicographic)`; the underlying BFS inherits the source's batch order.
//! Looplessness is by construction (root-path nodes are banned from each
//! spur search).

use std::collections::HashSet;

use arcgraph_core::{NodeId, RelId};

use crate::error::TraversalError;
use crate::shortest::{PathResult, constrained_shortest};
use crate::source::{EdgeFilter, EdgeSource, TraversalDirection};

/// Lexicographic key for deterministic candidate ordering.
fn path_key(p: &PathResult) -> (usize, Vec<u64>) {
    (p.hops(), p.nodes.iter().map(|n| n.raw()).collect())
}

/// Yen's k shortest loopless paths from `src` to `dst`. Returns up to `k`
/// paths sorted by `(hops, node-sequence)`; fewer when the graph runs out
/// of distinct loopless routes within `max_hops`. `k == 0` is a
/// structured [`TraversalError::InvalidRequest`].
///
/// Prefer calling through [`crate::GraphTraversalHandle::k_shortest_paths`].
#[allow(clippy::too_many_arguments)] // 7 args mirror the roadmap-bound handle method
// (src, dst, k, max_hops, direction, filter); a request struct would diverge the free fn
// from the V11-S-04 vocabulary for one arg of savings.
pub fn k_shortest_paths<S: EdgeSource + ?Sized>(
    source: &S,
    src: NodeId,
    dst: NodeId,
    k: usize,
    max_hops: u32,
    direction: TraversalDirection,
    filter: &EdgeFilter,
) -> Result<Vec<PathResult>, TraversalError<S::Error>> {
    if k == 0 {
        return Err(TraversalError::InvalidRequest {
            reason: "k_shortest_paths requires k >= 1",
        });
    }
    tracing::debug!(
        src = src.raw(),
        dst = dst.raw(),
        k,
        max_hops,
        "k_shortest_paths"
    );

    let no_nodes: HashSet<NodeId> = HashSet::new();
    let no_edges: HashSet<RelId> = HashSet::new();
    let Some(first) = constrained_shortest(
        source, src, dst, max_hops, direction, filter, &no_nodes, &no_edges,
    )?
    else {
        return Ok(Vec::new());
    };

    let mut accepted: Vec<PathResult> = vec![first];
    // Candidate pool B (Yen's notation): kept sorted-on-pop; deduped by
    // node sequence so the same deviation found from two spur roots is
    // considered once.
    let mut candidates: Vec<PathResult> = Vec::new();

    while accepted.len() < k {
        let prev = accepted
            .last()
            .expect("accepted is non-empty after the first path")
            .clone();

        // Each node of the previous path except its terminal is a spur
        // root; deviate after the shared root prefix.
        for spur_idx in 0..prev.rels.len() {
            let spur_node = prev.nodes[spur_idx];
            let root_nodes = &prev.nodes[..=spur_idx];
            let root_rels = &prev.rels[..spur_idx];

            // Ban every accepted path's NEXT edge where it shares this
            // exact root prefix — forcing the spur to deviate.
            let mut banned_edges: HashSet<RelId> = HashSet::new();
            for p in &accepted {
                if p.nodes.len() > spur_idx && p.nodes[..=spur_idx] == *root_nodes {
                    if let Some(&rel) = p.rels.get(spur_idx) {
                        banned_edges.insert(rel);
                    }
                }
            }
            // Looplessness: the root's interior nodes may not reappear in
            // the spur (the spur node itself stays usable as its start).
            let banned_nodes: HashSet<NodeId> = root_nodes[..spur_idx].iter().copied().collect();

            let remaining_hops = max_hops.saturating_sub(spur_idx as u32);
            if remaining_hops == 0 {
                continue;
            }
            if let Some(spur) = constrained_shortest(
                source,
                spur_node,
                dst,
                remaining_hops,
                direction,
                filter,
                &banned_nodes,
                &banned_edges,
            )? {
                // Total = root prefix + spur (spur.nodes[0] == spur_node,
                // shared with the root's tail — skip it on extend).
                let mut nodes = root_nodes.to_vec();
                nodes.extend_from_slice(&spur.nodes[1..]);
                let mut rels = root_rels.to_vec();
                rels.extend_from_slice(&spur.rels);
                let cand = PathResult { nodes, rels };
                let dup = accepted
                    .iter()
                    .chain(candidates.iter())
                    .any(|p| p.nodes == cand.nodes);
                if !dup {
                    candidates.push(cand);
                }
            }
        }

        if candidates.is_empty() {
            break; // the graph is out of distinct loopless routes
        }
        candidates.sort_by_key(path_key);
        accepted.push(candidates.remove(0));
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEdgeSource;

    /// Two short routes + one long route between 1 and 5.
    fn tri_route() -> MemoryEdgeSource {
        let mut g = MemoryEdgeSource::new();
        // Route A (2 hops): 1 → 2 → 5
        g.add_edge(NodeId::new(1), NodeId::new(2), None);
        g.add_edge(NodeId::new(2), NodeId::new(5), None);
        // Route B (2 hops): 1 → 3 → 5
        g.add_edge(NodeId::new(1), NodeId::new(3), None);
        g.add_edge(NodeId::new(3), NodeId::new(5), None);
        // Route C (3 hops): 1 → 4 → 6 → 5
        g.add_edge(NodeId::new(1), NodeId::new(4), None);
        g.add_edge(NodeId::new(4), NodeId::new(6), None);
        g.add_edge(NodeId::new(6), NodeId::new(5), None);
        g
    }

    fn ids(p: &PathResult) -> Vec<u64> {
        p.nodes.iter().map(|n| n.raw()).collect()
    }

    #[test]
    fn returns_k_distinct_loopless_paths_sorted_by_length() {
        let g = tri_route();
        let paths = k_shortest_paths(
            &g,
            NodeId::new(1),
            NodeId::new(5),
            3,
            10,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert_eq!(paths.len(), 3);
        assert_eq!(
            ids(&paths[0]),
            vec![1, 2, 5],
            "shortest, lexicographic first"
        );
        assert_eq!(ids(&paths[1]), vec![1, 3, 5]);
        assert_eq!(ids(&paths[2]), vec![1, 4, 6, 5]);
        // Loopless + distinct invariants.
        for p in &paths {
            let mut seen = std::collections::HashSet::new();
            assert!(p.nodes.iter().all(|n| seen.insert(*n)), "loopless");
        }
    }

    #[test]
    fn exhausts_gracefully_when_fewer_than_k_routes_exist() {
        let g = tri_route();
        let paths = k_shortest_paths(
            &g,
            NodeId::new(1),
            NodeId::new(5),
            10,
            10,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert_eq!(paths.len(), 3, "only 3 distinct loopless routes exist");
    }

    #[test]
    fn no_path_yields_empty_and_k_zero_is_an_error() {
        let g = tri_route();
        let none = k_shortest_paths(
            &g,
            NodeId::new(5),
            NodeId::new(1),
            2,
            10,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert!(none.is_empty(), "no directed 5→1 route");
        assert!(matches!(
            k_shortest_paths(
                &g,
                NodeId::new(1),
                NodeId::new(5),
                0,
                10,
                TraversalDirection::Outbound,
                &EdgeFilter::any(),
            ),
            Err(TraversalError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn max_hops_excludes_long_routes() {
        let g = tri_route();
        let paths = k_shortest_paths(
            &g,
            NodeId::new(1),
            NodeId::new(5),
            3,
            2,
            TraversalDirection::Outbound,
            &EdgeFilter::any(),
        )
        .expect("traversal");
        assert_eq!(paths.len(), 2, "the 3-hop route C exceeds max_hops=2");
    }
}
