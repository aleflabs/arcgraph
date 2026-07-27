//! Local-only multi-source `k_hop` result merge.
//!
//! Budget (PD#5): O(Σ|parts| log Σ|parts|) for the deterministic sort;
//! hash-probe dedupe per node/edge.

use std::collections::{HashMap, HashSet};

use arcgraph_core::RelId;

use crate::khop::{KHopEdge, KHopNode, KHopResult, Truncation};

/// Merge per-source [`KHopResult`]s into one: min-depth dedupe by node id,
/// `RelId` dedupe for edges, deterministic `(depth, NodeId)` order,
/// truncation to `limit` retained nodes. Every returned edge's endpoints
/// are present in the returned node set.
///
/// `truncation` of the merged result is the strongest of (a) any part's
/// own truncation, (b) a `LimitReached` introduced by the merge cap.
/// `cost_consumed` sums the parts (each part charged its own budget when
/// it ran; the merge itself charges nothing).
#[must_use]
pub fn merge_top_k<N: Clone, E: Clone>(
    parts: Vec<KHopResult<N, E>>,
    limit: usize,
) -> KHopResult<N, E> {
    let mut cost_consumed: u64 = 0;
    let mut truncation = Truncation::None;
    // First sighting wins for payloads; depth is minimized across parts.
    let mut best: HashMap<u64, KHopNode<N>> = HashMap::new();
    let mut edges: Vec<KHopEdge<E>> = Vec::new();
    let mut seen_edges: HashMap<RelId, ()> = HashMap::new();

    for part in parts {
        cost_consumed = cost_consumed.saturating_add(part.cost_consumed);
        truncation = strongest(truncation, part.truncation);
        for node in part.nodes {
            match best.get_mut(&node.node.raw()) {
                Some(existing) if existing.depth <= node.depth => {}
                Some(existing) => *existing = node,
                None => {
                    best.insert(node.node.raw(), node);
                }
            }
        }
        for edge in part.edges {
            if seen_edges.insert(edge.rel_id, ()).is_none() {
                edges.push(edge);
            }
        }
    }

    let mut nodes: Vec<KHopNode<N>> = best.into_values().collect();
    nodes.sort_by_key(|n| (n.depth, n.node.raw()));
    if nodes.len() > limit {
        nodes.truncate(limit);
        truncation = strongest(truncation, Truncation::LimitReached);
    }
    let retained_nodes: HashSet<_> = nodes.iter().map(|n| n.node).collect();
    // The node cap is authoritative: over-limit merges can reduce the
    // edge count because edges to truncated-away endpoints are invalid.
    edges.retain(|edge| retained_nodes.contains(&edge.src) && retained_nodes.contains(&edge.dst));

    KHopResult {
        nodes,
        edges,
        truncation,
        cost_consumed,
    }
}

/// Truncation severity order: `LimitReached > FrontierCapped > None`
/// (report the strongest binding constraint).
fn strongest(a: Truncation, b: Truncation) -> Truncation {
    fn rank(t: Truncation) -> u8 {
        // Deliberately NO wildcard arm: `Truncation` is `#[non_exhaustive]`
        // for downstream crates, but in-crate exhaustive matching means a
        // future variant FAILS COMPILATION here until it is ranked —
        // a merge can never silently under-report a new truncation class.
        match t {
            Truncation::None => 0,
            Truncation::FrontierCapped => 1,
            Truncation::LimitReached => 2,
        }
    }
    if rank(b) > rank(a) { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::NodeId;
    use proptest::prelude::*;

    fn node(id: u64, depth: u32) -> KHopNode<()> {
        KHopNode {
            node: NodeId::new(id),
            depth,
            data: (),
        }
    }

    fn edge(id: u64, src: u64, dst: u64) -> KHopEdge<()> {
        KHopEdge {
            rel_id: RelId::new(id),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            data: (),
        }
    }

    fn retained_ids(r: &KHopResult<(), ()>) -> HashSet<NodeId> {
        r.nodes.iter().map(|n| n.node).collect()
    }

    fn assert_endpoint_closure(r: &KHopResult<(), ()>) {
        let retained = retained_ids(r);
        assert!(
            r.edges
                .iter()
                .all(|e| retained.contains(&e.src) && retained.contains(&e.dst)),
            "every edge endpoint must be present in retained nodes"
        );
    }

    fn edge_ids(r: &KHopResult<(), ()>) -> Vec<u64> {
        r.edges.iter().map(|e| e.rel_id.raw()).collect()
    }

    fn arb_part() -> impl Strategy<Value = KHopResult<(), ()>> {
        (
            proptest::collection::vec((1u64..16, 0u32..5), 0..12),
            proptest::collection::vec((1u64..64, 1u64..16, 1u64..16), 0..24),
        )
            .prop_map(|(nodes, edges)| KHopResult {
                nodes: nodes
                    .into_iter()
                    .map(|(id, depth)| node(id, depth))
                    .collect(),
                edges: edges
                    .into_iter()
                    .map(|(rel, src, dst)| edge(rel, src, dst))
                    .collect(),
                truncation: Truncation::None,
                cost_consumed: 0,
            })
    }

    #[test]
    fn merges_min_depth_dedupes_edges_and_truncates_deterministically() {
        let a = KHopResult {
            nodes: vec![node(1, 0), node(2, 1), node(3, 2)],
            edges: vec![edge(10, 1, 2), edge(11, 2, 3)],
            truncation: Truncation::None,
            cost_consumed: 5,
        };
        let b = KHopResult {
            nodes: vec![node(3, 1), node(4, 1)],
            edges: vec![edge(11, 2, 3), edge(12, 3, 4)],
            truncation: Truncation::FrontierCapped,
            cost_consumed: 7,
        };
        let merged = merge_top_k(vec![a, b], 3);
        let got: Vec<(u64, u32)> = merged
            .nodes
            .iter()
            .map(|n| (n.node.raw(), n.depth))
            .collect();
        // node 3 takes min depth 1; order (depth, id); limit 3 drops node 3? No:
        // sorted = (1,0),(2,1),(3,1),(4,1) → truncate to 3 keeps 1,2,3.
        assert_eq!(got, vec![(1, 0), (2, 1), (3, 1)]);
        assert_endpoint_closure(&merged);
        assert_eq!(
            edge_ids(&merged),
            vec![10, 11],
            "edge 11 deduped; edge 12 is dropped with truncated node 4"
        );
        assert_eq!(merged.cost_consumed, 12);
        assert_eq!(
            merged.truncation,
            Truncation::LimitReached,
            "merge cap is the strongest"
        );
    }

    #[test]
    fn empty_parts_merge_to_empty() {
        let merged: KHopResult<(), ()> = merge_top_k(vec![], 10);
        assert!(merged.nodes.is_empty() && merged.edges.is_empty());
        assert_eq!(merged.truncation, Truncation::None);
    }

    #[test]
    fn truncation_filters_edges_to_retained_endpoints_issue_1107() {
        let a = KHopResult {
            nodes: vec![node(1, 0), node(2, 1), node(3, 2)],
            edges: vec![edge(10, 1, 2), edge(11, 2, 3)],
            truncation: Truncation::None,
            cost_consumed: 0,
        };
        let b = KHopResult {
            nodes: vec![node(3, 1), node(4, 2)],
            edges: vec![edge(11, 2, 3), edge(12, 3, 4)],
            truncation: Truncation::None,
            cost_consumed: 0,
        };

        let merged = merge_top_k(vec![a, b], 3);

        assert_eq!(
            merged
                .nodes
                .iter()
                .map(|n| n.node.raw())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_endpoint_closure(&merged);
        assert_eq!(edge_ids(&merged), vec![10, 11]);
    }

    #[test]
    fn under_limit_merge_keeps_all_edges() {
        let a = KHopResult {
            nodes: vec![node(1, 0), node(2, 1)],
            edges: vec![edge(10, 1, 2)],
            truncation: Truncation::None,
            cost_consumed: 1,
        };
        let b = KHopResult {
            nodes: vec![node(3, 1), node(4, 2)],
            edges: vec![edge(11, 2, 3), edge(12, 3, 4)],
            truncation: Truncation::None,
            cost_consumed: 2,
        };

        let merged = merge_top_k(vec![a, b], 4);

        assert_endpoint_closure(&merged);
        assert_eq!(edge_ids(&merged), vec![10, 11, 12]);
        assert_eq!(merged.truncation, Truncation::None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn merge_top_k_preserves_endpoint_closure(
            parts in proptest::collection::vec(arb_part(), 0..8),
            limit in 0usize..16,
        ) {
            let merged = merge_top_k(parts, limit);
            let retained = retained_ids(&merged);

            for edge in &merged.edges {
                prop_assert!(retained.contains(&edge.src));
                prop_assert!(retained.contains(&edge.dst));
            }
        }
    }
}
