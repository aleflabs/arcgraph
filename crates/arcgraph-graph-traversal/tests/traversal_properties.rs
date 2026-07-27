//! Property suite for the V11-S-04 traversal invariants (ADR-205):
//!
//! - `k_hop` BFS depths equal an independently-implemented reference BFS.
//! - `bidirectional_shortest` length equals the reference BFS distance
//!   (the meet-in-the-middle correctness oracle).
//! - Yen: sorted by hops, pairwise-distinct, loopless, first == shortest.
//! - Reservoir mode: deterministic under a fixed seed; capacity respected;
//!   retained set ⊆ true k-hop ball.

use std::collections::{HashMap, HashSet, VecDeque};

use arcgraph_core::NodeId;
use arcgraph_graph_traversal::{
    EdgeFilter, EdgeSource, GraphTraversalHandle, KHopRequest, MemoryEdgeSource, SamplingStrategy,
    TraversalDirection,
};
use proptest::prelude::*;

/// Reference single-source BFS over the same `EdgeSource` — written
/// independently of the crate's traversal internals (the oracle).
fn reference_bfs_depths(
    g: &MemoryEdgeSource,
    src: NodeId,
    k: u32,
    direction: TraversalDirection,
) -> HashMap<u64, u32> {
    let mut depth: HashMap<u64, u32> = HashMap::from([(src.raw(), 0)]);
    let mut q: VecDeque<(NodeId, u32)> = VecDeque::from([(src, 0)]);
    while let Some((u, d)) = q.pop_front() {
        if d >= k {
            continue;
        }
        for nb in g
            .neighbors(u, direction, &EdgeFilter::any())
            .expect("infallible")
        {
            if let std::collections::hash_map::Entry::Vacant(slot) = depth.entry(nb.dst.raw()) {
                slot.insert(d + 1);
                q.push_back((nb.dst, d + 1));
            }
        }
    }
    depth
}

/// Random small digraph: `n` nodes (ids 1..=n), arbitrary edge pairs.
fn arb_graph() -> impl Strategy<Value = (MemoryEdgeSource, u64)> {
    (2u64..24).prop_flat_map(|n| {
        proptest::collection::vec((1..=n, 1..=n), 0..96).prop_map(move |pairs| {
            let mut g = MemoryEdgeSource::new();
            for (a, b) in pairs {
                g.add_edge(NodeId::new(a), NodeId::new(b), None);
            }
            (g, n)
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn khop_bfs_depths_match_reference((g, n) in arb_graph(), k in 0u32..5) {
        let src = NodeId::new(1);
        let oracle = reference_bfs_depths(&g, src, k, TraversalDirection::Outbound);
        let handle = GraphTraversalHandle::new(&g);
        let r = handle
            .k_hop(src, (), 0, &KHopRequest::new(k, TraversalDirection::Outbound))
            .expect("traversal");
        let got: HashMap<u64, u32> = r.nodes.iter().map(|x| (x.node.raw(), x.depth)).collect();
        prop_assert_eq!(got, oracle);
        prop_assert!(n >= 2);
    }

    #[test]
    fn bidirectional_length_matches_reference_distance(
        (g, n) in arb_graph(),
        direction in prop_oneof![
            Just(TraversalDirection::Outbound),
            Just(TraversalDirection::Undirected),
        ],
    ) {
        let src = NodeId::new(1);
        let dst = NodeId::new(n);
        let max_hops = 32;
        let oracle = reference_bfs_depths(&g, src, max_hops, direction)
            .get(&dst.raw())
            .copied();
        let handle = GraphTraversalHandle::new(&g);
        let got = handle
            .bidirectional_shortest(src, dst, max_hops, direction, &EdgeFilter::any())
            .expect("traversal")
            .map(|p| p.hops() as u32);
        prop_assert_eq!(got, oracle);
    }

    #[test]
    fn yen_paths_are_sorted_distinct_loopless_and_first_is_shortest((g, n) in arb_graph()) {
        let src = NodeId::new(1);
        let dst = NodeId::new(n);
        let handle = GraphTraversalHandle::new(&g);
        let paths = handle
            .k_shortest_paths(src, dst, 4, 16, TraversalDirection::Outbound, &EdgeFilter::any())
            .expect("traversal");
        // Sorted by hop count.
        prop_assert!(paths.windows(2).all(|w| w[0].hops() <= w[1].hops()));
        // Pairwise distinct node sequences.
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                prop_assert_ne!(&paths[i].nodes, &paths[j].nodes);
            }
        }
        // Loopless; endpoints correct; node/rel arity.
        for p in &paths {
            let mut seen = HashSet::new();
            prop_assert!(p.nodes.iter().all(|x| seen.insert(*x)));
            prop_assert_eq!(p.nodes.len(), p.rels.len() + 1);
            prop_assert_eq!(*p.nodes.first().expect("non-empty"), src);
            prop_assert_eq!(*p.nodes.last().expect("non-empty"), dst);
        }
        // First path is a true shortest path.
        let oracle = reference_bfs_depths(&g, src, 16, TraversalDirection::Outbound)
            .get(&dst.raw())
            .copied();
        prop_assert_eq!(paths.first().map(|p| p.hops() as u32), oracle);
    }

    #[test]
    fn reservoir_is_deterministic_capacity_bounded_and_within_the_ball(
        (g, _n) in arb_graph(),
        seed in any::<u64>(),
        cap in 1usize..6,
        k in 1u32..4,
    ) {
        let src = NodeId::new(1);
        let mut req = KHopRequest::new(k, TraversalDirection::Outbound);
        req.sampling = SamplingStrategy::ReservoirVitterL;
        req.per_hop_frontier_cap = Some(cap);
        req.seed = seed;
        let handle = GraphTraversalHandle::new(&g);
        let a = handle.k_hop(src, (), 0, &req).expect("traversal");
        let b = handle.k_hop(src, (), 0, &req).expect("traversal");
        let ids = |r: &arcgraph_graph_traversal::KHopResult<(), ()>| -> Vec<u64> {
            r.nodes.iter().map(|x| x.node.raw()).collect()
        };
        prop_assert_eq!(ids(&a), ids(&b), "fixed seed must reproduce");
        // Per-hop capacity respected.
        let mut per_hop: HashMap<u32, usize> = HashMap::new();
        for node in &a.nodes {
            if node.depth > 0 {
                *per_hop.entry(node.depth).or_default() += 1;
            }
        }
        prop_assert!(per_hop.values().all(|&c| c <= cap));
        // Sampled nodes lie within the true k-hop ball, at >= true depth
        // (sampling can only lengthen apparent discovery, never shorten).
        let ball = reference_bfs_depths(&g, src, k, TraversalDirection::Outbound);
        for node in &a.nodes {
            let true_depth = ball.get(&node.node.raw());
            prop_assert!(true_depth.is_some(), "sampled node outside the k-hop ball");
            prop_assert!(*true_depth.expect("checked") <= node.depth);
        }
    }
}
