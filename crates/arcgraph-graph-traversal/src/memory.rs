//! In-memory reference [`EdgeSource`] adapter.
//!
//! The simplest possible adapter: insertion-ordered adjacency over
//! `arcgraph-core` IDs, topology-only payloads (`NodeData = EdgeData = ()`),
//! uniform per-item costs. Used by this crate's tests + the V11-S-04 exit
//! bench, and available to embedders that hold a materialized graph
//! so callers can populate an in-memory graph without a storage adapter.

use std::collections::HashMap;
use std::convert::Infallible;

use arcgraph_core::{NodeId, RelId, TypeId};

use crate::source::{EdgeFilter, EdgeSource, Neighbor, TraversalDirection};

/// One stored directed edge.
#[derive(Debug, Clone, Copy)]
struct StoredEdge {
    rel_id: RelId,
    other: NodeId,
    rel_type: Option<TypeId>,
}

/// Insertion-ordered in-memory adjacency (deterministic neighbor batches —
/// the [`EdgeSource`] determinism contract).
#[derive(Debug, Default)]
pub struct MemoryEdgeSource {
    out_adj: HashMap<NodeId, Vec<StoredEdge>>,
    in_adj: HashMap<NodeId, Vec<StoredEdge>>,
    next_rel: u64,
    /// Cost charged per first-seen edge (default 0 = cost budgets inert).
    edge_cost: u64,
    /// Cost charged per first-retained node (default 0).
    node_cost: u64,
}

impl MemoryEdgeSource {
    /// Empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Uniform cost units charged per edge / node (lets budget tests and
    /// byte-less embedders exercise the H-1 trip discipline).
    pub fn set_uniform_costs(&mut self, edge_cost: u64, node_cost: u64) {
        self.edge_cost = edge_cost;
        self.node_cost = node_cost;
    }

    /// Add a directed edge `src -> dst`, auto-allocating its [`RelId`].
    /// Returns the allocated id (useful for ban-list tests).
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, rel_type: Option<TypeId>) -> RelId {
        self.next_rel += 1;
        let rel_id = RelId::new(self.next_rel);
        self.out_adj.entry(src).or_default().push(StoredEdge {
            rel_id,
            other: dst,
            rel_type,
        });
        self.in_adj.entry(dst).or_default().push(StoredEdge {
            rel_id,
            other: src,
            rel_type,
        });
        rel_id
    }

    fn collect(
        &self,
        map: &HashMap<NodeId, Vec<StoredEdge>>,
        node: NodeId,
        filter: &EdgeFilter,
        out: &mut Vec<Neighbor<(), ()>>,
    ) {
        if let Some(edges) = map.get(&node) {
            for e in edges {
                let type_ok = match filter.rel_type {
                    None => true,
                    Some(t) => e.rel_type == Some(t),
                };
                if type_ok {
                    out.push(Neighbor {
                        rel_id: e.rel_id,
                        dst: e.other,
                        dst_data: (),
                        edge_data: (),
                        edge_cost: self.edge_cost,
                        node_cost: self.node_cost,
                    });
                }
            }
        }
    }
}

impl EdgeSource for MemoryEdgeSource {
    type NodeData = ();
    type EdgeData = ();
    type Error = Infallible;

    fn neighbors(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<Vec<Neighbor<(), ()>>, Infallible> {
        let mut out = Vec::new();
        match direction {
            TraversalDirection::Outbound => self.collect(&self.out_adj, node, filter, &mut out),
            TraversalDirection::Inbound => self.collect(&self.in_adj, node, filter, &mut out),
            TraversalDirection::Undirected => {
                // Outbound batch then inbound batch; a self-loop appears in
                // both with the same RelId and the traversal's per-RelId
                // dedupe collapses it (documented; mirrors the substrate's
                // undirected union semantics).
                self.collect(&self.out_adj, node, filter, &mut out);
                self.collect(&self.in_adj, node, filter, &mut out);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inbound_mirrors_outbound() {
        let mut g = MemoryEdgeSource::new();
        let rel = g.add_edge(NodeId::new(1), NodeId::new(2), None);
        let inb = g
            .neighbors(
                NodeId::new(2),
                TraversalDirection::Inbound,
                &EdgeFilter::any(),
            )
            .expect("infallible");
        assert_eq!(inb.len(), 1);
        assert_eq!(inb[0].rel_id, rel);
        assert_eq!(inb[0].dst, NodeId::new(1));
    }

    #[test]
    fn default_degree_counts_filtered_edges() {
        use arcgraph_core::TypeId;
        let mut g = MemoryEdgeSource::new();
        g.add_edge(NodeId::new(1), NodeId::new(2), Some(TypeId::new(7)));
        g.add_edge(NodeId::new(1), NodeId::new(3), Some(TypeId::new(8)));
        let d_all = g
            .degree(
                NodeId::new(1),
                TraversalDirection::Outbound,
                &EdgeFilter::any(),
            )
            .expect("infallible");
        let d_7 = g
            .degree(
                NodeId::new(1),
                TraversalDirection::Outbound,
                &EdgeFilter::rel_type(TypeId::new(7)),
            )
            .expect("infallible");
        assert_eq!((d_all, d_7), (2, 1));
    }
}
