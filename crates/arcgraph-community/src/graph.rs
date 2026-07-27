//! Undirected weighted graph in CSR layout, used by GVE-Leiden.
//!
//! Per Sahu §III.A the algorithm operates on a **static frozen
//! snapshot** — all changes between refreshes are absorbed by the
//! incremental DF Leiden phase (M3.d-2). The snapshot is per-tenant;
//! cross-tenant edges are caller error and rejected at the
//! construction boundary (the caller of [`Graph::from_edges_undirected`]
//! is responsible for filtering edges to a single tenant before
//! handing them to this module).
//!
//! # Layout
//!
//! - `offsets[u..=u+1]` slices a contiguous run of neighbours for
//!   vertex `u`. Length `n + 1`; `offsets[0] == 0` and
//!   `offsets[n] == m * 2` for an undirected graph (each edge
//!   appears twice, once in each direction).
//! - `neighbors` is parallel to `weights` and stores the destination
//!   vertex of each half-edge.
//! - `degrees` is the per-vertex sum of incident edge weights;
//!   cached so the modularity-delta computation in
//!   [`super::leiden_static::GveLeiden`] doesn't have to re-sum on
//!   every move check.
//! - `total_weight` is `2m` for an undirected weighted graph
//!   (sum over half-edges); the modularity formula divides by this.
//!
//! # Why CSR not adjacency-list-of-Vec
//!
//! CSR is one big allocation rather than `n` small ones — better
//! cache locality on the sequential local-moving sweep, and a
//! `&[u32]` neighbour slice can be processed without further
//! pointer-chasing. Aggregation in the Leiden multi-pass rebuilds
//! a fresh CSR from a `Vec<(u32, u32, f32)>` each time;
//! [`Graph::from_edges_undirected`] is the construction entry
//! point.

use arcgraph_core::NodeId;

/// Undirected weighted graph in CSR layout.
#[derive(Clone, Debug)]
pub struct Graph {
    /// Number of vertices.
    n: u32,
    /// CSR offset array, length `n + 1`. `offsets[u]` is the
    /// inclusive start of `u`'s neighbour run; `offsets[u+1]` the
    /// exclusive end.
    offsets: Vec<u32>,
    /// CSR neighbour array. Half-edges; each undirected edge
    /// appears twice (once for each endpoint).
    neighbors: Vec<u32>,
    /// CSR weight array, parallel to `neighbors`.
    weights: Vec<f32>,
    /// Per-vertex degree (sum of incident edge weights). Cached
    /// for the modularity-delta hot path in `local_moving_phase`.
    degrees: Vec<f32>,
    /// Total edge weight: `2m` for an undirected graph.
    total_weight: f64,
}

impl Graph {
    /// Build from `(u, v, w)` triples. Edges are interpreted as
    /// **undirected**: each `(u, v, w)` inserts both `u → v` and
    /// `v → u` half-edges. Self-loops (`u == v`) are allowed and
    /// carried at full weight per Newman 2006 modularity convention.
    ///
    /// # Panics
    ///
    /// Panics if any endpoint `u >= n` or `v >= n`. Caller is
    /// responsible for dense-indexed vertices in `0..n`.
    #[must_use]
    pub fn from_edges_undirected(n: u32, edges: &[(u32, u32, f32)]) -> Self {
        // Step 1: count per-vertex out-degree (number of half-edges).
        // Self-loops count once on this side; we pad them later.
        let mut counts = vec![0u32; n as usize];
        for &(u, v, _) in edges {
            assert!(u < n, "edge endpoint {u} out of range (n = {n})");
            assert!(v < n, "edge endpoint {v} out of range (n = {n})");
            counts[u as usize] = counts[u as usize].saturating_add(1);
            if u != v {
                counts[v as usize] = counts[v as usize].saturating_add(1);
            }
        }

        // Step 2: prefix-sum into offsets.
        let mut offsets = vec![0u32; (n as usize) + 1];
        let mut acc = 0u32;
        for i in 0..(n as usize) {
            offsets[i] = acc;
            acc = acc.checked_add(counts[i]).expect("CSR offset overflow");
        }
        offsets[n as usize] = acc;

        // Step 3: scatter half-edges. We re-use `counts` as a
        // per-vertex write cursor (decrementing as we fill each
        // half-edge slot) so we don't need a second prefix sum.
        let total_half_edges = acc as usize;
        let mut neighbors = vec![0u32; total_half_edges];
        let mut weights = vec![0.0f32; total_half_edges];
        let mut cursor = offsets.clone();
        for &(u, v, w) in edges {
            let i = cursor[u as usize] as usize;
            neighbors[i] = v;
            weights[i] = w;
            cursor[u as usize] += 1;
            if u != v {
                let j = cursor[v as usize] as usize;
                neighbors[j] = u;
                weights[j] = w;
                cursor[v as usize] += 1;
            }
        }

        // Step 4: compute per-vertex weighted degree.
        let mut degrees = vec![0.0f32; n as usize];
        let mut total_weight = 0.0f64;
        for u in 0..(n as usize) {
            let lo = offsets[u] as usize;
            let hi = offsets[u + 1] as usize;
            let s: f32 = weights[lo..hi].iter().sum();
            degrees[u] = s;
            total_weight += f64::from(s);
        }

        Self {
            n,
            offsets,
            neighbors,
            weights,
            degrees,
            total_weight,
        }
    }

    /// Number of vertices.
    #[inline]
    #[must_use]
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Iterate neighbours of `u` as `(neighbor, weight)` pairs.
    /// Half-edges; each undirected edge surfaces twice across the
    /// graph (once per endpoint).
    #[inline]
    pub fn neighbors(&self, u: u32) -> impl Iterator<Item = (u32, f32)> + '_ {
        let lo = self.offsets[u as usize] as usize;
        let hi = self.offsets[u as usize + 1] as usize;
        self.neighbors[lo..hi]
            .iter()
            .copied()
            .zip(self.weights[lo..hi].iter().copied())
    }

    /// Per-vertex weighted degree. Cached at construction.
    #[inline]
    #[must_use]
    pub fn degree(&self, u: u32) -> f32 {
        self.degrees[u as usize]
    }

    /// Total edge weight `2m` for an undirected graph. Used as the
    /// denominator of the Newman 2006 modularity formula.
    #[inline]
    #[must_use]
    pub fn total_weight_2m(&self) -> f64 {
        self.total_weight
    }

    /// Convert internal `u32` vertex index to `NodeId`. The mapping
    /// is identity at v1.0 — caller responsibility to maintain
    /// dense indexing in `0..n`. At v1.1+ a sparse-NodeId remapping
    /// table is added (ADR-040 open question).
    #[inline]
    #[must_use]
    pub fn vertex_to_node_id(v: u32) -> NodeId {
        NodeId::new(u64::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny graph: two disconnected edges, n=4.
    fn two_edges() -> Graph {
        Graph::from_edges_undirected(4, &[(0, 1, 1.0), (2, 3, 2.0)])
    }

    #[test]
    fn from_edges_undirected_offsets_are_prefix_sum() {
        let g = two_edges();
        assert_eq!(g.n(), 4);
        // 4 half-edges total: 2 edges × 2 endpoints.
        // Per vertex: 1, 1, 1, 1.
        // Offsets: [0, 1, 2, 3, 4]
        let collected: Vec<(u32, f32)> = g.neighbors(0).collect();
        assert_eq!(collected, vec![(1, 1.0)]);
        let collected: Vec<(u32, f32)> = g.neighbors(2).collect();
        assert_eq!(collected, vec![(3, 2.0)]);
    }

    #[test]
    fn degrees_match_edge_weights() {
        let g = two_edges();
        assert!((g.degree(0) - 1.0).abs() < 1e-6);
        assert!((g.degree(1) - 1.0).abs() < 1e-6);
        assert!((g.degree(2) - 2.0).abs() < 1e-6);
        assert!((g.degree(3) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn total_weight_is_2m_for_undirected() {
        let g = two_edges();
        // Two edges of weight 1 and 2 ⇒ 2m = 2*(1+2) = 6.
        assert!((g.total_weight_2m() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn neighbors_are_symmetric() {
        let g = Graph::from_edges_undirected(3, &[(0, 1, 1.0), (1, 2, 2.0)]);
        // 0 sees 1; 1 sees 0 and 2; 2 sees 1.
        let n0: Vec<u32> = g.neighbors(0).map(|(v, _)| v).collect();
        let n1: Vec<u32> = g.neighbors(1).map(|(v, _)| v).collect();
        let n2: Vec<u32> = g.neighbors(2).map(|(v, _)| v).collect();
        assert_eq!(n0, vec![1]);
        // n1 has both endpoints; order is insertion order.
        assert_eq!(n1.len(), 2);
        assert!(n1.contains(&0) && n1.contains(&2));
        assert_eq!(n2, vec![1]);
    }

    #[test]
    fn self_loop_is_allowed_and_counted_once() {
        // Self-loop on vertex 0 with weight 1; one regular edge 0-1.
        let g = Graph::from_edges_undirected(2, &[(0, 0, 1.0), (0, 1, 2.0)]);
        // Half-edges for 0: self + 1 = 2; for 1: 0 = 1.
        let n0: Vec<(u32, f32)> = g.neighbors(0).collect();
        let n1: Vec<(u32, f32)> = g.neighbors(1).collect();
        assert_eq!(n0.len(), 2);
        assert_eq!(n1.len(), 1);
        // Degree(0) sums both half-edges out of it: 1 (self) + 2 = 3.
        assert!((g.degree(0) - 3.0).abs() < 1e-6);
        assert!((g.degree(1) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn empty_graph_has_zero_total_weight() {
        let g = Graph::from_edges_undirected(5, &[]);
        assert_eq!(g.n(), 5);
        for u in 0..5 {
            assert_eq!(g.neighbors(u).count(), 0);
            assert!((g.degree(u) - 0.0).abs() < 1e-6);
        }
        assert!((g.total_weight_2m() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn vertex_to_node_id_is_identity() {
        assert_eq!(Graph::vertex_to_node_id(0), NodeId::ZERO);
        assert_eq!(Graph::vertex_to_node_id(42), NodeId::new(42));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_endpoint_panics() {
        let _ = Graph::from_edges_undirected(2, &[(0, 5, 1.0)]);
    }
}
