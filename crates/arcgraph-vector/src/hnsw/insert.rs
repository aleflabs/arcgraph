//! HNSW incremental insert — Malkov & Yashunin TPAMI 2018
//! Algorithms 1 & 4.
//!
//! Per-vector flow (paper §4 Algorithm 1):
//!
//! 1. Sample the new node's `level` from the geometric
//!    distribution `floor(-ln(uniform()) * mL)` where
//!    `mL = 1/ln(M)`.
//! 2. From `entry_point` greedy-zoom through layers
//!    `[max_level..level+1]` at `ef=1`.
//! 3. From the resulting closest node, run `search_layer` at
//!    `ef_construction` for each layer `[level..0]`, picking the
//!    M closest as the candidate set, then prune via Algorithm 4
//!    (`select_neighbors_heuristic`).
//! 4. Bidirectional edge insert: also update each chosen
//!    neighbor's adjacency list (and prune *that* list back to
//!    `m_max` at upper layers, `m_max0` at layer 0).
//! 5. If `level > max_level`, the new node becomes the
//!    `entry_point`.
//!
//! ## Determinism
//!
//! Every randomness source — level sampling, neighbor
//! tie-breaking — is fed by the graph's seeded `StdRng`, so an
//! `(insert sequence, params, seed)` triple is reproducible.
//! Tests pin recall via this property.

use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;

use super::graph::{HnswGraph, NodeNeighbors};
use super::search::{Candidate, search_layer};

impl HnswGraph {
    /// Insert a single encoded vector. Idempotent on repeat
    /// `id`s only in the sense that a duplicate replaces the
    /// previous bytes + connectivity (the previous node is
    /// torn down first, then re-inserted at a fresh level).
    /// In practice catalog allocators do not reuse `VectorId`s
    /// during normal operation; the duplicate path exists for
    /// MN-RU repair which re-inserts unreachable nodes after
    /// clearing their adjacency.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] if `vector_bytes`
    ///   length does not equal `bytes_per_vector`.
    pub fn insert(
        &mut self,
        id: VectorId,
        vector_bytes: &[u8],
        kernel: &dyn DistanceKernel,
    ) -> Result<(), VectorIndexError> {
        self.validate_vector_bytes(vector_bytes)?;

        // If `id` is already present, treat insert as
        // "replace" — strip prior adjacency entries that
        // mention `id` so we don't end up with stale
        // back-pointers, then re-insert at a fresh level.
        if self.vectors.contains_key(&id) {
            self.detach_node(id);
        }
        self.vectors.insert(id, vector_bytes.to_vec());
        // Insert wakes up a tombstoned slot (the MN-RU repair
        // path uses this to revive a previously-deleted vector
        // after re-attaching it to the graph).
        self.tombstones.insert(id, false);

        let level = self.sample_level();
        // Initialize the node's per-layer adjacency with empty
        // vectors capped at the right cardinality.
        let mut node = NodeNeighbors {
            neighbors: vec![Vec::new(); level + 1],
        };

        // Special case: empty graph → this becomes the entry.
        if self.entry_point.is_none() {
            self.nodes.insert(id, node);
            self.entry_point = Some(id);
            self.max_level = level;
            return Ok(());
        }

        let entry = self.entry_point.expect("entry_point checked above");
        let entry_level = self.max_level;

        // Greedy zoom from top-of-graph down to the inserted
        // node's level + 1 (exclusive of `level` itself; that
        // layer is part of the M-pick loop below).
        let mut eps: Vec<VectorId> = vec![entry];
        if entry_level > level {
            for l in ((level + 1)..=entry_level).rev() {
                let next = search_layer(self, vector_bytes, &eps, l, 1, kernel);
                if let Some(c) = next.first() {
                    eps = vec![c.id];
                }
            }
        }

        // M-pick loop — for each layer the node participates in,
        // run `search_layer` at `ef_construction`, prune to
        // `m_max`, install bidirectional edges.
        for l in (0..=level.min(entry_level)).rev() {
            let cap = if l == 0 {
                self.params.m_max0()
            } else {
                self.params.m_max()
            };
            let candidates = search_layer(
                self,
                vector_bytes,
                &eps,
                l,
                self.params.ef_construction,
                kernel,
            );
            // Per Algorithm 4 (`select_neighbors_heuristic`,
            // `extendCandidates=false`, `keepPrunedConnections=true`).
            let chosen =
                select_neighbors_heuristic(self, vector_bytes, &candidates, self.params.m, kernel);

            // Forward edges: node[l] ← chosen.
            node.neighbors[l] = chosen.iter().map(|c| c.id).collect();

            // Backward edges: each chosen neighbor adds `id` to
            // its layer-l adjacency, then re-prunes if it
            // exceeds the layer cap.
            for c in &chosen {
                if let Some(other) = self.nodes.get_mut(&c.id) {
                    // Defensively grow the other node's per-layer
                    // adjacency if it doesn't extend up to `l`
                    // (this should not happen — `l <=
                    // min(level, entry_level) <=
                    // other.level()` — but the bound check
                    // protects against future protocol drift).
                    while other.neighbors.len() <= l {
                        other.neighbors.push(Vec::new());
                    }
                    other.neighbors[l].push(id);
                    // Re-prune if over cap. We do this by
                    // collecting candidates with distances to
                    // the *neighbor's* vector (not the new
                    // node's), then re-running the heuristic.
                    if other.neighbors[l].len() > cap {
                        let neighbor_bytes = self
                            .vectors
                            .get(&c.id)
                            .expect("chosen neighbor must exist")
                            .clone();
                        // Build the candidate set: every back-edge
                        // entry, with distance from neighbor → it.
                        let mut prune_set: Vec<Candidate> = Vec::with_capacity(
                            self.nodes
                                .get(&c.id)
                                .map(|n| n.neighbors[l].len())
                                .unwrap_or(0),
                        );
                        if let Some(other_node) = self.nodes.get(&c.id) {
                            for &m in &other_node.neighbors[l] {
                                if let Some(mb) = self.vectors.get(&m) {
                                    let d = kernel.distance(&neighbor_bytes, mb);
                                    prune_set.push(Candidate::new(d, m));
                                }
                            }
                        }
                        // Sort ascending so the heuristic can
                        // walk in distance order.
                        prune_set.sort();
                        let kept = select_neighbors_heuristic(
                            self,
                            &neighbor_bytes,
                            &prune_set,
                            cap,
                            kernel,
                        );
                        if let Some(other_mut) = self.nodes.get_mut(&c.id) {
                            other_mut.neighbors[l] = kept.iter().map(|x| x.id).collect();
                        }
                    }
                }
            }

            eps = chosen.iter().map(|c| c.id).collect();
            if eps.is_empty() {
                // Pathological case: no candidates at this
                // layer. Should not happen on a non-empty graph
                // — fall back to entry and keep going.
                eps = vec![entry];
            }
        }

        // Persist the new node's adjacency. We deferred this
        // until after the loop so that during the M-pick loop
        // the `nodes` map does not yet contain `id` and
        // `search_layer` cannot find self-loops.
        self.nodes.insert(id, node);

        // Promote the entry point if the new node lives higher
        // than the current top.
        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }

        Ok(())
    }

    /// Strip every back-edge that mentions `id` and remove `id`
    /// from `vectors` + `nodes`. Used by `insert` when the
    /// caller is replacing a previously-inserted vector — and by
    /// `mn_ru_repair` to clear unreachable nodes' stale
    /// adjacencies before re-attaching.
    pub(crate) fn detach_node(&mut self, id: VectorId) {
        // Snapshot the per-layer neighbor list so we can mutate
        // their lists without aliasing borrows.
        let snap: Vec<Vec<VectorId>> = self
            .nodes
            .get(&id)
            .map(|n| n.neighbors.clone())
            .unwrap_or_default();
        for (l, layer_adj) in snap.into_iter().enumerate() {
            for n in layer_adj {
                if let Some(other) = self.nodes.get_mut(&n) {
                    if let Some(other_layer) = other.neighbors.get_mut(l) {
                        other_layer.retain(|x| *x != id);
                    }
                }
            }
        }
        self.nodes.remove(&id);
        self.vectors.remove(&id);
        self.tombstones.remove(&id);

        // Repair entry_point / max_level if we just removed the
        // top. This is conservative: a global recompute scan,
        // but `detach_node` runs only at MN-RU / replacement
        // boundaries — not the hot path.
        if self.entry_point == Some(id) {
            self.recompute_entry_point();
        }
    }

    fn recompute_entry_point(&mut self) {
        let mut best: Option<(VectorId, usize)> = None;
        for (&nid, node) in self.nodes.iter() {
            let lvl = node.level();
            match best {
                None => best = Some((nid, lvl)),
                Some((_, bl)) if lvl > bl => best = Some((nid, lvl)),
                _ => {}
            }
        }
        match best {
            Some((nid, lvl)) => {
                self.entry_point = Some(nid);
                self.max_level = lvl;
            }
            None => {
                self.entry_point = None;
                self.max_level = 0;
            }
        }
    }
}

/// Algorithm 4 (`select_neighbors_heuristic`).
///
/// Walk the candidate set in distance-ascending order; keep a
/// candidate only if it is closer to the query than to *any*
/// already-kept neighbor. This produces a diverse out-edge set
/// (each direction in vector space gets at most one edge),
/// which empirically yields better recall than the naive
/// "M closest" rule for the same out-degree.
///
/// Per the paper §4: `extendCandidates=false`,
/// `keepPrunedConnections=true` is the default that ships in
/// hnswlib + qdrant + this v1.0 baseline. Pruned-but-not-kept
/// connections are appended at the tail when fewer than `m`
/// were chosen by the heuristic — the "keep pruned" branch.
pub(crate) fn select_neighbors_heuristic(
    graph: &HnswGraph,
    query: &[u8],
    candidates: &[Candidate],
    m: usize,
    kernel: &dyn DistanceKernel,
) -> Vec<Candidate> {
    let _ = query; // referenced only via the input candidates' distance to query
    if candidates.is_empty() || m == 0 {
        return Vec::new();
    }

    // Candidates expected to arrive sorted ascending; defensive
    // re-sort is cheap and protects future callers.
    let mut sorted = candidates.to_vec();
    sorted.sort();

    let mut kept: Vec<Candidate> = Vec::with_capacity(m.min(sorted.len()));
    let mut pruned: Vec<Candidate> = Vec::new();
    for cand in sorted.iter() {
        if kept.len() >= m {
            break;
        }
        // Diversity rule: keep only if `cand` is closer to the
        // query than to any already-kept neighbor.
        let cand_bytes = match graph.vector_bytes(cand.id) {
            Some(b) => b,
            None => continue,
        };
        let mut keep = true;
        for k in &kept {
            let kbytes = match graph.vector_bytes(k.id) {
                Some(b) => b,
                None => continue,
            };
            let d_cand_to_kept = kernel.distance(cand_bytes, kbytes);
            // `cand.distance.0` is the kernel-output distance
            // from query → cand; `d_cand_to_kept` is from
            // cand → kept. The diversity rule keeps cand only
            // if it's *closer to the query than to a kept
            // neighbor* — i.e., the kept neighbor doesn't
            // already cover this region of vector space.
            if d_cand_to_kept < cand.distance.0 {
                keep = false;
                break;
            }
        }
        if keep {
            kept.push(*cand);
        } else {
            pruned.push(*cand);
        }
    }

    // `keepPrunedConnections=true` branch: backfill from the
    // pruned tail until `kept.len() == m` (or pruned exhausted).
    while kept.len() < m {
        match pruned.first() {
            None => break,
            Some(p) => {
                kept.push(*p);
                pruned.remove(0);
            }
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2F32;
    use crate::hnsw::graph::{HnswGraph, HnswParams};

    fn bytes(vs: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(vs).to_vec()
    }

    #[test]
    fn insert_first_node_sets_entry_point() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        g.insert(VectorId::new(7), &bytes(&[1.0, 0.0, 0.0]), &L2F32)
            .unwrap();
        assert_eq!(g.entry_point, Some(VectorId::new(7)));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn insert_validates_dimension() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        // 4-d vector against 3-d graph → DimensionMismatch
        let r = g.insert(VectorId::new(0), &bytes(&[1.0, 2.0, 3.0, 4.0]), &L2F32);
        assert!(matches!(
            r,
            Err(VectorIndexError::DimensionMismatch { expected: 3, .. })
        ));
    }

    #[test]
    fn replace_existing_id_preserves_count() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        g.insert(VectorId::new(0), &bytes(&[1.0, 2.0, 3.0]), &L2F32)
            .unwrap();
        g.insert(VectorId::new(1), &bytes(&[4.0, 5.0, 6.0]), &L2F32)
            .unwrap();
        // Re-insert id 0 with new bytes.
        g.insert(VectorId::new(0), &bytes(&[7.0, 8.0, 9.0]), &L2F32)
            .unwrap();
        assert_eq!(g.len(), 2);
        // Verify the bytes were swapped.
        assert_eq!(
            g.vector_bytes(VectorId::new(0)).unwrap(),
            bytes(&[7.0, 8.0, 9.0]).as_slice()
        );
    }

    #[test]
    fn select_neighbors_heuristic_caps_at_m() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        // Build a 5-node graph then directly call the heuristic
        // on its candidate set to verify the M-cap.
        for i in 0..5u32 {
            let v = bytes(&[i as f32, 0.0, 0.0]);
            g.insert(VectorId::new(i), &v, &L2F32).unwrap();
        }
        let q = bytes(&[0.0, 0.0, 0.0]);
        // Construct a candidate list manually (every node sorted
        // ascending by distance from query).
        let mut cs: Vec<Candidate> = (0..5u32)
            .map(|i| {
                let d = L2F32.distance(&q, g.vector_bytes(VectorId::new(i)).unwrap());
                Candidate::new(d, VectorId::new(i))
            })
            .collect();
        cs.sort();
        let kept = select_neighbors_heuristic(&g, &q, &cs, 3, &L2F32);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn detach_node_removes_back_edges() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        for i in 0..6u32 {
            g.insert(VectorId::new(i), &bytes(&[i as f32, 0.0, 0.0]), &L2F32)
                .unwrap();
        }
        let target = VectorId::new(3);
        g.detach_node(target);
        for (id, node) in g.nodes.iter() {
            for layer_adj in &node.neighbors {
                assert!(
                    !layer_adj.contains(&target),
                    "{id:?} still references detached {target:?}"
                );
            }
        }
        assert!(!g.contains(target));
    }
}
