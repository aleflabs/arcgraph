//! MN-RU connectivity repair (ADR-003 Strategy 2).
//!
//! Naive HNSW degrades catastrophically under repeated
//! delete-and-reinsert cycles: per Xiao et al. arxiv 2407.07871
//! Figure 3, ~3–4 % of points become unreachable from the entry
//! point on Gist after ~150 cycles. The MN-RU repair restores
//! reachability by detecting unreachable nodes and re-inserting
//! them.
//!
//! ## What "unreachable" means here
//!
//! A node is unreachable iff it cannot be reached from the
//! current entry point by walking the per-layer adjacency lists
//! (top layer down to layer 0, taking every outbound edge at
//! each step). This is the layered analogue of "BFS from
//! `entry_point`" — Xiao 2024 §3.1 phrases it as the same.
//!
//! ## What this repair does NOT do
//!
//! - **Back-up index variant.** Xiao 2024's full proposal keeps
//!   a parallel "MN-RU back-up" index and uses a dual-search
//!   strategy at query time. v1.0's MN-RU is the simpler
//!   "detect + reinsert" flavor — ADR-003 §Decision Strategy 2
//!   says preserved, and the full back-up-index variant is a
//!   v1.1 follow-up.
//! - **Concurrent reinsertion.** The repair runs synchronously
//!   on `&mut self`. A concurrent-search variant would need
//!   epoch-based reclamation (per ADR-002 Tokio thread-pool
//!   discipline) which lives in Slice F.1's per-tenant arena
//!   routing.
//! - **Tombstone-aware filtering.** MN-RU restores reachability
//!   for the routing graph; tombstoned vectors stay tombstoned
//!   (so they don't leak back into search results) but are
//!   re-attached so they keep serving as routing hubs.

use std::collections::{HashSet, VecDeque};

use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;

use super::graph::HnswGraph;

impl HnswGraph {
    /// Walk the graph from `entry_point` and return every
    /// `VectorId` that is **unreachable** under per-layer
    /// adjacency traversal.
    ///
    /// Time complexity: `O(V + E)` where `V = self.len()` and
    /// `E = Σᵢ |adj_i|` over all per-layer adjacencies. Hot
    /// callers should cache the result; the repair path always
    /// recomputes after a delete-heavy phase.
    ///
    /// Returns an empty vec on an empty graph.
    #[must_use]
    pub fn detect_unreachable(&self) -> Vec<VectorId> {
        let Some(entry) = self.entry_point else {
            // Empty graph → trivially nothing unreachable.
            return Vec::new();
        };
        let mut reachable: HashSet<VectorId> = HashSet::with_capacity(self.len());
        reachable.insert(entry);
        let mut queue: VecDeque<VectorId> = VecDeque::new();
        queue.push_back(entry);
        while let Some(cur) = queue.pop_front() {
            let Some(node) = self.nodes.get(&cur) else {
                continue;
            };
            for layer_adj in &node.neighbors {
                for &n in layer_adj {
                    if reachable.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }
        // Anything in `vectors` but not in `reachable` is
        // unreachable.
        self.vectors
            .keys()
            .filter(|id| !reachable.contains(id))
            .copied()
            .collect()
    }

    /// Re-insert each `VectorId` in `unreachable` so the graph
    /// regains connectivity. Tombstoned status is preserved
    /// (we re-attach the routing edges, but if the vector was
    /// tombstoned before the repair, it stays tombstoned).
    ///
    /// Per ADR-003 Strategy 2 the cost target is ≤ 1.5× memory,
    /// ≤ 1.3× insert latency. v1.0's repair is the simpler
    /// detect+reinsert; the back-up-index variant is v1.1.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during reinsertion.
    /// Callers should treat any error as fatal — the graph state
    /// after a partial repair is consistent (reachable subset
    /// grew monotonically) but possibly under-repaired.
    pub fn mn_ru_repair(
        &mut self,
        unreachable: &[VectorId],
        kernel: &dyn DistanceKernel,
    ) -> Result<(), VectorIndexError> {
        for &id in unreachable {
            // Snapshot the bytes + tombstone state, detach,
            // re-insert. Vector ownership transfers through a
            // local clone because `insert` re-validates the
            // byte slice and stores a fresh copy.
            let bytes = match self.vector_bytes(id) {
                Some(b) => b.to_vec(),
                None => continue, // already gone — repair is a no-op for this id.
            };
            let was_tombstoned = self.is_tombstoned(id);
            self.detach_node(id);
            // Insert restarts level assignment from the seeded
            // RNG, so the repaired graph differs deterministically
            // from the pre-repair graph but recall is preserved.
            self.insert(id, &bytes, kernel)?;
            if was_tombstoned {
                // Re-tombstone the vector: it's reachable now,
                // but search must still skip it from the
                // result list.
                self.tombstones.insert(id, true);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2F32;
    use crate::hnsw::graph::HnswParams;

    fn bytes(vs: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(vs).to_vec()
    }

    #[test]
    fn empty_graph_has_no_unreachable() {
        let g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        assert!(g.detect_unreachable().is_empty());
    }

    #[test]
    fn fully_connected_small_graph_reports_no_unreachable() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        for i in 0..16u32 {
            g.insert(VectorId::new(i), &bytes(&[i as f32, 0.0, 0.0]), &L2F32)
                .unwrap();
        }
        let u = g.detect_unreachable();
        assert!(
            u.is_empty(),
            "expected no unreachable nodes; got {} (sample: {:?})",
            u.len(),
            u.iter().take(5).collect::<Vec<_>>()
        );
    }

    #[test]
    fn synthetic_orphan_is_detected_and_repaired() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        for i in 0..8u32 {
            g.insert(VectorId::new(i), &bytes(&[i as f32, 0.0, 0.0]), &L2F32)
                .unwrap();
        }
        // Synthetically orphan id=4: clear every back-edge that
        // mentions it, AND clear its own outbound edges (so it
        // can't even hop back to a reachable node).
        let target = VectorId::new(4);
        for node in g.nodes.values_mut() {
            for layer_adj in node.neighbors.iter_mut() {
                layer_adj.retain(|x| *x != target);
            }
        }
        if let Some(target_node) = g.nodes.get_mut(&target) {
            for layer_adj in target_node.neighbors.iter_mut() {
                layer_adj.clear();
            }
        }
        // If the orphan happened to be the entry_point, swap the
        // entry to a different surviving node so the BFS doesn't
        // start at the orphan and trivially reach itself.
        if g.entry_point == Some(target) {
            // Pick any other node. Walk in deterministic order.
            let alternate = (0..8u32)
                .map(VectorId::new)
                .find(|id| *id != target && g.nodes.contains_key(id))
                .expect("alternate entry must exist");
            g.entry_point = Some(alternate);
            g.max_level = g.nodes[&alternate].level();
        }

        let u = g.detect_unreachable();
        assert!(
            u.contains(&target),
            "expected {target:?} unreachable; got {u:?}"
        );

        // Repair and verify.
        g.mn_ru_repair(&u, &L2F32).unwrap();
        let u2 = g.detect_unreachable();
        assert!(
            !u2.contains(&target),
            "{target:?} still unreachable after MN-RU repair; got {u2:?}"
        );
    }

    #[test]
    fn repair_preserves_tombstone_status() {
        let mut g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        for i in 0..8u32 {
            g.insert(VectorId::new(i), &bytes(&[i as f32, 0.0, 0.0]), &L2F32)
                .unwrap();
        }
        let target = VectorId::new(2);
        g.mark_deleted(target);
        // Tear edges (synthetic orphan).
        for node in g.nodes.values_mut() {
            for layer_adj in node.neighbors.iter_mut() {
                layer_adj.retain(|x| *x != target);
            }
        }
        if let Some(t) = g.nodes.get_mut(&target) {
            for layer_adj in t.neighbors.iter_mut() {
                layer_adj.clear();
            }
        }
        // Re-anchor entry if it was target.
        if g.entry_point == Some(target) {
            let alt = (0..8u32)
                .map(VectorId::new)
                .find(|id| *id != target)
                .unwrap();
            g.entry_point = Some(alt);
            g.max_level = g.nodes[&alt].level();
        }
        let u = g.detect_unreachable();
        g.mn_ru_repair(&u, &L2F32).unwrap();
        assert!(
            g.is_tombstoned(target),
            "tombstone bit must be preserved through MN-RU repair"
        );
    }
}
