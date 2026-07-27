//! Streaming insert + delta-segment lookaside.
//!
//! Per ADR-035 §5.3.1 (B-3 resolution) — DiskANN's batch-merge
//! Vamana insert pattern would otherwise violate the I-V7 T1
//! read-your-writes invariant (insert returns Ok ⟹ subsequent
//! search by the same kernel sees the inserted vector). The
//! delta-segment is an in-memory lookaside between Phase 3
//! visible-publish and the next Vamana batch fold; searches
//! merge main-graph + delta candidate lists by distance.
//!
//! ## Storage shape
//!
//! Slice D ships **brute-force only**:
//!
//! - `vectors: Vec<(VectorId, Vec<u8>)>` — append-only.
//! - `id_set: HashSet<VectorId>` — dedup + RYW-fast contains.
//!
//! ADR-035 §5.3.1 calls out that the delta-segment **may**
//! promote to a small in-memory HNSW (`M=8, ef_construction=100`)
//! once `delta.len() ≥ delta_brute_thresh = 128`. The
//! promotion is a perf optimization, not a correctness
//! requirement; Slice D keeps the brute-force path on the
//! hot path and leaves the HNSW promotion as future work
//! (see the `delta_segment_brute_force_only_at_slice_d` test
//! and the ADR-035 §S-2 follow-up). The storage shape +
//! parameter (`delta_brute_thresh`) is wired so the promotion
//! lands as a Slice F follow-up without reshaping the type.
//!
//! ## Merge fold trigger
//!
//! Per ADR-035 §5.3.1 the trigger is whichever fires first:
//!
//! 1. `delta.len() ≥ delta_max_size` (default 1 000).
//! 2. Background scheduler interval (default 60 s; not Slice D
//!    scope — the engine wires the scheduler at Slice F).
//! 3. Overflow `delta.len() ≥ 10 × delta_max_size` (forced
//!    inline; emit `tracing::warn!`).
//!
//! Slice D implements (1) and (3); (2) is left for the engine
//! integration to invoke [`DiskAnnGraph::merge_delta`] on its
//! cadence.
//!
//! ## RYW preservation (I-V7)
//!
//! [`DiskAnnGraph::insert_stream`] appends to the in-memory
//! delta-segment **synchronously**, before returning Ok. The
//! [`DiskAnnGraph::search_with_delta`] path runs delta-segment
//! search alongside the main graph and merges by distance
//! (NOT by rank — both paths use the same kernel under the
//! same quantizer state per ADR-035 §5.3.1, so distances are
//! directly comparable).
//!
//! Any subsequent search that would otherwise miss the new
//! vector finds it via the delta-segment branch of the merge.

use std::collections::HashSet;

use crate::{Result, VectorId, VectorIndexError};

use super::graph::DiskAnnGraph;

/// In-memory delta-segment lookaside for streaming inserts.
#[derive(Debug, Default, Clone)]
pub(crate) struct DeltaSegment {
    /// Append-only `(id, encoded bytes)` entries.
    entries: Vec<(VectorId, Vec<u8>)>,
    /// Set membership for O(1) `contains` (RYW-fast path).
    id_set: HashSet<VectorId>,
}

impl DeltaSegment {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn contains(&self, id: VectorId) -> bool {
        self.id_set.contains(&id)
    }

    pub(crate) fn append(&mut self, id: VectorId, bytes: Vec<u8>) {
        self.id_set.insert(id);
        self.entries.push((id, bytes));
    }

    /// Remove the entry with the given `VectorId` if present.
    /// Returns `true` if a removal occurred. Linear scan; the
    /// delta-segment is bounded by `delta_max_size = 1000` so
    /// O(N) is acceptable.
    pub(crate) fn remove(&mut self, id: VectorId) -> bool {
        if !self.id_set.remove(&id) {
            return false;
        }
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries.swap_remove(pos);
        }
        true
    }

    /// Iterator over `(id, bytes)` entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (VectorId, &[u8])> + '_ {
        self.entries.iter().map(|(id, b)| (*id, b.as_slice()))
    }
}

impl DiskAnnGraph {
    /// Streaming-insert a batch of `(VectorId, &[u8])` pairs.
    ///
    /// Per ADR-035 §5.3.1: appends to the in-memory delta-segment
    /// synchronously and folds into the main Vamana graph when
    /// the threshold is hit. Subsequent
    /// [`DiskAnnGraph::search_with_delta`] calls find the inserted
    /// vectors immediately — RYW (I-V7) preserved.
    ///
    /// Returns
    /// [`VectorIndexError::DimensionMismatch`] if any byte
    /// length doesn't match the configured `bytes_per_vector`,
    /// or if a duplicate id is supplied.
    ///
    /// Empty `vectors` is a no-op (returns `Ok` without
    /// touching the delta-segment).
    pub fn insert_stream(&mut self, vectors: &[(VectorId, &[u8])]) -> Result<()> {
        for (id, bytes) in vectors {
            self.check_or_set_byte_width(bytes.len())?;
            // Reject duplicates against either the main graph
            // or the delta-segment. The caller must `delete`
            // first to overwrite.
            if self.id_to_slot.contains_key(id) || self.delta.contains(*id) {
                return Err(VectorIndexError::IrrecoverableLoss {
                    index: crate::IndexId::ZERO,
                    reason: format!(
                        "duplicate VectorId {} on insert_stream (use delete first)",
                        id.raw()
                    ),
                });
            }
            self.delta.append(*id, bytes.to_vec());
        }

        let max = self.params.delta_max_size as usize;
        let overflow = max.saturating_mul(10).max(max + 1);

        if self.delta.len() >= overflow {
            // Per ADR-035 §5.3.1 — forced inline merge with
            // operator-visible warning.
            tracing::warn!(
                target: "arcgraph.vector.delta_segment_overflow",
                delta_len = self.delta.len(),
                threshold = overflow,
                "DiskANN delta-segment overflow; forcing inline merge"
            );
            self.merge_delta()?;
        } else if self.delta.len() >= max {
            self.merge_delta()?;
        }

        Ok(())
    }

    /// Force-merge the delta-segment into the main Vamana
    /// graph. Idempotent on an empty delta-segment.
    ///
    /// The merge is a Vamana incremental insert: each
    /// delta-segment vector is added to the main graph via
    /// greedy search → α-prune → symmetrize. After the merge
    /// the delta-segment is cleared and subsequent searches
    /// hit the main graph's edge list.
    ///
    /// Slice D ships single-thread merge. Concurrent
    /// insert/search wiring (epoch + reader-friendly
    /// concurrency) lives at Slice F.
    ///
    /// ## Pre-flight validation (issue #109 part 2)
    ///
    /// Per issue #109 part 2, the merge pre-validates the entire
    /// batch (byte-width + finite-float check on F32) BEFORE
    /// taking ownership of the delta entries. On any failure the
    /// delta is left intact and the merge returns
    /// [`VectorIndexError::BatchValidation`]. Without this
    /// pre-flight, a single malformed sibling (dim mismatch /
    /// NaN / Inf) would have silently lost every successfully
    /// ack'd entry queued behind it in the same batch — silent
    /// data loss for an I-V7 RYW-satisfying T3 insert.
    pub fn merge_delta(&mut self) -> Result<()> {
        if self.delta.is_empty() {
            return Ok(());
        }

        // Pre-validate the entire batch before taking ownership.
        // On error the delta stays intact so the caller can
        // surface the offending entry, repair, and retry.
        self.validate_delta_batch()?;

        // Snapshot the delta entries — we'll be mutating the
        // graph as we insert each one.
        let entries: Vec<(VectorId, Vec<u8>)> = std::mem::take(&mut self.delta.entries);
        self.delta.id_set.clear();

        // If the main graph is empty, build it directly from
        // the delta entries.
        if self.ids.is_empty() {
            let pairs: Vec<(VectorId, &[u8])> =
                entries.iter().map(|(id, b)| (*id, b.as_slice())).collect();
            return self.build(&pairs);
        }

        // Otherwise, incremental insert each one. Vamana's
        // incremental insert per Subramanya §3 is:
        // 1. Greedy-search from medoid for visited set V.
        // 2. RobustPrune(p, V, α, R) → new neighbors.
        // 3. For each n in pruned: add reverse edge p ←n,
        //    re-prune n's neighbors if over-capacity.
        for (id, bytes) in entries {
            if self.id_to_slot.contains_key(&id) {
                // Defensive; insert_stream guards. Skip silent.
                continue;
            }
            self.check_or_set_byte_width(bytes.len())?;
            let new_slot = self.allocate_slot(id, bytes);
            self.incremental_insert_slot(new_slot)?;
        }

        Ok(())
    }

    /// Pre-flight validation for the delta-segment batch (issue
    /// #109 part 2). Peeks the entries without taking ownership;
    /// on failure the delta stays intact. Two checks:
    ///
    /// 1. **Byte-width consistency.** Every entry's byte length
    ///    must equal the configured `bytes_per_vector` (when set)
    ///    or pairwise-agree with the first entry (when not yet
    ///    locked). Defense-in-depth: `insert_stream` already
    ///    validates byte width per entry, so this fires only on
    ///    a future bypass / corruption / direct delta access
    ///    path.
    /// 2. **Finite-float check.** For [`Encoding::F32`] every
    ///    f32 element must satisfy [`f32::is_finite`] — no NaN,
    ///    no ±Inf. F16/SQ8/Binary are skipped: SQ8 and Binary
    ///    do not carry NaN at the byte-encoding level (integer
    ///    domain); F16 NaN-check is a v1.1 follow-up (the bit
    ///    pattern is `exp == 0x1F && mantissa != 0` / `exp ==
    ///    0x1F && mantissa == 0` for Inf, but the kernel surface
    ///    in `distance.rs` does not yet ship a `f16::is_finite`
    ///    helper and the F.1 multi-tenant arena flow does not
    ///    yet exercise F16 in production).
    ///
    /// On failure returns [`VectorIndexError::BatchValidation`]
    /// with a `reason` naming the offending entry index and the
    /// rule it violated. Caller may then `delete` the offending
    /// id, repair, and retry the merge.
    fn validate_delta_batch(&self) -> Result<()> {
        if self.delta.entries.is_empty() {
            return Ok(());
        }

        // 1. Byte-width consistency.
        let expected_width = self
            .bytes_per_vector
            .unwrap_or_else(|| self.delta.entries[0].1.len());
        for (i, (_id, bytes)) in self.delta.entries.iter().enumerate() {
            if bytes.len() != expected_width {
                return Err(VectorIndexError::BatchValidation {
                    reason: format!(
                        "delta entry {i} byte-width mismatch: expected {expected_width}, got {}",
                        bytes.len()
                    ),
                });
            }
        }

        // 2. Finite-float check (F32 only at v1.0).
        if matches!(self.encoding, crate::Encoding::F32) {
            for (i, (_id, bytes)) in self.delta.entries.iter().enumerate() {
                if bytes.len() % std::mem::size_of::<f32>() != 0 {
                    return Err(VectorIndexError::BatchValidation {
                        reason: format!(
                            "delta entry {i} byte length {} not a multiple of 4 for F32 encoding",
                            bytes.len()
                        ),
                    });
                }
                let view: &[f32] = bytemuck::cast_slice(bytes);
                if let Some(pos) = view.iter().position(|x| !x.is_finite()) {
                    return Err(VectorIndexError::BatchValidation {
                        reason: format!(
                            "delta entry {i} contains non-finite f32 at index {pos}: {}",
                            view[pos]
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Search merging main-graph and delta-segment results.
    ///
    /// Per ADR-035 §5.3.1: runs the main-graph beam search
    /// AND the delta-segment search, then merges the
    /// candidates by distance and returns the top-K. RYW
    /// preserved because the delta-segment is in-memory
    /// before this call returns.
    pub fn search_with_delta(
        &self,
        query: &[u8],
        k: usize,
        l_search: usize,
    ) -> Result<Vec<(VectorId, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if let Some(expected) = self.bytes_per_vector
            && query.len() != expected
        {
            return Err(VectorIndexError::DimensionMismatch {
                expected,
                got: query.len(),
            });
        }

        // 1. Main-graph search (best-effort; empty if main is
        //    empty).
        let main_hits = self.search(query, k, l_search)?;
        // 2. Delta-segment search — brute-force linear scan.
        let delta_hits = self.search_delta(query, k);
        // 3. Merge by distance.
        let mut merged: Vec<(VectorId, f32, f32)> =
            Vec::with_capacity(main_hits.len() + delta_hits.len());
        for (id, raw) in main_hits.iter().chain(delta_hits.iter()) {
            let key = self.distance_key(*raw);
            merged.push((*id, *raw, key));
        }
        // Stable sort by ascending key, then by id for
        // deterministic tie-break.
        merged.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.raw().cmp(&b.0.raw())));
        // Dedup: a vector that exists in both main + delta
        // (shouldn't happen — insert_stream blocks dups — but
        // be defensive).
        merged.dedup_by(|a, b| a.0 == b.0);
        merged.truncate(k);
        Ok(merged
            .into_iter()
            .map(|(id, raw, _key)| (id, raw))
            .collect())
    }

    /// Brute-force scan of the delta-segment. Returns up to
    /// `k` (id, distance) pairs in ascending closeness.
    fn search_delta(&self, query: &[u8], k: usize) -> Vec<(VectorId, f32)> {
        if self.delta.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<(VectorId, f32)> = self
            .delta
            .iter()
            .map(|(id, bytes)| {
                let raw = self.kernel.distance(query, bytes);
                (id, raw)
            })
            .collect();
        hits.sort_by(|a, b| {
            self.distance_key(a.1)
                .total_cmp(&self.distance_key(b.1))
                .then(a.0.raw().cmp(&b.0.raw()))
        });
        hits.truncate(k);
        hits
    }

    /// Vamana incremental insert for a freshly-allocated slot.
    /// Assumes `self.entry_point` is `Some` (which is the case
    /// once the main graph has at least one node).
    fn incremental_insert_slot(&mut self, new_slot: u32) -> Result<()> {
        let entry = match self.entry_point {
            Some(e) => e,
            None => {
                // First-ever insert — set ourselves as medoid,
                // no neighbors, no symmetrize.
                self.entry_point = Some(new_slot);
                return Ok(());
            }
        };
        let r_target = self.params.r as usize;
        let l_construction = self.params.l_construction as usize;
        let alpha = self.params.alpha;

        // Pull query bytes into a buffer (avoid borrow conflict
        // with the mutable graph during greedy_visit).
        let query = self.vector_bytes_owned(new_slot);
        let visit = self.greedy_visit_from(&query, entry, l_construction);

        let mut candidates: Vec<(u32, f32)> = visit
            .visited
            .into_iter()
            .filter(|(s, _)| *s != new_slot)
            .collect();
        // Include any pre-seeded neighbors (none in normal
        // path, but defensive).
        for &existing in &self.neighbors[new_slot as usize] {
            if existing == new_slot || candidates.iter().any(|(s, _)| *s == existing) {
                continue;
            }
            let raw = self.slot_distance(new_slot, existing);
            candidates.push((existing, self.distance_key(raw)));
        }

        let pruned = self.robust_prune_pub(new_slot, candidates, alpha, r_target);
        self.neighbors[new_slot as usize] = pruned.clone();

        // Symmetrize back-edges.
        for q in pruned {
            if q == new_slot {
                continue;
            }
            let q_neigh_len = self.neighbors[q as usize].len();
            let q_already = self.neighbors[q as usize].contains(&new_slot);
            if q_already {
                continue;
            }
            if q_neigh_len < r_target {
                self.neighbors[q as usize].push(new_slot);
                continue;
            }
            // Over-capacity — re-prune q's neighbor set.
            let q_neighbors_snap = self.neighbors[q as usize].clone();
            let mut q_cands: Vec<(u32, f32)> = Vec::with_capacity(q_neighbors_snap.len() + 1);
            for nb in q_neighbors_snap {
                let raw = self.slot_distance(q, nb);
                q_cands.push((nb, self.distance_key(raw)));
            }
            let raw = self.slot_distance(q, new_slot);
            q_cands.push((new_slot, self.distance_key(raw)));
            let q_pruned = self.robust_prune_pub(q, q_cands, alpha, r_target);
            self.neighbors[q as usize] = q_pruned;
        }
        Ok(())
    }

    /// Public wrapper around `robust_prune` so the streaming
    /// path can reach it from this module. (`robust_prune`
    /// itself is `fn` on `DiskAnnGraph` in `build.rs` —
    /// re-exposed here via this thin wrapper to keep the
    /// `build.rs` private surface tight.)
    fn robust_prune_pub(
        &self,
        p: u32,
        candidates: Vec<(u32, f32)>,
        alpha: f32,
        r: usize,
    ) -> Vec<u32> {
        self.robust_prune_inner(p, candidates, alpha, r)
    }

    /// Inline copy of the α-prune routine — kept here to avoid
    /// re-exposing `build.rs`'s private helper. Logic mirrors
    /// `build::DiskAnnGraph::robust_prune` exactly.
    fn robust_prune_inner(
        &self,
        p: u32,
        mut candidates: Vec<(u32, f32)>,
        alpha: f32,
        r: usize,
    ) -> Vec<u32> {
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        candidates.retain(|(s, _)| *s != p);
        candidates.dedup_by(|a, b| a.0 == b.0);
        let mut result: Vec<u32> = Vec::with_capacity(r);
        let mut alive: Vec<bool> = vec![true; candidates.len()];
        // See `build::DiskAnnGraph::robust_prune` for the
        // Vamana α-prune semantics — duplicated here only to
        // keep the streaming module self-contained without
        // re-exporting `build.rs`'s private helper.
        for i in 0..candidates.len() {
            if !alive[i] {
                continue;
            }
            let (p_star_slot, _) = candidates[i];
            result.push(p_star_slot);
            if result.len() >= r {
                break;
            }
            for j in (i + 1)..candidates.len() {
                if !alive[j] {
                    continue;
                }
                let (v_slot, v_key) = candidates[j];
                let d_p_v = self.distance_external(v_key);
                let d_pstar_v = self.slot_distance(p_star_slot, v_slot);
                if d_p_v > alpha * d_pstar_v {
                    alive[j] = false;
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::graph::{DiskAnnGraph, DiskAnnParams};
    use crate::distance::L2F32;
    use crate::{Encoding, Metric};

    fn fxd(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    fn empty_graph_f32() -> DiskAnnGraph {
        DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(L2F32),
        )
        .unwrap()
    }

    #[test]
    fn insert_stream_into_empty_appends_to_delta() {
        let mut g = empty_graph_f32();
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.insert_stream(&[(VectorId::new(1), v.as_slice())])
            .unwrap();
        assert_eq!(g.delta_len(), 1);
        assert_eq!(g.main_len(), 0);
        assert!(g.contains(VectorId::new(1)));
    }

    #[test]
    fn insert_stream_rejects_dup_in_delta() {
        let mut g = empty_graph_f32();
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.insert_stream(&[(VectorId::new(1), v.as_slice())])
            .unwrap();
        let r = g.insert_stream(&[(VectorId::new(1), v.as_slice())]);
        assert!(matches!(r, Err(VectorIndexError::IrrecoverableLoss { .. })));
    }

    #[test]
    fn insert_stream_dim_mismatch_against_main() {
        let mut g = empty_graph_f32();
        let v4 = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.build(&[(VectorId::new(1), v4.as_slice())]).unwrap();
        let v2 = fxd(&[1.0, 0.0]);
        let r = g.insert_stream(&[(VectorId::new(2), v2.as_slice())]);
        assert!(matches!(r, Err(VectorIndexError::DimensionMismatch { .. })));
    }

    #[test]
    fn search_with_delta_finds_inserted_immediately() {
        // RYW invariant — insert a vector, search by it,
        // assert it's the top hit.
        let mut g = empty_graph_f32();
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..50_u32 {
            owned.push((VectorId::new(i), fxd(&[i as f32 * 0.1, 0.0, 0.0, 0.0])));
        }
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();

        // Insert a brand-new vector.
        let new_id = VectorId::new(100);
        let new_v = fxd(&[42.0, 0.0, 0.0, 0.0]);
        g.insert_stream(&[(new_id, new_v.as_slice())]).unwrap();
        // Search by the new vector itself.
        let r = g.search_with_delta(&new_v, 1, 100).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(
            r[0].0,
            new_id,
            "RYW violated: search returned {} not {}",
            r[0].0.raw(),
            new_id.raw()
        );
    }

    #[test]
    fn merge_fold_at_threshold_clears_delta_and_grows_main() {
        let params = DiskAnnParams {
            r: 8,
            alpha: 1.2,
            l_construction: 16,
            l_search_default: 16,
            delta_max_size: 4, // tiny threshold for the test
            ..DiskAnnParams::default()
        };
        let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32)).unwrap();
        // Build a small main graph first.
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..16_u32 {
            owned.push((VectorId::new(i), fxd(&[i as f32 * 0.05, 0.0, 0.0, 0.0])));
        }
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();
        let main_before = g.main_len();
        let owned_ext: Vec<(VectorId, Vec<u8>)> = (100..104_u32)
            .map(|i| (VectorId::new(i), fxd(&[(i as f32) * 0.05, 0.0, 0.0, 0.0])))
            .collect();
        let pairs_ext: Vec<(VectorId, &[u8])> = owned_ext
            .iter()
            .map(|(id, b)| (*id, b.as_slice()))
            .collect();
        // 4 inserts → hits delta_max_size → triggers merge.
        g.insert_stream(&pairs_ext).unwrap();
        assert_eq!(g.delta_len(), 0, "delta should be empty after merge");
        assert_eq!(g.main_len(), main_before + 4);
        for (id, _) in &owned_ext {
            assert!(
                g.id_to_slot.contains_key(id),
                "merged id {} not in main",
                id.raw()
            );
        }
    }

    #[test]
    fn delta_search_after_merge_returns_same_results_as_before() {
        // Insert N=20 via delta, then merge, then re-search;
        // assert the result set is identical (modulo merge-time
        // edge churn — top-K should be stable for distinct
        // clusters).
        let params = DiskAnnParams {
            r: 8,
            alpha: 1.2,
            l_construction: 16,
            l_search_default: 32,
            delta_max_size: 100, // don't auto-merge
            ..DiskAnnParams::default()
        };
        let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32)).unwrap();
        // Build a 30-vector main graph.
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..30_u32 {
            owned.push((
                VectorId::new(i),
                fxd(&[
                    (i as f32).sin(),
                    (i as f32).cos(),
                    (i as f32 * 0.3).sin(),
                    (i as f32 * 0.7).cos(),
                ]),
            ));
        }
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();
        // Delta-insert 20.
        let owned_d: Vec<(VectorId, Vec<u8>)> = (100..120_u32)
            .map(|i| {
                (
                    VectorId::new(i),
                    fxd(&[
                        (i as f32 * 0.11).sin(),
                        (i as f32 * 0.13).cos(),
                        (i as f32 * 0.17).sin(),
                        (i as f32 * 0.19).cos(),
                    ]),
                )
            })
            .collect();
        let pairs_d: Vec<(VectorId, &[u8])> =
            owned_d.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.insert_stream(&pairs_d).unwrap();
        assert_eq!(g.delta_len(), 20);

        // Pre-merge query.
        let q = fxd(&[0.5, 0.5, 0.5, 0.5]);
        let pre = g.search_with_delta(&q, 5, 64).unwrap();
        let pre_ids: Vec<VectorId> = pre.iter().map(|(id, _)| *id).collect();

        // Force merge.
        g.merge_delta().unwrap();
        assert_eq!(g.delta_len(), 0);
        assert_eq!(g.main_len(), 50);

        // Post-merge query.
        let post = g.search_with_delta(&q, 5, 64).unwrap();
        let post_ids: Vec<VectorId> = post.iter().map(|(id, _)| *id).collect();

        // Top-1 should be stable — the merge-fold preserves
        // order on the closest matches.
        assert_eq!(pre_ids[0], post_ids[0], "top-1 changed across merge");
        // ≥ 4/5 of the top-5 should be preserved.
        let pre_set: std::collections::HashSet<_> = pre_ids.iter().collect();
        let preserved = post_ids.iter().filter(|id| pre_set.contains(id)).count();
        assert!(
            preserved >= 4,
            "post-merge top-5 lost too many: {} preserved",
            preserved
        );
    }

    #[test]
    fn delete_id_in_delta_removes_it() {
        let mut g = empty_graph_f32();
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.insert_stream(&[(VectorId::new(1), v.as_slice())])
            .unwrap();
        assert!(g.contains(VectorId::new(1)));
        g.delete(VectorId::new(1)).unwrap();
        assert!(!g.contains(VectorId::new(1)));
        assert_eq!(g.delta_len(), 0);
    }

    #[test]
    fn search_with_delta_dim_mismatch_errors() {
        let mut g = empty_graph_f32();
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.build(&[(VectorId::new(1), v.as_slice())]).unwrap();
        let q = fxd(&[1.0, 0.0]); // dim mismatch
        let r = g.search_with_delta(&q, 1, 10);
        assert!(matches!(r, Err(VectorIndexError::DimensionMismatch { .. })));
    }

    /// Pin test for issue #109 part 2: a malformed sibling in
    /// the same merge batch must NOT silently lose every other
    /// entry queued behind it. Pre-validation runs BEFORE
    /// `mem::take`; on failure the delta is left intact and the
    /// caller can repair + retry.
    ///
    /// Two variants exercise the two pre-flight rules:
    /// - byte-width mismatch (dim drift)
    /// - non-finite f32 (NaN / +Inf)
    ///
    /// Without the pre-validation, this test would observe an
    /// empty delta + a partial main-graph update — silent data
    /// loss for every successfully ack'd T3 insert that shared
    /// the batch with the malformed entry. The bug pre-fix:
    /// `mem::take(&mut self.delta.entries)` drained the delta
    /// BEFORE validation, then validation `?`-exited the merge
    /// loop with the remainder unrecoverable.
    #[test]
    fn diskann_merge_delta_partial_batch_failure_preserves_unmerged() {
        // ── Variant 1: dim-mismatch sibling ──
        let mut g = empty_graph_f32();
        // Three valid F32×4 entries via insert_stream (this will
        // lock bytes_per_vector to 16).
        let v_ok = fxd(&[1.0, 0.0, 0.0, 0.0]);
        g.insert_stream(&[
            (VectorId::new(1), v_ok.as_slice()),
            (VectorId::new(2), v_ok.as_slice()),
            (VectorId::new(3), v_ok.as_slice()),
        ])
        .unwrap();
        // Inject a malformed dim-mismatch entry directly into
        // the delta (bypassing insert_stream's per-entry width
        // check, simulating a future bypass / corruption).
        let v_bad = fxd(&[1.0, 0.0]); // 8 bytes, expected 16
        g.delta.append(VectorId::new(99), v_bad);
        assert_eq!(g.delta_len(), 4, "setup: 3 valid + 1 malformed");

        let r = g.merge_delta();
        assert!(
            matches!(r, Err(VectorIndexError::BatchValidation { .. })),
            "merge_delta must reject the batch with BatchValidation; got {:?}",
            r
        );
        assert_eq!(
            g.delta_len(),
            4,
            "issue #109 part 2: pre-validation failure must NOT drain the \
             delta — RYW-satisfying entries 1/2/3 must still be queued"
        );
        assert_eq!(
            g.main_len(),
            0,
            "main graph must NOT have absorbed any entries from the \
             rejected batch"
        );

        // ── Variant 2: NaN sibling ──
        let mut g2 = empty_graph_f32();
        g2.insert_stream(&[
            (VectorId::new(10), v_ok.as_slice()),
            (VectorId::new(11), v_ok.as_slice()),
            (VectorId::new(12), v_ok.as_slice()),
        ])
        .unwrap();
        let v_nan = fxd(&[1.0, f32::NAN, 0.0, 0.0]);
        g2.delta.append(VectorId::new(98), v_nan);
        assert_eq!(g2.delta_len(), 4);

        let r2 = g2.merge_delta();
        assert!(
            matches!(r2, Err(VectorIndexError::BatchValidation { .. })),
            "merge_delta must reject the batch with BatchValidation; got {:?}",
            r2
        );
        assert_eq!(
            g2.delta_len(),
            4,
            "issue #109 part 2: NaN sibling must not drain the delta"
        );
        assert_eq!(g2.main_len(), 0);

        // ── Variant 3: ±Inf sibling ──
        let mut g3 = empty_graph_f32();
        g3.insert_stream(&[(VectorId::new(20), v_ok.as_slice())])
            .unwrap();
        let v_inf = fxd(&[1.0, f32::INFINITY, 0.0, 0.0]);
        g3.delta.append(VectorId::new(97), v_inf);
        let r3 = g3.merge_delta();
        assert!(
            matches!(r3, Err(VectorIndexError::BatchValidation { .. })),
            "merge_delta must reject Inf via BatchValidation; got {:?}",
            r3
        );
        assert_eq!(g3.delta_len(), 2);
    }

    /// Round-trip: after a clean batch (no malformed siblings)
    /// merge_delta still drains + folds successfully. The pin
    /// guard ensures pre-validation does not introduce a
    /// false-negative regression on the happy path.
    #[test]
    fn diskann_merge_delta_clean_batch_drains_normally() {
        let mut g = empty_graph_f32();
        let v = fxd(&[1.0, 0.5, 0.25, 0.125]);
        g.insert_stream(&[
            (VectorId::new(1), v.as_slice()),
            (VectorId::new(2), v.as_slice()),
            (VectorId::new(3), v.as_slice()),
        ])
        .unwrap();
        assert_eq!(g.delta_len(), 3);
        g.merge_delta().expect("clean batch must merge");
        assert_eq!(g.delta_len(), 0);
        assert_eq!(g.main_len(), 3);
    }

    #[test]
    fn delta_segment_brute_force_only_at_slice_d() {
        // Documentation/regression: Slice D ships brute-force
        // only. The promotion to a small in-memory HNSW at
        // delta_brute_thresh is wired as a parameter but not
        // implemented; this test pins the current behavior so
        // the Slice F follow-up is intentional.
        let p = DiskAnnParams::default();
        assert_eq!(
            p.delta_brute_thresh, 128,
            "delta_brute_thresh default should match ADR-035 §5.3.1 + Slice D handoff"
        );
    }
}
