//! Vamana α-pruning bulk build (Algorithm 1, Subramanya et al. NeurIPS 2019).
//!
//! Two passes over the points (`α = 1.0` then `α = params.alpha`)
//! per the original algorithm — the first pass forms the
//! Greedy-Nearest-Neighbor (GNN) graph, the second adds α-pruned
//! "long edges" that improve recall on out-of-distribution
//! queries.
//!
//! ## Latency / memory budget
//!
//! Build is `O(n · L_construction · log L)` distance ops
//! amortized; at `n = 1 M`, `L_construction = 100`, `R = 70`
//! the inner-loop kernel call count is ~`n × L × 2 passes ×
//! avg_neighbor_count` ≈ 1.4 × 10⁹, which at a simsimd-dispatched
//! L2/F32 d=128 throughput of ~40 ns/op fits in ~60 s on
//! c7i.2xlarge (single-thread). Slice D ships single-thread;
//! parallelization (rayon over the build permutation) is a
//! Slice F.5 follow-up.
//!
//! Memory: `n × R × 4 B` for the edge list + `n × bytes_per_vector`
//! for the vector array. At `n = 1 M`, `R = 70`, `dim = 128`,
//! `f32` encoding: ~280 MB edges + ~512 MB vectors = ~800 MB.
//! Within the c7i.2xlarge 16 GB envelope.
//!
//! ## Determinism
//!
//! Build is deterministic given a fixed input order: the
//! random permutation in the original Vamana paper is
//! seeded — we use a simple xorshift PRNG with a fixed seed
//! so the same inputs produce the same graph (essential for
//! the snapshot path's CRC stability in Slice G.2).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::{Result, VectorId, VectorIndexError};

use super::graph::DiskAnnGraph;

/// Completed nodes between throttled Vamana build-progress logs. A 10M build
/// emits ~100 progress lines per pass (one per 100K completed nodes) — enough
/// for a live ETA on a multi-hour build, cheap enough that the per-node cost is
/// a single relaxed `AtomicUsize::fetch_add` + a modulo compare (no allocation,
/// no formatting, no lock on the non-logging path).
const PROGRESS_LOG_INTERVAL: usize = 100_000;

/// Record one completed Vamana node and, every [`PROGRESS_LOG_INTERVAL`]
/// completions (plus the final node, so even a sub-interval build logs once per
/// pass), emit a throttled `tracing::info!` carrying a derived rate + ETA so an
/// operator can watch an otherwise-opaque multi-hour 10M build.
///
/// This is the build's ONLY per-node instrumentation. The hot-path cost is the
/// single `fetch_add` with [`Ordering::Relaxed`]: the count is monotone and is
/// read only here (the throttle + final emit), so there is no happens-before
/// dependency on other memory and `Relaxed` is sufficient. The `start.elapsed()`
/// + rate/ETA arithmetic + `tracing::info!` only run on the throttled path.
#[inline]
fn vamana_progress_tick(
    counter: &AtomicUsize,
    pass_num: u8,
    alpha: f32,
    total: usize,
    start: Instant,
) {
    let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
    // Fast path: nothing to do unless this is a throttle boundary or the final
    // node of the pass. Keeps the per-node cost to the atomic + this compare.
    if done % PROGRESS_LOG_INTERVAL != 0 && done != total {
        return;
    }
    let elapsed_secs = start.elapsed().as_secs_f64();
    let nodes_per_sec = if elapsed_secs > 0.0 {
        done as f64 / elapsed_secs
    } else {
        0.0
    };
    let eta_secs = if nodes_per_sec > 0.0 {
        // `done <= total` always (we emit at most once at `done == total`), so
        // `total - done` never underflows.
        (total - done) as f64 / nodes_per_sec
    } else {
        0.0
    };
    // Test-only probe mirroring this emission for the unit test's deterministic
    // oracle (see `record_progress_probe`); compiles to nothing in release.
    #[cfg(test)]
    record_progress_probe(pass_num, done, total);
    tracing::info!(
        pass = pass_num,
        alpha = alpha as f64,
        nodes_done = done,
        total,
        elapsed_secs,
        nodes_per_sec,
        eta_secs,
        "diskann vamana build progress"
    );
}

// Test-only thread-local mirror of throttled progress emissions:
// (pass_num, nodes_done, total) per emission, recorded at the exact point (and
// only when) `vamana_progress_tick` fires its `tracing::info!`. The unit test
// drains this for a deterministic, concurrency-safe oracle — tracing's
// process-global per-callsite Interest cache makes a per-thread capture
// subscriber unreliable when sibling tests hit these callsites first under the
// no-op global subscriber. `#[cfg(test)]`, so it is absent from release builds.
#[cfg(test)]
thread_local! {
    static PROGRESS_PROBE: std::cell::RefCell<Vec<(u8, usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn record_progress_probe(pass_num: u8, nodes_done: usize, total: usize) {
    PROGRESS_PROBE.with(|p| p.borrow_mut().push((pass_num, nodes_done, total)));
}

impl DiskAnnGraph {
    /// Bulk-build the Vamana graph from `(VectorId, &[u8])`
    /// pairs. Replaces any existing main-graph state and resets
    /// the delta-segment.
    ///
    /// Per Subramanya et al. NeurIPS 2019 Algorithm 1. `R`,
    /// `alpha`, and `L_construction` come from
    /// `self.params` (configured at [`DiskAnnGraph::new`]).
    ///
    /// Returns:
    /// - [`VectorIndexError::DimensionMismatch`] if any vector
    ///   byte length does not match the configured `bytes_per_vector`,
    ///   or if the input contains duplicate `VectorId`s.
    pub fn build(&mut self, vectors: &[(VectorId, &[u8])]) -> Result<()> {
        // Delegate to the owned-input path. This entry COPIES the borrowed
        // bytes; [`DiskAnnGraph::build_owned`] MOVES them. The bounded
        // SSD-resident build (ADR-195 §3) uses `build_owned` to avoid a
        // transient 2× copy of the in-RAM SQ8 array (7.68 GB vs 15.36 GB at
        // 10M×768d). Behaviour is otherwise identical — same passes, same
        // determinism, same errors (a parity test asserts byte-equal graphs).
        let owned: Vec<(VectorId, Vec<u8>)> =
            vectors.iter().map(|(id, b)| (*id, b.to_vec())).collect();
        self.build_owned(owned)
    }

    /// Bulk-build the Vamana graph by **moving** owned `(VectorId, Vec<u8>)`
    /// byte payloads into the parallel arrays (no per-vector copy). Identical
    /// to [`DiskAnnGraph::build`] in every observable way — same medoid, same
    /// seeded permutation, same α-prune passes, same error taxonomy — but the
    /// ingest avoids cloning. The SSD-resident serving tier (ADR-195 §3) builds
    /// with this so the SQ8 nav array is not transiently doubled at 10M scale.
    pub fn build_owned(&mut self, vectors: Vec<(VectorId, Vec<u8>)>) -> Result<()> {
        if let Some((medoid, mut prng)) = self.reset_ingest_init(vectors)? {
            // Two refinement passes: α = 1.0 (GNN graph) then α = params.alpha
            // (long-edge augmentation).
            self.vamana_pass(medoid, 1.0_f32, &mut prng, 1);
            if self.params.alpha > 1.0 {
                self.vamana_pass(medoid, self.params.alpha, &mut prng, 2);
            }
        }
        Ok(())
    }

    /// Bulk-build like [`DiskAnnGraph::build_owned`] but with the Vamana
    /// refinement passes PARALLELISED over the build permutation (ADR-195 §3 /
    /// F.5 #112) so the 10M GA-gate build is iterable (single-thread @100K×768
    /// ≈ 481 s → hours at 10M otherwise).
    ///
    /// Each batch of `batch_size` nodes computes its forward edges IN PARALLEL
    /// against a read-only snapshot of the current graph (the expensive
    /// greedy-search and α-prune — both `&self`), then APPLIES them (forward
    /// edges, then back-edge symmetrisation) SEQUENTIALLY. There are no data
    /// races by construction: the parallel phase only reads, the apply phase
    /// only mutates. `batch_size` trades parallelism for sequential-fidelity:
    /// smaller batches see more intra-pass updates (closer to the sequential
    /// graph, higher recall); larger batches parallelise more. A recall-parity test
    /// asserts the parallel build matches the sequential build's recall within
    /// tolerance (the strong oracle against a silent quality regression).
    pub fn build_owned_parallel(
        &mut self,
        vectors: Vec<(VectorId, Vec<u8>)>,
        batch_size: usize,
    ) -> Result<()> {
        if let Some((medoid, mut prng)) = self.reset_ingest_init(vectors)? {
            self.vamana_pass_parallel(medoid, 1.0_f32, &mut prng, batch_size, 1);
            if self.params.alpha > 1.0 {
                self.vamana_pass_parallel(medoid, self.params.alpha, &mut prng, batch_size, 2);
            }
        }
        Ok(())
    }

    /// Reset state, MOVE-ingest the owned byte payloads, choose the medoid, and
    /// random-init the neighbour lists — the shared prelude of the sequential
    /// and parallel builds. Returns `Some((medoid, prng))` to run the refinement
    /// passes, or `None` for the already-finalised trivial graphs (empty /
    /// single vector). Determinism is preserved (the prng is seeded by `n`).
    fn reset_ingest_init(
        &mut self,
        vectors: Vec<(VectorId, Vec<u8>)>,
    ) -> Result<Option<(u32, XorShift32)>> {
        // Reset: build is wholesale; existing state is dropped.
        self.ids.clear();
        self.vectors.clear();
        self.neighbors.clear();
        self.id_to_slot.clear();
        self.entry_point = None;
        self.delta = super::stream::DeltaSegment::new();
        self.label_idx = super::graph::LabelIndex::default();
        self.tombstones.clear();
        self.bytes_per_vector = None;
        // Per ADR-041 §D-3a: LSN parallel arrays reset alongside
        // the slot-indexed state. `allocate_slot` re-grows them
        // in lockstep with default (Lsn::ZERO, Lsn::MAX) values.
        self.commit_lsns.clear();
        self.expired_lsns.clear();
        self.delta_lsns.clear();

        if vectors.is_empty() {
            return Ok(None);
        }

        // Validate + ingest into parallel arrays (MOVE the bytes).
        for (id, bytes) in vectors {
            self.check_or_set_byte_width(bytes.len())?;
            if self.id_to_slot.contains_key(&id) {
                return Err(VectorIndexError::IrrecoverableLoss {
                    index: crate::IndexId::ZERO,
                    reason: format!("duplicate VectorId {} in build input", id.raw()),
                });
            }
            self.allocate_slot(id, bytes);
        }
        let n = self.ids.len();

        // Trivial case: a single node is its own entry point.
        if n == 1 {
            self.entry_point = Some(0);
            return Ok(None);
        }

        // Step 1: choose entry-point medoid via stride-sampled
        // approximation. The medoid is the vector minimizing
        // sum-of-distances to a representative sample.
        let all_slots: Vec<u32> = (0..n as u32).collect();
        let medoid = self.medoid_within(&all_slots);
        self.entry_point = Some(medoid);

        // Step 2: random init — assign R random neighbors per
        // node. We avoid pulling rand as a workspace dep by
        // using a deterministic xorshift; the first build pass
        // overwrites these edges anyway.
        let r_target = self.params.r as usize;
        let mut prng = XorShift32::seed(0xC0FF_EE42 ^ n as u32);
        for slot in 0..n {
            let mut chosen: Vec<u32> = Vec::with_capacity(r_target.min(n - 1));
            let mut attempts = 0usize;
            // Sample R distinct neighbors uniformly. For tiny
            // n (<= R+1) just take all other slots.
            if n <= r_target + 1 {
                for other in 0..n {
                    if other != slot {
                        chosen.push(other as u32);
                    }
                }
            } else {
                while chosen.len() < r_target && attempts < r_target * 16 {
                    let pick = (prng.next_u32() as usize) % n;
                    let pick_u32 = pick as u32;
                    if pick != slot && !chosen.contains(&pick_u32) {
                        chosen.push(pick_u32);
                    }
                    attempts += 1;
                }
            }
            self.neighbors[slot] = chosen;
        }

        Ok(Some((medoid, prng)))
    }

    /// One Vamana refinement pass — for each node in a random
    /// permutation, run greedy-search to get the visited set,
    /// α-prune to R neighbors, then symmetrize.
    fn vamana_pass(&mut self, medoid: u32, alpha: f32, prng: &mut XorShift32, pass_num: u8) {
        let n = self.ids.len();
        let r_target = self.params.r as usize;
        let l_construction = self.params.l_construction as usize;
        let bytes_per_vector = self
            .bytes_per_vector
            .expect("bytes_per_vector set before refinement");

        // Build-progress instrumentation (observability only — no effect on the
        // graph): pass-start timer + completed-node counter feeding the throttled
        // ETA log. `pass_num` (1 = α 1.0 GNN graph, 2 = α params.alpha long-edge
        // augmentation) lets an operator tell the two refinement passes apart.
        let pass_start = Instant::now();
        let progress = AtomicUsize::new(0);
        tracing::info!(
            pass = pass_num,
            alpha = alpha as f64,
            total = n,
            "diskann vamana build pass start"
        );

        // Random permutation of slots (Fisher-Yates).
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = (prng.next_u32() as usize) % (i + 1);
            perm.swap(i, j);
        }

        // Reusable buffers — avoid per-iteration allocs in the
        // hot loop.
        let mut query_buf: Vec<u8> = Vec::with_capacity(bytes_per_vector);

        for &p in &perm {
            // Pull query bytes into a buffer (cannot borrow
            // `self` across the mutable greedy-search path
            // because greedy-search reads other vectors).
            query_buf.clear();
            query_buf.extend_from_slice(self.vector_bytes(p));

            // Greedy-search from medoid for the visited set V.
            let visit = self.greedy_visit_from(&query_buf, medoid, l_construction);
            let mut candidates: Vec<(u32, f32)> =
                Vec::with_capacity(visit.visited.len() + r_target);
            for (slot, key) in visit.visited {
                if slot == p {
                    continue;
                }
                candidates.push((slot, key));
            }
            // Include p's existing neighbors (with re-computed
            // keys from current state). The prune fold-in
            // ensures we don't lose recall when a previously
            // good edge was dominated by a transiently-better
            // candidate during pass 1.
            for &existing in &self.neighbors[p as usize] {
                if existing == p || candidates.iter().any(|(s, _)| *s == existing) {
                    continue;
                }
                let raw = self.slot_distance(p, existing);
                let key = self.distance_key(raw);
                candidates.push((existing, key));
            }

            // α-prune.
            let pruned = self.robust_prune(p, candidates, alpha, r_target);
            self.neighbors[p as usize] = pruned.clone();

            // Symmetrize: add reverse edges from each pruned neighbor back to
            // p (re-pruning when the back-edge would push q past R).
            for q in pruned {
                self.symmetrize_back_edge(p, q, alpha, r_target);
            }

            // Node complete — record it (single relaxed atomic increment; the
            // throttled log fires only every PROGRESS_LOG_INTERVAL + final node).
            vamana_progress_tick(&progress, pass_num, alpha, n, pass_start);
        }

        tracing::info!(
            pass = pass_num,
            alpha = alpha as f64,
            nodes_done = progress.load(Ordering::Relaxed),
            total = n,
            elapsed_secs = pass_start.elapsed().as_secs_f64(),
            "diskann vamana build pass complete"
        );
    }

    /// Add the reverse edge `q → p` for a freshly-pruned forward edge `p → q`,
    /// re-pruning `q`'s neighbour set (including `p`) when `q` is already at the
    /// `r_target` degree cap. Shared by the sequential and parallel refinement
    /// passes' (sequential) apply phase. No-op when `q == p` or the edge exists.
    fn symmetrize_back_edge(&mut self, p: u32, q: u32, alpha: f32, r_target: usize) {
        if q == p {
            return;
        }
        // Decision-only borrow of q's neighbor list: peek at length + membership
        // without holding a mutable borrow across the distance calls.
        let q_neigh_len = self.neighbors[q as usize].len();
        if self.neighbors[q as usize].contains(&p) {
            return;
        }
        if q_neigh_len < r_target {
            self.neighbors[q as usize].push(p);
            return;
        }
        // Over capacity — re-prune q's neighbor set including p. Snapshot the
        // neighbor list so we can iterate while computing distances on `self`.
        let q_neighbors_snap = self.neighbors[q as usize].clone();
        let mut q_cands: Vec<(u32, f32)> = Vec::with_capacity(q_neighbors_snap.len() + 1);
        for nb in q_neighbors_snap {
            let raw = self.slot_distance(q, nb);
            q_cands.push((nb, self.distance_key(raw)));
        }
        let raw = self.slot_distance(q, p);
        q_cands.push((p, self.distance_key(raw)));
        let q_pruned = self.robust_prune(q, q_cands, alpha, r_target);
        self.neighbors[q as usize] = q_pruned;
    }

    /// Parallel analogue of [`DiskAnnGraph::vamana_pass`] (ADR-195 §3 / #112):
    /// each batch's forward edges are computed in parallel against a read-only
    /// snapshot, then applied (+ symmetrised) sequentially. See
    /// [`DiskAnnGraph::build_owned_parallel`] for the rationale + tradeoff.
    fn vamana_pass_parallel(
        &mut self,
        medoid: u32,
        alpha: f32,
        prng: &mut XorShift32,
        batch_size: usize,
        pass_num: u8,
    ) {
        let n = self.ids.len();
        let r_target = self.params.r as usize;
        let l_construction = self.params.l_construction as usize;

        // Build-progress instrumentation (observability only): see
        // [`DiskAnnGraph::vamana_pass`]. Nodes are counted in the SEQUENTIAL
        // apply phase below — NOT in the parallel read-only region — so the hot
        // par_iter map is untouched and the count is exact (one per applied
        // node) and on the main thread (a thread-local test subscriber sees it).
        let pass_start = Instant::now();
        let progress = AtomicUsize::new(0);
        tracing::info!(
            pass = pass_num,
            alpha = alpha as f64,
            total = n,
            "diskann vamana build pass start"
        );

        // Random permutation of slots (Fisher-Yates) — same prng stream as the
        // sequential pass so the build stays deterministic given a batch size.
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = (prng.next_u32() as usize) % (i + 1);
            perm.swap(i, j);
        }

        let batch_size = batch_size.max(1);
        for chunk in perm.chunks(batch_size) {
            // PARALLEL read-only phase: each node's forward edges against the
            // current snapshot. greedy_visit_from + robust_prune are `&self`;
            // the closure only reads — no data races by construction. The
            // immutable borrow ends with the block, before the apply mutates.
            let new_edges: Vec<(u32, Vec<u32>)> = {
                let this: &DiskAnnGraph = &*self;
                chunk
                    .par_iter()
                    .map(|&p| {
                        let query = this.vector_bytes_owned(p);
                        let visit = this.greedy_visit_from(&query, medoid, l_construction);
                        let mut candidates: Vec<(u32, f32)> =
                            Vec::with_capacity(visit.visited.len() + r_target);
                        for (slot, key) in visit.visited {
                            if slot != p {
                                candidates.push((slot, key));
                            }
                        }
                        for &existing in &this.neighbors[p as usize] {
                            if existing == p || candidates.iter().any(|(s, _)| *s == existing) {
                                continue;
                            }
                            let raw = this.slot_distance(p, existing);
                            candidates.push((existing, this.distance_key(raw)));
                        }
                        let pruned = this.robust_prune(p, candidates, alpha, r_target);
                        (p, pruned)
                    })
                    .collect()
            };

            // SEQUENTIAL apply phase: set forward edges + symmetrise back-edges.
            for (p, pruned) in new_edges {
                self.neighbors[p as usize] = pruned.clone();
                for q in pruned {
                    self.symmetrize_back_edge(p, q, alpha, r_target);
                }

                // Node complete (edges applied). Counting here — not in the
                // parallel map above — leaves the read-only hot region untouched.
                vamana_progress_tick(&progress, pass_num, alpha, n, pass_start);
            }
        }

        tracing::info!(
            pass = pass_num,
            alpha = alpha as f64,
            nodes_done = progress.load(Ordering::Relaxed),
            total = n,
            elapsed_secs = pass_start.elapsed().as_secs_f64(),
            "diskann vamana build pass complete"
        );
    }

    /// α-pruning per Vamana §3 Algorithm 2.
    ///
    /// Given a candidate list `V` (slot, key) ordered by
    /// ascending key (closer first), return up to `R` slots
    /// such that no two retained slots `p_i, p_j` satisfy
    /// `key(p, p_j) > α · key(p_i, p_j)` — i.e., longer
    /// candidates dominated by closer ones (under α scaling)
    /// are pruned.
    fn robust_prune(
        &self,
        p: u32,
        mut candidates: Vec<(u32, f32)>,
        alpha: f32,
        r: usize,
    ) -> Vec<u32> {
        // Sort ascending by key (smaller is closer).
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        // Remove p itself if it slipped in.
        candidates.retain(|(s, _)| *s != p);
        // Dedup.
        candidates.dedup_by(|a, b| a.0 == b.0);

        let mut result: Vec<u32> = Vec::with_capacity(r);
        let mut alive: Vec<bool> = vec![true; candidates.len()];

        // Vamana α-prune (Subramanya §3 Algorithm 2):
        // for the closest-to-p candidate `p_star`, prune any
        // remaining `v` where `d(p, v) > α · d(p_star, v)`.
        // The comparison is in the metric's natural-distance
        // domain (Subramanya defines Vamana for L2 — non-
        // negative distances; the comparison is geometrically
        // the "occlusion test"). For Cosine, simsimd returns
        // `1 − cos(θ)` (lower-is-closer), so the L2-shape
        // comparator is correct-by-accident. For IP it is
        // inverted (raw kernel similarities are higher-is-
        // closer); `Metric::Ip` is rejected at
        // `DiskAnnGraph::new` per issue #109 defensive (a)
        // until the v1.1 sign-aware comparator + IP-recall
        // regression test land.
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

/// Tiny xorshift32 PRNG — deterministic, no workspace dep on
/// `rand`. Used during build for the random permutation and
/// the random initial neighbor list (both of which are
/// overwritten by α-pruning passes).
pub(crate) struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    pub(crate) fn seed(s: u32) -> Self {
        // Avoid the all-zero state, which is fixed under
        // xorshift.
        Self {
            state: if s == 0 { 0xDEAD_BEEF } else { s },
        }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::graph::{DiskAnnGraph, DiskAnnParams};
    use crate::distance::L2F32;
    use crate::{DistanceKernel, Encoding, Metric};

    fn fxd(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    fn empty_graph_f32_with(params: DiskAnnParams) -> DiskAnnGraph {
        DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32)).unwrap()
    }

    #[test]
    fn build_empty_input_leaves_graph_empty() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        g.build(&[]).unwrap();
        assert!(g.is_empty());
        assert_eq!(g.entry_point_id(), None);
    }

    #[test]
    fn build_single_vector_sets_entry_point() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[1.0, 2.0, 3.0, 4.0]);
        g.build(&[(VectorId::new(7), v.as_slice())]).unwrap();
        assert_eq!(g.main_len(), 1);
        assert_eq!(g.entry_point_id(), Some(VectorId::new(7)));
    }

    #[test]
    fn build_rejects_duplicate_ids() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let err = g
            .build(&[
                (VectorId::new(1), v.as_slice()),
                (VectorId::new(1), v.as_slice()),
            ])
            .unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn build_rejects_dim_mismatch() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v4 = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = fxd(&[1.0, 0.0]);
        let err = g
            .build(&[
                (VectorId::new(1), v4.as_slice()),
                (VectorId::new(2), v2.as_slice()),
            ])
            .unwrap_err();
        assert!(matches!(err, VectorIndexError::DimensionMismatch { .. }));
    }

    #[test]
    fn build_owned_matches_build_byte_for_byte() {
        // Strong-oracle parity (ADR-195 §3): build_owned (MOVE ingest) must
        // produce a graph BYTE-IDENTICAL to build (COPY ingest) on the same
        // input — same medoid, same seeded permutation, same α-prune passes,
        // same edges. Anything weaker would let the bounded SSD build path
        // silently diverge from the reference. dim=6, N=40 exercises the
        // multi-pass path (n > R+1).
        let params = DiskAnnParams {
            r: 5,
            alpha: 1.2,
            l_construction: 24,
            l_search_default: 24,
            ..DiskAnnParams::default()
        };
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..40u32 {
            let fi = i as f32;
            owned.push((
                VectorId::new(i),
                fxd(&[
                    fi * 0.1,
                    fi.sin(),
                    fi.cos(),
                    (fi * 0.3).sin(),
                    -fi * 0.05,
                    0.2,
                ]),
            ));
        }

        let mut g_ref = empty_graph_f32_with(params);
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g_ref.build(&pairs).unwrap();

        let mut g_owned = empty_graph_f32_with(params);
        g_owned.build_owned(owned.clone()).unwrap();

        // Byte-equal parallel arrays + identical entry point.
        assert_eq!(g_ref.ids, g_owned.ids, "ids differ");
        assert_eq!(g_ref.vectors, g_owned.vectors, "vectors differ");
        assert_eq!(
            g_ref.neighbors, g_owned.neighbors,
            "neighbors (graph edges) differ"
        );
        assert_eq!(
            g_ref.entry_point, g_owned.entry_point,
            "entry point differs"
        );
        assert_eq!(g_ref.bytes_per_vector, g_owned.bytes_per_vector);
    }

    #[test]
    fn build_with_small_dataset_produces_connected_graph() {
        // Build with N=20, R=4. Every node should reach every
        // other node from the medoid in ≤ a few hops.
        let params = DiskAnnParams {
            r: 4,
            alpha: 1.2,
            l_construction: 16,
            l_search_default: 16,
            ..DiskAnnParams::default()
        };
        let mut g = empty_graph_f32_with(params);
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..20 {
            owned.push((
                VectorId::new(i as u32),
                fxd(&[i as f32 * 0.1, (i as f32).sin(), (i as f32).cos(), 0.0]),
            ));
        }
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();
        assert_eq!(g.main_len(), 20);
        // Every node should have ≤ R out-edges, and at least
        // one edge.
        for slot in 0..g.main_len() as u32 {
            let neigh = &g.neighbors[slot as usize];
            assert!(
                neigh.len() <= params.r as usize,
                "slot {slot} has {} neighbors > R={}",
                neigh.len(),
                params.r
            );
            // No self-loops.
            assert!(!neigh.contains(&slot), "slot {slot} self-loops");
        }
        // Reachability: BFS from medoid should reach all live
        // slots.
        let entry = g.entry_point.unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut queue = vec![entry];
        seen.insert(entry);
        while let Some(s) = queue.pop() {
            for &n in &g.neighbors[s as usize] {
                if seen.insert(n) {
                    queue.push(n);
                }
            }
        }
        assert_eq!(
            seen.len(),
            g.main_len(),
            "graph not connected from medoid: reached {}/{}",
            seen.len(),
            g.main_len()
        );
    }

    #[test]
    fn build_recall_at_10_on_random_dim_8() {
        // Recall sanity at small scale — full SIFT runs in
        // tests/diskann.rs. This is a smoke test that the
        // build + search path produces sane recall.
        let params = DiskAnnParams {
            r: 16,
            alpha: 1.2,
            l_construction: 32,
            l_search_default: 64,
            ..DiskAnnParams::default()
        };
        let mut g = empty_graph_f32_with(params);
        let n = 200_usize;
        let dim = 8_usize;
        let mut rng = XorShift32::seed(42);
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0_f32; dim];
            for x in v.iter_mut() {
                let r = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                *x = r;
            }
            owned.push((VectorId::new(i as u32), fxd(&v)));
        }
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();

        // Brute-force ground truth top-10 for 20 random queries.
        let n_queries = 20;
        let mut hits = 0_usize;
        let mut total = 0_usize;
        for q_idx in 0..n_queries {
            let mut q = vec![0.0_f32; dim];
            for x in q.iter_mut() {
                let r = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                *x = r;
            }
            let q_bytes = fxd(&q);

            let mut bf: Vec<(VectorId, f32)> = owned
                .iter()
                .map(|(id, v)| {
                    let raw = L2F32.distance(v.as_slice(), q_bytes.as_slice());
                    (*id, raw)
                })
                .collect();
            bf.sort_by(|a, b| a.1.total_cmp(&b.1));
            let bf_top10: std::collections::HashSet<u32> =
                bf.iter().take(10).map(|(id, _)| id.raw()).collect();

            let res = g.search(&q_bytes, 10, 64).unwrap();
            for (id, _) in res {
                if bf_top10.contains(&id.raw()) {
                    hits += 1;
                }
            }
            total += 10;
            let _ = q_idx;
        }
        let recall = hits as f64 / total as f64;
        // Smoke threshold: ≥ 0.85 at 200 random vectors dim=8.
        // The slice-D acceptance criterion (≥ 0.95) is checked
        // on the SIFT subset in the integration test.
        assert!(recall >= 0.85, "smoke recall@10 {recall} < 0.85");
    }

    /// Measure recall@k of a built graph vs the exhaustive brute-force oracle.
    fn recall_of(
        g: &DiskAnnGraph,
        owned: &[(VectorId, Vec<u8>)],
        queries: &[Vec<u8>],
        k: usize,
        l: usize,
    ) -> f64 {
        let mut hits = 0usize;
        for q in queries {
            let mut bf: Vec<(VectorId, f32)> = owned
                .iter()
                .map(|(id, v)| (*id, L2F32.distance(v.as_slice(), q.as_slice())))
                .collect();
            bf.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: std::collections::HashSet<u32> =
                bf.iter().take(k).map(|(id, _)| id.raw()).collect();
            for (id, _) in g.search(q, k, l).unwrap() {
                if gt.contains(&id.raw()) {
                    hits += 1;
                }
            }
        }
        hits as f64 / (k * queries.len()) as f64
    }

    #[test]
    fn parallel_build_recall_matches_sequential_within_tolerance() {
        // Strong oracle (ADR-195 §3 / #112): build_owned_parallel must NOT
        // silently regress recall vs the sequential build_owned. We build the
        // SAME data both ways and assert the parallel recall is within a small
        // tolerance of (and not far below) the sequential recall. The batched
        // snapshot-parallelism changes the graph slightly (intra-batch nodes
        // don't see each other's updates), so the gate is parity-within-tol,
        // not byte-equality.
        let params = DiskAnnParams {
            r: 16,
            alpha: 1.2,
            l_construction: 48,
            l_search_default: 64,
            ..DiskAnnParams::default()
        };
        let n = 600_usize;
        let dim = 16_usize;
        let mut rng = XorShift32::seed(2024);
        // Clustered corpus so recall is well-posed (10 centers × 60 pts).
        let mut centers = vec![vec![0.0_f32; dim]; 10];
        for c in centers.iter_mut() {
            for x in c.iter_mut() {
                *x = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
        }
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::with_capacity(n);
        for i in 0..n {
            let c = &centers[i % centers.len()];
            let v: Vec<f32> = c
                .iter()
                .map(|&cc| cc + ((rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05)
                .collect();
            owned.push((VectorId::new(i as u32), fxd(&v)));
        }

        let mut g_seq = empty_graph_f32_with(params);
        g_seq.build_owned(owned.clone()).unwrap();
        let mut g_par = empty_graph_f32_with(params);
        // Batch of 64 over 600 nodes → ~10 batches, real parallelism.
        g_par.build_owned_parallel(owned.clone(), 64).unwrap();

        // Both must have ingested every vector identically (the prelude is
        // shared) — only the edges may differ.
        assert_eq!(g_seq.ids, g_par.ids);
        assert_eq!(g_seq.vectors, g_par.vectors);
        assert_eq!(g_seq.entry_point, g_par.entry_point);

        // In-distribution queries.
        let mut queries: Vec<Vec<u8>> = Vec::new();
        for _ in 0..40 {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let v: Vec<f32> = c
                .iter()
                .map(|&cc| cc + ((rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05)
                .collect();
            queries.push(fxd(&v));
        }

        let seq_recall = recall_of(&g_seq, &owned, &queries, 10, 64);
        let par_recall = recall_of(&g_par, &owned, &queries, 10, 64);
        // Parity: the parallel build is within 5pp of sequential AND clears a
        // high absolute floor (no silent quality cliff).
        assert!(
            par_recall >= seq_recall - 0.05,
            "parallel recall {par_recall:.4} regressed > 5pp below sequential {seq_recall:.4}"
        );
        assert!(
            par_recall >= 0.90,
            "parallel recall {par_recall:.4} below the 0.90 floor (seq={seq_recall:.4})"
        );
    }

    // --- Build-progress probe drain (deterministic, concurrency-safe oracle) ---

    /// Drain the thread-local progress probe (see [`super::record_progress_probe`]):
    /// the `(pass_num, nodes_done, total)` of every throttled emission recorded on
    /// this thread since the last drain.
    fn drain_progress_probe() -> Vec<(u8, usize, usize)> {
        super::PROGRESS_PROBE.with(|p| std::mem::take(&mut *p.borrow_mut()))
    }

    #[test]
    fn build_emits_progress_reaching_total_each_pass() {
        // Active verification (ADR-133 Index-class) of the build-progress
        // instrumentation. STRONG oracle: a test-only probe records the
        // (pass, nodes_done, total) of every throttled emission — at the exact
        // point `vamana_progress_tick` fires its `tracing::info!` — and we assert
        // (a) progress emissions happened, (b) `nodes_done` reaches `total` for
        // BOTH passes (α=1.0 and α=params.alpha), (c) both passes emit — on BOTH
        // the sequential and parallel build paths. A counter that silently stalls
        // below n fails here (we assert the value reached n, not merely "an
        // emission happened"). We probe rather than capture `tracing` events
        // because tracing's process-global per-callsite Interest cache makes a
        // thread-local capture subscriber unreliable under concurrent tests.
        let params = DiskAnnParams {
            r: 5,
            alpha: 1.2, // > 1.0 ⇒ both refinement passes run
            l_construction: 24,
            l_search_default: 24,
            ..DiskAnnParams::default()
        };
        // n = 40 > r + 1 ⇒ the full multi-pass refinement runs (not the trivial
        // empty / single-vector paths that return before any pass).
        let mut owned: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..40u32 {
            let fi = i as f32;
            owned.push((
                VectorId::new(i),
                fxd(&[
                    fi * 0.1,
                    fi.sin(),
                    fi.cos(),
                    (fi * 0.3).sin(),
                    -fi * 0.05,
                    0.2,
                ]),
            ));
        }
        let total = owned.len();

        for path in ["sequential", "parallel"] {
            let _ = drain_progress_probe(); // clear any prior ticks on this thread
            let mut g = empty_graph_f32_with(params);
            if path == "sequential" {
                g.build_owned(owned.clone()).unwrap();
            } else {
                // Small batch over n=40 ⇒ several batches, real parallelism.
                g.build_owned_parallel(owned.clone(), 8).unwrap();
            }
            let ticks = drain_progress_probe();

            // (a) progress emissions happened.
            assert!(!ticks.is_empty(), "[{path}] no progress emissions recorded");

            // (c) both passes (1 and 2) emitted.
            let passes: std::collections::HashSet<u8> = ticks.iter().map(|&(p, _, _)| p).collect();
            assert!(
                passes.contains(&1) && passes.contains(&2),
                "[{path}] expected pass 1 and pass 2 to emit, saw {passes:?}"
            );

            // (b) final nodes_done == total for EACH pass (the counter actually
            // reached n — strong oracle), and every emission reported `total`.
            for pass in [1u8, 2] {
                let max_done = ticks
                    .iter()
                    .filter(|&&(p, _, _)| p == pass)
                    .map(|&(_, d, _)| d)
                    .max();
                assert_eq!(
                    max_done,
                    Some(total),
                    "[{path}] pass {pass}: final nodes_done {max_done:?} != total {total}"
                );
                assert!(
                    ticks
                        .iter()
                        .filter(|&&(p, _, _)| p == pass)
                        .all(|&(_, _, t)| t == total),
                    "[{path}] pass {pass}: an emission reported total != {total}"
                );
            }
        }
    }
}
