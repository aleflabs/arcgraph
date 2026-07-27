//! Vamana beam search (Algorithm 2 of Subramanya et al. NeurIPS 2019).
//!
//! Two flavors live here:
//!
//! - [`DiskAnnGraph::search`] — public top-K beam search on the
//!   main graph only. Used by callers that have already merged
//!   the delta-segment elsewhere or do not stream-insert.
//! - `DiskAnnGraph::greedy_visit_from` — internal helper
//!   exposing the "visited set V" output used by the build
//!   path (Algorithm 1) before α-pruning.
//!
//! Both routines maintain the **L-frontier** (best-first
//! priority queue of candidates) and the **visited set** as
//! they walk the graph from the medoid (or supplied entry
//! point). Tombstoned slots are skipped.
//!
//! ## Ordering convention
//!
//! Internally the algorithm orders by ascending
//! `distance_key` — smaller is closer for every metric.
//! Externally we report the kernel's natural distance via
//! `distance_external`. This lets a single implementation
//! handle L2 (lower-is-closer) and IP (higher-is-closer)
//! without per-metric code paths.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distance::DistanceKernel;
use crate::quantizer::{self, RaBitQQuery};
use crate::{Result, VectorId, VectorIndexError};

use super::graph::DiskAnnGraph;

/// Heap entry — `key` is the search-time ordering key; the
/// natural-orientation kernel distance is recovered via
/// [`DiskAnnGraph::distance_external`] at result time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SearchCandidate {
    pub(crate) slot: u32,
    /// Smaller is closer.
    pub(crate) key: f32,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want a min-heap for
        // ascending-key semantics. Reverse the natural
        // partial order. NaN inputs would break Ord, so we
        // canonicalize via `total_cmp` (handles NaN
        // deterministically) and reverse.
        other
            .key
            .total_cmp(&self.key)
            .then(self.slot.cmp(&other.slot))
    }
}

/// "Worst-first" heap entry — used to track the L-frontier's
/// largest key for cheap eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WorstFirst {
    slot: u32,
    /// Smaller is closer.
    key: f32,
}

impl Eq for WorstFirst {}

impl PartialOrd for WorstFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on key — larger keys at top. (Native
        // BinaryHeap behavior.)
        self.key
            .total_cmp(&other.key)
            .then(self.slot.cmp(&other.slot))
    }
}

/// Result of [`DiskAnnGraph::greedy_visit`].
#[derive(Debug, Clone, Default)]
pub(crate) struct GreedyVisit {
    /// Visited slots in ascending-key order (best first).
    pub(crate) visited: Vec<(u32, f32)>,
}

#[derive(Debug, Default)]
struct GenerationBitset {
    marks: Vec<u32>,
    generation: u32,
}

impl GenerationBitset {
    #[inline]
    fn reset(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    #[inline]
    fn contains(&self, slot: u32) -> bool {
        self.marks[slot as usize] == self.generation
    }

    #[inline]
    fn insert(&mut self, slot: u32) -> bool {
        let mark = &mut self.marks[slot as usize];
        if *mark == self.generation {
            return false;
        }
        *mark = self.generation;
        true
    }

    #[inline]
    fn remove(&mut self, slot: u32) {
        let mark = &mut self.marks[slot as usize];
        if *mark == self.generation {
            *mark = 0;
        }
    }
}

#[derive(Debug, Default)]
struct SearchScratch {
    visited: GenerationBitset,
    in_frontier: GenerationBitset,
}

thread_local! {
    static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
}

impl DiskAnnGraph {
    /// Beam-search top-`k` against the main Vamana graph (no
    /// delta-segment). Returns `(VectorId, distance)` pairs in
    /// ascending closeness — i.e., the kernel's natural
    /// distance with the metric direction respected (L2 lowest
    /// → most similar; IP largest → most similar).
    ///
    /// Per Vamana §3 / Subramanya et al. 2019 Algorithm 2.
    /// Falls through `search_with_delta` if the caller wants
    /// the delta-segment merge.
    pub fn search(&self, query: &[u8], k: usize, l_search: usize) -> Result<Vec<(VectorId, f32)>> {
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
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        let l = l_search.max(k).max(1);
        let visit = self.greedy_visit_from(query, entry, l);
        let mut out = Vec::with_capacity(k.min(visit.visited.len()));
        for (slot, key) in visit.visited.into_iter().take(k) {
            out.push((self.ids[slot as usize], self.distance_external(key)));
        }
        Ok(out)
    }

    /// Internal: greedy beam search from a specific entry slot.
    /// Returns the visited set ordered ascending by key — the
    /// build path consumes this for α-pruning.
    pub(crate) fn greedy_visit_from(&self, query: &[u8], entry: u32, l: usize) -> GreedyVisit {
        self.greedy_visit_core(|slot| self.query_to_slot_distance(query, slot), entry, l)
    }

    /// Internal: greedy beam search from a prepared RaBitQ query.
    pub(crate) fn greedy_visit_rabitq(
        &self,
        query: &RaBitQQuery,
        entry: u32,
        l: usize,
    ) -> GreedyVisit {
        self.greedy_visit_core(
            |slot| quantizer::estimate_l2_sq(query, self.vector_bytes(slot)),
            entry,
            l,
        )
    }

    fn greedy_visit_core<F>(&self, dist: F, entry: u32, l: usize) -> GreedyVisit
    where
        F: Fn(u32) -> f32,
    {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            self.greedy_visit_core_with_scratch(dist, entry, l, &mut scratch)
        })
    }

    fn greedy_visit_core_with_scratch<F>(
        &self,
        dist: F,
        entry: u32,
        l: usize,
        scratch: &mut SearchScratch,
    ) -> GreedyVisit
    where
        F: Fn(u32) -> f32,
    {
        debug_assert!(l > 0, "beam width must be > 0");
        debug_assert!((entry as usize) < self.ids.len(), "entry slot out of range");
        if l >= self.ids.len() {
            let mut visited: Vec<(u32, f32)> = (0..self.ids.len() as u32)
                .filter(|&slot| !self.is_tombstoned(slot))
                .map(|slot| (slot, self.distance_key(dist(slot))))
                .collect();
            visited.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            return GreedyVisit { visited };
        }
        // L-frontier — closed set of the best-l candidates we
        // have seen so far. Maintained as a max-heap on key so
        // the largest-key entry can be popped in O(log L) when
        // a closer neighbor is found.
        let mut frontier: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(l + 1);
        // Min-heap of unvisited candidates (best-first walk).
        let mut to_visit: BinaryHeap<SearchCandidate> = BinaryHeap::with_capacity(l + 1);
        let n_slots = self.ids.len();
        scratch.visited.reset(n_slots);
        scratch.in_frontier.reset(n_slots);

        let entry_key = self.distance_key(dist(entry));
        let seed = SearchCandidate {
            slot: entry,
            key: entry_key,
        };
        to_visit.push(seed);
        if !self.is_tombstoned(entry) {
            frontier.push(WorstFirst {
                slot: entry,
                key: entry_key,
            });
            scratch.in_frontier.insert(entry);
        }

        while let Some(curr) = to_visit.pop() {
            if !scratch.visited.insert(curr.slot) {
                continue;
            }
            // Expand neighbors.
            let neigh_slice = &self.neighbors[curr.slot as usize];
            for (i, &n) in neigh_slice.iter().enumerate() {
                if let Some(&next) = neigh_slice.get(i + 1) {
                    self.prefetch_vector_bytes(next);
                }
                if scratch.visited.contains(n) || scratch.in_frontier.contains(n) {
                    continue;
                }
                let raw = dist(n);
                let key = self.distance_key(raw);
                // Push onto best-first walk regardless — the
                // visited check above prevents re-expansion.
                to_visit.push(SearchCandidate { slot: n, key });
                // Frontier insertion respects tombstones —
                // tombstoned slots are still walked for graph
                // traversal but never returned to the caller.
                if self.is_tombstoned(n) {
                    continue;
                }
                if frontier.len() < l {
                    frontier.push(WorstFirst { slot: n, key });
                    scratch.in_frontier.insert(n);
                } else if let Some(worst) = frontier.peek()
                    && key < worst.key
                {
                    let popped = frontier.pop().expect("peek non-empty");
                    scratch.in_frontier.remove(popped.slot);
                    frontier.push(WorstFirst { slot: n, key });
                    scratch.in_frontier.insert(n);
                }
            }
            // Termination — if all best-first candidates are
            // strictly worse than the worst frontier key, we
            // cannot improve the frontier and are done.
            // `BinaryHeap::peek` for `to_visit` returns the
            // smallest-key (we encoded SearchCandidate with
            // reversed Ord); for `frontier` the largest key.
            if let Some(next) = to_visit.peek()
                && let Some(worst) = frontier.peek()
                && next.key > worst.key
            {
                break;
            }
        }

        // Drain frontier into ascending-key order.
        let mut visited_vec: Vec<(u32, f32)> =
            frontier.into_iter().map(|w| (w.slot, w.key)).collect();
        visited_vec.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        GreedyVisit {
            visited: visited_vec,
        }
    }

    /// Beam-search on the primary (quantized) graph with a
    /// full-precision rescore step against the merged
    /// main-graph + delta-segment candidate set.
    ///
    /// Per ADR-035 §3.3 + §5.4 step 6 + AC-1a:
    ///
    /// 1. Run the existing main-graph + delta-segment search
    ///    (via [`DiskAnnGraph::search_with_delta`]) with the
    ///    candidate budget enlarged to `k * rescore_factor` and
    ///    the beam width enlarged to `l_search * rescore_factor`.
    ///    The graph's internal kernel computes the quantized
    ///    distances during the walk; `primary_kernel` is accepted
    ///    for symmetry with the HNSW rescore surface (Slice E.2)
    ///    and validated against the graph's built-in
    ///    encoding + metric.
    /// 2. For each merged candidate, look up the full-precision
    ///    bytes via `full_precision_vectors`. Compute
    ///    `full_precision_kernel.distance(rescore_query, fp_bytes)`.
    ///    A `None` answer raises
    ///    [`VectorIndexError::RescoreVectorMissing`] (stale rescore
    ///    section vs the index — operator must `REINDEX`).
    /// 3. Sort the rescored candidates ascending by
    ///    metric-direction-aware distance key (via
    ///    `DiskAnnGraph::distance_key`) and truncate to `k`.
    ///    Returned distances are in the metric's natural
    ///    orientation (matches [`DiskAnnGraph::search`] output).
    ///
    /// `delta_segment` integration is automatic — Slice D's
    /// [`DiskAnnGraph::search_with_delta`] merges main + delta by
    /// distance under the primary kernel, and the rescore step
    /// runs uniformly over the merged set per ADR-035 §5.3.1
    /// B-3 ("rescore on the merged top-K"). RYW (I-V7) is
    /// preserved because the delta-segment is consulted on the
    /// primary path before rescore.
    ///
    /// ## Two queries (SQ8 primary + F32 rescore)
    ///
    /// The signature takes both `primary_query` (encoded for the
    /// graph's primary encoding) and `rescore_query` (encoded for
    /// `full_precision_kernel.encoding()`). Per ADR-035 §3.3 the
    /// recommended SQ8 default is the primary in i8 and the
    /// rescore section in f32 — the simsimd kernel surface
    /// requires both kernel inputs to share the encoding, so the
    /// caller must hand in two byte slices. For `Encoding::F32`
    /// indexes (rescore as a no-op sanity check) the two slices
    /// may alias the same buffer.
    ///
    /// ## Edge cases
    ///
    /// - `k == 0` — returns `Ok(Vec::new())` (no rescore lookup).
    /// - `rescore_factor == 1` — short-circuits to
    ///   [`DiskAnnGraph::search_with_delta`]; the result distances
    ///   are in primary-kernel space and the `full_precision_*`
    ///   arguments are unused. Operators choose this for
    ///   latency-sensitive workloads accepting the AC-1b
    ///   "best-effort" recall floor.
    /// - `rescore_factor == 0` — rejected with
    ///   [`VectorIndexError::InvalidRescoreFactor`]
    ///   (`factor: 0`); rescore_factor must be ≥ 1 by
    ///   definition. Per ADR-035 D-4, `rescore_factor = 1` is
    ///   the operator opt-out (SQ8-alone, AC-1b best-effort),
    ///   not an error.
    /// - `k * rescore_factor > graph_capacity` — the candidate
    ///   pool saturates at `main_len() + delta_len()` (the beam
    ///   search returns at most that many anyway). Recall
    ///   degrades gracefully on tiny graphs without erroring.
    /// - `primary_kernel` / `full_precision_kernel` mismatch with
    ///   the graph's metric (or encoding, for the primary kernel)
    ///   — rejected with [`VectorIndexError::UnsupportedFlags`].
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::InvalidRescoreFactor`] when
    ///   `rescore_factor == 0`.
    /// - [`VectorIndexError::DimensionMismatch`] on primary-query
    ///   byte width mismatch.
    /// - [`VectorIndexError::UnsupportedFlags`] on kernel mismatch.
    /// - [`VectorIndexError::RescoreVectorMissing`] when the
    ///   `full_precision_vectors` lookup returns `None` for a
    ///   primary candidate (rescore arena out of sync with the
    ///   primary index).
    // The wide signature is the prompt-specified rescore surface
    // (E.2 / E.3 mirror): two queries (primary-encoded +
    // full-precision-encoded), two kernels, the lookup callback,
    // plus the standard `(k, l_search, rescore_factor)` tuning
    // triple. Each parameter is load-bearing — no helper struct
    // would shrink it without hiding the contract that the
    // planner / arena layer must honor at the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_rescore<'fp, F>(
        &self,
        primary_query: &[u8],
        rescore_query: &[u8],
        k: usize,
        l_search: usize,
        rescore_factor: usize,
        primary_kernel: &dyn DistanceKernel,
        full_precision_kernel: &dyn DistanceKernel,
        full_precision_vectors: F,
    ) -> Result<Vec<(VectorId, f32)>>
    where
        F: Fn(VectorId) -> Option<&'fp [u8]>,
    {
        // 1. Validate rescore_factor up front. `usize` → `0` is the
        //    only invalid value; per ADR-035 D-4, `rescore_factor = 1`
        //    is the operator opt-out (SQ8-alone, AC-1b best-effort),
        //    not an error. The dedicated parameter-domain variant
        //    keeps the error message operator-actionable (vs reusing
        //    `DimensionMismatch`, which conflates parameter
        //    validation with codec-shape mismatches).
        if rescore_factor == 0 {
            return Err(VectorIndexError::InvalidRescoreFactor { factor: 0 });
        }

        // 2. Validate kernel compatibility. Primary kernel must
        //    match the graph's stored kernel in (encoding, metric)
        //    — a mismatch would compute distances against bytes
        //    that the graph stored under a different convention.
        //    Full-precision kernel must match the graph's metric
        //    (encoding may differ — that is the entire point of
        //    rescore: SQ8 primary + F32 rescore).
        if primary_kernel.encoding() != self.encoding() || primary_kernel.metric() != self.metric()
        {
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: primary_kernel.encoding(),
                metric: primary_kernel.metric(),
            });
        }
        if full_precision_kernel.metric() != self.metric() {
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: full_precision_kernel.encoding(),
                metric: full_precision_kernel.metric(),
            });
        }

        // 3. Short-circuit on rescore_factor == 1: skip rescore,
        //    run the merged primary search and return its result
        //    distances (primary-kernel space). The
        //    full_precision_* arguments are intentionally unused
        //    on this path.
        if rescore_factor == 1 {
            return self.search_with_delta(primary_query, k, l_search);
        }

        // 4. Empty-k early return AFTER parameter validation so
        //    that a caller passing rescore_factor=0 + k=0 still
        //    sees the parameter error.
        if k == 0 {
            return Ok(Vec::new());
        }

        // 5. Validate primary-query byte width against the graph.
        //    The full-precision query width is validated by
        //    `full_precision_kernel.distance()` itself
        //    (debug_assert in release; SIMD length check in any
        //    profile via simsimd) on every distance call.
        if let Some(expected) = self.bytes_per_vector()
            && primary_query.len() != expected
        {
            return Err(VectorIndexError::DimensionMismatch {
                expected,
                got: primary_query.len(),
            });
        }

        // 6. Enlarge candidate budget. Saturate at the graph's
        //    total slot count (main + delta); the primary beam
        //    search returns at most that many results regardless,
        //    so capping avoids spurious work without changing
        //    semantics.
        let graph_cap = self.main_len().saturating_add(self.delta_len());
        if graph_cap == 0 {
            return Ok(Vec::new());
        }
        let enlarged_k = k.saturating_mul(rescore_factor).min(graph_cap);
        let enlarged_l = l_search
            .saturating_mul(rescore_factor)
            .max(enlarged_k)
            .min(graph_cap);

        // 7. Primary beam search — merges main graph + delta-segment
        //    by distance under the primary kernel per ADR-035
        //    §5.3.1.
        let primary_hits = self.search_with_delta(primary_query, enlarged_k, enlarged_l)?;

        // 8. Rescore each candidate against the full-precision
        //    section. A None lookup means the rescore section is
        //    stale relative to the primary index — surface as
        //    RescoreVectorMissing so the call site can route to
        //    the §3.3 REINDEX path.
        let mut rescored: Vec<(VectorId, f32, f32)> = Vec::with_capacity(primary_hits.len());
        for (vector_id, _primary_dist) in primary_hits {
            let fp_bytes = full_precision_vectors(vector_id)
                .ok_or(VectorIndexError::RescoreVectorMissing { vector_id })?;
            let raw = full_precision_kernel.distance(rescore_query, fp_bytes);
            let key = self.distance_key(raw);
            rescored.push((vector_id, raw, key));
        }

        // 9. Sort ascending by metric-direction-aware key with
        //    deterministic tie-break on VectorId.
        rescored.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.raw().cmp(&b.0.raw())));
        rescored.truncate(k);

        Ok(rescored
            .into_iter()
            .map(|(id, raw, _key)| (id, raw))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::graph::{DiskAnnGraph, DiskAnnParams};
    use crate::distance::L2F32;
    use crate::{Encoding, Metric};
    use std::collections::HashSet;
    use std::time::Instant;

    fn fxd(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    fn graph_with_three_clusters() -> DiskAnnGraph {
        // 3 clusters of 4 vectors each in dim=4.
        let mut g = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(L2F32),
        )
        .unwrap();
        let raw: Vec<(VectorId, Vec<u8>)> = (0..12)
            .map(|i| {
                let cluster = i / 4;
                let mut v = vec![0.0_f32; 4];
                v[cluster] = 1.0 + (i as f32) * 0.001;
                (VectorId::new(i as u32), fxd(&v))
            })
            .collect();
        let pairs: Vec<(VectorId, &[u8])> = raw.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        g.build(&pairs).unwrap();
        g
    }

    fn deterministic_vec(i: usize, dim: usize) -> Vec<f32> {
        let cluster = i % 8;
        (0..dim)
            .map(|d| {
                let base = if d == cluster { 3.0 } else { 0.0 };
                base + (((i * 31 + d * 17) % 101) as f32) * 0.0007
            })
            .collect()
    }

    fn graph_for_identity_test() -> DiskAnnGraph {
        let params = DiskAnnParams {
            r: 8,
            l_construction: 16,
            l_search_default: 16,
            ..DiskAnnParams::default()
        };
        let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32)).unwrap();
        let raw: Vec<(VectorId, Vec<u8>)> = (0..128)
            .map(|i| (VectorId::new(i as u32), fxd(&deterministic_vec(i, 8))))
            .collect();
        g.build_owned(raw).unwrap();
        g
    }

    fn greedy_visit_hashset_baseline(
        g: &DiskAnnGraph,
        query: &[u8],
        entry: u32,
        l: usize,
    ) -> GreedyVisit {
        if l >= g.ids.len() {
            let mut visited: Vec<(u32, f32)> = (0..g.ids.len() as u32)
                .filter(|&slot| !g.is_tombstoned(slot))
                .map(|slot| (slot, g.distance_key(g.query_to_slot_distance(query, slot))))
                .collect();
            visited.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            return GreedyVisit { visited };
        }

        let mut frontier: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(l + 1);
        let mut to_visit: BinaryHeap<SearchCandidate> = BinaryHeap::with_capacity(l + 1);
        let mut visited: HashSet<u32> = HashSet::new();
        let mut in_frontier: HashSet<u32> = HashSet::new();

        let entry_key = g.distance_key(g.query_to_slot_distance(query, entry));
        to_visit.push(SearchCandidate {
            slot: entry,
            key: entry_key,
        });
        if !g.is_tombstoned(entry) {
            frontier.push(WorstFirst {
                slot: entry,
                key: entry_key,
            });
            in_frontier.insert(entry);
        }

        while let Some(curr) = to_visit.pop() {
            if !visited.insert(curr.slot) {
                continue;
            }
            for &n in &g.neighbors[curr.slot as usize] {
                if visited.contains(&n) || in_frontier.contains(&n) {
                    continue;
                }
                let key = g.distance_key(g.query_to_slot_distance(query, n));
                to_visit.push(SearchCandidate { slot: n, key });
                if g.is_tombstoned(n) {
                    continue;
                }
                if frontier.len() < l {
                    frontier.push(WorstFirst { slot: n, key });
                    in_frontier.insert(n);
                } else if let Some(worst) = frontier.peek()
                    && key < worst.key
                {
                    let popped = frontier.pop().expect("peek non-empty");
                    in_frontier.remove(&popped.slot);
                    frontier.push(WorstFirst { slot: n, key });
                    in_frontier.insert(n);
                }
            }
            if let Some(next) = to_visit.peek()
                && let Some(worst) = frontier.peek()
                && next.key > worst.key
            {
                break;
            }
        }

        let mut visited_vec: Vec<(u32, f32)> =
            frontier.into_iter().map(|w| (w.slot, w.key)).collect();
        visited_vec.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        GreedyVisit {
            visited: visited_vec,
        }
    }

    #[test]
    fn search_empty_graph_returns_empty() {
        let g = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(L2F32),
        )
        .unwrap();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.search(&q, 5, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_zero_k_returns_empty() {
        let g = graph_with_three_clusters();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.search(&q, 0, 100).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_dim_mismatch_errors() {
        let g = graph_with_three_clusters();
        let q = fxd(&[1.0, 0.0, 0.0]); // dim 3 vs expected 4
        let r = g.search(&q, 5, 100);
        assert!(matches!(r, Err(VectorIndexError::DimensionMismatch { .. })));
    }

    #[test]
    fn search_returns_top_k_in_ascending_distance() {
        let g = graph_with_three_clusters();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.search(&q, 4, 100).unwrap();
        assert!(!r.is_empty(), "search returned empty for non-empty graph");
        for w in r.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results not ascending: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn search_top_result_is_in_query_cluster() {
        let g = graph_with_three_clusters();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.search(&q, 4, 100).unwrap();
        let top_id = r.first().unwrap().0;
        // Cluster 0 vectors have ids 0..4.
        assert!(
            top_id.raw() < 4,
            "top result {} not in query cluster (0..4)",
            top_id.raw()
        );
        // Top-4 should all be cluster-0 (most-similar group).
        let top_cluster: Vec<u32> = r.iter().map(|(id, _)| id.raw() / 4).collect();
        assert_eq!(top_cluster, vec![0, 0, 0, 0]);
    }

    #[test]
    fn search_skips_tombstoned_slots() {
        let mut g = graph_with_three_clusters();
        // Mark two of cluster-0 as tombstoned.
        g.delete(VectorId::new(0)).unwrap();
        g.delete(VectorId::new(1)).unwrap();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.search(&q, 4, 100).unwrap();
        for (id, _) in &r {
            assert_ne!(id.raw(), 0, "tombstoned 0 leaked into results");
            assert_ne!(id.raw(), 1, "tombstoned 1 leaked into results");
        }
    }

    #[test]
    fn search_l_below_k_clamps() {
        let g = graph_with_three_clusters();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        // Pass l_search = 1 with k=4: implementation must clamp
        // l up to k internally so we can return 4 results.
        let r = g.search(&q, 4, 1).unwrap();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn traversal_results_identical_to_hashset_baseline() {
        let g = graph_for_identity_test();
        let entry = g.entry_point.expect("built graph has entry point");
        for qi in [0usize, 1, 7, 18, 31, 64, 95, 127] {
            let query = fxd(&deterministic_vec(qi, 8));
            let optimized = g.greedy_visit_from(&query, entry, 16);
            let baseline = greedy_visit_hashset_baseline(&g, &query, entry, 16);
            assert_eq!(
                optimized.visited, baseline.visited,
                "greedy visit changed for query {qi}"
            );

            let optimized_hits = g.search(&query, 10, 16).unwrap();
            let baseline_hits: Vec<(VectorId, f32)> = baseline
                .visited
                .iter()
                .take(10)
                .map(|(slot, key)| (g.ids[*slot as usize], g.distance_external(*key)))
                .collect();
            assert_eq!(
                optimized_hits, baseline_hits,
                "search changed for query {qi}"
            );
        }
    }

    #[test]
    #[ignore = "local perf probe; run explicitly for traversal-overhead deltas"]
    fn traversal_microbench_reports_per_query_time() {
        let g = graph_for_identity_test();
        let entry = g.entry_point.expect("built graph has entry point");
        let queries: Vec<Vec<u8>> = (0..64)
            .map(|i| fxd(&deterministic_vec((i * 3) % 128, 8)))
            .collect();
        let iters = 256usize;

        let start = Instant::now();
        let mut optimized_count = 0usize;
        for _ in 0..iters {
            for q in &queries {
                optimized_count += g.greedy_visit_from(q, entry, 16).visited.len();
            }
        }
        let optimized = start.elapsed();

        let start = Instant::now();
        let mut baseline_count = 0usize;
        for _ in 0..iters {
            for q in &queries {
                baseline_count += greedy_visit_hashset_baseline(&g, q, entry, 16)
                    .visited
                    .len();
            }
        }
        let baseline = start.elapsed();
        assert_eq!(optimized_count, baseline_count);

        let n = (iters * queries.len()) as f64;
        println!(
            "diskann traversal microbench: optimized={:.3}us/query baseline={:.3}us/query speedup={:.2}x",
            optimized.as_secs_f64() * 1_000_000.0 / n,
            baseline.as_secs_f64() * 1_000_000.0 / n,
            baseline.as_secs_f64() / optimized.as_secs_f64()
        );
    }
}
