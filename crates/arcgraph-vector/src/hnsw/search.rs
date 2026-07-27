//! HNSW search — Malkov & Yashunin TPAMI 2018 Algorithms 2 & 5.
//!
//! Two routines anchor the search code:
//!
//! - [`search_layer`] — Algorithm 2. A bounded-beam-width best-first
//!   search at one layer, returning the `ef` nearest visited
//!   nodes. Used by both insert (to populate the candidate set
//!   for `select_neighbors_heuristic`) and search (to walk
//!   layer-by-layer).
//! - [`HnswGraph::search`] — Algorithm 5. Greedy descent through
//!   the upper layers (each at `ef = 1`, so just zoom-in)
//!   followed by a single `search_layer` at layer 0 with the
//!   user-supplied `ef`.
//!
//! ## Distance ordering convention
//!
//! Per [`crate::Metric`], L2 / Hamming are "lower is closer" and
//! IP / Cosine are "higher is closer". simsimd's
//! `SpatialSimilarity::cosine` returns `1 - cos(θ)` so the
//! cosine kernel here is also "lower is closer". Inner-product
//! (`Ip`) is the only metric where lower-is-closer is wrong — and
//! v1.0 callers using IP for top-k pre-normalize their vectors so
//! IP becomes equivalent to cosine. This v1.0 baseline therefore
//! ranks **monotonically by raw kernel output ascending**; if a
//! caller hands in a "higher-is-closer" kernel they get the
//! inverse ranking. Slice E.2 wires kernel-aware ordering into
//! the rescore path; Slice C ships the lower-is-closer baseline
//! that L2 / Cosine want directly.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use ordered_float::OrderedF32;

use crate::Metric;
use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;

use super::graph::HnswGraph;

/// Lightweight `OrderedF32` wrapper used by `BinaryHeap`.
/// `f32` does not implement `Ord` (NaN is unordered); we lift
/// it into a totally-ordered shim that panics on NaN. NaN can
/// only enter via the kernel; tests pin NaN behavior of the
/// cosine-on-zero-vector OQ-V5 case explicitly.
pub(crate) mod ordered_float {
    use std::cmp::Ordering;

    /// Total-order wrapper around `f32`. NaN compares as the
    /// **largest** value (so the closest-neighbors min-heap
    /// pushes NaN to the end of the ranking — they cannot
    /// shadow real top-k results). The zero-vector cosine
    /// regression test pins this so future simsimd upgrades
    /// surface as a behavior change rather than a silent
    /// recall regression.
    #[derive(Debug, Clone, Copy)]
    pub struct OrderedF32(pub f32);

    impl PartialEq for OrderedF32 {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            // NaN == NaN under the total-order projection so
            // BinaryHeap's invariants hold even when the kernel
            // returns NaN.
            (self.0.is_nan() && other.0.is_nan()) || self.0 == other.0
        }
    }

    impl Eq for OrderedF32 {}

    impl PartialOrd for OrderedF32 {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for OrderedF32 {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            match (self.0.is_nan(), other.0.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => self.0.partial_cmp(&other.0).expect("non-NaN compared"),
            }
        }
    }
}

/// One step of the search: a `(distance, VectorId)` pair.
/// Distance ordering uses [`OrderedF32`] for total order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Candidate {
    pub(crate) distance: OrderedF32,
    pub(crate) id: VectorId,
}

impl Candidate {
    #[inline]
    pub(crate) fn new(distance: f32, id: VectorId) -> Self {
        Self {
            distance: OrderedF32(distance),
            id,
        }
    }
}

/// Algorithm 2: search-layer.
///
/// Bounded best-first search at `layer`. Starts from the entry
/// points `eps`; expands the closest unvisited node at each
/// step; stops when no candidate is closer than the worst kept
/// result. Returns the `ef` nearest visited nodes.
///
/// **Input invariant:** `eps` MUST be at this layer (or higher);
/// `search_layer` does not validate that.
///
/// **Tombstone discipline:** tombstoned vectors are still used
/// as routing hubs — the walk treats them like any other node,
/// preserving ADR-003 Strategy 1's "lazy delete preserves
/// connectivity" property. The result list, however, is filtered
/// by the caller (`HnswGraph::search`) before returning to the
/// user.
pub(crate) fn search_layer(
    graph: &HnswGraph,
    query: &[u8],
    eps: &[VectorId],
    layer: usize,
    ef: usize,
    kernel: &dyn DistanceKernel,
) -> Vec<Candidate> {
    debug_assert!(ef >= 1, "search_layer ef must be ≥ 1");

    let mut visited: HashSet<VectorId> = HashSet::new();
    // candidates: min-heap by distance — "the next node to expand
    // is the closest unexplored one". `BinaryHeap` is a max-heap;
    // we wrap in `Reverse` so peek = min.
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    // results: max-heap by distance — "drop the worst kept when
    // we exceed `ef`". `BinaryHeap` is naturally a max-heap.
    let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

    for &ep in eps {
        if !visited.insert(ep) {
            continue;
        }
        // Skip phantom entry points (e.g., MN-RU detected the
        // graph held a stale entry pointer — we still need
        // search to remain robust).
        let Some(bytes) = graph.vector_bytes(ep) else {
            continue;
        };
        let d = kernel.distance(query, bytes);
        let c = Candidate::new(d, ep);
        candidates.push(Reverse(c));
        results.push(c);
    }

    while let Some(Reverse(c)) = candidates.pop() {
        // Termination: if the closest unexpanded candidate is
        // farther than the worst kept result, no further
        // expansion can improve `results`.
        if let Some(furthest_kept) = results.peek() {
            if results.len() >= ef && c.distance > furthest_kept.distance {
                break;
            }
        }

        // Expand `c.id`'s `layer`-adjacency.
        let Some(node) = graph.nodes.get(&c.id) else {
            continue;
        };
        let Some(layer_adj) = node.neighbors.get(layer) else {
            continue;
        };
        for &n in layer_adj {
            if !visited.insert(n) {
                continue;
            }
            let Some(nbytes) = graph.vector_bytes(n) else {
                continue;
            };
            let nd = kernel.distance(query, nbytes);
            let cand = Candidate::new(nd, n);
            // Either the result is not yet full, or this is
            // strictly closer than the worst kept.
            let push = match results.peek() {
                Some(worst) => results.len() < ef || cand.distance < worst.distance,
                None => true,
            };
            if push {
                candidates.push(Reverse(cand));
                results.push(cand);
                if results.len() > ef {
                    let _ = results.pop();
                }
            }
        }
    }

    let mut out: Vec<Candidate> = results.into_iter().collect();
    // Sorted ascending — closest first.
    out.sort();
    out
}

impl HnswGraph {
    /// Algorithm 5: top-`k` HNSW search.
    ///
    /// Returns up to `k` `(VectorId, distance)` pairs sorted by
    /// closeness (ascending — closest first per the kernel's
    /// metric convention). Tombstoned vectors are filtered out
    /// of the result set per ADR-003 Strategy 1.
    ///
    /// `ef` is the beam width at the bottom layer; per Malkov &
    /// Yashunin §4 it must be `>= k` for a meaningful result.
    /// The implementation does not enforce that — a caller
    /// passing `ef < k` will simply receive at most `ef` results
    /// (correct, but degraded recall). If `ef == 0`, the
    /// `params.ef_search` default is used.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] if `query.len()`
    ///   does not equal `bytes_per_vector`.
    pub fn search(
        &self,
        query: &[u8],
        k: usize,
        ef: usize,
        kernel: &dyn DistanceKernel,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        self.validate_vector_bytes(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let Some(entry) = self.entry_point else {
            // Empty graph or all entries swept — return empty,
            // not an error. Callers must tolerate empty results.
            return Ok(Vec::new());
        };

        let ef_use = if ef == 0 { self.params.ef_search } else { ef };
        let ef_layer0 = ef_use.max(k);

        // Greedy zoom: at each upper layer, run search with
        // `ef = 1` and replace the entry point with the closest
        // result. Per Algorithm 5 lines 2–6.
        let mut eps: Vec<VectorId> = vec![entry];
        for layer in (1..=self.max_level).rev() {
            let next = search_layer(self, query, &eps, layer, 1, kernel);
            if let Some(c) = next.first() {
                eps = vec![c.id];
            }
        }

        // Layer 0: the meaty beam search.
        let mut results = search_layer(self, query, &eps, 0, ef_layer0, kernel);

        // ADR-003 Strategy 1: skip tombstoned vectors in result
        // list, preserving them in the routing graph.
        results.retain(|c| !self.is_tombstoned(c.id));

        // Truncate to top-`k` (already sorted ascending).
        results.truncate(k);

        Ok(results.into_iter().map(|c| (c.id, c.distance.0)).collect())
    }

    /// Top-`k` HNSW search with full-precision rescore per
    /// ADR-035 §3.3 / AC-1a.
    ///
    /// Two-stage retrieval:
    ///
    /// 1. **Primary search.** Beam search at the graph's storage
    ///    encoding (typically the quantized form — SQ8 or binary)
    ///    via `primary_kernel`, collecting `k * rescore_factor`
    ///    candidates. Cheap distance, approximate ranking.
    /// 2. **Full-precision rescore.** For each primary-search
    ///    candidate, fetch the full-precision (F32 / F16) vector
    ///    bytes via `full_precision_vectors` and recompute the
    ///    distance using `full_precision_kernel`. Sort ascending
    ///    by the **metric-direction-aware** key (negate the raw
    ///    distance for `Metric::Ip` so larger inner-product = smaller
    ///    key = closer; raw distance for L2 / Hamming / Cosine
    ///    where lower = closer per the kernel surface convention)
    ///    and return the top `k`. The returned tuples carry the
    ///    natural-orientation `(VectorId, raw_distance)` so callers
    ///    see the kernel's native distance value, not the internal
    ///    sort key.
    ///
    /// This is the canonical SQ8 / binary deployment path: the
    /// SQ8-alone recall floor (AC-1b ≥ 0.92) can dip below the
    /// ship-blocking 0.95 bar on harder datasets, but rescoring
    /// against the full-precision arena recovers recall to
    /// ≥ 0.95 (AC-1a) at the cost of `~rescore_factor × k`
    /// full-precision distance ops per query — negligible vs the
    /// hundreds of beam-search distance ops saved by quantizing
    /// the primary path.
    ///
    /// ## Why two queries
    ///
    /// `query` and `full_precision_query` carry the SAME logical
    /// query but in DIFFERENT encodings: `query` matches the
    /// graph's storage encoding (validated by the underlying
    /// [`HnswGraph::search`] call against `bytes_per_vector`);
    /// `full_precision_query` matches whatever encoding
    /// `full_precision_kernel` consumes. For SQ8 + F32 at
    /// dim=768 the byte lengths are 768 vs 3072 — incompatible
    /// in a single byte slice. The caller is responsible for
    /// pre-encoding both: in production this is the per-tenant
    /// arena (Slice F.1) holding the codebook; in tests it is
    /// the test harness invoking `crate::Sq8Codebook::encode`
    /// directly.
    ///
    /// The base [`ADR-035 §3.3`] sketch refers to a single
    /// "query" without distinguishing the two encodings. Slice
    /// E.2 surfaces them explicitly so the implementation is
    /// type-safe (the alternative — a single `query` slice fed
    /// to two kernels with different per-element widths — would
    /// underflow the rescore kernel's length check and produce
    /// nonsense distances).
    ///
    /// ## Edge cases
    ///
    /// - `rescore_factor == 0` → [`VectorIndexError::InvalidRescoreFactor`].
    ///   Per ADR-035 D-4, `rescore_factor = 1` is the operator
    ///   opt-out (SQ8-alone, AC-1b best-effort), not an error;
    ///   `0` is the only invalid value at the public surface.
    /// - `rescore_factor == 1` → short-circuits to a direct
    ///   [`HnswGraph::search`] call; `full_precision_query`,
    ///   `full_precision_kernel`, and `full_precision_vectors`
    ///   are NOT invoked (no rescore overhead). This honors the
    ///   D-4 opt-out: latency-sensitive operators that accept
    ///   the 0.92 SQ8-alone floor pay no rescore cost.
    /// - `k == 0` → returns `Ok(vec![])` (matches `search`).
    /// - `k * rescore_factor > self.len()` → saturates at
    ///   `self.len()` (oversampling beyond the graph size is
    ///   meaningless; `usize` overflow is also handled via
    ///   `saturating_mul`).
    ///
    /// ## Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] if `query.len()`
    ///   does not equal the graph's `bytes_per_vector`
    ///   (propagated from `search`).
    /// - [`VectorIndexError::InvalidRescoreFactor`] if
    ///   `rescore_factor == 0`.
    /// - [`VectorIndexError::RescoreVectorMissing`] if
    ///   `full_precision_vectors(vector_id)` returns `None` for
    ///   any primary-search candidate (rescore arena out of sync
    ///   with primary index — an operator-visible inconsistency).
    // Argument count exceeds clippy's default 6 because the rescore
    // path is fundamentally a join across two arenas (primary +
    // full-precision) each requiring its own (query, kernel) pair
    // plus the rescore-arena lookup callback. Bundling these into a
    // struct adds ceremony without clarity at the only call site
    // pattern (per-tenant arena dispatcher, Slice F.1+); follows
    // the same shape as `arcgraph-storage::crud::*` which carries
    // the same allow per the existing codebase precedent.
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_rescore<'a>(
        &self,
        query: &[u8],
        full_precision_query: &[u8],
        k: usize,
        ef_search: usize,
        rescore_factor: usize,
        primary_kernel: &dyn DistanceKernel,
        full_precision_kernel: &dyn DistanceKernel,
        full_precision_vectors: &dyn Fn(VectorId) -> Option<&'a [u8]>,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        // `0` is the only invalid factor; `1` is the operator
        // opt-out per D-4 (handled below as the short-circuit
        // path) and `≥ 2` is the rescore path proper.
        if rescore_factor == 0 {
            return Err(VectorIndexError::InvalidRescoreFactor { factor: 0 });
        }

        // Operator opt-out: factor=1 is "SQ8-alone" / "binary-alone".
        // No rescore work; defer to the primary search verbatim.
        // The full-precision query / kernel / lookup callback are
        // unused — documented contract so callers may pass dummy
        // values (e.g., a placeholder closure that always panics).
        if rescore_factor == 1 {
            return self.search(query, k, ef_search, primary_kernel);
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        // `k * rescore_factor` candidates from primary; saturate at
        // graph size so oversampling beyond the corpus does not
        // waste search effort and so usize overflow does not
        // silently truncate to zero.
        let oversample = k.saturating_mul(rescore_factor).min(self.len());

        // The graph may hold fewer vectors than `k` — in that
        // case `oversample` collapses to `self.len()` (possibly
        // less than k), and the underlying search returns at most
        // that many results. We still call through so the
        // tombstone filter + dim validation in `search` runs.
        let oversample = oversample.max(1);

        // Step 1: primary-kernel beam search.
        let primary_results = self.search(query, oversample, ef_search, primary_kernel)?;

        // Steps 2 + 3: rescore each candidate against
        // full-precision. We pre-allocate at the primary result
        // size; this is at most `oversample` and typically equals
        // it (the tombstone filter may shrink it).
        //
        // We carry a third field `distance_key` per candidate so
        // the sort below is **metric-direction-aware**. The raw
        // kernel distance is what callers see at the API boundary;
        // the key is what we sort by internally.
        let metric = full_precision_kernel.metric();
        let mut rescored: Vec<(VectorId, f32, f32)> = Vec::with_capacity(primary_results.len());
        for (vector_id, _quantized_distance) in primary_results {
            let full_precision_bytes = full_precision_vectors(vector_id)
                .ok_or(VectorIndexError::RescoreVectorMissing { vector_id })?;
            let raw = full_precision_kernel.distance(full_precision_query, full_precision_bytes);
            // Metric-direction-aware sort key: ascending key = closer
            // for every metric. L2 / Hamming / Cosine kernels return
            // "lower is closer" already; IP returns "higher is closer"
            // per `crate::Metric::Ip`'s convention, so we negate to
            // align it with the same ascending-= -closer ordering.
            // Without this, ascending raw IP would return the most
            // anti-aligned candidates first — the W-1 correctness gap
            // surfaced in PR #115 review and back-ported here from
            // the Slice E.3 (DiskANN rescore) sibling.
            let distance_key = match metric {
                Metric::L2 | Metric::Hamming | Metric::Cosine => raw,
                Metric::Ip => -raw,
            };
            rescored.push((vector_id, raw, distance_key));
        }

        // Step 4: sort ascending by `distance_key` (smaller key =
        // closer for every metric). `OrderedF32` keeps the same
        // NaN-as-largest discipline as the primary search so a
        // corrupt vector that produces NaN cannot shadow real
        // top-k regardless of metric. Tie-break on `VectorId.raw()`
        // for deterministic output across runs.
        rescored.sort_by(|a, b| {
            ordered_float::OrderedF32(a.2)
                .cmp(&ordered_float::OrderedF32(b.2))
                .then_with(|| a.0.raw().cmp(&b.0.raw()))
        });

        // Step 5: truncate to top-k and project back to the
        // natural-orientation `(VectorId, raw_distance)` shape so
        // callers see their kernel's native distance value (not
        // the internal `distance_key`).
        rescored.truncate(k);
        Ok(rescored
            .into_iter()
            .map(|(id, raw, _key)| (id, raw))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::ordered_float::OrderedF32;

    #[test]
    fn ordered_f32_total_orders_nan_as_largest() {
        let nan = OrderedF32(f32::NAN);
        let one = OrderedF32(1.0);
        assert!(one < nan, "NaN must compare as ≥ a real value");
        assert!(
            nan == nan,
            "NaN must compare equal to itself for heap soundness"
        );
    }

    #[test]
    fn ordered_f32_orders_normally_otherwise() {
        let lo = OrderedF32(0.5);
        let hi = OrderedF32(1.5);
        assert!(lo < hi);
    }

    #[test]
    fn ordered_f32_handles_negative_values() {
        let neg = OrderedF32(-1.0);
        let zero = OrderedF32(0.0);
        let pos = OrderedF32(1.0);
        assert!(neg < zero);
        assert!(zero < pos);
    }
}
