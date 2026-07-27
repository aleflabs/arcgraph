//! `HnswGraph` data structure.
//!
//! Layered Hierarchical Navigable Small World index per Malkov &
//! Yashunin TPAMI 2018 §3.1. The bottom layer (layer 0) holds
//! every vector; upper layers form a sparse navigation
//! hierarchy where each higher layer contains a stochastically
//! shrinking subset.
//!
//! The data layout is hash-table-backed rather than array-backed
//! because `VectorId`s come straight from the catalog allocator
//! (sparse if a delete-and-reinsert sequence holes them out).
//! Slice F.1 (per-tenant arena routing) layers a contiguous
//! arena on top; this slice keeps the single-graph layout
//! algorithm-faithful and dependency-free.
//!
//! ## Latency / memory budget (back-of-envelope)
//!
//! For a 1 M-vector × 768-dim × f32 graph with `M=32`, every
//! base-layer node holds 32 × `u32` = 128 B of edges, plus 3 KB
//! of vector bytes, plus the upper-layer edges (~1.5×
//! amplification across layers per the paper's geometric mean).
//! Per-vector memory: ~3.5 KB; total ≈ 3.5 GB at 1 M, 35 GB at
//! 10 M — fits the 64 GB budget per design-v2 §A.1 with the
//! per-tenant arena routing of Slice F.1 splitting the graph.
//!
//! Search hot path is `O(ef_search · M · log N)` distance
//! evaluations per query; at `ef_search=128`, `M=32`, `N=1 M`
//! that's ~80 K evaluations. With simsimd's f32 L2 at ≈ 5 ns /
//! 768-dim op, ≈ 0.4 ms / query — comfortably inside the
//! P95 ≤ 8 ms budget per ADR-003 / ADR-035 NFR-15.

use std::collections::HashMap;

use parking_lot::RwLock;
use rand::{SeedableRng, rngs::StdRng};

use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;

/// Tunable parameters that shape the HNSW graph topology.
///
/// Per ADR-003 the v1.0 default is `M=32`, `ef_construction=200`,
/// `ef_search=128` (the workload-validated set from
/// ANN-Benchmarks + Codastra "Vector DB Showdown" 2025). The
/// default impl ships those numbers; tests routinely override
/// to smaller values for speed.
///
/// `seed` controls the stochastic level assignment + neighbor
/// tie-break RNG; pinning it makes proptests reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    /// Target out-degree for upper layers. Layer 0 caps at
    /// `M_max0 = 2*M` per the paper (Malkov & Yashunin §4
    /// Algorithm 1 line 17).
    pub m: usize,
    /// Beam width during build (Algorithm 1 line 18).
    pub ef_construction: usize,
    /// Default beam width during search if the caller passes 0
    /// — in practice every caller passes an explicit value, but
    /// the field documents the v1.0 commitment per ADR-003.
    pub ef_search: usize,
    /// RNG seed for level assignment + tie-breaking.
    pub seed: u64,
}

impl Default for HnswParams {
    /// Per ADR-003: `M=32`, `ef_construction=200`,
    /// `ef_search=128`. Seed 0 — deterministic by default; tests
    /// or production callers that want reseeding pass a
    /// different value.
    #[inline]
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            ef_search: 128,
            seed: 0,
        }
    }
}

impl HnswParams {
    /// `M_max` for non-zero layers. Equals `m` per the paper.
    #[inline]
    #[must_use]
    pub const fn m_max(self) -> usize {
        self.m
    }

    /// `M_max0` for layer 0 — `2*M` per the paper. Layer 0 is
    /// where every vector lives, so we allow a higher fan-out
    /// to improve recall for the leaf-most distance evaluations.
    #[inline]
    #[must_use]
    pub const fn m_max0(self) -> usize {
        2 * self.m
    }

    /// Inverse-log normalization factor for the level-assignment
    /// distribution: `mL = 1 / ln(M)`. The expected number of
    /// nodes at level `l` is then `N · M^(-l)` for a graph of
    /// size `N`. Per Malkov & Yashunin 2018 §4 — the value that
    /// keeps the layered structure log-spaced.
    #[inline]
    #[must_use]
    pub fn level_norm(self) -> f64 {
        // `m = 1` would make `ln(m) = 0` and the level
        // assignment ill-defined; we forbid that at construction
        // (HnswGraph::new panics on invalid params).
        debug_assert!(self.m >= 2, "HnswParams::m must be ≥ 2");
        1.0 / (self.m as f64).ln()
    }
}

/// One node's per-layer adjacency.
///
/// Index `0` is the base layer (always populated for every
/// node); higher indices are populated only up to the node's
/// stochastically-assigned `level`.
#[derive(Debug, Default, Clone)]
pub(crate) struct NodeNeighbors {
    /// `neighbors[layer]` is the adjacency list at that layer.
    /// `neighbors.len() == level + 1`.
    pub(crate) neighbors: Vec<Vec<VectorId>>,
}

impl NodeNeighbors {
    /// The highest layer this node participates in.
    #[inline]
    pub(crate) fn level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }
}

/// Hierarchical Navigable Small World graph (in-memory baseline).
///
/// Owns the encoded vector bytes, the per-layer adjacency, and
/// the tombstone bitmap. Distance-kernel-agnostic: every method
/// that needs to compute distances takes a `&dyn DistanceKernel`
/// at the call site, so the same graph can be searched with
/// different kernels (e.g., L2F32 build, CosineF32 query) at the
/// caller's discretion. This is the same indirection the
/// quantized-rescore path (Slice E.2) uses.
pub struct HnswGraph {
    pub(crate) params: HnswParams,
    /// Vector dimension (number of f32 components). Cached at
    /// construction so per-insert validation is `O(1)`.
    pub(crate) dim: usize,
    /// Bytes-per-vector (cached for slice-length validation).
    pub(crate) bytes_per_vector: usize,

    /// Encoded vector bytes, keyed by `VectorId`.
    pub(crate) vectors: HashMap<VectorId, Vec<u8>>,

    /// Per-node adjacency lists (per-layer).
    pub(crate) nodes: HashMap<VectorId, NodeNeighbors>,

    /// Soft-delete bitmap. Set bits mean "this VectorId is
    /// tombstoned; skip it when collecting search results, but
    /// still use it as a routing hub" — preserves graph
    /// connectivity until rebuild fires per ADR-003 Strategy 1.
    pub(crate) tombstones: HashMap<VectorId, bool>,

    /// Top-level entry point for search. `None` iff the graph
    /// has no inserted (non-tombstoned routing-still-alive)
    /// nodes. Updated whenever a freshly-inserted vector lands
    /// at a strictly higher level than the current entry's.
    pub(crate) entry_point: Option<VectorId>,

    /// Highest level any node currently occupies. Invariant:
    /// `entry_point.is_some() ⇒ self.nodes[entry_point].level() == max_level`.
    pub(crate) max_level: usize,

    /// RNG used for stochastic level assignment. Wrapped in a
    /// lock so `&self` query methods can still mutate the
    /// generator state if a future variant needs randomized
    /// tie-breaking; today only `&mut self` insert needs it.
    pub(crate) rng: RwLock<StdRng>,
}

impl HnswGraph {
    /// Construct an empty HNSW graph with the given parameters
    /// and vector dimension. The `kernel` argument is consulted
    /// only for `bytes_per_vector` derivation (so that insert /
    /// search can validate slice lengths).
    ///
    /// # Panics
    ///
    /// - If `params.m < 2`. The level-assignment distribution
    ///   needs `ln(M) > 0` and `M < 2` collapses to a flat
    ///   list (which is not what the caller wants).
    /// - If `params.ef_construction == 0`. Beam width zero
    ///   degenerates to greedy descent only and is not the v1.0
    ///   commitment.
    /// - If `dim == 0`. Empty vectors are not insertable; rather
    ///   than discovering this on the first insert, fail loud at
    ///   construction.
    #[must_use]
    pub fn new(params: HnswParams, dim: usize, kernel: &dyn DistanceKernel) -> Self {
        assert!(
            params.m >= 2,
            "HnswParams::m must be ≥ 2 (got {})",
            params.m
        );
        assert!(
            params.ef_construction >= 1,
            "HnswParams::ef_construction must be ≥ 1 (got {})",
            params.ef_construction
        );
        assert!(dim > 0, "HnswGraph dim must be > 0");

        let bytes_per_vector = kernel.encoding().bytes_per_vector_unaligned(dim);
        Self {
            params,
            dim,
            bytes_per_vector,
            vectors: HashMap::new(),
            nodes: HashMap::new(),
            tombstones: HashMap::new(),
            entry_point: None,
            max_level: 0,
            rng: RwLock::new(StdRng::seed_from_u64(params.seed)),
        }
    }

    /// Number of vectors currently in the graph (tombstoned
    /// vectors counted — they still occupy slots and serve as
    /// routing hubs per ADR-003 Strategy 1).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the graph holds zero vectors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Whether `id` is present (tombstoned or not).
    #[inline]
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }

    /// Whether `id` is currently tombstoned.
    #[inline]
    #[must_use]
    pub fn is_tombstoned(&self, id: VectorId) -> bool {
        self.tombstones.get(&id).copied().unwrap_or(false)
    }

    /// Number of tombstoned vectors.
    #[inline]
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.values().filter(|t| **t).count()
    }

    /// `tombstone_count() / len()`, in `[0, 1]`. Returns `0.0` on
    /// an empty graph (no vectors → no ratio to report). ADR-003
    /// Strategy 1 wires the rebuild trigger at `> 0.30`; ADR-035
    /// §5.3 cites `> 0.10` as the operational alert threshold.
    #[inline]
    #[must_use]
    pub fn tombstone_ratio(&self) -> f64 {
        let total = self.len();
        if total == 0 {
            return 0.0;
        }
        self.tombstone_count() as f64 / total as f64
    }

    /// Return the `(VectorId, byte-slice)` pair for an existing
    /// node — `None` if not present. Used by the search /
    /// insert paths and by the MN-RU repair walk; intentionally
    /// not exposed publicly because callers should go through
    /// `search` / `insert`.
    #[inline]
    pub(crate) fn vector_bytes(&self, id: VectorId) -> Option<&[u8]> {
        self.vectors.get(&id).map(Vec::as_slice)
    }

    /// Validate that a query / insert byte slice matches the
    /// arena's `bytes_per_vector`. The error type is the v1.0
    /// codec-local one; callers translate at the public boundary
    /// per `docs/codec-error-translation.md`.
    pub(crate) fn validate_vector_bytes(&self, bytes: &[u8]) -> Result<(), VectorIndexError> {
        if bytes.len() != self.bytes_per_vector {
            // Best-effort: surface as a dim-mismatch in units of
            // dim (more useful than units of bytes for callers).
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim,
                got: bytes.len() / std::mem::size_of::<f32>().max(1),
            });
        }
        Ok(())
    }

    // ─── tombstone bitmap (ADR-003 Strategy 1) ─────────────────

    /// Mark `id` deleted. The vector + adjacency stay in place
    /// (preserving graph connectivity); subsequent searches skip
    /// it from the result list. Idempotent: marking an
    /// already-tombstoned vector is a no-op. Calling on a
    /// non-existent vector is a no-op (graceful degradation —
    /// the vector may have been swept by a concurrent rebuild).
    pub fn mark_deleted(&mut self, id: VectorId) {
        if self.vectors.contains_key(&id) {
            self.tombstones.insert(id, true);
        }
    }

    /// Clear `id`'s tombstone bit. Used by the MN-RU repair path
    /// (re-inserting a previously-tombstoned vector revives it)
    /// and by tests — production callers normally rebuild rather
    /// than un-tombstone.
    pub fn clear_tombstone(&mut self, id: VectorId) {
        self.tombstones.insert(id, false);
    }

    // ─── RNG access ────────────────────────────────────────────

    /// Sample the level-assignment distribution per Malkov &
    /// Yashunin 2018 §4 algorithm 1: `floor(-ln(uniform()) * mL)`.
    /// The closed form is the inverse CDF of the geometric
    /// distribution with parameter `1 - 1/M`; per the paper §4
    /// it keeps each upper layer's expected size geometrically
    /// shrinking by `1/M`.
    pub(crate) fn sample_level(&self) -> usize {
        use rand::RngExt;
        let mut rng = self.rng.write();
        // sample u in (0, 1] — open at 0 to avoid -ln(0) = +inf.
        // rng.random_range::<f64, _>(..) yields [0, 1); flip via
        // `1.0 - …` so the result is (0, 1].
        let u: f64 = 1.0 - rng.random_range::<f64, _>(0.0..1.0);
        let l = (-(u.ln()) * self.params.level_norm()).floor() as i64;
        l.max(0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2F32;

    #[test]
    fn default_params_match_adr_003() {
        let p = HnswParams::default();
        assert_eq!(p.m, 32);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.ef_search, 128);
    }

    #[test]
    fn m_max_and_m_max0_match_paper() {
        let p = HnswParams::default();
        assert_eq!(p.m_max(), 32);
        assert_eq!(p.m_max0(), 64);
    }

    #[test]
    fn level_norm_is_inverse_log_m() {
        let p = HnswParams::default();
        let expected = 1.0 / (32.0_f64).ln();
        assert!((p.level_norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn empty_graph_has_zero_len_and_no_entry_point() {
        let g = HnswGraph::new(HnswParams::default(), 4, &L2F32);
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.entry_point, None);
    }

    #[test]
    fn tombstone_ratio_zero_on_empty() {
        let g = HnswGraph::new(HnswParams::default(), 4, &L2F32);
        assert!((g.tombstone_ratio() - 0.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "HnswParams::m must be ≥ 2")]
    fn new_panics_on_m_below_two() {
        let p = HnswParams {
            m: 1,
            ..HnswParams::default()
        };
        let _ = HnswGraph::new(p, 4, &L2F32);
    }

    #[test]
    #[should_panic(expected = "HnswParams::ef_construction must be ≥ 1")]
    fn new_panics_on_zero_ef_construction() {
        let p = HnswParams {
            ef_construction: 0,
            ..HnswParams::default()
        };
        let _ = HnswGraph::new(p, 4, &L2F32);
    }

    #[test]
    #[should_panic(expected = "HnswGraph dim must be > 0")]
    fn new_panics_on_zero_dim() {
        let _ = HnswGraph::new(HnswParams::default(), 0, &L2F32);
    }

    #[test]
    fn sample_level_is_non_negative() {
        let g = HnswGraph::new(HnswParams::default(), 4, &L2F32);
        for _ in 0..1000 {
            // Non-negative by construction; but we also assert
            // the distribution is finite — `usize` can't be ∞,
            // so we'll catch any future arithmetic bug.
            let l = g.sample_level();
            assert!(l < 100, "level {l} suspiciously large");
        }
    }

    #[test]
    fn sample_level_distribution_log_spaced() {
        // E[level=0] should dominate; only a small fraction of
        // samples lands at level ≥ 1, and even fewer at ≥ 2.
        // This is a coarse sanity check on the distribution
        // formula — drift on this would indicate a regression.
        let g = HnswGraph::new(HnswParams::default(), 4, &L2F32);
        let n = 10_000;
        let mut at_zero = 0usize;
        let mut at_one_plus = 0usize;
        for _ in 0..n {
            let l = g.sample_level();
            if l == 0 {
                at_zero += 1;
            } else {
                at_one_plus += 1;
            }
        }
        // For M=32, expected fraction at level 0 is roughly
        // 1 - 1/M = 31/32 ≈ 0.97. Allow ±3 % tolerance.
        let frac_zero = at_zero as f64 / n as f64;
        assert!(
            (0.94..0.99).contains(&frac_zero),
            "frac_zero = {frac_zero}; expected ~0.97; level-1+ count = {at_one_plus}"
        );
    }
}
