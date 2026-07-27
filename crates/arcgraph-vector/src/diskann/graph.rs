//! DiskANN graph data structures.
//!
//! [`DiskAnnGraph`] holds the Vamana directed graph (parallel
//! arrays indexed by an internal slot `u32`), the entry-point
//! medoid, the streaming delta-segment, and the per-label entry
//! cache for Filtered-DiskANN scaffolding.
//!
//! Per ADR-035 §3.1 + §5.3 + Subramanya et al. NeurIPS 2019.
//!
//! ## Storage shape
//!
//! Vectors are stored as encoded byte slices (whatever the
//! arena's [`crate::Encoding`] dictates — F32, F16, SQ8,
//! Binary, RaBitQ). The graph itself is **encoding-agnostic**: distance
//! is computed by the configured [`crate::DistanceKernel`],
//! which takes care of the byte interpretation.
//!
//! Three parallel arrays plus a hash index implement the
//! `slot u32 ↔ VectorId u32 ↔ &[u8]` mapping:
//!
//! - `ids: Vec<VectorId>` — slot → external id.
//! - `vectors: Vec<u8>` — slot → encoded bytes in one contiguous arena.
//! - `neighbors: Vec<Vec<u32>>` — slot → out-edges (slot indices).
//! - `id_to_slot: HashMap<VectorId, u32>` — reverse lookup.
//!
//! Deletes set the tombstone bit; the slot is not reclaimed at
//! Slice D (ADR-003 Strategy 1 preserved). Slice F's rebuild
//! path compacts.

use std::collections::HashMap;

use arcgraph_core::Lsn;

use crate::{DistanceKernel, Encoding, Metric, Result, VectorId, VectorIndexError};

use super::stream::DeltaSegment;

/// Label identifier used by Filtered-DiskANN scaffolding.
///
/// Slice D ships the per-label entry-point cache so the Slice
/// F.3 (filter-aware Vamana build + search) lands without
/// reshaping `DiskAnnGraph`. The concrete planner-level
/// `LabelId` newtype lives in `arcgraph-core::ids`; the vector
/// crate accepts an opaque `u32` here so it does not pull
/// label semantics across the bounded-context boundary at v1.0.
pub type DiskAnnLabelId = u32;

/// Tunable parameters for Vamana build + search.
///
/// Defaults match the Slice D handoff prompt + ADR-035 §5.3.
/// The planner sets these at `DEFINE INDEX` time per ADR-035
/// §5.1; Slice D consumes them as-is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskAnnParams {
    /// Maximum out-degree per node (R in Subramanya 2019). The
    /// build prunes the candidate frontier to this size after
    /// each greedy walk; on-disk graph layout (Slice G) packs
    /// `R` 4-byte node ids per vertex.
    pub r: u32,
    /// α-pruning parameter (Vamana §3). `α = 1` gives the GNN
    /// graph; `α > 1` increases edge density for better recall
    /// at the cost of build time + graph size.
    pub alpha: f32,
    /// Beam width during build (`L_construction` in Vamana §3).
    /// Wider beams find higher-quality neighbor candidates at
    /// the cost of build wall-clock.
    pub l_construction: u32,
    /// Default beam width during search (`l_search` in
    /// Vamana §3). Caller may override per query.
    pub l_search_default: u32,
    /// Auto-merge threshold for the streaming delta-segment.
    /// When `delta.len() ≥ delta_max_size`, the next
    /// [`crate::diskann::DiskAnnGraph::insert_stream`] folds
    /// the delta into the main graph. Per ADR-035 §5.3.1
    /// the production default is 1 000.
    pub delta_max_size: u32,
    /// Brute-force vs in-memory-HNSW pivot for the
    /// delta-segment. While `delta.len() < delta_brute_thresh`,
    /// the delta is searched by linear scan (matches Faiss
    /// `IndexFlatL2` baseline). Slice D ships brute-force only;
    /// the small-HNSW promotion branch is wired but defaults to
    /// brute-force pending Slice F's HNSW reuse.
    pub delta_brute_thresh: u32,
    /// Random sample size for medoid approximation.
    /// Computing the exact medoid is `O(n²)`; per the Microsoft
    /// DiskANN reference impl we sample `min(n,
    /// medoid_sample_size)` vectors and pick the sample point
    /// with the smallest sum-of-distances to the rest of the
    /// sample. Quality of entry-point is a constant-factor
    /// regression target, not a recall blocker.
    pub medoid_sample_size: u32,
}

impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            r: 70,
            alpha: 1.2,
            l_construction: 100,
            l_search_default: 100,
            delta_max_size: 1000,
            delta_brute_thresh: 128,
            medoid_sample_size: 1000,
        }
    }
}

impl DiskAnnParams {
    /// Validate parameter sanity. Called from
    /// [`DiskAnnGraph::new`]; caller may surface a
    /// [`VectorIndexError`] of their choice.
    pub(crate) fn validate(self) -> Result<()> {
        if self.r == 0 {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: crate::IndexId::ZERO,
                reason: "DiskAnnParams.r must be > 0".into(),
            });
        }
        if !(self.alpha.is_finite() && self.alpha >= 1.0) {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: crate::IndexId::ZERO,
                reason: format!(
                    "DiskAnnParams.alpha must be finite and ≥ 1.0 (got {})",
                    self.alpha
                ),
            });
        }
        if self.l_construction < self.r {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: crate::IndexId::ZERO,
                reason: format!(
                    "DiskAnnParams.l_construction ({}) must be ≥ r ({})",
                    self.l_construction, self.r
                ),
            });
        }
        if self.l_search_default == 0 {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: crate::IndexId::ZERO,
                reason: "DiskAnnParams.l_search_default must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Per-label entry-point cache, populated by
/// [`DiskAnnGraph::set_label_index`].
///
/// Filtered-DiskANN scaffolding only at Slice D — Slice F.3
/// adds the label-aware Vamana α-pruning rule (Microsoft WWW
/// 2023) which uses these entry points during filtered beam
/// search. Slice D does NOT propagate labels through the build
/// edge selection.
#[derive(Debug, Clone, Default)]
pub(crate) struct LabelIndex {
    /// `label → entry-point slot` (medoid of the label-matching
    /// subgraph).
    pub(crate) entry_per_label: HashMap<DiskAnnLabelId, u32>,
    /// Optional per-node label list — `labels[slot]` returns
    /// `None` if the node carries no label or the label was
    /// never registered.
    pub(crate) labels: Vec<Option<DiskAnnLabelId>>,
}

/// In-memory Vamana graph + streaming delta-segment + per-label
/// index hooks.
///
/// Construct via [`DiskAnnGraph::new`]; bulk-build via
/// [`DiskAnnGraph::build`]; stream-insert via
/// [`DiskAnnGraph::insert_stream`]; query via
/// [`DiskAnnGraph::search`] or
/// [`DiskAnnGraph::search_with_delta`].
///
/// The graph stores its [`DistanceKernel`] internally so the
/// streaming and trait-impl paths can compute distances without
/// the caller threading the kernel through every method.
///
/// ## MVCC visibility (per ADR-041)
///
/// Two parallel arrays — `commit_lsns` and `expired_lsns` — pair
/// with `ids`/`vectors`/`neighbors`. Each slot's `(commit_lsn,
/// expired_lsn)` pair is consulted at search time when
/// `read_lsn` is supplied (see `filtered_search`). Defaults are
/// `(Lsn::ZERO, Lsn::MAX)` — always-visible — so callers without
/// LSN context preserve read-latest behavior. Mirrors the BM25
/// per-doc convention from ADR-039 §D-2 and the HNSW
/// `Payload.commit_lsn` / `Payload.expired_lsn` fields.
pub struct DiskAnnGraph {
    pub(crate) params: DiskAnnParams,
    pub(crate) encoding: Encoding,
    pub(crate) metric: Metric,
    pub(crate) kernel: Box<dyn DistanceKernel>,

    /// Slot → external [`VectorId`].
    pub(crate) ids: Vec<VectorId>,
    /// Contiguous encoded vector payloads. Slot `s` occupies
    /// `vectors[s * bytes_per_vector..(s + 1) * bytes_per_vector]`.
    pub(crate) vectors: Vec<u8>,
    /// Slot → out-neighbor slot list (Vamana edges).
    pub(crate) neighbors: Vec<Vec<u32>>,

    /// `VectorId → slot` reverse lookup.
    pub(crate) id_to_slot: HashMap<VectorId, u32>,

    /// Medoid slot; `None` while the graph is empty.
    pub(crate) entry_point: Option<u32>,

    /// Streaming delta-segment lookaside (per ADR-035 §5.3.1
    /// B-3 resolution).
    pub(crate) delta: DeltaSegment,

    /// Per-label entry-point cache (Slice F.3 scaffolding).
    pub(crate) label_idx: LabelIndex,

    /// Tombstone bitmap; one bit per slot. Search skips
    /// tombstoned slots.
    pub(crate) tombstones: Vec<u64>,

    /// Configured byte length of an encoded vector. Set on the
    /// first ingest (build/insert); subsequent ingests must
    /// match.
    pub(crate) bytes_per_vector: Option<usize>,

    /// Slot → MVCC commit LSN (per ADR-041 §D-3a). Defaults to
    /// `Lsn::ZERO` for every newly-allocated slot; populated
    /// explicitly via [`DiskAnnGraph::set_lsn_window`] when the
    /// caller has snapshot context.
    pub(crate) commit_lsns: Vec<Lsn>,
    /// Slot → MVCC expired LSN (per ADR-041 §D-3a). Defaults to
    /// `Lsn::MAX` for every newly-allocated slot. v1.0 in-place
    /// upserts keep this structurally `MAX` (mirrors ADR-039
    /// §D-2 for BM25). Populated alongside `commit_lsns` via
    /// [`DiskAnnGraph::set_lsn_window`].
    pub(crate) expired_lsns: Vec<Lsn>,
    /// Per-`VectorId` MVCC LSN window for delta-segment entries
    /// (per ADR-041 §D-3a). Slot allocation has not happened
    /// yet for delta entries; we store under the external id so
    /// the visibility filter can consult either store uniformly.
    /// On `merge_delta` the entry is migrated to the
    /// slot-indexed `commit_lsns` / `expired_lsns` arrays.
    pub(crate) delta_lsns: HashMap<VectorId, (Lsn, Lsn)>,
}

impl std::fmt::Debug for DiskAnnGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskAnnGraph")
            .field("params", &self.params)
            .field("encoding", &self.encoding)
            .field("metric", &self.metric)
            .field("len_main", &self.ids.len())
            .field("len_delta", &self.delta.len())
            .field("entry_point", &self.entry_point)
            .field("bytes_per_vector", &self.bytes_per_vector)
            .field("tombstone_count", &self.live_tombstone_count())
            .finish()
    }
}

impl DiskAnnGraph {
    /// Construct an empty Vamana graph for the given
    /// `(encoding, metric)` pair, using the given distance
    /// kernel and tuning parameters.
    ///
    /// The kernel's [`DistanceKernel::encoding`] /
    /// [`DistanceKernel::metric`] **must** match the supplied
    /// `encoding` / `metric` arguments. Mismatches surface as
    /// [`VectorIndexError::UnsupportedFlags`] — this is the
    /// arena/index assembly contract from ADR-035 §6.
    pub fn new(
        params: DiskAnnParams,
        encoding: Encoding,
        metric: Metric,
        kernel: Box<dyn DistanceKernel>,
    ) -> Result<Self> {
        params.validate()?;
        if !metric.is_valid_for(encoding) {
            return Err(VectorIndexError::UnsupportedFlags { encoding, metric });
        }
        if encoding == Encoding::RaBitQ && metric != Metric::L2 {
            // TODO(#758): OQ-4 — full-vector IP/cosine need centroid cross-terms
            // plus the alpha-prune comparator design. Slice 2 validates
            // bulk-build + search for L2 RaBitQ nav only; streaming under RaBitQ
            // remains mechanically coherent but unvalidated.
            return Err(VectorIndexError::UnsupportedFlags { encoding, metric });
        }
        // Per issue #109 defensive (a): reject Metric::Ip at
        // construction. The Vamana α-prune comparator in
        // `build::DiskAnnGraph::robust_prune` (and the streaming
        // mirror in `stream::robust_prune_inner`) is correct for
        // L2/Hamming and correct-by-accident for Cosine (simsimd
        // returns `1 − cos(θ)`, lower-is-closer), but inverted for
        // IP (raw kernel similarities are higher-is-closer). The
        // sign-aware comparator + IP-recall regression is tracked
        // for v1.1.
        if metric == Metric::Ip {
            return Err(VectorIndexError::UnsupportedFlags { encoding, metric });
        }
        if kernel.encoding() != encoding || kernel.metric() != metric {
            return Err(VectorIndexError::UnsupportedFlags { encoding, metric });
        }
        Ok(Self {
            params,
            encoding,
            metric,
            kernel,
            ids: Vec::new(),
            vectors: Vec::new(),
            neighbors: Vec::new(),
            id_to_slot: HashMap::new(),
            entry_point: None,
            delta: DeltaSegment::new(),
            label_idx: LabelIndex::default(),
            tombstones: Vec::new(),
            bytes_per_vector: None,
            commit_lsns: Vec::new(),
            expired_lsns: Vec::new(),
            delta_lsns: HashMap::new(),
        })
    }

    /// Total live (non-tombstoned) vector count across the
    /// main graph and the delta-segment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.main_len() - self.live_tombstone_count() + self.delta.len()
    }

    /// `true` when the graph has zero live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector count in the main Vamana graph (including
    /// tombstoned slots).
    #[must_use]
    pub fn main_len(&self) -> usize {
        self.ids.len()
    }

    /// Vector count in the delta-segment (always non-tombstoned).
    #[must_use]
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Read-only access to the configured tuning parameters.
    #[must_use]
    pub fn params(&self) -> DiskAnnParams {
        self.params
    }

    /// Encoding the graph was constructed for.
    #[must_use]
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Metric the graph was constructed for.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Configured byte width of an encoded vector (set on first
    /// ingest). Returns `None` while the graph is empty.
    #[must_use]
    pub fn bytes_per_vector(&self) -> Option<usize> {
        self.bytes_per_vector
    }

    /// Internal slot for the given `VectorId`, looking through
    /// both the main graph and the delta-segment.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.id_to_slot.contains_key(&id) || self.delta.contains(id)
    }

    /// Mark `id` as tombstoned. Returns
    /// [`VectorIndexError::ArenaNotFound`] if the id is not
    /// in the main graph.
    ///
    /// Per ADR-003 Strategy 1 (preserved at v1.0): the slot is
    /// not reclaimed; subsequent searches skip it. Slice F's
    /// rebuild path compacts when `tombstone_ratio` crosses
    /// 30 %.
    pub fn delete(&mut self, id: VectorId) -> Result<()> {
        // Tombstones apply to main-graph slots. Delta-segment
        // entries are removed directly; the merge-fold path
        // never re-introduces a deleted entry.
        if let Some(&slot) = self.id_to_slot.get(&id) {
            self.set_tombstone(slot);
            return Ok(());
        }
        if self.delta.remove(id) {
            // Per ADR-041 §D-3a: cull staged LSN window so a
            // subsequent same-id insert doesn't inherit the
            // pre-delete window.
            self.delta_lsns.remove(&id);
            return Ok(());
        }
        Err(VectorIndexError::ArenaNotFound {
            tenant: arcgraph_core::TenantId::DEFAULT,
            index: crate::IndexId::ZERO,
        })
    }

    /// `true` if the slot is tombstoned.
    #[inline]
    #[must_use]
    pub(crate) fn is_tombstoned(&self, slot: u32) -> bool {
        let word = (slot >> 6) as usize;
        let bit = slot & 63;
        match self.tombstones.get(word) {
            Some(w) => (w >> bit) & 1 == 1,
            None => false,
        }
    }

    /// Set the tombstone bit for `slot` (idempotent).
    #[inline]
    pub(crate) fn set_tombstone(&mut self, slot: u32) {
        let word = (slot >> 6) as usize;
        let bit = slot & 63;
        if word >= self.tombstones.len() {
            self.tombstones.resize(word + 1, 0);
        }
        self.tombstones[word] |= 1u64 << bit;
    }

    /// Live tombstone count across the graph.
    pub(crate) fn live_tombstone_count(&self) -> usize {
        self.tombstones
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Tombstone ratio across the main graph (0.0 ≤ x ≤ 1.0).
    /// Used by the rebuild trigger (per ADR-035 §5.3 + ADR-003
    /// Strategy 3); the trigger itself is Slice F's rebuild
    /// path.
    #[must_use]
    pub fn tombstone_ratio(&self) -> f32 {
        if self.ids.is_empty() {
            return 0.0;
        }
        self.live_tombstone_count() as f32 / self.ids.len() as f32
    }

    /// Validate that `bytes` matches the configured
    /// `bytes_per_vector`, or set the latter on first ingest.
    pub(crate) fn check_or_set_byte_width(&mut self, bytes_len: usize) -> Result<()> {
        match self.bytes_per_vector {
            Some(expected) if expected == bytes_len => Ok(()),
            Some(expected) => Err(VectorIndexError::DimensionMismatch {
                expected,
                got: bytes_len,
            }),
            None => {
                self.bytes_per_vector = Some(bytes_len);
                Ok(())
            }
        }
    }

    /// Borrow the encoded payload for `slot`. Caller has
    /// already validated `slot < self.ids.len()`.
    #[inline]
    #[must_use]
    pub(crate) fn vector_bytes(&self, slot: u32) -> &[u8] {
        let width = self
            .bytes_per_vector
            .expect("bytes_per_vector set for populated graph");
        let start = slot as usize * width;
        let end = start + width;
        &self.vectors[start..end]
    }

    /// Copy one slot's encoded payload into an owned buffer.
    #[inline]
    #[must_use]
    pub(crate) fn vector_bytes_owned(&self, slot: u32) -> Vec<u8> {
        self.vector_bytes(slot).to_vec()
    }

    /// Software-prefetch the encoded payload for `slot` when the
    /// target exposes a stable prefetch intrinsic. Unsupported
    /// architectures compile this as a no-op; correctness never
    /// depends on the prefetch.
    #[inline]
    pub(crate) fn prefetch_vector_bytes(&self, slot: u32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let ptr = self.vector_bytes(slot).as_ptr().cast::<i8>();
            #[cfg(target_arch = "x86")]
            use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

            #[allow(unsafe_code)]
            // SAFETY: `_mm_prefetch` is a CPU hint. The pointer is derived from
            // a live `&[u8]` owned by `self`, is not dereferenced by Rust, and
            // may legally be ignored by the processor. The function has a
            // no-op fallback on non-x86 targets.
            unsafe {
                _mm_prefetch(ptr, _MM_HINT_T0);
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let _ = slot;
        }
    }

    /// Compute the kernel distance between two main-graph
    /// slots. Inlined hot-path helper.
    #[inline]
    pub(crate) fn slot_distance(&self, a: u32, b: u32) -> f32 {
        self.kernel
            .distance(self.vector_bytes(a), self.vector_bytes(b))
    }

    /// Compute the kernel distance between a query byte slice
    /// and a main-graph slot.
    #[inline]
    pub(crate) fn query_to_slot_distance(&self, query: &[u8], slot: u32) -> f32 {
        self.kernel.distance(query, self.vector_bytes(slot))
    }

    /// Convert a metric distance into a comparison key where
    /// **smaller is better** (ascending). For L2/Hamming the
    /// kernel value is already smaller-is-better; for IP and
    /// Cosine higher-is-better, so we negate.
    ///
    /// Per ADR-035 §3.1 + the byte-slice contract in
    /// `distance.rs`. The DiskANN search path orders candidates
    /// by ascending key and prunes by descending key; this
    /// helper hides the metric-direction sign so the rest of
    /// the algorithm is metric-agnostic.
    #[inline]
    #[must_use]
    pub(crate) fn distance_key(&self, raw: f32) -> f32 {
        match self.metric {
            // Lower-is-closer kernels — pass through.
            Metric::L2 | Metric::Hamming => raw,
            // Higher-is-closer kernels — negate so smaller
            // becomes closer in the ordering.
            // Cosine is computed as `1 - cos(θ)` by simsimd
            // (lower-is-closer in practice), but the
            // [`Metric::Cosine`] convention says "higher is
            // closer" because the kernel surface returns
            // similarity in some adapters; we negate here only
            // for the IP case. Per `distance.rs` simsimd
            // returns `1 - cos(θ)` so cosine is already
            // lower-is-closer at the byte-slice contract level.
            Metric::Cosine => raw,
            Metric::Ip => -raw,
        }
    }

    /// External-facing distance — converts the search-time
    /// ordering key back to the metric's natural value (the
    /// number that the simsimd kernel returned).
    #[inline]
    #[must_use]
    pub(crate) fn distance_external(&self, key: f32) -> f32 {
        match self.metric {
            Metric::L2 | Metric::Hamming | Metric::Cosine => key,
            Metric::Ip => -key,
        }
    }

    /// Internal helper — register a label index for already-built
    /// nodes. Used by [`DiskAnnGraph::set_label_index`] which
    /// computes per-label medoids from the supplied per-node
    /// labels.
    pub(crate) fn install_label_index(
        &mut self,
        labels_per_node: &[Option<DiskAnnLabelId>],
    ) -> Result<()> {
        if labels_per_node.len() != self.ids.len() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.ids.len(),
                got: labels_per_node.len(),
            });
        }
        let mut buckets: HashMap<DiskAnnLabelId, Vec<u32>> = HashMap::new();
        for (slot, lbl) in labels_per_node.iter().enumerate() {
            if let Some(label) = lbl {
                buckets
                    .entry(*label)
                    .or_default()
                    .push(u32::try_from(slot).expect("slot < u32::MAX"));
            }
        }
        let mut entry_per_label = HashMap::with_capacity(buckets.len());
        for (label, slots) in buckets {
            // Per-label medoid: pick the slot whose sum of
            // distances to the other slots in the bucket is
            // smallest. For very large buckets we sample down
            // to params.medoid_sample_size to keep this
            // bounded — same approximation we use for the
            // global medoid. Slice F.3 will replace this with
            // a label-aware Vamana α-pruning entry-point
            // selector, but the cache shape is stable.
            let entry = self.medoid_within(&slots);
            entry_per_label.insert(label, entry);
        }
        self.label_idx = LabelIndex {
            entry_per_label,
            labels: labels_per_node.to_vec(),
        };
        Ok(())
    }

    /// Register a per-node label vector and (re-)compute the
    /// per-label entry-point cache. `labels_per_node[slot]`
    /// gives the label for the `slot`-th node, or `None` if
    /// the node carries no label. Length must equal
    /// [`DiskAnnGraph::main_len`].
    ///
    /// Slice D scaffolding only — Slice F.3 (Filtered-DiskANN
    /// per Microsoft WWW 2023) replaces this with a label-aware
    /// Vamana α-pruning rule that propagates labels through
    /// edge selection. The cache shape (one entry-point slot
    /// per label) is stable; the algorithm that populates it
    /// changes.
    ///
    /// Returns
    /// [`VectorIndexError::DimensionMismatch`] if the labels
    /// vector length does not match the main-graph node count.
    pub fn set_label_index(&mut self, labels_per_node: &[Option<DiskAnnLabelId>]) -> Result<()> {
        self.install_label_index(labels_per_node)
    }

    /// Per-label entry-point lookup. Returns `None` if the
    /// label has no entry registered.
    #[must_use]
    pub fn entry_for_label(&self, label: DiskAnnLabelId) -> Option<VectorId> {
        let slot = *self.label_idx.entry_per_label.get(&label)?;
        Some(self.ids[slot as usize])
    }

    /// Number of distinct labels registered in the per-label
    /// index.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.label_idx.entry_per_label.len()
    }

    /// Label registered for a given external `VectorId`, or
    /// `None` if the id has no label or the label index is
    /// empty. Slice F.3 (filtered Vamana build) uses this on
    /// the query path; Slice D ships only the lookup.
    #[must_use]
    pub fn label_of(&self, id: VectorId) -> Option<DiskAnnLabelId> {
        let slot = *self.id_to_slot.get(&id)?;
        self.label_idx.labels.get(slot as usize).copied().flatten()
    }

    /// Medoid slot — the global entry point. `None` while the
    /// graph is empty.
    #[must_use]
    pub fn entry_point_id(&self) -> Option<VectorId> {
        self.entry_point.map(|s| self.ids[s as usize])
    }

    /// Approximate medoid within a slice of slots. Sample-based
    /// when the slice is larger than `medoid_sample_size`. See
    /// the Microsoft DiskANN reference impl note in §5.1 of the
    /// Subramanya paper.
    pub(crate) fn medoid_within(&self, slots: &[u32]) -> u32 {
        debug_assert!(!slots.is_empty(), "medoid of empty bucket");
        let sample_size = self.params.medoid_sample_size as usize;
        let n = slots.len();
        let stride = if n > sample_size {
            (n / sample_size).max(1)
        } else {
            1
        };
        // Deterministic stride sampling — no RNG needed; the
        // medoid choice does not need entropy at v1.0 (Slice
        // F.3 may revisit if filtered Vamana wants random
        // sample diversity).
        let sampled: Vec<u32> = slots.iter().copied().step_by(stride).collect();
        let mut best_slot = sampled[0];
        let mut best_sum = f64::INFINITY;
        for &candidate in &sampled {
            let mut sum = 0.0_f64;
            for &other in &sampled {
                if candidate == other {
                    continue;
                }
                let raw = self.slot_distance(candidate, other);
                let key = self.distance_key(raw);
                sum += key as f64;
            }
            if sum < best_sum {
                best_sum = sum;
                best_slot = candidate;
            }
        }
        best_slot
    }

    /// Allocate a slot for `(id, bytes)` and return the slot
    /// index. Updates `id_to_slot`.
    ///
    /// Per ADR-041 §D-3a, the LSN parallel arrays grow in
    /// lockstep with `ids` / `vectors` / `neighbors`. If the
    /// caller has staged an LSN window for `id` (via
    /// `delta_lsns` carry-over or [`DiskAnnGraph::stage_lsn_window`]),
    /// it is consumed here; otherwise the slot defaults to
    /// `(Lsn::ZERO, Lsn::MAX)` — always-visible.
    pub(crate) fn allocate_slot(&mut self, id: VectorId, bytes: Vec<u8>) -> u32 {
        let slot = u32::try_from(self.ids.len()).expect("DiskAnn slot exhausted (4 B vectors)");
        self.ids.push(id);
        self.vectors.extend_from_slice(&bytes);
        self.neighbors.push(Vec::new());
        self.id_to_slot.insert(id, slot);
        // Migrate any staged LSN window from delta_lsns; otherwise
        // default to always-visible (ADR-041 §D-3a).
        let (commit, expired) = self.delta_lsns.remove(&id).unwrap_or((Lsn::ZERO, Lsn::MAX));
        self.commit_lsns.push(commit);
        self.expired_lsns.push(expired);
        slot
    }

    /// Stamp the MVCC visibility window for `id` (per ADR-041
    /// §D-3a). If `id` is in the main graph, updates the slot's
    /// LSN entries; if it's in the delta-segment, updates
    /// `delta_lsns`. Returns `Ok(())` on either path; returns
    /// [`VectorIndexError::ArenaNotFound`] when `id` is unknown.
    ///
    /// Production callers stamp a concrete `(commit_lsn,
    /// expired_lsn)` pair at insert time so the visibility
    /// filter at search time enforces snapshot isolation.
    /// Tests / not-yet-wired callers leave the defaults
    /// (`Lsn::ZERO` / `Lsn::MAX`) in place.
    pub fn set_lsn_window(
        &mut self,
        id: VectorId,
        commit_lsn: Lsn,
        expired_lsn: Lsn,
    ) -> Result<()> {
        if let Some(&slot) = self.id_to_slot.get(&id) {
            self.commit_lsns[slot as usize] = commit_lsn;
            self.expired_lsns[slot as usize] = expired_lsn;
            return Ok(());
        }
        if self.delta.contains(id) {
            self.delta_lsns.insert(id, (commit_lsn, expired_lsn));
            return Ok(());
        }
        Err(VectorIndexError::ArenaNotFound {
            tenant: arcgraph_core::TenantId::DEFAULT,
            index: crate::IndexId::ZERO,
        })
    }

    /// Look up the MVCC visibility window for a slot. Returns
    /// `(commit_lsn, expired_lsn)` — the slot is visible at
    /// `read_lsn` iff `commit ≤ read_lsn ∧ read_lsn < expired`.
    ///
    /// Slots out of bounds default to `(Lsn::ZERO, Lsn::MAX)`
    /// (always-visible) — defensive fallback only; the caller
    /// has typically already validated `slot < ids.len()`.
    ///
    /// `#[allow(dead_code)]` — consumed at Commit 3 (visibility-
    /// filter wiring through `filtered_search`); the helper
    /// lands here in Commit 2 (structural fields) so the surface
    /// is reviewable as a self-contained slice.
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn slot_lsn_window(&self, slot: u32) -> (Lsn, Lsn) {
        let s = slot as usize;
        let commit = self.commit_lsns.get(s).copied().unwrap_or(Lsn::ZERO);
        let expired = self.expired_lsns.get(s).copied().unwrap_or(Lsn::MAX);
        (commit, expired)
    }

    /// Look up the MVCC visibility window for a delta-segment
    /// id. Defaults to `(Lsn::ZERO, Lsn::MAX)` when the id has
    /// no staged LSN window (always-visible).
    ///
    /// `#[allow(dead_code)]` — see [`Self::slot_lsn_window`].
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn delta_lsn_window(&self, id: VectorId) -> (Lsn, Lsn) {
        self.delta_lsns
            .get(&id)
            .copied()
            .unwrap_or((Lsn::ZERO, Lsn::MAX))
    }

    /// Whether `slot` is visible at `read_lsn` per ADR-041 §D-3a
    /// — `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn`. Mirror
    /// of BM25's `mvcc.rs::build_visibility_filter`.
    ///
    /// `#[allow(dead_code)]` — see [`Self::slot_lsn_window`].
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn slot_visible_at(&self, slot: u32, read_lsn: Lsn) -> bool {
        let (commit, expired) = self.slot_lsn_window(slot);
        let read = read_lsn.raw();
        let expired_lower = read.saturating_add(1);
        commit.raw() <= read && expired.raw() >= expired_lower
    }

    /// Whether `id` is visible at `read_lsn` in the
    /// delta-segment per ADR-041 §D-3a (mirror of
    /// [`Self::slot_visible_at`] for delta entries).
    ///
    /// `#[allow(dead_code)]` — see [`Self::slot_lsn_window`].
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn delta_visible_at(&self, id: VectorId, read_lsn: Lsn) -> bool {
        let (commit, expired) = self.delta_lsn_window(id);
        let read = read_lsn.raw();
        let expired_lower = read.saturating_add(1);
        commit.raw() <= read && expired.raw() >= expired_lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{L2F32, L2RaBitQSym, L2Sq8};

    fn empty_graph_f32() -> DiskAnnGraph {
        DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(L2F32),
        )
        .expect("default params + matching kernel must construct")
    }

    #[test]
    fn default_params_pass_validation() {
        let p = DiskAnnParams::default();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn params_reject_zero_r() {
        let p = DiskAnnParams {
            r: 0,
            ..DiskAnnParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn params_reject_alpha_below_one() {
        let p = DiskAnnParams {
            alpha: 0.5,
            ..DiskAnnParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn params_reject_l_construction_below_r() {
        let p = DiskAnnParams {
            r: 70,
            l_construction: 50,
            ..DiskAnnParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn new_rejects_kernel_metric_mismatch() {
        // Build a kernel for Cosine/F32, declare graph as L2/F32
        // — mismatched metric must raise UnsupportedFlags. (We use
        // Cosine here because Metric::Ip is rejected unconditionally
        // at construction per issue #109 defensive (a) and so cannot
        // exercise the kernel/metric pair-check.)
        let r = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(crate::distance::CosineF32),
        );
        assert!(matches!(r, Err(VectorIndexError::UnsupportedFlags { .. })));
    }

    #[test]
    fn new_rejects_invalid_encoding_metric_pair() {
        // Hamming on F32 is rejected by Metric::is_valid_for.
        let r = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::Hamming,
            Box::new(L2F32),
        );
        assert!(matches!(r, Err(VectorIndexError::UnsupportedFlags { .. })));
    }

    #[test]
    fn new_accepts_rabitq_l2_and_rejects_other_metrics_or_kernel_mismatch() {
        let ok = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::RaBitQ,
            Metric::L2,
            Box::new(L2RaBitQSym::new(16)),
        );
        assert!(ok.is_ok());

        let cosine = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::RaBitQ,
            Metric::Cosine,
            Box::new(L2RaBitQSym::new(16)),
        );
        assert!(matches!(
            cosine,
            Err(VectorIndexError::UnsupportedFlags {
                encoding: Encoding::RaBitQ,
                metric: Metric::Cosine
            })
        ));

        let mismatch = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::RaBitQ,
            Metric::L2,
            Box::new(L2Sq8),
        );
        assert!(matches!(
            mismatch,
            Err(VectorIndexError::UnsupportedFlags {
                encoding: Encoding::RaBitQ,
                metric: Metric::L2
            })
        ));
    }

    #[test]
    fn empty_graph_reports_zero() {
        let g = empty_graph_f32();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
        assert_eq!(g.main_len(), 0);
        assert_eq!(g.delta_len(), 0);
        assert_eq!(g.entry_point_id(), None);
        assert_eq!(g.bytes_per_vector(), None);
        assert_eq!(g.tombstone_ratio(), 0.0);
    }

    #[test]
    fn distance_key_passes_through_l2() {
        let g = empty_graph_f32();
        assert_eq!(g.distance_key(1.5), 1.5);
        assert_eq!(g.distance_external(1.5), 1.5);
    }

    #[test]
    fn diskann_rejects_ip_metric_on_construction() {
        // Per issue #109 defensive (a): Metric::Ip is rejected at
        // DiskAnnGraph::new because the Vamana α-prune comparator
        // (build::DiskAnnGraph::robust_prune + the streaming mirror
        // in stream::robust_prune_inner) is sign-correct for L2 /
        // Hamming and correct-by-accident for Cosine, but inverted
        // for IP. The sign-aware comparator + IP-recall regression
        // is tracked for v1.1.
        let r = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::Ip,
            Box::new(crate::distance::IpF32),
        );
        assert!(
            matches!(
                r,
                Err(VectorIndexError::UnsupportedFlags {
                    encoding: Encoding::F32,
                    metric: Metric::Ip,
                })
            ),
            "expected UnsupportedFlags{{F32, Ip}}, got {:?}",
            r.as_ref().err()
        );
    }

    #[test]
    fn tombstone_bitmap_round_trip() {
        let mut g = empty_graph_f32();
        g.set_tombstone(0);
        g.set_tombstone(63);
        g.set_tombstone(64);
        g.set_tombstone(200);
        assert!(g.is_tombstoned(0));
        assert!(g.is_tombstoned(63));
        assert!(g.is_tombstoned(64));
        assert!(g.is_tombstoned(200));
        assert!(!g.is_tombstoned(1));
        assert!(!g.is_tombstoned(199));
        // Idempotent.
        g.set_tombstone(0);
        assert_eq!(g.live_tombstone_count(), 4);
    }

    #[test]
    fn check_or_set_byte_width_locks_first_value() {
        let mut g = empty_graph_f32();
        assert_eq!(g.bytes_per_vector(), None);
        g.check_or_set_byte_width(3072).unwrap();
        assert_eq!(g.bytes_per_vector(), Some(3072));
        // Same width is OK.
        g.check_or_set_byte_width(3072).unwrap();
        // Mismatch raises.
        let err = g.check_or_set_byte_width(1536);
        assert!(matches!(
            err,
            Err(VectorIndexError::DimensionMismatch {
                expected: 3072,
                got: 1536
            })
        ));
    }

    #[test]
    fn delete_unknown_id_errors() {
        let mut g = empty_graph_f32();
        let r = g.delete(VectorId::new(42));
        assert!(matches!(r, Err(VectorIndexError::ArenaNotFound { .. })));
    }

    // ─── ADR-041 §D-3a — MVCC visibility LSN tracking ───────

    /// PIN: empty graph has empty LSN parallel arrays; the
    /// invariant is they grow in lockstep with `ids`.
    #[test]
    fn empty_graph_has_empty_lsn_arrays() {
        let g = empty_graph_f32();
        assert!(g.commit_lsns.is_empty());
        assert!(g.expired_lsns.is_empty());
        assert!(g.delta_lsns.is_empty());
    }

    /// PIN: `set_lsn_window` on an unknown id returns
    /// `ArenaNotFound`. The contract symmetrizes
    /// [`DiskAnnGraph::delete`].
    #[test]
    fn set_lsn_window_unknown_id_errors() {
        let mut g = empty_graph_f32();
        let r = g.set_lsn_window(VectorId::new(42), Lsn::new(10), Lsn::new(20));
        assert!(matches!(r, Err(VectorIndexError::ArenaNotFound { .. })));
    }

    /// PIN: `slot_visible_at` mirrors ADR-039 §D-3 BM25 boundary
    /// — `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn`.
    /// Saturating-add on the upper bound prevents wrap.
    #[test]
    fn slot_visible_at_pins_inclusive_boundary() {
        let mut g = empty_graph_f32();
        g.allocate_slot(VectorId::new(1), vec![0u8; 16]);
        // Default at allocation: (Lsn::ZERO, Lsn::MAX) →
        // always-visible.
        assert!(g.slot_visible_at(0, Lsn::ZERO));
        assert!(g.slot_visible_at(0, Lsn::new(1_000)));
        assert!(g.slot_visible_at(0, Lsn::MAX));

        // Stamp a window: visible only on read_lsn ∈ [10, 19].
        g.set_lsn_window(VectorId::new(1), Lsn::new(10), Lsn::new(20))
            .unwrap();
        assert!(!g.slot_visible_at(0, Lsn::new(9)));
        assert!(g.slot_visible_at(0, Lsn::new(10)));
        assert!(g.slot_visible_at(0, Lsn::new(19)));
        assert!(!g.slot_visible_at(0, Lsn::new(20)));
        assert!(!g.slot_visible_at(0, Lsn::new(21)));
    }

    /// PIN: `delta_visible_at` mirrors `slot_visible_at` for
    /// delta-segment entries; the staged LSN window is consumed
    /// by `allocate_slot` on merge.
    #[test]
    fn delta_visible_at_pins_inclusive_boundary() {
        let mut g = empty_graph_f32();
        let v = vec![0u8; 16];
        g.insert_stream(&[(VectorId::new(1), v.as_slice())])
            .unwrap();

        // Default at insert: (Lsn::ZERO, Lsn::MAX) →
        // always-visible (no staged window yet).
        assert!(g.delta_visible_at(VectorId::new(1), Lsn::ZERO));
        assert!(g.delta_visible_at(VectorId::new(1), Lsn::new(1_000)));

        // Stage a window via set_lsn_window — the id is in delta,
        // so set_lsn_window updates `delta_lsns`.
        g.set_lsn_window(VectorId::new(1), Lsn::new(10), Lsn::new(20))
            .unwrap();
        assert!(!g.delta_visible_at(VectorId::new(1), Lsn::new(9)));
        assert!(g.delta_visible_at(VectorId::new(1), Lsn::new(15)));
        assert!(!g.delta_visible_at(VectorId::new(1), Lsn::new(20)));
    }
}
