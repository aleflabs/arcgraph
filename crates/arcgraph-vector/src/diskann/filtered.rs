//! Filtered-DiskANN — label-aware Vamana build + filter-aware
//! beam search per Gollapudi et al. WWW 2023.
//!
//! Per ADR-035 §6.4 D-5 + AC-6 (≥ 0.85 recall@10 across all
//! selectivities) + impl-plan §3 Slice F task 3. This module
//! layers on top of Slice D's `DiskAnnGraph` (in `graph.rs` /
//! `build.rs` / `search.rs` / `stream.rs`) without reshaping
//! that scaffolding.
//!
//! ## Algorithm (Microsoft Filtered-DiskANN, WWW 2023)
//!
//! The base Vamana α-prune (Subramanya et al. NeurIPS 2019
//! Algorithm 2) is replaced with **FilteredRobustPrune**
//! (Gollapudi et al. Algorithm 5) at build time, and the beam
//! search is replaced with **FilteredGreedySearch** (Algorithm 3)
//! at query time.
//!
//! ### FilteredRobustPrune (build, paper Algorithm 5 + §5)
//!
//! Given source point `p` with label set `F_p`, candidates `V`,
//! degree budget `R`, and α-pruning factor:
//!
//! ```text
//! result := []
//! sort V by distance to p ascending
//!
//! // Pass 1 — standard filtered α-prune up to R-1 slots,
//! // reserving the last slot for the §5 connectivity guarantee.
//! for each p* in V:
//!   if not alive(p*): continue
//!   if |result| >= R - 1: break
//!   result.push(p*)
//!   if F_{p*} contains the label F_p carries:
//!     label_co_located_added := true
//!   for each v in V after p*:
//!     if not alive(v): continue
//!     // Filter-aware occlusion: only allow p* to occlude v if
//!     // p* covers F_p ∩ F_v.
//!     if F_p ∩ F_v ⊆ F_{p*} AND d(p, v) > α · d(p*, v):
//!       alive(v) := false
//!
//! // Pass 2 — reserved-slot label-co-located edge (paper §5
//! // connectivity guarantee). Without this, a sparse-label `p`
//! // can have its R out-edges entirely populated with
//! // non-label-co-located vertices, partitioning the per-label
//! // sub-graph.
//! if not label_co_located_added and F_p ≠ ∅ and |result| < R:
//!   for each c in V:
//!     if alive(c) and c ∉ result and F_c contains a label of F_p:
//!       result.push(c)
//!       break
//!
//! // Pass 3 — fill any residual slots from alive candidates.
//! while |result| < R and ∃ alive c ∈ V \ result:
//!   result.push(closest alive c)
//!
//! return result
//! ```
//!
//! For the v1.0 single-label-per-vector format (per Slice D's
//! `LabelIndex.labels: Vec<Option<DiskAnnLabelId>>`):
//!
//! - `F_p ∩ F_v` is `{l}` when `p_label == v_label == Some(l)`,
//!   else `∅`.
//! - `F_p ∩ F_v ⊆ F_{p*}` reduces to: when `p` and `v` share
//!   label `l`, `p*` must also carry `l`; when they don't share a
//!   label, the intersection is empty and the cover trivially
//!   holds.
//!
//! This preserves per-label connectivity: for any label `l`, the
//! sub-graph induced by label-`l` vertices and label-`l` edges
//! (i.e., edges between two label-`l` vertices) stays connected.
//!
//! ### FilteredGreedySearch (search, paper Algorithm 3)
//!
//! From per-label entry points, beam-search visits only
//! filter-matching nodes. Expansion follows the existing Vamana
//! adjacency list but skips neighbors whose label set fails the
//! filter check. Tombstoned slots are still walked for graph
//! traversal (matches the unfiltered `greedy_visit_from`
//! tombstone rule) but never returned to the caller.
//!
//! ## Coordination with Slice F.2 (filter-HNSW)
//!
//! Per ADR-035 amendment-03 (issue #127), F.2 (HNSW) and F.3
//! (DiskANN) share the canonical [`Filter`] enum at
//! [`crate::query`]. F.3's filtered_search dispatches on the
//! variant at the public boundary:
//!
//! - [`Filter::Any`] — unfiltered fast path (`required = None`).
//! - [`Filter::LabelEq`] — per-label entry-point cache hot path
//!   (`required = Some(label.raw())`).
//! - Every other variant — returns
//!   [`crate::VectorIndexError::UnsupportedFilter`]. The Phase
//!   6 F.4 dispatcher routes such filters to HNSW; F.5 / G.4
//!   adds a per-label inverted index that lets DiskANN handle
//!   [`Filter::LabelIn`] / `And` / `Or` directly, at which
//!   point the variants currently rejected become supported.
//!
//! Internally, F.3 still operates on [`Option<DiskAnnLabelId>`]
//! (the per-slot label format on [`DiskAnnGraph`]); the
//! conversion happens at the public `filtered_search*` entry
//! points and at the `brute_force_filtered_with_delta`
//! recovery path.
//!
//! ## Latency / memory budget
//!
//! Build cost: ~2× plain Vamana (Gollapudi et al. §6 Table 4).
//! At `n = 1 M`, `R = 70`, `dim = 128`, `f32` encoding the
//! plain-Vamana build runs ~60 s; filtered ≈ 120 s. Memory
//! overhead: a `HashMap<DiskAnnLabelId, u32>` for per-label
//! entry-points + `Vec<Option<DiskAnnLabelId>>` for per-node
//! labels. At `n = 1 M`, `card(L) = 1 K`, `4 B` per entry =
//! ~4 KB for the entry map + ~5 MB for per-node labels (8 B
//! `Option<u32>`). Negligible vs the Vamana edge budget
//! (~280 MB).
//!
//! Search cost: identical to plain DiskANN (the filter check is
//! a single `Option<u32>::eq` per neighbor expansion); the
//! per-label entry point shortens the warm-up phase for
//! high-cardinality filter dispatches.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use arcgraph_core::Lsn;

use crate::distance::DistanceKernel;
use crate::query::Filter;
use crate::{Result, VectorId, VectorIndexError};

use super::build::XorShift32;
use super::graph::{DiskAnnGraph, DiskAnnLabelId, LabelIndex};

// ─── Filter dispatch (Slice F.3 v1.0 surface) ────────────────────

/// DiskANN v1.0 capability gate — translates a canonical
/// [`Filter`] into the per-label `Option<DiskAnnLabelId>` shape
/// the rest of this module operates on.
///
/// Per ADR-035 amendment-03 the per-backend Filter is unified;
/// the variant set DiskANN supports at v1.0 is narrower than the
/// canonical enum, so this helper is the single boundary that
/// rejects unsupported variants and routes the supported ones
/// onto the existing internal `required: Option<DiskAnnLabelId>`
/// shape.
///
/// Returns:
///
/// - `Ok(None)` for [`Filter::Any`] — F.3 dispatches to its
///   unfiltered search path (no per-label restriction).
/// - `Ok(Some(label))` for [`Filter::LabelEq`] — F.3 dispatches
///   to the per-label entry-point cache hot path.
/// - [`Err(VectorIndexError::UnsupportedFilter)`] for every
///   other variant; the F.4 dispatcher routes such filters to
///   HNSW.
fn diskann_required_label(
    filter: &Filter,
) -> std::result::Result<Option<DiskAnnLabelId>, VectorIndexError> {
    match filter {
        Filter::Any => Ok(None),
        Filter::LabelEq(l) => Ok(Some(l.raw())),
        Filter::Tenant(_) => Err(VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 does not support tenant filters; F.4 dispatcher routes to HNSW per ADR-035 amendment-03"
                .to_owned(),
        }),
        Filter::LabelIn(_) => Err(VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 supports only single-label equality; multi-label dispatch awaits F.5 / G.4 per-label inverted index per ADR-035 amendment-03"
                .to_owned(),
        }),
        Filter::PropertyEq(_, _) => Err(VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 does not support property filters; F.4 dispatcher routes to HNSW per ADR-035 amendment-03"
                .to_owned(),
        }),
        Filter::And(_) => Err(VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 does not support compound `And` filters; F.4 dispatcher routes to HNSW per ADR-035 amendment-03"
                .to_owned(),
        }),
        Filter::Or(_) => Err(VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 does not support compound `Or` filters; F.4 dispatcher routes to HNSW per ADR-035 amendment-03"
                .to_owned(),
        }),
    }
}

/// Whether a slot's recorded label matches the requirement
/// derived from the canonical [`Filter`] via
/// [`diskann_required_label`]. `None` requirement (universal)
/// always matches; `Some(l)` requires the slot's label to equal
/// `l`. Slots with no recorded label NEVER match a `Some(l)`
/// requirement (consistent with arena's `labels_for(id) == None`
/// semantics: the vector carries no payload to filter on).
#[inline]
fn label_matches_required(required: Option<DiskAnnLabelId>, label: Option<DiskAnnLabelId>) -> bool {
    match required {
        None => true,
        Some(want) => label == Some(want),
    }
}

// ─── Internal heap entry types (mirror search.rs) ────────────────
//
// search.rs's `WorstFirst` is private to that module; rather
// than expose it we ship local equivalents so filtered.rs stays
// self-contained per the F.3 boundary list (no edits to
// search.rs / build.rs / stream.rs / graph.rs from this slice).

#[derive(Debug, Clone, Copy, PartialEq)]
struct BestFirst {
    slot: u32,
    /// Smaller is closer.
    key: f32,
}

impl Eq for BestFirst {}

impl PartialOrd for BestFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BestFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap — reverse so smallest-key
        // (closest) pops first. NaN-safe via `total_cmp`.
        other
            .key
            .total_cmp(&self.key)
            .then(self.slot.cmp(&other.slot))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WorstFirstFiltered {
    slot: u32,
    /// Smaller is closer.
    key: f32,
}

impl Eq for WorstFirstFiltered {}

impl PartialOrd for WorstFirstFiltered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirstFiltered {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on key — largest key at top.
        self.key
            .total_cmp(&other.key)
            .then(self.slot.cmp(&other.slot))
    }
}

// ─── Public API on DiskAnnGraph ──────────────────────────────────

impl DiskAnnGraph {
    /// Bulk-build the Vamana graph with **filter-aware
    /// α-pruning** per Gollapudi et al. WWW 2023 Algorithm 4.
    ///
    /// Replaces any existing main-graph state and resets the
    /// delta-segment, mirroring [`DiskAnnGraph::build`].
    ///
    /// `labels` is parallel to `vectors`: `labels[i]` is the
    /// optional [`DiskAnnLabelId`] for `vectors[i]`. `None`
    /// marks an "unlabeled" vector (it carries no payload to
    /// filter on; behaves universally in filter-aware
    /// traversal — its α-prune does not enforce label-cover
    /// constraints, and it is never returned by a label-eq
    /// search).
    ///
    /// `kernel` is validated against the graph's stored encoding
    /// and metric (matching the
    /// [`DiskAnnGraph::search_with_rescore`] kernel-validation
    /// pattern); the graph's own kernel is used for distance
    /// computation.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] when
    ///   `labels.len() != vectors.len()` or when a vector byte
    ///   width drifts.
    /// - [`VectorIndexError::IrrecoverableLoss`] on duplicate
    ///   `VectorId` (matches plain [`DiskAnnGraph::build`]).
    /// - [`VectorIndexError::UnsupportedFlags`] when the
    ///   supplied kernel's encoding / metric don't match the
    ///   graph's.
    pub fn build_filtered(
        &mut self,
        vectors: &[(VectorId, &[u8])],
        labels: &[Option<DiskAnnLabelId>],
        kernel: &dyn DistanceKernel,
    ) -> Result<()> {
        if kernel.encoding() != self.encoding() || kernel.metric() != self.metric() {
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: kernel.encoding(),
                metric: kernel.metric(),
            });
        }
        if vectors.len() != labels.len() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: vectors.len(),
                got: labels.len(),
            });
        }

        // Reset graph state — full wholesale build per the base
        // `DiskAnnGraph::build` contract.
        self.ids.clear();
        self.vectors.clear();
        self.neighbors.clear();
        self.id_to_slot.clear();
        self.entry_point = None;
        self.delta = super::stream::DeltaSegment::new();
        self.label_idx = LabelIndex::default();
        self.tombstones.clear();
        self.bytes_per_vector = None;
        // Per ADR-041 §D-3a: LSN parallel arrays reset alongside
        // the slot-indexed state (`ids` / `vectors` / `neighbors`).
        // `allocate_slot` re-grows them in lockstep with default
        // (Lsn::ZERO, Lsn::MAX) values so the post-build graph is
        // always-visible until `set_lsn_window` is called.
        self.commit_lsns.clear();
        self.expired_lsns.clear();
        self.delta_lsns.clear();

        if vectors.is_empty() {
            return Ok(());
        }

        // Validate + ingest into parallel arrays.
        for (id, bytes) in vectors {
            self.check_or_set_byte_width(bytes.len())?;
            if self.id_to_slot.contains_key(id) {
                return Err(VectorIndexError::IrrecoverableLoss {
                    index: crate::IndexId::ZERO,
                    reason: format!("duplicate VectorId {} in build_filtered input", id.raw()),
                });
            }
            self.allocate_slot(*id, bytes.to_vec());
        }
        let n = self.ids.len();

        // Trivial cases.
        if n == 1 {
            self.entry_point = Some(0);
            let labels_vec: Vec<Option<DiskAnnLabelId>> = labels.to_vec();
            let mut entry_per_label: HashMap<DiskAnnLabelId, u32> = HashMap::new();
            if let Some(Some(l)) = labels.first() {
                entry_per_label.insert(*l, 0);
            }
            self.label_idx = LabelIndex {
                entry_per_label,
                labels: labels_vec,
            };
            return Ok(());
        }

        // Step 1: global medoid (entry point for unlabeled queries
        // and fallback when a label has no per-label entry).
        let all_slots: Vec<u32> = (0..n as u32).collect();
        let medoid = self.medoid_within(&all_slots);
        self.entry_point = Some(medoid);

        // Step 2: bucket slots by label and compute per-label
        // medoid via the same stride-sampled approximation used
        // for the global medoid.
        let mut buckets: HashMap<DiskAnnLabelId, Vec<u32>> = HashMap::new();
        for (slot, lbl) in labels.iter().enumerate() {
            if let Some(label) = lbl {
                buckets
                    .entry(*label)
                    .or_default()
                    .push(u32::try_from(slot).expect("slot < u32::MAX"));
            }
        }
        let mut entry_per_label: HashMap<DiskAnnLabelId, u32> =
            HashMap::with_capacity(buckets.len());
        for (label, slots) in &buckets {
            // `medoid_within` requires non-empty input; buckets
            // only enter the map with at least one slot, so this
            // is safe.
            let entry = self.medoid_within(slots);
            entry_per_label.insert(*label, entry);
        }
        self.label_idx = LabelIndex {
            entry_per_label,
            labels: labels.to_vec(),
        };

        // Step 3: random-init neighbor lists (overwritten by the
        // refinement passes; same convention as base
        // `DiskAnnGraph::build`).
        let r_target = self.params.r as usize;
        let mut prng = XorShift32::seed(0xF115_8210 ^ n as u32);
        for slot in 0..n {
            let mut chosen: Vec<u32> = Vec::with_capacity(r_target.min(n - 1));
            if n <= r_target + 1 {
                for other in 0..n {
                    if other != slot {
                        chosen.push(other as u32);
                    }
                }
            } else {
                let mut attempts = 0_usize;
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

        // Step 4: two filtered refinement passes (α=1.0 then
        // α=params.alpha, matching plain Vamana's two-pass
        // schedule; FilteredVamana paper §6 Algorithm 4 is also
        // two-pass).
        self.vamana_pass_filtered(medoid, 1.0_f32, &mut prng);
        if self.params.alpha > 1.0 {
            self.vamana_pass_filtered(medoid, self.params.alpha, &mut prng);
        }

        Ok(())
    }

    /// Beam-search top-`k` against the main Vamana graph with a
    /// label filter applied during traversal.
    ///
    /// Per ADR-035 §6.2 + Gollapudi et al. WWW 2023 Algorithm 3:
    ///
    /// 1. Resolve the start slot — `Filter::label_eq(l)` consults
    ///    the per-label entry-point cache (set by
    ///    [`DiskAnnGraph::build_filtered`] or
    ///    [`DiskAnnGraph::set_label_index`]) and falls back to the
    ///    global medoid if the label has no entry.
    /// 2. Run beam search; expansion skips neighbors whose label
    ///    fails the filter check. Tombstoned slots are still
    ///    walked for graph traversal but never returned.
    /// 3. Return up to `k` filter-matching candidates ordered by
    ///    ascending closeness in the metric's natural orientation.
    ///
    /// `kernel` is validated against the graph's stored
    /// encoding + metric for symmetry with
    /// [`DiskAnnGraph::search_with_rescore`]; the graph's own
    /// kernel performs the distance computation.
    ///
    /// ## Edge cases
    ///
    /// - `k == 0` — returns `Ok(Vec::new())`.
    /// - `Filter::any()` — behaves identically to
    ///   [`DiskAnnGraph::search`] (no label restriction; entry
    ///   from the global medoid).
    /// - `Filter::label_eq(l)` with no vector carrying `l` —
    ///   returns `Ok(Vec::new())`. The per-label entry-point cache
    ///   is consulted up front; an absent entry is the
    ///   "zero matching vectors" signal.
    /// - Empty graph — returns `Ok(Vec::new())`.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] on `query` byte
    ///   width mismatch.
    /// - [`VectorIndexError::UnsupportedFlags`] on kernel
    ///   mismatch.
    ///
    /// `read_lsn` is the MVCC visibility key per ADR-041 §D-3a.
    /// Slots whose `(commit_lsn, expired_lsn)` window does not
    /// cover `read_lsn` are excluded from results AND from the
    /// beam-search frontier (the snapshot-invisible neighbor is
    /// still walked for connectivity per the same discipline as
    /// `is_tombstoned` — see `Self::greedy_visit_filtered_multi_start`).
    /// Callers without snapshot context pass `Lsn::MAX` (most-
    /// permissive read; everything visible).
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes signature past clippy default; documented widening
    pub fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        l_search: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>> {
        if kernel.encoding() != self.encoding() || kernel.metric() != self.metric() {
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: kernel.encoding(),
                metric: kernel.metric(),
            });
        }
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

        // Canonical Filter → DiskANN-internal `Option<DiskAnnLabelId>`.
        // Unsupported variants short-circuit out per ADR-035
        // amendment-03 — F.4 routes them to HNSW.
        let required = diskann_required_label(filter)?;
        // For label-eq with no vectors carrying the label, return
        // empty without walking the graph. The per-label entry
        // cache is the authoritative "label has at least one
        // vector" signal at v1.0.
        if let Some(l) = required
            && !self.label_idx.entry_per_label.contains_key(&l)
        {
            return Ok(Vec::new());
        }

        let entry = match required {
            Some(l) => self
                .label_idx
                .entry_per_label
                .get(&l)
                .copied()
                .or(self.entry_point),
            None => self.entry_point,
        };
        let Some(entry) = entry else {
            return Ok(Vec::new());
        };

        let l = l_search.max(k).max(1);
        let visited = self.beam_search_filtered(query, entry, l, required, read_lsn);

        let mut out = Vec::with_capacity(k.min(visited.len()));
        for (slot, key) in visited.into_iter().take(k) {
            out.push((self.ids[slot as usize], self.distance_external(key)));
        }
        Ok(out)
    }

    /// Filtered beam search across the main graph **and** the
    /// streaming delta-segment, merged by distance.
    ///
    /// Mirrors [`DiskAnnGraph::search_with_delta`] but applies
    /// the filter to both stores. The delta-segment carries no
    /// internal label index (the v1.0 `DeltaSegment` from Slice D
    /// stores `(VectorId, Vec<u8>)` only); the caller passes
    /// `delta_label_lookup` — a closure resolving each
    /// delta-segment `VectorId` to its label, populated from the
    /// arena's [`crate::VectorArena::labels_for`] in production
    /// (Slice F.1) or from a test-side sidecar in unit tests.
    /// Returning `None` from the closure means "this delta entry
    /// has no recorded label" — equivalent to a `None` slot in
    /// `LabelIndex.labels` for the main graph.
    ///
    /// I-V7 (T1 read-your-writes) is preserved for filtered
    /// queries: the delta-segment is consulted synchronously,
    /// before the function returns, so an immediately-following
    /// `filtered_search_with_delta` for a just-inserted vector
    /// finds it even before the next merge fires.
    ///
    /// # Errors
    ///
    /// Same as [`DiskAnnGraph::filtered_search`] (kernel
    /// validation + dimension mismatch).
    // The signature carries the full filter-search surface
    // (query, k, filter, l_search, kernel, delta_label_lookup)
    // — each is load-bearing per the Slice F.3 + ADR-035 §6.2
    // contract. Trimming any of them would push the contract
    // back onto the caller; the rescore-pattern precedent in
    // search.rs::search_with_rescore documents the same
    // multi-arg trade-off.
    #[allow(clippy::too_many_arguments)]
    pub fn filtered_search_with_delta<L>(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        l_search: usize,
        kernel: &dyn DistanceKernel,
        delta_label_lookup: L,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>>
    where
        L: Fn(VectorId) -> Option<DiskAnnLabelId>,
    {
        if k == 0 {
            return Ok(Vec::new());
        }
        // `filtered_search` already validates kernel + dim AND
        // performs the canonical → DiskANN-internal Filter
        // dispatch (rejecting unsupported variants with
        // `UnsupportedFilter`); we call it first so the
        // downstream delta scan only runs once we know the
        // filter is supported.
        let main_hits = self.filtered_search(query, k, filter, l_search, kernel, read_lsn)?;
        // Re-derive the per-label requirement for the delta
        // scan — `filtered_search` already validated it, so the
        // re-translation here is infallible at this point.
        let required = diskann_required_label(filter)?;
        let delta_hits =
            self.filtered_search_delta(query, k, required, &delta_label_lookup, read_lsn);

        // Merge by ascending distance key (smaller-is-closer per
        // metric direction); de-dup by `VectorId` defensively
        // (insert-stream rejects duplicates against either store
        // but a future overwrite-on-delete path could surface
        // both).
        let mut merged: Vec<(VectorId, f32, f32)> =
            Vec::with_capacity(main_hits.len() + delta_hits.len());
        for (id, raw) in main_hits.iter().chain(delta_hits.iter()) {
            let key = self.distance_key(*raw);
            merged.push((*id, *raw, key));
        }
        merged.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.raw().cmp(&b.0.raw())));
        merged.dedup_by(|a, b| a.0 == b.0);
        merged.truncate(k);
        Ok(merged.into_iter().map(|(id, raw, _)| (id, raw)).collect())
    }

    /// Selectivity-aware filtered-search dispatch per
    /// ADR-035 §6.2 + impl-plan §3 Slice F task 4.
    ///
    /// Routes the query to one of three paths based on the
    /// caller-supplied `selectivity` (fraction of arena matching
    /// the filter, in `[0.0, 1.0]`):
    ///
    /// - **`selectivity < 0.01` (low):** brute-force scan across
    ///   the filter-matching vectors in main graph + delta-segment.
    ///   At very-low selectivity, beam search wastes work on
    ///   non-matching neighbors; brute-force over the small
    ///   matching subset is faster.
    /// - **`selectivity > 0.5` (high):** filter-bound beam
    ///   search at the default `l_search`. The matching subgraph
    ///   is dense enough that the unmodified beam search achieves
    ///   the AC-6 recall floor.
    /// - **mid-range (0.01 ≤ s ≤ 0.5):** filter-bound beam search
    ///   with adaptive `l_search` enlarged by `1 + ln(1/s)`
    ///   capped at `4 × default_l_search`, compensating for
    ///   skipped non-matching neighbors per the §6.2 dispatch
    ///   pseudocode.
    ///
    /// The selectivity estimator that produces `selectivity` is
    /// the planner-side heuristic at v1.0 (`Property-equality
    /// → 0.01`, `Range → 0.10`, `Tenant → 1.0` per impl-plan
    /// §F.4); the secondary B-tree histogram wiring is the
    /// post-M3.a follow-up.
    ///
    /// `delta_label_lookup` is consulted on both the
    /// brute-force and beam-search paths (the latter via
    /// [`DiskAnnGraph::filtered_search_with_delta`]).
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::IrrecoverableLoss`] when
    ///   `selectivity` is outside `[0.0, 1.0]` or NaN — the
    ///   planner is buggy.
    /// - Plus the kernel / dim errors propagated from the
    ///   downstream search.
    // The signature mirrors `filtered_search_with_delta` plus
    // the planner-supplied `selectivity` — same load-bearing
    // surface argument as the wide-arg comment above.
    #[allow(clippy::too_many_arguments)]
    pub fn filtered_search_dispatch<L>(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        selectivity: f64,
        kernel: &dyn DistanceKernel,
        delta_label_lookup: L,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>>
    where
        L: Fn(VectorId) -> Option<DiskAnnLabelId> + Copy,
    {
        if !selectivity.is_finite() || !(0.0..=1.0).contains(&selectivity) {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: crate::IndexId::ZERO,
                reason: format!(
                    "filtered_search_dispatch: selectivity {selectivity} not in [0.0, 1.0]"
                ),
            });
        }

        let base_l = self.params.l_search_default as usize;

        if selectivity < 0.01 {
            // Low-selectivity: brute-force across filter-matching
            // vectors in main + delta. Cheaper than walking a
            // beam mostly through non-matching neighbors.
            self.brute_force_filtered_with_delta(
                query,
                k,
                filter,
                kernel,
                delta_label_lookup,
                read_lsn,
            )
        } else if selectivity > 0.5 {
            // High-selectivity: filter-bound beam at default
            // `l_search`.
            self.filtered_search_with_delta(
                query,
                k,
                filter,
                base_l,
                kernel,
                delta_label_lookup,
                read_lsn,
            )
        } else {
            // Mid-range: enlarge `l_search` to compensate for
            // skipped neighbors. Cap at 4× to bound worst-case
            // latency.
            let factor = 1.0_f64 + (1.0 / selectivity).ln();
            let factor = factor.max(1.0);
            let enlarged = ((base_l as f64) * factor).ceil() as usize;
            let l_capped = enlarged.min(base_l.saturating_mul(4)).max(base_l);
            self.filtered_search_with_delta(
                query,
                k,
                filter,
                l_capped,
                kernel,
                delta_label_lookup,
                read_lsn,
            )
        }
    }

    /// Brute-force filtered scan against main + delta-segment.
    ///
    /// Public entry to the low-selectivity dispatch leg of
    /// [`DiskAnnGraph::filtered_search_dispatch`]; also exposed
    /// for fallback use by the operator-warned recovery path
    /// (`arcgraph.vector.filter_heuristic_miss` per impl-plan
    /// §F.4).
    ///
    /// # Errors
    ///
    /// Same as [`DiskAnnGraph::filtered_search`].
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes signature past clippy default; documented widening
    pub fn brute_force_filtered_with_delta<L>(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        kernel: &dyn DistanceKernel,
        delta_label_lookup: L,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>>
    where
        L: Fn(VectorId) -> Option<DiskAnnLabelId>,
    {
        if kernel.encoding() != self.encoding() || kernel.metric() != self.metric() {
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: kernel.encoding(),
                metric: kernel.metric(),
            });
        }
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

        // Canonical Filter → DiskANN-internal label requirement
        // (ADR-035 amendment-03). Unsupported variants short-
        // circuit out before any scan work.
        let required = diskann_required_label(filter)?;

        let mut hits: Vec<(VectorId, f32, f32)> = Vec::new();

        // Main-graph scan. Skip tombstones + filter mismatches +
        // MVCC-invisible slots (per ADR-041 §D-3a).
        let n = self.ids.len();
        for slot in 0..n as u32 {
            if self.is_tombstoned(slot) {
                continue;
            }
            if !self.slot_visible_at(slot, read_lsn) {
                continue;
            }
            let label = self.label_idx.labels.get(slot as usize).copied().flatten();
            if !label_matches_required(required, label) {
                continue;
            }
            let raw = self.query_to_slot_distance(query, slot);
            let key = self.distance_key(raw);
            hits.push((self.ids[slot as usize], self.distance_external(key), key));
        }

        // Delta-segment scan via caller-provided label lookup.
        for (id, bytes) in self.delta.iter() {
            if !self.delta_visible_at(id, read_lsn) {
                continue;
            }
            let label = delta_label_lookup(id);
            if !label_matches_required(required, label) {
                continue;
            }
            let raw = self.kernel.distance(query, bytes);
            let key = self.distance_key(raw);
            hits.push((id, self.distance_external(key), key));
        }

        hits.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.raw().cmp(&b.0.raw())));
        hits.dedup_by(|a, b| a.0 == b.0);
        hits.truncate(k);
        Ok(hits.into_iter().map(|(id, raw, _)| (id, raw)).collect())
    }
}

// ─── Internal helpers (graph-level filtered build / search) ──────

impl DiskAnnGraph {
    /// One filtered Vamana refinement pass per Gollapudi et al.
    /// Algorithm 4. Mirrors `build::DiskAnnGraph::vamana_pass`
    /// but uses
    /// [`DiskAnnGraph::greedy_visit_filtered_multi_start`] and
    /// [`DiskAnnGraph::robust_prune_filtered`] in place of the
    /// unfiltered helpers.
    fn vamana_pass_filtered(&mut self, medoid: u32, alpha: f32, prng: &mut XorShift32) {
        let n = self.ids.len();
        let r_target = self.params.r as usize;
        let l_construction = self.params.l_construction as usize;
        let bytes_per_vector = self
            .bytes_per_vector
            .expect("bytes_per_vector set before refinement");

        // Random permutation of slots (Fisher-Yates).
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = (prng.next_u32() as usize) % (i + 1);
            perm.swap(i, j);
        }

        let mut query_buf: Vec<u8> = Vec::with_capacity(bytes_per_vector);

        for &p in &perm {
            // Pull query bytes; we cannot borrow `self` across
            // the mutable graph traversal in the greedy visit
            // + symmetrize loop.
            query_buf.clear();
            query_buf.extend_from_slice(self.vector_bytes(p));
            let p_label = self.label_idx.labels.get(p as usize).copied().flatten();

            // Filtered greedy search from {global medoid} ∪
            // {start_per_label[l] for l in F_p}. For single-label
            // F_p = {p_label}, this is at most 2 starts.
            let mut starts: Vec<u32> = Vec::with_capacity(2);
            starts.push(medoid);
            if let Some(l) = p_label
                && let Some(&entry) = self.label_idx.entry_per_label.get(&l)
                && entry != medoid
            {
                starts.push(entry);
            }

            // Build-time greedy visit uses Lsn::MAX so every
            // already-built slot is in scope (per ADR-041
            // §D-3a; build-time has no snapshot — the LSN
            // window is consulted only at search time).
            let visit = self.greedy_visit_filtered_multi_start(
                &query_buf,
                &starts,
                p_label,
                l_construction,
                Lsn::MAX,
            );

            let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(visit.len() + r_target);
            for (slot, key) in visit {
                if slot == p {
                    continue;
                }
                candidates.push((slot, key));
            }
            // Include p's existing neighbors (with re-computed
            // keys) so the prune fold-in doesn't lose recall on
            // a previously-good edge that was dominated mid-walk.
            for &existing in &self.neighbors[p as usize] {
                if existing == p || candidates.iter().any(|(s, _)| *s == existing) {
                    continue;
                }
                let raw = self.slot_distance(p, existing);
                let key = self.distance_key(raw);
                candidates.push((existing, key));
            }

            // Filter-aware α-prune.
            let pruned = self.robust_prune_filtered(p, p_label, candidates, alpha, r_target);
            self.neighbors[p as usize] = pruned.clone();

            // Symmetrize back-edges. Same shape as the unfiltered
            // pass (`build.rs::vamana_pass`) but the re-prune
            // step uses `robust_prune_filtered` so q's edge set
            // stays filter-cover-compliant. Per the PR #126
            // review (Gollapudi 2023 §5 connectivity guarantee):
            // when the re-prune drops the new back-edge `p`, we
            // force-include it (displacing the worst residual
            // neighbor) so the per-label sub-graph keeps the
            // bidirectional structure that `filtered_search`
            // relies on for reachability.
            for q in pruned {
                if q == p {
                    continue;
                }
                let q_neigh_len = self.neighbors[q as usize].len();
                let q_already = self.neighbors[q as usize].contains(&p);
                if q_already {
                    continue;
                }
                if q_neigh_len < r_target {
                    self.neighbors[q as usize].push(p);
                    continue;
                }
                let q_label = self.label_idx.labels.get(q as usize).copied().flatten();
                let q_neighbors_snap = self.neighbors[q as usize].clone();
                let mut q_cands: Vec<(u32, f32)> = Vec::with_capacity(q_neighbors_snap.len() + 1);
                for nb in q_neighbors_snap {
                    let raw = self.slot_distance(q, nb);
                    q_cands.push((nb, self.distance_key(raw)));
                }
                let raw = self.slot_distance(q, p);
                q_cands.push((p, self.distance_key(raw)));
                let mut q_pruned = self.robust_prune_filtered(q, q_label, q_cands, alpha, r_target);
                // Force-include the new back-edge `p`. The
                // re-prune may α-occlude `p` against q's
                // existing neighbors, but the symmetrize
                // contract (a → b ⇒ b ∈ N_out(b)+{a} after
                // re-prune) is what gives Vamana its
                // bidirectional reachability structure.
                // Dropping the back-edge can isolate `p` in
                // the directed sub-graph that `filtered_search`
                // walks — see PR #126 review notes / Gollapudi
                // 2023 §5.
                if !q_pruned.contains(&p) {
                    if q_pruned.len() >= r_target {
                        q_pruned.pop();
                    }
                    q_pruned.push(p);
                }
                self.neighbors[q as usize] = q_pruned;
            }
        }
    }

    /// Filtered α-prune per Gollapudi et al. WWW 2023
    /// Algorithm 5 + paper §5 connectivity guarantee. Mirrors
    /// the unfiltered `build::DiskAnnGraph::robust_prune` plus
    /// the cover check `F_p ∩ F_v ⊆ F_{p*}` before the
    /// geometric occlusion test, AND reserves the last neighbor
    /// slot for at least one label-co-located edge.
    ///
    /// ## Connectivity guarantee (paper §5)
    ///
    /// > "Preserve label-co-located edges until R is exceeded."
    ///
    /// Without the reservation, a sparse-label vertex `p` whose
    /// closest geometric neighbors all carry a different label
    /// can have its R out-edges entirely populated with
    /// non-label-co-located vertices — the cover check protects
    /// label-co-located candidates from occlusion, but doesn't
    /// guarantee they reach the R-prefix. With the reservation,
    /// Pass 1 fills up to R - 1 slots via standard filtered
    /// α-prune; Pass 2 reserves the final slot for the closest
    /// remaining label-co-located candidate (provably alive at
    /// this point — see proof below); Pass 3 fills any residual
    /// slots from the unprocessed candidate tail.
    ///
    /// **Proof that Pass 2 always finds an alive label-co-located
    /// candidate when one exists:** if `label_co_located_added ==
    /// false` after Pass 1, no `p*` in `result` carries `p`'s
    /// label. The cover check therefore fails for any attempted
    /// occlusion of a label-co-located `v` (intersection
    /// `{p_label}` is not a subset of `F_{p*}`). All
    /// label-co-located candidates are still alive when Pass 2
    /// runs; Pass 2 picks the closest one in candidate order.
    ///
    /// PR #126 review (2026-04-26) surfaced the bug at proptest
    /// seed=76603; the regression case is captured in
    /// `tests/diskann_filtered.proptest-regressions`. Without
    /// this fix, `f3_filtered_alpha_prune_preserves_label_connectivity`
    /// deterministically failed (label-2 vertex 75 unreachable
    /// from label-2 vertex 1 even at l_search=248).
    fn robust_prune_filtered(
        &self,
        p: u32,
        p_label: Option<DiskAnnLabelId>,
        mut candidates: Vec<(u32, f32)>,
        alpha: f32,
        r: usize,
    ) -> Vec<u32> {
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        candidates.retain(|(s, _)| *s != p);
        candidates.dedup_by(|a, b| a.0 == b.0);

        let mut result: Vec<u32> = Vec::with_capacity(r);
        let mut alive: Vec<bool> = vec![true; candidates.len()];
        let mut label_co_located_added = false;

        // Pass 1: standard filtered α-prune capped at `r - 1`
        // slots so Pass 2 can reserve the last one for a
        // label-co-located edge (paper §5 connectivity
        // guarantee). When `r <= 1` the cap is 0 and Pass 1
        // is a no-op; Pass 2 / Pass 3 handle the degenerate
        // `r == 1` case naturally.
        let pass1_cap = r.saturating_sub(1);
        for i in 0..candidates.len() {
            if !alive[i] {
                continue;
            }
            if result.len() >= pass1_cap {
                break;
            }
            let (p_star_slot, _) = candidates[i];
            result.push(p_star_slot);
            let p_star_label = self
                .label_idx
                .labels
                .get(p_star_slot as usize)
                .copied()
                .flatten();
            if p_label.is_some() && p_star_label == p_label {
                label_co_located_added = true;
            }

            for j in (i + 1)..candidates.len() {
                if !alive[j] {
                    continue;
                }
                let (v_slot, v_key) = candidates[j];
                let v_label = self
                    .label_idx
                    .labels
                    .get(v_slot as usize)
                    .copied()
                    .flatten();

                // Filter-cover check (the F_p ∩ F_v ⊆ F_{p*}
                // clause from Algorithm 5). For the single-label
                // v1.0 format, the intersection is {l} when
                // p_label == v_label == Some(l) and ∅ otherwise.
                // ∅ ⊆ anything; for {l} we need p* to also carry
                // `l`.
                let intersection_label = match (p_label, v_label) {
                    (Some(lp), Some(lv)) if lp == lv => Some(lp),
                    _ => None,
                };
                let cover_holds = match intersection_label {
                    None => true,
                    Some(l) => p_star_label == Some(l),
                };
                if !cover_holds {
                    continue;
                }

                // Geometric occlusion (same shape as base Vamana
                // α-prune): only prune v if v is α-dominated by
                // p*.
                let d_p_v = self.distance_external(v_key);
                let d_pstar_v = self.slot_distance(p_star_slot, v_slot);
                if d_p_v > alpha * d_pstar_v {
                    alive[j] = false;
                }
            }
        }

        // Pass 2: reserve the final slot for a label-co-located
        // edge per paper §5 if Pass 1 didn't add one. We scan
        // candidates in ascending-distance order so the closest
        // alive label-co-located vertex wins. The proof above
        // guarantees every label-co-located candidate is alive
        // when this fires; we still gate on `alive[i]`
        // defensively in case future cover-rule extensions
        // alter the invariant.
        if !label_co_located_added && p_label.is_some() && result.len() < r {
            for i in 0..candidates.len() {
                if !alive[i] {
                    continue;
                }
                let (cand_slot, _) = candidates[i];
                if result.contains(&cand_slot) {
                    continue;
                }
                let cand_label = self
                    .label_idx
                    .labels
                    .get(cand_slot as usize)
                    .copied()
                    .flatten();
                if cand_label == p_label {
                    result.push(cand_slot);
                    break;
                }
            }
        }

        // Pass 3: fill any remaining slots from alive candidates
        // not yet in `result`. Mirrors the prompt's "fill
        // remaining slots from un-pruned candidates" leg —
        // typically a no-op when Pass 1 saturated at r - 1 and
        // Pass 2 added the reserved label-co-located edge. Fires
        // when (a) `r <= 1`, (b) no label-co-located candidate
        // existed for Pass 2 to add, or (c) Pass 1 stopped
        // early because all remaining candidates were occluded.
        if result.len() < r {
            for i in 0..candidates.len() {
                if result.len() >= r {
                    break;
                }
                if !alive[i] {
                    continue;
                }
                let (cand_slot, _) = candidates[i];
                if !result.contains(&cand_slot) {
                    result.push(cand_slot);
                }
            }
        }
        result
    }

    /// Multi-start filtered greedy beam search returning the
    /// visited set (`(slot, key)` pairs in ascending-key order)
    /// among label-matching, non-tombstoned, MVCC-visible slots.
    ///
    /// `required` is the filter target — `None` accepts every
    /// slot (degenerates to plain beam search), `Some(l)`
    /// restricts both the frontier and the start-point seeding
    /// to label-`l` slots.
    ///
    /// `read_lsn` is the MVCC visibility key per ADR-041 §D-3a;
    /// slots whose `(commit_lsn, expired_lsn)` window does not
    /// cover `read_lsn` are skipped from the result frontier
    /// but still walked for graph connectivity (mirrors the
    /// tombstone discipline above; matches HNSW's policy in
    /// [`super::super::hnsw::filtered::FilteredHnsw::filtered_search_layer0`]).
    fn greedy_visit_filtered_multi_start(
        &self,
        query: &[u8],
        starts: &[u32],
        required: Option<DiskAnnLabelId>,
        l: usize,
        read_lsn: Lsn,
    ) -> Vec<(u32, f32)> {
        debug_assert!(l > 0, "beam width must be > 0");
        debug_assert!(!starts.is_empty(), "at least one start point");

        let mut frontier: BinaryHeap<WorstFirstFiltered> = BinaryHeap::with_capacity(l + 1);
        let mut to_visit: BinaryHeap<BestFirst> = BinaryHeap::with_capacity(l + 1);
        let mut visited: HashSet<u32> = HashSet::new();
        let mut in_frontier: HashSet<u32> = HashSet::new();

        // Seed `to_visit` with each start; seed `frontier` only
        // with label-matching, MVCC-visible starts so the
        // visited set never contains a non-matching /
        // snapshot-invisible slot at termination.
        for &s in starts {
            if (s as usize) >= self.ids.len() {
                continue;
            }
            if to_visit.iter().any(|c| c.slot == s) {
                continue;
            }
            let key = self.distance_key(self.query_to_slot_distance(query, s));
            to_visit.push(BestFirst { slot: s, key });
            if !self.is_tombstoned(s)
                && self.slot_visible_at(s, read_lsn)
                && self.label_matches_slot(s, required)
            {
                frontier.push(WorstFirstFiltered { slot: s, key });
                in_frontier.insert(s);
            }
        }

        while let Some(curr) = to_visit.pop() {
            if !visited.insert(curr.slot) {
                continue;
            }
            let neigh_slice = &self.neighbors[curr.slot as usize];
            for &n in neigh_slice {
                if visited.contains(&n) || in_frontier.contains(&n) {
                    continue;
                }
                let raw = self.query_to_slot_distance(query, n);
                let key = self.distance_key(raw);
                // Always push onto best-first walk — this
                // preserves connectivity for label-matching /
                // MVCC-visible descendants reachable via non-
                // matching / snapshot-invisible intermediate
                // nodes.
                to_visit.push(BestFirst { slot: n, key });
                if self.is_tombstoned(n)
                    || !self.slot_visible_at(n, read_lsn)
                    || !self.label_matches_slot(n, required)
                {
                    continue;
                }
                if frontier.len() < l {
                    frontier.push(WorstFirstFiltered { slot: n, key });
                    in_frontier.insert(n);
                } else if let Some(worst) = frontier.peek()
                    && key < worst.key
                {
                    let popped = frontier.pop().expect("peek non-empty");
                    in_frontier.remove(&popped.slot);
                    frontier.push(WorstFirstFiltered { slot: n, key });
                    in_frontier.insert(n);
                }
            }
            // Termination: if every remaining unvisited
            // candidate is strictly worse than the worst
            // frontier entry, we cannot improve.
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
        visited_vec
    }

    /// Single-start filtered beam search — the search-time entry
    /// point used by [`DiskAnnGraph::filtered_search`].
    fn beam_search_filtered(
        &self,
        query: &[u8],
        entry: u32,
        l: usize,
        required: Option<DiskAnnLabelId>,
        read_lsn: Lsn,
    ) -> Vec<(u32, f32)> {
        self.greedy_visit_filtered_multi_start(query, &[entry], required, l, read_lsn)
    }

    /// Whether the slot's recorded label matches the filter
    /// requirement. `None` requirement (universal) always
    /// matches; `Some(l)` requires the slot's label to equal
    /// `l`. Slots with no recorded label NEVER match a
    /// `Some(l)` requirement (consistent with arena's
    /// `labels_for(id) == None` semantics: the vector carries
    /// no payload to filter on).
    #[inline]
    fn label_matches_slot(&self, slot: u32, required: Option<DiskAnnLabelId>) -> bool {
        match required {
            None => true,
            Some(want) => self.label_idx.labels.get(slot as usize).copied().flatten() == Some(want),
        }
    }

    /// Brute-force filtered scan of the delta-segment only.
    ///
    /// Internal — the public surface is
    /// [`DiskAnnGraph::filtered_search_with_delta`] (main +
    /// delta merged) or
    /// [`DiskAnnGraph::brute_force_filtered_with_delta`] (both
    /// stores brute-forced). Takes the already-translated
    /// `required: Option<DiskAnnLabelId>` rather than the
    /// canonical [`Filter`]; the public callers run
    /// [`diskann_required_label`] once at the boundary.
    fn filtered_search_delta<L>(
        &self,
        query: &[u8],
        k: usize,
        required: Option<DiskAnnLabelId>,
        delta_label_lookup: &L,
        read_lsn: Lsn,
    ) -> Vec<(VectorId, f32)>
    where
        L: Fn(VectorId) -> Option<DiskAnnLabelId>,
    {
        if self.delta.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<(VectorId, f32)> = Vec::new();
        for (id, bytes) in self.delta.iter() {
            // ADR-041 §D-3a: skip MVCC-invisible delta entries
            // before the (cheaper) label predicate runs.
            if !self.delta_visible_at(id, read_lsn) {
                continue;
            }
            let label = delta_label_lookup(id);
            if !label_matches_required(required, label) {
                continue;
            }
            let raw = self.kernel.distance(query, bytes);
            hits.push((id, raw));
        }
        hits.sort_by(|a, b| {
            self.distance_key(a.1)
                .total_cmp(&self.distance_key(b.1))
                .then(a.0.raw().cmp(&b.0.raw()))
        });
        hits.truncate(k);
        hits
    }
}

// ─── F.4 dispatcher trait impl ───────────────────────────────────
//
// Per ADR-035 amendment-04 D-7. Pure delegation: the trait body
// forwards verbatim to the existing public `filtered_search`
// method (which already returns
// `VectorIndexError::UnsupportedFilter` for variants outside
// `Any` + `LabelEq` per amendment-03). Adding this impl does NOT
// change the F.3 search body — the dispatcher logic lives in
// `crate::dispatcher` and consumes `&dyn FilteredVectorIndex`
// references.

impl crate::dispatcher::FilteredVectorIndex for DiskAnnGraph {
    #[inline]
    fn kind(&self) -> crate::dispatcher::BackendKind {
        crate::dispatcher::BackendKind::DiskAnn
    }

    #[inline]
    fn len(&self) -> usize {
        DiskAnnGraph::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        DiskAnnGraph::is_empty(self)
    }

    #[inline]
    fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>> {
        DiskAnnGraph::filtered_search(self, query, k, filter, ef, kernel, read_lsn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::DiskAnnParams;
    use crate::distance::L2F32;
    use crate::{Encoding, Metric};

    fn fxd(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    fn empty_graph_f32_with(params: DiskAnnParams) -> DiskAnnGraph {
        DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32))
            .expect("default params + matching kernel must construct")
    }

    // ─── canonical Filter → DiskANN dispatch ────────────────
    //
    // The structural Filter API (constructors, `is_any`,
    // `required_label`) is unit-tested in `crate::query`. The
    // tests below exercise the DiskANN-local boundary helpers
    // introduced by ADR-035 amendment-03.

    #[test]
    fn diskann_required_label_any_yields_none() {
        let r = diskann_required_label(&Filter::any()).expect("Any is supported");
        assert_eq!(r, None);
    }

    #[test]
    fn diskann_required_label_label_eq_yields_inner_u32() {
        let r = diskann_required_label(&Filter::label_eq(42_u32)).expect("LabelEq is supported");
        assert_eq!(r, Some(42_u32));
    }

    #[test]
    fn diskann_required_label_rejects_tenant_filter() {
        let f = Filter::Tenant(arcgraph_core::TenantId::new(7));
        let err = diskann_required_label(&f).unwrap_err();
        assert!(matches!(err, VectorIndexError::UnsupportedFilter { .. }));
    }

    #[test]
    fn diskann_required_label_rejects_label_in() {
        let f = Filter::LabelIn(vec![arcgraph_core::LabelId::new(1)]);
        let err = diskann_required_label(&f).unwrap_err();
        assert!(matches!(err, VectorIndexError::UnsupportedFilter { .. }));
    }

    #[test]
    fn diskann_required_label_rejects_property_eq() {
        let f = Filter::PropertyEq(
            arcgraph_core::StringId::new(0),
            crate::query::PropertyValue::U32(0),
        );
        let err = diskann_required_label(&f).unwrap_err();
        assert!(matches!(err, VectorIndexError::UnsupportedFilter { .. }));
    }

    #[test]
    fn diskann_required_label_rejects_compound_and_or() {
        let and = Filter::And(vec![Filter::any()]);
        assert!(matches!(
            diskann_required_label(&and).unwrap_err(),
            VectorIndexError::UnsupportedFilter { .. }
        ));
        let or = Filter::Or(vec![Filter::any()]);
        assert!(matches!(
            diskann_required_label(&or).unwrap_err(),
            VectorIndexError::UnsupportedFilter { .. }
        ));
    }

    #[test]
    fn label_matches_required_universal_accepts_all() {
        assert!(label_matches_required(None, None));
        assert!(label_matches_required(None, Some(7)));
    }

    #[test]
    fn label_matches_required_specific_requires_match() {
        assert!(label_matches_required(Some(7), Some(7)));
        assert!(!label_matches_required(Some(7), Some(8)));
        assert!(!label_matches_required(Some(7), None));
    }

    #[test]
    fn build_filtered_empty_input_is_ok() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        g.build_filtered(&[], &[], &L2F32).unwrap();
        assert!(g.is_empty());
        assert_eq!(g.entry_point_id(), None);
        assert_eq!(g.label_count(), 0);
    }

    #[test]
    fn build_filtered_single_vector_with_label() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[1.0, 2.0, 3.0, 4.0]);
        g.build_filtered(&[(VectorId::new(7), v.as_slice())], &[Some(11)], &L2F32)
            .unwrap();
        assert_eq!(g.main_len(), 1);
        assert_eq!(g.entry_point_id(), Some(VectorId::new(7)));
        assert_eq!(g.entry_for_label(11), Some(VectorId::new(7)));
        assert_eq!(g.label_count(), 1);
        assert_eq!(g.label_of(VectorId::new(7)), Some(11));
    }

    #[test]
    fn build_filtered_rejects_label_length_mismatch() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let err = g
            .build_filtered(
                &[
                    (VectorId::new(1), v.as_slice()),
                    (VectorId::new(2), v.as_slice()),
                ],
                &[Some(0)],
                &L2F32,
            )
            .unwrap_err();
        assert!(matches!(err, VectorIndexError::DimensionMismatch { .. }));
    }

    #[test]
    fn build_filtered_rejects_kernel_mismatch() {
        let mut g = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::F32,
            Metric::L2,
            Box::new(L2F32),
        )
        .unwrap();
        let v = fxd(&[1.0, 0.0, 0.0, 0.0]);
        // Construct a kernel for a different metric.
        let err = g
            .build_filtered(
                &[(VectorId::new(0), v.as_slice())],
                &[None],
                &crate::distance::IpF32,
            )
            .unwrap_err();
        assert!(matches!(err, VectorIndexError::UnsupportedFlags { .. }));
    }

    #[test]
    fn filtered_search_label_with_no_vectors_returns_empty() {
        let mut g = empty_graph_f32_with(DiskAnnParams {
            r: 4,
            alpha: 1.2,
            l_construction: 16,
            l_search_default: 16,
            ..DiskAnnParams::default()
        });
        let owned: Vec<(VectorId, Vec<u8>)> = (0..8_u32)
            .map(|i| (VectorId::new(i), fxd(&[i as f32, 0.0, 0.0, 0.0])))
            .collect();
        let pairs: Vec<(VectorId, &[u8])> =
            owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        // All vectors carry label 0; filtered search on label 99
        // hits no entry-point and returns empty.
        let labels: Vec<Option<DiskAnnLabelId>> = vec![Some(0); 8];
        g.build_filtered(&pairs, &labels, &L2F32).unwrap();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g
            .filtered_search(&q, 5, &Filter::label_eq(99), 32, &L2F32, Lsn::MAX)
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn filtered_search_dispatch_rejects_invalid_selectivity() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[0.0, 0.0, 0.0, 0.0]);
        g.build_filtered(&[(VectorId::new(0), v.as_slice())], &[None], &L2F32)
            .unwrap();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g.filtered_search_dispatch(&q, 5, &Filter::any(), -0.1, &L2F32, |_| None, Lsn::MAX);
        assert!(matches!(r, Err(VectorIndexError::IrrecoverableLoss { .. })));
        let r = g.filtered_search_dispatch(&q, 5, &Filter::any(), 1.5, &L2F32, |_| None, Lsn::MAX);
        assert!(matches!(r, Err(VectorIndexError::IrrecoverableLoss { .. })));
        let r =
            g.filtered_search_dispatch(&q, 5, &Filter::any(), f64::NAN, &L2F32, |_| None, Lsn::MAX);
        assert!(matches!(r, Err(VectorIndexError::IrrecoverableLoss { .. })));
    }

    #[test]
    fn filtered_search_zero_k_returns_empty() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[0.0, 0.0, 0.0, 0.0]);
        g.build_filtered(&[(VectorId::new(0), v.as_slice())], &[None], &L2F32)
            .unwrap();
        let q = fxd(&[1.0, 0.0, 0.0, 0.0]);
        let r = g
            .filtered_search(&q, 0, &Filter::any(), 16, &L2F32, Lsn::MAX)
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn filtered_search_dim_mismatch_errors() {
        let mut g = empty_graph_f32_with(DiskAnnParams::default());
        let v = fxd(&[0.0, 0.0, 0.0, 0.0]);
        g.build_filtered(&[(VectorId::new(0), v.as_slice())], &[None], &L2F32)
            .unwrap();
        let q_short = fxd(&[1.0, 0.0]);
        let r = g.filtered_search(&q_short, 1, &Filter::any(), 16, &L2F32, Lsn::MAX);
        assert!(matches!(r, Err(VectorIndexError::DimensionMismatch { .. })));
    }
}
