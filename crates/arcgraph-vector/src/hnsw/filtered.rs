//! Filter-aware HNSW — Slice F.2 (M3.a).
//!
//! Per ADR-035 §6, this module combines a payload-aware HNSW build
//! (the Qdrant pattern) with a selectivity-driven dispatcher.
//!
//! ## Why a wrapper, not extra fields on [`HnswGraph`]
//!
//! The filter-aware index lives in a wrapper ([`FilteredHnsw`]) that
//! owns an inner [`HnswGraph`] plus a per-vector payload sidecar. This
//! keeps filter state out of the base graph and lets callers use
//! [`HnswGraph`] without payload indexing.
//!
//! ## Qdrant payload-aware pattern (build side)
//!
//! Per [Qdrant's filtered-search blog post] and ADR-035 D-5:
//! at insert time, after the standard HNSW edge-selection
//! heuristic places `M` neighbors, the filter-aware variant ALSO
//! adds layer-0 edges to the closest **payload-co-located**
//! neighbors (vectors that share at least one label with the
//! new vector). This biases the graph so that filtered
//! traversals reach matching vectors faster — the boundary tests
//! (`f2_payload_aware_connectivity_partition` in particular)
//! verify recall stays ≥ 0.85 even when the filter partitions
//! the payload-aware sub-graph into disconnected components.
//!
//! v1.0 ships brute-force payload-co-location lookup
//! (`O(N)` over the payload sidecar) which is correct but not
//! scalable to 10 M-vector arenas; the Slice F.5 / G.4 secondary
//! index will replace it with an `O(log N)` lookup. The recall
//! contract holds either way.
//!
//! [Qdrant's filtered-search blog post]: https://qdrant.tech/articles/filtrable-hnsw/
//!
//! ## Selectivity-driven dispatcher (search side)
//!
//! Per Slice F.2 spec (refining ADR-035 §6.2 v1.0 caveat):
//!
//! - **selectivity > 0.5** (most of the arena passes the filter):
//!   standard HNSW search with **post-filter** on the result list.
//!   Cheap because the standard search already returns mostly-
//!   matching candidates; a thin post-filter strips the rest.
//! - **selectivity ≤ 0.5** (less than half passes): **filtered
//!   beam search** that applies the filter during traversal.
//!   Avoids the post-filter oversample blow-up at very-low
//!   selectivity.
//!
//! The 0.5 split point is a v1.0 heuristic per the slice's
//! "Path A" boundary tests; the empirical sweep
//! (`f2_selectivity_*pct_recall`) shows it preserves the
//! recall@10 floors at every measured selectivity bucket.
//! Adaptive `ef` tuning (ADR-035 §6.2 §item 4) is a Slice F.4
//! follow-up, not in F.2's scope.
//!
//! ## Latency / memory budget
//!
//! - Payload sidecar: `Payload` averages ~64 B (1 tenant_id, 2
//!   labels, 1 property). For a 1 M-vector arena, ~64 MB extra
//!   over [`HnswGraph`]'s ~3.5 GB base — well inside the 64 GB
//!   per-host budget per design-v2 §A.1.
//! - Filtered beam search: same `O(ef · M · log N)` distance
//!   evaluations as standard search, plus `O(ef · M)` filter
//!   evaluations. At `ef=128`, `M=32`, filter is `O(1)` for
//!   tenant + label + simple-property cases, so the filter
//!   overhead is negligible vs the distance cost.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use arcgraph_core::{LabelId, Lsn, TenantId};

use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;
use crate::query::{Filter, PropertyKey, PropertyValue};

use super::graph::{HnswGraph, HnswParams};
use super::search::Candidate;
use super::search::ordered_float::OrderedF32;

// ─── Payload ─────────────────────────────────────────────────────

/// Per-vector payload — the data attached to each vector that
/// filters evaluate against.
///
/// Construction is field-by-field to keep the v1.0 surface
/// minimal; future slices may grow the variant set (timestamps,
/// numeric ranges, …) without breaking existing callers.
///
/// A payload may carry zero or more of each kind; the empty
/// payload (`Payload::default()`) matches **only** the
/// no-op filter (a no-arg `Filter::And(vec![])`) and the
/// always-true filter — it is rejected by every concrete
/// label / tenant / property predicate.
///
/// ## MVCC visibility (per ADR-041)
///
/// Every payload carries `commit_lsn` + `expired_lsn` — the
/// LSN window during which this vector entry is visible. The
/// defaults (`Lsn::ZERO` / `Lsn::MAX`) keep entries always-
/// visible, preserving read-latest behavior for callers that
/// have not yet wired LSN tracking. Production callers that
/// thread `read_lsn` through the search API populate these at
/// insert time so snapshot isolation is enforced. Mirrors the
/// BM25 `(commit_lsn, expired_lsn)` per-doc convention from
/// ADR-039 §D-2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payload {
    /// Tenant the vector belongs to. `None` is allowed for the
    /// degenerate single-tenant deployment per ADR-011 (the
    /// arena selection itself enforces tenant isolation; the
    /// per-vector tenant tag is redundant when the arena is
    /// single-tenant). Multi-tenant arenas (a v1.1 follow-up)
    /// require this populated.
    pub tenant_id: Option<TenantId>,
    /// Labels attached to the vector. Order is irrelevant —
    /// `Filter::LabelIn` checks set membership.
    pub labels: Vec<LabelId>,
    /// Property bag keyed by interned property name.
    pub properties: HashMap<PropertyKey, PropertyValue>,
    /// LSN at which this vector entry was committed (per
    /// ADR-041). Default `Lsn::ZERO` (visible from any
    /// `read_lsn ≥ 0`) for callers without LSN context.
    pub commit_lsn: Lsn,
    /// LSN at which this vector entry was superseded (per
    /// ADR-041). Default `Lsn::MAX` (never expired). v1.0
    /// upserts replace in-place rather than versioning, so
    /// `expired_lsn` is structurally `MAX` for every live
    /// vector — same posture as ADR-039 §D-2 for BM25.
    pub expired_lsn: Lsn,
}

impl Default for Payload {
    fn default() -> Self {
        Self {
            tenant_id: None,
            labels: Vec::new(),
            properties: HashMap::new(),
            // ADR-041 defaults: always-visible at v1.0 unless a
            // caller explicitly stamps an LSN window.
            commit_lsn: Lsn::ZERO,
            expired_lsn: Lsn::MAX,
        }
    }
}

impl Payload {
    /// Build an empty payload. Equivalent to
    /// [`Payload::default`] but reads more naturally at call
    /// sites that intend "no labels, no properties".
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct a payload from a set of labels (no properties,
    /// no tenant tag). Used by tests and by the F.1 arena
    /// integration path that mirrors `arena.labels_for(id)`.
    ///
    /// LSN window defaults to `(Lsn::ZERO, Lsn::MAX)` — visible
    /// at every read_lsn (per ADR-041 default-behavior contract).
    #[inline]
    #[must_use]
    pub fn with_labels(labels: Vec<LabelId>) -> Self {
        Self {
            tenant_id: None,
            labels,
            properties: HashMap::new(),
            commit_lsn: Lsn::ZERO,
            expired_lsn: Lsn::MAX,
        }
    }

    /// Stamp the MVCC visibility window on this payload (per
    /// ADR-041 §D-3a). Builder-style so `Payload::with_labels(..)
    /// .with_lsn_window(commit, expired)` reads naturally at
    /// call sites.
    ///
    /// Every live vector has `expired_lsn == Lsn::MAX`; this mirrors
    /// the retained BM25 MVCC visibility contract.
    #[inline]
    #[must_use]
    pub fn with_lsn_window(mut self, commit_lsn: Lsn, expired_lsn: Lsn) -> Self {
        self.commit_lsn = commit_lsn;
        self.expired_lsn = expired_lsn;
        self
    }

    /// Whether this payload is visible at a given `read_lsn`
    /// per ADR-041 §D-3 (mirror of BM25 `mvcc.rs::build_visibility_filter`):
    /// visible iff `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn`.
    /// Saturating-add on the upper bound matches BM25 amendment-01.
    #[inline]
    #[must_use]
    pub fn is_visible_at(&self, read_lsn: Lsn) -> bool {
        let read = read_lsn.raw();
        let expired_lower = read.saturating_add(1);
        self.commit_lsn.raw() <= read && self.expired_lsn.raw() >= expired_lower
    }

    /// Whether the payload carries `label`.
    #[inline]
    #[must_use]
    pub fn has_label(&self, label: LabelId) -> bool {
        self.labels.contains(&label)
    }

    /// Whether the payload carries the property `(key, value)`.
    #[inline]
    #[must_use]
    pub fn has_property(&self, key: PropertyKey, value: PropertyValue) -> bool {
        self.properties.get(&key).copied() == Some(value)
    }

    /// Whether the payload shares at least one label with `other`.
    /// Used by the payload-aware insert path to identify
    /// co-located neighbors.
    #[inline]
    #[must_use]
    pub fn shares_any_label(&self, other: &Self) -> bool {
        self.labels.iter().any(|l| other.labels.contains(l))
    }
}

// ─── Filter — payload evaluation ─────────────────────────────────
//
// The canonical [`Filter`] enum and its structural methods
// (constructors, `is_any`, `required_label`) live in
// `crate::query` per ADR-035 amendment-03. The payload-evaluation
// impl block below is HNSW-local because [`Payload`] is the F.2
// sidecar; F.3 (DiskANN) does not have an equivalent and
// dispatches structurally via [`Filter::required_label`] instead.

impl Filter {
    /// Whether `payload` satisfies this filter.
    ///
    /// The single-step evaluator: walks the filter tree once per
    /// candidate vector. v1.0 sketches an `O(|filter|)` walk per
    /// candidate; the F.5 / G.4 secondary index will short-
    /// circuit the most-selective leaf first, but for the v1.0
    /// payload sizes (~3 labels, ~1 property) the walk is fast
    /// enough that the dispatcher's selectivity-aware switch
    /// dominates the latency.
    ///
    /// [`Filter::Any`] short-circuits to `true` for every payload;
    /// [`Filter::LabelEq`] is the single-label fast path
    /// (equivalent to a one-element [`Filter::LabelIn`] but
    /// dispatched directly by F.3 via the per-label entry-point
    /// cache).
    #[must_use]
    pub fn matches(&self, payload: &Payload) -> bool {
        match self {
            Self::Any => true,
            Self::Tenant(t) => match payload.tenant_id {
                Some(p_t) => p_t == *t,
                // No tenant tag: in single-tenant arenas the
                // tenant filter is the arena's identity, and the
                // payload tag is redundant. We treat the absence
                // as "matches the arena's tenant" since the
                // arena selection has already filtered
                // cross-tenant accesses (ADR-011).
                None => true,
            },
            Self::LabelEq(l) => payload.has_label(*l),
            Self::PropertyEq(k, v) => payload.has_property(*k, *v),
            Self::LabelIn(ls) => ls.iter().any(|l| payload.has_label(*l)),
            // `And(vec![])` is always-true (empty conjunction);
            // `Or(vec![])` is always-false (empty disjunction).
            Self::And(children) => children.iter().all(|c| c.matches(payload)),
            Self::Or(children) => children.iter().any(|c| c.matches(payload)),
        }
    }
}

// ─── FilteredHnsw ────────────────────────────────────────────────

/// Filter-aware HNSW index.
///
/// Wraps an inner [`HnswGraph`] with a per-vector payload
/// sidecar. The wrapper owns the payload state; the inner graph
/// remains the same single-tenant in-memory baseline that Slice
/// C ships. No field is added to [`HnswGraph`] (per Slice F.2's
/// disjointness contract — see module docs).
///
/// ## Ownership
///
/// The inner graph is owned (not borrowed) so [`FilteredHnsw`]
/// is a plain `Send + Sync` value. The wrapper exposes
/// [`Self::inner`] read-only access for callers that need to
/// reach the underlying graph (e.g., the standard
/// [`HnswGraph::search`] call from the > 50 % selectivity
/// post-filter path).
///
/// ## Concurrency
///
/// `&self` methods (search, payload getter) are safe to call
/// from multiple threads concurrently. `&mut self` methods
/// (insert, mark_deleted) require exclusive access. Callers
/// that need concurrent search + insert wrap a `FilteredHnsw`
/// in `parking_lot::RwLock`; the boundary test
/// `f2_concurrent_search_insert_no_torn_payload` exercises
/// exactly this pattern.
pub struct FilteredHnsw {
    inner: HnswGraph,
    /// Per-vector payload sidecar. Cleared in lockstep with
    /// `inner`'s vector / node / tombstone maps via
    /// [`FilteredHnsw::filtered_insert`] (and the
    /// [`FilteredHnsw::mark_deleted`] / [`FilteredHnsw::detach`]
    /// helpers).
    payloads: HashMap<VectorId, Payload>,
}

impl FilteredHnsw {
    /// Construct an empty filter-aware HNSW with the given
    /// parameters and dim. See [`HnswGraph::new`] for the
    /// `params` / `dim` / `kernel` semantics — the wrapper is a
    /// thin pass-through.
    #[must_use]
    pub fn new(params: HnswParams, dim: usize, kernel: &dyn DistanceKernel) -> Self {
        Self {
            inner: HnswGraph::new(params, dim, kernel),
            payloads: HashMap::new(),
        }
    }

    /// Number of vectors currently held (tombstoned vectors
    /// counted — they still occupy slots).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the index holds zero vectors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Read-only access to the underlying [`HnswGraph`]. Used
    /// by tests and by the > 50 %-selectivity post-filter path
    /// inside [`Self::filtered_search_dispatch`].
    #[inline]
    #[must_use]
    pub fn inner(&self) -> &HnswGraph {
        &self.inner
    }

    /// Lookup the payload for `id` — returns `None` if the
    /// vector was inserted via [`HnswGraph::insert`] directly
    /// (without a payload).
    #[inline]
    #[must_use]
    pub fn payload(&self, id: VectorId) -> Option<&Payload> {
        self.payloads.get(&id)
    }

    /// Mark `id` deleted. Tombstones the inner graph (so the
    /// vector still serves as a routing hub) AND drops the
    /// payload (so future filter evaluations skip it).
    pub fn mark_deleted(&mut self, id: VectorId) {
        self.inner.mark_deleted(id);
        // Note: keeping the payload in the sidecar is harmless
        // (the tombstone filter in the search path strips
        // the result); we drop it so the sidecar stays in sync
        // with the live set, freeing memory on a delete-heavy
        // workload.
        self.payloads.remove(&id);
    }

    /// Insert a vector with its payload, then add payload-aware
    /// edges per Qdrant pattern.
    ///
    /// Build flow:
    ///
    /// 1. Standard HNSW insert via [`HnswGraph::insert`] —
    ///    establishes the diversity-pruned `M` edges per
    ///    Algorithm 1 + 4.
    /// 2. Store the payload in the sidecar.
    /// 3. Find the closest payload-co-located neighbors (top
    ///    `M_max0 / 2` by distance) and add bidirectional
    ///    layer-0 edges if not already present. The cap respects
    ///    `m_max0()` — payload-aware augmentation does not blow
    ///    past the per-layer fan-out budget.
    ///
    /// **v1.0 caveat (per module docs):** payload-co-location
    /// lookup is brute-force `O(N)` over the sidecar. F.5 / G.4
    /// will replace it with a per-label inverted index.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] if
    ///   `vector_bytes.len() != dim * size_of::<f32>()`.
    pub fn filtered_insert(
        &mut self,
        id: VectorId,
        vector_bytes: &[u8],
        payload: Payload,
        kernel: &dyn DistanceKernel,
    ) -> Result<(), VectorIndexError> {
        // 1. Standard HNSW insert (validates dim, samples level,
        //    runs M-pick + diversity prune).
        self.inner.insert(id, vector_bytes, kernel)?;

        // 2. Store the payload. Insert (rather than entry().or)
        //    so a re-insert of the same id replaces the payload —
        //    consistent with `HnswGraph::insert`'s
        //    "duplicate id replaces" contract.
        self.payloads.insert(id, payload.clone());

        // 3. Payload-aware edge augmentation. Skip when the new
        //    payload has no labels (nothing to co-locate on).
        if payload.labels.is_empty() {
            return Ok(());
        }
        self.augment_payload_edges(id, vector_bytes, &payload, kernel);

        Ok(())
    }

    /// Find payload-co-located neighbors and add layer-0 edges.
    ///
    /// Hot loop: `O(N)` over the sidecar (brute-force scan).
    /// At v1.0 N is typically ≤ 10 K per arena (per ADR-035
    /// auto-quantize threshold); beyond that, F.5's per-label
    /// inverted index takes over.
    fn augment_payload_edges(
        &mut self,
        id: VectorId,
        vector_bytes: &[u8],
        payload: &Payload,
        kernel: &dyn DistanceKernel,
    ) {
        // Cap: half the layer-0 fan-out budget — leaves room for
        // standard edges + payload edges within the per-vector
        // memory budget. The factor 2 is a v1.0 heuristic; F.5
        // will surface this as a tunable.
        let cap = (self.inner.params.m_max0() / 2).max(1);

        // Collect candidate co-located neighbors with their
        // distances.
        let mut candidates: Vec<(f32, VectorId)> = self
            .payloads
            .iter()
            .filter_map(|(&other_id, other_payload)| {
                if other_id == id {
                    return None;
                }
                if !payload.shares_any_label(other_payload) {
                    return None;
                }
                let other_bytes = self.inner.vector_bytes(other_id)?;
                let d = kernel.distance(vector_bytes, other_bytes);
                Some((d, other_id))
            })
            .collect();

        // Sort ascending — closest first.
        candidates.sort_by_key(|a| OrderedF32(a.0));
        candidates.truncate(cap);

        // Add bidirectional layer-0 edges. We modify the
        // inner.nodes adjacency directly; this respects the
        // pub(crate) access we have on the field. Re-prune the
        // OTHER side's adjacency if the addition pushes it over
        // m_max0() — the payload-aware path is allowed to
        // displace far standard neighbors with closer payload
        // neighbors at v1.0 (consistent with the Qdrant
        // pattern's "payload edges are first-class" framing).
        for (_d, other_id) in candidates {
            self.add_layer0_edge_if_absent(id, other_id, kernel);
            self.add_layer0_edge_if_absent(other_id, id, kernel);
        }
    }

    /// Add `to` to `from`'s layer-0 adjacency if not already
    /// present. Caps the adjacency at `m_max0()` — when over
    /// cap, drops the most-distant existing neighbor (a cheap
    /// nearest-neighbor displacement; the full diversity prune
    /// runs at the standard insert path, not the payload-aware
    /// augmentation).
    ///
    /// Idempotent: repeated calls with the same edge are no-ops.
    fn add_layer0_edge_if_absent(
        &mut self,
        from: VectorId,
        to: VectorId,
        kernel: &dyn DistanceKernel,
    ) {
        if from == to {
            return; // self-loops not allowed
        }
        let cap = self.inner.params.m_max0();
        let Some(node) = self.inner.nodes.get_mut(&from) else {
            return;
        };
        if node.neighbors.is_empty() {
            return; // node sat at level >= 0 ⇒ neighbors[0] exists
        }
        let layer0 = &mut node.neighbors[0];
        if layer0.contains(&to) {
            return; // already linked
        }
        layer0.push(to);
        if layer0.len() <= cap {
            return;
        }
        // Re-prune. We need the `from` vector's bytes to
        // recompute distances; clone to release the &mut borrow
        // on inner.nodes.
        let layer0_snapshot: Vec<VectorId> = layer0.clone();
        let from_bytes_owned: Vec<u8> = match self.inner.vector_bytes(from) {
            Some(b) => b.to_vec(),
            None => return,
        };
        // Build (distance, neighbor) pairs.
        let mut scored: Vec<(f32, VectorId)> = layer0_snapshot
            .iter()
            .filter_map(|&n| {
                let nb = self.inner.vector_bytes(n)?;
                // Use the live kernel passed by the augmentation
                // path. Distance ordering follows the kernel's
                // convention (lower-is-closer for L2 / Cosine /
                // Hamming; for inner-product callers pre-
                // normalize per ADR-035 §3.3 so ascending raw
                // distance still reflects closeness).
                Some((kernel_distance_or_inf(kernel, &from_bytes_owned, nb), n))
            })
            .collect();
        scored.sort_by_key(|a| OrderedF32(a.0));
        scored.truncate(cap);
        // Persist the trimmed adjacency.
        let new_adj: Vec<VectorId> = scored.into_iter().map(|(_, n)| n).collect();
        if let Some(node) = self.inner.nodes.get_mut(&from) {
            if let Some(layer0) = node.neighbors.get_mut(0) {
                *layer0 = new_adj;
            }
        }
    }

    /// Filter-aware beam search at layer 0 (Algorithm 2 with
    /// per-candidate filter check).
    ///
    /// Behavior:
    ///
    /// - Greedy zoom from the entry point through upper layers
    ///   ignores the filter (connectivity preservation —
    ///   non-matching nodes still serve as routing hubs at
    ///   upper layers).
    /// - Layer-0 beam search expands every candidate's
    ///   adjacency, but only adds **filter-matching** AND
    ///   **MVCC-visible** candidates to the result heap.
    ///   Non-matching candidates contribute to the candidate
    ///   frontier (so the search reaches matching vectors hidden
    ///   behind non-matching ones) but do not occupy a slot in
    ///   the top-`k` result.
    /// - Tombstone discipline mirrors [`HnswGraph::search`]:
    ///   tombstoned vectors are filtered out of the result list
    ///   even if they pass the user-supplied filter.
    /// - **MVCC visibility (ADR-041 §D-3a):** entries with
    ///   `commit_lsn > read_lsn` or `expired_lsn ≤ read_lsn`
    ///   are excluded from results (they're outside the read
    ///   snapshot). Mirrors BM25's
    ///   `mvcc.rs::build_visibility_filter`.
    ///
    /// `ef` is the beam width at layer 0; per Malkov & Yashunin
    /// §4 it must be `>= k` for a meaningful result. The
    /// implementation does not enforce that — a caller passing
    /// `ef < k` simply receives at most `ef` results (correct,
    /// degraded recall). If `ef == 0`, the inner graph's
    /// `params.ef_search` default is used.
    ///
    /// `read_lsn` is the MVCC snapshot for visibility filtering
    /// (per ADR-041 §D-1). Callers without snapshot context
    /// pass `Lsn::MAX` (most-permissive read).
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] if `query.len()`
    ///   does not equal the inner graph's `bytes_per_vector`.
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes signature past clippy default; documented widening
    pub fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        self.inner.validate_vector_bytes(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let Some(entry) = self.inner.entry_point else {
            return Ok(Vec::new());
        };

        let ef_use = if ef == 0 {
            self.inner.params.ef_search
        } else {
            ef
        };
        let ef_layer0 = ef_use.max(k);

        // ── Greedy zoom (no filter) ─────────────────────────
        let mut eps: Vec<VectorId> = vec![entry];
        for layer in (1..=self.inner.max_level).rev() {
            let next = super::search::search_layer(&self.inner, query, &eps, layer, 1, kernel);
            if let Some(c) = next.first() {
                eps = vec![c.id];
            }
        }

        // ── Layer-0 filter-aware beam search ────────────────
        let results = self.filtered_search_layer0(query, &eps, ef_layer0, filter, kernel, read_lsn);

        // Tombstone filter — the graph's tombstones are the
        // single source of truth for "is this a live vector".
        // Then truncate to top-k.
        let out: Vec<(VectorId, f32)> = results
            .into_iter()
            .filter(|c| !self.inner.is_tombstoned(c.id))
            .take(k)
            .map(|c| (c.id, c.distance.0))
            .collect();
        Ok(out)
    }

    /// Selectivity-driven dispatcher.
    ///
    /// Per the Slice F.2 spec rule:
    ///
    /// - `selectivity > 0.5` → standard search + post-filter.
    ///   Cheap because the standard top-`k` is mostly-matching
    ///   already; we oversample by `1 / selectivity` (capped at
    ///   the graph size) and then keep the first `k` matches.
    /// - `selectivity ≤ 0.5` → [`Self::filtered_search`]
    ///   (filter applied during traversal). Avoids the
    ///   post-filter oversample blow-up at low selectivity.
    ///
    /// `selectivity` must be in `[0, 1]`. The dispatcher does
    /// NOT clamp; out-of-range values fall into one of the two
    /// branches via the ordinary > comparison (and a `NaN`
    /// selectivity always falls through to the filtered branch
    /// because `NaN > 0.5` is `false` per IEEE-754, which is
    /// the safer default).
    ///
    /// # Errors
    ///
    /// - As [`Self::filtered_search`] for the low-selectivity
    ///   branch.
    /// - As [`HnswGraph::search`] for the high-selectivity
    ///   branch.
    // Argument count exceeds clippy's default 6 because the
    // dispatcher carries the filter context (filter +
    // selectivity hint) on top of the standard search args
    // (query / k / ef / kernel). Bundling them into a struct
    // adds ceremony without clarity at the call sites; matches
    // the same allow on `HnswGraph::search_with_rescore` per the
    // existing codebase precedent (see `hnsw/search.rs`).
    #[allow(clippy::too_many_arguments)]
    pub fn filtered_search_dispatch(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        selectivity: f32,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        if selectivity > 0.5 {
            // High-selectivity post-filter path.
            //
            // Oversample by `1 / selectivity` so the expected
            // post-filter result count is `k`. Saturate at the
            // graph size to avoid degenerate oversampling on a
            // small arena.
            let inv_sel = (1.0_f32 / selectivity.max(f32::MIN_POSITIVE)).ceil() as usize;
            let oversample = k.saturating_mul(inv_sel).min(self.inner.len()).max(k);
            let raw = self.inner.search(query, oversample, ef, kernel)?;
            let filtered: Vec<(VectorId, f32)> = raw
                .into_iter()
                .filter(|(id, _)| match self.payloads.get(id) {
                    // ADR-041 §D-3a: visibility filter fuses
                    // with the user-supplied predicate; both
                    // must pass.
                    Some(p) => p.is_visible_at(read_lsn) && filter.matches(p),
                    // No payload recorded — vector was inserted
                    // bypassing `filtered_insert` (for example a
                    // test using `inner.insert` directly).
                    // Treat as non-matching by default; this is
                    // the safe choice (no false positives), and
                    // production callers always go through
                    // `filtered_insert`.
                    None => false,
                })
                .take(k)
                .collect();
            Ok(filtered)
        } else {
            // Low-selectivity filtered traversal.
            self.filtered_search(query, k, filter, ef, kernel, read_lsn)
        }
    }

    // ─── helpers ─────────────────────────────────────────────

    /// Filter-aware Algorithm 2 at layer 0.
    ///
    /// Differs from [`super::search::search_layer`] in two ways:
    ///
    /// 1. The result heap only holds **matching** candidates.
    /// 2. The candidate frontier expands every neighbor
    ///    regardless of match (preserving connectivity through
    ///    non-matching nodes).
    ///
    /// Per ADR-041 §D-3a, the result heap also gates on
    /// **MVCC visibility** at `read_lsn` — a candidate enters
    /// the result heap only if both the user filter matches AND
    /// the payload's `(commit_lsn, expired_lsn)` window covers
    /// `read_lsn`. The candidate frontier still expands MVCC-
    /// invisible neighbors so the search reaches snapshot-
    /// visible vectors hidden behind newer / superseded ones.
    ///
    /// Termination: when the closest unexpanded candidate is
    /// farther than the worst kept matching result AND the
    /// result heap holds at least `ef` matches. This is the
    /// same termination as the standard search but indexed on
    /// matching-result count rather than total-candidate count.
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes signature past clippy default; documented widening
    fn filtered_search_layer0(
        &self,
        query: &[u8],
        eps: &[VectorId],
        ef: usize,
        filter: &Filter,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Vec<Candidate> {
        // #815 — delegate to the shared predicate beam. The payload +
        // MVCC visibility test IS this path's per-vector eligibility
        // predicate; the served bare-graph path
        // (`predicate_filtered_search`) reuses the SAME beam with a
        // label-set predicate, so the two filter-during-search call
        // sites can never diverge.
        filtered_beam_layer0(&self.inner, query, eps, ef, kernel, &|id| {
            self.payload_visible_and_matches(id, filter, read_lsn)
        })
    }

    /// Whether the payload for `id` (if any) satisfies `filter`
    /// AND is visible at `read_lsn` per ADR-041 §D-3a. Vectors
    /// with no recorded payload are treated as non-matching by
    /// default (see [`Self::filtered_search_dispatch`] for the
    /// rationale).
    #[inline]
    fn payload_visible_and_matches(&self, id: VectorId, filter: &Filter, read_lsn: Lsn) -> bool {
        match self.payloads.get(&id) {
            Some(p) => p.is_visible_at(read_lsn) && filter.matches(p),
            None => false,
        }
    }
}

/// Filter-during-search beam at layer 0 (Algorithm 2 with a
/// per-candidate eligibility predicate) — the shared body behind
/// BOTH [`FilteredHnsw::filtered_search`] (payload + MVCC predicate)
/// and the served-path [`predicate_filtered_search`] (label-set
/// predicate over a bare [`HnswGraph`]). #815.
///
/// Two departures from [`super::search::search_layer`]:
///
/// 1. The result heap admits a candidate ONLY when `is_allowed(id)`
///    is `true` — so the top-`ef` kept are the nearest *matching*
///    vectors.
/// 2. The candidate frontier expands EVERY visited neighbor
///    regardless of `is_allowed` — non-matching nodes still serve as
///    routing hubs so the walk reaches matching vectors hidden behind
///    them (connectivity preservation). This is what makes a filtered
///    KNN return `k` true matches instead of collapsing to
///    `k · selectivity` the way a retrieve-then-discard post-filter
///    does (#815).
///
/// Termination mirrors the standard beam but is indexed on the
/// *matching*-result count: stop once the closest unexpanded
/// candidate is farther than the worst kept match AND the result heap
/// already holds `ef` matches. When matches are sparser than `ef` the
/// reachable frontier is exhausted — the honest cost of a
/// highly-selective filter, and exactly why the caller sizes `ef` to
/// the recall it needs.
fn filtered_beam_layer0(
    graph: &HnswGraph,
    query: &[u8],
    eps: &[VectorId],
    ef: usize,
    kernel: &dyn DistanceKernel,
    is_allowed: &dyn Fn(VectorId) -> bool,
) -> Vec<Candidate> {
    debug_assert!(ef >= 1, "filtered_beam_layer0 ef must be ≥ 1");

    let mut visited: HashSet<VectorId> = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

    // Seed the frontier from the upper-layer entry points. Each ep
    // contributes to BOTH the candidate frontier (always) AND the
    // result heap (only if it passes the eligibility predicate).
    for &ep in eps {
        if !visited.insert(ep) {
            continue;
        }
        let Some(bytes) = graph.vector_bytes(ep) else {
            continue;
        };
        let d = kernel.distance(query, bytes);
        let c = Candidate::new(d, ep);
        candidates.push(Reverse(c));
        if is_allowed(ep) {
            results.push(c);
            if results.len() > ef {
                let _ = results.pop();
            }
        }
    }

    while let Some(Reverse(c)) = candidates.pop() {
        // Termination: if the closest unexpanded candidate is farther
        // than the worst kept matching result AND we have ef matches,
        // no further expansion can improve the result heap.
        if let Some(furthest) = results.peek() {
            if results.len() >= ef && c.distance > furthest.distance {
                break;
            }
        }
        let Some(node) = graph.nodes.get(&c.id) else {
            continue;
        };
        let Some(layer_adj) = node.neighbors.first() else {
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
            let n_allowed = is_allowed(n);

            // Frontier-add discipline: explore a neighbor if results
            // are not yet full OR it is strictly closer than the worst
            // kept — the SAME budget as the standard search_layer, so
            // the expansion cost matches the unfiltered beam. The
            // result heap, by contrast, admits only matching nodes.
            let should_explore = match results.peek() {
                Some(worst) => results.len() < ef || cand.distance < worst.distance,
                None => true,
            };
            if should_explore {
                candidates.push(Reverse(cand));
            }
            if n_allowed && should_explore {
                results.push(cand);
                if results.len() > ef {
                    let _ = results.pop();
                }
            }
        }
    }

    let mut out: Vec<Candidate> = results.into_iter().collect();
    out.sort();
    out
}

/// Filter-during-search over a bare [`HnswGraph`] with a caller-
/// supplied per-vector predicate — the served-HNSW entry point for
/// #815. Unlike [`FilteredHnsw`] this requires NO payload sidecar and
/// NO payload-aware insert: the caller (the served
/// `HnswVectorSearchProvider`) already holds a dense `VectorId → label`
/// map, so it supplies `is_allowed` directly and keeps the cheap
/// standard [`HnswGraph::insert`] (no `O(N)` payload-edge
/// augmentation) on its incremental read-after-write path (#787).
///
/// Returns up to `k` `(VectorId, distance)` pairs, closest-first, each
/// satisfying `is_allowed`. Tombstoned vectors are excluded even if
/// they pass the predicate (mirrors [`HnswGraph::search`]). If
/// `ef == 0` the graph's `params.ef_search` default is used; the
/// effective layer-0 beam is `max(ef, k)` per Malkov & Yashunin §4.
///
/// # Complexity vs post-filtering (#815)
///
/// A post-filter retrieves `k` by distance then discards
/// non-matching, returning only ~`k · selectivity` true matches (the
/// recall collapse). This path keeps the result heap full of `k`
/// *matching* vectors while the frontier still walks through
/// non-matching ones, so it returns `k` true matches at any
/// selectivity — at the cost of a wider reachable-set walk when the
/// filter is selective (bounded by `ef`/graph connectivity).
///
/// # Errors
///
/// - [`VectorIndexError::DimensionMismatch`] if `query.len()` does not
///   equal the graph's `bytes_per_vector`.
pub fn predicate_filtered_search(
    graph: &HnswGraph,
    query: &[u8],
    k: usize,
    ef: usize,
    kernel: &dyn DistanceKernel,
    is_allowed: &dyn Fn(VectorId) -> bool,
) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
    graph.validate_vector_bytes(query)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let Some(entry) = graph.entry_point else {
        return Ok(Vec::new());
    };

    let ef_use = if ef == 0 { graph.params.ef_search } else { ef };
    let ef_layer0 = ef_use.max(k);

    // Greedy zoom through the upper layers ignores the predicate —
    // non-matching nodes still route at the upper layers (same as
    // FilteredHnsw::filtered_search).
    let mut eps: Vec<VectorId> = vec![entry];
    for layer in (1..=graph.max_level).rev() {
        let next = super::search::search_layer(graph, query, &eps, layer, 1, kernel);
        if let Some(c) = next.first() {
            eps = vec![c.id];
        }
    }

    let results = filtered_beam_layer0(graph, query, &eps, ef_layer0, kernel, is_allowed);
    let out: Vec<(VectorId, f32)> = results
        .into_iter()
        .filter(|c| !graph.is_tombstoned(c.id))
        .take(k)
        .map(|c| (c.id, c.distance.0))
        .collect();
    Ok(out)
}

/// Helper: compute distance, returning +inf on a length
/// mismatch (so the prune sort puts the offending neighbor
/// last). Cosmetic; the calling path validates lengths
/// upstream.
#[inline]
fn kernel_distance_or_inf(kernel: &dyn DistanceKernel, a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    kernel.distance(a, b)
}

// ─── F.4 dispatcher trait impl ───────────────────────────────────
//
// Per ADR-035 amendment-04 D-7. Pure delegation: the trait body
// forwards verbatim to the existing public `filtered_search`
// method. Adding this impl does NOT change the F.2 search body —
// the dispatcher logic lives in `crate::dispatcher` and consumes
// `&dyn FilteredVectorIndex` references.

impl crate::dispatcher::FilteredVectorIndex for FilteredHnsw {
    #[inline]
    fn kind(&self) -> crate::dispatcher::BackendKind {
        crate::dispatcher::BackendKind::Hnsw
    }

    #[inline]
    fn len(&self) -> usize {
        FilteredHnsw::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        FilteredHnsw::is_empty(self)
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
    ) -> Result<Vec<(VectorId, f32)>, VectorIndexError> {
        FilteredHnsw::filtered_search(self, query, k, filter, ef, kernel, read_lsn)
    }
}

// ─── unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2F32;
    use arcgraph_core::{StringId, TenantId};

    fn bytes_of(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    // ─── Filter::matches (HNSW-local payload evaluation) ─────
    //
    // The Filter constructors / `is_any` / `required_label`
    // are unit-tested in `crate::query`; this module owns the
    // [`Payload`]-aware match evaluator.

    #[test]
    fn filter_matches_any_accepts_every_payload() {
        let f = Filter::Any;
        assert!(f.matches(&Payload::default()));
        assert!(f.matches(&Payload::with_labels(vec![LabelId::new(1)])));
        let mut props = HashMap::new();
        props.insert(StringId::new(0), PropertyValue::U32(0));
        let with_props = Payload {
            tenant_id: Some(TenantId::new(7)),
            labels: vec![LabelId::new(2)],
            properties: props,
            ..Payload::default()
        };
        assert!(f.matches(&with_props));
    }

    #[test]
    fn filter_matches_label_eq_when_payload_carries_label() {
        let f = Filter::LabelEq(LabelId::new(7));
        let p = Payload::with_labels(vec![LabelId::new(7)]);
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_label_eq_rejects_when_payload_lacks_label() {
        let f = Filter::LabelEq(LabelId::new(7));
        let p = Payload::with_labels(vec![LabelId::new(8)]);
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_matches_tenant_when_payload_carries_same_tenant() {
        let f = Filter::Tenant(TenantId::DEFAULT);
        let p = Payload {
            tenant_id: Some(TenantId::DEFAULT),
            ..Default::default()
        };
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_matches_tenant_when_payload_has_no_tenant_tag() {
        // Single-tenant arena: payload tenant tag is redundant
        // because arena selection IS the tenant filter.
        let f = Filter::Tenant(TenantId::DEFAULT);
        let p = Payload::default();
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_rejects_tenant_when_payload_carries_different_tenant() {
        let f = Filter::Tenant(TenantId::new(7));
        let p = Payload {
            tenant_id: Some(TenantId::new(8)),
            ..Default::default()
        };
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_label_in_matches_when_payload_has_any_listed_label() {
        let f = Filter::LabelIn(vec![LabelId::new(1), LabelId::new(2)]);
        let p = Payload::with_labels(vec![LabelId::new(2)]);
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_label_in_rejects_when_payload_has_no_listed_label() {
        let f = Filter::LabelIn(vec![LabelId::new(1)]);
        let p = Payload::with_labels(vec![LabelId::new(2)]);
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_label_in_empty_list_rejects_everything() {
        let f = Filter::LabelIn(vec![]);
        let p = Payload::with_labels(vec![LabelId::new(1)]);
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_property_eq_matches_when_payload_has_property() {
        let f = Filter::PropertyEq(StringId::new(10), PropertyValue::U32(42));
        let mut props = HashMap::new();
        props.insert(StringId::new(10), PropertyValue::U32(42));
        let p = Payload {
            properties: props,
            ..Default::default()
        };
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_property_eq_rejects_when_value_differs() {
        let f = Filter::PropertyEq(StringId::new(10), PropertyValue::U32(42));
        let mut props = HashMap::new();
        props.insert(StringId::new(10), PropertyValue::U32(43));
        let p = Payload {
            properties: props,
            ..Default::default()
        };
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_and_empty_is_always_true() {
        let f = Filter::And(vec![]);
        let p = Payload::default();
        assert!(f.matches(&p));
    }

    #[test]
    fn filter_or_empty_is_always_false() {
        let f = Filter::Or(vec![]);
        let p = Payload::default();
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_and_short_circuits_on_first_failing_child() {
        let f = Filter::And(vec![
            Filter::LabelIn(vec![LabelId::new(1)]),
            Filter::LabelIn(vec![LabelId::new(2)]),
        ]);
        // Has label 1 but not 2 → fails.
        let p = Payload::with_labels(vec![LabelId::new(1)]);
        assert!(!f.matches(&p));
    }

    #[test]
    fn filter_or_passes_with_any_passing_child() {
        let f = Filter::Or(vec![
            Filter::LabelIn(vec![LabelId::new(1)]),
            Filter::LabelIn(vec![LabelId::new(2)]),
        ]);
        let p = Payload::with_labels(vec![LabelId::new(2)]);
        assert!(f.matches(&p));
    }

    // ─── ADR-041 §D-3a — MVCC visibility on Payload ─────────

    /// PIN: `Payload::default()` is always-visible at any
    /// `read_lsn` (commit_lsn = ZERO, expired_lsn = MAX). Mirrors
    /// ADR-039 §D-2 v1.0 BM25 posture.
    #[test]
    fn payload_default_is_always_visible() {
        let p = Payload::default();
        assert_eq!(p.commit_lsn, Lsn::ZERO);
        assert_eq!(p.expired_lsn, Lsn::MAX);
        assert!(p.is_visible_at(Lsn::ZERO));
        assert!(p.is_visible_at(Lsn::new(1_000)));
        assert!(p.is_visible_at(Lsn::new(u64::MAX - 1)));
    }

    /// PIN: ADR-041 §D-3 — `commit_lsn ≤ read_lsn` is INCLUSIVE
    /// on the commit side; mirrors ADR-039 §D-3 BM25 boundary.
    #[test]
    fn payload_visibility_inclusive_at_commit_lsn() {
        let p = Payload::default().with_lsn_window(Lsn::new(10), Lsn::MAX);
        assert!(!p.is_visible_at(Lsn::new(9)), "before commit invisible");
        assert!(
            p.is_visible_at(Lsn::new(10)),
            "at commit visible (inclusive)"
        );
        assert!(p.is_visible_at(Lsn::new(11)), "after commit visible");
    }

    /// PIN: ADR-041 §D-3 — `read_lsn < expired_lsn` (i.e., the
    /// upper bound is EXCLUSIVE on the expired_lsn side; the
    /// row at `expired_lsn = expire` is invisible at
    /// `read_lsn = expire`).
    #[test]
    fn payload_visibility_exclusive_at_expired_lsn() {
        let p = Payload::default().with_lsn_window(Lsn::new(10), Lsn::new(20));
        assert!(p.is_visible_at(Lsn::new(19)), "before expiry visible");
        assert!(!p.is_visible_at(Lsn::new(20)), "at expiry invisible");
        assert!(!p.is_visible_at(Lsn::new(21)), "after expiry invisible");
    }

    /// PIN: ADR-041 §D-3 saturating-add upper bound: `read_lsn =
    /// MAX` does not panic / wrap; the saturating_add(1) guards
    /// the boundary. Mirrors ADR-039 amendment-01 semantic.
    #[test]
    fn payload_visibility_saturates_at_lsn_max() {
        let p = Payload::default(); // commit ZERO, expired MAX
        // At read_lsn = MAX, expired_lower saturates at MAX. Row
        // is visible because expired_lsn == MAX ≥ MAX.
        assert!(p.is_visible_at(Lsn::MAX));
    }

    /// PIN: ADR-041 §D-3a — `with_lsn_window` is the canonical
    /// builder for stamping a visibility window. Round-trip
    /// confirmed.
    #[test]
    fn payload_with_lsn_window_round_trips() {
        let p = Payload::with_labels(vec![LabelId::new(7)])
            .with_lsn_window(Lsn::new(42), Lsn::new(100));
        assert_eq!(p.commit_lsn, Lsn::new(42));
        assert_eq!(p.expired_lsn, Lsn::new(100));
        assert_eq!(p.labels, vec![LabelId::new(7)]);
    }

    // ─── FilteredHnsw — basic ────────────────────────────────

    #[test]
    fn filtered_hnsw_is_empty_at_construction() {
        let g = FilteredHnsw::new(HnswParams::default(), 4, &L2F32);
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn filtered_insert_populates_payload_sidecar() {
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        let payload = Payload::with_labels(vec![LabelId::new(1)]);
        g.filtered_insert(
            VectorId::new(7),
            &bytes_of(&[1.0, 2.0, 3.0]),
            payload.clone(),
            &L2F32,
        )
        .unwrap();
        assert_eq!(g.payload(VectorId::new(7)), Some(&payload));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn filtered_insert_validates_dim() {
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        let r = g.filtered_insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 2.0, 3.0, 4.0]),
            Payload::default(),
            &L2F32,
        );
        assert!(matches!(
            r,
            Err(VectorIndexError::DimensionMismatch { expected: 3, .. })
        ));
    }

    #[test]
    fn filtered_search_empty_graph_returns_empty() {
        let g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0]),
                5,
                &Filter::And(vec![]),
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn filtered_search_zero_k_returns_empty() {
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        g.filtered_insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 0.0, 0.0]),
            Payload::with_labels(vec![LabelId::new(1)]),
            &L2F32,
        )
        .unwrap();
        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0]),
                0,
                &Filter::LabelIn(vec![LabelId::new(1)]),
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn filtered_search_returns_only_matching_vectors() {
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        let label_a = LabelId::new(1);
        let label_b = LabelId::new(2);
        // 4 vectors: ids 0, 1 with label A; ids 2, 3 with label B.
        g.filtered_insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 0.0, 0.0]),
            Payload::with_labels(vec![label_a]),
            &L2F32,
        )
        .unwrap();
        g.filtered_insert(
            VectorId::new(1),
            &bytes_of(&[0.99, 0.01, 0.0]),
            Payload::with_labels(vec![label_a]),
            &L2F32,
        )
        .unwrap();
        g.filtered_insert(
            VectorId::new(2),
            &bytes_of(&[0.98, 0.02, 0.0]),
            Payload::with_labels(vec![label_b]),
            &L2F32,
        )
        .unwrap();
        g.filtered_insert(
            VectorId::new(3),
            &bytes_of(&[0.97, 0.03, 0.0]),
            Payload::with_labels(vec![label_b]),
            &L2F32,
        )
        .unwrap();

        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0]),
                4,
                &Filter::LabelIn(vec![label_a]),
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        // Only label-A vectors (0 and 1) should appear.
        assert_eq!(r.len(), 2);
        let ids: Vec<VectorId> = r.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&VectorId::new(0)));
        assert!(ids.contains(&VectorId::new(1)));
    }

    #[test]
    fn filtered_search_dispatch_high_selectivity_uses_post_filter() {
        // selectivity = 1.0 → high path. All vectors match the
        // filter; result count == k regardless.
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        for i in 0..5u32 {
            g.filtered_insert(
                VectorId::new(i),
                &bytes_of(&[i as f32 * 0.01, 0.0, 0.0]),
                Payload::with_labels(vec![LabelId::new(1)]),
                &L2F32,
            )
            .unwrap();
        }
        let r = g
            .filtered_search_dispatch(
                &bytes_of(&[0.0, 0.0, 0.0]),
                3,
                &Filter::LabelIn(vec![LabelId::new(1)]),
                1.0,
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn filtered_search_dispatch_low_selectivity_uses_filtered_traversal() {
        // selectivity = 0.1 → low path. Only some vectors match.
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        let match_label = LabelId::new(1);
        let other_label = LabelId::new(2);
        for i in 0..10u32 {
            let label = if i < 2 { match_label } else { other_label };
            g.filtered_insert(
                VectorId::new(i),
                &bytes_of(&[i as f32 * 0.01, 0.0, 0.0]),
                Payload::with_labels(vec![label]),
                &L2F32,
            )
            .unwrap();
        }
        let r = g
            .filtered_search_dispatch(
                &bytes_of(&[0.0, 0.0, 0.0]),
                5,
                &Filter::LabelIn(vec![match_label]),
                0.1,
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        // Only the 2 matching vectors should appear.
        assert_eq!(r.len(), 2);
        for (id, _) in &r {
            assert!(id.raw() < 2, "non-matching id {id:?} returned");
        }
    }

    #[test]
    fn filtered_insert_with_no_labels_skips_payload_aware_edges() {
        // Smoke test: an empty-payload insert is equivalent to a
        // standard `inner.insert`. Verifies the early-return in
        // `filtered_insert` doesn't break the standard insert
        // path.
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        g.filtered_insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 0.0, 0.0]),
            Payload::default(),
            &L2F32,
        )
        .unwrap();
        assert_eq!(g.len(), 1);
        // Payload still recorded (as empty); the augmentation
        // is what's skipped, not the sidecar.
        assert_eq!(g.payload(VectorId::new(0)), Some(&Payload::default()));
    }

    #[test]
    fn mark_deleted_drops_payload_and_tombstones_inner() {
        let mut g = FilteredHnsw::new(HnswParams::default(), 3, &L2F32);
        g.filtered_insert(
            VectorId::new(0),
            &bytes_of(&[1.0, 0.0, 0.0]),
            Payload::with_labels(vec![LabelId::new(1)]),
            &L2F32,
        )
        .unwrap();
        g.mark_deleted(VectorId::new(0));
        assert!(g.payload(VectorId::new(0)).is_none());
        assert!(g.inner.is_tombstoned(VectorId::new(0)));
        // search returns empty (only tombstoned vector).
        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0]),
                5,
                &Filter::LabelIn(vec![LabelId::new(1)]),
                10,
                &L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert!(r.is_empty());
    }
}
