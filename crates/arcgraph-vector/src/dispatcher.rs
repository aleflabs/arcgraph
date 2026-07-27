//! Selectivity dispatcher — Slice F.4 (M3.a Phase 6).
//!
//! Per ADR-035 §6.2 + [`amendment-04`]. The F.4 dispatcher routes
//! a filtered vector search across the F.2 (HNSW) and F.3
//! (DiskANN) backends based on the canonical [`Filter`] variant.
//! v1.0 routing is **variant-based** — no cost model, no
//! statistics — and the dispatcher is a stateless pure function:
//! `(filter, backends) → backend choice → search result`.
//!
//! ## What this slice ships
//!
//! - [`FilteredVectorIndex`] — object-safe trait that both
//!   `FilteredHnsw` and `DiskAnnGraph` implement (via shim
//!   blocks in their respective modules — neither backend's
//!   `filtered_search` body changes per the F.4 hard-boundary
//!   contract; the shim is pure delegation).
//! - [`BackendKind`] — backend identity for tracing /
//!   observability.
//! - [`BackendSet`] — caller-owned wrapper grouping the
//!   `Option<&dyn FilteredVectorIndex>` slots for HNSW + DiskANN.
//! - [`BackendSet::dispatch_filtered_search`] — the routing
//!   function itself.
//!
//! ## v1.0 routing policy (variant-based)
//!
//! Per [`amendment-04`] D-2:
//!
//! | Filter variant         | Primary  | Fallback on `UnsupportedFilter` |
//! |------------------------|----------|----------------------------------|
//! | [`Filter::Any`]        | DiskANN if present, else HNSW | the other |
//! | [`Filter::LabelEq`]    | DiskANN (per-label entry-point cache) | HNSW |
//! | [`Filter::Tenant`]     | HNSW     | (none — DiskANN unsupported)     |
//! | [`Filter::LabelIn`]    | HNSW     | (none)                           |
//! | [`Filter::PropertyEq`] | HNSW     | (none)                           |
//! | [`Filter::And`]        | HNSW     | (none)                           |
//! | [`Filter::Or`]         | HNSW     | (none)                           |
//!
//! The dispatcher catches [`VectorIndexError::UnsupportedFilter`]
//! from the primary and routes to the fallback when present;
//! every other error propagates unchanged (a `DimensionMismatch`
//! is a real failure that the fallback would also encounter, so
//! masking it would hide correctness bugs).
//!
//! ## What this slice does NOT ship
//!
//! - **Cost-model dispatch.** No selectivity estimation, no
//!   histograms, no statistics. The per-backend internal
//!   dispatchers (`FilteredHnsw::filtered_search_dispatch`,
//!   `DiskAnnGraph::filtered_search_dispatch`) handle their own
//!   intra-backend selectivity-aware switch (post-filter vs
//!   filtered-traversal vs brute-force). F.4 picks the BACKEND;
//!   the backends pick their own intra-backend strategy.
//!   Cost-model dispatch lifts to amendment-05 once the secondary
//!   B-tree histogram interface lands (S-8 caveat in ADR-035 §6.2).
//! - **State or caching.** The dispatcher holds zero state. Every
//!   call recomputes the routing decision (single-branch
//!   `match` on the filter discriminant — measured at ~543 ps in
//!   the prior PR #132 `filter_dispatch` bench).
//!
//! ## Latency / memory budget
//!
//! - Routing decision: single `match` on the filter
//!   discriminant. Constant-time, branch-predictable.
//! - Trait dispatch: one v-table lookup per backend invocation.
//!   Amortized over the search body (microsecond-millisecond
//!   range); the dispatcher overhead is < 1 % of the surrounding
//!   filtered-search hot path per `benches/dispatcher.rs`.
//! - State: zero. The dispatcher takes references and returns
//!   them; no per-query allocation in the routing path itself
//!   (the search result allocation is the backend's
//!   responsibility).
//!
//! [`amendment-04`]: ../../docs/adr/amendments/ADR-035-amendment-04.md

use arcgraph_core::Lsn;

use crate::Result;
use crate::distance::DistanceKernel;
use crate::error::VectorIndexError;
use crate::ids::VectorId;
use crate::query::Filter;

// ─── BackendKind ─────────────────────────────────────────────────

/// Backend identity for tracing / observability.
///
/// Returned by [`FilteredVectorIndex::kind`]. The dispatcher does
/// NOT use this for routing decisions — routing is structural
/// (the caller places each backend in the matching slot of a
/// [`BackendSet`]). `BackendKind` exists for `tracing` field
/// values and debug prints.
///
/// `#[non_exhaustive]` so future backends (Phase 7+ vector
/// engines, mock backends in F.5 proptests) can add variants
/// without breaking external match arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// Hierarchical Navigable Small World (Slice F.2).
    Hnsw,
    /// DiskANN / Vamana (Slice F.3).
    DiskAnn,
}

impl BackendKind {
    /// Stable string label for `tracing` field values.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::DiskAnn => "diskann",
        }
    }
}

// ─── FilteredVectorIndex trait ───────────────────────────────────

/// Object-safe vector backend abstraction.
///
/// Both `FilteredHnsw` and `DiskAnnGraph` implement this trait
/// via pure-delegation shim blocks in their respective modules
/// (`hnsw/filtered.rs` and `diskann/filtered.rs`). The shim does
/// NOT change the underlying `filtered_search` body — F.4's
/// hard-boundary contract is "trait impl is delegation; routing
/// lives in `dispatcher.rs`."
///
/// ## Object-safety
///
/// Every method has a concrete (non-`Self`) return type and uses
/// trait-object kernel (`&dyn DistanceKernel`) for distance
/// dispatch. `&dyn FilteredVectorIndex` is therefore valid; the
/// dispatcher consumes exactly that.
///
/// ## What the trait does NOT promise
///
/// - **Universal filter coverage.** Per ADR-035 amendment-03 D-3,
///   each backend's v1.0 capability is narrower than the canonical
///   [`Filter`] enum. Implementations return
///   [`VectorIndexError::UnsupportedFilter`] for variants they
///   don't support; the dispatcher catches that error and
///   escalates to the fallback per amendment-04 D-3.
/// - **Identical results across backends.** Two backends searching
///   the same dataset with the same filter MAY return different
///   top-k orderings due to graph topology differences (HNSW's
///   stochastic levels vs Vamana's deterministic α-prune). The
///   recall floor (≥ 0.85 per AC-5/AC-6) is preserved; the exact
///   ranking can differ. The cross-backend correctness pins in
///   `tests/filter_unification.rs` (PR #132) document the
///   tolerated divergence.
///
/// ## Why a trait, not a function
///
/// Per ADR-035-amendment-04 §"Rejected alternatives" Option B:
/// the trait approach (a) opens the door to future backends
/// without a signature change, (b) gives F.5 multi-tenant proptest
/// a clean mock seam, and (c) lets the M4 query layer take
/// `&dyn FilteredVectorIndex` references from the catalog without
/// hard-coding the two backend types.
pub trait FilteredVectorIndex {
    /// Backend identity for tracing / observability. Used by the
    /// dispatcher's `tracing::debug!` field values; not consulted
    /// by the routing decision (routing is positional).
    fn kind(&self) -> BackendKind;

    /// Live vector count. Used by the dispatcher's empty-graph
    /// short-circuit (an empty-graph dispatch returns
    /// `Ok(Vec::new())` without invoking the backend's search).
    ///
    /// "Live" excludes tombstones; for HNSW
    /// (`FilteredHnsw::len`) this includes tombstoned slots
    /// because they still serve as routing hubs, but the
    /// short-circuit semantics are uniform across backends:
    /// `len() == 0` ⇔ no addressable vectors ⇒ empty result.
    fn len(&self) -> usize;

    /// `true` when the backend holds zero live vectors. Default
    /// impl delegates to `len() == 0`; backends with a cheaper
    /// emptiness check may override.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run a filtered search against this backend.
    ///
    /// Mirrors the per-backend public surfaces:
    ///
    /// - `FilteredHnsw::filtered_search` — `ef` is HNSW's
    ///   layer-0 beam width.
    /// - `DiskAnnGraph::filtered_search` — `ef` is DiskANN's
    ///   `l_search` beam width.
    ///
    /// `read_lsn` is the MVCC visibility key per ADR-041 §D-1.
    /// Each backend filters the result list to entries with
    /// `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn`. Callers
    /// without snapshot context pass `Lsn::MAX` (the most-
    /// permissive read; everything is visible). v1.0 production
    /// callers source `read_lsn` from `TransactionManager::current_lsn()`
    /// via the executor's transaction context.
    ///
    /// Returns [`VectorIndexError::UnsupportedFilter`] for
    /// variants the backend doesn't support at v1.0 (per ADR-035
    /// amendment-03 D-3); the dispatcher catches this and routes
    /// to the fallback. Other errors (`DimensionMismatch`,
    /// `Rebuilding`, …) propagate unchanged.
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes signature past clippy default; documented widening
    fn filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>>;
}

// ─── DispatchPreference ──────────────────────────────────────────

/// Routing preference computed from a [`Filter`] variant.
///
/// Per ADR-035-amendment-04 D-2. The dispatcher consults this
/// value to pick the primary backend for a given filter; the
/// caller does not interact with it directly (it's a public type
/// only so [`dispatch_preference`] can return it for diagnostics
/// and for the F.5 multi-tenant proptest's routing assertions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DispatchPreference {
    /// Primary: DiskANN. Fallback: HNSW.
    /// Applies to [`Filter::Any`] and [`Filter::LabelEq`].
    DiskAnnPreferred,
    /// Primary: HNSW. No DiskANN fallback (DiskANN v1.0 cannot
    /// handle these variants per ADR-035 amendment-03 D-3).
    /// Applies to every other [`Filter`] variant.
    HnswOnly,
}

/// Compute the v1.0 routing preference for a filter variant.
///
/// Pure function over the discriminant — single-branch O(1).
/// Exposed for the F.5 multi-tenant proptest's routing
/// assertions and for tracing field values; the dispatcher
/// inlines the same match internally.
#[inline]
#[must_use]
pub const fn dispatch_preference(filter: &Filter) -> DispatchPreference {
    match filter {
        Filter::Any | Filter::LabelEq(_) => DispatchPreference::DiskAnnPreferred,
        Filter::Tenant(_)
        | Filter::LabelIn(_)
        | Filter::PropertyEq(_, _)
        | Filter::And(_)
        | Filter::Or(_) => DispatchPreference::HnswOnly,
    }
}

// ─── BackendSet ──────────────────────────────────────────────────

/// Caller-owned grouping of backend handles.
///
/// `BackendSet` carries `Option<&dyn FilteredVectorIndex>` slots
/// for HNSW + DiskANN. The dispatcher uses positional routing —
/// the `hnsw` slot must hold an HNSW-shaped backend, the
/// `diskann` slot must hold a DiskANN-shaped backend. The trait
/// `kind()` method exists for observability, not for runtime
/// type-checking the slot.
///
/// At v1.0 a single arena holds one of each (HNSW for the
/// payload-aware path, DiskANN for the per-label hot path); the
/// catalog populates both slots when both are built. Single-
/// backend deployments populate only one slot; the dispatcher
/// honors that and uses the single available backend for every
/// filter (subject to the variant capability matrix —
/// e.g., `Tenant` on a DiskANN-only deployment surfaces as
/// `UnsupportedFilter` because DiskANN can't handle it).
///
/// ## Lifetime
///
/// `'a` is the borrow of the underlying backend handles. The
/// `BackendSet` is itself a thin value (~16 B); construct it
/// per-query at the dispatcher call site.
#[derive(Default)]
pub struct BackendSet<'a> {
    /// HNSW backend slot. `None` when the arena has no HNSW
    /// instance built (e.g., DiskANN-only deployments).
    pub hnsw: Option<&'a dyn FilteredVectorIndex>,
    /// DiskANN backend slot. `None` when the arena has no
    /// DiskANN instance built (e.g., HNSW-only deployments, or
    /// during the build window before DiskANN bulk-load
    /// completes).
    pub diskann: Option<&'a dyn FilteredVectorIndex>,
}

impl<'a> BackendSet<'a> {
    /// Construct an empty backend set. Equivalent to
    /// [`Default::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            hnsw: None,
            diskann: None,
        }
    }

    /// Set the HNSW slot. Builder-style for ergonomic
    /// per-query construction.
    #[must_use]
    pub fn with_hnsw(mut self, hnsw: &'a dyn FilteredVectorIndex) -> Self {
        self.hnsw = Some(hnsw);
        self
    }

    /// Set the DiskANN slot. Builder-style for ergonomic
    /// per-query construction.
    #[must_use]
    pub fn with_diskann(mut self, diskann: &'a dyn FilteredVectorIndex) -> Self {
        self.diskann = Some(diskann);
        self
    }

    /// Whether at least one backend slot is populated.
    #[inline]
    #[must_use]
    pub fn has_any(&self) -> bool {
        self.hnsw.is_some() || self.diskann.is_some()
    }

    /// Dispatch a filtered search per the v1.0 variant-based
    /// routing policy (ADR-035-amendment-04 D-2).
    ///
    /// Routing decision tree:
    ///
    /// 1. Compute the [`DispatchPreference`] from the filter
    ///    variant.
    /// 2. Select the primary + fallback per the preference and
    ///    backend availability (per amendment-04 D-3 escalation
    ///    contract).
    /// 3. If the primary's `is_empty() == true`, short-circuit to
    ///    `Ok(Vec::new())` (per amendment-04 D-4 empty-graph
    ///    short-circuit). Otherwise invoke the primary's
    ///    `filtered_search`.
    /// 4. On `Err(UnsupportedFilter)` from the primary, try the
    ///    fallback if present (with the same empty-graph
    ///    short-circuit). On any other error, propagate.
    /// 5. If both backends are absent, or both reject with
    ///    `UnsupportedFilter`, return
    ///    [`VectorIndexError::UnsupportedFilter`] with a
    ///    descriptive `reason`.
    ///
    /// ## Determinism
    ///
    /// The dispatcher is a pure function over its inputs. Same
    /// inputs (same backend handles, same filter, same query)
    /// produce the same output across runs. The underlying
    /// backends own their own determinism (HNSW's seeded layer
    /// assignment, DiskANN's deterministic medoid selection); the
    /// dispatcher does not introduce any non-determinism.
    ///
    /// ## Tenant isolation
    ///
    /// The dispatcher does NOT enforce tenant isolation directly
    /// — that's the arena's responsibility (per ADR-011 / ADR-035
    /// §9.11, the owning arena is selected before the dispatcher
    /// sees it). When the filter
    /// carries a [`Filter::Tenant`] variant, the dispatcher routes
    /// to HNSW (per the variant table) and HNSW's payload sidecar
    /// evaluates the tenant predicate per-vector; the F.5
    /// multi-tenant proptest validates the end-to-end isolation.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::UnsupportedFilter`] when neither
    ///   backend is present, or when both are present but both
    ///   reject the filter (impossible at v1.0 per the variant
    ///   table; reachable in Phase 7+ if a new variant lands
    ///   without HNSW coverage).
    /// - Any error returned by the chosen backend's
    ///   `filtered_search` method that is NOT
    ///   [`VectorIndexError::UnsupportedFilter`] —
    ///   `DimensionMismatch`, `Rebuilding`, etc., propagate
    ///   unchanged.
    ///
    /// `read_lsn` (per ADR-041 §D-1) is the MVCC visibility key.
    /// The dispatcher threads it verbatim into both the primary
    /// and the fallback backend's `filtered_search`; each backend
    /// applies its own visibility filter (HNSW via `Payload`,
    /// DiskANN via the slot-indexed LSN parallel arrays).
    #[allow(clippy::too_many_arguments)] // ADR-041 read_lsn pushes over default 7-arg threshold; documented widening
    pub fn dispatch_filtered_search(
        &self,
        query: &[u8],
        k: usize,
        filter: &Filter,
        ef: usize,
        kernel: &dyn DistanceKernel,
        read_lsn: Lsn,
    ) -> Result<Vec<(VectorId, f32)>> {
        let preference = dispatch_preference(filter);
        let (primary, fallback) = self.select(preference);

        // No backends at all — caller misconfiguration. Surface
        // as UnsupportedFilter (rather than panic) so the
        // dispatcher's contract stays total. v1.0 catalog never
        // produces this state; defensive handling guards future
        // refactors.
        let Some(primary) = primary else {
            return Err(VectorIndexError::UnsupportedFilter {
                reason: format!(
                    "F.4 dispatcher has no backend available for filter \
                     preference {preference:?}; arena likely misconfigured \
                     or both build paths in flight"
                ),
            });
        };

        // Empty-graph short-circuit. The trait's `is_empty()`
        // has a default impl as `len() == 0`; backends MAY
        // override for cheaper checks. An empty primary with a
        // present fallback still short-circuits (empty arena =
        // empty result, regardless of which backend would have
        // run); this matches the contract that an empty arena
        // returns `Ok(Vec::new())` per backend, lifted to the
        // dispatcher.
        if primary.is_empty() {
            // If the fallback exists and is non-empty, an empty
            // primary is anomalous (the catalog should keep both
            // backends in sync) — but we still short-circuit on
            // primary because that's the routing choice the
            // variant table makes. Catalog drift is operator-
            // visible elsewhere; the dispatcher honors the table.
            //
            // Defense-in-depth (codex retro V2): the asymmetric
            // case `primary.is_empty() && !fallback.is_empty()`
            // would silently drop queries that have valid answers
            // in the fallback if the catalog ever drifts. The
            // production short-circuit semantics are pinned by
            // `dispatcher_asymmetric_empty_violates_catalog_contract`
            // — we keep `Ok(vec![])` so amendment-04 D-4 holds —
            // but emit a `tracing::error!` and a `debug_assert!`
            // so the drift surfaces operationally and in test
            // runs without changing release-mode behavior.
            if let Some(fb) = fallback
                && !fb.is_empty()
            {
                tracing::error!(
                    target: "arcgraph.vector.dispatcher",
                    catalog_invariant = "primary_empty_fallback_nonempty",
                    primary_kind = ?primary.kind(),
                    fallback_kind = ?fb.kind(),
                    "amendment-04 D-4 violation: dispatcher short-circuits Ok(vec![]) but fallback has data — silent dropped-query risk"
                );
                debug_assert!(
                    fb.is_empty(),
                    "amendment-04 D-4 violated: primary empty but fallback non-empty (primary_kind={:?}, fallback_kind={:?})",
                    primary.kind(),
                    fb.kind(),
                );
            }
            return Ok(Vec::new());
        }

        match primary.filtered_search(query, k, filter, ef, kernel, read_lsn) {
            Ok(hits) => Ok(hits),
            Err(VectorIndexError::UnsupportedFilter {
                reason: primary_reason,
            }) => {
                // Escalate to fallback per amendment-04 D-3.
                let Some(fallback) = fallback else {
                    return Err(VectorIndexError::UnsupportedFilter {
                        reason: format!(
                            "F.4 dispatcher primary backend rejected filter \
                             ({primary_reason}); no fallback configured"
                        ),
                    });
                };
                if fallback.is_empty() {
                    return Ok(Vec::new());
                }
                match fallback.filtered_search(query, k, filter, ef, kernel, read_lsn) {
                    Ok(hits) => Ok(hits),
                    Err(VectorIndexError::UnsupportedFilter {
                        reason: fallback_reason,
                    }) => Err(VectorIndexError::UnsupportedFilter {
                        reason: format!(
                            "F.4 dispatcher: both backends rejected filter \
                             (primary: {primary_reason}; fallback: {fallback_reason})"
                        ),
                    }),
                    Err(other) => Err(other),
                }
            }
            Err(other) => Err(other),
        }
    }

    /// Resolve the (primary, fallback) backend pair per the
    /// preference and current slot availability.
    ///
    /// - [`DispatchPreference::DiskAnnPreferred`] → primary =
    ///   DiskANN if present else HNSW; fallback = the other if
    ///   present.
    /// - [`DispatchPreference::HnswOnly`] → primary = HNSW;
    ///   fallback = `None` (DiskANN v1.0 unsupported for these
    ///   variants per the capability matrix).
    ///
    /// Pure function over `self` + preference; no side effects.
    #[inline]
    fn select(
        &self,
        preference: DispatchPreference,
    ) -> (
        Option<&dyn FilteredVectorIndex>,
        Option<&dyn FilteredVectorIndex>,
    ) {
        match preference {
            DispatchPreference::DiskAnnPreferred => match (self.diskann, self.hnsw) {
                (Some(d), Some(h)) => (Some(d), Some(h)),
                (Some(d), None) => (Some(d), None),
                (None, Some(h)) => (Some(h), None),
                (None, None) => (None, None),
            },
            DispatchPreference::HnswOnly => (self.hnsw, None),
        }
    }
}

// ─── unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VectorId;
    use arcgraph_core::{LabelId, Lsn, StringId, TenantId};

    // ─── BackendKind ────────────────────────────────────────

    #[test]
    fn backend_kind_as_str_returns_stable_label() {
        assert_eq!(BackendKind::Hnsw.as_str(), "hnsw");
        assert_eq!(BackendKind::DiskAnn.as_str(), "diskann");
    }

    // ─── dispatch_preference ────────────────────────────────

    #[test]
    fn dispatch_preference_any_prefers_diskann() {
        assert_eq!(
            dispatch_preference(&Filter::Any),
            DispatchPreference::DiskAnnPreferred
        );
    }

    #[test]
    fn dispatch_preference_label_eq_prefers_diskann() {
        assert_eq!(
            dispatch_preference(&Filter::LabelEq(LabelId::new(1))),
            DispatchPreference::DiskAnnPreferred
        );
    }

    #[test]
    fn dispatch_preference_tenant_routes_hnsw_only() {
        assert_eq!(
            dispatch_preference(&Filter::Tenant(TenantId::DEFAULT)),
            DispatchPreference::HnswOnly
        );
    }

    #[test]
    fn dispatch_preference_label_in_routes_hnsw_only() {
        assert_eq!(
            dispatch_preference(&Filter::LabelIn(vec![LabelId::new(1)])),
            DispatchPreference::HnswOnly
        );
    }

    #[test]
    fn dispatch_preference_property_eq_routes_hnsw_only() {
        assert_eq!(
            dispatch_preference(&Filter::PropertyEq(
                StringId::new(0),
                crate::query::PropertyValue::U32(42)
            )),
            DispatchPreference::HnswOnly
        );
    }

    #[test]
    fn dispatch_preference_and_routes_hnsw_only() {
        assert_eq!(
            dispatch_preference(&Filter::And(vec![Filter::Any])),
            DispatchPreference::HnswOnly
        );
    }

    #[test]
    fn dispatch_preference_or_routes_hnsw_only() {
        assert_eq!(
            dispatch_preference(&Filter::Or(vec![Filter::Any])),
            DispatchPreference::HnswOnly
        );
    }

    #[test]
    fn dispatch_preference_empty_and_routes_hnsw_only() {
        // Empty And is logically always-true but structurally
        // a compound filter; routes to HNSW per the table.
        assert_eq!(
            dispatch_preference(&Filter::And(vec![])),
            DispatchPreference::HnswOnly
        );
    }

    // ─── BackendSet construction ────────────────────────────

    #[test]
    fn backend_set_default_has_no_backends() {
        let s = BackendSet::default();
        assert!(s.hnsw.is_none());
        assert!(s.diskann.is_none());
        assert!(!s.has_any());
    }

    #[test]
    fn backend_set_new_has_no_backends() {
        let s = BackendSet::new();
        assert!(!s.has_any());
    }

    // ─── Dispatcher behavior with mock backends ─────────────
    //
    // The integration tests in `tests/dispatcher.rs` use real
    // FilteredHnsw + DiskAnnGraph instances; these unit tests
    // use a minimal mock so the routing decision tree can be
    // exercised in isolation without backend builds.

    /// Test-only mock backend: records every `filtered_search`
    /// call and returns a configured response.
    struct MockBackend {
        kind: BackendKind,
        len: usize,
        response: std::sync::Mutex<MockResponse>,
        call_count: std::sync::Mutex<usize>,
    }

    enum MockResponse {
        Ok(Vec<(VectorId, f32)>),
        UnsupportedFilter,
        DimensionMismatch,
    }

    impl MockBackend {
        fn new(kind: BackendKind, len: usize, response: MockResponse) -> Self {
            Self {
                kind,
                len,
                response: std::sync::Mutex::new(response),
                call_count: std::sync::Mutex::new(0),
            }
        }
        fn calls(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    impl FilteredVectorIndex for MockBackend {
        fn kind(&self) -> BackendKind {
            self.kind
        }
        fn len(&self) -> usize {
            self.len
        }
        fn filtered_search(
            &self,
            _query: &[u8],
            _k: usize,
            _filter: &Filter,
            _ef: usize,
            _kernel: &dyn crate::distance::DistanceKernel,
            _read_lsn: Lsn,
        ) -> Result<Vec<(VectorId, f32)>> {
            *self.call_count.lock().unwrap() += 1;
            // Take the response, replace with a default that
            // wouldn't match real test expectations (so a missing
            // setup fails loud).
            let mut guard = self.response.lock().unwrap();
            let taken = std::mem::replace(&mut *guard, MockResponse::DimensionMismatch);
            match taken {
                MockResponse::Ok(hits) => Ok(hits),
                MockResponse::UnsupportedFilter => Err(VectorIndexError::UnsupportedFilter {
                    reason: format!("mock {} rejected", self.kind.as_str()),
                }),
                MockResponse::DimensionMismatch => Err(VectorIndexError::DimensionMismatch {
                    expected: 4,
                    got: 0,
                }),
            }
        }
    }

    #[test]
    fn dispatch_routes_label_eq_to_diskann_first() {
        let h = MockBackend::new(BackendKind::Hnsw, 10, MockResponse::Ok(vec![]));
        let d = MockBackend::new(
            BackendKind::DiskAnn,
            10,
            MockResponse::Ok(vec![(VectorId::new(7), 0.0)]),
        );
        let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

        let r = set
            .dispatch_filtered_search(
                &[0u8; 16],
                1,
                &Filter::LabelEq(LabelId::new(1)),
                10,
                &crate::distance::L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(r, vec![(VectorId::new(7), 0.0)]);
        assert_eq!(d.calls(), 1, "DiskANN should be called");
        assert_eq!(h.calls(), 0, "HNSW should not be called");
    }

    #[test]
    fn dispatch_routes_tenant_to_hnsw_only() {
        let h = MockBackend::new(
            BackendKind::Hnsw,
            10,
            MockResponse::Ok(vec![(VectorId::new(3), 0.0)]),
        );
        let d = MockBackend::new(BackendKind::DiskAnn, 10, MockResponse::Ok(vec![]));
        let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

        let r = set
            .dispatch_filtered_search(
                &[0u8; 16],
                1,
                &Filter::Tenant(TenantId::DEFAULT),
                10,
                &crate::distance::L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(r, vec![(VectorId::new(3), 0.0)]);
        assert_eq!(h.calls(), 1, "HNSW should be called");
        assert_eq!(d.calls(), 0, "DiskANN should NOT be called for Tenant");
    }

    #[test]
    fn dispatch_escalates_unsupported_filter_diskann_to_hnsw() {
        // Primary (DiskANN per LabelEq routing) returns
        // UnsupportedFilter — dispatcher escalates to HNSW.
        let h = MockBackend::new(
            BackendKind::Hnsw,
            10,
            MockResponse::Ok(vec![(VectorId::new(2), 0.5)]),
        );
        let d = MockBackend::new(BackendKind::DiskAnn, 10, MockResponse::UnsupportedFilter);
        let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

        let r = set
            .dispatch_filtered_search(
                &[0u8; 16],
                1,
                &Filter::LabelEq(LabelId::new(1)),
                10,
                &crate::distance::L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(r, vec![(VectorId::new(2), 0.5)]);
        assert_eq!(d.calls(), 1, "DiskANN tried first");
        assert_eq!(h.calls(), 1, "HNSW called after escalation");
    }

    #[test]
    fn dispatch_does_not_escalate_dimension_mismatch() {
        // DimensionMismatch is a real error; do NOT mask it by
        // falling through to the fallback.
        let h = MockBackend::new(
            BackendKind::Hnsw,
            10,
            MockResponse::Ok(vec![(VectorId::new(2), 0.0)]),
        );
        let d = MockBackend::new(BackendKind::DiskAnn, 10, MockResponse::DimensionMismatch);
        let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

        let r = set.dispatch_filtered_search(
            &[0u8; 16],
            1,
            &Filter::LabelEq(LabelId::new(1)),
            10,
            &crate::distance::L2F32,
            Lsn::MAX,
        );
        assert!(matches!(r, Err(VectorIndexError::DimensionMismatch { .. })));
        assert_eq!(d.calls(), 1, "DiskANN tried first");
        assert_eq!(
            h.calls(),
            0,
            "HNSW must NOT be called — DimensionMismatch is not a routing escalation"
        );
    }

    #[test]
    fn dispatch_diskann_only_falls_back_to_diskann_for_hnsw_only_variants() {
        // DiskANN-only deployment + a Tenant filter: per the
        // variant table HnswOnly → primary is HNSW → no HNSW
        // available → preference says no fallback → returns
        // UnsupportedFilter.
        let d = MockBackend::new(BackendKind::DiskAnn, 10, MockResponse::Ok(vec![]));
        let set = BackendSet::new().with_diskann(&d);

        let r = set.dispatch_filtered_search(
            &[0u8; 16],
            1,
            &Filter::Tenant(TenantId::DEFAULT),
            10,
            &crate::distance::L2F32,
            Lsn::MAX,
        );
        assert!(
            matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })),
            "DiskANN-only deployment + Tenant filter must surface UnsupportedFilter; got {r:?}"
        );
        assert_eq!(d.calls(), 0, "DiskANN must NOT be called for Tenant");
    }

    #[test]
    fn dispatch_hnsw_only_handles_label_eq_via_fallback_to_primary() {
        // HNSW-only deployment + a LabelEq filter: per the
        // variant table DiskAnnPreferred → DiskANN absent → use
        // HNSW as the (only) primary. Preference still names
        // DiskANN-preferred but the absence of DiskANN promotes
        // HNSW into the primary slot.
        let h = MockBackend::new(
            BackendKind::Hnsw,
            10,
            MockResponse::Ok(vec![(VectorId::new(5), 0.1)]),
        );
        let set = BackendSet::new().with_hnsw(&h);

        let r = set
            .dispatch_filtered_search(
                &[0u8; 16],
                1,
                &Filter::LabelEq(LabelId::new(1)),
                10,
                &crate::distance::L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(r, vec![(VectorId::new(5), 0.1)]);
        assert_eq!(h.calls(), 1);
    }

    #[test]
    fn dispatch_no_backends_returns_unsupported_filter() {
        let set = BackendSet::new();
        let r = set.dispatch_filtered_search(
            &[0u8; 16],
            1,
            &Filter::Any,
            10,
            &crate::distance::L2F32,
            Lsn::MAX,
        );
        assert!(matches!(r, Err(VectorIndexError::UnsupportedFilter { .. })));
    }

    #[test]
    fn dispatch_empty_primary_short_circuits_to_empty_result() {
        // DiskANN slot is empty (len=0). The dispatcher must
        // short-circuit to Ok(Vec::new()) without invoking the
        // backend's search.
        let d = MockBackend::new(BackendKind::DiskAnn, 0, MockResponse::Ok(vec![]));
        let set = BackendSet::new().with_diskann(&d);
        let r = set
            .dispatch_filtered_search(
                &[0u8; 16],
                5,
                &Filter::Any,
                10,
                &crate::distance::L2F32,
                Lsn::MAX,
            )
            .unwrap();
        assert!(r.is_empty());
        assert_eq!(d.calls(), 0, "Empty backend should not be called");
    }

    #[test]
    fn dispatch_both_unsupported_propagates_combined_reason() {
        // Pathological case (impossible at v1.0; reachable in
        // Phase 7+ if a new variant lands without HNSW coverage).
        // Both backends reject; dispatcher returns
        // UnsupportedFilter naming both.
        let h = MockBackend::new(BackendKind::Hnsw, 10, MockResponse::UnsupportedFilter);
        let d = MockBackend::new(BackendKind::DiskAnn, 10, MockResponse::UnsupportedFilter);
        let set = BackendSet::new().with_hnsw(&h).with_diskann(&d);

        let r = set.dispatch_filtered_search(
            &[0u8; 16],
            1,
            &Filter::Any, // DiskAnnPreferred → tries DiskANN first
            10,
            &crate::distance::L2F32,
            Lsn::MAX,
        );
        match r {
            Err(VectorIndexError::UnsupportedFilter { reason }) => {
                assert!(
                    reason.contains("primary") && reason.contains("fallback"),
                    "Both-rejection must mention both arms; got {reason:?}"
                );
            }
            other => panic!("expected UnsupportedFilter; got {other:?}"),
        }
        assert_eq!(d.calls(), 1, "DiskANN tried first");
        assert_eq!(h.calls(), 1, "HNSW tried as fallback");
    }
}
