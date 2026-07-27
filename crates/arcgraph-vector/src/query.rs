//! Canonical filter type for vector index query.
//!
//! Per ADR-035 §6 + amendment-03 (issue #127). The [`Filter`]
//! enum is the single canonical input to filter-aware vector
//! search across both HNSW (Slice F.2) and DiskANN (Slice F.3)
//! backends, and the input contract for the Phase 6 F.4
//! selectivity-driven dispatcher.
//!
//! ## Why one canonical type
//!
//! Pre-amendment-03, F.2 and F.3 each defined their own `Filter`
//! type — F.2's was a public 5-variant enum (`Tenant`,
//! `PropertyEq`, `LabelIn`, `And`, `Or`); F.3's was a public
//! struct wrapping a 2-variant private enum (`Any`, `LabelEq`).
//! Different kinds, different shapes, different capabilities.
//! The compiler accepted both because they lived in separate
//! modules, but the duplication blocked Phase 6 F.4: the
//! selectivity-driven dispatcher needs ONE canonical filter
//! contract so it can route a single query to either backend
//! without translating between two surfaces.
//!
//! Issue #127 resolved this with **Option A** (canonical enum,
//! adopted from F.2's shape with the addition of an explicit
//! `Any` variant for F.3's "no filter" fast path). This module
//! is the resolution: F.2 and F.3 both consume [`Filter`]; the
//! per-backend `filtered_search` methods adapt the variant set
//! to their own dispatch paths.
//!
//! ## Backend support matrix at v1.0
//!
//! | Variant                       | HNSW (F.2)         | DiskANN (F.3)            |
//! |-------------------------------|--------------------|--------------------------|
//! | [`Filter::Any`]               | full               | full (unfiltered fast path) |
//! | [`Filter::Tenant`]            | full               | unsupported              |
//! | [`Filter::LabelEq`]           | full               | full (per-label entry-point cache) |
//! | [`Filter::LabelIn`]           | full               | unsupported              |
//! | [`Filter::PropertyEq`]        | full               | unsupported              |
//! | [`Filter::And`]               | full               | unsupported              |
//! | [`Filter::Or`]                | full               | unsupported              |
//!
//! "unsupported" returns
//! [`crate::VectorIndexError::UnsupportedFilter`]. The Phase 6
//! F.4 dispatcher routes such filters to HNSW; the F.5 / G.4
//! follow-up adds a per-label inverted index that lets DiskANN
//! handle [`Filter::LabelIn`] and the `And` / `Or` closure
//! directly.
//!
//! ## Latency / memory budget
//!
//! [`Filter`] is a small recursive enum; the deepest commonly
//! constructed shape (`And(vec![Tenant, LabelIn(k=5),
//! PropertyEq])`) occupies ~120 B on the stack. Pattern matching
//! on the discriminant is a single-branch O(1) operation;
//! benched at < 1 % of the surrounding filtered-search cost in
//! `benches/filter_dispatch.rs`.

use arcgraph_core::{LabelId, StringId, TenantId};

// ─── Property types ──────────────────────────────────────────────

/// Interned property-name id.
///
/// Aliased from [`arcgraph_core::StringId`] so the filter
/// language reads naturally (`Filter::PropertyEq(prop_key,
/// value)`) without callers having to remember which crate owns
/// the interning table. The underlying type is the same as
/// `arcgraph_index::PropertyValue`'s key field, so when Slice
/// F.5 wires the filter into the secondary B-tree both sides
/// speak the same key type.
pub type PropertyKey = StringId;

/// Property-value variants supported by the v1.0 filter.
///
/// Mirrors the secondary B-tree's `arcgraph_index::PropertyValue`
/// variant set deliberately — when Slice F.5 wires filter
/// dispatch into the secondary index, the same variants flow
/// through both sides without a translation layer. Bytes / bool /
/// i32 / f64 widening is out of scope for v1.0 per ADR-035 §6 +
/// the secondary-index charter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PropertyValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// Interned string-id (categorical / enum property value).
    StringId(StringId),
}

// ─── Filter ──────────────────────────────────────────────────────

/// Boolean filter expression.
///
/// Variants are deliberately the v1.0-minimum set per ADR-035
/// §6 + amendment-03. Range, prefix, and `Not` are v1.1 follow-
/// ups; the [`Filter::Or`] + [`Filter::And`] combinators are
/// expressive enough for the boolean closure of the v1.0
/// variants over the v1.0 payload schema.
///
/// Empty `And(vec![])` is the **always-true** identity (every
/// payload satisfies an empty conjunction); empty `Or(vec![])`
/// is the **always-false** identity. Both follow from the
/// standard logical definitions and the boundary tests pin both
/// edges.
///
/// [`Filter::Any`] and [`Filter::And(vec![])`] are
/// behaviorally equivalent (both accept every payload), but
/// [`Filter::Any`] is the idiomatic constructor for "no filter"
/// and the F.3 fast path dispatches on it explicitly.
/// [`Filter::LabelEq`] is the F.3 single-label fast-path
/// special case; [`Filter::LabelIn`] covers F.2's broader
/// multi-label set membership case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Filter {
    /// Universal filter — every payload matches. F.3 dispatches
    /// to its unfiltered search path; F.2 short-circuits the
    /// match check.
    Any,
    /// Tenant-scoped filter — vector's payload tenant matches.
    /// At v1.0 in single-tenant arenas this is the universal
    /// short-circuit per ADR-011 (the arena selection IS the
    /// tenant filter); the predicate is structural rather than
    /// evaluated.
    Tenant(TenantId),
    /// Single-label equality. F.3 hot path via per-label entry-
    /// point cache (Gollapudi et al. WWW 2023 §5).
    LabelEq(LabelId),
    /// Label set-membership — at least one of `payload.labels`
    /// is in the listed set.
    LabelIn(Vec<LabelId>),
    /// Property equality — `payload.properties[key] == value`.
    PropertyEq(PropertyKey, PropertyValue),
    /// Conjunction — every child filter must pass.
    /// `And(vec![])` ≡ always-true.
    And(Vec<Filter>),
    /// Disjunction — at least one child filter must pass.
    /// `Or(vec![])` ≡ always-false.
    Or(Vec<Filter>),
}

impl Filter {
    /// Universal filter — every payload matches.
    #[inline]
    #[must_use]
    pub const fn any() -> Self {
        Self::Any
    }

    /// Tenant-scoped filter.
    #[inline]
    #[must_use]
    pub const fn tenant(t: TenantId) -> Self {
        Self::Tenant(t)
    }

    /// Single-label equality. Accepts any type that converts
    /// into [`LabelId`] (notably `u32` via `From<u32>`); this
    /// keeps the F.3 callsite ergonomics identical to the
    /// pre-amendment-03 `DiskAnnFilter::label_eq(u32)` while
    /// unifying the underlying type.
    #[inline]
    #[must_use]
    pub fn label_eq(label: impl Into<LabelId>) -> Self {
        Self::LabelEq(label.into())
    }

    /// Label set-membership. Accepts any iterator yielding
    /// types that convert into [`LabelId`].
    #[inline]
    #[must_use]
    pub fn label_in<L: Into<LabelId>>(labels: impl IntoIterator<Item = L>) -> Self {
        Self::LabelIn(labels.into_iter().map(Into::into).collect())
    }

    /// Property equality.
    #[inline]
    #[must_use]
    pub const fn property_eq(key: PropertyKey, value: PropertyValue) -> Self {
        Self::PropertyEq(key, value)
    }

    /// Conjunction. `Filter::and(std::iter::empty())` is the
    /// always-true identity.
    #[inline]
    #[must_use]
    pub fn and(children: impl IntoIterator<Item = Filter>) -> Self {
        Self::And(children.into_iter().collect())
    }

    /// Disjunction. `Filter::or(std::iter::empty())` is the
    /// always-false identity.
    #[inline]
    #[must_use]
    pub fn or(children: impl IntoIterator<Item = Filter>) -> Self {
        Self::Or(children.into_iter().collect())
    }

    /// Whether this filter is the universal [`Filter::Any`]
    /// variant. F.3's filtered_search dispatches to its
    /// unfiltered fast path when true.
    #[inline]
    #[must_use]
    pub const fn is_any(&self) -> bool {
        matches!(self, Self::Any)
    }

    /// Required label, if the filter is the
    /// [`Filter::LabelEq`] single-label variant.
    ///
    /// Returns `Some(label)` only for `LabelEq(_)`; every other
    /// variant (including [`Filter::LabelIn`]) returns `None`.
    /// F.3's filtered_search consults the per-label entry-point
    /// cache via this hook for its hot path.
    #[inline]
    #[must_use]
    pub const fn required_label(&self) -> Option<LabelId> {
        match self {
            Self::LabelEq(l) => Some(*l),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Constructors ────────────────────────────────────────

    #[test]
    fn any_constructor_yields_any_variant() {
        assert!(matches!(Filter::any(), Filter::Any));
    }

    #[test]
    fn tenant_constructor_carries_tenant_id() {
        let t = TenantId::new(7);
        assert_eq!(Filter::tenant(t), Filter::Tenant(t));
    }

    #[test]
    fn label_eq_accepts_u32_via_into() {
        let f = Filter::label_eq(42_u32);
        assert_eq!(f, Filter::LabelEq(LabelId::new(42)));
    }

    #[test]
    fn label_eq_accepts_label_id_directly() {
        let f = Filter::label_eq(LabelId::new(7));
        assert_eq!(f, Filter::LabelEq(LabelId::new(7)));
    }

    #[test]
    fn label_in_accepts_u32_iterator() {
        let f = Filter::label_in([1u32, 2, 3]);
        assert_eq!(
            f,
            Filter::LabelIn(vec![LabelId::new(1), LabelId::new(2), LabelId::new(3)])
        );
    }

    #[test]
    fn label_in_accepts_label_id_iterator() {
        let f = Filter::label_in(vec![LabelId::new(1), LabelId::new(2)]);
        assert_eq!(f, Filter::LabelIn(vec![LabelId::new(1), LabelId::new(2)]));
    }

    #[test]
    fn property_eq_constructor_carries_args() {
        let f = Filter::property_eq(StringId::new(10), PropertyValue::U32(42));
        assert_eq!(
            f,
            Filter::PropertyEq(StringId::new(10), PropertyValue::U32(42))
        );
    }

    #[test]
    fn and_constructor_collects_children() {
        let f = Filter::and([Filter::any(), Filter::tenant(TenantId::DEFAULT)]);
        assert_eq!(
            f,
            Filter::And(vec![Filter::Any, Filter::Tenant(TenantId::DEFAULT)])
        );
    }

    #[test]
    fn or_constructor_collects_children() {
        let f = Filter::or(std::iter::empty());
        assert_eq!(f, Filter::Or(vec![]));
    }

    // ─── is_any / required_label ────────────────────────────

    #[test]
    fn is_any_true_only_for_any_variant() {
        assert!(Filter::Any.is_any());
        assert!(!Filter::Tenant(TenantId::DEFAULT).is_any());
        assert!(!Filter::LabelEq(LabelId::new(1)).is_any());
        assert!(!Filter::LabelIn(vec![LabelId::new(1)]).is_any());
        assert!(!Filter::And(vec![]).is_any()); // empty And ≡ true logically, NOT structurally Any
        assert!(!Filter::Or(vec![]).is_any());
    }

    #[test]
    fn required_label_some_only_for_label_eq() {
        assert_eq!(Filter::Any.required_label(), None);
        assert_eq!(Filter::Tenant(TenantId::DEFAULT).required_label(), None);
        assert_eq!(
            Filter::LabelEq(LabelId::new(7)).required_label(),
            Some(LabelId::new(7))
        );
        assert_eq!(
            Filter::LabelIn(vec![LabelId::new(1)]).required_label(),
            None
        );
        assert_eq!(Filter::And(vec![]).required_label(), None);
        assert_eq!(Filter::Or(vec![]).required_label(), None);
    }
}
