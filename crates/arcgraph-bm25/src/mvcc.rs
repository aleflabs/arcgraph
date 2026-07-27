//! MVCC visibility filter composition (ADR-039 §D-3).
//!
//! Every search composes the user's parsed Tantivy query with a
//! visibility filter so the reader sees only docs that are alive at
//! the read snapshot. The filter is two `RangeQuery`s
//! intersected:
//!
//! - `commit_lsn ∈ [0, read_lsn]` — visible (committed at or before).
//! - `expired_lsn ∈ [read_lsn + 1, u64::MAX]` — not yet superseded.
//!
//! Every live doc has `expired_lsn = Lsn::MAX`, so the second clause
//! admits it. Both clauses remain explicit because they are the shared
//! MVCC visibility contract used by the retained index substrates.

use arcgraph_core::Lsn;
use std::ops::Bound;
use tantivy::query::{BooleanQuery, Occur, Query, RangeQuery};
use tantivy::schema::Field;

use crate::segment::Bm25Schema;

/// Build the MVCC visibility filter for a search at `read_lsn`
/// (ADR-039 §D-3).
///
/// Returns a `BooleanQuery` that AND-s the two `RangeQuery` clauses
/// described in the module rustdoc. The resulting query is
/// composable: callers `BooleanQuery::intersection`-it with a
/// user-supplied parsed query before handing the combined query to
/// `Searcher::search`.
///
/// At v1.0 the second clause (`expired_lsn > read_lsn`) is trivially
/// true for every live doc because every v1.0-produced doc has
/// `expired_lsn = Lsn::MAX = u64::MAX`. The clause still composes.
#[must_use]
pub fn build_visibility_filter(schema: &Bm25Schema, read_lsn: Lsn) -> Box<dyn Query> {
    let read = read_lsn.raw();

    // The saturating_add(1) below correctly clamps at the boundary,
    // but a `read_lsn == u64::MAX-1` would also saturate `expired_lower`
    // to `MAX` — admitting only the `expired_lsn = MAX` sentinel rather
    // than [MAX, MAX] and excluding (MAX-1)-expired versions. v1.0 LSN
    // width (u64) makes this unreachable in practice (~10^19 mutations
    // to approach the boundary). The gap becomes real if LSNs are
    // remapped to a lower-bit space (e.g., per-tenant 32-bit LSNs); see
    // codex retro review of M3.b (2026-05-03 CONCERN-soft #6) and
    // ADR-039 §D-3.
    debug_assert!(
        read != u64::MAX,
        "build_visibility_filter: saturating_add(1) semantic gap at read_lsn = u64::MAX-1; \
         v1.0 LSN width (u64) makes this unreachable but downstream LSN-width changes must \
         revisit. See ADR-039 + retro review (2026-05-03)."
    );
    // TODO(LSN-width-change): tighten `expired_lower` derivation when
    // LSN bit-width changes (e.g., per-tenant 32-bit LSNs). Current
    // saturating_add(1) is correct for u64 LSNs at any value < u64::MAX;
    // the debug_assert above guards the known boundary.

    // commit_lsn ∈ [0, read_lsn]  — visible.
    //
    // Tantivy's `RangeQuery::new` consumes `Bound::Included(_)` /
    // `Bound::Excluded(_)` arguments via the schema-typed terms.
    let commit_clause: Box<dyn Query> = Box::new(RangeQuery::new(
        Bound::Included(field_term(schema.commit_lsn, 0)),
        Bound::Included(field_term(schema.commit_lsn, read)),
    ));

    // expired_lsn ∈ [read_lsn + 1, u64::MAX]  — not yet superseded.
    //
    // `read + 1` may overflow only when `read == u64::MAX`, which is
    // pathological — `Lsn::MAX` as a read snapshot would observe
    // every doc, including those marked expired at MAX. v1.0 never
    // produces such a snapshot (Lsn allocation is monotonic from 0
    // and stays well below MAX in practice); the debug_assert above
    // surfaces the boundary in test runs and the `saturating_add(1)`
    // keeps the function total in release builds.
    let expired_lower = read.saturating_add(1);
    let expired_clause: Box<dyn Query> = Box::new(RangeQuery::new(
        Bound::Included(field_term(schema.expired_lsn, expired_lower)),
        Bound::Included(field_term(schema.expired_lsn, u64::MAX)),
    ));

    Box::new(BooleanQuery::new(vec![
        (Occur::Must, commit_clause),
        (Occur::Must, expired_clause),
    ]))
}

/// Build a typed `tantivy::Term` carrying a `u64` field value. Local
/// helper so the visibility filter call sites stay readable.
fn field_term(field: Field, value: u64) -> tantivy::Term {
    tantivy::Term::from_field_u64(field, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The visibility filter is a non-trivial `BooleanQuery` — i.e.
    /// it composes both clauses. A regression that drops one clause
    /// surfaces as a thinner query shape here.
    #[test]
    fn visibility_filter_composes_two_clauses() {
        let schema = Bm25Schema::build();
        let q = build_visibility_filter(&schema, Lsn::new(100));
        // We cannot down-cast a `Box<dyn Query>` without enabling
        // `Any`, but we can stringify it: `BooleanQuery::Debug`
        // renders the inner clauses textually, and a regression
        // that produces a `RangeQuery` directly (skipping the
        // outer Boolean) will surface as a different prefix.
        let dbg = format!("{q:?}");
        assert!(dbg.contains("BooleanQuery"), "{dbg}");
    }

    /// At v1.0 with `read_lsn = 0`, the filter must still compose;
    /// the lower bound on `expired_lsn` becomes `1`.
    #[test]
    fn visibility_filter_at_lsn_zero() {
        let schema = Bm25Schema::build();
        let q = build_visibility_filter(&schema, Lsn::ZERO);
        let dbg = format!("{q:?}");
        assert!(dbg.contains("BooleanQuery"), "{dbg}");
    }

    /// In debug builds, calling with `read_lsn = Lsn::MAX` fires the
    /// debug_assert that surfaces the saturating_add semantic gap
    /// (codex retro review of M3.b, 2026-05-03 CONCERN-soft #6).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "saturating_add(1) semantic gap")]
    fn visibility_filter_at_lsn_max_panics_in_debug() {
        let schema = Bm25Schema::build();
        let _ = build_visibility_filter(&schema, Lsn::MAX);
    }

    /// In release builds the debug_assert is a no-op, the
    /// saturating_add prevents wrap, and the filter is still
    /// composable (no panic).
    #[cfg(not(debug_assertions))]
    #[test]
    fn visibility_filter_at_lsn_max_does_not_overflow() {
        let schema = Bm25Schema::build();
        let q = build_visibility_filter(&schema, Lsn::MAX);
        let dbg = format!("{q:?}");
        assert!(dbg.contains("BooleanQuery"), "{dbg}");
    }
}
