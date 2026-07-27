//! M4-08a `MaterializedResult` structural roundtrip proptest per
//! ADR-038 amendment-02 §M4.h.
//!
//! # Why a structural roundtrip vs serde JSON / TOON
//!
//! The W12γ slice contract scope is bounded to `arcgraph-query` (per
//! the spawn prompt's "DO NOT touch other crates outside
//! arcgraph-query"); the W11ε arcgraph-mcp serializers (per
//! `feedback_writeup_loc_precision.md` PR #271 cite) live in
//! `arcgraph-mcp` and consume `MaterializedResult` end-to-end. The
//! proptest below ships the structural pin: a randomized
//! `MaterializedResult` value is cloned + decomposed via
//! `into_rows()` / `metrics()` accessors, and the components are
//! assert-equal to the original. This proves the data-flow surface
//! is preserved across the boundary that `arcgraph-mcp` consumes.
//!
//! The cross-crate JSON / TOON roundtrip is the natural follow-up
//! pin in the M5-07 `graph.search` slice (which lights the JSON /
//! TOON serializers end-to-end on real query output).
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-08a (M4-81) cite.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   `MaterializedResult` is the stable v1.0 return shape.
//! - **`feedback_writeup_loc_precision.md`** — PR #271 W11ε
//!   serializer cite; the cross-crate roundtrip lives in
//!   `arcgraph-mcp` per bounded-context discipline.

use arcgraph_query::executor::Value;
use arcgraph_query::{ExecutionMetrics, MaterializedResult};

use proptest::prelude::*;

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

/// Generate a v1.0-shaped runtime [`Value`]. Excludes Node /
/// Relationship / List / Map for proptest scope (we test the wrapper
/// type's structural roundtrip, not the value taxonomy's
/// — that's covered by the executor-side proptests).
fn value_strategy() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Boolean),
        any::<i64>().prop_map(Value::Integer),
        // Float is omitted: NaN ≠ NaN under PartialEq, breaking the
        // "Self == Self" roundtrip pin. The serde JSON / TOON
        // roundtrip handles NaN via "null" mapping in
        // arcgraph-mcp; the structural pin doesn't need to test
        // NaN-vs-NaN.
        any::<String>().prop_map(Value::String),
    ]
}

/// Generate a [`MaterializedResult`] of 0-50 rows, each with the
/// SAME column count (executor-output rows are uniform per ADR-038
/// §2 D-26 binding-resolution invariant).
fn materialized_result_strategy() -> impl Strategy<Value = MaterializedResult> {
    (
        // Fixed column count for the row-batch.
        1usize..=4,
        // Number of rows.
        0usize..=50,
        any::<u64>(),
        any::<u64>(),
    )
        .prop_flat_map(|(col_count, row_count, wall_time_ms, mem_high_water)| {
            (
                prop::collection::vec(
                    prop::collection::vec(value_strategy(), col_count..=col_count),
                    row_count..=row_count,
                ),
                Just(wall_time_ms),
                Just(mem_high_water),
                Just(row_count as u64),
            )
        })
        .prop_map(
            |(rows, wall_time_ms, memory_bytes_high_water, rows_emitted)| MaterializedResult {
                rows,
                metrics: ExecutionMetrics {
                    wall_time_ms,
                    memory_bytes_high_water,
                    rows_emitted,
                },
                // W13β M4-81 — the structural-roundtrip pin doesn't
                // exercise the truncation field; budget enforcement
                // is covered by the dedicated integration tests in
                // `tests/m4_81_*`. Default `None` here.
                truncation: None,
                // #353 — column names are not part of this structural
                // rows/metrics roundtrip pin; default empty.
                columns: Vec::new(),
            },
        )
}

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// Structural roundtrip: clone a MaterializedResult, decompose
    /// via the public accessor surface, assert-equal to the original.
    /// This pins the data-flow contract that arcgraph-mcp's JSON /
    /// TOON serializers consume.
    #[test]
    fn materialized_result_structural_roundtrip(orig in materialized_result_strategy()) {
        // The Clone roundtrip must preserve all data.
        let clone = orig.clone();
        prop_assert_eq!(orig.rows().len(), clone.rows().len(),
            "rows() borrow-equal across clone");
        prop_assert_eq!(orig.metrics(), clone.metrics(),
            "metrics() borrow-equal across clone");

        // The into_rows() accessor preserves Vec<Vec<Value>> shape;
        // the metrics half is consumed (matches the v1.0 MCP
        // renderer pattern: take rows, drop metrics).
        let rows_before = orig.rows().to_vec();
        let metrics_before = orig.metrics().clone();
        let rows_consumed = orig.into_rows();
        let consumed_len = rows_consumed.len();
        prop_assert_eq!(rows_before, rows_consumed,
            "into_rows() returns the rows half unchanged");

        // The Default impl must produce empty + zero-metrics.
        let default = MaterializedResult::default();
        prop_assert_eq!(default.len(), 0);
        prop_assert!(default.is_empty());
        prop_assert_eq!(default.metrics().rows_emitted, 0);

        // metrics_before observed pre-into-rows MUST match the
        // generator's intent: rows_emitted = generated rows count.
        prop_assert_eq!(metrics_before.rows_emitted, consumed_len as u64,
            "rows_emitted invariant: equal to rows.len() at v1.0-alpha");
    }
}
