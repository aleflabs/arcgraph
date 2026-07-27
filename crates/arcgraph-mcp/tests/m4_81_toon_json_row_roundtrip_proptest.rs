//! W13β M4-81 proptest — random `Row` (`Vec<Value>`) round-trips
//! through both the JSON pivot AND the TOON serializer per ADR-038
//! amendment-02 §M4.h "TOON + JSON serialization for MCP".
//!
//! # Why this lives in arcgraph-mcp (not arcgraph-query)
//!
//! The bounded-context rule assigns TOON to `arcgraph-mcp`; the JSON pivot
//! lives on arcgraph-query's `Value::to_json_value` /
//! `Value::try_from_json_value` (in
//! `crates/arcgraph-query/src/executor/value.rs`); the TOON encoder /
//! decoder live here in arcgraph-mcp's serializers. The proptest
//! exercises both halves end-to-end:
//!
//!   Row → JsonValue array → TOON text → JsonValue array → Row
//!
//! and asserts byte-equal round-trip across the full pipeline.
//!
//! # Strategy
//!
//! Random `Vec<Value>` rows are generated from a v1.0-shaped strategy
//! that excludes:
//! - Non-finite floats (NaN / ±Inf coerce to JSON null per the W11ε
//!   TOON encoder convention; round-trip is lossy by design).
//! - `Value::Node` / `Value::Relationship` whose property-key strings
//!   contain characters outside the W11ε `is_valid_unquoted_key`
//!   regex `^[A-Za-z_][A-Za-z0-9_.]*$`. Quoted keys aren't supported
//!   in the W11ε serializer slice; the encoder surfaces
//!   `ToonError::Unencodable` per its module docs.
//! - Strings containing un-escapable control chars per spec §7.1
//!   (the W11ε encoder's five escape sequences cover `\\, \", \n,
//!   \r, \t`; other control chars surface `Unencodable`).
//!
//! The exclusion list mirrors the existing `tests/toon_proptest.rs`
//! envelope (which exercises the TOON serializer directly on
//! arbitrary serde_json::Value trees). The W13β proptest inherits
//! that envelope and stacks the Value ↔ JsonValue bridge on top.
//!
//! # Cases
//!
//! `PROPTEST_CASES=10000` per the spawn prompt's exact requirement.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-81 cite.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   `MaterializedResult` is the stable v1.0 return shape; the
//!   TOON / JSON serializers consume `MaterializedResult::rows`.
//! - **`crates/arcgraph-mcp/tests/toon_proptest.rs`** — sister
//!   proptest exercising serde_json::Value directly; this proptest
//!   stacks the Value-bridge on top.

use arcgraph_mcp::serializers::{from_toon, to_toon};
use arcgraph_query::executor::Value;

use proptest::prelude::*;

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

/// Generate a finite f64 EXCLUDING integer-valued floats.
///
/// # Why exclude integer-valued floats
///
/// The W11ε TOON encoder canonicalizes integer-valued floats (e.g.,
/// `1.0` → `"1"`) per spec §2: this is the "shortest decimal /
/// integer-valued-float promotion" rule documented in
/// `crates/arcgraph-mcp/src/serializers/toon.rs::encode_number`. On
/// decode, the integer-style token reads back as
/// `serde_json::Number::from(i64)`, so `Value::Float(1.0)` round-
/// trips as `Value::Integer(1)`. Per the W11ε precedent in
/// `crates/arcgraph-mcp/tests/common/mod.rs` ("Float roundtripping
/// is exercised in the unit tests next to each serializer instead"),
/// the proptest envelope excludes the lossy edge so the test signal
/// stays on row-shape preservation, not float-canonical-form.
///
/// Non-integer finite floats (`1.5`, `-0.001`, etc.) round-trip
/// cleanly through both JSON and TOON.
///
/// # Construction
///
/// Constructive (NOT filter-based): `any::<f64>` is dominated by
/// special values (0.0, MIN, MAX, ±epsilon) that fail the
/// non-integer filter, exhausting the proptest reject budget. This
/// strategy composes an integer base (`-1_000_000..1_000_000`) with
/// a fractional offset on `(0.001, 0.999)` to guarantee
/// `f.fract() != 0.0`. The range covers enough magnitude variation
/// to exercise the IEEE-754 mantissa + exponent paths without
/// hitting the integer-coercion edge.
fn finite_non_integer_f64() -> impl Strategy<Value = f64> {
    (-1_000_000_i64..1_000_000_i64, 0.001_f64..0.999_f64).prop_map(|(i, frac)| (i as f64) + frac)
}

/// Generate a string that is safe for both TOON quoted-string
/// encoding AND TOON unquoted-string decoding. The W11ε encoder
/// surfaces `Unencodable` for control chars outside `\n \r \t`; the
/// proptest envelope mirrors that exclusion.
fn safe_string() -> impl Strategy<Value = String> {
    // ASCII-printable + the three escape-able control chars; matches
    // the existing toon_proptest's string strategy for compatibility.
    "[\\x20-\\x7E\\n\\r\\t]{0,40}".prop_map(String::from)
}

/// Generate a v1.0-shaped scalar [`Value`].
fn scalar_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Boolean),
        any::<i64>().prop_map(Value::Integer),
        finite_non_integer_f64().prop_map(Value::Float),
        safe_string().prop_map(Value::String),
    ]
}

/// Generate a heterogeneous list of v1.0 scalars. The proptest
/// envelope excludes `Value::Node` / `Value::Relationship` from the
/// list elements because those project to JSON objects whose keys
/// (`id`, `label`, etc.) must satisfy the W11ε `is_valid_unquoted_key`
/// regex — they always do, but the envelope keeps cells purely
/// scalar to mirror the M5-07 `graph.search` v1.0 result shape (uniform
/// rows of primitive cells).
///
/// # Composite-variant coverage (W13β fix-up N-4)
///
/// Per PR #287 review NIT-4: `Value::Node` / `Value::Relationship`
/// ARE exercised but NOT at the 10K-case proptest density of this
/// file. They are pinned at unit-test density at:
/// - `crates/arcgraph-query/src/executor/value.rs::tests::*` (Value
///   ↔ JsonValue bridge — Node/Relationship round-trip cases).
/// - `crates/arcgraph-mcp/tests/m4_81_materialize_serializer_unit.rs::tests::*`
///   (TOON serialization — Node/Relationship cell shapes).
///
/// Promoting Node/Relationship to proptest density is forward-deferred
/// to a v1.1 slice that adds a property-key generator constrained to
/// the W11ε `is_valid_unquoted_key` charset; v1.0-alpha density on
/// these two surfaces is the unit-test count.
fn list_value() -> impl Strategy<Value = Value> {
    prop::collection::vec(scalar_value(), 0..=8).prop_map(Value::List)
}

/// Generate a cell (scalar OR list).
fn cell_value() -> impl Strategy<Value = Value> {
    prop_oneof![scalar_value(), list_value()]
}

/// Generate a uniform-shape row (all rows in a result share column
/// count per ADR-038 §2 D-26).
fn row_strategy(col_count: usize) -> impl Strategy<Value = Vec<Value>> {
    prop::collection::vec(cell_value(), col_count..=col_count)
}

/// Generate a multi-row uniform-shape batch.
fn batch_strategy() -> impl Strategy<Value = Vec<Vec<Value>>> {
    (1usize..=4, 0usize..=20).prop_flat_map(|(col_count, row_count)| {
        prop::collection::vec(row_strategy(col_count), row_count..=row_count)
    })
}

// ---------------------------------------------------------------------
// Bridge helpers
// ---------------------------------------------------------------------

/// Project a row [`Vec<Value>`] into a [`serde_json::Value::Array`].
fn row_to_json(row: &[Value]) -> serde_json::Value {
    serde_json::Value::Array(row.iter().map(Value::to_json_value).collect())
}

/// Project a [`serde_json::Value::Array`] back into a row
/// [`Vec<Value>`].
fn row_from_json(v: &serde_json::Value) -> Vec<Value> {
    let serde_json::Value::Array(cells) = v else {
        panic!("expected JsonValue::Array")
    };
    cells
        .iter()
        .map(|c| Value::try_from_json_value(c).expect("decode"))
        .collect()
}

/// Project a batch [`Vec<Vec<Value>>`] into a JSON array of arrays.
fn batch_to_json(batch: &[Vec<Value>]) -> serde_json::Value {
    serde_json::Value::Array(batch.iter().map(|r| row_to_json(r)).collect())
}

/// Project a JSON array of arrays back into a [`Vec<Vec<Value>>`].
fn batch_from_json(v: &serde_json::Value) -> Vec<Vec<Value>> {
    let serde_json::Value::Array(rows) = v else {
        panic!("expected JsonValue::Array")
    };
    rows.iter().map(row_from_json).collect()
}

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        // Per the spawn prompt's exact requirement.
        cases: 10_000,
        ..ProptestConfig::default()
    })]

    /// JSON pivot round-trip — random batch → JSON → batch is byte-
    /// equal modulo the lossy edge for non-finite floats (excluded
    /// from the strategy).
    #[test]
    fn json_pivot_round_trip_preserves_row_shape(batch in batch_strategy()) {
        let json = batch_to_json(&batch);
        let back = batch_from_json(&json);
        prop_assert_eq!(back, batch, "JSON pivot must round-trip cleanly");
    }

    /// TOON pivot round-trip — random batch → JSON → TOON → JSON → batch.
    /// Stacks the W11ε TOON encoder/decoder on top of the JSON pivot.
    /// Per the spawn prompt's exact "TOON ↔ JSON ↔ Row" requirement.
    #[test]
    fn toon_pivot_round_trip_preserves_row_shape(batch in batch_strategy()) {
        let json = batch_to_json(&batch);
        let toon = to_toon(&json).expect("TOON encode");
        let back_json: serde_json::Value = from_toon(&toon).expect("TOON decode");
        let back = batch_from_json(&back_json);
        prop_assert_eq!(back, batch, "TOON pivot must round-trip cleanly");
    }
}
