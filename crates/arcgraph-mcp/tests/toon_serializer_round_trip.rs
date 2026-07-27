//! W26-γ-3 / ADR-136 — TOON serializer round-trip across MCP tool
//! response shapes.
//!
//! # Surface
//!
//! [`arcgraph_mcp::serializers::to_toon`] +
//! [`arcgraph_mcp::serializers::from_toon`] +
//! [`arcgraph_mcp::serializers::render_response`].
//!
//! # Coverage
//!
//! The existing `crates/arcgraph-mcp/tests/toon_proptest.rs` covers
//! the random-`Value`-shape oracle. This suite adds **per-tool
//! response-shape pinning** — every public MCP tool returns a structured response;
//! TOON ser/deser MUST round-trip that response shape.
//!
//! # Tool response shapes pinned
//!
//! 1. `graph.schema` — nested label/type map (YAML class).
//! 2. `graph.inspect` — single-row entity card.
//! 3. `graph.explore` — uniform tabular row set (TOON token-savings class).
//! 4. `graph.search` — top-K ranked hit array (TOON tabular).
//! 5. `graph.ingest` — ingest report (counts + ack).
//! 6. `graph.raw_query` — generic row set (TOON tabular).
//!
//! For each, we pin: (a) `to_toon(value)` succeeds; (b)
//! `from_toon::<Value>(...)` round-trips; (c) the round-trip output
//! is structurally equal to the input.

use arcgraph_mcp::serializers::{from_toon, to_toon};
use serde_json::{Value, json};

/// Round-trip a JSON value through TOON and assert structural equality.
fn assert_toon_roundtrip(name: &str, v: Value) {
    let encoded = to_toon(&v).unwrap_or_else(|e| panic!("{name}: to_toon failed: {e:?}"));
    let decoded: Value = from_toon(&encoded)
        .unwrap_or_else(|e| panic!("{name}: from_toon failed on output:\n--- TOON ---\n{encoded}\n--- end ---\nerror: {e:?}"));
    assert_eq!(
        v,
        decoded,
        "{name}: TOON round-trip diverged\n--- TOON ---\n{encoded}\n--- expected ---\n{}\n--- actual ---\n{}",
        serde_json::to_string_pretty(&v).unwrap(),
        serde_json::to_string_pretty(&decoded).unwrap()
    );
}

// =====================================================================
// 1. graph.schema — nested label/type map
// =====================================================================

#[test]
fn graph_schema_shape_round_trip() {
    let response = json!({
        "labels": {
            "Person": {
                "props": [
                    {"name": "age", "type": "Int"},
                    {"name": "name", "type": "Str"}
                ]
            },
            "Company": {
                "props": [
                    {"name": "employees", "type": "Int"}
                ]
            }
        },
        "rel_types": ["KNOWS", "WORKS_AT", "LIKES"]
    });
    assert_toon_roundtrip("graph.schema", response);
}

// =====================================================================
// 2. graph.inspect — single-row entity card
// =====================================================================

#[test]
fn graph_inspect_shape_round_trip() {
    let response = json!({
        "node_id": 1234,
        "label": "Person",
        "properties": {
            "name": "Alice",
            "age": 30,
            "active": true,
            "salary": 75000.50
        },
        "out_degree": 5,
        "in_degree": 3
    });
    assert_toon_roundtrip("graph.inspect", response);
}

// =====================================================================
// 3. graph.explore — uniform tabular row set
// =====================================================================

#[test]
fn graph_explore_uniform_table_round_trip() {
    let response = json!({
        "rows": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Carol", "age": 40},
            {"id": 4, "name": "Dave", "age": 35},
            {"id": 5, "name": "Eve", "age": 28}
        ],
        "total": 5
    });
    assert_toon_roundtrip("graph.explore", response);
}

#[test]
fn graph_explore_empty_table_round_trip() {
    let response = json!({"rows": [], "total": 0});
    assert_toon_roundtrip("graph.explore-empty", response);
}

// =====================================================================
// 4. graph.search — top-K ranked hit array
// =====================================================================

#[test]
fn graph_search_ranked_hits_round_trip() {
    let response = json!({
        "hits": [
            {"id": 100, "score": 0.95, "label": "Post"},
            {"id": 101, "score": 0.89, "label": "Post"},
            {"id": 102, "score": 0.82, "label": "Post"}
        ],
        "k": 3
    });
    assert_toon_roundtrip("graph.search", response);
}

// =====================================================================
// 5. graph.ingest — ingest report
// =====================================================================

#[test]
fn graph_ingest_report_round_trip() {
    let response = json!({
        "tenant": "demo",
        "nodes_created": 1000,
        "rels_created": 5432,
        "duration_ms": 120,
        "lsn": 8675309
    });
    assert_toon_roundtrip("graph.ingest", response);
}

// =====================================================================
// 6. graph.raw_query — generic row set
// =====================================================================

#[test]
fn graph_raw_query_generic_round_trip() {
    let response = json!({
        "columns": ["name", "age"],
        "rows": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ],
        "row_count": 2
    });
    assert_toon_roundtrip("graph.raw_query", response);
}

// =====================================================================
// 7. Adversarial — mixed-type rows in a "tabular" shape
// =====================================================================

#[test]
fn mixed_type_rows_round_trip_via_block_list() {
    // Per toon.rs §"Tabular" — when rows are NOT structurally
    // uniform, the encoder falls back to block-list form. Either
    // shape must round-trip.
    let response = json!({
        "rows": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob", "extra": "info"},
            {"id": 3, "name": "Carol"}
        ]
    });
    assert_toon_roundtrip("mixed-rows", response);
}

#[test]
fn deep_nesting_round_trip() {
    let response = json!({
        "a": {
            "b": {
                "c": {
                    "d": [{"e": 1}, {"e": 2}],
                    "f": "leaf"
                }
            }
        }
    });
    assert_toon_roundtrip("deep-nest", response);
}

#[test]
fn null_and_bool_values_round_trip() {
    let response = json!({
        "active": true,
        "deleted": false,
        "deleted_at": Value::Null,
        "items": [null, true, false, null]
    });
    assert_toon_roundtrip("null-bool", response);
}

#[test]
fn unicode_string_values_round_trip() {
    let response = json!({
        "name": "你好世界",
        "emoji": "🚀",
        "mixed": "Hello, 世界! 👋"
    });
    assert_toon_roundtrip("unicode", response);
}

#[test]
fn large_uniform_table_round_trip() {
    // 100 rows of uniform shape — the TOON token-savings sweet-spot.
    // Per toon.rs §"Encoding strategy": integer-valued floats normalize
    // to integers on encode. We avoid `i * 1.5` (which produces some
    // integer-valued floats like 0.0, 3.0, 6.0, …) and use a stride
    // that guarantees no integer-valued float.
    let rows: Vec<Value> = (0..100)
        .map(|i| {
            // i * 1.25 produces 0.0, 1.25, 2.5, 3.75, 5.0, …
            // Still has integer-valued floats at multiples of 4.
            // Add an offset so no value is integer-valued:
            // (i * 1.0 + 0.5) → 0.5, 1.5, 2.5, …
            let score = i as f64 + 0.5;
            json!({"id": i, "name": format!("User{i}"), "score": score})
        })
        .collect();
    let response = json!({"rows": rows, "total": 100});
    assert_toon_roundtrip("large-table-100", response);
}

#[test]
fn integer_valued_floats_normalize_to_int_one_way() {
    // Documented one-way normalization per toon.rs:
    // `2.0` encoded as `2` decodes back to `2` (Number::Int), not
    // `2.0` (Number::Float). Pin the asymmetry so future encoder
    // changes that "fix" it (a wire-incompatible bump) are caught.
    let response = json!({"score": 3.0});
    let encoded = to_toon(&response).expect("encode");
    let decoded: Value = from_toon(&encoded).expect("decode");
    // The encoded number ends up as Number(3), not Number(3.0).
    assert_eq!(decoded, json!({"score": 3}));
}

#[test]
fn integer_boundary_values_round_trip() {
    let response = json!({
        "max": i64::MAX,
        "min": i64::MIN,
        "zero": 0,
        "positive": 12345,
        "negative": -12345
    });
    assert_toon_roundtrip("int-boundary", response);
}

#[test]
fn empty_string_round_trip_via_quote() {
    // Per toon.rs §"Encoding strategy": empty string needs quoting.
    let response = json!({"empty": "", "non_empty": "x"});
    assert_toon_roundtrip("empty-string", response);
}

#[test]
fn numeric_string_values_round_trip_via_quote() {
    // Per toon.rs §"Encoding strategy": numeric-looking strings need quoting.
    let response = json!({"id_string": "42", "phone": "+1-555-1234"});
    assert_toon_roundtrip("numeric-string", response);
}
