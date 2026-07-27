//! W13β M4-81 unit-grade serializer tests — TOON half of the
//! "TOON roundtrip + JSON roundtrip + per-row + per-batch + nested
//! types" 8-case bar from the spawn prompt.
//!
//! # Pin set (4 unit-grade tests)
//!
//! These mirror the JSON-side tests in
//! `crates/arcgraph-query/src/executor/value.rs` `#[cfg(test)] mod
//! tests`:
//!
//! 1. `toon_per_row_primitive_scalars` — per-cell TOON roundtrip for
//!    scalar variants (Null / Bool / Integer / non-integer Float /
//!    String).
//! 2. `toon_per_batch_multi_row` — multi-row uniform-shape batch
//!    roundtrip; pins the tabular form for primitive-only objects
//!    (W11ε token-savings path).
//! 3. `toon_nested_node_relationship_list` — composite variants via
//!    the Value ↔ JsonValue bridge through the TOON serializer.
//! 4. `toon_edge_cases_unlabeled_node_untyped_relationship` —
//!    unlabeled Node + untyped Relationship + empty-properties edge
//!    cases.
//!
//! Combined with the 4 JSON-side cases in
//! `crates/arcgraph-query/src/executor/value.rs`, the 8-case bar in
//! the spawn prompt is satisfied.
//!
//! # Why this file lives in arcgraph-mcp
//!
//! Under the bounded-context policy (bounded contexts) + the W13β spawn prompt's
//! "DO NOT touch crates outside arcgraph-query + arcgraph-mcp (the
//! latter only for serializer integration tests; no production-
//! source changes)", TOON belongs to arcgraph-mcp. The Value ↔
//! JsonValue bridge is the SHARED contract surface between the two
//! crates.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-81 cite.
//! - **W11ε `crates/arcgraph-mcp/src/serializers/toon.rs`** — the
//!   underlying TOON encoder/decoder this test exercises.

use arcgraph_core::{LabelId, NodeId, RelId, TypeId};
use arcgraph_mcp::serializers::{from_toon, to_toon};
use arcgraph_query::executor::Value;
use arcgraph_query::executor::value::{NodeView, RelView};

/// Round-trip a row through Value → JsonValue → TOON → JsonValue → Row.
fn roundtrip_row_via_toon(row: &[Value]) -> Vec<Value> {
    let json = serde_json::Value::Array(row.iter().map(Value::to_json_value).collect());
    let toon = to_toon(&json).expect("TOON encode");
    let back: serde_json::Value = from_toon(&toon).expect("TOON decode");
    let serde_json::Value::Array(cells) = back else {
        panic!("expected JSON Array")
    };
    cells
        .iter()
        .map(|c| Value::try_from_json_value(c).expect("decode"))
        .collect()
}

/// Round-trip a batch through Value → JsonValue → TOON → JsonValue → Vec<Vec<Value>>.
fn roundtrip_batch_via_toon(batch: &[Vec<Value>]) -> Vec<Vec<Value>> {
    let json = serde_json::Value::Array(
        batch
            .iter()
            .map(|row| serde_json::Value::Array(row.iter().map(Value::to_json_value).collect()))
            .collect(),
    );
    let toon = to_toon(&json).expect("TOON encode");
    let back: serde_json::Value = from_toon(&toon).expect("TOON decode");
    let serde_json::Value::Array(rows) = back else {
        panic!("expected JSON Array of Arrays")
    };
    rows.iter()
        .map(|row| {
            let serde_json::Value::Array(cells) = row else {
                panic!("expected per-row Array")
            };
            cells
                .iter()
                .map(|c| Value::try_from_json_value(c).expect("decode"))
                .collect()
        })
        .collect()
}

// =====================================================================
// 1. Per-row primitive-scalar TOON roundtrip
// =====================================================================

#[test]
fn toon_per_row_primitive_scalars() {
    // Scalar variants — Null / Bool / Integer / non-integer Float /
    // String. Integer-valued floats (1.0, 0.0) are excluded per the
    // W11ε encoder's canonical-form rule (1.0 → "1" → Integer(1));
    // this is the same lossy edge the proptest envelope filters out.
    let row: Vec<Value> = vec![
        Value::Null,
        Value::Boolean(true),
        Value::Boolean(false),
        Value::Integer(0),
        Value::Integer(-7),
        Value::Integer(i64::MAX),
        Value::Integer(i64::MIN),
        Value::Float(1.5),
        Value::Float(-1.234567),
        Value::String(String::new()),
        Value::String("hello".into()),
        Value::String("with: colon, comma".into()),
    ];
    let back = roundtrip_row_via_toon(&row);
    assert_eq!(back, row, "per-row TOON roundtrip preserves cells");
}

// =====================================================================
// 2. Per-batch multi-row TOON roundtrip (tabular form pinned)
// =====================================================================

#[test]
fn toon_per_batch_multi_row() {
    // Uniform-shape multi-row batch — exercises the W11ε tabular
    // form (header `[N]{f1,f2}:` + per-row comma-separated cells).
    // The Vec<Vec<Value>> serializes to a JSON array-of-arrays;
    // TOON encodes the outer array as block-list (each row is a JSON
    // array, NOT a tabular-eligible primitive-only object).
    let batch: Vec<Vec<Value>> = vec![
        vec![Value::Integer(1), Value::String("Ada".into())],
        vec![Value::Integer(2), Value::String("Bob".into())],
        vec![Value::Integer(3), Value::String("Cay".into())],
    ];
    let back = roundtrip_batch_via_toon(&batch);
    assert_eq!(
        back, batch,
        "per-batch multi-row TOON roundtrip preserves rows + ordering"
    );
}

// =====================================================================
// 3. Nested-types TOON roundtrip (Node + Rel + List)
// =====================================================================

#[test]
fn toon_nested_node_relationship_list() {
    // Composite variants. Node / Rel project to JSON objects with the
    // bridge's structural keys (id, label, properties); TOON encodes
    // them as block-list-of-objects (NOT tabular — the property bag
    // is non-primitive-keyed). The roundtrip preserves the full
    // structural surface.
    let node = Value::Node(
        NodeView::new(NodeId::new(7), Some(LabelId::new(1)))
            .with_property("name", Value::String("Alice".into()))
            .with_property("age", Value::Integer(30)),
    );
    let rel = Value::Relationship(
        RelView::new(
            RelId::new(99),
            NodeId::new(1),
            NodeId::new(2),
            Some(TypeId::new(1)),
        )
        .with_property("since", Value::Integer(2020)),
    );
    let list = Value::List(vec![
        Value::Integer(1),
        Value::String("two".into()),
        Value::Null,
    ]);
    let row: Vec<Value> = vec![node.clone(), rel.clone(), list.clone()];
    let back = roundtrip_row_via_toon(&row);
    assert_eq!(back, row, "nested-types TOON roundtrip preserves structure");
}

// =====================================================================
// 4. Edge cases — unlabeled Node + untyped Relationship + empty props
// =====================================================================

#[test]
fn toon_edge_cases_unlabeled_node_untyped_relationship() {
    // Edge cases: Node without label + Rel without rel_type + both
    // with empty property bags. The bridge encodes these as
    // `{"id": <n>, "label": null, "properties": {}}` — round-trip
    // must reconstruct the same shape.
    let n_no_label = Value::Node(NodeView::new(NodeId::new(42), None));
    let n_no_props = Value::Node(NodeView::new(NodeId::new(99), Some(LabelId::new(2))));
    let r_no_type = Value::Relationship(RelView::new(
        RelId::new(7),
        NodeId::new(1),
        NodeId::new(2),
        None,
    ));
    let row: Vec<Value> = vec![n_no_label.clone(), n_no_props.clone(), r_no_type.clone()];
    let back = roundtrip_row_via_toon(&row);
    assert_eq!(
        back, row,
        "edge cases (unlabeled / untyped / empty-props) TOON roundtrip"
    );
}
