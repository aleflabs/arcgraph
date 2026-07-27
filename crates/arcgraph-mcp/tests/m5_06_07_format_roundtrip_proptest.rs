//! W14β M5-06 + M5-07 — TOON ↔ JSON ↔ Result roundtrip proptest.
//!
//! Per the spawn prompt's "1 proptest" acceptance: prove that the
//! [`Neighborhood`] (M5-06) and [`SearchResult`] (M5-07) wire shapes
//! survive every renderable format. Oracle:
//!
//!   from_json(to_toon(v).decode())   == v
//!   from_yaml(to_yaml(v))            == v
//!   from_json(to_json(v))            == v
//!
//! In practice we encode each type through `crate::tools::render_response`
//! at every [`ResponseFormat`], parse the body back through the
//! sibling [`from_toon`] / [`from_yaml`] / `serde_json::from_str`
//! family, and assert structural equality on the `serde_json::Value`
//! pivot.
//!
//! Why pivot through `Value`: per [`crate::serializers`] doc, each
//! encoder uses `serde_json::Value` as its canonical intermediate
//! representation; the round-trip oracle therefore lives at the
//! `Value` layer where the lattice is shared.

use std::collections::BTreeMap;

use arcgraph_mcp::ResponseFormat;
use arcgraph_mcp::render_response;
use arcgraph_mcp::serializers::{from_toon, from_yaml};
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::inspect::NeighborDirection;
use arcgraph_mcp::tools::search::{SearchHit, SearchResult};
use proptest::prelude::*;
use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────
// Strategy: bounded Neighborhood values. The Value lattice that the
// TOON encoder normalizes onto excludes floats per
// `tests/common/mod.rs` discipline; we mirror that discipline here
// (property values are bools, ints, and strings — same lattice as
// the shared TOON proptest).
// ─────────────────────────────────────────────────────────────────────

fn arb_neighbor_direction() -> impl Strategy<Value = NeighborDirection> {
    prop_oneof![
        Just(NeighborDirection::Out),
        Just(NeighborDirection::In),
        Just(NeighborDirection::Undirected),
    ]
}

fn arb_label() -> impl Strategy<Value = Option<String>> {
    prop_oneof![
        Just(None),
        // Identifier-shaped labels only — TOON unquoted-key syntax.
        "[A-Za-z][A-Za-z0-9_]{0,7}".prop_map(Some),
    ]
}

fn arb_rel_type() -> impl Strategy<Value = Option<String>> {
    arb_label()
}

fn arb_simple_value() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        any::<bool>().prop_map(serde_json::Value::from),
        any::<i64>().prop_map(serde_json::Value::from),
        // ASCII-only short strings — avoids the encoder's quoting
        // edge cases the upstream proptest handles separately.
        "[A-Za-z0-9 _]{0,12}".prop_map(serde_json::Value::from),
    ]
}

fn arb_property_bag() -> impl Strategy<Value = BTreeMap<String, serde_json::Value>> {
    prop::collection::btree_map(
        // Identifier-shaped property keys only — TOON unquoted-key
        // syntax (see `crate::tools::explore` for the contract).
        "[A-Za-z][A-Za-z0-9_]{0,7}",
        arb_simple_value(),
        0..3,
    )
}

fn arb_neighborhood_node(depth: u32) -> impl Strategy<Value = NeighborhoodNode> {
    (any::<u64>(), arb_label(), arb_property_bag()).prop_map(move |(id, label, properties)| {
        NeighborhoodNode {
            id,
            label,
            depth,
            properties,
        }
    })
}

fn arb_neighborhood_edge() -> impl Strategy<Value = NeighborhoodEdge> {
    (
        any::<u64>(),
        any::<u64>(),
        arb_rel_type(),
        arb_neighbor_direction(),
    )
        .prop_map(|(from, to, rel_type, direction)| NeighborhoodEdge {
            from,
            to,
            rel_type,
            direction,
        })
}

fn arb_neighborhood() -> impl Strategy<Value = Neighborhood> {
    (
        any::<u64>(),
        0u32..=5,
        prop::collection::vec(arb_neighborhood_node(0), 0..4),
        prop::collection::vec(arb_neighborhood_node(1), 0..4),
        prop::collection::vec(arb_neighborhood_edge(), 0..4),
    )
        .prop_map(|(seed, max_depth, depth0, depth1, edges)| {
            let mut nodes = depth0;
            nodes.extend(depth1);
            Neighborhood {
                seed,
                max_depth,
                truncated: false,
                nodes,
                edges,
            }
        })
}

fn arb_search_hit() -> impl Strategy<Value = SearchHit> {
    (any::<u64>(), arb_label(), -1000i32..=1000).prop_map(|(node_id, label, raw_score)| {
        // Bias every score to a non-integer-valued f64 so the TOON
        // encoder's "integer-valued floats emitted as integers"
        // normalization rule (spec §2) does NOT change the wire
        // shape mid-roundtrip. Adding 0.25 gives a quarter-integer
        // float that is exactly representable in f64 (0.25 has a
        // terminating binary expansion) and survives TOON ↔ JSON ↔
        // YAML unchanged.
        let score = f64::from(raw_score) + 0.25;
        SearchHit {
            node_id,
            label,
            score,
        }
    })
}

fn arb_search_result() -> impl Strategy<Value = SearchResult> {
    (1u32..=20, prop::collection::vec(arb_search_hit(), 0..6))
        .prop_map(|(k, hits)| SearchResult { k, hits })
}

fn decode_body(format: ResponseFormat, body: &str) -> Value {
    match format {
        ResponseFormat::Toon => from_toon::<Value>(body).expect("toon decode"),
        ResponseFormat::Yaml => from_yaml::<Value>(body).expect("yaml decode"),
        ResponseFormat::Json => serde_json::from_str::<Value>(body).expect("json decode"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Roundtrip oracle: Neighborhood → ResponseFormat → bytes →
    /// Value, vs serde_json::to_value(Neighborhood). The two MUST
    /// match on every format. This is the W14β M5-06 wire-shape
    /// invariant: clients that decode `result.body` via the format
    /// slug observe the same structural value the producer encoded.
    #[test]
    fn neighborhood_roundtrips_through_every_format(n in arb_neighborhood()) {
        let canonical = serde_json::to_value(&n).expect("canonical");
        for format in [ResponseFormat::Toon, ResponseFormat::Yaml, ResponseFormat::Json] {
            let env = render_response(format, &canonical)
                .expect("render");
            let body = env["body"].as_str().expect("body string");
            let decoded = decode_body(format, body);
            prop_assert_eq!(
                decoded,
                canonical.clone(),
                "Neighborhood mismatch under format {:?}",
                format
            );
        }
    }

    /// Same oracle for SearchResult — M5-07 wire-shape invariant.
    #[test]
    fn search_result_roundtrips_through_every_format(r in arb_search_result()) {
        let canonical = serde_json::to_value(&r).expect("canonical");
        for format in [ResponseFormat::Toon, ResponseFormat::Yaml, ResponseFormat::Json] {
            let env = render_response(format, &canonical)
                .expect("render");
            let body = env["body"].as_str().expect("body string");
            let decoded = decode_body(format, body);
            prop_assert_eq!(
                decoded,
                canonical.clone(),
                "SearchResult mismatch under format {:?}",
                format
            );
        }
    }
}

/// PR #292 review LOW-3 — pin the TOON spec §2 integer-normalization
/// quirk EXPLICITLY for the integer-valued `SearchHit.score: f64`
/// case. The proptest above biases scores to non-integer-valued floats
/// (raw + 0.25) to side-step this normalization rule, so the
/// integer-valued case is NOT covered by the proptest oracle.
///
/// Production producers (RRF / weighted-RRF / CombSUM rank-fusion
/// scores) naturally have non-zero fractional parts, so this is a
/// producer-contract concern in practice. But a stub fixture or a
/// future code path that emits `score: 1.0` would round-trip through
/// TOON as `score: 1` (i64). This test documents and pins that
/// behavior so a future reader grepping for `score: 1.0` finds the
/// integer-normalization explanation.
#[test]
fn search_hit_integer_valued_score_normalizes_to_int_through_toon() {
    let canonical = serde_json::to_value(SearchResult {
        k: 1,
        hits: vec![SearchHit {
            node_id: 42,
            label: Some("Document".into()),
            score: 1.0,
        }],
    })
    .expect("canonical");

    // JSON + YAML preserve the f64 → Number::F64 distinction.
    for format in [ResponseFormat::Yaml, ResponseFormat::Json] {
        let env = render_response(format, &canonical).expect("render");
        let body = env["body"].as_str().expect("body string");
        let decoded = decode_body(format, body);
        assert_eq!(
            decoded, canonical,
            "{format:?} round-trip preserves f64=1.0 as canonical Number"
        );
    }

    // TOON spec §2 normalizes integer-valued floats to integer
    // wire-shape. Decoding back through `from_toon::<Value>` therefore
    // yields a Number::I64(1), not Number::F64(1.0). The decoded value
    // is NOT structurally equal to the canonical f64-bearing Value.
    let env = render_response(ResponseFormat::Toon, &canonical).expect("render");
    let body = env["body"].as_str().expect("body string");
    let decoded = decode_body(ResponseFormat::Toon, body);
    assert_ne!(
        decoded, canonical,
        "TOON normalizes integer-valued floats; decoded should NOT equal f64-bearing canonical"
    );
    let decoded_score = &decoded["hits"][0]["score"];
    assert!(
        decoded_score.is_i64() || decoded_score.is_u64(),
        "TOON-normalized score must be an integer Number variant: {decoded_score:?}"
    );
    assert_eq!(decoded_score.as_i64(), Some(1), "value preserved as 1");
}
