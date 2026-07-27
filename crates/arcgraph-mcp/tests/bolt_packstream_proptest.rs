//! W14δ M5-13 proptest: PackStream encode → decode roundtrip is
//! lossless for every value the lattice admits.
//!
//! The unit tests in `transport::bolt::packstream::tests` pin
//! specific shapes (each primitive, narrowing-band boundaries,
//! struct arity caps). This proptest generalizes that pin: for any
//! [`PackValue`] the strategy can synthesize, encode + decode MUST
//! return the input verbatim.
//!
//! Per the spawn prompt's testing section, the proptest is the
//! single proptest the slice ships. Generation is deliberately
//! bounded:
//!
//! - Bytes / strings ≤ 256 bytes (so the test fits in the default
//!   proptest case-count without blowing CI time).
//! - List / Map nesting depth ≤ 4 (avoid exponential growth).
//! - Struct arity ≤ 15 (the PackStream STRUCT marker only admits
//!   TINY_STRUCT; arities ≥ 16 are out-of-contract per
//!   §"Struct" and rejected at encode-time by a separate unit pin).

use arcgraph_mcp::transport::bolt::{PackValue, decode, encode};
use proptest::prelude::*;

/// Recursive PackValue strategy. The leaf primitives are
/// straightforward; the composite leaves (List / Map / Struct)
/// recurse with depth budget enforced via proptest's `Strategy`
/// combinators.
fn pack_value_strategy() -> impl Strategy<Value = PackValue> {
    let leaf = prop_oneof![
        Just(PackValue::Null),
        any::<bool>().prop_map(PackValue::Boolean),
        any::<i64>().prop_map(PackValue::Integer),
        // NaN is excluded because `NaN != NaN` would defeat the
        // roundtrip equality assertion below. ±Inf IS roundtripped
        // losslessly through `to_be_bytes`/`from_be_bytes` (the
        // bit-pattern is preserved and `f64::INFINITY == f64::INFINITY`
        // evaluates `true`), so the proptest covers it. The companion
        // unit pin `primitives_roundtrip_lossless` in
        // `transport/bolt/packstream.rs` also pins ±Inf explicitly.
        any::<f64>()
            .prop_filter("not NaN", |f| !f.is_nan())
            .prop_map(PackValue::Float),
        ".{0,256}".prop_map(PackValue::String),
        proptest::collection::vec(any::<u8>(), 0..=256).prop_map(PackValue::Bytes),
    ];

    leaf.prop_recursive(
        4,  // max depth
        64, // max total nodes per case
        16, // max items per inner collection
        |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 0..=16).prop_map(PackValue::List),
                proptest::collection::btree_map(".{0,32}", inner.clone(), 0..=16)
                    .prop_map(PackValue::Map),
                (any::<u8>(), proptest::collection::vec(inner, 0..=15))
                    .prop_map(|(tag, fields)| PackValue::Struct { tag, fields }),
            ]
        },
    )
}

proptest! {
    /// Property: encode(decode(value)) == value (with the codec
    /// reading exactly `encoded.len()` bytes).
    #[test]
    fn encode_decode_roundtrip(v in pack_value_strategy()) {
        let mut buf = Vec::new();
        encode(&mut buf, &v).expect("encode ok");
        let (decoded, n) = decode(&buf, 0).expect("decode ok");
        prop_assert_eq!(n, buf.len(), "consumed all bytes");
        prop_assert_eq!(decoded, v);
    }
}
