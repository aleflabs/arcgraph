//! Shared integration-test helpers for the W11ε serializer proptests.
//!
//! Per Rust testing convention, `tests/common/` is NOT compiled as a
//! standalone test crate; consumer files pull it in with
//! `mod common;` at the top level.
//!
//! The shared `arb_value` strategy generates a `serde_json::Value` tree
//! whose every variant is roundtrip-safe across BOTH serializers
//! (TOON and YAML). The deliberate exclusions:
//!   - Floats: TOON canonicalizes `1.0` → `"1"` (integer) per spec §2,
//!     so a float-bearing roundtrip would drift Number-typing on
//!     decode. Float roundtripping is exercised in the unit tests next
//!     to each serializer instead.
//!   - Control characters outside `{\n, \r, \t}`: TOON spec §7.1
//!     defines exactly five escape sequences; other control bytes
//!     cannot be encoded losslessly. The encoder errors on them
//!     (`ToonError::Unencodable`) and the proptest filters them at
//!     generation time so the test signal stays on roundtrip
//!     correctness, not on the Unencodable error path.
//!   - Map keys outside `^[a-zA-Z_][a-zA-Z0-9_.]*$`: this slice does
//!     not implement quoted-key support (see `serializers/toon.rs`
//!     module docs §"Quoted keys"). Restricted-key generation keeps
//!     the proptest exercising the full structural surface without
//!     hitting the deliberately-unimplemented code path.

#![allow(dead_code)]

use proptest::prelude::*;
use serde_json::Value;

/// Strategy: a single character drawn from a balanced alphabet.
///
/// The alphabet mixes:
///   - identifier-class bytes (heavy weight, common case),
///   - structural punctuation that triggers TOON quoting rules
///     (`:`, `,`, `"`, `\`, `[`, `]`, `{`, `}`),
///   - whitespace including the 3 control chars TOON's escape table
///     accepts (`\n`, `\r`, `\t`),
///   - a small unicode tail (BMP + emoji) so quoting + UTF-8 paths
///     get coverage too.
pub fn arb_char() -> impl Strategy<Value = char> {
    prop_oneof![
        // Heavy weight on identifier characters so collisions with
        // the must-quote enumeration stay representative rather than
        // dominating the input distribution.
        20 => prop::sample::select(
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                .iter()
                .map(|&b| b as char)
                .collect::<Vec<_>>()
        ),
        // Quoting-trigger structural characters.
        5 => prop::sample::select(vec![
            ' ', ':', ',', '"', '\\', '[', ']', '{', '}', '-', '.',
        ]),
        // The three TOON-allowed control chars.
        2 => prop::sample::select(vec!['\n', '\r', '\t']),
        // Multi-byte unicode tail.
        2 => prop::sample::select(vec!['é', '漢', '🎉', 'Ω', 'ñ']),
    ]
}

pub fn arb_string() -> impl Strategy<Value = String> {
    prop::collection::vec(arb_char(), 0..=24).prop_map(|cs| cs.into_iter().collect())
}

/// Valid-unquoted-key identifier per TOON spec §7.3.
pub fn arb_key() -> impl Strategy<Value = String> {
    "[a-zA-Z_][a-zA-Z0-9_.]{0,7}"
}

/// Recursive value strategy, depth-bounded so 10K cases run in
/// reasonable wall-time on consumer hardware.
pub fn arb_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|i| Value::Number(i.into())),
        arb_string().prop_map(Value::String),
    ];
    leaf.prop_recursive(
        // Max recursion depth — bounded so the encoder's
        // depth-tracking machinery is exercised but proptest doesn't
        // pay quadratic indent-emission cost.
        4,
        // Max total nodes in the generated tree (proptest's
        // soft-cap; rejects beyond and re-rolls).
        32,
        // Max children per array / object.
        4,
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..=4).prop_map(Value::Array),
                prop::collection::vec((arb_key(), inner), 0..=4).prop_map(|kvs| {
                    let mut map = serde_json::Map::new();
                    for (k, v) in kvs {
                        map.insert(k, v);
                    }
                    Value::Object(map)
                }),
            ]
        },
    )
}
