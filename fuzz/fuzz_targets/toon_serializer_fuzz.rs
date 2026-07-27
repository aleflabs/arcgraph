#![no_main]
//! W22-DB-ε: TOON serializer + parser fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::serializers::to_toon`] + the corresponding
//! `from_toon` decoder. The encoder pivots through `serde_json::Value`
//! per the v1.0 design (`crates/arcgraph-mcp/src/serializers/toon.rs:127`).
//!
//! # Assertion
//!
//! Two complementary surfaces are fuzzed:
//!
//! 1. **Decoder fuzz.** `from_toon::<Value>(input)` MUST NOT panic on
//!    ANY UTF-8 input. Returns either `Ok(Value)` or `Err(ToonError)`.
//! 2. **Encoder roundtrip (value-equivalent).** When the input is a
//!    valid TOON document that decodes to `Value`,
//!    `to_toon(&value).and_then(from_toon)` MUST return a value that
//!    is value-equivalent (per the [`value_equivalent`] helper) to
//!    the first decode result. Strict-`==` divergence is acceptable
//!    on numeric-representation boundaries (TOON emits `400` for
//!    `400.0`); the value-equivalent check treats `Number(400) ==
//!    Number(400.0)` so this benign representational divergence is
//!    NOT a panic — see
//!    [`docs/chaos/v1-alpha-chaos-known-issues.md`] / TOON-1 for the
//!    P2 observation.
//!
//! The encoder MAY reject some `Value`s (e.g., maps with non-
//! identifier keys per spec §"Quoted keys") with `ToonError::Unencodable`.
//! That is a legitimate reject, not a roundtrip failure.
//!
//! [`docs/chaos/v1-alpha-chaos-known-issues.md`]: ../../docs/chaos/v1-alpha-chaos-known-issues.md

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    let decoded: Result<serde_json::Value, _> =
        arcgraph_mcp::serializers::from_toon(s);
    if let Ok(value) = decoded {
        // Roundtrip: encode the decoded value; if the encoder
        // accepts it, decoding the encoded form MUST be value-
        // equivalent to the first decode result.
        if let Ok(encoded) = arcgraph_mcp::serializers::to_toon(&value) {
            let re_decoded: Result<serde_json::Value, _> =
                arcgraph_mcp::serializers::from_toon(&encoded);
            if let Ok(re_value) = re_decoded {
                assert!(
                    value_equivalent(&value, &re_value),
                    "TOON roundtrip diverged: original={value} \
                     encoded={encoded} re_decoded={re_value}"
                );
            }
        }
    }
});

/// Value-equivalent comparison treating `Number(400)` == `Number(400.0)`.
///
/// JSON has a single `Number` type but `serde_json::Value` distinguishes
/// integer vs float representations internally. The TOON encoder
/// canonicalizes whole-number floats to integer form (`400.0` → `400`).
/// This is benign representational divergence — both values
/// round-trip to the same JSON document.
fn value_equivalent(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => {
            if let (Some(xf), Some(yf)) = (x.as_f64(), y.as_f64()) {
                // NaN never compares equal; for fuzz purposes treat
                // NaN == NaN as equivalent (the TOON encoder emits
                // canonical NaN regardless of input bit pattern).
                if xf.is_nan() && yf.is_nan() {
                    return true;
                }
                xf == yf
            } else {
                x == y
            }
        }
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len()
                && x.iter().zip(y.iter()).all(|(a, b)| value_equivalent(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, vx)| {
                    y.get(k).is_some_and(|vy| value_equivalent(vx, vy))
                })
        }
        _ => false,
    }
}
