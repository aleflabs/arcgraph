#![no_main]
//! W28-S604: MCP YAML serializer/deserializer fuzz target (testing strategy;
//! full-canonical-set audit Task #604).
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::serializers::from_yaml`] + the matching `to_yaml`
//! encoder at `crates/arcgraph-mcp/src/serializers/yaml.rs`. This is
//! the YAML sibling of the already-covered TOON surface
//! (`toon_serializer_fuzz`): per design-v2 §9.3 the MCP layer emits
//! YAML for nested/heterogeneous result trees and accepts YAML on the
//! agent-facing tool surfaces. `from_yaml` is a thin wrapper over
//! `serde_yaml`, which transitively decodes through `unsafe-libyaml`
//! (a c2rust port of the libyaml C parser). The module docstring
//! explicitly notes the wrapper accepts untrusted YAML and adds NO
//! schema/DoS mitigation — making this the highest-value previously-
//! UNCOVERED serializer surface (untrusted input → C-derived unsafe
//! parser).
//!
//! # Assertions
//!
//! 1. **No panic.** `from_yaml::<Value>(s)` MUST NOT panic on ANY
//!    UTF-8 input. Returns `Ok(Value)` or `Err(YamlError)`; both are
//!    valid.
//! 2. **Encoder round-trip (value-equivalent).** When the input decodes
//!    to a `Value`, `to_yaml(&value).and_then(from_yaml)` MUST be
//!    value-equivalent to the first decode (same helper rationale as
//!    `toon_serializer_fuzz`: `Number(400)` ≡ `Number(400.0)`; NaN is
//!    treated as equivalent to NaN). The encoder may legitimately
//!    reject some values, which is not a round-trip failure.
//!
//! # Scope boundary / known dependency limits
//!
//! Input is capped at 16 KiB (TIGHTER than the 64 KiB used elsewhere)
//! to bound the YAML-DoS amplification classes the `from_yaml` wrapper
//! deliberately does NOT mitigate (anchor amplification / deep alias
//! chains / billion-laughs, per the `yaml.rs` docstring). Those are
//! upstream `serde_yaml`/`unsafe-libyaml` resource-exhaustion classes,
//! not first-party logic bugs; the cap keeps the fuzzer focused on the
//! first-party wrapper's no-panic + round-trip contract. A genuine
//! panic (not OOM/timeout) inside the decode/encode path on a small
//! input IS a finding and is checked in to `fuzz/artifacts/`.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 16 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    let decoded: Result<serde_json::Value, _> = arcgraph_mcp::serializers::from_yaml(s);
    if let Ok(value) = decoded {
        // Round-trip: encode the decoded value; if the encoder accepts
        // it, decoding the encoded form MUST be value-equivalent to the
        // first decode result.
        if let Ok(encoded) = arcgraph_mcp::serializers::to_yaml(&value) {
            let re_decoded: Result<serde_json::Value, _> =
                arcgraph_mcp::serializers::from_yaml(&encoded);
            if let Ok(re_value) = re_decoded {
                assert!(
                    value_equivalent(&value, &re_value),
                    "YAML roundtrip diverged: original={value} \
                     encoded={encoded} re_decoded={re_value}"
                );
            }
        }
    }
});

/// Value-equivalent comparison treating `Number(400)` == `Number(400.0)`
/// and `NaN` == `NaN`. Mirrors the helper in `toon_serializer_fuzz`:
/// `serde_json::Value` distinguishes int vs float representations
/// internally, but YAML round-trips through a single numeric domain, so
/// whole-number-float canonicalization is benign representational
/// divergence rather than a parser bug.
fn value_equivalent(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => {
            if let (Some(xf), Some(yf)) = (x.as_f64(), y.as_f64()) {
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
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| value_equivalent(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, vx)| y.get(k).is_some_and(|vy| value_equivalent(vx, vy)))
        }
        _ => false,
    }
}
