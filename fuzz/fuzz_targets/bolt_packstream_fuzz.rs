#![no_main]
//! W28-S4: Bolt / PackStream parser fuzz target (testing-strategy §2.4
//! minimum target `bolt`; gap analysis PR #510 §1c; ADR-165 M6).
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::transport::bolt::decode`] — the Bolt 5.0 PackStream
//! decoder at `crates/arcgraph-mcp/src/transport/bolt/packstream.rs:437`.
//! It is the wire-format parser the Bolt server feeds peer bytes into.
//! This extends the bounded property test
//! (`crates/arcgraph-mcp/tests/bolt_packstream_proptest.rs`) to
//! coverage-guided fuzzing over arbitrary, hostile byte streams.
//!
//! # Assertions (testing-strategy §2.4 oracle)
//!
//! - **No panic / no UB** on ANY byte sequence — the libfuzzer contract.
//!   Truncated headers, unknown markers, invalid UTF-8, non-string map
//!   keys, unsupported struct tags, and over-deep nesting must all
//!   surface as `Err(PackError)`, never a panic.
//! - **Over-read guard.** When `decode` succeeds it MUST NOT report
//!   having consumed more bytes than the input held.
//! - **Round-trip is a canonical fixed point.** For any value that
//!   decodes, `encode` then `decode` then `encode` again must reproduce
//!   the SAME bytes. We compare at the *byte* level rather than the
//!   value level so that IEEE-754 `NaN` floats — which decode/encode
//!   bit-for-bit but compare unequal (`NaN != NaN`) — do not produce a
//!   spurious failure. This mirrors the proptest's documented NaN
//!   rationale (it excludes NaN to use value-equality; the fuzzer cannot
//!   exclude it, so it uses byte-equality instead).
//!
//! Input is capped at 64 KiB to bound per-iteration wall time; the Bolt
//! framing layer (`transport::bolt::chunking`) caps a reassembled
//! message well below this, but the decoder is fuzzed directly per the
//! §2.4 "parser" contract.

use libfuzzer_sys::fuzz_target;

use arcgraph_mcp::transport::bolt::{PackValue, decode, encode};

const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let (value, consumed) = match decode(data, 0) {
        Ok(ok) => ok,
        // Structured reject: UnknownMarker / UnexpectedEof / InvalidUtf8 /
        // NonStringMapKey / UnsupportedStructTag / DepthExceeded. Expected
        // for the overwhelming majority of arbitrary inputs.
        Err(_) => return,
    };

    // Over-read guard — `decode` must never claim more bytes than it was
    // given (the trailing bytes after `consumed` are a framing concern,
    // not a decode-consumed concern).
    assert!(
        consumed <= data.len(),
        "decode consumed {consumed} bytes from a {}-byte input",
        data.len()
    );

    // Canonical re-encode. `encode` is total over any value `decode`
    // produces: decoded structs have arity ≤ 15, and decoded
    // string/bytes/list lengths are bounded by the ≤ 64 KiB input — both
    // well inside `encode`'s limits — so a failure here is itself an
    // encoder/decoder asymmetry bug worth surfacing.
    let mut canonical = Vec::new();
    encode(&mut canonical, &value).expect("encode of a decoded PackValue must succeed");

    // Re-decode the canonical bytes: must consume them exactly and yield
    // a value whose re-encoding is byte-identical (encode∘decode is the
    // identity on canonical wire bytes — the fixed-point round-trip).
    let (value2, consumed2) = match decode(&canonical, 0) {
        Ok(ok) => ok,
        Err(e) => panic!("re-decode of canonical PackStream encoding failed: {e:?}"),
    };
    assert_eq!(
        consumed2,
        canonical.len(),
        "canonical re-encode left {} trailing byte(s) on re-decode",
        canonical.len() - consumed2
    );

    let mut canonical2 = Vec::new();
    encode(&mut canonical2, &value2).expect("re-encode of a re-decoded PackValue must succeed");
    assert_eq!(
        canonical, canonical2,
        "PackStream canonical round-trip is not a fixed point"
    );

    // Belt-and-braces value-equality WHEN the value holds no NaN float
    // (the only value that legitimately breaks reflexive equality). This
    // is strictly stronger than the byte oracle for the non-NaN case.
    if !contains_nan(&value) {
        assert_eq!(
            value, value2,
            "PackStream value round-trip diverged (non-NaN value)"
        );
    }
});

/// Recursively test whether a decoded value carries an IEEE-754 `NaN`
/// float anywhere in its structure. Bounded by the decoded value's depth,
/// which the decoder caps at `MAX_PACKSTREAM_DEPTH`.
fn contains_nan(v: &PackValue) -> bool {
    match v {
        PackValue::Float(f) => f.is_nan(),
        PackValue::List(items) => items.iter().any(contains_nan),
        PackValue::Map(entries) => entries.values().any(contains_nan),
        PackValue::Struct { fields, .. } => fields.iter().any(contains_nan),
        _ => false,
    }
}
