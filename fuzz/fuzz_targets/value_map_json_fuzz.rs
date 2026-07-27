#![no_main]
//! ADR-191 (#620) — `Value::Map` JSON-decode fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_query::executor::value::Value::try_from_json_value`] — the
//! network-reachable reverse JSON bridge (the MCP `raw_query` / Bolt
//! decode paths feed it untrusted JSON). With the recursive
//! `Value::Map` variant (ADR-191 D-7), an adversarially-deep nested-map
//! / nested-list JSON tree could overflow the stack absent the
//! `MAX_JSON_DECODE_DEPTH` bound (D-12). This target drives arbitrary
//! JSON at the decoder per
//! `feedback_security_class_first_network_surface.md` (recursion-depth +
//! input-bytes discipline for any value reachable from network input).
//!
//! # Construction strategy
//!
//! The byte stream is interpreted as UTF-8 JSON; if it parses to a
//! `serde_json::Value`, it is fed to `try_from_json_value`. A
//! successfully-decoded value is then round-tripped back through
//! `to_json_value` → `try_from_json_value`. (`serde_json`'s own 128-deep
//! parse limit bounds the input tree; OUR 64-deep decode cap rejects the
//! 65..=128 band with `NestingTooDeep` — both paths are exercised.)
//!
//! # Assertions
//!
//! - **No panic / no stack overflow.** `try_from_json_value` MUST return
//!   `Ok` / `Err` for ANY input — never overflow. Over-deep input is
//!   rejected with `ValueJsonError::NestingTooDeep`.
//! - **Round-trip idempotence.** Encoding then decoding a decoded value
//!   reaches a stable fixpoint (the first encode normalizes non-finite
//!   floats → null and large u64 → f64; subsequent round-trips are
//!   value-stable).

use libfuzzer_sys::fuzz_target;

use arcgraph_query::executor::value::Value;

fuzz_target!(|data: &[u8]| {
    // Interpret the bytes as JSON. Most inputs won't parse; the fuzzer
    // learns toward valid JSON via the corpus.
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };
    // The load-bearing invariant: decode MUST NOT panic / overflow on
    // ANY input (the depth cap turns over-deep trees into a clean error).
    let Ok(decoded) = Value::try_from_json_value(&json) else {
        // A rejection (UnsupportedShape / NumberOutOfRange /
        // NestingTooDeep) is a valid, panic-free outcome.
        return;
    };
    // Re-encode + re-decode the decoder's OWN output. The first decode
    // already normalized the tree, so this round-trip is a stable
    // fixpoint — a non-finite-float / large-u64 input is collapsed by
    // the FIRST encode, so `once` and `twice` must be EQUAL.
    let once = Value::try_from_json_value(&decoded.to_json_value())
        .expect("re-decode of own output must succeed");
    let twice = Value::try_from_json_value(&once.to_json_value())
        .expect("second re-decode must succeed");
    assert_eq!(once, twice, "Value::Map JSON round-trip must be idempotent");
});
