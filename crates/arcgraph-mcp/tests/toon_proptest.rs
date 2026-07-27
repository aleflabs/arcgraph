//! Roundtrip proptest for the W11ε M5-09 TOON serializer.
//!
//! Oracle: `from_toon::<Value>(to_toon(v)?) == v` for every `v` the
//! shared `arb_value` strategy produces (per `tests/common/mod.rs`,
//! the strategy excludes float-typed numbers and quoted-key shapes
//! that the encoder is documented to NOT support — see toon.rs
//! module docs).
//!
//! Per the spawn-prompt §GAUNTLET step 4, this file is meant to run
//! at `PROPTEST_CASES=10000` to satisfy the M5-09 acceptance bar. The
//! default `cases=256` still runs in CI; the env-var override turns
//! it up for the heavy gauntlet pass.

mod common;

use arcgraph_mcp::serializers::{ToonError, from_toon, to_toon};
use common::arb_value;
use proptest::prelude::*;
use serde_json::Value;

proptest! {
    #[test]
    fn toon_value_roundtrip(v in arb_value()) {
        let encoded = match to_toon(&v) {
            Ok(s) => s,
            // The strategy is engineered to never produce Unencodable
            // inputs; if it does (e.g., an unforeseen control-char
            // escape), the test should fail loudly so we tighten the
            // strategy rather than silently skip.
            Err(ToonError::Unencodable(msg)) => {
                prop_assert!(false, "encoder rejected strategy-generated value: {msg}");
                unreachable!()
            }
            Err(e) => {
                prop_assert!(false, "unexpected encode error: {e:?}");
                unreachable!()
            }
        };
        let back: Value = match from_toon(&encoded) {
            Ok(v) => v,
            Err(e) => {
                prop_assert!(
                    false,
                    "decode failed on encoder output:\n--- TOON ---\n{encoded}\n--- end ---\nerror: {e:?}"
                );
                unreachable!()
            }
        };
        prop_assert_eq!(
            &back,
            &v,
            "roundtrip mismatch.\nTOON:\n{}\noriginal: {:?}\ndecoded: {:?}",
            encoded,
            v,
            back
        );
    }
}
