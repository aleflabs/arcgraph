//! Roundtrip proptest for the W11ε M5-10 YAML serializer.
//!
//! Oracle: `from_yaml::<Value>(to_yaml(v)?) == v` for every `v` the
//! shared `arb_value` strategy produces. Like the TOON proptest, this
//! intentionally excludes floats — not because YAML can't roundtrip
//! them (it can), but because we share one strategy across both
//! formats so the same input distribution exercises both wrappers.
//! Float roundtripping is covered in the YAML unit tests.
//!
//! Run heavy via `PROPTEST_CASES=10000 cargo test ... --test yaml_proptest`.

mod common;

use arcgraph_mcp::serializers::{from_yaml, to_yaml};
use common::arb_value;
use proptest::prelude::*;
use serde_json::Value;

proptest! {
    #[test]
    fn yaml_value_roundtrip(v in arb_value()) {
        let encoded = match to_yaml(&v) {
            Ok(s) => s,
            Err(e) => {
                prop_assert!(false, "encode failed: {e:?}");
                unreachable!()
            }
        };
        let back: Value = match from_yaml(&encoded) {
            Ok(v) => v,
            Err(e) => {
                prop_assert!(
                    false,
                    "decode failed on encoder output:\n--- YAML ---\n{encoded}\n--- end ---\nerror: {e:?}"
                );
                unreachable!()
            }
        };
        prop_assert_eq!(
            &back,
            &v,
            "roundtrip mismatch.\nYAML:\n{}\noriginal: {:?}\ndecoded: {:?}",
            encoded,
            v,
            back
        );
    }
}
