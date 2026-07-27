//! M5-10 — YAML serializer for nested results.
//!
//! Thin wrapper over `serde_yaml`. Per spawn prompt §M5-10 the slice
//! intentionally does NOT hand-roll a parser — `serde_yaml` is the
//! mainstream Serde↔YAML bridge, dual-licensed MIT/Apache-2.0
//! (Prime Directive #1), and is already in the workspace dep tree.
//!
//! Public API: `to_yaml` / `from_yaml`, both generic over Serde traits.
//! Errors translate at this boundary into `YamlError` per the codec
//! error-translation discipline (see `docs/codec-error-translation.md`):
//! callers never see `serde_yaml::Error` directly.
//!
//! Why YAML at all (when TOON exists)? Per design-v2 §9.3:
//!   - **Tabular** (uniform rows) → TOON.
//!   - **Nested** (subgraphs with heterogeneous label types) → YAML.
//!   - **JSON** as fallback.
//!
//! YAML is the right shape for `graph.schema()` style trees because the
//! tree depth doesn't penalize tokens the way JSON's brace duplication
//! does (design-v2 §9.3 cites 30% efficiency gain over JSON and a
//! 17.7pp accuracy advantage over XML on nested-tree inputs).
//!
//! ## Dependency unsafe surface
//!
//! This module is safe Rust at the first-party level — no `unsafe`
//! blocks are introduced here, and the file is compatible with a
//! module-level `#![forbid(unsafe_code)]` attribute. However, it
//! transitively depends on `unsafe-libyaml` via `serde_yaml`:
//!
//! ```text
//! arcgraph-mcp -> serde_yaml -> unsafe-libyaml (c2rust port of libyaml)
//! ```
//!
//! `unsafe-libyaml` is a c2rust-translated port of the upstream libyaml
//! C library and by design preserves C-style unsafe machinery. We treat
//! the trust boundary at the dep edge: the upstream `unsafe-libyaml`
//! audit is the authority for memory-safety of the YAML parser; this
//! wrapper does not re-audit it. The Apache-2.0/MIT compatibility
//! check lives in the workspace `deny.toml` allow-list, and `cargo deny
//! check` is part of the W11ε gauntlet.
//!
//! For agent-facing surfaces that accept untrusted YAML input (M5-04+
//! Tier-1 tools, M5-02 HTTP transport), the consumer boundary should
//! add (a) an input-size cap, (b) explicit handling of YAML-DoS classes
//! the wrapper does NOT mitigate (anchor amplification, deep alias
//! chains, billion-laughs); none of those are addressed at the
//! `from_yaml` boundary. A migration to `serde_norway` or
//! `serde_yaml_bw` (both pure-safe-Rust YAML implementations) remains
//! forward debt.

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::error::YamlError;

/// Encode `value` as a YAML document.
///
/// Returns the canonical YAML string `serde_yaml` would produce. The
/// roundtrip oracle is `from_yaml::<T>(to_yaml(&v)) == v` for every
/// `T: Serialize + DeserializeOwned + PartialEq`; the proptest at
/// `tests/yaml_proptest.rs` exercises the oracle at
/// `PROPTEST_CASES=10000`.
///
/// # Errors
///
/// Returns `YamlError::Encode` if the value's `Serialize` impl fails
/// or the resulting YAML cannot be emitted (e.g., a non-string map key
/// in a context the YAML spec forbids — `serde_yaml` accepts arbitrary
/// keys, so this is rare in practice).
pub fn to_yaml<T: Serialize>(value: &T) -> Result<String, YamlError> {
    serde_yaml::to_string(value).map_err(YamlError::Encode)
}

/// Decode a YAML document into `T`.
///
/// Accepts any string `serde_yaml` accepts — the wrapper does not add
/// schema validation; consumers that need strictness should layer it
/// at the tool boundary (see ADR-004 §"all inputs validated").
///
/// # Errors
///
/// Returns `YamlError::Decode` on any parse / shape mismatch.
pub fn from_yaml<T: DeserializeOwned>(s: &str) -> Result<T, YamlError> {
    serde_yaml::from_str(s).map_err(YamlError::Decode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_scalar_primitives() {
        // Spec breadth check: every JSON primitive should survive a
        // YAML roundtrip via the Value IR. Done as a single test
        // because the failure mode (`serde_yaml` mishandling a primitive
        // class) would be uniform across all five.
        for v in [
            json!(null),
            json!(true),
            json!(false),
            json!(42_i64),
            json!(-2.5_f64),
            json!("hello"),
            json!(""),
        ] {
            let yaml = to_yaml(&v).expect("encode");
            let back: serde_json::Value = from_yaml(&yaml).expect("decode");
            assert_eq!(v, back, "roundtrip mismatch on {v:?} -> {yaml:?}");
        }
    }

    #[test]
    fn roundtrip_nested_object() {
        // Mirrors the design-v2 §9.3 "nested results" use-case:
        // heterogeneous depth, mixed primitive types per branch.
        let v = json!({
            "schema": {
                "labels": ["Person", "Comment"],
                "props": { "Person": ["id", "name"] },
            },
            "version": 1,
        });
        let yaml = to_yaml(&v).expect("encode");
        let back: serde_json::Value = from_yaml(&yaml).expect("decode");
        assert_eq!(v, back);
    }

    #[test]
    fn roundtrip_array_of_mixed_objects() {
        let v = json!([
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob", "extra": [1, 2, 3]},
        ]);
        let yaml = to_yaml(&v).expect("encode");
        let back: serde_json::Value = from_yaml(&yaml).expect("decode");
        assert_eq!(v, back);
    }

    #[test]
    fn roundtrip_unicode_and_special_chars() {
        // Validates that YAML's quoting machinery handles the same
        // edge cases TOON's quoting rules do (control chars, unicode).
        let v = json!({
            "emoji": "🎉🎊",
            "with_colon": "a: b",
            "newline_inside": "line1\nline2",
            "quoted": "she said \"hi\"",
            "leading_space": " indented",
        });
        let yaml = to_yaml(&v).expect("encode");
        let back: serde_json::Value = from_yaml(&yaml).expect("decode");
        assert_eq!(v, back);
    }

    #[test]
    fn decode_error_carries_context() {
        // Hostile-regex pin: a known-bad YAML must surface as
        // `YamlError::Decode`, not bubble up as `serde_yaml::Error`.
        // Re-asserts the codec error-translation discipline.
        let bad = "key: [unterminated";
        let err = from_yaml::<serde_json::Value>(bad).unwrap_err();
        assert!(matches!(err, YamlError::Decode(_)), "got {err:?}");
    }
}
