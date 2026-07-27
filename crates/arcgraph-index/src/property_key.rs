//! **RC-4 / RC-5 (#1366)** — canonical property-index key derivation.
//!
//! The user-visible property index keys node property VALUES. This
//! module owns the single load-bearing rule of Phase 1:
//!
//! > **The index-lookup key of a value MUST match ArcQL `=` coercion
//! > EXACTLY.** Any divergence between the key a WRITE stores and the
//! > key a LOOKUP computes for a value that `=` calls equal is a
//! > *silent wrong-results* bug (the int/float false-split class NN-4
//! > #1387 hit on MERGE lock keys).
//!
//! [`canonical_row_key`] is the sole entry point. Given a typed
//! [`IndexKeyInput`] (the MCP layer maps its property `Value` into this
//! JSON-opaque-boundary type — Prime Directive 7, storage/index never
//! see JSON), it returns:
//!
//! - `Some(PropertyValue)` — the canonical B+tree key for a SUPPORTED,
//!   representable value.
//! - `None` — an UNSUPPORTED value (composite/list/map/blob, a negative
//!   integer the `u56` slot can't hold, a non-integral / out-of-range
//!   float now that Float is dropped as an indexed type). The caller
//!   MUST take the unsupported-value path: the write still succeeds,
//!   the property is simply ABSENT from the index, the planner keeps a
//!   residual filter, and a warning/metric fires. `None` is NEVER a
//!   reason to reject a write.
//!
//! # RC-5 type decisions (spelled out, per the design §types RC-5)
//!
//! - **String** → [`PropertyValue::StrHash`] via
//!   [`crate::secondary_btree::hash_str_56`] (RC-4 — no intern growth,
//!   collisions absorbed by candidate-then-verify).
//! - **Boolean** → `U32(0|1)` (a tiny, exact, order-stable domain).
//! - **Integer (i64)** — the coercion-critical case (see below).
//! - **Float** is DROPPED as an *indexed value type* (RC-5): a stored
//!   *fractional* float is unsupported → `None`. But a float *lookup
//!   literal* that is INTEGRAL (`n.age = 42.0`) MUST canonicalize to
//!   the integer key so it finds the stored int `42` — that path lives
//!   here too, because `=` coerces `(x as f64) == y`.
//!
//! # The int/float boundary (mirrors NN-4 #1387 F4 exactly)
//!
//! ArcQL `=` compares a numeric predicate against a stored numeric via
//! `(x as f64) == y` (LOSSY on the i64 side). So the key rule splits at
//! `2^53` — the largest magnitude for which every i64 has a UNIQUE f64
//! image:
//!
//! - **`|n| < 2^53`:** the integer (and any integral float that
//!   round-trips to it) has a unique f64 image → key through the
//!   INTEGER bucket (`U64(n)`), exact and collision-free.
//! - **`|n| >= 2^53`:** `(x as f64)` starts merging distinct integers
//!   (`2^53` and `2^53+1` share the image `2^53.0`). To keep the index
//!   key consistent with `=`, BOTH such an integer and an integral
//!   float at that magnitude key through the SAME "float-image bucket":
//!   `StrHash(hash of the f64 bit pattern's canonical form)`. This
//!   guarantees `Integer(2^53) ≡ Integer(2^53+1) ≡ Float(2^53.0)` all
//!   land on one key — the exact set `=` treats as mutually equal via
//!   the float image (the #1387 F4 "float>2^53 bucket").
//!
//! Because Float is dropped, a stored *fractional* float is never
//! indexed; only the lookup-literal / integer-image coercion above is
//! in play, so the boundary rule is closed.

use crate::secondary_btree::{PropertyValue, hash_str_56};

/// `2^53` — the exact boundary of the contiguous integer range an `f64`
/// mantissa represents losslessly. Mirrors `merge.rs`'s
/// `F64_EXACT_INT_BOUND` (NN-4 #1387): at/above `2^53` the `(x as f64)`
/// coercion `=` applies starts merging distinct integers.
const F64_EXACT_INT_BOUND: i64 = 1_i64 << 53; // 9_007_199_254_740_992

/// **RC-5 (#1366)** — a typed, JSON-OPAQUE value handed across the
/// storage/index boundary for property-index key derivation.
///
/// The MCP layer (which owns JSON) maps its `Value` into exactly these
/// variants and calls [`canonical_row_key`]; storage/index never touch
/// a JSON blob (Prime Directive 7). The variant set is the RC-5
/// supported domain (`String` / `Integer` / `Boolean`) PLUS `Float`
/// (needed for the integral-float lookup-literal coercion) PLUS a
/// catch-all `Unsupported` the MCP uses for composite/list/map/blob.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexKeyInput<'a> {
    /// A UTF-8 string value (keyed by 56-bit hash — RC-4).
    Str(&'a str),
    /// A signed 64-bit integer value.
    Int(i64),
    /// A boolean value.
    Bool(bool),
    /// A float value. A stored fractional float is unsupported (Float
    /// dropped, RC-5) → `None`; an INTEGRAL float canonicalizes to the
    /// integer key so `n.age = 42.0` finds stored int `42`.
    Float(f64),
    /// Any value type the index does not support (composite / list /
    /// map / blob). Always → `None` (absent-from-index path).
    Unsupported,
}

/// **RC-4 / RC-5 (#1366)** — derive the canonical B+tree key for an
/// index-eligible property value, or `None` for an unsupported /
/// unrepresentable value.
///
/// See the module docs for the full rule set. This function is the
/// SINGLE place the "index key ≡ `=` coercion" invariant is enforced;
/// a WRITE and a LOOKUP that both route their value through here are
/// guaranteed to agree.
#[must_use]
pub fn canonical_row_key(input: IndexKeyInput<'_>) -> Option<PropertyValue> {
    match input {
        // RC-4: strings key by 56-bit hash. No intern growth; a
        // never-seen lookup value hashes in place (no read-path
        // mutation). Collisions absorbed by candidate-then-verify.
        IndexKeyInput::Str(s) => Some(PropertyValue::StrHash(hash_str_56(s))),
        // Boolean → tiny exact domain.
        IndexKeyInput::Bool(b) => Some(PropertyValue::U32(u32::from(b))),
        // Integer: the coercion-critical case.
        IndexKeyInput::Int(n) => int_key(n),
        // Float: only an INTEGRAL, in-range float is index-eligible,
        // and it routes through the SAME `int_key` path so a lookup
        // literal `42.0` produces the identical key to stored int `42`.
        // A fractional / non-finite / out-of-i64-range float is
        // unsupported (Float dropped, RC-5) → None.
        IndexKeyInput::Float(f) => {
            if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                let as_int = f as i64;
                // Round-trip guard: only treat as the integer it images
                // to when the cast is exact (rejects e.g. a float whose
                // truncation loses precision).
                if as_int as f64 == f {
                    return int_key(as_int);
                }
            }
            None
        }
        IndexKeyInput::Unsupported => None,
    }
}

/// Map an `i64` to its canonical index key per the int/float boundary.
///
/// - **Negative** → `None`. The `u56` value slot caps at `2^56 − 1` and
///   has no sign bit; a negative i64 is not representable. Per RC-5 a
///   negative integer takes the unsupported-value path (write succeeds,
///   property absent from index, residual filter + warning). This is
///   stated explicitly so "i64/u56-compatible range" is never silently
///   read as "non-negative only".
/// - **`0 <= n < 2^53`** → `U64(n)` (the integer bucket — exact,
///   f64-unique, collision-free).
/// - **`2^53 <= n < 2^56`** → the FLOAT-IMAGE bucket: key by the hash
///   of the canonical decimal of `(n as f64)`, so this integer and the
///   integral float that shares its f64 image (which `=` calls equal)
///   land on ONE key (#1387 F4 "float>2^53 bucket"). Using the float
///   image's canonical form — not `n` itself — is what makes
///   `Integer(2^53)`, `Integer(2^53+1)`, and `Float(2^53.0)` collide.
/// - **`n >= 2^56`** → still keyed through the float-image bucket (the
///   `u56` slot could not hold `n` directly anyway, and `=` has already
///   merged it with its float image).
fn int_key(n: i64) -> Option<PropertyValue> {
    if n < 0 {
        // Negative → unsupported (u56 slot is unsigned; RC-5).
        return None;
    }
    if n < F64_EXACT_INT_BOUND {
        // Exact integer bucket — every value here has a unique f64
        // image, so no float can spuriously collide.
        return Some(PropertyValue::U64(n as u64));
    }
    // At/above 2^53: route through the float-image bucket so this
    // integer collides with the integral float `=` treats it as equal
    // to. We key by a 56-bit hash of the f64 image's canonical bit
    // pattern; the StrHash variant reuses the 7-byte slot. Two i64s
    // that share an f64 image (`2^53` and `2^53+1`) hash identically
    // because `(n as f64).to_bits()` is identical for both.
    Some(PropertyValue::StrHash(float_image_key(n as f64)))
}

/// 56-bit key for a float image (the `>= 2^53` integer bucket). Keys by
/// the f64's raw bit pattern so any two i64/float inputs that share an
/// f64 image (which `=` treats as equal) collide onto one key.
///
/// **On-disk-key-stability caveat (#1366 R1 NIT-2):** like
/// [`hash_str_56`], this uses [`std::collections::hash_map::DefaultHasher`]
/// — currently **SipHash-1-3 seeded `(0, 0)`** (per the current std impl)
/// — and the result becomes an ON-DISK secondary-index key. That output
/// is stable within a toolchain but is **not contractually guaranteed
/// across Rust toolchain upgrades**; a std change would make old keys
/// unfindable (a candidate-verify false-negative, WAL-replay hygiene, not
/// wrong results). The pinned canary test `float_image_key_canary_is_stable`
/// locks the current value on a fixed f64 so any such change FAILS loudly.
///
/// [`hash_str_56`]: crate::secondary_btree::hash_str_56
fn float_image_key(f: f64) -> u64 {
    // Hash the 8 raw bytes of the bit pattern, then mask to 56 bits (the
    // same slot the StrHash variant occupies). Collisions are absorbed
    // by candidate-then-verify like any StrHash collision.
    //
    // `DefaultHasher` = SipHash-1-3 seeded (0, 0) per current std; on-disk
    // stability is only within-toolchain — see the doc caveat + the
    // `float_image_key_canary_is_stable` regression test.
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Canonicalize -0.0 → +0.0 so both bit patterns key alike (they are
    // `=`-equal); NaN never reaches here (fract()==0.0 excludes it).
    let bits = if f == 0.0 {
        0.0f64.to_bits()
    } else {
        f.to_bits()
    };
    bits.hash(&mut h);
    h.finish() & ((1u64 << 56) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the current `float_image_key` value on a fixed f64. Like
    /// [`hash_str_56`](crate::secondary_btree::hash_str_56)'s canary,
    /// this guards an ON-DISK secondary-index key against a silent
    /// `DefaultHasher` (SipHash-1-3 seeded (0, 0)) change across Rust
    /// toolchain upgrades — a change FAILS loudly instead of degrading
    /// to candidate-verify false-negatives. Uses `2^53` (the smallest
    /// float-image-bucketed integer), a representative live input.
    #[test]
    fn float_image_key_canary_is_stable() {
        // 2^53 = 9007199254740992.0
        assert_eq!(
            float_image_key(9_007_199_254_740_992.0),
            9_377_163_672_145_492,
            "DefaultHasher output changed for float_image_key — on-disk secondary-index \
             keys are no longer stable; migrate keys before shipping this toolchain"
        );
        assert!(float_image_key(9_007_199_254_740_992.0) < (1u64 << 56));
    }

    #[test]
    fn string_keys_by_hash_not_intern() {
        // A never-seen string hashes in place; no InternTable side.
        let k = canonical_row_key(IndexKeyInput::Str("nobody@example.com")).unwrap();
        match k {
            PropertyValue::StrHash(h) => assert!(h < (1u64 << 56), "56-bit masked"),
            other => panic!("expected StrHash, got {other:?}"),
        }
    }

    #[test]
    fn same_string_keys_identically() {
        let a = canonical_row_key(IndexKeyInput::Str("alice@x.com")).unwrap();
        let b = canonical_row_key(IndexKeyInput::Str("alice@x.com")).unwrap();
        assert_eq!(a, b, "identical strings must key identically");
    }

    #[test]
    fn boolean_keys() {
        assert_eq!(
            canonical_row_key(IndexKeyInput::Bool(true)),
            Some(PropertyValue::U32(1))
        );
        assert_eq!(
            canonical_row_key(IndexKeyInput::Bool(false)),
            Some(PropertyValue::U32(0))
        );
    }

    #[test]
    fn small_integer_keys_through_u64_bucket() {
        assert_eq!(
            canonical_row_key(IndexKeyInput::Int(42)),
            Some(PropertyValue::U64(42))
        );
        assert_eq!(
            canonical_row_key(IndexKeyInput::Int(0)),
            Some(PropertyValue::U64(0))
        );
    }

    #[test]
    fn integral_float_normalizes_to_integer_key_rc5() {
        // THE RC-5 int-float boundary: `n.age = 42.0` must find stored
        // int 42. The float literal and the stored int produce the SAME
        // key.
        let int_key = canonical_row_key(IndexKeyInput::Int(42)).unwrap();
        let float_key = canonical_row_key(IndexKeyInput::Float(42.0)).unwrap();
        assert_eq!(
            int_key, float_key,
            "42.0 must key identically to stored int 42 (= coercion)"
        );
    }

    #[test]
    fn fractional_float_is_unsupported() {
        // Float dropped (RC-5): a fractional float is not indexable.
        assert_eq!(canonical_row_key(IndexKeyInput::Float(1.5)), None);
    }

    #[test]
    fn non_finite_float_is_unsupported() {
        assert_eq!(canonical_row_key(IndexKeyInput::Float(f64::NAN)), None);
        assert_eq!(canonical_row_key(IndexKeyInput::Float(f64::INFINITY)), None);
        assert_eq!(
            canonical_row_key(IndexKeyInput::Float(f64::NEG_INFINITY)),
            None
        );
    }

    #[test]
    fn negative_integer_is_unsupported_rc5() {
        // The u56 slot has no sign bit → negative i64 takes the
        // unsupported-value path (write succeeds, absent from index).
        assert_eq!(canonical_row_key(IndexKeyInput::Int(-1)), None);
        assert_eq!(canonical_row_key(IndexKeyInput::Int(i64::MIN)), None);
    }

    #[test]
    fn unsupported_input_is_none() {
        assert_eq!(canonical_row_key(IndexKeyInput::Unsupported), None);
    }

    #[test]
    fn float_gt_2_53_buckets_with_its_integer_image() {
        // #1387 F4: at/above 2^53, `(x as f64)` merges distinct integers.
        // `2^53` and `2^53+1` share the f64 image `2^53.0`, so ALL THREE
        // (both integers + the float) must key identically — the exact
        // set `=` treats as mutually equal via the float image.
        let two53: i64 = 1_i64 << 53;
        let k_int_at = canonical_row_key(IndexKeyInput::Int(two53)).unwrap();
        let k_int_plus1 = canonical_row_key(IndexKeyInput::Int(two53 + 1)).unwrap();
        let k_float = canonical_row_key(IndexKeyInput::Float(two53 as f64)).unwrap();
        assert_eq!(
            k_int_at, k_int_plus1,
            "2^53 and 2^53+1 share an f64 image → one key"
        );
        assert_eq!(
            k_int_at, k_float,
            "the float image 2^53.0 keys with its integer images"
        );
        // And it is the StrHash (float-image) bucket, NOT the exact
        // U64 bucket — so it can never collide with a small integer.
        assert!(
            matches!(k_int_at, PropertyValue::StrHash(_)),
            "float-image bucket uses StrHash, got {k_int_at:?}"
        );
    }

    #[test]
    fn below_boundary_integer_uses_exact_bucket() {
        // Just below the boundary stays in the exact U64 bucket.
        let just_below: i64 = (1_i64 << 53) - 1;
        assert_eq!(
            canonical_row_key(IndexKeyInput::Int(just_below)),
            Some(PropertyValue::U64(just_below as u64))
        );
    }

    #[test]
    fn distinct_small_integers_never_collide() {
        // The exact bucket is collision-free below 2^53.
        let a = canonical_row_key(IndexKeyInput::Int(100)).unwrap();
        let b = canonical_row_key(IndexKeyInput::Int(101)).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn neg_zero_float_keys_like_zero() {
        // -0.0 and 0.0 are `=`-equal and both integral → both key as
        // Integer(0).
        assert_eq!(
            canonical_row_key(IndexKeyInput::Float(-0.0)),
            canonical_row_key(IndexKeyInput::Int(0))
        );
    }
}
