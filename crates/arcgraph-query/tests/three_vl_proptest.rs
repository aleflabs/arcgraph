//! 3VL truth-table proptest.
//!
//! Strategy: generate randomly mixed `True/False/Null` operand pairs
//! and assert the M4-22 helpers `apply_and_3vl` / `apply_or_3vl` /
//! `apply_not_3vl` agree with the openCypher 9 §6.4 truth table.
//!
//! The truth table is enumerated INSIDE this file as a reference
//! — the production helpers MUST match. This is the
//! "two implementations agree" property test pattern.
//!
//! # ADR provenance
//! - ADR-038 §2 D-20 — 3VL contract.
//! - openCypher 9 §6.4 — TRUE / FALSE / NULL truth tables.

use arcgraph_query::semantic::type_check::{
    BoolOrNull, apply_and_3vl, apply_not_3vl, apply_or_3vl,
};
use proptest::prelude::*;

/// Reference implementation: AND truth table.
///
/// This independently encodes the table per openCypher 9 §6.4.
fn reference_and(a: BoolOrNull, b: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    if a == False || b == False {
        return False;
    }
    if a == Null || b == Null {
        return Null;
    }
    True
}

/// Reference implementation: OR truth table.
fn reference_or(a: BoolOrNull, b: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    if a == True || b == True {
        return True;
    }
    if a == Null || b == Null {
        return Null;
    }
    False
}

/// Reference implementation: NOT truth table.
fn reference_not(a: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    match a {
        True => False,
        False => True,
        Null => Null,
    }
}

fn arb_bool_or_null() -> impl Strategy<Value = BoolOrNull> {
    prop::sample::select(vec![BoolOrNull::True, BoolOrNull::False, BoolOrNull::Null])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `apply_and_3vl` must agree with the reference truth table on
    /// every pair of operands.
    #[test]
    fn three_vl_and_matches_reference(
        a in arb_bool_or_null(),
        b in arb_bool_or_null()
    ) {
        prop_assert_eq!(apply_and_3vl(a, b), reference_and(a, b));
    }

    /// `apply_or_3vl` must agree with the reference truth table.
    #[test]
    fn three_vl_or_matches_reference(
        a in arb_bool_or_null(),
        b in arb_bool_or_null()
    ) {
        prop_assert_eq!(apply_or_3vl(a, b), reference_or(a, b));
    }

    /// `apply_not_3vl` must agree with the reference table.
    #[test]
    fn three_vl_not_matches_reference(a in arb_bool_or_null()) {
        prop_assert_eq!(apply_not_3vl(a), reference_not(a));
    }

    /// AND is commutative under 3VL.
    #[test]
    fn three_vl_and_is_commutative(
        a in arb_bool_or_null(),
        b in arb_bool_or_null()
    ) {
        prop_assert_eq!(apply_and_3vl(a, b), apply_and_3vl(b, a));
    }

    /// OR is commutative under 3VL.
    #[test]
    fn three_vl_or_is_commutative(
        a in arb_bool_or_null(),
        b in arb_bool_or_null()
    ) {
        prop_assert_eq!(apply_or_3vl(a, b), apply_or_3vl(b, a));
    }

    /// NOT is involutive (NOT NOT a == a) for True / False; for Null
    /// it stays Null.
    #[test]
    fn three_vl_not_is_involutive(a in arb_bool_or_null()) {
        prop_assert_eq!(apply_not_3vl(apply_not_3vl(a)), a);
    }
}
