//! Cypher 3VL truth tables (M4-62) per ADR-038 §2 D-20.
//!
//! [`ThreeValued`] is the predicate-evaluation result type — distinct
//! from [`crate::executor::Value::Boolean`] so the type system enforces
//! the 3VL discipline at every consumer (filter operator, JOIN ON
//! clause, `IS NULL` check, AND/OR/NOT folds).
//!
//! # Truth tables (ADR-038 §2 D-20)
//!
//! | AND    | T       | F       | UNK     |
//! |--------|---------|---------|---------|
//! | **T**  | T       | F       | UNK     |
//! | **F**  | F       | F       | F       |
//! | **UNK**| UNK     | F       | UNK     |
//!
//! | OR     | T       | F       | UNK     |
//! |--------|---------|---------|---------|
//! | **T**  | T       | T       | T       |
//! | **F**  | T       | F       | UNK     |
//! | **UNK**| T       | UNK     | UNK     |
//!
//! | XOR    | T       | F       | UNK     |
//! |--------|---------|---------|---------|
//! | **T**  | F       | T       | UNK     |
//! | **F**  | T       | F       | UNK     |
//! | **UNK**| UNK     | UNK     | UNK     |
//!
//! XOR (openCypher v9 §boolean; #621) is true iff EXACTLY ONE operand
//! is True. Unlike AND/OR it has no short-circuit: any `Unknown`
//! operand yields `Unknown` (the result depends on the exact value of
//! both sides), so `_ XOR null = null` and `null XOR _ = null`.
//!
//! | NOT    |
//! |--------|
//! | **T → F** |
//! | **F → T** |
//! | **UNK → UNK** |
//!
//! # `IS NULL` (and `IS NOT NULL`) tunnel through
//!
//! `expr IS NULL` and `expr IS NOT NULL` are the ONLY predicates that
//! treat NULL as a real value (returning `True` / `False`, never
//! `Unknown`). Per Cypher 9 §6.2.5 + ADR-038 §2 D-20.
//!
//! # ADR provenance
//! - **ADR-038 §2 D-20** — Cypher 3VL definitive lock.
//! - **ADR-038 amendment-03 §TIER-2-b** — M4-62 3VL implementation
//!   slice scope.
//! - **ADR-006 amendment-01 §A-2** — OPTIONAL MATCH null-row emission;
//!   the rows the 3VL truth-tables consume.

use crate::executor::value::Value;

/// Cypher 3VL predicate-evaluation result.
///
/// Distinct from [`Value::Boolean`] so the type system catches a
/// predicate-evaluation routine that accidentally returns
/// `Boolean(false)` when the openCypher 3VL contract demands
/// `Unknown` (the masking-by-coercion bug class).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeValued {
    /// Predicate is definitely true.
    True,
    /// Predicate is definitely false.
    False,
    /// Predicate cannot be determined (operand reduces to NULL).
    /// Treated as "false" by WHERE filter (rows are dropped) but as
    /// distinct in AND/OR algebra per the truth tables.
    Unknown,
}

impl ThreeValued {
    /// Lift a [`Value`] to a [`ThreeValued`]. `Boolean(true)` → `True`,
    /// `Boolean(false)` → `False`, `Null` → `Unknown`. Any other
    /// `Value` is a type-check failure that the planner SHOULD have
    /// caught — the executor surfaces a runtime evaluation error via
    /// [`ThreeValued::from_value_strict`].
    #[inline]
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Boolean(true) => Self::True,
            Value::Boolean(false) => Self::False,
            Value::Null => Self::Unknown,
            // Defensive: a non-boolean predicate operand reaching here
            // is a type-check escape; surface `Unknown` per the 3VL
            // "could be anything" semantics. Strict callers use
            // `from_value_strict` to catch the planner-escape bug.
            _ => Self::Unknown,
        }
    }

    /// Strict variant of [`Self::from_value`] returning `None` for a
    /// non-Boolean / non-Null value. Operators that have a
    /// type-checked predicate operand assert this returns `Some(..)`
    /// in debug builds.
    #[inline]
    #[must_use]
    pub fn from_value_strict(v: &Value) -> Option<Self> {
        match v {
            Value::Boolean(true) => Some(Self::True),
            Value::Boolean(false) => Some(Self::False),
            Value::Null => Some(Self::Unknown),
            _ => None,
        }
    }

    /// Project to a Boolean for filter discrimination — `True` is
    /// `true`; `False` and `Unknown` are both `false`. Per Cypher 9
    /// §6.2 + ADR-038 §2 D-20: WHERE rows where the predicate is
    /// Unknown are FILTERED (treated as false at the row-emission
    /// boundary).
    #[inline]
    #[must_use]
    pub fn passes_filter(self) -> bool {
        matches!(self, Self::True)
    }

    /// Cypher 3VL `AND`. Truth-table from ADR-038 §2 D-20:
    /// `False AND _ = False`; `_ AND False = False`; `True AND True =
    /// True`; `True AND Unknown = Unknown`; `Unknown AND True =
    /// Unknown`; `Unknown AND Unknown = Unknown`.
    #[inline]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }

    /// Cypher 3VL `OR`. Truth-table from ADR-038 §2 D-20:
    /// `True OR _ = True`; `_ OR True = True`; `False OR False =
    /// False`; `False OR Unknown = Unknown`; `Unknown OR False =
    /// Unknown`; `Unknown OR Unknown = Unknown`.
    #[inline]
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, Self::False) => Self::False,
            _ => Self::Unknown,
        }
    }

    /// Cypher 3VL `XOR` (openCypher v9 §boolean; #621). True iff
    /// EXACTLY ONE operand is True. Truth-table:
    /// `True XOR False = True`; `False XOR True = True`;
    /// `True XOR True = False`; `False XOR False = False`;
    /// any `Unknown` operand → `Unknown`.
    ///
    /// Unlike [`Self::and`] / [`Self::or`], XOR has NO short-circuit:
    /// its result depends on the exact value of BOTH operands, so an
    /// `Unknown` (NULL-derived) operand makes the result `Unknown`
    /// regardless of the other side. This is the load-bearing 3VL
    /// property that Boolean3 `1` (`_ XOR null = null`) + `5`/`7`
    /// (commutativity / associativity on null) discriminate.
    #[inline]
    #[must_use]
    pub fn xor(self, other: Self) -> Self {
        match (self, other) {
            // Any Unknown operand → Unknown (no short-circuit).
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            // Exactly one True → True.
            (Self::True, Self::False) | (Self::False, Self::True) => Self::True,
            // Both True or both False → False.
            (Self::True, Self::True) | (Self::False, Self::False) => Self::False,
        }
    }

    /// Cypher 3VL `NOT`. `True ↔ False`; `Unknown` is its own image
    /// per ADR-038 §2 D-20 — UNK NOT-flipping to UNK is the
    /// load-bearing 3VL property that DISTINGUISHES it from a
    /// 2VL-projected boolean.
    ///
    /// Named `not` (not `negate`) to match the Cypher spec
    /// language; `clippy::should_implement_trait` is allowed
    /// because [`std::ops::Not::not`]'s required identity laws (`a
    /// = NOT NOT a`) hold for True/False but the Unknown-fixed-
    /// point breaks the 2VL involution that the std trait expects.
    /// Keeping the inherent-impl form preserves the 3VL discipline
    /// at the type level — a future caller writing `!tv` (Std::Not)
    /// would expect 2VL semantics and silently misread Unknown.
    #[inline]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Unknown => Self::Unknown,
        }
    }

    /// `expr IS NULL` — returns `True` iff the operand is NULL.
    /// Crucially returns `False` (NOT `Unknown`) for non-NULL operands;
    /// `IS NULL` is the canonical "tunnel through 3VL" predicate per
    /// Cypher 9 §6.2.5.
    #[inline]
    #[must_use]
    pub fn is_null(v: &Value) -> Self {
        if v.is_null() { Self::True } else { Self::False }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- AND truth table (full 3×3) ----------

    #[test]
    fn three_vl_and_truth_table() {
        // Row 1: True AND _
        assert_eq!(ThreeValued::True.and(ThreeValued::True), ThreeValued::True);
        assert_eq!(
            ThreeValued::True.and(ThreeValued::False),
            ThreeValued::False
        );
        assert_eq!(
            ThreeValued::True.and(ThreeValued::Unknown),
            ThreeValued::Unknown
        );
        // Row 2: False AND _
        assert_eq!(
            ThreeValued::False.and(ThreeValued::True),
            ThreeValued::False
        );
        assert_eq!(
            ThreeValued::False.and(ThreeValued::False),
            ThreeValued::False
        );
        assert_eq!(
            ThreeValued::False.and(ThreeValued::Unknown),
            ThreeValued::False
        );
        // Row 3: Unknown AND _
        assert_eq!(
            ThreeValued::Unknown.and(ThreeValued::True),
            ThreeValued::Unknown
        );
        assert_eq!(
            ThreeValued::Unknown.and(ThreeValued::False),
            ThreeValued::False
        );
        assert_eq!(
            ThreeValued::Unknown.and(ThreeValued::Unknown),
            ThreeValued::Unknown
        );
    }

    // ---------- OR truth table (full 3×3) ----------

    #[test]
    fn three_vl_or_truth_table() {
        // Row 1: True OR _
        assert_eq!(ThreeValued::True.or(ThreeValued::True), ThreeValued::True);
        assert_eq!(ThreeValued::True.or(ThreeValued::False), ThreeValued::True);
        assert_eq!(
            ThreeValued::True.or(ThreeValued::Unknown),
            ThreeValued::True
        );
        // Row 2: False OR _
        assert_eq!(ThreeValued::False.or(ThreeValued::True), ThreeValued::True);
        assert_eq!(
            ThreeValued::False.or(ThreeValued::False),
            ThreeValued::False
        );
        assert_eq!(
            ThreeValued::False.or(ThreeValued::Unknown),
            ThreeValued::Unknown
        );
        // Row 3: Unknown OR _
        assert_eq!(
            ThreeValued::Unknown.or(ThreeValued::True),
            ThreeValued::True
        );
        assert_eq!(
            ThreeValued::Unknown.or(ThreeValued::False),
            ThreeValued::Unknown
        );
        assert_eq!(
            ThreeValued::Unknown.or(ThreeValued::Unknown),
            ThreeValued::Unknown
        );
    }

    // ---------- XOR truth table (full 3×3) ----------

    #[test]
    fn three_vl_xor_truth_table() {
        use ThreeValued::{False, True, Unknown};
        // Row 1: True XOR _
        assert_eq!(True.xor(True), False);
        assert_eq!(True.xor(False), True);
        assert_eq!(True.xor(Unknown), Unknown);
        // Row 2: False XOR _
        assert_eq!(False.xor(True), True);
        assert_eq!(False.xor(False), False);
        assert_eq!(False.xor(Unknown), Unknown);
        // Row 3: Unknown XOR _ — ALWAYS Unknown (no short-circuit).
        // This is the load-bearing 3VL property: unlike AND (which
        // short-circuits on False) and OR (on True), XOR depends on
        // the exact value of BOTH operands, so any Unknown → Unknown.
        assert_eq!(Unknown.xor(True), Unknown);
        assert_eq!(Unknown.xor(False), Unknown);
        assert_eq!(Unknown.xor(Unknown), Unknown);
    }

    #[test]
    fn xor_is_commutative_and_associative() {
        let states = [ThreeValued::True, ThreeValued::False, ThreeValued::Unknown];
        for &a in &states {
            for &b in &states {
                assert_eq!(a.xor(b), b.xor(a), "XOR commutativity {a:?} vs {b:?}");
                for &c in &states {
                    assert_eq!(
                        a.xor(b).xor(c),
                        a.xor(b.xor(c)),
                        "XOR associativity {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    // ---------- NOT truth table ----------

    #[test]
    fn three_vl_not_inverts_only_definitive() {
        assert_eq!(ThreeValued::True.not(), ThreeValued::False);
        assert_eq!(ThreeValued::False.not(), ThreeValued::True);
        // Critical 3VL property: NOT Unknown = Unknown (NOT False).
        // A 2VL-projection bug would flip this to True; the test
        // pins the correct semantics.
        assert_eq!(ThreeValued::Unknown.not(), ThreeValued::Unknown);
    }

    // ---------- IS NULL tunnels through 3VL ----------

    #[test]
    fn is_null_returns_true_for_null_false_for_non_null() {
        // Critical: IS NULL never returns Unknown — it tunnels
        // through 3VL per Cypher 9 §6.2.5.
        assert_eq!(ThreeValued::is_null(&Value::Null), ThreeValued::True);
        assert_eq!(
            ThreeValued::is_null(&Value::Boolean(false)),
            ThreeValued::False
        );
        assert_eq!(ThreeValued::is_null(&Value::Integer(0)), ThreeValued::False);
    }

    // ---------- Lift Value → ThreeValued ----------

    #[test]
    fn lift_value_to_threevalued() {
        assert_eq!(
            ThreeValued::from_value(&Value::Boolean(true)),
            ThreeValued::True
        );
        assert_eq!(
            ThreeValued::from_value(&Value::Boolean(false)),
            ThreeValued::False
        );
        // CRITICAL: Null lifts to Unknown — distinct from Boolean(false).
        assert_eq!(ThreeValued::from_value(&Value::Null), ThreeValued::Unknown);
    }

    #[test]
    fn from_value_strict_returns_none_for_non_boolean_non_null() {
        assert_eq!(ThreeValued::from_value_strict(&Value::Integer(42)), None);
        assert_eq!(
            ThreeValued::from_value_strict(&Value::String("nope".into())),
            None
        );
        // Boolean and Null pass through.
        assert_eq!(
            ThreeValued::from_value_strict(&Value::Boolean(true)),
            Some(ThreeValued::True)
        );
        assert_eq!(
            ThreeValued::from_value_strict(&Value::Null),
            Some(ThreeValued::Unknown)
        );
    }

    // ---------- passes_filter discipline ----------

    #[test]
    fn passes_filter_treats_unknown_as_false() {
        // Critical: WHERE filters rows where the predicate is Unknown.
        // A bug that lets Unknown through would surface NULL-rows in
        // strict-WHERE queries, violating Cypher 9 §6.2.
        assert!(ThreeValued::True.passes_filter());
        assert!(!ThreeValued::False.passes_filter());
        assert!(!ThreeValued::Unknown.passes_filter());
    }

    // ---------- 3VL is associative + commutative ----------

    #[test]
    fn and_is_commutative_and_associative() {
        let states = [ThreeValued::True, ThreeValued::False, ThreeValued::Unknown];
        for &a in &states {
            for &b in &states {
                // Commutativity.
                assert_eq!(a.and(b), b.and(a), "AND commutativity {a:?} vs {b:?}");
                for &c in &states {
                    // Associativity.
                    assert_eq!(
                        a.and(b).and(c),
                        a.and(b.and(c)),
                        "AND associativity {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn or_is_commutative_and_associative() {
        let states = [ThreeValued::True, ThreeValued::False, ThreeValued::Unknown];
        for &a in &states {
            for &b in &states {
                assert_eq!(a.or(b), b.or(a), "OR commutativity");
                for &c in &states {
                    assert_eq!(a.or(b).or(c), a.or(b.or(c)), "OR associativity");
                }
            }
        }
    }

    #[test]
    fn de_morgan_holds_for_definitive_states() {
        // De Morgan's laws hold for True / False but DEFAULTS to a
        // weaker form for Unknown — `NOT (a AND b) = (NOT a) OR (NOT
        // b)` requires `Unknown.not() == Unknown` which we have.
        for &a in &[ThreeValued::True, ThreeValued::False, ThreeValued::Unknown] {
            for &b in &[ThreeValued::True, ThreeValued::False, ThreeValued::Unknown] {
                assert_eq!(
                    a.and(b).not(),
                    a.not().or(b.not()),
                    "De Morgan AND→OR for {a:?},{b:?}"
                );
                assert_eq!(
                    a.or(b).not(),
                    a.not().and(b.not()),
                    "De Morgan OR→AND for {a:?},{b:?}"
                );
            }
        }
    }
}
