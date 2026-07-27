//! ADR-152-amendment-02 (W28) — composite `List`-literal lift for the
//! write-op property-bag materialization path.
//!
//! The W26-θ write-op family (CREATE node — ADR-147; CREATE rel —
//! ADR-148; SET — ADR-150; MERGE — ADR-151) materializes literal
//! property values into runtime [`Value`]s via a per-op
//! `literal_to_value` helper. At v1.0-α those helpers narrowed EVERY
//! composite literal (`List` / `Map` / temporal / decimal) to a
//! rejected `None` per ADR-147 §D-4 + ADR-152 §"Forward-deferred" #5
//! (the parser admits the literal; the executor rejects it).
//!
//! This module is the single source for that materialization. It
//! **lifts the `List` narrowing**: a `List` literal whose every
//! (recursively-nested) element is a losslessly-round-tripping scalar
//! (`Null` / `Bool` / `Integer` / `Float` / `String`) or a nested
//! `List` of the same materializes to [`Value::List`]; it then
//! round-trips losslessly through the JSON property-bag path
//! ([`Value::to_json_value`] → `properties_to_property_data` →
//! `PropertyData::Blob` → scan-decode via [`Value::try_from_json_value`])
//! per ADR-152 §D-1/§D-3.
//!
//! # Scope (ADR-152-amendment-02 §D-2): `List` only
//!
//! - **`Map` is FENCED OUT — permanently, not deferred** (ADR-191 D-11).
//!   A `Value::Map` runtime variant now EXISTS (ADR-191), but openCypher
//!   forbids maps as stored property values ("Property values can only be
//!   of primitive types or arrays thereof. Encountered: Map"). So unlike
//!   `List` (which WAS correctly lifted — lists are valid property
//!   values), `Map` MUST STAY rejected here: lifting it would persist a
//!   map property and introduce a NEW conformance violation. The
//!   read/expression admittance of `Value::Map` (issue #356) does NOT
//!   extend to write-op property persistence.
//! - **`Temporal` / `LocalDateTime` / `Date` / `Duration` / `Decimal`
//!   are deferred too.** Although `eval.rs`'s read-path lifts them, their
//!   JSON round-trip is **LOSSY** at v1.0-α: they encode as bare
//!   ISO-8601 / decimal strings ([`Value::to_json_value`]) and decode
//!   back as [`Value::String`] ([`Value::try_from_json_value`]), so a
//!   persisted `date('2026-01-01')` would silently re-materialize as the
//!   string `"2026-01-01"`. The lossless typed `{ "_type": .., "value":
//!   .. }` encoding is v1.2-reserved (see `value.rs` §"Wire shape" +
//!   issue #356). Admitting a lossy round-trip would violate the
//!   strong-oracle discipline, so we reject (the caller surfaces a clean
//!   error) rather than corrupt. A `List` carrying any such element is
//!   rejected for the same reason — we never silently persist a
//!   `Value::String` where the user wrote a temporal literal.
//!
//! Strict-schema property typing remains tracked separately from this
//! literal-lifting path.

use crate::ast::{Expression, Literal, UnaryOp};
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate, negate_const_value};
use crate::executor::ops::schema_index;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// Materialize a [`Literal`] into a runtime [`Value`] for write-op
/// property-bag persistence.
///
/// Returns `None` for any literal that may not be persisted as a
/// property value: `Map` (FENCED — openCypher forbids map property
/// values, ADR-191 D-11; rejected even though `Value::Map` now exists)
/// and the temporal / decimal family (lossy JSON string round-trip at
/// v1.0-α, per ADR-152-amendment-02 §D-2). `List` recurses element-wise
/// via [`list_element_value`] and rejects (`None`) the whole list if ANY
/// element is non-liftable.
///
/// `None` is the caller's signal to surface its existing literal-only
/// rejection error: the type-check pass admits the OUTER property value
/// when it is a `BoundExpression::Literal` but does **not** recurse into
/// a list's inner elements, so a list with a non-literal element (a
/// function call, a parameter) or a deferred-composite element is
/// genuinely reachable here.
pub(super) fn literal_value(lit: &Literal) -> Option<Value> {
    Some(match lit {
        Literal::Null => Value::Null,
        Literal::Bool(b) => Value::Boolean(*b),
        Literal::Integer(i) => Value::Integer(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        // ADR-152-amendment-02 §D-1 — `List` lifts when every element
        // is a losslessly-round-tripping scalar or a nested list of the
        // same. A single non-liftable element rejects the whole list
        // (`collect::<Option<Vec<_>>>()` short-circuits to `None`) — we
        // never silently persist a `Value::Null` hole for an element we
        // cannot round-trip.
        Literal::List(elems) => {
            let lifted: Option<Vec<Value>> = elems.iter().map(list_element_value).collect();
            Value::List(lifted?)
        }
        // `Map` is FENCED (ADR-191 D-11): a `Value::Map` runtime variant
        // now exists, but openCypher forbids maps as STORED property
        // values, so the write-op path keeps rejecting them — lifting it
        // (the way `List` was correctly lifted) would persist a map
        // property = a NEW conformance violation. The temporal / decimal
        // family is deferred for a different reason (ADR-152-amendment-02
        // §D-2): it round-trips LOSSILY through JSON (encode → string,
        // decode → `Value::String`), forward-pinned to the v1.2 typed
        // encoding (issue #356). Both `return None`; the caller surfaces
        // a clean literal-only rejection error.
        Literal::Map(_)
        | Literal::Temporal(_)
        | Literal::LocalDateTime(_)
        | Literal::Date(_)
        | Literal::Duration(_)
        | Literal::Decimal(_) => return None,
    })
}

/// Lift one `List` element to a [`Value`].
///
/// A list literal carries raw AST [`Expression`]s (the parser emits the
/// inner shape directly — it does NOT pre-lower them to
/// `BoundExpression`s; see `eval.rs::literal_to_value`). Only a
/// `Expression::Literal` element is liftable; any other shape (function
/// call, parameter, property access, …) returns `None` so the enclosing
/// list is rejected rather than persisting a silent `Value::Null` hole.
fn list_element_value(e: &Expression) -> Option<Value> {
    match e {
        Expression::Literal(lit) => literal_value(lit),
        // #870 — a negative/unary-`+` numeric literal element parses as
        // `UnaryOp(Neg/Pos, <numeric literal>)` (`[-5]` ⇒ `[UnaryOp(Neg,5)]`);
        // fold it to a persistable constant instead of rejecting the whole
        // list. `negate_const_value` returns `None` for a non-numeric operand,
        // preserving the reject-non-liftable contract.
        Expression::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => negate_const_value(list_element_value(operand)?),
        Expression::UnaryOp {
            op: UnaryOp::Pos,
            operand,
        } => list_element_value(operand),
        _ => None,
    }
}

/// Materialize a CONSTANT [`BoundExpression`] property value into a runtime
/// [`Value`] for write-op persistence (CREATE / SET / MERGE), accepting both
/// a bare `Literal` and a `UnaryOp(Neg/Pos, <numeric literal>)` — i.e. a
/// NEGATIVE numeric literal, which parses as a `UnaryOp`, not a `Literal`
/// (#870 — `CREATE (n {x: -5})` was rejected as "not a literal"). Returns
/// `None` for any other shape (a genuine non-literal property value, or a
/// fenced/deferred composite), which the caller surfaces as its existing
/// literal-only rejection error. The single BoundExpression entry point the
/// three write ops (`create_node` / `create_rel` / `set`) share.
pub(super) fn bound_literal_value(e: &BoundExpression) -> Option<Value> {
    match e {
        BoundExpression::Literal { value, .. } => literal_value(value),
        BoundExpression::ListLiteral { elements, .. } => {
            let lifted: Option<Vec<Value>> = elements.iter().map(bound_literal_value).collect();
            Some(Value::List(lifted?))
        }
        BoundExpression::UnaryOp {
            op: UnaryOp::Neg,
            operand,
            ..
        } => negate_const_value(bound_literal_value(operand)?),
        BoundExpression::UnaryOp {
            op: UnaryOp::Pos,
            operand,
            ..
        } => bound_literal_value(operand),
        _ => None,
    }
}

/// ADR-147-amendment-03 (D-1) — hard cap on the element count of a list
/// value evaluated for a CREATE property. This gates the FINAL
/// materialized list; it is NOT the concat-amplification backstop. A
/// bracketed `BinOp::Add` doubling tree (`{x: (($a+$a)+($a+$a))+…}`) would
/// allocate every intermediate up to ~2^depth elements and OOM BEFORE
/// this result-level check runs — the real backstop for that amplifier is
/// the per-op cap enforced inside `eval::add_or_concat`
/// (`eval::MAX_CONCAT_LIST_LEN`, kept equal to this const so a value
/// clearing every intermediate also clears this result gate; see
/// ADR-147-amendment-03 §B1). This const remains the defense-in-depth
/// bound on any single non-amplifying list value (e.g. a huge list
/// arriving via `$p`). Back-of-envelope: 1M
/// `Value::Integer`s ≈ 24 MB in-memory before the JSON-blob encode. The
/// read-path `range()` cap (`eval::MAX_RANGE_LEN`) matches this bound.
pub(crate) const MAX_CREATE_PROP_LIST_LEN: usize = 1_000_000;

/// Materialize a CREATE / CREATE-path property bag into runtime `Value`s
/// (ADR-147-amendment-03 §D-1) — the SINGLE shared implementation for
/// `CreateSpineOp` (the live path), `CreateNodeOp`, and `CreateRelOp`.
///
/// Const literals take the [`bound_literal_value`] fast path (which keeps
/// the ADR-191/D-11 map fence + Null-hole fence for the const case);
/// everything else is [`evaluate`]d against the current work `row` + the
/// parameter bag — the SAME engine `UnwindOp` / `ProjectOp` use.
///
/// # The runtime value-type gate (the load-bearing guard)
///
/// AST-shape admittance at type-check is NOT sufficient: `$p` / `r.x`
/// resolve at RUNTIME to arbitrary `Value`s, and `$p` bound to a map
/// passes the type-check shape check. This gate enforces the SAME fence
/// the const path enforces, at the value layer, BEFORE the substrate
/// write:
/// - `Value::Null` → property ABSENT (openCypher: a null-valued property
///   is not stored, not a stored-null).
/// - `Value::Map` / `Node` / `Relationship` / `Path` → typed execution
///   error (openCypher forbids map/entity property values; ADR-191 D-11).
/// - temporal / decimal family → typed error (their JSON round-trip is
///   LOSSY at v1.0-α; `literal_lift` §D-2 / issue #356).
/// - `Value::List` → each element recursively vetted + length-capped at
///   [`MAX_CREATE_PROP_LIST_LEN`].
///
/// `context` is a caller-supplied label (e.g. `"CreateSpineOp: node"`)
/// prefixed onto error messages.
pub(super) fn materialize_create_properties(
    context: &str,
    properties: &[(String, BoundExpression)],
    row: &[Value],
    row_schema: &[BindingId],
    params: &Parameters,
) -> Result<Vec<(String, Value)>, ExecutionError> {
    // Owned clone so the lookup closure does not borrow `row_schema`
    // (which the caller may still hold mutably) — mirrors unwind.rs:173.
    let schema = row_schema.to_vec();
    let lookup = |b: BindingId| schema_index(&schema, b);

    let mut materialized = Vec::with_capacity(properties.len());
    for (key, expr) in properties {
        // Const fast path: pure literals fold without touching `evaluate`
        // and keep the const-path map / Null-hole fence.
        let value = match bound_literal_value(expr) {
            Some(v) => v,
            None => evaluate(expr, row, &lookup, params)
                .map_err(|e| ExecutionError::Eval(format!("{context} property `{key}`: {e}")))?,
        };

        // === RUNTIME VALUE-TYPE GATE (ADR-147-amendment-03 §D-1) ===
        match &value {
            // openCypher: a null-valued property is ABSENT, not stored.
            Value::Null => continue,
            Value::List(items) => {
                if items.len() > MAX_CREATE_PROP_LIST_LEN {
                    return Err(ExecutionError::Eval(format!(
                        "{context} property `{key}`: list length {} exceeds cap {MAX_CREATE_PROP_LIST_LEN}",
                        items.len()
                    )));
                }
                if let Some(bad) = first_non_persistable(items) {
                    return Err(ExecutionError::Eval(format!(
                        "{context} property `{key}`: list contains {bad}, not a valid stored \
                         property element (openCypher forbids map/entity/temporal property \
                         values; ADR-191 D-11 / lossy-round-trip §D-2)"
                    )));
                }
            }
            other => {
                if let Some(bad) = non_persistable_scalar(other) {
                    return Err(ExecutionError::Eval(format!(
                        "{context} property `{key}`: {bad} is not a valid stored property value \
                         (openCypher forbids map/entity property values; ADR-191 D-11 / \
                         lossy-round-trip §D-2)"
                    )));
                }
            }
        }
        materialized.push((key.clone(), value));
    }
    Ok(materialized)
}

/// Return the offending type name if a scalar `Value` may NOT be
/// persisted as a property value (ADR-147-amendment-03 §D-1 value gate).
///
/// Persistable scalars: `Boolean` / `Integer` / `Float` / `String`
/// (`Null` is handled as absence by the caller, `List` is handled
/// recursively). NON-persistable: entity shapes (`Node` / `Relationship`
/// / `Path`), `Map` (openCypher forbids map property values, ADR-191
/// D-11), and the temporal / decimal family (lossy JSON round-trip at
/// v1.0-α, §D-2 / issue #356).
fn non_persistable_scalar(v: &Value) -> Option<&'static str> {
    match v {
        Value::Null
        | Value::Boolean(_)
        | Value::Integer(_)
        | Value::Float(_)
        | Value::String(_) => None,
        Value::List(_) => None, // vetted recursively by the caller
        Value::Node(_) => Some("a Node"),
        Value::Relationship(_) => Some("a Relationship"),
        Value::Path(_) => Some("a Path"),
        Value::Map(_) => Some("a Map"),
        Value::Temporal(_) => Some("a Timestamp (lossy round-trip, deferred)"),
        Value::LocalDateTime(_) => Some("a LocalDateTime (lossy round-trip, deferred)"),
        Value::Date(_) => Some("a Date (lossy round-trip, deferred)"),
        Value::Duration(_) => Some("a Duration (lossy round-trip, deferred)"),
        Value::Decimal(_) => Some("a Decimal (lossy round-trip, deferred)"),
    }
}

/// Recurse a list, returning the first element (name) that may not be
/// persisted as a property element. A `Value::Null` INSIDE a list is a
/// non-persistable hole (unlike a top-level null property, which is
/// absence) — the const path rejects lists carrying a null element for
/// the same reason (`literal_value` never persists a `Value::Null` hole).
fn first_non_persistable(items: &[Value]) -> Option<&'static str> {
    for item in items {
        match item {
            Value::Null => return Some("a null hole"),
            Value::List(inner) => {
                if inner.len() > MAX_CREATE_PROP_LIST_LEN {
                    return Some("an over-length nested list");
                }
                if let Some(bad) = first_non_persistable(inner) {
                    return Some(bad);
                }
            }
            other => {
                if let Some(bad) = non_persistable_scalar(other) {
                    return Some(bad);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Expression {
        Expression::Literal(Literal::Integer(n))
    }
    fn string(s: &str) -> Expression {
        Expression::Literal(Literal::String(s.to_string()))
    }

    #[test]
    fn scalars_lift_identity() {
        assert_eq!(literal_value(&Literal::Null), Some(Value::Null));
        assert_eq!(
            literal_value(&Literal::Bool(true)),
            Some(Value::Boolean(true))
        );
        assert_eq!(
            literal_value(&Literal::Integer(42)),
            Some(Value::Integer(42))
        );
        assert_eq!(
            literal_value(&Literal::String("x".into())),
            Some(Value::String("x".into()))
        );
        match literal_value(&Literal::Float(2.5)) {
            Some(Value::Float(f)) => assert!((f - 2.5).abs() < 1e-9),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn list_of_scalars_lifts() {
        let lit = Literal::List(vec![string("a"), string("b"), string("c")]);
        assert_eq!(
            literal_value(&lit),
            Some(Value::List(vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
            ]))
        );
    }

    #[test]
    fn heterogeneous_list_preserves_element_types() {
        // Cypher 9 §3.5 admits heterogeneous lists; the lift preserves
        // each element's runtime type.
        let lit = Literal::List(vec![
            int(1),
            string("x"),
            Expression::Literal(Literal::Bool(true)),
        ]);
        assert_eq!(
            literal_value(&lit),
            Some(Value::List(vec![
                Value::Integer(1),
                Value::String("x".into()),
                Value::Boolean(true),
            ]))
        );
    }

    #[test]
    fn nested_list_recurses() {
        // [[1, 2], [3]]
        let lit = Literal::List(vec![
            Expression::Literal(Literal::List(vec![int(1), int(2)])),
            Expression::Literal(Literal::List(vec![int(3)])),
        ]);
        assert_eq!(
            literal_value(&lit),
            Some(Value::List(vec![
                Value::List(vec![Value::Integer(1), Value::Integer(2)]),
                Value::List(vec![Value::Integer(3)]),
            ]))
        );
    }

    #[test]
    fn empty_list_lifts_to_empty_value_list() {
        assert_eq!(
            literal_value(&Literal::List(vec![])),
            Some(Value::List(vec![]))
        );
    }

    #[test]
    fn map_is_fenced_from_property_persistence() {
        // ADR-191 D-11 — `Value::Map` now EXISTS, but a map MUST STAY
        // rejected as a write-op property value (openCypher forbids map
        // property values). Both the empty map and a populated map reject.
        assert_eq!(literal_value(&Literal::Map(vec![])), None);
        let populated = Literal::Map(vec![("a".to_string(), int(1))]);
        assert_eq!(
            literal_value(&populated),
            None,
            "a populated map literal must NOT lift to a persistable property value"
        );
    }

    #[test]
    fn temporal_family_is_deferred_lossy_round_trip() {
        // ADR-152-amendment-02 §D-2 — these round-trip LOSSILY through
        // JSON (encode → string, decode → Value::String); deferred to
        // the v1.2 typed encoding (issue #356).
        let d = arcgraph_core::Date::new(2026, 1).expect("valid date");
        assert_eq!(literal_value(&Literal::Date(d)), None);
    }

    #[test]
    fn list_with_deferred_element_rejects_whole_list() {
        // A list carrying a temporal element is rejected wholesale — we
        // never silently persist a String where the user wrote a date.
        let d = arcgraph_core::Date::new(2026, 1).expect("valid date");
        let lit = Literal::List(vec![int(1), Expression::Literal(Literal::Date(d))]);
        assert_eq!(literal_value(&lit), None);
    }

    #[test]
    fn list_with_non_literal_element_rejects_whole_list() {
        // A non-literal list element (here a bare identifier/parameter-
        // shaped expression the type-check does not recurse into)
        // rejects the whole list rather than persisting a Null hole.
        let lit = Literal::List(vec![int(1), Expression::Parameter("p".into())]);
        assert_eq!(literal_value(&lit), None);
    }
}
