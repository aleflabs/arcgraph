//! GA Lane BINDER-VALIDATIONS — active end-to-end verification (#618).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! This slice adds the MISSING COMPILE-time semantic validations that
//! let the engine EXECUTE (or reject at the wrong phase) a class of
//! queries openCypher requires it to REJECT at bind / type-check time.
//! The openCypher TCK `Then a <X>Error should be raised at compile time:
//! <detail>` lines are the oracle (the W28 full-eligible harness scores
//! these by compile PHASE; this lane took the ratchet 421 → 457).
//!
//! Every test pairs, per validation class:
//! - an INVALID query → asserts it rejects at COMPILE (bind / type-check)
//!   time with the EXACT expected error variant (NOT merely "errored",
//!   and NOT `NotImplemented` / a runtime error — the wrong phase); and
//! - a VALID NEIGHBOUR → asserts it still binds + type-checks cleanly.
//!
//! The valid-neighbour assertions are as LOAD-BEARING as the rejections:
//! they are the per-test proof that the new check rejects EXACTLY the
//! invalid form and does NOT over-reject (the failure mode for this
//! lane). They mirror the before/after passing-set diff (zero regression)
//! at the unit level.

use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::semantic::{ArcQLError, BindingError, StubCatalogProvider, TypeCheckError};
use arcgraph_query::{ExplainError, QueryEngine};

/// Bind + type-check `query` through the full engine against an EMPTY
/// stub substrate (the W28 harness path) and return the `ArcQLError` on
/// rejection. Panics if the query did NOT error — that is the
/// `ExpectedErrorGotRows` failure the lane closes, so a test that calls
/// this expects a genuine rejection.
fn reject(query: &str) -> ArcQLError {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    match engine.execute(query, &substrate) {
        Ok(res) => panic!(
            "expected COMPILE-time rejection for `{query}`, but it returned {} row(s)",
            res.rows.len()
        ),
        Err(ExplainError::ArcQL(e)) => e,
        Err(other) => panic!(
            "expected `ExplainError::ArcQL(..)` for `{query}`, got a different ExplainError: {other}"
        ),
    }
}

/// Assert `query` binds + type-checks cleanly — i.e. it is NOT rejected
/// at compile/bind/type-check time. A later executor `NotImplemented`
/// (e.g. for a valid literal `SKIP 5`, whose executor support is a
/// separate slice) is ACCEPTED here: this proves only that THIS lane's
/// new validations do not OVER-reject the valid neighbour. An
/// `ArcQLError::Binding` / `TypeCheck` / `CrossSubstrate` / `LogicalPlan`
/// (a genuine compile rejection) FAILS the assertion (that would be
/// over-rejection); a `NotImplemented` / runtime error / success passes.
fn assert_not_compile_rejected(query: &str) {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    match engine.execute(query, &substrate) {
        Ok(_) => {}
        Err(ExplainError::ArcQL(ArcQLError::NotImplemented { .. })) => {}
        Err(ExplainError::ArcQL(ArcQLError::Internal { .. })) => {}
        Err(ExplainError::ArcQL(ArcQLError::ResourceExhausted { .. })) => {}
        Err(ExplainError::ExecutionEval(_))
        | Err(ExplainError::MissingParameter { .. })
        | Err(ExplainError::Substrate(_)) => {}
        Err(ExplainError::ArcQL(
            compile @ (ArcQLError::Binding(_)
            | ArcQLError::TypeCheck(_)
            | ArcQLError::CrossSubstrate(_)
            | ArcQLError::LogicalPlan(_)),
        )) => panic!(
            "OVER-REJECTION: valid neighbour `{query}` was wrongly rejected at compile time: {compile}"
        ),
        Err(other) => panic!("unexpected error for valid neighbour `{query}`: {other}"),
    }
}

fn assert_binding(query: &str, pred: impl Fn(&BindingError) -> bool, what: &str) {
    match reject(query) {
        ArcQLError::Binding(b) if pred(&b) => {}
        other => panic!("expected BindingError ({what}) for `{query}`, got: {other}"),
    }
}

fn assert_type_check(query: &str, pred: impl Fn(&TypeCheckError) -> bool, what: &str) {
    match reject(query) {
        ArcQLError::TypeCheck(t) if pred(&t) => {}
        other => panic!("expected TypeCheckError ({what}) for `{query}`, got: {other}"),
    }
}

// ===================================================================
// Class A — duplicate / rebound variable (VariableTypeConflict /
// RelationshipUniquenessViolation / VariableAlreadyBound).
// ===================================================================

#[test]
fn class_a_rel_var_reused_as_node_is_type_conflict() {
    // Match1 [7] — relationship var `r` reused as a node in a later MATCH.
    assert_binding(
        "MATCH ()-[r]-()\nMATCH (r)\nRETURN r",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict rel→node",
    );
    // Match1 [9] — relationship var reused as a node in the SAME pattern.
    assert_binding(
        "MATCH ()-[r]-(r)\nRETURN r",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict rel→node same-pattern",
    );
}

#[test]
fn class_a_path_var_reused_is_type_conflict() {
    // Match6 [23] — path var `p` reused as a node in the same pattern.
    assert_binding(
        "MATCH p = (p)-[]-()\nRETURN p",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict path→node",
    );
    // Match2 [10] — path var reused as a relationship in a later MATCH.
    assert_binding(
        "MATCH r = ()-[]-()\nMATCH ()-[r]-()\nRETURN r",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict path→rel",
    );
}

#[test]
fn class_a_value_bound_var_matched_as_node_is_type_conflict() {
    // Match1 [11] — a value-bound var (`WITH 1 AS n`) matched as a node.
    assert_binding(
        "WITH 1 AS n\nMATCH (n)\nRETURN n",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict value→node",
    );
    // Match3 [30] — a list-bound var matched as a node.
    assert_binding(
        "MATCH (n)\nWITH [n] AS users\nMATCH (users)-->(m)\nRETURN m",
        |b| matches!(b, BindingError::VariableTypeConflict { .. }),
        "VariableTypeConflict list→node",
    );
}

#[test]
fn class_a_rel_uniqueness_violation() {
    // Match3 [29] — the SAME relationship var twice in one pattern.
    assert_binding(
        "MATCH (a)-[r]->()-[r]->(a)\nRETURN r",
        |b| matches!(b, BindingError::RelationshipUniquenessViolation { .. }),
        "RelationshipUniquenessViolation",
    );
}

#[test]
fn class_a_valid_neighbours_not_over_rejected() {
    // A node var re-referenced across MATCHes is LEGAL (re-reference).
    assert_not_compile_rejected("MATCH (a)-[r]->(b)\nMATCH (a)-[s]->(c)\nRETURN a");
    // The same node var twice in one pattern is LEGAL (`(a)-[r]-(a)`).
    assert_not_compile_rejected("MATCH (a)-[r]-(a)\nRETURN a");
    // A node var passed through a WITH stays a node and re-references.
    assert_not_compile_rejected("MATCH (a)\nWITH a\nMATCH (a)-->(b)\nRETURN b");
    // Distinct relationship vars in one pattern are LEGAL.
    assert_not_compile_rejected("MATCH (a)-[r]->()-[s]->(b)\nRETURN r, s");
}

// ===================================================================
// Class B — aggregation in an illegal position (InvalidAggregation /
// NestedAggregation).
// ===================================================================

#[test]
fn class_b_aggregation_in_where_rejected() {
    assert_binding(
        "MATCH (a)\nWHERE count(a) > 10\nRETURN a",
        |b| matches!(b, BindingError::InvalidAggregation { .. }),
        "InvalidAggregation WHERE",
    );
}

#[test]
fn class_b_aggregation_in_order_by_rejected() {
    // ReturnOrderBy2 [14].
    assert_binding(
        "MATCH (n)\nRETURN n.num1\n  ORDER BY max(n.num2)",
        |b| matches!(b, BindingError::InvalidAggregation { .. }),
        "InvalidAggregation ORDER BY",
    );
    // WithOrderBy2 [25].
    assert_binding(
        "MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(1)\nRETURN foo AS foo",
        |b| matches!(b, BindingError::InvalidAggregation { .. }),
        "InvalidAggregation WITH ORDER BY",
    );
}

#[test]
fn class_b_aggregation_in_list_comprehension_rejected() {
    // List12 [7].
    assert_binding(
        "MATCH (n)\nRETURN [x IN [1, 2, 3] | count(*)]",
        |b| matches!(b, BindingError::InvalidAggregation { .. }),
        "InvalidAggregation list comprehension",
    );
}

#[test]
fn class_b_nested_aggregation_rejected() {
    // Return6 [14] — was WrongErrorPhase (NotImplemented) before.
    assert_binding(
        "RETURN count(count(*))",
        |b| matches!(b, BindingError::NestedAggregation { .. }),
        "NestedAggregation",
    );
}

#[test]
fn class_b_valid_neighbours_not_over_rejected() {
    // A non-aggregating WHERE is fine.
    assert_not_compile_rejected("MATCH (a)\nWHERE a.age > 10\nRETURN a");
    // A non-aggregating ORDER BY is fine.
    assert_not_compile_rejected("MATCH (n)\nRETURN n.num1\n  ORDER BY n.num2");
    // Aggregation in a RETURN projection term is fine (the legal home).
    assert_not_compile_rejected("MATCH (n)\nRETURN count(n)");
    // A non-aggregating list comprehension is fine.
    assert_not_compile_rejected("MATCH (n)\nRETURN [x IN [1, 2, 3] | x + 1]");
}

// ===================================================================
// Class C — projection / scope (ColumnNameConflict / NoVariablesInScope
// / NoExpressionAlias).
// ===================================================================

#[test]
fn class_c_duplicate_result_column_rejected() {
    // Return4 [10].
    assert_binding(
        "RETURN 1 AS a, 2 AS a",
        |b| matches!(b, BindingError::ColumnNameConflict { .. }),
        "ColumnNameConflict",
    );
}

#[test]
fn class_c_return_star_no_vars_rejected() {
    // Return7 [2].
    assert_binding(
        "MATCH ()\nRETURN *",
        |b| matches!(b, BindingError::NoVariablesInScope { .. }),
        "NoVariablesInScope",
    );
}

#[test]
fn class_c_unaliased_with_expr_rejected() {
    // With4 [5].
    assert_binding(
        "MATCH (a)\nWITH a, count(*)\nRETURN a",
        |b| matches!(b, BindingError::NoExpressionAlias { .. }),
        "NoExpressionAlias",
    );
}

#[test]
fn class_c_valid_neighbours_not_over_rejected() {
    // Distinct column names are fine.
    assert_not_compile_rejected("RETURN 1 AS a, 2 AS b");
    // RETURN * WITH in-scope variables is fine.
    assert_not_compile_rejected("MATCH (n)\nRETURN *");
    // An ALIASED WITH expression is fine.
    assert_not_compile_rejected("MATCH (a)\nWITH a, count(*) AS c\nRETURN a, c");
    // A bare passthrough `WITH a` (no alias) is fine.
    assert_not_compile_rejected("MATCH (a)\nWITH a\nRETURN a");
}

// ===================================================================
// Class D — operator / function argument types (InvalidArgumentType /
// FloatingPointOverflow / UndefinedVariable in a map literal).
// ===================================================================

#[test]
fn class_d_boolean_op_non_boolean_operand_rejected() {
    // Boolean1 [8] — non-boolean operand, both operands concrete.
    assert_type_check(
        "RETURN 123 AND true",
        |t| matches!(t, TypeCheckError::TypeMismatch { .. }),
        "TypeMismatch AND non-bool",
    );
    // Boolean2/3 [8] — the LOAD-BEARING null-operand case: `<non-bool> OR
    // null` must reject at COMPILE time even though the other operand is
    // a `null` literal (the 3VL short-circuit must NOT mask the type
    // error). Was `OK rows=1` before this slice.
    assert_type_check(
        "RETURN 123.4 OR null",
        |t| matches!(t, TypeCheckError::TypeMismatch { .. }),
        "TypeMismatch OR non-bool with null operand",
    );
    assert_type_check(
        "RETURN 123.4 XOR null",
        |t| matches!(t, TypeCheckError::TypeMismatch { .. }),
        "TypeMismatch XOR non-bool with null operand",
    );
}

#[test]
fn class_d_function_arg_kind_rejected() {
    // Graph4 [7] — type() on a node (rel-only).
    assert_type_check(
        "MATCH (r)\nRETURN type(r)",
        |t| matches!(t, TypeCheckError::FunctionArgumentTypeMismatch { name, .. } if name == "type"),
        "type() on node",
    );
    // Path3 [2]/[3] — length() on a node / relationship (path-only).
    assert_type_check(
        "MATCH (n)\nRETURN length(n)",
        |t| matches!(t, TypeCheckError::FunctionArgumentTypeMismatch { name, .. } if name == "length"),
        "length() on node",
    );
    assert_type_check(
        "MATCH ()-[r]->()\nRETURN length(r)",
        |t| matches!(t, TypeCheckError::FunctionArgumentTypeMismatch { name, .. } if name == "length"),
        "length() on rel",
    );
    // List6 [5] — size() on a path (list/string-only).
    assert_type_check(
        "MATCH p = (a)-[*]->(b)\nRETURN size(p)",
        |t| matches!(t, TypeCheckError::FunctionArgumentTypeMismatch { name, .. } if name == "size"),
        "size() on path",
    );
}

#[test]
fn class_d_float_overflow_rejected() {
    // Literals5 [27].
    assert_binding(
        "RETURN 1.34E999",
        |b| matches!(b, BindingError::FloatingPointOverflow { .. }),
        "FloatingPointOverflow",
    );
}

#[test]
fn class_d_map_unquoted_identifier_rejected() {
    // Literals8 [22] — `{k1: k2}` where `k2` is an unbound identifier.
    assert_binding(
        "RETURN {k1: k2} AS literal",
        |b| matches!(b, BindingError::UndeclaredVariable { name, .. } if name == "k2"),
        "UndeclaredVariable in map literal",
    );
}

#[test]
fn class_d_valid_neighbours_not_over_rejected() {
    // A boolean conjunction of booleans is fine.
    assert_not_compile_rejected("RETURN true AND false");
    // type() on a relationship is fine.
    assert_not_compile_rejected("MATCH ()-[r]->()\nRETURN type(r)");
    // length() on a path is fine.
    assert_not_compile_rejected("MATCH p = (a)-[*]->(b)\nRETURN length(p)");
    // size() on a list is fine.
    assert_not_compile_rejected("RETURN size([1, 2, 3])");
    // size() over a property (dynamic-typed) must NOT be over-rejected.
    assert_not_compile_rejected("MATCH (n)\nRETURN size(n.numbers)");
    // Property access on a node is fine.
    assert_not_compile_rejected("MATCH (n)\nRETURN n.num");
    // A bound identifier inside a map literal is fine.
    assert_not_compile_rejected("MATCH (n)\nRETURN {k1: n} AS m");
    // A finite float literal is fine.
    assert_not_compile_rejected("RETURN 1.5");
}

// ===================================================================
// Class E — SKIP / LIMIT constant-ness (NonConstantExpression /
// NegativeIntegerArgument / NonIntegerSkipLimit). All were
// WrongErrorPhase (NotImplemented) before this slice.
// ===================================================================

#[test]
fn class_e_non_constant_skip_limit_rejected() {
    // ReturnSkipLimit1 [5]/[10].
    assert_binding(
        "MATCH (n) RETURN n SKIP n.count",
        |b| matches!(b, BindingError::NonConstantExpression { clause, .. } if *clause == "SKIP"),
        "NonConstantExpression SKIP",
    );
    // ReturnSkipLimit2 [9].
    assert_binding(
        "MATCH (n) RETURN n LIMIT n.count",
        |b| matches!(b, BindingError::NonConstantExpression { clause, .. } if *clause == "LIMIT"),
        "NonConstantExpression LIMIT",
    );
}

#[test]
fn class_e_negative_skip_limit_rejected() {
    // ReturnSkipLimit1 [11] — `SKIP -1` (parsed as unary-neg of `1`).
    assert_binding(
        "MATCH (n)\nRETURN n\n  SKIP -1",
        |b| matches!(b, BindingError::NegativeIntegerArgument { clause, value, .. } if *clause == "SKIP" && *value == -1),
        "NegativeIntegerArgument SKIP",
    );
    // ReturnSkipLimit2 [12] — `LIMIT -1`.
    assert_binding(
        "MATCH (n)\nRETURN n\n  LIMIT -1",
        |b| matches!(b, BindingError::NegativeIntegerArgument { clause, value, .. } if *clause == "LIMIT" && *value == -1),
        "NegativeIntegerArgument LIMIT",
    );
}

#[test]
fn class_e_float_limit_rejected() {
    // ReturnSkipLimit2 [16] — `LIMIT 1.7`.
    assert_binding(
        "MATCH (n)\nRETURN n\n  LIMIT 1.7",
        |b| matches!(b, BindingError::NonIntegerSkipLimit { clause, actual, .. } if *clause == "LIMIT" && *actual == "float"),
        "NonIntegerSkipLimit LIMIT float",
    );
}

#[test]
fn class_e_valid_neighbours_not_over_rejected() {
    // A non-negative integer literal SKIP / LIMIT must NOT be rejected at
    // COMPILE time by this lane (the executor's own SKIP/dynamic-LIMIT
    // NotImplemented is a separate, pre-existing concern — accepted by
    // `assert_not_compile_rejected`).
    assert_not_compile_rejected("MATCH (n)\nRETURN n SKIP 5");
    assert_not_compile_rejected("MATCH (n)\nRETURN n LIMIT 5");
    // Zero is a valid non-negative count.
    assert_not_compile_rejected("MATCH (n)\nRETURN n LIMIT 0");
    // A parameter SKIP/LIMIT is a query-constant (valid at compile time).
    assert_not_compile_rejected("MATCH (n)\nRETURN n SKIP $s");
}
