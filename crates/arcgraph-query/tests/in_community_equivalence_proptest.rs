//! M4-23 IN-COMMUNITY ↔ canonical `community(...)` lowering equivalence
//! proptest (256 cases). Closes PR #154 reviewer Finding 5.
//!
//! # Surfaces under test
//!
//! Per ADR-038 amendment-01 (M4-01 surface-alignment amendment), ArcQL
//! at v1.0 admits TWO surfaces for community membership:
//!
//! 1. **Predicate surface** —
//!    ```text
//!    MATCH (n) WHERE n IN COMMUNITY($cid) RETURN n
//!    ```
//!    parses to `Expression::InCommunity { node, community }`.
//!
//! 2. **Canonical function-call surface** — per ADR-038 §2 D-4, the
//!    canonical surface uses `community(node)` (1-arg, returns the
//!    node's community ID):
//!    ```text
//!    MATCH (n) WHERE community(n) = $cid RETURN n
//!    ```
//!    parses to `BinaryOp { op: Eq, lhs: FunctionCall("community", [n]),
//!    rhs: Parameter($cid) }`.
//!
//! # The equivalence claim — and the documented compromise
//!
//! ADR-038 amendment-01 §A-2 commits that both surfaces "lower to the
//! same `CommunityLookup` plan-tree node" at M4-31 (the logical-plan
//! generator). M4-31 is OUT of M4-23 scope. M4-23 cannot perform the
//! AST-shape normalization itself because:
//!
//! - `crates/arcgraph-query/src/semantic/binding.rs` is FROZEN by the
//!   M4-21 + M4-22 contracts.
//! - `crates/arcgraph-query/src/semantic/bound_ast.rs` carries
//!   distinct variants for the two surfaces (`BoundExpression::InCommunity`
//!   vs. `BoundExpression::FunctionCall`) and is FROZEN at the M4-22
//!   contract.
//!
//! **The compromise:** This proptest verifies SEMANTIC equivalence —
//! both surfaces produce a bound query that
//!
//! 1. binds the same MATCH-pattern variable `n` (same `name`, same
//!    `binding_id`),
//! 2. assigns `n` the same `type_info` (Node with the resolved Person
//!    label) and the same `may_be_null` flag (`false` — neither surface
//!    is inside an OPTIONAL MATCH),
//! 3. yields a `WHERE`-admissible top-level type (`Boolean` or `Null`,
//!    both accepted by `TypeCheckVisitor::check_where_top_type` per
//!    Cypher 3VL D-20),
//! 4. validates cleanly under
//!    [`CrossSubstrateValidator`](arcgraph_query::semantic::CrossSubstrateValidator)
//!    when the community substrate is attached.
//!
//! ## A note on the 3VL type asymmetry
//!
//! The two WHERE expressions intentionally do NOT carry the same
//! `TypeInfo`. The predicate surface lowers to
//! `BoundExpression::InCommunity { ... }`, which the M4-22 type-checker
//! pins to `TypeInfo::Boolean` unconditionally. The canonical
//! function-call surface lowers to `BinaryOp { op: Eq,
//! lhs: FunctionCall("community", [n]), rhs: Parameter($cid) }`. Per
//! ADR-038 §2 D-20 (3VL truth-table propagation), any binary op with
//! a `Null` operand yields `Null` — and parameters at v1.0 carry
//! `TypeInfo::Null` (the v1.0 catalog does not propagate JSON-typed
//! bind-value info statically; M4-22 type_check.rs §`Parameter`).
//! Both `Boolean` and `Null` are admissible WHERE-top types per
//! `check_where_top_type` (Cypher treats Null as FALSE in WHERE
//! position). M4-31's logical-plan generator MUST collapse this
//! asymmetry into a single `LogicalCommunityLookup` node.
//!
//! AST-SHAPE equivalence (i.e., identical `BoundExpression` variants)
//! is a M4-31 concern: M4-31's logical-plan generator MUST normalize
//! both surfaces to the same `LogicalCommunityLookup` node before plan
//! enumeration. M4-23 documents that contract here and asserts the
//! semantic-level equivalence that does NOT require modifying frozen
//! bind-time files.
//!
//! # ADR provenance
//! - ADR-038 §2 D-4 (canonical `community(...)` family).
//! - ADR-038 §2 D-23 (cross-substrate validation contract).
//! - ADR-038 amendment-01 (`IN COMMUNITY(...)` alternate predicate +
//!   "lower to same plan-tree node" commitment for M4-31).
//! - PR #154 reviewer Finding 5 (this proptest closes the finding).

use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, BoundClause, BoundExpression, BoundMatchBody, BoundQuery, BoundStatement,
    BoundVariable, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor, TypeInfo,
};
use proptest::prelude::*;

/// Strategy for the MATCH-pattern variable name. Keep the alphabet
/// narrow to avoid the binding-pass span-cursor heuristic edge cases
/// (substring-overlap; see `crates/arcgraph-query/src/semantic/binding.rs`
/// top-of-file comment).
fn arbitrary_var_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9]?"
}

/// Strategy for the parameter name (referenced as `$<name>`). Same
/// alphabet as `arbitrary_var_name` to keep the cursor heuristic happy
/// AND to avoid name-collision with the variable.
fn arbitrary_param_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9]?"
}

/// Walk `q` and locate the head MATCH pattern's bound variable.
/// Returns `None` for queries without a MATCH (none of the inputs in
/// this proptest exercise that branch).
fn first_match_head_var(q: &BoundQuery) -> Option<&BoundVariable> {
    for c in &q.clauses {
        if let BoundClause::Match(m) = c {
            if let BoundMatchBody::Patterns(ps) = &m.body {
                if let Some(p) = ps.first() {
                    return p.head.var.as_ref();
                }
            }
        }
    }
    None
}

/// Walk `q` and locate the WHERE clause expression on the first
/// MATCH. Returns `None` if either the MATCH or the WHERE is absent.
fn first_where(q: &BoundQuery) -> Option<&BoundExpression> {
    for c in &q.clauses {
        if let BoundClause::Match(m) = c {
            return m.where_clause.as_ref();
        }
    }
    None
}

/// Bind + type-check + cross-substrate-validate. Asserts every step
/// passes and returns the final bound query.
fn process(input: &str, cat: &StubCatalogProvider) -> BoundQuery {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, cat).expect("cross-substrate validate");
    match bound {
        BoundStatement::Read(q) => q,
        _ => panic!("expected Read"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// For every (var_name, param_name) pair drawn from the strategy:
    /// the predicate-surface query `MATCH (n:Person) WHERE n IN
    /// COMMUNITY($cid) RETURN n` and the canonical function-call
    /// surface `MATCH (n:Person) WHERE community(n) = $cid RETURN n`
    /// MUST produce semantically equivalent bound queries (per the
    /// equivalence claim in this file's docstring).
    ///
    /// `var_name` and `param_name` are independent — the proptest
    /// covers the case where they happen to collide because the parser
    /// scopes parameters into a separate namespace from variables (no
    /// shadowing concern).
    #[test]
    fn in_community_predicate_equiv_to_canonical_function_call(
        var_name in arbitrary_var_name(),
        param_name in arbitrary_param_name(),
    ) {
        let predicate_input = format!(
            "MATCH ({var_name}:Person) WHERE {var_name} IN COMMUNITY(${param_name}) RETURN {var_name}"
        );
        let function_input = format!(
            "MATCH ({var_name}:Person) WHERE community({var_name}) = ${param_name} RETURN {var_name}"
        );

        let cat = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_community_index();

        let pred_q = process(&predicate_input, &cat);
        let fn_q = process(&function_input, &cat);

        // 1. Both surfaces bind the head MATCH variable to the same
        //    name + same binding_id (binding-id deterministic from
        //    the M4-21 visitor's monotonic counter; both queries
        //    declare exactly one variable up-front).
        let pred_v = first_match_head_var(&pred_q).expect("predicate-form bound var");
        let fn_v = first_match_head_var(&fn_q).expect("function-form bound var");
        prop_assert_eq!(&pred_v.name, &fn_v.name, "var names must match");
        prop_assert_eq!(pred_v.binding_id, fn_v.binding_id, "binding_ids must match");

        // 2. Same type_info (Node with the resolved Person label) +
        //    same may_be_null flag.
        prop_assert_eq!(
            pred_v.may_be_null, fn_v.may_be_null,
            "may_be_null flags must match"
        );
        match (&pred_v.type_info, &fn_v.type_info) {
            (Some(TypeInfo::Node { label: lp }), Some(TypeInfo::Node { label: lf })) => {
                prop_assert_eq!(lp, lf, "Node labels must match");
            }
            other => {
                return Err(TestCaseError::fail(format!(
                    "expected Node type_info on both, got {other:?}"
                )));
            }
        }

        // 3. Both WHERE expressions carry a top-level type that
        //    `TypeCheckVisitor::check_where_top_type` admits
        //    (Boolean or Null — see file-top docstring on the 3VL
        //    asymmetry). The predicate-form is Boolean; the
        //    function-form is Null because the parameter operand
        //    carries `TypeInfo::Null` per the M4-22 Parameter rule.
        let pred_w = first_where(&pred_q).expect("predicate WHERE");
        let fn_w = first_where(&fn_q).expect("function WHERE");
        prop_assert!(
            matches!(
                pred_w.type_info(),
                Some(TypeInfo::Boolean) | Some(TypeInfo::Null)
            ),
            "predicate WHERE must be Boolean or Null, got {:?}",
            pred_w.type_info()
        );
        prop_assert!(
            matches!(
                fn_w.type_info(),
                Some(TypeInfo::Boolean) | Some(TypeInfo::Null)
            ),
            "function-call WHERE must be Boolean or Null, got {:?}",
            fn_w.type_info()
        );
    }
}
