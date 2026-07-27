//! Integration tests for M4-22 (M4-02b) type-check pass.
//!
//! These tests exercise the public API end-to-end:
//!     `parse → BindingVisitor::bind → TypeCheckVisitor::check`.
//!
//! # Reserved-variant rejection pins
//!
//! The 6 reserved-variant rejection pins live HERE (not in
//! `src/semantic/type_check.rs::tests`) because they exercise the
//! full parse → bind → type-check pipeline, which is more
//! representative of how downstream callers (M4-31 logical planner;
//! M4-MCP wire layer) consume the surface than the unit-level
//! invocation. Each pin asserts:
//!
//!   (a) parse + bind succeeds (M4-21 doesn't reject the variant),
//!   (b) M4-22's type-check rejects with `ArcQLError::NotImplemented`,
//!   (c) the error message names the D-section + `target_version`
//!       per ADR-038 §2 D-16.
//!
//! Additional unit-level coverage of `TypeCheckVisitor` lives in
//! `src/semantic/type_check.rs::tests` (12 tests for individual
//! visitor methods + 3VL helpers).

use arcgraph_query::parse;
use arcgraph_query::semantic::{
    ArcQLError, BindingVisitor, BoundClause, BoundMatchBody, BoundProjectionKind, BoundStatement,
    StubCatalogProvider, TypeCheckVisitor,
};

/// Parse + bind + type-check; return the bound statement on success.
fn check_ok(input: &str, cat: &StubCatalogProvider) -> BoundStatement {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
    bound
}

/// Parse + bind + type-check; return the type-check errors.
fn check_err(input: &str, cat: &StubCatalogProvider) -> Vec<ArcQLError> {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect_err("expected type-check errors")
}

// =====================================================================
// Happy-path end-to-end
// =====================================================================

#[test]
fn end_to_end_match_where_return_succeeds() {
    let input =
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN b.name AS friend_name";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name"]);
    let bound = check_ok(input, &cat);
    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!("expected Read"),
    };

    // Confirm `a` got Node type-info populated.
    let m = match &q.clauses[0] {
        BoundClause::Match(m) => m,
        _ => panic!(),
    };
    let path = match &m.body {
        BoundMatchBody::Patterns(ps) => &ps[0],
        _ => panic!(),
    };
    let a_var = path.head.var.as_ref().unwrap();
    assert!(a_var.type_info.is_some(), "a's type_info populated");
}

#[test]
fn end_to_end_optional_match_propagates_may_be_null() {
    let input = "MATCH (a:Person) OPTIONAL MATCH (b:Doc) RETURN a, b";
    let cat = StubCatalogProvider::new().with_labels(["Person", "Doc"]);
    let bound = check_ok(input, &cat);
    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!(),
    };
    let opt = match &q.clauses[1] {
        BoundClause::Match(m) => m,
        _ => panic!(),
    };
    assert!(opt.is_optional);
    let b_var = match &opt.body {
        BoundMatchBody::Patterns(ps) => ps[0].head.var.as_ref().unwrap(),
        _ => panic!(),
    };
    assert!(b_var.may_be_null, "OPTIONAL MATCH var must be nullable");
}

#[test]
fn end_to_end_function_call_resolves_return_type() {
    let input = "MATCH (n:Person) RETURN count(n)";
    let cat = StubCatalogProvider::new().with_labels(["Person"]);
    let bound = check_ok(input, &cat);
    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!(),
    };
    let r = match q.clauses.last().unwrap() {
        BoundClause::Return(r) => r,
        _ => panic!(),
    };
    let e = match &r.items[0].kind {
        BoundProjectionKind::Expr(e) => e,
        _ => panic!(),
    };
    // count() returns Integer.
    use arcgraph_query::semantic::TypeInfo;
    assert_eq!(e.type_info(), Some(&TypeInfo::Integer));
}

#[test]
fn end_to_end_3vl_in_where_succeeds() {
    // `n.age = NULL` is admissible in WHERE (returns Null, treated as
    // FALSE). The type-check pass must NOT reject this query.
    let input = "MATCH (n:Person) WHERE n.age = NULL RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);
    let _ = check_ok(input, &cat);
}

#[test]
fn reserved_variant_quantified_length_range_rejected_with_d9() {
    // GQL `{1,3}` form. The grammar accepts the quantifier; M4-22
    // rejects with NotImplemented.
    let input = "MATCH (a)-[:KNOWS]->{1,3}(b) RETURN a, b";
    let cat = StubCatalogProvider::new().with_rel_types(["KNOWS"]);
    let errs = check_err(input, &cat);
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::NotImplemented { feature, .. }
                if feature == "LengthRange::Quantified ({N,M})"
        )),
        "expected NotImplemented for LengthRange::Quantified, got {errs:?}"
    );
}
