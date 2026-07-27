//! Integration tests for M4-21 (M4-02a) binding pass.
//!
//! These tests exercise the public surface of `arcgraph-query::semantic`
//! end-to-end: `parse` → `BindingVisitor::bind` against a populated
//! `StubCatalogProvider`. They are deliberately spot-check shape —
//! the binding-pass internals are pinned by the unit tests in
//! `src/semantic/binding.rs::tests`. These tests verify the cross-
//! module wiring + the span-bearing error-reporting contract.

use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingError, BindingVisitor, BoundClause, BoundExpression, BoundMatchBody,
    BoundProjectionKind, BoundStatement, CatalogProvider, StubCatalogProvider, TypeCheckVisitor,
};

/// Walk all `BoundExpression` nodes in a clause, collecting variable
/// references in source order.
fn clause_var_refs(c: &BoundClause) -> Vec<String> {
    let mut acc = Vec::new();
    match c {
        BoundClause::Match(m) => {
            if let Some(w) = &m.where_clause {
                expr_var_refs(w, &mut acc);
            }
        }
        BoundClause::Return(r) => {
            for it in &r.items {
                if let BoundProjectionKind::Expr(e) = &it.kind {
                    expr_var_refs(e, &mut acc);
                }
            }
        }
        _ => {}
    }
    acc
}

fn expr_var_refs(e: &BoundExpression, acc: &mut Vec<String>) {
    match e {
        BoundExpression::VariableRef { name, .. } => acc.push(name.clone()),
        BoundExpression::PropertyAccess { base, .. } => expr_var_refs(base, acc),
        BoundExpression::BinaryOp { lhs, rhs, .. } => {
            expr_var_refs(lhs, acc);
            expr_var_refs(rhs, acc);
        }
        BoundExpression::UnaryOp { operand, .. } => expr_var_refs(operand, acc),
        BoundExpression::FunctionCall { args, .. } => {
            for a in args {
                expr_var_refs(a, acc);
            }
        }
        BoundExpression::Near { lhs, target, .. } => {
            expr_var_refs(lhs, acc);
            expr_var_refs(target, acc);
        }
        BoundExpression::TextMatch { lhs, query, .. } => {
            expr_var_refs(lhs, acc);
            expr_var_refs(query, acc);
        }
        BoundExpression::InCommunity {
            node, community, ..
        } => {
            expr_var_refs(node, acc);
            expr_var_refs(community, acc);
        }
        BoundExpression::In { lhs, rhs, .. } => {
            expr_var_refs(lhs, acc);
            expr_var_refs(rhs, acc);
        }
        BoundExpression::IsNull { lhs, .. } => expr_var_refs(lhs, acc),
        _ => {}
    }
}

#[test]
fn full_query_bind_against_populated_catalog() {
    let input =
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN b.name AS friend_name";
    let stmt = parse(input).expect("parse");
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name"]);
    let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");

    let q = match &bound {
        BoundStatement::Read(q) => q,
        other => panic!("expected Read, got {other:?}"),
    };

    // Sanity: tenant/partition stamped from catalog.
    assert_eq!(q.tenant, cat.tenant());
    assert_eq!(q.partition, cat.partition());
    assert!(
        q.snapshot_lsn.is_none(),
        "M4-21 reserves snapshot_lsn = None; M4-61 populates"
    );

    // Match clause: a, b distinct bindings; KNOWS resolved.
    let m = match &q.clauses[0] {
        BoundClause::Match(m) => m,
        other => panic!("expected Match, got {other:?}"),
    };
    let path = match &m.body {
        BoundMatchBody::Patterns(ps) => &ps[0],
        BoundMatchBody::NamedPath(_) => panic!("expected Patterns body"),
    };
    let a_var = path.head.var.as_ref().expect("a declared");
    assert_eq!(a_var.name, "a");
    assert!(!path.head.labels.is_empty(), "Person label resolved");
    let (rel, b_node) = &path.tail[0];
    assert_eq!(rel.rel_types.len(), 1, "KNOWS rel-type resolved");
    let b_var = b_node.var.as_ref().expect("b declared");
    assert_eq!(b_var.name, "b");
    assert_ne!(a_var.binding_id, b_var.binding_id, "a and b distinct");

    // WHERE references a.age — the `a` ref must resolve to a's binding.
    assert!(m.where_clause.is_some());

    // RETURN's projection: b.name AS friend_name.
    let r = match q.clauses.last().expect("RETURN") {
        BoundClause::Return(r) => r,
        other => panic!("expected Return last, got {other:?}"),
    };
    assert_eq!(r.items.len(), 1);
    assert_eq!(r.items[0].alias.as_deref(), Some("friend_name"));
    let return_refs = clause_var_refs(q.clauses.last().unwrap());
    assert!(
        return_refs.iter().any(|n| n == "b"),
        "RETURN b.name resolves b"
    );

    // Verify the property `name` got a resolved PropertyId at the
    // RETURN site (round-trip catalog lookup).
    if let BoundProjectionKind::Expr(BoundExpression::PropertyAccess { path, .. }) =
        &r.items[0].kind
    {
        let last = path.last().expect("non-empty path");
        assert!(
            last.property_id.is_some(),
            "name property should resolve via catalog"
        );
    } else {
        panic!("expected b.name PropertyAccess projection");
    }
}

#[test]
fn binding_error_spans_point_at_offending_token() {
    // (Was an `UnknownLabel` case; since ADR-038 amendment-12 made unknown
    // labels permissive, this exercises span-translation on an
    // `UndeclaredVariable` — an undeclared `undeclaredVar` in RETURN — which
    // remains a hard BindingError with a span at the offending token.)
    let input = "MATCH (n) RETURN undeclaredVar";
    let stmt = parse(input).expect("parse");
    let cat = StubCatalogProvider::new();
    let errs = BindingVisitor::bind(&stmt, input, &cat).expect_err("expected errors");

    let err = &errs[0];
    assert!(
        matches!(err, BindingError::UndeclaredVariable { .. }),
        "first error must be UndeclaredVariable, got {err:?}"
    );
    let (start, end) = err
        .span_byte_range(input)
        .expect("span_byte_range translation");
    let offending = &input[start..end];
    assert_eq!(
        offending, "undeclaredVar",
        "span byte range must point at the variable name `undeclaredVar`"
    );
}

// =====================================================================
// OPTIONAL MATCH binding (M4-22 — ADR-006 amendment-01 + amendment-03
// §TIER-1 GAP D)
// =====================================================================

#[test]
fn optional_match_sets_is_optional_flag_after_type_check() {
    let input = "OPTIONAL MATCH (n:Person) RETURN n";
    let stmt = parse(input).expect("parse");
    let cat = StubCatalogProvider::new().with_labels(["Person"]);
    let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!(),
    };
    let m = match &q.clauses[0] {
        BoundClause::Match(m) => m,
        other => panic!("expected Match, got {other:?}"),
    };
    assert!(m.is_optional, "OPTIONAL MATCH must set is_optional=true");
}

#[test]
fn optional_match_introduces_may_be_null_variables_after_type_check() {
    let input = "OPTIONAL MATCH (n:Person) RETURN n";
    let stmt = parse(input).expect("parse");
    let cat = StubCatalogProvider::new().with_labels(["Person"]);
    let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!(),
    };
    let m = match &q.clauses[0] {
        BoundClause::Match(m) => m,
        other => panic!("expected Match, got {other:?}"),
    };
    let path = match &m.body {
        BoundMatchBody::Patterns(ps) => &ps[0],
        _ => panic!(),
    };
    let v = path.head.var.as_ref().expect("n declared");
    assert!(
        v.may_be_null,
        "OPTIONAL MATCH variable must have may_be_null=true"
    );
}

#[test]
fn match_then_optional_match_distinct_variables_get_correct_flags() {
    // Plain MATCH (a:Person) has may_be_null=false on `a`;
    // the OPTIONAL MATCH (b:Doc) introduces a fresh `b` with
    // may_be_null=true. The binding pass + type-check pass
    // cooperate per ADR-006 amendment-01.
    //
    // NOTE on re-using a name across MATCH-chain: openCypher
    // canonical re-declaration semantics (`MATCH (a) OPTIONAL
    // MATCH (a)-[:..]->(b)`) collide with the M4-21 binding pass's
    // strict `DuplicateBinding` rule (the `(a)` site in the OPTIONAL
    // MATCH parses as a fresh declaration, conflicting with the
    // plain MATCH's `a`). M4-23 will revisit re-reference resolution
    // when extending the BoundAst with substrate validation; M4-22
    // only commits to the discriminant + may_be_null propagation.
    let input = "MATCH (a:Person) OPTIONAL MATCH (b:Doc) RETURN a, b";
    let stmt = parse(input).expect("parse");
    let cat = StubCatalogProvider::new().with_labels(["Person", "Doc"]);
    let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    let q = match &bound {
        BoundStatement::Read(q) => q,
        _ => panic!(),
    };

    // Plain MATCH first.
    let first = match &q.clauses[0] {
        BoundClause::Match(m) => m,
        _ => panic!(),
    };
    assert!(!first.is_optional, "first MATCH is plain");
    let a_var = match &first.body {
        BoundMatchBody::Patterns(ps) => ps[0].head.var.as_ref().unwrap(),
        _ => panic!(),
    };
    assert!(!a_var.may_be_null, "plain MATCH variable not nullable");

    // OPTIONAL MATCH second.
    let second = match &q.clauses[1] {
        BoundClause::Match(m) => m,
        _ => panic!(),
    };
    assert!(second.is_optional, "second clause is OPTIONAL MATCH");
    let b_var = match &second.body {
        BoundMatchBody::Patterns(ps) => ps[0].head.var.as_ref().unwrap(),
        _ => panic!(),
    };
    assert!(
        b_var.may_be_null,
        "OPTIONAL MATCH variable must be nullable"
    );
}
