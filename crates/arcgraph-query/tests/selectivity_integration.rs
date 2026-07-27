//! Integration tests for M4-42 (M4-04b) selectivity estimators.
//!
//! These tests exercise the public API of [`SelectivityEstimator`]
//! against a `StubCatalogProvider` configured to mimic three
//! production scenarios:
//!
//! 1. **End-to-end pipeline + estimator query** — parse a query, run
//!    the M4-21/M4-22/M4-23 semantic pipeline to extract a
//!    representative bound predicate, then query the estimator. Pins
//!    that the estimator surface composes cleanly with the bound-AST
//!    types (`BindingId`, `LabelId`) the planner will hand it.
//!
//! 2. **Multi-tenant isolation** — two `StubCatalogProvider`s with
//!    distinct cardinality stats produce distinct selectivity
//!    estimates. Pins that the estimator carries no per-tenant state
//!    of its own (it reads through to whichever catalog it was
//!    constructed with), so the M4-05 cost planner can build a fresh
//!    estimator per query without a per-tenant cache invalidation
//!    surface.
//!
//! 3. **Cold-start fallback** — a catalog with no stats collected
//!    (a fresh tenant whose commit pipeline has never fired, per
//!    M4-41 §D-25) returns the documented `DEFAULT_*_SELECTIVITY`
//!    constant for every estimator. Pins the graceful-degradation
//!    contract end-to-end.
//!
//! # ADR provenance
//! - ADR-038 §2 D-25 — M4-41 catalog stats schema (the read source).
//! - ADR-038 §2 D-27 — M4-42 selectivity estimators per predicate
//!   class.
//! - ADR-038 amendment-03 §M4-42 row — slice scope + test artifacts.

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingId, BindingVisitor, BoundClause, BoundExpression, BoundMatchBody, BoundStatement,
    DEFAULT_EQ_SELECTIVITY, DEFAULT_IN_SELECTIVITY, DEFAULT_LABEL_SELECTIVITY,
    DEFAULT_LT_SELECTIVITY, DEFAULT_REL_TYPE_SELECTIVITY, SelectivityEstimator,
    StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// 1. End-to-end pipeline + representative-predicate estimator query.
// ---------------------------------------------------------------------

/// Walk the bound query and return the `BindingId` of the first
/// node-pattern variable plus its first declared label (if any). Used
/// by the end-to-end test to extract a representative `(var, label)`
/// pair from the post-binding AST and feed it to the estimator.
fn first_node_var_and_label(bound: &BoundStatement) -> Option<(BindingId, Option<LabelId>)> {
    let q = match bound {
        BoundStatement::Read(q) => q,
        _ => return None,
    };
    for clause in &q.clauses {
        if let BoundClause::Match(m) = clause {
            if let BoundMatchBody::Patterns(paths) = &m.body {
                let head = &paths.first()?.head;
                let var = head.var.as_ref().map(|v| v.binding_id)?;
                let label = head.labels.first().map(|l| l.label_id);
                return Some((var, label));
            }
        }
    }
    None
}

#[test]
fn end_to_end_parse_bind_typecheck_then_estimate_eq() {
    // Parse a simple node-equality query. The bound AST gives us a
    // real `BindingId` + `LabelId` pair that the M4-05 planner will
    // pass to the estimator at cost-evaluation time.
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"])
        .with_label_cardinality(LabelId::new(1), 250)
        .with_total_node_count(1_000)
        .with_total_rel_count(2_000);

    let input = "MATCH (n:Person) WHERE n.age = 42 RETURN n";
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    let (var, label) = first_node_var_and_label(&bound).expect("var + label");
    assert_eq!(label, Some(LabelId::new(1)));

    let est = SelectivityEstimator::new(&cat);

    // estimate_eq: 1 / total_node_count = 1 / 1000 = 0.001.
    let s_eq = est.estimate_eq(var, label);
    assert!((s_eq - 0.001).abs() < 1e-12, "got {s_eq}");

    // estimate_label: 250 / 1000 = 0.25 — the predicate `(n:Person)`
    // narrows to a quarter of the tenant's nodes.
    let s_label = est.estimate_label(label.unwrap());
    assert!((s_label - 0.25).abs() < 1e-12, "got {s_label}");

    // Bound-shape sanity: every estimator returns a finite value in
    // [0, 1] (the planner cost-monotonicity invariant).
    for s in [s_eq, s_label] {
        assert!(
            s.is_finite() && (0.0..=1.0).contains(&s),
            "out of [0,1]: {s}"
        );
    }
}

// ---------------------------------------------------------------------
// 2. Multi-tenant isolation — distinct catalogs → distinct estimates.
// ---------------------------------------------------------------------

#[test]
fn multi_tenant_isolation_distinct_estimates() {
    // Two tenants with the SAME schema (Person label, KNOWS rel-type)
    // but different stats. The estimator carries no per-tenant cache;
    // each constructor reads through the per-tenant catalog. Cross-
    // tenant pollution is structurally impossible.
    let l = LabelId::new(1);
    let t = TypeId::new(1);

    let tenant_a = StubCatalogProvider::new()
        .with_tenant(TenantId::new(1))
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_label_cardinality(l, 100)
        .with_rel_type_cardinality(t, 50)
        .with_total_node_count(500)
        .with_total_rel_count(1_000);

    let tenant_b = StubCatalogProvider::new()
        .with_tenant(TenantId::new(2))
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_label_cardinality(l, 800)
        .with_rel_type_cardinality(t, 1_500)
        .with_total_node_count(2_000)
        .with_total_rel_count(2_500);

    let est_a = SelectivityEstimator::new(&tenant_a);
    let est_b = SelectivityEstimator::new(&tenant_b);

    // Tenant A: rare-label tenant — :Person covers 100/500 = 20%.
    let label_a = est_a.estimate_label(l);
    assert!((label_a - 0.2).abs() < 1e-12, "tenant A label: {label_a}");

    // Tenant B: dense-label tenant — :Person covers 800/2000 = 40%.
    let label_b = est_b.estimate_label(l);
    assert!((label_b - 0.4).abs() < 1e-12, "tenant B label: {label_b}");
    assert_ne!(label_a, label_b, "estimates must isolate per tenant");

    // Tenant A: KNOWS covers 50/1000 = 5%.
    let rel_a = est_a.estimate_rel_type(t);
    assert!((rel_a - 0.05).abs() < 1e-12, "tenant A rel: {rel_a}");

    // Tenant B: KNOWS covers 1500/2500 = 60%.
    let rel_b = est_b.estimate_rel_type(t);
    assert!((rel_b - 0.6).abs() < 1e-12, "tenant B rel: {rel_b}");
    assert_ne!(rel_a, rel_b);

    // estimate_eq is also tenant-scoped: 1/500 vs 1/2000.
    let eq_a = est_a.estimate_eq(BindingId::new(0), Some(l));
    let eq_b = est_b.estimate_eq(BindingId::new(0), Some(l));
    assert!((eq_a - 0.002).abs() < 1e-12);
    assert!((eq_b - 0.0005).abs() < 1e-12);
    assert_ne!(eq_a, eq_b);
}

// ---------------------------------------------------------------------
// 3. Cold-start fallback — empty catalog → DEFAULT_* constants.
// ---------------------------------------------------------------------

#[test]
fn cold_start_catalog_returns_defaults_for_every_estimator() {
    // Bind a representative query against a catalog with NO stats
    // (empty `StubCatalogProvider` — the production analogue is a
    // fresh tenant whose commit pipeline has never fired per M4-41
    // §D-25). Every estimator MUST return its `DEFAULT_*_SELECTIVITY`
    // constant. End-to-end pin of the graceful-degradation contract.
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age"]);

    let input = "MATCH (n:Person)-[r:KNOWS]->(m:Person) WHERE n.age = 42 RETURN n";
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    let (var, label) = first_node_var_and_label(&bound).expect("var + label");

    let est = SelectivityEstimator::new(&cat);

    assert_eq!(est.estimate_eq(var, label), DEFAULT_EQ_SELECTIVITY);
    assert_eq!(est.estimate_lt(var, label), DEFAULT_LT_SELECTIVITY);
    assert_eq!(est.estimate_in(var, label, 5), DEFAULT_IN_SELECTIVITY);
    assert_eq!(
        est.estimate_label(label.unwrap()),
        DEFAULT_LABEL_SELECTIVITY
    );
    assert_eq!(
        est.estimate_rel_type(TypeId::new(1)),
        DEFAULT_REL_TYPE_SELECTIVITY
    );

    // Sanity: the bound query also exposes a relationship pattern,
    // and the estimator returns the rel-type default for IT too.
    if let BoundStatement::Read(q) = &bound {
        for clause in &q.clauses {
            if let BoundClause::Match(m) = clause {
                if let BoundMatchBody::Patterns(paths) = &m.body {
                    for path in paths {
                        for (rel, _next) in &path.tail {
                            if let Some(rt) = rel.rel_types.first() {
                                let s = est.estimate_rel_type(rt.type_id);
                                assert_eq!(s, DEFAULT_REL_TYPE_SELECTIVITY);
                            }
                        }
                    }
                }
            }
        }
    }

    // Also pin: the WHERE-clause variable reference resolves to the
    // same BindingId we extracted from the head — i.e., the estimator
    // input and the planner input are wired through the same bound
    // identifier (no second binding pass for predicate sub-expressions).
    if let BoundStatement::Read(q) = &bound {
        for clause in &q.clauses {
            if let BoundClause::Match(m) = clause {
                if let Some(BoundExpression::BinaryOp { lhs, .. }) = m.where_clause.as_ref() {
                    if let BoundExpression::PropertyAccess { base, .. } = lhs.as_ref() {
                        if let BoundExpression::VariableRef { binding_id, .. } = base.as_ref() {
                            assert_eq!(*binding_id, var);
                        }
                    }
                }
            }
        }
    }
}
