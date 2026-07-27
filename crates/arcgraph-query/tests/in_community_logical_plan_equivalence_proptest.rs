//! M4-32 IN-COMMUNITY ↔ canonical `community(n) = $cid` LOGICAL-PLAN
//! equivalence proptest (256 cases). Closes PR #154 reviewer Finding 5
//! at the logical-plan level (M4-23's proptest pinned semantic
//! equivalence; M4-32 extends to the LogicalPlan tree).
//!
//! # The equivalence claim
//!
//! Per ADR-038 §2 D-26 + amendment-01 §A-2, both surfaces MUST lower
//! to IDENTICAL `LogicalCommunityLookup` plan trees, modulo span
//! coordinates. M4-32's [`LogicalPlanLoweringVisitor`] recognizes both
//! the `BoundExpression::InCommunity` predicate shape AND the
//! canonical `BinaryOp(Eq, FunctionCall("community", [VariableRef]),
//! Parameter)` (and its symmetric form) and produces identical
//! `LogicalCommunityLookup` nodes — same `node_var`, same
//! `community_id` payload, identical `input` subtrees.
//!
//! # ADR provenance
//! - ADR-038 §2 D-4 (canonical `community(...)` family).
//! - ADR-038 §2 D-23 (cross-substrate validation contract).
//! - ADR-038 §2 D-26 (M4-32 hybrid retrieval lowering + OPTIONAL
//!   MATCH; THIS file's primary spec).
//! - ADR-038 amendment-01 §A-2 (`IN COMMUNITY(...)` ↔ canonical
//!   "lower to same plan-tree node" commitment).
//! - PR #154 reviewer Finding 5 (closes at logical-plan level).
//! - M4-23 `tests/in_community_equivalence_proptest.rs` (the
//!   semantic-level prequel this proptest extends).

use arcgraph_query::ast::{BinOp, Literal};
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{JoinCondition, LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, BoundExpression, BoundMapProjectionItem, BoundProjectionItem,
    BoundProjectionKind, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------
// Strategy generators (re-using the M4-23 alphabet)
// ---------------------------------------------------------------------

/// Strategy for the MATCH-pattern variable name. Narrow alphabet to
/// avoid the binding-pass span-cursor heuristic edge cases.
fn arbitrary_var_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9]?"
}

/// Strategy for the parameter name (referenced as `$<name>`).
fn arbitrary_param_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9]?"
}

// ---------------------------------------------------------------------
// Pipeline helper
// ---------------------------------------------------------------------

/// Run parse → bind → type-check → cross-substrate → lower. Panics on
/// any prior-stage failure since those are out of M4-32 scope.
fn parse_bind_typecheck_validate_lower(input: &str, cat: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, cat).expect("cross-substrate validate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

// ---------------------------------------------------------------------
// Span-blind structural equivalence
// ---------------------------------------------------------------------

/// Compare two [`LogicalPlan`] trees for structural equivalence,
/// IGNORING span coordinates throughout. Returns `true` only when the
/// two trees are byte-equal modulo every `Span` field.
///
/// The function recursively walks both trees in tandem; it returns
/// `false` on the first structural mismatch (variant kind, BindingId,
/// LabelId, projection-item shape, etc.) — i.e., the only acceptable
/// difference between the two inputs is the Span coordinates.
fn logical_plan_structurally_equivalent(a: &LogicalPlan, b: &LogicalPlan) -> bool {
    let mut na = a.clone();
    let mut nb = b.clone();
    normalize_plan(&mut na);
    normalize_plan(&mut nb);
    na == nb
}

const Z: Span = Span {
    start_line: 1,
    start_col: 1,
    end_line: 1,
    end_col: 1,
};

fn normalize_plan(p: &mut LogicalPlan) {
    match p {
        LogicalPlan::Scan(s) => {
            s.span = Z;
        }
        LogicalPlan::PropertyIndexScan(p) => {
            p.span = Z;
        }
        LogicalPlan::CountStore(c) => {
            c.span = Z;
        }
        LogicalPlan::Expand(e) => {
            e.span = Z;
        }
        LogicalPlan::Filter(f) => {
            f.span = Z;
            normalize_expr(&mut f.predicate);
            normalize_plan(&mut f.input);
        }
        LogicalPlan::Project(pr) => {
            pr.span = Z;
            for it in &mut pr.items {
                normalize_projection_item(it);
            }
            normalize_plan(&mut pr.input);
        }
        LogicalPlan::Join(j) => {
            j.span = Z;
            normalize_plan(&mut j.left);
            normalize_plan(&mut j.right);
            normalize_join_condition(&mut j.on);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            j.span = Z;
            normalize_plan(&mut j.left);
            normalize_plan(&mut j.right);
            normalize_join_condition(&mut j.on);
        }
        LogicalPlan::Limit(l) => {
            l.span = Z;
            normalize_plan(&mut l.input);
        }
        LogicalPlan::Skip(s) => {
            s.span = Z;
            normalize_plan(&mut s.input);
        }
        LogicalPlan::RankByHybrid(r) => {
            r.span = Z;
            for op in &mut r.operands {
                op.span = Z;
                normalize_expr(&mut op.query);
            }
        }
        LogicalPlan::Fusion(f) => {
            f.span = Z;
            f.spec.span = Z;
            for inp in &mut f.inputs {
                normalize_plan(inp);
            }
        }
        LogicalPlan::Union(u) => {
            u.span = Z;
            for arm in &mut u.arms {
                normalize_plan(arm);
            }
        }
        LogicalPlan::CommunityLookup(c) => {
            c.span = Z;
            normalize_plan(&mut c.input);
            normalize_expr(&mut c.community_id);
        }
        LogicalPlan::VectorNear(v) => {
            v.span = Z;
            normalize_expr(&mut v.query_vector);
        }
        LogicalPlan::TextMatch(t) => {
            t.span = Z;
            normalize_expr(&mut t.query_text);
        }
        LogicalPlan::Aggregate(a) => {
            a.span = Z;
            for it in &mut a.group_by {
                normalize_projection_item(it);
            }
            for spec in &mut a.aggregations {
                spec.span = Z;
                normalize_expr(&mut spec.arg);
            }
            normalize_plan(&mut a.input);
        }
        LogicalPlan::Sort(s) => {
            s.span = Z;
            for it in &mut s.order_by {
                it.span = Z;
                normalize_expr(&mut it.expr);
            }
            normalize_plan(&mut s.input);
        }
        LogicalPlan::Distinct(d) => {
            d.span = Z;
            normalize_plan(&mut d.input);
        }
        LogicalPlan::Unwind(u) => {
            u.span = Z;
            normalize_expr(&mut u.list_expr);
            normalize_plan(&mut u.input);
        }
        LogicalPlan::ProcedureCall(p) => {
            p.span = Z;
            for a in &mut p.args {
                normalize_expr(a);
            }
            normalize_plan(&mut p.input);
        }
        LogicalPlan::NamedPath(np) => {
            np.span = Z;
            normalize_plan(&mut np.input);
        }
        LogicalPlan::DynamicLimit(l) => {
            l.span = Z;
            normalize_expr(&mut l.count_expr);
            normalize_plan(&mut l.input);
        }
        LogicalPlan::CreateNode(c) => {
            c.span = Z;
            for (_, v) in &mut c.properties {
                normalize_expr(v);
            }
        }
        // #830 / ADR-200: CREATE VECTOR INDEX — normalize the span only
        // (its OPTIONS is an unbound ast::Expression carried verbatim;
        // it has no span to zero + no bound sub-expressions to walk).
        LogicalPlan::CreateVectorIndex(c) => {
            c.span = Z;
        }
        LogicalPlan::CreatePropertyIndex(c) => {
            c.span = Z;
        }
        LogicalPlan::CreateRel(c) => {
            c.span = Z;
            for (_, v) in &mut c.properties {
                normalize_expr(v);
            }
            normalize_plan(&mut c.source_plan);
            normalize_plan(&mut c.target_plan);
        }
        LogicalPlan::Delete(d) => {
            d.span = Z;
            for item in &mut d.items {
                item.span = Z;
            }
            normalize_plan(&mut d.input);
        }
        LogicalPlan::Set(s) => {
            s.span = Z;
            for item in &mut s.items {
                item.span = Z;
                normalize_set_mutation(&mut item.mutation);
            }
            normalize_plan(&mut s.input);
        }
        LogicalPlan::Remove(r) => {
            r.span = Z;
            for item in &mut r.items {
                item.span = Z;
            }
            normalize_plan(&mut r.input);
        }
        LogicalPlan::Merge(m) => {
            m.span = Z;
            normalize_plan(&mut m.match_branch);
            normalize_plan(&mut m.create_branch);
            for item in &mut m.on_create {
                item.span = Z;
                normalize_set_mutation(&mut item.mutation);
            }
            for item in &mut m.on_match {
                item.span = Z;
                normalize_set_mutation(&mut item.mutation);
            }
        }
        LogicalPlan::Empty(e) => {
            e.span = Z;
        }
        // ADR-192 (#623): normalize the CALL{} span + recurse into BOTH
        // the driving input and the subquery body; the seed has only a
        // span. (Not produced by this test's query set today, but the
        // span-normalization recursion is the structurally correct arm.)
        LogicalPlan::Call(c) => {
            c.span = Z;
            normalize_plan(&mut c.input);
            normalize_plan(&mut c.body);
        }
        LogicalPlan::CorrelationSeed(s) => {
            s.span = Z;
        }
    }
}

fn normalize_set_mutation(m: &mut arcgraph_query::logical_plan::LogicalSetMutation) {
    use arcgraph_query::logical_plan::LogicalSetMutation;
    match m {
        LogicalSetMutation::PropertyAssign { value, .. } => normalize_expr(value),
        LogicalSetMutation::PropertyReplace(entries)
        | LogicalSetMutation::PropertyMerge(entries) => {
            for (_, v) in entries {
                normalize_expr(v);
            }
        }
        LogicalSetMutation::LabelAdd(_) => {}
    }
}

fn normalize_join_condition(jc: &mut JoinCondition) {
    let JoinCondition::SharedBindings(_) = jc;
    // SharedBindings carries no spans; the BindingId vec is part of
    // the structural identity. No-op.
}

fn normalize_projection_item(it: &mut BoundProjectionItem) {
    it.span = Z;
    if let BoundProjectionKind::Expr(e) = &mut it.kind {
        normalize_expr(e);
    }
}

fn normalize_expr(e: &mut BoundExpression) {
    match e {
        BoundExpression::Literal { span, value, .. } => {
            *span = Z;
            normalize_literal(value);
        }
        BoundExpression::ListLiteral { span, elements, .. } => {
            *span = Z;
            for element in elements {
                normalize_expr(element);
            }
        }
        BoundExpression::MapLiteral { span, entries, .. } => {
            *span = Z;
            for (_, value) in entries {
                normalize_expr(value);
            }
        }
        BoundExpression::Parameter { span, .. } => {
            *span = Z;
        }
        BoundExpression::VariableRef { span, .. } => {
            *span = Z;
        }
        BoundExpression::UnresolvedVariable { span, .. } => {
            *span = Z;
        }
        BoundExpression::PropertyAccess {
            span, base, path, ..
        } => {
            *span = Z;
            normalize_expr(base);
            for p in path {
                p.span = Z;
            }
        }
        BoundExpression::BinaryOp { span, lhs, rhs, .. } => {
            *span = Z;
            normalize_expr(lhs);
            normalize_expr(rhs);
        }
        BoundExpression::UnaryOp { span, operand, .. } => {
            *span = Z;
            normalize_expr(operand);
        }
        BoundExpression::FunctionCall { span, args, .. } => {
            *span = Z;
            for a in args {
                normalize_expr(a);
            }
        }
        BoundExpression::Near {
            span, lhs, target, ..
        } => {
            *span = Z;
            normalize_expr(lhs);
            normalize_expr(target);
        }
        BoundExpression::TextMatch {
            span, lhs, query, ..
        } => {
            *span = Z;
            normalize_expr(lhs);
            normalize_expr(query);
        }
        BoundExpression::InCommunity {
            span,
            node,
            community,
            ..
        } => {
            *span = Z;
            normalize_expr(node);
            normalize_expr(community);
        }
        BoundExpression::In { span, lhs, rhs, .. } => {
            *span = Z;
            normalize_expr(lhs);
            normalize_expr(rhs);
        }
        BoundExpression::IsNull { span, lhs, .. } => {
            *span = Z;
            normalize_expr(lhs);
        }
        // ADR-188 — list-predicate / reduce special forms. (None of
        // this proptest's queries exercise them, but the match must be
        // total.) Zero the span + recurse into every child.
        BoundExpression::ListPredicate {
            span,
            list,
            predicate,
            ..
        } => {
            *span = Z;
            normalize_expr(list);
            normalize_expr(predicate);
        }
        BoundExpression::Reduce {
            span,
            init,
            list,
            expr,
            ..
        } => {
            *span = Z;
            normalize_expr(init);
            normalize_expr(list);
            normalize_expr(expr);
        }
        // ADR-188 (#620 list-half) — list comprehension. (None of this
        // proptest's queries exercise it, but the match must be total.)
        // Zero the span + recurse into every child (list + optional
        // predicate + optional projection).
        BoundExpression::ListComprehension {
            span,
            list,
            predicate,
            projection,
            ..
        } => {
            *span = Z;
            normalize_expr(list);
            if let Some(p) = predicate {
                normalize_expr(p);
            }
            if let Some(e) = projection {
                normalize_expr(e);
            }
        }
        // #621 — list subscript / slice (§3.4). (None of this proptest's
        // queries exercise them, but the match must be total.) Zero the
        // span + recurse into every child.
        BoundExpression::Subscript {
            span, base, index, ..
        } => {
            *span = Z;
            normalize_expr(base);
            normalize_expr(index);
        }
        BoundExpression::Slice {
            span,
            base,
            start,
            end,
            ..
        } => {
            *span = Z;
            normalize_expr(base);
            if let Some(s) = start {
                normalize_expr(s);
            }
            if let Some(e) = end {
                normalize_expr(e);
            }
        }
        // ADR-191 D-6 (#620 map-half) — map projection. (None of this
        // proptest's queries exercise it, but the match must be total.)
        // Zero the span + recurse into the base + every literal-entry value.
        BoundExpression::MapProjection {
            span, base, items, ..
        } => {
            *span = Z;
            normalize_expr(base);
            for item in items {
                if let BoundMapProjectionItem::Literal { value, .. } = item {
                    normalize_expr(value);
                }
            }
        }
        // #621 — CASE expression (§3.6). (None of this proptest's queries
        // exercise it, but the match must be total.) Zero the span + recurse
        // into the test + every WHEN/THEN + the ELSE.
        BoundExpression::Case {
            span,
            test,
            branches,
            default,
            ..
        } => {
            *span = Z;
            if let Some(t) = test {
                normalize_expr(t);
            }
            for (when, then) in branches {
                normalize_expr(when);
                normalize_expr(then);
            }
            if let Some(d) = default {
                normalize_expr(d);
            }
        }
    }
}

fn normalize_literal(l: &mut Literal) {
    if let Literal::List(xs) = l {
        for x in xs {
            normalize_ast_expr(x);
        }
    }
    if let Literal::Map(entries) = l {
        for (_, v) in entries {
            normalize_ast_expr(v);
        }
    }
}

/// Defensive: `Literal::List` / `Literal::Map` carry untyped
/// `crate::ast::Expression` rather than `BoundExpression`. None of
/// the proptest queries exercise these, but leaving the helper as a
/// no-op keeps the recursion total.
fn normalize_ast_expr(_: &mut arcgraph_query::ast::Expression) {
    // No spans on the AST type — the M4-01 100 ratified tests pin
    // that contract. No-op.
}

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// For every (var_name, param_name) drawn from the strategy: the
    /// predicate-form query `MATCH (n:Person) WHERE n IN
    /// COMMUNITY($cid) RETURN n` and the canonical-form query
    /// `MATCH (n:Person) WHERE community(n) = $cid RETURN n` MUST
    /// lower to STRUCTURALLY-IDENTICAL [`LogicalPlan`] trees, modulo
    /// span coordinates.
    ///
    /// Per ADR-038 §2 D-26 + amendment-01 §A-2 + PR #154 reviewer
    /// Finding 5: the asymmetry the M4-23 proptest documented (the
    /// two surfaces produce different `BoundExpression` variants but
    /// semantically-equivalent bound queries) MUST collapse at the
    /// logical-plan level.
    #[test]
    fn in_community_predicate_logical_plan_equiv_to_canonical_function_call(
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

        let plan_pred = parse_bind_typecheck_validate_lower(&predicate_input, &cat);
        let plan_fn = parse_bind_typecheck_validate_lower(&function_input, &cat);

        prop_assert!(
            logical_plan_structurally_equivalent(&plan_pred, &plan_fn),
            "predicate vs canonical lowering must be structurally equivalent\n\
             predicate: {plan_pred:#?}\n\
             canonical: {plan_fn:#?}"
        );
    }

    /// Symmetric form: `$cid = community(n)` (community call on the
    /// RHS of the equality) MUST lower to the same tree as
    /// `community(n) = $cid` (call on the LHS).
    #[test]
    fn community_equality_is_order_invariant(
        var_name in arbitrary_var_name(),
        param_name in arbitrary_param_name(),
    ) {
        let lhs_form = format!(
            "MATCH ({var_name}:Person) WHERE community({var_name}) = ${param_name} RETURN {var_name}"
        );
        let rhs_form = format!(
            "MATCH ({var_name}:Person) WHERE ${param_name} = community({var_name}) RETURN {var_name}"
        );

        let cat = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_community_index();

        let plan_lhs = parse_bind_typecheck_validate_lower(&lhs_form, &cat);
        let plan_rhs = parse_bind_typecheck_validate_lower(&rhs_form, &cat);

        prop_assert!(
            logical_plan_structurally_equivalent(&plan_lhs, &plan_rhs),
            "community(n) = $cid must be order-invariant under equality\n\
             lhs-form: {plan_lhs:#?}\n\
             rhs-form: {plan_rhs:#?}"
        );
    }
}

// ---------------------------------------------------------------------
// Additional unit oracle: the predicate form's tree contains a
// `LogicalCommunityLookup` (vs. a generic `LogicalFilter`).
// ---------------------------------------------------------------------

#[test]
fn predicate_form_lowers_to_logical_community_lookup() {
    let input = "MATCH (n:Person) WHERE n IN COMMUNITY($cid) RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let plan = parse_bind_typecheck_validate_lower(input, &cat);
    assert!(
        contains_community_lookup(&plan),
        "expected LogicalCommunityLookup in plan; got: {plan:#?}"
    );
}

#[test]
fn canonical_form_lowers_to_logical_community_lookup() {
    let input = "MATCH (n:Person) WHERE community(n) = $cid RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let plan = parse_bind_typecheck_validate_lower(input, &cat);
    assert!(
        contains_community_lookup(&plan),
        "expected LogicalCommunityLookup in plan; got: {plan:#?}"
    );
}

#[test]
fn canonical_form_does_not_emit_generic_filter_for_community_predicate() {
    let input = "MATCH (n:Person) WHERE community(n) = $cid RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let plan = parse_bind_typecheck_validate_lower(input, &cat);
    assert!(
        !contains_filter(&plan),
        "expected NO LogicalFilter (community predicate must lower to \
         LogicalCommunityLookup); got: {plan:#?}"
    );
}

#[test]
fn canonical_form_with_param_on_lhs_lowers_to_logical_community_lookup() {
    // Direct (non-transitive) verification of reviewer's F2 concern:
    // the symmetric ordering `$cid = community(n)` (param on LHS,
    // community-call on RHS) MUST itself lower to LogicalCommunityLookup
    // — NOT a generic Eq Filter. The 256-case
    // `community_equality_is_order_invariant` proptest above asserts
    // structural equivalence between the two orderings; this oracle
    // pins the absolute claim ("symmetric form produces a
    // CommunityLookup, period") without leaning on transitivity.
    let input = "MATCH (n:Person) WHERE $cid = community(n) RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let plan = parse_bind_typecheck_validate_lower(input, &cat);
    assert!(
        contains_community_lookup(&plan),
        "expected LogicalCommunityLookup for symmetric ordering; got: {plan:#?}"
    );
    assert!(
        !contains_filter(&plan),
        "symmetric ordering must NOT emit a generic Eq Filter; got: {plan:#?}"
    );
}

#[test]
fn unrelated_equality_predicate_keeps_lowering_as_filter() {
    // Sanity check that the canonical-form recognizer is precise:
    // an unrelated equality (`n.age = 30`) MUST stay a Filter, not
    // become a CommunityLookup.
    let input = "MATCH (n:Person) WHERE n.age = 30 RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);
    let plan = parse_bind_typecheck_validate_lower(input, &cat);
    assert!(
        contains_filter(&plan),
        "expected LogicalFilter for non-community equality; got: {plan:#?}"
    );
    assert!(
        !contains_community_lookup(&plan),
        "expected NO LogicalCommunityLookup; got: {plan:#?}"
    );
}

#[test]
fn binding_id_carried_into_community_lookup_matches_match_pattern() {
    // Stronger oracle: the LogicalCommunityLookup's node_var must
    // equal the BindingId of the `n` declared by the MATCH pattern.
    let input = "MATCH (n:Person) WHERE n IN COMMUNITY($cid) RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let plan = parse_bind_typecheck_validate_lower(input, &cat);

    let lookup = find_community_lookup(&plan).expect("LogicalCommunityLookup present");
    let scan = find_scan(&plan).expect("LogicalScan present");
    assert_eq!(
        lookup.node_var, scan.var,
        "lookup.node_var must equal the MATCH pattern's BindingId"
    );

    // And the carried community_id MUST be the parameter expression.
    match &lookup.community_id {
        BoundExpression::Parameter { name, .. } => assert_eq!(name, "cid"),
        other => panic!("expected Parameter community_id; got: {other:?}"),
    }
}

#[test]
fn predicate_and_canonical_carry_same_binop_eq_or_in_community_at_bind_time() {
    // Cross-check with the M4-23 prequel: the BoundAst-level shapes
    // ARE different (the closure is at the LogicalPlan level only).
    // This test pins the documented asymmetry M4-23 caught.
    let pred_input = "MATCH (n:Person) WHERE n IN COMMUNITY($cid) RETURN n";
    let fn_input = "MATCH (n:Person) WHERE community(n) = $cid RETURN n";
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_community_index();
    let pred_stmt = parse(pred_input).expect("parse");
    let fn_stmt = parse(fn_input).expect("parse");
    let mut pred_b = BindingVisitor::bind(&pred_stmt, pred_input, &cat).expect("bind");
    let mut fn_b = BindingVisitor::bind(&fn_stmt, fn_input, &cat).expect("bind");
    TypeCheckVisitor::check(&mut pred_b, &cat).expect("type-check");
    TypeCheckVisitor::check(&mut fn_b, &cat).expect("type-check");

    let pred_q = match &pred_b {
        arcgraph_query::semantic::BoundStatement::Read(q) => q,
        _ => panic!("expected Read"),
    };
    let fn_q = match &fn_b {
        arcgraph_query::semantic::BoundStatement::Read(q) => q,
        _ => panic!("expected Read"),
    };

    // Predicate form: WHERE is BoundExpression::InCommunity.
    let pred_where = match &pred_q.clauses[0] {
        arcgraph_query::semantic::BoundClause::Match(m) => m.where_clause.as_ref().unwrap(),
        _ => panic!(),
    };
    assert!(matches!(pred_where, BoundExpression::InCommunity { .. }));

    // Canonical form: WHERE is BoundExpression::BinaryOp(Eq, ...).
    let fn_where = match &fn_q.clauses[0] {
        arcgraph_query::semantic::BoundClause::Match(m) => m.where_clause.as_ref().unwrap(),
        _ => panic!(),
    };
    assert!(matches!(
        fn_where,
        BoundExpression::BinaryOp { op: BinOp::Eq, .. }
    ));
}

// ---------------------------------------------------------------------
// Helpers used by the unit oracles
// ---------------------------------------------------------------------

fn contains_community_lookup(p: &LogicalPlan) -> bool {
    find_community_lookup(p).is_some()
}

fn find_community_lookup(
    p: &LogicalPlan,
) -> Option<&arcgraph_query::logical_plan::LogicalCommunityLookup> {
    match p {
        LogicalPlan::CommunityLookup(c) => Some(c),
        LogicalPlan::Filter(f) => find_community_lookup(&f.input),
        LogicalPlan::Project(pr) => find_community_lookup(&pr.input),
        LogicalPlan::Join(j) => {
            find_community_lookup(&j.left).or_else(|| find_community_lookup(&j.right))
        }
        LogicalPlan::LeftOuterJoin(j) => {
            find_community_lookup(&j.left).or_else(|| find_community_lookup(&j.right))
        }
        LogicalPlan::Limit(l) => find_community_lookup(&l.input),
        LogicalPlan::Skip(s) => find_community_lookup(&s.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_community_lookup(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_community_lookup),
        LogicalPlan::Aggregate(a) => find_community_lookup(&a.input),
        LogicalPlan::Sort(s) => find_community_lookup(&s.input),
        LogicalPlan::Distinct(d) => find_community_lookup(&d.input),
        LogicalPlan::Unwind(u) => find_community_lookup(&u.input),
        LogicalPlan::ProcedureCall(p) => find_community_lookup(&p.input),
        LogicalPlan::NamedPath(np) => find_community_lookup(&np.input),
        LogicalPlan::DynamicLimit(l) => find_community_lookup(&l.input),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => None,
    }
}

fn contains_filter(p: &LogicalPlan) -> bool {
    match p {
        LogicalPlan::Filter(_) => true,
        LogicalPlan::Project(pr) => contains_filter(&pr.input),
        LogicalPlan::Join(j) => contains_filter(&j.left) || contains_filter(&j.right),
        LogicalPlan::LeftOuterJoin(j) => contains_filter(&j.left) || contains_filter(&j.right),
        LogicalPlan::Limit(l) => contains_filter(&l.input),
        LogicalPlan::Skip(s) => contains_filter(&s.input),
        LogicalPlan::CommunityLookup(c) => contains_filter(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| contains_filter(inp)),
        LogicalPlan::Union(u) => u.arms.iter().any(contains_filter),
        LogicalPlan::Aggregate(a) => contains_filter(&a.input),
        LogicalPlan::Sort(s) => contains_filter(&s.input),
        LogicalPlan::Distinct(d) => contains_filter(&d.input),
        LogicalPlan::Unwind(u) => contains_filter(&u.input),
        LogicalPlan::ProcedureCall(p) => contains_filter(&p.input),
        LogicalPlan::NamedPath(np) => contains_filter(&np.input),
        LogicalPlan::DynamicLimit(l) => contains_filter(&l.input),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => false,
    }
}

fn find_scan(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalScan> {
    match p {
        LogicalPlan::Scan(s) => Some(s),
        LogicalPlan::PropertyIndexScan(_) => None,
        LogicalPlan::Filter(f) => find_scan(&f.input),
        LogicalPlan::Project(pr) => find_scan(&pr.input),
        LogicalPlan::Join(j) => find_scan(&j.left).or_else(|| find_scan(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => find_scan(&j.left).or_else(|| find_scan(&j.right)),
        LogicalPlan::Limit(l) => find_scan(&l.input),
        LogicalPlan::Skip(s) => find_scan(&s.input),
        LogicalPlan::CommunityLookup(c) => find_scan(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_scan(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_scan),
        LogicalPlan::Aggregate(a) => find_scan(&a.input),
        LogicalPlan::Sort(s) => find_scan(&s.input),
        LogicalPlan::Distinct(d) => find_scan(&d.input),
        LogicalPlan::Unwind(u) => find_scan(&u.input),
        LogicalPlan::ProcedureCall(p) => find_scan(&p.input),
        LogicalPlan::NamedPath(np) => find_scan(&np.input),
        LogicalPlan::DynamicLimit(l) => find_scan(&l.input),
        LogicalPlan::Expand(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => None,
    }
}
