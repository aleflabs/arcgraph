//! M4-32 OPTIONAL MATCH lowering integration tests per ADR-006
//! amendment-01 §A-2 + ADR-038 §2 D-26 + Cypher 9 §6.5.
//!
//! M4-31 deferred OPTIONAL MATCH lowering with a
//! `NotImplementedAtM4_31 { surface: "OPTIONAL MATCH",
//! target_slice: "M4-32 (left-outer join lowering)" }` marker. M4-32
//! replaces that emission with [`LogicalPlan::LeftOuterJoin`]
//! construction.
//!
//! # Pin set
//!
//! 1. `optional_match_lowers_to_left_outer_join` — single OPTIONAL
//!    MATCH after a plain MATCH.
//! 2. `optional_match_join_condition_carries_shared_bindings` — the
//!    LeftOuterJoin's `on` field MUST carry the bindings shared
//!    between the plain MATCH and the OPTIONAL MATCH (re-references
//!    of pre-existing bindings).
//! 3. `chained_optional_match_lowers_to_nested_left_outer_joins` —
//!    `MATCH (a) OPTIONAL MATCH (a)-[:R]-(b) OPTIONAL MATCH
//!    (a)-[:S]-(c)` — three clauses → two LeftOuterJoins, each
//!    sharing `a`.
//! 4. `optional_match_at_query_head_lowers_without_outer_join` — an
//!    OPTIONAL MATCH at the very head (no prior plain MATCH)
//!    degenerates to its pattern subtree per Cypher 9 §6.5
//!    (left-outer-join over an empty left side reduces to the
//!    right).
//! 5. `optional_match_propagates_may_be_null_into_lowered_tree` —
//!    cross-checks that the binding-time `may_be_null` flag M4-22b
//!    sets is preserved across the lowering boundary (M4-32 reads
//!    the flag; it does not recompute nullability).
//! 6. `ldbc_is5_shape_lowers_cleanly` — pin the LDBC SNB IS5 query
//!    shape (a single OPTIONAL MATCH on top of a plain MATCH)
//!    succeeds without deferral.
//! 7. `ldbc_is6_shape_lowers_cleanly` — pin the LDBC SNB IS6 query
//!    shape (a graph-traversal MATCH followed by an OPTIONAL MATCH
//!    re-referencing the seed node) succeeds without deferral.
//!
//! # ADR provenance
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH at v1.0 lowers to
//!   left-outer join per Cypher 9 §6.5.
//! - ADR-038 §2 D-26 — M4-32 hybrid retrieval + OPTIONAL MATCH
//!   lowering contract.
//! - ADR-038 §2 D-21 (M4-22b refinement) — `may_be_null` is set at
//!   binding time; M4-32 reads it at lowering time and does NOT
//!   recompute.

use arcgraph_query::logical_plan::{JoinCondition, LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingId, BindingVisitor, BoundClause, BoundMatchBody, BoundQuery, BoundStatement,
    CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// Pipeline + catalog helpers
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Comment", "Post", "Forum", "Place"])
        .with_rel_types([
            "KNOWS",
            "REPLY_OF",
            "HAS_CREATOR",
            "HAS_MEMBER",
            "IS_LOCATED_IN",
            "R",
            "S",
            "OTHER",
        ])
        .with_properties(["age", "name", "creationDate", "id"])
}

/// Run the full pipeline. Panics on any pre-lowering failure.
fn lower_ok(input: &str) -> LogicalPlan {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat()).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat()).expect("cross-substrate validate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

fn query_of(b: &BoundStatement) -> &BoundQuery {
    match b {
        BoundStatement::Read(q) => q,
        _ => panic!("expected Read"),
    }
}

/// Walk and collect node-kind labels in pre-order.
fn shape(plan: &LogicalPlan) -> Vec<&'static str> {
    let mut out = Vec::new();
    walk(plan, &mut out);
    out
}

fn walk(p: &LogicalPlan, out: &mut Vec<&'static str>) {
    match p {
        LogicalPlan::Scan(_) => out.push("Scan"),
        LogicalPlan::PropertyIndexScan(_) => out.push("PropertyIndexScan"),
        LogicalPlan::CountStore(_) => out.push("CountStore"),
        LogicalPlan::Expand(_) => out.push("Expand"),
        LogicalPlan::Filter(f) => {
            out.push("Filter");
            walk(&f.input, out);
        }
        LogicalPlan::Project(pr) => {
            out.push("Project");
            walk(&pr.input, out);
        }
        LogicalPlan::Join(j) => {
            out.push("Join");
            walk(&j.left, out);
            walk(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            out.push("LeftOuterJoin");
            walk(&j.left, out);
            walk(&j.right, out);
        }
        LogicalPlan::Limit(l) => {
            out.push("Limit");
            walk(&l.input, out);
        }
        LogicalPlan::Skip(s) => {
            out.push("Skip");
            walk(&s.input, out);
        }
        LogicalPlan::RankByHybrid(_) => out.push("RankByHybrid"),
        LogicalPlan::Fusion(f) => {
            out.push("Fusion");
            for inp in &f.inputs {
                walk(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            out.push("Union");
            for arm in &u.arms {
                walk(arm, out);
            }
        }
        LogicalPlan::CommunityLookup(c) => {
            out.push("CommunityLookup");
            walk(&c.input, out);
        }
        LogicalPlan::VectorNear(_) => out.push("VectorNear"),
        LogicalPlan::TextMatch(_) => out.push("TextMatch"),
        LogicalPlan::Aggregate(a) => {
            out.push("Aggregate");
            walk(&a.input, out);
        }
        LogicalPlan::Sort(s) => {
            out.push("Sort");
            walk(&s.input, out);
        }
        LogicalPlan::Distinct(d) => {
            out.push("Distinct");
            walk(&d.input, out);
        }
        LogicalPlan::Unwind(u) => {
            out.push("Unwind");
            walk(&u.input, out);
        }
        LogicalPlan::ProcedureCall(p) => {
            out.push("Unwind");
            walk(&p.input, out);
        }
        LogicalPlan::NamedPath(np) => {
            out.push("NamedPath");
            walk(&np.input, out);
        }
        LogicalPlan::DynamicLimit(l) => {
            out.push("DynamicLimit");
            walk(&l.input, out);
        }
        LogicalPlan::CreateNode(_) => out.push("CreateNode"),
        LogicalPlan::CreateVectorIndex(_) => out.push("CreateVectorIndex"),
        LogicalPlan::CreatePropertyIndex(_) => out.push("CreatePropertyIndex"),
        LogicalPlan::CreateRel(c) => {
            out.push("CreateRel");
            walk(&c.source_plan, out);
            walk(&c.target_plan, out);
        }
        LogicalPlan::Delete(d) => {
            out.push("Delete");
            walk(&d.input, out);
        }
        LogicalPlan::Set(s) => {
            out.push("Set");
            walk(&s.input, out);
        }
        LogicalPlan::Remove(r) => {
            out.push("Remove");
            walk(&r.input, out);
        }
        LogicalPlan::Merge(m) => {
            out.push("Merge");
            walk(&m.match_branch, out);
            walk(&m.create_branch, out);
        }
        LogicalPlan::Empty(_) => out.push("Empty"),
        LogicalPlan::Call(_) => out.push("Call"),
        LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
    }
}

fn count_left_outer_joins(p: &LogicalPlan) -> usize {
    let mut n = 0;
    count_loj(p, &mut n);
    n
}

fn count_loj(p: &LogicalPlan, n: &mut usize) {
    if let LogicalPlan::LeftOuterJoin(_) = p {
        *n += 1;
    }
    match p {
        LogicalPlan::Filter(f) => count_loj(&f.input, n),
        LogicalPlan::Project(pr) => count_loj(&pr.input, n),
        LogicalPlan::Join(j) => {
            count_loj(&j.left, n);
            count_loj(&j.right, n);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            count_loj(&j.left, n);
            count_loj(&j.right, n);
        }
        LogicalPlan::Limit(l) => count_loj(&l.input, n),
        LogicalPlan::Skip(s) => count_loj(&s.input, n),
        LogicalPlan::CommunityLookup(c) => count_loj(&c.input, n),
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                count_loj(inp, n);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                count_loj(arm, n);
            }
        }
        LogicalPlan::Aggregate(a) => count_loj(&a.input, n),
        LogicalPlan::Sort(s) => count_loj(&s.input, n),
        LogicalPlan::Distinct(d) => count_loj(&d.input, n),
        LogicalPlan::Unwind(u) => count_loj(&u.input, n),
        LogicalPlan::ProcedureCall(p) => count_loj(&p.input, n),
        LogicalPlan::NamedPath(np) => count_loj(&np.input, n),
        LogicalPlan::DynamicLimit(l) => count_loj(&l.input, n),
        LogicalPlan::Delete(d) => count_loj(&d.input, n),
        LogicalPlan::Set(s) => count_loj(&s.input, n),
        LogicalPlan::Remove(r) => count_loj(&r.input, n),
        LogicalPlan::Merge(m) => {
            count_loj(&m.match_branch, n);
            count_loj(&m.create_branch, n);
        }
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
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => {}
    }
}

fn first_left_outer_join(
    p: &LogicalPlan,
) -> Option<&arcgraph_query::logical_plan::LogicalLeftOuterJoin> {
    match p {
        LogicalPlan::LeftOuterJoin(j) => Some(j),
        LogicalPlan::Filter(f) => first_left_outer_join(&f.input),
        LogicalPlan::Project(pr) => first_left_outer_join(&pr.input),
        LogicalPlan::Join(j) => {
            first_left_outer_join(&j.left).or_else(|| first_left_outer_join(&j.right))
        }
        LogicalPlan::Limit(l) => first_left_outer_join(&l.input),
        LogicalPlan::Skip(s) => first_left_outer_join(&s.input),
        LogicalPlan::CommunityLookup(c) => first_left_outer_join(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| first_left_outer_join(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(first_left_outer_join),
        LogicalPlan::Aggregate(a) => first_left_outer_join(&a.input),
        LogicalPlan::Sort(s) => first_left_outer_join(&s.input),
        LogicalPlan::Distinct(d) => first_left_outer_join(&d.input),
        LogicalPlan::Unwind(u) => first_left_outer_join(&u.input),
        LogicalPlan::ProcedureCall(p) => first_left_outer_join(&p.input),
        LogicalPlan::NamedPath(np) => first_left_outer_join(&np.input),
        LogicalPlan::DynamicLimit(l) => first_left_outer_join(&l.input),
        LogicalPlan::Delete(d) => first_left_outer_join(&d.input),
        LogicalPlan::Set(s) => first_left_outer_join(&s.input),
        LogicalPlan::Remove(r) => first_left_outer_join(&r.input),
        LogicalPlan::Merge(m) => first_left_outer_join(&m.match_branch)
            .or_else(|| first_left_outer_join(&m.create_branch)),
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
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => None,
    }
}

// =====================================================================
// Pin 1 — Single OPTIONAL MATCH after a plain MATCH lowers to LeftOuterJoin
// =====================================================================

#[test]
fn optional_match_lowers_to_left_outer_join() {
    let input =
        "MATCH (a:Person)-[:KNOWS]-(b:Person) OPTIONAL MATCH (a)-[:OTHER]-(c) RETURN a, b, c";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(
        s.contains(&"LeftOuterJoin"),
        "expected LeftOuterJoin in plan; got: {s:?}"
    );
    assert_eq!(
        count_left_outer_joins(&plan),
        1,
        "expected exactly one LeftOuterJoin for one OPTIONAL MATCH"
    );
}

// =====================================================================
// Pin 2 — JoinCondition carries shared bindings
// =====================================================================

#[test]
fn optional_match_join_condition_carries_shared_bindings() {
    let input =
        "MATCH (a:Person)-[:KNOWS]-(b:Person) OPTIONAL MATCH (a)-[:OTHER]-(c) RETURN a, b, c";
    let plan = lower_ok(input);
    let loj = first_left_outer_join(&plan).expect("LeftOuterJoin present");

    // Resolve `a`'s BindingId from the bound query.
    let stmt = parse(input).expect("parse");
    let mut b = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut b, &cat()).expect("type-check");
    let q = query_of(&b);
    let a_id = head_var_id(q, 0).expect("a in match clause 0");

    let JoinCondition::SharedBindings(ids) = &loj.on;
    assert!(
        ids.contains(&a_id),
        "expected `a` ({a_id:?}) in shared bindings, got: {ids:?}"
    );
}

fn head_var_id(q: &BoundQuery, clause_idx: usize) -> Option<BindingId> {
    match &q.clauses[clause_idx] {
        BoundClause::Match(m) => match &m.body {
            BoundMatchBody::Patterns(ps) => ps.first()?.head.var.as_ref().map(|v| v.binding_id),
            _ => None,
        },
        _ => None,
    }
}

// =====================================================================
// Pin 3 — Chained OPTIONAL MATCH lowers to nested LeftOuterJoins
// =====================================================================

#[test]
fn chained_optional_match_lowers_to_nested_left_outer_joins() {
    let input =
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:R]-(b) OPTIONAL MATCH (a)-[:S]-(c) RETURN a, b, c";
    let plan = lower_ok(input);
    assert_eq!(
        count_left_outer_joins(&plan),
        2,
        "expected two LeftOuterJoins for two OPTIONAL MATCH clauses; got plan: {plan:#?}"
    );
}

// =====================================================================
// Pin 4 — OPTIONAL MATCH at query head — left-outer-join over the unit
// (single-empty-row) driving table (openCypher 9 §6.5). #996-followup:
// the prior `None => filtered` shortcut dropped the unit row and returned
// 0 rows over an empty graph; the head OPTIONAL MATCH now roots on a
// leading-clause `LogicalEmpty` unit row under a `LeftOuterJoin` so that
// an empty match still emits one null-extended row.
// =====================================================================

#[test]
fn optional_match_at_query_head_lowers_to_unit_row_left_outer_join() {
    let input = "OPTIONAL MATCH (n:Person) RETURN n";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(s.contains(&"Scan"), "expected Scan; got: {s:?}");
    assert!(
        s.contains(&"LeftOuterJoin"),
        "head OPTIONAL MATCH must root on a unit-row LeftOuterJoin so an \
         empty match emits one null row (openCypher 9 §6.5); got: {s:?}"
    );
    assert!(
        s.contains(&"Empty"),
        "the LeftOuterJoin's left side must be the unit-row `LogicalEmpty` \
         driving table; got: {s:?}"
    );
    // The binding's `may_be_null` flag is still set at bind time, so
    // downstream consumers see the nullability.
    let stmt = parse(input).expect("parse");
    let mut b = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut b, &cat()).expect("type-check");
    let q = query_of(&b);
    let n_var = match &q.clauses[0] {
        BoundClause::Match(m) => match &m.body {
            BoundMatchBody::Patterns(ps) => ps[0].head.var.as_ref().unwrap(),
            _ => panic!(),
        },
        _ => panic!(),
    };
    assert!(
        n_var.may_be_null,
        "head OPTIONAL MATCH still flags fresh declarations as nullable"
    );
}

// =====================================================================
// Pin 5 — `may_be_null` propagation across the lowering boundary.
// M4-22b binding-time semantics: re-references of non-nullable
// bindings stay non-nullable; fresh declarations in OPTIONAL MATCH
// are nullable. M4-32 reads — does not recompute.
// =====================================================================

#[test]
fn optional_match_propagates_may_be_null_into_lowered_tree() {
    let input = "MATCH (a:Person)-[:KNOWS]-(b) OPTIONAL MATCH (a)-[:OTHER]-(c) RETURN a, b, c";
    let stmt = parse(input).expect("parse");
    let mut b = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut b, &cat()).expect("type-check");
    let q = query_of(&b);

    // First clause variables.
    let a_match = first_var_with_name(q, 0, "a").expect("a in clause 0");
    let b_match = first_var_with_name(q, 0, "b").expect("b in clause 0");
    // Second-clause variables.
    let a_opt = first_var_with_name(q, 1, "a").expect("a in clause 1");
    let c_opt = first_var_with_name(q, 1, "c").expect("c in clause 1");

    assert!(!a_match.may_be_null);
    assert!(!b_match.may_be_null);
    assert!(
        !a_opt.may_be_null,
        "re-reference of `a` stays non-nullable per M4-22b"
    );
    assert!(c_opt.may_be_null, "fresh `c` in OPTIONAL MATCH is nullable");

    // Stronger oracle: re-reference shares binding_id.
    assert_eq!(a_match.binding_id, a_opt.binding_id);

    // M4-32 lowering does not mutate the binding-time flags.
    let mut b2 = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut b2, &cat()).expect("type-check");
    CrossSubstrateValidator::validate(&b2, &cat()).expect("validate");
    let _ = LogicalPlanLoweringVisitor::lower(&b2).expect("lower");
    let q_post = query_of(&b2);
    let a_match_post = first_var_with_name(q_post, 0, "a").expect("a post-lower");
    let c_opt_post = first_var_with_name(q_post, 1, "c").expect("c post-lower");
    assert_eq!(a_match.may_be_null, a_match_post.may_be_null);
    assert_eq!(c_opt.may_be_null, c_opt_post.may_be_null);
}

fn first_var_with_name<'q>(
    q: &'q BoundQuery,
    clause_idx: usize,
    name: &str,
) -> Option<&'q arcgraph_query::semantic::BoundVariable> {
    let m = match &q.clauses[clause_idx] {
        BoundClause::Match(m) => m,
        _ => return None,
    };
    let pp = match &m.body {
        BoundMatchBody::Patterns(ps) => &ps[0],
        _ => return None,
    };
    if let Some(v) = pp.head.var.as_ref() {
        if v.name == name {
            return Some(v);
        }
    }
    for (rel, node) in &pp.tail {
        if let Some(v) = rel.var.as_ref() {
            if v.name == name {
                return Some(v);
            }
        }
        if let Some(v) = node.var.as_ref() {
            if v.name == name {
                return Some(v);
            }
        }
    }
    None
}

// =====================================================================
// Pin 6 — LDBC SNB IS5 query shape lowers cleanly.
//
// IS5 (LDBC SNB Interactive Short 5): given a Comment id, return the
// person who created it. The Cypher form uses an OPTIONAL MATCH over
// the creator pattern (per LDBC reference impl). We approximate the
// shape with the v1.0 surface; the goal is to verify M4-32 lowers it
// without deferral, not to exactly replicate the LDBC parameter
// binding.
// =====================================================================

#[test]
fn ldbc_is5_shape_lowers_cleanly() {
    let input = "MATCH (c:Comment) OPTIONAL MATCH (c)-[:HAS_CREATOR]->(p:Person) RETURN c, p";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(
        s.contains(&"LeftOuterJoin"),
        "IS5 shape must lower to a LeftOuterJoin; got: {s:?}"
    );
    assert!(s.contains(&"Scan"), "expected Scan; got: {s:?}");
    assert!(s.contains(&"Expand"), "expected Expand; got: {s:?}");
}

// =====================================================================
// Pin 7 — LDBC SNB IS6 shape: graph-traversal MATCH followed by
// OPTIONAL MATCH re-referencing the seed.
// =====================================================================

#[test]
fn ldbc_is6_shape_lowers_cleanly() {
    let input = "MATCH (m:Comment)-[:REPLY_OF]->(p:Post) OPTIONAL MATCH (p)-[:HAS_CREATOR]->(creator:Person) RETURN m, p, creator";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(
        s.contains(&"LeftOuterJoin"),
        "IS6 shape must lower to a LeftOuterJoin; got: {s:?}"
    );
    assert_eq!(
        count_left_outer_joins(&plan),
        1,
        "IS6 has one OPTIONAL MATCH"
    );
}
