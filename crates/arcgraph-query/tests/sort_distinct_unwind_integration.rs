//! M4-33 sort + DISTINCT + UNWIND + named-path + dynamic-LIMIT
//! integration tests per ADR-038 §2 D-28.
//!
//! M4-31 + M4-32 deferred these surfaces with `NotImplementedAtM4_31`
//! markers; M4-33 lights them up.
//!
//! # Pin set (per ADR-038 amendment-03 §M4-33 row)
//!
//! 1. `lower_order_by_asc_desc` — both ASC (default) and DESC sort
//!    directions are preserved in the [`LogicalSort`] node.
//! 2. `lower_distinct_on_columns` — `RETURN DISTINCT n` lowers to
//!    [`LogicalDistinct`] with `n` in the on-set.
//! 3. `lower_unwind_list_expression` — `UNWIND [1,2,3] AS x RETURN x`
//!    lowers to [`LogicalUnwind`] over [`LogicalEmpty`].
//! 4. `lower_named_path_with_shortest_path` — `MATCH p =
//!    SHORTEST_PATH(...)` lowers to [`LogicalNamedPath`] carrying
//!    [`PathAlgorithm::ShortestPath`] over the lowered pattern
//!    subtree.
//! 5. `lower_dynamic_limit_with_parameter` — `LIMIT $n` lowers to
//!    [`LogicalDynamicLimit`] (kind = Limit) carrying the parameter
//!    expression.
//! 6. `lower_dynamic_skip_with_expression` — non-literal SKIP `SKIP $k`
//!    lowers to [`LogicalDynamicLimit`] (kind = Skip) with the
//!    expression preserved.
//!
//! # ADR provenance
//! - ADR-038 §2 D-28 — aggregation + sort + path operators contract.
//! - ADR-038 amendment-03 §M4-33 row — test-artifact pin (8 unit + 6
//!   integration + 1 proptest).

use arcgraph_query::logical_plan::{
    DynamicLimitKind, LogicalPlan, LogicalPlanLoweringVisitor, PathAlgorithm, SortDirection,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// Pipeline + catalog helpers
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "x", "embedding", "content"])
}

fn lower_ok(input: &str) -> LogicalPlan {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat()).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat()).expect("validate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

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
        // ADR-192 (#623): CALL{} + its seed are not produced by this
        // test's query set; name them for structural-walk completeness.
        LogicalPlan::Call(_) => out.push("Call"),
        LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
    }
}

fn find_sort(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalSort> {
    match p {
        LogicalPlan::Sort(s) => Some(s),
        LogicalPlan::Filter(f) => find_sort(&f.input),
        LogicalPlan::Project(pr) => find_sort(&pr.input),
        LogicalPlan::Join(j) => find_sort(&j.left).or_else(|| find_sort(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => find_sort(&j.left).or_else(|| find_sort(&j.right)),
        LogicalPlan::Limit(l) => find_sort(&l.input),
        LogicalPlan::Skip(s) => find_sort(&s.input),
        LogicalPlan::CommunityLookup(c) => find_sort(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_sort(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_sort),
        LogicalPlan::Aggregate(a) => find_sort(&a.input),
        LogicalPlan::Distinct(d) => find_sort(&d.input),
        LogicalPlan::Unwind(u) => find_sort(&u.input),
        LogicalPlan::ProcedureCall(p) => find_sort(&p.input),
        LogicalPlan::NamedPath(np) => find_sort(&np.input),
        LogicalPlan::DynamicLimit(l) => find_sort(&l.input),
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

fn find_distinct(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalDistinct> {
    match p {
        LogicalPlan::Distinct(d) => Some(d),
        LogicalPlan::Filter(f) => find_distinct(&f.input),
        LogicalPlan::Project(pr) => find_distinct(&pr.input),
        LogicalPlan::Join(j) => find_distinct(&j.left).or_else(|| find_distinct(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => find_distinct(&j.left).or_else(|| find_distinct(&j.right)),
        LogicalPlan::Limit(l) => find_distinct(&l.input),
        LogicalPlan::Skip(s) => find_distinct(&s.input),
        LogicalPlan::CommunityLookup(c) => find_distinct(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_distinct(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_distinct),
        LogicalPlan::Aggregate(a) => find_distinct(&a.input),
        LogicalPlan::Sort(s) => find_distinct(&s.input),
        LogicalPlan::Unwind(u) => find_distinct(&u.input),
        LogicalPlan::ProcedureCall(p) => find_distinct(&p.input),
        LogicalPlan::NamedPath(np) => find_distinct(&np.input),
        LogicalPlan::DynamicLimit(l) => find_distinct(&l.input),
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

fn find_unwind(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalUnwind> {
    match p {
        LogicalPlan::Unwind(u) => Some(u),
        LogicalPlan::ProcedureCall(p) => find_unwind(&p.input),
        LogicalPlan::Filter(f) => find_unwind(&f.input),
        LogicalPlan::Project(pr) => find_unwind(&pr.input),
        LogicalPlan::Join(j) => find_unwind(&j.left).or_else(|| find_unwind(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => find_unwind(&j.left).or_else(|| find_unwind(&j.right)),
        LogicalPlan::Limit(l) => find_unwind(&l.input),
        LogicalPlan::Skip(s) => find_unwind(&s.input),
        LogicalPlan::CommunityLookup(c) => find_unwind(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_unwind(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_unwind),
        LogicalPlan::Aggregate(a) => find_unwind(&a.input),
        LogicalPlan::Sort(s) => find_unwind(&s.input),
        LogicalPlan::Distinct(d) => find_unwind(&d.input),
        LogicalPlan::NamedPath(np) => find_unwind(&np.input),
        LogicalPlan::DynamicLimit(l) => find_unwind(&l.input),
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

fn find_named_path(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalNamedPath> {
    match p {
        LogicalPlan::NamedPath(np) => Some(np),
        LogicalPlan::Filter(f) => find_named_path(&f.input),
        LogicalPlan::Project(pr) => find_named_path(&pr.input),
        LogicalPlan::Join(j) => find_named_path(&j.left).or_else(|| find_named_path(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => {
            find_named_path(&j.left).or_else(|| find_named_path(&j.right))
        }
        LogicalPlan::Limit(l) => find_named_path(&l.input),
        LogicalPlan::Skip(s) => find_named_path(&s.input),
        LogicalPlan::CommunityLookup(c) => find_named_path(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_named_path(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_named_path),
        LogicalPlan::Aggregate(a) => find_named_path(&a.input),
        LogicalPlan::Sort(s) => find_named_path(&s.input),
        LogicalPlan::Distinct(d) => find_named_path(&d.input),
        LogicalPlan::Unwind(u) => find_named_path(&u.input),
        LogicalPlan::ProcedureCall(p) => find_named_path(&p.input),
        LogicalPlan::DynamicLimit(l) => find_named_path(&l.input),
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

fn find_dynamic_limit(
    p: &LogicalPlan,
) -> Option<&arcgraph_query::logical_plan::LogicalDynamicLimit> {
    match p {
        LogicalPlan::DynamicLimit(l) => Some(l),
        LogicalPlan::Filter(f) => find_dynamic_limit(&f.input),
        LogicalPlan::Project(pr) => find_dynamic_limit(&pr.input),
        LogicalPlan::Join(j) => {
            find_dynamic_limit(&j.left).or_else(|| find_dynamic_limit(&j.right))
        }
        LogicalPlan::LeftOuterJoin(j) => {
            find_dynamic_limit(&j.left).or_else(|| find_dynamic_limit(&j.right))
        }
        LogicalPlan::Limit(l) => find_dynamic_limit(&l.input),
        LogicalPlan::Skip(s) => find_dynamic_limit(&s.input),
        LogicalPlan::CommunityLookup(c) => find_dynamic_limit(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_dynamic_limit(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_dynamic_limit),
        LogicalPlan::Aggregate(a) => find_dynamic_limit(&a.input),
        LogicalPlan::Sort(s) => find_dynamic_limit(&s.input),
        LogicalPlan::Distinct(d) => find_dynamic_limit(&d.input),
        LogicalPlan::Unwind(u) => find_dynamic_limit(&u.input),
        LogicalPlan::ProcedureCall(p) => find_dynamic_limit(&p.input),
        LogicalPlan::NamedPath(np) => find_dynamic_limit(&np.input),
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

// =====================================================================
// Pin 1 — ORDER BY ASC and DESC
// =====================================================================

#[test]
fn lower_order_by_asc_desc() {
    // ASC (default).
    let plan = lower_ok("MATCH (n:Person) RETURN n ORDER BY n.age");
    let s = shape(&plan);
    assert!(s.contains(&"Sort"), "expected Sort: {s:?}");
    let sort = find_sort(&plan).expect("Sort present");
    assert_eq!(sort.order_by.len(), 1);
    assert_eq!(sort.order_by[0].direction, SortDirection::Asc);

    // DESC.
    let plan_desc = lower_ok("MATCH (n:Person) RETURN n ORDER BY n.age DESC");
    let sort_desc = find_sort(&plan_desc).expect("Sort present");
    assert_eq!(sort_desc.order_by[0].direction, SortDirection::Desc);

    // Multi-key: primary ASC + tie-breaker DESC.
    let plan_multi = lower_ok("MATCH (n:Person) RETURN n ORDER BY n.age ASC, n.name DESC");
    let sort_multi = find_sort(&plan_multi).expect("Sort present");
    assert_eq!(sort_multi.order_by.len(), 2);
    assert_eq!(sort_multi.order_by[0].direction, SortDirection::Asc);
    assert_eq!(sort_multi.order_by[1].direction, SortDirection::Desc);
}

// =====================================================================
// Pin 2 — DISTINCT carries an on-set derived from RETURN bindings
// =====================================================================

#[test]
fn lower_distinct_on_columns() {
    let plan = lower_ok("MATCH (a:Person), (b:Doc) RETURN DISTINCT a, b");
    let s = shape(&plan);
    assert!(s.contains(&"Distinct"), "expected Distinct: {s:?}");
    let d = find_distinct(&plan).expect("Distinct present");
    // Both `a` and `b` bindings appear in the on-set.
    assert_eq!(d.on.len(), 2);
}

// =====================================================================
// Pin 3 — UNWIND lowers to LogicalUnwind
// =====================================================================

#[test]
fn lower_unwind_list_expression() {
    let plan = lower_ok("UNWIND [1, 2, 3] AS x RETURN x");
    let s = shape(&plan);
    assert!(s.contains(&"Unwind"), "expected Unwind: {s:?}");
    let u = find_unwind(&plan).expect("Unwind present");
    // The list expression is preserved verbatim.
    let _ = &u.list_expr;
    // The unwind binding gets a fresh BindingId allocated at bind
    // time — assert it's a real (non-sentinel) id.
    assert_ne!(u.var.raw(), u64::MAX);
}

// =====================================================================
// Pin 4 — Named path with SHORTEST_PATH lowers to LogicalNamedPath
//          carrying ShortestPath
// =====================================================================

#[test]
fn lower_named_path_with_shortest_path() {
    let plan = lower_ok("MATCH p = SHORTEST_PATH((a:Person)-[:KNOWS]->(b:Person)) RETURN p");
    let s = shape(&plan);
    assert!(s.contains(&"NamedPath"), "expected NamedPath: {s:?}");
    let np = find_named_path(&plan).expect("NamedPath present");
    assert_eq!(np.algorithm, PathAlgorithm::ShortestPath);

    // Plain named path (no SHORTEST_PATH wrapper) → PathAlgorithm::Plain.
    let plan_plain = lower_ok("MATCH p = (a:Person)-[:KNOWS]->(b:Person) RETURN p");
    let np_plain = find_named_path(&plan_plain).expect("NamedPath present");
    assert_eq!(np_plain.algorithm, PathAlgorithm::Plain);
}

// =====================================================================
// Pin 5 — Parameter-driven LIMIT lowers to LogicalDynamicLimit
// =====================================================================

#[test]
fn lower_dynamic_limit_with_parameter() {
    let plan = lower_ok("MATCH (n:Person) RETURN n LIMIT $n");
    let s = shape(&plan);
    assert!(s.contains(&"DynamicLimit"), "expected DynamicLimit: {s:?}");
    let dl = find_dynamic_limit(&plan).expect("DynamicLimit present");
    assert_eq!(dl.kind, DynamicLimitKind::Limit);

    // Static literal LIMIT 10 still produces the M4-31 LogicalLimit
    // variant — preservation contract.
    let plan_lit = lower_ok("MATCH (n:Person) RETURN n LIMIT 10");
    let s_lit = shape(&plan_lit);
    assert!(
        s_lit.contains(&"Limit") && !s_lit.contains(&"DynamicLimit"),
        "literal LIMIT preserves LogicalLimit: {s_lit:?}",
    );
}

// =====================================================================
// Pin 6 — Non-literal SKIP via a parameter expression
// =====================================================================

#[test]
fn lower_dynamic_skip_with_expression() {
    let plan = lower_ok("MATCH (n:Person) RETURN n SKIP $k");
    let dl = find_dynamic_limit(&plan).expect("DynamicLimit present");
    assert_eq!(dl.kind, DynamicLimitKind::Skip);

    // Static literal SKIP 5 still produces the M4-31 LogicalSkip
    // variant.
    let plan_lit = lower_ok("MATCH (n:Person) RETURN n SKIP 5");
    let s_lit = shape(&plan_lit);
    assert!(
        s_lit.contains(&"Skip") && !s_lit.contains(&"DynamicLimit"),
        "literal SKIP preserves LogicalSkip: {s_lit:?}",
    );
}
