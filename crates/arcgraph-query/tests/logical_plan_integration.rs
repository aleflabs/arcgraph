//! Integration tests for M4-31 (M4-03a) logical-plan lowering.
//!
//! These tests exercise the full public API end-to-end:
//!     `parse → BindingVisitor::bind → TypeCheckVisitor::check
//!     → CrossSubstrateValidator::validate → LogicalPlanLoweringVisitor::lower`.
//!
//! # Pin set (per ADR-038 amendment-03 §M4-31 row + M4-32 row)
//!
//! 1. `lower_full_simple_query_end_to_end` — composes Scan + Expand +
//!    Filter + Project + Limit + Skip in a single ValidatedQuery and
//!    asserts the resulting LogicalPlan tree contains all six node
//!    kinds.
//! 2. `lower_rank_by_hybrid_to_logical_rank_by_hybrid` — M4-32 (this
//!    slice) replaces the M4-31 NotImplementedAtM4_31 deferral marker
//!    with proper lowering: `RANK BY HYBRID(VECTOR(...), TEXT(...))
//!    WITH FUSION = RRF(k = 60)` lowers to `Fusion +
//!    RankByHybrid` per ADR-038 §2 D-26.
//! 3. `lower_emits_not_implemented_for_aggregation` — verifies M4-31
//!    correctly defers aggregation functions (`count` / `sum` / `avg` /
//!    `min` / `max` / `collect`) to M4-33 with `target_slice="M4-33"`.
//! 4. `lower_preserves_span_on_every_node` — verifies the IDE
//!    error-reporting contract: every plan node carries a `span`
//!    pointing back into the original input.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 (the M4-31 contract).
//! - ADR-038 §2 D-26 (the M4-32 hybrid + OPTIONAL MATCH contract).
//! - ADR-038 §2 D-23 (visitor-trait discipline lock; M4-31 + M4-32
//!   ship custom walker).
//! - ADR-038 amendment-03 §M4-31 + §M4-32 rows (test-artifact pinning).

use arcgraph_query::logical_plan::{
    FusionKind, HybridOperandKind, LogicalPlan, LogicalPlanLoweringVisitor,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    ArcQLError, BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// Pipeline + catalog helpers
// ---------------------------------------------------------------------

fn cat_full() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "x", "embedding", "content"])
        .with_vector_index()
        .with_bm25_index()
        .with_community_index()
}

fn cat_bare() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "x", "embedding", "content"])
}

/// Run the full pipeline through M4-31 lowering. Returns the lowered
/// `LogicalPlan` on success or the `Vec<ArcQLError>` if lowering
/// emits any deferred-surface markers. Panics on any pre-lowering
/// failure (parse / bind / type-check / cross-substrate) since those
/// are out of M4-31 scope.
fn lower(input: &str, cat: &StubCatalogProvider) -> Result<LogicalPlan, Vec<ArcQLError>> {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, cat).expect("validate");
    LogicalPlanLoweringVisitor::lower(&bound)
}

/// Walk the tree, collecting node-kind labels in pre-order.
fn shape(plan: &LogicalPlan) -> Vec<&'static str> {
    let mut out = Vec::new();
    walk_kinds(plan, &mut out);
    out
}

fn walk_kinds(p: &LogicalPlan, out: &mut Vec<&'static str>) {
    match p {
        LogicalPlan::Scan(_) => out.push("Scan"),
        LogicalPlan::PropertyIndexScan(_) => out.push("PropertyIndexScan"),
        LogicalPlan::CountStore(_) => out.push("CountStore"),
        LogicalPlan::Expand(_) => out.push("Expand"),
        LogicalPlan::Filter(f) => {
            out.push("Filter");
            walk_kinds(&f.input, out);
        }
        LogicalPlan::Project(p) => {
            out.push("Project");
            walk_kinds(&p.input, out);
        }
        LogicalPlan::Join(j) => {
            out.push("Join");
            walk_kinds(&j.left, out);
            walk_kinds(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            out.push("LeftOuterJoin");
            walk_kinds(&j.left, out);
            walk_kinds(&j.right, out);
        }
        LogicalPlan::Limit(l) => {
            out.push("Limit");
            walk_kinds(&l.input, out);
        }
        LogicalPlan::Skip(s) => {
            out.push("Skip");
            walk_kinds(&s.input, out);
        }
        LogicalPlan::RankByHybrid(_) => out.push("RankByHybrid"),
        LogicalPlan::Fusion(f) => {
            out.push("Fusion");
            for inp in &f.inputs {
                walk_kinds(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            out.push("Union");
            for arm in &u.arms {
                walk_kinds(arm, out);
            }
        }
        LogicalPlan::CommunityLookup(c) => {
            out.push("CommunityLookup");
            walk_kinds(&c.input, out);
        }
        LogicalPlan::VectorNear(_) => out.push("VectorNear"),
        LogicalPlan::TextMatch(_) => out.push("TextMatch"),
        // M4-33 variants — walk through `input`.
        LogicalPlan::Aggregate(a) => {
            out.push("Aggregate");
            walk_kinds(&a.input, out);
        }
        LogicalPlan::Sort(s) => {
            out.push("Sort");
            walk_kinds(&s.input, out);
        }
        LogicalPlan::Distinct(d) => {
            out.push("Distinct");
            walk_kinds(&d.input, out);
        }
        LogicalPlan::Unwind(u) => {
            out.push("Unwind");
            walk_kinds(&u.input, out);
        }
        LogicalPlan::ProcedureCall(p) => {
            out.push("ProcedureCall");
            walk_kinds(&p.input, out);
        }
        LogicalPlan::NamedPath(np) => {
            out.push("NamedPath");
            walk_kinds(&np.input, out);
        }
        LogicalPlan::DynamicLimit(l) => {
            out.push("DynamicLimit");
            walk_kinds(&l.input, out);
        }
        LogicalPlan::CreateNode(_) => out.push("CreateNode"),
        LogicalPlan::CreateVectorIndex(_) => out.push("CreateVectorIndex"),
        LogicalPlan::CreatePropertyIndex(_) => out.push("CreatePropertyIndex"),
        LogicalPlan::CreateRel(c) => {
            out.push("CreateRel");
            walk_kinds(&c.source_plan, out);
            walk_kinds(&c.target_plan, out);
        }
        LogicalPlan::Delete(d) => {
            out.push("Delete");
            walk_kinds(&d.input, out);
        }
        LogicalPlan::Set(s) => {
            out.push("Set");
            walk_kinds(&s.input, out);
        }
        LogicalPlan::Remove(r) => {
            out.push("Remove");
            walk_kinds(&r.input, out);
        }
        LogicalPlan::Merge(m) => {
            out.push("Merge");
            walk_kinds(&m.match_branch, out);
            walk_kinds(&m.create_branch, out);
        }
        LogicalPlan::Empty(_) => out.push("Empty"),
        LogicalPlan::Call(_) => out.push("Call"),
        LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
    }
}

// =====================================================================
// Pin 1 — Full simple-query end-to-end
// =====================================================================

#[test]
fn lower_full_simple_query_end_to_end() {
    // Scan(:Person) + Expand(KNOWS) + Filter(WHERE) + Project(RETURN)
    // + Skip(SKIP 5) + Limit(LIMIT 10).
    let input =
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a, b SKIP 5 LIMIT 10";
    let plan = lower(input, &cat_full()).expect("lower");
    let s = shape(&plan);

    // Every required operator kind appears in the tree.
    for kind in ["Scan", "Expand", "Filter", "Project", "Skip", "Limit"] {
        assert!(
            s.contains(&kind),
            "expected `{kind}` in plan shape; got: {s:?}"
        );
    }
}

// =====================================================================
// Pin 2 — RANK BY HYBRID + WITH FUSION = RRF lowers to LogicalRankByHybrid
// + LogicalFusion (M4-32 lights this surface; per ADR-038 §2 D-26).
// =====================================================================

#[test]
fn lower_rank_by_hybrid_to_logical_rank_by_hybrid() {
    let input = "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) WITH FUSION = RRF(k = 60) RETURN n";
    let plan = lower(input, &cat_full()).expect("M4-32 lowers RANK BY HYBRID + RRF");
    let s = shape(&plan);

    for kind in ["Scan", "RankByHybrid", "Fusion", "Project"] {
        assert!(
            s.contains(&kind),
            "expected `{kind}` in plan shape; got: {s:?}"
        );
    }

    // The Fusion node MUST carry FusionKind::Rrf with k = 60.
    let fusion = find_fusion(&plan).expect("Fusion present");
    assert_eq!(fusion.spec.kind, FusionKind::Rrf);
    assert_eq!(fusion.spec.k, 60);

    // The RankByHybrid node MUST carry one Vector + one Text operand,
    // each with K = 20.
    let hybrid = find_rank_by_hybrid(&plan).expect("RankByHybrid present");
    assert_eq!(hybrid.operands.len(), 2);
    let vec_op = hybrid
        .operands
        .iter()
        .find(|o| o.kind == HybridOperandKind::Vector)
        .expect("vector operand");
    let txt_op = hybrid
        .operands
        .iter()
        .find(|o| o.kind == HybridOperandKind::Text)
        .expect("text operand");
    assert_eq!(vec_op.k, 20);
    assert_eq!(txt_op.k, 20);
    assert_eq!(vec_op.property, "embedding");
    assert_eq!(txt_op.property, "content");
}

fn find_fusion(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalFusion> {
    match p {
        LogicalPlan::Fusion(f) => Some(f),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_fusion),
        LogicalPlan::Filter(f) => find_fusion(&f.input),
        LogicalPlan::Project(pr) => find_fusion(&pr.input),
        LogicalPlan::Join(j) => find_fusion(&j.left).or_else(|| find_fusion(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => find_fusion(&j.left).or_else(|| find_fusion(&j.right)),
        LogicalPlan::Limit(l) => find_fusion(&l.input),
        LogicalPlan::Skip(s) => find_fusion(&s.input),
        LogicalPlan::CommunityLookup(c) => find_fusion(&c.input),
        LogicalPlan::Aggregate(a) => find_fusion(&a.input),
        LogicalPlan::Sort(s) => find_fusion(&s.input),
        LogicalPlan::Distinct(d) => find_fusion(&d.input),
        LogicalPlan::Unwind(u) => find_fusion(&u.input),
        LogicalPlan::ProcedureCall(p) => find_fusion(&p.input),
        LogicalPlan::NamedPath(np) => find_fusion(&np.input),
        LogicalPlan::DynamicLimit(l) => find_fusion(&l.input),
        LogicalPlan::Delete(d) => find_fusion(&d.input),
        LogicalPlan::Set(s) => find_fusion(&s.input),
        LogicalPlan::Remove(r) => find_fusion(&r.input),
        LogicalPlan::Merge(m) => {
            find_fusion(&m.match_branch).or_else(|| find_fusion(&m.create_branch))
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
        | LogicalPlan::CorrelationSeed(_) => None,
    }
}

fn find_rank_by_hybrid(
    p: &LogicalPlan,
) -> Option<&arcgraph_query::logical_plan::LogicalRankByHybrid> {
    match p {
        LogicalPlan::RankByHybrid(r) => Some(r),
        LogicalPlan::Filter(f) => find_rank_by_hybrid(&f.input),
        LogicalPlan::Project(pr) => find_rank_by_hybrid(&pr.input),
        LogicalPlan::Join(j) => {
            find_rank_by_hybrid(&j.left).or_else(|| find_rank_by_hybrid(&j.right))
        }
        LogicalPlan::LeftOuterJoin(j) => {
            find_rank_by_hybrid(&j.left).or_else(|| find_rank_by_hybrid(&j.right))
        }
        LogicalPlan::Limit(l) => find_rank_by_hybrid(&l.input),
        LogicalPlan::Skip(s) => find_rank_by_hybrid(&s.input),
        LogicalPlan::CommunityLookup(c) => find_rank_by_hybrid(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_rank_by_hybrid(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_rank_by_hybrid),
        LogicalPlan::Aggregate(a) => find_rank_by_hybrid(&a.input),
        LogicalPlan::Sort(s) => find_rank_by_hybrid(&s.input),
        LogicalPlan::Distinct(d) => find_rank_by_hybrid(&d.input),
        LogicalPlan::Unwind(u) => find_rank_by_hybrid(&u.input),
        LogicalPlan::ProcedureCall(p) => find_rank_by_hybrid(&p.input),
        LogicalPlan::NamedPath(np) => find_rank_by_hybrid(&np.input),
        LogicalPlan::DynamicLimit(l) => find_rank_by_hybrid(&l.input),
        LogicalPlan::Delete(d) => find_rank_by_hybrid(&d.input),
        LogicalPlan::Set(s) => find_rank_by_hybrid(&s.input),
        LogicalPlan::Remove(r) => find_rank_by_hybrid(&r.input),
        LogicalPlan::Merge(m) => {
            find_rank_by_hybrid(&m.match_branch).or_else(|| find_rank_by_hybrid(&m.create_branch))
        }
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
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
// Pin 3 — Aggregation lowers to LogicalAggregate (M4-33 lights this)
//
// Modified in-place per the prompt's pre-exit re-verify discipline:
// the M4-31 + M4-32 deferral assertion is replaced with the M4-33
// proper-lowering assertion. Modified-in-place tests do NOT reduce
// gross test-count; the modification corresponds to D-26-style
// `NotImplementedAtM4_31` marker removal per ADR-038 §2 D-28.
// =====================================================================

#[test]
fn lower_aggregation_to_logical_aggregate() {
    use arcgraph_query::logical_plan::AggregationKind;

    // count() in RETURN — single-row aggregate (no group_by keys).
    let input = "MATCH (n:Person) RETURN count(n)";
    let plan = lower(input, &cat_bare()).expect("aggregation lowers at M4-33");
    let s = shape(&plan);
    assert!(
        s.contains(&"Aggregate"),
        "expected Aggregate in plan: {s:?}"
    );
    assert!(s.contains(&"Project"), "expected Project in plan: {s:?}");

    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(
        agg.group_by.len(),
        0,
        "single-row aggregate has empty group_by"
    );
    assert_eq!(agg.aggregations.len(), 1, "one aggregation function call");
    assert_eq!(agg.aggregations[0].function, AggregationKind::Count);

    // collect() in RETURN — same path, different aggregation kind.
    // Uses a Node argument (`n` not `n.prop`) to side-step v1.0
    // stub-catalog property-type inference.
    let input2 = "MATCH (n:Person) RETURN collect(n)";
    let plan2 = lower(input2, &cat_bare()).expect("collect lowers at M4-33");
    let agg2 = find_aggregate(&plan2).expect("Aggregate present for collect()");
    assert_eq!(agg2.aggregations.len(), 1);
    assert_eq!(agg2.aggregations[0].function, AggregationKind::Collect);

    // Mixed group_by + aggregation: `RETURN n.name, count(n)` →
    // group_by=[n.name], aggregations=[count(n)].
    let input3 = "MATCH (n:Person) RETURN n.name, count(n)";
    let plan3 = lower(input3, &cat_bare()).expect("mixed group_by + agg lowers");
    let agg3 = find_aggregate(&plan3).expect("Aggregate present");
    assert_eq!(
        agg3.group_by.len(),
        1,
        "expected 1 group_by item (n.name); got: {:?}",
        agg3.group_by
    );
    assert_eq!(
        agg3.aggregations.len(),
        1,
        "expected 1 aggregation (count); got: {:?}",
        agg3.aggregations
    );
}

fn find_aggregate(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalAggregate> {
    match p {
        LogicalPlan::Aggregate(a) => Some(a),
        LogicalPlan::Filter(f) => find_aggregate(&f.input),
        LogicalPlan::Project(pr) => find_aggregate(&pr.input),
        LogicalPlan::Join(j) => find_aggregate(&j.left).or_else(|| find_aggregate(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => {
            find_aggregate(&j.left).or_else(|| find_aggregate(&j.right))
        }
        LogicalPlan::Limit(l) => find_aggregate(&l.input),
        LogicalPlan::Skip(s) => find_aggregate(&s.input),
        LogicalPlan::CommunityLookup(c) => find_aggregate(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_aggregate(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_aggregate),
        LogicalPlan::Sort(s) => find_aggregate(&s.input),
        LogicalPlan::Distinct(d) => find_aggregate(&d.input),
        LogicalPlan::Unwind(u) => find_aggregate(&u.input),
        LogicalPlan::ProcedureCall(p) => find_aggregate(&p.input),
        LogicalPlan::NamedPath(np) => find_aggregate(&np.input),
        LogicalPlan::DynamicLimit(l) => find_aggregate(&l.input),
        LogicalPlan::Delete(d) => find_aggregate(&d.input),
        LogicalPlan::Set(s) => find_aggregate(&s.input),
        LogicalPlan::Remove(r) => find_aggregate(&r.input),
        LogicalPlan::Merge(m) => {
            find_aggregate(&m.match_branch).or_else(|| find_aggregate(&m.create_branch))
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
        | LogicalPlan::CorrelationSeed(_) => None,
    }
}

// =====================================================================
// Pin 4 — Spans preserved on every plan node (IDE contract)
// =====================================================================

#[test]
fn lower_preserves_span_on_every_node() {
    let input =
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a, b SKIP 5 LIMIT 10";
    let plan = lower(input, &cat_full()).expect("lower");

    // Every node's span MUST be translatable to a byte range that
    // points into the input. This is the M4-31 IDE contract.
    let mut all_spans = Vec::new();
    collect_spans(&plan, &mut all_spans);
    assert!(!all_spans.is_empty(), "expected at least one span");

    for sp in all_spans {
        // The span must have non-zero coordinates.
        assert!(
            sp.start_line >= 1 && sp.start_col >= 1,
            "span has zero coordinates: {sp:?}"
        );
        assert!(
            sp.end_line >= sp.start_line,
            "span end_line precedes start_line: {sp:?}"
        );
        // Translate to byte range — since this is a single-line input,
        // the offset must fit within the input length.
        if sp.start_line == 1 {
            let off = sp.start_col.saturating_sub(1);
            assert!(
                off <= input.len(),
                "span start_col exceeds input length: span={sp:?}, input_len={}",
                input.len()
            );
        }
    }
}

fn collect_spans(p: &LogicalPlan, out: &mut Vec<arcgraph_query::Span>) {
    out.push(p.span().clone());
    match p {
        LogicalPlan::Filter(f) => collect_spans(&f.input, out),
        LogicalPlan::Project(pr) => collect_spans(&pr.input, out),
        LogicalPlan::Join(j) => {
            collect_spans(&j.left, out);
            collect_spans(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_spans(&j.left, out);
            collect_spans(&j.right, out);
        }
        LogicalPlan::Limit(l) => collect_spans(&l.input, out),
        LogicalPlan::Skip(s) => collect_spans(&s.input, out),
        LogicalPlan::CommunityLookup(c) => collect_spans(&c.input, out),
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_spans(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_spans(arm, out);
            }
        }
        LogicalPlan::Aggregate(a) => collect_spans(&a.input, out),
        LogicalPlan::Sort(s) => collect_spans(&s.input, out),
        LogicalPlan::Distinct(d) => collect_spans(&d.input, out),
        LogicalPlan::Unwind(u) => collect_spans(&u.input, out),
        LogicalPlan::ProcedureCall(p) => collect_spans(&p.input, out),
        LogicalPlan::NamedPath(np) => collect_spans(&np.input, out),
        LogicalPlan::DynamicLimit(l) => collect_spans(&l.input, out),
        LogicalPlan::Delete(d) => collect_spans(&d.input, out),
        LogicalPlan::Set(s) => collect_spans(&s.input, out),
        LogicalPlan::Remove(r) => collect_spans(&r.input, out),
        LogicalPlan::Merge(m) => {
            collect_spans(&m.match_branch, out);
            collect_spans(&m.create_branch, out);
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
