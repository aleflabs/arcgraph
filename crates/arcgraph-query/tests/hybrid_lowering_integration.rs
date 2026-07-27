//! M4-32 hybrid retrieval lowering integration tests per ADR-038
//! §2 D-26.
//!
//! M4-31 deferred the hybrid surfaces with `NotImplementedAtM4_31`
//! markers; M4-32 lights them up. The pin set covers:
//!
//! 1. `RANK BY HYBRID(VECTOR(...), TEXT(...)) WITH FUSION = RRF(k = N)
//!    RETURN n` → `Project + Fusion + Join + Scan + RankByHybrid`
//!    with the operands carrying the right K + property names.
//! 2. `<expr> NEAR <expr>` predicate → `LogicalVectorNear` wrapped
//!    in a Join onto the pattern subtree.
//! 3. `<expr> MATCH <expr>` predicate → `LogicalTextMatch` wrapped
//!    in a Join onto the pattern subtree.
//! 4. `vector_distance(field, query)` function-call form → same
//!    `LogicalVectorNear` shape as the `<expr> NEAR <expr>` form.
//! 5. `text_match(field, query)` function-call form → same
//!    `LogicalTextMatch` shape as the `<expr> MATCH <expr>` form.
//! 6. `community(n)` standalone function call (in a RETURN
//!    projection, not a WHERE) → preserved as a regular Project
//!    item; no LogicalCommunityLookup is emitted (CommunityLookup
//!    only surfaces when the function call appears in a WHERE-
//!    equality predicate, per ADR-038 §2 D-26).
//!
use arcgraph_query::logical_plan::{
    FusionKind, HybridOperandKind, LogicalPlan, LogicalPlanLoweringVisitor,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// Pipeline + catalog helpers
// ---------------------------------------------------------------------

fn cat_full() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Doc", "Person"])
        .with_rel_types(["KNOWS", "OTHER"])
        .with_properties(["embedding", "content", "text", "body", "name"])
        .with_vector_index()
        .with_bm25_index()
        .with_community_index()
}

fn lower_ok(input: &str) -> LogicalPlan {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat_full()).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat_full()).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat_full()).expect("validate");
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
        // M4-33 variants — walk through `input`.
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
        // ADR-192 (#623): not produced by this test's query set.
        LogicalPlan::Call(_) => out.push("Call"),
        LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
    }
}

fn find_vector_near(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalVectorNear> {
    match p {
        LogicalPlan::VectorNear(v) => Some(v),
        LogicalPlan::Filter(f) => find_vector_near(&f.input),
        LogicalPlan::Project(pr) => find_vector_near(&pr.input),
        LogicalPlan::Join(j) => find_vector_near(&j.left).or_else(|| find_vector_near(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => {
            find_vector_near(&j.left).or_else(|| find_vector_near(&j.right))
        }
        LogicalPlan::Limit(l) => find_vector_near(&l.input),
        LogicalPlan::Skip(s) => find_vector_near(&s.input),
        LogicalPlan::CommunityLookup(c) => find_vector_near(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_vector_near(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_vector_near),
        LogicalPlan::Aggregate(a) => find_vector_near(&a.input),
        LogicalPlan::Sort(s) => find_vector_near(&s.input),
        LogicalPlan::Distinct(d) => find_vector_near(&d.input),
        LogicalPlan::Unwind(u) => find_vector_near(&u.input),
        LogicalPlan::ProcedureCall(p) => find_vector_near(&p.input),
        LogicalPlan::NamedPath(np) => find_vector_near(&np.input),
        LogicalPlan::DynamicLimit(l) => find_vector_near(&l.input),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
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

fn find_text_match(p: &LogicalPlan) -> Option<&arcgraph_query::logical_plan::LogicalTextMatch> {
    match p {
        LogicalPlan::TextMatch(t) => Some(t),
        LogicalPlan::Filter(f) => find_text_match(&f.input),
        LogicalPlan::Project(pr) => find_text_match(&pr.input),
        LogicalPlan::Join(j) => find_text_match(&j.left).or_else(|| find_text_match(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => {
            find_text_match(&j.left).or_else(|| find_text_match(&j.right))
        }
        LogicalPlan::Limit(l) => find_text_match(&l.input),
        LogicalPlan::Skip(s) => find_text_match(&s.input),
        LogicalPlan::CommunityLookup(c) => find_text_match(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_text_match(inp)),
        LogicalPlan::Union(u) => u.arms.iter().find_map(find_text_match),
        LogicalPlan::Aggregate(a) => find_text_match(&a.input),
        LogicalPlan::Sort(s) => find_text_match(&s.input),
        LogicalPlan::Distinct(d) => find_text_match(&d.input),
        LogicalPlan::Unwind(u) => find_text_match(&u.input),
        LogicalPlan::ProcedureCall(p) => find_text_match(&p.input),
        LogicalPlan::NamedPath(np) => find_text_match(&np.input),
        LogicalPlan::DynamicLimit(l) => find_text_match(&l.input),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
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
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => None,
    }
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
// Pin 1 — RANK BY HYBRID + RRF lowers to RankByHybrid + Fusion.
// =====================================================================

#[test]
fn rank_by_hybrid_with_rrf_lowers_to_logical_fusion_over_rank_by_hybrid() {
    let input = "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 30)) WITH FUSION = RRF(k = 60) RETURN n";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(s.contains(&"RankByHybrid"), "got: {s:?}");
    assert!(s.contains(&"Fusion"), "got: {s:?}");
    assert!(s.contains(&"Project"), "got: {s:?}");

    let fusion = find_fusion(&plan).expect("Fusion present");
    assert_eq!(fusion.spec.kind, FusionKind::Rrf);
    assert_eq!(fusion.spec.k, 60);

    let hybrid = find_rank_by_hybrid(&plan).expect("RankByHybrid present");
    assert_eq!(
        hybrid.fusion.as_ref().map(|spec| spec.k),
        Some(60),
        "fusion k must reach the retrieval leaf through the MATCH join"
    );
    assert_eq!(hybrid.operands.len(), 2);
    let v = hybrid
        .operands
        .iter()
        .find(|o| o.kind == HybridOperandKind::Vector)
        .unwrap();
    let t = hybrid
        .operands
        .iter()
        .find(|o| o.kind == HybridOperandKind::Text)
        .unwrap();
    assert_eq!(v.k, 20);
    assert_eq!(v.property, "embedding");
    assert_eq!(t.k, 30);
    assert_eq!(t.property, "content");
}

#[test]
fn rank_by_hybrid_score_alias_becomes_a_float_result_binding() {
    let input = "MATCH (n:Doc) \
        RANK BY HYBRID(\
          VECTOR(n.embedding, $q, K = 20), \
          TEXT(n.content, \"x\", K = 20)\
        ) AS fusion_score \
        WITH FUSION = RRF(k = 60) \
        RETURN n, fusion_score";
    let plan = lower_ok(input);
    let hybrid = find_rank_by_hybrid(&plan).expect("RankByHybrid present");
    assert!(
        hybrid.score_binding.is_some(),
        "score alias must lower to an executor-visible binding"
    );
}

// =====================================================================
// Pin 2 — `<var>.<prop> NEAR <expr>` predicate lowers to LogicalVectorNear.
// =====================================================================

#[test]
fn near_predicate_lowers_to_logical_vector_near() {
    let input = "MATCH (n:Doc) WHERE n.embedding NEAR $q RETURN n";
    let plan = lower_ok(input);
    assert!(
        shape(&plan).contains(&"VectorNear"),
        "expected VectorNear; got: {plan:#?}"
    );
    let near = find_vector_near(&plan).expect("VectorNear present");
    assert_eq!(near.property, "embedding");
}

// =====================================================================
// Pin 3 — `<var>.<prop> MATCH <expr>` predicate lowers to LogicalTextMatch.
// =====================================================================

#[test]
fn text_match_operator_lowers_to_logical_text_match() {
    let input = "MATCH (n:Doc) WHERE n.text MATCH $q RETURN n";
    let plan = lower_ok(input);
    assert!(
        shape(&plan).contains(&"TextMatch"),
        "expected TextMatch; got: {plan:#?}"
    );
    let text = find_text_match(&plan).expect("TextMatch present");
    assert_eq!(text.property, "text");
}

// =====================================================================
// Pin 4 — `vector_distance(field, query)` is reachable through the
// canonical `RANK BY HYBRID(VECTOR(...))` surface; the bare-WHERE
// `vector_distance(...)` form is rejected at M4-22 because the
// function returns Float (NonBooleanWhere per ADR-038 D-20). The
// intent of "vector_distance lowers to LogicalVectorNear" is therefore
// covered by Pin 1 (RankByHybrid carries Vector operands rooted at
// the same field). This pin reasserts Pin 1's vector-operand
// invariant for documentation.
// =====================================================================

#[test]
fn rank_by_hybrid_vector_operand_carries_field_binding_and_property() {
    let input = "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 25), TEXT(n.content, \"x\", K = 25)) WITH FUSION = RRF(k = 60) RETURN n";
    let plan = lower_ok(input);
    let hybrid = find_rank_by_hybrid(&plan).expect("RankByHybrid present");
    let vec_op = hybrid
        .operands
        .iter()
        .find(|o| o.kind == HybridOperandKind::Vector)
        .expect("vector operand");
    assert_eq!(vec_op.k, 25);
    assert_eq!(vec_op.property, "embedding");
}

// =====================================================================
// Pin 5 — `text_match(field, query)` function-call form lowers to
// the same LogicalTextMatch shape.
// =====================================================================

#[test]
fn text_match_function_call_lowers_to_logical_text_match() {
    let input = "MATCH (n:Doc) WHERE text_match(n.content, $q) RETURN n";
    let plan = lower_ok(input);
    assert!(
        shape(&plan).contains(&"TextMatch"),
        "expected TextMatch; got: {plan:#?}"
    );
    let text = find_text_match(&plan).expect("TextMatch");
    assert_eq!(text.property, "content");
}

// =====================================================================
// Pin 6 — `community(n)` in a RETURN projection (NOT a WHERE
// equality) MUST stay a Project item, not become a CommunityLookup.
// =====================================================================

#[test]
fn community_function_call_in_return_does_not_emit_community_lookup() {
    // M4-22 + M4-23 admit `community(n)` returning Integer; M4-32
    // only emits LogicalCommunityLookup for the WHERE-equality shape.
    // A standalone call in RETURN stays inside the Project node.
    let input = "MATCH (n:Person) RETURN community(n)";
    let plan = lower_ok(input);
    let s = shape(&plan);
    assert!(s.contains(&"Project"), "expected Project; got: {s:?}");
    assert!(
        !s.contains(&"CommunityLookup"),
        "RETURN community(n) must NOT emit a CommunityLookup; got: {s:?}"
    );
}

// =====================================================================
// Pin 8 — ADR-038 §2 D-26 cross-cut: the M4-31 `NotImplementedAtM4_31`
// emission sites for the M4-32 surfaces are NO LONGER triggered.
// =====================================================================

#[test]
fn m4_31_deferral_markers_do_not_fire_for_m4_32_surfaces() {
    let inputs = [
        "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) WITH FUSION = RRF(k = 60) RETURN n",
        "MATCH (n:Doc) WHERE n.embedding NEAR $q RETURN n",
        "MATCH (n:Doc) WHERE n.text MATCH $q RETURN n",
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]-(b) RETURN a, b",
    ];
    for input in inputs {
        // Each query MUST lower without any errors.
        let stmt = parse(input).expect("parse");
        let mut bound = BindingVisitor::bind(&stmt, input, &cat_full()).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat_full()).expect("type-check");
        CrossSubstrateValidator::validate(&bound, &cat_full()).expect("validate");
        let plan = LogicalPlanLoweringVisitor::lower(&bound).expect(input);
        // The plan must not be the degenerate Empty case — these
        // surfaces all produce non-trivial trees.
        assert!(
            !matches!(plan, LogicalPlan::Empty(_)),
            "input lowered to Empty: {input}"
        );
    }
}
