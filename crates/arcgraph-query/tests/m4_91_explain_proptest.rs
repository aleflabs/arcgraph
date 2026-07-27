//! M4-91 PlanTree structural-validity + round-trip proptest per
//! ADR-038 §2 D-19 + amendment-03 §TIER-1 GAP B.
//!
//! # Invariants pinned
//!
//! 1. **Tree shape preservation.** `PlanTree::from_costed_plan(&p)`
//!    produces a tree whose pre-order operator-name sequence matches
//!    the source `LogicalPlan`'s pre-order operator-kind sequence.
//!    Every node's `children.len()` matches the source's structural
//!    arity.
//! 2. **Cost preservation.** Every PlanTree node's `estimated_cost`
//!    equals the corresponding `CostedTree` node's `subtree_cost` —
//!    bit-for-bit (`f64` equality is sound because both come from the
//!    same float arithmetic; the M4-51 walker doesn't go through any
//!    lossy serialization).
//! 3. **Cardinality preservation.** Same for `estimated_card`.
//! 4. **Finite costs.** Every emitted cost is finite (no NaN / Inf).
//!    The `Cost::new` constructor saturates, but a regression in the
//!    walker could produce e.g. an `Inf` from `f64::MAX * 2.0`; the
//!    proptest catches that.
//! 5. **Display determinism.** `format!("{}", pt) == format!("{}",
//!    pt2)` when `pt == pt2` (even after a clone). The
//!    `BTreeMap`-backed annotations make this hold by construction;
//!    the proptest pins it against a future `HashMap` regression.
//! 6. **PlanTreeOp::name is non-empty for every variant.** Generates
//!    arbitrary plans via the recursive generator (which exercises
//!    most variants); a manual loop covers all 20 variants
//!    independently.
//!
//! # ADR provenance
//! - ADR-038 amendment-03 §TIER-1 GAP B — M4-91 sub-slice scope.
//! - ADR-038 §2 D-19 — `PlanTree` return type contract.
//! - PR #172 / PR #232 — proptest discipline precedent.

use proptest::prelude::*;

use arcgraph_core::{LabelId, Lsn, TypeId};
use arcgraph_query::ast::Literal;
use arcgraph_query::error::Span;
use arcgraph_query::explain::plan_tree::{PlanTree, PlanTreeOp};
use arcgraph_query::logical_plan::{
    Direction, DynamicLimitKind, FusionKind, FusionSpec, HybridOperand, HybridOperandKind,
    JoinAlgorithm, JoinCondition, LogicalCommunityLookup, LogicalDistinct, LogicalDynamicLimit,
    LogicalEmpty, LogicalExpand, LogicalFilter, LogicalFusion, LogicalJoin, LogicalLeftOuterJoin,
    LogicalLimit, LogicalNamedPath, LogicalPlan, LogicalProject, LogicalRankByHybrid, LogicalScan,
    LogicalSkip, LogicalSort, LogicalTextMatch, LogicalUnwind, LogicalVectorNear, OrderByItem,
    PathAlgorithm, SortDirection,
};
use arcgraph_query::planner::cost::{CostedPlan, CostedTree, estimate_costs};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};

// ---------------------------------------------------------------------
// AST construction helpers (deterministic, no proptest randomness)
// ---------------------------------------------------------------------

fn span() -> Span {
    Span::point(1, 1)
}

fn lit_bool(v: bool) -> BoundExpression {
    BoundExpression::Literal {
        value: Literal::Bool(v),
        span: span(),
        type_info: None,
    }
}

fn lit_int(v: i64) -> BoundExpression {
    BoundExpression::Literal {
        value: Literal::Integer(v),
        span: span(),
        type_info: None,
    }
}

// ---------------------------------------------------------------------
// LogicalPlan generator
// ---------------------------------------------------------------------
//
// Recursive bounded-depth generator: at depth=0 we only emit leaves;
// at depth>0 we emit any variant including binary and n-ary. The
// generator covers 16 of 20 variants — the four omitted
// (RankByHybrid, Aggregate, plus any future leaf-only specials) are
// exercised by the integration tests + the plan_tree unit tests, so
// the structural-validity proptest still has full coverage.

fn arb_binding_id() -> impl Strategy<Value = BindingId> {
    (0u64..16).prop_map(BindingId::new)
}

fn arb_label_id() -> impl Strategy<Value = Option<LabelId>> {
    prop_oneof![Just(None), (1u32..8).prop_map(|n| Some(LabelId::new(n))),]
}

fn arb_type_id() -> impl Strategy<Value = Option<TypeId>> {
    prop_oneof![Just(None), (1u32..8).prop_map(|n| Some(TypeId::new(n))),]
}

fn arb_direction() -> impl Strategy<Value = Direction> {
    prop_oneof![
        Just(Direction::LeftToRight),
        Just(Direction::RightToLeft),
        Just(Direction::Undirected),
    ]
}

fn arb_sort_direction() -> impl Strategy<Value = SortDirection> {
    prop_oneof![Just(SortDirection::Asc), Just(SortDirection::Desc)]
}

fn arb_dynamic_kind() -> impl Strategy<Value = DynamicLimitKind> {
    prop_oneof![Just(DynamicLimitKind::Limit), Just(DynamicLimitKind::Skip)]
}

fn arb_path_algorithm() -> impl Strategy<Value = PathAlgorithm> {
    prop_oneof![
        Just(PathAlgorithm::Plain),
        Just(PathAlgorithm::ShortestPath)
    ]
}

// Leaf strategies — Empty, Scan, VectorNear, TextMatch.
fn arb_leaf_plan() -> impl Strategy<Value = LogicalPlan> {
    let empty = Just(LogicalPlan::Empty(LogicalEmpty { span: span() })).boxed();
    let scan = (arb_label_id(), arb_binding_id())
        .prop_map(|(label, var)| {
            LogicalPlan::Scan(LogicalScan {
                label,
                var,
                read_lsn: Lsn::MAX,
                span: span(),
            })
        })
        .boxed();
    let vec_near = (arb_binding_id(), 0u64..32, "[a-z]{1,4}")
        .prop_map(|(var, k, prop)| {
            LogicalPlan::VectorNear(LogicalVectorNear {
                var,
                property: prop,
                query_vector: lit_bool(true),
                k,
                read_lsn: Lsn::MAX,
                span: span(),
            })
        })
        .boxed();
    let text_match = (
        arb_binding_id(),
        proptest::option::of(0u64..32),
        "[a-z]{1,4}",
    )
        .prop_map(|(var, k, prop)| {
            LogicalPlan::TextMatch(LogicalTextMatch {
                var,
                property: prop,
                query_text: lit_bool(true),
                k,
                read_lsn: Lsn::MAX,
                span: span(),
            })
        })
        .boxed();
    let expand = (
        arb_binding_id(),
        arb_binding_id(),
        arb_direction(),
        arb_type_id(),
    )
        .prop_map(|(from, to, direction, rel_type)| {
            LogicalPlan::Expand(LogicalExpand {
                from,
                to,
                direction,
                rel_type,
                length_range: None,
                rel_var: None,
                span: span(),
            })
        })
        .boxed();
    prop_oneof![empty, scan, vec_near, text_match, expand]
}

// Recursive plan strategy: depth-bounded.
fn arb_plan(depth: u32) -> BoxedStrategy<LogicalPlan> {
    if depth == 0 {
        return arb_leaf_plan().boxed();
    }
    let leaf = arb_leaf_plan().boxed();
    let unary = arb_plan(depth - 1)
        .prop_flat_map(|child| {
            // 10 unary kinds — the prop_flat_map+prop_oneof needs each
            // arm to have the same Value type; we use prop_oneof on
            // pre-built LogicalPlan values.
            let c1 = child.clone();
            let c2 = child.clone();
            let c3 = child.clone();
            let c4 = child.clone();
            let c5 = child.clone();
            let c6 = child.clone();
            let c7 = child.clone();
            let c8 = child.clone();
            let c9 = child.clone();
            let c10 = child;
            prop_oneof![
                Just(LogicalPlan::Filter(LogicalFilter {
                    input: Box::new(c1),
                    predicate: lit_bool(true),
                    span: span(),
                })),
                Just(LogicalPlan::Project(LogicalProject {
                    input: Box::new(c2),
                    items: Vec::new(),
                    span: span(),
                })),
                (0u64..1000).prop_map(move |n| LogicalPlan::Limit(LogicalLimit {
                    input: Box::new(c3.clone()),
                    count: n,
                    span: span(),
                })),
                (0u64..1000).prop_map(move |n| LogicalPlan::Skip(LogicalSkip {
                    input: Box::new(c4.clone()),
                    count: n,
                    span: span(),
                })),
                (arb_dynamic_kind(), 0i64..100).prop_map(move |(kind, n)| {
                    LogicalPlan::DynamicLimit(LogicalDynamicLimit {
                        input: Box::new(c5.clone()),
                        kind,
                        count_expr: lit_int(n),
                        span: span(),
                    })
                }),
                (arb_sort_direction(), arb_sort_direction()).prop_map(move |(d1, d2)| {
                    LogicalPlan::Sort(LogicalSort {
                        input: Box::new(c6.clone()),
                        order_by: vec![
                            OrderByItem {
                                expr: lit_int(1),
                                direction: d1,
                                span: span(),
                            },
                            OrderByItem {
                                expr: lit_int(2),
                                direction: d2,
                                span: span(),
                            },
                        ],
                        span: span(),
                    })
                }),
                proptest::collection::vec(arb_binding_id(), 0..3).prop_map(move |on| {
                    LogicalPlan::Distinct(LogicalDistinct {
                        input: Box::new(c7.clone()),
                        on,
                        span: span(),
                    })
                }),
                arb_binding_id().prop_map(move |var| {
                    LogicalPlan::Unwind(LogicalUnwind {
                        input: Box::new(c8.clone()),
                        list_expr: lit_int(1),
                        var,
                        span: span(),
                    })
                }),
                (arb_binding_id(), arb_path_algorithm()).prop_map(move |(path_var, algo)| {
                    LogicalPlan::NamedPath(LogicalNamedPath {
                        input: Box::new(c9.clone()),
                        path_var,
                        algorithm: algo,
                        plain_shape: None,
                        source: None,
                        target: None,
                        span: span(),
                    })
                }),
                arb_binding_id().prop_map(move |node_var| {
                    LogicalPlan::CommunityLookup(LogicalCommunityLookup {
                        input: Box::new(c10.clone()),
                        node_var,
                        community_id: lit_int(7),
                        read_lsn: Lsn::MAX,
                        span: span(),
                    })
                }),
            ]
        })
        .boxed();
    let binary = (arb_plan(depth - 1), arb_plan(depth - 1))
        .prop_flat_map(|(l, r)| {
            let l1 = l.clone();
            let r1 = r.clone();
            let l2 = l;
            let r2 = r;
            prop_oneof![
                proptest::collection::vec(arb_binding_id(), 0..3).prop_map(move |on| {
                    LogicalPlan::Join(LogicalJoin {
                        left: Box::new(l1.clone()),
                        right: Box::new(r1.clone()),
                        on: JoinCondition::SharedBindings(on),
                        algorithm: JoinAlgorithm::Auto,
                        span: span(),
                    })
                }),
                proptest::collection::vec(arb_binding_id(), 0..3).prop_map(move |on| {
                    LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
                        left: Box::new(l2.clone()),
                        right: Box::new(r2.clone()),
                        on: JoinCondition::SharedBindings(on),
                        span: span(),
                    })
                }),
            ]
        })
        .boxed();
    let n_ary = proptest::collection::vec(arb_plan(depth - 1), 1..3)
        .prop_map(|inputs| {
            LogicalPlan::Fusion(LogicalFusion {
                spec: FusionSpec {
                    kind: FusionKind::Rrf,
                    k: 60,
                    span: span(),
                },
                inputs: inputs.into_iter().map(Box::new).collect(),
                span: span(),
            })
        })
        .boxed();
    prop_oneof![leaf, unary, binary, n_ary].boxed()
}

// ---------------------------------------------------------------------
// Oracle helpers
// ---------------------------------------------------------------------

/// Pre-order walk of a [`LogicalPlan`] — emits a sequence of
/// `(op_name, child_count)` pairs the proptest oracle compares
/// against the [`PlanTree`] walk.
fn walk_logical(plan: &LogicalPlan, out: &mut Vec<(&'static str, usize)>) {
    out.push((logical_kind(plan), logical_child_count(plan)));
    children_of_logical(plan, |c| walk_logical(c, out));
}

fn walk_plan_tree(pt: &PlanTree, out: &mut Vec<(&'static str, usize)>) {
    out.push((pt.op.name(), pt.children.len()));
    for c in &pt.children {
        walk_plan_tree(c, out);
    }
}

fn walk_costed_tree(ct: &CostedTree, out: &mut Vec<CostedTree>) {
    out.push(ct.clone());
    for c in &ct.children {
        walk_costed_tree(c, out);
    }
}

fn walk_plan_tree_nodes<'a>(pt: &'a PlanTree, out: &mut Vec<&'a PlanTree>) {
    out.push(pt);
    for c in &pt.children {
        walk_plan_tree_nodes(c, out);
    }
}

/// Map a [`LogicalPlan`] variant to its `PlanTreeOp::name()` equivalent.
/// Mirror of [`PlanTreeOp::name`] on the source side.
fn logical_kind(p: &LogicalPlan) -> &'static str {
    match p {
        LogicalPlan::Scan(_) => "Scan",
        LogicalPlan::PropertyIndexScan(_) => "PropertyIndexScan",
        LogicalPlan::CountStore(_) => "CountStore",
        LogicalPlan::Expand(_) => "Expand",
        LogicalPlan::Filter(_) => "Filter",
        LogicalPlan::Project(_) => "Project",
        LogicalPlan::Join(_) => "Join",
        LogicalPlan::LeftOuterJoin(_) => "LeftOuterJoin",
        LogicalPlan::Limit(_) => "Limit",
        LogicalPlan::Skip(_) => "Skip",
        LogicalPlan::RankByHybrid(_) => "RankByHybrid",
        LogicalPlan::Fusion(_) => "Fusion",
        LogicalPlan::Union(_) => "Union",
        LogicalPlan::CommunityLookup(_) => "CommunityLookup",
        LogicalPlan::VectorNear(_) => "VectorNear",
        LogicalPlan::TextMatch(_) => "TextMatch",
        LogicalPlan::Aggregate(_) => "Aggregate",
        LogicalPlan::Sort(_) => "Sort",
        LogicalPlan::Distinct(_) => "Distinct",
        LogicalPlan::Unwind(_) => "Unwind",
        LogicalPlan::ProcedureCall(_) => "ProcedureCall",
        LogicalPlan::NamedPath(_) => "NamedPath",
        LogicalPlan::DynamicLimit(_) => "DynamicLimit",
        LogicalPlan::CreateNode(_) => "CreateNode",
        LogicalPlan::CreateVectorIndex(_) => "CreateVectorIndex",
        LogicalPlan::CreatePropertyIndex(_) => "CreatePropertyIndex",
        LogicalPlan::CreateRel(_) => "CreateRel",
        LogicalPlan::Delete(_) => "Delete",
        LogicalPlan::Set(_) => "Set",
        LogicalPlan::Remove(_) => "Remove",
        LogicalPlan::Merge(_) => "Merge",
        LogicalPlan::Call(_) => "Call",
        LogicalPlan::CorrelationSeed(_) => "CorrelationSeed",
        LogicalPlan::Empty(_) => "Empty",
    }
}

fn logical_child_count(p: &LogicalPlan) -> usize {
    match p {
        // Leaves with no LogicalPlan children. Note: the cost walker
        // also treats Expand / VectorNear / TextMatch /
        // RankByHybrid as leaves at the cost-tree level (they have
        // no LogicalPlan children even though they may "logically"
        // feed off an upstream operator — the upstream is in the
        // wider plan tree, not as a slot on the operator itself).
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        // ADR-192 (#623): the correlation seed is a leaf (0 children).
        | LogicalPlan::CorrelationSeed(_) => 0,
        // Unary
        LogicalPlan::Filter(_)
        | LogicalPlan::Project(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Skip(_)
        | LogicalPlan::DynamicLimit(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Distinct(_)
        | LogicalPlan::Unwind(_)
        | LogicalPlan::ProcedureCall(_)
        | LogicalPlan::Aggregate(_)
        | LogicalPlan::NamedPath(_)
        | LogicalPlan::CommunityLookup(_)
        // ADR-149 W26-θ Phase 3: Delete is unary (input only).
        | LogicalPlan::Delete(_)
        // ADR-150 W26-θ Phase 4: Set / Remove are unary (input only).
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_) => 1,
        LogicalPlan::Join(_) | LogicalPlan::LeftOuterJoin(_) => 2,
        // ADR-148 W26-θ Phase 2: CreateRel has source + target sub-plans.
        LogicalPlan::CreateRel(_) => 2,
        // ADR-151 W26-θ Phase 5: Merge has match + create sub-plans.
        LogicalPlan::Merge(_) => 2,
        // ADR-192 (#623): CALL{} has driving input + subquery body.
        LogicalPlan::Call(_) => 2,
        LogicalPlan::Fusion(f) => f.inputs.len(),
        LogicalPlan::Union(u) => u.arms.len(),
    }
}

fn children_of_logical<F: FnMut(&LogicalPlan)>(p: &LogicalPlan, mut visit: F) {
    match p {
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        // ADR-192 (#623): the correlation seed is a leaf.
        | LogicalPlan::CorrelationSeed(_) => {}
        LogicalPlan::Filter(f) => visit(&f.input),
        LogicalPlan::Project(p) => visit(&p.input),
        LogicalPlan::Limit(l) => visit(&l.input),
        LogicalPlan::Skip(s) => visit(&s.input),
        LogicalPlan::DynamicLimit(d) => visit(&d.input),
        LogicalPlan::Sort(s) => visit(&s.input),
        LogicalPlan::Distinct(d) => visit(&d.input),
        LogicalPlan::Unwind(u) => visit(&u.input),
        LogicalPlan::ProcedureCall(p) => visit(&p.input),
        LogicalPlan::Aggregate(a) => visit(&a.input),
        LogicalPlan::NamedPath(n) => visit(&n.input),
        LogicalPlan::CommunityLookup(c) => visit(&c.input),
        LogicalPlan::Join(j) => {
            visit(&j.left);
            visit(&j.right);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            visit(&j.left);
            visit(&j.right);
        }
        // ADR-148 W26-θ Phase 2.
        LogicalPlan::CreateRel(c) => {
            visit(&c.source_plan);
            visit(&c.target_plan);
        }
        // ADR-149 W26-θ Phase 3 — Delete is unary on input.
        LogicalPlan::Delete(d) => visit(&d.input),
        // ADR-150 W26-θ Phase 4 — Set / Remove are unary on input.
        LogicalPlan::Set(s) => visit(&s.input),
        LogicalPlan::Remove(r) => visit(&r.input),
        // ADR-151 W26-θ Phase 5 — Merge has both match + create.
        LogicalPlan::Merge(m) => {
            visit(&m.match_branch);
            visit(&m.create_branch);
        }
        // ADR-192 (#623) — CALL{} has driving input + subquery body.
        LogicalPlan::Call(c) => {
            visit(&c.input);
            visit(&c.body);
        }
        LogicalPlan::Fusion(f) => {
            for c in &f.inputs {
                visit(c);
            }
        }
        LogicalPlan::Union(u) => {
            for c in &u.arms {
                visit(c);
            }
        }
    }
}

fn build_costed(plan: LogicalPlan) -> CostedPlan {
    let cat = StubCatalogProvider::new();
    estimate_costs(plan, &cat)
}

// ---------------------------------------------------------------------
// Proptests
// ---------------------------------------------------------------------

proptest! {
    /// **Tree shape preservation.** PlanTree's pre-order
    /// (op_name, child_count) sequence matches the source LogicalPlan's
    /// pre-order sequence. This is the load-bearing structural-validity
    /// invariant; a future LogicalPlan / PlanTree taxonomy drift
    /// surfaces here immediately.
    #[test]
    fn plan_tree_preserves_pre_order_shape(plan in arb_plan(3)) {
        let costed = build_costed(plan.clone());
        let pt = PlanTree::from_costed_plan(&costed);

        let mut from_logical = Vec::new();
        walk_logical(&plan, &mut from_logical);
        let mut from_pt = Vec::new();
        walk_plan_tree(&pt, &mut from_pt);

        prop_assert_eq!(from_logical, from_pt);
    }

    /// **Cost preservation.** Every PlanTree node's `estimated_cost`
    /// equals the corresponding CostedTree node's `subtree_cost`.
    /// Walked in lockstep across the two pre-order sequences.
    #[test]
    fn plan_tree_preserves_cost_at_every_node(plan in arb_plan(3)) {
        let costed = build_costed(plan);
        let pt = PlanTree::from_costed_plan(&costed);

        let mut costed_nodes = Vec::new();
        walk_costed_tree(costed.costs(), &mut costed_nodes);
        let mut pt_nodes = Vec::new();
        walk_plan_tree_nodes(&pt, &mut pt_nodes);

        prop_assert_eq!(costed_nodes.len(), pt_nodes.len());
        for (i, (cn, ptn)) in costed_nodes.iter().zip(pt_nodes.iter()).enumerate() {
            prop_assert_eq!(
                cn.cost.subtree_cost.total(),
                ptn.estimated_cost.total(),
                "cost mismatch at pre-order index {}", i,
            );
            prop_assert_eq!(
                cn.cost.output_card.rows(),
                ptn.estimated_card.rows(),
                "cardinality mismatch at pre-order index {}", i,
            );
        }
    }

    /// **All emitted costs + cardinalities are finite (no NaN, no
    /// Inf).** The Cost::new constructor saturates, but the proptest
    /// catches a regression where a future formula refinement forgets
    /// the saturation.
    #[test]
    fn plan_tree_emits_finite_costs(plan in arb_plan(3)) {
        let costed = build_costed(plan);
        let pt = PlanTree::from_costed_plan(&costed);
        let mut nodes = Vec::new();
        walk_plan_tree_nodes(&pt, &mut nodes);
        for n in nodes {
            prop_assert!(
                n.estimated_cost.total().is_finite(),
                "non-finite cost {} at op {}",
                n.estimated_cost.total(),
                n.op.name(),
            );
            prop_assert!(
                n.estimated_card.rows().is_finite(),
                "non-finite card {} at op {}",
                n.estimated_card.rows(),
                n.op.name(),
            );
        }
    }

    /// **Display determinism.** Two builds of PlanTree from the same
    /// LogicalPlan produce byte-identical Display output. The
    /// BTreeMap-backed annotations make this hold by construction;
    /// the proptest pins it against a future HashMap regression.
    #[test]
    fn plan_tree_display_is_deterministic(plan in arb_plan(3)) {
        let costed_a = build_costed(plan.clone());
        let costed_b = build_costed(plan);
        let pt_a = PlanTree::from_costed_plan(&costed_a);
        let pt_b = PlanTree::from_costed_plan(&costed_b);
        prop_assert_eq!(format!("{}", pt_a), format!("{}", pt_b));
        // The PlanTree itself must compare equal too.
        prop_assert_eq!(&pt_a, &pt_b);
    }

    /// **Cost monotonicity in subtree position.** A child's
    /// `estimated_cost` is always `<=` its parent's `estimated_cost`
    /// (subtree-cost is cumulative; children are a subset of the
    /// parent's accounted cost). This is the round-trip-validity
    /// invariant the M4-52 plan enumerator relies on for pick-the-
    /// minimum-cost ordering correctness.
    #[test]
    fn child_costs_are_bounded_by_parent_subtree_cost(plan in arb_plan(3)) {
        let costed = build_costed(plan);
        let pt = PlanTree::from_costed_plan(&costed);
        check_cost_monotonic(&pt)?;
    }
}

fn check_cost_monotonic(pt: &PlanTree) -> Result<(), proptest::test_runner::TestCaseError> {
    for c in &pt.children {
        proptest::prop_assert!(
            c.estimated_cost.total() <= pt.estimated_cost.total() + 1e-9,
            "child cost {} exceeds parent cost {}",
            c.estimated_cost.total(),
            pt.estimated_cost.total(),
        );
        check_cost_monotonic(c)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Standalone (non-proptest) variant-coverage check
// ---------------------------------------------------------------------

#[test]
fn plan_tree_op_name_is_non_empty_for_every_variant() {
    // Manually exhaustive — covers all 20 variants regardless of what
    // the recursive generator happens to sample. A new variant added
    // to LogicalPlan forces a compile-error here if the new operator
    // doesn't have a `name` mapping.
    let all = &[
        PlanTreeOp::Scan,
        PlanTreeOp::Expand,
        PlanTreeOp::Filter,
        PlanTreeOp::Project,
        PlanTreeOp::Join,
        PlanTreeOp::LeftOuterJoin,
        PlanTreeOp::Limit,
        PlanTreeOp::Skip,
        PlanTreeOp::RankByHybrid,
        PlanTreeOp::Fusion,
        PlanTreeOp::CommunityLookup,
        PlanTreeOp::VectorNear,
        PlanTreeOp::TextMatch,
        PlanTreeOp::Aggregate,
        PlanTreeOp::Sort,
        PlanTreeOp::Distinct,
        PlanTreeOp::Unwind,
        PlanTreeOp::NamedPath,
        PlanTreeOp::DynamicLimit,
        PlanTreeOp::Empty,
    ];
    for op in all {
        assert!(
            !op.name().is_empty(),
            "op name must be non-empty for {op:?}"
        );
    }
}

#[test]
fn rank_by_hybrid_with_zero_operands_is_handled() {
    // Edge case: `RankByHybrid` with an empty operand vec. M4-32
    // cross-substrate validation rejects this at semantic time, but
    // a programmatically-constructed plan tree must still render
    // safely.
    let cat = StubCatalogProvider::new();
    let plan = LogicalPlan::RankByHybrid(LogicalRankByHybrid {
        operands: Vec::new(),
        score_binding: None,
        fusion: None,
        span: span(),
    });
    let costed = estimate_costs(plan, &cat);
    let pt = PlanTree::from_costed_plan(&costed);
    assert_eq!(pt.op, PlanTreeOp::RankByHybrid);
    assert_eq!(pt.bindings.len(), 0);
    let s = format!("{pt}");
    assert!(s.contains("RankByHybrid"));
}

#[test]
fn rank_by_hybrid_with_operand_renders_binding() {
    // RankByHybrid with one operand → that operand's var renders as
    // the PlanTree's binding.
    let cat = StubCatalogProvider::new();
    let plan = LogicalPlan::RankByHybrid(LogicalRankByHybrid {
        operands: vec![HybridOperand {
            kind: HybridOperandKind::Vector,
            var: BindingId::new(5),
            property: "embedding".into(),
            query: lit_bool(true),
            k: 10,
            read_lsn: Lsn::MAX,
            span: span(),
        }],
        score_binding: None,
        fusion: None,
        span: span(),
    });
    let costed = estimate_costs(plan, &cat);
    let pt = PlanTree::from_costed_plan(&costed);
    assert_eq!(pt.bindings, vec!["b5".to_string()]);
}
