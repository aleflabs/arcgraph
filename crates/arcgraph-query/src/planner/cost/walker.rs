//! Cost-walker — the M4-51 entry point that walks a [`LogicalPlan`]
//! and produces a [`CostedPlan`].
//!
//! # Walker shape (custom struct, not a trait)
//!
//! Per the 7-slice 3-strike pattern (M4-21 / M4-22 / M4-22b / M4-23 /
//! M4-31 / M4-32 / M4-33 — every walker concretized; zero speculative
//! traits across the chain), the walker is a CONCRETE STRUCT, not a
//! `pub trait LogicalPlanVisitor`. M4-52's plan-enumeration walker
//! is the candidate for trait extraction once its consumer surface
//! is known.
//!
//! # Snapshot capture discipline
//!
//! Per ADR-038 §2 D-25 + amendment-03 §M4-04e (issue #210), the
//! walker calls [`crate::semantic::CatalogProvider::snapshot`] EXACTLY
//! ONCE at plan-start. Every per-operator cost function reads from
//! the captured snapshot — preserving cross-key consistency
//! (`sum(label_cards) ≤ total_nodes`, etc.) across the entire plan
//! walk. Reading the catalog per-operator would race against
//! concurrent commits and produce non-monotonic cost estimates.
//!
//! # Exhaustive-match contract
//!
//! Per [`crate::logical_plan::types::LogicalPlan`] rustdoc, the
//! enum is NOT `#[non_exhaustive]`; every consumer MUST exhaustively
//! match. The walker honors this — adding a new variant to
//! [`LogicalPlan`] forces a compile-error here, which is the design
//! intent. (The future M4-33-codex-review-note 2 sentinel test
//! pattern would augment this with a compile-fail proptest, but
//! that's an additive M4-52+ concern — the exhaustive `match`
//! already provides the load-bearing guarantee.)
//!
//! # Budget
//!
//! Per ADR-036 §D-25 the M4-05 plan-build budget is 5 ms. The walker
//! is `O(plan-nodes × predicate-tree-size)` — at v1.0 plan sizes
//! (≤ 50 nodes per plan, ≤ 10 predicates per filter) the wall-clock
//! is dominated by catalog snapshot capture (~1 µs per the M4-04e
//! catalog_stats_snapshot bench) plus a few µs of arithmetic per
//! node. Total budget consumption: ~10–100 µs at v1.0 scale.

use crate::ast::LengthRange;
use crate::logical_plan::LogicalPlan;
use crate::planner::enumeration::FrozenCatalog;
use crate::semantic::CatalogProvider;
use crate::semantic::CatalogSnapshot;
use crate::semantic::SelectivityEstimator;

use super::COST_HINT_HIGH;
use super::operator;
use super::predicate;
use super::{Cardinality, Cost, CostNode, CostedPlan, CostedTree};

const SUPERNODE_DEGREE_RATIO: f64 = 10.0;

/// Crate-private helper: cost the subtree + return only the root
/// output cardinality. Used by the W25-M4-61b
/// [`crate::planner::pick_join_algorithms`] pass for the per-side
/// cardinality estimate at join-algorithm-picking time.
///
/// Forwards to [`walk`] internally — the picker can share the exact
/// same cardinality the cost walker would compute for the same plan
/// + snapshot.
#[must_use]
pub(crate) fn walker_card_for<C: CatalogProvider + ?Sized>(
    plan: LogicalPlan,
    snapshot: &CatalogSnapshot,
    estimator: &SelectivityEstimator<'_, C>,
) -> Cardinality {
    walk(&plan, snapshot, estimator).output_card()
}

/// Walk a [`LogicalPlan`] and produce a cost-annotated
/// [`CostedPlan`].
///
/// # Arguments
///
/// - `plan` — the logical plan to cost. Consumed by-value because
///   the [`CostedPlan`] wraps the plan; if callers need both the
///   original and the costed form, they should clone before
///   calling.
/// - `catalog` — borrowed [`CatalogProvider`]. The walker calls
///   `catalog.snapshot()` once and reads from the snapshot for
///   every per-operator cost call.
///
/// # Determinism
///
/// The walk is fully deterministic — same plan + same catalog
/// snapshot → same cost output. Concurrent commits to the catalog
/// CAN affect successive snapshots (and hence the cost output),
/// but within a single walk the snapshot is fixed.
#[must_use]
pub fn estimate_costs(plan: LogicalPlan, catalog: &dyn CatalogProvider) -> CostedPlan {
    let snapshot = super::capture_snapshot(catalog);
    let estimator = SelectivityEstimator::new(catalog);
    let costs = walk(&plan, &snapshot, &estimator);
    let diagnostics = supernode_firewall_diagnostics(&plan, &snapshot);
    CostedPlan::with_diagnostics(plan, costs, diagnostics)
}

/// [`estimate_costs`] sibling that reuses an externally-captured
/// [`FrozenCatalog`] instead of taking its own snapshot.
///
/// # Why
///
/// Per issue #261 (W9d retro Agent A §A-LOW-1), the EXPLAIN pipeline
/// previously captured two independent [`CatalogSnapshot`]s — one for
/// the M4-52 DP enumerator and one for the cost walker. Threading a
/// single [`FrozenCatalog`] through both stages produces apples-to-
/// apples cost annotations against the same per-key cardinalities; the
/// DP-side entry is
/// [`crate::planner::enumeration::enumerate_join_order_with_frozen`].
///
/// # Snapshot-once
///
/// `frozen.snapshot()` returns a clone of the externally-captured
/// value, so the walker still satisfies its single-snapshot contract
/// (per ADR-038 §2 D-25 + amendment-03 §M4-04e). The
/// [`SelectivityEstimator`] is built against the frozen catalog
/// (delegates everything except `snapshot()`); predicate selectivity
/// reads remain consistent across the walk.
#[must_use]
pub(crate) fn estimate_costs_with_frozen(
    plan: LogicalPlan,
    frozen: &FrozenCatalog<'_>,
) -> CostedPlan {
    let snapshot = frozen.snapshot();
    let estimator = SelectivityEstimator::new(frozen);
    let costs = walk(&plan, &snapshot, &estimator);
    let diagnostics = supernode_firewall_diagnostics(&plan, &snapshot);
    CostedPlan::with_diagnostics(plan, costs, diagnostics)
}

fn supernode_firewall_diagnostics(plan: &LogicalPlan, snapshot: &CatalogSnapshot) -> Vec<String> {
    let mut rel_types = Vec::new();
    collect_k3_rel_types(plan, &mut rel_types);
    if rel_types.is_empty() {
        return Vec::new();
    }

    snapshot
        .max_out_degree_entries()
        .iter()
        .filter(|entry| rel_types.contains(&Some(entry.rel_type)) || rel_types.contains(&None))
        .filter(|entry| {
            let avg = avg_out_degree(snapshot, entry.label, entry.rel_type);
            avg > 0.0 && entry.degree as f64 >= avg * SUPERNODE_DEGREE_RATIO
        })
        .map(|entry| {
            format!(
                "{COST_HINT_HIGH}: supernode vertex {} degree {} for label L{} rel_type T{}; per_hop_frontier_cap required for k>=3 degradation",
                entry.vertex.raw(),
                entry.degree,
                entry.label.raw(),
                entry.rel_type.raw()
            )
        })
        .collect()
}

fn avg_out_degree(
    snapshot: &CatalogSnapshot,
    label: arcgraph_core::LabelId,
    rel_type: arcgraph_core::TypeId,
) -> f64 {
    // Cross-label approximation: storage tracks exact rel-type totals,
    // not per-(source-label, rel-type) edge totals, so this is only a
    // coarse baseline for detecting clear supernodes.
    match (snapshot.label_card(label), snapshot.rel_type_card(rel_type)) {
        (Some(nodes), Some(edges)) if nodes > 0 => edges as f64 / nodes as f64,
        _ => 0.0,
    }
}

fn collect_k3_rel_types(plan: &LogicalPlan, out: &mut Vec<Option<arcgraph_core::TypeId>>) {
    match plan {
        LogicalPlan::Expand(expand) if length_range_reaches_k3(expand.length_range.as_ref()) => {
            out.push(expand.rel_type);
        }
        LogicalPlan::Filter(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Project(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Limit(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Skip(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::DynamicLimit(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Sort(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Distinct(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Unwind(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::ProcedureCall(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Aggregate(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::CommunityLookup(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::NamedPath(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Join(p) => {
            collect_k3_rel_types(&p.left, out);
            collect_k3_rel_types(&p.right, out);
        }
        LogicalPlan::LeftOuterJoin(p) => {
            collect_k3_rel_types(&p.left, out);
            collect_k3_rel_types(&p.right, out);
        }
        LogicalPlan::Fusion(p) => {
            for input in &p.inputs {
                collect_k3_rel_types(input, out);
            }
        }
        LogicalPlan::Union(p) => {
            for arm in &p.arms {
                collect_k3_rel_types(arm, out);
            }
        }
        LogicalPlan::CreateNode(p) => {
            if let Some(input) = &p.input {
                collect_k3_rel_types(input, out);
            }
        }
        LogicalPlan::CreateRel(p) => {
            collect_k3_rel_types(&p.source_plan, out);
            collect_k3_rel_types(&p.target_plan, out);
            if let Some(input) = &p.input {
                collect_k3_rel_types(input, out);
            }
        }
        LogicalPlan::Delete(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Set(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Remove(p) => collect_k3_rel_types(&p.input, out),
        LogicalPlan::Merge(p) => {
            collect_k3_rel_types(&p.match_branch, out);
            collect_k3_rel_types(&p.create_branch, out);
        }
        LogicalPlan::Call(p) => {
            collect_k3_rel_types(&p.input, out);
            collect_k3_rel_types(&p.body, out);
        }
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CorrelationSeed(_)
        | LogicalPlan::Expand(_) => {}
    }
}

fn length_range_reaches_k3(length_range: Option<&LengthRange>) -> bool {
    match length_range {
        Some(LengthRange::Unbounded) => true,
        Some(LengthRange::Cypher { max, .. }) | Some(LengthRange::Quantified { max, .. }) => {
            max.unwrap_or(u32::MAX) >= 3
        }
        None => false,
    }
}

/// Recursively cost a [`LogicalPlan`] subtree.
fn walk<C: CatalogProvider + ?Sized>(
    plan: &LogicalPlan,
    snapshot: &CatalogSnapshot,
    estimator: &SelectivityEstimator<'_, C>,
) -> CostedTree {
    match plan {
        // -----------------------------------------------------------
        // Leaf operators.
        // -----------------------------------------------------------
        LogicalPlan::Scan(scan) => {
            let (local_cost, output_card) = operator::cost_scan(scan, snapshot);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        // #1366 (Phase 2): indexed point-lookup leaf — flat lookup cost
        // + cardinality 1, decisively below the anchor scan it replaces.
        LogicalPlan::PropertyIndexScan(lookup) => {
            let (local_cost, output_card) = operator::cost_property_index_scan(lookup);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        LogicalPlan::CountStore(_) => {
            CostedTree::leaf(CostNode::leaf(Cost::new(1.0), Cardinality::new(1.0)))
        }
        LogicalPlan::Empty(_) => {
            let (local_cost, output_card) = operator::cost_empty();
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }

        // -----------------------------------------------------------
        // Hybrid retrieval leaves (no LogicalPlan children).
        // -----------------------------------------------------------
        LogicalPlan::RankByHybrid(rank) => {
            let (local_cost, output_card) = operator::cost_rank_by_hybrid(rank);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        LogicalPlan::VectorNear(near) => {
            let (local_cost, output_card) = operator::cost_vector_near(near);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        LogicalPlan::TextMatch(text) => {
            let (local_cost, output_card) = operator::cost_text_match(text);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }

        // -----------------------------------------------------------
        // Unary-input operators.
        // -----------------------------------------------------------
        LogicalPlan::Expand(expand) => {
            // Expand requires an input cardinality but does NOT carry
            // a child LogicalPlan slot — the M4-31 lowering shape pairs
            // Expand with an upstream Scan via the surrounding tree
            // structure (nested patterns lower into a chain). v1.0
            // walker treats Expand as a leaf with input_card derived
            // from the upstream Scan's cardinality estimate. To keep
            // the walker tree correct, we approximate Expand at root
            // as having the catalog's total_nodes as input.
            let synthetic_input = match snapshot.total_nodes() {
                Some(n) => Cardinality::new(n as f64),
                None => Cardinality::new(operator::FALLBACK_TENANT_NODE_COUNT),
            };
            let (local_cost, output_card) =
                operator::cost_expand(expand, synthetic_input, snapshot);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        LogicalPlan::Filter(filter) => {
            let child = walk(&filter.input, snapshot, estimator);
            let input_card = child.output_card();
            let predicate_sel = predicate::predicate_selectivity(
                &filter.predicate,
                estimator,
                // The filter's predicate is rooted at no specific
                // binding from the LogicalFilter type; pass a
                // sentinel binding (zero) — the v1.0 estimator
                // ignores it. v1.1 sketches refine.
                crate::semantic::bound_ast::BindingId::new(0),
            );
            let (local_cost, output_card) =
                operator::cost_filter(filter, input_card, predicate_sel);
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Project(project) => {
            let child = walk(&project.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_project(project, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Limit(limit) => {
            let child = walk(&limit.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_limit(limit, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Skip(skip) => {
            let child = walk(&skip.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_skip(skip, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::DynamicLimit(dyn_lim) => {
            let child = walk(&dyn_lim.input, snapshot, estimator);
            let (local_cost, output_card) =
                operator::cost_dynamic_limit(dyn_lim, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Sort(sort) => {
            let child = walk(&sort.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_sort(sort, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Distinct(distinct) => {
            let child = walk(&distinct.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_distinct(distinct, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Unwind(unwind) => {
            let child = walk(&unwind.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_unwind(unwind, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        // ADR-197 (#802): procedure / SHOW — small fixed catalog-sized
        // output over the (unit-row) input.
        LogicalPlan::ProcedureCall(p) => {
            let child = walk(&p.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_procedure_call(child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::Aggregate(aggr) => {
            let child = walk(&aggr.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_aggregate(aggr, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::CommunityLookup(lookup) => {
            let child = walk(&lookup.input, snapshot, estimator);
            let (local_cost, output_card) =
                operator::cost_community_lookup(lookup, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }
        LogicalPlan::NamedPath(named) => {
            let child = walk(&named.input, snapshot, estimator);
            let (local_cost, output_card) = operator::cost_named_path(named, child.output_card());
            let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![child],
            }
        }

        // -----------------------------------------------------------
        // Binary-input operators.
        // -----------------------------------------------------------
        LogicalPlan::Join(join) => {
            let left = walk(&join.left, snapshot, estimator);
            let right = walk(&join.right, snapshot, estimator);
            let (local_cost, output_card) =
                operator::cost_join(join, left.output_card(), right.output_card());
            let cost_node = CostNode::n_ary(
                local_cost,
                output_card,
                &[left.total_cost(), right.total_cost()],
            );
            CostedTree {
                cost: cost_node,
                children: vec![left, right],
            }
        }
        LogicalPlan::LeftOuterJoin(join) => {
            let left = walk(&join.left, snapshot, estimator);
            let right = walk(&join.right, snapshot, estimator);
            let (local_cost, output_card) =
                operator::cost_left_outer_join(join, left.output_card(), right.output_card());
            let cost_node = CostNode::n_ary(
                local_cost,
                output_card,
                &[left.total_cost(), right.total_cost()],
            );
            CostedTree {
                cost: cost_node,
                children: vec![left, right],
            }
        }

        // -----------------------------------------------------------
        // n-ary input — Fusion.
        // -----------------------------------------------------------
        LogicalPlan::Fusion(fusion) => {
            let children: Vec<CostedTree> = fusion
                .inputs
                .iter()
                .map(|input| walk(input, snapshot, estimator))
                .collect();
            let input_cards: Vec<Cardinality> =
                children.iter().map(CostedTree::output_card).collect();
            let (local_cost, output_card) = operator::cost_fusion(fusion, &input_cards);
            let child_costs: Vec<Cost> = children.iter().map(CostedTree::total_cost).collect();
            let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
            CostedTree {
                cost: cost_node,
                children,
            }
        }
        // ADR-185 (#649-A1, W28) — UNION ALL is an n-ary concat.
        // Back-of-envelope (PD#5): output cardinality = Σ arm cards;
        // local cost ≈ one streaming pass over the concatenated rows
        // (O(1) memory — NOT a materialization point; the dedup/sort
        // materialization is a separate Distinct/Sort node above this).
        LogicalPlan::Union(union) => {
            let children: Vec<CostedTree> = union
                .arms
                .iter()
                .map(|arm| walk(arm, snapshot, estimator))
                .collect();
            let output_rows: f64 = children.iter().map(|c| c.output_card().rows()).sum();
            let output_card = Cardinality::new(output_rows);
            let local_cost = Cost::new(output_rows);
            let child_costs: Vec<Cost> = children.iter().map(CostedTree::total_cost).collect();
            let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
            CostedTree {
                cost: cost_node,
                children,
            }
        }
        // ADR-147 W26-θ Phase 1: CreateNode is a write-op — O(1) write
        // per upstream row (the substrate dominates; the executor-side
        // cost is constant). #832: a multi-item leading
        // `CREATE (a),(b),(c)` lowers to a left-deep chain via the
        // optional `input` child; the walker MUST recurse it (input at
        // child index 0) so the CostedTree shape stays in LOCKSTEP with
        // the EXPLAIN plan-tree (`explain/plan_tree.rs`) + row-count
        // (`observer/row_count.rs`) walkers. Reverting this re-opens a
        // REACHABLE `EXPLAIN`/`PROFILE` panic: those walkers index
        // `child_at(costs, 0)`, which panics on a 0-child leaf. Streaming
        // semantic: one create per upstream row, emit the row extended
        // with the new binding → output card tracks the input's (1 for
        // the leading-CREATE chain leaf).
        LogicalPlan::CreateNode(c) => {
            let local_cost = Cost::new(1.0);
            match &c.input {
                Some(input) => {
                    let child = walk(input, snapshot, estimator);
                    let output_card = child.output_card();
                    let cost_node = CostNode::unary(local_cost, output_card, child.total_cost());
                    CostedTree {
                        cost: cost_node,
                        children: vec![child],
                    }
                }
                None => {
                    let output_card = Cardinality::new(1.0);
                    CostedTree::leaf(CostNode::leaf(local_cost, output_card))
                }
            }
        }
        // #830 / ADR-200: CREATE VECTOR INDEX is a write-op leaf — a
        // single O(1) catalog metadata insert, ZERO output rows. Cost
        // mirrors CreateNode (constant); cardinality is 0 (DDL returns
        // no rows). LEAF in lockstep with the EXPLAIN plan-tree +
        // row-count walkers (zero children — no input child to recurse).
        LogicalPlan::CreateVectorIndex(_) => {
            let local_cost = Cost::new(1.0);
            let output_card = Cardinality::new(0.0);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        // #1366 (task #248, Phase 1): CREATE INDEX (property index) is a
        // leaf write-op with ZERO output rows, same cost shape as the
        // vector-index CREATE (a constant catalog register + a bounded
        // backfill; the backfill is not modeled as per-row query cost).
        LogicalPlan::CreatePropertyIndex(_) => {
            let local_cost = Cost::new(1.0);
            let output_card = Cardinality::new(0.0);
            CostedTree::leaf(CostNode::leaf(local_cost, output_card))
        }
        // ADR-148 W26-θ Phase 2: CreateRel is a write-op — walks source
        // + target endpoint sub-plans then writes (rel-write cost mirrors
        // CreateNode at the executor layer; the M4-05 cost-planner is
        // free to specialize). #832: appends the optional chain `input`
        // as the THIRD child (after source + target), mirroring the
        // executor pipeline build order + the EXPLAIN plan-tree /
        // row-count walkers — so a multi-path
        // `CREATE (a)-[:R]->(b),(c)-[:R]->(d)` keeps lockstep shape and
        // doesn't panic in `child_at(costs, 2)`.
        LogicalPlan::CreateRel(c) => {
            let source = walk(&c.source_plan, snapshot, estimator);
            let target = walk(&c.target_plan, snapshot, estimator);
            let local_cost = Cost::new(1.0);
            match &c.input {
                Some(input) => {
                    let input_subtree = walk(input, snapshot, estimator);
                    let output_card = input_subtree.output_card();
                    let child_costs = [
                        source.total_cost(),
                        target.total_cost(),
                        input_subtree.total_cost(),
                    ];
                    let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
                    CostedTree {
                        cost: cost_node,
                        children: vec![source, target, input_subtree],
                    }
                }
                None => {
                    let output_card = Cardinality::new(1.0);
                    let child_costs = [source.total_cost(), target.total_cost()];
                    let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
                    CostedTree {
                        cost: cost_node,
                        children: vec![source, target],
                    }
                }
            }
        }
        // ADR-149 W26-θ Phase 3: Delete is a 1-input write-op — its
        // cost is the input subtree's cost plus N×constant per item
        // per upstream row. Output cardinality is 0 (Delete is a
        // terminal clause; it produces no downstream rows at Phase 3
        // per ADR-149 §D-9).
        LogicalPlan::Delete(d) => {
            let input = walk(&d.input, snapshot, estimator);
            let upstream_card = input.output_card().rows();
            let item_count = d.items.len() as f64;
            let local_cost = Cost::new(upstream_card * item_count.max(1.0));
            let output_card = Cardinality::new(0.0);
            let cost_node = CostNode::unary(local_cost, output_card, input.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![input],
            }
        }
        // ADR-150 W26-θ Phase 4: Set / Remove follow the Delete shape
        // — 1-input write-op terminal at Phase 4 with output_card = 0.
        LogicalPlan::Set(s) => {
            let input = walk(&s.input, snapshot, estimator);
            let upstream_card = input.output_card().rows();
            let item_count = s.items.len() as f64;
            let local_cost = Cost::new(upstream_card * item_count.max(1.0));
            let output_card = Cardinality::new(0.0);
            let cost_node = CostNode::unary(local_cost, output_card, input.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![input],
            }
        }
        LogicalPlan::Remove(r) => {
            let input = walk(&r.input, snapshot, estimator);
            let upstream_card = input.output_card().rows();
            let item_count = r.items.len() as f64;
            let local_cost = Cost::new(upstream_card * item_count.max(1.0));
            let output_card = Cardinality::new(0.0);
            let cost_node = CostNode::unary(local_cost, output_card, input.total_cost());
            CostedTree {
                cost: cost_node,
                children: vec![input],
            }
        }
        // ADR-151 W26-θ Phase 5: Merge is a 2-input write-op (match
        // probe + create branch). Cost is the sum of both sub-trees +
        // a constant for the action firing (proportional to the number
        // of action items × the row count of whichever branch fires —
        // at v1.0-α we approximate using the match-branch's
        // cardinality as the upper bound).
        //
        // Output cardinality (ADR-151-amendment-01 §D-4): a node-shape
        // NAMED merge (`output_binding = Some`) emits the matched-or-
        // created binding row(s) — `max(match_card, 1)` (≥1 because the
        // create branch always emits exactly one when the match misses;
        // when the match hits, it emits `match_card`). Path-shape /
        // anonymous merges stay terminal at `0` (the §D-9 forward-pin,
        // unchanged). No correctness/plan-shape impact either way — the
        // pipeline is linear — accuracy only (so a parent `Project`
        // estimates ~1 input row instead of 0).
        LogicalPlan::Merge(m) => {
            let match_subtree = walk(&m.match_branch, snapshot, estimator);
            let create_subtree = walk(&m.create_branch, snapshot, estimator);
            let match_card = match_subtree.output_card().rows();
            let action_items = (m.on_create.len().max(m.on_match.len())) as f64;
            let local_cost = Cost::new(match_card.max(1.0) * action_items.max(1.0));
            let output_card = if m.output_binding.is_some() {
                Cardinality::new(match_card.max(1.0))
            } else {
                Cardinality::new(0.0)
            };
            let child_costs = [match_subtree.total_cost(), create_subtree.total_cost()];
            let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
            CostedTree {
                cost: cost_node,
                children: vec![match_subtree, create_subtree],
            }
        }
        // ADR-192 (#623) — CALL { … } correlated subquery. Cost ≈
        // outer-cardinality × body-cost (the per-driving-row re-execution;
        // OQ-192-4 rough v1.0-α estimate, refined post-LDBC). Output
        // cardinality ≈ outer-card × body-output-card (the D-7 multiply;
        // an aggregating body's 1:1 preservation is the body's own
        // output-card of 1, so the product collapses to outer-card). The
        // child order [input, body] mirrors the plan-tree / observer
        // walkers (lockstep).
        LogicalPlan::Call(c) => {
            let input_subtree = walk(&c.input, snapshot, estimator);
            let body_subtree = walk(&c.body, snapshot, estimator);
            let input_card = input_subtree.output_card().rows();
            let body_card = body_subtree.output_card().rows();
            let local_cost = Cost::new(input_card.max(1.0) * body_card.max(1.0));
            let output_card = Cardinality::new(input_card * body_card);
            let child_costs = [input_subtree.total_cost(), body_subtree.total_cost()];
            let cost_node = CostNode::n_ary(local_cost, output_card, &child_costs);
            CostedTree {
                cost: cost_node,
                children: vec![input_subtree, body_subtree],
            }
        }
        // ADR-192 (#623) — the one-row correlation seed (a leaf: one row,
        // negligible cost — like `Empty` but one row of imported cells).
        LogicalPlan::CorrelationSeed(_) => {
            let (local_cost, _) = operator::cost_empty();
            CostedTree::leaf(CostNode::leaf(local_cost, Cardinality::new(1.0)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::{BindingId, BoundExpression};
    use arcgraph_core::{LabelId, Lsn, NodeId, TypeId};
    use proptest::prelude::*;

    fn span() -> Span {
        Span::point(1, 1)
    }

    #[test]
    fn estimate_costs_on_empty_plan_returns_zero_cost() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let costed = estimate_costs(plan, &cat);
        assert_eq!(costed.total_cost().total(), 0.0);
        assert_eq!(costed.output_card().rows(), 0.0);
    }

    #[test]
    fn estimate_costs_chained_scan_filter_project_threads_cardinality() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 1_000);
        let scan = LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let filter = LogicalFilter {
            input: Box::new(LogicalPlan::Scan(scan)),
            predicate: BoundExpression::Literal {
                value: crate::ast::Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        };
        let project = LogicalProject {
            input: Box::new(LogicalPlan::Filter(filter)),
            items: Vec::new(),
            span: span(),
        };
        let plan = LogicalPlan::Project(project);
        let costed = estimate_costs(plan, &cat);

        // Scan: 1000 rows (label_card=1000)
        // Filter on boolean-true literal → selectivity 1.0 → 1000 rows
        // Project preserves cardinality → 1000 rows
        assert_eq!(costed.output_card().rows(), 1_000.0);
        // Subtree cost = scan + filter + project.
        // Scan: 1000 * 1 = 1000
        // Filter: 1000 * 0.1 = 100
        // Project: 1000 * 0.05 = 50
        // Total: 1150
        assert!((costed.total_cost().total() - 1150.0).abs() < 1e-9);
    }

    fn k3_expand(rel_type: TypeId) -> LogicalPlan {
        LogicalPlan::Expand(LogicalExpand {
            from: BindingId::new(1),
            to: BindingId::new(2),
            direction: Direction::LeftToRight,
            rel_type: Some(rel_type),
            length_range: Some(LengthRange::Cypher {
                min: 1,
                max: Some(3),
            }),
            rel_var: None,
            span: span(),
        })
    }

    #[test]
    fn adversarial_supernode_promotes_to_cost_hint_high_in_explain() {
        let label = LabelId::new(7);
        let rel_type = TypeId::new(9);
        let hub = NodeId::new(42);
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_001)
            .with_total_rel_count(10_000)
            .with_label_cardinality(label, 10_001)
            .with_rel_type_cardinality(rel_type, 10_000)
            .with_max_out_degree(label, rel_type, hub, 10_000);

        let costed = estimate_costs(k3_expand(rel_type), &cat);
        assert!(
            costed
                .diagnostics()
                .iter()
                .any(|d| d.contains("COST_HINT 'high'") && d.contains("vertex 42"))
        );
        let explain = crate::explain::PlanTree::from_costed_plan(&costed).to_string();
        assert!(explain.contains("COST_HINT 'high'"), "{explain}");
        assert!(explain.contains("vertex 42"), "{explain}");
        assert!(explain.contains("degree 10000"), "{explain}");
    }

    proptest! {
        #[test]
        /// This property pins the plan-time firewall diagnostic only.
        /// Executor-side cap wiring is deferred until var-len MATCH
        /// adopts the traversal crate.
        fn planted_supernode_k3_flagged_in_plan_diagnostics(
            node_count in 1_000_u64..20_000,
            extra_edges in 0_u64..1_000,
            hub_raw in 1_u64..100_000,
        ) {
            let label = LabelId::new(1);
            let rel_type = TypeId::new(1);
            let hub = NodeId::new(hub_raw);
            let degree = 10_000 + extra_edges;
            let rel_count = degree + node_count;
            let cat = StubCatalogProvider::new()
                .with_total_node_count(node_count)
                .with_total_rel_count(rel_count)
                .with_label_cardinality(label, node_count)
                .with_rel_type_cardinality(rel_type, rel_count)
                .with_max_out_degree(label, rel_type, hub, degree);

            let costed = estimate_costs(k3_expand(rel_type), &cat);
            prop_assert!(
                costed.diagnostics().iter().any(|d| {
                    d.contains("COST_HINT 'high'")
                        && d.contains("per_hop_frontier_cap")
                        && d.contains(&format!("vertex {}", hub.raw()))
                }),
                "missing supernode degradation diagnostic: {:?}",
                costed.diagnostics()
            );
        }
    }
}
