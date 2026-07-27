//! Cost-based join-algorithm picker (W25-M4-61b / ADR-097).
//!
//! Walks a [`LogicalPlan`] tree and resolves every
//! `LogicalJoin::algorithm` field set to `JoinAlgorithm::Auto`
//! into a concrete [`JoinAlgorithm::HashJoin`] or
//! [`JoinAlgorithm::MergeJoin`] based on per-side cardinality
//! estimates and the [`crate::planner::cost::operator::cost_hash_join`]
//! vs [`crate::planner::cost::operator::cost_merge_join`] comparison.
//!
//! # Why a separate pass (not inlined into the cost walker)
//!
//! The M4-51 [`crate::planner::cost::estimate_costs`] walker consumes
//! the plan read-only (per the
//! `crate::planner::cost::walker` rustdoc invariant). Mutating the
//! `algorithm` field during costing would violate that contract.
//! Instead, ADR-097 ships a dedicated `pick_join_algorithms` pass that
//! is called BEFORE the executor consumes the plan — the
//! [`crate::execute`] / [`crate::QueryEngine::execute`] entry points
//! call this pass in their plan-build sequence.
//!
//! The pass is a one-shot rewrite walker; running it twice on the
//! same plan is a no-op (the second pass sees only concrete
//! algorithms, no `Auto` left to resolve).
//!
//! # Cardinality threading
//!
//! Per-side cardinalities flow from the [`CatalogSnapshot`] via the
//! same per-operator cost functions the M4-51 walker uses
//! ([`crate::planner::cost::operator::cost_scan`] /
//! [`cost_expand`](crate::planner::cost::operator::cost_expand)
//! / `cost_filter` / etc.). This guarantees the picker's decision
//! is consistent with the cost-walker's annotations for the SAME
//! plan + SAME snapshot.
//!
//! # Cartesian rule
//!
//! `SharedBindings([])` ALWAYS resolves to [`JoinAlgorithm::HashJoin`]
//! regardless of cost — merge-join is structurally undefined without
//! join keys. The cost function returns `f64::MAX` for the merge cost
//! in this case so `min(hash, merge) == hash`, but the picker enforces
//! the rule explicitly for clarity + defense-in-depth.
//!
//! # ADR provenance
//! - **ADR-097** — W25-M4-61b executor JOIN cost-based picker.
//! - **ADR-038 §2 D-24** — `LogicalJoin` lowering surface.
//! - **ADR-038 amendment-02 §M4.e** — M4-05 cost-model + plan
//!   enumeration decomposition; this pass is M4-05c-adjacent (the
//!   "physical-algorithm-resolution" sub-step, parallel to M4-52's
//!   join-ordering DP).

use crate::logical_plan::{JoinAlgorithm, JoinCondition, LogicalPlan};
use crate::planner::cost::operator::{cost_hash_join, cost_merge_join};
use crate::planner::cost::{Cardinality, capture_snapshot};
use crate::semantic::{CatalogProvider, CatalogSnapshot, SelectivityEstimator};

/// Walk a [`LogicalPlan`] and rewrite every
/// [`crate::logical_plan::LogicalJoin`] whose algorithm is `Auto` to
/// the cost-optimal concrete variant.
///
/// Joins whose `algorithm` field is already concrete
/// ([`JoinAlgorithm::HashJoin`] / [`JoinAlgorithm::MergeJoin`]) are
/// left untouched — tests + EXPLAIN consumers that pin a specific
/// algorithm get a no-op pass.
///
/// # Determinism
///
/// Same plan + same catalog snapshot → same algorithm choices. The
/// pass calls [`CatalogProvider::snapshot`] EXACTLY ONCE at entry
/// (matching the M4-51 walker's single-snapshot discipline per
/// ADR-038 §2 D-25); every per-join cost comparison reads from the
/// captured snapshot.
#[must_use]
pub fn pick_join_algorithms(plan: LogicalPlan, catalog: &dyn CatalogProvider) -> LogicalPlan {
    let snapshot = capture_snapshot(catalog);
    let estimator = SelectivityEstimator::new(catalog);
    walk_and_pick(plan, &snapshot, &estimator)
}

fn walk_and_pick<C: CatalogProvider + ?Sized>(
    plan: LogicalPlan,
    snapshot: &CatalogSnapshot,
    estimator: &SelectivityEstimator<'_, C>,
) -> LogicalPlan {
    match plan {
        // -------------------------------------------------------------
        // Joins — the rewrite payload.
        // -------------------------------------------------------------
        LogicalPlan::Join(mut j) => {
            // Recurse into children first (post-order rewrite — children
            // need to be resolved before we estimate their cardinality
            // because nested joins may pick their algorithm based on
            // their own subtree's output cardinality).
            j.left = Box::new(walk_and_pick(*j.left, snapshot, estimator));
            j.right = Box::new(walk_and_pick(*j.right, snapshot, estimator));
            // Resolve Auto only when still Auto post-recursion.
            if matches!(j.algorithm, JoinAlgorithm::Auto) {
                let JoinCondition::SharedBindings(ref keys) = j.on;
                if keys.is_empty() {
                    // Cartesian routes to HashJoin (defense-in-depth on
                    // top of cost_merge_join returning f64::MAX).
                    j.algorithm = JoinAlgorithm::HashJoin;
                } else {
                    let l_card = estimate_subtree_card(&j.left, snapshot, estimator);
                    let r_card = estimate_subtree_card(&j.right, snapshot, estimator);
                    let h = cost_hash_join(l_card.rows(), r_card.rows());
                    let m = cost_merge_join(l_card.rows(), r_card.rows(), &j.on);
                    // Tie-breaking: hash wins on equal costs (preserves
                    // the W17α default behavior under hostile-tie inputs).
                    j.algorithm = if m < h {
                        JoinAlgorithm::MergeJoin
                    } else {
                        JoinAlgorithm::HashJoin
                    };
                }
            }
            LogicalPlan::Join(j)
        }
        // -------------------------------------------------------------
        // Pass-through unary + binary parents — recurse into children.
        // -------------------------------------------------------------
        LogicalPlan::LeftOuterJoin(mut j) => {
            j.left = Box::new(walk_and_pick(*j.left, snapshot, estimator));
            j.right = Box::new(walk_and_pick(*j.right, snapshot, estimator));
            LogicalPlan::LeftOuterJoin(j)
        }
        LogicalPlan::Filter(mut f) => {
            f.input = Box::new(walk_and_pick(*f.input, snapshot, estimator));
            LogicalPlan::Filter(f)
        }
        LogicalPlan::Project(mut p) => {
            p.input = Box::new(walk_and_pick(*p.input, snapshot, estimator));
            LogicalPlan::Project(p)
        }
        LogicalPlan::Limit(mut l) => {
            l.input = Box::new(walk_and_pick(*l.input, snapshot, estimator));
            LogicalPlan::Limit(l)
        }
        LogicalPlan::Skip(mut s) => {
            s.input = Box::new(walk_and_pick(*s.input, snapshot, estimator));
            LogicalPlan::Skip(s)
        }
        LogicalPlan::DynamicLimit(mut d) => {
            d.input = Box::new(walk_and_pick(*d.input, snapshot, estimator));
            LogicalPlan::DynamicLimit(d)
        }
        LogicalPlan::Aggregate(mut a) => {
            a.input = Box::new(walk_and_pick(*a.input, snapshot, estimator));
            LogicalPlan::Aggregate(a)
        }
        LogicalPlan::Sort(mut s) => {
            s.input = Box::new(walk_and_pick(*s.input, snapshot, estimator));
            LogicalPlan::Sort(s)
        }
        LogicalPlan::Distinct(mut d) => {
            d.input = Box::new(walk_and_pick(*d.input, snapshot, estimator));
            LogicalPlan::Distinct(d)
        }
        LogicalPlan::Unwind(mut u) => {
            u.input = Box::new(walk_and_pick(*u.input, snapshot, estimator));
            LogicalPlan::Unwind(u)
        }
        // ADR-197 (#802): no joins inside a procedure call; just walk
        // the (unit-row) input for uniformity.
        LogicalPlan::ProcedureCall(mut p) => {
            p.input = Box::new(walk_and_pick(*p.input, snapshot, estimator));
            LogicalPlan::ProcedureCall(p)
        }
        LogicalPlan::NamedPath(mut n) => {
            n.input = Box::new(walk_and_pick(*n.input, snapshot, estimator));
            LogicalPlan::NamedPath(n)
        }
        LogicalPlan::CommunityLookup(mut c) => {
            c.input = Box::new(walk_and_pick(*c.input, snapshot, estimator));
            LogicalPlan::CommunityLookup(c)
        }
        LogicalPlan::Fusion(mut f) => {
            f.inputs = f
                .inputs
                .into_iter()
                .map(|input| Box::new(walk_and_pick(*input, snapshot, estimator)))
                .collect();
            LogicalPlan::Fusion(f)
        }
        // ADR-185 (#649-A1, W28) — UNION ALL: recurse into each arm so
        // joins INSIDE a union arm still get their algorithm picked
        // (each arm is an independent sub-plan).
        LogicalPlan::Union(mut u) => {
            u.arms = u
                .arms
                .into_iter()
                .map(|arm| walk_and_pick(arm, snapshot, estimator))
                .collect();
            LogicalPlan::Union(u)
        }
        // -------------------------------------------------------------
        // CreateRel (ADR-148 W26-θ Phase 2) — recurse into source +
        // target sub-plans (each typically a CreateNode at Phase 2 —
        // MATCH-bound resolution forward-pinned to Phase 5).
        // -------------------------------------------------------------
        LogicalPlan::CreateRel(mut c) => {
            c.source_plan = Box::new(walk_and_pick(*c.source_plan, snapshot, estimator));
            c.target_plan = Box::new(walk_and_pick(*c.target_plan, snapshot, estimator));
            LogicalPlan::CreateRel(c)
        }
        // -------------------------------------------------------------
        // Delete (ADR-149 W26-θ Phase 3) — recurse into the input
        // sub-plan (typically the MATCH-produced upstream plan).
        // -------------------------------------------------------------
        LogicalPlan::Delete(mut d) => {
            d.input = Box::new(walk_and_pick(*d.input, snapshot, estimator));
            LogicalPlan::Delete(d)
        }
        // -------------------------------------------------------------
        // Set / Remove (ADR-150 W26-θ Phase 4) — recurse into the
        // input sub-plan (typically the MATCH-produced upstream plan).
        // -------------------------------------------------------------
        LogicalPlan::Set(mut s) => {
            s.input = Box::new(walk_and_pick(*s.input, snapshot, estimator));
            LogicalPlan::Set(s)
        }
        LogicalPlan::Remove(mut r) => {
            r.input = Box::new(walk_and_pick(*r.input, snapshot, estimator));
            LogicalPlan::Remove(r)
        }
        // -------------------------------------------------------------
        // Merge (ADR-151 W26-θ Phase 5) — recurse into BOTH the match
        // and create sub-plans. Per-branch picker decisions (e.g.,
        // join algorithm on path-shape match-branch) propagate into
        // the picked plan.
        // -------------------------------------------------------------
        LogicalPlan::Merge(mut m) => {
            m.match_branch = Box::new(walk_and_pick(*m.match_branch, snapshot, estimator));
            m.create_branch = Box::new(walk_and_pick(*m.create_branch, snapshot, estimator));
            LogicalPlan::Merge(m)
        }
        // -------------------------------------------------------------
        // CALL { … } (ADR-192 #623) — recurse into BOTH the driving
        // `input` and the subquery `body` so joins inside either get
        // their algorithm picked (the body is an independent sub-plan
        // re-executed per driving row).
        // -------------------------------------------------------------
        LogicalPlan::Call(mut c) => {
            c.input = Box::new(walk_and_pick(*c.input, snapshot, estimator));
            c.body = Box::new(walk_and_pick(*c.body, snapshot, estimator));
            LogicalPlan::Call(c)
        }
        // -------------------------------------------------------------
        // Leaves — no recursion needed.
        // -------------------------------------------------------------
        p @ (LogicalPlan::Scan(_)
        // #1366 (Phase 2): the indexed point-lookup is a read leaf.
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        // ADR-147 W26-θ Phase 1: CreateNode is a leaf at Phase 1.
        | LogicalPlan::CreateNode(_)
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf write-op.
        | LogicalPlan::CreateVectorIndex(_)
        // #1366: CREATE INDEX (property index) is a leaf write-op.
        | LogicalPlan::CreatePropertyIndex(_)
        // ADR-192 (#623): the correlation seed is a leaf.
        | LogicalPlan::CorrelationSeed(_)
        | LogicalPlan::Empty(_)) => p,
    }
}

/// Estimate the output cardinality of an arbitrary [`LogicalPlan`]
/// subtree.
///
/// Delegates to the M4-51 [`crate::planner::cost::estimate_costs`]
/// machinery — we clone the subtree, run a cost walk, and read off
/// the root output_card. This is more expensive than threading the
/// cardinality through a custom mini-walker, but it guarantees the
/// picker's decision matches the cost walker's annotations for the
/// SAME plan + SAME snapshot (i.e., no drift between picker and
/// observer at M4-71).
fn estimate_subtree_card<C: CatalogProvider + ?Sized>(
    plan: &LogicalPlan,
    snapshot: &CatalogSnapshot,
    estimator: &SelectivityEstimator<'_, C>,
) -> Cardinality {
    // Delegate to the internal cost walker. We MUST clone the plan
    // here — the public M4-51 walker consumes the plan by-value;
    // re-cloning lets the picker call it without taking ownership.
    crate::planner::cost::walker_card_for(plan.clone(), snapshot, estimator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::{
        Direction, JoinCondition, LogicalEmpty, LogicalExpand, LogicalJoin, LogicalScan,
    };
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::BindingId;
    use arcgraph_core::{LabelId, Lsn, TypeId};

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn scan(label_raw: u32, var_raw: u64) -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(label_raw)),
            var: BindingId::new(var_raw),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    }

    fn join(left: LogicalPlan, right: LogicalPlan, on: Vec<BindingId>) -> LogicalPlan {
        LogicalPlan::Join(LogicalJoin {
            left: Box::new(left),
            right: Box::new(right),
            on: JoinCondition::SharedBindings(on),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        })
    }

    #[test]
    fn picker_resolves_auto_for_equi_join() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 100);
        let plan = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let resolved = pick_join_algorithms(plan, &cat);
        match resolved {
            LogicalPlan::Join(j) => assert!(
                matches!(
                    j.algorithm,
                    JoinAlgorithm::HashJoin | JoinAlgorithm::MergeJoin
                ),
                "Auto must be resolved to concrete algorithm; got {:?}",
                j.algorithm
            ),
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn picker_pins_cartesian_to_hash() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 100);
        let plan = join(scan(1, 0), scan(2, 0), Vec::new()); // Cartesian
        let resolved = pick_join_algorithms(plan, &cat);
        match resolved {
            LogicalPlan::Join(j) => {
                assert_eq!(j.algorithm, JoinAlgorithm::HashJoin);
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn picker_preserves_concrete_algorithm_choices() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let plan = LogicalPlan::Join(LogicalJoin {
            left: Box::new(scan(1, 0)),
            right: Box::new(scan(2, 0)),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::MergeJoin, // explicit pin
            span: span(),
        });
        let resolved = pick_join_algorithms(plan, &cat);
        match resolved {
            LogicalPlan::Join(j) => assert_eq!(j.algorithm, JoinAlgorithm::MergeJoin),
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn picker_is_idempotent() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 50);
        let plan = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let once = pick_join_algorithms(plan, &cat);
        let twice = pick_join_algorithms(once.clone(), &cat);
        assert_eq!(once, twice, "picker must be idempotent");
    }

    #[test]
    fn picker_recurses_into_nested_joins() {
        // (A ⋈ B) ⋈ C — outer join Auto, inner join Auto. Both must
        // resolve.
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 100)
            .with_label_cardinality(LabelId::new(3), 100);
        let inner = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let outer = join(inner, scan(3, 0), vec![BindingId::new(0)]);
        let resolved = pick_join_algorithms(outer, &cat);
        match resolved {
            LogicalPlan::Join(outer_j) => {
                assert!(!matches!(outer_j.algorithm, JoinAlgorithm::Auto));
                match outer_j.left.as_ref() {
                    LogicalPlan::Join(inner_j) => {
                        assert!(!matches!(inner_j.algorithm, JoinAlgorithm::Auto));
                    }
                    other => panic!("inner join must remain a join, got {other:?}"),
                }
            }
            other => panic!("outer join must remain a join, got {other:?}"),
        }
    }

    #[test]
    fn picker_descends_through_expand_join_chain() {
        // A common LDBC-style shape: Join(Scan, Expand(Scan)) where
        // the picker should not get confused by Expand inside the
        // right side.
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_total_rel_count(5_000)
            .with_label_cardinality(LabelId::new(1), 100);
        let scan_a = scan(1, 0);
        let expand = LogicalPlan::Expand(LogicalExpand {
            from: BindingId::new(0),
            to: BindingId::new(1),
            direction: Direction::LeftToRight,
            rel_type: Some(TypeId::new(1)),
            length_range: None,
            rel_var: None,
            span: span(),
        });
        let plan = join(scan_a, expand, vec![BindingId::new(0)]);
        let resolved = pick_join_algorithms(plan, &cat);
        match resolved {
            LogicalPlan::Join(j) => assert!(!matches!(j.algorithm, JoinAlgorithm::Auto)),
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn picker_leaves_no_op_subtrees_untouched() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        // No joins in the plan; the picker traverses but mutates nothing.
        let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let resolved = pick_join_algorithms(plan.clone(), &cat);
        assert_eq!(resolved, plan);
    }
}
