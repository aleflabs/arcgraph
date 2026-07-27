//! Leaf extraction + left-deep tree construction for the M4-52 DP
//! enumerator.
//!
//! # Leaf extraction
//!
//! Given a [`LogicalPlan::Join`] sub-tree, `extract_inner_join_leaves`
//! flattens contiguous inner-join nodes into a list of "leaf"
//! relations. A "leaf" is any [`LogicalPlan`] variant other than
//! [`LogicalPlan::Join`] — note that [`LogicalPlan::LeftOuterJoin`]
//! IS a leaf for enumeration purposes (the DP does not cross
//! outer-join boundaries; outer-join ordering is preserved per Cypher
//! 9 §6.5).
//!
//! The leaves preserve their original order from the input plan's
//! pre-order traversal. This is the canonical "input order" used as
//! a tie-break by `super::dp::enumerate`.
//!
//! # Left-deep construction
//!
//! `build_left_deep` produces the canonical left-deep
//! [`LogicalPlan::Join`] tree from a permutation of leaves. Each
//! join's [`crate::logical_plan::JoinCondition::SharedBindings`] is
//! re-derived from the binding-set overlap of the left sub-tree and
//! the right leaf — this matches what M4-31 lowering would produce
//! if it had emitted the chosen ordering directly.
//!
//! Both helpers are `pub(super)`-scoped — DP module use only; not
//! part of the public API.

use crate::error::Span;
use crate::logical_plan::{JoinAlgorithm, JoinCondition, LogicalEmpty, LogicalJoin, LogicalPlan};

use super::{bindings_in, join_condition_for};

/// Flatten a [`LogicalPlan::Join`] sub-tree into its constituent
/// leaves.
///
/// "Inner-join" here means [`LogicalPlan::Join`] specifically — NOT
/// [`LogicalPlan::LeftOuterJoin`], which is opaque (its ordering is
/// preserved).
///
/// # Returns
///
/// Vec of leaves in the order they were encountered during a left-
/// first depth-first traversal — preserving the input plan's
/// "natural" leaf order. Tie-breaking the DP back to this order
/// produces deterministic results when multiple orderings have
/// identical cost.
pub(super) fn extract_inner_join_leaves(plan: LogicalPlan) -> Vec<LogicalPlan> {
    let mut out = Vec::new();
    collect(plan, &mut out);
    out
}

fn collect(plan: LogicalPlan, out: &mut Vec<LogicalPlan>) {
    match plan {
        LogicalPlan::Join(j) => {
            collect(*j.left, out);
            collect(*j.right, out);
        }
        leaf => out.push(leaf),
    }
}

/// Build a left-deep [`LogicalPlan::Join`] tree from a permutation
/// of leaves.
///
/// `leaves[0]` becomes the leftmost leaf; subsequent leaves are
/// joined onto the accumulated left sub-tree one at a time. Each
/// join's [`JoinCondition`] is the sorted intersection of the left
/// sub-tree's bindings with the right leaf's bindings (per M4-31
/// lowering convention).
///
/// # Cartesian handling
///
/// If a leaf shares no bindings with the accumulated left sub-tree,
/// the resulting join is a Cartesian product
/// (`SharedBindings(empty)`). This is admissible per [`JoinCondition`]
/// rustdoc — both Cartesian and equi-join shapes are encoded by the
/// same variant. The DP rejects this candidate during enumeration
/// (Cartesian is high-cost); it is built here defensively for the
/// degenerate-input fallback path.
///
/// # Span
///
/// The new join's span is the supplied `span` argument (typically
/// the span of the original Join node being replaced).
///
/// # Panics
///
/// Panics if `leaves` is empty. Caller MUST guarantee `leaves.len() ≥ 1`.
pub(super) fn build_left_deep(leaves: Vec<LogicalPlan>, span: &Span) -> LogicalPlan {
    let mut iter = leaves.into_iter();
    let mut acc = iter
        .next()
        .expect("build_left_deep called with empty leaves; caller bug");

    for leaf in iter {
        let left_bindings = bindings_in(&acc);
        let right_bindings = bindings_in(&leaf);
        let shared = join_condition_for(&left_bindings, &right_bindings);
        acc = LogicalPlan::Join(LogicalJoin {
            left: Box::new(acc),
            right: Box::new(leaf),
            on: JoinCondition::SharedBindings(shared),
            algorithm: JoinAlgorithm::Auto,
            span: span.clone(),
        });
    }

    acc
}

/// Construct an empty plan placeholder. Used by the rewriter's
/// degenerate-input fallback.
pub(super) fn empty_plan(span: Span) -> LogicalPlan {
    LogicalPlan::Empty(LogicalEmpty { span })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::semantic::bound_ast::BindingId;
    use arcgraph_core::{LabelId, Lsn};

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

    fn ij(left: LogicalPlan, right: LogicalPlan, on: Vec<BindingId>) -> LogicalPlan {
        LogicalPlan::Join(LogicalJoin {
            left: Box::new(left),
            right: Box::new(right),
            on: JoinCondition::SharedBindings(on),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        })
    }

    /// Single-leaf input → single leaf output (no Join).
    #[test]
    fn extract_single_scan_yields_single_leaf() {
        let plan = scan(1, 0);
        let leaves = extract_inner_join_leaves(plan.clone());
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0], plan);
    }

    /// 2-way join: two leaves, in left-then-right order.
    #[test]
    fn extract_two_way_join_yields_two_leaves() {
        let plan = ij(scan(1, 0), scan(2, 1), vec![]);
        let leaves = extract_inner_join_leaves(plan);
        assert_eq!(leaves.len(), 2);
        match &leaves[0] {
            LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(1))),
            _ => panic!("first leaf should be scan(1)"),
        }
        match &leaves[1] {
            LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(2))),
            _ => panic!("second leaf should be scan(2)"),
        }
    }

    /// 4-way left-deep input → 4 leaves in left-to-right order.
    #[test]
    fn extract_four_way_left_deep_yields_four_leaves_in_order() {
        // ((a JOIN b) JOIN c) JOIN d
        let plan = ij(
            ij(ij(scan(1, 0), scan(2, 1), vec![]), scan(3, 2), vec![]),
            scan(4, 3),
            vec![],
        );
        let leaves = extract_inner_join_leaves(plan);
        assert_eq!(leaves.len(), 4);
        let labels: Vec<_> = leaves
            .iter()
            .map(|leaf| match leaf {
                LogicalPlan::Scan(s) => s.label.unwrap().raw(),
                _ => panic!("expected Scan"),
            })
            .collect();
        assert_eq!(labels, vec![1_u32, 2, 3, 4]);
    }

    /// Bushy input: (a JOIN b) JOIN (c JOIN d) → 4 leaves still
    /// flatten left-to-right (DFS pre-order).
    #[test]
    fn extract_bushy_yields_dfs_preorder_leaves() {
        let plan = ij(
            ij(scan(1, 0), scan(2, 1), vec![]),
            ij(scan(3, 2), scan(4, 3), vec![]),
            vec![],
        );
        let leaves = extract_inner_join_leaves(plan);
        let labels: Vec<_> = leaves
            .iter()
            .map(|leaf| match leaf {
                LogicalPlan::Scan(s) => s.label.unwrap().raw(),
                _ => panic!("expected Scan"),
            })
            .collect();
        assert_eq!(labels, vec![1_u32, 2, 3, 4]);
    }

    /// Outer-join sub-tree is treated as an opaque LEAF (not
    /// recursed).
    #[test]
    fn extract_treats_outer_join_as_opaque_leaf() {
        let outer = LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
            left: Box::new(scan(1, 0)),
            right: Box::new(scan(2, 1)),
            on: JoinCondition::SharedBindings(vec![]),
            span: span(),
        });
        // Inner Join wraps the outer.
        let plan = ij(outer.clone(), scan(3, 2), vec![]);
        let leaves = extract_inner_join_leaves(plan);
        assert_eq!(leaves.len(), 2);
        // First leaf is the outer-join (preserved opaque).
        match &leaves[0] {
            LogicalPlan::LeftOuterJoin(_) => {}
            _ => panic!("outer join should be preserved as a leaf"),
        }
    }

    /// build_left_deep on 2 leaves with shared binding → equi-join.
    #[test]
    fn build_left_deep_two_leaves_with_shared_binding() {
        let leaves = vec![scan(1, 0), scan(2, 0)]; // both bind id 0
        let plan = build_left_deep(leaves, &span());
        match plan {
            LogicalPlan::Join(j) => match j.on {
                JoinCondition::SharedBindings(ids) => {
                    assert_eq!(ids, vec![BindingId::new(0)]);
                }
            },
            _ => panic!("expected Join at root"),
        }
    }

    /// build_left_deep on 3 leaves all chained → produces left-deep
    /// tree with appropriate per-join SharedBindings.
    #[test]
    fn build_left_deep_three_leaves_chain() {
        // a—[k]—b, b—[k]—c, c standalone
        let leaves = vec![
            scan(1, 0), // {0}
            scan(2, 0), // {0} ← shares with leaf0
            scan(3, 0), // {0} ← shares with accumulated
        ];
        let plan = build_left_deep(leaves, &span());
        // Tree shape: Join(Join(scan1, scan2), scan3)
        match plan {
            LogicalPlan::Join(outer) => {
                match outer.on {
                    JoinCondition::SharedBindings(ids) => {
                        assert_eq!(ids, vec![BindingId::new(0)]);
                    }
                }
                match *outer.left {
                    LogicalPlan::Join(inner) => match inner.on {
                        JoinCondition::SharedBindings(ids) => {
                            assert_eq!(ids, vec![BindingId::new(0)]);
                        }
                    },
                    _ => panic!("inner left should be Join"),
                }
            }
            _ => panic!("expected Join at root"),
        }
    }

    /// build_left_deep on a single leaf → the leaf, unchanged.
    #[test]
    fn build_left_deep_single_leaf_returns_leaf() {
        let plan = build_left_deep(vec![scan(1, 0)], &span());
        match plan {
            LogicalPlan::Scan(s) => assert_eq!(s.var, BindingId::new(0)),
            _ => panic!("single-leaf input should pass through"),
        }
    }

    /// build_left_deep on disjoint leaves → Cartesian (empty
    /// SharedBindings).
    #[test]
    fn build_left_deep_disjoint_leaves_cartesian() {
        let leaves = vec![scan(1, 0), scan(2, 1)]; // disjoint bindings
        let plan = build_left_deep(leaves, &span());
        match plan {
            LogicalPlan::Join(j) => match j.on {
                JoinCondition::SharedBindings(ids) => assert!(ids.is_empty()),
            },
            _ => panic!("expected Join at root"),
        }
    }

    /// build_left_deep panics on empty input (caller bug).
    #[test]
    #[should_panic(expected = "empty leaves")]
    fn build_left_deep_panics_on_empty() {
        let _ = build_left_deep(Vec::new(), &span());
    }

    /// empty_plan returns a `LogicalPlan::Empty` with the given span.
    #[test]
    fn empty_plan_carries_span() {
        let s = Span::point(7, 9);
        let plan = empty_plan(s.clone());
        assert_eq!(plan.span(), &s);
    }
}
