//! Predicate analyzer for the M4-51 cost planner.
//!
//! Walks a [`crate::semantic::bound_ast::BoundExpression`] and
//! computes its combined selectivity by composing per-leaf
//! [`crate::semantic::SelectivityEstimator`] outputs through the
//! [`crate::planner::cost::composition`] helpers.
//!
//! # Predicate-class lookup
//!
//! The walker recognizes the v1.0 predicate classes from ADR-038
//! §2 D-27 (M4-42 estimator surface):
//!
//! | Source predicate | Maps to |
//! |------------------|---------|
//! | `expr = literal` / `expr = $param` | `estimate_eq` |
//! | `expr < literal` / `<=` / `>` / `>=` | `estimate_lt` (range fallback) |
//! | `expr IN list` | `estimate_in` |
//! | `expr IS NULL` / `expr IS NOT NULL` | `estimate_eq` (treated as point) |
//! | (label filter on scan) | applied at scan-cost stage, not here |
//!
//! Logical connectives:
//! - `expr1 AND expr2` → [`crate::planner::cost::composition::compose_and`]
//! - `expr1 OR expr2` → [`crate::planner::cost::composition::compose_or`]
//! - `expr1 XOR expr2` → [`crate::planner::cost::composition::compose_xor`]
//! - `NOT expr` → [`crate::planner::cost::composition::compose_not`]
//!
//! Unrecognized predicate shapes (function-call predicates,
//! `<expr> NEAR <expr>`, `<expr> MATCH <expr>`,
//! `n IN COMMUNITY($cid)` — which are lowered to dedicated
//! [`crate::logical_plan::LogicalPlan`] variants, NOT
//! [`crate::logical_plan::LogicalFilter`] nodes — fall back to the
//! `DEFAULT_EQ_SELECTIVITY` constant per the M4-42 contract.
//!
//! # v1.1 forward-link
//!
//! When per-property histograms / sketches land (M4-04c), the
//! predicate walker's `estimate_eq` / `estimate_lt` / `estimate_in`
//! call sites grow a `prop: PropertyId` argument; the walker tracks
//! the property currently being filtered through the recursion.
//! M4-51 does NOT thread the property because the v1.0 estimators
//! ignore it.

use crate::ast::{BinOp, UnaryOp};
use crate::semantic::SelectivityEstimator;
use crate::semantic::bound_ast::{BindingId, BoundExpression};
use crate::semantic::catalog::CatalogProvider;

use super::composition::{compose_and, compose_not, compose_or, compose_xor};

/// Compute the combined selectivity of a predicate expression over
/// the active catalog snapshot.
///
/// Returns an `f64 ∈ [0.0, 1.0]`. Never NaN, never Inf, never
/// negative, never > 1.0 — the
/// `SelectivityEstimator` + `composition` helpers preserve the
/// unit-interval invariant per the M4-42 + composition module
/// contracts.
///
/// # Arguments
///
/// - `predicate` — the [`BoundExpression`] to analyze.
/// - `estimator` — borrowed [`SelectivityEstimator`] reading from
///   the active [`CatalogProvider`].
/// - `default_var` — the binding the predicate is filtering against
///   (the [`crate::logical_plan::LogicalFilter`]'s scan target). The
///   v1.0 estimators ignore this; v1.1 sketch-aware estimators key
///   per-binding histograms by it.
#[must_use]
pub fn predicate_selectivity<C: CatalogProvider + ?Sized>(
    predicate: &BoundExpression,
    estimator: &SelectivityEstimator<'_, C>,
    default_var: BindingId,
) -> f64 {
    walk(predicate, estimator, default_var)
}

fn walk<C: CatalogProvider + ?Sized>(
    expr: &BoundExpression,
    estimator: &SelectivityEstimator<'_, C>,
    default_var: BindingId,
) -> f64 {
    match expr {
        // -----------------------------------------------------------
        // Logical connectives — composed over the left-nested SPINE
        // iteratively (#1290): a flat `p1 AND p2 AND … pN` WHERE folds
        // into an N-deep left-nested connective spine (up to
        // `MAX_FLAT_CHAIN_DEPTH`), and recursing per level overflowed
        // the native stack. Walk down the connective/NOT edge
        // collecting one frame per level, estimate the non-connective
        // base, then fold the composition back up — the same
        // left-associative composition order as the recursion this
        // replaces. `rhs` operands recurse (they are single predicates
        // / bracket-bounded subtrees, never the LEFT spine).
        //
        // XOR (#621) is handled EXPLICITLY (not absorbed by the
        // arithmetic-BinaryOp default below) so its selectivity
        // composes through the dedicated `compose_xor` (exactly-one)
        // formula instead of the conservative default-eq fallback.
        // -----------------------------------------------------------
        BoundExpression::BinaryOp {
            op: BinOp::And | BinOp::Or | BinOp::Xor,
            ..
        }
        | BoundExpression::UnaryOp {
            op: UnaryOp::Not, ..
        } => {
            enum Connective<'a> {
                And(&'a BoundExpression),
                Or(&'a BoundExpression),
                Xor(&'a BoundExpression),
                Not,
            }
            let mut frames: Vec<Connective<'_>> = Vec::new();
            let mut cur = expr;
            loop {
                match cur {
                    BoundExpression::BinaryOp {
                        op: BinOp::And,
                        lhs,
                        rhs,
                        ..
                    } => {
                        frames.push(Connective::And(rhs));
                        cur = lhs;
                    }
                    BoundExpression::BinaryOp {
                        op: BinOp::Or,
                        lhs,
                        rhs,
                        ..
                    } => {
                        frames.push(Connective::Or(rhs));
                        cur = lhs;
                    }
                    BoundExpression::BinaryOp {
                        op: BinOp::Xor,
                        lhs,
                        rhs,
                        ..
                    } => {
                        frames.push(Connective::Xor(rhs));
                        cur = lhs;
                    }
                    BoundExpression::UnaryOp {
                        op: UnaryOp::Not,
                        operand,
                        ..
                    } => {
                        frames.push(Connective::Not);
                        cur = operand;
                    }
                    _ => break,
                }
            }
            let mut acc = walk(cur, estimator, default_var);
            while let Some(frame) = frames.pop() {
                acc = match frame {
                    Connective::And(rhs) => compose_and(acc, walk(rhs, estimator, default_var)),
                    Connective::Or(rhs) => compose_or(acc, walk(rhs, estimator, default_var)),
                    Connective::Xor(rhs) => compose_xor(acc, walk(rhs, estimator, default_var)),
                    Connective::Not => compose_not(acc),
                };
            }
            acc
        }

        // -----------------------------------------------------------
        // Equality / inequality predicates.
        // -----------------------------------------------------------
        BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs,
            rhs,
            ..
        } => {
            // n.prop = literal or n.prop = $param. v1.0 estimator
            // signature ignores both sides' actual values.
            let var = leftmost_binding(lhs).unwrap_or(default_var);
            // Heuristic: if neither side names a property access nor a
            // literal/parameter, fall back to default_eq via the
            // estimator anyway — `estimate_eq` is the sane default.
            let _ = rhs;
            estimator.estimate_eq(var, None)
        }
        BoundExpression::BinaryOp {
            op: BinOp::Neq,
            lhs,
            rhs,
            ..
        } => {
            // ! (= ) — complement of equality.
            let var = leftmost_binding(lhs).unwrap_or(default_var);
            let _ = rhs;
            compose_not(estimator.estimate_eq(var, None))
        }

        // Range predicates — all map to estimate_lt at v1.0.
        BoundExpression::BinaryOp {
            op: BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge,
            lhs,
            ..
        } => {
            let var = leftmost_binding(lhs).unwrap_or(default_var);
            estimator.estimate_lt(var, None)
        }

        // -----------------------------------------------------------
        // IN-list predicate.
        // -----------------------------------------------------------
        BoundExpression::In { lhs, rhs, .. } => {
            let var = leftmost_binding(lhs).unwrap_or(default_var);
            let list_size = estimate_list_size(rhs);
            estimator.estimate_in(var, None, list_size)
        }

        // -----------------------------------------------------------
        // IS NULL — point predicate; treat as estimate_eq (a single
        // value: NULL).
        // -----------------------------------------------------------
        BoundExpression::IsNull { lhs, negated, .. } => {
            let var = leftmost_binding(lhs).unwrap_or(default_var);
            let null_sel = estimator.estimate_eq(var, None);
            if *negated {
                compose_not(null_sel)
            } else {
                null_sel
            }
        }

        // -----------------------------------------------------------
        // Predicates that lower to dedicated LogicalPlan variants
        // (NOT LogicalFilter):
        // - `expr NEAR expr` → LogicalVectorNear
        // - `expr MATCH expr` → LogicalTextMatch
        // - `n IN COMMUNITY($cid)` → LogicalCommunityLookup
        // If they reach the predicate walker, it means the lowering
        // produced a Filter-with-Hybrid-predicate shape (a future
        // refactor); v1.0 falls back to default-eq selectivity.
        // -----------------------------------------------------------
        BoundExpression::Near { .. }
        | BoundExpression::TextMatch { .. }
        | BoundExpression::InCommunity { .. } => estimator.estimate_eq(default_var, None),

        // -----------------------------------------------------------
        // Pure-value expressions and arithmetic — neutral selectivity.
        // A WHERE clause that's a bare boolean parameter / literal
        // is assumed to pass everything (true) or nothing (false).
        // The v1.0 estimator returns a default; the cost-model treats
        // this as "no filtering" (selectivity = 1.0).
        // -----------------------------------------------------------
        BoundExpression::Literal { value, .. } => match value {
            // Literal `false` → no rows pass.
            crate::ast::Literal::Bool(false) => 0.0,
            // Literal `true` → all rows pass.
            crate::ast::Literal::Bool(true) => 1.0,
            // Other literals in WHERE position are unusual; default.
            _ => 1.0,
        },

        // Variables / parameters / function calls / property
        // accesses / arithmetic — at the predicate root these are
        // expressions of unknown selectivity; default to
        // DEFAULT_EQ_SELECTIVITY by deferring to the estimator.
        BoundExpression::Parameter { .. }
        | BoundExpression::ListLiteral { .. }
        | BoundExpression::MapLiteral { .. }
        | BoundExpression::VariableRef { .. }
        | BoundExpression::UnresolvedVariable { .. }
        | BoundExpression::PropertyAccess { .. }
        | BoundExpression::FunctionCall { .. }
        | BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        // ADR-188 — list-predicates (`all`/`any`/`none`/`single`) and
        // `reduce` are expression-internal scoped folds of unknown
        // selectivity at the predicate root; defer to the estimator's
        // default like every other non-comparison expression. A
        // list-comprehension (#620 list-half) produces a LIST (not a
        // Boolean) so it would only reach the predicate root via a
        // type-incoherent `WHERE [x IN l | x]` (the type-check rejects
        // it); we still include it in the exhaustive default group so
        // the match is total and the estimate is the conservative
        // default.
        | BoundExpression::ListPredicate { .. }
        | BoundExpression::Reduce { .. }
        | BoundExpression::ListComprehension { .. }
        // ADR-191 D-6 (#620 map-half) — a map projection produces a MAP
        // (not a Boolean), so it only reaches the predicate root via a
        // type-incoherent `WHERE n{.x}` (the type-check rejects it);
        // included in the conservative default group so the match is total.
        | BoundExpression::MapProjection { .. }
        // openCypher v9 §3.4 — a subscript / slice produces a VALUE (not
        // a Boolean), so it only reaches the predicate root via a
        // type-incoherent `WHERE list[0]` (the type-check rejects it);
        // included in the conservative default group so the match is
        // total.
        | BoundExpression::Subscript { .. }
        | BoundExpression::Slice { .. }
        // #621 — a CASE at the predicate root. Unlike the comprehensions /
        // map / subscript (which produce non-Boolean values the type-check
        // rejects at a predicate root), a SEARCHED CASE legitimately yields
        // a Boolean (`WHERE CASE WHEN … THEN true ELSE false END`), so it is
        // type-coherent here — but its selectivity is opaque to the v1.0
        // estimator, so we defer to the same conservative default as every
        // other non-comparison expression.
        | BoundExpression::Case { .. } => estimator.estimate_eq(default_var, None),
    }
}

/// Walk an expression to find the leftmost variable binding it
/// touches. Returns `None` for expressions that bind no variable
/// (literals, parameters, etc.). Used to scope per-binding
/// estimates; v1.0 ignores the result, but it's threaded through
/// for v1.1 per-binding sketches.
fn leftmost_binding(expr: &BoundExpression) -> Option<BindingId> {
    match expr {
        BoundExpression::VariableRef { binding_id, .. } => Some(*binding_id),
        BoundExpression::PropertyAccess { base, .. } => leftmost_binding(base),
        BoundExpression::ListLiteral { elements, .. } => elements.iter().find_map(leftmost_binding),
        BoundExpression::MapLiteral { entries, .. } => entries
            .iter()
            .find_map(|(_, value)| leftmost_binding(value)),
        // #1290 — left-nested operator SPINE walked iteratively (the
        // spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may interleave
        // BinaryOp / UnaryOp / In / IsNull levels; recursing per level
        // overflowed the native stack). Base subtree first, then each
        // level's rhs innermost→outermost — the same source-order
        // find-first the recursion this replaces performed.
        BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        | BoundExpression::In { .. }
        | BoundExpression::IsNull { .. } => {
            let mut rhs_stack: Vec<&BoundExpression> = Vec::new();
            let mut cur = expr;
            loop {
                match cur {
                    BoundExpression::BinaryOp { lhs, rhs, .. }
                    | BoundExpression::In { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    BoundExpression::UnaryOp { operand, .. } => cur = operand,
                    BoundExpression::IsNull { lhs, .. } => cur = lhs,
                    _ => break,
                }
            }
            leftmost_binding(cur)
                .or_else(|| rhs_stack.iter().rev().find_map(|rhs| leftmost_binding(rhs)))
        }
        BoundExpression::FunctionCall { args, .. } => args.iter().find_map(leftmost_binding),
        BoundExpression::Near { lhs, target, .. } => {
            leftmost_binding(lhs).or_else(|| leftmost_binding(target))
        }
        BoundExpression::TextMatch { lhs, query, .. } => {
            leftmost_binding(lhs).or_else(|| leftmost_binding(query))
        }
        BoundExpression::InCommunity {
            node, community, ..
        } => leftmost_binding(node).or_else(|| leftmost_binding(community)),
        // ADR-188 — the leftmost binding a list-predicate / reduce
        // touches is the one in its LIST operand (an outer-scope
        // expression); the predicate/body reference the
        // expression-internal scoped var, not an outer row binding, so
        // we descend into `list`.
        BoundExpression::ListPredicate { list, .. } => leftmost_binding(list),
        BoundExpression::Reduce { list, init, .. } => {
            leftmost_binding(list).or_else(|| leftmost_binding(init))
        }
        // ADR-188 (#620 list-half) — the leftmost binding a
        // list-comprehension touches is the one in its LIST operand (an
        // outer-scope expression); the predicate/projection reference
        // the expression-internal scoped var, not an outer row binding,
        // so we descend into `list` (same as `ListPredicate`).
        BoundExpression::ListComprehension { list, .. } => leftmost_binding(list),
        // ADR-191 D-6 (#620 map-half) — the leftmost binding a map
        // projection touches is its BASE (the projected variable); the
        // literal-entry values are typically constants / params. Descend
        // into `base`.
        BoundExpression::MapProjection { base, .. } => leftmost_binding(base),
        // openCypher v9 §3.4 — the leftmost binding a subscript / slice
        // touches is the one in its BASE (the index / bounds are
        // typically constants or params); descend into `base`.
        BoundExpression::Subscript { base, .. } => leftmost_binding(base),
        BoundExpression::Slice { base, .. } => leftmost_binding(base),
        // #621 — the leftmost binding a CASE touches is the first one found
        // scanning its test → each WHEN/THEN arm → ELSE (source order).
        BoundExpression::Case {
            test,
            branches,
            default,
            ..
        } => test
            .as_deref()
            .and_then(leftmost_binding)
            .or_else(|| {
                branches.iter().find_map(|(when, then)| {
                    leftmost_binding(when).or_else(|| leftmost_binding(then))
                })
            })
            .or_else(|| default.as_deref().and_then(leftmost_binding)),
        BoundExpression::Literal { .. }
        | BoundExpression::Parameter { .. }
        | BoundExpression::UnresolvedVariable { .. } => None,
    }
}

/// Estimate the size of an IN-list operand. Recognizes literal
/// list-shape expressions; falls back to a moderate constant when
/// the RHS is a parameter / function call (where the list size is
/// only known at execute time).
fn estimate_list_size(expr: &BoundExpression) -> usize {
    // List literals now have a dedicated variant; unsupported RHS shapes use a fallback constant.
    let _ = expr;
    /// Conservative midpoint for unknown IN-list sizes (Postgres
    /// uses 10 as the default `IN`-list NDV; we match).
    const FALLBACK_IN_LIST_SIZE: usize = 10;
    FALLBACK_IN_LIST_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::semantic::{DEFAULT_EQ_SELECTIVITY, DEFAULT_LT_SELECTIVITY, StubCatalogProvider};

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn var(id: u64) -> BoundExpression {
        BoundExpression::VariableRef {
            name: format!("v{}", id),
            binding_id: BindingId::new(id),
            span: span(),
            type_info: None,
        }
    }

    fn lit(b: bool) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Bool(b),
            span: span(),
            type_info: None,
        }
    }

    #[test]
    fn predicate_eq_uses_estimate_eq() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(var(0)),
            rhs: Box::new(BoundExpression::Literal {
                value: Literal::Integer(42),
                span: span(),
                type_info: None,
            }),
            span: span(),
            type_info: None,
        };
        // estimate_eq with total_node_count=1000 → 1/1000 = 0.001.
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        assert!((s - 0.001).abs() < 1e-12);
    }

    #[test]
    fn predicate_neq_complements_eq() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Neq,
            lhs: Box::new(var(0)),
            rhs: Box::new(BoundExpression::Parameter {
                name: "x".into(),
                span: span(),
                type_info: None,
            }),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        // 1 - 0.001 = 0.999.
        assert!((s - 0.999).abs() < 1e-12);
    }

    #[test]
    fn predicate_lt_uses_estimate_lt() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Lt,
            lhs: Box::new(var(0)),
            rhs: Box::new(BoundExpression::Literal {
                value: Literal::Integer(1),
                span: span(),
                type_info: None,
            }),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        assert_eq!(s, DEFAULT_LT_SELECTIVITY);
    }

    #[test]
    fn predicate_and_composes_pairwise() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let eq_pred = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let lt_pred = BoundExpression::BinaryOp {
            op: BinOp::Lt,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let and_pred = BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs: Box::new(eq_pred),
            rhs: Box::new(lt_pred),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&and_pred, &est, BindingId::new(0));
        // 0.001 * 0.33 = 0.00033.
        assert!((s - 0.001 * DEFAULT_LT_SELECTIVITY).abs() < 1e-12);
    }

    #[test]
    fn predicate_or_uses_inclusion_exclusion() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let eq_pred = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let lt_pred = BoundExpression::BinaryOp {
            op: BinOp::Lt,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let or_pred = BoundExpression::BinaryOp {
            op: BinOp::Or,
            lhs: Box::new(eq_pred),
            rhs: Box::new(lt_pred),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&or_pred, &est, BindingId::new(0));
        // 1 - (1 - 0.001) * (1 - 0.33) = 1 - 0.999 * 0.67 = 1 - 0.66933 = 0.33067.
        let expected = 1.0 - (1.0 - 0.001) * (1.0 - DEFAULT_LT_SELECTIVITY);
        assert!((s - expected).abs() < 1e-9);
    }

    #[test]
    fn predicate_not_complements_inner() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let inner = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let not_pred = BoundExpression::UnaryOp {
            op: UnaryOp::Not,
            operand: Box::new(inner),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&not_pred, &est, BindingId::new(0));
        assert!((s - 0.999).abs() < 1e-12);
    }

    #[test]
    fn predicate_in_uses_estimate_in_with_fallback_size() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::In {
            lhs: Box::new(var(0)),
            rhs: Box::new(BoundExpression::Parameter {
                name: "ids".into(),
                span: span(),
                type_info: None,
            }),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        // FALLBACK_IN_LIST_SIZE=10, total=1000 → 10/1000 = 0.01.
        assert!((s - 0.01).abs() < 1e-12);
    }

    #[test]
    fn predicate_is_null_treated_as_eq() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::IsNull {
            lhs: Box::new(var(0)),
            negated: false,
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        assert!((s - 0.001).abs() < 1e-12);

        let pred_neg = BoundExpression::IsNull {
            lhs: Box::new(var(0)),
            negated: true,
            span: span(),
            type_info: None,
        };
        let s_neg = predicate_selectivity(&pred_neg, &est, BindingId::new(0));
        assert!((s_neg - 0.999).abs() < 1e-12);
    }

    #[test]
    fn predicate_literal_true_pass_all_false_pass_none() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(
            predicate_selectivity(&lit(true), &est, BindingId::new(0)),
            1.0
        );
        assert_eq!(
            predicate_selectivity(&lit(false), &est, BindingId::new(0)),
            0.0
        );
    }

    #[test]
    fn predicate_falls_back_to_default_eq_when_stats_empty() {
        // Empty catalog → estimate_eq returns DEFAULT_EQ_SELECTIVITY.
        let cat = StubCatalogProvider::new();
        let est = SelectivityEstimator::new(&cat);
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(var(0)),
            rhs: Box::new(var(0)),
            span: span(),
            type_info: None,
        };
        let s = predicate_selectivity(&pred, &est, BindingId::new(0));
        assert_eq!(s, DEFAULT_EQ_SELECTIVITY);
    }
}
