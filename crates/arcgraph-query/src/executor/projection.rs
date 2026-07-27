//! v2 M2 — plan-time property-projection pushdown (design §M2.3;
//! ADR-230 row M2).
//!
//! Derives, for a `Project(Scan)` / `Project(Filter(Scan))` chain, the
//! COMPLETE set of property names the plan consumes from the scanned
//! variable — so the scan can materialize only those properties
//! (`PropBlockView` touches only the projected key_ids; untouched
//! properties cost zero — "the cost of a read is O(|projection|), not
//! O(|bag|)").
//!
//! # Safety polarity (the load-bearing design rule)
//!
//! Pushing a projection UNDER-fetches if any consumption site is
//! missed — a silent wrong-read. Every uncertainty in this module
//! therefore resolves to **no pushdown** (full-bag scan — over-fetch,
//! never wrong):
//!
//! - Only the exact chain shapes `Project(Scan)` and
//!   `Project(Filter(Scan))` are analyzed (the dominant point-lookup /
//!   filtered-scan shapes; e.g. `MATCH (n:Incident) WHERE
//!   n.severity = 'high' RETURN n.title, n.opened_at`). Any other
//!   plan shape → `None`.
//! - A `Project` here is an ESCAPE BARRIER: its output schema carries
//!   only the projected items, so nothing above it can reference the
//!   scan variable — the analysis is complete by construction once
//!   every chain-local expression is classified. A projection item
//!   that passes the variable WHOLE (`RETURN n`, `WITH n`, `RETURN *`)
//!   classifies as whole-entity → `None`.
//! - The expression classifier walks the same variant surface as
//!   `collect_referenced_bindings` (the lowering's exhaustive
//!   reference walker); any variant it does not explicitly handle —
//!   including future additions — classifies as whole-entity via the
//!   catch-all (`_ => Whole`). An unknown expression form can only
//!   DISABLE pushdown, never mis-scope it.
//! - `Distinct` is deliberately NOT admitted between the barrier and
//!   the scan: it compares WHOLE rows, and two nodes equal under a
//!   projected bag may differ under the full bag — deduping them
//!   would be a wrong result.
//!
//! # Budget (PD#5)
//!
//! One recursive walk over the chain's expressions at plan-build time
//! — O(plan size), zero per-row cost. The name→key_id resolution
//! happens once per scan call on the substrate side (design §M2.3
//! "resolved at plan time, never per row").

use std::collections::BTreeSet;

use crate::logical_plan::{LogicalPlan, LogicalScan};
use crate::semantic::bound_ast::{
    BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind,
};

/// How a chain's expressions use the scanned variable.
#[derive(Debug, PartialEq, Eq)]
enum VarUse {
    /// Only `var.prop` accesses — the collected top-level names.
    PropsOnly(BTreeSet<String>),
    /// The variable escapes whole (bare reference, unknown expression
    /// form, wildcard, …) — pushdown is off.
    Whole,
}

impl VarUse {
    fn merge(self, other: VarUse) -> VarUse {
        match (self, other) {
            (VarUse::PropsOnly(mut a), VarUse::PropsOnly(b)) => {
                a.extend(b);
                VarUse::PropsOnly(a)
            }
            _ => VarUse::Whole,
        }
    }

    fn empty() -> VarUse {
        VarUse::PropsOnly(BTreeSet::new())
    }
}

/// Detect a pushdown-eligible chain rooted at `plan` and return the
/// scan variable + the sorted property-name set to push into the scan.
///
/// `None` = no pushdown (the safe default). `Some((var, names))` =
/// every consumption of `var` in the chain is a `var.<name>` access
/// with `<name>` ∈ `names`, and `var` cannot escape the chain (the
/// `Project` barrier).
#[must_use]
pub(crate) fn scan_projection_for_chain(plan: &LogicalPlan) -> Option<(BindingId, Vec<String>)> {
    let LogicalPlan::Project(p) = plan else {
        return None;
    };
    // The admitted input shapes: Scan, Filter(Scan).
    let (scan, filter_predicate): (&LogicalScan, Option<&BoundExpression>) = match p.input.as_ref()
    {
        LogicalPlan::Scan(s) => (s, None),
        LogicalPlan::Filter(f) => match f.input.as_ref() {
            LogicalPlan::Scan(s) => (s, Some(&f.predicate)),
            _ => return None,
        },
        _ => return None,
    };
    let var = scan.var;

    let mut usage = VarUse::empty();
    for item in &p.items {
        usage = usage.merge(classify_projection_item(item, var));
        if usage == VarUse::Whole {
            return None;
        }
    }
    if let Some(pred) = filter_predicate {
        usage = usage.merge(classify_expr(pred, var));
    }
    match usage {
        VarUse::PropsOnly(names) if !names.is_empty() => Some((var, names.into_iter().collect())),
        // An empty consumption set means the plan reads NO properties
        // of `var` (e.g. `RETURN 1`): push the EMPTY projection —
        // zero-property materialization is exactly the design's
        // "a count/existence check materializes nothing".
        VarUse::PropsOnly(_) => Some((var, Vec::new())),
        VarUse::Whole => None,
    }
}

/// Classify one projection item's use of `var`.
fn classify_projection_item(item: &BoundProjectionItem, var: BindingId) -> VarUse {
    match &item.kind {
        // `RETURN *` passes every binding whole.
        BoundProjectionKind::Wildcard { .. } => VarUse::Whole,
        BoundProjectionKind::Expr(e) => classify_expr(e, var),
    }
}

/// Classify how `e` uses `var`. Mirrors the variant surface of the
/// lowering's `collect_referenced_bindings`; every form not explicitly
/// handled classifies as [`VarUse::Whole`] (the safe polarity — see
/// the module docs).
fn classify_expr(e: &BoundExpression, var: BindingId) -> VarUse {
    use BoundExpression as BE;
    match e {
        // `var.prop[.sub…]` with the base being EXACTLY the variable:
        // consumes only the FIRST path segment's top-level property
        // (nested segments resolve inside that materialized value).
        BE::PropertyAccess { base, path, .. } => match base.as_ref() {
            BE::VariableRef { binding_id, .. } if *binding_id == var => match path.first() {
                Some(seg) => {
                    let mut s = BTreeSet::new();
                    s.insert(seg.name.clone());
                    VarUse::PropsOnly(s)
                }
                // A property access with an empty path cannot name
                // what it reads — treat as whole (defensive).
                None => VarUse::Whole,
            },
            // `f(x).prop` etc. — classify the base itself.
            other => classify_expr(other, var),
        },
        // A bare reference to the variable = whole-entity escape.
        BE::VariableRef { binding_id, .. } => {
            if *binding_id == var {
                VarUse::Whole
            } else {
                VarUse::empty()
            }
        }
        BE::Literal { .. } | BE::Parameter { .. } | BE::UnresolvedVariable { .. } => {
            VarUse::empty()
        }
        BE::ListLiteral { elements, .. } => elements
            .iter()
            .fold(VarUse::empty(), |acc, el| acc.merge(classify_expr(el, var))),
        BE::MapLiteral { entries, .. } => entries.iter().fold(VarUse::empty(), |acc, (_, v)| {
            acc.merge(classify_expr(v, var))
        }),
        BE::BinaryOp { lhs, rhs, .. } | BE::In { lhs, rhs, .. } => {
            classify_expr(lhs, var).merge(classify_expr(rhs, var))
        }
        BE::UnaryOp { operand, .. } => classify_expr(operand, var),
        BE::IsNull { lhs, .. } => classify_expr(lhs, var),
        BE::FunctionCall { args, .. } => args
            .iter()
            .fold(VarUse::empty(), |acc, a| acc.merge(classify_expr(a, var))),
        BE::Near { lhs, target, .. } => classify_expr(lhs, var).merge(classify_expr(target, var)),
        BE::TextMatch { lhs, query, .. } => {
            classify_expr(lhs, var).merge(classify_expr(query, var))
        }
        BE::InCommunity {
            node, community, ..
        } => classify_expr(node, var).merge(classify_expr(community, var)),
        BE::ListPredicate {
            list, predicate, ..
        } => classify_expr(list, var).merge(classify_expr(predicate, var)),
        BE::Reduce {
            init, list, expr, ..
        } => classify_expr(init, var)
            .merge(classify_expr(list, var))
            .merge(classify_expr(expr, var)),
        BE::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            let mut u = classify_expr(list, var);
            if let Some(p) = predicate {
                u = u.merge(classify_expr(p, var));
            }
            if let Some(p) = projection {
                u = u.merge(classify_expr(p, var));
            }
            u
        }
        BE::Subscript { base, index, .. } => {
            classify_expr(base, var).merge(classify_expr(index, var))
        }
        BE::Slice {
            base, start, end, ..
        } => {
            let mut u = classify_expr(base, var);
            if let Some(s) = start {
                u = u.merge(classify_expr(s, var));
            }
            if let Some(s) = end {
                u = u.merge(classify_expr(s, var));
            }
            u
        }
        // Map projection reads the base's properties dynamically
        // (`.*` copies ALL) — conservative whole-entity when it
        // targets `var`; otherwise classify its item expressions.
        BE::MapProjection { base, .. } => match base.as_ref() {
            BE::VariableRef { binding_id, .. } if *binding_id == var => VarUse::Whole,
            other => classify_expr(other, var),
        },
        // Any variant not explicitly handled — INCLUDING variants
        // added after this module — disables pushdown (safe polarity;
        // e.g. `Case` carries nested branch expressions whose scoping
        // is easy to get subtly wrong, so it stays conservative until
        // a consumer needs it).
        _ => VarUse::Whole,
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::Lsn;

    use super::*;
    use crate::error::Span;
    use crate::logical_plan::{LogicalFilter, LogicalProject};
    use crate::semantic::bound_ast::BoundPropertyRef;

    fn span() -> Span {
        Span::point(0, 0)
    }

    fn var_ref(id: u64) -> BoundExpression {
        BoundExpression::VariableRef {
            name: format!("v{id}"),
            binding_id: BindingId::new(id),
            span: span(),
            type_info: None,
        }
    }

    fn prop(id: u64, name: &str) -> BoundExpression {
        BoundExpression::PropertyAccess {
            base: Box::new(var_ref(id)),
            path: vec![BoundPropertyRef {
                name: name.to_string(),
                property_id: None,
                span: span(),
            }],
            span: span(),
            type_info: None,
        }
    }

    fn item(e: BoundExpression) -> BoundProjectionItem {
        BoundProjectionItem {
            kind: BoundProjectionKind::Expr(e),
            alias: None,
            output_id: Some(BindingId::new(900)),
            source_text: None,
            span: span(),
        }
    }

    fn scan(var: u64) -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: None,
            var: BindingId::new(var),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    }

    fn project(items: Vec<BoundProjectionItem>, input: LogicalPlan) -> LogicalPlan {
        LogicalPlan::Project(LogicalProject {
            input: Box::new(input),
            items,
            span: span(),
        })
    }

    #[test]
    fn project_over_scan_props_only_pushes() {
        let plan = project(vec![item(prop(1, "title")), item(prop(1, "sev"))], scan(1));
        let (var, names) = scan_projection_for_chain(&plan).expect("pushdown");
        assert_eq!(var, BindingId::new(1));
        assert_eq!(names, vec!["sev".to_string(), "title".to_string()]);
    }

    #[test]
    fn filter_predicate_props_join_the_set() {
        let filtered = LogicalPlan::Filter(LogicalFilter {
            input: Box::new(scan(1)),
            predicate: BoundExpression::BinaryOp {
                op: crate::ast::BinOp::Eq,
                lhs: Box::new(prop(1, "severity")),
                rhs: Box::new(BoundExpression::Parameter {
                    name: "p".into(),
                    span: span(),
                    type_info: None,
                }),
                span: span(),
                type_info: None,
            },
            span: span(),
        });
        let plan = project(vec![item(prop(1, "title"))], filtered);
        let (_, names) = scan_projection_for_chain(&plan).expect("pushdown");
        assert_eq!(names, vec!["severity".to_string(), "title".to_string()]);
    }

    #[test]
    fn whole_entity_return_disables_pushdown() {
        let plan = project(vec![item(var_ref(1))], scan(1));
        assert!(scan_projection_for_chain(&plan).is_none());
    }

    #[test]
    fn wildcard_disables_pushdown() {
        let plan = project(
            vec![BoundProjectionItem {
                kind: BoundProjectionKind::wildcard(),
                alias: None,
                output_id: None,
                source_text: None,
                span: span(),
            }],
            scan(1),
        );
        assert!(scan_projection_for_chain(&plan).is_none());
    }

    #[test]
    fn entity_inside_function_disables_pushdown() {
        let plan = project(
            vec![item(BoundExpression::FunctionCall {
                name: "properties".into(),
                args: vec![var_ref(1)],
                distinct: false,
                star: false,
                span: span(),
                type_info: None,
            })],
            scan(1),
        );
        assert!(scan_projection_for_chain(&plan).is_none());
    }

    #[test]
    fn other_bindings_props_do_not_pollute_the_scan_vars_set() {
        // n.title + m.other — only n (var 1) is the scan var; m's
        // access neither pushes a name nor disables (m is not scanned
        // here).
        let plan = project(
            vec![item(prop(1, "title")), item(prop(2, "other"))],
            scan(1),
        );
        let (_, names) = scan_projection_for_chain(&plan).expect("pushdown");
        assert_eq!(names, vec!["title".to_string()]);
    }

    #[test]
    fn no_property_consumption_pushes_the_empty_projection() {
        // `MATCH (n) RETURN 1` — zero properties needed: the empty
        // pushdown (existence-only materialization, design §M2.2).
        let plan = project(
            vec![item(BoundExpression::Literal {
                value: crate::ast::Literal::Integer(1),
                span: span(),
                type_info: None,
            })],
            scan(1),
        );
        let (_, names) = scan_projection_for_chain(&plan).expect("pushdown");
        assert!(names.is_empty());
    }

    #[test]
    fn non_chain_shapes_disable_pushdown() {
        // Bare scan (no Project barrier) — nothing above bounds the
        // consumption set.
        assert!(scan_projection_for_chain(&scan(1)).is_none());
    }
}
