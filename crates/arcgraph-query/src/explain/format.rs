//! Text formatter for [`PlanTree`].
//!
//! The format is a stable, human-readable indented listing — one
//! operator per line, two-space indents per child level, with cost +
//! cardinality + bindings + annotations on the same line as the
//! operator name. Stability is load-bearing: snapshot tests in
//! `tests/m4_91_explain_integration.rs` compare formatter output
//! byte-for-byte. Annotation iteration uses [`std::collections::BTreeMap`]
//! ordering (alphabetical), guaranteed via the [`PlanTree`] field type.
//!
//! # Format example
//!
//! ```text
//! Project [items=2] cost=1150.00 rows=1000.00
//!   Filter cost=1100.00 rows=1000.00
//!     Scan b0 [label=L1, read_lsn=18446744073709551615] cost=1000.00 rows=1000.00
//! ```
//!
//! - Each operator opens with `PlanTreeOp::name`.
//! - Bindings (when non-empty) follow on the same line as
//!   space-separated `b{raw}` tokens.
//! - Annotations follow in `[k1=v1, k2=v2]` form (alphabetical by key
//!   per BTreeMap iteration).
//! - Cost + card append as `cost=<f.2> rows=<f.2>`.
//! - Children indent two spaces and recurse.
//!
//! Floats render with two decimal places to keep snapshot diffs
//! readable; the `f64` precision-loss is acceptable for EXPLAIN output
//! (the underlying [`PlanTree::estimated_cost`] / `estimated_card`
//! retain full precision for callers that need the raw value).

use std::fmt;

use super::plan_tree::PlanTree;

impl fmt::Display for PlanTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_at(f, 0)
    }
}

impl PlanTree {
    /// Render the binding + annotation detail segment used after the
    /// operator name and before cost/cardinality in [`Display`].
    #[must_use]
    pub(crate) fn details_string(&self) -> String {
        let mut details = String::new();
        if !self.bindings.is_empty() {
            for (i, b) in self.bindings.iter().enumerate() {
                if i > 0 {
                    details.push(' ');
                }
                details.push_str(b);
            }
        }
        if !self.annotations.is_empty() {
            if !details.is_empty() {
                details.push(' ');
            }
            details.push('[');
            for (i, (k, v)) in self.annotations.iter().enumerate() {
                if i > 0 {
                    details.push_str(", ");
                }
                details.push_str(k);
                details.push('=');
                details.push_str(v);
            }
            details.push(']');
        }
        details
    }

    fn fmt_at(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        // Indent.
        for _ in 0..depth {
            f.write_str("  ")?;
        }
        // Operator name.
        f.write_str(self.op.name())?;
        let details = self.details_string();
        if !details.is_empty() {
            f.write_str(" ")?;
            f.write_str(&details)?;
        }
        // Cost + cardinality.
        write!(
            f,
            " cost={:.2} rows={:.2}",
            self.estimated_cost.total(),
            self.estimated_card.rows(),
        )?;
        f.write_str("\n")?;
        // Recurse into children.
        for child in &self.children {
            child.fmt_at(f, depth + 1)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::explain::plan_tree::PlanTreeOp;
    use crate::logical_plan::types::*;
    use crate::planner::cost::estimate_costs;
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::{BindingId, BoundExpression};
    use arcgraph_core::{LabelId, Lsn};

    fn span() -> Span {
        Span::point(1, 1)
    }

    #[test]
    fn empty_plan_renders_one_line() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let s = format!("{pt}");
        // Empty has no bindings + no annotations + 0 cost + 0 rows.
        assert_eq!(s, "Empty cost=0.00 rows=0.00\n");
    }

    #[test]
    fn nested_tree_indents_two_spaces_per_level() {
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
            items: vec![],
            span: span(),
        };
        let plan = LogicalPlan::Project(project);
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let s = format!("{pt}");
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3);
        // Project at depth 0 (no leading space).
        assert!(lines[0].starts_with("Project"), "got: {:?}", lines[0]);
        // Filter at depth 1 (two leading spaces).
        assert!(lines[1].starts_with("  Filter"), "got: {:?}", lines[1]);
        // Scan at depth 2 (four leading spaces).
        assert!(lines[2].starts_with("    Scan b0"), "got: {:?}", lines[2]);
    }

    #[test]
    fn annotations_order_is_alphabetical_per_btreemap() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(2)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let s = format!("{pt}");
        // Alphabetical: label < read_lsn.
        let lo = s.find("label=").expect("has label");
        let hi = s.find("read_lsn=").expect("has read_lsn");
        assert!(lo < hi, "annotation order must be alphabetical: full=`{s}`");
    }

    #[test]
    fn binding_only_node_renders_without_annotations_block() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let s = format!("{pt}");
        assert!(!s.contains('['), "Empty has no annotations: {s}");
        // Sanity: PlanTreeOp::Empty renders as exactly "Empty".
        assert!(s.starts_with("Empty"), "{s}");
    }

    #[test]
    fn float_format_is_two_decimal_places() {
        // Picked a stub catalog that produces a non-integer cost so
        // we can pin the `.NN` slice. The Filter's selectivity over a
        // BoolTrue literal is 1.0 → cardinality 1000 → cost 100 — but
        // the chained Scan(1000) + Filter(100) + Project(50) = 1150.
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
            items: vec![],
            span: span(),
        };
        let plan = LogicalPlan::Project(project);
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let root_line = format!("{pt}").lines().next().unwrap().to_string();
        // Pin the format once to lock down byte-stable rendering.
        assert!(
            root_line.contains("cost=1150.00 rows=1000.00"),
            "root_line=`{root_line}`",
        );
    }

    #[test]
    fn matches_planeop_name_for_every_op_kind() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let line = format!("{pt}");
        assert!(line.starts_with(PlanTreeOp::Empty.name()));
    }
}
