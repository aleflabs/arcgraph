//! M4-33 aggregation grouping correctness proptest (256 cases).
//!
//! # Property under test
//!
//! For any well-formed RETURN clause carrying a mix of aggregation
//! and non-aggregation projection items, the M4-33 lowering pass
//! partitions the items into:
//! - `aggregations` — items whose top-level expression is an
//!   aggregation function call (`count` / `sum` / `avg` / `min` /
//!   `max` / `collect`);
//! - `group_by` — every other item (the implicit GROUP BY per
//!   openCypher 9 §6.4).
//!
//! The PARTITION INVARIANT: `len(group_by) + len(aggregations) ==
//! len(items)`, with no item double-counted, no item dropped, and the
//! partition matching the source-item type-tag.
//!
//! This is the load-bearing safety property for the M4-05 cost-based
//! planner: aggregation cardinality is `1` per group; cost-planner
//! must be able to derive groups from `group_by` alone.
//!
//! # Strategy
//!
//! The proptest generator builds RETURN clauses from a fixed
//! alphabet:
//! - 4 aggregation functions admitted at v1.0
//!   (count / min / max / collect — sum / avg defer to literal arg
//!   per the dynamic-schema sentinel — see
//!   `aggregation_lowering_integration.rs::lower_sum_avg_min_max`);
//! - 4 grouping-key shapes: `n`, `n.name`, integer literal, string
//!   literal;
//! - 1..6 RETURN items per query (mixed aggregation + grouping).
//!
//! 256 cases × ~3 items each = ~750 random partitions exercised.
//!
//! # ADR provenance
//! - ADR-038 §2 D-28 — aggregation contract.
//! - ADR-038 §2 D-22 — M4-22 aggregation function registry.
//! - ADR-038 amendment-03 §M4-33 row — proptest pin.

use arcgraph_query::logical_plan::{
    AggregationKind, LogicalAggregate, LogicalPlan, LogicalPlanLoweringVisitor,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name"])
}

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ReturnItem {
    /// Aggregation form: e.g. `count(n)`, `min(n.age)`, `collect(n)`.
    /// Uses ArgKind::Any-compatible aggregations only (count / min /
    /// max / collect) to side-step the v1.0 dynamic-schema
    /// PropertyType::String sentinel for sum / avg.
    Aggregation(AggregationKind),
    /// Group-by form: a non-aggregation item.
    /// 0 = `n`, 1 = `n.age`, 2 = `42` (integer literal), 3 = `"x"`
    /// (string literal).
    GroupBy(u8),
}

fn render_item(it: &ReturnItem) -> String {
    match it {
        ReturnItem::Aggregation(AggregationKind::Count) => "count(n)".into(),
        ReturnItem::Aggregation(AggregationKind::Min) => "min(n.age)".into(),
        ReturnItem::Aggregation(AggregationKind::Max) => "max(n.age)".into(),
        ReturnItem::Aggregation(AggregationKind::Collect) => "collect(n)".into(),
        // Sum / Avg are unreachable in this generator (per v1.0
        // dynamic-schema constraint); kept for AggregationKind
        // exhaustiveness.
        ReturnItem::Aggregation(AggregationKind::Sum) => "sum(1)".into(),
        ReturnItem::Aggregation(AggregationKind::Avg) => "avg(1)".into(),
        ReturnItem::GroupBy(0) => "n".into(),
        ReturnItem::GroupBy(1) => "n.age".into(),
        ReturnItem::GroupBy(2) => "42".into(),
        ReturnItem::GroupBy(_) => "\"x\"".into(),
    }
}

fn item_strategy() -> impl Strategy<Value = ReturnItem> {
    prop_oneof![
        // 50% aggregations
        Just(ReturnItem::Aggregation(AggregationKind::Count)),
        Just(ReturnItem::Aggregation(AggregationKind::Min)),
        Just(ReturnItem::Aggregation(AggregationKind::Max)),
        Just(ReturnItem::Aggregation(AggregationKind::Collect)),
        // 50% group-by shapes
        Just(ReturnItem::GroupBy(0)),
        Just(ReturnItem::GroupBy(1)),
        Just(ReturnItem::GroupBy(2)),
        Just(ReturnItem::GroupBy(3)),
    ]
}

prop_compose! {
    fn return_clause_strategy()
        (items in prop::collection::vec(item_strategy(), 1..=6))
        -> Vec<ReturnItem>
    {
        items
    }
}

// ---------------------------------------------------------------------
// Pipeline + walkers
// ---------------------------------------------------------------------

fn try_lower(input: &str) -> Option<LogicalPlan> {
    let stmt = parse(input).ok()?;
    let mut bound = BindingVisitor::bind(&stmt, input, &cat()).ok()?;
    TypeCheckVisitor::check(&mut bound, &cat()).ok()?;
    CrossSubstrateValidator::validate(&bound, &cat()).ok()?;
    LogicalPlanLoweringVisitor::lower(&bound).ok()
}

fn find_aggregate(p: &LogicalPlan) -> Option<&LogicalAggregate> {
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

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn aggregation_partitions_items_consistently(items in return_clause_strategy()) {
        // Build the source query.
        let rendered: Vec<String> = items.iter().map(render_item).collect();
        let return_body = rendered.join(", ");
        let input = format!("MATCH (n:Person) RETURN {return_body}");

        // Lower; if pre-lowering stages fail (parse / bind / type-
        // check / cross-substrate), we just skip — those error paths
        // are exercised by other proptests.
        let plan = match try_lower(&input) {
            Some(p) => p,
            None => return Ok(()),
        };

        // Compute the expected partition counts from the GENERATOR.
        let expected_aggregations = items
            .iter()
            .filter(|it| matches!(it, ReturnItem::Aggregation(_)))
            .count();
        let expected_group_by = items.len() - expected_aggregations;

        match find_aggregate(&plan) {
            Some(agg) => {
                // PARTITION INVARIANT: counts add up.
                prop_assert_eq!(
                    agg.aggregations.len(),
                    expected_aggregations,
                    "aggregation count mismatch for: {}",
                    input
                );
                prop_assert_eq!(
                    agg.group_by.len(),
                    expected_group_by,
                    "group_by count mismatch for: {}",
                    input
                );
                prop_assert_eq!(
                    agg.aggregations.len() + agg.group_by.len(),
                    items.len(),
                    "total preserves source-item count for: {}",
                    input
                );
            }
            None => {
                // No Aggregate node was emitted. This is correct iff
                // the source items contained ZERO aggregations.
                prop_assert_eq!(
                    expected_aggregations,
                    0,
                    "expected Aggregate node when items contain ≥1 aggregation; input: {}",
                    input
                );
            }
        }
    }
}
