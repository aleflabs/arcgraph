//! M4-33 aggregation lowering integration tests per ADR-038 §2 D-28.
//!
//! M4-31 + M4-32 deferred RETURN-aggregation surfaces with
//! `NotImplementedAtM4_31 { surface: "aggregation function",
//! target_slice: "M4-33", .. }`; M4-33 lights them up.
//!
//! # Pin set (per ADR-038 amendment-03 §M4-33 row)
//!
//! 1. `lower_count_with_group_by` — `RETURN n.name, count(n)` →
//!    Aggregate { group_by=[n.name], aggregations=[Count(n)] }.
//! 2. `lower_sum_avg_min_max_aggregations` — each numeric aggregation
//!    function resolves to its [`AggregationKind`] and preserves the
//!    arg expression.
//! 3. `lower_collect_aggregation` — `collect(n)` lowers to
//!    AggregationKind::Collect.
//! 4. `lower_count_star_special_case` — `count(n)` (the v1.0 surrogate
//!    for `count(*)`) lowers to a single-row aggregate (group_by
//!    empty).
//! 5. `lower_aggregation_with_optional_match_null_handling` — OPTIONAL
//!    MATCH + count() composes a LeftOuterJoin under Aggregate, with
//!    NULL-bearing bindings carried through unchanged.
//! 6. `lower_no_aggregation_returns_logical_project_unchanged` —
//!    queries with no aggregation function calls lower without an
//!    Aggregate node (M4-31 / M4-32 contract preserved).
//!
//! # ADR provenance
//! - ADR-038 §2 D-28 — aggregation + sort + path operators contract
//!   (this slice's primary spec).
//! - ADR-038 §2 D-22 — M4-22 aggregation function registry (consumed
//!   by [`AggregationKind::from_function_name`]).
//! - ADR-038 §2 D-26 — M4-32 hybrid + OPTIONAL MATCH baseline
//!   (Pin 5 composes OPTIONAL MATCH lowering with M4-33 aggregation).
//! - ADR-038 amendment-03 §M4-33 row — test-artifact pin (8 unit + 6
//!   integration + 1 proptest).

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::{
    AggregationKind, LogicalAggregate, LogicalPlan, LogicalPlanLoweringVisitor,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

// ---------------------------------------------------------------------
// Pipeline + catalog helpers
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "x", "price", "embedding", "content"])
}

fn lower_ok(input: &str) -> LogicalPlan {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat()).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat()).expect("validate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
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
        LogicalPlan::CreateRel(c) => {
            find_aggregate(&c.source_plan).or_else(|| find_aggregate(&c.target_plan))
        }
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
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => None,
    }
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
        LogicalPlan::Call(_) => out.push("Call"),
        LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
    }
}

// =====================================================================
// Pin 1 — count() with explicit GROUP BY key
// =====================================================================

#[test]
fn lower_count_with_group_by() {
    let plan = lower_ok("MATCH (n:Person) RETURN n.name, count(n)");
    let s = shape(&plan);
    assert!(s.contains(&"Aggregate"), "expected Aggregate: {s:?}");
    assert!(s.contains(&"Project"), "expected Project: {s:?}");

    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(
        agg.group_by.len(),
        1,
        "expected 1 group_by key (n.name); got: {}",
        agg.group_by.len()
    );
    assert_eq!(
        agg.aggregations.len(),
        1,
        "expected 1 aggregation (count(n)); got: {}",
        agg.aggregations.len()
    );
    assert_eq!(agg.aggregations[0].function, AggregationKind::Count);
}

// =====================================================================
// Pin 2 — sum / avg / min / max all resolve correctly
// =====================================================================

#[test]
fn lower_sum_avg_min_max_aggregations() {
    // Note: sum / avg require ArgKind::Numeric, which under v1.0
    // dynamic-schema does NOT admit `Property { value_type: String }`
    // (the catalog's default property-value type per
    // [`crate::semantic::type_check::TypeCheckVisitor`] —
    // PropertyType::String is the dynamic-schema sentinel until v1.1
    // strict-mode lights `lookup_property_type`). To keep this pin
    // schema-agnostic, we feed sum / avg an integer literal; min /
    // max accept ArgKind::Any so they take a property reference.
    let cases: &[(&str, AggregationKind)] = &[
        ("MATCH (n:Person) RETURN sum(1)", AggregationKind::Sum),
        ("MATCH (n:Person) RETURN avg(1)", AggregationKind::Avg),
        ("MATCH (n:Person) RETURN min(n.age)", AggregationKind::Min),
        ("MATCH (n:Person) RETURN max(n.age)", AggregationKind::Max),
    ];
    for (input, expected_kind) in cases {
        let plan = lower_ok(input);
        let agg = find_aggregate(&plan).unwrap_or_else(|| panic!("Aggregate present for: {input}"));
        assert_eq!(agg.aggregations.len(), 1, "input={input}");
        assert_eq!(
            agg.aggregations[0].function, *expected_kind,
            "input={input}",
        );
    }
}

// =====================================================================
// Pin 3 — collect() resolves to AggregationKind::Collect
// =====================================================================

#[test]
fn lower_collect_aggregation() {
    let plan = lower_ok("MATCH (n:Person) RETURN collect(n)");
    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(agg.aggregations.len(), 1);
    assert_eq!(agg.aggregations[0].function, AggregationKind::Collect);
    // Single-row aggregate has empty group_by.
    assert!(agg.group_by.is_empty());
}

// =====================================================================
// Pin 4 — count(n) over a Node binding lowers to single-row aggregate
//
// `count(n)` where `n` is the bound node variable lowers to a single-row
// aggregate (empty group_by). NB: `count(*)` is now ALSO a grammar surface
// (#773 G4 — `star_arg`); it lowers to the SAME single-row shape with the
// `AggregationSpec::star` flag set, and is covered end-to-end in
// `tests/cz773_count_star_distinct_e2e.rs`. This pin exercises the
// `count(n)` (non-star) path specifically.
// =====================================================================

#[test]
fn lower_count_star_special_case() {
    let plan = lower_ok("MATCH (n:Person) RETURN count(n)");
    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(agg.aggregations.len(), 1);
    assert_eq!(agg.aggregations[0].function, AggregationKind::Count);
    assert!(
        agg.group_by.is_empty(),
        "single-row aggregate has empty group_by"
    );
}

// =====================================================================
// Pin 5 — OPTIONAL MATCH + count() composes LeftOuterJoin under
// Aggregate. NULL-bearing bindings flow through; per-aggregation NULL
// semantics apply at execution time per openCypher 9 §6.4.
// =====================================================================

#[test]
fn lower_aggregation_with_optional_match_null_handling() {
    let plan =
        lower_ok("MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) RETURN a, count(b)");
    let s = shape(&plan);
    assert!(s.contains(&"Aggregate"), "expected Aggregate: {s:?}");
    assert!(
        s.contains(&"LeftOuterJoin"),
        "expected LeftOuterJoin under Aggregate (M4-32 OPTIONAL MATCH lowering): {s:?}",
    );

    let agg = find_aggregate(&plan).expect("Aggregate present");
    // group_by = [a]; aggregations = [count(b)].
    assert_eq!(agg.group_by.len(), 1);
    assert_eq!(agg.aggregations.len(), 1);
    assert_eq!(agg.aggregations[0].function, AggregationKind::Count);
}

// =====================================================================
// Pin 6 — No aggregation: plan stays Project-only (no Aggregate node).
// =====================================================================

#[test]
fn lower_no_aggregation_returns_logical_project_unchanged() {
    let plan = lower_ok("MATCH (n:Person) RETURN n.name");
    let s = shape(&plan);
    assert!(
        !s.contains(&"Aggregate"),
        "expected NO Aggregate when no aggregation in items: {s:?}",
    );
    assert!(s.contains(&"Project"), "expected Project: {s:?}");
    assert!(find_aggregate(&plan).is_none());
}

// =====================================================================
// #910 — aggregation NESTED in an expression (two-phase lowering)
//
// CZ AI-native parity: `RETURN count(n)*2`, `RETURN size(collect(x))`,
// `RETURN sum(x)+1`, `RETURN toString(count(n))`, `100.0*count(a)/count(b)`,
// `head/last(collect(x))`, `collect(x)[i]` — and the same in `WITH`. Before
// #910 these errored at row-eval with the MISLEADING `NotImplemented`
// (`aggregation function … reserved per ADR-038 §2 D-28`, MCP `-32005`):
// `try_lift_aggregation` only matched a BARE aggregate, so the OUTER
// expression (a `BinaryOp` / non-aggregate `FunctionCall` / `Subscript`)
// fell through to the implicit-GROUP-BY-key path and the embedded aggregate
// was (mis)evaluated row-wise.
//
// The fix lifts each embedded aggregate into the `Aggregate` node under a
// fresh HIDDEN binding id (reusing the #746/#864 Aggregate→Project hidden-
// column tunnel) and rewrites the outer expression to read those hidden
// columns — composing the EXISTING `AggregateOp` + `ProjectOp` (NO new op).
//
// Golden oracles = openCypher v9 §6.4 + vendored TCK `Return6` [2]/[4]/[5]/
// [9]/[17]/[18]/[19] (pass) vs [14] (`NestedAggregation`, stays rejected).
// These round-trip through the FULL pipeline (parse → bind → type-check →
// cross-substrate → lower → execute) via `QueryEngine::execute` — the same
// path the TCK ratchet uses — and assert EXACT result rows, not "no error".
// =====================================================================

/// Execute `cypher` through the FULL pipeline over an empty substrate
/// (UNWIND-driven queries are fully hermetic + deterministic). Panics on a
/// pipeline error — so a query that hits the pre-#910 `-32005`
/// `NotImplemented` fails the test loudly (the RED-on-revert signal).
fn exec_rows(cypher: &str) -> Vec<Vec<Value>> {
    let cat = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    QueryEngine::new(&cat)
        .execute(cypher, &s)
        .unwrap_or_else(|e| panic!("execute `{cypher}`: {e:?}"))
        .rows
}

/// Execute `cypher`, returning `Ok(rows)` / `Err(debug-rendered error)` so a
/// test can assert a SPECIFIC compile-time rejection (e.g. `count(count(*))`
/// must STAY `NestedAggregation`, not silently become lifted).
fn try_exec(cypher: &str) -> Result<Vec<Vec<Value>>, String> {
    let cat = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    QueryEngine::new(&cat)
        .execute(cypher, &s)
        .map(|r| r.rows)
        .map_err(|e| format!("{e:?}"))
}

/// Sort rows by their first cell's debug rendering (group-emission order is
/// unspecified) for a deterministic grouped-aggregate oracle.
fn sorted_rows(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.sort_by_key(|r| format!("{:?}", r.first()));
    rows
}

// ---- arithmetic over an aggregate (Return6 [2]/[4]/[9]) ----

#[test]
fn nested_count_times_two() {
    // `count(x)` (=5) lifted to the Aggregate; the outer `* 2` runs in the
    // Project over the hidden count column → 10.
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3, 4, 5] AS x RETURN count(x) * 2"),
        vec![vec![Value::Integer(10)]]
    );
}

#[test]
fn nested_sum_plus_one() {
    // sum([10,20,30]) = 60; + 1 = 61.
    assert_eq!(
        exec_rows("UNWIND [10, 20, 30] AS x RETURN sum(x) + 1"),
        vec![vec![Value::Integer(61)]]
    );
}

#[test]
fn nested_count_multiple_divisions() {
    // Return6 [4] shape — `count(n) / a / b` (multiple divisions OVER the
    // single lifted count). count = 7251; 7251/60/60 = 2 (integer division).
    assert_eq!(
        exec_rows("UNWIND range(0, 7250) AS i RETURN count(i) / 60 / 60 AS count"),
        vec![vec![Value::Integer(2)]]
    );
}

// ---- aggregate inside a normal scalar function (Return6 [5]) ----

#[test]
fn nested_size_collect() {
    // `collect(x)` (=[1,2,3]) lifted; outer `size(...)` runs over the hidden
    // collect column → 3. The canonical `size(collect(x))` group-cardinality
    // idiom from the issue.
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3] AS x RETURN size(collect(x))"),
        vec![vec![Value::Integer(3)]]
    );
}

#[test]
fn nested_tostring_count() {
    // `toString(count(x))` → "5" (aggregate inside a type-conversion fn).
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3, 4, 5] AS x RETURN toString(count(x))"),
        vec![vec![Value::String("5".to_string())]]
    );
}

#[test]
fn nested_head_collect() {
    // `head(collect(x))` — aggregate inside a list-scalar fn → first element.
    assert_eq!(
        exec_rows("UNWIND [10, 20, 30] AS x RETURN head(collect(x))"),
        vec![vec![Value::Integer(10)]]
    );
}

#[test]
fn nested_collect_subscript() {
    // `collect(x)[1]` — aggregate inside a subscript operand → 2nd element.
    assert_eq!(
        exec_rows("UNWIND [10, 20, 30] AS x RETURN collect(x)[1]"),
        vec![vec![Value::Integer(20)]]
    );
}

// ---- MULTIPLE aggregates in one expression (issue: `100.0*count(a)/count(b)`) ----

#[test]
fn nested_multiple_aggregates_percentage() {
    // TWO aggregates in one expression, each lifted to its OWN hidden column:
    // count(x) excludes the NULL (=3); count(*) counts all rows (=4).
    // 100.0 * 3 / 4 = 75.0 (float arithmetic). Strong oracle: proves the
    // count(expr)-vs-count(*) distinction survives the two-phase lift AND
    // that multiple embedded aggregates each get a distinct hidden id.
    assert_eq!(
        exec_rows("UNWIND [1, 2, null, 4] AS x RETURN 100.0 * count(x) / count(*)"),
        vec![vec![Value::Float(75.0)]]
    );
}

// ---- constants / parameters alongside the aggregate (Return6 [17]) ----

#[test]
fn nested_aggregate_with_literal_null_propagation() {
    // avg over an EMPTY input is NULL; `1 + null - 1000` propagates to NULL
    // (Return6 [17] shape, without the parameter). Single-row aggregate (no
    // group key) → exactly one NULL row.
    assert_eq!(
        exec_rows("UNWIND [] AS x RETURN 1 + avg(x) - 1000"),
        vec![vec![Value::Null]]
    );
}

// ---- the SAME nesting in WITH (issue: "and the same in WITH") ----

#[test]
fn nested_aggregate_in_with_clause() {
    // `WITH count(x) * 2 AS c` — the nested-aggregate lift must also fire on
    // the WITH projection path (shared `lower_aggregation_clause`).
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3] AS x WITH count(x) * 2 AS c RETURN c"),
        vec![vec![Value::Integer(6)]]
    );
}

#[test]
fn with_collect_then_size_baseline_still_works() {
    // The documented WITH-stage WORKAROUND must keep working unchanged
    // (regression guard): compute the aggregate in WITH, wrap it in RETURN.
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3] AS x WITH collect(x) AS xs RETURN size(xs)"),
        vec![vec![Value::Integer(3)]]
    );
}

// ---- grouping-key REFERENCE inside the aggregating expression (Return6 [18]/[19]) ----

#[test]
fn grouping_key_reference_with_rows() {
    // `RETURN x, x + count(*)` — `x` is the implicit grouping key AND is
    // referenced in the aggregating projection. Grouped by x: x=1 has 2 rows
    // (1 + 2 = 3); x=2 has 1 row (2 + 1 = 3). The fix rewrites the outer-
    // expression `x` reference to the key's precomputed Aggregate column
    // (the empty-graph Return6 [18]/[19] never exercise this with rows; this
    // is the stronger rows-bearing oracle).
    assert_eq!(
        sorted_rows(exec_rows("UNWIND [1, 1, 2] AS x RETURN x, x + count(*)")),
        vec![
            vec![Value::Integer(1), Value::Integer(3)],
            vec![Value::Integer(2), Value::Integer(3)],
        ]
    );
}

// ---- regression: bare + grouped aggregates are UNCHANGED (no over-lift) ----

#[test]
fn bare_aggregate_unchanged() {
    // A bare aggregate still lowers exactly as before (no double-lift).
    assert_eq!(
        exec_rows("UNWIND [1, 2, 3] AS x RETURN count(x)"),
        vec![vec![Value::Integer(3)]]
    );
}

#[test]
fn nested_agg_in_agg_stays_rejected() {
    // `count(count(*))` (aggregate INSIDE an aggregate's argument) is a
    // genuine error — it must STAY rejected at compile time (don't over-lift
    // into a silent success). Return6 [14] `NestedAggregation`.
    let err = try_exec("MATCH (n) RETURN count(count(*))").expect_err("count(count(*)) must error");
    assert!(
        err.contains("NestedAggregation"),
        "expected NestedAggregation, got: {err}"
    );
}

// ---- lowering-shape pins (this file's structural-oracle style) ----

#[test]
fn nested_count_times_two_shape_is_project_over_aggregate() {
    // The two-phase lower composes EXISTING ops: a Project (outer `* 2`) over
    // a single-aggregation Aggregate (the lifted count) — NOT a new operator.
    let plan = lower_ok("MATCH (n:Person) RETURN count(n) * 2");
    let s = shape(&plan);
    assert!(s.contains(&"Project"), "expected Project: {s:?}");
    assert!(s.contains(&"Aggregate"), "expected Aggregate: {s:?}");
    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(
        agg.aggregations.len(),
        1,
        "the embedded count(n) is the sole lifted aggregation"
    );
    assert_eq!(agg.aggregations[0].function, AggregationKind::Count);
    assert!(
        agg.group_by.is_empty(),
        "no non-aggregate projection ⇒ single-row aggregate (empty group_by)"
    );
}

#[test]
fn nested_multiple_aggregates_lift_to_two_specs() {
    // `100.0 * count(x) / count(*)` lifts BOTH aggregates into the SAME
    // Aggregate node (two specs, distinct hidden ids).
    let plan = lower_ok("MATCH (n:Person) RETURN 100.0 * count(n) / count(*)");
    let agg = find_aggregate(&plan).expect("Aggregate present");
    assert_eq!(
        agg.aggregations.len(),
        2,
        "two embedded aggregates ⇒ two aggregation specs"
    );
}
