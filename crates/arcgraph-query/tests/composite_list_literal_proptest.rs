//! ADR-152-amendment-02 (W28) — composite `List`-literal proptest.
//!
//! Oracle #7: random `List`-of-scalars property values (including nested
//! lists + heterogeneous elements) round-trip EXACTLY through the full
//! query pipeline — parse → bind → type-check → lower → execute → stub
//! scan — and back as `n.v`. The generated list is rendered to ArcQL
//! text with **internal whitespace** (`[a, b, c]`), so each iteration
//! also exercises the §D-5 `list_literal` grammar whitespace fix in the
//! compound-atomic property-bag position.
//!
//! Excludes non-finite floats and `u64 > i64` per the inherited
//! `Value::to_json_value` lossy edges (amendment-02 §D-4); floats are
//! covered separately by the MCP-crate JSON-property-bag round-trip
//! proptest (`mcp_composite_list_e2e.rs`) which constructs `Value`s
//! directly (no ArcQL float-rendering ambiguity).

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

use proptest::prelude::*;

const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower(query: &str) -> Result<LogicalPlan, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    match &stmt {
        Statement::Read(_) => {}
        other => return Err(format!("not Read: {other:?}")),
    }
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("type-check: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat).map_err(|e| format!("cross: {e:?}"))?;
    LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))
}

fn execute(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Result<Vec<Vec<Value>>, String> {
    let mut op = Pipeline::build(plan).map_err(|e| format!("build: {e:?}"))?;
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op
            .next_batch(ctx, substrate)
            .map_err(|e| format!("batch: {e:?}"))?;
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    Ok(out)
}

/// Render a (scalar / nested-list) `Value` to ArcQL literal text.
/// Strings are `[a-z]{1,8}` so no escaping is required; lists render
/// with `", "` separators (internal whitespace → §D-5 exercise).
fn render(v: &Value) -> String {
    match v {
        Value::Boolean(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Integer(n) => n.to_string(),
        Value::String(s) => format!("\"{s}\""),
        Value::List(xs) => {
            let inner = xs.iter().map(render).collect::<Vec<_>>().join(", ");
            format!("[{inner}]")
        }
        other => unreachable!("proptest generates only bool/int/string/list; got {other:?}"),
    }
}

/// A scalar element (no `Null`/`Float` — see module doc) or a nested
/// list thereof, bounded depth 3.
///
/// Integers are **non-negative** (`0..=i64::MAX`): a negative numeric
/// literal parses as a `UnaryOp { Neg, Literal }` expression — NOT a
/// `Literal` — so it is (correctly) rejected as a non-literal list
/// element, CONSISTENT with the pre-existing rejection of a negative
/// SCALAR property value (`CREATE (n {x: -1})` fails type-check with
/// `CreatePropertyValueNotLiteral`). Negative numeric literals in
/// property values are a pre-existing parser/AST gap (affecting scalars
/// too), out of this `List`-of-LITERALS slice's scope; the
/// `composite_list_literal_smoke.rs::negative_number_in_list_rejected`
/// test pins that boundary explicitly.
fn element_strategy() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(Value::Boolean),
        (0i64..=i64::MAX).prop_map(Value::Integer),
        "[a-z]{1,8}".prop_map(Value::String),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        proptest::collection::vec(inner, 0..4).prop_map(Value::List)
    })
}

fn list_value_strategy() -> impl Strategy<Value = Value> {
    proptest::collection::vec(element_strategy(), 0..5).prop_map(Value::List)
}

proptest! {
    /// A random `List`-of-scalars property value round-trips EXACTLY
    /// through CREATE → MATCH … RETURN n.v.
    #[test]
    fn list_property_round_trips_through_full_pipeline(list in list_value_strategy()) {
        let rendered = render(&list);
        let create_q = format!("CREATE (n:User {{v: {rendered}}}) RETURN n");

        let substrate = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let create_plan = lower(&create_q)
            .map_err(|e| TestCaseError::fail(format!("lower CREATE failed for `{create_q}`: {e}")))?;
        execute(&create_plan, &substrate, &ctx)
            .map_err(|e| TestCaseError::fail(format!("execute CREATE failed: {e}")))?;

        let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let match_plan = lower("MATCH (n:User) RETURN n.v")
            .map_err(|e| TestCaseError::fail(format!("lower MATCH failed: {e}")))?;
        let rows = execute(&match_plan, &substrate, &ctx2)
            .map_err(|e| TestCaseError::fail(format!("execute MATCH failed: {e}")))?;

        prop_assert_eq!(rows.len(), 1, "exactly one node round-trips");
        prop_assert_eq!(&rows[0][0], &list, "n.v round-trips to the generated list EXACTLY");
    }
}
