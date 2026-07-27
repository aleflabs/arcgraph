//! #960 — dynamic `LIMIT $k` / `SKIP $s` execution.
//!
//! The lowering already produces `LogicalDynamicLimit` for non-literal
//! row-window counts. These tests pin the executor's parameter-aware
//! build path: evaluate the count expression once against the per-query
//! parameter bag, then run the same physical Limit/Skip operators used
//! by literal counts.

use arcgraph_query::executor::eval::Parameters;
use arcgraph_query::executor::{ExecutionContext, ExecutionError, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
}

fn lower(query: &str, c: &StubCatalogProvider) -> arcgraph_query::logical_plan::LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

fn run_with_params(query: &str, parameters: Parameters) -> Result<Vec<Vec<Value>>, ExecutionError> {
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let plan = lower(query, &c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition()).with_parameters(parameters);
    materialize::materialize(&plan, &s, &ctx).map(|r| r.rows().to_vec())
}

fn run(query: &str) -> Vec<Vec<Value>> {
    run_with_params(query, Parameters::new()).expect("materialize")
}

fn params(entries: &[(&str, Value)]) -> Parameters {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn rows_of_ints(xs: &[i64]) -> Vec<Vec<Value>> {
    xs.iter().map(|n| vec![Value::Integer(*n)]).collect()
}

#[test]
fn limit_parameter_keeps_first_k_rows() {
    let rows = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k",
        params(&[("k", Value::Integer(2))]),
    )
    .expect("dynamic LIMIT");
    assert_eq!(rows, rows_of_ints(&[1, 2]));
}

#[test]
fn limit_parameter_zero_returns_no_rows() {
    let rows = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k",
        params(&[("k", Value::Integer(0))]),
    )
    .expect("dynamic LIMIT zero");
    assert_eq!(rows, Vec::<Vec<Value>>::new());
}

#[test]
fn skip_parameter_drops_first_s_rows() {
    let rows = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x SKIP $s",
        params(&[("s", Value::Integer(1))]),
    )
    .expect("dynamic SKIP");
    assert_eq!(rows, rows_of_ints(&[2, 3, 4]));
}

#[test]
fn skip_parameter_then_limit_parameter_pages_rows() {
    let rows = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x SKIP $s LIMIT $k",
        params(&[("s", Value::Integer(1)), ("k", Value::Integer(2))]),
    )
    .expect("dynamic SKIP + LIMIT");
    assert_eq!(rows, rows_of_ints(&[2, 3]));
}

#[test]
fn limit_parameter_arithmetic_expression_is_evaluated() {
    let rows = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k + 1",
        params(&[("k", Value::Integer(1))]),
    )
    .expect("dynamic LIMIT expression");
    assert_eq!(rows, rows_of_ints(&[1, 2]));
}

#[test]
fn dynamic_limit_rejects_negative_count() {
    let err = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k",
        params(&[("k", Value::Integer(-1))]),
    )
    .expect_err("negative dynamic LIMIT must error");
    assert!(
        matches!(err, ExecutionError::Eval(message) if message.contains("LIMIT value must be non-negative"))
    );
}

#[test]
fn dynamic_limit_rejects_null_count() {
    let err = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k",
        params(&[("k", Value::Null)]),
    )
    .expect_err("null dynamic LIMIT must error");
    assert!(
        matches!(err, ExecutionError::Eval(message) if message.contains("LIMIT value must not be null"))
    );
}

#[test]
fn dynamic_limit_rejects_float_count_even_when_integral() {
    let err = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $k",
        params(&[("k", Value::Float(2.0))]),
    )
    .expect_err("float dynamic LIMIT must error");
    assert!(
        matches!(err, ExecutionError::Eval(message) if message.contains("LIMIT value must be an integer"))
    );
}

#[test]
fn dynamic_limit_missing_parameter_uses_missing_parameter_error() {
    let err = run_with_params(
        "UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT $missing",
        Parameters::new(),
    )
    .expect_err("missing dynamic LIMIT parameter must error");
    assert_eq!(
        err,
        ExecutionError::MissingParameter {
            name: "missing".into()
        }
    );
}

#[test]
fn literal_limit_and_skip_still_work() {
    let limit_rows = run("UNWIND [1, 2, 3, 4] AS x RETURN x LIMIT 2");
    assert_eq!(limit_rows, rows_of_ints(&[1, 2]));

    let skip_rows = run("UNWIND [1, 2, 3, 4] AS x RETURN x SKIP 1");
    assert_eq!(skip_rows, rows_of_ints(&[2, 3, 4]));
}
