//! #952 regression coverage: EXPLAIN through execute is side-effect-free.

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId};
use arcgraph_query::executor::substrate::HeldTxnHandle;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{ExplainError, QueryEngine};

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["ET", "X", "PT"])
        .with_properties(["id"])
}

fn catalog_for_first_created_label(label: &str) -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_label_id(label, LabelId::new(1024))
        .with_properties(["id"])
}

fn engine<'a>(catalog: &'a StubCatalogProvider) -> QueryEngine<'a, StubCatalogProvider> {
    QueryEngine::new(catalog)
}

fn plan_columns() -> Vec<String> {
    ["operator", "details", "est_cost", "est_rows", "depth"]
        .map(str::to_string)
        .to_vec()
}

fn count_label(
    engine: &QueryEngine<'_, StubCatalogProvider>,
    substrate: &StubExecutorSubstrate,
    label: &str,
) -> i64 {
    let query = format!("MATCH (n:{label}) RETURN count(n)");
    let result = engine.execute(&query, substrate).expect("count query");
    match result.rows.as_slice() {
        [row] => match row.as_slice() {
            [Value::Integer(n)] => *n,
            other => panic!("count row must be one integer, got {other:?}"),
        },
        other => panic!("count query must return one row, got {other:?}"),
    }
}

fn operators(result: &arcgraph_query::MaterializedResult) -> Vec<&str> {
    result
        .rows
        .iter()
        .map(|row| match row.first() {
            Some(Value::String(op)) => op.as_str(),
            other => panic!("plan row must start with operator string, got {other:?}"),
        })
        .collect()
}

#[derive(Debug)]
struct FakeHeldTxn;

impl HeldTxnHandle for FakeHeldTxn {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn snapshot_lsn(&self) -> Lsn {
        Lsn::new(7)
    }
}

#[test]
fn execute_explain_create_returns_plan_rows_and_does_not_create_node() {
    let catalog = catalog_for_first_created_label("ET");
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new();

    let result = engine
        .execute("EXPLAIN CREATE (n:ET {id: 1}) RETURN n.id", &substrate)
        .expect("execute EXPLAIN CREATE");

    assert_eq!(result.columns, plan_columns());
    assert!(!result.rows.is_empty(), "EXPLAIN must return plan rows");
    let ops = operators(&result);
    assert!(
        ops.contains(&"CreateNode"),
        "plan must include CreateNode operator, got {ops:?}"
    );
    assert_eq!(
        count_label(&engine, &substrate, "ET"),
        0,
        "EXPLAIN CREATE must not mutate"
    );
}

#[test]
fn execute_explain_read_returns_plan_rows_not_query_rows() {
    let catalog = catalog();
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new();

    let result = engine
        .execute("EXPLAIN MATCH (n) RETURN count(n)", &substrate)
        .expect("execute EXPLAIN read");

    assert_eq!(result.columns, plan_columns());
    assert_ne!(result.columns, vec!["count(n)".to_string()]);
    assert!(
        !result.rows.is_empty(),
        "EXPLAIN read must return plan rows"
    );
}

#[test]
fn execute_explain_detach_delete_returns_plan_rows_and_keeps_node() {
    let catalog = catalog();
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(2))),
    );

    let result = engine
        .execute("EXPLAIN MATCH (n:X) DETACH DELETE n", &substrate)
        .expect("execute EXPLAIN DETACH DELETE");

    assert_eq!(result.columns, plan_columns());
    assert!(
        operators(&result).contains(&"Delete"),
        "plan must include Delete operator"
    );
    assert_eq!(
        count_label(&engine, &substrate, "X"),
        1,
        "EXPLAIN DETACH DELETE must not delete"
    );
}

#[test]
fn profile_create_still_executes() {
    let catalog = catalog_for_first_created_label("PT");
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new();

    let (_plan, metrics) = engine
        .profile("PROFILE CREATE (n:PT {id: 1}) RETURN n", &substrate)
        .expect("PROFILE CREATE");

    assert_eq!(metrics.rows_emitted, 1);
    assert_eq!(
        count_label(&engine, &substrate, "PT"),
        1,
        "PROFILE CREATE must keep execute semantics"
    );
}

#[test]
fn explicit_txn_explain_create_returns_plan_rows_and_does_not_create_node() {
    let catalog = catalog_for_first_created_label("ET");
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new();
    let held: Box<dyn HeldTxnHandle> = Box::new(FakeHeldTxn);

    let (result, _held) = engine.execute_in_txn(
        "EXPLAIN CREATE (n:ET {id: 1}) RETURN n.id",
        &substrate,
        held,
        std::time::Duration::from_secs(30),
    );
    let result = result.expect("execute_in_txn EXPLAIN CREATE");

    assert_eq!(result.columns, plan_columns());
    assert!(
        operators(&result).contains(&"CreateNode"),
        "plan must include CreateNode operator"
    );
    assert_eq!(
        count_label(&engine, &substrate, "ET"),
        0,
        "txn EXPLAIN CREATE must not mutate"
    );
}

#[test]
fn internal_execute_multi_rejects_explain_instead_of_stripping_and_executing() {
    let catalog = catalog_for_first_created_label("ET");
    let engine = engine(&catalog);
    let substrate = StubExecutorSubstrate::new();

    let err = engine
        .execute_multi(
            "EXPLAIN CREATE (n:ET {id: 1}); MATCH (n:ET) RETURN n",
            &substrate,
        )
        .expect_err("multi EXPLAIN must fail closed");

    assert!(
        matches!(err, ExplainError::ArcQL(_)),
        "expected ArcQL rejection, got {err:?}"
    );
    assert_eq!(
        count_label(&engine, &substrate, "ET"),
        0,
        "rejected multi EXPLAIN must not mutate"
    );
}
