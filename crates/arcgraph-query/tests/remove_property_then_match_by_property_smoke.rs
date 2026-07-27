//! ADR-152 W27-α — REMOVE property + MATCH-by-(removed property)
//! returns 0 rows.
//!
//! Walks the full query-side pipeline:
//!
//! 1. CREATE (n:User {id: 42, name: "Alice"}).
//! 2. MATCH (n:User {id: 42}) REMOVE n.name.
//! 3. MATCH (n:User {name: "Alice"}) RETURN n — observes 0 rows
//!    (the removed property no longer matches).
//! 4. MATCH (n:User {id: 42}) RETURN n.name — observes 1 row but
//!    `n.name` cell is `Value::Null` (the property is gone).

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn execute(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Vec<Vec<Value>> {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op.next_batch(ctx, substrate).expect("batch OK");
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    out
}

#[test]
fn remove_property_then_match_by_removed_returns_zero_rows() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE.
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#),
        &substrate,
        &ctx,
    );

    // REMOVE n.name.
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower("MATCH (n:User {id: 42}) REMOVE n.name"),
        &substrate,
        &ctx2,
    );

    // MATCH-by-(removed-property) returns 0 rows.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(
        &lower(r#"MATCH (n:User {name: "Alice"}) RETURN n"#),
        &substrate,
        &ctx3,
    );
    assert_eq!(
        rows.len(),
        0,
        "post-REMOVE, MATCH-by-(removed property value) returns 0 rows"
    );
}

#[test]
fn remove_property_other_properties_preserved() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE with id + name.
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#),
        &substrate,
        &ctx,
    );

    // REMOVE n.name (id stays).
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower("MATCH (n:User {id: 42}) REMOVE n.name"),
        &substrate,
        &ctx2,
    );

    // id still matches.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(
        &lower("MATCH (n:User {id: 42}) RETURN n.name, n.id"),
        &substrate,
        &ctx3,
    );
    assert_eq!(
        rows.len(),
        1,
        "post-REMOVE n.name, n.id still matches the original value"
    );
    assert!(
        rows[0][0].is_null(),
        "post-REMOVE, the removed `n.name` projects as Value::Null \
         per the PropertyAccess fallback in eval.rs"
    );
    assert_eq!(
        rows[0][1],
        Value::Integer(42),
        "post-REMOVE n.name, the preserved n.id property reads back"
    );
}
