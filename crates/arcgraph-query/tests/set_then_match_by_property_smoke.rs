//! ADR-152 W27-α — SET property + MATCH-by-(set property) round-trip.
//!
//! Walks the full query-side pipeline:
//!
//! 1. CREATE (n:User {id: 42}) — observes 1 row.
//! 2. MATCH (n:User {id: 42}) SET n.name = "Alice" — observes 0 rows
//!    (SET is terminal at Phase 4 per ADR-150 §"Forward-deferred").
//! 3. MATCH (n:User {name: "Alice"}) RETURN n.id — observes 1 row
//!    whose `n.id` is `Value::Integer(42)`.
//!
//! Closes the SET → MATCH-by-(post-SET-property) gap per ADR-152
//! §D-2 + §D-4.

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
fn set_then_match_by_property_round_trip() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE (n:User {id: 42}).
    let _ = execute(
        &lower("CREATE (n:User {id: 42}) RETURN n"),
        &substrate,
        &ctx,
    );

    // SET name="Alice" on the matching node.
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower(r#"MATCH (n:User {id: 42}) SET n.name = "Alice""#),
        &substrate,
        &ctx2,
    );

    // MATCH-by-(post-SET-property) returns the node + emits its id.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(
        &lower(r#"MATCH (n:User {name: "Alice"}) RETURN n.id"#),
        &substrate,
        &ctx3,
    );
    assert_eq!(
        rows.len(),
        1,
        "MATCH-by-(post-SET-name) returns the SET-touched node \
         (ADR-152 §D-2 + §D-4)"
    );
    assert_eq!(
        rows[0][0],
        Value::Integer(42),
        "MATCH-by-(post-SET-property) reads back the CREATE-time `id` \
         + the SET-applied `name` round-trip"
    );
}

#[test]
fn set_overwrites_existing_property() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE with id=42, name="OldName".
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 42, name: "OldName"}) RETURN n"#),
        &substrate,
        &ctx,
    );

    // SET name="NewName" — overwrite the existing property.
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower(r#"MATCH (n:User {id: 42}) SET n.name = "NewName""#),
        &substrate,
        &ctx2,
    );

    // Old name no longer matches.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let old_rows = execute(
        &lower(r#"MATCH (n:User {name: "OldName"}) RETURN n"#),
        &substrate,
        &ctx3,
    );
    assert_eq!(
        old_rows.len(),
        0,
        "post-SET, the pre-SET property value no longer matches"
    );

    // New name matches.
    let ctx4 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let new_rows = execute(
        &lower(r#"MATCH (n:User {name: "NewName"}) RETURN n"#),
        &substrate,
        &ctx4,
    );
    assert_eq!(
        new_rows.len(),
        1,
        "post-SET, the new property value matches"
    );
}

#[test]
fn return_after_match_carries_property_bag() {
    // ADR-152 §D-3 — scan_nodes populates NodeView.properties from
    // the persisted bag; RETURN-after-MATCH projects to a property
    // access on the scanned node.
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#),
        &substrate,
        &ctx,
    );

    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(
        &lower("MATCH (n:User {id: 42}) RETURN n.name, n.id"),
        &substrate,
        &ctx2,
    );
    assert_eq!(rows.len(), 1, "single match row");
    assert_eq!(
        rows[0][0],
        Value::String("Alice".into()),
        "RETURN n.name reads from the scanned NodeView.properties bag"
    );
    assert_eq!(
        rows[0][1],
        Value::Integer(42),
        "RETURN n.id reads from the scanned NodeView.properties bag"
    );
}
