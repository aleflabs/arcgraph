//! ADR-152 W27-α — MERGE-by-property idempotency.
//!
//! Closes the ADR-151 §"Risks" always-creates production narrowing —
//! at HEAD afff463 MERGE in production always took the create branch
//! because `scan_nodes` returned nodes filtered by label only.
//! Post-ADR-152 §D-4, the match-branch's Scan is wrapped with a
//! property-filter so the second MERGE matches the first.
//!
//! Walks:
//!
//! 1. MERGE (n:User {id: 42}) — first call creates the node.
//! 2. MERGE (n:User {id: 42}) — second call matches the first.
//! 3. MATCH (n:User) RETURN n — observes exactly 1 row (not 2;
//!    MERGE was idempotent).

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
fn merge_by_property_second_call_matches_first() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // First MERGE — create branch.
    let _ = execute(&lower("MERGE (n:User {id: 42})"), &substrate, &ctx);

    // Second MERGE — must take match branch (NOT create branch).
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(&lower("MERGE (n:User {id: 42})"), &substrate, &ctx2);

    // MATCH all User nodes — must be exactly 1.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(&lower("MATCH (n:User) RETURN n"), &substrate, &ctx3);
    assert_eq!(
        rows.len(),
        1,
        "MERGE-by-property is idempotent post-ADR-152 — second MERGE \
         matches first (closes ADR-151 §Risks always-creates narrowing)"
    );
}

#[test]
fn merge_with_different_property_creates_distinct_node() {
    // Sanity-check: MERGE with id=42 then MERGE with id=99 creates
    // 2 distinct nodes (the property predicate discriminates).
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    let _ = execute(&lower("MERGE (n:User {id: 42})"), &substrate, &ctx);
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(&lower("MERGE (n:User {id: 99})"), &substrate, &ctx2);

    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(&lower("MATCH (n:User) RETURN n"), &substrate, &ctx3);
    assert_eq!(
        rows.len(),
        2,
        "MERGE-by-property with distinct ids creates two distinct nodes"
    );
}

#[test]
fn merge_followed_by_match_round_trips_property_bag() {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    let _ = execute(
        &lower(r#"MERGE (n:User {id: 42, name: "Alice"})"#),
        &substrate,
        &ctx,
    );

    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let rows = execute(
        &lower(r#"MATCH (n:User {id: 42}) RETURN n.name"#),
        &substrate,
        &ctx2,
    );
    assert_eq!(
        rows.len(),
        1,
        "MERGE persists the property bag; MATCH-by-property reads it back"
    );
    assert_eq!(
        rows[0][0],
        Value::String("Alice".into()),
        "RETURN n.name reads the MERGE-persisted property"
    );
}
