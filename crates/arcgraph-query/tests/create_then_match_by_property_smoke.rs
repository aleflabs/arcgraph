//! ADR-152 W27-α — CREATE-then-MATCH-by-property smoke test.
//!
//! The audit-identified smoking-gun case at 2026-05-27 post-θ-5:
//!
//! ```text
//! arcql> CREATE (n:User {id: 42, name: "Alice"}) RETURN n;
//! ok: 1 node created
//! arcql> MATCH (n:User {id: 42}) RETURN n;
//! 0 rows                               ← BUG at HEAD afff463
//! ```
//!
//! Post-ADR-152 this round-trip MUST return 1 row carrying the
//! materialized property bag.
//!
//! The test walks the full query-side pipeline twice against the
//! SAME `StubExecutorSubstrate` instance:
//!
//! 1. CREATE (n:User {id: 42, name: "Alice"}) — observes 1 row
//!    emitted; the substrate's `create_state.nodes` carries the new
//!    NodeId; the substrate's `node_properties` sidecar carries the
//!    `id` + `name` bag.
//! 2. MATCH (n:User {id: 42}) RETURN n.name — observes 1 row whose
//!    `n.name` cell is `Value::String("Alice")`.

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

/// LabelId the StubExecutorSubstrate allocates for the first
/// interned label name (per its `next_label` allocator that starts
/// at 1024). The smoke tests pre-bind the catalog via
/// [`StubCatalogProvider::with_label_id`] so the MATCH-lowered Scan
/// emits the SAME LabelId the substrate's `create_node` will
/// assign — closing the catalog↔substrate id-divergence the
/// `with_label_id` doc-comment warns about.
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
fn create_then_match_by_property_round_trip_returns_the_node() {
    // ADR-152 §D-1 + §D-3 + §D-4 — the audit-identified smoking-gun
    // case must round-trip post-ADR-152.
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE (n:User {id: 42, name: "Alice"}) RETURN n.
    let create_plan = lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#);
    let create_rows = execute(&create_plan, &substrate, &ctx);
    assert_eq!(
        create_rows.len(),
        1,
        "CREATE emits exactly one row (openCypher TCK \"1 node created\")"
    );

    // MATCH (n:User {id: 42}) RETURN n.name — the round-trip read.
    // Fresh ExecutionContext so the snapshot LSN handshake re-arms.
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_plan = lower("MATCH (n:User {id: 42}) RETURN n.name");
    let match_rows = execute(&match_plan, &substrate, &ctx2);
    assert_eq!(
        match_rows.len(),
        1,
        "MATCH-by-property returns the persisted node (the audit's \
         smoking-gun case must PASS post-ADR-152)"
    );
    let name_cell = &match_rows[0][0];
    assert_eq!(
        name_cell,
        &Value::String("Alice".into()),
        "MATCH-by-property's RETURN n.name carries the persisted \
         literal"
    );
}

#[test]
fn create_then_match_by_property_no_match_returns_zero_rows() {
    // Negative case: CREATE n with id=42, MATCH n with id=99 → 0 rows.
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let create_plan = lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#);
    let _ = execute(&create_plan, &substrate, &ctx);

    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_plan = lower("MATCH (n:User {id: 99}) RETURN n");
    let match_rows = execute(&match_plan, &substrate, &ctx2);
    assert_eq!(
        match_rows.len(),
        0,
        "MATCH-by-property with non-matching predicate returns 0 rows"
    );
}

#[test]
fn create_then_match_by_property_multi_predicate_all_must_match() {
    // CREATE n with id=42 + name="Alice"; MATCH-by-(id=42, name="Bob") → 0 rows.
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let create_plan = lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n"#);
    let _ = execute(&create_plan, &substrate, &ctx);

    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_plan = lower(r#"MATCH (n:User {id: 42, name: "Bob"}) RETURN n"#);
    let match_rows = execute(&match_plan, &substrate, &ctx2);
    assert_eq!(
        match_rows.len(),
        0,
        "MATCH-by-property multi-predicate requires ALL keys to match \
         (AND-conjunction per ADR-152 §D-4)"
    );

    // Same MATCH with all-matching predicate returns 1 row.
    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_plan2 = lower(r#"MATCH (n:User {id: 42, name: "Alice"}) RETURN n"#);
    let match_rows2 = execute(&match_plan2, &substrate, &ctx3);
    assert_eq!(
        match_rows2.len(),
        1,
        "MATCH-by-property multi-predicate with all-matching values \
         returns the node"
    );
}

#[test]
fn match_without_property_predicate_returns_all_nodes() {
    // Sanity-check: MATCH without property literals returns ALL nodes
    // of the label (i.e. the property filter is OPT-IN per ADR-152
    // §D-4 — empty BoundPropertyMap skips the filter wrap).
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    // Create two distinct nodes — same label, different ids.
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 1, name: "Alice"}) RETURN n"#),
        &substrate,
        &ctx,
    );
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower(r#"CREATE (n:User {id: 2, name: "Bob"}) RETURN n"#),
        &substrate,
        &ctx2,
    );

    let ctx3 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_plan = lower("MATCH (n:User) RETURN n");
    let match_rows = execute(&match_plan, &substrate, &ctx3);
    assert_eq!(
        match_rows.len(),
        2,
        "MATCH without property literals returns ALL label-matching \
         nodes"
    );
}

#[test]
fn return_after_create_carries_materialized_property_bag() {
    // ADR-152 §D-1 — CreateNodeOp emits NodeView with materialized
    // properties so RETURN-after-CREATE shows them.
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let create_plan = lower(r#"CREATE (n:User {id: 42, name: "Alice"}) RETURN n.name"#);
    let rows = execute(&create_plan, &substrate, &ctx);
    assert_eq!(rows.len(), 1, "CREATE … RETURN n.name emits one row");
    assert_eq!(
        rows[0][0],
        Value::String("Alice".into()),
        "RETURN n.name reads the materialized property bag from \
         the CreateNodeOp's emitted NodeView"
    );
}
