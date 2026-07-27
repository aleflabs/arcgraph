//! M4-64a per-tenant memory-budget integration tests per ADR-038
//! amendment-02 §M4.f + amendment-03 §Structural-1.
//!
//! # Pin set (per amendment-03 §Structural-1 M4-64a row)
//!
//! 1. `multi_tenant_memory_isolation` — two tenants with distinct caps;
//!    one tenant's budget exhaustion does NOT affect the other's
//!    available bytes.
//! 2. `budget_exceeded_surfaces_resource_exhausted` — a configured
//!    cap below the operator's working set surfaces
//!    `ArcQLError::ResourceExhausted` rather than OOM.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-64a cite.
//! - **ADR-038 amendment-03 §Structural-1** — split out from M4-64
//!   bundled SIMD; correctness primitive.

use arcgraph_core::Lsn;
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ExecutionError;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::ops::{AggregateCall, AggregateOp, PhysicalOperator, ScanOp};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, MemoryBudget, Value};
use arcgraph_query::logical_plan::AggregationKind;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};
use arcgraph_query::semantic::error::ArcQLError;

fn person_scan() -> ScanOp {
    ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
}

fn var_n() -> BoundExpression {
    BoundExpression::VariableRef {
        name: "n".into(),
        binding_id: BindingId::new(0),
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn make_persons(n: u64, tenant: TenantId) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }
    s
}

#[test]
fn multi_tenant_memory_isolation() {
    // amendment-03 §Structural-1: tenants are isolated against each
    // other's budget exhaustion. Budget A gets a tight cap (will
    // reject); Budget B gets a generous cap (admits).
    //
    // The two ExecutionContexts share the same MemoryBudget instance
    // — that's the M5-12 forward-binding shape (one process-wide
    // budget configured per-tenant). Tenant A's exhaustion does NOT
    // bleed into Tenant B's accounting.
    let tenant_a = TenantId::DEFAULT;
    let tenant_b = TenantId::new(42);
    let mut s = StubExecutorSubstrate::new();
    // Tenant A has 100 person rows; tenant B has 5.
    for i in 1..=100_u64 {
        s = s.with_node(
            tenant_a,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }
    for i in 1..=5_u64 {
        s = s.with_node(
            tenant_b,
            NodeView::new(NodeId::new(1000 + i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }
    // Shared budget instance. Tenant A: 256-byte cap (well below
    // ~100 × 24-byte rows = 2400 bytes). Tenant B: 100KiB cap (well
    // above 5 rows).
    let budget = MemoryBudget::new();
    budget.set_per_tenant_cap(tenant_a, 256);
    budget.set_per_tenant_cap(tenant_b, 100_000);

    // Tenant A's aggregate hits the cap.
    let ctx_a = ExecutionContext::new(tenant_a, PartitionId::ZERO).with_budget(budget.clone());
    let mut op_a = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let r_a = op_a.next_batch(&ctx_a, &s);
    // Either the aggregate succeeds (because count-only doesn't
    // require buffering rows), OR it fails with ResourceExhausted —
    // the M4-63 single-row aggregate stores ONE output row, so it
    // depends on whether estimate_row_bytes(single row) > cap.
    // For a tighter test, run the aggregate with GROUP BY n.id which
    // produces one row per group; that creates 100 output rows and
    // WILL exceed the 256-byte cap.
    drop(r_a);
    let mut op_a2 = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        vec![arcgraph_query::semantic::bound_ast::BoundProjectionItem {
            kind: arcgraph_query::semantic::bound_ast::BoundProjectionKind::Expr(
                BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: Span::point(1, 1),
                    type_info: None,
                },
            ),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        }],
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let r_a2 = op_a2.next_batch(&ctx_a, &s);
    // 100 distinct group rows × ~hundreds-of-bytes-per-row ≫ 256 cap.
    assert!(matches!(
        r_a2,
        Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { .. }))
    ));

    // Tenant B's aggregate (with the SAME budget instance) succeeds —
    // 5 rows × ~hundreds-of-bytes-per-row ≪ 100KiB cap.
    let ctx_b = ExecutionContext::new(tenant_b, PartitionId::ZERO).with_budget(budget.clone());
    let mut op_b = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        vec![arcgraph_query::semantic::bound_ast::BoundProjectionItem {
            kind: arcgraph_query::semantic::bound_ast::BoundProjectionKind::Expr(
                BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: Span::point(1, 1),
                    type_info: None,
                },
            ),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        }],
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let r_b = op_b.next_batch(&ctx_b, &s);
    assert!(
        r_b.is_ok(),
        "tenant B's budget is unaffected by tenant A's exhaustion: {r_b:?}"
    );
    let b = r_b.unwrap();
    assert_eq!(b.row_count(), 5);

    // Per-tenant accounting separation pin: tenant B's current
    // bytes count is independent of tenant A's exhaustion.
    let bytes_b = budget.current_bytes(tenant_b);
    assert!(bytes_b > 0, "tenant B's accounting was bumped: {bytes_b}");
}

#[test]
fn budget_exceeded_surfaces_resource_exhausted() {
    // Configure a tenant-scoped cap below the working set; verify
    // that the operator surfaces ArcQLError::ResourceExhausted via
    // ExecutionError::Plan rather than OOMing or returning Eval.
    let tenant = TenantId::DEFAULT;
    let s = make_persons(50, tenant);
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 64); // 64 bytes — far too small
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        vec![arcgraph_query::semantic::bound_ast::BoundProjectionItem {
            kind: arcgraph_query::semantic::bound_ast::BoundProjectionKind::Expr(
                BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: Span::point(1, 1),
                    type_info: None,
                },
            ),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        }],
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let r = op.next_batch(&ctx, &s);
    match r {
        Err(ExecutionError::Plan(ArcQLError::ResourceExhausted {
            feature, cap_bytes, ..
        })) => {
            assert!(feature.contains("Aggregate") || feature.contains("Spillover"));
            assert_eq!(cap_bytes, 64);
        }
        other => panic!("expected ResourceExhausted; got {other:?}"),
    }
}
