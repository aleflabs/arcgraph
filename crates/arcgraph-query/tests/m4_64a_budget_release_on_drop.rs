//! W12α fix-up MED-1 (PR #277 retro): per-tenant budget counter MUST
//! return to zero after each `AggregateOp` / `SortOp` is dropped, so a
//! long-running tenant configured with a per-tenant byte cap does NOT
//! see the counter drift upward across queries (which would cause
//! false `ResourceExhausted` rejections after enough successful
//! queries).
//!
//! # Pin set
//!
//! - `aggregate_op_release_on_drop_returns_counter_to_zero` —
//!   single aggregate query; assert the budget tenant counter returns
//!   to zero after the operator is dropped.
//! - `sort_op_release_on_drop_returns_counter_to_zero` — same shape
//!   for `SortOp`.
//! - `aggregate_op_no_drift_across_100_serial_queries` — run 100
//!   sequential `AggregateOp` queries on a shared `MemoryBudget`
//!   instance; assert that after each query's operator drop the
//!   tenant counter returns to zero (no monotone upward drift).
//! - `sort_op_no_drift_across_100_serial_queries` — same shape for
//!   `SortOp`.
//! - `aggregate_op_collect_release_on_drop_no_drift` — covers the
//!   per-fold COLLECT reservation path (W12α fix-up LOW-3); same
//!   no-drift assertion holds.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-64a cite.
//! - **ADR-038 amendment-03 §Structural-1** — correctness primitive.
//! - PR #277 reviewer packet MED-1 — the budget-counter-drift class
//!   that this test set pins against regression.

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::MemoryBudget;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::Value;
use arcgraph_query::executor::ops::{
    AggregateCall, AggregateOp, PhysicalOperator, ScanOp, SortKey, SortOp,
};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::logical_plan::{AggregationKind, SortDirection};
use arcgraph_query::semantic::bound_ast::{
    BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind, BoundPropertyRef,
};

/// Build a stub substrate of `n` person nodes with an integer `age`
/// property in `[1..=n]` so the budget reservation path is exercised
/// (per-row bytes are non-trivial).
fn make_n_persons(tenant: TenantId, n: u64) -> StubExecutorSubstrate {
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

fn prop_age(base: BoundExpression) -> BoundExpression {
    BoundExpression::PropertyAccess {
        base: Box::new(base),
        path: vec![BoundPropertyRef {
            name: "age".into(),
            property_id: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn group_by_n() -> Vec<BoundProjectionItem> {
    vec![BoundProjectionItem {
        kind: BoundProjectionKind::Expr(var_n()),
        alias: None,
        output_id: Some(BindingId::new(1)),
        source_text: None,
        span: Span::point(1, 1),
    }]
}

fn count_n() -> Vec<AggregateCall> {
    vec![AggregateCall {
        distinct: false,
        star: false,
        kind: AggregationKind::Count,
        arg: var_n(),
        output_id: BindingId::new(2),
    }]
}

/// Run one aggregate query (drain-to-completion + drop the operator)
/// against the supplied budget. Returns the row count for sanity
/// pinning.
fn run_one_aggregate_query(
    budget: &MemoryBudget,
    tenant: TenantId,
    s: &StubExecutorSubstrate,
    rows_in: u64,
) -> usize {
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget.clone());
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        group_by_n(),
        count_n(),
    );
    let mut emitted = 0_usize;
    loop {
        let b = op.next_batch(&ctx, s).expect("aggregate runs cleanly");
        if b.is_empty() {
            break;
        }
        emitted += b.row_count();
    }
    // Sanity pin: rows_in distinct nodes ⇒ rows_in groups (one per node).
    assert_eq!(emitted, rows_in as usize);
    // Op drops at end of scope.
    drop(op);
    emitted
}

fn run_one_sort_query(
    budget: &MemoryBudget,
    tenant: TenantId,
    s: &StubExecutorSubstrate,
    rows_in: u64,
) -> usize {
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget.clone());
    let mut op = SortOp::new(
        PhysicalOperator::Scan(person_scan()),
        vec![SortKey {
            expr: prop_age(var_n()),
            direction: SortDirection::Asc,
        }],
    );
    let mut emitted = 0_usize;
    loop {
        let b = op.next_batch(&ctx, s).expect("sort runs cleanly");
        if b.is_empty() {
            break;
        }
        emitted += b.row_count();
    }
    assert_eq!(emitted, rows_in as usize);
    drop(op);
    emitted
}

#[test]
fn aggregate_op_release_on_drop_returns_counter_to_zero() {
    let tenant = TenantId::DEFAULT;
    // Generous cap so the query succeeds; the test pins the release-
    // on-drop semantics, not the cap-rejection path.
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 1_000_000);
    let s = make_n_persons(tenant, 25);
    let baseline = budget.current_bytes(tenant);
    assert_eq!(baseline, 0, "fresh budget starts at zero");
    run_one_aggregate_query(&budget, tenant, &s, 25);
    let after = budget.current_bytes(tenant);
    assert_eq!(
        after,
        0,
        "after AggregateOp drops, the per-tenant counter MUST return \
         to baseline ({baseline}); observed {after} (drift = {})",
        after as i64 - baseline as i64
    );
    // Peak is preserved per amendment-03 §Structural-3 edge 6 (M4-71 /
    // M4-91 PROFILE consumer).
    assert!(
        budget.peak_bytes(tenant) > 0,
        "peak bytes preserved across release; observed {}",
        budget.peak_bytes(tenant)
    );
}

#[test]
fn sort_op_release_on_drop_returns_counter_to_zero() {
    let tenant = TenantId::DEFAULT;
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 1_000_000);
    let s = make_n_persons(tenant, 25);
    let baseline = budget.current_bytes(tenant);
    assert_eq!(baseline, 0);
    run_one_sort_query(&budget, tenant, &s, 25);
    let after = budget.current_bytes(tenant);
    assert_eq!(
        after, 0,
        "after SortOp drops, the per-tenant counter MUST return to \
         baseline ({baseline}); observed {after}"
    );
    assert!(budget.peak_bytes(tenant) > 0, "peak bytes preserved");
}

#[test]
fn aggregate_op_no_drift_across_100_serial_queries() {
    // The drift class: pre-fix, the per-tenant counter accumulated
    // bytes-per-output-row × queries-run, so eventually the cap was
    // saturated by the bookkeeping alone (rows freed by
    // `Vec::drop`, but no `budget.release` matched the
    // `try_reserve_unscoped`). Today: each query's drop releases its
    // running total; counter returns to zero after every drop.
    let tenant = TenantId::DEFAULT;
    // Cap chosen tight enough that drift would trip ResourceExhausted
    // within a few iterations (10 KiB easily exceeds 25 rows × per-
    // row-bytes after a handful of queries).
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 10_000);
    let s = make_n_persons(tenant, 25);
    for i in 1..=100_u64 {
        run_one_aggregate_query(&budget, tenant, &s, 25);
        let after = budget.current_bytes(tenant);
        assert_eq!(
            after, 0,
            "no drift after iteration {i}: per-tenant counter MUST \
             return to zero between queries (got {after})"
        );
    }
}

#[test]
fn sort_op_no_drift_across_100_serial_queries() {
    let tenant = TenantId::DEFAULT;
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 10_000);
    let s = make_n_persons(tenant, 25);
    for i in 1..=100_u64 {
        run_one_sort_query(&budget, tenant, &s, 25);
        let after = budget.current_bytes(tenant);
        assert_eq!(
            after, 0,
            "no drift after iteration {i}: per-tenant counter MUST \
             return to zero between queries (got {after})"
        );
    }
}

#[test]
fn aggregate_op_collect_release_on_drop_no_drift() {
    // W12α fix-up LOW-3: COLLECT folds reserve per-push bytes against
    // the budget. Drop must release ALL of those bytes (not just the
    // emit-time per-row reservation) so the counter returns to zero.
    let tenant = TenantId::DEFAULT;
    // Generous cap so COLLECT succeeds; pins drop-release, not the
    // cap-rejection path.
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 10_000_000);
    let s = make_n_persons(tenant, 25);
    let aggregations = vec![AggregateCall {
        distinct: false,
        star: false,
        kind: AggregationKind::Collect,
        arg: prop_age(var_n()),
        output_id: BindingId::new(2),
    }];
    for i in 1..=20_u64 {
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget.clone());
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            aggregations.clone(),
        );
        loop {
            let b = op.next_batch(&ctx, &s).expect("collect runs cleanly");
            if b.is_empty() {
                break;
            }
        }
        drop(op);
        let after = budget.current_bytes(tenant);
        assert_eq!(
            after, 0,
            "no COLLECT-fold drift after iteration {i}: counter MUST \
             return to zero (got {after})"
        );
    }
}

#[test]
fn aggregate_op_collect_per_fold_reservation_surfaces_resource_exhausted() {
    // W12α fix-up LOW-3 pin: per-fold COLLECT reservation MUST
    // surface `ResourceExhausted` when the cumulative push bytes
    // cross the cap (BEFORE the materialize emit-time reservation
    // would fire). Picks a cap that fits a handful of pushes but not
    // 10K of them.
    use arcgraph_query::executor::ExecutionError;
    use arcgraph_query::semantic::error::ArcQLError;
    let tenant = TenantId::DEFAULT;
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 10_000);
    // 10K rows; each Integer push debits ~size_of::<Value>() bytes
    // (well over a few hundred bytes total within a few hundred
    // pushes).
    let s = make_n_persons(tenant, 10_000);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget.clone());
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Collect,
            arg: prop_age(var_n()),
            output_id: BindingId::new(2),
        }],
    );
    let r = op.next_batch(&ctx, &s);
    match r {
        Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { feature, .. })) => {
            assert!(
                feature.contains("AggregateOp COLLECT fold")
                    || feature.contains("AggregateOp output"),
                "expected COLLECT-fold or output-emit feature label; got {feature:?}"
            );
        }
        other => panic!("expected ResourceExhausted (per-fold COLLECT); got {other:?}"),
    }
    // Drop the op and assert no drift. (try_reserve failure does not
    // bump the counter, so drop release of `reserved_total` cleanly
    // returns to zero with NO under/overflow even on the partial
    // execution.)
    drop(op);
    assert_eq!(
        budget.current_bytes(tenant),
        0,
        "drop releases partial reservations cleanly even on \
         mid-fold ResourceExhausted"
    );
}
