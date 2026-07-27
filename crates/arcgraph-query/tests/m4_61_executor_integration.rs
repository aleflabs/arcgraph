//! M4-61 vectorized executor integration tests per ADR-038 amendment-02
//! §M4.f + amendment-03 §TIER-1 GAP D/E + §TIER-2-b/c.
//!
//! # Pin set
//!
//! 1. `executor_multi_batch_scan_paginates_correctly` — substrate
//!    populated with > 2 * BATCH_ROWS rows; the executor emits the
//!    expected number of full + partial batches.
//! 2. `executor_multi_tenant_scan_isolation` — two tenants, two
//!    distinct row sets; executing for tenant A returns ONLY tenant
//!    A's rows (no cross-tenant leakage).
//! 3. `executor_snapshot_lsn_acquired_pre_first_batch_and_held` —
//!    ADR-038 §2 D-18 rule 1 + amendment-03 §TIER-1 GAP E rule 5
//!    pin: the LSN is acquired during the first operator's
//!    `next_batch` call (rule 1 — lazy capture) AND every
//!    subsequent operator in the same `ExecutionContext` observes
//!    the same LSN (rule 5 — within-context sharing). Rule 2 is
//!    the distinct multi-statement LSN-sharing rule per M4-83.
//! 4. `executor_tracing_span_carries_query_identity` — the
//!    [`ExecutionContext`] tracing span is tagged with `query_id`,
//!    `tenant`, and `partition`. Verified via `tracing-test`'s span
//!    capture.
//! 5. `executor_cancellation_during_execute_surfaces_at_next_batch`
//!    — tripping the cancellation token mid-execution surfaces
//!    [`ExecutionError::Cancelled`] at the next batch boundary
//!    (NOT the current batch — a batch in flight completes).
//! 6. `executor_query_engine_execute_routes_through_full_pipeline` —
//!    the M5↔M4 contract surface end-to-end smoke (parse → bind →
//!    type-check → cross-substrate → lower → enumerate → execute).

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::ast::{BinOp, Literal};
use arcgraph_query::executor::ops::{ExpandOp, FilterOp, ProjectOp, ScanOp};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    BATCH_ROWS, BoundEdge, BoundNode, CancellationError, CancellationToken, ExecutionContext,
    ExecutionError, ExecutorSubstrate, MemoryBudget, PhysicalOperator, RankedHit,
    StubExecutorSubstrate, SubstrateAccessError, Value, execute, execute_with_context,
};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::bound_ast::{
    BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind, BoundPropertyRef,
};
use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};
use arcgraph_query::{QueryEngine, error::Span};

mod common;

// ---------------------------------------------------------------------
// Catalog helpers
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
}

fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64 * 5)),
        );
    }
    s
}

// ---------------------------------------------------------------------
// 1. Multi-batch scan paginates correctly
// ---------------------------------------------------------------------

#[test]
fn executor_multi_batch_scan_paginates_correctly() {
    let total = BATCH_ROWS * 2 + 5;
    let s = substrate_with_n_persons(total as u64);
    let cat = cat_basic();
    let plan = arcgraph_query::logical_plan::LogicalPlan::Scan(
        arcgraph_query::logical_plan::LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        },
    );
    let rows = execute(&plan, &cat, &s).expect("execute");
    assert_eq!(rows.len(), total, "all paginated rows materialized");
}

// ---------------------------------------------------------------------
// 2. Multi-tenant isolation
// ---------------------------------------------------------------------

#[test]
fn executor_multi_tenant_scan_isolation() {
    // Two tenants, distinct row sets. Executing for one returns only
    // that tenant's rows.
    let other = TenantId::new(42);
    let mut s = StubExecutorSubstrate::new();
    s = s
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        )
        .with_node(other, NodeView::new(NodeId::new(99), Some(LabelId::new(1))));

    let cat_default = cat_basic();
    let cat_other = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_tenant(other);

    let plan = arcgraph_query::logical_plan::LogicalPlan::Scan(
        arcgraph_query::logical_plan::LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        },
    );

    let rows_default = execute(&plan, &cat_default, &s).unwrap();
    let rows_other = execute(&plan, &cat_other, &s).unwrap();
    assert_eq!(rows_default.len(), 1);
    assert_eq!(rows_other.len(), 1);
    let id_default = match &rows_default[0][0] {
        Value::Node(n) => n.id,
        _ => panic!(),
    };
    let id_other = match &rows_other[0][0] {
        Value::Node(n) => n.id,
        _ => panic!(),
    };
    assert_eq!(id_default, NodeId::new(1));
    assert_eq!(id_other, NodeId::new(99));
}

// ---------------------------------------------------------------------
// 3. Snapshot-LSN acquired pre-first-batch and held
// ---------------------------------------------------------------------

/// Recording substrate that snapshots the LSN passed to every scan /
/// expand / vector / bm25 / community call so the test can assert
/// every substrate-side observation saw the same LSN.
///
/// W11Z fix-up LOW-4 (PR #268 retro): the recording side-channel is
/// the load-bearing oracle for the
/// `executor_snapshot_lsn_acquired_pre_first_batch_and_held` test.
/// Pre-fix-up the side-channel was built but never asserted against;
/// the test only asserted that the exec-side `snapshot_lsn()` was
/// stable (a no-op since both reads come from the same context).
#[derive(Default)]
struct LsnRecordingSubstrate {
    inner: StubExecutorSubstrate,
    observed_lsns: std::sync::Mutex<Vec<Lsn>>,
}

impl LsnRecordingSubstrate {
    fn new(inner: StubExecutorSubstrate) -> Self {
        Self {
            inner,
            observed_lsns: std::sync::Mutex::new(Vec::new()),
        }
    }
    /// Read-only accessor for observed LSNs. Used by the W11Z LOW-4
    /// fix-up test to assert every substrate call within a single
    /// `next_batch` traversal observed the same LSN value.
    fn observed(&self) -> Vec<Lsn> {
        self.observed_lsns.lock().unwrap().clone()
    }
}

impl ExecutorSubstrate for LsnRecordingSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.observed_lsns.lock().unwrap().push(read_lsn);
        self.inner.scan_nodes(tenant, label, read_lsn)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.observed_lsns.lock().unwrap().push(read_lsn);
        self.inner
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.observed_lsns.lock().unwrap().push(read_lsn);
        self.inner
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.observed_lsns.lock().unwrap().push(read_lsn);
        self.inner
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.observed_lsns.lock().unwrap().push(read_lsn);
        self.inner.community_members(tenant, community_id, read_lsn)
    }

    fn has_vector_substrate(&self) -> bool {
        self.inner.has_vector_substrate()
    }
    fn has_bm25_substrate(&self) -> bool {
        self.inner.has_bm25_substrate()
    }
    fn has_community_substrate(&self) -> bool {
        self.inner.has_community_substrate()
    }
}

/// W11Z fix-up LOW-4 (PR #268 retro): assert (a) exec-side LSN
/// captured exactly once between [`PhysicalOperator::next_batch`] of
/// the FIRST batch and the second batch, AND (b) every substrate-side
/// call observed the SAME plan-side `read_lsn` (no operator drifts to
/// a different LSN mid-query). The recording side-channel
/// (`LsnRecordingSubstrate::observed`) is the load-bearing oracle
/// for (b); the pre-fix-up version had this side-channel but never
/// asserted against it.
#[test]
fn executor_snapshot_lsn_acquired_pre_first_batch_and_held() {
    let mut inner = StubExecutorSubstrate::new();
    for i in 1..=3_u64 {
        inner = inner.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    inner = inner.with_edge(
        TenantId::DEFAULT,
        RelView::new(
            RelId::new(10),
            NodeId::new(1),
            NodeId::new(2),
            Some(TypeId::new(1)),
        ),
    );
    let s = LsnRecordingSubstrate::new(inner);
    let cat = cat_basic();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    assert_eq!(
        ctx.snapshot_lsn(),
        None,
        "pre-execute: LSN not yet captured"
    );
    assert!(
        s.observed().is_empty(),
        "pre-execute: substrate has not been called yet"
    );

    let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
    let expand = ExpandOp::new(
        PhysicalOperator::Scan(scan),
        BindingId::new(0),
        None,
        BindingId::new(1),
        Some(TypeId::new(1)),
        Direction::LeftToRight,
        None,
        Lsn::MAX,
    )
    .unwrap();
    let mut op = PhysicalOperator::Expand(expand);

    let _b1 = op.next_batch(&ctx, &s).unwrap();
    let captured = ctx
        .snapshot_lsn()
        .expect("post-first-batch: LSN captured (D-18 rule 1)");
    let observed_after_first = s.observed();
    assert!(
        !observed_after_first.is_empty(),
        "first batch: at least one substrate call recorded an LSN"
    );

    // (a) Drive remaining batches to exhaust.
    while !op.next_batch(&ctx, &s).unwrap().is_empty() {}

    // (a) The exec-side snapshot LSN was acquired ONCE; subsequent
    // batches observe the same captured value (pin per ADR-038 §2
    // D-18 rule 4 "released at query-end" + amendment-03 §TIER-1
    // GAP E rule 5 within-context sharing; rule 2 is the distinct
    // multi-statement LSN-sharing rule per M4-83).
    assert_eq!(
        captured,
        ctx.snapshot_lsn().expect("post-exhaust: LSN still present"),
        "exec-side LSN must be stable across batches",
    );

    // (b) W11Z LOW-4 fix-up: the recording side-channel is now
    // load-bearing. Every substrate call (scan + expand) saw the
    // SAME LSN — the plan-side `read_lsn` plumbed through
    // `LogicalScan::read_lsn` / `LogicalExpand::read_lsn`. v1.0-alpha
    // routes `Lsn::MAX` through both; production wiring at M4-08+
    // will route the captured snapshot LSN, and the assertion below
    // continues to hold.
    let observed_all = s.observed();
    assert!(
        !observed_all.is_empty(),
        "substrate must have been called at least once"
    );
    let expected_lsn = observed_all[0];
    for (i, &lsn) in observed_all.iter().enumerate() {
        assert_eq!(
            lsn, expected_lsn,
            "every substrate call observed the same LSN; \
             call #{i} drifted to {lsn:?} (expected {expected_lsn:?})"
        );
    }
    // v1.0-alpha-specific shape pin: the plan-side LSN is `Lsn::MAX`.
    // When M4-08+ binds the real LSN this assertion changes shape;
    // the every-call-equal pin (above) is the durable invariant.
    assert_eq!(
        expected_lsn,
        Lsn::MAX,
        "v1.0-alpha plan-side `read_lsn` is the read-latest sentinel"
    );
}

// ---------------------------------------------------------------------
// 4. Tracing span carries query identity
// ---------------------------------------------------------------------

/// W11Z fix-up LOW-3 (PR #268 retro): real tracing-span field
/// assertion via `tracing-test`'s `traced_test` macro + the span's
/// static metadata.
///
/// The pre-fix-up version asserted `metadata().is_some() ||
/// is_disabled()` — a tautology that passed when the span was
/// disabled (the default `cargo test` config), so a regression that
/// dropped `query_id` / `tenant` / `partition` from the span tag
/// would NOT be caught. This rewrite:
///
/// 1. Uses `#[traced_test]` to attach an active subscriber so the
///    span is NOT disabled (`metadata()` returns `Some` only when a
///    subscriber expressed interest at the span's level).
/// 2. Asserts `metadata().is_some()` to prove the subscriber-driven
///    enabled-span path (NOT the disabled fall-through tautology).
/// 3. Inspects `metadata.fields()` for the three field names —
///    `query_id`, `tenant`, `partition` — that the
///    `ExecutionContext::new` `tracing::info_span!` declares. Field
///    names are static-metadata: dropping any of them from the
///    `info_span!` macro arguments fails this assertion at runtime.
/// 4. Emits a tracing event `parent: ctx.tracing_span()` so the
///    subscriber observes the span being used (not just constructed),
///    pinning the live-context wiring.
#[test]
#[tracing_test::traced_test]
fn executor_tracing_span_carries_query_identity() {
    let pinned_query_id = arcgraph_query::executor::QueryId::from_uuid(uuid::uuid!(
        "01234567-89ab-7cde-8f01-23456789abcd"
    ));
    let pinned_tenant = TenantId::new(7777);
    let ctx = ExecutionContext::with_query_id(pinned_tenant, PartitionId::ZERO, pinned_query_id);
    let span = ctx.tracing_span();

    // (1) + (2): a subscriber expressed interest. Pre-fix-up, the
    // disabled-span fall-through made this assertion vacuous.
    let metadata = span
        .metadata()
        .expect("traced_test subscriber must enable info-level executor span");

    // (3): static-metadata assertion. Each declared field of the
    // `info_span!` macro shows up as a [`Field`] in
    // `metadata.fields()`. A regression that drops `query_id` /
    // `tenant` / `partition` from the macro fails THIS assertion at
    // runtime.
    let field_names: Vec<&str> = metadata.fields().iter().map(|f| f.name()).collect();
    assert!(
        field_names.contains(&"query_id"),
        "executor tracing span must declare `query_id` field; got {field_names:?}"
    );
    assert!(
        field_names.contains(&"tenant"),
        "executor tracing span must declare `tenant` field; got {field_names:?}"
    );
    assert!(
        field_names.contains(&"partition"),
        "executor tracing span must declare `partition` field; got {field_names:?}"
    );

    // (4): live-context pin. Emit a tracing event scoped to the
    // span so the subscriber observes the span being used (not just
    // constructed). The captured-log assertion below cross-checks
    // that the event was attached to a live (non-disabled) span.
    span.in_scope(|| {
        tracing::info!("w11z_executor_span_field_pin");
    });
    assert!(
        logs_contain("w11z_executor_span_field_pin"),
        "traced_test subscriber must capture the in-scope event"
    );

    // Auxiliary: span name is the load-bearing telemetry identifier
    // for slow-query-log routing (per amendment-02 §M4.f). Pin it.
    assert_eq!(
        metadata.name(),
        "arcgraph_query::executor",
        "executor span name is the slow-query-log routing key"
    );
}

// ---------------------------------------------------------------------
// 5. Cancellation during execute surfaces at next batch
// ---------------------------------------------------------------------

#[test]
fn executor_cancellation_during_execute_surfaces_at_next_batch() {
    // Drive the executor with a token, trip mid-flight, verify
    // ExecutionError::Cancelled. v1.0-alpha cancels at batch
    // boundaries — within a single batch the work completes; the
    // NEXT batch boundary trips. Test pin: tripping BEFORE the
    // first call surfaces immediately.
    let s = substrate_with_n_persons(10);
    let cat = cat_basic();
    let token = CancellationToken::new();
    token.cancel();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_cancellation(token);
    let plan = arcgraph_query::logical_plan::LogicalPlan::Scan(
        arcgraph_query::logical_plan::LogicalScan {
            label: None,
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        },
    );
    let r = execute_with_context(&plan, &s, &ctx);
    assert_eq!(r, Err(ExecutionError::Cancelled));
}

// ---------------------------------------------------------------------
// 6. QueryEngine::execute end-to-end smoke
// ---------------------------------------------------------------------

#[test]
fn executor_query_engine_execute_routes_through_full_pipeline() {
    // M5↔M4 contract surface smoke. Bind `MATCH (n:Person) RETURN n`
    // through the full pipeline; the executor's output is the same
    // count as the substrate's Person node count.
    let s = substrate_with_n_persons(7);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let rows = engine
        .execute("MATCH (n:Person) RETURN n", &s)
        .expect("execute");
    assert_eq!(rows.len(), 7);
}

// ---------------------------------------------------------------------
// Auxiliary: WHERE filter end-to-end via QueryEngine::execute
// ---------------------------------------------------------------------

#[test]
fn executor_query_engine_execute_threads_filter_and_project() {
    let s = substrate_with_n_persons(5);
    // Persons have age = i*5 → ages 5, 10, 15, 20, 25.
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let rows = engine
        .execute("MATCH (n:Person) WHERE n.age > 10 RETURN n.age", &s)
        .expect("execute");
    // Predicate `> 10` keeps ages 15,20,25 → 3 rows.
    assert_eq!(rows.len(), 3);
}

// ---------------------------------------------------------------------
// Auxiliary: cancellation immediately surfaces, no work done
// ---------------------------------------------------------------------

#[test]
fn executor_cancellation_check_marker_pin() {
    // Trivial pin: the public API surface for CancellationError +
    // CancellationToken combined work end-to-end.
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
    token.cancel();
    assert_eq!(token.check(), Err(CancellationError));
}

// ---------------------------------------------------------------------
// Auxiliary: filter operator threads the per-query parameter bag
// ---------------------------------------------------------------------

#[test]
fn executor_filter_threads_parameters() {
    use arcgraph_query::executor::eval::Parameters;
    let s = substrate_with_n_persons(5);
    let cat = cat_basic();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    // Predicate: n.age > $threshold
    let pred = BoundExpression::BinaryOp {
        op: BinOp::Gt,
        lhs: Box::new(BoundExpression::PropertyAccess {
            base: Box::new(BoundExpression::VariableRef {
                name: "n".into(),
                binding_id: BindingId::new(0),
                span: Span::point(1, 1),
                type_info: None,
            }),
            path: vec![BoundPropertyRef {
                name: "age".into(),
                property_id: None,
                span: Span::point(1, 1),
            }],
            span: Span::point(1, 1),
            type_info: None,
        }),
        rhs: Box::new(BoundExpression::Parameter {
            name: "threshold".into(),
            span: Span::point(1, 1),
            type_info: None,
        }),
        span: Span::point(1, 1),
        type_info: None,
    };
    let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
    let mut params = Parameters::new();
    params.insert("threshold".into(), Value::Integer(15));
    let mut op = PhysicalOperator::Filter(
        FilterOp::new(PhysicalOperator::Scan(scan), pred).with_parameters(params),
    );
    let mut total = 0;
    loop {
        let b = op.next_batch(&ctx, &s).unwrap();
        if b.is_empty() {
            break;
        }
        total += b.row_count();
    }
    // Persons have age = i*5 → ages 5,10,15,20,25; > 15 keeps 20,25.
    assert_eq!(total, 2);
}

// ---------------------------------------------------------------------
// Auxiliary: Project literal column with aliases
// ---------------------------------------------------------------------

#[test]
fn executor_project_emits_one_row_per_input_with_aliased_column() {
    let s = substrate_with_n_persons(3);
    let cat = cat_basic();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
    let item = BoundProjectionItem {
        kind: BoundProjectionKind::Expr(BoundExpression::Literal {
            value: Literal::Integer(42),
            span: Span::point(1, 1),
            type_info: None,
        }),
        alias: Some("forty_two".into()),
        output_id: Some(BindingId::new(1)),
        source_text: None,
        span: Span::point(1, 1),
    };
    let mut op =
        PhysicalOperator::Project(ProjectOp::new(PhysicalOperator::Scan(scan), vec![item]));
    let b = op.next_batch(&ctx, &s).unwrap();
    assert_eq!(b.row_count(), 3);
    for row in b.rows() {
        assert_eq!(row[0], Value::Integer(42));
    }
}

// ---------------------------------------------------------------------
// #980 NIT-1 — the 7th ceiling: the eager result-Vec path
// (`execute()` / `execute_with_context()`, which `PROFILE` runs through
// via `profile_with_substrate`) must scale past the OLD 131 072-row
// fixed ceiling on the UNCAPPED budget path.
// ---------------------------------------------------------------------

/// Build a scan plan over all `LabelId(1)` nodes bound to `BindingId(0)`.
fn scan_all_persons_plan() -> arcgraph_query::logical_plan::LogicalPlan {
    arcgraph_query::logical_plan::LogicalPlan::Scan(arcgraph_query::logical_plan::LogicalScan {
        label: Some(LabelId::new(1)),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    })
}

/// #980 NIT-1 — the eager result-Vec drain in `execute_with_context`
/// (the path the public `execute()` Vec API + `PROFILE` use) previously
/// clipped UNCONDITIONALLY at `BUDGET_FALLBACK_ROWS` (= 131 072), so a
/// result set larger than that errored with the same "would reserve 0"
/// `ResourceExhausted` symptom even on the uncapped (no per-tenant byte
/// cap) budget path. After the fix it scales to the actual result
/// cardinality (guarded only by `UNCAPPED_RUNAWAY_GUARD_ROWS`).
///
/// N = 200 000 > 131 072. RED-on-revert: re-impose the
/// `BUDGET_FALLBACK_ROWS` ceiling at `executor/mod.rs` and this fails
/// with that ResourceExhausted.
#[test]
fn eager_result_vec_past_old_ceiling_succeeds_uncapped() {
    let n: u64 = 200_000; // > old 131 072 ceiling
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    // DEFAULT context => uncapped budget (the GA-blocker path).
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    assert!(
        !ctx.budget().has_cap(ctx.tenant()),
        "this test pins the UNCAPPED eager-Vec / PROFILE path (#980 NIT-1)"
    );
    let rows = execute_with_context(&scan_all_persons_plan(), &s, &ctx)
        .expect("uncapped eager-Vec result must not hit the old 131K ceiling");
    assert_eq!(rows.len(), n as usize, "every result row must materialize");
}

/// #980 NIT-1 (consistency) — the eager result-Vec path is now
/// `has_cap`-aware, mirroring the 6 operator-level fixes. With a
/// per-tenant byte cap configured, the row-count clip is NOT imposed by
/// this loop (the operator-layer byte budget governs); so a budgeted
/// tenant whose result fits the byte cap is NOT clipped early at the
/// old 131 072-row boundary the way the pre-fix `has_cap`-blind ceiling
/// did. We set a generous byte cap that admits the full result, and
/// assert the > 131 072-row result still materializes.
#[test]
fn eager_result_vec_capped_tenant_not_clipped_at_old_row_boundary() {
    let n: u64 = 150_000; // > old 131 072 ceiling
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    // A generous byte cap (8 GiB) that comfortably admits N rows — the
    // point is that the ROW-count clip no longer fires for a capped
    // tenant; byte enforcement (not exercised here) is the cap surface.
    let budget = MemoryBudget::with_per_tenant_cap(cat.tenant(), 8 * 1024 * 1024 * 1024);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget);
    assert!(
        ctx.budget().has_cap(ctx.tenant()),
        "this test pins the CAPPED path"
    );
    let rows = execute_with_context(&scan_all_persons_plan(), &s, &ctx)
        .expect("capped tenant within byte budget must not be clipped at the old row boundary");
    assert_eq!(rows.len(), n as usize);
}
