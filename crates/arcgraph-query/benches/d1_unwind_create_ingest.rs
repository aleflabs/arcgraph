//! Criterion bench for the ADR-147-amendment-03 (D-1) UNWIND-ingest
//! lever: `UNWIND $rows AS r CREATE (n {…: r.…})`.
//!
//! # Measured surface
//!
//! End-to-end parse → bind → type-check → cross-substrate → lower →
//! execute of a single `UNWIND $rows AS r CREATE` statement over an
//! in-memory `StubExecutorSubstrate`, swept over batch size. The single
//! statement fans out to N substrate writes with ONE parse/plan/dispatch
//! (the O(N)→O(1) collapse D-1 delivers) — this bench measures that per-
//! statement amortization against the pre-D-1 one-statement-per-row shape.
//!
//! # Honesty
//!
//! The "51×" and "47K/s" figures in the PR/ADR are PROJECTIONS. This
//! bench measures the parse/dispatch amortization on a stub substrate
//! (no fsync, no durable commit) — it is NOT a durable-throughput
//! number. The residual toward 47K/s is commit-amortization on the
//! held-tx path, which a stub substrate does not model. Treat the
//! numbers here as the parse/plan/dispatch-elimination contribution
//! only.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench d1_unwind_create_ingest --quick`

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::eval::Parameters;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{executor::Pipeline, parse};

const STUB_FIRST_LABEL_ID: u32 = 1024;
const QUERY: &str = "UNWIND $rows AS r CREATE (n:User {v: r.v})";

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID))
}

/// Build the `$rows` param: an `n`-element list of `{v: i}` maps.
fn rows_param(n: usize) -> Parameters {
    let list = Value::List(
        (0..n)
            .map(|i| {
                let mut m = std::collections::BTreeMap::new();
                m.insert("v".to_string(), Value::Integer(i as i64));
                Value::Map(m)
            })
            .collect(),
    );
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("rows".to_string(), list);
    params
}

fn lower_query() -> LogicalPlan {
    let stmt = parse(QUERY).expect("parse");
    let cat = catalog();
    let mut bound = BindingVisitor::bind(&stmt, QUERY, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

fn drain(
    op: &mut arcgraph_query::executor::ops::PhysicalOperator,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) {
    loop {
        let b = op.next_batch(ctx, substrate).expect("batch");
        if b.is_empty() {
            break;
        }
    }
}

fn bench_unwind_create(c: &mut Criterion) {
    let plan = lower_query();
    let mut group = c.benchmark_group("d1_unwind_create_ingest");
    for &n in &[1usize, 16, 256, 2048] {
        let params = rows_param(n);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                // Fresh substrate per iter so node-id space doesn't grow
                // unbounded across iterations (keeps the measured work
                // the N writes, not an ever-growing index).
                let substrate = StubExecutorSubstrate::new();
                let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                let mut op =
                    Pipeline::build_with_parameters(&plan, &params).expect("pipeline build");
                drain(&mut op, &substrate, &ctx);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_unwind_create);
criterion_main!(benches);
