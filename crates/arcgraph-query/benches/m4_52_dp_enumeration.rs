//! Criterion benchmark for M4-52 (M4-05b) DP join-ordering
//! enumeration time vs join-width.
//!
//! Per ADR-036 §D-25 the M4-05 plan-build budget is **5 ms**
//! end-to-end. This bench measures DP enumeration time across
//! join-widths N ∈ {2..=8} (the v1.0 ceiling per
//! [`MAX_DP_RELATIONS`]). The combined budget includes the M4-51
//! cost walker (~5–50 µs) plus M4-52 DP enumeration; the bench
//! pin verifies M4-52 does NOT consume more than its share.
//!
//! # Empirical pin
//!
//! The roadmap exit criterion (M4-52 row): "1 Criterion bench (DP
//! enumeration time vs join-width)". The bench number outputs go
//! into `target/criterion/m4_52_dp_enumeration/` for the M4-52
//! review packet's empirical-gauntlet section.
//!
//! # Cost-model integration
//!
//! Each bench case builds a `StubCatalogProvider` with realistic
//! LDBC SNB SF-1-like cardinalities so the estimate_costs walks
//! exercise the standard cost-model code paths. The DP itself
//! reads `cat.snapshot()` once and then uses the per-operator
//! `cost_join` for incremental candidate costing.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench m4_52_dp_enumeration`

use arcgraph_core::{LabelId, Lsn};
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{
    JoinAlgorithm, JoinCondition, LogicalJoin, LogicalPlan, LogicalScan,
};
use arcgraph_query::planner::enumerate_join_order;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::BindingId;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn span() -> Span {
    Span::point(1, 1)
}

/// Build a connected join input with N leaves all sharing var=0
/// (the typical multi-pattern shared-anchor shape from LDBC SNB
/// IS3 / IS5 / IS6).
fn build_input(n: usize) -> LogicalPlan {
    assert!(n >= 2);
    let leaf = |i: usize| {
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(i as u32)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    };
    // Build a left-deep input plan in order [1, 2, ..., n].
    let mut acc = leaf(1);
    for i in 2..=n {
        acc = LogicalPlan::Join(LogicalJoin {
            left: Box::new(acc),
            right: Box::new(leaf(i)),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        });
    }
    acc
}

/// Catalog with N labels stamped at varying cardinalities (10×
/// spread to make ordering matter).
fn build_catalog(n: usize) -> StubCatalogProvider {
    let mut cat = StubCatalogProvider::new()
        .with_total_node_count(1_000_000)
        .with_total_rel_count(5_000_000);
    for i in 1..=n {
        let card = (10 * i) as u64 * 100;
        cat = cat.with_label_cardinality(LabelId::new(i as u32), card);
    }
    cat
}

fn bench_enumeration_by_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("m4_52_dp_enumeration_by_width");
    for n in 2..=8 {
        let plan = build_input(n);
        let cat = build_catalog(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let out = enumerate_join_order(black_box(plan.clone()), black_box(&cat));
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_enumeration_by_width);
criterion_main!(benches);
