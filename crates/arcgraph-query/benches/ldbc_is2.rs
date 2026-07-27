//! Criterion bench for LDBC SNB Interactive-Short IS2 query (Recent
//! Messages of a Person) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # design-v2 §10.5 target
//!
//! IS2 P50 = **200µs**, P99 = 2ms.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is2 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is2_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    let _warm = engine.explain(ldbc_fixture::IS2).expect("warm-up explain");
    c.bench_function("ldbc_is2_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS2).expect("LDBC IS2 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is2_explain);
criterion_main!(benches);
