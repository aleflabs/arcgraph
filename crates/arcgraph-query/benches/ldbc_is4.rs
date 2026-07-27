//! Criterion bench for LDBC SNB Interactive-Short IS4 query (Content
//! of a Message) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # design-v2 §10.5 target
//!
//! IS4 P50 = **2ms**, P99 = 20ms (the IS4-7 group's shared target).
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is4 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is4_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    let _warm = engine.explain(ldbc_fixture::IS4).expect("warm-up explain");
    c.bench_function("ldbc_is4_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS4).expect("LDBC IS4 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is4_explain);
criterion_main!(benches);
