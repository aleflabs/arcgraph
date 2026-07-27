//! Criterion bench for LDBC SNB Interactive-Short IS3 query (Friends
//! of a Person) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # design-v2 §10.5 target
//!
//! IS3 P50 = **500µs**, P99 = 5ms.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is3 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is3_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    let _warm = engine.explain(ldbc_fixture::IS3).expect("warm-up explain");
    c.bench_function("ldbc_is3_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS3).expect("LDBC IS3 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is3_explain);
criterion_main!(benches);
