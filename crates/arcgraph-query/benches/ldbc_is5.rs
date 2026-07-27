//! Criterion bench for LDBC SNB Interactive-Short IS5 query (Author
//! of a Message) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # design-v2 §10.5 target
//!
//! IS5 P50 = **2ms**, P99 = 20ms (the IS4-7 group's shared target).
//!
//! IS5 + IS6 are the OPTIONAL MATCH queries in the LDBC SNB spec; per
//! amendment-03 §TIER-1 GAP D the OPTIONAL MATCH path is lit at
//! v1.0. Per W13γ fix-up LOW-4 (closes review-pr-285-final.md LOW-4)
//! the LDBC IS5 query bank now uses OPTIONAL MATCH per the LDBC SNB
//! driver contract — see `tests/common/ldbc_fixture.rs::IS5`.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is5 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is5_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    let _warm = engine.explain(ldbc_fixture::IS5).expect("warm-up explain");
    c.bench_function("ldbc_is5_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS5).expect("LDBC IS5 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is5_explain);
criterion_main!(benches);
