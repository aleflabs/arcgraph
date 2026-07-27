//! Criterion bench for LDBC SNB Interactive-Short IS6 query (Forum +
//! Moderator of a Message) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # design-v2 §10.5 target
//!
//! IS6 P50 = **2ms**, P99 = 20ms (the IS4-7 group's shared target).
//!
//! IS6 is the canonical OPTIONAL MATCH path in the LDBC SNB spec; per
//! amendment-03 §TIER-1 GAP D it ships at v1.0. Per W13γ fix-up LOW-4
//! (closes review-pr-285-final.md LOW-4) the LDBC IS6 query bank now
//! uses OPTIONAL MATCH per the LDBC SNB driver contract — see
//! `tests/common/ldbc_fixture.rs::IS6`.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is6 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is6_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    let _warm = engine.explain(ldbc_fixture::IS6).expect("warm-up explain");
    c.bench_function("ldbc_is6_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS6).expect("LDBC IS6 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is6_explain);
criterion_main!(benches);
