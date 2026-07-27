//! Criterion bench for LDBC SNB Interactive-Short IS1 query (Profile
//! of a Person) per design-v2 §10.5 + LDBC SNB Interactive
//! Specification §3.5.
//!
//! # Measured surface (v1.0-alpha)
//!
//! Plan-build wall-time: parse + bind + type-check + cross-substrate
//! validate + lower + DP enumerate + cost walker + PlanTree render via
//! `QueryEngine::explain`. Per `tests/m4_84_ldbc_perf_gate.rs`, the
//! full-execute LDBC harness is forward-deferred to M4-64 / M6 LDBC
//! perf milestone.
//!
//! # design-v2 §10.5 target
//!
//! IS1 P50 = **50µs**, P99 = 500µs.
//!
//! # Run
//!
//! `cargo bench -p arcgraph-query --bench ldbc_is1 --quick`

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "../tests/common/ldbc_fixture.rs"]
mod ldbc_fixture;

use arcgraph_query::QueryEngine;

fn ldbc_is1_explain(c: &mut Criterion) {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    // Warm-up: per Criterion convention, the first iteration includes
    // cold-cache + initial heap allocation noise; we explicitly run
    // one EXPLAIN before measurement to amortize.
    let _warm = engine.explain(ldbc_fixture::IS1).expect("warm-up explain");
    c.bench_function("ldbc_is1_explain_sf_0_0001", |b| {
        b.iter(|| {
            let _ = engine.explain(ldbc_fixture::IS1).expect("LDBC IS1 EXPLAIN");
        });
    });
}

criterion_group!(benches, ldbc_is1_explain);
criterion_main!(benches);
