//! V11-S-04 exit-criterion bench (`docs/roadmap.md` V11-S-04 row):
//! `k_hop` k=3 LIMIT=1000 on a 1M-node fixture — **P95 ≤ 100 ms**.
//!
//! Fixture: 1M nodes, deterministic bounded-degree (8 out-edges per node,
//! SplitMix64-derived targets) — ~8M edges via `MemoryEdgeSource`. The
//! fixture is built ONCE outside the measured closure; the measurement is
//! the traversal alone (the in-crate cost; production end-to-end adds the
//! substrate's `expand` I/O, which this crate does not own).
//!
//! Run: `cargo bench -p arcgraph-graph-traversal --bench k_hop_1m`
//! Criterion emits to `target/criterion/` (testing strategy: regressions >10%
//! block merges).

use arcgraph_core::NodeId;
use arcgraph_graph_traversal::{
    GraphTraversalHandle, KHopRequest, MemoryEdgeSource, SamplingStrategy, TraversalDirection,
};
use criterion::{Criterion, criterion_group, criterion_main};

const NODES: u64 = 1_000_000;
const OUT_DEGREE: u64 = 8;

/// SplitMix64 step (kept local: the bench must not depend on crate
/// internals beyond the public API).
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn build_fixture() -> MemoryEdgeSource {
    let mut g = MemoryEdgeSource::new();
    let mut rng_state: u64 = 0xA5C3_9D1E_F012_3456;
    for src in 1..=NODES {
        for _ in 0..OUT_DEGREE {
            let dst = (splitmix(&mut rng_state) % NODES) + 1;
            g.add_edge(NodeId::new(src), NodeId::new(dst), None);
        }
    }
    g
}

fn bench_k_hop(c: &mut Criterion) {
    let handle = GraphTraversalHandle::new(build_fixture());
    let mut group = c.benchmark_group("k_hop_1m");
    // 1M-node fixture: keep sampling modest; the exit criterion is a P95
    // latency bound, not throughput.
    group.sample_size(20);

    let mut bfs_req = KHopRequest::new(3, TraversalDirection::Outbound);
    bfs_req.limit = 1000;
    group.bench_function("bfs_k3_limit1000", |b| {
        b.iter(|| {
            let r = handle
                .k_hop(std::hint::black_box(NodeId::new(42)), (), 0, &bfs_req)
                .expect("traversal");
            std::hint::black_box(r.nodes.len())
        });
    });

    let mut res_req = KHopRequest::new(3, TraversalDirection::Outbound);
    res_req.limit = 1000;
    res_req.sampling = SamplingStrategy::ReservoirVitterL;
    res_req.per_hop_frontier_cap = Some(256);
    res_req.seed = 7;
    group.bench_function("reservoir_k3_limit1000_cap256", |b| {
        b.iter(|| {
            let r = handle
                .k_hop(std::hint::black_box(NodeId::new(42)), (), 0, &res_req)
                .expect("traversal");
            std::hint::black_box(r.nodes.len())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_k_hop);
criterion_main!(benches);
