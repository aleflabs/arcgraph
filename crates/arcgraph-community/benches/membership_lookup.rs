//! Criterion benches for [`BTreeMembershipIndex`] per ADR-040 §D-9.
//!
//! Three sub-benches:
//!
//! 1. `lookup` — point lookup of `membership(node_id, level)` against
//!    a 1 M-node, 10 K-community population. ADR-040 §D-9 commits
//!    P50 < 200 ns; the bench reports Criterion's measured P50 and
//!    P99.
//! 2. `members` — range scan returning all members of one
//!    community. Cost shape O(community_size).
//! 3. `rank_by_seeds` — composite operation. ADR-040 §D-9 commits
//!    O(k_seeds · log(communities) + k_out).
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p arcgraph-community --bench membership_lookup -- --quick
//! ```
//!
//! `--quick` cuts wall time to ~5 s with looser confidence
//! intervals; the full bench (~60 s) emits HTML reports under
//! `target/criterion/`.

use std::hint::black_box;

use arcgraph_community::{BTreeMembershipIndex, CommunityId, Level, MembershipIndex};
use arcgraph_core::{Lsn, NodeId, TenantId};
use criterion::{Criterion, criterion_group, criterion_main};

const N: u64 = 1_000_000;
const K: u64 = 10_000; // communities

fn populate() -> BTreeMembershipIndex {
    let idx = BTreeMembershipIndex::new();
    let assignment: Vec<(NodeId, CommunityId)> = (0..N)
        .map(|i| (NodeId::new(i), CommunityId::new(i % K)))
        .collect();
    // ADR-041 §D-3b: single install at LSN=1; queries use Lsn::MAX.
    idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(1), &assignment);
    idx
}

fn bench_lookup(c: &mut Criterion) {
    let idx = populate();
    let mut group = c.benchmark_group("membership_lookup");
    // Round-robin a small probe set so every iteration hits a
    // different B-tree leaf (avoids degenerate "always hot in L1"
    // numbers).
    let probes: Vec<NodeId> = (0..1024).map(|i| NodeId::new((i * 991) % N)).collect();
    let mut i = 0usize;
    group.bench_function("lookup_1m_10k", |b| {
        b.iter(|| {
            let n = probes[i % probes.len()];
            i = i.wrapping_add(1);
            let r = idx
                .lookup(
                    black_box(TenantId::DEFAULT),
                    black_box(n),
                    Level::FINEST,
                    Lsn::MAX,
                )
                .expect("ok");
            black_box(r);
        });
    });
    group.finish();
}

fn bench_members(c: &mut Criterion) {
    let idx = populate();
    let mut group = c.benchmark_group("membership_lookup");
    // Each community has N / K = 100 members.
    let mut i = 0u64;
    group.bench_function("members_size_100", |b| {
        b.iter(|| {
            let cid = CommunityId::new(i % K);
            i = i.wrapping_add(1);
            let r = idx
                .members(
                    black_box(TenantId::DEFAULT),
                    black_box(cid),
                    Level::FINEST,
                    Lsn::MAX,
                )
                .expect("ok");
            black_box(r);
        });
    });
    group.finish();
}

fn bench_rank_by_seeds(c: &mut Criterion) {
    let idx = populate();
    let mut group = c.benchmark_group("membership_lookup");
    let seeds = [NodeId::new(1), NodeId::new(2), NodeId::new(3)];
    group.bench_function("rank_by_seeds_3_to_top10", |b| {
        b.iter(|| {
            let r = idx
                .rank_by_seeds(
                    black_box(TenantId::DEFAULT),
                    black_box(&seeds),
                    Level::FINEST,
                    10,
                    Lsn::MAX,
                )
                .expect("ok");
            black_box(r);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_lookup, bench_members, bench_rank_by_seeds);
criterion_main!(benches);
