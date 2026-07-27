//! Membership lookup latency integration test.
//!
//! Per ADR-040 §D-9: `membership(node_id, level)` lookup is
//! committed to **<200 ns P50** in release builds. This test
//! populates the index with 100 K nodes across 1 K communities at
//! level 0, runs 100 K lookups, and asserts the P50 budget.
//!
//! In debug builds the strict <200 ns budget does not hold (the
//! BTreeMap::get path is not inlined / vectorized the same way),
//! so we relax to <1 µs as a sanity floor — strict enforcement is
//! gated behind `cfg(not(debug_assertions))` (i.e. release).
//!
//! ## Why this is an integration test, not a Criterion bench
//!
//! Criterion warms up, samples thousands of times per measurement,
//! and reports per-group statistics — that is what
//! `benches/membership_lookup.rs` is for. Here we run a single
//! shot of 100 K lookups against a populated index, sort the
//! observed latencies, and assert the P50. Single-shot is the
//! right shape because the test is about the unloaded P50 floor;
//! the bench tracks regressions over time.

use std::time::Instant;

use arcgraph_community::{BTreeMembershipIndex, CommunityId, Level, MembershipIndex};
use arcgraph_core::{Lsn, NodeId, TenantId};

const N: usize = 100_000;
const K: usize = 1_000; // communities

#[test]
fn membership_lookup_p50_under_200ns_release() {
    // Populate the index: node `i` belongs to community `i % K`.
    let idx = BTreeMembershipIndex::new();
    let assignment: Vec<(NodeId, CommunityId)> = (0..N as u64)
        .map(|i| (NodeId::new(i), CommunityId::new(i % K as u64)))
        .collect();
    // ADR-041 §D-3b: install at LSN=1; lookups at Lsn::MAX see
    // the latest install (single-snapshot index for this latency
    // pin).
    idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(1), &assignment);

    // Warmup pass — primes the B-tree node caches.
    let mut warm = 0u64;
    for i in 0..N as u64 {
        if let Ok(Some(c)) = idx.lookup(TenantId::DEFAULT, NodeId::new(i), Level::FINEST, Lsn::MAX)
        {
            warm = warm.wrapping_add(c.raw());
        }
    }
    // Defensive: ensure the warmup is not optimized out.
    std::hint::black_box(warm);

    // Measured pass — record per-call latency in nanoseconds.
    let mut latencies = Vec::with_capacity(N);
    for i in 0..N as u64 {
        let t0 = Instant::now();
        let r = idx.lookup(TenantId::DEFAULT, NodeId::new(i), Level::FINEST, Lsn::MAX);
        let dt = t0.elapsed().as_nanos() as u64;
        // Confirm the hit so the compiler doesn't elide the call.
        match r {
            Ok(Some(_)) => {}
            other => panic!("unexpected miss: {other:?}"),
        }
        latencies.push(dt);
    }

    latencies.sort_unstable();
    let p50 = latencies[N / 2];
    let p99 = latencies[(N * 99) / 100];
    let mean = latencies.iter().sum::<u64>() / latencies.len() as u64;

    eprintln!(
        "membership_lookup latency over {N} calls: P50 = {p50} ns, P99 = {p99} ns, mean = {mean} ns",
    );

    // ADR-040 §D-9: <200 ns P50 in release builds. In debug, the
    // floor is much higher because the standard library's
    // BTreeMap::get is uninlined and bounds checks fire on every
    // node hop; relax to 1 µs as a sanity floor.
    #[cfg(not(debug_assertions))]
    {
        assert!(
            p50 < 200,
            "release P50 must be < 200 ns per ADR-040 §D-9, got {p50} ns"
        );
    }
    #[cfg(debug_assertions)]
    {
        assert!(
            p50 < 1_000,
            "debug P50 must be < 1 µs (relaxed floor), got {p50} ns"
        );
    }
}
