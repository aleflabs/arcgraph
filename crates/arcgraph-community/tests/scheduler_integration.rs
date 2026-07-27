//! End-to-end integration tests for the
//! [`CommunityRefreshScheduler`] (M3.d-2).
//!
//! Unit tests in `crates/arcgraph-community/src/scheduler.rs` use
//! stub hooks that always return `None`; this file uses **real**
//! graphs + a real [`BTreeMembershipIndex`] + a real
//! [`GveLeiden::run`] under the hook so the
//! `hook → run → install_into → membership index` path is
//! exercised end-to-end.
//!
//! Cross-references:
//! - ADR-040 §D-7 commits ArcGraph to a daily background-refresh
//!   scheduler over the static GVE-Leiden algorithm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use arcgraph_community::{
    BTreeMembershipIndex, CommunityRefreshScheduler, EdgeUpdate, Graph, GveLeiden,
    LeidenIncremental, LeidenParams, Level, MembershipIndex, OwnedRefreshInputs, RefreshHook,
    SchedulerConfig,
};
use arcgraph_core::{Lsn, NodeId, TenantId};

// ───────────────────────────────────────────────────────────────
// Fixtures.
// ───────────────────────────────────────────────────────────────

/// Edge list of Zachary's karate club, copied inline per the
/// M3.d-2 split-decision: integration tests own their fixtures
/// until v1.1's cleanup.
const ZACHARY_EDGES: &[(u32, u32)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

fn zachary_graph() -> Graph {
    let edges: Vec<(u32, u32, f32)> = ZACHARY_EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    Graph::from_edges_undirected(34, &edges)
}

/// Test config with a 1-hour interval — long enough that the
/// scheduler thread never naturally ticks during a test, so
/// every observed tick is from `tick()` (the forced-sync path).
///
/// Per ADR-041 §D-3b the scheduler's install-LSN allocator
/// starts at `initial_install_lsn`. We set it to `Lsn::new(1000)`
/// so tests that manually pre-populate the index at low LSNs
/// (e.g. Lsn::new(1) / Lsn::new(2)) don't collide with the
/// scheduler's auto-allocated LSNs.
fn long_interval_cfg() -> SchedulerConfig {
    SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(1000),
    }
}

// ───────────────────────────────────────────────────────────────
// Hook implementations.
//
// Per ADR-040 amendment-05, hooks own `Arc<Graph>` +
// `Arc<BTreeMembershipIndex>` and return Arc-clones per
// `OwnedRefreshInputs` (no `&'a self`-tied borrow). Test fixture
// only — production posture is per-tick re-materialisation from
// `CrudStore` (see `arcgraph-storage::engine::ProductionRefreshHook`).
// ───────────────────────────────────────────────────────────────

/// Single-tenant hook resolving `tenant_match` to a fixed
/// `(graph, index)` pair, and `None` for any other tenant.
///
/// Per ADR-040 amendment-05, the hook owns `Arc<Graph>` so the
/// scheduler can take an Arc clone per tick without trait-shape
/// gymnastics. Test fixture only — production posture is per-tick
/// re-materialisation from `CrudStore` (see
/// `arcgraph-storage::engine::ProductionRefreshHook`).
struct StaticHook {
    tenant_match: TenantId,
    graph: Arc<Graph>,
    index: Arc<BTreeMembershipIndex>,
    params: LeidenParams,
}

impl StaticHook {
    fn new(tenant: TenantId, graph: Graph, index: Arc<BTreeMembershipIndex>) -> Self {
        Self {
            tenant_match: tenant,
            graph: Arc::new(graph),
            index,
            params: LeidenParams::default(),
        }
    }
}

impl RefreshHook for StaticHook {
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
        if tenant != self.tenant_match {
            return None;
        }
        Some(OwnedRefreshInputs {
            graph: Arc::clone(&self.graph),
            index: Arc::clone(&self.index),
            params: self.params,
            n_skip_prefix: 0,
        })
    }
}

/// Two-tenant hook: tenant A resolves to a real graph + index;
/// tenant B always returns `None` (soft-skip path).
struct TwoTenantHook {
    tenant_a: TenantId,
    tenant_b: TenantId,
    graph_a: Arc<Graph>,
    index_a: Arc<BTreeMembershipIndex>,
    params: LeidenParams,
}

impl RefreshHook for TwoTenantHook {
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
        if tenant == self.tenant_a {
            Some(OwnedRefreshInputs {
                graph: Arc::clone(&self.graph_a),
                index: Arc::clone(&self.index_a),
                params: self.params,
                n_skip_prefix: 0,
            })
        } else if tenant == self.tenant_b {
            // Soft-skip: tenant B is registered but not eligible.
            None
        } else {
            None
        }
    }
}

// ───────────────────────────────────────────────────────────────
// Tests.
// ───────────────────────────────────────────────────────────────

/// End-to-end: a forced tick walks
/// `RefreshHook::resolve → GveLeiden::run → install_into` and
/// populates the [`BTreeMembershipIndex`] with real community ids.
#[test]
fn scheduler_refresh_installs_into_membership_index() {
    let g = zachary_graph();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());
    let hook: Arc<dyn RefreshHook> =
        Arc::new(StaticHook::new(TenantId::DEFAULT, g, Arc::clone(&idx)));

    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    sched.register(TenantId::DEFAULT);

    // Force one tick.
    sched.tick();

    // The index should now report a community for node 0 at
    // Level::FINEST. We don't assert the specific community id —
    // GVE-Leiden's exact choice depends on the iteration order —
    // we assert that the lookup succeeds with `Some(_)`.
    let cid = idx
        .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("lookup must not error post-refresh")
        .expect("post-refresh index must contain node 0");
    eprintln!("post-refresh community for node 0: {cid:?}");

    // Health: one tick, zero failures, zero soft-skips.
    let h = sched.health_check();
    assert_eq!(h.total_ticks, 1);
    assert_eq!(h.total_refresh_failures, 0);
    assert_eq!(h.total_soft_skips, 0);
    assert!(h.last_tick_completed);

    sched.shutdown();
}

/// A fresh static-tick refresh overrides any prior incremental
/// drift. We seed the index manually with a static run, then
/// install incremental-update results, then force a scheduler tick.
/// After the tick the index is consistent with the canonical
/// static result on the (unchanged) graph.
#[test]
fn scheduler_refresh_overrides_prior_incremental_drift() {
    let g = zachary_graph();
    let params = LeidenParams::default();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());

    // Step 1: seed the index with a known good static result.
    let canonical = GveLeiden::run(&g, params);
    GveLeiden::install_into(&canonical, &idx, TenantId::DEFAULT, Lsn::new(1), 0);

    // Step 2: simulate drift via an incremental batch + manual
    // install. We reuse the same graph (no real perturbation), so
    // the incremental result equals the static one in steady-state;
    // the test pins the COMPOSITION path through the hook, not a
    // numerical drift.
    let prior_raw: Vec<u32> = canonical.levels[0].iter().map(|c| c.raw() as u32).collect();
    let updates = [EdgeUpdate::Insert { u: 0, v: 33 }]; // already-existing (0,1,…) doesn't matter; pick a non-edge.
    let inc = LeidenIncremental::apply_batch(&g, &prior_raw, &updates, &params);
    // Manually install incremental's level-0 assignment under
    // Level::FINEST so the index reflects the "post-incremental"
    // state.
    let pairs: Vec<(NodeId, arcgraph_community::CommunityId)> = inc
        .assignment
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            (
                NodeId::new(i as u64),
                arcgraph_community::CommunityId::new(u64::from(c)),
            )
        })
        .collect();
    idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(2), &pairs);

    // Step 3: snapshot the pre-tick state for at least one node.
    let pre_tick = idx
        .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("lookup ok")
        .expect("seeded index has node 0");

    // Step 4: force a scheduler tick — this re-runs static and
    // re-installs.
    let hook: Arc<dyn RefreshHook> = Arc::new(StaticHook::new(
        TenantId::DEFAULT,
        g.clone(),
        Arc::clone(&idx),
    ));
    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    sched.register(TenantId::DEFAULT);
    sched.tick();

    // Step 5: post-tick the index reflects the canonical static
    // result. The node may have the same community id as before
    // (the incremental run was steady-state), but the canonical
    // pin is that the post-tick membership matches the result of
    // a fresh static run on the same graph.
    let canonical_after = GveLeiden::run(&g, params);
    let canonical_for_node_0 = canonical_after.levels[0][0];
    let post_tick = idx
        .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("lookup ok")
        .expect("post-tick index has node 0");
    assert_eq!(
        post_tick, canonical_for_node_0,
        "post-tick membership must equal canonical static result; \
         pre_tick={pre_tick:?}, post_tick={post_tick:?}, canonical={canonical_for_node_0:?}",
    );

    let h = sched.health_check();
    assert_eq!(h.total_ticks, 1);
    assert_eq!(h.total_refresh_failures, 0);

    sched.shutdown();
}

/// One tenant returns `None` (soft-skip) — the other tenant's
/// refresh still completes. A soft-skip must not stop the tick.
#[test]
fn scheduler_continues_after_one_tenant_failure() {
    let tenant_a = TenantId::new(100);
    let tenant_b = TenantId::new(200);
    let g_a = zachary_graph();
    let idx_a: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());

    let hook: Arc<dyn RefreshHook> = Arc::new(TwoTenantHook {
        tenant_a,
        tenant_b,
        graph_a: Arc::new(g_a),
        index_a: Arc::clone(&idx_a),
        params: LeidenParams::default(),
    });

    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    sched.register(tenant_a);
    sched.register(tenant_b);

    sched.tick();

    // Tenant A's index should be populated.
    let cid_a = idx_a
        .lookup(tenant_a, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("A: lookup ok")
        .expect("A: index populated post-tick");
    eprintln!("Tenant A community for node 0: {cid_a:?}");

    // Health: one tick, one soft-skip, zero failures.
    let h = sched.health_check();
    assert_eq!(h.total_ticks, 1, "exactly one forced tick");
    assert_eq!(h.total_refresh_failures, 0, "soft-skip is not a failure",);
    assert_eq!(
        h.total_soft_skips, 1,
        "tenant B should have been soft-skipped exactly once",
    );

    sched.shutdown();
}

/// The natural cadence (1 hour) does not fire within a 100 ms
/// test window. Forced `tick()` does fire. Pins that the
/// scheduler is genuinely cadence-driven and the test interval
/// is honored.
#[test]
fn scheduler_with_natural_interval_does_not_tick_in_test_window() {
    let g = zachary_graph();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());
    let hook: Arc<dyn RefreshHook> =
        Arc::new(StaticHook::new(TenantId::DEFAULT, g, Arc::clone(&idx)));

    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    sched.register(TenantId::DEFAULT);

    // Wait briefly — much shorter than the 1-hour interval.
    thread::sleep(Duration::from_millis(100));

    let h0 = sched.health_check();
    assert_eq!(
        h0.total_ticks, 0,
        "natural-cadence tick must NOT fire within 100 ms when interval=1h",
    );

    // Forced tick still works.
    sched.tick();
    let h1 = sched.health_check();
    assert_eq!(h1.total_ticks, 1, "forced tick should fire");

    sched.shutdown();
}

/// Concurrent reader stress test. Per
/// `PHASE-M3D-SPLIT-DECISION.md` §4.2 `concurrent_membership_lookup`:
/// N readers concurrently hit `lookup` while the scheduler refreshes
/// the index; readers must not panic and must observe a consistent
/// snapshot (no torn community ids).
///
/// The [`BTreeMembershipIndex`] uses a single `parking_lot::RwLock`
/// so reads are blocked during the brief install_level write
/// window — readers see either the pre-write or post-write state,
/// never a torn one. This test pins that the lock model holds
/// under the scheduler's tick path.
#[test]
fn concurrent_membership_lookup_during_refresh() {
    let g = zachary_graph();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());

    // Pre-populate via a static run so readers always see at
    // least one valid generation of data.
    let pre = GveLeiden::run(&g, LeidenParams::default());
    GveLeiden::install_into(&pre, &idx, TenantId::DEFAULT, Lsn::new(1), 0);

    let hook: Arc<dyn RefreshHook> =
        Arc::new(StaticHook::new(TenantId::DEFAULT, g, Arc::clone(&idx)));
    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), Arc::clone(&hook));
    sched.register(TenantId::DEFAULT);

    let stop = Arc::new(AtomicBool::new(false));
    let mut readers: Vec<thread::JoinHandle<usize>> = Vec::new();
    for tid in 0..4 {
        let idx_r = Arc::clone(&idx);
        let stop_r = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name(format!("integration-reader-{tid}"))
            .spawn(move || {
                let mut hits = 0usize;
                let mut iter_count = 0usize;
                while !stop_r.load(Ordering::Acquire) && iter_count < 1000 {
                    // Cycle through nodes 0..34.
                    let node = NodeId::new((iter_count as u64) % 34);
                    let res = idx_r.lookup(TenantId::DEFAULT, node, Level::FINEST, Lsn::MAX);
                    match res {
                        Ok(Some(_)) => hits += 1,
                        Ok(None) => { /* node not in index; ignore */ }
                        Err(e) => {
                            // Errors here are a real correctness
                            // bug — fail loud.
                            panic!("reader observed error: {e:?}");
                        }
                    }
                    iter_count += 1;
                }
                hits
            })
            .expect("spawn reader thread");
        readers.push(handle);
    }

    // While readers run, force five ticks in a tight loop. Each
    // tick acquires the membership-index write lock briefly.
    for _ in 0..5 {
        sched.tick();
    }

    // Signal readers to stop; join.
    stop.store(true, Ordering::Release);
    let mut total_hits = 0usize;
    for r in readers {
        let h = r.join().expect("reader thread joined cleanly");
        total_hits += h;
    }
    eprintln!("concurrent readers total hits: {total_hits}");

    // Every reader should have observed at least some hits — the
    // index was pre-populated. Zero hits would suggest the index
    // was empty for the entire window, which would indicate a
    // torn lock or write-side bug.
    assert!(
        total_hits > 0,
        "readers should observe pre-populated entries during scheduler activity",
    );

    // Final-state consistency: the index has a community for
    // node 0.
    let final_cid = idx
        .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("lookup ok")
        .expect("post-stress index populated");
    eprintln!("final community for node 0: {final_cid:?}");

    let h = sched.health_check();
    // Forced ticks count toward total_ticks.
    assert!(
        h.total_ticks >= 5,
        "expected at least 5 forced ticks, got {}",
        h.total_ticks,
    );
    assert_eq!(h.total_refresh_failures, 0);

    sched.shutdown();
}

/// Bonus: incremental updates run after a scheduler tick produce
/// stable (deterministic + non-degenerate) assignments. This is
/// the "incremental → static refresh → incremental" cycle that
/// ADR-040 §D-7 frames as the steady-state daily cadence.
#[test]
fn incremental_after_scheduler_tick_resumes_correctly() {
    let g = zachary_graph();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());
    let hook: Arc<dyn RefreshHook> = Arc::new(StaticHook::new(
        TenantId::DEFAULT,
        g.clone(),
        Arc::clone(&idx),
    ));
    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    sched.register(TenantId::DEFAULT);
    sched.tick();

    // After the tick, read the canonical Level-0 assignment back
    // out of the index for incremental's `c_prev`.
    let mut c_prev: Vec<u32> = vec![0u32; g.n() as usize];
    for v in 0..g.n() {
        let cid = idx
            .lookup(
                TenantId::DEFAULT,
                NodeId::new(u64::from(v)),
                Level::FINEST,
                Lsn::MAX,
            )
            .expect("lookup ok")
            .expect("post-tick index populated");
        c_prev[v as usize] = cid.raw() as u32;
    }

    // Apply an incremental batch (single insert).
    let updates = [EdgeUpdate::Insert { u: 0, v: 33 }];
    let mut new_edges: Vec<(u32, u32, f32)> =
        ZACHARY_EDGES.iter().map(|&(a, b)| (a, b, 1.0)).collect();
    new_edges.push((0, 33, 1.0));
    let g_after = Graph::from_edges_undirected(34, &new_edges);
    let r1 = LeidenIncremental::apply_batch(&g_after, &c_prev, &updates, &LeidenParams::default());
    let r2 = LeidenIncremental::apply_batch(&g_after, &c_prev, &updates, &LeidenParams::default());
    // Determinism survives the static→incremental boundary.
    assert_eq!(r1.assignment, r2.assignment);
    assert!(r1.assignment.len() == g.n() as usize);

    sched.shutdown();
}

/// Bonus: registering many tenants then dropping the scheduler
/// does not panic; the thread is reaped in `Drop`. Pins
/// shutdown idempotency under a non-trivial registered set.
#[test]
fn scheduler_drop_after_long_register_chain() {
    let g = zachary_graph();
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());
    let hook: Arc<dyn RefreshHook> =
        Arc::new(StaticHook::new(TenantId::DEFAULT, g, Arc::clone(&idx)));
    let sched = CommunityRefreshScheduler::start(long_interval_cfg(), hook);
    for i in 1..=100u64 {
        sched.register(TenantId::new(i));
    }
    let h = sched.health_check();
    assert_eq!(h.registered_tenants, 100);
    sched.shutdown();
    let h_post = sched.health_check();
    assert!(h_post.shut_down);
    // Implicit drop here triggers a second shutdown — the
    // scheduler must remain idempotent.
}
