//! First fault-injection regressions for the DF-Leiden incremental
//! community surface (W28 Slice S5).
//!
//! # Why this file exists
//!
//! Per the W28 gap analysis (PR #510 §2 rank-4) and ADR-165 §M3,
//! `arcgraph-community` (DF-Leiden incremental + the daily-refresh
//! scheduler + the LSN-versioned membership index) shipped with
//! **zero fault-injection coverage**. Every load-bearing surface in
//! ArcGraph carries ≥1 fault-injection regression per failure mode
//! (`feedback_load_bearing_pr_requires_fault_injection_tests`); this
//! file closes that gap with one test per failure mode:
//!
//! 1. [`incremental_survives_scheduler_crash_mid_refresh`] — a crash
//!    that strikes *during* a daily-refresh reset (after a partial,
//!    torn membership-index write but before the generation lands in
//!    full). The scheduler's `catch_unwind` containment
//!    (`scheduler.rs::refresh_one_tenant`) must keep the thread alive,
//!    and the next clean refresh ("restart") must reconstruct a
//!    **valid partition** (every node in exactly one community) whose
//!    modularity is **≥ the pre-crash floor**. The torn intermediate
//!    must be fully superseded — it must never surface at the
//!    post-restart visible snapshot.
//!
//! 2. [`apply_batch_torn_update_is_atomic_or_rejected`] — a torn
//!    edge-update batch (the cumulative graph reflects only a prefix
//!    of the batch's edges) and a torn commit (a crash between the
//!    pure `apply_batch` compute and the `install_level` commit). The
//!    membership effect must be **full-or-none**: either the whole
//!    computed generation lands (read-back binary-equal to the
//!    computed assignment) or none of it does (read-back binary-equal
//!    to the prior generation). A half-applied membership — some nodes
//!    on the new community, some on the old — must be structurally
//!    impossible. Oracle is **binary-equal vs a clean re-run**
//!    (`feedback_determinism_oracle_concurrency_tests`), not
//!    dedupe-/length-consistency.
//!
//! 3. [`mvcc_boundary_install_vs_read`] — a concurrent install at LSN
//!    N racing readers pinned to snapshot N-1. The readers MUST see
//!    the pre-install partition **exactly** (snapshot isolation): an
//!    install at a strictly later LSN is invisible to the older
//!    snapshot, so a read pinned to N-1 cannot tear, whether or not the
//!    LSN-N install has landed. Extends the sequential happy-path pins
//!    in `community_mvcc_visibility.rs` with a real concurrent
//!    writer/reader race. Narrowed per triage #595: the earlier
//!    "any read at the N boundary observes a whole snapshot" assertion
//!    over-asserted cross-node multi-key snapshot atomicity that the
//!    per-call read API does not promise (`read_back_level0` is `n`
//!    separate `lookup` calls); atomic cross-node snapshot reads (a
//!    snapshot-pinned batch-read handle) are a future feature, not the
//!    v1.0 contract.
//!
//! # Oracle discipline (ENGINEERING_DOCTRINE §3 "Strong oracles")
//!
//! - "Valid partition" is asserted **explicitly** via
//!   [`assert_valid_partition`] — every graph vertex maps to exactly
//!   one community in BOTH the reverse (`lookup`) and forward
//!   (`members`) structures, and the union covers `0..n` with no
//!   omissions and no duplicates. It is NOT "the call didn't panic".
//! - Atomicity is asserted by **binary-equal** read-back vs the
//!   computed assignment (full) or the prior generation (none), NOT by
//!   "result is non-empty / has the right length".
//! - The MVCC pin compares each concurrent read against a captured
//!   reference partition with `==` over the whole `Vec<CommunityId>`.
//!
//! # Test-only fault injection — zero production change
//!
//! All three faults are injected purely at the test layer: a
//! fault-injecting [`RefreshHook`] that commits a partial install then
//! panics (the scheduler already contains hook panics via
//! `catch_unwind` per `scheduler.rs`), a `catch_unwind` around the
//! apply/commit pipeline, and a `std::thread`-driven writer/reader
//! race. No production seam, feature flag, or `#[cfg(test)]` hook is
//! added to `scheduler.rs` / `leiden_incremental.rs` / the index.
//!
//! Fixtures are Zachary's karate club (tests 1 + 3; deterministic,
//! well-known community structure, small `n` for a fast concurrency
//! race) and a 4-block SBM (test 2; denser graph where the deletion
//! batch provably re-partitions). Both are copied inline per the
//! M3.d-2 split-decision convention (shared fixtures are a v1.1
//! cleanup).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::Duration;

use arcgraph_community::{
    BTreeMembershipIndex, CommunityId, CommunityRefreshScheduler, EdgeUpdate, Graph, GveLeiden,
    LeidenIncremental, LeidenParams, Level, MembershipIndex, OwnedRefreshInputs, RefreshHook,
    SchedulerConfig, modularity,
};
use arcgraph_core::{Lsn, NodeId, TenantId};

// ───────────────────────────────────────────────────────────────
// Fixtures.
// ───────────────────────────────────────────────────────────────

/// Edge list of Zachary's karate club (34 nodes, 78 undirected
/// edges). Copied inline per the M3.d-2 split-decision (shared
/// fixtures are a v1.1 cleanup); identical to the encoding used by
/// `scheduler_integration.rs` and `leiden_incremental_consistency.rs`.
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

const ZACHARY_N: u32 = 34;

fn zachary_graph() -> Graph {
    let edges: Vec<(u32, u32, f32)> = ZACHARY_EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    Graph::from_edges_undirected(ZACHARY_N, &edges)
}

/// Linear congruential generator step — same constants as the
/// `leiden_incremental_consistency.rs` fixtures; reproduces a
/// deterministic pseudo-random sequence across runs.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Uniform [0, 1) sample using the top 53 bits of an LCG step.
fn lcg_unit(state: &mut u64) -> f64 {
    let raw = lcg_next(state);
    ((raw >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Deterministic 4-block stochastic block model. Returns the graph;
/// the block ground-truth is unused here (we draw `c_prev` from a
/// real static run instead). Same shape as the
/// `leiden_incremental_consistency.rs` SBM helper.
fn sbm(n: u32, k: u32, p_in: f64, p_out: f64, seed: u64) -> Graph {
    assert!(n % k == 0, "n must be divisible by k for clean blocks");
    let block_size = n / k;
    let block_of = |v: u32| v / block_size;

    let mut state = seed;
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if block_of(u) == block_of(v) {
                p_in
            } else {
                p_out
            };
            if lcg_unit(&mut state) < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    Graph::from_edges_undirected(n, &edges)
}

// ───────────────────────────────────────────────────────────────
// Shared oracles + helpers.
// ───────────────────────────────────────────────────────────────

/// Read the Level-0 partition back out of the index at `read_lsn`,
/// one `Option<CommunityId>` per vertex `0..n`.
fn read_back_level0(
    idx: &BTreeMembershipIndex,
    tenant: TenantId,
    n: u32,
    read_lsn: Lsn,
) -> Vec<Option<CommunityId>> {
    (0..n)
        .map(|v| {
            idx.lookup(tenant, NodeId::new(u64::from(v)), Level::FINEST, read_lsn)
                .expect("lookup must not error")
        })
        .collect()
}

/// Convert a raw `u32` assignment to the `[CommunityId]`
/// representation `modularity` consumes.
fn raw_to_cid(raw: &[u32]) -> Vec<CommunityId> {
    raw.iter()
        .map(|&c| CommunityId::new(u64::from(c)))
        .collect()
}

/// Install a flat Level-0 `[CommunityId]` partition (one entry per
/// vertex `0..n`) at `install_lsn`.
fn install_level0(
    idx: &BTreeMembershipIndex,
    tenant: TenantId,
    partition: &[CommunityId],
    install_lsn: Lsn,
) {
    let pairs: Vec<(NodeId, CommunityId)> = partition
        .iter()
        .enumerate()
        .map(|(i, &c)| (NodeId::new(i as u64), c))
        .collect();
    idx.install_level(tenant, Level::FINEST, install_lsn, &pairs);
}

/// **Strong "valid partition" oracle.** Assert that the Level-0
/// snapshot visible at `read_lsn` is a genuine partition of `0..n`:
/// every vertex maps to exactly one community in BOTH the reverse
/// (`lookup`) and forward (`members`) structures, and the union of
/// all communities' members covers `0..n` exactly once (no omission,
/// no duplicate). Returns the by-vertex assignment for downstream
/// modularity computation.
///
/// This is the explicit every-node-in-exactly-one-community guard the
/// W28-S5 oracle discipline requires — not "the call didn't panic".
fn assert_valid_partition(
    idx: &BTreeMembershipIndex,
    tenant: TenantId,
    n: u32,
    read_lsn: Lsn,
) -> Vec<CommunityId> {
    let nn = n as usize;
    // Reverse view: lookup must yield Some(_) for every vertex.
    let mut by_vertex: Vec<CommunityId> = Vec::with_capacity(nn);
    for v in 0..n {
        let c = idx
            .lookup(tenant, NodeId::new(u64::from(v)), Level::FINEST, read_lsn)
            .expect("lookup must not error");
        assert!(
            c.is_some(),
            "VALID-PARTITION VIOLATION: vertex {v} maps to NO community at \
             read_lsn={} — a torn/partial install survived recovery",
            read_lsn.raw(),
        );
        by_vertex.push(c.expect("checked Some above"));
    }

    // Forward view: walk every distinct community's member list and
    // confirm each vertex is covered exactly once, and that the
    // forward (members) and reverse (lookup) views agree per vertex.
    let distinct: BTreeSet<CommunityId> = by_vertex.iter().copied().collect();
    let mut covered = vec![false; nn];
    let mut total = 0usize;
    for &c in &distinct {
        let members = idx
            .members(tenant, c, Level::FINEST, read_lsn)
            .expect("members must not error");
        for m in members {
            let mi = m.raw() as usize;
            assert!(mi < nn, "member {mi} out of range (n={n})");
            assert!(
                !covered[mi],
                "VALID-PARTITION VIOLATION: vertex {mi} appears in more than \
                 one community at read_lsn={}",
                read_lsn.raw(),
            );
            covered[mi] = true;
            assert_eq!(
                by_vertex[mi],
                c,
                "VALID-PARTITION VIOLATION: forward(members) and \
                 reverse(lookup) disagree for vertex {mi} at read_lsn={}",
                read_lsn.raw(),
            );
            total += 1;
        }
    }
    assert_eq!(
        total, nn,
        "VALID-PARTITION VIOLATION: forward index covers {total} vertices, \
         expected {nn}",
    );
    assert!(
        covered.iter().all(|&b| b),
        "VALID-PARTITION VIOLATION: at least one vertex is absent from the \
         forward index at read_lsn={}",
        read_lsn.raw(),
    );
    by_vertex
}

/// Test config whose interval (1 hour) is far longer than any test
/// window, so the dedicated scheduler thread never ticks naturally —
/// every observed tick is the synchronous forced `tick()`. The
/// install-LSN allocator starts high (1000) so it never collides with
/// the manual low-LSN installs the tests pre-stage.
fn restart_cfg() -> SchedulerConfig {
    SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(1000),
    }
}

// ───────────────────────────────────────────────────────────────
// Fault-injecting hooks.
// ───────────────────────────────────────────────────────────────

/// A [`RefreshHook`] that crashes *mid-refresh*: it begins the
/// refresh reset (recomputes the static partition), commits a
/// **torn, partial** Level-0 install (only `partial_nodes` of the `n`
/// vertices land), and then panics — modelling a process death that
/// struck after the write began but before the full generation
/// committed.
///
/// The scheduler wraps `do_refresh` (which calls `resolve`) in
/// `std::panic::catch_unwind` (`scheduler.rs::refresh_one_tenant`), so
/// this panic is contained: the scheduler thread survives and
/// `total_refresh_failures` increments by one. The torn snapshot is
/// left committed at `crash_lsn`; a later clean refresh must supersede
/// it. `parking_lot::RwLock` does not poison on panic, and the panic
/// fires only after `install_level` has returned (lock released), so
/// no lock is held across the unwind.
struct CrashMidRefreshHook {
    tenant: TenantId,
    graph: Arc<Graph>,
    index: Arc<BTreeMembershipIndex>,
    params: LeidenParams,
    crash_lsn: Lsn,
    partial_nodes: usize,
}

impl RefreshHook for CrashMidRefreshHook {
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
        if tenant != self.tenant {
            return None;
        }
        // Begin the refresh "reset": recompute the static partition.
        let fresh = GveLeiden::run(&self.graph, self.params);
        let level0 = &fresh.levels[0];
        // Commit a TORN, partial Level-0 install — only the first
        // `partial_nodes` (node, community) pairs land. This is the
        // half-written generation a crash would leave behind.
        let pairs: Vec<(NodeId, CommunityId)> = level0
            .iter()
            .enumerate()
            .take(self.partial_nodes)
            .map(|(i, &c)| (NodeId::new(i as u64), c))
            .collect();
        self.index
            .install_level(self.tenant, Level::FINEST, self.crash_lsn, &pairs);
        // … then the process dies mid-refresh.
        panic!("simulated scheduler crash mid-refresh (after partial Level-0 install)");
    }
}

/// A clean single-tenant hook used for the "restart" refresh: resolves
/// the tenant to a fixed `(graph, index)` and `None` otherwise. Per
/// ADR-040 amendment-05 the hook owns `Arc<Graph>` + `Arc<index>`.
struct CleanHook {
    tenant: TenantId,
    graph: Arc<Graph>,
    index: Arc<BTreeMembershipIndex>,
    params: LeidenParams,
}

impl RefreshHook for CleanHook {
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
        if tenant != self.tenant {
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

// ───────────────────────────────────────────────────────────────
// Failure mode 1: crash mid-refresh.
// ───────────────────────────────────────────────────────────────

/// **Failure mode: a crash strikes during the daily-refresh reset.**
///
/// Timeline:
/// 1. A prior good refresh (generation G0) is committed at LSN 1 —
///    the "last clean daily refresh". Its modularity is the
///    pre-crash floor.
/// 2. A new refresh begins, commits a **torn** partial Level-0 install
///    at LSN 2 (only half the vertices land), then the process dies.
///    The scheduler contains the panic (thread survives;
///    `total_refresh_failures == 1`).
/// 3. **Restart**: a fresh scheduler runs one clean refresh, fully
///    re-installing every level at LSN 1000.
///
/// Assertions on the post-restart visible snapshot (`Lsn::MAX`):
/// - **(a) valid partition** — every vertex in exactly one community
///   (the torn 17-vertex intermediate must NOT survive; if it did,
///   vertices 17..34 would be missing and the oracle fires).
/// - **(b) modularity ≥ the pre-crash floor** — recovery does not
///   degrade community quality.
#[test]
fn incremental_survives_scheduler_crash_mid_refresh() {
    let tenant = TenantId::DEFAULT;
    let params = LeidenParams::default();
    let graph = Arc::new(zachary_graph());
    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());

    // ── Step 1: last clean refresh G0 at LSN 1; capture the floor. ──
    let g0 = GveLeiden::run(&graph, params);
    GveLeiden::install_into(&g0, &idx, tenant, Lsn::new(1), 0);
    let pre_crash = assert_valid_partition(&idx, tenant, ZACHARY_N, Lsn::MAX);
    let floor = modularity(&graph, &pre_crash, params.gamma);
    eprintln!("pre-crash floor modularity = {floor:.6}");

    // ── Step 2: a refresh crashes mid-write (torn partial install). ──
    let crash_hook: Arc<dyn RefreshHook> = Arc::new(CrashMidRefreshHook {
        tenant,
        graph: Arc::clone(&graph),
        index: Arc::clone(&idx),
        params,
        crash_lsn: Lsn::new(2),
        partial_nodes: 17, // half of 34 — a genuinely torn generation.
    });
    {
        let crashed_sched = CommunityRefreshScheduler::start(restart_cfg(), crash_hook);
        crashed_sched.register(tenant);
        // The forced tick runs `do_refresh` on THIS thread, inside the
        // scheduler's `catch_unwind`. The hook panics; the scheduler
        // contains it and the test thread survives.
        crashed_sched.tick();

        let h = crashed_sched.health_check();
        assert_eq!(h.total_ticks, 1, "the crash tick fired exactly once");
        assert_eq!(
            h.total_refresh_failures, 1,
            "the mid-refresh crash must be contained as exactly one failure",
        );
        assert!(
            h.last_tick_completed,
            "the scheduler thread must SURVIVE a mid-refresh crash (not die)",
        );
        // Drop the crashed scheduler == the crashed process exits.
    }

    // The torn intermediate is what is currently visible — confirm the
    // fault actually corrupted the live snapshot (so the recovery
    // assertion below is non-vacuous): vertex 17 fell off the torn
    // Level-0 install at LSN 2.
    let torn_snapshot = read_back_level0(&idx, tenant, ZACHARY_N, Lsn::MAX);
    assert!(
        torn_snapshot[17].is_none(),
        "pre-condition: the torn install should have dropped vertex 17 \
         (got {:?}) — otherwise the fault is not exercised",
        torn_snapshot[17],
    );

    // ── Step 3: restart — one clean refresh re-installs every level. ──
    let clean_hook: Arc<dyn RefreshHook> = Arc::new(CleanHook {
        tenant,
        graph: Arc::clone(&graph),
        index: Arc::clone(&idx),
        params,
    });
    let restart_sched = CommunityRefreshScheduler::start(restart_cfg(), clean_hook);
    restart_sched.register(tenant);
    restart_sched.tick();

    let h = restart_sched.health_check();
    assert_eq!(h.total_ticks, 1, "restart fired exactly one tick");
    assert_eq!(
        h.total_refresh_failures, 0,
        "the restart refresh must complete cleanly",
    );

    // ── Assertions on the recovered (post-restart) snapshot. ──
    // (a) valid partition — the torn intermediate is fully superseded.
    let recovered = assert_valid_partition(&idx, tenant, ZACHARY_N, Lsn::MAX);
    // (b) modularity ≥ pre-crash floor (deterministic recompute ⇒ equal).
    let recovered_q = modularity(&graph, &recovered, params.gamma);
    eprintln!("post-restart recovered modularity = {recovered_q:.6}");
    assert!(
        recovered_q >= floor - 1e-9,
        "recovered modularity {recovered_q:.6} dropped below the pre-crash \
         floor {floor:.6}",
    );

    restart_sched.shutdown();
}

// ───────────────────────────────────────────────────────────────
// Failure mode 2: torn edge-update batch / torn commit atomicity.
// ───────────────────────────────────────────────────────────────

/// **Failure mode: a torn edge-update batch + a torn commit.**
///
/// The production incremental-apply pipeline is two steps:
/// `apply_batch` (pure compute) then `install_level` (atomic commit).
/// This test pins that the membership effect is **full-or-none**:
///
/// - **Torn graph**: a crash during cumulative-graph materialisation
///   hands `apply_batch` a graph reflecting only a *prefix* of the
///   batch. `apply_batch` must be total (no panic) and deterministic
///   on it — binary-equal across re-runs. (It never half-mutates: it
///   returns a fresh assignment vector.)
/// - **Torn commit (NONE)**: a crash between compute and commit leaves
///   the index holding the **prior generation exactly** — read-back
///   binary-equal to `c_prev`, never a half-applied membership.
/// - **Clean commit (FULL)**: the computed generation lands in full —
///   read-back binary-equal to the computed assignment.
///
/// The oracle is **binary equality vs a clean re-run**
/// (`feedback_determinism_oracle_concurrency_tests`), not
/// length/non-empty consistency. A half-applied membership (some
/// vertices new, some old) is structurally impossible because
/// `install_level` commits a whole snapshot under one write lock; this
/// test would catch a regression that broke that property.
#[test]
fn apply_batch_torn_update_is_atomic_or_rejected() {
    let tenant = TenantId::DEFAULT;
    let params = LeidenParams::default();

    // Prior committed generation over a 4-block SBM (denser than
    // Zachary; the deletion batch below provably re-partitions it).
    let n = 200u32;
    let g_old = sbm(n, 4, 0.3, 0.02, 0xC0FFEE);
    let prior = GveLeiden::run(&g_old, params);
    let c_prev: Vec<u32> = prior.levels[0].iter().map(|c| c.raw() as u32).collect();

    // Pick a victim vertex `v` that is NOT its own community
    // representative (`c_prev[v] != v`) and has at least one edge. A
    // deletion batch removing EVERY edge incident to `v` isolates it:
    // after the singleton reset (Sahu §6), `v` has no neighbours, so it
    // stays in its own singleton (`r_clean[v] == v`). Because
    // `c_prev[v] != v`, the batch PROVABLY changes the final partition,
    // which makes the FULL-vs-NONE atomicity oracle non-vacuous (a
    // half-application would differ from BOTH `c_prev` and `r_clean`).
    // This is a stronger fixture than a random mixed batch, which can
    // re-converge to `c_prev` (affected vertices reset then move back),
    // leaving FULL == NONE and the oracle blind.
    let victim = (0..n)
        .find(|&v| c_prev[v as usize] != v && g_old.neighbors(v).next().is_some())
        .expect("SBM must have a non-representative vertex with edges");

    // Canonical (lo, hi) edge set of g_old; the batch deletes every
    // edge incident to `victim`.
    let mut all_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for u in 0..n {
        for (w, _) in g_old.neighbors(u) {
            let (lo, hi) = if u <= w { (u, w) } else { (w, u) };
            all_edges.insert((lo, hi));
        }
    }
    let incident_to_victim: Vec<(u32, u32)> = all_edges
        .iter()
        .copied()
        .filter(|&(a, b)| a == victim || b == victim)
        .collect();
    assert!(
        !incident_to_victim.is_empty(),
        "victim {victim} must have incident edges",
    );
    let updates: Vec<EdgeUpdate> = incident_to_victim
        .iter()
        .map(|&(a, b)| EdgeUpdate::Delete { u: a, v: b })
        .collect();

    // g_new = g_old with every edge incident to `victim` removed.
    let g_new_edges: Vec<(u32, u32, f32)> = all_edges
        .iter()
        .copied()
        .filter(|&(a, b)| a != victim && b != victim)
        .map(|(a, b)| (a, b, 1.0))
        .collect();
    let g_new = Graph::from_edges_undirected(n, &g_new_edges);

    // ── Reference (clean) compute + binary-equal determinism oracle. ──
    let r_clean = LeidenIncremental::apply_batch(&g_new, &c_prev, &updates, &params);
    let r_again = LeidenIncremental::apply_batch(&g_new, &c_prev, &updates, &params);
    assert_eq!(
        r_clean.assignment, r_again.assignment,
        "apply_batch must be byte-identical across re-runs (determinism oracle)",
    );
    assert_eq!(r_clean.iterations, r_again.iterations);
    assert_eq!(r_clean.vertices_moved, r_again.vertices_moved);
    assert_eq!(r_clean.frontier_visits, r_again.frontier_visits);
    // The isolation batch provably re-partitions: the victim is now a
    // lone singleton (`r_clean[victim] == victim`) whereas it was a
    // non-representative member before (`c_prev[victim] != victim`), so
    // FULL and NONE are observably different (the oracle is non-vacuous).
    assert_eq!(
        r_clean.assignment[victim as usize], victim,
        "isolated victim {victim} must end in its own singleton community",
    );
    assert_ne!(
        r_clean.assignment, c_prev,
        "the isolation batch must change the assignment so the FULL-vs-NONE \
         oracle bites",
    );

    // ── Torn graph: crash during cumulative-graph materialisation. ──
    // `g_torn` reflects only the FIRST deletion of the batch applied;
    // the full update list is still handed to apply_batch. The function
    // must be total (no panic) and deterministic — never a half-state.
    let first_del = incident_to_victim[0];
    let g_torn_edges: Vec<(u32, u32, f32)> = all_edges
        .iter()
        .copied()
        .filter(|&e| e != first_del)
        .map(|(a, b)| (a, b, 1.0))
        .collect();
    let g_torn = Graph::from_edges_undirected(n, &g_torn_edges);
    let r_torn1 = LeidenIncremental::apply_batch(&g_torn, &c_prev, &updates, &params);
    let r_torn2 = LeidenIncremental::apply_batch(&g_torn, &c_prev, &updates, &params);
    assert_eq!(
        r_torn1.assignment, r_torn2.assignment,
        "apply_batch on a TORN graph must be total + byte-identical across \
         re-runs — never a half-state or a panic",
    );
    assert_eq!(
        r_torn1.assignment.len(),
        n as usize,
        "apply_batch always returns a full-length assignment (one entry per vertex)",
    );

    // ── Install atomicity: FULL-or-NONE against a binary-equal oracle. ──
    // The prior generation `c_prev` is committed at LSN 1 in both
    // probe indexes.
    let prior_cid = raw_to_cid(&c_prev);

    // (NONE) — a crash strikes AFTER compute, BEFORE commit. The index
    // must keep the prior generation EXACTLY.
    let idx_none = BTreeMembershipIndex::new();
    install_level0(&idx_none, tenant, &prior_cid, Lsn::new(1));
    let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _computed = LeidenIncremental::apply_batch(&g_new, &c_prev, &updates, &params);
        // The commit (install_level) would happen here — but the
        // process dies first. `_computed` is dropped, never installed.
        panic!("simulated crash between apply_batch compute and commit");
    }));
    assert!(crashed.is_err(), "the injected crash must unwind");
    let none_readback = read_back_level0(&idx_none, tenant, n, Lsn::MAX);
    let none_expected: Vec<Option<CommunityId>> = prior_cid.iter().copied().map(Some).collect();
    assert_eq!(
        none_readback, none_expected,
        "TORN COMMIT (NONE): a crash before commit must leave the index at \
         the prior generation EXACTLY — binary-equal to c_prev, never a \
         half-applied membership",
    );

    // (FULL) — a clean commit installs the whole computed generation.
    let idx_full = BTreeMembershipIndex::new();
    install_level0(&idx_full, tenant, &prior_cid, Lsn::new(1));
    let r_committed = LeidenIncremental::apply_batch(&g_new, &c_prev, &updates, &params);
    install_level0(
        &idx_full,
        tenant,
        &raw_to_cid(&r_committed.assignment),
        Lsn::new(2),
    );
    let full_readback = read_back_level0(&idx_full, tenant, n, Lsn::MAX);
    let full_expected: Vec<Option<CommunityId>> = r_clean
        .assignment
        .iter()
        .map(|&c| Some(CommunityId::new(u64::from(c))))
        .collect();
    assert_eq!(
        full_readback, full_expected,
        "CLEAN COMMIT (FULL): the read-back must be binary-equal to the \
         computed assignment — the whole generation lands, no half-application",
    );

    // The only two reachable states are NONE (== c_prev) and FULL
    // (== r_clean); they are distinct (asserted above), so a
    // half-applied membership is excluded by construction.
}

// ───────────────────────────────────────────────────────────────
// Failure mode 3: MVCC boundary — concurrent install vs read.
// ───────────────────────────────────────────────────────────────

/// **Failure mode: an install at LSN N races readers pinned to the
/// N-1 snapshot — they must never observe the in-flight install.**
///
/// Extends the sequential pins in `community_mvcc_visibility.rs` with a
/// real concurrent writer/reader race. Generation A is installed at
/// LSN 10 (= N-1); a writer thread installs a clearly-distinct
/// generation B (singletons) at LSN 20 (= N) while reader threads spin
/// reads pinned to the N-1 snapshot:
///
/// - A read at `read_lsn = 10` (= N-1) MUST see generation A **exactly**,
///   regardless of whether the concurrent LSN-20 install has landed —
///   the install is invisible to the older snapshot (`snapshot_at`
///   resolves to the latest install ≤ 10, which is always the LSN-10
///   install). This is the snapshot-isolation contract the single
///   `parking_lot::RwLock` actually provides: each `lookup` is atomic
///   and independently resolves to the immutable LSN-10 snapshot, so a
///   read pinned to N-1 cannot tear — an install at a strictly later LSN
///   cannot perturb it. Sequential pins (pre-install at LSN 10 → A;
///   post-install, after the writer joins, at LSN 10 → A and at LSN 20
///   → B) bracket the race with the whole-generation reads the
///   write-lock guarantees for a non-racing read.
///
/// # Narrowed per triage #595 (test over-assertion, NOT a community bug)
///
/// An earlier revision also asserted that a read at the N **boundary**
/// (`read_lsn = install_lsn = 20`) mid-race observed a *whole* snapshot
/// — exactly A or exactly B, never a torn mix. That over-asserts a
/// **cross-node multi-key snapshot-atomicity** guarantee the per-call
/// read API does NOT promise: [`read_back_level0`] is `n` *separate*
/// [`MembershipIndex::lookup`] calls, each taking its own read lock and
/// resolving `snapshot_at` independently. A `read_back_level0(.., 20)`
/// whose iteration straddles the writer's install legitimately observes
/// a prefix on generation A (lookups that ran before the install
/// landed) and a suffix on generation B (lookups that ran after) — a
/// torn *multi-call* read that is correct behaviour, not an MVCC
/// violation. (`install_level` swaps a single snapshot atomically under
/// its write-lock — that is *per-call* atomicity, not multi-key
/// snapshot atomicity across `n` independent reads.) The exact-boundary
/// race (`read_lsn == install_lsn == 20`) is itself unrealistic: a
/// correct reader's snapshot only advances past 20 AFTER the commit it
/// is reading has been observed to land. Atomic cross-node snapshot
/// reads (a snapshot-pinned batch-read handle that takes ONE read lock
/// for the whole partition) are a **future** feature, not the v1.0
/// contract; the over-asserting boundary assertion was removed per #595.
///
/// Oracle: each concurrent N-1 read is compared with `==` against the
/// captured reference partition (whole `Vec<CommunityId>`), not a
/// per-element or length check.
#[test]
fn mvcc_boundary_install_vs_read() {
    let tenant = TenantId::DEFAULT;
    let params = LeidenParams::default();
    let graph = zachary_graph();
    let n = ZACHARY_N;

    // Generation A = the canonical static partition (a real multi-
    // community partition). Generation B = singletons (every vertex its
    // own community) — a valid partition that is clearly distinct from A.
    let gen_a: Vec<CommunityId> = GveLeiden::run(&graph, params).levels[0].clone();
    let gen_b: Vec<CommunityId> = (0..n).map(|v| CommunityId::new(u64::from(v))).collect();
    assert_ne!(gen_a, gen_b, "the two generations must differ");

    let idx: Arc<BTreeMembershipIndex> = Arc::new(BTreeMembershipIndex::new());
    // Pre-install generation A at LSN 10 (= N-1).
    install_level0(&idx, tenant, &gen_a, Lsn::new(10));

    let ref_a: Vec<Option<CommunityId>> = gen_a.iter().copied().map(Some).collect();
    let ref_b: Vec<Option<CommunityId>> = gen_b.iter().copied().map(Some).collect();

    // Sequential pre-install pins (the contract install_level's write-lock
    // provides for a read that does NOT race the install): a whole read
    // at LSN 10 (= N-1) and at LSN 20 (= N, before any LSN-20 install
    // exists) both resolve to generation A.
    assert_eq!(
        read_back_level0(&idx, tenant, n, Lsn::new(10)),
        ref_a,
        "sanity: generation A is visible at LSN 10 pre-race",
    );
    assert_eq!(
        read_back_level0(&idx, tenant, n, Lsn::new(20)),
        ref_a,
        "pre-install: at LSN 20 the latest install ≤ 20 is the LSN-10 \
         install — generation A is visible until the LSN-20 install lands",
    );

    let n_readers = 4;
    let iters_per_reader = 4000usize;
    let barrier = Arc::new(Barrier::new(n_readers + 1)); // readers + writer.

    let mut readers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n_readers);
    for tid in 0..n_readers {
        let idx_r = Arc::clone(&idx);
        let barrier_r = Arc::clone(&barrier);
        let ref_a_r = ref_a.clone();
        let handle = thread::Builder::new()
            .name(format!("mvcc-reader-{tid}"))
            .spawn(move || {
                barrier_r.wait();
                for _ in 0..iters_per_reader {
                    // N-1 snapshot isolation: a read pinned to LSN 10 MUST
                    // be generation A exactly, whether or not the concurrent
                    // LSN-20 install has landed. Each `lookup` resolves
                    // `snapshot_at` to the latest install ≤ 10 (always the
                    // LSN-10 install) independently, so this read cannot
                    // tear — an install at a strictly later LSN is invisible
                    // to the older snapshot. (The N-boundary read at LSN 20
                    // was removed per #595: it over-asserted cross-node
                    // multi-call snapshot atomicity the per-call API does
                    // not promise — see the fn doc.)
                    let at_nm1 = read_back_level0(&idx_r, tenant, n, Lsn::new(10));
                    assert_eq!(
                        at_nm1, ref_a_r,
                        "MVCC VIOLATION: a read at the N-1 snapshot (LSN 10) \
                         observed something other than generation A — the \
                         LSN-20 install leaked into an older snapshot",
                    );
                }
            })
            .expect("spawn mvcc reader");
        readers.push(handle);
    }

    // Writer: install generation B at LSN 20 (= N) once the readers are
    // spinning, so the install lands mid-race against the N-1 readers.
    let idx_w = Arc::clone(&idx);
    let barrier_w = Arc::clone(&barrier);
    let gen_b_w = gen_b.clone();
    let writer = thread::Builder::new()
        .name("mvcc-writer".to_owned())
        .spawn(move || {
            barrier_w.wait();
            install_level0(&idx_w, tenant, &gen_b_w, Lsn::new(20));
        })
        .expect("spawn mvcc writer");

    writer.join().expect("writer joined cleanly");
    for r in readers {
        r.join().expect("a reader observed an MVCC violation");
    }

    // Post-race sequential pins (after the writer joins — no race): the
    // N-1 snapshot is immutable (still generation A); the N snapshot is
    // generation B (the install landed). A non-racing read_back at a
    // given LSN is a whole generation per the write-lock contract.
    assert_eq!(
        read_back_level0(&idx, tenant, n, Lsn::new(10)),
        ref_a,
        "post-race: the N-1 snapshot is immutable — still generation A",
    );
    assert_eq!(
        read_back_level0(&idx, tenant, n, Lsn::new(20)),
        ref_b,
        "post-race: the N snapshot is generation B (the install landed)",
    );
}
