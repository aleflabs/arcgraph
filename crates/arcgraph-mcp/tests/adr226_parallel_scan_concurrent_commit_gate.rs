//! ADR-226 rc-GATE **T1** — the load-bearing combined-concurrency test.
//!
//! Converts the two invariants the combined-concurrency ultracode
//! verdict rested "CONCURRENCY-BUILD-SOUND" on — but which had ZERO
//! tests — from *argued* to *enforced*:
//!
//! 1. **materialize-before-fan-out** (`parallel_scan.rs:386 → 390`):
//!    `substrate.scan_nodes_with_context(...)` returns the whole owned
//!    `Vec<BoundNode>` BEFORE `filter_parallel(nodes)` opens the rayon
//!    region, so morsel workers hold NO page ref and make ZERO substrate
//!    calls. This keeps seams (a) morsel×buffer-fault-in,
//!    (b) SI-uniformity-across-morsels and (d) eviction×morsel-page-ref
//!    UNREACHABLE.
//! 2. **one-frozen-snapshot** (`substrate.rs` `scan_id_range` →
//!    `txn_manager.begin` → `transaction.rs` snapshot frozen for the
//!    txn's life): the whole id-range read runs under ONE snapshot
//!    pinned once, on the calling thread, before any morsel splits.
//!
//! # The gate condition (verdict §5)
//!
//! "A green rc whose soundness rests on inspection-only invariants is
//! one refactor away from silent corruption." T1 drives the REAL
//! [`arcgraph_mcp::storage::CrudExecutorSubstrate`] (over the real MVCC
//! kernel) with a FORCED multi-morsel fan-out (`ARCGRAPH_SCAN_MORSEL_SIZE`
//! tiny, `ARCGRAPH_SCAN_PARALLEL_THRESHOLD=0`, `ARCGRAPH_PARALLEL_SCAN=1`)
//! WHILE a second thread commits mid-scan
//! mutations — BOTH in-place slot updates (mutate an existing node's
//! `counter` property) AND inserts of brand-new nodes. The oracle is the
//! SERIAL [`arcgraph_query::executor::ops::ScanOp`] run against the SAME
//! substrate under the SAME concurrent-commit schedule. If the
//! morsel-driven parallel scan ever drops / dupes / reorders / TEARS a
//! row (a partial-morsel view that mixes pre- and post-mutation state),
//! `parallel_result != serial_result` and the gate FAILS.
//!
//! # Two gates: (A) parallel == serial, and (B) the no-leak positive proof
//!
//! Because the scan reads ONE frozen snapshot (before any morsel), neither
//! the serial nor the parallel scan observes a commit that lands after the
//! snapshot is pinned (snapshot isolation — the CORRECT behavior). T1's
//! point is NOT that the scan sees the concurrent writes; it is that the
//! parallel materialize-before-fan path yields the SAME frozen-snapshot
//! result as serial, EVEN under concurrent-commit page / version-chain
//! write pressure.
//!
//! **Gate A (`parallel == serial`)** is compared over the SEEDED id range
//! `1..=SEED_NODES`, a fixed committed prefix (committed serially + joined
//! BEFORE either scan). The concurrent committer touches only a DISJOINT
//! region — it UPDATES **sacrificial** nodes (ids `> SEED_NODES`) and
//! INSERTS brand-new nodes (ids above the run's pre-scan high-water) — so
//! the seeded-range projection is timing-INDEPENDENT and the equality is
//! flake-free (no "did the snapshot race the commit").
//!
//! But Gate A ALONE is INSUFFICIENT as an rc-gate for materialize-before-fan
//! (R1 #1378): because the compared seeded range is IMMUTABLE during the
//! scan, a BROKEN impl that moved the substrate scan INTO the per-morsel
//! closure (each morsel a fresh `begin`/snapshot) would STILL pass Gate A —
//! every per-morsel snapshot sees the same immutable seeded rows regardless
//! of timing. A gate test that passes even when its named invariant breaks
//! does not enforce it.
//!
//! **Gate B (no-leak positive proof)** closes that circuit. It captures the
//! FULL scan output (before the seeded-range filter) and asserts NO
//! during-scan-INSERTED id (one above the run's pre-scan high-water)
//! appears in it. The threshold is captured PER RUN at the snapshot-pin
//! instant — not a static constant — because the two runs share the fixture
//! (the second run starts with the first run's during-inserts already
//! committed, i.e. a higher high-water).
//!
//! To make Gate B a HARD, flake-free assertion on the correct path (rather
//! than relying on the scan's internal `begin` winning a race against the
//! committer), each scan runs under a **held transaction** whose snapshot is
//! pinned by `TxnManager::begin_owned` at a point the test CONTROLS —
//! strictly BEFORE the committer is released — and the high-water is
//! captured at that SAME instant. The correct `ParallelScanOp` / `ScanOp`
//! reads the whole owned `Vec` through that held, pre-insert snapshot
//! (`scan_nodes_with_context` → `scan_id_range_in_tx` on the held txn) and
//! materializes it fully BEFORE fanning to morsels — so no during-insert id
//! can be in the buffer, no matter when the inserts land. A BROKEN impl that
//! moved the substrate scan into the per-morsel closure as a FRESH `begin`
//! (a morsel making a substrate call — the exact "morsels make zero
//! substrate calls" invariant violation) would read the LIVE snapshot
//! mid-burst and catch a just-inserted id above the captured high-water →
//! Gate B TRIPS. Gate B needs NO mutation of the seeded range, so it is
//! flake-free while being sensitive to the exact seam it is the rc-gate for.
//! (Verified RED-on-revert: transiently rewriting the scan's first-batch to
//! fresh-`begin` re-scan the LIVE substrate — with a small delay so inserts
//! land — trips Gate B; restored to byte-identical production.)
//!
//! Run:
//!   cargo test -p arcgraph-mcp --test adr226_parallel_scan_concurrent_commit_gate
//!
//! # ADR provenance
//! - **ADR-226 §4 slice S4 / gate CONC-D** — morsel-driven parallel scan.
//! - **combined-concurrency ultracode verdict §4 T1 + §5** — the gate
//!   condition this test discharges.
//! - **ADR-163** — the reused Jepsen-style ArcQL→MVCC fixture
//!   (`CrudExecutorSubstrate` over the real kernel).

#[path = "jepsen_arcql_common/mod.rs"]
mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId};
use arcgraph_mcp::storage::BoltHeldTxn;
use arcgraph_storage::crud;

use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::ops::{ParallelScanOp, PhysicalOperator, ScanOp};
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::bound_ast::BindingId;

use common::{JepsenArcqlFixture, counter_property_data};

// ── Env-var names (the ADR-226 §4 S4 knobs; literal per the spec) ──────
const ENV_PARALLEL_SCAN: &str = "ARCGRAPH_PARALLEL_SCAN";
const ENV_MORSEL_SIZE: &str = "ARCGRAPH_SCAN_MORSEL_SIZE";
const ENV_ROW_THRESHOLD: &str = "ARCGRAPH_SCAN_PARALLEL_THRESHOLD";

/// Serialize env-mutating tests: Rust runs `#[test]`s on shared process
/// threads, and this suite pokes the process-global `ARCGRAPH_SCAN_*`
/// env vars that [`ParallelScanOp`] reads at scan time. Mirrors the
/// `ENV_TEST_LOCK` pattern the in-crate `parallel_scan` unit tests use.
static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Tiny morsel size (records) forced so the seeded dataset spans MANY
/// morsels — the true multi-morsel fan-out path (not the single-morsel
/// degenerate case). 8 over `SEED_NODES` seeded rows ⇒ several morsels.
const FORCED_MORSEL_SIZE: usize = 8;

/// Seeded node count (the compared id range `1..=SEED_NODES`). Chosen so
/// `SEED_NODES / FORCED_MORSEL_SIZE` is comfortably `> 1` (multi-morsel).
const SEED_NODES: u64 = 64;

/// Sacrificial nodes (created before the scans, ids
/// `SEED_NODES+1 ..= SEED_NODES+SACRIFICIAL_NODES`) that the concurrent
/// committer UPDATES during each scan. Pre-existing, so they always appear
/// in the frozen view regardless of timing.
const SACRIFICIAL_NODES: u64 = 8;

/// The id count that exists BEFORE the FIRST scan starts (seed +
/// sacrificial) — a pre-scan sanity anchor. The ACTUAL per-run leak
/// threshold is captured DYNAMICALLY at each run's snapshot-pin instant
/// (`run_scan_under_concurrent_commits` returns it), because the two runs
/// share the fixture and the second run starts with the first run's
/// during-inserts already committed (a higher high-water). Every node the
/// concurrent committer INSERTS during a run gets an id strictly `>` that
/// run's captured high-water.
const PRESCAN_HIGH_WATER: u64 = SEED_NODES + SACRIFICIAL_NODES;

/// During-scan inserts per burst. Large enough that a BROKEN per-morsel
/// re-scan (each morsel a fresh `begin`) has a wide window to catch a
/// freshly-inserted id `>` the run's prescan high-water mid-burst, so the
/// leak assertion is RED-on-revert (see the module + fn docs).
const DURING_INSERTS: u64 = 64;

/// The single label every node (seeded + sacrificial + inserted) carries,
/// so the scan's label filter engages identically for both paths.
fn label() -> LabelId {
    LabelId::new(1)
}

/// The scan's node binding (index 0). A bare label scan binds exactly one
/// variable; the id only tags the schema slot, so a stable `0` mirrors
/// the in-crate `parallel_scan` unit tests.
fn binding0() -> BindingId {
    BindingId::new(0)
}

/// RAII guard: set the three ADR-226 §4 S4 env vars to force the
/// multi-morsel parallel path, restoring the prior values on drop.
///
/// SAFETY (edition-2024 `set_var`/`remove_var` unsafe): all env mutation
/// happens while the caller holds [`ENV_TEST_LOCK`], so no concurrent
/// reader of these vars runs; the guard restores each var (to its prior
/// value, or removed) before the lock is released.
struct ScanEnvGuard {
    prior: Vec<(&'static str, Option<String>)>,
}

impl ScanEnvGuard {
    fn force_multi_morsel_parallel() -> Self {
        let keys = [ENV_PARALLEL_SCAN, ENV_MORSEL_SIZE, ENV_ROW_THRESHOLD];
        let prior: Vec<(&'static str, Option<String>)> =
            keys.iter().map(|&k| (k, std::env::var(k).ok())).collect();
        // SAFETY: see the struct doc — guarded by ENV_TEST_LOCK; restored
        // on drop.
        unsafe {
            std::env::set_var(ENV_PARALLEL_SCAN, "1");
            std::env::set_var(ENV_MORSEL_SIZE, FORCED_MORSEL_SIZE.to_string());
            std::env::set_var(ENV_ROW_THRESHOLD, "0");
        }
        Self { prior }
    }
}

impl Drop for ScanEnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.prior {
            // SAFETY: see the struct doc — guarded by ENV_TEST_LOCK; this
            // restores each var to its captured prior value (or removes it).
            unsafe {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
}

/// One scanned row projected to `(id, counter)` — the observable identity
/// plus the in-place-mutable property. A torn view (a morsel that mixed
/// pre- and post-update `counter` values) shows up as a mismatched
/// `counter` even when the id set matches; a dropped / duped row shows up
/// as a differing id multiset; a reorder shows up in the ordered `Vec`.
type ScanRow = (u64, Option<i64>);

/// Drive a physical operator to EOS against the substrate, projecting
/// EVERY emitted node to `(id, counter)` in EMISSION order (the FULL,
/// UNFILTERED buffer). The caller derives two views from this:
///
/// - the SEEDED-range subset (`id <= SEED_NODES`) — timing-independent, so
///   `parallel == serial` is a flake-free equality oracle; and
/// - the FULL set — used by the LEAK assertion, which checks that no
///   during-scan-inserted id (`> PRESCAN_HIGH_WATER`) appears, directly
///   pinning materialize-before-fan (see [`seeded`] + the leak assert).
fn drain_all_rows(
    mut op: PhysicalOperator,
    ctx: &ExecutionContext,
    substrate: &arcgraph_mcp::storage::CrudExecutorSubstrate,
) -> Vec<ScanRow> {
    let mut rows = Vec::new();
    loop {
        let batch = op.next_batch(ctx, substrate).expect("next_batch OK");
        if batch.is_empty() {
            break;
        }
        for row in batch.rows() {
            if let Value::Node(n) = &row[0] {
                let counter = match n.properties.get("counter") {
                    Some(Value::Integer(v)) => Some(*v),
                    _ => None,
                };
                rows.push((n.id.raw(), counter));
            }
        }
    }
    rows
}

/// The SEEDED-range subset of a full scan output (`id <= SEED_NODES`),
/// preserving emission order. Timing-independent (the committer never
/// touches the seeded range), so `parallel == serial` over it is flake
/// free.
fn seeded(rows: &[ScanRow]) -> Vec<ScanRow> {
    rows.iter()
        .copied()
        .filter(|(id, _)| *id <= SEED_NODES)
        .collect()
}

/// A deterministic burst of concurrent commits that runs on a SECOND
/// thread WHILE a scan executes on the main thread. It performs BOTH
/// mutation shapes the T1 spec requires, targeting ONLY the disjoint
/// region (ids `> SEED_NODES`) so it never perturbs the compared seeded
/// projection while still pounding the version chains / pages:
///
/// - **in-place slot updates** — `crud::update_node` bumps each
///   sacrificial node's `counter` (the in-place version-chain write path).
/// - **new-node inserts** — `crud::create_node` allocates fresh nodes past
///   the current high-water (ids `> PRESCAN_HIGH_WATER`).
///
/// Each mutation is its own `begin → mutate → commit`, so the committer
/// races the scan's morsel fan-out. The WORK is deterministic (fixed
/// count); only the interleaving is racy — which is the point: the scan's
/// snapshot is pinned (by the held txn — see
/// [`run_scan_under_concurrent_commits`]) strictly BEFORE this burst runs,
/// so by SI the scan must see NONE of these commits regardless of when each
/// lands. `scan_started` gates the FIRST commit on the scanner having
/// entered its drain path so the commits genuinely OVERLAP the fan-out
/// (rather than racing ahead and finishing before the scan even starts) —
/// giving a broken per-morsel re-scan a live window to leak.
fn concurrent_commit_burst(
    fixture: &JepsenArcqlFixture,
    sacrificial: &[NodeId],
    inserts: u64,
    counter_base: i64,
    scan_started: &AtomicBool,
) {
    let (mgr, crud, tenant) = (&fixture.mgr, &fixture.crud, fixture.tenant);
    // Overlap the fan-out: don't start committing until the scanner is in
    // its drain path. (Correctness of the leak gate does NOT depend on this
    // — the held-txn snapshot is already pinned pre-burst — but it ensures
    // the commits pressure the morsels rather than finishing first.)
    while !scan_started.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    // In-place slot updates on the sacrificial (disjoint) nodes.
    for (i, &node) in sacrificial.iter().enumerate() {
        let mut tx = mgr.begin(tenant);
        let next = counter_base + i as i64;
        if crud::update_node(crud, &mut tx, node, &counter_property_data(next)).is_ok() {
            let _ = crud::commit(tx, crud);
        } else {
            crud.discard_pending(tx.id());
            crud.discard_pending_installs(tx.id());
        }
    }
    // New-node inserts past the current high-water (ids > PRESCAN_HIGH_WATER).
    for j in 0..inserts {
        let mut tx = mgr.begin(tenant);
        let seed_val = counter_base + 10_000 + j as i64;
        if crud::create_node(
            crud,
            &mut tx,
            tenant,
            label(),
            &counter_property_data(seed_val),
        )
        .is_ok()
        {
            let _ = crud::commit(tx, crud);
        } else {
            crud.discard_pending(tx.id());
            crud.discard_pending_installs(tx.id());
        }
    }
}

/// Run ONE scan (built by `build_op`) to EOS on the main thread WHILE a
/// [`concurrent_commit_burst`] runs on a second thread. Returns the FULL
/// scanned `(id, counter)` rows in emission order (the caller derives the
/// seeded-range subset + the leak set from it).
///
/// # Deterministic pre-insert snapshot (the hard leak-gate guarantee)
///
/// The scan runs under a **held transaction** whose snapshot is pinned by
/// `TxnManager::begin_owned` HERE — strictly before the committer is
/// released. The correct scan reads the whole owned `Vec` through that
/// held, pre-insert snapshot (`scan_nodes_with_context` →
/// `scan_id_range_in_tx`), so the frozen buffer CANNOT contain a
/// during-insert id no matter when the inserts land. A broken per-morsel
/// re-scan that made a FRESH `begin` substrate call would instead read the
/// LIVE snapshot mid-burst and leak — which the caller's Gate B catches.
///
/// A fresh [`ExecutionContext`] is built per scan (with the held txn
/// attached) because `with_held_txn` consumes `self`.
///
/// Returns `(full_rows, prescan_high_water)`: `prescan_high_water` is the
/// node high-water captured at the instant the snapshot was pinned. Every
/// id in the correct frozen buffer is `<= prescan_high_water`; every
/// during-insert gets an id `> prescan_high_water` (the per-run leak
/// threshold — the two runs share a fixture, so the second run's threshold
/// is HIGHER than the first, hence per-run capture rather than a global
/// `PRESCAN_HIGH_WATER`).
fn run_scan_under_concurrent_commits<F>(
    fixture: &JepsenArcqlFixture,
    sacrificial: &[NodeId],
    during_inserts: u64,
    counter_base: i64,
    build_op: F,
) -> (Vec<ScanRow>, u64)
where
    F: FnOnce() -> PhysicalOperator,
{
    // Pin the read snapshot NOW (pre-burst): begin_owned freezes it, and
    // BoltHeldTxn carries it opaquely onto the ExecutionContext so the
    // substrate scan reads through it (scan_id_range_in_tx). Capture the
    // high-water at the SAME instant — every id at or below it is
    // pre-existing (in the frozen view); every during-insert exceeds it.
    let owned = fixture.mgr.begin_owned(fixture.tenant);
    let prescan_high_water = fixture.crud.node_high_water(fixture.tenant);
    let held = BoltHeldTxn::new(owned);
    let ctx =
        ExecutionContext::new(fixture.tenant, PartitionId::ZERO).with_held_txn(Box::new(held));

    // Barrier of 2: main (scanner) + committer. Both rendezvous; the
    // committer waits on `scan_started` (set right before the scanner
    // drains) so the commits OVERLAP the fan-out.
    let start = Barrier::new(2);
    let scan_started = AtomicBool::new(false);

    let rows = std::thread::scope(|scope| {
        let committer = scope.spawn(|| {
            start.wait();
            concurrent_commit_burst(
                fixture,
                sacrificial,
                during_inserts,
                counter_base,
                &scan_started,
            );
        });

        // Release the committer and begin the scan on this thread at the
        // same rendezvous point.
        start.wait();
        let op = build_op();
        scan_started.store(true, Ordering::Release);
        let rows = drain_all_rows(op, &ctx, &fixture.substrate);

        committer.join().expect("committer thread panicked");
        rows
    });

    // Reclaim + abort the held txn (read-only; we only used its snapshot).
    if let Some(mut held) = ctx.take_held_txn() {
        if let Some(owned) = held
            .as_any_mut()
            .downcast_mut::<BoltHeldTxn>()
            .and_then(BoltHeldTxn::take_owned)
        {
            owned.abort();
        }
    }
    (rows, prescan_high_water)
}

/// The DISTINCT committed node ids currently visible at a fresh snapshot
/// (sanity for the seed / concurrent-insert accounting).
fn live_ids(fixture: &JepsenArcqlFixture) -> Vec<u64> {
    let tx = fixture.mgr.begin(fixture.tenant);
    let hw = fixture.crud.node_high_water(fixture.tenant);
    (1..=hw)
        .filter(|&raw| matches!(crud::read_node(&tx, NodeId::new(raw)), Ok(Some(_))))
        .collect()
}

/// **T1 — the load-bearing gate.** Parallel morsel-driven scan ≡ serial
/// scan against the REAL `CrudExecutorSubstrate`, under a forced
/// multi-morsel fan-out, WHILE a second thread commits mid-scan in-place
/// updates + new-node inserts. `parallel_result == serial_result` over the
/// seeded id range.
#[test]
fn parallel_scan_equals_serial_under_concurrent_commit_multimorsel() {
    // Hold the env lock for the whole test body so no sibling env-mutating
    // test observes our forced `ARCGRAPH_SCAN_*` vars, and keep the env
    // guard alive so the vars are restored when the body exits.
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _env = ScanEnvGuard::force_multi_morsel_parallel();

    let fixture = JepsenArcqlFixture::new();
    let (mgr, crud, tenant) = (
        Arc::clone(&fixture.mgr),
        Arc::clone(&fixture.crud),
        fixture.tenant,
    );

    // ── Seed the COMPARED range: SEED_NODES nodes (ids 1..=SEED_NODES),
    //    each with a distinct starting `counter` = its id. Then apply a
    //    fixed set of in-place UPDATES to the first 16 (counter → id+100)
    //    so the compared frozen view is non-trivial (a scan that dropped
    //    property re-reads would miss them → the counter equality has real
    //    teeth). All committed serially + joined, so the seeded prefix is
    //    FIXED before either scan begins. ──────────────────────────────
    let mut seeded_nodes: Vec<NodeId> = Vec::with_capacity(SEED_NODES as usize);
    for i in 1..=SEED_NODES {
        let mut tx = mgr.begin(tenant);
        let node = crud::create_node(
            &crud,
            &mut tx,
            tenant,
            label(),
            &counter_property_data(i as i64),
        )
        .expect("seed create");
        crud::commit(tx, &crud).expect("seed commit");
        seeded_nodes.push(node);
    }
    assert_eq!(seeded_nodes.len(), SEED_NODES as usize);
    for (idx, &node) in seeded_nodes.iter().take(16).enumerate() {
        let mut tx = mgr.begin(tenant);
        let id = idx as i64 + 1;
        crud::update_node(&crud, &mut tx, node, &counter_property_data(id + 100))
            .expect("pre-scan update");
        crud::commit(tx, &crud).expect("pre-scan update commit");
    }
    assert_eq!(
        live_ids(&fixture).len(),
        SEED_NODES as usize,
        "exactly the seeded nodes are visible pre-scan (no strays yet)"
    );

    // ── Sacrificial DISJOINT nodes (ids > SEED_NODES) the concurrent
    //    committer will UPDATE during each scan. They are never in the
    //    compared seeded projection, so their during-scan mutation cannot
    //    perturb the equality. ──────────────────────────────────────────
    let mut sacrificial: Vec<NodeId> = Vec::with_capacity(SACRIFICIAL_NODES as usize);
    for s in 0..SACRIFICIAL_NODES {
        let mut tx = mgr.begin(tenant);
        let node = crud::create_node(
            &crud,
            &mut tx,
            tenant,
            label(),
            &counter_property_data(s as i64),
        )
        .expect("sacrificial create");
        crud::commit(tx, &crud).expect("sacrificial commit");
        sacrificial.push(node);
    }
    let live_before_scans = live_ids(&fixture).len();
    assert_eq!(
        live_before_scans, PRESCAN_HIGH_WATER as usize,
        "seed + sacrificial visible before the scans (= PRESCAN_HIGH_WATER)"
    );

    // Each scan builds its OWN held-txn-pinned context internally (the
    // snapshot is frozen pre-burst inside the helper) and returns the
    // high-water captured at that instant — its per-run leak threshold. The
    // two runs share the fixture, so the SECOND run's threshold is HIGHER
    // (the first run's during-inserts are now committed + pre-existing).

    // ── Run the SERIAL scan (the oracle) under a concurrent-commit race. ─
    let (serial_full, serial_hw) = run_scan_under_concurrent_commits(
        &fixture,
        &sacrificial,
        DURING_INSERTS,
        /* counter_base = */ 500,
        || PhysicalOperator::Scan(ScanOp::new(binding0(), Some(label()), Lsn::MAX)),
    );

    // ── Run the PARALLEL (morsel-driven) scan under the SAME shape of
    //    race (another deterministic during-burst). The compared seeded
    //    prefix is identical to the serial run's, so the frozen view is
    //    identical. ─────────────────────────────────────────────────────
    let (parallel_full, parallel_hw) = run_scan_under_concurrent_commits(
        &fixture,
        &sacrificial,
        DURING_INSERTS,
        /* counter_base = */ 900,
        || PhysicalOperator::ParallelScan(ParallelScanOp::new(binding0(), Some(label()), Lsn::MAX)),
    );

    let serial_rows = seeded(&serial_full);
    let parallel_rows = seeded(&parallel_full);

    // ── The gate assertions ──────────────────────────────────────────

    // (1) The multi-morsel path genuinely engaged: the compared prefix has
    //     > FORCED_MORSEL_SIZE rows, so the parallel scan fanned across
    //     ≥ 2 morsels (not the single-morsel degenerate path). The full
    //     frozen buffer is even larger (+ sacrificial + prior inserts), so
    //     morsel count is strictly greater still.
    assert!(
        parallel_rows.len() > FORCED_MORSEL_SIZE,
        "multi-morsel path must engage: {} compared rows over morsel size {}",
        parallel_rows.len(),
        FORCED_MORSEL_SIZE
    );

    // (2) THE gate: parallel ≡ serial under concurrent commit. Same rows,
    //     same order, same in-place counter values — no torn / dropped /
    //     duped / reordered row across the morsel fan-out.
    assert_eq!(
        parallel_rows, serial_rows,
        "parallel morsel scan diverged from serial oracle under concurrent commit \
         (materialize-before-fan / one-frozen-snapshot invariant broken)"
    );

    // (3) THE materialize-before-fan LEAK GATE (R1 #1378): NO
    //     during-scan-INSERTED id (`> this run's prescan high-water`) may
    //     appear in the FULL scan output. Each scan reads through a held
    //     snapshot pinned (with the high-water captured) strictly BEFORE its
    //     concurrent-commit burst, so under correct materialize-before-fan +
    //     one-frozen-snapshot the owned `Vec` — materialized fully before the
    //     fan — CANNOT contain a freshly-inserted id. A BROKEN impl that
    //     moved the substrate scan INTO the per-morsel closure as a FRESH
    //     `begin` (a morsel making a substrate call — the exact "morsels make
    //     zero substrate calls" violation) would read the LIVE snapshot
    //     mid-burst and catch a just-inserted id `> high-water`, TRIPPING this
    //     assertion. This is the positive, flake-free proof that the equality
    //     in (2) is actually GATED ON the seam (the disjoint-region equality
    //     alone is insensitive to it — the compared seeded range is immutable
    //     during the scan).
    for &(id, _) in &parallel_full {
        assert!(
            id <= parallel_hw,
            "materialize-before-fan LEAK: id {id} (> the parallel run's prescan \
             high-water {parallel_hw}) was inserted DURING the scan yet appeared in \
             the PARALLEL scan buffer — indicates a per-morsel re-scan from a fresh \
             `begin`, breaking the one-frozen-snapshot / materialize-before-fan invariant"
        );
    }
    // The serial oracle must also be leak-free (same one-frozen-snapshot
    // property; a serial scan that leaked would be an equally real defect).
    for &(id, _) in &serial_full {
        assert!(
            id <= serial_hw,
            "one-frozen-snapshot LEAK: id {id} (> the serial run's prescan high-water \
             {serial_hw}) inserted during the scan appeared in the SERIAL buffer"
        );
    }

    // (4) Non-vacuity for the leak gate: the frozen buffer is EXACTLY the
    //     pre-scan population (every labelled id `1..=high-water`). Proves
    //     the scan materialized every pre-existing node (no drops) AND
    //     excluded ALL during-inserts — the leak gate is non-trivially
    //     satisfied (not vacuous because the buffer was empty). All nodes
    //     carry the single `label()`, so the labelled population == the
    //     high-water.
    assert_eq!(
        parallel_full.len(),
        parallel_hw as usize,
        "parallel frozen buffer must be exactly the pre-scan population (1..=high-water)"
    );
    assert_eq!(
        serial_full.len(),
        serial_hw as usize,
        "serial frozen buffer must be exactly the pre-scan population (1..=high-water)"
    );
    // The parallel run began AFTER the serial run's burst committed, so its
    // high-water is strictly greater — confirms the runs share the fixture
    // and the per-run thresholds are correct (not a stale global constant).
    assert!(
        parallel_hw > serial_hw,
        "parallel run's prescan high-water ({parallel_hw}) must exceed the serial \
         run's ({serial_hw}) — the serial burst's inserts are committed by then"
    );

    // (5) Non-vacuity: the compared seeded view is EXACTLY the seeded range
    //     (no drops) with no dupes + strict ascending order across morsels.
    assert_eq!(
        serial_rows.len(),
        SEED_NODES as usize,
        "compared frozen view must be exactly the seeded range"
    );
    let mut ids: Vec<u64> = parallel_rows.iter().map(|(id, _)| *id).collect();
    ids.dedup();
    assert_eq!(
        ids.len(),
        SEED_NODES as usize,
        "no duplicate ids across morsels (a duped row would inflate this)"
    );
    assert!(
        parallel_rows.windows(2).all(|w| w[0].0 < w[1].0),
        "morsel concatenation must preserve strict ascending id order"
    );

    // (6) Non-vacuity: the 16 pre-scan in-place UPDATES are reflected in
    //     the frozen view, so the counter equality in (2) is load-bearing
    //     (a scan that ignored the updated property versions would fail).
    let updated_seen = parallel_rows
        .iter()
        .filter(|(id, ctr)| *id <= 16 && *ctr == Some(*id as i64 + 100))
        .count();
    assert_eq!(
        updated_seen, 16,
        "all 16 pre-scan counter updates must be visible in the frozen view"
    );

    // (7) The concurrent committer really did commit its during-scan
    //     INSERTS (the race was live, not skipped): live count grew past
    //     the pre-scan state by both during-bursts' DURING_INSERTS each. If
    //     the race had been skipped, (3)'s leak gate would be vacuously true
    //     — this proves the inserts genuinely happened (and were correctly
    //     EXCLUDED by the frozen snapshot, per (3)+(4)).
    let final_live = live_ids(&fixture).len();
    assert_eq!(
        final_live,
        live_before_scans + 2 * DURING_INSERTS as usize,
        "both during-scan insert bursts committed (concurrent race was live)"
    );
}
