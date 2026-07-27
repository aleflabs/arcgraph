//! Issue #247 Phase 4.2 — per-tenant fault isolation under parallel
//! cold-start rebuild (per ADR-038 amendment-06 §2.5.1).
//!
//! # What this test pins
//!
//! The composition of two pre-existing primitives MUST tolerate a
//! per-tenant panic without:
//!
//! 1. propagating the panic across the rayon worker boundary into
//!    other workers' rebuilds OR the parent thread,
//! 2. violating the per-tenant SeqLock invariant
//!    (`commits_started == commits_observed`) for ANY tenant,
//! 3. surfacing the failed tenant as anything other than
//!    `TenantRebuildOutcome::PartialFailure { panic_message: <msg> }`,
//! 4. blocking other tenants' rebuilds.
//!
//! The two primitives:
//!
//! - **The 4-invariant SeqLock pattern** in
//!   `crate::recovery::stats_rebuild::rebuild_catalog_stats_for_tenant`
//!   (codified by `feedback_seqlock_panic_safety_primitive.md`):
//!   begin OUTSIDE catch_unwind → walk INSIDE `AssertUnwindSafe` →
//!   observe UNCONDITIONALLY OUTSIDE → panic SWALLOWED for per-tenant
//!   isolation.
//! - **Rayon's `into_par_iter().map(...).collect()`** which receives
//!   the per-tenant primitive's `TenantRebuildOutcome` value and
//!   never observes a propagated panic (the per-tenant primitive
//!   converts panics to `PartialFailure` before returning).
//!
//! # Why the test mirrors the production pattern instead of using it
//!
//! The production [`rebuild_catalog_stats_for_tenant`] is hard-wired
//! to a graceful walk that does NOT panic at the byte-decode boundary
//! (decode failures fall through to `tracing::warn!` per PR #170
//! reviewer Finding 3). To exercise the panic-and-recover composition
//! end-to-end, this test ships its own `parallel_rebuild_with_walk_hook`
//! harness that:
//!
//! - mirrors the production [`rebuild_all_tenant_stats`] driver (rayon
//!   ParIter + per-tenant SeqLock + outcome collection + sort),
//! - exposes an injectable per-tenant `walk_hook` invoked INSIDE the
//!   per-tenant catch_unwind boundary BEFORE the for-each walk,
//! - returns the production [`RebuildReport`] / [`TenantRebuildOutcome`]
//!   types so assertions inherit the production contract verbatim.
//!
//! The harness's SeqLock pattern is **byte-for-byte identical** to the
//! production primitive's; the test's `panic!` injection lives in the
//! `walk_hook` (which the production driver wires to a no-op closure).
//! This composition demonstrates that:
//!
//! - the SeqLock pattern's "observe runs unconditionally" invariant
//!   continues to hold under rayon's worker scheduling,
//! - rayon's worker pool does not surface a hidden panic-propagation
//!   path that would defeat the per-tenant isolation,
//! - the aggregate report's deterministic ordering is preserved
//!   even when N of K tenants surface as `PartialFailure`.
//!
//! # Phase 4.2 controlled-mutation discipline
//!
//! Per the spawn prompt's Phase 4.2:
//!
//! 1. inject `if tenant.raw() == 25 { panic!("REVIEWER PROBE: forced panic"); }`
//!    into the per-tenant walk closure,
//! 2. run K=50 stress; verify tenant 25 = `PartialFailure`, all 49
//!    others = `Success`, no propagated panic, SeqLock invariant
//!    holds for all 50,
//! 3. restore (no-op hook); re-run; verify all 50 succeed.
//!
//! Steps (1) + (2) are pinned by `K50_panic_at_tenant_25_isolated`.
//! Step (3) (the "restore" leg) is pinned by
//! `K50_no_panic_all_succeed` — running the SAME harness with a no-op
//! hook MUST produce 50 successful rebuilds. Both tests run as a pair
//! to demonstrate the panic-injection is the load-bearing variable.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;

use arcgraph_core::{LabelId, Lsn, NodeRecord, TenantId};
use arcgraph_storage::catalog::CatalogStats;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::recovery::stats_rebuild::{RebuildReport, TenantRebuildOutcome};
use arcgraph_storage::transaction::{ReplayApplyOutcome, TxnManager};
use bytes::Bytes;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// ──────────────────────────────────────────────────────────────────
// Workload setup (mirrors `tests/m4_41_chain_index_stress_K50_parallel.rs`
// at smaller N to keep the fault-isolation test within unit-test
// runtime).
// ──────────────────────────────────────────────────────────────────

const K_TENANTS: u64 = 50;
const N_PER_TENANT: u64 = 5_000; // 250K aggregate — fast enough for a fault-isolation test

fn canonical_node_bytes() -> &'static [u8; NodeRecord::SIZE] {
    use std::sync::OnceLock;
    static BYTES: OnceLock<[u8; NodeRecord::SIZE]> = OnceLock::new();
    BYTES.get_or_init(|| {
        let rec = NodeRecord::new(arcgraph_core::NodeId::new(1), LabelId::new(1), Lsn::new(1));
        rec.to_bytes()
    })
}

fn populate_tenant(mgr: &TxnManager, tenant_raw: u64, n: u64) {
    let t = TenantId::new(tenant_raw);
    let value = Bytes::from_static(canonical_node_bytes());
    let lsn_base = tenant_raw.saturating_mul(n);
    for k in 0..n {
        let commit_lsn = Lsn::new(lsn_base.saturating_add(k).saturating_add(1));
        let outcome = mgr.apply_replay_mvcc_write(commit_lsn, t, k, Some(value.clone()));
        debug_assert_eq!(outcome, ReplayApplyOutcome::Applied);
        let _ = outcome;
    }
}

fn build_seeded_manager(n_per_tenant: u64) -> Arc<TxnManager> {
    let mgr = Arc::new(TxnManager::new());
    let setup_start = Instant::now();
    let handles: Vec<_> = (0..K_TENANTS)
        .map(|raw| {
            let mgr = Arc::clone(&mgr);
            std::thread::spawn(move || populate_tenant(&mgr, raw, n_per_tenant))
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let max_lsn = Lsn::new(K_TENANTS.saturating_mul(n_per_tenant));
    mgr.seed_after_replay(max_lsn);
    eprintln!(
        "issue-247-fault-iso K={} N={} setup: {} chains in {:?}",
        K_TENANTS,
        n_per_tenant,
        K_TENANTS.saturating_mul(n_per_tenant),
        setup_start.elapsed()
    );
    mgr
}

// ──────────────────────────────────────────────────────────────────
// Test harness: parallel rebuild with injectable per-tenant walk hook.
//
// Mirrors `crate::recovery::stats_rebuild::rebuild_all_tenant_stats`
// + `rebuild_catalog_stats_for_tenant` shape verbatim, with one
// addition: a `walk_hook(tenant)` callback fires INSIDE the per-tenant
// catch_unwind boundary BEFORE the for-each walk. A panic in the hook
// is caught by the per-tenant SeqLock primitive's catch_unwind exactly
// as a panic in the walk closure would be — proving the production
// primitive's panic-safety contract composes correctly with rayon's
// parallel iteration.
// ──────────────────────────────────────────────────────────────────

fn parallel_rebuild_with_walk_hook<F>(
    recovered_lsn: Lsn,
    txn_mgr: &TxnManager,
    store: &CrudStore,
    walk_hook: F,
) -> RebuildReport
where
    F: Fn(TenantId) + Sync + Send,
{
    let tenants = txn_mgr.tenants_with_chains();
    let walk_hook_ref = &walk_hook;

    let mut outcomes: Vec<(TenantId, TenantRebuildOutcome)> = tenants
        .into_par_iter()
        .map(|tenant| {
            // ── 4-invariant SeqLock pattern (byte-for-byte mirror of
            //    `rebuild_catalog_stats_for_tenant`) ──
            //
            //  1. begin OUTSIDE catch_unwind
            let stats = store.init_catalog_stats(tenant);
            stats.begin_commit_observation();

            let mut nodes_walked: u64 = 0;
            let stats_ref: &CatalogStats = stats.as_ref();
            let nodes_ref = &mut nodes_walked;

            //  2. walk INSIDE AssertUnwindSafe — INCLUDING the
            //     injectable hook
            let walk_result = catch_unwind(AssertUnwindSafe(|| {
                walk_hook_ref(tenant);
                txn_mgr.for_each_visible_record(tenant, recovered_lsn, |_key, _bytes| {
                    // Every record in the harness uses the canonical
                    // shared NodeRecord bytes; we increment the
                    // total_nodes counter directly to mirror the
                    // production rebuild's per-record stat update,
                    // without needing the crate-private decode helper.
                    stats_ref.increment_label(LabelId::new(1));
                    stats_ref.increment_total_nodes();
                    *nodes_ref += 1;
                });
            }));

            //  3. observe UNCONDITIONALLY OUTSIDE
            stats.observe_commit();

            //  4. panic SWALLOWED for per-tenant isolation
            let outcome = match walk_result {
                Ok(()) => TenantRebuildOutcome::Success {
                    nodes_walked,
                    rels_walked: 0,
                },
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<&'static str>()
                        .copied()
                        .map(str::to_string)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "<non-string panic payload>".to_string());
                    TenantRebuildOutcome::PartialFailure { panic_message: msg }
                }
            };

            (tenant, outcome)
        })
        .collect();

    // Belt-and-braces deterministic ordering — mirrors production
    // `rebuild_all_tenant_stats`.
    outcomes.sort_by_key(|(tenant, _)| tenant.raw());

    let mut report = RebuildReport::default();
    for (tenant, outcome) in outcomes {
        match outcome {
            TenantRebuildOutcome::Success {
                nodes_walked,
                rels_walked,
            } => {
                report.successful.push((tenant, nodes_walked, rels_walked));
            }
            TenantRebuildOutcome::PartialFailure { panic_message } => {
                report.failed.push((tenant, panic_message));
            }
        }
    }
    report
}

// ──────────────────────────────────────────────────────────────────
// Phase 4.2 controlled-mutation: panic in tenant 25 under parallel
// execution.
// ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "stress test: requires release build, ~50 MB peak RSS"]
#[allow(non_snake_case)]
fn K50_panic_at_tenant_25_isolated() {
    let mgr = build_seeded_manager(N_PER_TENANT);
    let store = Arc::new(CrudStore::new());
    let recovered_lsn = mgr.current_lsn();

    let report = parallel_rebuild_with_walk_hook(recovered_lsn, &mgr, &store, |tenant| {
        if tenant.raw() == 25 {
            panic!("REVIEWER PROBE: forced panic");
        }
    });

    // ── Failure surface ──
    //
    // Tenant 25 surfaces as PartialFailure with the verbatim panic
    // message — the panic was caught inside the per-tenant primitive's
    // catch_unwind, NOT propagated across the rayon worker boundary.
    assert_eq!(
        report.failed.len(),
        1,
        "exactly one tenant must fail; got {}",
        report.failed.len()
    );
    let (failed_tenant, failed_msg) = &report.failed[0];
    assert_eq!(failed_tenant.raw(), 25);
    assert!(
        failed_msg.contains("REVIEWER PROBE: forced panic"),
        "panic_message must contain the injected sentinel: got {failed_msg:?}"
    );

    // ── Success surface ──
    //
    // All 49 other tenants completed normally. Each walked exactly
    // N_PER_TENANT records (the harness's canonical shared bytes
    // encode a single label, so `nodes_walked` equals the chain count
    // for each tenant).
    assert_eq!(
        report.successful.len(),
        (K_TENANTS - 1) as usize,
        "exactly K-1 tenants must succeed; got {}",
        report.successful.len()
    );
    for (tenant, nodes_walked, rels_walked) in &report.successful {
        assert_ne!(tenant.raw(), 25);
        assert_eq!(*nodes_walked, N_PER_TENANT);
        assert_eq!(*rels_walked, 0);
    }

    // ── Deterministic report ordering ──
    //
    // `successful` sorted by raw TenantId ascending; tenant 25 is
    // EXPECTED to be missing (it landed in `failed`).
    let success_order: Vec<u64> = report.successful.iter().map(|(t, _, _)| t.raw()).collect();
    let mut expected_order: Vec<u64> = (0..K_TENANTS).collect();
    expected_order.retain(|&r| r != 25);
    assert_eq!(success_order, expected_order);

    // ── SeqLock invariant for ALL 50 tenants ──
    //
    // For every tenant (including the panicked one), the per-tenant
    // SeqLock primitive's "observe runs unconditionally" contract must
    // hold: `commits_started == commits_observed == 1`. We assert this
    // BOTH directly (commits_observed_count) AND indirectly (snapshot()
    // returns cleanly — if commits_started > commits_observed, the
    // unbounded retry loop would spin forever).
    for raw in 0..K_TENANTS {
        let tenant = TenantId::new(raw);
        let stats = store
            .catalog_stats(tenant)
            .unwrap_or_else(|| panic!("tenant {raw} CatalogStats missing post-rebuild"));

        // Direct: commits_observed_count == 1 — observe_commit ran
        // exactly once for every tenant, including tenant 25 whose
        // walk panicked.
        assert_eq!(
            stats.commits_observed_count(),
            1,
            "tenant {raw}: observe_commit must have run unconditionally \
             (panic-tenant: {})",
            raw == 25,
        );

        // Indirect: snapshot() returns. If commits_started >
        // commits_observed (SeqLock invariant violated), this loops
        // forever; the test would hang. A clean return proves the
        // invariant holds.
        let snap = stats.snapshot();
        if raw != 25 {
            // Non-panicked tenant: full per-record walk completed; the
            // canonical shared bytes mean every record contributes one
            // increment to LabelId(1) + total_nodes.
            assert_eq!(snap.total_nodes(), Some(N_PER_TENANT));
            assert_eq!(snap.label_card(LabelId::new(1)), Some(N_PER_TENANT));
        } else {
            // Panicked tenant: panic fired BEFORE the for_each walk
            // (the hook runs at the top of the catch_unwind closure);
            // no per-record increments. Counters are zero but
            // observe_commit ran, so total_nodes == Some(0) (NOT None).
            assert_eq!(snap.total_nodes(), Some(0));
            // The cross-key invariant `sum(label_cards) ≤ total_nodes`
            // holds vacuously (both sides are 0).
            let sum_labels: u64 = snap.label_cards().iter().map(|(_, c)| *c).sum();
            assert_eq!(sum_labels, 0);
        }
    }

    eprintln!(
        "issue-247-fault-iso K=50 panic@25: report.failed={} report.successful={} \
         all 50 SeqLock invariants intact",
        report.failed.len(),
        report.successful.len()
    );
}

// ──────────────────────────────────────────────────────────────────
// Phase 4.2 step 3 — restore (no-op hook): all 50 succeed.
//
// Demonstrates the panic injection from the previous test is the
// LOAD-BEARING variable; with a no-op hook the same harness produces
// 50 successful rebuilds, identical to what the production
// `rebuild_all_tenant_stats` driver would.
// ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "stress test: requires release build, ~50 MB peak RSS"]
#[allow(non_snake_case)]
fn K50_no_panic_all_succeed() {
    let mgr = build_seeded_manager(N_PER_TENANT);
    let store = Arc::new(CrudStore::new());
    let recovered_lsn = mgr.current_lsn();

    let report = parallel_rebuild_with_walk_hook(recovered_lsn, &mgr, &store, |_tenant| {
        // No-op — restoration of step 3.
    });

    assert!(
        report.failed.is_empty(),
        "no tenant must fail: {:?}",
        report.failed
    );
    assert_eq!(report.successful.len() as u64, K_TENANTS);

    // Per-tenant snapshot consistency (cross-key invariant) on each
    // of the 50 tenants. The canonical shared bytes mean every
    // tenant's snapshot has `total_nodes == Some(N_PER_TENANT)` and
    // `label_card(1) == Some(N_PER_TENANT)`.
    for (tenant, nodes_walked, rels_walked) in &report.successful {
        assert_eq!(*nodes_walked, N_PER_TENANT);
        assert_eq!(*rels_walked, 0);

        let stats = store.catalog_stats(*tenant).unwrap();
        assert_eq!(stats.commits_observed_count(), 1);
        let snap = stats.snapshot();
        assert_eq!(snap.total_nodes(), Some(N_PER_TENANT));
        assert_eq!(snap.label_card(LabelId::new(1)), Some(N_PER_TENANT));
    }

    eprintln!(
        "issue-247-fault-iso K=50 restore: all {} tenants Success",
        report.successful.len()
    );
}

// ──────────────────────────────────────────────────────────────────
// Production-path smoke: the stress harness's `rebuild_all_tenant_stats`
// (the actual production driver) on the same shape MUST also produce
// 50 successful rebuilds. This locks the harness's pattern fidelity
// to the production code: any divergence between the two would surface
// as a divergent rebuild outcome.
// ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "stress test: requires release build, ~50 MB peak RSS"]
#[allow(non_snake_case)]
fn K50_production_driver_all_succeed() {
    use arcgraph_storage::recovery::stats_rebuild::rebuild_all_tenant_stats;

    let mgr = build_seeded_manager(N_PER_TENANT);
    let store = Arc::new(CrudStore::new());
    let recovered_lsn = mgr.current_lsn();

    let report = rebuild_all_tenant_stats(recovered_lsn, &mgr, &store);

    assert!(report.failed.is_empty());
    assert_eq!(report.successful.len() as u64, K_TENANTS);
    for (tenant, nodes_walked, rels_walked) in &report.successful {
        assert_eq!(*nodes_walked, N_PER_TENANT);
        assert_eq!(*rels_walked, 0);

        let stats = store.catalog_stats(*tenant).unwrap();
        let snap = stats.snapshot();
        assert_eq!(snap.total_nodes(), Some(N_PER_TENANT));
        assert_eq!(snap.label_card(LabelId::new(1)), Some(N_PER_TENANT));
    }

    eprintln!(
        "issue-247-fault-iso K=50 production driver: all {} tenants Success",
        report.successful.len()
    );
}
