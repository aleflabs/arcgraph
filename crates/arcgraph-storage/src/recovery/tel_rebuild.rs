//! P0 #780 — cold-start TEL **adjacency** rebuild.
//!
//! Sibling of [`crate::recovery::stats_rebuild`]. Where `stats_rebuild`
//! repopulates the per-tenant `CatalogStats` cardinality counters after
//! WAL replay, THIS module repopulates the per-tenant in-memory TEL
//! adjacency chains (`CrudStore::tel_chains` / `reverse_tel_chains`) that
//! `scan_out` / `scan_in` walk to serve `MATCH ()-[r]->()` traversals.
//!
//! # The bug (#780)
//!
//! After a durable `--data` restart, node data + intern names recover
//! (#776 / #782) and the relationship RECORDS recover into the MVCC +
//! record stores (the CommitBundle carries them). But the TEL adjacency
//! index does **not** participate in the CommitBundle — its MVCC↔TEL
//! atomicity gap is issue #20 (`crud::commit` §"Drain TEL appends AFTER
//! commit") — and `CrudStore::tel_append` had **no replay caller**. So
//! after recovery the adjacency chains are empty, `scan_out` yields
//! nothing, and `MATCH ()-[r]->() RETURN count(r)` reads **0** of N
//! durably-committed relationships. A database that loses acknowledged
//! relationships on restart is not GA-ready.
//!
//! The rel record surviving (so the typed query returns `Ok(0)`, not a
//! `-32005` name error) while traversal counts read 0 is exactly the
//! shape the #782 forward-pin test
//! (`durable_intern_restart_776::relationship_traversal_recovery_is_forward_pinned_780`)
//! documented and deferred to this PR.
//!
//! # The fix
//!
//! Mirror `stats_rebuild`: AFTER `recover_from_wal` reinstates the rel
//! records into the MVCC store, walk each tenant's MVCC-visible
//! relationships at the recovered LSN and, for each, call the
//! recovery-time analogue of the live commit drain
//! (`CrudStore::reinstate_rel_adjacency`, which performs the identical
//! forward `CrudStore::tel_append` + reverse `tel_append_reverse` with
//! the identical `channel = LabelId::new(ty.raw())` projection). This
//! reinstates BOTH the forward and reverse adjacency so out-edge
//! (`Direction::LeftToRight`) and in-edge (`RightToLeft` / `Undirected`)
//! expands traverse correctly post-restart.
//!
//! # Architecture (same posture as `stats_rebuild`, ADR-038
//! amendment-06 §Context / locked invariant I-Q17)
//!
//! 1. Runs SYNCHRONOUSLY at recovery time, AFTER `recover_from_wal`
//!    completes (so the MVCC chains exist) and AFTER / alongside
//!    `rebuild_all_tenant_stats`, BEFORE the first user query.
//! 2. Walks the recovered MVCC primary store at the recovered LSN via
//!    [`TxnManager::for_each_visible_record`] — the SAME per-tenant walk
//!    `stats_rebuild` uses (the chain-index fast path from issue #238).
//!    Tombstoned-at-recovery rels are skipped automatically: the walk
//!    only yields the version visible at `recovered_lsn`, and a delete is
//!    a `value: None` tombstone the walk does not surface. So a rel
//!    created then deleted pre-restart is NOT re-added to the TEL.
//! 3. Dispatches node vs. rel by the MVCC key tag bit
//!    ([`crate::crud::REL_TAG_BIT`]); only rel keys are reinstated.
//! 4. Per-tenant fault-isolated (ADR-038 amendment-06 §2.5.1): the
//!    per-tenant walk is wrapped in `catch_unwind`; a panic mid-walk is
//!    SWALLOWED, the tenant is marked [`AdjacencyRebuildOutcome::PartialFailure`],
//!    and other tenants' rebuilds are unaffected.
//!
//! # Why `recovered_lsn` is the reinstated entry's visibility LSN
//!
//! The reinstated [`arcgraph_core::TelEntry`] carries `created_lsn =
//! recovered_lsn` (= `applied_commit_lsn`). This is correct for every
//! reachable post-recovery read: the visible watermark resumes at
//! `applied_commit_lsn`, so every reader snapshot is `>= recovered_lsn`.
//! An entry stamped at the watermark is thus
//! visible at every reachable snapshot, while the MVCC kernel probe in
//! `scan_out` (`tx.read(rel_mvcc_key(..))`) remains the authoritative
//! tombstone / visibility filter. Using a single coalesced LSN (rather
//! than threading each rel's original commit LSN, which `for_each_visible_record`
//! does not expose) is the honest statement of the rebuild contract:
//! "every relationship live at recovery is traversable from the
//! recovery point forward."

use std::panic::{AssertUnwindSafe, catch_unwind};

use arcgraph_core::{Lsn, TenantId};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::crud::{CrudStore, REL_TAG_BIT, decode_rel_bytes};
use crate::transaction::TxnManager;

/// Outcome of a per-tenant cold-start adjacency rebuild. Mirrors
/// [`crate::recovery::TenantRebuildOutcome`] for the TEL path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjacencyRebuildOutcome {
    /// Rebuild ran cleanly to completion. `rels_reinstated` is the number
    /// of relationships re-appended into the forward (and, when the
    /// reverse index is enabled, reverse) adjacency chains.
    Success {
        /// Number of relationship records reinstated into the TEL.
        rels_reinstated: u64,
    },
    /// Rebuild panicked mid-walk. The tenant is marked `recovery_failed`;
    /// admin remediation is required (per amendment-06 §2.5.1). Other
    /// tenants' rebuilds are unaffected.
    PartialFailure {
        /// Captured panic message.
        panic_message: String,
    },
}

/// Aggregate report from [`rebuild_all_tenant_adjacency`]. Mirrors
/// [`crate::recovery::RebuildReport`] for the TEL path. Both lists are
/// sorted by raw `TenantId` for deterministic iteration.
#[derive(Debug, Clone, Default)]
pub struct AdjacencyRebuildReport {
    /// Tenants whose rebuilds completed successfully, each carrying the
    /// per-tenant `rels_reinstated` count.
    pub successful: Vec<(TenantId, u64)>,
    /// Tenants whose rebuilds panicked mid-walk, each carrying the
    /// captured panic message.
    pub failed: Vec<(TenantId, String)>,
}

impl AdjacencyRebuildReport {
    /// Total number of tenants walked (success + failure).
    #[must_use]
    pub fn tenants_walked(&self) -> usize {
        self.successful.len() + self.failed.len()
    }

    /// Sum of relationships reinstated across all successful per-tenant
    /// rebuilds. Failed tenants do not contribute (their walks were
    /// truncated at the panic point).
    #[must_use]
    pub fn total_rels_reinstated(&self) -> u64 {
        self.successful.iter().map(|(_, r)| *r).sum()
    }
}

/// Rebuild a single tenant's TEL adjacency from the recovered MVCC state
/// at `recovered_lsn`.
///
/// Walks every MVCC key in `tenant`'s slice whose latest version visible
/// at `recovered_lsn` is a live relationship record, decodes it, and
/// reinstates the forward + reverse adjacency via
/// `CrudStore::reinstate_rel_adjacency`. Node keys (tag bit clear) are
/// skipped — they carry no adjacency.
///
/// Decode failures (should be impossible for records committed via
/// `crud::commit`'s codec, but defended against) and per-edge reinstate
/// errors log a `tracing::warn!` and skip that record; they do NOT fail
/// the rebuild. A panic mid-walk is caught and surfaced as
/// [`AdjacencyRebuildOutcome::PartialFailure`] (the panic is swallowed,
/// per amendment-06 §D-25.1 step 2 / §2.5.1).
pub fn rebuild_adjacency_for_tenant(
    tenant: TenantId,
    recovered_lsn: Lsn,
    txn_mgr: &TxnManager,
    store: &CrudStore,
) -> AdjacencyRebuildOutcome {
    let mut rels_reinstated: u64 = 0;
    let mut rels_failed: u64 = 0;

    let reinstated_ref = &mut rels_reinstated;
    let failed_ref = &mut rels_failed;

    // catch_unwind for per-tenant fault isolation (mirrors
    // `stats_rebuild::rebuild_catalog_stats_for_tenant`). AssertUnwindSafe
    // is sound: `&CrudStore` mutates only DashMap entries + Mutex-guarded
    // TEL chains (both unwind-safe), `&TxnManager` is read-only here, and
    // the local counters are `u64` (Copy).
    let walk_result = catch_unwind(AssertUnwindSafe(|| {
        txn_mgr.for_each_visible_record(tenant, recovered_lsn, |key, bytes| {
            // Only relationships carry adjacency. `key & REL_TAG_BIT == 0`
            // is a node — no TEL entry to reinstate.
            if key & REL_TAG_BIT == 0 {
                return;
            }
            match decode_rel_bytes(bytes) {
                Ok(rec) => {
                    // Reinstate forward + reverse adjacency for this
                    // recovered rel at the recovered watermark (see module
                    // rustdoc for the visibility-LSN rationale).
                    match store.reinstate_rel_adjacency(tenant, &rec, recovered_lsn) {
                        Ok(()) => *reinstated_ref += 1,
                        Err(e) => {
                            *failed_ref += 1;
                            tracing::warn!(
                                tenant_raw = tenant.raw(),
                                rel_id = rec.id,
                                error = ?e,
                                "#780 cold-start TEL rebuild: reinstate failed for rel; \
                                 edge will not be traversable until next write",
                            );
                        }
                    }
                }
                Err(e) => {
                    *failed_ref += 1;
                    tracing::warn!(
                        tenant_raw = tenant.raw(),
                        error = ?e,
                        recovered_lsn_raw = recovered_lsn.raw(),
                        "#780 cold-start TEL rebuild: decode failure for rel record; \
                         adjacency not reinstated for this entry",
                    );
                }
            }
        });
    }));

    match walk_result {
        Ok(()) => {
            if rels_failed > 0 {
                tracing::error!(
                    tenant_raw = tenant.raw(),
                    rels_reinstated,
                    rels_failed,
                    "#780 cold-start TEL rebuild: completed with per-edge failures \
                     (some relationships not traversable post-restart)",
                );
            }
            AdjacencyRebuildOutcome::Success { rels_reinstated }
        }
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<&'static str>()
                .copied()
                .map(str::to_string)
                .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            tracing::error!(
                tenant_raw = tenant.raw(),
                recovered_lsn_raw = recovered_lsn.raw(),
                panic_message = %msg,
                "#780 cold-start TEL rebuild: panic mid-walk; tenant marked recovery_failed \
                 (per ADR-038 amendment-06 §2.5.1). Other tenants' rebuilds unaffected.",
            );
            AdjacencyRebuildOutcome::PartialFailure { panic_message: msg }
        }
    }
}

/// Walk every tenant in the recovered MVCC state and rebuild its TEL
/// adjacency in parallel. Per-tenant fault-isolated (a panic in tenant
/// T's rebuild does not affect U, V, …).
///
/// Mirrors [`crate::recovery::rebuild_all_tenant_stats`]: drives
/// [`TxnManager::tenants_with_chains`] (a deterministic `Vec<TenantId>`
/// sorted by raw id) through [`rayon::iter::IntoParallelIterator`].
/// Per-tenant TEL chains are keyed by `(tenant, …)` so parallel tenants
/// touch disjoint chains — no cross-tenant contention, no concurrent
/// append to the same block (the per-tenant walk is single-threaded).
///
/// Both `successful` and `failed` lists are sorted by raw `TenantId`
/// after collection so two consecutive invocations on the same
/// (`recovered_lsn`, `tenants`, `store`) return identically-ordered
/// reports.
pub fn rebuild_all_tenant_adjacency(
    recovered_lsn: Lsn,
    txn_mgr: &TxnManager,
    store: &CrudStore,
) -> AdjacencyRebuildReport {
    let tenants = txn_mgr.tenants_with_chains();
    let mut outcomes: Vec<(TenantId, AdjacencyRebuildOutcome)> = tenants
        .into_par_iter()
        .map(|tenant| {
            let outcome = rebuild_adjacency_for_tenant(tenant, recovered_lsn, txn_mgr, store);
            (tenant, outcome)
        })
        .collect();
    outcomes.sort_by_key(|(tenant, _)| tenant.raw());

    let mut report = AdjacencyRebuildReport::default();
    for (tenant, outcome) in outcomes {
        match outcome {
            AdjacencyRebuildOutcome::Success { rels_reinstated } => {
                report.successful.push((tenant, rels_reinstated));
            }
            AdjacencyRebuildOutcome::PartialFailure { panic_message } => {
                report.failed.push((tenant, panic_message));
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};

    use super::*;
    use crate::crud::{
        CrudStore, PropertyData, commit, create_node, create_rel, scan_in, scan_out,
    };
    use crate::transaction::TxnManager;

    /// Build a store with a live edge `(src)-[ty]->(dst)`, then DROP all
    /// TEL state (simulating a restart where rel records survive in MVCC
    /// but adjacency chains are empty), and prove the rebuild reinstates
    /// it so `scan_out` traverses again.
    ///
    /// We model the post-restart state directly: a FRESH `CrudStore`
    /// (empty TEL) whose `TxnManager` has the rel records reinstated into
    /// MVCC (what WAL replay produces). The rebuild must repopulate the
    /// adjacency from those records alone.
    #[allow(clippy::too_many_arguments)]
    fn install_rel_into_mvcc(
        mgr: &TxnManager,
        tenant: TenantId,
        rel: RelId,
        ty: TypeId,
        src: NodeId,
        dst: NodeId,
        commit_lsn: u64,
    ) {
        use crate::crud::rel_mvcc_key;
        use bytes::Bytes;
        let rec = arcgraph_core::RelRecord::new(rel, ty, src, dst, Lsn::new(commit_lsn));
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(commit_lsn),
            tenant,
            rel_mvcc_key(rel),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        mgr.seed_after_replay(Lsn::new(commit_lsn));
    }

    #[test]
    fn rebuild_reinstates_forward_adjacency_from_recovered_rels() {
        // Post-restart shape: rel records live in MVCC, TEL is empty.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let src = NodeId::new(1);
        let mid = NodeId::new(2);
        let dst = NodeId::new(3);
        let ty = TypeId::new(7);
        // Two edges: (1)-[7]->(2), (2)-[7]->(3).
        install_rel_into_mvcc(&mgr, tenant, RelId::new(10), ty, src, mid, 100);
        install_rel_into_mvcc(&mgr, tenant, RelId::new(11), ty, mid, dst, 101);

        let recovered_lsn = mgr.current_lsn();

        // Pre-rebuild: TEL is empty → scan_out yields nothing.
        {
            let tx = mgr.begin(tenant);
            assert_eq!(
                scan_out(&store, &tx, src, None).count(),
                0,
                "pre-rebuild adjacency must be empty (the #780 bug)"
            );
        }

        // Rebuild.
        let report = rebuild_all_tenant_adjacency(recovered_lsn, &mgr, &store);
        assert_eq!(report.total_rels_reinstated(), 2);
        assert!(report.failed.is_empty());

        // Post-rebuild: scan_out traverses both edges.
        let tx = mgr.begin(tenant);
        let out_src: Vec<_> = scan_out(&store, &tx, src, None).collect();
        assert_eq!(out_src.len(), 1, "src 1 has one out-edge after rebuild");
        assert_eq!(out_src[0].dst_id, mid.raw());
        assert_eq!(out_src[0].rel_id, 10);

        let out_mid: Vec<_> = scan_out(&store, &tx, mid, None).collect();
        assert_eq!(out_mid.len(), 1, "node 2 has one out-edge after rebuild");
        assert_eq!(out_mid[0].dst_id, dst.raw());

        // Type-filtered scan also resolves the channel projection.
        let out_typed: Vec<_> = scan_out(&store, &tx, src, Some(ty)).collect();
        assert_eq!(out_typed.len(), 1);

        // REVERSE adjacency (scan_in) is rebuilt too: node `mid` (2) has an
        // in-edge from `src` (1) via rel 10. Reverse entries store the original
        // SRC in the `dst_id` field (per ADR-131 reverse-chain semantics).
        let in_mid = scan_in(&store, &tx, mid, None).expect("reverse index enabled");
        assert_eq!(in_mid.len(), 1, "node 2 has one in-edge after rebuild");
        assert_eq!(in_mid[0].dst_id, src.raw(), "in-edge of 2 originates at 1");
        assert_eq!(in_mid[0].rel_id, 10);
        // Node `dst` (3) has one in-edge from `mid` (2) via rel 11.
        let in_dst = scan_in(&store, &tx, dst, None).expect("reverse index enabled");
        assert_eq!(in_dst.len(), 1);
        assert_eq!(in_dst[0].dst_id, mid.raw());
        assert_eq!(in_dst[0].rel_id, 11);
        // Node `src` (1) has NO in-edge (it is only a source).
        let in_src = scan_in(&store, &tx, src, None).expect("reverse index enabled");
        assert_eq!(in_src.len(), 0, "node 1 is a pure source — no in-edges");
    }

    #[test]
    fn rebuild_skips_tombstoned_rels() {
        // A rel created at lsn 100 then DELETED (tombstoned) at lsn 200.
        // The walk at recovered_lsn=200 sees the tombstone (value=None),
        // so the rebuild must NOT reinstate it.
        use crate::crud::rel_mvcc_key;
        use bytes::Bytes;
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let src = NodeId::new(1);

        let rec = arcgraph_core::RelRecord::new(
            RelId::new(10),
            TypeId::new(7),
            src,
            NodeId::new(2),
            Lsn::new(100),
        );
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(100),
            tenant,
            rel_mvcc_key(RelId::new(10)),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        // Tombstone (delete).
        let _ =
            mgr.apply_replay_mvcc_write(Lsn::new(200), tenant, rel_mvcc_key(RelId::new(10)), None);
        mgr.seed_after_replay(Lsn::new(200));

        let report = rebuild_all_tenant_adjacency(Lsn::new(200), &mgr, &store);
        assert_eq!(
            report.total_rels_reinstated(),
            0,
            "a rel deleted before recovery must not be reinstated into the TEL"
        );
        let tx = mgr.begin(tenant);
        assert_eq!(scan_out(&store, &tx, src, None).count(), 0);
    }

    #[test]
    fn rebuild_roundtrip_through_live_commit_matches_in_session() {
        // Stronger oracle: build a real edge through the LIVE commit path
        // (which populates the TEL), snapshot the in-session scan_out
        // result, then rebuild a SECOND fresh store from the SAME recovered
        // MVCC and assert the adjacency is equivalent (same dst/rel set).
        let mgr = TxnManager::new();
        let live = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let a = create_node(
            &live,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let b = create_node(
            &live,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let ty = TypeId::new(5);
        let rel = create_rel(&live, &mut tx, tenant, a, b, ty, &PropertyData::Empty).unwrap();
        commit(tx, &live).unwrap();

        // In-session (live TEL) out-edges of `a`.
        let live_tx = mgr.begin(tenant);
        let live_out: Vec<(u64, u64)> = scan_out(&live, &live_tx, a, None)
            .map(|e| (e.dst_id, e.rel_id))
            .collect();
        assert_eq!(live_out, vec![(b.raw(), rel.raw())]);
        drop(live_tx);

        // Rebuild a fresh store from the same recovered MVCC.
        let recovered = CrudStore::new();
        let report = rebuild_all_tenant_adjacency(mgr.current_lsn(), &mgr, &recovered);
        assert_eq!(report.total_rels_reinstated(), 1);

        let rec_tx = mgr.begin(tenant);
        let rebuilt_out: Vec<(u64, u64)> = scan_out(&recovered, &rec_tx, a, None)
            .map(|e| (e.dst_id, e.rel_id))
            .collect();
        assert_eq!(
            rebuilt_out, live_out,
            "rebuilt adjacency must match the live in-session adjacency"
        );
    }

    /// P0 #812 (DURABLE variant): a supernode whose rel records survive a
    /// restart in MVCC must rebuild its FULL adjacency — past the
    /// overflow boundary. The rebuild path
    /// ([`CrudStore::reinstate_rel_adjacency`]) drives the same
    /// `tel_append` growth+overflow logic as the live commit drain, so
    /// the `grown()`-drops-prev bug manifested identically on recovery:
    /// a 5000-edge supernode rebuilt to only the still-growing newest
    /// block (~906 edges) and `scan_out`/`MATCH ()-[r]->()` under-counted
    /// after every durable restart. Strong oracle: exact (rel_id -> dst)
    /// set round-trips through the rebuild. RED pre-fix.
    #[test]
    fn rebuild_reinstates_supernode_past_overflow_boundary() {
        const FANOUT: u64 = 5000; // > 2 × MAX_ENTRIES (4094): three blocks.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let hub = NodeId::new(1);
        let ty = TypeId::new(7);

        // Post-restart shape: rel records in MVCC, TEL empty. rel ids
        // 100.. and dst ids 10_000.. keep them disjoint from the hub.
        let mut expected: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
        for i in 0..FANOUT {
            let rel = RelId::new(100 + i);
            let dst = NodeId::new(10_000 + i);
            install_rel_into_mvcc(&mgr, tenant, rel, ty, hub, dst, 100 + i);
            expected.insert(rel.raw(), dst.raw());
        }
        let recovered_lsn = mgr.current_lsn();

        let report = rebuild_all_tenant_adjacency(recovered_lsn, &mgr, &store);
        assert_eq!(
            report.total_rels_reinstated(),
            FANOUT,
            "every recovered rel must be reinstated into the rebuilt TEL"
        );
        assert!(report.failed.is_empty(), "no rel may fail reinstatement");

        let tx = mgr.begin(tenant);
        let out: Vec<_> = scan_out(&store, &tx, hub, Some(ty)).collect();
        assert_eq!(
            out.len(),
            FANOUT as usize,
            "rebuilt supernode must expose ALL out-edges (no overflow drop on recovery)"
        );
        let got: std::collections::BTreeMap<u64, u64> =
            out.iter().map(|e| (e.rel_id, e.dst_id)).collect();
        assert_eq!(got, expected, "every rebuilt edge maps to its original dst");

        // Reverse adjacency is rebuilt symmetrically: one arbitrary dst
        // has exactly one in-edge originating at the hub.
        let probe_dst = NodeId::new(10_000);
        let in_edges = scan_in(&store, &tx, probe_dst, Some(ty)).expect("reverse index enabled");
        assert_eq!(
            in_edges.len(),
            1,
            "each leaf dst has one in-edge from the hub"
        );
        assert_eq!(in_edges[0].dst_id, hub.raw());
    }

    /// #835 hardening (DURABLE reverse-overflow coverage gap): a REVERSE
    /// supernode — one SINK with `FANOUT` distinct sources all pointing at
    /// it — whose rel records survive a restart in MVCC must rebuild its
    /// FULL *reverse* adjacency PAST the overflow boundary, so
    /// `scan_in(sink)` reads ALL N inbound edges.
    ///
    /// Closes the exact coverage gap the #835 report named ("Stage-2 …
    /// exercised reverse LINKAGE, not reverse OVERFLOW"): the sibling
    /// `rebuild_reinstates_supernode_past_overflow_boundary` exercises a
    /// FORWARD overflow chain but its reverse probe is trivial (each leaf
    /// dst has ONE in-edge), so the durable REVERSE overflow chain
    /// (>2 × MAX_ENTRIES) was previously untested on the rebuild path.
    ///
    /// Verdict at current `main` (`aedfecb0`, with #826): GREEN — the
    /// reinstate path (`reinstate_rel_adjacency` → `tel_append_reverse`)
    /// builds the reverse overflow chain correctly, and `scan_in` walks
    /// it fully. This is therefore a regression GUARD (it RED-flips only
    /// if a future change re-breaks the durable reverse re-link), NOT a
    /// fix for the #835 customer-reported symptom — which the
    /// investigation localized to the reverse-direction READ path through
    /// the MCP/query stack, NOT the storage TEL (the same rels read via
    /// FORWARD expand return in full; storage `scan_in` returns all N in
    /// every storage config — in-mem, dual-write, full-WAL+CDC). See the
    /// PR body for the full bisect.
    /// Strong oracle: the exact source set round-trips through the
    /// rebuilt reverse chain.
    #[test]
    fn rebuild_reinstates_reverse_supernode_past_overflow_boundary() {
        const FANOUT: u64 = 5000; // > 2 × MAX_ENTRIES (4094): three blocks.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let sink = NodeId::new(1);
        let ty = TypeId::new(7);

        // Post-restart shape: rel records in MVCC, TEL empty. All FANOUT
        // rels point at the SAME sink (reverse fan-in supernode); rel ids
        // 100.. and src ids 10_000.. keep them disjoint from the sink.
        let mut expected_srcs: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for i in 0..FANOUT {
            let rel = RelId::new(100 + i);
            let src = NodeId::new(10_000 + i);
            install_rel_into_mvcc(&mgr, tenant, rel, ty, src, sink, 100 + i);
            expected_srcs.insert(src.raw());
        }
        let recovered_lsn = mgr.current_lsn();

        let report = rebuild_all_tenant_adjacency(recovered_lsn, &mgr, &store);
        assert_eq!(
            report.total_rels_reinstated(),
            FANOUT,
            "every recovered rel must be reinstated into the rebuilt TEL"
        );
        assert!(report.failed.is_empty(), "no rel may fail reinstatement");

        let tx = mgr.begin(tenant);
        let in_edges = scan_in(&store, &tx, sink, Some(ty)).expect("reverse index enabled");
        assert_eq!(
            in_edges.len(),
            FANOUT as usize,
            "rebuilt reverse supernode must expose ALL in-edges (no reverse overflow drop on recovery)"
        );
        // Reverse entries store the ORIGINAL SRC in `dst_id` (ADR-131).
        let got_srcs: std::collections::BTreeSet<u64> = in_edges.iter().map(|e| e.dst_id).collect();
        assert_eq!(
            got_srcs, expected_srcs,
            "every inbound source preserved through the reverse rebuild"
        );
    }
}
