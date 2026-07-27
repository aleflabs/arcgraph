//! #1380 — cold-start PRIMARY + SECONDARY index reconciliation.
//!
//! Sibling of [`crate::recovery::stats_rebuild`] and
//! [`crate::recovery::tel_rebuild`]. Where `stats_rebuild` repopulates
//! the per-tenant `CatalogStats` cardinality counters after WAL replay,
//! and `tel_rebuild` repopulates the per-tenant TEL adjacency chains,
//! THIS module reconciles the per-`(tenant, kind, id)` PRIMARY index
//! (id → `(page, slot)`) and the per-node SECONDARY index (property →
//! node) from the recovered MVCC-visible records.
//!
//! # The bug (#1380 — dual-write split-brain)
//!
//! The live commit drain (`crud::commit`) commits the MVCC record in
//! Phase 1, then attempts the primary/secondary index install as a dual
//! write. Per ADR-023 an index-install FAILURE **degrades but does not
//! fail** the commit — the drain logs a `tracing::warn!` and `continue`s
//! (`crud.rs` ~line 3766, `Err(e) => // Per ADR-023: index install
//! failure degrades`). So under index pressure the MVCC record commits
//! and durifies, but its primary-id (and any secondary-label) index
//! entry is MISSING. The node is then SCAN-visible (MVCC is
//! authoritative — a full scan walks the version chain) yet PERMANENTLY
//! absent from `read_node_with_store`'s primary fast-path / secondary
//! label lookup: reachability ≠ readability by id.
//!
//! Worse, recovery previously rebuilt only stats (`stats_rebuild`) and
//! TEL adjacency (`tel_rebuild`) from MVCC — NOT the primary/secondary
//! index. So a warn-and-continue split-brain SURVIVED restart forever: a
//! node stuck in "scannable but never id/label-lookup-able", including
//! on every existing corrupt data-dir. A database that permanently loses
//! id-lookup of an acknowledged, durably-committed node is not GA-ready.
//!
//! # The fix
//!
//! Mirror `tel_rebuild`: AFTER `recover_from_wal` reinstates the records
//! into the MVCC store, walk each tenant's MVCC-visible records at the
//! recovered LSN and, for each, call
//! `CrudStore::reinstate_record_index`, which re-installs the record
//! page + primary entry + (for nodes) secondary property entries IFF the
//! primary entry is missing. Records whose index entry is already present
//! (the overwhelming majority — every non-degraded commit) are a no-op.
//! This heals any warn-and-continue split-brain on restart, including on
//! existing corrupt data-dirs, and — because it re-derives from the
//! authoritative MVCC store every restart — stays healed.
//!
//! # Architecture (same posture as `stats_rebuild` / `tel_rebuild`,
//! ADR-038 amendment-06 §Context / locked invariant I-Q17)
//!
//! 1. Runs SYNCHRONOUSLY at recovery time, AFTER `recover_from_wal`
//!    completes (so the MVCC chains exist) and AFTER / alongside
//!    `rebuild_all_tenant_stats` + `rebuild_all_tenant_adjacency`, BEFORE
//!    the first user query.
//! 2. Walks the recovered MVCC primary store at the recovered LSN via
//!    [`TxnManager::for_each_visible_record_with_created_lsn`] — the SAME
//!    per-tenant walk the sibling rebuilds use (the chain-index fast path
//!    from issue #238), carrying the authoritative version LSN.
//!    Tombstoned-at-recovery records are skipped automatically: the walk
//!    only yields the version visible at `recovered_lsn`, and a delete is
//!    a `value: None` tombstone the walk does not surface. So a record
//!    created then deleted pre-restart is NOT re-installed.
//! 3. Both node and rel keys are reconciled (both live in the primary
//!    index); the kind is discriminated inside
//!    `CrudStore::reinstate_record_index` via the MVCC key tag bit.
//! 4. Per-tenant fault-isolated (ADR-038 amendment-06 §2.5.1): the
//!    per-tenant walk is wrapped in `catch_unwind`; a panic mid-walk is
//!    SWALLOWED, the tenant is marked [`IndexRebuildOutcome::PartialFailure`],
//!    and other tenants' rebuilds are unaffected.

use std::panic::{AssertUnwindSafe, catch_unwind};

use arcgraph_core::{Lsn, TenantId};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::crud::CrudStore;
use crate::transaction::TxnManager;

/// Outcome of a per-tenant cold-start index reconciliation. Mirrors
/// [`crate::recovery::TenantRebuildOutcome`] /
/// [`crate::recovery::AdjacencyRebuildOutcome`] for the index path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexRebuildOutcome {
    /// Reconciliation ran cleanly to completion.
    Success {
        /// Number of MVCC-visible records walked (nodes + rels).
        records_walked: u64,
        /// Number of records whose primary index entry was MISSING and
        /// got reinstated (the healed split-brain population). Zero on a
        /// data-dir with no degraded commits — the common case.
        records_reinstated: u64,
    },
    /// Reconciliation panicked mid-walk. The tenant is marked
    /// `recovery_failed`; admin remediation is required (per amendment-06
    /// §2.5.1). Other tenants' rebuilds are unaffected.
    PartialFailure {
        /// Captured panic message.
        panic_message: String,
    },
}

/// Aggregate report from [`rebuild_all_tenant_index`]. Mirrors
/// [`crate::recovery::RebuildReport`] /
/// [`crate::recovery::AdjacencyRebuildReport`] for the index path. Both
/// lists are sorted by raw `TenantId` for deterministic iteration.
#[derive(Debug, Clone, Default)]
pub struct IndexRebuildReport {
    /// Tenants whose reconciliations completed successfully, each carrying
    /// the per-tenant `(records_walked, records_reinstated)` counts.
    pub successful: Vec<(TenantId, u64, u64)>,
    /// Tenants whose reconciliations panicked mid-walk, each carrying the
    /// captured panic message.
    pub failed: Vec<(TenantId, String)>,
}

impl IndexRebuildReport {
    /// Total number of tenants walked (success + failure).
    #[must_use]
    pub fn tenants_walked(&self) -> usize {
        self.successful.len() + self.failed.len()
    }

    /// Sum of records walked across all successful per-tenant
    /// reconciliations.
    #[must_use]
    pub fn total_records_walked(&self) -> u64 {
        self.successful.iter().map(|(_, w, _)| *w).sum()
    }

    /// Sum of records whose primary index entry was reinstated (the
    /// healed split-brain population) across all successful per-tenant
    /// reconciliations. Non-zero ONLY on a data-dir that suffered a
    /// warn-and-continue index degrade (#1380).
    #[must_use]
    pub fn total_records_reinstated(&self) -> u64 {
        self.successful.iter().map(|(_, _, r)| *r).sum()
    }
}

/// Reconcile a single tenant's primary + secondary index from the
/// recovered MVCC state at `recovered_lsn`.
///
/// Walks every MVCC key in `tenant`'s slice whose latest version visible
/// at `recovered_lsn` is a live record, and calls
/// `CrudStore::reinstate_record_index` for each. That call is a no-op
/// when the primary entry is already present (the normal case); it
/// re-installs the record page + primary entry + (for nodes) secondary
/// property entries when the entry is MISSING (the #1380 split-brain).
///
/// Per-record reinstall errors (should be impossible for records
/// committed via `crud::commit`'s codec, but defended against) log a
/// `tracing::warn!` and skip that record; they do NOT fail the
/// reconciliation. A panic mid-walk is caught and surfaced as
/// [`IndexRebuildOutcome::PartialFailure`] (the panic is swallowed, per
/// amendment-06 §D-25.1 step 2 / §2.5.1).
pub fn rebuild_index_for_tenant(
    tenant: TenantId,
    recovered_lsn: Lsn,
    txn_mgr: &TxnManager,
    store: &CrudStore,
) -> IndexRebuildOutcome {
    let mut records_walked: u64 = 0;
    let mut records_reinstated: u64 = 0;
    let mut records_failed: u64 = 0;

    let walked_ref = &mut records_walked;
    let reinstated_ref = &mut records_reinstated;
    let failed_ref = &mut records_failed;

    // catch_unwind for per-tenant fault isolation (mirrors
    // `stats_rebuild` / `tel_rebuild`). AssertUnwindSafe is sound:
    // `&CrudStore` mutates only DashMap entries + Mutex/latch-guarded
    // pages (all unwind-safe), `&TxnManager` is read-only here, and the
    // local counters are `u64` (Copy).
    let walk_result = catch_unwind(AssertUnwindSafe(|| {
        // #1616: take the visibility LSN from the MVCC version, not the
        // payload's canonical ZERO placeholder.
        txn_mgr.for_each_visible_record_with_created_lsn(
            tenant,
            recovered_lsn,
            |key, bytes, created_lsn| {
                *walked_ref += 1;
                match store.reinstate_record_index(tenant, key, bytes, created_lsn) {
                    Ok(true) => *reinstated_ref += 1,
                    Ok(false) => {}
                    Err(e) => {
                        *failed_ref += 1;
                        tracing::warn!(
                            tenant_raw = tenant.raw(),
                            error = ?e,
                            recovered_lsn_raw = recovered_lsn.raw(),
                            "#1380 cold-start index reconcile: reinstate failed for record; \
                             id/label lookup for this record will miss until next write",
                        );
                    }
                }
            },
        );
    }));

    match walk_result {
        Ok(()) => {
            if records_failed > 0 {
                tracing::error!(
                    tenant_raw = tenant.raw(),
                    records_reinstated,
                    records_failed,
                    "#1380 cold-start index reconcile: completed with per-record failures \
                     (some records not id/label-lookup-able post-restart)",
                );
            }
            if records_reinstated > 0 {
                // A non-zero reinstall count means this data-dir suffered a
                // warn-and-continue index degrade (#1380) that we just
                // healed. Surface it at info so operators see the heal.
                tracing::info!(
                    tenant_raw = tenant.raw(),
                    records_reinstated,
                    records_walked,
                    "#1380 cold-start index reconcile: healed dual-write split-brain \
                     (records that were scan-visible but missing from the primary/secondary \
                     index have been reinstalled from MVCC)",
                );
            }
            IndexRebuildOutcome::Success {
                records_walked,
                records_reinstated,
            }
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
                "#1380 cold-start index reconcile: panic mid-walk; tenant marked recovery_failed \
                 (per ADR-038 amendment-06 §2.5.1). Other tenants' reconciliations unaffected.",
            );
            IndexRebuildOutcome::PartialFailure { panic_message: msg }
        }
    }
}

/// Walk every tenant in the recovered MVCC state and reconcile its
/// primary + secondary index in parallel. Per-tenant fault-isolated (a
/// panic in tenant T's reconciliation does not affect U, V, …).
///
/// Mirrors [`crate::recovery::rebuild_all_tenant_stats`] /
/// [`crate::recovery::rebuild_all_tenant_adjacency`]: drives
/// [`TxnManager::tenants_with_chains`] (a deterministic `Vec<TenantId>`
/// sorted by raw id) through [`rayon::iter::IntoParallelIterator`].
/// Per-tenant primary/secondary index keys are `(tenant, …)`-scoped so
/// parallel tenants touch disjoint keys — no cross-tenant contention.
///
/// Both `successful` and `failed` lists are sorted by raw `TenantId`
/// after collection so two consecutive invocations on the same
/// (`recovered_lsn`, `tenants`, `store`) return identically-ordered
/// reports.
pub fn rebuild_all_tenant_index(
    recovered_lsn: Lsn,
    txn_mgr: &TxnManager,
    store: &CrudStore,
) -> IndexRebuildReport {
    let tenants = txn_mgr.tenants_with_chains();
    let mut outcomes: Vec<(TenantId, IndexRebuildOutcome)> = tenants
        .into_par_iter()
        .map(|tenant| {
            let outcome = rebuild_index_for_tenant(tenant, recovered_lsn, txn_mgr, store);
            (tenant, outcome)
        })
        .collect();
    outcomes.sort_by_key(|(tenant, _)| tenant.raw());

    let mut report = IndexRebuildReport::default();
    for (tenant, outcome) in outcomes {
        match outcome {
            IndexRebuildOutcome::Success {
                records_walked,
                records_reinstated,
            } => {
                report
                    .successful
                    .push((tenant, records_walked, records_reinstated));
            }
            IndexRebuildOutcome::PartialFailure { panic_message } => {
                report.failed.push((tenant, panic_message));
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
    use bytes::Bytes;

    use super::*;
    use crate::crud::{
        CrudStore, PropertyData, commit, create_node, node_mvcc_key, read_node_with_store,
        rel_mvcc_key,
    };
    use crate::page_alloc::PageAllocator;
    use crate::primary_index::{PageSlot, PrimaryIndex, PrimaryKey, RecordKind};
    use crate::transaction::TxnManager;

    // ─────────────────────────────────────────────────────────────────
    // Fixtures. We model the #1380 split-brain directly: a dual-write
    // `CrudStore` whose MVCC store has a record reinstated (what WAL
    // replay produces) but whose primary index has NO entry for it (what
    // the warn-and-continue degrade leaves). The reconcile pass must
    // re-install the primary entry so id lookup finds the record again.
    //
    // NOTE ON THE SECONDARY LEG. The concrete secondary index lives in
    // `arcgraph-index` (which depends on `-storage`, never the reverse —
    // bounded-context, `docs/bounded-contexts.md`), so a REAL secondary
    // index cannot be constructed from an `arcgraph-storage` unit test.
    // The secondary reinstall leg of the #1380 oracle is therefore
    // covered by the sibling integration test
    // `crates/arcgraph-index/tests/index_reconcile_secondary_1380.rs`,
    // which wires a real `SecondaryIndex` into `new_with_indices` and
    // asserts the label/property lookup finds the healed node. The unit
    // tests here cover the PRIMARY leg (both directions), rel reconcile,
    // idempotency, survives-restart, tombstone-skip, and the vacuous
    // dual-write-disabled path.
    // ─────────────────────────────────────────────────────────────────

    /// Build a dual-write store (primary index + record store, no
    /// secondary) and return the shared `TxnManager` + `Arc<PrimaryIndex>`
    /// so tests can probe the primary index directly.
    fn build_dual_write() -> (Arc<TxnManager>, CrudStore, Arc<PrimaryIndex>) {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let primary =
            Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
        let store = CrudStore::new_with_index(None, Arc::clone(&primary), Arc::clone(&alloc));
        (txn_mgr, store, primary)
    }

    fn primary_lookup(primary: &PrimaryIndex, pk: PrimaryKey) -> Option<PageSlot> {
        primary.lookup(pk).expect("primary lookup must not error")
    }

    /// Install a node record directly into MVCC (what WAL replay does),
    /// WITHOUT touching the primary/secondary index — modelling the
    /// #1380 warn-and-continue degrade where the MVCC side committed but
    /// the index install was skipped. Always installs under
    /// [`TenantId::DEFAULT`] (single-tenant is sufficient for the
    /// split-brain oracle; cross-tenant fault isolation is inherited from
    /// the sibling `tel_rebuild` / `stats_rebuild` parallel drivers).
    ///
    /// #1616 fixture shape: the payload carries the canonical `Lsn::ZERO`
    /// placeholder and `commit_lsn` goes only onto the MVCC version,
    /// matching `crud::commit`'s v8 / non-delta path. Stamping the commit
    /// LSN into both places makes the fixture richer than production and
    /// hollows the reinstate-LSN oracle.
    fn install_node_into_mvcc_only(
        mgr: &TxnManager,
        node: NodeId,
        label: u32,
        prop_a: u32,
        prop_b: u32,
        commit_lsn: u64,
    ) {
        let mut rec = arcgraph_core::NodeRecord::new(node, LabelId::new(label), Lsn::ZERO);
        rec.inline_u32a = prop_a;
        rec.inline_u32b = prop_b;
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(commit_lsn),
            TenantId::DEFAULT,
            node_mvcc_key(node),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        mgr.seed_after_replay(Lsn::new(commit_lsn));
    }

    /// Relationship sibling of [`install_node_into_mvcc_only`]: zero in
    /// the payload, authoritative `commit_lsn` on the MVCC version.
    fn install_rel_into_mvcc_only(
        mgr: &TxnManager,
        rel: RelId,
        ty: u32,
        src: NodeId,
        dst: NodeId,
        commit_lsn: u64,
    ) {
        let rec = arcgraph_core::RelRecord::new(rel, TypeId::new(ty), src, dst, Lsn::ZERO);
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(commit_lsn),
            TenantId::DEFAULT,
            rel_mvcc_key(rel),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        mgr.seed_after_replay(Lsn::new(commit_lsn));
    }

    /// THE ORACLE (#1380, primary leg): a node whose MVCC record is
    /// present but whose primary index entry is ABSENT (the
    /// warn-and-continue split-brain) must, after the recovery reconcile
    /// pass, be found by the primary (id) lookup AND materialise via
    /// `read_node_with_store`'s id fast-path.
    ///
    /// RED-on-revert: emptying `reinstate_record_index`'s body to
    /// `Ok(false)` (or removing the `rebuild_all_tenant_index` call) →
    /// the primary lookup returns not-found post-recovery → the
    /// `primary_lookup(...).is_some()` assertion FAILS. Verbatim both
    /// ways: reconcile → lookup-finds; revert → lookup-missing.
    #[test]
    fn reconcile_heals_missing_primary_from_mvcc() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;

        let node = NodeId::new(42);
        let label = 7u32;
        // Post-degrade shape: MVCC record present, index empty.
        install_node_into_mvcc_only(&mgr, node, label, 100, 200, 500);
        let recovered_lsn = mgr.current_lsn();

        // Pre-reconcile: primary lookup MISSES (the #1380 split-brain).
        let pk = PrimaryKey::new(tenant, RecordKind::Node, node.raw());
        assert!(
            primary_lookup(&primary, pk).is_none(),
            "pre-reconcile the split-brained node must be ABSENT from the primary index"
        );

        // Reconcile.
        let report = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert!(report.failed.is_empty(), "no tenant may fail reconcile");
        assert_eq!(
            report.total_records_reinstated(),
            1,
            "the split-brained node must be reinstated exactly once"
        );

        // Post-reconcile: PRIMARY (id) lookup FINDS the node. This is the
        // assertion that goes RED when the reconcile pass is reverted.
        assert!(
            primary_lookup(&primary, pk).is_some(),
            "post-reconcile primary (id) lookup must FIND the healed node (#1380 oracle)"
        );

        // And the id fast-path materialises the correct record.
        let tx = mgr.begin(tenant);
        let rec = read_node_with_store(&store, &tx, node)
            .expect("read must not error")
            .expect("post-reconcile the node must be readable by id");
        assert_eq!(rec.id, node.raw());
        assert_eq!(rec.label_id, label);
        assert_eq!(
            rec.created_lsn, 500,
            "#1616: the reinstalled slot carries the MVCC version's committed LSN"
        );
    }

    /// #1616 — the reinstalled slot's `created_lsn` must come from the
    /// MVCC version, not from the record payload.
    ///
    /// RED-on-revert: derive `created_lsn` from `bytes` inside
    /// `reinstate_record_index` and this reads back a slot stamped `0`
    /// instead of `777`.
    #[test]
    fn reinstated_slot_takes_created_lsn_from_version_not_payload() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;
        let node = NodeId::new(1234);
        let commit_lsn = 777_u64;

        let mut rec = arcgraph_core::NodeRecord::new(node, LabelId::new(3), Lsn::ZERO);
        rec.inline_u32a = 11;
        rec.inline_u32b = 22;
        assert_eq!(
            rec.created_lsn, 0,
            "fixture precondition: payload carries the ZERO placeholder"
        );
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(commit_lsn),
            tenant,
            node_mvcc_key(node),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        mgr.seed_after_replay(Lsn::new(commit_lsn));

        let pk = PrimaryKey::new(tenant, RecordKind::Node, node.raw());
        assert!(primary_lookup(&primary, pk).is_none());

        let report = rebuild_all_tenant_index(mgr.current_lsn(), &mgr, &store);
        assert!(report.failed.is_empty());
        assert_eq!(report.total_records_reinstated(), 1);

        let tx = mgr.begin(tenant);
        let healed = read_node_with_store(&store, &tx, node)
            .expect("read must not error")
            .expect("healed node must be readable by id");
        assert_eq!(
            healed.created_lsn, commit_lsn,
            "#1616: the reinstalled slot must carry the MVCC version's created_lsn"
        );
    }

    /// Rels are reconciled too: a rel record present in MVCC but missing
    /// from the primary index must be found by id lookup post-reconcile.
    #[test]
    fn reconcile_heals_missing_rel_primary_from_mvcc() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;
        let rel = RelId::new(99);
        install_rel_into_mvcc_only(&mgr, rel, 5, NodeId::new(1), NodeId::new(2), 600);
        let recovered_lsn = mgr.current_lsn();

        let pk = PrimaryKey::new(tenant, RecordKind::Rel, rel.raw());
        assert!(primary_lookup(&primary, pk).is_none());

        let report = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert_eq!(report.total_records_reinstated(), 1);
        assert!(
            primary_lookup(&primary, pk).is_some(),
            "post-reconcile rel primary (id) lookup must FIND the healed rel"
        );
    }

    /// No-regression + idempotency: a node committed through the LIVE
    /// dual-write path (index ALREADY present) is a NO-OP on reconcile —
    /// zero reinstalls, and the primary entry is unchanged. Running the
    /// reconcile a SECOND time is also a no-op (idempotent — re-installing
    /// a present entry is never a dup).
    #[test]
    fn reconcile_is_noop_for_normally_committed_node_and_idempotent() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;

        // LIVE commit — populates MVCC + primary atomically.
        let mut tx = mgr.begin(tenant);
        let node = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(3),
            &PropertyData::Empty,
        )
        .expect("create_node");
        commit(tx, &store).expect("commit");
        let recovered_lsn = mgr.current_lsn();

        let pk = PrimaryKey::new(tenant, RecordKind::Node, node.raw());
        let slot_before =
            primary_lookup(&primary, pk).expect("normally-committed node is already indexed");

        // First reconcile: the entry is present → NO reinstall.
        let r1 = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert_eq!(
            r1.total_records_reinstated(),
            0,
            "a normally-committed node must NOT be reinstated (idempotency: present entry is \
             a no-op)"
        );
        assert_eq!(
            primary_lookup(&primary, pk),
            Some(slot_before),
            "the present primary entry must be UNCHANGED by a no-op reconcile (no dup slot)"
        );

        // Second reconcile: still a no-op.
        let r2 = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert_eq!(
            r2.total_records_reinstated(),
            0,
            "re-reconcile is idempotent"
        );
        assert_eq!(
            primary_lookup(&primary, pk),
            Some(slot_before),
            "second reconcile leaves the entry unchanged"
        );
    }

    /// Reconcile of a healed node PERSISTS across a second restart: after
    /// the first reconcile installs the entry, a second recovery pass sees
    /// it present and re-reconciles cleanly (zero reinstalls the second
    /// time), and the node stays id-lookup-able.
    #[test]
    fn reconciled_index_survives_a_second_restart() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;
        let node = NodeId::new(77);
        install_node_into_mvcc_only(&mgr, node, 4, 11, 22, 700);
        let recovered_lsn = mgr.current_lsn();

        // First restart: heals the split-brain.
        let r1 = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert_eq!(r1.total_records_reinstated(), 1);
        let pk = PrimaryKey::new(tenant, RecordKind::Node, node.raw());
        assert!(primary_lookup(&primary, pk).is_some());

        // Second restart: entry already present → zero reinstalls, still
        // id-lookup-able (the heal is durable across restarts).
        let r2 = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
        assert_eq!(
            r2.total_records_reinstated(),
            0,
            "the reconciled index must survive a second restart (re-reconcile is a no-op)"
        );
        assert!(
            primary_lookup(&primary, pk).is_some(),
            "post-second-restart the healed node stays id-lookup-able"
        );
    }

    /// A record created then TOMBSTONED before recovery must NOT be
    /// reinstated (the walk only yields the visible version; a delete is a
    /// `value: None` tombstone the walk skips).
    #[test]
    fn reconcile_skips_tombstoned_record() {
        let (mgr, store, primary) = build_dual_write();
        let tenant = TenantId::DEFAULT;
        let node = NodeId::new(88);

        // Create at lsn 100.
        let rec = arcgraph_core::NodeRecord::new(node, LabelId::new(1), Lsn::new(100));
        let _ = mgr.apply_replay_mvcc_write(
            Lsn::new(100),
            tenant,
            node_mvcc_key(node),
            Some(Bytes::copy_from_slice(&rec.to_bytes())),
        );
        // Tombstone at lsn 200.
        let _ = mgr.apply_replay_mvcc_write(Lsn::new(200), tenant, node_mvcc_key(node), None);
        mgr.seed_after_replay(Lsn::new(200));

        let report = rebuild_all_tenant_index(Lsn::new(200), &mgr, &store);
        assert_eq!(
            report.total_records_reinstated(),
            0,
            "a node deleted before recovery must not be reinstated into the index"
        );
        let pk = PrimaryKey::new(tenant, RecordKind::Node, node.raw());
        assert!(
            primary_lookup(&primary, pk).is_none(),
            "the tombstoned node must remain absent from the primary index"
        );
    }

    /// A store with dual-write DISABLED (no primary index) reconciles
    /// vacuously: the report walks records but reinstalls nothing (there
    /// is no index to reconcile into).
    #[test]
    fn reconcile_is_vacuous_when_dual_write_disabled() {
        let mgr = TxnManager::new();
        let store = CrudStore::new(); // no primary / no records
        install_node_into_mvcc_only(&mgr, NodeId::new(1), 1, 0, 0, 100);

        let report = rebuild_all_tenant_index(mgr.current_lsn(), &mgr, &store);
        assert!(report.failed.is_empty());
        assert_eq!(
            report.total_records_reinstated(),
            0,
            "no primary index means nothing to reconcile — vacuous success"
        );
        // The walk still visited the record.
        assert_eq!(report.total_records_walked(), 1);
    }
}
