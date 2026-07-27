//! #1380 — recovery-time index reconcile, SECONDARY leg.
//!
//! The primary leg of the #1380 oracle is unit-tested inside
//! `arcgraph-storage` (`recovery::index_rebuild::tests`). The SECONDARY
//! leg requires a REAL `SecondaryIndex` — which lives in `arcgraph-index`
//! (dependency graph is `index → storage`, never the reverse) — so it
//! is tested HERE, mirroring `crud_dual_write_secondary.rs`.
//!
//! # The bug (#1380)
//!
//! On a dual-write index install failure the live commit drain
//! warn-and-continues (ADR-023): the MVCC record commits but its
//! primary/secondary index entries are MISSING. Recovery previously
//! rebuilt only stats + TEL adjacency from MVCC — NOT the
//! primary/secondary index — so the split-brain survived restart: a node
//! SCAN-visible via MVCC but never found by a `label + property` lookup.
//!
//! # The oracle
//!
//! Model the post-degrade state directly: install a node record into MVCC
//! (what WAL replay produces) WITHOUT touching the primary/secondary
//! index (the degrade), then run the recovery reconcile pass and assert
//! BOTH the primary (id) lookup AND the SECONDARY (label + property)
//! lookup FIND the node. RED-on-revert (neuter `reinstate_record_index`):
//! the secondary lookup returns empty → the assertion FAILS.

use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, StringId, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudStore, INLINE_U32A_PROPERTY_KEY, INLINE_U32B_PROPERTY_KEY, node_mvcc_key,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::recovery::rebuild_all_tenant_index;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;

fn build_store() -> (
    Arc<TxnManager>,
    CrudStore,
    Arc<PrimaryIndex>,
    Arc<SecondaryIndex>,
) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store =
        CrudStore::new_with_indices(None, Arc::clone(&primary), Some(handle), Arc::clone(&alloc));
    (txn_mgr, store, primary, secondary)
}

fn sec_key(label: u32, pk: StringId, v: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        LabelId::new(label),
        pk,
        PropertyValue::U32(v),
    )
}

/// Install a node record into MVCC ONLY (no primary/secondary index) —
/// modelling the #1380 warn-and-continue degrade.
///
/// #1616 fixture shape: the payload carries the canonical `Lsn::ZERO`
/// placeholder and `commit_lsn` goes only onto the MVCC version, matching
/// `crud::commit`'s v8 / non-delta path. Stamping the LSN into both places
/// would hollow the reinstate-LSN oracle.
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
        Some(bytes::Bytes::copy_from_slice(&rec.to_bytes())),
    );
    mgr.seed_after_replay(Lsn::new(commit_lsn));
}

/// THE ORACLE (#1380, secondary leg): a node whose MVCC record is present
/// but whose primary AND secondary index entries are ABSENT must, after
/// the recovery reconcile pass, be found by BOTH the primary (id) lookup
/// AND the secondary (label + property) lookup.
#[test]
fn reconcile_heals_missing_secondary_from_mvcc() {
    let (mgr, store, primary, secondary) = build_store();
    let node = NodeId::new(42);
    let label = 7u32;
    let (prop_a, prop_b) = (100u32, 200u32);

    // Post-degrade shape: MVCC record present, index empty.
    install_node_into_mvcc_only(&mgr, node, label, prop_a, prop_b, 500);
    let recovered_lsn = mgr.current_lsn();

    // Pre-reconcile: BOTH primary and secondary MISS.
    let pk = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, node.raw());
    assert!(
        primary.lookup(pk).unwrap().is_none(),
        "pre-reconcile the split-brained node must be absent from the primary index"
    );
    assert!(
        secondary
            .lookup(sec_key(label, INLINE_U32A_PROPERTY_KEY, prop_a))
            .unwrap()
            .is_empty(),
        "pre-reconcile the split-brained node must be absent from the secondary index (the \
         #1380 bug — a label+property lookup finds nothing)"
    );

    // Reconcile.
    let report = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
    assert!(report.failed.is_empty());
    assert_eq!(
        report.total_records_reinstated(),
        1,
        "the split-brained node must be reinstated exactly once"
    );

    // Post-reconcile: PRIMARY (id) lookup FINDS the node.
    assert!(
        primary.lookup(pk).unwrap().is_some(),
        "post-reconcile primary (id) lookup must find the healed node"
    );

    // Post-reconcile: SECONDARY (label + property) lookup FINDS the node —
    // BOTH positional properties. This is the assertion that goes RED when
    // `reinstate_record_index` (or its secondary leg) is reverted.
    let hits_a = secondary
        .lookup(sec_key(label, INLINE_U32A_PROPERTY_KEY, prop_a))
        .unwrap();
    let hits_b = secondary
        .lookup(sec_key(label, INLINE_U32B_PROPERTY_KEY, prop_b))
        .unwrap();
    assert_eq!(
        hits_a,
        vec![node],
        "post-reconcile secondary lookup on inline_u32a must find the healed node (#1380 \
         oracle, secondary leg)"
    );
    assert_eq!(
        hits_b,
        vec![node],
        "post-reconcile secondary lookup on inline_u32b must find the healed node"
    );
}

/// No-regression: a node committed through the LIVE dual-write path (its
/// secondary entries ALREADY present) is a NO-OP on reconcile — zero
/// reinstalls, and the secondary lookup still resolves to exactly one
/// entry (not a duplicate).
#[test]
fn reconcile_is_noop_for_normally_committed_node_secondary() {
    use arcgraph_storage::crud::{PropertyData, commit, create_node};

    let (mgr, store, _primary, secondary) = build_store();

    // LIVE commit — populates MVCC + primary + secondary atomically.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let node = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(3),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    let recovered_lsn = mgr.current_lsn();

    // Pre-reconcile the secondary already resolves to the node.
    assert_eq!(
        secondary
            .lookup(sec_key(3, INLINE_U32A_PROPERTY_KEY, 11))
            .unwrap(),
        vec![node]
    );

    // Reconcile is a NO-OP (entry present).
    let report = rebuild_all_tenant_index(recovered_lsn, &mgr, &store);
    assert_eq!(
        report.total_records_reinstated(),
        0,
        "a normally-committed node must not be reinstated (idempotency)"
    );

    // The secondary lookup still resolves to EXACTLY the one node — no dup.
    assert_eq!(
        secondary
            .lookup(sec_key(3, INLINE_U32A_PROPERTY_KEY, 11))
            .unwrap(),
        vec![node],
        "no-op reconcile must not duplicate the secondary entry"
    );
}
