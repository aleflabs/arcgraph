//! v2 M2 A4 — the intern-durability RED-on-revert gate (L1 review,
//! provider-diverse leg; promoted from the QA scratch repro
//! `l1_m2_wrong_read_and_intern_order_scratch.rs`).
//!
//! # The defect this pins closed
//!
//! `intern_logged` publishes `(name, id)` in the shared table
//! (`intern_is_new`) BEFORE appending the `InternString` WAL record.
//! Pre-fix the append was gated on `was_new`, so a racing loser —
//! observing the winner's publish — skipped the log and could commit
//! a typed block referencing the id. Crash before the winner's append
//! ⇒ recovery has the acked node but NO intern binding ⇒ the unseeded
//! allocator re-hands the id to the next unrelated name ⇒ the old
//! committed property silently materializes under the WRONG name.
//!
//! # The fix (durable-logged-set protocol)
//!
//! `InternTable` carries a durable-proof set; `intern_logged` gates
//! the append on THAT (inserted only after the fsync-blocking append
//! returns), never on the in-memory publish. Losers re-append an
//! idempotent duplicate instead of trusting the publish (see
//! `arcgraph_storage::intern::intern_logged` rustdoc for the LSN-order
//! proof).
//!
//! # RED-on-revert
//!
//! Revert the gate in `intern_logged` from the durable-proof set back
//! to `was_new` and this test fails 100% deterministically: the loser
//! encoder emits no record, the fresh-table recovery cannot resolve
//! the id, and the bag read errors (or, post-reuse, materializes the
//! wrong name). No timing dependence — the "descheduled winner" is
//! simulated by publishing via `intern_is_new` directly and never
//! appending.

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_mcp::storage::property_payload::{
    properties_to_property_data_typed, record_property_bag_checked,
};
use arcgraph_query::executor::value::Value;
use arcgraph_storage::crud::{CrudStore, commit, create_node, read_node_with_store};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig,
    WalWriter, recover_from_wal,
};
use tempfile::tempdir;

fn wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(1),
        group_commit_max_batch: 1,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn wal_stack(dir: &std::path::Path) -> (WalWriter, Arc<TxnManager>, Arc<CrudStore>) {
    let writer = WalWriter::spawn(wal_config(dir)).expect("spawn WAL");
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone()))
            .expect("primary"),
    );
    let store = Arc::new(CrudStore::new_with_index(Some(handle), primary, alloc));
    (writer, mgr, store)
}

fn recover_stack(
    dir: &std::path::Path,
    intern: Arc<InternTable>,
) -> (Arc<TxnManager>, Arc<CrudStore>) {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let store = Arc::new(CrudStore::new_with_index(
        None,
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> =
        Arc::clone(store.records().expect("record store")) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_intern_table(intern);
    recover_from_wal(dir, Arc::clone(&mgr), target, None).expect("recover");
    (mgr, store)
}

/// The A4 gate. Split point: the winning thread has completed
/// `intern_is_new` (published id=1) but is "descheduled" before its
/// WAL append — simulated by never appending on the winner's behalf.
/// A racing production encoder then builds + commits a typed block
/// referencing the id. The commit is acknowledged; the process
/// "crashes" (writer shutdown); recovery starts from a FRESH
/// `InternTable`.
///
/// Post-fix contract asserted here:
/// 1. the loser's encode path itself appended the `InternString`
///    record (durable proof was absent), so recovery reconstructs the
///    binding — the acked commit's reference is never dangling;
/// 2. the acked node's bag materializes under the CORRECT name;
/// 3. the recovered allocator is seeded past the id — an unrelated
///    intern gets a FRESH id (no silent wrong-name reuse).
#[test]
fn racing_loser_commit_is_durable_before_it_references_a_fresh_intern() {
    let dir = tempdir().expect("tempdir");
    let tenant = TenantId::DEFAULT;
    let live_intern = Arc::new(InternTable::new());
    let (writer, mgr, store) = wal_stack(dir.path());

    // The winner: published, descheduled forever before its append.
    let (published_id, was_new) = live_intern.intern_is_new(tenant, "race_key").unwrap();
    assert!(was_new);

    // The racing loser: the production encoder. Under the durable-
    // logged-set protocol it must NOT trust the publish — it appends
    // the InternString record itself before the commit can reference
    // the id.
    let props = vec![("race_key".to_string(), Value::Integer(7))];
    let property_data =
        properties_to_property_data_typed(&props, &live_intern, store.wal(), tenant)
            .expect("racing encoder");
    let mut tx = mgr.begin(tenant);
    let node =
        create_node(&store, &mut tx, tenant, LabelId::new(0), &property_data).expect("create");
    commit(tx, &store).expect("referencing commit is acknowledged");
    writer.shutdown().expect("shutdown");

    // Crash-shaped recovery: FRESH InternTable (the pre-fix repro's
    // wrongness lived exactly here — no InternString record preceded
    // the acked commit).
    let recovered_intern = Arc::new(InternTable::new());
    let (recovered_mgr, recovered_store) = recover_stack(dir.path(), Arc::clone(&recovered_intern));

    // (1) The binding was reconstructed from the WAL.
    assert_eq!(
        recovered_intern
            .try_resolve(tenant, StringId::new(published_id.raw()))
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("race_key"),
        "the loser's own append made the binding durable before its commit (A4)",
    );

    // (2) The acked node materializes under the CORRECT name.
    let read_tx = recovered_mgr.begin(tenant);
    let record = read_node_with_store(&recovered_store, &read_tx, NodeId::new(node.raw()))
        .expect("read")
        .expect("acked node recovered");
    let bag = record_property_bag_checked(
        &record,
        recovered_store.blob_store(),
        &recovered_intern,
        tenant,
    )
    .expect("acked typed block resolves every referenced intern id");
    assert_eq!(bag.get("race_key"), Some(&Value::Integer(7)));
    assert_eq!(bag.len(), 1);

    // (3) The allocator was seeded by the replay — an unrelated intern
    // must NOT reuse the id (the pre-fix silent-wrong-name detonator).
    let unrelated = recovered_intern.intern(tenant, "unrelated_key").unwrap();
    assert_ne!(
        unrelated, published_id,
        "recovered allocator must be seeded past every replayed id",
    );
    let again = record_property_bag_checked(
        &record,
        recovered_store.blob_store(),
        &recovered_intern,
        tenant,
    )
    .expect("bag still reads");
    assert_eq!(
        again.get("race_key"),
        Some(&Value::Integer(7)),
        "the committed property never migrates to another name",
    );
    assert!(!again.contains_key("unrelated_key"));
}
