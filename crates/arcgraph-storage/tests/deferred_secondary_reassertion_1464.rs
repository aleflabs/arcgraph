//! #1464 — deferred secondary removals must not erase newer re-assertions.
//!
//! Production-shape gate: concrete primary + secondary stores, a real disk
//! WAL, public CRUD calls, a non-DEFAULT tenant, and Strict durability.  The
//! query helper mirrors the ADR-023 candidate-then-MVCC-verify contract so a
//! missing candidate is observed as the wrong-result false negative it is.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{DurabilityTier, LabelId, NodeId, TenantDurabilityLookup, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudStore, INLINE_U32A_PROPERTY_KEY, PropertyData, commit, create_node, read_node, update_node,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::TempDir;

const TENANT: TenantId = TenantId::new(1_464);

#[derive(Debug)]
struct Strict;

impl TenantDurabilityLookup for Strict {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Strict
    }
}

fn wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

struct StrictSecondaryStack {
    _dir: TempDir,
    writer: Option<WalWriter>,
    manager: Arc<TxnManager>,
    secondary: Arc<SecondaryIndex>,
    store: CrudStore,
}

impl StrictSecondaryStack {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let writer = WalWriter::spawn(wal_config(dir.path().to_path_buf())).unwrap();
        let wal = writer.handle();
        let mut manager = TxnManager::with_wal(wal.clone());
        manager.set_durability_lookup(Arc::new(Strict));
        let manager = Arc::new(manager);
        let allocator = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(
                Arc::clone(&manager),
                Arc::clone(&allocator),
                Some(wal.clone()),
            )
            .unwrap(),
        );
        let secondary = Arc::new(
            SecondaryIndex::new(
                Arc::clone(&manager),
                Arc::clone(&allocator),
                Some(wal.clone()),
            )
            .unwrap(),
        );
        let secondary_handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
        let store =
            CrudStore::new_with_indices(Some(wal), primary, Some(secondary_handle), allocator);
        Self {
            _dir: dir,
            writer: Some(writer),
            manager,
            secondary,
            store,
        }
    }

    fn create(&self, tenant: TenantId, label: LabelId, value: PropertyData) -> NodeId {
        let mut tx = self.manager.begin(tenant);
        let id = create_node(&self.store, &mut tx, tenant, label, &value).unwrap();
        commit(tx, &self.store).unwrap();
        id
    }

    fn update_inline_a(&self, tenant: TenantId, id: NodeId, value: u32) {
        let mut tx = self.manager.begin(tenant);
        update_node(
            &self.store,
            &mut tx,
            id,
            &PropertyData::InlineU32Pair(value, 0),
        )
        .unwrap();
        commit(tx, &self.store).unwrap();
    }

    /// Query the secondary candidate set and apply the mandatory ADR-023
    /// visibility + label + property equality recheck at one snapshot.
    fn query_inline_a(&self, tenant: TenantId, label: LabelId, value: u32) -> Vec<NodeId> {
        let key = SecondaryKey::new(
            tenant,
            label,
            INLINE_U32A_PROPERTY_KEY,
            PropertyValue::U32(value),
        );
        let reader = self.manager.begin(tenant);
        let mut found: Vec<NodeId> = self
            .secondary
            .lookup(key)
            .unwrap()
            .into_iter()
            .filter(|&id| {
                matches!(
                    read_node(&reader, id),
                    Ok(Some(rec))
                        if rec.label_id == label.raw() && rec.inline_u32a == value
                )
            })
            .collect();
        reader.abort();
        // PropertyIndexScan's public contract deduplicates candidates after
        // verification: the B-tree may retain both the original A slot and
        // the later A re-assertion slot, but the query returns one node row.
        found.sort_unstable_by_key(|id| id.raw());
        found.dedup();
        found
    }

    fn advance_horizon(&self) {
        self.create(TENANT, LabelId::new(9_999), PropertyData::Empty);
        assert_eq!(
            self.store.deferred_removal_queue_len(),
            0,
            "the horizon-advancing commit must fully drain ready removals"
        );
    }
}

impl Drop for StrictSecondaryStack {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            writer.shutdown().unwrap();
        }
    }
}

#[test]
fn strict_non_default_a_b_a_keeps_live_secondary_entry_after_full_drain() {
    const A: u32 = 101;
    const B: u32 = 202;
    let stack = StrictSecondaryStack::new();
    let label = LabelId::new(23);

    // Make the gate genuinely multi-tenant and make the two tenants' NodeIds
    // differ, so an accidental cross-tenant candidate cannot satisfy either
    // exact result oracle.
    let default_id = stack.create(TenantId::DEFAULT, label, PropertyData::InlineU32Pair(A, 0));
    stack.create(TENANT, LabelId::new(1), PropertyData::Empty);
    let id = stack.create(TENANT, label, PropertyData::InlineU32Pair(A, 0));
    assert_ne!(default_id, id);

    stack.update_inline_a(TENANT, id, B);
    stack.update_inline_a(TENANT, id, A);
    stack.advance_horizon();

    assert_eq!(
        stack.query_inline_a(TENANT, label, A),
        vec![id],
        "A→B→A must retain exactly the live A result after deferred removals drain"
    );
    assert_eq!(stack.query_inline_a(TENANT, label, B), Vec::<NodeId>::new());
    assert_eq!(
        stack.query_inline_a(TenantId::DEFAULT, label, A),
        vec![default_id],
        "the non-DEFAULT toggle must not disturb the DEFAULT tenant"
    );
}

#[test]
fn strict_non_default_a_b_c_removes_but_keeps_live_c_after_full_drain() {
    const A: u32 = 303;
    const B: u32 = 404;
    const C: u32 = 505;
    let stack = StrictSecondaryStack::new();
    let label = LabelId::new(29);

    let default_id = stack.create(TenantId::DEFAULT, label, PropertyData::InlineU32Pair(C, 0));
    stack.create(TENANT, LabelId::new(1), PropertyData::Empty);
    let id = stack.create(TENANT, label, PropertyData::InlineU32Pair(A, 0));
    assert_ne!(default_id, id);

    stack.update_inline_a(TENANT, id, B);
    stack.update_inline_a(TENANT, id, C);
    stack.advance_horizon();

    assert_eq!(stack.query_inline_a(TENANT, label, A), Vec::<NodeId>::new());
    assert_eq!(stack.query_inline_a(TENANT, label, B), Vec::<NodeId>::new());
    assert_eq!(
        stack.query_inline_a(TENANT, label, C),
        vec![id],
        "A→B→C must remove both stale values and retain exactly live C"
    );
    assert_eq!(
        stack.query_inline_a(TenantId::DEFAULT, label, C),
        vec![default_id]
    );
}
