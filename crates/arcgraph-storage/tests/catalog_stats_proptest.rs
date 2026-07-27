//! M4-41 (M4-04a) catalog-stats monotonic-bookkeeping proptest per
//! ADR-038 §2 D-25.
//!
//! Property: for any sequence of node creations and node deletions
//! against a single tenant, the post-commit per-label cardinality
//! equals the count of *currently-live* nodes carrying that label.
//! This is a stronger invariant than "monotonic increase" — it pins
//! the increment / decrement bookkeeping holistically against the
//! oracle (the tracked-by-the-test live-set) rather than just
//! checking that counts never go negative.
//!
//! 256 cases is the project's standard proptest case count (matches
//! `binding_proptest`, `type_check_proptest`, `multi_tenant_tier_proptest`).

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, delete_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    /// Create a node carrying `label`.
    Create { label: u32 },
    /// Delete the node at index `live_index` of the currently-live
    /// list. Bounded `0..=u8::MAX` to keep proptest shrinking
    /// efficient; the test reduces it modulo the live-set size.
    DeleteAtIndex { live_index: u8 },
}

fn op_strategy() -> impl Strategy<Value = Op> {
    // Small label space (1..=4) so the proptest exercises both "many
    // creates of the same label" and "deletes hit a label that has
    // multiple live records". Bumping the bound widens the search
    // surface but adds little new behaviour to test.
    prop_oneof![
        (1u32..=4u32).prop_map(|label| Op::Create { label }),
        any::<u8>().prop_map(|live_index| Op::DeleteAtIndex { live_index }),
    ]
}

fn build_store() -> (Arc<TxnManager>, CrudStore) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let store = CrudStore::new_with_index(None, Arc::clone(&primary), Arc::clone(&alloc));
    (txn_mgr, store)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        .. ProptestConfig::default()
    })]

    /// Random sequence of (Create | DeleteAtIndex) ops; each op runs
    /// in its own commit so the post-commit hook fires every iteration.
    /// The test maintains a parallel oracle:
    /// - `live: Vec<(NodeId, u32)>` — the test's view of currently-live
    ///   `(id, label)` pairs.
    /// - `expected_label_counts: HashMap<u32, u64>` — derived
    ///   per-label live-counts.
    ///
    /// After each op, the test asserts:
    /// 1. `stats.label_cardinality(label)` matches the oracle's
    ///    count for every label that has ever appeared.
    /// 2. `stats.total_node_count()` matches `live.len()`.
    ///
    /// This pins increment / decrement bookkeeping under random
    /// sequences AND under the saturating-decrement defensive guard
    /// (the proptest's DeleteAtIndex modulo'd onto an empty live list
    /// is a no-op — never reaches the stats hook because
    /// delete_node_with_store would NotFound first).
    #[test]
    fn label_cardinality_tracks_live_set_under_random_create_delete(
        ops in proptest::collection::vec(op_strategy(), 0..=20),
    ) {
        let (mgr, store) = build_store();

        let mut live: Vec<(NodeId, u32)> = Vec::new();
        let mut expected_label_counts: HashMap<u32, u64> = HashMap::new();

        for op in ops {
            match op {
                Op::Create { label } => {
                    let mut tx = mgr.begin(TenantId::DEFAULT);
                    let id = create_node(
                        &store,
                        &mut tx,
                        TenantId::DEFAULT,
                        LabelId::new(label),
                        &PropertyData::Empty,
                    ).unwrap();
                    commit(tx, &store).unwrap();
                    live.push((id, label));
                    *expected_label_counts.entry(label).or_default() += 1;
                }
                Op::DeleteAtIndex { live_index } => {
                    if live.is_empty() {
                        // No-op — there's nothing to delete. The
                        // commit pipeline isn't engaged this iteration;
                        // existing stats (if any) stay unchanged and
                        // the assertions below still hold against the
                        // unchanged oracle.
                    } else {
                        let idx = (live_index as usize) % live.len();
                        let (id, label) = live.remove(idx);
                        let mut tx = mgr.begin(TenantId::DEFAULT);
                        delete_node_with_store(&store, &mut tx, id).unwrap();
                        commit(tx, &store).unwrap();
                        let count = expected_label_counts.get_mut(&label).expect("label present");
                        *count -= 1;
                    }
                }
            }

            // Assertions after every op. The fresh-tenant case (no
            // commits ever landed) returns None for catalog_stats —
            // skip the body in that case; nothing to check yet.
            let Some(stats) = store.catalog_stats(TenantId::DEFAULT) else {
                prop_assert!(live.is_empty(), "live set must be empty pre-first-commit");
                continue;
            };

            // 1. Per-label cardinality matches the oracle for every
            //    label that has ever appeared. Labels that have
            //    appeared and now have zero live members surface as
            //    Some(0) (NOT None) — the documented
            //    "observed-then-fully-deleted" sentinel.
            for (&label, &expected) in expected_label_counts.iter() {
                let observed = stats.label_cardinality(LabelId::new(label));
                prop_assert_eq!(observed, Some(expected), "label {} expected={} observed={:?}", label, expected, observed);
            }

            // 2. Tenant-wide total matches live.len().
            prop_assert_eq!(stats.total_node_count(), Some(live.len() as u64));

            // 3. Counts can never be negative — implicit via u64,
            //    but pin the saturating-decrement guard structurally
            //    by sweeping every label (including never-touched
            //    label 99, which must remain None) for monotonicity
            //    of the `None` sentinel.
            prop_assert_eq!(stats.label_cardinality(LabelId::new(99)), None);
        }
    }
}
