//! MVCC GC safety plus the record-native M4 id/slot lifecycle family.
//!
//! *GC safety*: no version still reachable by an active snapshot is
//! reclaimed. Equivalently, running [`TxnManager::gc`] while a
//! transaction holds a snapshot S never changes what that transaction
//! reads.
//!
//! Gate: 5,000 cases in `--release`.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_gc_safety --nocapture

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::record::{NodeRecord, RelRecord};
use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
use arcgraph_storage::address::{AddressError, MAX_ID};
use arcgraph_storage::addressed_store::{AddressedRecordStore, AddressedStoreError};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit as crud_commit, create_node, delete_node_with_store, read_node,
};
use arcgraph_storage::primary_index::RecordKind;
use arcgraph_storage::records::{NODE_CAPACITY, REL_CAPACITY};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::bundle::{AllocatorAdvance, AllocatorKind};
use bytes::Bytes;
use proptest::prelude::*;

const KEY_SPACE: u64 = 32;
const M4_TENANTS: [TenantId; 2] = [TenantId::new(41), TenantId::new(73)];

fn addressed_crud() -> (TxnManager, CrudStore, Arc<AddressedRecordStore>) {
    let manager = TxnManager::new();
    let addressed = Arc::new(AddressedRecordStore::new());
    let store = CrudStore::new().with_addressed_record_store(Arc::clone(&addressed));
    (manager, store, addressed)
}

fn create_committed_node(
    manager: &TxnManager,
    store: &CrudStore,
    tenant: TenantId,
    label: LabelId,
) -> NodeId {
    let mut tx = manager.begin(tenant);
    let id = create_node(store, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
    crud_commit(tx, store).unwrap();
    id
}

fn delete_committed_node(manager: &TxnManager, store: &CrudStore, tenant: TenantId, id: NodeId) {
    let mut tx = manager.begin(tenant);
    delete_node_with_store(store, &mut tx, id).unwrap();
    crud_commit(tx, store).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 5_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn gc_preserves_every_active_snapshot_view(
        initial in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 0..16),
        later in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 0..64),
        gc_cycles in 0u32..5,
    ) {
        let m = TxnManager::new();

        // Seed committed state; record what each seeded commit makes
        // visible.
        let mut expected: HashMap<u64, u8> = HashMap::new();
        for (k, v) in &initial {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
            expected.insert(*k, *v);
        }

        // Begin the reader — this pins `expected` as its ground truth.
        let reader = m.begin(TenantId::DEFAULT);

        // Interleave more commits with GC cycles.
        for (i, (k, v)) in later.iter().enumerate() {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
            if gc_cycles > 0 && i % (later.len().max(1) / gc_cycles.max(1) as usize + 1) == 0 {
                let _ = m.gc();
            }
        }

        // Final GC under the held reader.
        let _ = m.gc();

        // Reader's view unchanged despite GC.
        for k in 0..KEY_SPACE {
            let got = reader.read(k).map(|b| b[0]);
            let want = expected.get(&k).copied();
            prop_assert_eq!(got, want, "GC corrupted reader view at key {}", k);
        }
    }

    #[test]
    fn gc_with_no_active_txns_keeps_latest_live_version(
        writes in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 1..64),
    ) {
        let m = TxnManager::new();
        let mut last: HashMap<u64, u8> = HashMap::new();
        for (k, v) in &writes {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
            last.insert(*k, *v);
        }
        let _ = m.gc();
        // With no active snapshots, GC may reclaim every expired
        // version, but must keep the single live version per key.
        let lsn = m.current_lsn();
        for (k, v) in &last {
            let got = m.read_at(TenantId::DEFAULT, *k, lsn).map(|b| b[0]);
            prop_assert_eq!(got, Some(*v), "latest live value lost at key {}", k);
        }
    }

    #[test]
    fn repeated_gc_is_idempotent_on_quiescent_state(
        writes in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 0..32),
    ) {
        let m = TxnManager::new();
        for (k, v) in &writes {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
        }
        let _ = m.gc();
        // Snapshot the post-GC chain lengths.
        let post_lengths: Vec<(u64, usize)> = (0..KEY_SPACE)
            .map(|k| (k, m.chain_len(TenantId::DEFAULT, k)))
            .collect();
        // Second pass: must reclaim nothing new.
        let stats = m.gc();
        prop_assert_eq!(stats.reclaimed, 0, "GC was not idempotent: {:?}", stats);
        for (k, len) in &post_lengths {
            prop_assert_eq!(m.chain_len(TenantId::DEFAULT, *k), *len);
        }
    }

    #[test]
    fn deleted_slot_is_a_permanent_tombstone(
        label_raw in 1u32..u32::MAX,
        pressure in 1usize..8,
    ) {
        let (manager, store, addressed) = addressed_crud();
        for tenant in M4_TENANTS {
            prop_assert_ne!(tenant, TenantId::DEFAULT);
            store.seed_node_from_advance(tenant, u64::from(NODE_CAPACITY - 2));
            let retired = create_committed_node(
                &manager,
                &store,
                tenant,
                LabelId::new(label_raw),
            );
            delete_committed_node(&manager, &store, tenant, retired);
            let _ = manager.gc();

            let mut issued = Vec::with_capacity(pressure);
            let mut tx = manager.begin(tenant);
            for offset in 0..pressure {
                issued.push(create_node(
                    &store,
                    &mut tx,
                    tenant,
                    LabelId::new(label_raw.wrapping_add(offset as u32)),
                    &PropertyData::Empty,
                ).unwrap());
            }
            crud_commit(tx, &store).unwrap();
            let _ = manager.gc();

            prop_assert!(issued.iter().all(|id| id.raw() > retired.raw()));
            prop_assert_eq!(addressed.read_node(tenant, retired).unwrap(), None);
        }
    }

    #[test]
    fn allocator_never_reissues_a_retired_id(
        cycle_widths in prop::collection::vec(1u8..5, 2..6),
    ) {
        for tenant in M4_TENANTS {
            prop_assert_ne!(tenant, TenantId::DEFAULT);
            let addressed = Arc::new(AddressedRecordStore::new());
            let mut store = CrudStore::new().with_addressed_record_store(Arc::clone(&addressed));
            let mut greatest_node = 0;
            let mut greatest_rel = 0;

            for width in &cycle_widths {
                for _ in 0..*width {
                    let node = store.alloc_node(tenant).unwrap();
                    prop_assert!(node.raw() > greatest_node);
                    greatest_node = node.raw();
                    addressed.write_node(
                        tenant,
                        &NodeRecord::new(node, LabelId::new(1), Lsn::new(greatest_node)),
                    ).unwrap();

                    let rel = store.alloc_rel(tenant).unwrap();
                    prop_assert!(rel.raw() > greatest_rel);
                    greatest_rel = rel.raw();
                    addressed.write_rel(
                        tenant,
                        &RelRecord::new(
                            rel,
                            TypeId::new(1),
                            node,
                            node,
                            Lsn::new(greatest_rel),
                        ),
                    ).unwrap();
                }

                prop_assert!(addressed.tombstone_node(tenant, NodeId::new(greatest_node)).unwrap());
                prop_assert!(addressed.tombstone_rel(tenant, RelId::new(greatest_rel)).unwrap());

                let recovered = CrudStore::new()
                    .with_addressed_record_store(Arc::clone(&addressed));
                recovered.apply_allocator_advance(AllocatorAdvance {
                    tenant,
                    kind: AllocatorKind::Node,
                    new_high_water: greatest_node,
                });
                recovered.apply_allocator_advance(AllocatorAdvance {
                    tenant,
                    kind: AllocatorKind::Rel,
                    new_high_water: greatest_rel,
                });
                store = recovered;
            }
        }
    }

    #[test]
    fn deleted_id_lookup_is_deterministic_notfound(label_raw in 1u32..u32::MAX) {
        let (manager, store, addressed) = addressed_crud();
        for tenant in M4_TENANTS {
            prop_assert_ne!(tenant, TenantId::DEFAULT);
            let id = create_committed_node(
                &manager,
                &store,
                tenant,
                LabelId::new(label_raw),
            );
            let before_delete = manager.begin(tenant);
            let expected = read_node(&before_delete, id).unwrap().unwrap();

            delete_committed_node(&manager, &store, tenant, id);
            let _ = manager.gc();

            prop_assert_eq!(addressed.read_node(tenant, id).unwrap(), None);
            let historic = read_node(&before_delete, id).unwrap().unwrap();
            prop_assert_eq!(historic.to_bytes(), expected.to_bytes());
            let current = manager.begin(tenant);
            prop_assert_eq!(read_node(&current, id).unwrap(), None);
            drop(current);
            drop(before_delete);
            let _ = manager.gc();
            prop_assert_eq!(addressed.read_node(tenant, id).unwrap(), None);
        }
    }

    #[test]
    fn dangling_ref_resolves_to_tombstone(retired_raw in 2u64..128) {
        for tenant in M4_TENANTS {
            prop_assert_ne!(tenant, TenantId::DEFAULT);
            let addressed = Arc::new(AddressedRecordStore::new());
            let retired = NodeId::new(retired_raw);
            addressed.write_node(
                tenant,
                &NodeRecord::new(retired, LabelId::new(7), Lsn::new(1)),
            ).unwrap();
            prop_assert!(addressed.tombstone_node(tenant, retired).unwrap());
            let dangling = RelRecord::new(
                RelId::new(1),
                TypeId::new(9),
                retired,
                NodeId::new(retired_raw + 1),
                Lsn::new(1),
            );

            let recovered = CrudStore::new()
                .with_addressed_record_store(Arc::clone(&addressed));
            recovered.apply_allocator_advance(AllocatorAdvance {
                tenant,
                kind: AllocatorKind::Node,
                new_high_water: retired_raw,
            });
            let occupant = recovered.alloc_node(tenant).unwrap();
            addressed.write_node(
                tenant,
                &NodeRecord::new(occupant, LabelId::new(99), Lsn::new(2)),
            ).unwrap();

            let resolved = addressed
                .read_node(tenant, NodeId::new(dangling.src_id))
                .unwrap();
            prop_assert!(
                resolved.is_none(),
                "dangling ref {} resolved to newly-issued occupant {}",
                dangling.src_id,
                occupant.raw(),
            );
            prop_assert!(occupant.raw() > retired_raw);
        }
    }

    #[test]
    fn gap_id_lookup_is_notfound_never_allocates(salt in 1u32..u32::MAX) {
        let addressed = AddressedRecordStore::new();
        for tenant in M4_TENANTS {
            prop_assert_ne!(tenant, TenantId::DEFAULT);
            let node_legal = [
                1,
                u64::from(NODE_CAPACITY - 1),
                u64::from(NODE_CAPACITY),
                u64::from(NODE_CAPACITY) + 1,
                MAX_ID - 1,
                MAX_ID,
            ];
            let rel_legal = [
                1,
                u64::from(REL_CAPACITY - 1),
                u64::from(REL_CAPACITY),
                u64::from(REL_CAPACITY) + 1,
                MAX_ID - 1,
                MAX_ID,
            ];
            for id in node_legal {
                prop_assert_eq!(addressed.read_node(tenant, NodeId::new(id)).unwrap(), None);
            }
            for id in rel_legal {
                prop_assert_eq!(addressed.read_rel(tenant, RelId::new(id)).unwrap(), None);
            }
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Node), 0);
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Rel), 0);

            prop_assert!(matches!(
                addressed.read_node(tenant, NodeId::new(0)),
                Err(AddressedStoreError::Address(AddressError::ReservedSentinel)),
            ));
            prop_assert!(matches!(
                addressed.read_rel(tenant, RelId::new(MAX_ID + 1)),
                Err(AddressedStoreError::Address(AddressError::OutOfRange)),
            ));
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Node), 0);
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Rel), 0);

            let node_id = NodeId::new(u64::from(NODE_CAPACITY) + 1);
            addressed.write_node(
                tenant,
                &NodeRecord::new(node_id, LabelId::new(salt), Lsn::new(1)),
            ).unwrap();
            prop_assert_eq!(
                addressed
                    .read_node(tenant, NodeId::new(u64::from(NODE_CAPACITY)))
                    .unwrap(),
                None,
            );
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Node), 1);

            let rel_id = RelId::new(u64::from(REL_CAPACITY) + 1);
            addressed.write_rel(
                tenant,
                &RelRecord::new(
                    rel_id,
                    TypeId::new(salt),
                    node_id,
                    node_id,
                    Lsn::new(1),
                ),
            ).unwrap();
            prop_assert_eq!(
                addressed
                    .read_rel(tenant, RelId::new(u64::from(REL_CAPACITY)))
                    .unwrap(),
                None,
            );
            prop_assert_eq!(addressed.page_count(tenant, RecordKind::Rel), 1);
        }
    }
}
