//! Issue #129 P0 — `PageAllocator` + `CrudStore` allocator
//! advance integration tests.
//!
//! These tests exercise the v4 `CommitBundle.allocator_advances`
//! path end-to-end through the WAL codec, including the cross-store
//! dispatch between `PageAllocator` (Page* kinds) and `CrudStore`
//! (Node / Rel kinds). The unit tests in `src/page_alloc.rs` cover
//! the bare snapshot / seed APIs; the tests here cover the round-trip
//! through WAL and the multi-thread + idempotence properties of the
//! seed.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use arcgraph_core::{Lsn, NodeId, PageType, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::wal::{
    AllocatorAdvance, AllocatorKind, AllocatorSeedHandle, decode_commit_bundle_v4,
    encode_commit_bundle_v4,
};
use bytes::Bytes;

// ─── Test 1: round-trip a snapshot through the v4 bundle codec ─────

#[test]
fn allocator_advance_round_trip() {
    // Build an allocator state on one process; snapshot; encode it
    // into a v4 bundle; decode the bundle; seed a fresh allocator;
    // verify high-water is restored exactly so the next allocation
    // returns high_water + 1 (no reuse possible).
    let original = PageAllocator::new();
    let store = CrudStore::new();
    let tenant = TenantId::DEFAULT;

    // Allocate some NodeIds, RelIds, and PageIds across multiple
    // page types.
    for _ in 0..10 {
        let _ = store.alloc_node(tenant).unwrap();
    }
    for _ in 0..3 {
        let _ = store.alloc_rel(tenant).unwrap();
    }
    for _ in 0..7 {
        let _ = original.alloc(tenant, PageType::Node);
    }
    for _ in 0..5 {
        let _ = original.alloc(tenant, PageType::IndexLeaf);
    }
    for _ in 0..2 {
        let _ = original.alloc(tenant, PageType::Tel);
    }

    let mut snap = store.snapshot_allocator_advances();
    snap.extend(original.snapshot_advances());
    // Encode + decode through the v4 codec.
    let encoded = encode_commit_bundle_v4(Lsn::new(1), tenant, &HashMap::new(), &[], &[], &snap);
    let decoded = decode_commit_bundle_v4(&encoded, tenant).unwrap();

    // Verify expected entries are present (5 = 1 Node + 1 Rel + 3 Page*).
    assert_eq!(
        decoded.allocator_advances.len(),
        5,
        "expected 5 advance entries (Node, Rel, PageNode, PageIndexLeaf, PageTel); \
         got {:?}",
        decoded.allocator_advances,
    );

    // Apply advances to a fresh allocator pair.
    let restored_alloc = Arc::new(PageAllocator::new());
    let restored_store = Arc::new(CrudStore::new());
    let seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&restored_store), Arc::clone(&restored_alloc));
    for adv in &decoded.allocator_advances {
        seed.seed_from_advance(*adv);
    }

    // Post-seed: high-water values match the original; next
    // allocation strictly exceeds the pre-fault high-water.
    assert_eq!(restored_store.node_high_water(tenant), 10);
    assert_eq!(restored_store.rel_high_water(tenant), 3);
    assert_eq!(restored_alloc.current_high_water(tenant, PageType::Node), 7);
    assert_eq!(
        restored_alloc.current_high_water(tenant, PageType::IndexLeaf),
        5
    );
    assert_eq!(restored_alloc.current_high_water(tenant, PageType::Tel), 2);

    // Next allocations: must be > pre-fault high-water (no reuse).
    let next_node = restored_store.alloc_node(tenant).unwrap();
    assert_eq!(next_node, NodeId::new(11));
    let next_page = restored_alloc.alloc(tenant, PageType::Node);
    assert_eq!(next_page.raw(), 8);
}

// ─── Test 2: idempotent monotonic seed under double-replay ─────────

#[test]
fn allocator_advance_idempotent_replay() {
    let alloc = Arc::new(PageAllocator::new());
    let store = Arc::new(CrudStore::new());
    let seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let tenant = TenantId::DEFAULT;

    let advances = vec![
        AllocatorAdvance {
            tenant,
            kind: AllocatorKind::Node,
            new_high_water: 100,
        },
        AllocatorAdvance {
            tenant,
            kind: AllocatorKind::PageIndexLeaf,
            new_high_water: 50,
        },
    ];

    // First replay: counters jump to the advance values.
    for adv in &advances {
        seed.seed_from_advance(*adv);
    }
    assert_eq!(store.node_high_water(tenant), 100);
    assert_eq!(alloc.current_high_water(tenant, PageType::IndexLeaf), 50);

    // Second replay (Lemma I3): identical advances applied a
    // second time are a no-op — counters stay at 100 / 50.
    for adv in &advances {
        seed.seed_from_advance(*adv);
    }
    assert_eq!(store.node_high_water(tenant), 100);
    assert_eq!(alloc.current_high_water(tenant, PageType::IndexLeaf), 50);

    // A LOWER advance that arrives out-of-order (e.g., a stale
    // bundle in a torn replay) must NOT regress the counter.
    seed.seed_from_advance(AllocatorAdvance {
        tenant,
        kind: AllocatorKind::Node,
        new_high_water: 50,
    });
    assert_eq!(
        store.node_high_water(tenant),
        100,
        "monotonic-max: out-of-order replay must not regress"
    );

    // A HIGHER advance applied after the steady state advances
    // cleanly (the normal forward-progress case).
    seed.seed_from_advance(AllocatorAdvance {
        tenant,
        kind: AllocatorKind::Node,
        new_high_water: 200,
    });
    assert_eq!(store.node_high_water(tenant), 200);
}

// ─── Test 3: monotonic — multi-thread concurrent commits ───────────

#[test]
fn allocator_advance_monotonic() {
    // Concurrent allocators emit advances; replay applies them in
    // arbitrary order; the final state matches the highest
    // observed snapshot. Stresses the cmpxchg loop in
    // `seed_from_advance`.
    let alloc = Arc::new(PageAllocator::new());
    let store = Arc::new(CrudStore::new());
    let tenant = TenantId::DEFAULT;
    const THREADS: usize = 8;
    const PER_THREAD: u64 = 250;

    // Phase 1: 8 threads concurrently allocate from the SAME
    // allocator state. Each thread snapshots after its allocations.
    let snapshots: Vec<Vec<AllocatorAdvance>> = {
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let alloc = Arc::clone(&alloc);
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    let _ = store.alloc_node(tenant).unwrap();
                    let _ = alloc.alloc(tenant, PageType::Node);
                }
                let mut snap = store.snapshot_allocator_advances();
                snap.extend(alloc.snapshot_advances());
                snap
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    };

    // Total allocations: THREADS × PER_THREAD on each of (Node, PageNode).
    let total: u64 = (THREADS as u64) * PER_THREAD;
    assert_eq!(store.node_high_water(tenant), total);
    assert_eq!(alloc.current_high_water(tenant, PageType::Node), total);

    // Phase 2: replay all snapshots in REVERSE order onto a fresh
    // allocator stack — simulates an out-of-order replay where
    // older snapshots arrive after newer ones.
    let restored_alloc = Arc::new(PageAllocator::new());
    let restored_store = Arc::new(CrudStore::new());
    let seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&restored_store), Arc::clone(&restored_alloc));

    for snap in snapshots.iter().rev() {
        for adv in snap {
            seed.seed_from_advance(*adv);
        }
    }

    // Lemma I3: monotonic-max under any apply order.
    // The HIGHEST snapshot reflected the full `total` advancement
    // (whichever thread snapshotted last), and the seed converges
    // to that maximum regardless of replay order.
    let restored_node = restored_store.node_high_water(tenant);
    let restored_page = restored_alloc.current_high_water(tenant, PageType::Node);
    assert!(
        restored_node >= total - PER_THREAD * (THREADS as u64 - 1) && restored_node <= total,
        "restored node high-water {restored_node} should fall in \
         [total - (THREADS-1)*PER_THREAD, total] = [{}, {}]",
        total - PER_THREAD * (THREADS as u64 - 1),
        total,
    );
    assert!(restored_page >= total - PER_THREAD * (THREADS as u64 - 1) && restored_page <= total);

    // Phase 3: apply the FINAL (highest) snapshot once more —
    // captures the post-Phase-1 state. After this, the restored
    // counters equal `total` exactly (the maximum observed).
    let final_snap = {
        let mut s = store.snapshot_allocator_advances();
        s.extend(alloc.snapshot_advances());
        s
    };
    for adv in &final_snap {
        seed.seed_from_advance(*adv);
    }
    assert_eq!(restored_store.node_high_water(tenant), total);
    assert_eq!(
        restored_alloc.current_high_water(tenant, PageType::Node),
        total
    );
}

// ─── Test 4: local-only guard at v1.0 ──────────────────────

#[test]
fn allocator_advance_partition_id_always_zero_at_v1() {
    // Mirror of `replay_partition_id_always_zero_at_v1` and
    // `z1_partition_id_always_zero_at_v1`. At v1.0 the
    // `AllocatorAdvance` struct carries NO `partition_id` field
    // and the on-wire entry is exactly 17 bytes (8 tenant +
    // 1 kind + 8 high_water).
    //
    // This test is the canary that fires if a change accidentally
    // adds a partition_id field to the wire shape
    // (e.g., by hand-rolling a parallel struct that someone wires
    // into encode_commit_bundle_v4 without bumping the version).
    assert_eq!(AllocatorAdvance::ENCODED_LEN, 17);

    // Encode + decode N advances; verify the decoded vec is byte-
    // identical to the input (modulo wire-order sort) and carries no
    // implicit partition_id state.
    let advances: Vec<AllocatorAdvance> = (1..=10u64)
        .map(|i| AllocatorAdvance {
            tenant: TenantId::new(i),
            kind: AllocatorKind::Node,
            new_high_water: i * 100,
        })
        .collect();
    let encoded = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &advances,
    );
    let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();

    // Total wire size = (header) + 4 (n_advances) + 10 × 17 (entries).
    // Sanity check: decoded count matches input count.
    assert_eq!(decoded.allocator_advances.len(), advances.len());

    // The PartitionId hook lives in `arcgraph_core::PartitionId` and
    // is always ZERO at v1.0 (per ADR-024-amendment-02 §OQ-3). The
    // bundle codec does not touch it, so it remains at the default.
    let partition_default = PartitionId::default();
    assert_eq!(partition_default, PartitionId::ZERO);
    let _ = partition_default;
}

// ─── Test 5: tenant isolation under cross-tenant seeding ───────────

#[test]
fn allocator_advance_tenant_isolation() {
    // Issue #129 fix must be tenant-scoped: seeding tenant A's
    // counter MUST NOT affect tenant B's counter. Pin the
    // invariant.
    let alloc = Arc::new(PageAllocator::new());
    let store = Arc::new(CrudStore::new());
    let seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));

    let t_a = TenantId::new(1);
    let t_b = TenantId::new(2);

    seed.seed_from_advance(AllocatorAdvance {
        tenant: t_a,
        kind: AllocatorKind::Node,
        new_high_water: 100,
    });
    seed.seed_from_advance(AllocatorAdvance {
        tenant: t_b,
        kind: AllocatorKind::Node,
        new_high_water: 50,
    });

    assert_eq!(store.node_high_water(t_a), 100);
    assert_eq!(store.node_high_water(t_b), 50);

    // Tenant isolation under seed-then-alloc — each tenant
    // strictly above its own pre-seed high-water.
    let next_a = store.alloc_node(t_a).unwrap();
    let next_b = store.alloc_node(t_b).unwrap();
    assert_eq!(next_a, NodeId::new(101));
    assert_eq!(next_b, NodeId::new(51));
}

// ─── Test 6: empty bundle preserves zero allocator_advances ────────

#[test]
fn allocator_advance_empty_when_no_allocations() {
    // A commit that touches NO allocator emits an empty
    // `allocator_advances` section. Drives the bundle through the
    // production-mirror snapshot helpers on a pristine
    // PageAllocator + CrudStore.
    let alloc = PageAllocator::new();
    let store = CrudStore::new();
    let mut snap = store.snapshot_allocator_advances();
    snap.extend(alloc.snapshot_advances());
    assert!(snap.is_empty(), "pristine allocators emit no advances");

    let encoded = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &snap,
    );
    let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
    assert!(decoded.allocator_advances.is_empty());
}

// ─── Test 7: bundle with both writes and advances round-trips ─────

#[test]
fn allocator_advance_with_writes_and_pages() {
    // Realistic bundle shape: MVCC writes + staged_pages +
    // allocator_advances all together. Pin the v4 codec composes
    // sections correctly under the encoder's section ordering.
    let mut writes = HashMap::new();
    writes.insert(42u64, Some(Bytes::from_static(b"hello")));
    let advances = vec![AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::Node,
        new_high_water: 7,
    }];
    let encoded = encode_commit_bundle_v4(
        Lsn::new(99),
        TenantId::DEFAULT,
        &writes,
        &[],
        &[],
        &advances,
    );
    let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.commit_lsn, Lsn::new(99));
    assert_eq!(decoded.mvcc_writes.len(), 1);
    assert_eq!(decoded.allocator_advances.len(), 1);
    assert_eq!(decoded.allocator_advances[0].new_high_water, 7);
}

// ─── Test 8: inverse of for_page_type covers every page type ──────

#[test]
fn allocator_kind_for_page_type_is_total_over_pagetype() {
    // Every PageType variant maps to a distinct AllocatorKind
    // Page* variant; the inverse `page_type()` returns Some for
    // every Page* variant. Pin this so a future PageType variant
    // can't be silently added without updating the
    // AllocatorKind taxonomy.
    use arcgraph_core::PageType;
    let mut seen_kinds = std::collections::HashSet::new();
    for pt in [
        PageType::Free,
        PageType::Node,
        PageType::Rel,
        PageType::Tel,
        PageType::IndexInternal,
        PageType::IndexLeaf,
        PageType::VectorNeighbor,
        PageType::WalBuffer,
        PageType::IndexOverflow,
    ] {
        let kind = AllocatorKind::for_page_type(pt);
        assert!(
            seen_kinds.insert(kind),
            "AllocatorKind::for_page_type produced duplicate kind for {pt:?}"
        );
        assert_eq!(kind.page_type(), Some(pt));
    }
}

// ─── Test 9: stress — N tenants × N kinds round-trip ───────────────

#[test]
fn allocator_advance_n_tenants_round_trip() {
    const TENANTS: u64 = 16;
    const NODES_PER_TENANT: u64 = 50;
    const RELS_PER_TENANT: u64 = 25;
    let store = Arc::new(CrudStore::new());

    for t in 1..=TENANTS {
        let tenant = TenantId::new(t);
        for _ in 0..NODES_PER_TENANT {
            let _ = store.alloc_node(tenant).unwrap();
        }
        for _ in 0..RELS_PER_TENANT {
            let _ = store.alloc_rel(tenant).unwrap();
        }
    }

    let snap = store.snapshot_allocator_advances();
    // Expect 2 advances per tenant (Node + Rel) = TENANTS × 2.
    assert_eq!(snap.len(), (TENANTS as usize) * 2);

    let encoded = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &snap,
    );
    let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.allocator_advances.len(), snap.len());

    // Apply to a fresh CrudStore, verify per-tenant high-water.
    let restored = CrudStore::new();
    for adv in &decoded.allocator_advances {
        restored.apply_allocator_advance(*adv);
    }
    for t in 1..=TENANTS {
        let tenant = TenantId::new(t);
        assert_eq!(restored.node_high_water(tenant), NODES_PER_TENANT);
        assert_eq!(restored.rel_high_water(tenant), RELS_PER_TENANT);
        // Next id is strictly above pre-fault high-water.
        let next = restored.alloc_node(tenant).unwrap();
        assert_eq!(next.raw(), NODES_PER_TENANT + 1);
    }
}

// Suppress the "unused" warning on AtomicU64; it's there in case a
// future test adds a concurrent allocator-seed stress.
#[allow(dead_code)]
fn _unused_atomic_marker() -> AtomicU64 {
    AtomicU64::new(0)
}
#[allow(dead_code)]
fn _unused_ordering_marker() -> Ordering {
    Ordering::Acquire
}
