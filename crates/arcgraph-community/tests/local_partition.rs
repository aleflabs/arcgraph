//! Local-partition pin for [`CommunityIndexHandle`].
//!
//! At v1.0 every public-API construction of a
//! [`CommunityIndexHandle`] MUST carry `partition_id ==
//! PartitionId::ZERO`. The only public constructor —
//! [`CommunityIndexHandle::for_tenant`] — enforces this.
//!
//! This integration test pins the community engine's local-only
//! construction path.

use std::sync::Arc;

use arcgraph_community::{
    BTreeMembershipIndex, CommunityIndexHandle, CommunityIndexId, MembershipIndex,
};
use arcgraph_core::{PartitionId, TenantId};

fn membership() -> Arc<dyn MembershipIndex> {
    Arc::new(BTreeMembershipIndex::new())
}

#[test]
fn for_tenant_pins_partition_zero() {
    let h =
        CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::new(1), membership());
    assert_eq!(
        h.partition(),
        PartitionId::ZERO,
        "v1.0 invariant: partition_id is always ZERO"
    );
    assert!(h.is_v1_local(), "v1.0 invariant: handle is v1-local");
}

#[test]
fn for_tenant_pins_partition_zero_across_many_tenants() {
    // Build a handle for every TenantId in 0..1000 (the system
    // and default tenants plus a sweep of user-tenant ids) and
    // verify the partition stays at ZERO.
    for raw in 0u64..1_000 {
        let h = CommunityIndexHandle::for_tenant(
            TenantId::new(raw),
            CommunityIndexId::new(raw),
            membership(),
        );
        assert_eq!(
            h.partition(),
            PartitionId::ZERO,
            "tenant {raw}: partition_id must be ZERO at v1.0"
        );
        assert!(h.is_v1_local(), "tenant {raw}: handle must be v1-local");
    }
}

#[test]
fn cloning_preserves_partition_zero() {
    // Clone semantics: a clone of a v1-local handle stays v1-local.
    let h1 =
        CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::new(7), membership());
    let h2 = h1.clone();
    assert!(h1.is_v1_local());
    assert!(h2.is_v1_local());
    assert_eq!(h1.partition(), h2.partition());
    assert_eq!(h1.tenant(), h2.tenant());
    assert_eq!(h1.index_id(), h2.index_id());
}

#[test]
fn distinct_index_ids_share_partition_zero() {
    // Even with distinct (tenant, index_id) pairs, partition is
    // always ZERO.
    let h1 =
        CommunityIndexHandle::for_tenant(TenantId::new(1), CommunityIndexId::new(1), membership());
    let h2 =
        CommunityIndexHandle::for_tenant(TenantId::new(2), CommunityIndexId::new(2), membership());
    assert_eq!(h1.partition(), PartitionId::ZERO);
    assert_eq!(h2.partition(), PartitionId::ZERO);
    assert_ne!(h1.tenant(), h2.tenant());
    assert_ne!(h1.index_id(), h2.index_id());
}
