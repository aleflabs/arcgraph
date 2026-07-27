//! Tenant-scoped community-index handle.
//!
//! `CommunityIndexHandle` is the **public API entry point** for
//! the community-detection engine. It is keyed by the
//! `(TenantId, PartitionId, CommunityIndexId)` tuple.
//!
//! At v1.0, `partition_id` is always [`PartitionId::ZERO`]; the
//! local-only guarantee is upheld by `Self::for_tenant`.
//!
//! At the M3.d-1 scaffold (this commit) the three retrieval
//! methods delegate directly to the [`MembershipIndex`] trait;
//! the trait's default impls return `unimplemented!` until task
//! #4 lands the B-tree-backed implementation.

use std::sync::Arc;

use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};

use crate::error::CommunityError;
use crate::ids::{CommunityId, CommunityIndexId, Level};
use crate::index::MembershipIndex;

/// Local handle to a community-detection index for a tenant.
///
/// Per ADR-040 §D-3 the API surface is the three retrieval
/// methods below; per ADR-035 §D-7 (referenced by ADR-040 §D-8
/// Q1) `partition_id` is always [`PartitionId::ZERO`] at v1.0.
///
/// `partition_id` is always [`PartitionId::ZERO`].
#[derive(Clone)]
pub struct CommunityIndexHandle {
    tenant_id: TenantId,
    partition_id: PartitionId,
    index_id: CommunityIndexId,
    membership: Arc<dyn MembershipIndex>,
}

impl CommunityIndexHandle {
    /// Construct a handle for a tenant + index. v1.0 always sets
    /// `partition_id` to [`PartitionId::ZERO`] per ADR-035 §D-7
    /// and ADR-040 §D-8 Q1.
    #[inline]
    #[must_use]
    pub fn for_tenant(
        tenant_id: TenantId,
        index_id: CommunityIndexId,
        membership: Arc<dyn MembershipIndex>,
    ) -> Self {
        Self {
            tenant_id,
            partition_id: PartitionId::ZERO,
            index_id,
            membership,
        }
    }

    /// Tenant the handle is scoped to.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant_id
    }

    /// Partition the handle is scoped to. v1.0 invariant:
    /// `partition_id == PartitionId::ZERO`.
    #[inline]
    #[must_use]
    pub fn partition(&self) -> PartitionId {
        self.partition_id
    }

    /// Catalog-allocated global community-index id.
    #[inline]
    #[must_use]
    pub fn index_id(&self) -> CommunityIndexId {
        self.index_id
    }

    /// Whether this handle obeys the local-only partition invariant.
    #[inline]
    #[must_use]
    pub fn is_v1_local(&self) -> bool {
        self.partition_id.raw() == PartitionId::ZERO.raw()
    }

    /// Membership lookup at the visible snapshot: which community
    /// does `node_id` belong to at hierarchy `level`, as visible
    /// at `read_lsn` (per ADR-041 §D-3b)? Returns `Ok(None)` if
    /// `node_id` is not present in the visible snapshot — either
    /// because the node has never been classified or because
    /// `read_lsn` predates every refresh for the
    /// `(tenant, level)`.
    ///
    /// Callers without snapshot context pass `Lsn::MAX` (most-
    /// permissive read; the latest install wins). Production
    /// callers source `read_lsn` from
    /// `TransactionManager::current_lsn()`.
    pub fn membership(
        &self,
        node_id: NodeId,
        level: Level,
        read_lsn: Lsn,
    ) -> Result<Option<CommunityId>, CommunityError> {
        self.membership
            .lookup(self.tenant_id, node_id, level, read_lsn)
    }

    /// Members of `community_id` at `level`, as visible at
    /// `read_lsn`. Sorted ascending by `NodeId` per ADR-040 §D-4
    /// (B-tree range-scan order).
    pub fn members(
        &self,
        community_id: CommunityId,
        level: Level,
        read_lsn: Lsn,
    ) -> Result<Vec<NodeId>, CommunityError> {
        self.membership
            .members(self.tenant_id, community_id, level, read_lsn)
    }

    /// Communities ranked by relevance to `seeds` per ADR-040
    /// §D-3 size-normalized seed-overlap score, computed against
    /// the snapshot visible at `read_lsn`. Returns top-`k`
    /// `(community_id, score)` pairs.
    pub fn rank_by_seeds(
        &self,
        seeds: &[NodeId],
        level: Level,
        k: usize,
        read_lsn: Lsn,
    ) -> Result<Vec<(CommunityId, f32)>, CommunityError> {
        self.membership
            .rank_by_seeds(self.tenant_id, seeds, level, k, read_lsn)
    }
}

impl std::fmt::Debug for CommunityIndexHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommunityIndexHandle")
            .field("tenant_id", &self.tenant_id)
            .field("partition_id", &self.partition_id)
            .field("index_id", &self.index_id)
            .field("membership", &"<dyn MembershipIndex>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty stub used to construct a handle for tests. Returns
    /// `Ok(None)` / empty vectors so the handle's identity-side
    /// methods can be exercised without panicking on the
    /// `unimplemented!` defaults.
    struct StubMembershipIndex;
    impl MembershipIndex for StubMembershipIndex {
        fn lookup(
            &self,
            _tenant: TenantId,
            _node_id: NodeId,
            _level: Level,
            _read_lsn: Lsn,
        ) -> Result<Option<CommunityId>, CommunityError> {
            Ok(None)
        }

        fn members(
            &self,
            _tenant: TenantId,
            _community_id: CommunityId,
            _level: Level,
            _read_lsn: Lsn,
        ) -> Result<Vec<NodeId>, CommunityError> {
            Ok(Vec::new())
        }

        fn rank_by_seeds(
            &self,
            _tenant: TenantId,
            _seeds: &[NodeId],
            _level: Level,
            _k: usize,
            _read_lsn: Lsn,
        ) -> Result<Vec<(CommunityId, f32)>, CommunityError> {
            Ok(Vec::new())
        }
    }

    fn stub() -> Arc<dyn MembershipIndex> {
        Arc::new(StubMembershipIndex)
    }

    #[test]
    fn for_tenant_uses_partition_zero() {
        let h =
            CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::new(1), stub());
        assert_eq!(h.tenant(), TenantId::DEFAULT);
        assert_eq!(h.partition(), PartitionId::ZERO);
        assert_eq!(h.index_id(), CommunityIndexId::new(1));
    }

    #[test]
    fn for_tenant_is_v1_local() {
        let h = CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::ZERO, stub());
        assert!(h.is_v1_local());
    }

    #[test]
    fn handle_carries_tenant_and_index_id() {
        let h =
            CommunityIndexHandle::for_tenant(TenantId::new(42), CommunityIndexId::new(7), stub());
        assert_eq!(h.tenant(), TenantId::new(42));
        assert_eq!(h.index_id(), CommunityIndexId::new(7));
    }

    #[test]
    fn clone_preserves_identity() {
        let h1 =
            CommunityIndexHandle::for_tenant(TenantId::new(3), CommunityIndexId::new(11), stub());
        let h2 = h1.clone();
        assert_eq!(h1.tenant(), h2.tenant());
        assert_eq!(h1.partition(), h2.partition());
        assert_eq!(h1.index_id(), h2.index_id());
        // The Arc is cloned, not duplicated; counts increment.
        assert!(h1.is_v1_local() && h2.is_v1_local());
    }

    #[test]
    fn membership_delegates_to_trait() {
        let h = CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::ZERO, stub());
        let got = h
            .membership(NodeId::ZERO, Level::FINEST, Lsn::MAX)
            .expect("stub");
        assert!(got.is_none());
    }

    #[test]
    fn members_delegates_to_trait() {
        let h = CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::ZERO, stub());
        let got = h
            .members(CommunityId::ZERO, Level::FINEST, Lsn::MAX)
            .expect("stub");
        assert!(got.is_empty());
    }

    #[test]
    fn rank_by_seeds_delegates_to_trait() {
        let h = CommunityIndexHandle::for_tenant(TenantId::DEFAULT, CommunityIndexId::ZERO, stub());
        let got = h
            .rank_by_seeds(&[NodeId::new(1)], Level::FINEST, 5, Lsn::MAX)
            .expect("stub");
        assert!(got.is_empty());
    }

    #[test]
    fn debug_impl_does_not_leak_membership_internals() {
        let h =
            CommunityIndexHandle::for_tenant(TenantId::new(5), CommunityIndexId::new(9), stub());
        let s = format!("{h:?}");
        // Identity fields show up.
        assert!(s.contains("tenant_id"), "got: {s}");
        assert!(s.contains("partition_id"), "got: {s}");
        assert!(s.contains("index_id"), "got: {s}");
        // The Arc<dyn MembershipIndex> is rendered as a placeholder,
        // not as the trait-object's internal representation.
        assert!(s.contains("MembershipIndex"), "got: {s}");
    }
}
