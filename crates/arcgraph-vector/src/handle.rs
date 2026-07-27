//! Tenant-scoped vector index handle.
//!
//! `VectorIndexHandle` is the **public API entry point** for the
//! vector engine. It is keyed by `(TenantId, PartitionId, IndexId)`.
//!
//! At v1.0, `partition_id` is always [`PartitionId::ZERO`]; the
//! local-only guarantee is upheld by `Self::for_tenant`.

use arcgraph_core::{PartitionId, TenantId};

use crate::IndexId;

/// Local handle to a vector index. `partition_id` is always
/// [`PartitionId::ZERO`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorIndexHandle {
    /// Tenant the index belongs to. Always set to a real tenant
    /// (never [`TenantId::SYSTEM`] for user vector data, though
    /// system-owned indexes may use the system tenant).
    pub tenant_id: TenantId,

    /// Partition the index lives on. v1.0 invariant:
    /// `partition_id == PartitionId::ZERO`.
    partition_id: PartitionId,

    /// Catalog-allocated global index id for this `(tenant,
    /// label/property)` pairing.
    pub index_id: IndexId,
}

impl VectorIndexHandle {
    /// Construct a handle for a tenant + index. v1.0 always sets
    /// `partition_id` to [`PartitionId::ZERO`] per ADR-035 D-7.
    ///
    #[inline]
    #[must_use]
    pub const fn for_tenant(tenant_id: TenantId, index_id: IndexId) -> Self {
        Self {
            tenant_id,
            partition_id: PartitionId::ZERO,
            index_id,
        }
    }

    /// Whether this handle obeys the local-only partition invariant.
    #[inline]
    #[must_use]
    pub const fn is_v1_local(&self) -> bool {
        self.partition_id.raw() == PartitionId::ZERO.raw()
    }

    /// Local partition sentinel.
    #[inline]
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_tenant_uses_partition_zero() {
        let h = VectorIndexHandle::for_tenant(TenantId::DEFAULT, IndexId::new(1));
        assert_eq!(h.tenant_id, TenantId::DEFAULT);
        assert_eq!(h.partition(), PartitionId::ZERO);
        assert_eq!(h.index_id, IndexId::new(1));
    }

    #[test]
    fn for_tenant_is_v1_local() {
        let h = VectorIndexHandle::for_tenant(TenantId::DEFAULT, IndexId::ZERO);
        assert!(h.is_v1_local());
    }

    #[test]
    fn handle_is_copy_and_eq() {
        let h1 = VectorIndexHandle::for_tenant(TenantId::DEFAULT, IndexId::new(7));
        let h2 = h1;
        assert_eq!(h1, h2);
    }
}
