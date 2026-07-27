//! Membership-index trait scaffold per ADR-040 §D-4.
//!
//! Both the forward `(tenant_id, node_id, level) → community_id`
//! lookup and the reverse `(tenant_id, community_id, level,
//! node_id)` range scan land behind the [`MembershipIndex`] trait.
//! At the M3.d-1 scaffold (this commit) every method is a stub
//! returning [`unimplemented!`]; the B-tree implementation lands
//! in M3.d-1 task #4 (`crates/arcgraph-community/src/membership_index.rs`).
//!
//! ## Why a trait at v1.0 with only one production impl?
//!
//! ADR-040 §6.2 commits a v1.1 per-(tenant, partition) sharded
//! variant `DashMap<(TenantId, PartitionId), Arc<RwLock<Inner>>>`
//! that replaces today's process-wide `RwLock<Inner>` when the
//! multi-arena partition orchestration slice lands (post-M4).
//! The `MembershipIndex` trait is the v1.0 → v1.1 substitution
//! seam: today's `BTreeMembershipIndex` impls it; the v1.1
//! sharded variant impls it; `CommunityIndexHandle` consumes
//! the trait via `Arc<dyn>` and is unchanged across the lift.
//! Ratified in ADR-040 amendment-04 D-1 (codex F-2 closure).
//!
//! ## Tenant isolation
//!
//! All methods MUST enforce tenant isolation per ADR-011 /
//! ADR-040 §D-8: a query for `tenant_a` MUST NOT surface community
//! state from `tenant_b`. The trait surface accepts a
//! `tenant: TenantId` parameter at every method so the
//! implementation can guard at the lowest level rather than relying
//! on the handle alone.

use arcgraph_core::{Lsn, NodeId, TenantId};

use crate::error::CommunityError;
use crate::ids::{CommunityId, Level};

/// Per-tenant membership-index trait.
///
/// All trait methods default to [`unimplemented!`] per the M3.d-1
/// scaffold contract. Subsequent task #4 overrides with the real
/// B-tree-backed body. Method bodies that return [`unimplemented!`]
/// are NEVER reached in production — every concrete implementor
/// overrides every method. The defaults exist so the trait is
/// constructable without a sibling impl during the M3.d-1 parallel
/// tasks (router accessor and B-tree implementation may merge in
/// either order).
///
/// ## MVCC visibility (per ADR-041 §D-3b)
///
/// Every retrieval method takes a `read_lsn: Lsn` parameter (last
/// positional argument; mirrors BM25 + vector-substrate
/// convention). At lookup time the implementation finds the
/// LATEST `install_lsn ≤ read_lsn` per `(tenant, level)` and
/// answers from THAT snapshot. A `read_lsn` strictly less than
/// the earliest install returns `Ok(None)` / empty (the read
/// predates every refresh). Callers without snapshot context
/// pass `Lsn::MAX` (most-permissive read; the latest install
/// wins). Production callers source `read_lsn` from
/// `TransactionManager::current_lsn()` via the executor's
/// transaction context.
pub trait MembershipIndex: Send + Sync {
    /// Forward lookup: which community does `node_id` belong to at
    /// hierarchy `level`, as visible at `read_lsn`? Returns
    /// `Ok(None)` if `node_id` is not present in the visible
    /// snapshot (e.g., orphan node from a fresh insert before the
    /// next refresh, or `read_lsn` predates every refresh for the
    /// `(tenant, level)`).
    fn lookup(
        &self,
        _tenant: TenantId,
        _node_id: NodeId,
        _level: Level,
        _read_lsn: Lsn,
    ) -> Result<Option<CommunityId>, CommunityError> {
        unimplemented!("MembershipIndex::lookup — concrete impl required (M3.d-1 task #4)")
    }

    /// Reverse range scan: members of `community_id` at `level`,
    /// as visible at `read_lsn`, sorted ascending by `NodeId` per
    /// ADR-040 §D-4 (B-tree natural order).
    fn members(
        &self,
        _tenant: TenantId,
        _community_id: CommunityId,
        _level: Level,
        _read_lsn: Lsn,
    ) -> Result<Vec<NodeId>, CommunityError> {
        unimplemented!("MembershipIndex::members — concrete impl required (M3.d-1 task #4)")
    }

    /// Communities ranked by relevance to `seeds` per ADR-040
    /// §D-3 size-normalized seed-overlap score, computed against
    /// the snapshot visible at `read_lsn`. Returns top-`k`
    /// `(community_id, score)` pairs.
    fn rank_by_seeds(
        &self,
        _tenant: TenantId,
        _seeds: &[NodeId],
        _level: Level,
        _k: usize,
        _read_lsn: Lsn,
    ) -> Result<Vec<(CommunityId, f32)>, CommunityError> {
        unimplemented!("MembershipIndex::rank_by_seeds — concrete impl required (M3.d-1 task #4)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial impl that overrides nothing — exists only to
    /// prove the trait is object-safe and that the unimplemented
    /// defaults compile against an arbitrary type.
    struct Empty;
    impl MembershipIndex for Empty {}

    #[test]
    #[should_panic(expected = "MembershipIndex::lookup")]
    fn default_lookup_is_unimplemented() {
        let e = Empty;
        let _ = e.lookup(TenantId::DEFAULT, NodeId::ZERO, Level::FINEST, Lsn::MAX);
    }

    #[test]
    #[should_panic(expected = "MembershipIndex::members")]
    fn default_members_is_unimplemented() {
        let e = Empty;
        let _ = e.members(
            TenantId::DEFAULT,
            CommunityId::ZERO,
            Level::FINEST,
            Lsn::MAX,
        );
    }

    #[test]
    #[should_panic(expected = "MembershipIndex::rank_by_seeds")]
    fn default_rank_by_seeds_is_unimplemented() {
        let e = Empty;
        let _ = e.rank_by_seeds(TenantId::DEFAULT, &[], Level::FINEST, 1, Lsn::MAX);
    }

    #[test]
    fn trait_is_object_safe() {
        // Compile-time check: dyn MembershipIndex is constructable.
        // The handle relies on this for `Arc<dyn MembershipIndex>`.
        let _: Box<dyn MembershipIndex> = Box::new(Empty);
    }
}
