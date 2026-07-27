//! Community detection ID newtypes per ADR-040 §D-3 / §D-4.
//!
//! These IDs are scoped to the community-detection engine and do
//! not collide with the workspace IDs in `arcgraph-core::ids`
//! (graph IDs: `NodeId`, `RelId`, `PageId`, `Lsn`, `LabelId`, …)
//! or with the vector-engine IDs in `arcgraph-vector::ids`
//! (`VectorId`, `IndexId`).
//!
//! ## Sizing rationale
//!
//! - [`CommunityId`] is a `u64` — per-tenant local. Allocated by
//!   the Leiden algorithm during a refresh pass; stable across
//!   queries until the next static refresh. `u64` headroom covers
//!   the worst-case 100 M-vertex tenant of ADR-040 §D-9 with
//!   billions of communities of headroom for the hierarchy.
//! - [`Level`] is a `u8` — per ADR-040 §D-5 v1.0 stores all
//!   Leiden agglomeration levels in the membership index; the
//!   tree depth never exceeds `log2(|V|)` ≤ 27 for |V| ≤ 100 M,
//!   so a `u8` carries 9× headroom.
//! - [`CommunityIndexId`] is a `u64` — global across tenants.
//!   Allocated by the catalog at `DEFINE INDEX … USING COMMUNITY`
//!   DDL time (M4); used as the third tuple component of
//!   [`crate::CommunityIndexHandle`]. Mirrors
//!   `arcgraph_vector::IndexId`; defined locally because
//!   `arcgraph-core` does not export an `IndexId` (each engine
//!   crate owns its own).

use serde::{Deserialize, Serialize};

/// Community identifier within a tenant.
///
/// Allocated by the Leiden algorithm during a refresh pass; stable
/// across queries until the next static refresh. Per ADR-040 §D-4
/// the membership index keys are `(tenant_id, community_id, level,
/// node_id)` — `CommunityId` is scoped to a `(tenant_id,
/// partition_id)` pair (`PartitionId::ZERO` at v1.0 per ADR-035
/// §D-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct CommunityId(pub u64);

impl CommunityId {
    /// The zero value; safe to use as a "null" sentinel only when
    /// the caller proves the tenant reserves community 0.
    pub const ZERO: Self = Self(0);

    /// Maximum addressable community id (`u64::MAX`).
    pub const MAX: Self = Self(u64::MAX);

    /// Construct from a raw `u64`.
    #[inline]
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw `u64` representation.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl From<u64> for CommunityId {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<CommunityId> for u64 {
    #[inline]
    fn from(v: CommunityId) -> Self {
        v.0
    }
}

/// Hierarchy level in the Leiden agglomeration tree.
///
/// Level 0 = finest (leaf communities); higher = coarser (multi-
/// community agglomerations). Per ADR-040 §D-5 v1.0 stores all
/// levels in the membership index; the tree depth never exceeds
/// `log2(|V|)` ≤ 27 for |V| ≤ 100 M, so a `u8` carries 9× headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Level(pub u8);

impl Level {
    /// The finest hierarchy level (leaf communities).
    pub const FINEST: Self = Self(0);

    /// Maximum addressable level (`u8::MAX`); a sentinel only.
    /// Real Leiden hierarchies bound out at `log2(|V|)` ≤ 27 per
    /// ADR-040 §D-5.
    pub const MAX: Self = Self(u8::MAX);

    /// Construct from a raw `u8`.
    #[inline]
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    /// Raw `u8` representation.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

impl From<u8> for Level {
    #[inline]
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl From<Level> for u8 {
    #[inline]
    fn from(v: Level) -> Self {
        v.0
    }
}

/// Global community-index identifier. Allocated by the catalog at
/// `DEFINE INDEX foo … USING COMMUNITY` time (M4); carried in
/// [`crate::CommunityIndexHandle`].
///
/// Mirrors `arcgraph_vector::IndexId`. Defined locally because
/// `arcgraph-core` does not export an `IndexId` (each engine
/// crate owns its own catalog id type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct CommunityIndexId(pub u64);

impl CommunityIndexId {
    /// The zero value; reserved by the catalog as a sentinel for
    /// "no index assigned".
    pub const ZERO: Self = Self(0);

    /// Maximum addressable index id (`u64::MAX`).
    pub const MAX: Self = Self(u64::MAX);

    /// Construct from a raw `u64`.
    #[inline]
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw `u64` representation.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl From<u64> for CommunityIndexId {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<CommunityIndexId> for u64 {
    #[inline]
    fn from(v: CommunityIndexId) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn community_id_const_zero_and_max() {
        assert_eq!(CommunityId::ZERO.raw(), 0);
        assert_eq!(CommunityId::MAX.raw(), u64::MAX);
    }

    #[test]
    fn level_finest_is_zero() {
        assert_eq!(Level::FINEST.raw(), 0);
    }

    #[test]
    fn level_max_is_u8_max() {
        assert_eq!(Level::MAX.raw(), u8::MAX);
    }

    #[test]
    fn community_index_id_const_zero_and_max() {
        assert_eq!(CommunityIndexId::ZERO.raw(), 0);
        assert_eq!(CommunityIndexId::MAX.raw(), u64::MAX);
    }

    #[test]
    fn community_id_ordering_is_numeric() {
        assert!(CommunityId::new(1) < CommunityId::new(2));
    }

    #[test]
    fn level_ordering_is_numeric() {
        assert!(Level::new(0) < Level::new(1));
        assert!(Level::FINEST < Level::new(1));
    }

    #[test]
    fn community_index_id_ordering_is_numeric() {
        assert!(CommunityIndexId::new(1) < CommunityIndexId::new(2));
    }

    #[test]
    fn community_id_from_u64() {
        let id = CommunityId::from(42u64);
        assert_eq!(id, CommunityId::new(42));
    }

    #[test]
    fn community_id_into_u64() {
        let id = CommunityId::new(42);
        let raw: u64 = id.into();
        assert_eq!(raw, 42);
    }

    proptest! {
        #[test]
        fn community_id_roundtrip(raw in any::<u64>()) {
            let id = CommunityId::new(raw);
            prop_assert_eq!(id.raw(), raw);
            prop_assert_eq!(CommunityId::from(raw), id);
            prop_assert_eq!(u64::from(id), raw);
        }

        #[test]
        fn level_roundtrip(raw in any::<u8>()) {
            let level = Level::new(raw);
            prop_assert_eq!(level.raw(), raw);
            prop_assert_eq!(Level::from(raw), level);
            prop_assert_eq!(u8::from(level), raw);
        }

        #[test]
        fn community_index_id_roundtrip(raw in any::<u64>()) {
            let id = CommunityIndexId::new(raw);
            prop_assert_eq!(id.raw(), raw);
            prop_assert_eq!(CommunityIndexId::from(raw), id);
            prop_assert_eq!(u64::from(id), raw);
        }
    }
}
