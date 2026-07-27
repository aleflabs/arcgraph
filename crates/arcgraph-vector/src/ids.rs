//! Vector-engine ID newtypes.
//!
//! These IDs are scoped to the vector engine and do not collide
//! with the workspace IDs in `arcgraph-core::ids` (which are graph
//! IDs: `NodeId`, `RelId`, `PageId`, `Lsn`, `LabelId`, etc.).
//!
//! ## Sizing rationale
//!
//! - [`VectorId`] is a `u32` — per-arena local. With 4 B addressable
//!   vectors per arena it exceeds the 1 B v1.0 sizing target with
//!   4× headroom. Per-arena identifiers never collide between arenas.
//! - [`IndexId`] is a `u64` — global across tenants. Allocated by
//!   the catalog at `DEFINE INDEX` DDL time; used as the third
//!   tuple component of [`crate::VectorIndexHandle`].

use serde::{Deserialize, Serialize};

/// Per-arena vector identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct VectorId(pub u32);

impl VectorId {
    /// The zero value; safe to use as a "null" sentinel only when
    /// the caller proves the arena reserves slot 0.
    pub const ZERO: Self = Self(0);

    /// Maximum addressable vector slot (`u32::MAX`).
    pub const MAX: Self = Self(u32::MAX);

    /// Construct from a raw `u32`.
    #[inline]
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` representation.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for VectorId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<VectorId> for u32 {
    #[inline]
    fn from(v: VectorId) -> Self {
        v.0
    }
}

/// Global vector index identifier. Allocated by the catalog at
/// `DEFINE INDEX foo ON Bar.embedding USING HNSW` time; carried
/// in [`crate::VectorIndexHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct IndexId(pub u64);

impl IndexId {
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

impl From<u64> for IndexId {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<IndexId> for u64 {
    #[inline]
    fn from(v: IndexId) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn vector_id_zero_is_zero() {
        assert_eq!(VectorId::ZERO.raw(), 0);
    }

    #[test]
    fn vector_id_max_is_u32_max() {
        assert_eq!(VectorId::MAX.raw(), u32::MAX);
    }

    #[test]
    fn index_id_zero_is_zero() {
        assert_eq!(IndexId::ZERO.raw(), 0);
    }

    #[test]
    fn index_id_max_is_u64_max() {
        assert_eq!(IndexId::MAX.raw(), u64::MAX);
    }

    #[test]
    fn vector_id_ordering_is_numeric() {
        assert!(VectorId::new(1) < VectorId::new(2));
    }

    #[test]
    fn index_id_ordering_is_numeric() {
        assert!(IndexId::new(1) < IndexId::new(2));
    }

    proptest! {
        #[test]
        fn vector_id_u32_roundtrip(raw in any::<u32>()) {
            let id = VectorId::new(raw);
            prop_assert_eq!(id.raw(), raw);
            prop_assert_eq!(VectorId::from(raw), id);
            prop_assert_eq!(u32::from(id), raw);
        }

        #[test]
        fn index_id_u64_roundtrip(raw in any::<u64>()) {
            let id = IndexId::new(raw);
            prop_assert_eq!(id.raw(), raw);
            prop_assert_eq!(IndexId::from(raw), id);
            prop_assert_eq!(u64::from(id), raw);
        }
    }
}
