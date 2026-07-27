//! Total arithmetic addressing for direct-addressed node and relationship stores.
//!
//! This module is the single `id -> (page_no, slot)` derivation required by
//! `m1-m2-m4-m5-impl-designs.md` §M4.1. Callers must pass the raw logical id;
//! the relationship MVCC namespace tag is not a record id and is rejected by
//! the 63-bit bound.

use thiserror::Error;

use crate::crud::REL_TAG_BIT;
use crate::primary_index::RecordKind;
use crate::records::{NODE_CAPACITY, REL_CAPACITY};

/// Largest legal raw node or relationship id.
///
/// Per design §M4.1, this is an id-space bound: bit 63 is reserved for
/// [`REL_TAG_BIT`]. It is deliberately not derived from page-byte arithmetic;
/// direct addressing produces logical page numbers, not byte offsets.
pub const MAX_ID: u64 = REL_TAG_BIT - 1;

/// Failure to derive a direct record address from a raw logical id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AddressError {
    /// Id zero is the permanently reserved 1-based allocator sentinel.
    #[error("record id zero is reserved")]
    ReservedSentinel,
    /// The id is outside the raw 63-bit node/relationship id space.
    #[error("record id is outside the 63-bit address space")]
    OutOfRange,
}

impl RecordKind {
    /// Derive the logical page number and slot for a raw record id.
    ///
    /// This is the one arithmetic derivation shared by live CRUD, WAL replay,
    /// recovery, and the future M5 loader. The capacity is selected only from
    /// the compile-pinned [`NODE_CAPACITY`] / [`REL_CAPACITY`] constants.
    /// There is intentionally no `id - 1` normalization: id zero owns the
    /// reserved `(page 0, slot 0)` address and is never written.
    ///
    /// # Errors
    ///
    /// Returns [`AddressError::ReservedSentinel`] for id zero and
    /// [`AddressError::OutOfRange`] for tagged or otherwise out-of-range ids.
    pub fn address(self, id: u64) -> Result<(u64, u16), AddressError> {
        if id == 0 {
            return Err(AddressError::ReservedSentinel);
        }
        if id > MAX_ID {
            return Err(AddressError::OutOfRange);
        }

        let capacity = match self {
            Self::Node => u64::from(NODE_CAPACITY),
            Self::Rel => u64::from(REL_CAPACITY),
        };
        let page_no = id.checked_div(capacity).ok_or(AddressError::OutOfRange)?;
        let slot = id.checked_rem(capacity).ok_or(AddressError::OutOfRange)?;
        let slot = u16::try_from(slot).map_err(|_| AddressError::OutOfRange)?;
        Ok((page_no, slot))
    }
}
