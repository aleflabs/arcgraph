//! Inline property discriminator + encode/decode (M2-30).
//!
//! M2.c wired `PropertyData::{Empty, InlineU32Pair, Blob}` directly onto
//! `NodeRecord`/`RelRecord` but the discriminator scheme was implicit.
//! This module formalizes it so M2-31 (BlobStore) and the M2-CUTOVER
//! dual-write have a single place to read and write the encoding.
//!
//! # Discriminator layout (design-v2 §3.2)
//!
//! `NodeRecord` / `RelRecord` carry three property-bearing fields:
//!
//! - `property_ref: u64`
//! - `inline_u32a: u32`
//! - `inline_u32b: u32`
//!
//! Bit 63 of `property_ref` is the **overflow bit**:
//!
//! - `0` = inline: the two `u32` fields hold the payload; `property_ref`
//!   is zero. This is the M2.c-supported path.
//! - `1` = overflow: the low 63 bits decode as a [`BlobRef`] pointing
//!   into the BlobStore (M2-31). `inline_u32a` / `inline_u32b` are
//!   reserved.
//!
//! The low 63 bits of an overflow `property_ref` decompose as:
//!
//! - bits 15..63 (48 bits) — `page_id` (see DEC-2).
//! - bits 0..15  (15 bits) — `slot_id`, zero for chained blobs.
//!
//! 48 bits of page id is 2^48 × 8 KiB = 2 PiB of addressable blob
//! pages, which is generous for alpha; 15 bits of slot id is 32 K
//! slots per small-blob page, exceeding the ~60-slot target in the
//! M2-31 small-blob design.
//!
//! # Relation to `node_flags::HAS_EXTENDED`
//!
//! `HAS_EXTENDED` mirrors the overflow bit so scanners can reject
//! inline-only records without touching `property_ref`. [`encode_overflow_node`]
//! sets it; [`encode_inline_node`] clears it. The two must agree; the
//! test `overflow_bit_and_has_extended_flag_are_coherent` proves it.
//!
//! # Public surface vs `PropertyData`
//!
//! `PropertyData` in `crud.rs` is the public façade — it carries the
//! caller-facing `Empty | InlineU32Pair | Blob` variants and absorbs
//! the [`PropError`](crate::crud::PropError) translation. This module
//! is the private mechanism beneath it.

use arcgraph_core::record::{NodeRecord, RelRecord, node_flags};

// ─────────────────────────────────────────────────────────────────────
// Bit-pack constants
// ─────────────────────────────────────────────────────────────────────

/// Top bit of `property_ref`. `1` = overflow (blob), `0` = inline.
pub const OVERFLOW_BIT: u64 = 1u64 << 63;

/// Width of the slot-id field in an overflow `property_ref`.
pub const OVERFLOW_SLOT_BITS: u32 = 15;

/// Width of the page-id field in an overflow `property_ref`.
pub const OVERFLOW_PAGE_BITS: u32 = 48;

/// Mask isolating the slot-id bits of an overflow `property_ref`.
pub const OVERFLOW_SLOT_MASK: u64 = (1u64 << OVERFLOW_SLOT_BITS) - 1;

/// Mask isolating the page-id bits after the slot-id has been shifted off.
pub const OVERFLOW_PAGE_MASK: u64 = (1u64 << OVERFLOW_PAGE_BITS) - 1;

const _: () = assert!(OVERFLOW_SLOT_BITS + OVERFLOW_PAGE_BITS < 64);

// ─────────────────────────────────────────────────────────────────────
// BlobRef
// ─────────────────────────────────────────────────────────────────────

/// Reference to a blob page/slot, stored inside `NodeRecord::property_ref`
/// when the overflow bit is set. Populated by M2-31's BlobStore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobRef {
    /// Page id of the head blob page. Bounded to 48 bits on encode.
    pub page_id: u64,
    /// Slot id inside the head page. Zero for chained (large) blobs.
    pub slot_id: u16,
}

impl BlobRef {
    /// Build a `BlobRef` from head-page + slot, clamping out-of-range
    /// bits. Callers MUST NOT pass ids that exceed the encoded widths
    /// — debug builds assert.
    #[must_use]
    pub fn new(page_id: u64, slot_id: u16) -> Self {
        debug_assert!(
            page_id <= OVERFLOW_PAGE_MASK,
            "blob page_id {page_id} exceeds 48 bits"
        );
        debug_assert!(
            u64::from(slot_id) <= OVERFLOW_SLOT_MASK,
            "blob slot_id {slot_id} exceeds {OVERFLOW_SLOT_BITS} bits"
        );
        Self { page_id, slot_id }
    }

    /// Pack into the on-record `property_ref` u64. Sets the overflow bit.
    #[must_use]
    pub fn encode(self) -> u64 {
        OVERFLOW_BIT
            | ((self.page_id & OVERFLOW_PAGE_MASK) << OVERFLOW_SLOT_BITS)
            | (u64::from(self.slot_id) & OVERFLOW_SLOT_MASK)
    }

    /// Unpack a `property_ref` u64. Returns `None` if the overflow bit
    /// is not set.
    #[must_use]
    pub fn decode(raw: u64) -> Option<Self> {
        if raw & OVERFLOW_BIT == 0 {
            return None;
        }
        let low = raw & !OVERFLOW_BIT;
        let slot_id = (low & OVERFLOW_SLOT_MASK) as u16;
        let page_id = (low >> OVERFLOW_SLOT_BITS) & OVERFLOW_PAGE_MASK;
        Some(Self { page_id, slot_id })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Inline shape
// ─────────────────────────────────────────────────────────────────────

/// Machine view of the inline payload. `PropertyData::Empty` and
/// `PropertyData::InlineU32Pair(0, 0)` both round-trip as
/// [`InlineShape::U32Pair(0, 0)`]; the distinction is purely a caller-
/// side mnemonic. See DEC-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineShape {
    /// Two u32s packed into `inline_u32a` / `inline_u32b`.
    U32Pair(u32, u32),
}

// ─────────────────────────────────────────────────────────────────────
// Encode / decode — NodeRecord
// ─────────────────────────────────────────────────────────────────────

/// Write an inline payload onto `rec`. Clears the overflow bit and the
/// `HAS_EXTENDED` flag.
pub fn encode_inline_node(shape: InlineShape, rec: &mut NodeRecord) {
    let InlineShape::U32Pair(a, b) = shape;
    rec.property_ref = 0;
    rec.inline_u32a = a;
    rec.inline_u32b = b;
    rec.flags &= !node_flags::HAS_EXTENDED;
}

/// Write a BlobRef onto `rec`. Sets the overflow bit and the
/// `HAS_EXTENDED` flag, zeros the inline u32s.
pub fn encode_overflow_node(blob: BlobRef, rec: &mut NodeRecord) {
    rec.property_ref = blob.encode();
    rec.inline_u32a = 0;
    rec.inline_u32b = 0;
    rec.flags |= node_flags::HAS_EXTENDED;
}

/// Decode the property payload from a `NodeRecord`.
#[must_use]
pub fn decode_node(rec: &NodeRecord) -> PropertyReadout {
    if let Some(blob) = BlobRef::decode(rec.property_ref) {
        PropertyReadout::Overflow(blob)
    } else {
        PropertyReadout::Inline(InlineShape::U32Pair(rec.inline_u32a, rec.inline_u32b))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Encode / decode — RelRecord
// ─────────────────────────────────────────────────────────────────────

/// Mirror of [`encode_inline_node`] for `RelRecord`.
pub fn encode_inline_rel(shape: InlineShape, rec: &mut RelRecord) {
    let InlineShape::U32Pair(a, b) = shape;
    rec.property_ref = 0;
    rec.inline_u32a = a;
    rec.inline_u32b = b;
    rec.flags &= !node_flags::HAS_EXTENDED;
}

/// Mirror of [`encode_overflow_node`] for `RelRecord`.
pub fn encode_overflow_rel(blob: BlobRef, rec: &mut RelRecord) {
    rec.property_ref = blob.encode();
    rec.inline_u32a = 0;
    rec.inline_u32b = 0;
    rec.flags |= node_flags::HAS_EXTENDED;
}

/// Decode the property payload from a `RelRecord`.
#[must_use]
pub fn decode_rel(rec: &RelRecord) -> PropertyReadout {
    if let Some(blob) = BlobRef::decode(rec.property_ref) {
        PropertyReadout::Overflow(blob)
    } else {
        PropertyReadout::Inline(InlineShape::U32Pair(rec.inline_u32a, rec.inline_u32b))
    }
}

// ─────────────────────────────────────────────────────────────────────
// PropertyReadout
// ─────────────────────────────────────────────────────────────────────

/// Result of decoding a property payload. `Overflow` callers resolve
/// via the BlobStore (M2-31); `Inline` callers are complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyReadout {
    /// Inline payload — no further I/O required.
    Inline(InlineShape),
    /// Overflow reference — caller looks up via BlobStore.
    Overflow(BlobRef),
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TypeId};
    use proptest::prelude::*;

    fn fresh_node() -> NodeRecord {
        NodeRecord::new(NodeId::new(1), LabelId::new(0), Lsn::new(1))
    }

    fn fresh_rel() -> RelRecord {
        RelRecord::new(
            RelId::new(1),
            TypeId::new(0),
            NodeId::new(2),
            NodeId::new(3),
            Lsn::new(1),
        )
    }

    // ---- bit constants ----

    #[test]
    fn overflow_bit_is_top_bit() {
        assert_eq!(OVERFLOW_BIT, 0x8000_0000_0000_0000);
    }

    #[test]
    fn slot_and_page_masks_are_disjoint() {
        let shifted_page = OVERFLOW_PAGE_MASK << OVERFLOW_SLOT_BITS;
        assert_eq!(shifted_page & OVERFLOW_SLOT_MASK, 0);
        assert_eq!(shifted_page & OVERFLOW_BIT, 0);
    }

    // ---- inline encode/decode ----

    #[test]
    fn inline_encode_decode_node_roundtrip() {
        let mut rec = fresh_node();
        encode_inline_node(InlineShape::U32Pair(11, 22), &mut rec);
        assert_eq!(rec.property_ref, 0);
        assert_eq!(rec.inline_u32a, 11);
        assert_eq!(rec.inline_u32b, 22);
        assert_eq!(rec.flags & node_flags::HAS_EXTENDED, 0);
        match decode_node(&rec) {
            PropertyReadout::Inline(InlineShape::U32Pair(a, b)) => {
                assert_eq!((a, b), (11, 22));
            }
            other => panic!("expected inline, got {other:?}"),
        }
    }

    #[test]
    fn inline_encode_decode_rel_roundtrip() {
        let mut rec = fresh_rel();
        encode_inline_rel(InlineShape::U32Pair(0xDEAD_BEEF, 0xCAFE_F00D), &mut rec);
        assert_eq!(rec.property_ref, 0);
        match decode_rel(&rec) {
            PropertyReadout::Inline(InlineShape::U32Pair(a, b)) => {
                assert_eq!((a, b), (0xDEAD_BEEF, 0xCAFE_F00D));
            }
            other => panic!("expected inline, got {other:?}"),
        }
    }

    #[test]
    fn transition_from_empty_to_inline_sets_discriminator_bit_0() {
        let mut rec = fresh_node();
        assert_eq!(rec.property_ref & OVERFLOW_BIT, 0);
        encode_inline_node(InlineShape::U32Pair(1, 2), &mut rec);
        assert_eq!(rec.property_ref & OVERFLOW_BIT, 0);
    }

    // ---- overflow encode/decode ----

    #[test]
    fn blob_ref_encode_decode_roundtrip() {
        let blob = BlobRef::new(0x1234_5678_9ABC, 0x3FFF);
        let raw = blob.encode();
        assert_eq!(raw & OVERFLOW_BIT, OVERFLOW_BIT);
        let back = BlobRef::decode(raw).unwrap();
        assert_eq!(back, blob);
    }

    #[test]
    fn blob_ref_decode_rejects_inline() {
        assert!(BlobRef::decode(0).is_none());
        assert!(BlobRef::decode(0x7FFF_FFFF_FFFF_FFFF).is_none());
    }

    #[test]
    fn overflow_encode_decode_node_roundtrip() {
        let mut rec = fresh_node();
        let blob = BlobRef::new(0xABCD, 7);
        encode_overflow_node(blob, &mut rec);
        assert!(rec.property_ref & OVERFLOW_BIT != 0);
        assert_eq!(rec.inline_u32a, 0);
        assert_eq!(rec.inline_u32b, 0);
        assert_ne!(rec.flags & node_flags::HAS_EXTENDED, 0);
        match decode_node(&rec) {
            PropertyReadout::Overflow(b) => assert_eq!(b, blob),
            other => panic!("expected overflow, got {other:?}"),
        }
    }

    #[test]
    fn overflow_encode_decode_rel_roundtrip() {
        let mut rec = fresh_rel();
        let blob = BlobRef::new(42, 3);
        encode_overflow_rel(blob, &mut rec);
        match decode_rel(&rec) {
            PropertyReadout::Overflow(b) => assert_eq!(b, blob),
            other => panic!("expected overflow, got {other:?}"),
        }
    }

    #[test]
    fn overflow_bit_and_has_extended_flag_are_coherent() {
        let mut rec = fresh_node();
        encode_overflow_node(BlobRef::new(1, 0), &mut rec);
        let overflow_set = rec.property_ref & OVERFLOW_BIT != 0;
        let extended_set = rec.flags & node_flags::HAS_EXTENDED != 0;
        assert_eq!(overflow_set, extended_set);

        encode_inline_node(InlineShape::U32Pair(5, 6), &mut rec);
        let overflow_set = rec.property_ref & OVERFLOW_BIT != 0;
        let extended_set = rec.flags & node_flags::HAS_EXTENDED != 0;
        assert_eq!(overflow_set, extended_set);
        assert!(!overflow_set);
    }

    // ---- bit-pack property ----

    proptest! {
        #[test]
        fn inline_discriminator_bit_pack_roundtrip(
            a in any::<u32>(),
            b in any::<u32>(),
        ) {
            let mut rec = fresh_node();
            encode_inline_node(InlineShape::U32Pair(a, b), &mut rec);
            prop_assert_eq!(rec.property_ref & OVERFLOW_BIT, 0);
            match decode_node(&rec) {
                PropertyReadout::Inline(InlineShape::U32Pair(da, db)) => {
                    prop_assert_eq!((da, db), (a, b));
                }
                other => prop_assert!(false, "expected inline, got {:?}", other),
            }
        }

        #[test]
        fn blob_ref_roundtrip_any_page_slot(
            page_id in 0u64..=OVERFLOW_PAGE_MASK,
            slot_id in 0u16..=(OVERFLOW_SLOT_MASK as u16),
        ) {
            let blob = BlobRef::new(page_id, slot_id);
            let raw = blob.encode();
            prop_assert!(raw & OVERFLOW_BIT != 0);
            let back = BlobRef::decode(raw).unwrap();
            prop_assert_eq!(back, blob);
        }
    }
}
