//! On-disk record layouts for ArcGraph.
//!
//! Layout budget (design-v2 §3.2):
//!
//! | Record       | Size | Alignment | Notes                                     |
//! |--------------|------|-----------|-------------------------------------------|
//! | `PageHeader` | 40 B | 8         | `#[repr(C)]`; offset 0 of a page; tenant_id added M1.5-02 |
//! | `NodeRecord` | 64 B | 64        | one full cache line                       |
//! | `RelRecord`  | 96 B | 8         | 1.5 cache lines                           |
//! | `TelEntry`   | 32 B | 8         | 2 per cache line for fast scan            |
//!
//! Every record has a dedicated little-endian `to_bytes` / `from_bytes`
//! pair. These are the on-disk contract and are stable across minor
//! versions. A `record_version` nibble in `NodeRecord::flags` and
//! `RelRecord::flags` provides a future migration path; the
//! `PageHeader::version` byte plays the same role for whole-page layout.

use crate::ids::{LabelId, Lsn, NodeId, PageId, RelId, TenantId, TypeId};
use crate::{ArcGraphError, Result};

// ---------- compile-time layout assertions (fail the build if broken) ----------

const _: () = assert!(core::mem::size_of::<PageHeader>() == 40);
const _: () = assert!(core::mem::align_of::<PageHeader>() == 8);
const _: () = assert!(core::mem::size_of::<NodeRecord>() == 64);
const _: () = assert!(core::mem::align_of::<NodeRecord>() == 64);
const _: () = assert!(core::mem::size_of::<RelRecord>() == 96);
const _: () = assert!(core::mem::size_of::<TelEntry>() == 32);

// ---------- magic + versioning ----------

/// On-disk page size. Matches a typical 4 KiB filesystem block × 2 for
/// graph-record locality (design-v2 §3.4).
pub const PAGE_SIZE: usize = 8192;

/// Bytes 0..4 of every page. ASCII "ARCG".
pub const PAGE_MAGIC: u32 = 0x4743_5241; // little-endian "ARCG"

/// Current page-header format version. Bumped on any on-disk layout change.
///
/// v2 introduces the 8-byte `tenant_id` field at offset 24..32, growing
/// the header from 32 to 40 bytes. Pre-GA change; no on-disk migration.
pub const PAGE_HEADER_VERSION: u8 = 2;

/// Current node-record layout version (stored in the low 3 bits of `flags`).
pub const NODE_RECORD_VERSION: u8 = 1;

/// Current relationship-record layout version.
pub const REL_RECORD_VERSION: u8 = 1;

// ---------- page type ----------

/// Enumeration of page types. Stored as a single byte in [`PageHeader::page_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PageType {
    /// Free / unallocated page.
    Free = 0,
    /// Page holding `NodeRecord` entries.
    Node = 1,
    /// Page holding `RelRecord` entries.
    Rel = 2,
    /// Page holding a `TelEntry` block.
    Tel = 3,
    /// B-tree internal node.
    IndexInternal = 4,
    /// B-tree leaf node.
    IndexLeaf = 5,
    /// HNSW neighbor-list page.
    VectorNeighbor = 6,
    /// WAL buffer page (ring buffer in-memory representation).
    WalBuffer = 7,
    /// Secondary-index duplicate-NodeId overflow page (M2-34, DEC-20).
    /// Payload layout:
    /// `40 B header | 2 B filled_count | 6 B pad | 8 B next_page_id |
    /// (PAGE_SIZE - 56)/8 × 8 B NodeId slots`.
    /// Zero NodeId slots are empty / tombstoned. `next = 0` terminates
    /// the chain.
    IndexOverflow = 8,
    /// Shared slotted property-bag heap page (v2 M1 — W-B1 slotted
    /// small-blob packing, ADR-230 / design §2.2). Packs many small
    /// property-bag payloads (DEC-6 JSON bytes at M1) into one page
    /// using the same slot-array layout as `Node`/`Rel` pages, but with
    /// variable-length payloads. Referenced from
    /// `NodeRecord::property_ref` / `RelRecord::property_ref` as a
    /// `BlobRef` whose slot field is load-bearing (1-based; `slot_id=0`
    /// remains the DEC-4 chained-blob discriminant).
    PropSlotted = 9,
}

impl PageType {
    /// Convert a byte back to a `PageType`, rejecting unknown values.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => Self::Free,
            1 => Self::Node,
            2 => Self::Rel,
            3 => Self::Tel,
            4 => Self::IndexInternal,
            5 => Self::IndexLeaf,
            6 => Self::VectorNeighbor,
            7 => Self::WalBuffer,
            8 => Self::IndexOverflow,
            9 => Self::PropSlotted,
            other => return Err(ArcGraphError::UnknownPageType(other)),
        })
    }

    /// Raw byte for storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

// ---------- PageHeader ----------

/// 40-byte header at offset 0 of every 8 KiB page.
///
/// Layout (`#[repr(C)]`, natural u64 alignment = 8):
/// ```text
///  0.. 4  magic      (u32)
///  4.. 5  version    (u8)   — PAGE_HEADER_VERSION
///  5.. 6  page_type  (u8)   — PageType byte
///  6.. 8  flags      (u16)
///  8..16  page_id    (u64)
/// 16..24  lsn        (u64)
/// 24..32  tenant_id  (u64)  — M1.5-02; new in page format v2
/// 32..36  checksum   (u32)  — CRC32C of page bytes 40..8192
/// 36..38  slot_count (u16)
/// 38..40  free_space (u16)
/// ```
///
/// A 40-byte header at page offset 0 fits entirely within the first
/// 64-byte cache line when the page is 8-KiB-aligned (offset 0 + 40 < 64).
/// Pages are always 8-KiB-aligned, so the full header is loaded in a
/// single cache-line access. See ADR-011 amendment-01 for the repr choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PageHeader {
    /// [`PAGE_MAGIC`] — validated on every read.
    pub magic: u32,
    /// [`PAGE_HEADER_VERSION`].
    pub version: u8,
    /// [`PageType`] as a byte.
    pub page_type: u8,
    /// Implementation-defined flags.
    pub flags: u16,
    /// This page's identifier.
    pub page_id: u64,
    /// Last LSN that modified this page.
    pub lsn: u64,
    /// Logical database (tenant) that owns this page. See [`TenantId`].
    pub tenant_id: u64,
    /// CRC32C of the page body (bytes `Self::SIZE`..8192).
    pub checksum: u32,
    /// Number of slot-directory entries currently in use.
    pub slot_count: u16,
    /// Bytes of free space on this page.
    pub free_space: u16,
}

impl PageHeader {
    /// On-disk size in bytes. Must equal `size_of::<PageHeader>()`.
    pub const SIZE: usize = 40;

    /// Fresh header for a page of the given type owned by `tenant`.
    #[must_use]
    pub fn new(page_id: PageId, page_type: PageType, tenant: TenantId) -> Self {
        Self {
            magic: PAGE_MAGIC,
            version: PAGE_HEADER_VERSION,
            page_type: page_type.as_byte(),
            flags: 0,
            page_id: page_id.raw(),
            lsn: 0,
            tenant_id: tenant.raw(),
            checksum: 0,
            slot_count: 0,
            free_space: 0,
        }
    }

    /// Serialize to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4] = self.version;
        buf[5] = self.page_type;
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.page_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.tenant_id.to_le_bytes());
        buf[32..36].copy_from_slice(&self.checksum.to_le_bytes());
        buf[36..38].copy_from_slice(&self.slot_count.to_le_bytes());
        buf[38..40].copy_from_slice(&self.free_space.to_le_bytes());
        buf
    }

    /// Deserialize from exactly [`Self::SIZE`] little-endian bytes.
    ///
    /// Validates `magic` and `version`; rejects unknown page types.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Result<Self> {
        let magic = u32::from_le_bytes(read_array::<4>(&bytes[0..4]));
        if magic != PAGE_MAGIC {
            return Err(ArcGraphError::BadPageMagic {
                got: magic,
                expected: PAGE_MAGIC,
            });
        }
        let version = bytes[4];
        if version != PAGE_HEADER_VERSION {
            return Err(ArcGraphError::UnsupportedRecordVersion(version));
        }
        let page_type_byte = bytes[5];
        // Validate the byte; keep the raw byte for zero-loss roundtrip.
        PageType::from_byte(page_type_byte)?;

        Ok(Self {
            magic,
            version,
            page_type: page_type_byte,
            flags: u16::from_le_bytes(read_array::<2>(&bytes[6..8])),
            page_id: u64::from_le_bytes(read_array::<8>(&bytes[8..16])),
            lsn: u64::from_le_bytes(read_array::<8>(&bytes[16..24])),
            tenant_id: u64::from_le_bytes(read_array::<8>(&bytes[24..32])),
            checksum: u32::from_le_bytes(read_array::<4>(&bytes[32..36])),
            slot_count: u16::from_le_bytes(read_array::<2>(&bytes[36..38])),
            free_space: u16::from_le_bytes(read_array::<2>(&bytes[38..40])),
        })
    }
}

// ---------- NodeRecord ----------

/// 64-byte cache-line-aligned node record (design-v2 §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct NodeRecord {
    /// Node identifier.
    pub id: u64,
    /// Interned label.
    pub label_id: u32,
    /// Flags byte. Low 3 bits are `record_version`; higher bits describe
    /// deletion, vector presence, and extended-properties state.
    pub flags: u8,
    _pad1: [u8; 3],
    /// Offset (or inline) reference into the property store.
    pub property_ref: u64,
    /// First inline cached numeric property.
    pub inline_u32a: u32,
    /// Second inline cached numeric property.
    pub inline_u32b: u32,
    /// Reference into the vector store (0 = no embedding).
    pub vector_ref: u64,
    /// Head of the outgoing TEL chain.
    pub out_tel_ref: u64,
    /// Head of the incoming TEL chain.
    pub in_tel_ref: u64,
    /// MVCC lower visibility bound (the commit LSN that created this version).
    pub created_lsn: u64,
}

/// Flag bits for [`NodeRecord::flags`] and [`RelRecord::flags`].
pub mod node_flags {
    /// Record version lives in the low 3 bits.
    pub const VERSION_MASK: u8 = 0b0000_0111;
    /// Set when the record is tombstoned.
    pub const DELETED: u8 = 0b0000_1000;
    /// Set when `vector_ref` points to a live embedding.
    pub const HAS_VECTOR: u8 = 0b0001_0000;
    /// Set when properties overflow into a secondary store.
    pub const HAS_EXTENDED: u8 = 0b0010_0000;
}

impl NodeRecord {
    /// On-disk size in bytes.
    pub const SIZE: usize = 64;

    /// Fresh live record.
    #[must_use]
    pub fn new(id: NodeId, label: LabelId, created_lsn: Lsn) -> Self {
        Self {
            id: id.raw(),
            label_id: label.raw(),
            flags: NODE_RECORD_VERSION & node_flags::VERSION_MASK,
            _pad1: [0; 3],
            property_ref: 0,
            inline_u32a: 0,
            inline_u32b: 0,
            vector_ref: 0,
            out_tel_ref: 0,
            in_tel_ref: 0,
            created_lsn: created_lsn.raw(),
        }
    }

    /// Extract the record-format version from `flags`.
    #[must_use]
    pub const fn version(self) -> u8 {
        self.flags & node_flags::VERSION_MASK
    }

    /// True if the tombstone bit is set.
    #[must_use]
    pub const fn is_deleted(self) -> bool {
        (self.flags & node_flags::DELETED) != 0
    }

    /// Is this physical record visible at `read_lsn`?
    ///
    /// Node records carry a lower MVCC bound plus the defensive deleted
    /// flag (the version-chain tombstone remains authoritative). Keeping the
    /// complete verdict here prevents physical accelerators from each
    /// re-deriving only part of the visibility rule.
    #[must_use]
    pub const fn is_visible_at(&self, read_lsn: Lsn) -> bool {
        self.created_lsn <= read_lsn.raw() && !self.is_deleted()
    }

    /// Serialize to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.label_id.to_le_bytes());
        buf[12] = self.flags;
        // bytes 13..16 are padding, left as zeros.
        buf[16..24].copy_from_slice(&self.property_ref.to_le_bytes());
        buf[24..28].copy_from_slice(&self.inline_u32a.to_le_bytes());
        buf[28..32].copy_from_slice(&self.inline_u32b.to_le_bytes());
        buf[32..40].copy_from_slice(&self.vector_ref.to_le_bytes());
        buf[40..48].copy_from_slice(&self.out_tel_ref.to_le_bytes());
        buf[48..56].copy_from_slice(&self.in_tel_ref.to_le_bytes());
        buf[56..64].copy_from_slice(&self.created_lsn.to_le_bytes());
        buf
    }

    /// Deserialize from exactly [`Self::SIZE`] bytes.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Result<Self> {
        let flags = bytes[12];
        let version = flags & node_flags::VERSION_MASK;
        if version != NODE_RECORD_VERSION {
            return Err(ArcGraphError::UnsupportedRecordVersion(version));
        }
        Ok(Self {
            id: u64::from_le_bytes(read_array::<8>(&bytes[0..8])),
            label_id: u32::from_le_bytes(read_array::<4>(&bytes[8..12])),
            flags,
            _pad1: [0; 3],
            property_ref: u64::from_le_bytes(read_array::<8>(&bytes[16..24])),
            inline_u32a: u32::from_le_bytes(read_array::<4>(&bytes[24..28])),
            inline_u32b: u32::from_le_bytes(read_array::<4>(&bytes[28..32])),
            vector_ref: u64::from_le_bytes(read_array::<8>(&bytes[32..40])),
            out_tel_ref: u64::from_le_bytes(read_array::<8>(&bytes[40..48])),
            in_tel_ref: u64::from_le_bytes(read_array::<8>(&bytes[48..56])),
            created_lsn: u64::from_le_bytes(read_array::<8>(&bytes[56..64])),
        })
    }
}

// ---------- RelRecord ----------

/// 96-byte relationship record. First 32 bytes = selection path (id,
/// type, flags, src, dst); remaining 64 bytes = materialization path
/// (properties + MVCC + weight). See design-v2 §3.2 for the motivation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RelRecord {
    /// Relationship identifier.
    pub id: u64,
    /// Interned relationship type.
    pub type_id: u32,
    /// Flags byte; same layout as [`NodeRecord::flags`].
    pub flags: u8,
    _pad1: [u8; 3],
    /// Source node identifier.
    pub src_id: u64,
    /// Destination node identifier.
    pub dst_id: u64,
    /// Reference into the property store.
    pub property_ref: u64,
    /// First inline cached numeric property.
    pub inline_u32a: u32,
    /// Second inline cached numeric property.
    pub inline_u32b: u32,
    /// MVCC lower visibility bound.
    pub created_lsn: u64,
    /// MVCC upper visibility bound. `Lsn::MAX` (`u64::MAX`) means alive.
    pub expired_lsn: u64,
    /// Edge weight (GDS-style).
    pub weight: f32,
    _pad2: [u8; 28],
}

impl RelRecord {
    /// On-disk size in bytes.
    pub const SIZE: usize = 96;

    /// Fresh live record.
    #[must_use]
    pub fn new(id: RelId, ty: TypeId, src: NodeId, dst: NodeId, created_lsn: Lsn) -> Self {
        Self {
            id: id.raw(),
            type_id: ty.raw(),
            flags: REL_RECORD_VERSION & node_flags::VERSION_MASK,
            _pad1: [0; 3],
            src_id: src.raw(),
            dst_id: dst.raw(),
            property_ref: 0,
            inline_u32a: 0,
            inline_u32b: 0,
            created_lsn: created_lsn.raw(),
            expired_lsn: Lsn::MAX.raw(),
            weight: 0.0,
            _pad2: [0; 28],
        }
    }

    /// Is this record alive at `read_lsn`?
    #[must_use]
    pub fn is_visible_at(&self, read_lsn: Lsn) -> bool {
        self.created_lsn <= read_lsn.raw() && read_lsn.raw() < self.expired_lsn
    }

    /// Serialize to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.type_id.to_le_bytes());
        buf[12] = self.flags;
        // bytes 13..16 padding.
        buf[16..24].copy_from_slice(&self.src_id.to_le_bytes());
        buf[24..32].copy_from_slice(&self.dst_id.to_le_bytes());
        buf[32..40].copy_from_slice(&self.property_ref.to_le_bytes());
        buf[40..44].copy_from_slice(&self.inline_u32a.to_le_bytes());
        buf[44..48].copy_from_slice(&self.inline_u32b.to_le_bytes());
        buf[48..56].copy_from_slice(&self.created_lsn.to_le_bytes());
        buf[56..64].copy_from_slice(&self.expired_lsn.to_le_bytes());
        buf[64..68].copy_from_slice(&self.weight.to_le_bytes());
        // bytes 68..96 padding.
        buf
    }

    /// Deserialize from exactly [`Self::SIZE`] bytes.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Result<Self> {
        let flags = bytes[12];
        let version = flags & node_flags::VERSION_MASK;
        if version != REL_RECORD_VERSION {
            return Err(ArcGraphError::UnsupportedRecordVersion(version));
        }
        Ok(Self {
            id: u64::from_le_bytes(read_array::<8>(&bytes[0..8])),
            type_id: u32::from_le_bytes(read_array::<4>(&bytes[8..12])),
            flags,
            _pad1: [0; 3],
            src_id: u64::from_le_bytes(read_array::<8>(&bytes[16..24])),
            dst_id: u64::from_le_bytes(read_array::<8>(&bytes[24..32])),
            property_ref: u64::from_le_bytes(read_array::<8>(&bytes[32..40])),
            inline_u32a: u32::from_le_bytes(read_array::<4>(&bytes[40..44])),
            inline_u32b: u32::from_le_bytes(read_array::<4>(&bytes[44..48])),
            created_lsn: u64::from_le_bytes(read_array::<8>(&bytes[48..56])),
            expired_lsn: u64::from_le_bytes(read_array::<8>(&bytes[56..64])),
            weight: f32::from_le_bytes(read_array::<4>(&bytes[64..68])),
            _pad2: [0; 28],
        })
    }
}

impl Eq for RelRecord {}

// ---------- TelEntry ----------

/// 32-byte TEL (Transactional Edge Log) entry. Two per cache line for
/// fast forward/backward scanning. See design-v2 §3.3 and LiveGraph
/// VLDB 2020.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TelEntry {
    /// Destination vertex for this edge.
    pub dst_id: u64,
    /// Relationship id, for full-record lookup in the rel store.
    pub rel_id: u64,
    /// MVCC lower visibility bound.
    pub created_lsn: u64,
    /// MVCC upper visibility bound. `u64::MAX` = alive.
    pub expired_lsn: u64,
}

impl TelEntry {
    /// On-disk size in bytes.
    pub const SIZE: usize = 32;

    /// Fresh live entry.
    #[must_use]
    pub fn new(dst: NodeId, rel: RelId, created_lsn: Lsn) -> Self {
        Self {
            dst_id: dst.raw(),
            rel_id: rel.raw(),
            created_lsn: created_lsn.raw(),
            expired_lsn: Lsn::MAX.raw(),
        }
    }

    /// Is this entry visible at `read_lsn`?
    #[must_use]
    pub const fn is_visible_at(&self, read_lsn: Lsn) -> bool {
        self.created_lsn <= read_lsn.raw() && read_lsn.raw() < self.expired_lsn
    }

    /// Serialize to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.dst_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.rel_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.created_lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.expired_lsn.to_le_bytes());
        buf
    }

    /// Deserialize from exactly [`Self::SIZE`] bytes. No versioning
    /// byte on TEL entries; they are gated by the containing TEL block.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            dst_id: u64::from_le_bytes(read_array::<8>(&bytes[0..8])),
            rel_id: u64::from_le_bytes(read_array::<8>(&bytes[8..16])),
            created_lsn: u64::from_le_bytes(read_array::<8>(&bytes[16..24])),
            expired_lsn: u64::from_le_bytes(read_array::<8>(&bytes[24..32])),
        }
    }
}

// ---------- helpers ----------

/// Copy the first `N` bytes of `slice` into a fixed-size array.
///
/// Panics only if called with a slice shorter than `N`; every call in
/// this module has a statically-sized bound, so this is total in practice.
#[inline]
fn read_array<const N: usize>(slice: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&slice[..N]);
    out
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    // ---- size / alignment ----

    #[test]
    fn page_header_size_is_40_and_align_is_8() {
        assert_eq!(core::mem::size_of::<PageHeader>(), 40);
        assert_eq!(core::mem::align_of::<PageHeader>(), 8);
    }

    #[test]
    fn node_record_size_is_64_and_cache_aligned() {
        assert_eq!(core::mem::size_of::<NodeRecord>(), 64);
        assert_eq!(core::mem::align_of::<NodeRecord>(), 64);
    }

    #[test]
    fn rel_record_size_is_96() {
        assert_eq!(core::mem::size_of::<RelRecord>(), 96);
    }

    #[test]
    fn tel_entry_size_is_32() {
        assert_eq!(core::mem::size_of::<TelEntry>(), 32);
    }

    // ---- page type ----

    #[test]
    fn page_type_roundtrip() {
        for byte in 0..=8u8 {
            let ty = PageType::from_byte(byte).unwrap();
            assert_eq!(ty.as_byte(), byte);
        }
    }

    #[test]
    fn index_overflow_variant_is_8() {
        assert_eq!(PageType::IndexOverflow.as_byte(), 8);
        assert_eq!(PageType::from_byte(8).unwrap(), PageType::IndexOverflow);
    }

    #[test]
    fn unknown_page_type_is_error() {
        let e = PageType::from_byte(99).unwrap_err();
        assert!(matches!(e, ArcGraphError::UnknownPageType(99)));
    }

    // ---- unit happy-path ----

    #[test]
    fn page_header_roundtrip_unit() {
        let h = PageHeader {
            magic: PAGE_MAGIC,
            version: PAGE_HEADER_VERSION,
            page_type: PageType::Node.as_byte(),
            flags: 0xA5A5,
            page_id: 0x1234_5678_9ABC_DEF0,
            lsn: 0xCAFE_F00D_1234_5678,
            tenant_id: TenantId::DEFAULT.raw(),
            checksum: 0xDEAD_BEEF,
            slot_count: 3,
            free_space: 8100,
        };
        let bytes = h.to_bytes();
        let back = PageHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn page_header_tenant_id_roundtrips() {
        for &tenant in &[
            TenantId::SYSTEM,
            TenantId::DEFAULT,
            TenantId::new(u64::MAX - 1),
        ] {
            let h = PageHeader::new(PageId::new(42), PageType::Tel, tenant);
            let bytes = h.to_bytes();
            let back = PageHeader::from_bytes(&bytes).unwrap();
            assert_eq!(back.tenant_id, tenant.raw());
        }
    }

    #[test]
    fn page_header_rejects_bad_magic() {
        let mut bytes =
            PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT).to_bytes();
        bytes[0] = 0;
        let err = PageHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::BadPageMagic { .. }));
    }

    #[test]
    fn page_header_rejects_bad_version() {
        let mut bytes =
            PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT).to_bytes();
        bytes[4] = 99;
        let err = PageHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnsupportedRecordVersion(99)));
    }

    #[test]
    fn page_header_rejects_bad_page_type() {
        let mut bytes =
            PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT).to_bytes();
        bytes[5] = 99;
        let err = PageHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnknownPageType(99)));
    }

    // ---- node record ----

    #[test]
    fn node_record_flags_helpers() {
        let mut n = NodeRecord::new(NodeId::new(7), LabelId::new(3), Lsn::new(42));
        assert_eq!(n.version(), NODE_RECORD_VERSION);
        assert!(!n.is_deleted());
        assert!(!n.is_visible_at(Lsn::new(41)));
        assert!(n.is_visible_at(Lsn::new(42)));
        n.flags |= node_flags::DELETED;
        assert!(n.is_deleted());
        assert!(!n.is_visible_at(Lsn::MAX));
    }

    // ---- rel record visibility ----

    #[test]
    fn rel_record_visibility() {
        let r = RelRecord::new(
            RelId::new(1),
            TypeId::new(2),
            NodeId::new(3),
            NodeId::new(4),
            Lsn::new(10),
        );
        assert!(!r.is_visible_at(Lsn::new(9)));
        assert!(r.is_visible_at(Lsn::new(10)));
        assert!(r.is_visible_at(Lsn::new(u64::MAX - 1)));
    }

    // ---- property: roundtrip ----

    proptest! {
        #[test]
        fn page_header_roundtrip(
            flags in any::<u16>(),
            page_id in any::<u64>(),
            lsn in any::<u64>(),
            tenant_id in any::<u64>(),
            checksum in any::<u32>(),
            slot_count in any::<u16>(),
            free_space in any::<u16>(),
            page_type_byte in 0u8..=7,
        ) {
            let h = PageHeader {
                magic: PAGE_MAGIC,
                version: PAGE_HEADER_VERSION,
                page_type: page_type_byte,
                flags,
                page_id,
                lsn,
                tenant_id,
                checksum,
                slot_count,
                free_space,
            };
            let bytes = h.to_bytes();
            let back = PageHeader::from_bytes(&bytes).unwrap();
            prop_assert_eq!(h, back);
        }

        #[test]
        fn node_record_roundtrip(
            id in any::<u64>(),
            label_id in any::<u32>(),
            high_flags in 0u8..32,
            property_ref in any::<u64>(),
            a in any::<u32>(),
            b in any::<u32>(),
            vector_ref in any::<u64>(),
            out_ref in any::<u64>(),
            in_ref in any::<u64>(),
            created_lsn in any::<u64>(),
        ) {
            let flags = (high_flags << 3) | (NODE_RECORD_VERSION & node_flags::VERSION_MASK);
            let n = NodeRecord {
                id,
                label_id,
                flags,
                _pad1: [0; 3],
                property_ref,
                inline_u32a: a,
                inline_u32b: b,
                vector_ref,
                out_tel_ref: out_ref,
                in_tel_ref: in_ref,
                created_lsn,
            };
            let bytes = n.to_bytes();
            let back = NodeRecord::from_bytes(&bytes).unwrap();
            prop_assert_eq!(n, back);
        }

        #[test]
        fn rel_record_roundtrip(
            id in any::<u64>(),
            type_id in any::<u32>(),
            high_flags in 0u8..32,
            src_id in any::<u64>(),
            dst_id in any::<u64>(),
            property_ref in any::<u64>(),
            a in any::<u32>(),
            b in any::<u32>(),
            created_lsn in any::<u64>(),
            expired_lsn in any::<u64>(),
            weight_bits in any::<u32>(),
        ) {
            // Skip NaN weights so equality holds.
            let weight = f32::from_bits(weight_bits);
            prop_assume!(!weight.is_nan());
            let flags = (high_flags << 3) | (REL_RECORD_VERSION & node_flags::VERSION_MASK);
            let r = RelRecord {
                id,
                type_id,
                flags,
                _pad1: [0; 3],
                src_id,
                dst_id,
                property_ref,
                inline_u32a: a,
                inline_u32b: b,
                created_lsn,
                expired_lsn,
                weight,
                _pad2: [0; 28],
            };
            let bytes = r.to_bytes();
            let back = RelRecord::from_bytes(&bytes).unwrap();
            prop_assert_eq!(r, back);
        }

        #[test]
        fn tel_entry_roundtrip(
            dst_id in any::<u64>(),
            rel_id in any::<u64>(),
            created_lsn in any::<u64>(),
            expired_lsn in any::<u64>(),
        ) {
            let t = TelEntry {
                dst_id,
                rel_id,
                created_lsn,
                expired_lsn,
            };
            let bytes = t.to_bytes();
            let back = TelEntry::from_bytes(&bytes);
            prop_assert_eq!(t, back);
        }

        #[test]
        fn corrupted_magic_is_rejected(
            bad in prop::num::u32::ANY.prop_filter("must not be PAGE_MAGIC", |v| *v != PAGE_MAGIC),
        ) {
            let mut bytes =
                PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT).to_bytes();
            bytes[0..4].copy_from_slice(&bad.to_le_bytes());
            let err = PageHeader::from_bytes(&bytes).unwrap_err();
            let is_bad_magic = matches!(err, ArcGraphError::BadPageMagic { .. });
            prop_assert!(is_bad_magic);
        }

        #[test]
        fn corrupted_node_version_is_rejected(
            bad_ver in prop::num::u8::ANY.prop_filter(
                "must not match NODE_RECORD_VERSION",
                |v| (*v & node_flags::VERSION_MASK) != NODE_RECORD_VERSION,
            ),
        ) {
            let n = NodeRecord::new(NodeId::new(1), LabelId::new(1), Lsn::new(1));
            let mut bytes = n.to_bytes();
            bytes[12] = bad_ver;
            let err = NodeRecord::from_bytes(&bytes).unwrap_err();
            let is_bad_version = matches!(err, ArcGraphError::UnsupportedRecordVersion(_));
            prop_assert!(is_bad_version);
        }

        #[test]
        fn corrupted_rel_version_is_rejected(
            bad_ver in prop::num::u8::ANY.prop_filter(
                "must not match REL_RECORD_VERSION",
                |v| (*v & node_flags::VERSION_MASK) != REL_RECORD_VERSION,
            ),
        ) {
            let r = RelRecord::new(
                RelId::new(1),
                TypeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                Lsn::new(4),
            );
            let mut bytes = r.to_bytes();
            bytes[12] = bad_ver;
            let err = RelRecord::from_bytes(&bytes).unwrap_err();
            let is_bad_version = matches!(err, ArcGraphError::UnsupportedRecordVersion(_));
            prop_assert!(is_bad_version);
        }

        #[test]
        fn corrupted_page_type_byte_is_rejected(
            // v2 M1: `9 = PropSlotted` joined the valid set; the highest
            // valid discriminant is now 9.
            bad_type in prop::num::u8::ANY.prop_filter("must be > 9", |v| *v > 9),
        ) {
            let mut bytes =
                PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT).to_bytes();
            bytes[5] = bad_type;
            let err = PageHeader::from_bytes(&bytes).unwrap_err();
            let is_bad = matches!(err, ArcGraphError::UnknownPageType(_));
            prop_assert!(is_bad);
        }

        #[test]
        fn node_record_decode_zeroes_padding(
            id in any::<u64>(),
            label in any::<u32>(),
        ) {
            // Encode with zero padding, then fill the padding bytes in
            // the serialized form with garbage and decode. The decoded
            // struct should have all-zero padding regardless.
            let n = NodeRecord::new(NodeId::new(id), LabelId::new(label), Lsn::new(42));
            let mut bytes = n.to_bytes();
            bytes[13] = 0xFF;
            bytes[14] = 0xAA;
            bytes[15] = 0x55;
            let back = NodeRecord::from_bytes(&bytes).unwrap();
            prop_assert_eq!(back, n, "padding garbage must not leak into the decoded struct");
        }

        #[test]
        fn rel_record_decode_zeroes_padding(
            id in any::<u64>(),
        ) {
            let r = RelRecord::new(
                RelId::new(id),
                TypeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                Lsn::new(4),
            );
            let mut bytes = r.to_bytes();
            // Fill every padding region (13..16 and 68..96) with garbage.
            for b in &mut bytes[13..16] {
                *b = 0xAA;
            }
            for b in &mut bytes[68..96] {
                *b = 0xBB;
            }
            let back = RelRecord::from_bytes(&bytes).unwrap();
            prop_assert_eq!(back, r, "rel padding garbage must not leak into decoded struct");
        }
    }

    // ---- flag-bit layout (independent of any record instance) ----

    #[test]
    fn node_flag_masks_are_disjoint_and_cover_low_bits() {
        use node_flags::*;
        // Version lives in the low 3 bits; the named flags start at bit 3
        // and must not overlap with VERSION_MASK or each other.
        assert_eq!(VERSION_MASK, 0b0000_0111);
        assert_eq!(VERSION_MASK & DELETED, 0);
        assert_eq!(VERSION_MASK & HAS_VECTOR, 0);
        assert_eq!(VERSION_MASK & HAS_EXTENDED, 0);
        assert_eq!(DELETED & HAS_VECTOR, 0);
        assert_eq!(DELETED & HAS_EXTENDED, 0);
        assert_eq!(HAS_VECTOR & HAS_EXTENDED, 0);
    }
}
