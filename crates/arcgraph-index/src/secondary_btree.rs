//! Secondary B-tree index `(TenantId, LabelId, StringId, PropertyValue)
//! → [NodeId]` (M2-34).
//!
//! A secondary index on node properties: given a property label
//! (`LabelId`), a property key (`StringId` interned via
//! [`arcgraph_storage::intern::InternTable`]), and a property value
//! (U32 / U64 / StringId — bytes / BLOBs are M3), return the set of
//! nodes whose property matches.
//!
//! # Design references
//!
//! - **DEC-15** — inline N=4 NodeIds in the leaf entry, overflow
//!   chain keyed by the same tuple for the 5th+ duplicate.
//! - **DEC-16** — we duplicate the page-ops scaffold from
//!   `arcgraph_storage::primary_index` and specialize for the
//!   secondary key + duplicate semantics. Extraction to a generic
//!   `BTree<K, V>` waits for M3 when HNSW becomes the third consumer.
//! - **DEC-17** — pages live in an in-memory
//!   `DashMap<PageId, Arc<RwLock<Box<[u8; PAGE_SIZE]>>>>` — BufferPool
//!   migration lands with M2.e WAL replay.
//! - **DEC-19** (this milestone) — key encoding: LE integers + a
//!   1-byte variant tag placed inside the 24 B key; in-memory
//!   ordering uses the derived `Ord` impl (LE bytes do not memcmp-
//!   sort, so we never memcmp the raw bytes). U64 values are capped
//!   at `2^56 - 1` to fit the 7-byte on-disk value slot; the cap is
//!   enforced at encode time.
//! - **ADR-023** — the secondary index is a read-accelerator under
//!   the same contract as the primary: readers pre-filter via the
//!   index then verify snapshot visibility by calling
//!   `crud::read_node(tx, id)` on each candidate before returning it.
//!
//! # Layout
//!
//! One shared B+tree across all tenants (same rationale as DEC-8 for
//! the primary). Keys sort tenant-major, then label, then
//! property_key, then `PropertyValue` variant tag (U32 < U64 <
//! StringId), then value.
//!
//! Leaf page (`PageType::IndexLeaf`):
//!
//! ```text
//!  0.. 40  PageHeader
//! 40.. 40 + 64 * N             N leaf entries, each:
//!                                  key: 24 B (see SecondaryKey encoding)
//!                                  inline: 4 × NodeId u64 LE    = 32 B
//!                                           (slot = 0 means "empty")
//!                                  overflow_head: PageId u64 LE = 8 B
//!                                           (0 = no overflow chain)
//! ```
//!
//! Capacity: 127 entries per leaf (`(8192 - 40) / 64 = 127.375`).
//!
//! # Delete policy
//!
//! Mark-only at the **NodeId slot** granularity: `remove(k, n)` zeros
//! the matching slot (in `inline[]` or somewhere in the overflow
//! chain) but leaves the leaf entry and the chain in place. A later
//! `insert(k, m)` reuses the first zero slot it finds. Overflow
//! pages are never reclaimed in M2.d.
//!
//! Per Philosophy §H, this keeps the write path simple. Compaction
//! is an M2.e+ task.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{
    ArcGraphError, LabelId, Lsn, NodeId, PageHeader, PageId, PageType, Result as CoreResult,
    StringId, TenantId,
};
use arcgraph_storage::mutation_log::{IndexHandle, PageStoreKind, TxnMutationLog};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::encode_index_page_payload;
use arcgraph_storage::secondary_handle::{
    IndexState, SecondaryIndexHandle, SecondaryIndexHandleError, SecondaryIndexValue,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::SecondaryPageStoreHandle;
use arcgraph_storage::wal::{WalHandle, WalRecordType};
use arcgraph_storage::{SideChannelWrite, StagedEmit};
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::{Mutex, RawRwLock, RwLock, RwLockWriteGuard};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────
// Public key / value types
// ─────────────────────────────────────────────────────────────────────

/// Property-value variants supported by the secondary index in M2.d.
///
/// `Bytes` / BLOBs / bool / i32 / f64 are intentionally out of scope —
/// see the module docs. Ord is derived: variant-major (U32 < U64 <
/// StringId < StrHash), then by the inner payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PropertyValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer. Values strictly greater than
    /// `(1 << 56) - 1` cannot be encoded; see the module-level
    /// DEC-19 note and [`SecondaryIndexError::ValueOverflow`].
    U64(u64),
    /// Interned string-id (e.g. a categorical / enum property value
    /// from M2-32's intern table). **RC-4 note (#1366):** the
    /// *user-visible property index* does NOT key strings by
    /// `StringId` — it uses [`Self::StrHash`] instead, to avoid the
    /// intern-memory blowup + the read-path intern-mutation hazard
    /// (RC-3/RC-4). This variant is retained for the M2.d
    /// storage-internal positional-field index which keys interned
    /// categorical ids directly.
    StringId(StringId),
    /// **RC-4 (#1366)** — a 56-bit hash of a UTF-8 string value's
    /// bytes ([`hash_str_56`]). The user-visible property index keys
    /// string VALUES here rather than as interned [`Self::StringId`]:
    ///
    /// - **No intern growth.** Keying by hash means an index lookup
    ///   for a never-seen value (`MATCH (n:User {email:"nobody@x"})`)
    ///   never inserts into `InternTable` — the value is hashed in
    ///   place, so the read path never mutates shared durable-adjacent
    ///   state (closes RC-3 for the value side; the property-KEY stays
    ///   interned, hence [`SecondaryKey::property_key`] is still a
    ///   `StringId`).
    /// - **Collisions are read-safe.** Two distinct strings that
    ///   56-bit-collide land on the same key → both return as
    ///   candidates from [`SecondaryIndex::lookup`]. The MANDATORY
    ///   candidate-then-verify equality recheck (ADR-023) hydrates each
    ///   candidate and re-compares the actual property value, dropping
    ///   the non-matching one. A collision is therefore just an extra
    ///   candidate that fails the recheck — never a wrong result.
    ///
    /// The 56-bit width fits the same 7-byte on-disk value slot as
    /// [`Self::U64`] (`< 2^56`), so no key-layout change is needed.
    StrHash(u64),
}

impl PropertyValue {
    /// Tag byte written at offset 16 of a `SecondaryKey`. Keeps
    /// variants strictly ordered under `<` on the tag alone: U32
    /// < U64 < StringId < StrHash.
    #[inline]
    const fn variant_tag(self) -> u8 {
        match self {
            Self::U32(_) => 0,
            Self::U64(_) => 1,
            Self::StringId(_) => 2,
            // RC-4 (#1366): a new tag disjoint from the M2.d set so a
            // hash-keyed value never aliases an interned-id-keyed one.
            Self::StrHash(_) => 3,
        }
    }
}

/// **RC-4 (#1366)** — hash a string value's UTF-8 bytes to a 56-bit key
/// for [`PropertyValue::StrHash`].
///
/// Uses [`std::collections::hash_map::DefaultHasher`] — currently
/// **SipHash-1-3 seeded with the fixed key `(0, 0)`** (per the current
/// std impl) — over the raw bytes, truncated to the low 56 bits to fit
/// the 7-byte on-disk value slot (DEC-19). The full 64-bit hash is
/// masked with `(1 << 56) - 1`; the truncation only *increases* the
/// collision rate relative to the full hash, and every collision is
/// absorbed by the mandatory candidate-then-verify recheck — so 56 bits
/// is correctness-safe (it is a probabilistic *candidate reducer*, never
/// an authority on equality).
///
/// Determinism: the fixed `(0, 0)` seed makes the same bytes hash to the
/// same 56-bit value across processes (unlike `RandomState`) — required
/// for the key to be stable across restart / WAL replay.
///
/// **On-disk-key-stability caveat (#1366 R1 NIT-2):** these 56-bit
/// values become ON-DISK secondary-index keys. `DefaultHasher`'s output
/// is stable *within* a toolchain but is **not contractually guaranteed
/// across Rust toolchain upgrades** — a future std change to the hash
/// algorithm or seed would make old on-disk keys unfindable (a
/// candidate-verify *false-negative*, not wrong results, but still a
/// WAL-replay hygiene hazard requiring a key migration). The pinned
/// canary test `hash_str_56_canary_is_stable` locks the current value of
/// `hash_str_56("arcgraph_canary")` so any such change FAILS loudly as a
/// signal to migrate keys rather than degrade silently.
#[must_use]
pub fn hash_str_56(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    // `DefaultHasher::new()` = SipHash-1-3 seeded (0, 0) per current std,
    // so the output is deterministic across processes (unlike
    // `RandomState`). See the doc comment's on-disk-stability caveat +
    // the `hash_str_56_canary_is_stable` regression test.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.as_bytes().hash(&mut h);
    h.finish() & ((1u64 << 56) - 1)
}

/// Secondary-index key: tenant-major, then label, then
/// `property_key` (the interned property name), then
/// [`PropertyValue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SecondaryKey {
    /// Tenant that owns the indexed record.
    pub tenant: TenantId,
    /// Node label that carries the indexed property.
    pub label: LabelId,
    /// Interned property-name id (via `InternTable`).
    pub property_key: StringId,
    /// Property value.
    pub value: PropertyValue,
}

impl SecondaryKey {
    /// On-disk key size (bytes).
    pub const SIZE: usize = 24;

    /// Convenience constructor.
    #[inline]
    #[must_use]
    pub const fn new(
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: PropertyValue,
    ) -> Self {
        Self {
            tenant,
            label,
            property_key,
            value,
        }
    }

    /// Encode this key into `out`, which must be exactly
    /// [`Self::SIZE`] bytes. Returns
    /// [`SecondaryIndexError::ValueOverflow`] if `value` is a U64 with
    /// its top byte set.
    ///
    /// Layout:
    /// ```text
    ///  0.. 8  tenant       u64 little-endian
    ///  8..12  label        u32 little-endian
    /// 12..16  property_key u32 little-endian
    /// 16..17  variant_tag  u8
    /// 17..24  value_bytes  7 bytes little-endian, zero-padded
    /// ```
    pub fn encode_into(&self, out: &mut [u8]) -> Result<(), SecondaryIndexError> {
        debug_assert_eq!(out.len(), Self::SIZE);
        out[0..8].copy_from_slice(&self.tenant.raw().to_le_bytes());
        out[8..12].copy_from_slice(&self.label.raw().to_le_bytes());
        out[12..16].copy_from_slice(&self.property_key.raw().to_le_bytes());
        out[16] = self.value.variant_tag();
        let mut value_bytes = [0u8; 7];
        match self.value {
            PropertyValue::U32(v) => {
                value_bytes[..4].copy_from_slice(&v.to_le_bytes());
            }
            PropertyValue::U64(v) => {
                if v > (1u64 << 56) - 1 {
                    return Err(SecondaryIndexError::ValueOverflow { value: v });
                }
                value_bytes.copy_from_slice(&v.to_le_bytes()[..7]);
            }
            PropertyValue::StringId(s) => {
                value_bytes[..4].copy_from_slice(&s.raw().to_le_bytes());
            }
            // RC-4 (#1366): a 56-bit hash fits the 7-byte slot by
            // construction (`hash_str_56` masks to `< 2^56`); the
            // debug-assert documents the invariant without a runtime
            // branch on the happy path.
            PropertyValue::StrHash(h) => {
                debug_assert!(h < (1u64 << 56), "StrHash must be masked to 56 bits");
                value_bytes.copy_from_slice(&h.to_le_bytes()[..7]);
            }
        }
        out[17..24].copy_from_slice(&value_bytes);
        Ok(())
    }

    /// Decode a 24-byte buffer into a `SecondaryKey`.
    pub fn decode(bytes: &[u8]) -> Result<Self, SecondaryIndexError> {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        let tenant = TenantId::new(u64::from_le_bytes(
            bytes[0..8].try_into().expect("slice size asserted above"),
        ));
        let label = LabelId::new(u32::from_le_bytes(
            bytes[8..12].try_into().expect("slice size asserted above"),
        ));
        let property_key = StringId::new(u32::from_le_bytes(
            bytes[12..16].try_into().expect("slice size asserted above"),
        ));
        let tag = bytes[16];
        let value = match tag {
            0 => {
                let raw = u32::from_le_bytes(
                    bytes[17..21].try_into().expect("slice size asserted above"),
                );
                if bytes[21..24] != [0u8; 3] {
                    return Err(SecondaryIndexError::CorruptKey {
                        reason: "U32 variant must have zero-padded high bytes".into(),
                    });
                }
                PropertyValue::U32(raw)
            }
            1 => {
                let mut full = [0u8; 8];
                full[..7].copy_from_slice(&bytes[17..24]);
                PropertyValue::U64(u64::from_le_bytes(full))
            }
            2 => {
                let raw = u32::from_le_bytes(
                    bytes[17..21].try_into().expect("slice size asserted above"),
                );
                if bytes[21..24] != [0u8; 3] {
                    return Err(SecondaryIndexError::CorruptKey {
                        reason: "StringId variant must have zero-padded high bytes".into(),
                    });
                }
                PropertyValue::StringId(StringId::new(raw))
            }
            // RC-4 (#1366): StrHash decodes the 7-byte LE value into
            // the low 56 bits (top byte implicitly zero).
            3 => {
                let mut full = [0u8; 8];
                full[..7].copy_from_slice(&bytes[17..24]);
                PropertyValue::StrHash(u64::from_le_bytes(full))
            }
            other => {
                return Err(SecondaryIndexError::CorruptKey {
                    reason: format!("unknown PropertyValue variant tag: {other}"),
                });
            }
        };
        Ok(Self {
            tenant,
            label,
            property_key,
            value,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Local error surface for secondary-index operations. Converted into
/// a caller-appropriate error at the arcgraph-storage boundary via the
/// `SecondaryIndexHandle` trait.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SecondaryIndexError {
    /// Page with this id is not tracked by the index.
    #[error("secondary index: page {0:?} not mapped")]
    MissingPage(PageId),
    /// Page header or body did not match expected invariants.
    #[error("secondary index: page {page_id:?} corrupt: {reason}")]
    CorruptPage {
        /// Offending page id.
        page_id: PageId,
        /// Human-readable reason.
        reason: String,
    },
    /// 24-byte key payload did not match the expected shape.
    #[error("secondary index: corrupt key: {reason}")]
    CorruptKey {
        /// Human-readable reason.
        reason: String,
    },
    /// Caller attempted to encode a `PropertyValue::U64` whose payload
    /// exceeds the 7-byte on-disk slot. See DEC-19.
    #[error(
        "secondary index: U64 value {value} exceeds 7-byte on-disk limit (2^56 - 1 = {max})",
        max = (1u64 << 56) - 1,
    )]
    ValueOverflow {
        /// The rejected value.
        value: u64,
    },
    /// Inline NodeId array is full and overflow-chain allocation is
    /// not yet implemented (reserved for commit 5 of M2-34).
    #[error(
        "secondary index: inline NodeId array full for key {key:?}; overflow chain not yet implemented"
    )]
    InlineFull {
        /// The key whose inline array is saturated.
        key: SecondaryKey,
    },
    /// Core layer rejected a page header (bubbled from
    /// `PageHeader::from_bytes`).
    #[error(transparent)]
    Core(#[from] ArcGraphError),
}

// ─────────────────────────────────────────────────────────────────────
// Page-level constants
// ─────────────────────────────────────────────────────────────────────

/// Number of `NodeId` slots stored inline in a leaf entry (DEC-15).
pub const INLINE_NODEID_COUNT: usize = 4;

/// `SecondaryKey::SIZE + INLINE_NODEID_COUNT × 8 + 8` (overflow_head).
pub const LEAF_ENTRY_SIZE: usize = SecondaryKey::SIZE + INLINE_NODEID_COUNT * 8 + 8;
const _: () = assert!(LEAF_ENTRY_SIZE == 64);

/// Offset of the first entry on a leaf page (immediately after the
/// standard 40-byte page header).
pub const LEAF_ENTRY_OFFSET: usize = PageHeader::SIZE;

/// Maximum leaf entries per page (DEC-15: 127 per `(PAGE_SIZE - 40) /
/// 64 = 127.375`).
pub const LEAF_CAPACITY: u16 = ((PAGE_SIZE - LEAF_ENTRY_OFFSET) / LEAF_ENTRY_SIZE) as u16;
const _: () = assert!(LEAF_CAPACITY == 127);

// ─────────────────────────────────────────────────────────────────────
// Raw page buffer type
// ─────────────────────────────────────────────────────────────────────

/// `[u8; PAGE_SIZE]` — the raw byte buffer each page is backed by.
pub type PageBuf = [u8; PAGE_SIZE];

/// Allocate a zero-initialized page buffer with the given
/// [`PageType`] header stamped at offset 0.
#[must_use]
pub fn fresh_page_buf(page_id: PageId, page_type: PageType) -> Box<PageBuf> {
    let mut buf: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
    let header = PageHeader::new(page_id, page_type, TenantId::SYSTEM);
    buf[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
    buf
}

fn read_header(buf: &PageBuf) -> CoreResult<PageHeader> {
    let arr: &[u8; PageHeader::SIZE] = (&buf[..PageHeader::SIZE])
        .try_into()
        .expect("header slice is 40 bytes");
    PageHeader::from_bytes(arr)
}

fn write_header(buf: &mut PageBuf, header: &PageHeader) {
    buf[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
}

// ─────────────────────────────────────────────────────────────────────
// LeafEntry (value type)
// ─────────────────────────────────────────────────────────────────────

/// In-memory representation of one leaf slot. Used by tests and by
/// the tree-level insert/split/merge code once the rest of the
/// scaffolding lands.
///
/// Slots set to `NodeId::ZERO` (raw 0) represent empty positions —
/// they're either never-used or tombstoned by `remove`. A leaf entry
/// is valid even when every inline slot is zero and `overflow_head ==
/// PageId::ZERO`; lookup returns an empty `Vec` in that case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafEntry {
    /// The secondary key this entry indexes.
    pub key: SecondaryKey,
    /// Inline NodeId slots; zero entries are empty.
    pub inline: [NodeId; INLINE_NODEID_COUNT],
    /// Head of the overflow chain, or [`PageId::ZERO`] if no chain
    /// has been allocated yet.
    pub overflow_head: PageId,
}

impl LeafEntry {
    /// Empty leaf entry for `key` with no nodes and no overflow.
    #[must_use]
    pub const fn empty(key: SecondaryKey) -> Self {
        Self {
            key,
            inline: [NodeId::ZERO; INLINE_NODEID_COUNT],
            overflow_head: PageId::ZERO,
        }
    }

    /// Encode into a 64-byte buffer.
    pub fn encode_into(&self, out: &mut [u8]) -> Result<(), SecondaryIndexError> {
        debug_assert_eq!(out.len(), LEAF_ENTRY_SIZE);
        self.key.encode_into(&mut out[..SecondaryKey::SIZE])?;
        for (i, n) in self.inline.iter().enumerate() {
            let off = SecondaryKey::SIZE + i * 8;
            out[off..off + 8].copy_from_slice(&n.raw().to_le_bytes());
        }
        let off = SecondaryKey::SIZE + INLINE_NODEID_COUNT * 8;
        out[off..off + 8].copy_from_slice(&self.overflow_head.raw().to_le_bytes());
        Ok(())
    }

    /// Decode a 64-byte buffer.
    pub fn decode(bytes: &[u8]) -> Result<Self, SecondaryIndexError> {
        debug_assert_eq!(bytes.len(), LEAF_ENTRY_SIZE);
        let key = SecondaryKey::decode(&bytes[..SecondaryKey::SIZE])?;
        let mut inline = [NodeId::ZERO; INLINE_NODEID_COUNT];
        for (i, slot) in inline.iter_mut().enumerate() {
            let off = SecondaryKey::SIZE + i * 8;
            *slot = NodeId::new(u64::from_le_bytes(
                bytes[off..off + 8]
                    .try_into()
                    .expect("slice size asserted above"),
            ));
        }
        let off = SecondaryKey::SIZE + INLINE_NODEID_COUNT * 8;
        let overflow_head = PageId::new(u64::from_le_bytes(
            bytes[off..off + 8]
                .try_into()
                .expect("slice size asserted above"),
        ));
        Ok(Self {
            key,
            inline,
            overflow_head,
        })
    }

    /// Live (non-zero) inline slots.
    pub fn live_inline(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inline.iter().copied().filter(|n| n.raw() != 0)
    }

    /// Index of the first empty inline slot (raw == 0), if any.
    #[must_use]
    pub fn first_empty_inline(&self) -> Option<usize> {
        self.inline.iter().position(|n| n.raw() == 0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// LeafPage: codec + on-page search
// ─────────────────────────────────────────────────────────────────────

/// Read-only accessor for a leaf page's contents.
pub struct LeafPageRef<'a> {
    bytes: &'a PageBuf,
    header: PageHeader,
}

impl fmt::Debug for LeafPageRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeafPageRef")
            .field("page_id", &self.page_id())
            .field("entry_count", &self.entry_count())
            .finish()
    }
}

impl<'a> LeafPageRef<'a> {
    /// Open a leaf view, validating the header.
    pub fn open(bytes: &'a PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexLeaf.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexLeaf page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Page id read from the header.
    #[must_use]
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Number of leaf entries (including entries whose NodeId slots
    /// are all zero).
    #[must_use]
    pub fn entry_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Decode the entry at `index`.
    pub fn entry(&self, index: u16) -> Result<LeafEntry, SecondaryIndexError> {
        let n = self.entry_count();
        if index >= n {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("entry index {index} out of range (count={n})"),
            });
        }
        let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE;
        LeafEntry::decode(&self.bytes[off..off + LEAF_ENTRY_SIZE])
    }

    /// Binary-search for `needle`. Returns the position of an exact
    /// match or the sorted insertion position.
    pub fn find(&self, needle: SecondaryKey) -> Result<LeafFindResult, SecondaryIndexError> {
        let n = self.entry_count();
        let mut lo: u32 = 0;
        let mut hi: u32 = u32::from(n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let e = self.entry(mid as u16)?;
            match e.key.cmp(&needle) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    return Ok(LeafFindResult::Found { index: mid as u16 });
                }
            }
        }
        Ok(LeafFindResult::Absent {
            insert_at: lo as u16,
        })
    }

    /// Point lookup: return the live inline `NodeId`s for `key` and
    /// its `overflow_head` (caller walks the chain). Returns `None`
    /// when the key is absent.
    pub fn lookup_entry(
        &self,
        key: SecondaryKey,
    ) -> Result<Option<LeafEntry>, SecondaryIndexError> {
        match self.find(key)? {
            LeafFindResult::Found { index } => Ok(Some(self.entry(index)?)),
            LeafFindResult::Absent { .. } => Ok(None),
        }
    }
}

/// Result of a binary-search on the leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafFindResult {
    /// Exact match at slot `index`.
    Found {
        /// Slot index within the leaf (0 ≤ index < entry_count).
        index: u16,
    },
    /// Absent; sorted insertion position is `insert_at`.
    Absent {
        /// Position where the entry would sit to preserve sort order.
        insert_at: u16,
    },
}

/// Mutable accessor. Callers are responsible for ensuring unique
/// access to `bytes`; the tree-level layer holds page latches.
#[derive(Debug)]
pub struct LeafPageMut<'a> {
    bytes: &'a mut PageBuf,
    header: PageHeader,
}

impl<'a> LeafPageMut<'a> {
    /// Initialize `bytes` as a fresh empty leaf page for `page_id`.
    pub fn init(bytes: &'a mut PageBuf, page_id: PageId) -> Self {
        let header = PageHeader::new(page_id, PageType::IndexLeaf, TenantId::SYSTEM);
        write_header(bytes, &header);
        Self { bytes, header }
    }

    /// Open an existing leaf page for mutation.
    pub fn open(bytes: &'a mut PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexLeaf.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexLeaf page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Page id stored in the header.
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Read-only view of this page.
    #[must_use]
    pub fn as_ref(&self) -> LeafPageRef<'_> {
        LeafPageRef {
            bytes: self.bytes,
            header: self.header,
        }
    }

    /// Current entry count.
    pub fn entry_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Is the page at leaf capacity? (i.e., no room for a new entry.)
    pub fn is_full(&self) -> bool {
        self.entry_count() >= LEAF_CAPACITY
    }

    /// Overwrite the entry at `index`. Caller guarantees `index <
    /// entry_count`.
    pub fn write_entry(
        &mut self,
        index: u16,
        entry: &LeafEntry,
    ) -> Result<(), SecondaryIndexError> {
        if index >= self.entry_count() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!(
                    "write_entry: index {index} out of range (count={})",
                    self.entry_count()
                ),
            });
        }
        let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE;
        entry.encode_into(&mut self.bytes[off..off + LEAF_ENTRY_SIZE])?;
        Ok(())
    }

    /// Insert `entry` at the sorted position for `entry.key`. Fails if
    /// the key already exists (use [`Self::upsert_entry`] for
    /// "replace" semantics) or if the page is full.
    pub fn insert_entry(&mut self, entry: LeafEntry) -> Result<(), SecondaryIndexError> {
        match self.as_ref().find(entry.key)? {
            LeafFindResult::Found { .. } => Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: "insert_entry called for key that already exists".into(),
            }),
            LeafFindResult::Absent { insert_at } => self.insert_at(insert_at, entry),
        }
    }

    /// Replace an existing entry or insert a new one at sorted
    /// position. Returns the prior entry (live or empty) if a key
    /// collision replaced it.
    pub fn upsert_entry(
        &mut self,
        entry: LeafEntry,
    ) -> Result<Option<LeafEntry>, SecondaryIndexError> {
        match self.as_ref().find(entry.key)? {
            LeafFindResult::Found { index } => {
                let prev = self.as_ref().entry(index)?;
                self.write_entry(index, &entry)?;
                Ok(Some(prev))
            }
            LeafFindResult::Absent { insert_at } => {
                self.insert_at(insert_at, entry)?;
                Ok(None)
            }
        }
    }

    /// Raw bytes of the underlying page. Used by the tree-level
    /// mutation code to emit WAL records without re-acquiring the
    /// page latch — parking_lot's `RwLock` is not re-entrant, so
    /// re-latching a page whose write guard is already held on the
    /// same thread deadlocks (the C-I defect surfaced by the M2-D5
    /// alt review; DEC-21).
    #[must_use]
    pub fn page_bytes(&self) -> &PageBuf {
        &*self.bytes
    }

    /// Split the leaf: move the upper half of entries (indices `[N/2,
    /// N)`) into a freshly allocated page under `new_page_id`, trim
    /// this page to the lower half, and return the promoted key
    /// (= first key of the new right page).
    ///
    /// "Empty" leaf entries (all inline slots zero + no overflow) are
    /// carried over unchanged so their sort-order position is
    /// preserved — the tree-level insert layer filters them out at
    /// lookup time, not here.
    pub fn split_into(
        &mut self,
        new_page_id: PageId,
    ) -> Result<(Box<PageBuf>, SecondaryKey), SecondaryIndexError> {
        let n = self.entry_count();
        if n < 2 {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("leaf split requires >=2 entries, have {n}"),
            });
        }
        let split_at = n / 2;
        let right_count = n - split_at;

        let mut new_buf = fresh_page_buf(new_page_id, PageType::IndexLeaf);
        let src_off = LEAF_ENTRY_OFFSET + (split_at as usize) * LEAF_ENTRY_SIZE;
        let src_end = LEAF_ENTRY_OFFSET + (n as usize) * LEAF_ENTRY_SIZE;
        let dst_off = LEAF_ENTRY_OFFSET;
        let dst_end = dst_off + (right_count as usize) * LEAF_ENTRY_SIZE;
        new_buf[dst_off..dst_end].copy_from_slice(&self.bytes[src_off..src_end]);

        let mut new_header = PageHeader::new(new_page_id, PageType::IndexLeaf, TenantId::SYSTEM);
        new_header.slot_count = right_count;
        write_header(&mut new_buf, &new_header);

        // Trim self to the lower half and zero the freed tail so
        // stale bytes never surface through a codec bug.
        self.header.slot_count = split_at;
        write_header(self.bytes, &self.header);
        for b in &mut self.bytes[src_off..src_end] {
            *b = 0;
        }

        let promoted_key = SecondaryKey::decode(
            &new_buf[LEAF_ENTRY_OFFSET..LEAF_ENTRY_OFFSET + SecondaryKey::SIZE],
        )?;
        Ok((new_buf, promoted_key))
    }

    fn insert_at(&mut self, position: u16, entry: LeafEntry) -> Result<(), SecondaryIndexError> {
        if self.is_full() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: "leaf insert into full page (split required before insert)".into(),
            });
        }
        let n = self.entry_count();
        let end = LEAF_ENTRY_OFFSET + (n as usize) * LEAF_ENTRY_SIZE;
        let pos_off = LEAF_ENTRY_OFFSET + (position as usize) * LEAF_ENTRY_SIZE;
        if (position as usize) < (n as usize) {
            self.bytes
                .copy_within(pos_off..end, pos_off + LEAF_ENTRY_SIZE);
        }
        entry.encode_into(&mut self.bytes[pos_off..pos_off + LEAF_ENTRY_SIZE])?;
        self.header.slot_count = n + 1;
        write_header(self.bytes, &self.header);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Internal page codec (same shape as arcgraph-storage::primary_index,
// DEC-16 intentional duplication; 24-byte key is identical)
// ─────────────────────────────────────────────────────────────────────

/// Offset of `first_child` on an internal page (immediately after the
/// 40-byte page header).
pub const INTERNAL_FIRST_CHILD_OFFSET: usize = PageHeader::SIZE;

/// Offset of the first `(key, child)` entry on an internal page.
pub const INTERNAL_ENTRY_OFFSET: usize = PageHeader::SIZE + 8;

/// Bytes in an internal entry (`SecondaryKey` + child `PageId`).
pub const INTERNAL_ENTRY_SIZE: usize = SecondaryKey::SIZE + 8;
const _: () = assert!(INTERNAL_ENTRY_SIZE == 32);

/// Maximum internal entries per page. Fanout = `INTERNAL_CAPACITY + 1`
/// children.
pub const INTERNAL_CAPACITY: u16 =
    ((PAGE_SIZE - INTERNAL_ENTRY_OFFSET) / INTERNAL_ENTRY_SIZE) as u16;
const _: () = assert!(INTERNAL_CAPACITY == 254);

/// Read-only accessor for internal-page contents.
///
/// An internal page holds a `first_child` pointer followed by `N`
/// `(key, child)` pairs, where `slot_count = N`. `first_child`'s
/// subtree holds keys strictly less than `key[0]`; the `i`-th pair
/// (for `0 ≤ i < N`) owns keys in `[key[i], key[i+1])` (and the last
/// pair owns `[key[N-1], +∞)`).
pub struct InternalPageRef<'a> {
    bytes: &'a PageBuf,
    header: PageHeader,
}

impl fmt::Debug for InternalPageRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InternalPageRef")
            .field("page_id", &self.page_id())
            .field("entry_count", &self.entry_count())
            .finish()
    }
}

impl<'a> InternalPageRef<'a> {
    /// Open an internal view, validating the header.
    pub fn open(bytes: &'a PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexInternal.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexInternal page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Page id from the header.
    #[must_use]
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Number of `(key, child)` pairs on this page.
    #[must_use]
    pub fn entry_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Is the page at internal capacity? (no room for another pair)
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.entry_count() >= INTERNAL_CAPACITY
    }

    /// The `first_child` pointer (subtree for keys `< key[0]`).
    #[must_use]
    pub fn first_child(&self) -> PageId {
        let raw = u64::from_le_bytes(
            self.bytes[INTERNAL_FIRST_CHILD_OFFSET..INTERNAL_FIRST_CHILD_OFFSET + 8]
                .try_into()
                .expect("slice size asserted above"),
        );
        PageId::new(raw)
    }

    /// Decode the `(key, child)` pair at `index`.
    pub fn entry(&self, index: u16) -> Result<(SecondaryKey, PageId), SecondaryIndexError> {
        let n = self.entry_count();
        if index >= n {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("entry index {index} out of range (count={n})"),
            });
        }
        let off = INTERNAL_ENTRY_OFFSET + (index as usize) * INTERNAL_ENTRY_SIZE;
        let key = SecondaryKey::decode(&self.bytes[off..off + SecondaryKey::SIZE])?;
        let child = PageId::new(u64::from_le_bytes(
            self.bytes[off + SecondaryKey::SIZE..off + INTERNAL_ENTRY_SIZE]
                .try_into()
                .expect("slice size asserted above"),
        ));
        Ok((key, child))
    }

    /// Locate the child subtree that may contain `needle`.
    ///
    /// Binary-searches keys for the largest index `i` such that
    /// `key[i] <= needle`. Returns `first_child` when no such index
    /// exists (i.e., `needle < key[0]` or the page has zero entries).
    pub fn find_child(&self, needle: SecondaryKey) -> Result<PageId, SecondaryIndexError> {
        let n = self.entry_count();
        if n == 0 {
            return Ok(self.first_child());
        }
        let mut lo: u32 = 0;
        let mut hi: u32 = u32::from(n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k, _) = self.entry(mid as u16)?;
            if k <= needle {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            Ok(self.first_child())
        } else {
            let (_, child) = self.entry((lo - 1) as u16)?;
            Ok(child)
        }
    }
}

/// Mutable internal-page accessor.
#[derive(Debug)]
pub struct InternalPageMut<'a> {
    bytes: &'a mut PageBuf,
    header: PageHeader,
}

impl<'a> InternalPageMut<'a> {
    /// Initialize `bytes` as a fresh empty internal page with the given
    /// `first_child` and no `(key, child)` entries.
    pub fn init(bytes: &'a mut PageBuf, page_id: PageId, first_child: PageId) -> Self {
        let header = PageHeader::new(page_id, PageType::IndexInternal, TenantId::SYSTEM);
        write_header(bytes, &header);
        let off = INTERNAL_FIRST_CHILD_OFFSET;
        bytes[off..off + 8].copy_from_slice(&first_child.raw().to_le_bytes());
        Self { bytes, header }
    }

    /// Open an existing internal page for in-place mutation.
    pub fn open(bytes: &'a mut PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexInternal.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexInternal page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Page id stored in the header.
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Read-only view of this page.
    #[must_use]
    pub fn as_ref(&self) -> InternalPageRef<'_> {
        InternalPageRef {
            bytes: self.bytes,
            header: self.header,
        }
    }

    /// Number of `(key, child)` pairs.
    pub fn entry_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Is the page at internal capacity?
    pub fn is_full(&self) -> bool {
        self.entry_count() >= INTERNAL_CAPACITY
    }

    /// Raw bytes of the underlying page (see [`LeafPageMut::page_bytes`]
    /// for the motivating re-entrance problem).
    #[must_use]
    pub fn page_bytes(&self) -> &PageBuf {
        &*self.bytes
    }

    /// Insert `(key, right_child)` in sort-order. `right_child` becomes
    /// the subtree rooted immediately after `key`. Returns
    /// `CorruptPage` on duplicate key or if the page is full — callers
    /// must split first.
    pub fn insert(
        &mut self,
        key: SecondaryKey,
        right_child: PageId,
    ) -> Result<(), SecondaryIndexError> {
        if self.is_full() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: "internal insert into full page (split required before insert)".into(),
            });
        }
        // Linear-but-bounded search: internal pages stay small on
        // average, and binary search on a mutating slice adds
        // complexity without speeding up the common case.
        let n = self.entry_count();
        let mut pos: u16 = n;
        for i in 0..n {
            let (k, _) = self.as_ref().entry(i)?;
            match k.cmp(&key) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => {
                    return Err(SecondaryIndexError::CorruptPage {
                        page_id: self.page_id(),
                        reason: "internal insert: duplicate key".into(),
                    });
                }
                core::cmp::Ordering::Greater => {
                    pos = i;
                    break;
                }
            }
        }
        let end = INTERNAL_ENTRY_OFFSET + (n as usize) * INTERNAL_ENTRY_SIZE;
        let pos_off = INTERNAL_ENTRY_OFFSET + (pos as usize) * INTERNAL_ENTRY_SIZE;
        if (pos as usize) < (n as usize) {
            self.bytes
                .copy_within(pos_off..end, pos_off + INTERNAL_ENTRY_SIZE);
        }
        key.encode_into(&mut self.bytes[pos_off..pos_off + SecondaryKey::SIZE])?;
        self.bytes[pos_off + SecondaryKey::SIZE..pos_off + INTERNAL_ENTRY_SIZE]
            .copy_from_slice(&right_child.raw().to_le_bytes());
        self.header.slot_count = n + 1;
        write_header(self.bytes, &self.header);
        Ok(())
    }
}

impl<'a> InternalPageMut<'a> {
    /// Split the internal node, promoting the middle key.
    ///
    /// Layout pre-split: `first_child | (k0,c0) (k1,c1) ... (k_{N-1},c_{N-1})`.
    ///
    /// Let `mid = N / 2`. After split:
    /// - Self keeps pairs `0..mid` (its `first_child` unchanged).
    /// - The promoted key is `key[mid]`.
    /// - The new right page has `first_child = child[mid]` and carries
    ///   pairs `mid+1..N`.
    ///
    /// Callers own the returned [`Box<PageBuf>`] and typically install
    /// it into the page store before inserting the promoted key into
    /// their own parent (or becoming a new root).
    pub fn split_into(
        &mut self,
        new_page_id: PageId,
    ) -> Result<(Box<PageBuf>, SecondaryKey), SecondaryIndexError> {
        let n = self.entry_count();
        if n < 2 {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("internal split requires >=2 pairs, have {n}"),
            });
        }
        let mid = n / 2;
        let right_pairs = n - mid - 1;

        // Pull out the promoted key and the right side's first_child
        // (= child of the promoted pair).
        let mid_off = INTERNAL_ENTRY_OFFSET + (mid as usize) * INTERNAL_ENTRY_SIZE;
        let promoted_key =
            SecondaryKey::decode(&self.bytes[mid_off..mid_off + SecondaryKey::SIZE])?;
        let right_first_child = PageId::new(u64::from_le_bytes(
            self.bytes[mid_off + SecondaryKey::SIZE..mid_off + INTERNAL_ENTRY_SIZE]
                .try_into()
                .expect("slice size asserted above"),
        ));

        // Build the new right page.
        let mut new_buf = fresh_page_buf(new_page_id, PageType::IndexInternal);
        new_buf[INTERNAL_FIRST_CHILD_OFFSET..INTERNAL_FIRST_CHILD_OFFSET + 8]
            .copy_from_slice(&right_first_child.raw().to_le_bytes());
        if right_pairs > 0 {
            let src_off = INTERNAL_ENTRY_OFFSET + ((mid + 1) as usize) * INTERNAL_ENTRY_SIZE;
            let src_end = INTERNAL_ENTRY_OFFSET + (n as usize) * INTERNAL_ENTRY_SIZE;
            let dst_off = INTERNAL_ENTRY_OFFSET;
            let dst_end = dst_off + (right_pairs as usize) * INTERNAL_ENTRY_SIZE;
            new_buf[dst_off..dst_end].copy_from_slice(&self.bytes[src_off..src_end]);
        }
        let mut new_header =
            PageHeader::new(new_page_id, PageType::IndexInternal, TenantId::SYSTEM);
        new_header.slot_count = right_pairs;
        write_header(&mut new_buf, &new_header);

        // Trim self to `mid` pairs and zero the freed tail.
        self.header.slot_count = mid;
        write_header(self.bytes, &self.header);
        let zero_off = INTERNAL_ENTRY_OFFSET + (mid as usize) * INTERNAL_ENTRY_SIZE;
        let zero_end = INTERNAL_ENTRY_OFFSET + (n as usize) * INTERNAL_ENTRY_SIZE;
        for b in &mut self.bytes[zero_off..zero_end] {
            *b = 0;
        }

        Ok((new_buf, promoted_key))
    }
}

/// Result of a child split propagated up to a parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitInfo {
    /// Key promoted to the parent — also the first key of the new
    /// right page (for leaves) or the median key (for internals).
    pub promoted_key: SecondaryKey,
    /// Page id of the newly allocated right page.
    pub new_right_page: PageId,
}

// ─────────────────────────────────────────────────────────────────────
// Overflow-page codec (DEC-15, DEC-20)
// ─────────────────────────────────────────────────────────────────────

/// Offset within an overflow page of the `filled_count` field.
pub const OVERFLOW_FILLED_COUNT_OFFSET: usize = PageHeader::SIZE;

/// Offset within an overflow page of the `next_page` pointer.
pub const OVERFLOW_NEXT_OFFSET: usize = PageHeader::SIZE + 8;

/// Offset within an overflow page of the first NodeId slot.
pub const OVERFLOW_SLOTS_OFFSET: usize = PageHeader::SIZE + 16;

/// NodeId slots per overflow page. Each slot is 8 bytes (`NodeId`'s
/// little-endian u64 encoding). `(8192 - 56) / 8 = 1017`.
pub const OVERFLOW_SLOTS_PER_PAGE: usize = (PAGE_SIZE - OVERFLOW_SLOTS_OFFSET) / 8;
const _: () = assert!(OVERFLOW_SLOTS_PER_PAGE == 1017);

/// Read-only accessor for an overflow-chain page.
pub struct OverflowPageRef<'a> {
    bytes: &'a PageBuf,
    header: PageHeader,
}

impl<'a> OverflowPageRef<'a> {
    /// Open an overflow view, validating the header.
    pub fn open(bytes: &'a PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexOverflow.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexOverflow page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Page id from the header.
    #[must_use]
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Number of slots `[0..filled_count)` that have been assigned.
    /// Slots inside that prefix can still be zero (tombstoned); slots
    /// outside it are unwritten and must read as zero.
    #[must_use]
    pub fn filled_count(&self) -> u16 {
        u16::from_le_bytes(
            self.bytes[OVERFLOW_FILLED_COUNT_OFFSET..OVERFLOW_FILLED_COUNT_OFFSET + 2]
                .try_into()
                .expect("slice size asserted above"),
        )
    }

    /// Page id of the next overflow page in the chain, or
    /// [`PageId::ZERO`] when this is the tail.
    #[must_use]
    pub fn next(&self) -> PageId {
        PageId::new(u64::from_le_bytes(
            self.bytes[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 8]
                .try_into()
                .expect("slice size asserted above"),
        ))
    }

    /// NodeId stored at `index`. Out-of-range indices return
    /// `CorruptPage`.
    pub fn slot(&self, index: u16) -> Result<NodeId, SecondaryIndexError> {
        if usize::from(index) >= OVERFLOW_SLOTS_PER_PAGE {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!(
                    "overflow slot index {index} out of range (max {})",
                    OVERFLOW_SLOTS_PER_PAGE
                ),
            });
        }
        let off = OVERFLOW_SLOTS_OFFSET + usize::from(index) * 8;
        Ok(NodeId::new(u64::from_le_bytes(
            self.bytes[off..off + 8]
                .try_into()
                .expect("slice size asserted above"),
        )))
    }

    /// Iterator over live (non-zero) NodeIds in `[0..filled_count)`.
    pub fn live_slots(&self) -> impl Iterator<Item = NodeId> + '_ {
        let filled = self.filled_count();
        (0..filled).filter_map(move |i| self.slot(i).ok().filter(|n| n.raw() != 0))
    }

    /// Is this page at capacity?
    #[must_use]
    pub fn is_full(&self) -> bool {
        usize::from(self.filled_count()) >= OVERFLOW_SLOTS_PER_PAGE
    }
}

/// Mutable accessor for an overflow-chain page.
pub struct OverflowPageMut<'a> {
    bytes: &'a mut PageBuf,
    header: PageHeader,
}

impl<'a> OverflowPageMut<'a> {
    /// Initialize `bytes` as a fresh empty overflow page for
    /// `page_id`. `filled_count = 0`, `next = 0`, all slots zero.
    pub fn init(bytes: &'a mut PageBuf, page_id: PageId) -> Self {
        let header = PageHeader::new(page_id, PageType::IndexOverflow, TenantId::SYSTEM);
        write_header(bytes, &header);
        // filled_count = 0
        bytes[OVERFLOW_FILLED_COUNT_OFFSET..OVERFLOW_FILLED_COUNT_OFFSET + 2]
            .copy_from_slice(&0u16.to_le_bytes());
        // next = 0
        bytes[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 8].copy_from_slice(&0u64.to_le_bytes());
        // slots are already zero via `fresh_page_buf`'s calloc
        Self { bytes, header }
    }

    /// Open an existing overflow page for mutation.
    pub fn open(bytes: &'a mut PageBuf) -> Result<Self, SecondaryIndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexOverflow.as_byte() {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexOverflow page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Read-only view of this page.
    #[must_use]
    pub fn as_ref(&self) -> OverflowPageRef<'_> {
        OverflowPageRef {
            bytes: self.bytes,
            header: self.header,
        }
    }

    /// Current `filled_count`.
    pub fn filled_count(&self) -> u16 {
        self.as_ref().filled_count()
    }

    /// Is the page at capacity?
    pub fn is_full(&self) -> bool {
        self.as_ref().is_full()
    }

    /// Next-page pointer.
    pub fn next(&self) -> PageId {
        self.as_ref().next()
    }

    /// Set the `next_page` pointer.
    pub fn set_next(&mut self, next: PageId) {
        self.bytes[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 8]
            .copy_from_slice(&next.raw().to_le_bytes());
    }

    fn set_filled(&mut self, filled: u16) {
        self.bytes[OVERFLOW_FILLED_COUNT_OFFSET..OVERFLOW_FILLED_COUNT_OFFSET + 2]
            .copy_from_slice(&filled.to_le_bytes());
    }

    fn write_slot(&mut self, index: u16, node: NodeId) {
        let off = OVERFLOW_SLOTS_OFFSET + usize::from(index) * 8;
        self.bytes[off..off + 8].copy_from_slice(&node.raw().to_le_bytes());
    }

    /// Append `node` at the current write head. Returns
    /// `CorruptPage` if the page is already full; caller should
    /// allocate the next chunk.
    pub fn append(&mut self, node: NodeId) -> Result<u16, SecondaryIndexError> {
        let filled = self.filled_count();
        if usize::from(filled) >= OVERFLOW_SLOTS_PER_PAGE {
            return Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::new(self.header.page_id),
                reason: "overflow page: append into full page".into(),
            });
        }
        self.write_slot(filled, node);
        self.set_filled(filled + 1);
        Ok(filled)
    }

    /// Find the first slot in `[0..filled_count)` holding `node` and
    /// zero it. Returns `true` when a slot was tombstoned. Slots are
    /// never shifted — tombstones remain as zeros in the written
    /// prefix (mark-only per DEC-12 / Philosophy §H).
    pub fn tombstone_first(&mut self, node: NodeId) -> Result<bool, SecondaryIndexError> {
        if node == NodeId::ZERO {
            return Ok(false);
        }
        let filled = self.filled_count();
        for i in 0..filled {
            if self.as_ref().slot(i)? == node {
                self.write_slot(i, NodeId::ZERO);
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory page store (DEC-17)
// ─────────────────────────────────────────────────────────────────────

/// Per-page latch (same pattern as `PrimaryPageStore`).
pub type PageLatch = Arc<RwLock<Box<PageBuf>>>;

/// In-memory page store for the secondary index. Alpha-only; the
/// permanent home is BufferPool once WAL replay lands in M2.e (DEC-17).
#[derive(Default)]
pub struct SecondaryPageStore {
    pages: DashMap<PageId, PageLatch>,
}

impl fmt::Debug for SecondaryPageStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecondaryPageStore")
            .field("pages", &self.pages.len())
            .finish()
    }
}

impl SecondaryPageStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pages: DashMap::new(),
        }
    }

    /// Number of pages currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Is the store empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Clone the latch for `page_id`.
    pub fn latch(&self, page_id: PageId) -> Result<PageLatch, SecondaryIndexError> {
        self.pages
            .get(&page_id)
            .map(|r| Arc::clone(r.value()))
            .ok_or(SecondaryIndexError::MissingPage(page_id))
    }

    /// Install a zero-initialized page of `page_type` under `page_id`.
    pub fn install_fresh(
        &self,
        page_id: PageId,
        page_type: PageType,
    ) -> Result<(), SecondaryIndexError> {
        self.install(page_id, fresh_page_buf(page_id, page_type))
    }

    /// Install a fresh page under `page_id`. Rejects duplicate ids.
    pub fn install(&self, page_id: PageId, page: Box<PageBuf>) -> Result<(), SecondaryIndexError> {
        use dashmap::mapref::entry::Entry;
        match self.pages.entry(page_id) {
            Entry::Occupied(_) => Err(SecondaryIndexError::CorruptPage {
                page_id,
                reason: "page id already mapped in secondary index".into(),
            }),
            Entry::Vacant(v) => {
                v.insert(Arc::new(RwLock::new(page)));
                Ok(())
            }
        }
    }

    /// ADR-032 Slice 3 + PR #79 X-1 review fold-in: unconditional
    /// byte-copy install. Sibling of
    /// `PrimaryPageStore::install_or_replace` — see that method's
    /// rustdoc for the Lemma-I2-is-bundle-level rationale.
    pub fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), SecondaryIndexError> {
        use dashmap::mapref::entry::Entry;
        match self.pages.entry(page_id) {
            Entry::Occupied(e) => {
                let latch = Arc::clone(e.get());
                let mut guard = latch.write();
                let bytes: &mut PageBuf = guard.as_mut();
                bytes.copy_from_slice(page.as_ref());
                Ok(())
            }
            Entry::Vacant(v) => {
                v.insert(Arc::new(RwLock::new(page)));
                Ok(())
            }
        }
    }

    /// ADR-033 Z-1 (b): install a fresh page and record it in the
    /// transaction's mutation log so rollback can remove it. Parallel
    /// to [`arcgraph_storage::primary_index::PrimaryPageStore::install_for_txn`].
    pub fn install_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), SecondaryIndexError> {
        self.install(page_id, page)?;
        log.new_pages.push((PageStoreKind::Secondary, page_id));
        Ok(())
    }

    /// ADR-033 Z-1 (b): capture pre-mutation bytes from an
    /// already-held `ArcRwLock` write guard into `log`. Parallel to
    /// `PrimaryPageStore::capture_from_guard`.
    pub fn capture_from_guard(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        guard: &ArcPageWriteGuard,
    ) {
        self.capture_bytes(log, page_id, guard.as_ref());
    }

    /// ADR-033 Z-1 (b): capture pre-mutation bytes from a plain
    /// (non-`Arc`) write guard — used on the overflow-chain remove
    /// walk, which latches with `RwLock::write()` rather than
    /// `write_arc()`.
    pub fn capture_from_write_guard(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        guard: &RwLockWriteGuard<Box<PageBuf>>,
    ) {
        self.capture_bytes(log, page_id, guard.as_ref());
    }

    /// Shared Z-1 F-1 capture core: snapshot `bytes` (the current, i.e.
    /// pre-mutation, contents of `page_id`) into `log.page_mutations`
    /// under `PageStoreKind::Secondary`, deduped by the compound
    /// `(kind, page_id)` key (Y-2) so the FIRST capture per page wins
    /// and rollback restores the genuine pre-W state.
    fn capture_bytes(&self, log: &mut TxnMutationLog, page_id: PageId, bytes: &PageBuf) {
        // Y-2: (PageStoreKind::Secondary, page_id) compound dedup so
        // secondary captures do not collide with primary / record
        // captures on shared numeric PageIds.
        if log.has_captured(PageStoreKind::Secondary, page_id) {
            return;
        }
        let mut snapshot: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
        snapshot.copy_from_slice(bytes);
        log.page_mutations
            .push((PageStoreKind::Secondary, page_id, snapshot));
    }

    /// ADR-033 Z-1 (b) rollback primitive: remove a page from the
    /// secondary store's DashMap. Parallel to
    /// `PrimaryPageStore::remove_page`.
    pub fn remove_page(&self, page_id: PageId) -> Option<PageLatch> {
        self.pages.remove(&page_id).map(|(_, latch)| latch)
    }

    /// ADR-033 Z-1 (b) rollback primitive: restore a page's bytes
    /// from a pre-captured snapshot. Parallel to
    /// `PrimaryPageStore::restore_page_bytes`.
    pub fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), SecondaryIndexError> {
        let latch = self.latch(page_id)?;
        let mut guard = latch.write();
        let bytes: &mut PageBuf = guard.as_mut();
        bytes.copy_from_slice(pre_bytes);
        Ok(())
    }

    /// Test/observability: does the store currently map `page_id`?
    #[doc(hidden)]
    #[must_use]
    pub fn contains(&self, page_id: PageId) -> bool {
        self.pages.contains_key(&page_id)
    }
}

/// ADR-032 Slice 3 replay-handle impl. Idempotent byte-copy install
/// (Lemma I2) via a latch probe: if the page is already mapped,
/// verify bytes match and return `Ok(())`; otherwise call
/// `SecondaryPageStore::install`. Replay is single-threaded so the
/// latch/install pair has no TOCTOU window.
impl SecondaryPageStoreHandle for SecondaryPageStore {
    fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<[u8; PAGE_SIZE]>,
    ) -> arcgraph_core::Result<()> {
        // PR #79 X-1 review fold-in: unconditional overwrite.
        // Lemma I2 is bundle-level, not entry-level — see the
        // primary handle's rustdoc for the full rationale.
        self.install_or_replace(page_id, page).map_err(|e| {
            arcgraph_core::ArcGraphError::WalCorruption {
                lsn: arcgraph_core::Lsn::ZERO,
                reason: format!(
                    "secondary page_store.install_or_replace({:?}) on replay: {}",
                    page_id, e
                ),
            }
        })
    }

    fn contains(&self, page_id: PageId) -> bool {
        self.pages.contains_key(&page_id)
    }
}

// ─────────────────────────────────────────────────────────────────────
// SecondaryIndex — public tree surface (DEC-10 sibling, DEC-13, DEC-17)
// ─────────────────────────────────────────────────────────────────────

/// MVCC key under `TenantId::SYSTEM` that stores the secondary-index
/// root page id. Sibling of the primary-index root key (1) and the
/// catalog tenants key (0); encoded as 8 little-endian bytes.
pub const SECONDARY_INDEX_ROOT_KEY: u64 = 2;

/// Write operation kind used by the internal `write` dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteOp {
    Insert,
    Remove,
}

// Owning guards (same pattern as primary) — `'static` lifetime because
// they retain an internal `Arc` to the underlying lock. Lets us store
// guards in a `Vec` for iterative crabbing without lifetime gymnastics.
type ArcPageWriteGuard = parking_lot::ArcRwLockWriteGuard<RawRwLock, Box<PageBuf>>;
type ArcPageReadGuard = parking_lot::ArcRwLockReadGuard<RawRwLock, Box<PageBuf>>;

/// Secondary B+tree index with `(TenantId, LabelId, StringId,
/// PropertyValue) → [NodeId]` semantics (M2-34).
///
/// See module-level docs for layout; see ADR-023 for the read-
/// accelerator visibility contract — readers MUST verify candidate
/// NodeIds against `crud::read_node(tx, id)` before returning them.
pub struct SecondaryIndex {
    page_store: Arc<SecondaryPageStore>,
    txn_mgr: Arc<TxnManager>,
    wal: Option<WalHandle>,
    allocator: Arc<PageAllocator>,
    // `0` means "not loaded" — every other value is a valid root page id.
    root_cache: AtomicU64,
    // Serializes all mutations. Readers bypass this and use per-page
    // read locks. Same coarse-gate + per-page-crabbing shape as
    // `PrimaryIndex` (DEC-17 amendment).
    write_gate: Mutex<()>,
    /// Cached tail-page id per overflow-chain head (DEC-22).
    /// `overflow_head → current_tail`. Updated when a successor page
    /// is allocated in `append_to_overflow_chain`. Eliminates the
    /// O(chain_length) walk-to-tail the alt review flagged as a
    /// quadratic hazard on duplicate-heavy workloads (C-II).
    ///
    /// Safe to be in-memory only: the overflow chain IS durable
    /// (every page has a WAL record), and on M2.e restart the cache
    /// can be rebuilt by walking each chain once from head → tail as
    /// pages are encountered during WAL replay.
    overflow_tail_cache: DashMap<PageId, PageId>,
    /// Latest `grow_root` new-root id that still needs a SYSTEM-tenant
    /// MVCC persist. `0` = nothing pending. Drained by
    /// [`Self::persist_pending_root_update`] AFTER any enclosing
    /// `Transaction::commit_with_bundle` builder returns so the
    /// persist's own inner commit doesn't nest inside an outer's
    /// Phase 2 and deadlock on `install_order`.
    ///
    /// #1200 doc-rot fix: this was formerly described as a "mirror of
    /// `PrimaryIndex::pending_root`", but the primary's `pending_root`
    /// slot was RETIRED in ADR-032 Slice-2 (the primary now pushes
    /// SideChannelWrites directly into the outer CommitBundle). The
    /// secondary kept this slot; the slot inherently coalesces
    /// last-wins (an `AtomicU64::store` overwrites), so the secondary
    /// never had the #1200 multi-grow_root defect. The primary's lost
    /// last-wins coalescing is now restored on the bundle-folded path by
    /// `TxnManager::coalesce_sidechannel_writes` (#1200), not by a slot.
    pending_root: AtomicU64,
    /// **RC-2 (#1366)** — declared-index lifecycle state, encoded as
    /// `0 = Building`, `1 = Online`. Defaults to `Online` (the Phase-0
    /// always-on secondary has no backfill, so it is complete by
    /// construction). Phase-1 `CREATE INDEX` starts a fresh index in
    /// `Building` and flips to `Online` in the same `CommitBundle` as
    /// the final backfill watermark. The write-follows-declare RULE
    /// ([`SecondaryIndexHandle::maintenance_active`]) applies
    /// maintenance in BOTH states — see [`IndexState`].
    index_state: AtomicU8,
}

/// Encoded `IndexState::Building` for [`SecondaryIndex::index_state`].
const INDEX_STATE_BUILDING: u8 = 0;
/// Encoded `IndexState::Online` for [`SecondaryIndex::index_state`].
const INDEX_STATE_ONLINE: u8 = 1;

impl fmt::Debug for SecondaryIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecondaryIndex")
            .field("root_cache", &self.root_cache.load(Ordering::Acquire))
            .field("page_store", &self.page_store)
            .finish()
    }
}

impl SecondaryIndex {
    /// Construct a secondary index.
    ///
    /// On first use in a database, allocates a fresh leaf root via
    /// `allocator`, installs it, and persists the pointer under
    /// `(TenantId::SYSTEM, SECONDARY_INDEX_ROOT_KEY = 2)`. Subsequent
    /// constructions against the same `TxnManager` recover the pointer
    /// from MVCC. Page contents are not repopulated until WAL replay
    /// lands with M2.e.
    pub fn new(
        txn_mgr: Arc<TxnManager>,
        allocator: Arc<PageAllocator>,
        wal: Option<WalHandle>,
    ) -> Result<Self, SecondaryIndexError> {
        let page_store = Arc::new(SecondaryPageStore::new());
        let this = Self {
            page_store,
            txn_mgr: Arc::clone(&txn_mgr),
            wal,
            allocator: Arc::clone(&allocator),
            root_cache: AtomicU64::new(0),
            write_gate: Mutex::new(()),
            overflow_tail_cache: DashMap::new(),
            pending_root: AtomicU64::new(0),
            // RC-2: the always-on Phase-0 secondary is Online by
            // construction (no backfill to run). A Phase-1 CREATE INDEX
            // will start fresh indexes in Building.
            index_state: AtomicU8::new(INDEX_STATE_ONLINE),
        };
        let root_id = match this.read_root_from_mvcc() {
            Some(existing) => existing,
            None => {
                let fresh = this.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
                // Stage-WAL-publish (DEC-21): WAL the fresh leaf
                // BEFORE install, since install is what lets another
                // thread observe it via `page_store.latch`.
                let fresh_buf = fresh_page_buf(fresh, PageType::IndexLeaf);
                this.emit_wal_for_bytes(fresh, fresh_buf.as_ref())?;
                this.page_store.install(fresh, fresh_buf)?;
                this.persist_root_to_mvcc(fresh)?;
                fresh
            }
        };
        this.root_cache.store(root_id.raw(), Ordering::Release);
        Ok(this)
    }

    /// Return the cached root page id, or load it from MVCC once.
    pub fn root(&self) -> Result<PageId, SecondaryIndexError> {
        let cached = self.root_cache.load(Ordering::Acquire);
        if cached != 0 {
            return Ok(PageId::new(cached));
        }
        match self.read_root_from_mvcc() {
            Some(id) => {
                self.root_cache.store(id.raw(), Ordering::Release);
                Ok(id)
            }
            None => Err(SecondaryIndexError::CorruptPage {
                page_id: PageId::ZERO,
                reason: "secondary-index root pointer missing from MVCC state".to_owned(),
            }),
        }
    }

    fn read_root_from_mvcc(&self) -> Option<PageId> {
        let txn = self.txn_mgr.begin(TenantId::SYSTEM);
        let bytes = txn.read(SECONDARY_INDEX_ROOT_KEY)?;
        txn.abort();
        if bytes.len() != 8 {
            return None;
        }
        let arr: [u8; 8] = bytes[..8].try_into().ok()?;
        let raw = u64::from_le_bytes(arr);
        if raw == 0 {
            None
        } else {
            Some(PageId::new(raw))
        }
    }

    fn persist_root_to_mvcc(&self, root_id: PageId) -> Result<(), SecondaryIndexError> {
        let mut txn = self.txn_mgr.begin(TenantId::SYSTEM);
        txn.write(
            SECONDARY_INDEX_ROOT_KEY,
            Bytes::copy_from_slice(&root_id.raw().to_le_bytes()),
        );
        txn.commit().map_err(SecondaryIndexError::Core)?;
        Ok(())
    }

    /// Shared page store — used by tests.
    #[doc(hidden)]
    #[must_use]
    pub fn page_store(&self) -> &Arc<SecondaryPageStore> {
        &self.page_store
    }

    // ────────── read path ──────────

    /// Return the live `NodeId`s currently indexed at `key` (ADR-023:
    /// callers MUST verify each id's snapshot visibility via
    /// `crud::read_node(tx, id)` before yielding it). Returns an empty
    /// vector when the key is absent or every slot has been tombstoned.
    ///
    /// Walks the inline array first, then the overflow chain starting
    /// at `overflow_head`.
    pub fn lookup(&self, key: SecondaryKey) -> Result<Vec<NodeId>, SecondaryIndexError> {
        let leaf_entry = self.descend_read(key)?;
        let Some(entry) = leaf_entry else {
            return Ok(Vec::new());
        };
        let mut out: Vec<NodeId> = entry.live_inline().collect();
        let mut next = entry.overflow_head;
        while next != PageId::ZERO {
            let latch = self.page_store.latch(next)?;
            let g = latch.read();
            let page = OverflowPageRef::open(g.as_ref())?;
            out.extend(page.live_slots());
            next = page.next();
        }
        Ok(out)
    }

    fn descend_read(&self, key: SecondaryKey) -> Result<Option<LeafEntry>, SecondaryIndexError> {
        let root_id = self.root()?;
        let root_latch = self.page_store.latch(root_id)?;
        let mut guard: ArcPageReadGuard = root_latch.read_arc();

        loop {
            let header = read_header(guard.as_ref())?;
            if header.page_type == PageType::IndexLeaf.as_byte() {
                let leaf = LeafPageRef::open(guard.as_ref())?;
                return leaf.lookup_entry(key);
            }
            if header.page_type != PageType::IndexInternal.as_byte() {
                return Err(SecondaryIndexError::CorruptPage {
                    page_id: PageId::new(header.page_id),
                    reason: format!("unexpected page_type {} in read descent", header.page_type),
                });
            }
            let child_id = {
                let internal = InternalPageRef::open(guard.as_ref())?;
                internal.find_child(key)?
            };
            let child_latch = self.page_store.latch(child_id)?;
            guard = child_latch.read_arc();
        }
    }

    // ────────── write path ──────────

    /// Insert `(key, node)` into the index. If `key` already has an
    /// entry, the NodeId is appended to the first empty inline slot.
    ///
    /// In commit 4 the inline array is the only storage; when all four
    /// inline slots are filled AND no overflow chain is attached,
    /// inserting a 5th NodeId returns [`SecondaryIndexError::InlineFull`].
    /// Commit 5 replaces that error with overflow-page allocation.
    ///
    /// Re-inserting a `NodeId` that's already present in the entry is
    /// a no-op (first-empty-slot scan returns the next zero, which is
    /// the first position after the duplicates — new duplicates just
    /// stack rather than dedup). Callers that need dedup semantics do
    /// a `lookup` first.
    pub fn insert(&self, key: SecondaryKey, node: NodeId) -> Result<(), SecondaryIndexError> {
        self.write(key, Some(node), WriteOp::Insert).map(|_| ())
    }

    /// Remove `node` from `key`'s entry. Returns `true` if the NodeId
    /// was found (and its inline slot zeroed), `false` if the key is
    /// absent or the node was not among its live slots.
    ///
    /// Commit 5 extends this to search the overflow chain as well.
    pub fn remove(&self, key: SecondaryKey, node: NodeId) -> Result<bool, SecondaryIndexError> {
        self.write(key, Some(node), WriteOp::Remove)
            .map(|r| r.removed)
    }

    /// Bundle-aware insert. Performs the mutation + in-memory install
    /// under `write_gate` + per-page latch, then returns the staged
    /// `IndexPage` byte snapshots to the caller (rather than draining
    /// them into the WAL internally). The caller folds them into a
    /// single `CommitBundle` record, typically via
    /// `Transaction::commit_with_bundle`.
    pub fn insert_deferred(
        &self,
        key: SecondaryKey,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, SecondaryIndexError> {
        let (_outcome, staged) = self.write_deferred(key, Some(node), WriteOp::Insert, log)?;
        Ok(staged)
    }

    /// Bundle-aware remove (ADR-031). See [`Self::insert_deferred`]
    /// for the durability contract. Returns `(removed_flag, staged)`
    /// — `true` if the NodeId was found, plus any staged `IndexPage`
    /// snapshots from the tombstone rewrites.
    pub fn remove_deferred(
        &self,
        key: SecondaryKey,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<(bool, Vec<StagedEmit>), SecondaryIndexError> {
        let (outcome, staged) = self.write_deferred(key, Some(node), WriteOp::Remove, log)?;
        Ok((outcome.removed, staged))
    }

    /// Internal write path: standalone emit variant. Mutates +
    /// installs under `write_gate` (via `write_deferred`), then emits
    /// ALL staged pages + any grow_root SYSTEM root-pointer update as
    /// ONE crash-atomic `CommitBundle`
    /// (`TxnManager::commit_index_pages_atomic`).
    ///
    /// Kept for non-bundle callers (`insert` / `remove` public
    /// wrappers + any future bootstrap use). #37 [A-1] closed the
    /// pre-existing split-record / overflow-successor crash hazard on
    /// this path: it no longer drains one `IndexPage` record per page,
    /// so a crash mid-op can no longer leave an orphan page on replay
    /// (ADR-031 realized for the standalone path). The bundle-aware
    /// hot path (`crud::commit`) still uses
    /// [`Self::persist_pending_root_update`]; only the standalone path
    /// is folded here.
    fn write(
        &self,
        key: SecondaryKey,
        node: Option<NodeId>,
        op: WriteOp,
    ) -> Result<WriteOutcome, SecondaryIndexError> {
        // Standalone path: this method commits its own crash-atomic
        // `CommitBundle` immediately below (there is no enclosing txn
        // whose rollback closure would drain a shared log), so the
        // mutation-log captures land in a throwaway log that is dropped
        // when this method returns. Z-1 (b) rollback for standalone
        // writes is provided by the atomic bundle itself: either the
        // whole bundle fsyncs or none of it does.
        let mut throwaway = TxnMutationLog::new();
        let (outcome, staged) = self.write_deferred(key, node, op, &mut throwaway)?;
        // #37 [A-1]: fold ALL staged index pages + the grow_root SYSTEM
        // root-pointer update (drained from `pending_root`) into ONE
        // crash-atomic `CommitBundle` (was: N `IndexPage` records via
        // `drain_staged_emits` + a separate SYSTEM MVCC commit via
        // `persist_pending_root_update`).
        let sc_writes = self.take_pending_root_sidechannel();
        self.txn_mgr
            .commit_index_pages_atomic(self.wal.as_ref(), &staged, &sc_writes)
            .map_err(SecondaryIndexError::Core)?;
        Ok(outcome)
    }

    /// #37 [A-1]: drain the grow_root root-pointer update stashed in
    /// `pending_root` (set during `write_deferred`'s grow_root) into a
    /// [`SideChannelWrite`] so the standalone `write` path can fold it
    /// into the SAME crash-atomic `CommitBundle` as the staged pages.
    /// Returns an empty vec when no grow_root happened. The encoding
    /// (SYSTEM tenant, `SECONDARY_INDEX_ROOT_KEY`, 8-byte LE PageId)
    /// matches [`Self::persist_root_to_mvcc`] so replay restores the
    /// root pointer identically to the bundle-aware path.
    fn take_pending_root_sidechannel(&self) -> Vec<SideChannelWrite> {
        let pending = self.pending_root.swap(0, Ordering::AcqRel);
        if pending == 0 {
            Vec::new()
        } else {
            vec![SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: SECONDARY_INDEX_ROOT_KEY,
                value: Some(Bytes::copy_from_slice(&pending.to_le_bytes())),
            }]
        }
    }

    /// Drain the stashed `grow_root` root-pointer update (if any)
    /// into a SYSTEM-tenant MVCC commit. Mirror of
    /// `PrimaryIndex::persist_pending_root_update`. MUST be called
    /// OUTSIDE any enclosing `Transaction::commit_with_bundle`
    /// builder — the inner MVCC commit this runs would otherwise
    /// deadlock on `install_order`.
    pub fn persist_pending_root_update(&self) -> Result<(), SecondaryIndexError> {
        let pending = self.pending_root.swap(0, Ordering::AcqRel);
        if pending != 0 {
            self.persist_root_to_mvcc(PageId::new(pending))?;
        }
        Ok(())
    }

    /// Internal write path: bundle-aware variant returning the
    /// staged `IndexPage` snapshots for the caller to fold into a
    /// `CommitBundle`. Mutation + install still happen under
    /// `write_gate` + per-page latches per ADR-030; the drop from
    /// this function leaves the caller holding NO index locks.
    fn write_deferred(
        &self,
        key: SecondaryKey,
        node: Option<NodeId>,
        op: WriteOp,
        log: &mut TxnMutationLog,
    ) -> Result<(WriteOutcome, Vec<StagedEmit>), SecondaryIndexError> {
        // Per ADR-030 + ADR-031: mutate+install under `write_gate` +
        // per-page latch; WAL emission is folded into ONE
        // `CommitBundle` — by the standalone `write` caller via
        // `commit_index_pages_atomic` (#37 [A-1]), or by the
        // bundle-aware `Transaction::commit_with_bundle` caller.
        //
        // Z-1 F-1 (#1366): `log` records every fresh page install and
        // captures the pre-mutation bytes of every in-place edit under
        // `PageStoreKind::Secondary`. On WAL fsync failure the Z-1 (b)
        // rollback closure (`crud.rs`) drains these so aborted-insert
        // secondary pages do not leak — closing the deferred F-1 gap.
        // Capture is under `write_gate` (which serializes all writers)
        // + the per-page write latch, so the snapshotted bytes are the
        // genuine pre-W state.
        let mut staged: Vec<StagedEmit> = Vec::with_capacity(4);
        let outcome = {
            let _gate = self.write_gate.lock();
            let root_id = self.root()?;

            // Descend with write-crabbing: push parents onto `path` and
            // drop all ancestors once a safe child (one with room for a
            // promoted entry) is reached.
            let mut path: Vec<(PageId, ArcPageWriteGuard)> = Vec::with_capacity(4);
            let root_latch = self.page_store.latch(root_id)?;
            let root_guard = root_latch.write_arc();
            // Z-1 F-1: capture root pre-W bytes before any mutation.
            self.page_store
                .capture_from_guard(log, root_id, &root_guard);
            path.push((root_id, root_guard));

            loop {
                let (_, top_guard) = path.last().expect("path is non-empty inside descent loop");
                let page_type_byte = read_header(top_guard.as_ref())?.page_type;
                if page_type_byte == PageType::IndexLeaf.as_byte() {
                    break;
                }
                if page_type_byte != PageType::IndexInternal.as_byte() {
                    return Err(SecondaryIndexError::CorruptPage {
                        page_id: path.last().expect("non-empty path").0,
                        reason: format!("unexpected page_type {page_type_byte} in write descent"),
                    });
                }
                let child_id = {
                    let internal = InternalPageRef::open(top_guard.as_ref())?;
                    internal.find_child(key)?
                };
                let child_latch = self.page_store.latch(child_id)?;
                let child_guard = child_latch.write_arc();
                // Z-1 F-1: capture child pre-W bytes before mutation.
                // When the crab clears ancestors below, an already-
                // captured page stays captured (has_captured dedup) —
                // rollback still restores the earliest pre-W state.
                self.page_store
                    .capture_from_guard(log, child_id, &child_guard);
                let child_is_safe = {
                    let hdr = read_header(child_guard.as_ref())?;
                    if hdr.page_type == PageType::IndexLeaf.as_byte() {
                        hdr.slot_count < LEAF_CAPACITY
                    } else if hdr.page_type == PageType::IndexInternal.as_byte() {
                        hdr.slot_count < INTERNAL_CAPACITY
                    } else {
                        return Err(SecondaryIndexError::CorruptPage {
                            page_id: child_id,
                            reason: format!("unexpected child page_type {}", hdr.page_type),
                        });
                    }
                };
                if child_is_safe {
                    path.clear();
                }
                path.push((child_id, child_guard));
            }

            let (leaf_id, mut leaf_guard) = path.pop().expect("leaf is at top of path");
            let (outcome, leaf_split) =
                self.apply_leaf_op(leaf_id, &mut leaf_guard, key, node, op, &mut staged, log)?;
            drop(leaf_guard);

            let mut pending = leaf_split;
            while let Some((parent_id, mut parent_guard)) = path.pop() {
                match pending {
                    None => {
                        drop(parent_guard);
                        break;
                    }
                    Some(split) => {
                        pending = self.apply_internal_insert(
                            parent_id,
                            &mut parent_guard,
                            split,
                            &mut staged,
                            log,
                        )?;
                        drop(parent_guard);
                    }
                }
            }
            if let Some(split) = pending {
                self.grow_root(split, &mut staged, log)?;
            }
            outcome
            // `_gate` drops here; all per-page guards have been
            // dropped already. The caller (bundle-aware path) folds
            // `staged` into the outer `CommitBundle`; the standalone
            // path (`write`) drains it immediately.
        };

        Ok((outcome, staged))
    }

    // Signature has seven params (self + 6) per ADR-030's staged
    // emit plumbing; see primary_index.rs::apply_leaf_op for the
    // mirror.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn apply_leaf_op(
        &self,
        leaf_id: PageId,
        guard: &mut ArcPageWriteGuard,
        key: SecondaryKey,
        node: Option<NodeId>,
        op: WriteOp,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(WriteOutcome, Option<SplitInfo>), SecondaryIndexError> {
        let leaf_bytes: &mut PageBuf = guard.as_mut();
        let mut leaf = LeafPageMut::open(leaf_bytes)?;

        match op {
            WriteOp::Insert => self.apply_leaf_insert(leaf_id, &mut leaf, key, node, staged, log),
            WriteOp::Remove => {
                let n = node.ok_or_else(|| SecondaryIndexError::CorruptPage {
                    page_id: leaf_id,
                    reason: "WriteOp::Remove without node".to_owned(),
                })?;
                let removed = match leaf.as_ref().find(key)? {
                    LeafFindResult::Found { index } => {
                        // C-IV fix: scan EVERY inline slot and EVERY
                        // overflow-chain page, zeroing every match.
                        // `insert` doesn't dedup (its docstring is
                        // explicit about this), so a NodeId may
                        // appear more than once under the same key;
                        // a "first match + stop" remove would leave
                        // phantom entries on subsequent lookups.
                        let mut entry = leaf.as_ref().entry(index)?;
                        let mut inline_changed = false;
                        for slot in entry.inline.iter_mut() {
                            if *slot == n {
                                *slot = NodeId::ZERO;
                                inline_changed = true;
                                // NOTE: no break — scan all 4 slots.
                            }
                        }
                        if inline_changed {
                            leaf.write_entry(index, &entry)?;
                            Self::stage_emit(staged, leaf_id, leaf.page_bytes());
                        }
                        // Walk the full overflow chain regardless of
                        // inline outcome — duplicates can straddle
                        // inline and chain.
                        let mut chain_changed = false;
                        let mut next_page = entry.overflow_head;
                        while next_page != PageId::ZERO {
                            let latch = self.page_store.latch(next_page)?;
                            let mut g = latch.write();
                            // Z-1 F-1: capture the overflow page's pre-W
                            // bytes before the tombstone rewrites it.
                            self.page_store.capture_from_write_guard(log, next_page, &g);
                            let mut page = OverflowPageMut::open(g.as_mut())?;
                            let mut page_changed = false;
                            // Loop tombstone_first until no more
                            // matches on this page.
                            while page.tombstone_first(n)? {
                                page_changed = true;
                                chain_changed = true;
                            }
                            let cont = page.next();
                            if page_changed {
                                Self::stage_emit(staged, next_page, g.as_ref());
                            }
                            drop(g);
                            next_page = cont;
                        }
                        inline_changed || chain_changed
                    }
                    LeafFindResult::Absent { .. } => false,
                };
                Ok((
                    WriteOutcome {
                        removed,
                        inserted_fresh_entry: false,
                    },
                    None,
                ))
            }
        }
    }

    // Seven params (self + leaf_id + leaf + key + node + staged + log):
    // the Z-1 F-1 rollback `log` threads through the ADR-030 staged-emit
    // plumbing; mirrors `apply_leaf_op`'s shape.
    #[allow(clippy::too_many_arguments)]
    fn apply_leaf_insert(
        &self,
        leaf_id: PageId,
        leaf: &mut LeafPageMut<'_>,
        key: SecondaryKey,
        node: Option<NodeId>,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(WriteOutcome, Option<SplitInfo>), SecondaryIndexError> {
        let node = node.ok_or_else(|| SecondaryIndexError::CorruptPage {
            page_id: leaf_id,
            reason: "WriteOp::Insert without node".to_owned(),
        })?;

        // Case 1: existing entry — fill the first empty inline slot,
        // or append into the overflow chain when inline is saturated.
        match leaf.as_ref().find(key)? {
            LeafFindResult::Found { index } => {
                let mut entry = leaf.as_ref().entry(index)?;
                if let Some(pos) = entry.first_empty_inline() {
                    entry.inline[pos] = node;
                    leaf.write_entry(index, &entry)?;
                    Self::stage_emit(staged, leaf_id, leaf.page_bytes());
                    return Ok((
                        WriteOutcome {
                            removed: false,
                            inserted_fresh_entry: false,
                        },
                        None,
                    ));
                }
                // Inline saturated — overflow path (DEC-15).
                let (new_head, leaf_updated) =
                    self.append_to_overflow_chain(leaf_id, &entry, node, staged, log)?;
                if let Some(new_head_pid) = new_head {
                    entry.overflow_head = new_head_pid;
                    leaf.write_entry(index, &entry)?;
                    Self::stage_emit(staged, leaf_id, leaf.page_bytes());
                }
                // If only an existing overflow page was mutated, its
                // WAL record was staged inside the helper; the leaf
                // page itself didn't change, so no leaf WAL.
                let _ = leaf_updated;
                Ok((
                    WriteOutcome {
                        removed: false,
                        inserted_fresh_entry: false,
                    },
                    None,
                ))
            }
            LeafFindResult::Absent { insert_at: _ } => {
                // Case 2: brand-new entry on a page with room.
                let fresh = LeafEntry {
                    key,
                    inline: [node, NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
                    overflow_head: PageId::ZERO,
                };
                if !leaf.is_full() {
                    leaf.insert_entry(fresh)?;
                    Self::stage_emit(staged, leaf_id, leaf.page_bytes());
                    return Ok((
                        WriteOutcome {
                            removed: false,
                            inserted_fresh_entry: true,
                        },
                        None,
                    ));
                }
                // Case 3: brand-new entry on a full page — split.
                let new_id = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
                let (mut new_buf, promoted_key) = leaf.split_into(new_id)?;
                if key < promoted_key {
                    leaf.insert_entry(fresh)?;
                } else {
                    let mut right = LeafPageMut::open(new_buf.as_mut())?;
                    right.insert_entry(fresh)?;
                }
                // ADR-030: stage both WAL emits; drain happens after
                // locks release.
                Self::stage_emit(staged, new_id, new_buf.as_ref());
                Self::stage_emit(staged, leaf_id, leaf.page_bytes());
                // Z-1 F-1: record the fresh split page so rollback can
                // remove it if the enclosing commit's WAL fsync fails.
                self.page_store.install_for_txn(log, new_id, new_buf)?;
                Ok((
                    WriteOutcome {
                        removed: false,
                        inserted_fresh_entry: true,
                    },
                    Some(SplitInfo {
                        promoted_key,
                        new_right_page: new_id,
                    }),
                ))
            }
        }
    }

    /// Append `node` to the overflow chain rooted at
    /// `leaf_entry.overflow_head`. If the chain is empty
    /// (`overflow_head == PageId::ZERO`), allocate a new head and
    /// return its page id so the caller can install it on the leaf
    /// entry. If an existing tail has room, append there. Otherwise
    /// allocate a new tail and link it from the previous tail.
    ///
    /// Returns `(Some(new_head), _)` iff a brand-new head page was
    /// allocated — the caller owes a `leaf_entry.overflow_head`
    /// update + a leaf WAL emission.
    fn append_to_overflow_chain(
        &self,
        _leaf_id: PageId,
        leaf_entry: &LeafEntry,
        node: NodeId,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(Option<PageId>, bool), SecondaryIndexError> {
        // Empty chain: allocate a fresh head page in memory, fill it,
        // stage the WAL snapshot, THEN install. Drain happens after
        // `write_gate` releases (ADR-030). Seed the tail cache (DEC-22)
        // with `new_head → new_head` — for a one-page chain the tail
        // IS the head — so the next duplicate append can jump straight
        // to the tail without walking.
        if leaf_entry.overflow_head == PageId::ZERO {
            let new_head = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
            let mut new_head_buf = fresh_page_buf(new_head, PageType::IndexOverflow);
            {
                let mut page = OverflowPageMut::open(new_head_buf.as_mut())?;
                page.append(node)?;
            }
            Self::stage_emit(staged, new_head, new_head_buf.as_ref());
            // Z-1 F-1: record the fresh overflow-head page for rollback.
            self.page_store
                .install_for_txn(log, new_head, new_head_buf)?;
            self.overflow_tail_cache.insert(new_head, new_head);
            return Ok((Some(new_head), false));
        }

        // Non-empty chain. Use the tail cache (DEC-22) to avoid the
        // O(chain_length) walk-to-tail per append — the alt review
        // flagged this as a quadratic hazard on duplicate-heavy
        // workloads (C-II). Cache miss falls back to walking from
        // head.
        let head = leaf_entry.overflow_head;
        let cached_tail = self.overflow_tail_cache.get(&head).map(|r| *r.value());
        let tail_candidate = cached_tail.unwrap_or(head);

        self.append_at_or_past_tail(head, tail_candidate, node, staged, log)
    }

    /// Append `node` at the tail of the overflow chain rooted at
    /// `head`, starting the search at `tail_candidate` (either the
    /// cached tail or the head on cache miss). On cache miss, walks
    /// forward once + populates the cache.
    fn append_at_or_past_tail(
        &self,
        head: PageId,
        tail_candidate: PageId,
        node: NodeId,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(Option<PageId>, bool), SecondaryIndexError> {
        let mut current = tail_candidate;
        loop {
            let latch = self.page_store.latch(current)?;
            let mut g = latch.write();
            // Z-1 F-1: capture the tail page's pre-W bytes before it is
            // mutated (append into it, or set_next linking a successor).
            // A pure forward-walk over a full page doesn't mutate it,
            // but capture is idempotent (deduped) and cheap, so
            // capturing every touched page is safe and simplest.
            self.page_store.capture_from_write_guard(log, current, &g);
            let mut page = OverflowPageMut::open(g.as_mut())?;
            if !page.is_full() {
                page.append(node)?;
                // ADR-030: stage the WAL snapshot under the held
                // write guard; the byte-copy captures post-append
                // state. Drain runs outside `write_gate`.
                Self::stage_emit(staged, current, g.as_ref());
                drop(g);
                // Refresh the cache — `current` is still the tail.
                // This is also the place where a stale cache entry
                // (from a cache that predates our knowledge of this
                // chain) gets corrected: `current` MIGHT have been
                // reached via a forward walk from the original
                // candidate, in which case we're now at the genuine
                // tail.
                self.overflow_tail_cache.insert(head, current);
                return Ok((None, false));
            }
            let cont = page.next();
            if cont != PageId::ZERO {
                // Cache was stale (the real tail is further along) or
                // the caller passed `head` on a cache miss — walk
                // forward and retry. Cache gets refreshed when we
                // eventually land on the real tail.
                drop(g);
                current = cont;
                continue;
            }
            // Tail is full and `next` is zero — allocate a successor
            // page in memory, fill it, stage its WAL snapshot,
            // install, THEN link the old tail's `next` pointer and
            // stage its updated snapshot. Ordering: reader-visible
            // state changes (install, set_next) are made-safe by the
            // staged WAL snapshots draining under ADR-023's
            // read-accelerator contract on restart.
            let new_tail = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
            let mut new_tail_buf = fresh_page_buf(new_tail, PageType::IndexOverflow);
            {
                let mut new_page = OverflowPageMut::open(new_tail_buf.as_mut())?;
                new_page.append(node)?;
            }
            Self::stage_emit(staged, new_tail, new_tail_buf.as_ref());
            // Z-1 F-1: record the fresh successor tail page for rollback.
            self.page_store
                .install_for_txn(log, new_tail, new_tail_buf)?;
            page.set_next(new_tail);
            Self::stage_emit(staged, current, g.as_ref());
            drop(g);
            // Update cache: new_tail is the current tail.
            self.overflow_tail_cache.insert(head, new_tail);
            return Ok((None, false));
        }
    }

    fn apply_internal_insert(
        &self,
        page_id: PageId,
        guard: &mut ArcPageWriteGuard,
        incoming: SplitInfo,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<Option<SplitInfo>, SecondaryIndexError> {
        let bytes: &mut PageBuf = guard.as_mut();
        let mut internal = InternalPageMut::open(bytes)?;
        if internal.is_full() {
            let new_id = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
            let (mut new_buf, promoted_key) = internal.split_into(new_id)?;
            if incoming.promoted_key < promoted_key {
                internal.insert(incoming.promoted_key, incoming.new_right_page)?;
            } else {
                let mut right = InternalPageMut::open(new_buf.as_mut())?;
                right.insert(incoming.promoted_key, incoming.new_right_page)?;
            }
            // ADR-030: stage both WAL emits; drain happens after
            // locks release.
            Self::stage_emit(staged, new_id, new_buf.as_ref());
            Self::stage_emit(staged, page_id, internal.page_bytes());
            // Z-1 F-1: record the fresh internal split page for rollback.
            self.page_store.install_for_txn(log, new_id, new_buf)?;
            return Ok(Some(SplitInfo {
                promoted_key,
                new_right_page: new_id,
            }));
        }
        internal.insert(incoming.promoted_key, incoming.new_right_page)?;
        Self::stage_emit(staged, page_id, internal.page_bytes());
        Ok(None)
    }

    fn grow_root(
        &self,
        split: SplitInfo,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(), SecondaryIndexError> {
        let old_root = self.root()?;
        let new_root_id = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
        let mut new_buf = fresh_page_buf(new_root_id, PageType::IndexInternal);
        {
            let mut new_root = InternalPageMut::init(new_buf.as_mut(), new_root_id, old_root);
            new_root.insert(split.promoted_key, split.new_right_page)?;
        }
        // ADR-031: stage the new-root IndexPage emit and stash the
        // pending MVCC root-pointer update. Deferring the persist
        // avoids nesting a SYSTEM-tenant MVCC commit inside an outer
        // commit's Phase 2 (which would deadlock on `install_order`).
        // See primary_index.rs::grow_root for the mirror rationale.
        Self::stage_emit(staged, new_root_id, new_buf.as_ref());
        // Z-1 F-1: record the OLD root under IndexHandle::SECONDARY so
        // rollback restores `root_cache` (root_changes drains BEFORE
        // new_pages per ADR-033 §5 — an in-flight reader that captured
        // the new root still finds the page mapped when it is removed).
        // MUST push root_change BEFORE the install_for_txn new_pages
        // entry so the drain order (root_changes → new_pages) holds.
        log.root_changes.push((IndexHandle::SECONDARY, old_root));
        self.page_store.install_for_txn(log, new_root_id, new_buf)?;
        self.root_cache.store(new_root_id.raw(), Ordering::Release);
        self.pending_root
            .store(new_root_id.raw(), Ordering::Release);
        Ok(())
    }

    // ────────── WAL emission ──────────

    /// Capture a byte-level snapshot of a page's post-mutation
    /// contents while the caller still holds its write latch, and
    /// push it onto `staged` for drain outside `write_gate`.
    ///
    /// See ADR-030. Mirror of `primary_index::PrimaryIndex::stage_emit`.
    #[inline]
    fn stage_emit(staged: &mut Vec<StagedEmit>, page_id: PageId, bytes: &[u8; PAGE_SIZE]) {
        let mut copy: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        copy.copy_from_slice(bytes);
        staged.push(StagedEmit {
            // Secondary-index pages carry the SecondaryIndex
            // discriminator so the v3 replay executor routes them
            // into `SecondaryPageStore` (PR #79 X-2 fold-in).
            kind: arcgraph_storage::wal::BundlePageKind::SecondaryIndex,
            page_id,
            bytes: copy,
        });
    }

    /// Emit a [`WalRecordType::IndexPage`] record carrying `bytes`
    /// as the post-write page image. The sole remaining caller is
    /// [`Self::new`], which emits the freshly-allocated root leaf
    /// (a single page — inherently atomic, so it does not need the
    /// `CommitBundle` folding that the multi-page write path uses
    /// since #37 [A-1]).
    ///
    /// **Locking contract (ADR-030).** This method must only be
    /// called with NO index locks held (`write_gate` released, no
    /// per-page write guards). The old DEC-21 contract required the
    /// caller to hold the write guard across the append; ADR-030
    /// narrows that rule for index pages because ADR-023 designates
    /// the secondary index (like the primary) as a read accelerator.
    /// Callers stage bytes via [`Self::stage_emit`] under the latch.
    ///
    /// The no-re-latch property of DEC-21 is preserved: this method
    /// does NOT latch the page store. Bytes are supplied directly.
    fn emit_wal_for_bytes(
        &self,
        page_id: PageId,
        bytes: &[u8; PAGE_SIZE],
    ) -> Result<(), SecondaryIndexError> {
        let Some(wal) = self.wal.as_ref() else {
            return Ok(());
        };
        self.emit_wal_for_bytes_inner(wal, page_id, bytes)
    }

    fn emit_wal_for_bytes_inner(
        &self,
        wal: &WalHandle,
        page_id: PageId,
        bytes: &[u8; PAGE_SIZE],
    ) -> Result<(), SecondaryIndexError> {
        let payload = encode_index_page_payload(page_id, TenantId::SYSTEM, bytes);
        wal.append(
            WalRecordType::IndexPage,
            /* txn_id = */ 0,
            now_millis(),
            TenantId::SYSTEM,
            payload,
        )
        .map_err(SecondaryIndexError::Core)
        .map(|_lsn: Lsn| ())
    }
}

/// Internal outcome from a `write` dispatch — not exported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriteOutcome {
    removed: bool,
    inserted_fresh_entry: bool,
}

impl From<SecondaryIndexValue> for PropertyValue {
    fn from(v: SecondaryIndexValue) -> Self {
        match v {
            SecondaryIndexValue::U32(x) => PropertyValue::U32(x),
            SecondaryIndexValue::U64(x) => PropertyValue::U64(x),
            SecondaryIndexValue::StringId(s) => PropertyValue::StringId(s),
            SecondaryIndexValue::StrHash(h) => PropertyValue::StrHash(h),
        }
    }
}

impl SecondaryIndexHandle for SecondaryIndex {
    fn insert_property(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
    ) -> Result<(), SecondaryIndexHandleError> {
        let key = SecondaryKey::new(tenant, label, property_key, value.into());
        self.insert(key, node)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))
    }

    fn remove_property(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
    ) -> Result<bool, SecondaryIndexHandleError> {
        let key = SecondaryKey::new(tenant, label, property_key, value.into());
        self.remove(key, node)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))
    }

    fn insert_property_deferred(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError> {
        let key = SecondaryKey::new(tenant, label, property_key, value.into());
        self.insert_deferred(key, node, log)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))
    }

    fn remove_property_deferred(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError> {
        let key = SecondaryKey::new(tenant, label, property_key, value.into());
        let (_removed, staged) = self
            .remove_deferred(key, node, log)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))?;
        Ok(staged)
    }

    fn persist_pending_root_update(&self) -> Result<(), SecondaryIndexHandleError> {
        Self::persist_pending_root_update(self)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))
    }

    // ─── Z-1 F-1 rollback dispatch ──────────────────────────────────

    fn rollback_remove_page(&self, page_id: PageId) {
        // Best-effort remove — an absent page is a no-op (already
        // removed by a sibling drain, or never installed).
        let _ = self.page_store.remove_page(page_id);
    }

    fn rollback_restore_page(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), SecondaryIndexHandleError> {
        self.page_store
            .restore_page_bytes(page_id, pre_bytes)
            .map_err(|e| SecondaryIndexHandleError::Backend(e.to_string()))
    }

    fn rollback_restore_root(&self, old_root_id: PageId) {
        // Restore the cached root pointer to its pre-grow_root value
        // and clear the pending (undurified) grow-root stash so the
        // aborted new root is never persisted to MVCC post-rollback.
        // Mirrors PrimaryIndex::restore_root_cache.
        self.root_cache.store(old_root_id.raw(), Ordering::Release);
        self.pending_root.store(0, Ordering::Release);
    }

    // ─── RC-2 write-follows-declare state machine (#1366) ────────────

    fn index_state(&self) -> IndexState {
        match self.index_state.load(Ordering::Acquire) {
            INDEX_STATE_BUILDING => IndexState::Building,
            // Any other encoding (only ONLINE is written) is Online.
            _ => IndexState::Online,
        }
    }

    fn set_index_state(&self, state: IndexState) {
        let encoded = match state {
            IndexState::Building => INDEX_STATE_BUILDING,
            IndexState::Online => INDEX_STATE_ONLINE,
        };
        self.index_state.store(encoded, Ordering::Release);
    }
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use proptest::prelude::*;

    use super::*;

    fn pv_u32(v: u32) -> PropertyValue {
        PropertyValue::U32(v)
    }

    fn k_u32(t: u64, l: u32, p: u32, v: u32) -> SecondaryKey {
        SecondaryKey::new(
            TenantId::new(t),
            LabelId::new(l),
            StringId::new(p),
            pv_u32(v),
        )
    }

    fn n(id: u64) -> NodeId {
        NodeId::new(id)
    }

    // ───── on-disk-key stability canary (#1366 R1 NIT-2) ─────

    /// Pins the current `hash_str_56("arcgraph_canary")` value. These
    /// 56-bit hashes are ON-DISK secondary-index keys; `DefaultHasher`
    /// (SipHash-1-3 seeded (0, 0)) is stable within a toolchain but is
    /// NOT contractually guaranteed across Rust upgrades. If a future
    /// toolchain changes the hash, this assert FAILS loudly — the signal
    /// to migrate on-disk keys rather than degrade to candidate-verify
    /// false-negatives silently.
    #[test]
    fn hash_str_56_canary_is_stable() {
        assert_eq!(
            hash_str_56("arcgraph_canary"),
            29_198_083_841_200_401,
            "DefaultHasher output changed for hash_str_56 — on-disk secondary-index \
             keys are no longer stable; migrate keys before shipping this toolchain"
        );
        // Sanity: masked to 56 bits.
        assert!(hash_str_56("arcgraph_canary") < (1u64 << 56));
    }

    // ───── SecondaryKey codec ─────

    #[test]
    fn secondary_key_roundtrip_u32() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        let key = k_u32(42, 7, 100, 12345);
        key.encode_into(&mut buf).unwrap();
        let back = SecondaryKey::decode(&buf).unwrap();
        assert_eq!(back, key);
        assert_eq!(buf[16], 0, "U32 variant tag byte is 0");
    }

    #[test]
    fn secondary_key_roundtrip_u64() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        let key = SecondaryKey::new(
            TenantId::new(1),
            LabelId::new(1),
            StringId::new(1),
            PropertyValue::U64(0x00FE_DCBA_9876_5432),
        );
        key.encode_into(&mut buf).unwrap();
        let back = SecondaryKey::decode(&buf).unwrap();
        assert_eq!(back, key);
        assert_eq!(buf[16], 1, "U64 variant tag byte is 1");
    }

    #[test]
    fn secondary_key_roundtrip_stringid() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        let key = SecondaryKey::new(
            TenantId::new(1),
            LabelId::new(1),
            StringId::new(1),
            PropertyValue::StringId(StringId::new(0xDEAD_BEEF)),
        );
        key.encode_into(&mut buf).unwrap();
        let back = SecondaryKey::decode(&buf).unwrap();
        assert_eq!(back, key);
        assert_eq!(buf[16], 2, "StringId variant tag byte is 2");
    }

    #[test]
    fn secondary_key_u64_value_overflow_rejected() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        let key = SecondaryKey::new(
            TenantId::DEFAULT,
            LabelId::new(1),
            StringId::new(1),
            PropertyValue::U64(1u64 << 56),
        );
        let err = key.encode_into(&mut buf).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::ValueOverflow { .. }));
    }

    #[test]
    fn secondary_key_u64_boundary_value_encodes() {
        // 2^56 - 1 is the maximum encodable U64.
        let mut buf = [0u8; SecondaryKey::SIZE];
        let max = (1u64 << 56) - 1;
        let key = SecondaryKey::new(
            TenantId::DEFAULT,
            LabelId::new(1),
            StringId::new(1),
            PropertyValue::U64(max),
        );
        key.encode_into(&mut buf).unwrap();
        let back = SecondaryKey::decode(&buf).unwrap();
        assert_eq!(back, key);
    }

    #[test]
    fn secondary_key_decode_rejects_nonzero_padding_u32() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        k_u32(1, 1, 1, 5).encode_into(&mut buf).unwrap();
        buf[22] = 0xFF;
        let err = SecondaryKey::decode(&buf).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptKey { .. }));
    }

    #[test]
    fn secondary_key_decode_rejects_unknown_variant() {
        let mut buf = [0u8; SecondaryKey::SIZE];
        k_u32(1, 1, 1, 5).encode_into(&mut buf).unwrap();
        buf[16] = 0xAB;
        let err = SecondaryKey::decode(&buf).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptKey { .. }));
    }

    // ───── SecondaryKey ordering ─────

    #[test]
    fn secondary_key_ordering_is_tenant_label_property_then_value() {
        // Same (label, property, value); differing tenant.
        let k_t1 = k_u32(1, 10, 100, 5);
        let k_t2 = k_u32(2, 10, 100, 5);
        assert!(k_t1 < k_t2, "tenant-major ordering");

        // Same tenant; differing label.
        let k_l1 = k_u32(1, 10, 100, 5);
        let k_l2 = k_u32(1, 11, 100, 5);
        assert!(k_l1 < k_l2, "label ordering within tenant");

        // Same tenant+label; differing property_key.
        let k_p1 = k_u32(1, 10, 100, 999);
        let k_p2 = k_u32(1, 10, 101, 0);
        assert!(k_p1 < k_p2, "property_key ordering within label");

        // Same tenant+label+property_key; differing value.
        let k_v1 = k_u32(1, 10, 100, 5);
        let k_v2 = k_u32(1, 10, 100, 6);
        assert!(k_v1 < k_v2, "value ordering within property_key");
    }

    #[test]
    fn secondary_key_variant_ordering_is_u32_then_u64_then_stringid() {
        let base = |v| SecondaryKey::new(TenantId::new(1), LabelId::new(1), StringId::new(1), v);
        let k_u32 = base(PropertyValue::U32(u32::MAX));
        let k_u64 = base(PropertyValue::U64(0));
        let k_sid = base(PropertyValue::StringId(StringId::ZERO));
        assert!(k_u32 < k_u64);
        assert!(k_u64 < k_sid);
    }

    // ───── LeafEntry codec ─────

    #[test]
    fn leaf_entry_roundtrip_no_overflow_no_nodes() {
        let mut buf = [0u8; LEAF_ENTRY_SIZE];
        let e = LeafEntry::empty(k_u32(1, 10, 100, 50));
        e.encode_into(&mut buf).unwrap();
        let back = LeafEntry::decode(&buf).unwrap();
        assert_eq!(back, e);
        assert_eq!(back.live_inline().count(), 0);
        assert_eq!(back.overflow_head, PageId::ZERO);
    }

    #[test]
    fn leaf_entry_roundtrip_with_partial_inline() {
        let mut buf = [0u8; LEAF_ENTRY_SIZE];
        let e = LeafEntry {
            key: k_u32(1, 10, 100, 50),
            inline: [n(1), n(2), NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };
        e.encode_into(&mut buf).unwrap();
        let back = LeafEntry::decode(&buf).unwrap();
        assert_eq!(back, e);
        assert_eq!(back.live_inline().collect::<Vec<_>>(), vec![n(1), n(2)]);
        assert_eq!(back.first_empty_inline(), Some(2));
    }

    #[test]
    fn leaf_entry_roundtrip_full_inline_and_overflow_head() {
        let mut buf = [0u8; LEAF_ENTRY_SIZE];
        let e = LeafEntry {
            key: k_u32(1, 10, 100, 50),
            inline: [n(1), n(2), n(3), n(4)],
            overflow_head: PageId::new(777),
        };
        e.encode_into(&mut buf).unwrap();
        let back = LeafEntry::decode(&buf).unwrap();
        assert_eq!(back, e);
        assert_eq!(back.first_empty_inline(), None);
        assert_eq!(back.overflow_head, PageId::new(777));
    }

    // ───── LeafPage codec ─────

    #[test]
    fn leaf_page_insert_and_find_roundtrip() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        assert_eq!(leaf.entry_count(), 0);

        let a = LeafEntry {
            key: k_u32(1, 10, 100, 10),
            inline: [n(1), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };
        let b = LeafEntry {
            key: k_u32(1, 10, 100, 20),
            inline: [n(2), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };
        let c = LeafEntry {
            key: k_u32(1, 10, 200, 0),
            inline: [n(3), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };

        // Out-of-order inserts; verify internal order preserved.
        leaf.insert_entry(b).unwrap();
        leaf.insert_entry(a).unwrap();
        leaf.insert_entry(c).unwrap();
        assert_eq!(leaf.entry_count(), 3);

        let view = leaf.as_ref();
        assert_eq!(view.entry(0).unwrap(), a);
        assert_eq!(view.entry(1).unwrap(), b);
        assert_eq!(view.entry(2).unwrap(), c);

        // find()
        assert!(matches!(
            view.find(a.key).unwrap(),
            LeafFindResult::Found { index: 0 }
        ));
        assert!(matches!(
            view.find(b.key).unwrap(),
            LeafFindResult::Found { index: 1 }
        ));
        assert!(matches!(
            view.find(c.key).unwrap(),
            LeafFindResult::Found { index: 2 }
        ));
        // Absent key between a and b.
        assert!(matches!(
            view.find(k_u32(1, 10, 100, 15)).unwrap(),
            LeafFindResult::Absent { insert_at: 1 }
        ));
    }

    #[test]
    fn leaf_page_duplicate_insert_rejected() {
        let mut page = fresh_page_buf(PageId::new(2), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        let e = LeafEntry {
            key: k_u32(1, 10, 100, 5),
            inline: [n(1), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };
        leaf.insert_entry(e).unwrap();
        let err = leaf.insert_entry(e).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    #[test]
    fn leaf_page_upsert_replaces_in_place() {
        let mut page = fresh_page_buf(PageId::new(3), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        let key = k_u32(1, 10, 100, 5);
        let first = LeafEntry {
            key,
            inline: [n(1), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        };
        let second = LeafEntry {
            key,
            inline: [n(1), n(2), n(3), n(4)],
            overflow_head: PageId::new(42),
        };
        assert_eq!(leaf.upsert_entry(first).unwrap(), None);
        assert_eq!(leaf.upsert_entry(second).unwrap(), Some(first));
        assert_eq!(leaf.entry_count(), 1);
        assert_eq!(leaf.as_ref().entry(0).unwrap(), second);
    }

    #[test]
    fn leaf_page_full_rejects_new_key() {
        let mut page = fresh_page_buf(PageId::new(4), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        for i in 0..LEAF_CAPACITY as u32 {
            let e = LeafEntry {
                key: k_u32(1, 10, 100, i),
                inline: [
                    n(u64::from(i + 1)),
                    NodeId::ZERO,
                    NodeId::ZERO,
                    NodeId::ZERO,
                ],
                overflow_head: PageId::ZERO,
            };
            leaf.insert_entry(e).unwrap();
        }
        assert!(leaf.is_full());
        let err = leaf
            .insert_entry(LeafEntry::empty(k_u32(1, 10, 100, u32::MAX)))
            .unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
        // Upsert of an existing key is still allowed — no new slot.
        leaf.upsert_entry(LeafEntry {
            key: k_u32(1, 10, 100, 0),
            inline: [n(999), NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            overflow_head: PageId::ZERO,
        })
        .unwrap();
    }

    #[test]
    fn wrong_page_type_rejected_on_open() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::Node);
        let err = LeafPageMut::open(page.as_mut()).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    // ───── Internal page codec ─────

    #[test]
    fn internal_capacity_constants_match_invariants() {
        assert_eq!(INTERNAL_ENTRY_SIZE, 32);
        assert_eq!(INTERNAL_CAPACITY, 254);
        assert_eq!(
            INTERNAL_ENTRY_OFFSET + (INTERNAL_CAPACITY as usize) * INTERNAL_ENTRY_SIZE,
            48 + 254 * 32
        );
    }

    #[test]
    fn internal_insert_roundtrip_single_page() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(42), PageId::new(100));
        assert_eq!(node.entry_count(), 0);
        assert_eq!(node.as_ref().first_child(), PageId::new(100));

        // Insert a few (key, child) pairs out of order.
        node.insert(k_u32(1, 10, 100, 30), PageId::new(230))
            .unwrap();
        node.insert(k_u32(1, 10, 100, 10), PageId::new(210))
            .unwrap();
        node.insert(k_u32(1, 10, 100, 20), PageId::new(220))
            .unwrap();
        assert_eq!(node.entry_count(), 3);

        let view = node.as_ref();
        let (k0, c0) = view.entry(0).unwrap();
        let (k1, c1) = view.entry(1).unwrap();
        let (k2, c2) = view.entry(2).unwrap();
        assert_eq!(k0, k_u32(1, 10, 100, 10));
        assert_eq!(k1, k_u32(1, 10, 100, 20));
        assert_eq!(k2, k_u32(1, 10, 100, 30));
        assert_eq!(c0, PageId::new(210));
        assert_eq!(c1, PageId::new(220));
        assert_eq!(c2, PageId::new(230));
        assert_eq!(view.first_child(), PageId::new(100));
    }

    #[test]
    fn internal_insert_ordering_preserved_across_tenants_and_labels() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        let keys = [
            k_u32(1, 10, 100, 5),
            k_u32(1, 10, 101, 0),
            k_u32(2, 10, 100, 7),
            k_u32(1, 11, 100, 100),
        ];
        for (i, key) in keys.iter().copied().enumerate() {
            node.insert(key, PageId::new(1000 + i as u64)).unwrap();
        }
        let view = node.as_ref();
        let mut seen = Vec::new();
        for i in 0..view.entry_count() {
            let (key, _) = view.entry(i).unwrap();
            seen.push(key);
        }
        let mut sorted = keys.to_vec();
        sorted.sort();
        assert_eq!(seen, sorted);
    }

    #[test]
    fn internal_find_child_is_binary_search_correct() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(50));
        // Keys (value) 10, 20, 30 with children 60, 70, 80.
        node.insert(k_u32(1, 10, 100, 10), PageId::new(60)).unwrap();
        node.insert(k_u32(1, 10, 100, 20), PageId::new(70)).unwrap();
        node.insert(k_u32(1, 10, 100, 30), PageId::new(80)).unwrap();
        let view = node.as_ref();
        // Strictly less than key[0] → first_child.
        assert_eq!(
            view.find_child(k_u32(1, 10, 100, 5)).unwrap(),
            PageId::new(50)
        );
        // Exactly key[0] → child[0].
        assert_eq!(
            view.find_child(k_u32(1, 10, 100, 10)).unwrap(),
            PageId::new(60)
        );
        // In [key[0], key[1]) → child[0].
        assert_eq!(
            view.find_child(k_u32(1, 10, 100, 15)).unwrap(),
            PageId::new(60)
        );
        // Exactly key[1] → child[1].
        assert_eq!(
            view.find_child(k_u32(1, 10, 100, 20)).unwrap(),
            PageId::new(70)
        );
        // After last key → last child.
        assert_eq!(
            view.find_child(k_u32(1, 10, 100, 1_000)).unwrap(),
            PageId::new(80)
        );
        // Empty internal node → first_child.
        let mut page2 = [0u8; PAGE_SIZE];
        let empty = InternalPageMut::init(&mut page2, PageId::new(2), PageId::new(42));
        assert_eq!(
            empty.as_ref().find_child(k_u32(1, 10, 100, 0)).unwrap(),
            PageId::new(42)
        );
    }

    #[test]
    fn internal_full_page_rejects_new_child() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        for i in 0..INTERNAL_CAPACITY as u32 {
            node.insert(k_u32(1, 10, 100, i), PageId::new(u64::from(i + 2)))
                .unwrap();
        }
        assert!(node.is_full());
        let err = node
            .insert(k_u32(1, 10, 100, u32::MAX), PageId::new(999_999))
            .unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    #[test]
    fn internal_duplicate_key_rejected() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        let k = k_u32(1, 10, 100, 5);
        node.insert(k, PageId::new(1)).unwrap();
        let err = node.insert(k, PageId::new(2)).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    #[test]
    fn internal_wrong_page_type_rejected_on_open() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let err = InternalPageMut::open(page.as_mut()).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    // ───── Split with median promotion ─────

    #[test]
    fn leaf_split_promotes_first_right_key_and_trims_left() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        for i in 0..10u32 {
            let e = LeafEntry {
                key: k_u32(1, 10, 100, i),
                inline: [
                    n(u64::from(i + 1)),
                    NodeId::ZERO,
                    NodeId::ZERO,
                    NodeId::ZERO,
                ],
                overflow_head: PageId::ZERO,
            };
            leaf.insert_entry(e).unwrap();
        }
        assert_eq!(leaf.entry_count(), 10);

        let (right_buf, promoted) = leaf.split_into(PageId::new(2)).unwrap();
        assert_eq!(promoted, k_u32(1, 10, 100, 5), "median key of [0..10)");
        assert_eq!(leaf.entry_count(), 5);
        // Left half carries keys [0, 5).
        for i in 0..5u16 {
            let e = leaf.as_ref().entry(i).unwrap();
            assert_eq!(e.key, k_u32(1, 10, 100, u32::from(i)));
        }
        // Right half carries keys [5, 10).
        let right = LeafPageRef::open(right_buf.as_ref()).unwrap();
        assert_eq!(right.entry_count(), 5);
        for i in 0..5u16 {
            let e = right.entry(i).unwrap();
            assert_eq!(e.key, k_u32(1, 10, 100, u32::from(i + 5)));
        }
        // Right page header carries the correct page_id + page_type.
        assert_eq!(right.page_id(), PageId::new(2));

        // Also verify the freed tail on the left half is zeroed (a
        // codec bug that reads past slot_count sees zeros, not stale).
        let _ = right_buf; // retain until after the left-tail check
        let tail_off = LEAF_ENTRY_OFFSET + 5 * LEAF_ENTRY_SIZE;
        for b in &page[tail_off..tail_off + 5 * LEAF_ENTRY_SIZE] {
            assert_eq!(*b, 0, "split should zero the freed left tail");
        }
    }

    #[test]
    fn leaf_split_preserves_total_order() {
        let mut page = fresh_page_buf(PageId::new(3), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        // Insert out of order across multiple labels + property keys.
        let keys = [
            k_u32(1, 10, 100, 50),
            k_u32(1, 10, 100, 10),
            k_u32(1, 10, 100, 30),
            k_u32(1, 10, 200, 5),
            k_u32(1, 11, 100, 0),
            k_u32(2, 10, 100, 9999),
            k_u32(1, 10, 100, 20),
            k_u32(1, 10, 100, 40),
        ];
        for k in keys {
            leaf.insert_entry(LeafEntry::empty(k)).unwrap();
        }

        let (right_buf, promoted) = leaf.split_into(PageId::new(4)).unwrap();
        let left_keys: Vec<SecondaryKey> = (0..leaf.entry_count())
            .map(|i| leaf.as_ref().entry(i).unwrap().key)
            .collect();
        let right_view = LeafPageRef::open(right_buf.as_ref()).unwrap();
        let right_keys: Vec<SecondaryKey> = (0..right_view.entry_count())
            .map(|i| right_view.entry(i).unwrap().key)
            .collect();
        // Left is strictly ascending.
        for w in left_keys.windows(2) {
            assert!(w[0] < w[1], "left half monotone");
        }
        // Right is strictly ascending.
        for w in right_keys.windows(2) {
            assert!(w[0] < w[1], "right half monotone");
        }
        // Left-max < promoted = right-min.
        assert_eq!(promoted, right_keys[0]);
        assert!(left_keys.last().unwrap() < &promoted);
    }

    #[test]
    fn leaf_split_requires_two_entries() {
        let mut page = fresh_page_buf(PageId::new(5), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        leaf.insert_entry(LeafEntry::empty(k_u32(1, 10, 100, 0)))
            .unwrap();
        let err = leaf.split_into(PageId::new(6)).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    #[test]
    fn internal_split_promotes_median_and_migrates_right_subtree() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(100));
        // 7 pairs: keys 10..70, children 101..107
        for i in 0..7u32 {
            let key = k_u32(1, 10, 100, (i + 1) * 10);
            node.insert(key, PageId::new(101 + u64::from(i))).unwrap();
        }
        assert_eq!(node.entry_count(), 7);

        // mid = 3, promoted = keys[3] = 40, right.first_child = children[3] = 104,
        // right pairs = 50→105, 60→106, 70→107
        let (right_buf, promoted) = node.split_into(PageId::new(200)).unwrap();
        assert_eq!(promoted, k_u32(1, 10, 100, 40));
        assert_eq!(node.entry_count(), 3);
        // Left keeps pairs 0..3 and original first_child.
        assert_eq!(node.as_ref().first_child(), PageId::new(100));
        for (i, exp_key_v) in [10, 20, 30].iter().enumerate() {
            let (k, c) = node.as_ref().entry(i as u16).unwrap();
            assert_eq!(k, k_u32(1, 10, 100, *exp_key_v));
            assert_eq!(c, PageId::new(101 + i as u64));
        }
        // Right carries pairs 4..7 with first_child = children[3] = 104.
        let right = InternalPageRef::open(right_buf.as_ref()).unwrap();
        assert_eq!(right.first_child(), PageId::new(104));
        assert_eq!(right.entry_count(), 3);
        for (i, exp_key_v) in [50, 60, 70].iter().enumerate() {
            let (k, c) = right.entry(i as u16).unwrap();
            assert_eq!(k, k_u32(1, 10, 100, *exp_key_v));
            assert_eq!(c, PageId::new(105 + i as u64));
        }
        assert_eq!(right.page_id(), PageId::new(200));

        // Freed tail on the left should be zeroed.
        let _ = right_buf;
        let tail_off = INTERNAL_ENTRY_OFFSET + 3 * INTERNAL_ENTRY_SIZE;
        for b in &page[tail_off..tail_off + 4 * INTERNAL_ENTRY_SIZE] {
            assert_eq!(*b, 0, "split should zero the freed left tail");
        }
    }

    #[test]
    fn internal_split_even_count_handles_right_first_child() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(100));
        // 4 pairs: mid = 2, right has 1 pair.
        for i in 0..4u32 {
            node.insert(k_u32(1, 10, 100, i), PageId::new(200 + u64::from(i)))
                .unwrap();
        }
        let (right_buf, promoted) = node.split_into(PageId::new(500)).unwrap();
        assert_eq!(promoted, k_u32(1, 10, 100, 2));
        assert_eq!(node.entry_count(), 2);
        let right = InternalPageRef::open(right_buf.as_ref()).unwrap();
        assert_eq!(right.entry_count(), 1);
        assert_eq!(right.first_child(), PageId::new(202));
        let (k, c) = right.entry(0).unwrap();
        assert_eq!(k, k_u32(1, 10, 100, 3));
        assert_eq!(c, PageId::new(203));
        let _ = right_buf;
    }

    #[test]
    fn internal_split_requires_two_pairs() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(100));
        node.insert(k_u32(1, 10, 100, 0), PageId::new(200)).unwrap();
        let err = node.split_into(PageId::new(500)).unwrap_err();
        assert!(matches!(err, SecondaryIndexError::CorruptPage { .. }));
    }

    #[test]
    fn split_info_struct_is_copyable() {
        // Lightweight sanity check on the split-propagation handoff
        // type used by the tree-level mutation code.
        let s = SplitInfo {
            promoted_key: k_u32(1, 10, 100, 5),
            new_right_page: PageId::new(42),
        };
        let s2 = s;
        assert_eq!(s, s2);
    }

    // ───── SecondaryIndex public API (commit 4) ─────

    fn build_index() -> SecondaryIndex {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        SecondaryIndex::new(txn_mgr, alloc, None).expect("fresh index builds")
    }

    #[test]
    fn new_installs_root_and_caches_page_id() {
        let idx = build_index();
        let root = idx.root().unwrap();
        assert!(root.raw() > 0, "root page id is allocated > 0");
        assert_eq!(idx.page_store().len(), 1);
    }

    #[test]
    fn insert_lookup_single_entry_roundtrip() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        idx.insert(k, n(42)).unwrap();
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits, vec![n(42)]);
        // Absent key → empty vec.
        assert!(idx.lookup(k_u32(1, 10, 100, 999)).unwrap().is_empty());
    }

    #[test]
    fn insert_four_duplicates_stay_inline() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        for i in 1u64..=4 {
            idx.insert(k, n(i)).unwrap();
        }
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits, vec![n(1), n(2), n(3), n(4)]);
    }

    #[test]
    fn fifth_duplicate_allocates_overflow_page() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        for i in 1u64..=4 {
            idx.insert(k, n(i)).unwrap();
        }
        let before_pages = idx.page_store().len();
        idx.insert(k, n(5)).unwrap();
        assert_eq!(
            idx.page_store().len(),
            before_pages + 1,
            "5th duplicate must allocate an overflow page"
        );
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits, (1..=5u64).map(n).collect::<Vec<_>>());
    }

    #[test]
    fn overflow_chain_grows_to_second_page_at_capacity() {
        // 4 inline + 1 017 in the first overflow page + 1 more in a
        // newly-allocated second overflow page = 1 022 NodeIds total.
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        let total = 4 + OVERFLOW_SLOTS_PER_PAGE + 1;
        for i in 1..=total as u64 {
            idx.insert(k, n(i)).unwrap();
        }
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits.len(), total);
        assert_eq!(
            hits,
            (1..=total as u64).map(n).collect::<Vec<_>>(),
            "order must match insertion"
        );
    }

    #[test]
    fn remove_walks_overflow_chain() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        // Fill inline + seed a few overflow entries.
        for i in 1u64..=10 {
            idx.insert(k, n(i)).unwrap();
        }
        // Remove a NodeId that only exists in the overflow chain
        // (slots 5..=10 live on the first overflow page).
        assert!(idx.remove(k, n(7)).unwrap());
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits.len(), 9);
        assert!(!hits.contains(&n(7)));
    }

    #[test]
    fn remove_and_reinsert_across_overflow_boundary() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        for i in 1u64..=6 {
            idx.insert(k, n(i)).unwrap();
        }
        // Remove an inline slot (first position).
        assert!(idx.remove(k, n(1)).unwrap());
        // Reinsert lands in the freed inline slot, not at the chain
        // tail — `first_empty_inline` returns 0.
        idx.insert(k, n(99)).unwrap();
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits.len(), 6);
        assert_eq!(hits[0], n(99), "reinsert re-used the vacated inline slot");
    }

    #[test]
    fn overflow_tail_cache_tracks_current_tail() {
        // DEC-22 regression. Fill 4 inline + 3 × OVERFLOW_SLOTS_PER_PAGE
        // duplicates, verify the cache points at the actual tail
        // (== 3rd overflow page). Before this fix, `append_to_overflow_chain`
        // walked every page per insert — O(N²/1017) total.
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        let total = 4 + 3 * OVERFLOW_SLOTS_PER_PAGE;
        for i in 1..=total as u64 {
            idx.insert(k, n(i)).unwrap();
        }
        // Lookup still returns everything.
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits.len(), total);

        // Dig into internals: find the leaf entry, inspect its
        // overflow_head, and verify the cached tail matches the real
        // tail (walk the chain once from head to cross-check).
        let leaf_entry = idx.descend_read(k).unwrap().expect("entry present");
        let head = leaf_entry.overflow_head;
        assert_ne!(head, PageId::ZERO);
        // Walk the chain to find the actual tail.
        let mut real_tail = head;
        loop {
            let latch = idx.page_store().latch(real_tail).unwrap();
            let g = latch.read();
            let p = OverflowPageRef::open(g.as_ref()).unwrap();
            let cont = p.next();
            if cont == PageId::ZERO {
                break;
            }
            real_tail = cont;
        }
        // Cache should match the walked tail.
        let cached = idx
            .overflow_tail_cache
            .get(&head)
            .map(|r| *r.value())
            .expect("cache populated");
        assert_eq!(cached, real_tail, "tail cache must point at current tail");
    }

    #[test]
    fn overflow_tail_cache_recovers_from_cache_miss() {
        // Simulate a cold cache (e.g., M2.e restart before replay
        // rebuilds the cache): start fresh, manually populate a
        // chain through the public API, then evict the cache entry
        // and insert again. The walk-from-head fallback must still
        // converge on the correct tail AND re-populate the cache.
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        for i in 1..=(4 + OVERFLOW_SLOTS_PER_PAGE as u64 + 5) {
            idx.insert(k, n(i)).unwrap();
        }
        // Find the chain head and blow away its cache entry.
        let leaf_entry = idx.descend_read(k).unwrap().expect("entry present");
        let head = leaf_entry.overflow_head;
        idx.overflow_tail_cache.remove(&head);

        // Next insert should walk from head, find the real tail,
        // append, and re-cache.
        idx.insert(k, n(99_999)).unwrap();
        let cached = idx
            .overflow_tail_cache
            .get(&head)
            .map(|r| *r.value())
            .expect("cache should be repopulated after miss");
        assert_ne!(cached, PageId::ZERO);
        let hits = idx.lookup(k).unwrap();
        assert!(hits.contains(&n(99_999)));
    }

    #[test]
    fn remove_drains_stacked_inline_and_chain_duplicates() {
        // C-IV regression. `insert` doesn't dedup, so the same
        // NodeId can end up in multiple inline slots AND the chain.
        // `remove` must zero every occurrence; "first-match-and-stop"
        // would leave phantom entries that subsequent lookups return.
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        // Stack NodeId(7) three times in inline: slots 0, 1, 2.
        idx.insert(k, n(7)).unwrap();
        idx.insert(k, n(7)).unwrap();
        idx.insert(k, n(7)).unwrap();
        // Fill the last inline slot with something else so the chain
        // allocates — then stack n(7) twice more in the chain.
        idx.insert(k, n(99)).unwrap();
        idx.insert(k, n(7)).unwrap(); // → chain
        idx.insert(k, n(7)).unwrap(); // → chain (same page)

        let pre = idx.lookup(k).unwrap();
        assert_eq!(pre.iter().filter(|x| **x == n(7)).count(), 5);
        assert_eq!(pre.iter().filter(|x| **x == n(99)).count(), 1);

        // A single remove drains ALL five copies of n(7).
        assert!(idx.remove(k, n(7)).unwrap());
        let post = idx.lookup(k).unwrap();
        assert!(
            !post.contains(&n(7)),
            "post-remove: every n(7) occurrence must be zeroed"
        );
        assert!(post.contains(&n(99)), "n(99) must survive");
    }

    #[test]
    fn lookup_empty_returns_empty_vec_after_all_tombstoned() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        idx.insert(k, n(1)).unwrap();
        idx.insert(k, n(2)).unwrap();
        assert!(idx.remove(k, n(1)).unwrap());
        assert!(idx.remove(k, n(2)).unwrap());
        assert!(
            idx.lookup(k).unwrap().is_empty(),
            "all-tombstoned lookup is empty"
        );
    }

    #[test]
    fn remove_zeros_inline_slot_and_allows_reinsert() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        idx.insert(k, n(1)).unwrap();
        idx.insert(k, n(2)).unwrap();
        assert!(idx.remove(k, n(1)).unwrap());
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits, vec![n(2)]);
        // Reinsert of the same node lands in the vacated slot
        // (first_empty_inline is now position 0).
        idx.insert(k, n(3)).unwrap();
        let hits = idx.lookup(k).unwrap();
        assert_eq!(hits, vec![n(3), n(2)]);
    }

    #[test]
    fn remove_missing_key_returns_false() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        assert!(!idx.remove(k, n(1)).unwrap());
    }

    #[test]
    fn remove_missing_node_in_live_key_returns_false() {
        let idx = build_index();
        let k = k_u32(1, 10, 100, 5);
        idx.insert(k, n(1)).unwrap();
        assert!(!idx.remove(k, n(999)).unwrap());
        // Original node still live.
        assert_eq!(idx.lookup(k).unwrap(), vec![n(1)]);
    }

    #[test]
    fn tenant_isolation_across_same_label_property_value() {
        let idx = build_index();
        let t1 = SecondaryKey::new(
            TenantId::new(1),
            LabelId::new(10),
            StringId::new(100),
            PropertyValue::U32(5),
        );
        let t2 = SecondaryKey::new(
            TenantId::new(2),
            LabelId::new(10),
            StringId::new(100),
            PropertyValue::U32(5),
        );
        idx.insert(t1, n(111)).unwrap();
        idx.insert(t2, n(222)).unwrap();
        assert_eq!(idx.lookup(t1).unwrap(), vec![n(111)]);
        assert_eq!(idx.lookup(t2).unwrap(), vec![n(222)]);
    }

    #[test]
    fn root_split_grows_new_internal_root() {
        let idx = build_index();
        let initial_root = idx.root().unwrap();
        // Fill the root leaf to capacity, then one more → split.
        for i in 0..LEAF_CAPACITY as u32 {
            idx.insert(k_u32(1, 10, 100, i), n(u64::from(i + 1)))
                .unwrap();
        }
        assert_eq!(idx.root().unwrap(), initial_root);
        // The (LEAF_CAPACITY + 1)-th unique insert triggers split +
        // root growth.
        idx.insert(
            k_u32(1, 10, 100, LEAF_CAPACITY as u32),
            n(u64::from(LEAF_CAPACITY) + 1),
        )
        .unwrap();
        let new_root = idx.root().unwrap();
        assert_ne!(new_root, initial_root, "root must have grown");
        // All inserts are still findable.
        for i in 0..=(LEAF_CAPACITY as u32) {
            let hits = idx.lookup(k_u32(1, 10, 100, i)).unwrap();
            assert_eq!(hits, vec![n(u64::from(i) + 1)], "key {i} must be found");
        }
    }

    #[test]
    fn lookup_across_split_returns_all_keys() {
        let idx = build_index();
        let total = 2 * LEAF_CAPACITY as u32 + 50; // forces multiple splits
        for i in 0..total {
            idx.insert(k_u32(1, 10, 100, i), n(u64::from(i + 1)))
                .unwrap();
        }
        for i in 0..total {
            let hits = idx.lookup(k_u32(1, 10, 100, i)).unwrap();
            assert_eq!(hits, vec![n(u64::from(i + 1))], "key {i} not found");
        }
        assert!(idx.lookup(k_u32(1, 10, 100, total + 1)).unwrap().is_empty());
    }

    #[test]
    fn insert_lookup_three_variants_coexist_at_same_prop_key() {
        let idx = build_index();
        let base = |v| SecondaryKey::new(TenantId::new(1), LabelId::new(10), StringId::new(100), v);
        let k_u32 = base(PropertyValue::U32(5));
        let k_u64 = base(PropertyValue::U64(5));
        let k_sid = base(PropertyValue::StringId(StringId::new(5)));
        idx.insert(k_u32, n(1)).unwrap();
        idx.insert(k_u64, n(2)).unwrap();
        idx.insert(k_sid, n(3)).unwrap();
        assert_eq!(idx.lookup(k_u32).unwrap(), vec![n(1)]);
        assert_eq!(idx.lookup(k_u64).unwrap(), vec![n(2)]);
        assert_eq!(idx.lookup(k_sid).unwrap(), vec![n(3)]);
    }

    // ─────────────────────────────────────────────────────────────────
    // W28-S1: model-based property tests + build-determinism.
    //
    // Closes the testing strategy / `docs/testing-strategy.md` §2.2 gap
    // flagged by the W28 test-slice plan (Slice S1; gap analysis
    // PR #510): `proptest` was declared in `Cargo.toml` but never used by
    // this crate — a silent §6 violation. These tests exercise only the
    // *implemented* surface: point insert / remove / lookup, mark-only
    // delete, leaf + internal splits, and the inline→overflow duplicate
    // chain. The delete is mark-only and the public API is point-lookup
    // only (no range scan, no merge / rebalance), so none of those are
    // tested — that code does not exist (scope guard: test only what is
    // implemented).
    //
    // Oracle design (deliberately model-equality, NOT a consistency /
    // dedupe check):
    //  * `prop_btree_matches_btreemap_model` compares the real
    //    `SecondaryIndex` against an independent `BTreeMap`-keyed
    //    reference model. The per-key model (`KeyModel`) faithfully
    //    reproduces the *documented* slot policy (module docs §"Delete
    //    policy"): the four inline slots are reused first-zero-first, the
    //    5th+ live duplicate appends to an append-only overflow region
    //    (mid-chain tombstones are never reused — `OverflowPageMut::
    //    append` uses a monotonic write head), and `remove` drains
    //    *every* matching slot (insert does not dedup). Equality is
    //    therefore exact, including lookup order AND multiplicity —
    //    multiplicity is preserved on purpose so the oracle never
    //    collapses into the dedupe check the spec warns against.
    //  * `btree_build_determinism` uses the binary-equal reference-
    //    snapshot oracle (strictly stronger than dedupe-consistency; per
    //    `feedback_determinism_oracle_concurrency_tests`): two indexes
    //    built from the same op sequence must be byte-identical across
    //    the whole page store. Page bytes are deterministic here — the
    //    only header nonce candidates (`lsn`, `crc`) are constant
    //    (`PageHeader::new` zeroes `lsn`; no WAL is attached), and page
    //    ids come from per-`(tenant, page_type)` monotonic counters that
    //    both fresh `PageAllocator`s replay identically.

    /// Independent per-key reference model of one key's `NodeId`
    /// storage. Reproduces the documented slot policy so the oracle can
    /// assert exact `lookup` equality (order + multiplicity).
    #[derive(Default, Clone)]
    struct KeyModel {
        inline: [Option<NodeId>; INLINE_NODEID_COUNT],
        overflow: Vec<Option<NodeId>>,
    }

    impl KeyModel {
        /// Mirror of `apply_leaf_insert`: fill the first empty inline
        /// slot, else append at the overflow tail (never backfill a
        /// tombstoned mid-chain slot).
        fn insert(&mut self, node: NodeId) {
            if let Some(slot) = self.inline.iter_mut().find(|s| s.is_none()) {
                *slot = Some(node);
            } else {
                self.overflow.push(Some(node));
            }
        }

        /// Mirror of the C-IV remove: zero EVERY matching slot across
        /// inline + overflow. Returns whether at least one was found.
        fn remove(&mut self, node: NodeId) -> bool {
            let mut found = false;
            for slot in self.inline.iter_mut().chain(self.overflow.iter_mut()) {
                if *slot == Some(node) {
                    *slot = None;
                    found = true;
                }
            }
            found
        }

        /// Mirror of `SecondaryIndex::lookup`: inline live slots in
        /// order, then overflow live slots in order.
        fn lookup(&self) -> Vec<NodeId> {
            self.inline
                .iter()
                .chain(self.overflow.iter())
                .filter_map(|s| *s)
                .collect()
        }
    }

    /// One generated operation against the index.
    #[derive(Debug, Clone)]
    enum ModelOp {
        Insert(SecondaryKey, NodeId),
        Remove(SecondaryKey, NodeId),
        Lookup(SecondaryKey),
    }

    /// Logical (allocation-independent) snapshot of one tree node, used
    /// for the readable structural-equality leg of the determinism
    /// oracle.
    #[derive(Debug, PartialEq, Eq)]
    enum LogicalNode {
        Leaf(Vec<LeafEntry>),
        Internal(Vec<SecondaryKey>),
    }

    /// In-order (left-to-right) traversal of the whole tree, descending
    /// internal nodes via `first_child` then each separator's child.
    fn walk_nodes(idx: &SecondaryIndex) -> Vec<LogicalNode> {
        let mut out = Vec::new();
        walk_rec(idx, idx.root().expect("root resolves"), &mut out);
        out
    }

    fn walk_rec(idx: &SecondaryIndex, page_id: PageId, out: &mut Vec<LogicalNode>) {
        let latch = idx.page_store().latch(page_id).expect("page latch");
        let g = latch.read();
        let header = read_header(g.as_ref()).expect("page header");
        if header.page_type == PageType::IndexLeaf.as_byte() {
            let leaf = LeafPageRef::open(g.as_ref()).expect("leaf open");
            let entries: Vec<LeafEntry> = (0..leaf.entry_count())
                .map(|i| leaf.entry(i).expect("leaf entry"))
                .collect();
            out.push(LogicalNode::Leaf(entries));
        } else if header.page_type == PageType::IndexInternal.as_byte() {
            let internal = InternalPageRef::open(g.as_ref()).expect("internal open");
            let mut children = vec![internal.first_child()];
            let mut separators = Vec::new();
            for i in 0..internal.entry_count() {
                let (k, c) = internal.entry(i).expect("internal entry");
                separators.push(k);
                children.push(c);
            }
            out.push(LogicalNode::Internal(separators));
            drop(g);
            for child in children {
                walk_rec(idx, child, out);
            }
        } else {
            panic!("unexpected page type {} at {page_id:?}", header.page_type);
        }
    }

    /// In-order leaf keys (one per stored entry, including tombstoned
    /// entries — the entry survives a full drain in this mark-only tree).
    fn walk_leaf_keys(idx: &SecondaryIndex) -> Vec<SecondaryKey> {
        let mut keys = Vec::new();
        for node in walk_nodes(idx) {
            if let LogicalNode::Leaf(entries) = node {
                keys.extend(entries.into_iter().map(|e| e.key));
            }
        }
        keys
    }

    /// Keys with at least one live (non-tombstoned) value — i.e. the set
    /// a caller can actually retrieve.
    fn live_keys_in_tree(idx: &SecondaryIndex) -> BTreeSet<SecondaryKey> {
        let mut set = BTreeSet::new();
        for node in walk_nodes(idx) {
            if let LogicalNode::Leaf(entries) = node {
                for e in entries {
                    if e.live_inline().next().is_some() || e.overflow_head != PageId::ZERO {
                        set.insert(e.key);
                    }
                }
            }
        }
        set
    }

    /// Byte-for-byte snapshot of every installed page, keyed by raw page
    /// id — the binary-equal reference snapshot for the determinism
    /// oracle.
    fn snapshot_all_pages(idx: &SecondaryIndex) -> BTreeMap<u64, Vec<u8>> {
        let mut out = BTreeMap::new();
        for entry in idx.page_store().pages.iter() {
            out.insert(entry.key().raw(), entry.value().read().as_ref().to_vec());
        }
        out
    }

    /// Replay an op sequence into a fresh index (lookups are no-ops on
    /// state). Used to build twice for the determinism oracle.
    fn build_from_ops(ops: &[ModelOp]) -> SecondaryIndex {
        let idx = build_index();
        for op in ops {
            match op {
                ModelOp::Insert(k, node) => {
                    idx.insert(*k, *node).expect("insert ok");
                }
                ModelOp::Remove(k, node) => {
                    idx.remove(*k, *node).expect("remove ok");
                }
                ModelOp::Lookup(_) => {}
            }
        }
        idx
    }

    // ── strategies (built on the existing `(tenant, label, property,
    //    value)` key encoders; values stay well under the U64 2^56-1
    //    encode cap so no `ValueOverflow` is generated) ──

    fn arb_pv_small() -> impl Strategy<Value = PropertyValue> {
        prop_oneof![
            (0u32..8).prop_map(PropertyValue::U32),
            (0u64..8).prop_map(PropertyValue::U64),
            (0u32..8).prop_map(|v| PropertyValue::StringId(StringId::new(v))),
        ]
    }

    fn arb_pv_medium() -> impl Strategy<Value = PropertyValue> {
        prop_oneof![
            (0u32..60).prop_map(PropertyValue::U32),
            (0u64..60).prop_map(PropertyValue::U64),
            (0u32..60).prop_map(|v| PropertyValue::StringId(StringId::new(v))),
        ]
    }

    fn arb_pv_wide() -> impl Strategy<Value = PropertyValue> {
        prop_oneof![
            (0u32..1_000_000).prop_map(PropertyValue::U32),
            (0u64..1_000_000).prop_map(PropertyValue::U64),
            (0u32..1_000_000).prop_map(|v| PropertyValue::StringId(StringId::new(v))),
        ]
    }

    fn arb_key_small() -> impl Strategy<Value = SecondaryKey> {
        (
            prop::sample::select(vec![1u64, 2]),
            prop::sample::select(vec![10u32, 11]),
            prop::sample::select(vec![100u32, 101]),
            arb_pv_small(),
        )
            .prop_map(|(t, l, p, v)| {
                SecondaryKey::new(TenantId::new(t), LabelId::new(l), StringId::new(p), v)
            })
    }

    fn arb_key_medium() -> impl Strategy<Value = SecondaryKey> {
        (
            prop::sample::select(vec![1u64, 2, 3]),
            prop::sample::select(vec![10u32, 11]),
            prop::sample::select(vec![100u32, 101]),
            arb_pv_medium(),
        )
            .prop_map(|(t, l, p, v)| {
                SecondaryKey::new(TenantId::new(t), LabelId::new(l), StringId::new(p), v)
            })
    }

    fn arb_key_wide() -> impl Strategy<Value = SecondaryKey> {
        (
            prop::sample::select(vec![1u64, 2]),
            prop::sample::select(vec![10u32, 11]),
            prop::sample::select(vec![100u32, 101]),
            arb_pv_wide(),
        )
            .prop_map(|(t, l, p, v)| {
                SecondaryKey::new(TenantId::new(t), LabelId::new(l), StringId::new(p), v)
            })
    }

    fn arb_node_small() -> impl Strategy<Value = NodeId> {
        // NodeId::ZERO is the empty-slot sentinel; never generate it.
        (1u64..=12).prop_map(NodeId::new)
    }

    fn arb_node_medium() -> impl Strategy<Value = NodeId> {
        (1u64..=30).prop_map(NodeId::new)
    }

    fn arb_op_seq_small() -> impl Strategy<Value = Vec<ModelOp>> {
        let op = prop_oneof![
            3 => (arb_key_small(), arb_node_small()).prop_map(|(k, n)| ModelOp::Insert(k, n)),
            2 => (arb_key_small(), arb_node_small()).prop_map(|(k, n)| ModelOp::Remove(k, n)),
            1 => arb_key_small().prop_map(ModelOp::Lookup),
        ];
        prop::collection::vec(op, 1..160)
    }

    fn arb_op_seq_medium() -> impl Strategy<Value = Vec<ModelOp>> {
        let op = prop_oneof![
            3 => (arb_key_medium(), arb_node_medium()).prop_map(|(k, n)| ModelOp::Insert(k, n)),
            2 => (arb_key_medium(), arb_node_medium()).prop_map(|(k, n)| ModelOp::Remove(k, n)),
            1 => arb_key_medium().prop_map(ModelOp::Lookup),
        ];
        prop::collection::vec(op, 1..256)
    }

    proptest! {
        // W28-S573 exceed-spec: 256 → 768 cases. Exhaustive enumeration
        // ([`btree_exhaustive_single_key_vs_model`] /
        // [`btree_exhaustive_two_key_vs_model`]) covers tiny domains
        // completely; this randomized model test broadens the SAMPLED
        // space (larger key domain, op sequences up to 160) where
        // exhaustive is infeasible — case count is the statistical lever.
        #![proptest_config(ProptestConfig {
            cases: 768,
            .. ProptestConfig::default()
        })]

        /// Model-equality oracle. A random insert / remove / lookup
        /// sequence over a small (collision-heavy) key domain must keep
        /// the real index in lockstep with an independent `BTreeMap`
        /// reference model after EVERY op — exact `lookup` equality
        /// (order + multiplicity), plus a periodic and final full-key
        /// sweep. This is model-equality, not a consistency / dedupe
        /// check.
        #[test]
        fn prop_btree_matches_btreemap_model(ops in arb_op_seq_small()) {
            let idx = build_index();
            let mut model: BTreeMap<SecondaryKey, KeyModel> = BTreeMap::new();

            for (i, op) in ops.iter().enumerate() {
                match op {
                    ModelOp::Insert(k, node) => {
                        idx.insert(*k, *node).expect("insert ok");
                        model.entry(*k).or_default().insert(*node);
                        prop_assert_eq!(
                            idx.lookup(*k).expect("lookup ok"),
                            model.get(k).map(KeyModel::lookup).unwrap_or_default(),
                            "post-insert mismatch at {:?}",
                            k
                        );
                    }
                    ModelOp::Remove(k, node) => {
                        let real = idx.remove(*k, *node).expect("remove ok");
                        let modeled = model.get_mut(k).is_some_and(|m| m.remove(*node));
                        prop_assert_eq!(real, modeled, "remove-flag mismatch at {:?}", k);
                        prop_assert_eq!(
                            idx.lookup(*k).expect("lookup ok"),
                            model.get(k).map(KeyModel::lookup).unwrap_or_default(),
                            "post-remove mismatch at {:?}",
                            k
                        );
                    }
                    ModelOp::Lookup(k) => {
                        prop_assert_eq!(
                            idx.lookup(*k).expect("lookup ok"),
                            model.get(k).map(KeyModel::lookup).unwrap_or_default(),
                            "lookup mismatch at {:?}",
                            k
                        );
                    }
                }

                // Periodic full sweep over every key the model has seen.
                if i % 32 == 0 {
                    for (k, m) in &model {
                        prop_assert_eq!(
                            idx.lookup(*k).expect("lookup ok"),
                            m.lookup(),
                            "periodic full-sweep mismatch at {:?}",
                            k
                        );
                    }
                }
            }

            // Final full sweep.
            for (k, m) in &model {
                prop_assert_eq!(
                    idx.lookup(*k).expect("lookup ok"),
                    m.lookup(),
                    "final full-sweep mismatch at {:?}",
                    k
                );
            }

            // And the in-order key walk stays strictly ascending.
            let walked = walk_leaf_keys(&idx);
            for w in walked.windows(2) {
                prop_assert!(w[0] < w[1], "leaf walk not ascending: {:?} !< {:?}", w[0], w[1]);
            }
        }
    }

    proptest! {
        // W28-S573 exceed-spec: 128 → 384 cases (broader sampled key sets).
        #![proptest_config(ProptestConfig { cases: 384, .. ProptestConfig::default() })]

        /// Total-order invariant — the property form of the example-based
        /// `leaf_split_preserves_total_order`. For any set of unique
        /// keys, an in-order walk yields strictly ascending keys equal to
        /// the sorted inserted set. Ordering is the derived `SecondaryKey`
        /// `Ord`; per DEC-19 the LE-encoded bytes do NOT memcmp-sort, so
        /// the tree orders on the decoded key, never the raw bytes.
        #[test]
        fn prop_key_ordering_total(keys in prop::collection::btree_set(arb_key_wide(), 1..200)) {
            let idx = build_index();
            for (i, k) in keys.iter().enumerate() {
                idx.insert(*k, NodeId::new(i as u64 + 1)).expect("insert ok");
            }
            let walked = walk_leaf_keys(&idx);
            for w in walked.windows(2) {
                prop_assert!(
                    w[0] < w[1],
                    "in-order walk not strictly ascending: {:?} !< {:?}",
                    w[0],
                    w[1]
                );
            }
            let expected: Vec<SecondaryKey> = keys.iter().copied().collect();
            prop_assert_eq!(walked, expected, "in-order walk must equal the sorted key set");
        }
    }

    proptest! {
        // W28-S573 exceed-spec: 48 → 96 cases. Each case inserts 256-360
        // keys (forces ≥2 splits), so this is the heaviest randomized
        // B-tree property; 96 is a measured 2× deepening that keeps the
        // CI debug run bounded while the adversarial cascade regressions
        // ([`btree_adversarial_ascending_insert_cascade`] /
        // [`..._descending_...`]) pin the worst-case monotonic orders.
        #![proptest_config(ProptestConfig { cases: 96, .. ProptestConfig::default() })]

        /// Split-membership invariant. ≥256 unique keys cannot fit in two
        /// 127-entry leaves, so the tree MUST grow to ≥3 leaves (≥2
        /// splits). The set of retrievable keys must then exactly equal
        /// the inserted set — no loss, no phantom — and each key resolves
        /// to its single inserted node.
        #[test]
        fn prop_split_preserves_membership(
            keys in prop::collection::btree_set(arb_key_wide(), 256..360),
        ) {
            let idx = build_index();
            let mut expected_node: BTreeMap<SecondaryKey, NodeId> = BTreeMap::new();
            for (i, k) in keys.iter().enumerate() {
                let node = NodeId::new(i as u64 + 1);
                idx.insert(*k, node).expect("insert ok");
                expected_node.insert(*k, node);
            }

            let leaf_count = walk_nodes(&idx)
                .iter()
                .filter(|node| matches!(node, LogicalNode::Leaf(_)))
                .count();
            prop_assert!(
                leaf_count >= 3,
                "expected >=3 leaves (>=2 splits), got {}",
                leaf_count
            );

            // No loss: every inserted key resolves to exactly its node.
            for (k, node) in &expected_node {
                prop_assert_eq!(
                    idx.lookup(*k).expect("lookup ok"),
                    vec![*node],
                    "lost key {:?}",
                    k
                );
            }
            // No phantom: the live key set equals the inserted key set.
            let inserted: BTreeSet<SecondaryKey> = keys.iter().copied().collect();
            prop_assert_eq!(
                live_keys_in_tree(&idx),
                inserted,
                "retrievable key set != inserted key set"
            );
        }
    }

    proptest! {
        // W28-S573 exceed-spec: 48 → 96 cases (each builds the index
        // twice + full byte-snapshot comparison).
        #![proptest_config(ProptestConfig { cases: 96, .. ProptestConfig::default() })]

        /// Build-determinism. Two indexes built from the SAME op sequence
        /// must be structurally identical. Asserted at two levels: the
        /// in-order leaf-key arrays + full `LogicalNode` structure
        /// (readable), and a byte-for-byte page-store snapshot (the
        /// binary-equal reference-snapshot oracle, strictly stronger than
        /// dedupe-consistency).
        #[test]
        fn btree_build_determinism(ops in arb_op_seq_medium()) {
            let a = build_from_ops(&ops);
            let b = build_from_ops(&ops);
            prop_assert_eq!(
                a.page_store().len(),
                b.page_store().len(),
                "page count diverged"
            );
            prop_assert_eq!(walk_leaf_keys(&a), walk_leaf_keys(&b), "in-order leaf keys diverged");
            prop_assert_eq!(walk_nodes(&a), walk_nodes(&b), "tree structure diverged");
            prop_assert_eq!(
                snapshot_all_pages(&a),
                snapshot_all_pages(&b),
                "page bytes diverged (build is non-deterministic)"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // W28-S573 (exceed-spec; Feature #570; gap analysis PR #510 §3;
    // ADR-165 M1/M2): deepen the B-tree gate to the FULL invariant space.
    //
    // S1 (#511) shipped the FIRST B-tree model-oracle proptests, but at
    // SAMPLED case counts over randomized op sequences. The
    // EXCEED-THE-SPEC mandate (`ENGINEERING_DOCTRINE` §3) requires the
    // test gate to be a STRICT SUPERSET of the spec: exhaustive where
    // feasible, adversarial elsewhere — "representative is a RED FLAG".
    // This block adds:
    //
    //  1. EXHAUSTIVE small-domain enumeration — every op sequence over a
    //     tiny key/node domain (NOT random sampling), each checked
    //     against the same `KeyModel` / `BTreeMap` oracle with EXACT
    //     equality (order + multiplicity).
    //  2. Adversarial insert-order corpora (ascending / descending /
    //     duplicate-heavy / boundary + cross-variant / multi-leaf split
    //     cascade) as named regression cases + an order-independence
    //     proptest.
    //
    // Oracle discipline is unchanged from S1: model EQUALITY, never
    // consistency / dedupe (`feedback_review_oracle_relaxations`). The
    // exhaustive tests assert the real index agrees with the model after
    // EVERY op; because the op space is prefix-closed (every length-≤L
    // sequence is a prefix of some enumerated length-L sequence, and is
    // therefore checked at that prefix point), the entire space of
    // sequences of length ≤ L is covered — not just terminal states.

    // ── 1. Exhaustive small-domain enumeration ───────────────────────

    /// **Exhaustive single-key op-sequence enumeration vs the model.**
    ///
    /// One fixed key; state-mutating alphabet
    /// `{Insert(n1), Insert(n2), Remove(n1), Remove(n2)}` (4 symbols).
    /// Lookups are state-INVARIANT reads, so the enumeration ranges over
    /// the mutating alphabet and the lookup oracle is applied after
    /// EVERY op — covering all length-≤7 prefix-states (prefix-closed).
    ///
    /// Feasibility (EXACT, not sampled): `4^7 = 16 384` length-7
    /// sequences ⇒ `Σ_{L=1}^{7} 4^L = 21 844` distinct prefix-states
    /// checked. Depth 7 reaches 7 consecutive inserts → 4 inline slots +
    /// an overflow chain of depth 3, so the inline→overflow transition,
    /// inline slot back-fill after a remove, and append-only overflow
    /// (mid-chain tombstones never reused) are ALL inside the enumerated
    /// space. Two node values suffice: `insert` is value-blind
    /// (first-empty-slot), and two distinct values still exercise the
    /// "remove drains EVERY matching slot" multiplicity branch.
    #[test]
    fn btree_exhaustive_single_key_vs_model() {
        const LEN: u32 = 7;
        let key = k_u32(1, 10, 100, 5);
        let n1 = n(1);
        let n2 = n(2);
        let total = 4usize.pow(LEN);
        for seq in 0..total {
            let idx = build_index();
            let mut model = KeyModel::default();
            let mut code = seq;
            for step in 0..LEN {
                // 2 bits per symbol (alphabet = 4).
                let sym = code & 0b11;
                code >>= 2;
                match sym {
                    0 => {
                        idx.insert(key, n1).expect("insert n1");
                        model.insert(n1);
                    }
                    1 => {
                        idx.insert(key, n2).expect("insert n2");
                        model.insert(n2);
                    }
                    2 => {
                        let real = idx.remove(key, n1).expect("remove n1");
                        let modeled = model.remove(n1);
                        assert_eq!(
                            real, modeled,
                            "remove-flag mismatch (seq={seq}, step={step}, Rm n1)"
                        );
                    }
                    3 => {
                        let real = idx.remove(key, n2).expect("remove n2");
                        let modeled = model.remove(n2);
                        assert_eq!(
                            real, modeled,
                            "remove-flag mismatch (seq={seq}, step={step}, Rm n2)"
                        );
                    }
                    _ => unreachable!("2-bit symbol is in 0..4"),
                }
                // Lookup oracle after EVERY op: exact order + multiplicity.
                assert_eq!(
                    idx.lookup(key).expect("lookup"),
                    model.lookup(),
                    "lookup mismatch (seq={seq}, step={step})"
                );
            }
        }
    }

    /// **Exhaustive two-key op-sequence enumeration vs the model.**
    ///
    /// Two ordered keys `kA < kB`; alphabet
    /// `{Ins,Rm} × {kA,kB} × {n1,n2}` (8 symbols). Verifies the FULL
    /// `BTreeMap<SecondaryKey, KeyModel>` oracle (per-key exact lookup
    /// equality on BOTH keys) AND the cross-key invariant (in-order leaf
    /// walk strictly ascending) after EVERY op. Prefix-closed ⇒
    /// exhaustive over ALL sequences of length ≤ 5.
    ///
    /// Feasibility (EXACT): `8^5 = 32 768` length-5 sequences ⇒
    /// `Σ_{L=1}^{5} 8^L = 37 448` distinct prefix-states. Single-key
    /// OVERFLOW depth is exhausted by
    /// [`btree_exhaustive_single_key_vs_model`]; this test's job is the
    /// cross-key ordering + multi-entry oracle that the single-key
    /// enumeration cannot reach.
    #[test]
    fn btree_exhaustive_two_key_vs_model() {
        const LEN: u32 = 5;
        let ka = k_u32(1, 10, 100, 1);
        let kb = k_u32(1, 10, 100, 2);
        assert!(ka < kb, "kA must order before kB");
        let keys = [ka, kb];
        let nodes = [n(1), n(2)];
        let total = 8usize.pow(LEN);
        for seq in 0..total {
            let idx = build_index();
            let mut model: BTreeMap<SecondaryKey, KeyModel> = BTreeMap::new();
            let mut code = seq;
            for step in 0..LEN {
                // 3 bits per symbol (alphabet = 8): bit0 op, bit1 key, bit2 node.
                let sym = code & 0b111;
                code >>= 3;
                let key = keys[(sym >> 1) & 1];
                let node = nodes[(sym >> 2) & 1];
                if sym & 1 == 1 {
                    let real = idx.remove(key, node).expect("remove");
                    let modeled = model.get_mut(&key).is_some_and(|m| m.remove(node));
                    assert_eq!(
                        real, modeled,
                        "remove-flag mismatch (seq={seq}, step={step})"
                    );
                } else {
                    idx.insert(key, node).expect("insert");
                    model.entry(key).or_default().insert(node);
                }
                // Per-key lookup oracle for BOTH keys.
                for k in &keys {
                    assert_eq!(
                        idx.lookup(*k).expect("lookup"),
                        model.get(k).map(KeyModel::lookup).unwrap_or_default(),
                        "lookup mismatch at {k:?} (seq={seq}, step={step})"
                    );
                }
                // Cross-key invariant: in-order leaf walk strictly ascending.
                let walked = walk_leaf_keys(&idx);
                for w in walked.windows(2) {
                    assert!(
                        w[0] < w[1],
                        "leaf walk not ascending (seq={seq}, step={step}): {:?} !< {:?}",
                        w[0],
                        w[1]
                    );
                }
            }
        }
    }

    // ── 2. Adversarial corpora ────────────────────────────────────────

    /// Multi-leaf split cascade under a strictly ASCENDING insert order
    /// — the pathological case that always splits the rightmost leaf.
    #[test]
    fn btree_adversarial_ascending_insert_cascade() {
        adversarial_monotonic_cascade(false);
    }

    /// Multi-leaf split cascade under a strictly DESCENDING insert order
    /// — always splits the leftmost leaf / first-child edge.
    #[test]
    fn btree_adversarial_descending_insert_cascade() {
        adversarial_monotonic_cascade(true);
    }

    /// 1 000 distinct keys inserted monotonically force
    /// `⌈1000/127⌉ ≥ 8` leaves under an internal root. Membership +
    /// per-key node mapping + ascending walk must survive the cascade
    /// regardless of which edge keeps splitting.
    fn adversarial_monotonic_cascade(descending: bool) {
        const N: u32 = 1_000;
        let idx = build_index();
        let mut model: BTreeMap<SecondaryKey, NodeId> = BTreeMap::new();
        let order: Vec<u32> = if descending {
            (0..N).rev().collect()
        } else {
            (0..N).collect()
        };
        for (i, &v) in order.iter().enumerate() {
            let key = k_u32(1, 10, 100, v);
            let node = n(i as u64 + 1);
            idx.insert(key, node).expect("insert");
            model.insert(key, node);
        }

        // The tree MUST have grown past a single leaf into an internal root.
        let root_id = idx.root().expect("root");
        let latch = idx.page_store().latch(root_id).expect("root latch");
        let g = latch.read();
        let header = read_header(g.as_ref()).expect("root header");
        assert_eq!(
            header.page_type,
            PageType::IndexInternal.as_byte(),
            "1000 keys must force an internal root (descending={descending})"
        );
        drop(g);

        let leaf_count = walk_nodes(&idx)
            .iter()
            .filter(|node| matches!(node, LogicalNode::Leaf(_)))
            .count();
        assert!(
            leaf_count >= 8,
            "expected >= 8 leaves after 1000-key cascade, got {leaf_count} (descending={descending})"
        );

        // No loss / exact node mapping.
        for (k, node) in &model {
            assert_eq!(
                idx.lookup(*k).expect("lookup"),
                vec![*node],
                "lost key {k:?} (descending={descending})"
            );
        }
        // No phantom: retrievable set == inserted set.
        let inserted: BTreeSet<SecondaryKey> = model.keys().copied().collect();
        assert_eq!(
            live_keys_in_tree(&idx),
            inserted,
            "retrievable set != inserted set (descending={descending})"
        );
        // Ascending walk regardless of insert order.
        let walked = walk_leaf_keys(&idx);
        let expected: Vec<SecondaryKey> = inserted.iter().copied().collect();
        assert_eq!(
            walked, expected,
            "in-order walk must equal sorted key set (descending={descending})"
        );
    }

    /// Duplicate-heavy single key forcing a MULTI-PAGE overflow chain
    /// (3 overflow pages). `4 inline + 2·OVERFLOW_SLOTS_PER_PAGE + 5 =
    /// 2 043` duplicate NodeIds. `lookup` must return ALL of them in
    /// insertion order (inline slots 0..3 then overflow append order),
    /// and the page store must hold exactly root-leaf + 3 overflow
    /// pages. Checked against `KeyModel` for exact order + multiplicity.
    #[test]
    fn btree_adversarial_duplicate_heavy_overflow_chain() {
        let idx = build_index();
        let mut model = KeyModel::default();
        let k = k_u32(1, 10, 100, 5);
        let total = (4 + 2 * OVERFLOW_SLOTS_PER_PAGE + 5) as u64; // 2 043
        for i in 1..=total {
            idx.insert(k, n(i)).expect("insert");
            model.insert(n(i));
        }
        let hits = idx.lookup(k).expect("lookup");
        assert_eq!(hits.len(), total as usize, "all duplicates retrievable");
        assert_eq!(
            hits,
            model.lookup(),
            "exact order + multiplicity vs KeyModel"
        );
        assert_eq!(
            hits,
            (1..=total).map(n).collect::<Vec<_>>(),
            "duplicates preserve insertion order across the overflow chain"
        );
        // root leaf (1) + 3 overflow pages = 4.
        assert_eq!(
            idx.page_store().len(),
            4,
            "expected root-leaf + 3 overflow pages for {total} duplicates"
        );
    }

    /// Boundary + cross-variant `PropertyValue` keys. Extreme encodable
    /// values (U64 max-encodable `2^56-1`, `U32::MAX`, `StringId`
    /// `u32::MAX`, and the zeros) across all three variants, inserted in
    /// REVERSE of their canonical `Ord` so the tree must re-sort them.
    /// The in-order leaf walk must equal the `BTreeSet`-canonical order
    /// (derived `SecondaryKey` Ord — DEC-19: decoded key, not memcmp of
    /// LE bytes), and every key resolves to its node.
    #[test]
    fn btree_adversarial_boundary_and_cross_variant_keys() {
        let max_u64 = (1u64 << 56) - 1; // maximum encodable U64
        let values = vec![
            PropertyValue::U32(0),
            PropertyValue::U32(1),
            PropertyValue::U32(u32::MAX),
            PropertyValue::U64(0),
            PropertyValue::U64(1),
            PropertyValue::U64(max_u64),
            PropertyValue::StringId(StringId::new(0)),
            PropertyValue::StringId(StringId::new(u32::MAX)),
        ];
        let keys: Vec<SecondaryKey> = values
            .into_iter()
            .map(|v| SecondaryKey::new(TenantId::new(1), LabelId::new(10), StringId::new(100), v))
            .collect();

        let idx = build_index();
        let mut model: BTreeMap<SecondaryKey, NodeId> = BTreeMap::new();
        // Insert in REVERSE canonical order to force the tree to re-sort
        // every boundary value.
        let mut sorted = keys.clone();
        sorted.sort();
        for (i, k) in sorted.iter().rev().enumerate() {
            let node = n(i as u64 + 1);
            idx.insert(*k, node).expect("insert boundary key");
            model.insert(*k, node);
        }

        let walked = walk_leaf_keys(&idx);
        let expected: Vec<SecondaryKey> = model.keys().copied().collect();
        assert_eq!(
            walked, expected,
            "boundary / cross-variant keys must sort by decoded Ord"
        );
        for (k, node) in &model {
            assert_eq!(
                idx.lookup(*k).expect("lookup"),
                vec![*node],
                "boundary key {k:?} lost"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 192, .. ProptestConfig::default() })]

        /// Insert-order independence of retrievable CONTENT. The final
        /// retrievable key set + per-key node mapping + ascending walk
        /// must be IDENTICAL regardless of insert order. Each adversarial
        /// permutation class is compared against the same
        /// order-independent `BTreeMap` model. (The B-tree SHAPE may
        /// legitimately differ by insert order — split points depend on
        /// arrival order — but retrievable CONTENT may not.)
        ///
        /// `order`: 0 ascending, 1 descending, 2 evens-then-odds,
        /// 3 odds-then-evens — all deterministic, no rng dependency.
        #[test]
        fn prop_btree_insert_order_independence(
            keys in prop::collection::btree_set(arb_key_wide(), 1..200),
            order in 0u8..4,
        ) {
            let sorted: Vec<SecondaryKey> = keys.iter().copied().collect();
            // Node assignment keyed by canonical position (order-independent).
            let mut model: BTreeMap<SecondaryKey, NodeId> = BTreeMap::new();
            for (i, k) in sorted.iter().enumerate() {
                model.insert(*k, n(i as u64 + 1));
            }

            // Build the adversarial insert order.
            let mut indices: Vec<usize> = (0..sorted.len()).collect();
            match order {
                0 => {}
                1 => indices.reverse(),
                2 => indices.sort_by_key(|&i| (i % 2, i)),
                3 => indices.sort_by_key(|&i| (1 - (i % 2), i)),
                _ => unreachable!("order in 0..4"),
            }

            let idx = build_index();
            for &i in &indices {
                let k = sorted[i];
                idx.insert(k, *model.get(&k).expect("modeled node")).expect("insert");
            }

            // Content is order-independent: matches the model exactly.
            for (k, node) in &model {
                prop_assert_eq!(idx.lookup(*k).expect("lookup"), vec![*node], "lost {:?}", k);
            }
            let inserted: BTreeSet<SecondaryKey> = model.keys().copied().collect();
            prop_assert_eq!(
                live_keys_in_tree(&idx),
                inserted.clone(),
                "retrievable set drifted under order {}",
                order
            );
            let walked = walk_leaf_keys(&idx);
            let expected: Vec<SecondaryKey> = inserted.iter().copied().collect();
            prop_assert_eq!(walked, expected, "ascending walk drifted under order {}", order);
        }
    }
}
