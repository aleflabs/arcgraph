//! Primary B-tree index `(TenantId, Kind, Id) → (PageId, SlotId)` (M2-33).
//!
//! Maps every committed node/rel record's logical identity to the
//! `(PageId, SlotId)` coordinate that holds its authoritative bytes
//! in the slotted-page store (`records::SlottedPage`). The index is a
//! **read accelerator**, not a visibility authority; on a miss, callers
//! fall through to the MVCC chain. See `docs/adr/ADR-023-primary-index-
//! read-accelerator.md`.
//!
//! # Layout
//!
//! One shared B+tree across all tenants (DEC-8). Keys sort
//! tenant-major, then kind, then id.
//!
//! Leaf page (`PageType::IndexLeaf`):
//!
//! ```text
//!  0.. 40  PageHeader           (magic, version, page_type=5,
//!                                flags, page_id, lsn,
//!                                tenant_id=TenantId::SYSTEM,
//!                                checksum=0, slot_count=N,
//!                                free_space=0)
//! 40.. 40 + 40 * N             N leaf entries, each:
//!                                  key: 24 B (tenant_u64 |
//!                                             kind_u8 + 7 B pad |
//!                                             id_u64)
//!                                  value: 16 B (page_u64 |
//!                                               slot_u16 |
//!                                               tombstone_u8 |
//!                                               5 B pad)
//! ```
//!
//! Internal page (`PageType::IndexInternal`):
//!
//! ```text
//!  0.. 40  PageHeader
//! 40.. 48  first_child: u64     (PageId of child subtree strictly
//!                                 less than key_0)
//! 48..     N × (key: 24 B | child: u64)
//!                                 slot_count stores N.
//! ```
//!
//! Capacity (DEC-9): leaf = 203 entries, internal = 255 children. Three
//! levels of internal + one leaf level supports 255 × 255 × 203 ≈
//! 13.2 M keys, comfortably above the 10 M M2.d exit criterion.
//!
//! # Concurrency (§F, DEC-17)
//!
//! Pages live in an internal
//! `DashMap<PageId, Arc<parking_lot::RwLock<Box<PageBuf>>>>` —
//! identical pattern to [`crate::blob::BlobStore`]. Crabbing is
//! top-down RwLock coupling; write path holds parent latches until
//! child is known "safe" (has room for another entry).
//!
//! # WAL (DEC-11)
//!
//! Every page mutation emits a [`crate::wal::WalRecordType::IndexPage`]
//! record carrying the page-id and the full post-write page bytes. WAL
//! replay is M2.e; emission is wired in the same commit as the WAL
//! variant definition.
//!
//! # Delete policy (DEC-12)
//!
//! `remove` sets the `tombstone_u8` byte on the leaf entry value.
//! `lookup` skips tombstoned entries. Inserts collide only with live
//! keys, never with tombstones.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{
    ArcGraphError, Lsn, PageHeader, PageId, PageType, Result as CoreResult, TenantId,
};
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::{Mutex, RawRwLock, RwLock};
use thiserror::Error;

use crate::mutation_log::{IndexHandle, PageStoreKind, TxnMutationLog};
use crate::page_alloc::PageAllocator;
use crate::records::SlotId;
use crate::transaction::TxnManager;
use crate::wal::bundle::{SideChannelWrite, StagedEmit};
use crate::wal::{WalHandle, WalRecordType};

// ─────────────────────────────────────────────────────────────────────
// Public key / value types
// ─────────────────────────────────────────────────────────────────────

/// Kind discriminator in the primary-index key. Nodes and rels share a
/// single tree but never share keys because this byte differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum RecordKind {
    /// Node record.
    Node = 0,
    /// Rel record.
    Rel = 1,
}

impl RecordKind {
    #[inline]
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    fn from_byte(b: u8) -> Result<Self, IndexError> {
        match b {
            0 => Ok(Self::Node),
            1 => Ok(Self::Rel),
            other => Err(IndexError::CorruptPage {
                page_id: PageId::ZERO,
                reason: format!("unknown record_kind byte: {other}"),
            }),
        }
    }
}

/// Primary-index key: tenant-major, then kind, then id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrimaryKey {
    /// Tenant that owns the record.
    pub tenant: TenantId,
    /// Node vs Rel.
    pub kind: RecordKind,
    /// Raw logical id (`NodeId::raw()` or `RelId::raw()`).
    pub id: u64,
}

impl PrimaryKey {
    /// On-disk key size.
    pub const SIZE: usize = 24;

    /// Convenience constructor.
    #[inline]
    #[must_use]
    pub const fn new(tenant: TenantId, kind: RecordKind, id: u64) -> Self {
        Self { tenant, kind, id }
    }

    fn encode_into(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), Self::SIZE);
        out[0..8].copy_from_slice(&self.tenant.raw().to_le_bytes());
        out[8] = self.kind.as_byte();
        out[9..16].fill(0);
        out[16..24].copy_from_slice(&self.id.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, IndexError> {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        let tenant = TenantId::new(u64::from_le_bytes(
            bytes[0..8].try_into().expect("slice size asserted above"),
        ));
        let kind = RecordKind::from_byte(bytes[8])?;
        let id = u64::from_le_bytes(bytes[16..24].try_into().expect("slice size asserted above"));
        Ok(Self { tenant, kind, id })
    }
}

/// Primary-index leaf value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageSlot {
    /// Page holding the authoritative record bytes.
    pub page: PageId,
    /// Slot within that page (see `records::SlotId`).
    pub slot: SlotId,
}

impl PageSlot {
    /// On-disk value size.
    pub const SIZE: usize = 16;

    /// Convenience constructor.
    #[inline]
    #[must_use]
    pub const fn new(page: PageId, slot: SlotId) -> Self {
        Self { page, slot }
    }

    fn encode_into(&self, out: &mut [u8], tombstone: bool) {
        debug_assert_eq!(out.len(), Self::SIZE);
        out[0..8].copy_from_slice(&self.page.raw().to_le_bytes());
        out[8..10].copy_from_slice(&self.slot.raw().to_le_bytes());
        out[10] = u8::from(tombstone);
        out[11..16].fill(0);
    }

    fn decode(bytes: &[u8]) -> (Self, bool) {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        let page = PageId::new(u64::from_le_bytes(
            bytes[0..8].try_into().expect("slice size asserted above"),
        ));
        let slot = SlotId(u16::from_le_bytes(
            bytes[8..10].try_into().expect("slice size asserted above"),
        ));
        let tombstone = bytes[10] != 0;
        (Self { page, slot }, tombstone)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Primary-index local error surface. Converted into `ArcGraphError`
/// at the `crud.rs` boundary where needed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexError {
    /// Page with this id is not tracked by the index.
    #[error("primary index: page {0:?} not mapped")]
    MissingPage(PageId),
    /// Page header or body did not match the expected invariants.
    #[error("primary index: page {page_id:?} corrupt: {reason}")]
    CorruptPage {
        /// Offending page id.
        page_id: PageId,
        /// Human-readable reason.
        reason: String,
    },
    /// Attempted to insert a duplicate live key.
    #[error("primary index: key {key:?} already present")]
    DuplicateKey {
        /// The conflicting key.
        key: PrimaryKey,
    },
    /// Core layer rejected a page header (bubbled up from `PageHeader::from_bytes`).
    #[error(transparent)]
    Core(#[from] ArcGraphError),
}

// ─────────────────────────────────────────────────────────────────────
// Page-level constants
// ─────────────────────────────────────────────────────────────────────

/// Bytes in a leaf entry (`PrimaryKey` + `PageSlot`).
pub const LEAF_ENTRY_SIZE: usize = PrimaryKey::SIZE + PageSlot::SIZE;
const _: () = assert!(LEAF_ENTRY_SIZE == 40);

/// Bytes in an internal entry (`PrimaryKey` + child `PageId`).
pub const INTERNAL_ENTRY_SIZE: usize = PrimaryKey::SIZE + 8;
const _: () = assert!(INTERNAL_ENTRY_SIZE == 32);

/// Offset of the first entry on a leaf page (immediately after the header).
pub const LEAF_ENTRY_OFFSET: usize = PageHeader::SIZE;

/// Offset of `first_child` on an internal page.
pub const INTERNAL_FIRST_CHILD_OFFSET: usize = PageHeader::SIZE;
/// Offset of the first `(key, child)` entry on an internal page.
pub const INTERNAL_ENTRY_OFFSET: usize = PageHeader::SIZE + 8;

/// Maximum leaf entries per page (DEC-9: 203).
pub const LEAF_CAPACITY: u16 = ((PAGE_SIZE - LEAF_ENTRY_OFFSET) / LEAF_ENTRY_SIZE) as u16;
const _: () = assert!(LEAF_CAPACITY == 203);

/// Maximum internal entries (`(key, child)` pairs) per page.
/// Internal fanout = `INTERNAL_CAPACITY + 1` children.
pub const INTERNAL_CAPACITY: u16 =
    ((PAGE_SIZE - INTERNAL_ENTRY_OFFSET) / INTERNAL_ENTRY_SIZE) as u16;
const _: () = assert!(INTERNAL_CAPACITY == 254);

// ─────────────────────────────────────────────────────────────────────
// Raw page buffer type
// ─────────────────────────────────────────────────────────────────────

type PageBuf = [u8; PAGE_SIZE];

/// Allocate a zero-initialized page buffer with the given header stamped
/// at offset 0. Used by `PrimaryPageStore::install_fresh` and by later
/// commits that materialize freshly allocated leaf / internal pages.
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
// Leaf page codec
// ─────────────────────────────────────────────────────────────────────

/// Read-only accessor for leaf-page contents.
pub struct LeafPageRef<'a> {
    bytes: &'a PageBuf,
    header: PageHeader,
}

impl<'a> LeafPageRef<'a> {
    /// Open a leaf view, validating the header.
    pub fn open(bytes: &'a PageBuf) -> Result<Self, IndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexLeaf.as_byte() {
            return Err(IndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexLeaf page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    /// Number of live+tombstoned entries.
    #[must_use]
    pub fn entry_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Page id stored in the header.
    #[must_use]
    pub fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Decode the entry at `index` into `(key, value, tombstone)`.
    pub fn entry(&self, index: u16) -> Result<(PrimaryKey, PageSlot, bool), IndexError> {
        let n = self.entry_count();
        if index >= n {
            return Err(IndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("entry index {index} out of range (count={n})"),
            });
        }
        let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE;
        let key = PrimaryKey::decode(&self.bytes[off..off + PrimaryKey::SIZE])?;
        let (value, tombstone) =
            PageSlot::decode(&self.bytes[off + PrimaryKey::SIZE..off + LEAF_ENTRY_SIZE]);
        Ok((key, value, tombstone))
    }

    /// Point lookup: find the live `PageSlot` for `key`, or `None`.
    pub fn lookup(&self, key: PrimaryKey) -> Result<Option<PageSlot>, IndexError> {
        let idx = self.binary_search(key)?;
        match idx {
            LeafFindResult::Found { index, tombstoned } => {
                if tombstoned {
                    Ok(None)
                } else {
                    let (_, value, _) = self.entry(index)?;
                    Ok(Some(value))
                }
            }
            LeafFindResult::Absent { .. } => Ok(None),
        }
    }

    /// Returns `true` iff `key` is present on the page as either a
    /// live entry or a tombstoned slot. Useful to the tree-level
    /// mutation logic: on a full leaf, `upsert` / `insert` with a
    /// pre-existing slot is in-place and does NOT require a split;
    /// only a brand-new-key insertion triggers one.
    pub fn contains_including_tombstoned(&self, key: PrimaryKey) -> Result<bool, IndexError> {
        match self.binary_search(key)? {
            LeafFindResult::Found { .. } => Ok(true),
            LeafFindResult::Absent { .. } => Ok(false),
        }
    }

    fn binary_search(&self, needle: PrimaryKey) -> Result<LeafFindResult, IndexError> {
        let n = self.entry_count();
        let mut lo: u32 = 0;
        let mut hi: u32 = u32::from(n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k, _, tombstoned) = self.entry(mid as u16)?;
            match k.cmp(&needle) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    return Ok(LeafFindResult::Found {
                        index: mid as u16,
                        tombstoned,
                    });
                }
            }
        }
        Ok(LeafFindResult::Absent {
            insert_at: lo as u16,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum LeafFindResult {
    Found { index: u16, tombstoned: bool },
    Absent { insert_at: u16 },
}

/// Mutable leaf-page accessor. Holds no page latches itself —
/// callers are responsible for ensuring unique access to `bytes`.
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

    /// Open an existing leaf page for in-place mutation.
    pub fn open(bytes: &'a mut PageBuf) -> Result<Self, IndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexLeaf.as_byte() {
            return Err(IndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexLeaf page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Read-only view of this page.
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

    /// Is the page at leaf capacity?
    pub fn is_full(&self) -> bool {
        self.entry_count() >= LEAF_CAPACITY
    }

    /// Insert `(key, value)` in sort-order.
    ///
    /// Behaviour:
    /// - If `key` already exists **live**, returns [`IndexError::DuplicateKey`].
    /// - If `key` already exists **tombstoned**, the slot is revived
    ///   in place and the old value overwritten — no shift, no change
    ///   to `entry_count`.
    /// - If `key` is absent and the page has room, inserts at the sorted
    ///   position, shifting following entries right.
    /// - If the page is full (no room for a new entry), returns
    ///   [`IndexError::CorruptPage`] — callers must split before
    ///   calling `insert`; pre-split logic lives in [`PrimaryIndex`].
    pub fn insert(&mut self, key: PrimaryKey, value: PageSlot) -> Result<(), IndexError> {
        match self.as_ref().binary_search(key)? {
            LeafFindResult::Found {
                index,
                tombstoned: true,
            } => {
                let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE + PrimaryKey::SIZE;
                value.encode_into(&mut self.bytes[off..off + PageSlot::SIZE], false);
                Ok(())
            }
            LeafFindResult::Found {
                tombstoned: false, ..
            } => Err(IndexError::DuplicateKey { key }),
            LeafFindResult::Absent { insert_at } => self.insert_at(insert_at, key, value),
        }
    }

    /// Upsert: same as `insert` but a duplicate-live key is replaced
    /// in place and the old value returned to the caller.
    pub fn upsert(
        &mut self,
        key: PrimaryKey,
        value: PageSlot,
    ) -> Result<Option<PageSlot>, IndexError> {
        match self.as_ref().binary_search(key)? {
            LeafFindResult::Found { index, tombstoned } => {
                let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE + PrimaryKey::SIZE;
                let (old, _was_tombstoned) =
                    PageSlot::decode(&self.bytes[off..off + PageSlot::SIZE]);
                value.encode_into(&mut self.bytes[off..off + PageSlot::SIZE], false);
                Ok(if tombstoned { None } else { Some(old) })
            }
            LeafFindResult::Absent { insert_at } => {
                self.insert_at(insert_at, key, value)?;
                Ok(None)
            }
        }
    }

    fn insert_at(
        &mut self,
        position: u16,
        key: PrimaryKey,
        value: PageSlot,
    ) -> Result<(), IndexError> {
        if self.is_full() {
            return Err(IndexError::CorruptPage {
                page_id: self.page_id(),
                reason: "leaf insert into full page (split required before insert)".into(),
            });
        }
        let n = self.entry_count();
        let end = LEAF_ENTRY_OFFSET + (n as usize) * LEAF_ENTRY_SIZE;
        let pos_off = LEAF_ENTRY_OFFSET + (position as usize) * LEAF_ENTRY_SIZE;
        // Shift the tail right by one entry to make room.
        if (position as usize) < (n as usize) {
            self.bytes
                .copy_within(pos_off..end, pos_off + LEAF_ENTRY_SIZE);
        }
        key.encode_into(&mut self.bytes[pos_off..pos_off + PrimaryKey::SIZE]);
        value.encode_into(
            &mut self.bytes[pos_off + PrimaryKey::SIZE..pos_off + LEAF_ENTRY_SIZE],
            false,
        );
        self.header.slot_count = n + 1;
        write_header(self.bytes, &self.header);
        Ok(())
    }

    /// Tombstone `key` if present-and-live. Returns the previous value,
    /// or `None` if absent / already tombstoned.
    pub fn remove(&mut self, key: PrimaryKey) -> Result<Option<PageSlot>, IndexError> {
        match self.as_ref().binary_search(key)? {
            LeafFindResult::Found {
                index,
                tombstoned: false,
            } => {
                let off = LEAF_ENTRY_OFFSET + (index as usize) * LEAF_ENTRY_SIZE + PrimaryKey::SIZE;
                let (old, _) = PageSlot::decode(&self.bytes[off..off + PageSlot::SIZE]);
                old.encode_into(&mut self.bytes[off..off + PageSlot::SIZE], true);
                Ok(Some(old))
            }
            _ => Ok(None),
        }
    }

    /// Raw bytes of the underlying page. Used by the tree-level
    /// mutation code to emit WAL records without re-acquiring the
    /// page latch (the caller already holds the write guard that
    /// produced `self`, and parking_lot's `RwLock` is not re-entrant).
    #[must_use]
    pub fn page_bytes(&self) -> &PageBuf {
        &*self.bytes
    }

    /// Split the leaf: move the upper half of entries (indices `[N/2,
    /// N)`) into a freshly allocated page under `new_page_id`, trim
    /// this page to the lower half, and return the promoted key
    /// (= first key of the new right page). Tombstoned entries are
    /// carried over unchanged so their key-ordering remains monotone.
    ///
    /// Callers own the returned [`Box<PageBuf>`] and typically install
    /// it into the page store before inserting the promoted key into
    /// the parent.
    pub fn split_into(
        &mut self,
        new_page_id: PageId,
    ) -> Result<(Box<PageBuf>, PrimaryKey), IndexError> {
        let n = self.entry_count();
        if n < 2 {
            return Err(IndexError::CorruptPage {
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

        // Trim self to the lower half and zero the freed tail so stale
        // bytes never surface through a codec bug.
        self.header.slot_count = split_at;
        write_header(self.bytes, &self.header);
        for b in &mut self.bytes[src_off..src_end] {
            *b = 0;
        }

        let promoted_key =
            PrimaryKey::decode(&new_buf[LEAF_ENTRY_OFFSET..LEAF_ENTRY_OFFSET + PrimaryKey::SIZE])?;
        Ok((new_buf, promoted_key))
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory page store (DEC-17)
// ─────────────────────────────────────────────────────────────────────

/// Per-page latch. Callers take the appropriate read / write guard
/// directly on the returned `Arc<RwLock<...>>`; crabbing holds multiple
/// latches simultaneously.
pub type PageLatch = Arc<RwLock<Box<PageBuf>>>;

/// In-memory page store for the primary index. Alpha-only; the
/// permanent home (BufferPool-backed) lands with M2.e WAL replay (DEC-17).
#[derive(Default)]
pub struct PrimaryPageStore {
    pages: DashMap<PageId, PageLatch>,
}

impl PrimaryPageStore {
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

    /// Snapshot every tracked `(PageId, latch)` pair into a `Vec`.
    /// Iteration order is arbitrary. Used by the ADR-229 checkpoint
    /// producer to capture the full page-image set of this store into
    /// the durable checkpoint snapshot (symmetric with
    /// [`crate::record_store::RecordPageStore::iter_pages`]).
    #[must_use]
    pub fn iter_pages(&self) -> Vec<(PageId, PageLatch)> {
        self.pages
            .iter()
            .map(|e| (*e.key(), Arc::clone(e.value())))
            .collect()
    }

    /// SVC-1 / #849 / ADR-229 REQ-2 — NON-FAULTING resident-page iterator
    /// for the checkpoint producer. Returns `(resident latches,
    /// evicted-page-ids)` WITHOUT ever reading from disk. The checkpoint
    /// producer holds the commit-freeze WRITE guard while it copies the
    /// resident page bytes (in-RAM only) and records the evicted ids, then
    /// reads the evicted pages' durable disk images AFTER releasing the
    /// guard — so a periodic checkpoint NEVER blocks foreground commits on
    /// synchronous disk fault-in (the availability regression the ULTRACODE
    /// re-verify flagged at 10M with an evicting buffer pool).
    ///
    /// `PrimaryPageStore` is a pure in-memory `DashMap` (no eviction), so
    /// EVERY page is resident and the evicted list is ALWAYS empty — the
    /// freeze is trivially fault-free. The `(resident, evicted)` shape is
    /// the seam the ADR-140 `BufferedRecordPageStore` wiring plugs into
    /// (skip+record evicted instead of `fault_in`-faulting them).
    #[must_use]
    pub fn iter_pages_resident_only(&self) -> (Vec<(PageId, PageLatch)>, Vec<PageId>) {
        // Pure DashMap: all pages resident, nothing evicted, no disk read.
        (self.iter_pages(), Vec::new())
    }

    /// #1404 M0.x FIX-B — the STREAMING resident-page capture: emit each
    /// resident page `(PageId, &PageLatch)` through `f` ONE at a time, so the
    /// checkpoint producer never pre-collects a whole `Vec` of latches (or the
    /// per-page byte copies it reads through them) under the freeze. The
    /// callback takes the latch read guard + copies the page bytes into the
    /// sink, then the borrow ends before the next page — ≤ one page-image
    /// transient. Returns the evicted page-ids (always empty for this pure
    /// DashMap; the seam the ADR-140 buffered store plugs into). Wire-emitted
    /// bytes are byte-identical to the whole-`Vec` path (same iteration + per-
    /// page encode).
    pub fn for_each_resident_page<F, E>(&self, mut f: F) -> std::result::Result<Vec<PageId>, E>
    where
        F: FnMut(PageId, &PageLatch) -> std::result::Result<(), E>,
    {
        for e in self.pages.iter() {
            f(*e.key(), e.value())?;
        }
        Ok(Vec::new())
    }

    /// #1404 M0.x FIX-B — resident page count for the streaming checkpoint
    /// count header (`for_each_resident_page` streams exactly this many under
    /// the freeze). Cheap DashMap length read.
    #[must_use]
    pub fn resident_page_count(&self) -> usize {
        self.pages.len()
    }

    /// Clone the latch for `page_id`. Callers own the returned `Arc` for
    /// the duration of their crabbing traversal and take read / write
    /// guards on it directly.
    pub fn latch(&self, page_id: PageId) -> Result<PageLatch, IndexError> {
        self.pages
            .get(&page_id)
            .map(|r| Arc::clone(r.value()))
            .ok_or(IndexError::MissingPage(page_id))
    }

    /// Install a zero-initialized page of `page_type` under `page_id`.
    /// Convenience wrapper around [`fresh_page_buf`] + [`Self::install`].
    pub fn install_fresh(&self, page_id: PageId, page_type: PageType) -> Result<(), IndexError> {
        self.install(page_id, fresh_page_buf(page_id, page_type))
    }

    /// Install a fresh page under `page_id`. Rejects duplicate ids.
    pub fn install(&self, page_id: PageId, page: Box<PageBuf>) -> Result<(), IndexError> {
        use dashmap::mapref::entry::Entry;
        match self.pages.entry(page_id) {
            Entry::Occupied(_) => Err(IndexError::CorruptPage {
                page_id,
                reason: "page id already mapped in primary index".into(),
            }),
            Entry::Vacant(v) => {
                v.insert(Arc::new(RwLock::new(page)));
                Ok(())
            }
        }
    }

    /// ADR-032 Slice 3 + PR #79 X-1 review fold-in: unconditional
    /// byte-copy install. Overwrites any existing page-id mapping;
    /// installs fresh if none. Called by the replay executor's
    /// [`crate::wal::PrimaryPageStoreHandle::install_or_replace`] impl.
    ///
    /// Lemma I2 is **bundle-level**: a later bundle's entry for the
    /// same page_id is a legitimate supersession, NOT a corruption.
    /// Byte-equality comparison at entry level (the pre-fold-in
    /// behaviour) was the X-1 bug — every real WAL on the branch
    /// failed replay because `PrimaryIndex::new()` emits a legacy
    /// `IndexPage = 11` record with empty-root bytes, and the first
    /// tree-mutating `CommitBundle` emits a different-bytes entry
    /// for the same `page_id`.
    pub fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), IndexError> {
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

    /// ADR-033 Z-1 (b): capture `page_id`'s pre-mutation bytes into
    /// `log` (if not already captured) and return a write-latch for
    /// in-place mutation. The builder phase calls this in place of
    /// raw `latch(page_id)` followed by `write()` / `write_arc()`.
    ///
    /// Idempotent within a transaction: the first call per `(tx, page_id)`
    /// pair snapshots the pre-W bytes into `log.page_mutations`;
    /// subsequent calls on the same page are no-ops (the log already
    /// has the pre-W bytes).
    ///
    /// **Concurrency.** This helper MUST be called under the owning
    /// index's `write_gate` — the capture reads under the page's
    /// `RwLock` read lock, drops it, then escalates to `write_arc()`.
    /// `write_gate` excludes concurrent writers; concurrent readers
    /// are permitted and observe the pre-mutation state until our
    /// write-lock acquires.
    ///
    /// On rollback, `crate::transaction::TxnManager::rollback_wal_failure`
    /// calls [`Self::restore_page_bytes`] with the captured pre-W bytes.
    pub fn capture_and_latch(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
    ) -> Result<ArcPageWriteGuard, IndexError> {
        let latch = self.latch(page_id)?;
        // Y-2: dedup MUST key on (kind, page_id). A bare page_id key
        // would collide with record-store captures on the same numeric
        // PageId (the common case for small tenants — both allocators
        // start at 1).
        if !log.has_captured(PageStoreKind::Primary, page_id) {
            let mut snapshot: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
            {
                let read = latch.read();
                snapshot.copy_from_slice(read.as_ref().as_ref());
            }
            log.page_mutations
                .push((PageStoreKind::Primary, page_id, snapshot));
        }
        Ok(latch.write_arc())
    }

    /// ADR-033 Z-1 (b): install a fresh page and record it in the
    /// transaction's mutation log so rollback can remove it. Same
    /// semantics as [`Self::install`] plus the rollback-hook entry.
    pub fn install_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), IndexError> {
        self.install(page_id, page)?;
        log.new_pages.push((PageStoreKind::Primary, page_id));
        Ok(())
    }

    /// ADR-033 Z-1 (b): `install_fresh` sibling that records the
    /// new page in the transaction's mutation log. For builder-
    /// phase code paths that allocate a fresh typed page (e.g.,
    /// B-tree split producing a new leaf) under a transaction.
    pub fn install_fresh_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page_type: PageType,
    ) -> Result<(), IndexError> {
        self.install_for_txn(log, page_id, fresh_page_buf(page_id, page_type))
    }

    /// ADR-033 Z-1 (b): capture pre-mutation bytes from an
    /// already-held write guard into `log` (if not already
    /// captured). Use this when descent has already acquired a
    /// `write_arc` on the page via the existing `latch()` API and
    /// you only want to snapshot bytes immediately before the
    /// first mutation — avoiding the extra latch round-trip of
    /// [`Self::capture_and_latch`].
    ///
    /// Idempotent within a transaction.
    pub fn capture_from_guard(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        guard: &ArcPageWriteGuard,
    ) {
        // Y-2: (PageStoreKind::Primary, page_id) compound dedup.
        if log.has_captured(PageStoreKind::Primary, page_id) {
            return;
        }
        let mut snapshot: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
        let bytes: &PageBuf = guard.as_ref();
        snapshot.copy_from_slice(bytes);
        log.page_mutations
            .push((PageStoreKind::Primary, page_id, snapshot));
    }

    /// ADR-033 Z-1 (b) rollback primitive: remove a page that was
    /// installed during a transaction that is being rolled back.
    /// Returns the removed latch (for test assertions) or `None` if
    /// the page was never mapped.
    ///
    /// Intended for `crate::transaction::TxnManager::rollback_wal_failure`;
    /// do NOT call from happy-path code. A caller who invokes this on
    /// a page referenced by a live primary-index traversal will surface
    /// `IndexError::MissingPage` at the reader's next descent step —
    /// which is fine in rollback context (the reader was observing
    /// pre-durable state that is being un-published) but a bug anywhere
    /// else.
    pub fn remove_page(&self, page_id: PageId) -> Option<PageLatch> {
        self.pages.remove(&page_id).map(|(_, latch)| latch)
    }

    /// ADR-033 Z-1 (b) rollback primitive: restore a page's in-memory
    /// bytes to a pre-captured snapshot. Write-locks the page's
    /// latch, overwrites bytes, releases. Returns `Err` if the page
    /// is not mapped.
    ///
    /// Intended for `crate::transaction::TxnManager::rollback_wal_failure`;
    /// do NOT call from happy-path code. Combined with the
    /// commit_gate held across the rollback, no concurrent Phase 1
    /// validator can observe a half-restored chain.
    pub fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), IndexError> {
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

// ─────────────────────────────────────────────────────────────────────
// Internal page codec
// ─────────────────────────────────────────────────────────────────────

/// Read-only accessor for internal-page contents.
///
/// An internal page holds a `first_child` pointer followed by `N`
/// `(key, child)` pairs, where `slot_count = N`. The `first_child`
/// subtree contains keys strictly less than `key[0]`; the `i`-th pair
/// (for `0 <= i < N`) owns keys in `[key[i], key[i+1])` (and the last
/// pair owns `[key[N-1], +inf)`).
pub struct InternalPageRef<'a> {
    bytes: &'a PageBuf,
    header: PageHeader,
}

impl<'a> InternalPageRef<'a> {
    /// Open an internal view, validating the header.
    pub fn open(bytes: &'a PageBuf) -> Result<Self, IndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexInternal.as_byte() {
            return Err(IndexError::CorruptPage {
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
    pub fn entry(&self, index: u16) -> Result<(PrimaryKey, PageId), IndexError> {
        let n = self.entry_count();
        if index >= n {
            return Err(IndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("entry index {index} out of range (count={n})"),
            });
        }
        let off = INTERNAL_ENTRY_OFFSET + (index as usize) * INTERNAL_ENTRY_SIZE;
        let key = PrimaryKey::decode(&self.bytes[off..off + PrimaryKey::SIZE])?;
        let child = PageId::new(u64::from_le_bytes(
            self.bytes[off + PrimaryKey::SIZE..off + INTERNAL_ENTRY_SIZE]
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
    pub fn find_child(&self, needle: PrimaryKey) -> Result<PageId, IndexError> {
        let n = self.entry_count();
        if n == 0 {
            return Ok(self.first_child());
        }
        // lo..hi is the search range; we're looking for the first index
        // where key[idx] > needle. `lo` at exit is `|keys <= needle|`.
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

    /// Is the page at internal capacity? (i.e., no room for another pair.)
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.entry_count() >= INTERNAL_CAPACITY
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
    /// `first_child` and no entries.
    pub fn init(bytes: &'a mut PageBuf, page_id: PageId, first_child: PageId) -> Self {
        let header = PageHeader::new(page_id, PageType::IndexInternal, TenantId::SYSTEM);
        write_header(bytes, &header);
        let off = INTERNAL_FIRST_CHILD_OFFSET;
        bytes[off..off + 8].copy_from_slice(&first_child.raw().to_le_bytes());
        Self { bytes, header }
    }

    /// Open an existing internal page for in-place mutation.
    pub fn open(bytes: &'a mut PageBuf) -> Result<Self, IndexError> {
        let header = read_header(bytes)?;
        if header.page_type != PageType::IndexInternal.as_byte() {
            return Err(IndexError::CorruptPage {
                page_id: PageId::new(header.page_id),
                reason: format!(
                    "expected IndexInternal page_type, got byte {}",
                    header.page_type
                ),
            });
        }
        Ok(Self { bytes, header })
    }

    fn page_id(&self) -> PageId {
        PageId::new(self.header.page_id)
    }

    /// Read-only view of this page.
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
    /// [`IndexError::DuplicateKey`] if `key` already appears; returns
    /// [`IndexError::CorruptPage`] if the page is full — callers must
    /// split first.
    pub fn insert(&mut self, key: PrimaryKey, right_child: PageId) -> Result<(), IndexError> {
        if self.is_full() {
            return Err(IndexError::CorruptPage {
                page_id: self.page_id(),
                reason: "internal insert into full page (split required before insert)".into(),
            });
        }
        // Linear-but-bounded search: internal pages stay small on avg,
        // and binary search on a mutating slice adds complexity without
        // speeding up the common case. Keep it explicit.
        let n = self.entry_count();
        let mut pos: u16 = n;
        for i in 0..n {
            let (k, _) = self.as_ref().entry(i)?;
            match k.cmp(&key) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => return Err(IndexError::DuplicateKey { key }),
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
        key.encode_into(&mut self.bytes[pos_off..pos_off + PrimaryKey::SIZE]);
        self.bytes[pos_off + PrimaryKey::SIZE..pos_off + INTERNAL_ENTRY_SIZE]
            .copy_from_slice(&right_child.raw().to_le_bytes());
        self.header.slot_count = n + 1;
        write_header(self.bytes, &self.header);
        Ok(())
    }

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
    ) -> Result<(Box<PageBuf>, PrimaryKey), IndexError> {
        let n = self.entry_count();
        if n < 2 {
            return Err(IndexError::CorruptPage {
                page_id: self.page_id(),
                reason: format!("internal split requires >=2 pairs, have {n}"),
            });
        }
        let mid = n / 2;
        let right_pairs = n - mid - 1;

        // Pull out the promoted key and the right side's first_child
        // (= child of the promoted pair).
        let mid_off = INTERNAL_ENTRY_OFFSET + (mid as usize) * INTERNAL_ENTRY_SIZE;
        let promoted_key = PrimaryKey::decode(&self.bytes[mid_off..mid_off + PrimaryKey::SIZE])?;
        let right_first_child = PageId::new(u64::from_le_bytes(
            self.bytes[mid_off + PrimaryKey::SIZE..mid_off + INTERNAL_ENTRY_SIZE]
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
    /// Key promoted to the parent — also the first key of the new right page.
    pub promoted_key: PrimaryKey,
    /// Page id of the newly allocated right page.
    pub new_right_page: PageId,
}

// ─────────────────────────────────────────────────────────────────────
// PrimaryIndex — public tree surface (DEC-10, DEC-13, DEC-14, DEC-17)
// ─────────────────────────────────────────────────────────────────────

/// MVCC key under `TenantId::SYSTEM` that stores the primary-index
/// root page id (DEC-10). Sibling of `catalog::CATALOG_TENANTS_KEY`
/// (=0); the root pointer is encoded as 8 little-endian bytes.
pub const PRIMARY_INDEX_ROOT_KEY: u64 = 1;

/// Which kind of write mutation is in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteOp {
    /// Insert — fails on a live duplicate.
    Insert,
    /// Upsert — replaces a live duplicate; returns the old value.
    Upsert,
    /// Remove — tombstones the entry; returns the previous value.
    Remove,
}

/// Result of a mutating operation. `previous` is the pre-op value (if
/// any). `existed` is true iff a live value existed before the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriteResult {
    /// Value that was at this key before the op, if any (live).
    pub previous: Option<PageSlot>,
    /// Whether a live value existed prior to the operation.
    pub existed: bool,
}

/// Stats returned by [`PrimaryIndex::bootstrap_from_mvcc`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BootstrapStats {
    /// Number of keys indexed by this bootstrap run.
    pub indexed: usize,
    /// Number of keys skipped because already indexed.
    pub skipped: usize,
}

// Owning guard types — have `'static` lifetime because they retain an
// internal `Arc` to the underlying lock. Lets us store guards in a
// `Vec` for iterative crabbing without lifetime gymnastics.
type ArcPageWriteGuard = parking_lot::ArcRwLockWriteGuard<RawRwLock, Box<PageBuf>>;
type ArcPageReadGuard = parking_lot::ArcRwLockReadGuard<RawRwLock, Box<PageBuf>>;

/// Primary B+tree index with `(TenantId, Kind, Id) → (PageId, SlotId)`
/// semantics. See module docs for layout; see ADR-023 for the read-
/// accelerator visibility contract.
pub struct PrimaryIndex {
    page_store: Arc<PrimaryPageStore>,
    txn_mgr: Arc<TxnManager>,
    wal: RwLock<Option<WalHandle>>,
    allocator: Arc<PageAllocator>,
    // `0` means "not loaded" — every other value is a valid root page id.
    root_cache: AtomicU64,
    // Serializes all mutations. Readers bypass this and use per-page
    // read locks; write_gate + per-page write locks together honor
    // §F's "pessimistic lock-coupling".
    //
    // ADR-032 §2: write_gate is ALSO the serialization primitive for
    // sidechannel SYSTEM root-pointer writes. grow_root pushes a
    // SideChannelWrite under write_gate; the outer commit's Phase 3
    // applies it via `TxnManager::apply_sidechannel_mvcc_write`. No
    // two grow_roots can concurrently target the SYSTEM root-pointer
    // key because write_gate serializes all grow_root callers.
    write_gate: Mutex<()>,
    // ADR-032 Slice 2: `pending_root: AtomicU64` + its companion
    // `persist_pending_root_update` are RETIRED. grow_root now pushes
    // a SideChannelWrite into the outer CommitBundle via the threaded
    // `&mut Vec<SideChannelWrite>` parameter on the bundle-aware
    // write path. The standalone write path runs a standalone SYSTEM
    // MVCC commit for each sidechannel write (matches pre-Slice-2
    // semantics; standalone is not the load-bearing hot path — only
    // bootstrap_from_mvcc + pre-bundle test code).
}

impl PrimaryIndex {
    /// Construct a primary index.
    ///
    /// On first use in a database, allocates `PageId(1)` for the root
    /// leaf, installs it, and persists the pointer under
    /// `(TenantId::SYSTEM, PRIMARY_INDEX_ROOT_KEY)`. Subsequent
    /// constructions against the same `TxnManager` recover the pointer
    /// from MVCC (page contents themselves are not repopulated until
    /// WAL replay lands in M2.e).
    pub fn new(
        txn_mgr: Arc<TxnManager>,
        allocator: Arc<PageAllocator>,
        wal: Option<WalHandle>,
    ) -> Result<Self, IndexError> {
        Self::with_page_store(txn_mgr, allocator, wal, Arc::new(PrimaryPageStore::new()))
    }

    /// Construct a primary index over an existing page store.
    ///
    /// Durable bootstrap uses this to recover pages into a raw
    /// [`PrimaryPageStore`] before the WAL writer opens, then wraps that same
    /// store for serving after torn-tail truncation and writer attachment.
    pub fn with_page_store(
        txn_mgr: Arc<TxnManager>,
        allocator: Arc<PageAllocator>,
        wal: Option<WalHandle>,
        page_store: Arc<PrimaryPageStore>,
    ) -> Result<Self, IndexError> {
        let this = Self {
            page_store,
            txn_mgr: Arc::clone(&txn_mgr),
            wal: RwLock::new(wal),
            allocator: Arc::clone(&allocator),
            root_cache: AtomicU64::new(0),
            write_gate: Mutex::new(()),
        };
        let root_id = match this.read_root_from_mvcc() {
            Some(existing) => existing,
            None => {
                let fresh = this.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
                // Stage-WAL-publish (DEC-21): build the fresh buffer,
                // emit WAL on its bytes, THEN install it in the page
                // store. Installation is what makes the page externally
                // discoverable; WAL must be durable first.
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
    pub fn root(&self) -> Result<PageId, IndexError> {
        let cached = self.root_cache.load(Ordering::Acquire);
        if cached != 0 {
            return Ok(PageId::new(cached));
        }
        match self.read_root_from_mvcc() {
            Some(id) => {
                self.root_cache.store(id.raw(), Ordering::Release);
                Ok(id)
            }
            None => Err(IndexError::CorruptPage {
                page_id: PageId::ZERO,
                reason: "root pointer missing from MVCC state".to_owned(),
            }),
        }
    }

    /// Access the underlying page store. Used by the ADR-033 Z-1 (b)
    /// rollback closure in `crud::commit` to drain the mutation log's
    /// new_pages / page_mutations entries. Not intended for hot-path
    /// use — readers should stick to [`Self::lookup`] and writers to
    /// [`Self::upsert_deferred`] & co.
    #[must_use]
    pub fn page_store(&self) -> &Arc<PrimaryPageStore> {
        &self.page_store
    }

    /// Attach a WAL handle after durable bootstrap has recovered the WAL prefix.
    ///
    /// The same `PrimaryIndex` must serve as both the replay target and the
    /// runtime index, so post-recovery writer attachment is intentionally
    /// in-place.
    pub fn attach_wal(&self, wal: WalHandle) {
        *self.wal.write() = Some(wal);
    }

    /// ADR-033 Z-1 (b) rollback primitive: atomically restore the
    /// `root_cache` pointer to `old_root_id`.
    ///
    /// Called by the rollback closure in `crud::commit` when the
    /// mutation log's `root_changes` vec contains an entry for the
    /// primary index. §5 ordering requires root_cache restoration
    /// BEFORE any `new_pages` removal of the new_root page.
    pub fn restore_root_cache(&self, old_root_id: PageId) {
        self.root_cache.store(old_root_id.raw(), Ordering::Release);
    }

    fn read_root_from_mvcc(&self) -> Option<PageId> {
        let txn = self.txn_mgr.begin(TenantId::SYSTEM);
        let bytes = txn.read(PRIMARY_INDEX_ROOT_KEY)?;
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

    fn persist_root_to_mvcc(&self, root_id: PageId) -> Result<(), IndexError> {
        let mut txn = self.txn_mgr.begin(TenantId::SYSTEM);
        txn.write(
            PRIMARY_INDEX_ROOT_KEY,
            Bytes::copy_from_slice(&root_id.raw().to_le_bytes()),
        );
        txn.commit().map_err(IndexError::Core)?;
        Ok(())
    }

    // ---- read path ----

    /// Snapshot-isolated is NOT a thing here — the index is a read
    /// accelerator per ADR-023. Lookup returns the latest committed
    /// entry or `None` if absent / tombstoned.
    pub fn lookup(&self, key: PrimaryKey) -> Result<Option<PageSlot>, IndexError> {
        let root_id = self.root()?;
        let root_latch = self.page_store.latch(root_id)?;
        let mut guard: ArcPageReadGuard = root_latch.read_arc();

        loop {
            let header = read_header(guard.as_ref())?;
            if header.page_type == PageType::IndexLeaf.as_byte() {
                let leaf = LeafPageRef::open(guard.as_ref())?;
                return leaf.lookup(key);
            }
            if header.page_type != PageType::IndexInternal.as_byte() {
                return Err(IndexError::CorruptPage {
                    page_id: PageId::new(header.page_id),
                    reason: format!("unexpected page_type {} in read descent", header.page_type),
                });
            }
            let child_id = {
                let internal = InternalPageRef::open(guard.as_ref())?;
                internal.find_child(key)?
            };
            // Standard read-crabbing: acquire child latch BEFORE dropping
            // the parent guard. `guard = child_latch.read_arc()` evaluates
            // the right-hand side first (holding parent via old `guard`),
            // then drops the old guard on assignment.
            let child_latch = self.page_store.latch(child_id)?;
            guard = child_latch.read_arc();
        }
    }

    // ---- write path ----

    /// Insert `(key, value)`. Fails if `key` already has a live value.
    ///
    /// Standalone emission path: the staged `IndexPage` snapshot is
    /// drained into its own `wal.append(IndexPage)` immediately after
    /// the index mutation. Callers already inside a bundle-aware
    /// commit (`Transaction::commit_with_bundle`) should use
    /// [`Self::insert_deferred`] instead, which returns the staged
    /// emits for folding into the outer `CommitBundle` (ADR-031).
    pub fn insert(&self, key: PrimaryKey, value: PageSlot) -> Result<(), IndexError> {
        self.write(key, Some(value), WriteOp::Insert).map(|_| ())
    }

    /// Upsert `(key, value)`. Returns the previous live value, if any.
    /// Standalone emission path; see [`Self::upsert_deferred`] for
    /// bundle-aware callers (ADR-031).
    pub fn upsert(&self, key: PrimaryKey, value: PageSlot) -> Result<Option<PageSlot>, IndexError> {
        self.write(key, Some(value), WriteOp::Upsert)
            .map(|r| r.previous)
    }

    /// Tombstone `key`. Returns the previous live value, if any.
    /// Standalone emission path; see [`Self::remove_deferred`] for
    /// bundle-aware callers (ADR-031).
    pub fn remove(&self, key: PrimaryKey) -> Result<Option<PageSlot>, IndexError> {
        self.write(key, None, WriteOp::Remove).map(|r| r.previous)
    }

    /// Bundle-aware insert (ADR-031). Performs the mutation +
    /// in-memory install under `write_gate` + per-page latch, then
    /// returns the staged `IndexPage` byte snapshots to the caller
    /// (rather than draining them into the WAL internally). The
    /// caller MUST ensure the staged emits reach a durable
    /// `CommitBundle` record before any snapshot ≥ the associated
    /// commit_lsn is published to readers outside this process —
    /// `TxnManager::commit_with_bundle_writes` guarantees this as
    /// Phase 2's `wal.append(CommitBundle)`.
    ///
    /// ADR-032 Slice 2: `sc_writes` is an in/out parameter — if the
    /// insert's split cascade triggers grow_root, the SYSTEM
    /// root-pointer update is pushed here. The caller forwards these
    /// to the outer CommitBundle's sidechannel list (typically by
    /// passing the `&mut Vec<SideChannelWrite>` handed to the
    /// `commit_with_bundle` builder directly).
    pub fn insert_deferred(
        &self,
        key: PrimaryKey,
        value: PageSlot,
        sc_writes: &mut Vec<SideChannelWrite>,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, IndexError> {
        let (_result, staged) =
            self.write_deferred(key, Some(value), WriteOp::Insert, sc_writes, log)?;
        Ok(staged)
    }

    /// Bundle-aware upsert (ADR-031 + ADR-032 Slice 2). See
    /// [`Self::insert_deferred`] for the durability contract.
    /// Returns the previous live value (if any) alongside the staged
    /// emits.
    ///
    /// ADR-033 Z-1 (b): `log` collects in-memory page mutations so a
    /// WAL fsync failure can unwind them via
    /// `crate::transaction::TxnManager::rollback_wal_failure`. Pass
    /// the `&mut TxnMutationLog` handed to the `commit_with_bundle`
    /// builder.
    pub fn upsert_deferred(
        &self,
        key: PrimaryKey,
        value: PageSlot,
        sc_writes: &mut Vec<SideChannelWrite>,
        log: &mut TxnMutationLog,
    ) -> Result<(Option<PageSlot>, Vec<StagedEmit>), IndexError> {
        let (result, staged) =
            self.write_deferred(key, Some(value), WriteOp::Upsert, sc_writes, log)?;
        Ok((result.previous, staged))
    }

    /// Bundle-aware tombstone (ADR-031 + ADR-032 Slice 2). See
    /// [`Self::insert_deferred`] for the durability contract.
    /// Returns the previous live value (if any) alongside the staged
    /// emits.
    pub fn remove_deferred(
        &self,
        key: PrimaryKey,
        sc_writes: &mut Vec<SideChannelWrite>,
        log: &mut TxnMutationLog,
    ) -> Result<(Option<PageSlot>, Vec<StagedEmit>), IndexError> {
        let (result, staged) = self.write_deferred(key, None, WriteOp::Remove, sc_writes, log)?;
        Ok((result.previous, staged))
    }

    /// Internal write path: standalone emit variant. Mutates + installs
    /// under `write_gate` (via `write_deferred`), then emits ALL staged
    /// pages + the grow_root SYSTEM root-pointer write as ONE
    /// crash-atomic `CommitBundle` (`commit_index_pages_atomic`).
    ///
    /// Kept for non-bundle callers (`insert` / `upsert` / `remove`
    /// public wrappers + `bootstrap_from_mvcc`). The standalone path
    /// is NOT the load-bearing hot path — `crud::commit` uses
    /// `insert_deferred` / `upsert_deferred` / `remove_deferred`
    /// which thread sc_writes into the outer CommitBundle atomically
    /// (ADR-032 §2). #37 [A-1] closed the pre-existing split-record /
    /// overflow-successor crash hazard on this path: it no longer
    /// drains one `IndexPage` record per page, so a crash mid-op can no
    /// longer leave an orphan page on replay (ADR-031 realized for the
    /// standalone path).
    fn write(
        &self,
        key: PrimaryKey,
        value: Option<PageSlot>,
        op: WriteOp,
    ) -> Result<WriteResult, IndexError> {
        let mut sc_writes: Vec<SideChannelWrite> = Vec::new();
        // ADR-033 Z-1 (b): the standalone `write` path is not inside
        // an outer commit's builder closure, so there is no shared
        // mutation log to populate. We pass a throwaway log; rollback
        // never sees it (WAL failure inside the standalone path
        // aborts the whole operation, which is acceptable because
        // standalone writers are bootstrap-only — see this fn's
        // rustdoc). A future amendment could wire the standalone
        // write into a per-write rollback if rollback's scope ever
        // extends to bootstrap; for now the trade-off is acceptable.
        let mut log = TxnMutationLog::new();
        let (result, staged) = self.write_deferred(key, value, op, &mut sc_writes, &mut log)?;
        // #37 [A-1]: fold ALL staged index pages + the grow_root SYSTEM
        // root-pointer writes into ONE crash-atomic CommitBundle (was:
        // N `IndexPage` records via `drain_staged_emits` + a standalone
        // SYSTEM MVCC commit per sidechannel write). A crash mid-split /
        // mid-grow_root can no longer leave an orphan page — replay
        // applies all pages or none. Realizes ADR-031 for the
        // standalone path; see `TxnManager::commit_index_pages_atomic`.
        let wal = self.wal.read().clone();
        self.txn_mgr
            .commit_index_pages_atomic(wal.as_ref(), &staged, &sc_writes)
            .map_err(IndexError::Core)?;
        Ok(result)
    }

    /// Internal write path: bundle-aware variant returning the
    /// staged `IndexPage` snapshots for the caller to fold into a
    /// `CommitBundle`. Mutation + install still happen under
    /// `write_gate` + per-page latches per ADR-030; the drop from
    /// this function leaves the caller holding NO index locks.
    ///
    /// ADR-032 Slice 2: `sc_writes` is threaded through so
    /// `grow_root` can push a SYSTEM-tenant root-pointer write
    /// directly into the outer CommitBundle's sidechannel list. The
    /// outer commit's Phase 2 encodes it into v2 bundle; Phase 3
    /// applies it atomically with the primary writes.
    fn write_deferred(
        &self,
        key: PrimaryKey,
        value: Option<PageSlot>,
        op: WriteOp,
        sc_writes: &mut Vec<SideChannelWrite>,
        log: &mut TxnMutationLog,
    ) -> Result<(WriteResult, Vec<StagedEmit>), IndexError> {
        // Per ADR-030 + ADR-031: mutate+install under `write_gate` +
        // per-page latch. In the bundle-aware path the staged emits
        // ride the outer commit's single `CommitBundle` `wal.append`;
        // in the standalone path `write()` folds them into one
        // crash-atomic `CommitBundle` via `commit_index_pages_atomic`
        // (#37 [A-1]).
        let mut staged: Vec<StagedEmit> = Vec::with_capacity(4);
        let result = {
            let _gate = self.write_gate.lock();
            let root_id = self.root()?;

            // Descend top-down with write-crabbing: push parents into
            // `path`; drop all ancestors once we find a safe child (one
            // that can absorb a promoted key without splitting).
            let mut path: Vec<(PageId, ArcPageWriteGuard)> = Vec::with_capacity(4);
            let root_latch = self.page_store.latch(root_id)?;
            path.push((root_id, root_latch.write_arc()));

            loop {
                let (_, top_guard) = path.last().expect("path is non-empty inside descent loop");
                let page_type_byte = read_header(top_guard.as_ref())?.page_type;
                if page_type_byte == PageType::IndexLeaf.as_byte() {
                    break;
                }
                if page_type_byte != PageType::IndexInternal.as_byte() {
                    return Err(IndexError::CorruptPage {
                        page_id: path.last().unwrap().0,
                        reason: format!("unexpected page_type {page_type_byte} in write descent"),
                    });
                }
                // Find child, then lock it, then decide on ancestor release.
                let child_id = {
                    let internal = InternalPageRef::open(top_guard.as_ref())?;
                    internal.find_child(key)?
                };
                let child_latch = self.page_store.latch(child_id)?;
                let child_guard = child_latch.write_arc();
                let child_is_safe = {
                    let hdr = read_header(child_guard.as_ref())?;
                    if hdr.page_type == PageType::IndexLeaf.as_byte() {
                        hdr.slot_count < LEAF_CAPACITY
                    } else if hdr.page_type == PageType::IndexInternal.as_byte() {
                        hdr.slot_count < INTERNAL_CAPACITY
                    } else {
                        return Err(IndexError::CorruptPage {
                            page_id: child_id,
                            reason: format!("unexpected child page_type {}", hdr.page_type),
                        });
                    }
                };
                if child_is_safe {
                    // Child can absorb any split propagation from below,
                    // so every ancestor is also safe in the chain; drop.
                    path.clear();
                }
                path.push((child_id, child_guard));
            }

            // Perform the op at the leaf. Pop the leaf frame.
            let (leaf_id, mut leaf_guard) = path.pop().expect("leaf is at top of path");
            let (result, leaf_split) =
                self.apply_leaf_op(leaf_id, &mut leaf_guard, key, value, op, &mut staged, log)?;
            drop(leaf_guard);

            // Propagate split upward through held ancestor chain.
            let mut pending = leaf_split;
            while let Some((parent_id, mut parent_guard)) = path.pop() {
                match pending {
                    None => {
                        // No further propagation; drop remaining ancestors.
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

            // If propagation reached past the root, grow a new root.
            // `grow_root` is still under `write_gate` — ADR-032 §2:
            // it pushes the SYSTEM root-pointer update into
            // `sc_writes` rather than triggering an inline or
            // stashed SYSTEM MVCC commit. The outer CommitBundle
            // folds the sidechannel write atomically, closing #66 by
            // construction.
            if let Some(split) = pending {
                self.grow_root(split, &mut staged, sc_writes, log)?;
            }

            result
            // `_gate` drops here; all per-page guards have been
            // dropped already. The caller (bundle-aware path) folds
            // `staged` into the outer `CommitBundle`; the standalone
            // path (`write`) drains it immediately.
        };

        Ok((result, staged))
    }

    // Signature has eight params (self + 7) per ADR-030's staged
    // emit plumbing + ADR-033's mutation-log plumbing; refactoring to
    // bundle params would obscure the (leaf_id, guard, op-triple,
    // staged, log) separation that the call site relies on.
    #[allow(clippy::too_many_arguments)]
    fn apply_leaf_op(
        &self,
        leaf_id: PageId,
        guard: &mut ArcPageWriteGuard,
        key: PrimaryKey,
        value: Option<PageSlot>,
        op: WriteOp,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<(WriteResult, Option<SplitInfo>), IndexError> {
        // ADR-033 Z-1 (b): snapshot pre-mutation bytes before the
        // first mutation of this leaf. Idempotent within a txn.
        self.page_store.capture_from_guard(log, leaf_id, guard);
        let leaf_bytes: &mut PageBuf = guard.as_mut();
        let mut leaf = LeafPageMut::open(leaf_bytes)?;

        let (result, needs_split) = match op {
            WriteOp::Insert => {
                let v = value.ok_or_else(|| IndexError::CorruptPage {
                    page_id: leaf_id,
                    reason: "WriteOp::Insert without value".to_owned(),
                })?;
                if leaf.is_full() {
                    (
                        WriteResult {
                            previous: None,
                            existed: false,
                        },
                        Some(v),
                    )
                } else {
                    // insert() handles tombstone-revival and duplicate-live rejection.
                    leaf.insert(key, v)?;
                    (
                        WriteResult {
                            previous: None,
                            existed: false,
                        },
                        None,
                    )
                }
            }
            WriteOp::Upsert => {
                let v = value.ok_or_else(|| IndexError::CorruptPage {
                    page_id: leaf_id,
                    reason: "WriteOp::Upsert without value".to_owned(),
                })?;
                // Only split if this key doesn't have an existing slot
                // to occupy (a live duplicate or a tombstone-revive is
                // in-place; a brand-new key on a full page needs split).
                let in_place_ok =
                    !leaf.is_full() || leaf.as_ref().contains_including_tombstoned(key)?;
                if in_place_ok {
                    let prev = leaf.upsert(key, v)?;
                    (
                        WriteResult {
                            previous: prev,
                            existed: prev.is_some(),
                        },
                        None,
                    )
                } else {
                    (
                        WriteResult {
                            previous: None,
                            existed: false,
                        },
                        Some(v),
                    )
                }
            }
            WriteOp::Remove => {
                let prev = leaf.remove(key)?;
                (
                    WriteResult {
                        previous: prev,
                        existed: prev.is_some(),
                    },
                    None,
                )
            }
        };

        if let Some(to_insert) = needs_split {
            // Split the leaf, then insert into the correct half.
            let new_id = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
            let (mut new_buf, promoted_key) = leaf.split_into(new_id)?;
            // `leaf` is now the left half, `new_buf` is the right.
            if key < promoted_key {
                leaf.insert(key, to_insert)?;
            } else {
                let mut right = LeafPageMut::open(new_buf.as_mut())?;
                right.insert(key, to_insert)?;
            }
            // ADR-030: stage WAL emits for both pages; drain runs
            // outside `write_gate` after locks release. Byte snapshots
            // are copied under the still-held leaf guard, so they
            // capture the post-split post-insert state. No re-latch;
            // no deadlock. Split-atomicity-across-crash remains a
            // pre-existing M2.e concern.
            //
            // ADR-033 Z-1 (b): install_for_txn records the new leaf
            // in `log.new_pages` so rollback removes it from the
            // store on WAL failure.
            Self::stage_emit(staged, new_id, new_buf.as_ref());
            Self::stage_emit(staged, leaf_id, leaf.page_bytes());
            self.page_store.install_for_txn(log, new_id, new_buf)?;
            return Ok((
                result,
                Some(SplitInfo {
                    promoted_key,
                    new_right_page: new_id,
                }),
            ));
        }

        // Non-splitting mutation — stage the WAL emit for drain
        // after locks release (ADR-030).
        Self::stage_emit(staged, leaf_id, leaf.page_bytes());
        Ok((result, None))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_internal_insert(
        &self,
        page_id: PageId,
        guard: &mut ArcPageWriteGuard,
        incoming: SplitInfo,
        staged: &mut Vec<StagedEmit>,
        log: &mut TxnMutationLog,
    ) -> Result<Option<SplitInfo>, IndexError> {
        // ADR-033 Z-1 (b): snapshot pre-mutation bytes of this
        // internal page before mutating.
        self.page_store.capture_from_guard(log, page_id, guard);
        let bytes: &mut PageBuf = guard.as_mut();
        let mut internal = InternalPageMut::open(bytes)?;
        if internal.is_full() {
            // Must split this internal before inserting the promoted pair.
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
        sc_writes: &mut Vec<SideChannelWrite>,
        log: &mut TxnMutationLog,
    ) -> Result<(), IndexError> {
        let old_root = self.root()?;
        let new_root_id = self.allocator.alloc(TenantId::SYSTEM, PageType::IndexLeaf);
        let mut new_buf = fresh_page_buf(new_root_id, PageType::IndexInternal);
        {
            let mut new_root = InternalPageMut::init(new_buf.as_mut(), new_root_id, old_root);
            new_root.insert(split.promoted_key, split.new_right_page)?;
        }
        // ADR-032 §2 — F1 FOLD: push the SYSTEM root-pointer update
        // into the outer CommitBundle's sidechannel list instead of
        // stashing into `pending_root` or calling
        // `persist_root_to_mvcc` inline. The outer commit's Phase 2
        // encodes this as a v2 bundle entry (per-entry tenant =
        // SYSTEM, key = PRIMARY_INDEX_ROOT_KEY); Phase 3 applies it
        // via `apply_sidechannel_mvcc_write` atomically with the
        // primary writes.
        //
        // This closes #66 by construction: the "crash between outer
        // fsync and SYSTEM persist" window does not exist — there is
        // only ONE fsync, one commit_lsn, one atomic bundle. MVCC
        // root_pointer = new_root_id iff the bundle is durable iff
        // the new-root IndexPage entry is durable, so MVCC having R
        // implies page_store has R installed. The pre-Slice-2
        // fallback "bootstrap_from_mvcc patches up missing pages on
        // replay" is no longer the recovery path — it remains as a
        // defensive pre-check for pre-ADR-032 legacy WALs (Slice 3
        // §R1 classification path).
        //
        // The in-memory tree-safety invariant from PR #67 Slice 4 is
        // unchanged: `old_root` remains reachable as `new_root`'s
        // `first_child` for the duration of this function; the
        // root_cache.store below is the explicit publish that tells
        // in-flight readers to traverse from R_B — they were already
        // traversing from R_A under read-crabbing, so R_A's
        // reachability via R_B's first_child guarantees no torn
        // traversal.
        // ADR-033 Z-1 (b): record the root-pointer change BEFORE
        // publishing the new root via `root_cache.store`. On
        // rollback §5 requires restoring root pointers first (so a
        // reader capturing new_root_id from root_cache finds it
        // still mapped in page_store when the rollback un-installs
        // it last). Recording old_root here gives rollback the
        // target to restore.
        log.root_changes.push((IndexHandle::PRIMARY, old_root));
        Self::stage_emit(staged, new_root_id, new_buf.as_ref());
        self.page_store.install_for_txn(log, new_root_id, new_buf)?;
        self.root_cache.store(new_root_id.raw(), Ordering::Release);
        sc_writes.push(SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: PRIMARY_INDEX_ROOT_KEY,
            value: Some(Bytes::copy_from_slice(&new_root_id.raw().to_le_bytes())),
        });
        Ok(())
    }

    // ---- WAL emission ----

    /// Capture a byte-level snapshot of a page's post-mutation
    /// contents while the caller still holds its write latch, and
    /// push it onto `staged` for drain outside `write_gate`.
    ///
    /// See ADR-030. The copy is mandatory because, after
    /// `write_gate` and per-page latches are released, another
    /// writer may mutate the same page in place; the staged bytes
    /// must be the snapshot-as-of-this-commit or the WAL record
    /// would log a post-conflict state.
    #[inline]
    fn stage_emit(staged: &mut Vec<StagedEmit>, page_id: PageId, bytes: &[u8; PAGE_SIZE]) {
        // SAFETY-less: `Box::new([0u8; PAGE_SIZE])` + copy_from_slice
        // is safe, not `unsafe`. Allocation is ~700 ns at 8 KiB on
        // the M3 bench host; total per-commit staging overhead is
        // < 5 µs for the typical 1–7 emits per commit.
        let mut copy: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        copy.copy_from_slice(bytes);
        staged.push(StagedEmit {
            kind: crate::wal::bundle::BundlePageKind::PrimaryIndex,
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
    /// per-page write guards). The old DEC-21 contract required
    /// the caller to hold the write guard across the append; ADR-030
    /// narrows that rule for index pages because ADR-023 designates
    /// the index as a read accelerator. Callers stage bytes via
    /// [`Self::stage_emit`] under the latch.
    ///
    /// The no-re-latch property of DEC-21 is preserved: this method
    /// does NOT latch the page store. Bytes are supplied directly.
    fn emit_wal_for_bytes(
        &self,
        page_id: PageId,
        bytes: &[u8; PAGE_SIZE],
    ) -> Result<(), IndexError> {
        let Some(wal) = self.wal.read().clone() else {
            return Ok(());
        };
        self.emit_wal_for_bytes_inner(&wal, page_id, bytes)
    }

    fn emit_wal_for_bytes_inner(
        &self,
        wal: &WalHandle,
        page_id: PageId,
        bytes: &[u8; PAGE_SIZE],
    ) -> Result<(), IndexError> {
        let payload = encode_index_page_payload(page_id, TenantId::SYSTEM, bytes);
        wal.append(
            WalRecordType::IndexPage,
            /* txn_id = */ 0,
            now_millis(),
            TenantId::SYSTEM,
            payload,
        )
        .map_err(IndexError::Core)
        .map(|_lsn: Lsn| ())
    }

    // ---- bootstrap ----

    /// Idempotent migration: consume an iterator of `(key, value)`
    /// pairs recovered from the MVCC state and install them into the
    /// index. Entries that already exist are skipped. See ADR-023.
    pub fn bootstrap_from_mvcc<I>(&self, entries: I) -> Result<BootstrapStats, IndexError>
    where
        I: IntoIterator<Item = (PrimaryKey, PageSlot)>,
    {
        let mut stats = BootstrapStats::default();
        for (key, value) in entries {
            if self.lookup(key)?.is_some() {
                stats.skipped += 1;
            } else {
                self.upsert(key, value)?;
                stats.indexed += 1;
            }
        }
        Ok(stats)
    }
}

/// Encode an [`WalRecordType::IndexPage`] payload.
///
/// Layout: `8 B page_id (LE u64) | 8 B tenant_id (LE u64) | 8192 B page bytes`.
#[must_use]
pub fn encode_index_page_payload(
    page_id: PageId,
    tenant: TenantId,
    page_bytes: &[u8; PAGE_SIZE],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + PAGE_SIZE);
    out.extend_from_slice(&page_id.raw().to_le_bytes());
    out.extend_from_slice(&tenant.raw().to_le_bytes());
    out.extend_from_slice(page_bytes);
    out
}

/// Decode an [`WalRecordType::IndexPage`] payload. Returns
/// `(page_id, tenant_id, page_bytes)`; used by M2.e WAL replay.
pub fn decode_index_page_payload(
    payload: &[u8],
) -> Result<(PageId, TenantId, Box<[u8; PAGE_SIZE]>), IndexError> {
    const EXPECTED: usize = 16 + PAGE_SIZE;
    if payload.len() != EXPECTED {
        return Err(IndexError::CorruptPage {
            page_id: PageId::ZERO,
            reason: format!(
                "IndexPage payload length {} != expected {}",
                payload.len(),
                EXPECTED
            ),
        });
    }
    let page_raw = u64::from_le_bytes(payload[0..8].try_into().expect("slice size asserted above"));
    let tenant_raw = u64::from_le_bytes(
        payload[8..16]
            .try_into()
            .expect("slice size asserted above"),
    );
    let mut bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    bytes.copy_from_slice(&payload[16..]);
    Ok((PageId::new(page_raw), TenantId::new(tenant_raw), bytes))
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────
// Tests (leaf-level only; tree ops land in later commits)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use proptest::prelude::*;

    use super::*;

    fn k(tenant: u64, kind: RecordKind, id: u64) -> PrimaryKey {
        PrimaryKey::new(TenantId::new(tenant), kind, id)
    }

    fn v(page: u64, slot: u16) -> PageSlot {
        PageSlot::new(PageId::new(page), SlotId(slot))
    }

    #[test]
    fn key_roundtrip_encodes_24_bytes() {
        let mut buf = [0u8; PrimaryKey::SIZE];
        let key = k(42, RecordKind::Rel, 1234);
        key.encode_into(&mut buf);
        let back = PrimaryKey::decode(&buf).unwrap();
        assert_eq!(back, key);
        assert_eq!(buf[8], 1, "RecordKind::Rel encodes to 1");
        assert_eq!(&buf[9..16], &[0u8; 7], "kind padding must be zeroed");
    }

    #[test]
    fn key_ordering_is_tenant_major_then_kind_then_id() {
        let a = k(1, RecordKind::Node, 999);
        let b = k(2, RecordKind::Node, 0);
        let c = k(1, RecordKind::Rel, 0);
        let d = k(1, RecordKind::Node, 1000);
        // tenant major: a < b
        assert!(a < b);
        // within tenant 1, kind Node < Rel regardless of id.
        assert!(a < c);
        assert!(d < c);
        // within tenant+kind, id monotone.
        assert!(a < d);
    }

    #[test]
    fn value_tombstone_roundtrip() {
        let mut buf = [0u8; PageSlot::SIZE];
        let val = v(9999, 17);
        val.encode_into(&mut buf, true);
        let (back, tomb) = PageSlot::decode(&buf);
        assert_eq!(back, val);
        assert!(tomb);

        val.encode_into(&mut buf, false);
        let (_, tomb2) = PageSlot::decode(&buf);
        assert!(!tomb2);
    }

    #[test]
    fn leaf_capacity_constants_match_invariants() {
        assert_eq!(LEAF_CAPACITY, 203);
        assert_eq!(INTERNAL_CAPACITY, 254);
        // Check the "capacity + 1 child" internal-fanout invariant.
        assert_eq!(
            INTERNAL_ENTRY_OFFSET + (INTERNAL_CAPACITY as usize) * INTERNAL_ENTRY_SIZE,
            48 + 254 * 32
        );
    }

    #[test]
    fn leaf_insert_lookup_roundtrip_single_page() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        assert_eq!(leaf.entry_count(), 0);
        let key_a = k(1, RecordKind::Node, 10);
        let key_b = k(1, RecordKind::Node, 20);
        let key_c = k(1, RecordKind::Rel, 5);

        leaf.insert(key_b, v(100, 0)).unwrap();
        leaf.insert(key_a, v(101, 1)).unwrap();
        leaf.insert(key_c, v(102, 2)).unwrap();
        assert_eq!(leaf.entry_count(), 3);

        // Verify sort order preserved.
        let view = leaf.as_ref();
        let (k0, _, _) = view.entry(0).unwrap();
        let (k1, _, _) = view.entry(1).unwrap();
        let (k2, _, _) = view.entry(2).unwrap();
        assert!(k0 < k1 && k1 < k2);
        assert_eq!(k0, key_a);
        assert_eq!(k1, key_b);
        assert_eq!(k2, key_c);

        assert_eq!(view.lookup(key_a).unwrap(), Some(v(101, 1)));
        assert_eq!(view.lookup(key_b).unwrap(), Some(v(100, 0)));
        assert_eq!(view.lookup(key_c).unwrap(), Some(v(102, 2)));
        assert_eq!(view.lookup(k(1, RecordKind::Node, 999)).unwrap(), None);
    }

    #[test]
    fn leaf_duplicate_live_key_rejected() {
        let mut page = fresh_page_buf(PageId::new(2), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        let key = k(1, RecordKind::Node, 7);
        leaf.insert(key, v(5, 0)).unwrap();
        let err = leaf.insert(key, v(5, 1)).unwrap_err();
        assert!(matches!(err, IndexError::DuplicateKey { .. }));
    }

    #[test]
    fn leaf_upsert_returns_old_value() {
        let mut page = fresh_page_buf(PageId::new(3), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        let key = k(1, RecordKind::Node, 7);
        assert_eq!(leaf.upsert(key, v(5, 0)).unwrap(), None);
        assert_eq!(leaf.upsert(key, v(6, 1)).unwrap(), Some(v(5, 0)));
        assert_eq!(leaf.as_ref().lookup(key).unwrap(), Some(v(6, 1)));
    }

    #[test]
    fn leaf_tombstone_blocks_lookup_and_allows_reinsert() {
        let mut page = fresh_page_buf(PageId::new(4), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        let key = k(1, RecordKind::Node, 7);
        leaf.insert(key, v(5, 0)).unwrap();
        assert_eq!(leaf.remove(key).unwrap(), Some(v(5, 0)));
        assert_eq!(leaf.as_ref().lookup(key).unwrap(), None);
        // Slot count unchanged (mark-only delete).
        assert_eq!(leaf.entry_count(), 1);
        // Reinserting the same key reuses the tombstoned slot in place.
        leaf.insert(key, v(9, 3)).unwrap();
        assert_eq!(leaf.entry_count(), 1);
        assert_eq!(leaf.as_ref().lookup(key).unwrap(), Some(v(9, 3)));
    }

    #[test]
    fn leaf_full_page_rejects_new_key() {
        let mut page = fresh_page_buf(PageId::new(5), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        for i in 0..LEAF_CAPACITY as u64 {
            leaf.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        assert!(leaf.is_full());
        let err = leaf
            .insert(k(1, RecordKind::Node, u64::MAX), v(1, 0))
            .unwrap_err();
        assert!(matches!(err, IndexError::CorruptPage { .. }));
        // But tombstone-reinsert is still allowed (doesn't need a new slot).
        leaf.remove(k(1, RecordKind::Node, 0)).unwrap();
        leaf.insert(k(1, RecordKind::Node, 0), v(99, 7)).unwrap();
        assert_eq!(
            leaf.as_ref().lookup(k(1, RecordKind::Node, 0)).unwrap(),
            Some(v(99, 7))
        );
    }

    #[test]
    fn page_store_basic_install_and_latch() {
        let store = PrimaryPageStore::new();
        let page = fresh_page_buf(PageId::new(7), PageType::IndexLeaf);
        store.install(PageId::new(7), page).unwrap();
        assert_eq!(store.len(), 1);
        {
            let latch = store.latch(PageId::new(7)).unwrap();
            let g = latch.read();
            let leaf = LeafPageRef::open(g.as_ref()).unwrap();
            assert_eq!(leaf.entry_count(), 0);
        }
        // Duplicate install is rejected.
        let err = store
            .install(
                PageId::new(7),
                fresh_page_buf(PageId::new(7), PageType::IndexLeaf),
            )
            .unwrap_err();
        assert!(matches!(err, IndexError::CorruptPage { .. }));
        // Missing page latch is an error.
        assert!(matches!(
            store.latch(PageId::new(999)).unwrap_err(),
            IndexError::MissingPage(_)
        ));
    }

    #[test]
    fn page_store_write_latch_persists_mutations() {
        let store = PrimaryPageStore::new();
        store
            .install(
                PageId::new(11),
                fresh_page_buf(PageId::new(11), PageType::IndexLeaf),
            )
            .unwrap();
        {
            let latch = store.latch(PageId::new(11)).unwrap();
            let mut g = latch.write();
            let mut leaf = LeafPageMut::open(g.as_mut()).unwrap();
            leaf.insert(k(1, RecordKind::Node, 1), v(1, 0)).unwrap();
            leaf.insert(k(1, RecordKind::Node, 2), v(2, 0)).unwrap();
        }
        let latch = store.latch(PageId::new(11)).unwrap();
        let g = latch.read();
        let leaf = LeafPageRef::open(g.as_ref()).unwrap();
        assert_eq!(leaf.entry_count(), 2);
    }

    // ─── ADR-033 Z-1 (b): PrimaryPageStore rollback helpers ───

    #[test]
    fn capture_and_latch_captures_pre_mutation_bytes_once() {
        let store = PrimaryPageStore::new();
        store
            .install(
                PageId::new(42),
                fresh_page_buf(PageId::new(42), PageType::IndexLeaf),
            )
            .unwrap();
        let mut log = TxnMutationLog::new();

        // First capture snapshots pre-W state (an empty leaf).
        {
            let mut guard = store.capture_and_latch(&mut log, PageId::new(42)).unwrap();
            let mut leaf = LeafPageMut::open(guard.as_mut()).unwrap();
            leaf.insert(k(1, RecordKind::Node, 1), v(1, 0)).unwrap();
        }
        assert_eq!(log.page_mutations.len(), 1);
        assert_eq!(log.page_mutations[0].0, PageStoreKind::Primary);
        assert_eq!(log.page_mutations[0].1, PageId::new(42));

        // Second capture on the same page is a no-op (idempotent).
        {
            let mut guard = store.capture_and_latch(&mut log, PageId::new(42)).unwrap();
            let mut leaf = LeafPageMut::open(guard.as_mut()).unwrap();
            leaf.insert(k(1, RecordKind::Node, 2), v(2, 0)).unwrap();
        }
        assert_eq!(
            log.page_mutations.len(),
            1,
            "second capture must be idempotent"
        );

        // The captured snapshot reflects pre-W bytes (empty leaf).
        let snapshot = &log.page_mutations[0].2;
        let leaf_view = LeafPageRef::open(snapshot.as_ref()).unwrap();
        assert_eq!(leaf_view.entry_count(), 0, "capture must precede mutation");
    }

    #[test]
    fn capture_and_latch_missing_page_errors() {
        let store = PrimaryPageStore::new();
        let mut log = TxnMutationLog::new();
        let err = store
            .capture_and_latch(&mut log, PageId::new(999))
            .unwrap_err();
        assert!(matches!(err, IndexError::MissingPage(_)));
        assert_eq!(log.page_mutations.len(), 0);
    }

    #[test]
    fn install_for_txn_records_new_pages_entry() {
        let store = PrimaryPageStore::new();
        let mut log = TxnMutationLog::new();
        let page = fresh_page_buf(PageId::new(77), PageType::IndexLeaf);
        store
            .install_for_txn(&mut log, PageId::new(77), page)
            .unwrap();
        assert_eq!(log.new_pages.len(), 1);
        assert_eq!(log.new_pages[0], (PageStoreKind::Primary, PageId::new(77)));
        assert!(store.contains(PageId::new(77)));
    }

    #[test]
    fn install_fresh_for_txn_records_new_pages_entry() {
        let store = PrimaryPageStore::new();
        let mut log = TxnMutationLog::new();
        store
            .install_fresh_for_txn(&mut log, PageId::new(88), PageType::IndexInternal)
            .unwrap();
        assert_eq!(log.new_pages.len(), 1);
        assert_eq!(log.new_pages[0], (PageStoreKind::Primary, PageId::new(88)));
    }

    #[test]
    fn remove_page_undoes_install() {
        let store = PrimaryPageStore::new();
        store
            .install(
                PageId::new(13),
                fresh_page_buf(PageId::new(13), PageType::IndexLeaf),
            )
            .unwrap();
        assert!(store.contains(PageId::new(13)));
        let removed = store.remove_page(PageId::new(13));
        assert!(removed.is_some());
        assert!(!store.contains(PageId::new(13)));
        // Idempotent second call.
        assert!(store.remove_page(PageId::new(13)).is_none());
    }

    #[test]
    fn restore_page_bytes_overwrites_current_state() {
        let store = PrimaryPageStore::new();
        store
            .install(
                PageId::new(21),
                fresh_page_buf(PageId::new(21), PageType::IndexLeaf),
            )
            .unwrap();
        // Mutate the page out-of-band.
        {
            let latch = store.latch(PageId::new(21)).unwrap();
            let mut g = latch.write();
            let mut leaf = LeafPageMut::open(g.as_mut()).unwrap();
            leaf.insert(k(1, RecordKind::Node, 1), v(1, 0)).unwrap();
        }
        // Restore to a known pre-image (fresh empty leaf).
        let pristine = fresh_page_buf(PageId::new(21), PageType::IndexLeaf);
        store
            .restore_page_bytes(PageId::new(21), pristine.as_ref())
            .unwrap();
        // Assert post-restore state matches pristine.
        let latch = store.latch(PageId::new(21)).unwrap();
        let g = latch.read();
        let leaf = LeafPageRef::open(g.as_ref()).unwrap();
        assert_eq!(leaf.entry_count(), 0);
    }

    #[test]
    fn wrong_page_type_rejected_on_open() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::Node);
        let err = LeafPageMut::open(page.as_mut()).unwrap_err();
        assert!(matches!(err, IndexError::CorruptPage { .. }));
    }

    // ─── internal-page codec ───

    #[test]
    fn internal_insert_lookup_roundtrip_single_page() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(42), PageId::new(100));
        assert_eq!(node.entry_count(), 0);
        assert_eq!(node.as_ref().first_child(), PageId::new(100));

        // Insert a few (key, child) pairs out of order.
        node.insert(k(1, RecordKind::Node, 30), PageId::new(230))
            .unwrap();
        node.insert(k(1, RecordKind::Node, 10), PageId::new(210))
            .unwrap();
        node.insert(k(1, RecordKind::Node, 20), PageId::new(220))
            .unwrap();
        assert_eq!(node.entry_count(), 3);

        let view = node.as_ref();
        let (k0, c0) = view.entry(0).unwrap();
        let (k1, c1) = view.entry(1).unwrap();
        let (k2, c2) = view.entry(2).unwrap();
        assert_eq!(k0.id, 10);
        assert_eq!(k1.id, 20);
        assert_eq!(k2.id, 30);
        assert_eq!(c0, PageId::new(210));
        assert_eq!(c1, PageId::new(220));
        assert_eq!(c2, PageId::new(230));
        // Header roundtrip check.
        assert_eq!(view.first_child(), PageId::new(100));
    }

    #[test]
    fn internal_ordering_preserved() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        // Insert across tenants and kinds to exercise the full key order.
        let keys = [
            k(1, RecordKind::Node, 5),
            k(1, RecordKind::Rel, 1),
            k(2, RecordKind::Node, 7),
            k(1, RecordKind::Node, 100),
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
        // Keys 10, 20, 30 with children 60, 70, 80.
        node.insert(k(1, RecordKind::Node, 10), PageId::new(60))
            .unwrap();
        node.insert(k(1, RecordKind::Node, 20), PageId::new(70))
            .unwrap();
        node.insert(k(1, RecordKind::Node, 30), PageId::new(80))
            .unwrap();
        let view = node.as_ref();
        // Strictly less than key[0] → first_child.
        assert_eq!(
            view.find_child(k(1, RecordKind::Node, 5)).unwrap(),
            PageId::new(50)
        );
        // Exactly key[0] → child[0].
        assert_eq!(
            view.find_child(k(1, RecordKind::Node, 10)).unwrap(),
            PageId::new(60)
        );
        // In [key[0], key[1]) → child[0].
        assert_eq!(
            view.find_child(k(1, RecordKind::Node, 15)).unwrap(),
            PageId::new(60)
        );
        // Exactly key[1] → child[1].
        assert_eq!(
            view.find_child(k(1, RecordKind::Node, 20)).unwrap(),
            PageId::new(70)
        );
        // After last key → last child.
        assert_eq!(
            view.find_child(k(1, RecordKind::Node, 1_000)).unwrap(),
            PageId::new(80)
        );
        // Empty internal node → first_child.
        let mut page2 = [0u8; PAGE_SIZE];
        let empty = InternalPageMut::init(&mut page2, PageId::new(2), PageId::new(42));
        assert_eq!(
            empty
                .as_ref()
                .find_child(k(1, RecordKind::Node, 0))
                .unwrap(),
            PageId::new(42)
        );
    }

    #[test]
    fn internal_full_page_rejects_new_child() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        for i in 0..INTERNAL_CAPACITY as u64 {
            node.insert(k(1, RecordKind::Node, i), PageId::new(1000 + i))
                .unwrap();
        }
        assert!(node.is_full());
        let err = node
            .insert(k(1, RecordKind::Node, u64::MAX), PageId::new(9999))
            .unwrap_err();
        assert!(matches!(err, IndexError::CorruptPage { .. }));
    }

    #[test]
    fn internal_duplicate_key_rejected() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(0));
        node.insert(k(1, RecordKind::Node, 10), PageId::new(200))
            .unwrap();
        let err = node
            .insert(k(1, RecordKind::Node, 10), PageId::new(201))
            .unwrap_err();
        assert!(matches!(err, IndexError::DuplicateKey { .. }));
    }

    #[test]
    fn internal_wrong_page_type_rejected_on_open() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let err = InternalPageMut::open(page.as_mut()).unwrap_err();
        assert!(matches!(err, IndexError::CorruptPage { .. }));
    }

    // ─── splits ───

    #[test]
    fn leaf_split_produces_two_sorted_halves() {
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        // Fill to capacity with ids 0, 1, ..., 202.
        for i in 0..LEAF_CAPACITY as u64 {
            leaf.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        assert_eq!(leaf.entry_count(), LEAF_CAPACITY);

        let (new_buf, promoted) = leaf.split_into(PageId::new(99)).unwrap();
        let expected_split = LEAF_CAPACITY / 2;

        // Left side shrunk; right side has the upper half.
        assert_eq!(leaf.entry_count(), expected_split);
        let right = LeafPageRef::open(new_buf.as_ref()).unwrap();
        assert_eq!(right.entry_count(), LEAF_CAPACITY - expected_split);

        // Promoted key = first key of right page.
        let (first_right_key, _, _) = right.entry(0).unwrap();
        assert_eq!(promoted, first_right_key);
        assert_eq!(promoted.id, u64::from(expected_split));

        // Ordering is preserved across both sides.
        let left = leaf.as_ref();
        let (last_left_key, _, _) = left.entry(expected_split - 1).unwrap();
        assert!(last_left_key < first_right_key);

        // Every original key is now reachable through exactly one of the two pages.
        for i in 0..LEAF_CAPACITY as u64 {
            let key = k(1, RecordKind::Node, i);
            let in_left = left.lookup(key).unwrap().is_some();
            let in_right = right.lookup(key).unwrap().is_some();
            assert!(in_left ^ in_right, "id {i} not uniquely placed");
        }

        // Right page inherits the correct header (page_id, page_type).
        assert_eq!(right.page_id(), PageId::new(99));
    }

    #[test]
    fn internal_split_promotes_middle_key() {
        let mut page = [0u8; PAGE_SIZE];
        let mut node = InternalPageMut::init(&mut page, PageId::new(1), PageId::new(500));
        // Fill to capacity (254 pairs) with keys 10, 20, 30, ..., and children starting at 501.
        for i in 0..INTERNAL_CAPACITY as u64 {
            node.insert(k(1, RecordKind::Node, 10 + i * 10), PageId::new(501 + i))
                .unwrap();
        }
        assert!(node.is_full());
        let mid = INTERNAL_CAPACITY / 2;
        let expected_promoted_id = 10 + u64::from(mid) * 10;

        let (new_buf, promoted) = node.split_into(PageId::new(999)).unwrap();
        assert_eq!(promoted.id, expected_promoted_id);

        // Left has `mid` pairs, same first_child.
        assert_eq!(node.entry_count(), mid);
        let left = node.as_ref();
        assert_eq!(left.first_child(), PageId::new(500));
        // Right's first_child is the promoted pair's child.
        let right = InternalPageRef::open(new_buf.as_ref()).unwrap();
        assert_eq!(right.first_child(), PageId::new(501 + u64::from(mid)));
        assert_eq!(right.entry_count(), INTERNAL_CAPACITY - mid - 1);

        // Every non-promoted pair ends up on exactly one side, ordering preserved.
        for i in 0..INTERNAL_CAPACITY as u64 {
            let key_id = 10 + i * 10;
            if key_id == expected_promoted_id {
                continue;
            }
            let target = k(1, RecordKind::Node, key_id);
            let child_left = (0..left.entry_count())
                .map(|idx| left.entry(idx).unwrap())
                .find(|(k, _)| *k == target);
            let child_right = (0..right.entry_count())
                .map(|idx| right.entry(idx).unwrap())
                .find(|(k, _)| *k == target);
            assert!(
                child_left.is_some() ^ child_right.is_some(),
                "key {key_id} placed on {}",
                if child_left.is_some() && child_right.is_some() {
                    "both"
                } else {
                    "neither"
                }
            );
        }
    }

    // ─── PrimaryIndex public API ───

    fn build_index() -> Arc<PrimaryIndex> {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        Arc::new(PrimaryIndex::new(txn_mgr, alloc, None).expect("fresh index"))
    }

    #[test]
    fn public_api_insert_lookup_roundtrip() {
        let idx = build_index();
        let key = k(1, RecordKind::Node, 42);
        let value = v(10, 5);
        assert_eq!(idx.lookup(key).unwrap(), None);
        idx.insert(key, value).unwrap();
        assert_eq!(idx.lookup(key).unwrap(), Some(value));
        // Duplicate-insert on live key → DuplicateKey error.
        let err = idx.insert(key, v(11, 0)).unwrap_err();
        assert!(matches!(err, IndexError::DuplicateKey { .. }));
    }

    #[test]
    fn public_api_upsert_returns_old_value() {
        let idx = build_index();
        let key = k(1, RecordKind::Node, 7);
        assert_eq!(idx.upsert(key, v(100, 0)).unwrap(), None);
        assert_eq!(idx.upsert(key, v(200, 1)).unwrap(), Some(v(100, 0)));
        assert_eq!(idx.lookup(key).unwrap(), Some(v(200, 1)));
    }

    #[test]
    fn public_api_remove_marks_tombstone() {
        let idx = build_index();
        let key = k(1, RecordKind::Node, 5);
        idx.insert(key, v(50, 0)).unwrap();
        assert_eq!(idx.remove(key).unwrap(), Some(v(50, 0)));
        assert_eq!(idx.lookup(key).unwrap(), None);
        // Reinsert the same key should succeed (tombstone revival).
        idx.insert(key, v(51, 1)).unwrap();
        assert_eq!(idx.lookup(key).unwrap(), Some(v(51, 1)));
        // Remove on absent key is a no-op (returns None).
        let absent = k(1, RecordKind::Node, 999);
        assert_eq!(idx.remove(absent).unwrap(), None);
    }

    #[test]
    fn root_split_allocates_new_root() {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let idx = PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap();
        // Fresh root is PageId(1).
        assert_eq!(idx.root().unwrap(), PageId::new(1));
        // Fill the root leaf. Inserting LEAF_CAPACITY+1 keys forces one split
        // that grows a new internal root.
        for i in 0..=u64::from(LEAF_CAPACITY) {
            idx.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        // Root now points to an internal page.
        let root_id = idx.root().unwrap();
        assert_ne!(root_id, PageId::new(1));
        let latch = idx.page_store.latch(root_id).unwrap();
        let g = latch.read();
        let hdr = read_header(g.as_ref()).unwrap();
        assert_eq!(hdr.page_type, PageType::IndexInternal.as_byte());
        drop(g);
        // Every inserted key is still findable.
        for i in 0..=u64::from(LEAF_CAPACITY) {
            assert_eq!(
                idx.lookup(k(1, RecordKind::Node, i)).unwrap(),
                Some(v(i, 0))
            );
        }
    }

    #[test]
    fn root_pointer_persisted_via_system_catalog() {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let idx = PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap();
        let root_before = idx.root().unwrap();
        // Read the MVCC-persisted pointer and confirm it matches.
        let txn = txn_mgr.begin(TenantId::SYSTEM);
        let bytes = txn.read(PRIMARY_INDEX_ROOT_KEY).unwrap();
        txn.abort();
        assert_eq!(bytes.len(), 8);
        let stored = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(stored, root_before.raw());

        // Force a root split, then verify the MVCC pointer was updated.
        for i in 0..=u64::from(LEAF_CAPACITY) {
            idx.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        let root_after = idx.root().unwrap();
        assert_ne!(root_after, root_before);
        let txn = txn_mgr.begin(TenantId::SYSTEM);
        let bytes = txn.read(PRIMARY_INDEX_ROOT_KEY).unwrap();
        txn.abort();
        let stored_after = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(stored_after, root_after.raw());
    }

    #[test]
    fn root_pointer_recoverable_across_new_primaryindex_instance() {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let idx1 = PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap();
        // Grow a bit so the root id is stable at whatever split picks.
        for i in 0..=u64::from(LEAF_CAPACITY) {
            idx1.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        let root1 = idx1.root().unwrap();
        drop(idx1);
        // Fresh PrimaryIndex against the same TxnManager — the pointer
        // survives (page contents don't, per DEC-17; that's M2.e).
        let idx2 = PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap();
        assert_eq!(idx2.root().unwrap(), root1);
    }

    #[test]
    fn insert_across_three_levels_is_consistent() {
        // With LEAF_CAPACITY = 203 and INTERNAL_CAPACITY = 254, inserting
        // ~40 K keys forces several splits and a deeper-than-two tree.
        let idx = build_index();
        let n: u64 = 40_000;
        for i in 0..n {
            idx.insert(k(1, RecordKind::Node, i), v(i, (i as u16) & 0xFF))
                .unwrap();
        }
        // Spot-check a handful.
        for &i in &[0_u64, 1, 1_000, 20_000, n - 1] {
            assert_eq!(
                idx.lookup(k(1, RecordKind::Node, i)).unwrap(),
                Some(v(i, (i as u16) & 0xFF))
            );
        }
        // Absent key is None.
        assert_eq!(idx.lookup(k(1, RecordKind::Node, u64::MAX)).unwrap(), None);
    }

    #[test]
    fn bootstrap_idempotent_on_empty_then_populated_mvcc() {
        let idx = build_index();
        // Pretend MVCC walk returns nothing.
        let stats0 = idx.bootstrap_from_mvcc(std::iter::empty()).unwrap();
        assert_eq!(stats0, BootstrapStats::default());
        // Populate by direct inserts.
        for i in 0..100u64 {
            idx.insert(k(1, RecordKind::Node, i), v(i, 0)).unwrap();
        }
        // Now bootstrap with the same entries (simulates re-index on restart).
        let entries: Vec<_> = (0..100u64)
            .map(|i| (k(1, RecordKind::Node, i), v(i, 0)))
            .collect();
        let stats1 = idx.bootstrap_from_mvcc(entries.clone()).unwrap();
        assert_eq!(stats1.indexed, 0, "already-indexed keys must be skipped");
        assert_eq!(stats1.skipped, 100);
        // Second run same result (idempotency).
        let stats2 = idx.bootstrap_from_mvcc(entries).unwrap();
        assert_eq!(stats2, stats1);
    }

    #[test]
    fn bootstrap_matches_manual_inserts() {
        let idx = build_index();
        let entries: Vec<_> = (0..500u64)
            .map(|i| (k(1, RecordKind::Node, i), v(i * 7, 0)))
            .collect();
        let stats = idx.bootstrap_from_mvcc(entries.clone()).unwrap();
        assert_eq!(stats.indexed, 500);
        assert_eq!(stats.skipped, 0);
        for (key, value) in entries {
            assert_eq!(idx.lookup(key).unwrap(), Some(value));
        }
    }

    #[test]
    fn index_page_payload_roundtrip() {
        let page_id = PageId::new(42);
        let tenant = TenantId::new(7);
        let mut page = [0u8; PAGE_SIZE];
        for (i, b) in page.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        let payload = encode_index_page_payload(page_id, tenant, &page);
        assert_eq!(payload.len(), 16 + PAGE_SIZE);
        let (pid, tid, bytes) = decode_index_page_payload(&payload).unwrap();
        assert_eq!(pid, page_id);
        assert_eq!(tid, tenant);
        assert_eq!(bytes.as_ref(), &page);
    }

    #[test]
    fn split_preserves_tenant_ordering_invariant() {
        // Mixed tenants in one leaf exercise the full 24-byte key order.
        let mut page = fresh_page_buf(PageId::new(1), PageType::IndexLeaf);
        let mut leaf = LeafPageMut::open(page.as_mut()).unwrap();
        // Interleave tenants 1 and 2 so the split crosses a tenant boundary.
        let mut all_keys = Vec::new();
        for i in 0..(LEAF_CAPACITY as u64 / 2) {
            all_keys.push(k(1, RecordKind::Node, i));
            all_keys.push(k(2, RecordKind::Node, i));
        }
        all_keys.sort();
        for (i, key) in all_keys.iter().copied().enumerate() {
            leaf.insert(key, v(i as u64, 0)).unwrap();
        }
        let (new_buf, promoted) = leaf.split_into(PageId::new(77)).unwrap();
        let right = LeafPageRef::open(new_buf.as_ref()).unwrap();
        let left = leaf.as_ref();

        // All left keys < promoted, all right keys >= promoted.
        for i in 0..left.entry_count() {
            let (lk, _, _) = left.entry(i).unwrap();
            assert!(lk < promoted, "left key {lk:?} >= promoted {promoted:?}");
        }
        for i in 0..right.entry_count() {
            let (rk, _, _) = right.entry(i).unwrap();
            assert!(rk >= promoted);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Property tests (W28 S602 — sibling of arcgraph-index
    // `secondary_btree.rs` §6 model-based suite; closes #511 R1 NIT +
    // ADR-165 M2). Same discipline, adapted to the PRIMARY tree.
    //
    // Scope guard (test ONLY the implemented surface): the primary index
    // exposes point `insert` / `upsert` / `remove` / `lookup`, leaf +
    // internal splits with grow-root, and a mark-only (tombstone) delete.
    // It is a UNIQUE-key map — at most one live `PageSlot` per key — and
    // has no range scan, no merge / rebalance, and no duplicate values,
    // so none of those are tested: that code does not exist. (No-op
    // trampolines forbidden per `feedback_noop_trampoline_anti_pattern`.)
    //
    // Oracle design (deliberately model-EQUALITY, NOT a consistency /
    // dedupe check):
    //  * `prop_primary_btree_matches_btreemap_model` keeps the real
    //    `PrimaryIndex` in lockstep with an independent
    //    `BTreeMap<PrimaryKey, PageSlot>` reference model, where a key's
    //    presence in the map means "has a live value". The model
    //    reproduces the documented op semantics EXACTLY: `insert` errors
    //    with `DuplicateKey` iff the key is live and otherwise revives a
    //    tombstone / inserts fresh; `upsert` returns the prior LIVE value
    //    (None over a tombstone) and always sets the value; `remove`
    //    tombstones and returns the prior live value. Equality is asserted
    //    on the full op return value AND on `lookup` after every op, plus
    //    a periodic + final sweep over EVERY key ever touched (so a
    //    tombstoned key that wrongly stayed live is caught). This is
    //    value-exact model-equality, never a dedupe / consistency check.
    //  * `primary_btree_build_determinism` uses the binary-equal
    //    reference-snapshot oracle (strictly stronger than dedupe-
    //    consistency; per `feedback_determinism_oracle_concurrency_tests`):
    //    two indexes built from the same op sequence must be byte-
    //    identical across the whole page store. Page bytes are
    //    deterministic here — the only header nonce candidates (`lsn`,
    //    `checksum`) are constant (`PageHeader::new` zeroes both and the
    //    in-memory codec never restamps them; `now_millis` feeds only the
    //    WAL record timestamp, and `build_index` attaches NO WAL), and
    //    page ids come from per-`(tenant, page_type)` monotonic counters
    //    that both fresh `PageAllocator`s replay identically. A readable
    //    structural-equality projection (in-order leaf entries + node
    //    shape) is asserted alongside the byte oracle.

    /// One generated operation against the index.
    #[derive(Debug, Clone)]
    enum ModelOp {
        Insert(PrimaryKey, PageSlot),
        Upsert(PrimaryKey, PageSlot),
        Remove(PrimaryKey),
        Lookup(PrimaryKey),
    }

    /// Logical (allocation-independent) snapshot of one tree node, used
    /// for the readable structural-equality leg of the determinism
    /// oracle. Leaf entries carry the value + tombstone flag so the
    /// projection is a strong structural oracle, not just a key list.
    #[derive(Debug, PartialEq, Eq)]
    enum LogicalNode {
        Leaf(Vec<(PrimaryKey, PageSlot, bool)>),
        Internal(Vec<PrimaryKey>),
    }

    /// In-order (left-to-right) traversal of the whole tree, descending
    /// internal nodes via `first_child` then each separator's child.
    fn walk_nodes(idx: &PrimaryIndex) -> Vec<LogicalNode> {
        let mut out = Vec::new();
        walk_rec(idx, idx.root().expect("root resolves"), &mut out);
        out
    }

    fn walk_rec(idx: &PrimaryIndex, page_id: PageId, out: &mut Vec<LogicalNode>) {
        let latch = idx.page_store().latch(page_id).expect("page latch");
        let g = latch.read();
        let header = read_header(g.as_ref()).expect("page header");
        if header.page_type == PageType::IndexLeaf.as_byte() {
            let leaf = LeafPageRef::open(g.as_ref()).expect("leaf open");
            let entries: Vec<(PrimaryKey, PageSlot, bool)> = (0..leaf.entry_count())
                .map(|i| leaf.entry(i).expect("leaf entry"))
                .collect();
            out.push(LogicalNode::Leaf(entries));
        } else if header.page_type == PageType::IndexInternal.as_byte() {
            let internal = InternalPageRef::open(g.as_ref()).expect("internal open");
            let mut children = vec![internal.first_child()];
            let mut separators = Vec::new();
            for i in 0..internal.entry_count() {
                let (key, child) = internal.entry(i).expect("internal entry");
                separators.push(key);
                children.push(child);
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
    /// entries — the entry survives a tombstone in this mark-only tree).
    fn walk_leaf_keys(idx: &PrimaryIndex) -> Vec<PrimaryKey> {
        let mut keys = Vec::new();
        for node in walk_nodes(idx) {
            if let LogicalNode::Leaf(entries) = node {
                keys.extend(entries.into_iter().map(|(key, _, _)| key));
            }
        }
        keys
    }

    /// Keys with a live (non-tombstoned) value — i.e. the set a caller
    /// can actually retrieve via `lookup`.
    fn live_keys_in_tree(idx: &PrimaryIndex) -> BTreeSet<PrimaryKey> {
        let mut set = BTreeSet::new();
        for node in walk_nodes(idx) {
            if let LogicalNode::Leaf(entries) = node {
                for (key, _, tombstoned) in entries {
                    if !tombstoned {
                        set.insert(key);
                    }
                }
            }
        }
        set
    }

    /// Byte-for-byte snapshot of every installed page, keyed by raw page
    /// id — the binary-equal reference snapshot for the determinism
    /// oracle.
    fn snapshot_all_pages(idx: &PrimaryIndex) -> BTreeMap<u64, Vec<u8>> {
        let mut out = BTreeMap::new();
        for entry in idx.page_store().pages.iter() {
            out.insert(entry.key().raw(), entry.value().read().as_ref().to_vec());
        }
        out
    }

    /// Replay an op sequence into a fresh index. Lookups are no-ops on
    /// state; `insert` over a live key legitimately errors (deterministic)
    /// so its result is discarded rather than `expect`ed. Used to build
    /// twice for the determinism oracle.
    fn build_from_ops(ops: &[ModelOp]) -> Arc<PrimaryIndex> {
        let idx = build_index();
        for op in ops {
            match op {
                ModelOp::Insert(key, value) => {
                    let _ = idx.insert(*key, *value);
                }
                ModelOp::Upsert(key, value) => {
                    let _ = idx.upsert(*key, *value);
                }
                ModelOp::Remove(key) => {
                    let _ = idx.remove(*key);
                }
                ModelOp::Lookup(_) => {}
            }
        }
        idx
    }

    // ── strategies ──
    //
    // `arb_key_small` / `_medium` keep the key domain small + collision-
    // heavy so insert-dup, tombstone-revive, and upsert-replace are all
    // exercised densely. `arb_key_wide` spreads ids so a large unique set
    // is easy to draw for the split test. Values vary so upsert-replace
    // is observable. The kind byte + both tenants exercise the full
    // 24-byte tenant-major / kind / id key order.

    fn arb_record_kind() -> impl Strategy<Value = RecordKind> {
        prop_oneof![Just(RecordKind::Node), Just(RecordKind::Rel)]
    }

    fn arb_slot() -> impl Strategy<Value = PageSlot> {
        // page 1.. avoids the PageId::ZERO sentinel; the value is opaque
        // to the index, but distinct values let upsert-replace be seen.
        (1u64..=64, 0u16..=15).prop_map(|(p, s)| PageSlot::new(PageId::new(p), SlotId(s)))
    }

    fn arb_key_small() -> impl Strategy<Value = PrimaryKey> {
        (
            prop::sample::select(vec![1u64, 2]),
            arb_record_kind(),
            0u64..6,
        )
            .prop_map(|(t, kind, id)| PrimaryKey::new(TenantId::new(t), kind, id))
    }

    fn arb_key_medium() -> impl Strategy<Value = PrimaryKey> {
        (
            prop::sample::select(vec![1u64, 2]),
            arb_record_kind(),
            0u64..30,
        )
            .prop_map(|(t, kind, id)| PrimaryKey::new(TenantId::new(t), kind, id))
    }

    fn arb_key_wide() -> impl Strategy<Value = PrimaryKey> {
        (
            prop::sample::select(vec![1u64, 2]),
            arb_record_kind(),
            0u64..1_000_000,
        )
            .prop_map(|(t, kind, id)| PrimaryKey::new(TenantId::new(t), kind, id))
    }

    fn arb_op_seq_small() -> impl Strategy<Value = Vec<ModelOp>> {
        let op = prop_oneof![
            3 => (arb_key_small(), arb_slot()).prop_map(|(k, v)| ModelOp::Insert(k, v)),
            2 => (arb_key_small(), arb_slot()).prop_map(|(k, v)| ModelOp::Upsert(k, v)),
            2 => arb_key_small().prop_map(ModelOp::Remove),
            1 => arb_key_small().prop_map(ModelOp::Lookup),
        ];
        prop::collection::vec(op, 1..160)
    }

    fn arb_op_seq_medium() -> impl Strategy<Value = Vec<ModelOp>> {
        let op = prop_oneof![
            3 => (arb_key_medium(), arb_slot()).prop_map(|(k, v)| ModelOp::Insert(k, v)),
            2 => (arb_key_medium(), arb_slot()).prop_map(|(k, v)| ModelOp::Upsert(k, v)),
            2 => arb_key_medium().prop_map(ModelOp::Remove),
            1 => arb_key_medium().prop_map(ModelOp::Lookup),
        ];
        prop::collection::vec(op, 1..256)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            .. ProptestConfig::default()
        })]

        /// Model-equality oracle. A random insert / upsert / remove /
        /// lookup sequence over a small (collision-heavy) key domain must
        /// keep the real index in lockstep with an independent
        /// `BTreeMap<PrimaryKey, PageSlot>` reference model after EVERY op
        /// — exact op-return equality + exact `lookup` equality, plus a
        /// periodic and final sweep over every key ever touched. This is
        /// value-exact model-equality, not a consistency / dedupe check.
        #[test]
        fn prop_primary_btree_matches_btreemap_model(ops in arb_op_seq_small()) {
            let idx = build_index();
            let mut model: BTreeMap<PrimaryKey, PageSlot> = BTreeMap::new();
            let mut seen: BTreeSet<PrimaryKey> = BTreeSet::new();

            for (i, op) in ops.iter().enumerate() {
                match op {
                    ModelOp::Insert(key, value) => {
                        seen.insert(*key);
                        let expect_dup = model.contains_key(key);
                        let res = idx.insert(*key, *value);
                        if expect_dup {
                            prop_assert!(
                                matches!(&res, Err(IndexError::DuplicateKey { key: dk }) if *dk == *key),
                                "insert on live key {:?} must be DuplicateKey, got {:?}",
                                key,
                                res
                            );
                        } else {
                            prop_assert!(
                                res.is_ok(),
                                "insert on absent/tombstoned key {:?} must succeed, got {:?}",
                                key,
                                res
                            );
                            model.insert(*key, *value);
                        }
                        prop_assert_eq!(
                            idx.lookup(*key).expect("lookup ok"),
                            model.get(key).copied(),
                            "post-insert lookup mismatch at {:?}",
                            key
                        );
                    }
                    ModelOp::Upsert(key, value) => {
                        seen.insert(*key);
                        let expect_prev = model.get(key).copied();
                        let real_prev = idx.upsert(*key, *value).expect("upsert ok");
                        prop_assert_eq!(
                            real_prev,
                            expect_prev,
                            "upsert previous-value mismatch at {:?}",
                            key
                        );
                        model.insert(*key, *value);
                        prop_assert_eq!(
                            idx.lookup(*key).expect("lookup ok"),
                            Some(*value),
                            "post-upsert lookup mismatch at {:?}",
                            key
                        );
                    }
                    ModelOp::Remove(key) => {
                        seen.insert(*key);
                        let expect_prev = model.remove(key);
                        let real_prev = idx.remove(*key).expect("remove ok");
                        prop_assert_eq!(
                            real_prev,
                            expect_prev,
                            "remove previous-value mismatch at {:?}",
                            key
                        );
                        prop_assert_eq!(
                            idx.lookup(*key).expect("lookup ok"),
                            None,
                            "post-remove lookup must be None at {:?}",
                            key
                        );
                    }
                    ModelOp::Lookup(key) => {
                        seen.insert(*key);
                        prop_assert_eq!(
                            idx.lookup(*key).expect("lookup ok"),
                            model.get(key).copied(),
                            "lookup mismatch at {:?}",
                            key
                        );
                    }
                }

                // Periodic full sweep over every key ever touched — a
                // tombstoned key that wrongly stayed live (or vice versa)
                // is caught here, not just on the key just operated on.
                if i % 32 == 0 {
                    for key in &seen {
                        prop_assert_eq!(
                            idx.lookup(*key).expect("lookup ok"),
                            model.get(key).copied(),
                            "periodic full-sweep mismatch at {:?}",
                            key
                        );
                    }
                }
            }

            // Final full sweep.
            for key in &seen {
                prop_assert_eq!(
                    idx.lookup(*key).expect("lookup ok"),
                    model.get(key).copied(),
                    "final full-sweep mismatch at {:?}",
                    key
                );
            }

            // And the in-order leaf-key walk stays strictly ascending
            // (tombstoned entries are retained but still ordered).
            let walked = walk_leaf_keys(&idx);
            for w in walked.windows(2) {
                prop_assert!(w[0] < w[1], "leaf walk not ascending: {:?} !< {:?}", w[0], w[1]);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

        /// Split-membership invariant. `LEAF_CAPACITY` = 203, so two
        /// leaves hold at most 406 keys; ≥407 unique keys MUST grow the
        /// tree to ≥3 leaves (≥2 splits). The retrievable set must then
        /// exactly equal the inserted set — no loss, no phantom — and
        /// each key must resolve to its single inserted value.
        #[test]
        fn prop_primary_split_preserves_membership(
            keys in prop::collection::btree_set(arb_key_wide(), 407..=470),
        ) {
            let idx = build_index();
            let mut expected: BTreeMap<PrimaryKey, PageSlot> = BTreeMap::new();
            for (i, key) in keys.iter().enumerate() {
                // Distinct value per key so a mis-routed split that keeps
                // the key but loses the value is also caught.
                let value = PageSlot::new(PageId::new(i as u64 + 1), SlotId((i as u16) & 0x0F));
                idx.insert(*key, value).expect("insert ok");
                expected.insert(*key, value);
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

            // No loss: every inserted key resolves to exactly its value.
            for (key, value) in &expected {
                prop_assert_eq!(
                    idx.lookup(*key).expect("lookup ok"),
                    Some(*value),
                    "lost key {:?}",
                    key
                );
            }
            // No phantom: the live key set equals the inserted key set.
            let inserted: BTreeSet<PrimaryKey> = keys.iter().copied().collect();
            prop_assert_eq!(
                live_keys_in_tree(&idx),
                inserted,
                "retrievable key set != inserted key set"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

        /// Build-determinism. Two indexes built from the SAME op sequence
        /// must be identical. Asserted at two levels: the readable
        /// structural projection (in-order leaf entries + full
        /// `LogicalNode` shape), and a byte-for-byte page-store snapshot
        /// (the binary-equal reference-snapshot oracle, strictly stronger
        /// than dedupe-consistency). No per-run nonce reaches the compared
        /// bytes — see the module-level oracle note above.
        #[test]
        fn primary_btree_build_determinism(ops in arb_op_seq_medium()) {
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
}
