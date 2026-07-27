//! Slot-array (slotted-page) codec for `NodeRecord` and `RelRecord`.
//!
//! Layout (design-v2 §3.2):
//!
//! ```text
//!  0..  40  PageHeader                      (fixed)
//! 40..  40 + 4*slot_count   Slot array       (grows forward)
//!          Slot = [u16 offset, u16 length]   (0,0 = tombstoned)
//!  ...                                       (free space)
//!  <-end>   Record[slot_count-1] ... Record[0]  (grows backward)
//! ```
//!
//! Records are fixed size: `NodeRecord = 64 B`, `RelRecord = 96 B`.
//! Capacity in slots per 8 KiB page is therefore:
//!
//! | Record     | Slot+Record | Capacity |
//! |------------|-------------|----------|
//! | NodeRecord |  4 + 64 = 68 |      119 |
//! | RelRecord  |  4 + 96 = 100 |      81 |
//!
//! Per-op latency budget (§4.4, 5 K TPS envelope): insert/read ≤ 40 ns —
//! slot lookup is two loads (slot_count, slot`i`) plus one memcpy-sized
//! record copy. The page buffer is pinned by the caller; this codec
//! performs no I/O.
//!
//! **Overflow policy.** This codec does not chain pages or compact
//! garbage. When `insert_*` cannot fit a record, it returns
//! [`PageError::Full`] and the caller (`crud.rs` in M2-21/22) allocates
//! a fresh page. Tombstoned slots are not reused here either — the
//! callers of M2-26 tombstone-delete writes a new version record via
//! MVCC and keeps the old slot tombstoned until a future compaction
//! pass (M2.d).

use std::mem::size_of;

use arcgraph_core::Lsn;
use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord};
use thiserror::Error;

// ---- constants ----------------------------------------------------------

/// On-disk size of a single slot directory entry: `u16 offset || u16 length`.
pub const SLOT_SIZE: usize = 4;
/// Durable provenance for a record that was deleted before the M4 swap.
///
/// `(0, 0)` remains a reusable sparse hole. `(0, u16::MAX)` is deliberately
/// outside every valid record length and therefore survives restart as a
/// never-revivable tombstone.
pub const PERMANENT_TOMBSTONE_LEN: u16 = u16::MAX;

/// First byte of the slot directory (immediately after the page header).
pub const SLOT_AREA_START: usize = PageHeader::SIZE;

/// Bytes of slot+record storage available on a freshly initialized page.
pub const PAGE_BODY_BYTES: usize = PAGE_SIZE - PageHeader::SIZE;

/// Maximum number of `NodeRecord`s that fit in one 8 KiB page.
pub const NODE_CAPACITY: u16 = (PAGE_BODY_BYTES / (SLOT_SIZE + NodeRecord::SIZE)) as u16;

/// Maximum number of `RelRecord`s that fit in one 8 KiB page.
pub const REL_CAPACITY: u16 = (PAGE_BODY_BYTES / (SLOT_SIZE + RelRecord::SIZE)) as u16;

const _: () = assert!(NODE_CAPACITY == 119);
const _: () = assert!(REL_CAPACITY == 81);

/// Maximum number of slot-directory entries whose 4-byte entries fit
/// inside a single page. A header's `slot_count` above this would let
/// `SlottedPageRef::slot_entry` compute a directory offset past the
/// page buffer and panic on an out-of-bounds slice index (#592).
///
/// Equals `(PAGE_SIZE - SLOT_AREA_START) / SLOT_SIZE`. `slot_count` is a
/// `u16` read verbatim from the on-disk header (max `u16::MAX`), so it
/// MUST be validated against this bound before any slot is addressed.
pub const MAX_SLOTS: u16 = ((PAGE_SIZE - SLOT_AREA_START) / SLOT_SIZE) as u16;

const _: () = assert!(MAX_SLOTS == 2038);

/// Maximum payload bytes a single property bag may occupy in a
/// [`PageType::PropSlotted`] page (v2 M1 — W-B1 slotted small-blob
/// packing). A bag must fit in a FRESH page alongside its 4-byte slot
/// entry: `PAGE_BODY_BYTES - SLOT_SIZE = 8148`. Bags larger than this
/// keep the DEC-4 chained-blob representation (`blob.rs`), which is the
/// design §M1.2 overflow tail ("a bag whose JSON exceeds a page's free
/// space keeps a chain").
pub const PROP_BAG_MAX_BYTES: usize = PAGE_BODY_BYTES - SLOT_SIZE;

const _: () = assert!(PROP_BAG_MAX_BYTES == 8148);
// The directory for `MAX_SLOTS` entries must end at or before the page
// boundary: the last entry (index MAX_SLOTS-1) spans
// `SLOT_AREA_START + (MAX_SLOTS-1)*SLOT_SIZE .. + SLOT_SIZE`, which must
// be `<= PAGE_SIZE`. This is the formal statement of why `slot_count`
// values `<= MAX_SLOTS` can never index out of bounds in `slot_entry`.
const _: () = assert!(SLOT_AREA_START + (MAX_SLOTS as usize) * SLOT_SIZE <= PAGE_SIZE);

// ---- slot id ------------------------------------------------------------

/// Zero-based slot directory index. Stable across tombstoning but not
/// across a future page-compaction pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u16);

impl SlotId {
    /// Raw u16 view.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

// ---- errors -------------------------------------------------------------

/// Codec-local errors. Translated to `ArcGraphError` at the `crud.rs`
/// boundary; kept local so the core error enum stays frozen.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PageError {
    /// Not enough bytes left for a new slot + record.
    #[error("slotted page full: needs {needed} bytes, free={free}")]
    Full {
        /// Bytes the insert needs (slot + record).
        needed: u16,
        /// Bytes actually free between slot-array end and record-area start.
        free: u16,
    },
    /// Slot index was ≥ `slot_count`.
    #[error("slot {slot} out of range (count={count})")]
    SlotOutOfRange {
        /// Caller-provided index.
        slot: u16,
        /// Current slot_count from the page header.
        count: u16,
    },
    /// Slot index referred to a tombstoned entry.
    #[error("slot {0} is tombstoned")]
    SlotTombstoned(u16),
    /// Page header reports a `page_type` that does not match the API called.
    #[error("wrong page type: got {got}, expected {expected}")]
    WrongPageType {
        /// Byte read from the header.
        got: u8,
        /// Byte this API required.
        expected: u8,
    },
    /// Internal structural invariant violated: slot offset outside record
    /// area or length does not match the fixed record size for the page type.
    #[error("page format error: {0}")]
    Format(String),
    /// Propagated decode error from `arcgraph-core` (bad magic, bad version).
    #[error("record decode: {0}")]
    RecordDecode(String),
    /// CRC32C over the page body (bytes 40..8192) did not match the
    /// `checksum` field in the header. Indicates bit-rot, a torn write,
    /// or an uninitialized page.
    #[error("page checksum mismatch: header=0x{stored:08x}, computed=0x{computed:08x}")]
    ChecksumMismatch {
        /// Value carried in `PageHeader::checksum`.
        stored: u32,
        /// Value recomputed over bytes 40..8192.
        computed: u32,
    },
    /// A direct-address publish arrived behind the slot's installed version.
    /// Rejecting it prevents an out-of-order post-fsync publisher from making
    /// an older committed version authoritative.
    #[error(
        "stale direct-address publish at slot {slot}: incoming LSN {incoming} < current LSN {current}"
    )]
    StalePublish {
        /// Direct-address slot.
        slot: u16,
        /// LSN already installed in the slot.
        current: u64,
        /// LSN carried by the attempted publish.
        incoming: u64,
    },
}

// ---- header validation -------------------------------------------------

/// Reject a header whose `slot_count` exceeds the page's slot capacity.
///
/// `slot_count` is an attacker-controllable `u16` read verbatim from the
/// on-disk header, while only [`MAX_SLOTS`] directory entries fit in a
/// page. The `slot_count` field lives at header bytes `36..38`, *outside*
/// the CRC body range (bytes `40..`), so forging it preserves the body
/// checksum — a crafted page can pass the CRC gate yet carry
/// `slot_count == u16::MAX`. Without this bound, [`SlottedPageRef::slot_entry`]
/// would compute `SLOT_AREA_START + slot * SLOT_SIZE` past the buffer and
/// panic on the slice index. Validating at attach time rejects the frame
/// up-front. See #592; same class as the #577/#594 PackStream
/// length-prefix fix (validate an untrusted length before indexing).
fn validate_slot_count(slot_count: u16) -> Result<(), PageError> {
    if slot_count > MAX_SLOTS {
        return Err(PageError::Format(format!(
            "slot_count {slot_count} exceeds page capacity (max {MAX_SLOTS})"
        )));
    }
    Ok(())
}

// ---- mutable slotted page ----------------------------------------------

/// Mutable view over a single 8 KiB page buffer.
///
/// The buffer is borrowed from the caller (typically a
/// `BufferPool::pin_write` frame); this type owns no heap allocation.
#[derive(Debug)]
pub struct SlottedPage<'a> {
    bytes: &'a mut [u8],
}

impl<'a> SlottedPage<'a> {
    /// Initialize an 8 KiB slice as a fresh slotted page holding records
    /// of `page_type`. Writes the header, zeroes the body. `page_type`
    /// must be [`PageType::Node`], [`PageType::Rel`], or
    /// [`PageType::PropSlotted`] (v2 M1 — variable-length property bags).
    ///
    /// # Errors
    /// Returns [`PageError::WrongPageType`] if `header.page_type` is not
    /// a supported slotted type, [`PageError::Format`] if `bytes.len() != PAGE_SIZE`.
    pub fn init(bytes: &'a mut [u8], header: PageHeader) -> Result<Self, PageError> {
        if bytes.len() != PAGE_SIZE {
            return Err(PageError::Format(format!(
                "page buffer must be {PAGE_SIZE} bytes, got {}",
                bytes.len()
            )));
        }
        match PageType::from_byte(header.page_type) {
            Ok(PageType::Node) | Ok(PageType::Rel) | Ok(PageType::PropSlotted) => {}
            Ok(other) => {
                return Err(PageError::WrongPageType {
                    got: other.as_byte(),
                    expected: PageType::Node.as_byte(),
                });
            }
            Err(_) => {
                return Err(PageError::WrongPageType {
                    got: header.page_type,
                    expected: PageType::Node.as_byte(),
                });
            }
        }

        let mut fresh = header;
        fresh.slot_count = 0;
        // Body spans SLOT_AREA_START..PAGE_SIZE; all of it is free.
        fresh.free_space = PAGE_BODY_BYTES as u16;
        let hdr = fresh.to_bytes();
        bytes[..PageHeader::SIZE].copy_from_slice(&hdr);
        for b in &mut bytes[PageHeader::SIZE..] {
            *b = 0;
        }
        let mut page = Self { bytes };
        page.recompute_checksum();
        Ok(page)
    }

    /// Attach to an already-initialized [`PageType::PropSlotted`] page
    /// WITHOUT the body-CRC pass — the mutable sibling of
    /// [`SlottedPageRef::open_prop_trusted`], for buffers the caller
    /// exclusively owns and whose CRC is valid by construction (every
    /// mutation path recomputes it; v2 M1 uses this for the
    /// txn-exclusive slotted-scratch append path where a per-append
    /// 8 KiB CRC verify would double the [`Self::insert_bag`] cost for
    /// zero trust gain). Validates length, header decode, the
    /// `PropSlotted` type byte, and the #592 `slot_count` bound.
    ///
    /// # Errors
    /// [`PageError::Format`] / [`PageError::WrongPageType`] as
    /// [`SlottedPageRef::open_prop_trusted`].
    pub fn open_prop_trusted(bytes: &'a mut [u8]) -> Result<Self, PageError> {
        SlottedPageRef::open_prop_trusted(bytes)?;
        Ok(Self { bytes })
    }

    /// Attach to an already-initialized page. Validates the header and
    /// the body CRC32C (see [`PageError::ChecksumMismatch`]).
    pub fn open(bytes: &'a mut [u8]) -> Result<Self, PageError> {
        Self::validate_len(bytes.len())?;
        let hdr_bytes: &[u8; PageHeader::SIZE] = (&bytes[..PageHeader::SIZE])
            .try_into()
            .expect("SLOT_AREA_START == PageHeader::SIZE");
        let hdr =
            PageHeader::from_bytes(hdr_bytes).map_err(|e| PageError::Format(e.to_string()))?;
        let computed = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
        if computed != hdr.checksum {
            return Err(PageError::ChecksumMismatch {
                stored: hdr.checksum,
                computed,
            });
        }
        // Reject a forged/corrupt `slot_count` before any reader can
        // address a slot past the page buffer (#592).
        validate_slot_count(hdr.slot_count)?;
        Ok(Self { bytes })
    }

    /// Decoded header.
    #[must_use]
    pub fn header(&self) -> PageHeader {
        SlottedPageRef::from_bytes_unchecked(self.bytes).header()
    }

    /// Full redo sub-LSN currently stamped on this page.
    #[must_use]
    pub fn page_lsn(&self) -> Lsn {
        Lsn::new(self.header().lsn)
    }

    /// Apply one physiological mutation iff `op_lsn` is newer than the
    /// page's full sub-LSN, then stamp exactly that sub-LSN. Same-page
    /// ops from one bundle therefore apply independently instead of the
    /// second and later ops self-skipping at a shared commit LSN.
    pub fn apply_redo_if_newer<E>(
        &mut self,
        op_lsn: Lsn,
        apply: impl FnOnce(&mut Self) -> Result<(), E>,
    ) -> Result<bool, E> {
        if op_lsn.raw() <= self.page_lsn().raw() {
            return Ok(false);
        }
        apply(self)?;
        let mut header = self.header();
        header.lsn = op_lsn.raw();
        self.bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        Ok(true)
    }

    /// Number of slot-directory entries currently in use (including tombstones).
    #[must_use]
    pub fn slot_count(&self) -> u16 {
        self.header().slot_count
    }

    /// Bytes free between slot-array end and record-area start.
    #[must_use]
    pub fn free_space(&self) -> u16 {
        self.header().free_space
    }

    /// Insert a `NodeRecord`, returning its freshly assigned slot id.
    ///
    /// # Errors
    /// [`PageError::WrongPageType`] if the page is not a `Node` page;
    /// [`PageError::Full`] if there is insufficient free space;
    /// [`PageError::Format`] if the header carries an inconsistent
    /// record-area watermark (#16).
    pub fn insert_node(&mut self, rec: &NodeRecord) -> Result<SlotId, PageError> {
        self.require_page_type(PageType::Node)?;
        let bytes = rec.to_bytes();
        self.insert_raw(&bytes)
    }

    /// Insert a `RelRecord`.
    ///
    /// # Errors
    /// As [`Self::insert_node`], but requires a `Rel` page.
    pub fn insert_rel(&mut self, rec: &RelRecord) -> Result<SlotId, PageError> {
        self.require_page_type(PageType::Rel)?;
        let bytes = rec.to_bytes();
        self.insert_raw(&bytes)
    }

    /// Write a node record at its direct-addressed slot.
    ///
    /// Unlike [`Self::insert_node`], both the directory entry and record
    /// offset are pure functions of `slot`; write arrival order cannot alter
    /// page bytes. Growing past the current high-water mark reserves and
    /// zeroes every intermediate fixed-size slot. A zero `(0, 0)` entry is a
    /// reusable sparse hole; migrated deletions use the distinct durable
    /// `(0, u16::MAX)` provenance marker.
    ///
    /// # Errors
    ///
    /// [`PageError::WrongPageType`] if this is not a node page,
    /// [`PageError::SlotOutOfRange`] if `slot` is outside
    /// [`NODE_CAPACITY`], or [`PageError::Format`] if an existing page does
    /// not have the canonical fixed-slot layout.
    pub fn write_node_at_slot(&mut self, slot: SlotId, rec: &NodeRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Node)?;
        self.guard_monotone_publish(slot, rec.created_lsn, |page, slot| {
            page.read_node(slot)
                .map(|record| record.map(|record| record.created_lsn))
        })?;
        self.write_fixed_at_slot(
            slot,
            &rec.to_bytes(),
            NodeRecord::SIZE as u16,
            NODE_CAPACITY,
        )
    }

    /// Write a relationship record at its direct-addressed slot.
    ///
    /// This is the relationship-store counterpart of
    /// [`Self::write_node_at_slot`].
    ///
    /// # Errors
    ///
    /// As [`Self::write_node_at_slot`], using [`REL_CAPACITY`].
    pub fn write_rel_at_slot(&mut self, slot: SlotId, rec: &RelRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Rel)?;
        self.guard_monotone_publish(slot, rec.created_lsn, |page, slot| {
            page.read_rel(slot)
                .map(|record| record.map(|record| record.created_lsn))
        })?;
        self.write_fixed_at_slot(slot, &rec.to_bytes(), RelRecord::SIZE as u16, REL_CAPACITY)
    }

    fn guard_monotone_publish(
        &self,
        slot: SlotId,
        incoming: u64,
        read_lsn: impl FnOnce(&SlottedPageRef<'_>, SlotId) -> Result<Option<u64>, PageError>,
    ) -> Result<(), PageError> {
        if slot.0 >= self.slot_count() {
            return Ok(());
        }
        if self.as_ref().is_permanent_tombstone(slot)? {
            return Err(PageError::SlotTombstoned(slot.0));
        }
        if let Some(current) = read_lsn(&self.as_ref(), slot)?
            && incoming < current
        {
            return Err(PageError::StalePublish {
                slot: slot.0,
                current,
                incoming,
            });
        }
        Ok(())
    }

    /// Overwrite a previously-inserted node record in place. Slot id and
    /// on-disk offset are preserved; only the record bytes change.
    ///
    /// # Errors
    /// [`PageError::SlotOutOfRange`], [`PageError::SlotTombstoned`],
    /// [`PageError::WrongPageType`], or [`PageError::Format`] if the
    /// existing slot length mismatches `NodeRecord::SIZE`.
    pub fn update_node(&mut self, slot: SlotId, rec: &NodeRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Node)?;
        let bytes = rec.to_bytes();
        self.overwrite(slot, &bytes, NodeRecord::SIZE as u16)
    }

    /// Overwrite a previously-inserted rel record in place.
    ///
    /// # Errors
    /// As [`Self::update_node`], but requires a `Rel` page.
    pub fn update_rel(&mut self, slot: SlotId, rec: &RelRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Rel)?;
        let bytes = rec.to_bytes();
        self.overwrite(slot, &bytes, RelRecord::SIZE as u16)
    }

    /// Install a node at an exact physiological redo target.
    ///
    /// A target equal to `slot_count` appends a new record; an earlier
    /// live target overwrites in place. Reservation aborts can leave a
    /// legal sparse target or a released tombstone, so redo materializes
    /// intervening tombstone directory entries and may revive a tombstone.
    pub fn put_node_at(&mut self, slot: SlotId, rec: &NodeRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Node)?;
        if slot.0 < self.slot_count() && self.as_ref().is_permanent_tombstone(slot)? {
            return Err(PageError::SlotTombstoned(slot.0));
        }
        let bytes = rec.to_bytes();
        self.put_fixed_at(slot, &bytes, NodeRecord::SIZE as u16)
    }

    /// Install a relationship at an exact physiological redo target.
    pub fn put_rel_at(&mut self, slot: SlotId, rec: &RelRecord) -> Result<(), PageError> {
        self.require_page_type(PageType::Rel)?;
        if slot.0 < self.slot_count() && self.as_ref().is_permanent_tombstone(slot)? {
            return Err(PageError::SlotTombstoned(slot.0));
        }
        let bytes = rec.to_bytes();
        self.put_fixed_at(slot, &bytes, RelRecord::SIZE as u16)
    }

    /// Persist a pre-M4 node deletion at an exact direct-addressed slot.
    ///
    /// The marker is distinct from a sparse hole and cannot be overwritten by
    /// either a runtime direct publish or physiological redo.
    pub fn permanent_tombstone_node_at_slot(
        &mut self,
        slot: SlotId,
        tombstone_lsn: Lsn,
    ) -> Result<(), PageError> {
        self.require_page_type(PageType::Node)?;
        self.write_fixed_permanent_tombstone(
            slot,
            tombstone_lsn,
            NodeRecord::SIZE as u16,
            NODE_CAPACITY,
        )
    }

    /// Persist a pre-M4 relationship deletion at an exact direct-addressed
    /// slot. See [`Self::permanent_tombstone_node_at_slot`].
    pub fn permanent_tombstone_rel_at_slot(
        &mut self,
        slot: SlotId,
        tombstone_lsn: Lsn,
    ) -> Result<(), PageError> {
        self.require_page_type(PageType::Rel)?;
        self.write_fixed_permanent_tombstone(
            slot,
            tombstone_lsn,
            RelRecord::SIZE as u16,
            REL_CAPACITY,
        )
    }

    /// Materialize the same durable tombstone provenance for a fixed-size
    /// `PropSlotted` owner cell. This deliberately delegates to PRE-B.4's
    /// single `(offset=0, length=u16::MAX)` encoding; owner stores do not
    /// invent a second retirement marker.
    pub(crate) fn permanent_tombstone_fixed_bag_at_slot(
        &mut self,
        slot: SlotId,
        tombstone_lsn: Lsn,
        fixed_len: u16,
        capacity: u16,
    ) -> Result<(), PageError> {
        self.require_page_type(PageType::PropSlotted)?;
        self.write_fixed_permanent_tombstone(slot, tombstone_lsn, fixed_len, capacity)
    }

    /// Mark a slot tombstoned (offset=0, length=0). The underlying
    /// record bytes remain until a future compaction pass; they are
    /// never read because [`Self::read_node`] / [`Self::read_rel`]
    /// return `Ok(None)` once a slot is tombstoned.
    ///
    /// # Errors
    /// [`PageError::SlotOutOfRange`] if `slot` is past the high-water mark.
    /// Calling tombstone on an already-tombstoned slot is idempotent and returns `Ok`.
    pub fn tombstone(&mut self, slot: SlotId) -> Result<(), PageError> {
        let count = self.slot_count();
        if slot.0 >= count {
            return Err(PageError::SlotOutOfRange {
                slot: slot.0,
                count,
            });
        }
        let slot_off = SLOT_AREA_START + (slot.0 as usize) * SLOT_SIZE;
        self.bytes[slot_off..slot_off + 2].copy_from_slice(&0u16.to_le_bytes());
        self.bytes[slot_off + 2..slot_off + 4].copy_from_slice(&0u16.to_le_bytes());
        self.recompute_checksum();
        Ok(())
    }

    /// Read-only reinterpretation of this page.
    #[must_use]
    pub fn as_ref(&self) -> SlottedPageRef<'_> {
        SlottedPageRef::from_bytes_unchecked(self.bytes)
    }

    /// Read a node via the immutable view.
    ///
    /// # Errors
    /// See [`SlottedPageRef::read_node`].
    pub fn read_node(&self, slot: SlotId) -> Result<Option<NodeRecord>, PageError> {
        self.as_ref().read_node(slot)
    }

    /// Read a rel via the immutable view.
    ///
    /// # Errors
    /// See [`SlottedPageRef::read_rel`].
    pub fn read_rel(&self, slot: SlotId) -> Result<Option<RelRecord>, PageError> {
        self.as_ref().read_rel(slot)
    }

    // ---- property bags (v2 M1 — W-B1 slotted small-blob packing) --------

    /// Insert a variable-length property-bag payload (the DEC-6 JSON
    /// bytes at M1) into a [`PageType::PropSlotted`] page, returning its
    /// freshly assigned slot id.
    ///
    /// Per-op latency budget (design-v2 §4.4, same envelope as the
    /// fixed-size record inserts above): one header read + one
    /// memcpy-sized payload copy + one slot-entry write + a body CRC32C
    /// recompute — the CRC dominates (~1–2 µs hw-accelerated over 8 KiB),
    /// paid once per bag APPEND on the write path, which is itself
    /// dominated by the commit fsync it batches under.
    ///
    /// # Errors
    /// [`PageError::WrongPageType`] if the page is not `PropSlotted`;
    /// [`PageError::Format`] if `bag` is empty or exceeds
    /// [`PROP_BAG_MAX_BYTES`] (callers route oversize bags to the DEC-4
    /// chain path instead); [`PageError::Full`] if this page's free
    /// space cannot hold `bag` + a slot entry (callers open a fresh
    /// page, exactly the `records.rs` `Full` → allocate-fresh shape).
    pub fn insert_bag(&mut self, bag: &[u8]) -> Result<SlotId, PageError> {
        self.require_page_type(PageType::PropSlotted)?;
        if bag.is_empty() {
            return Err(PageError::Format(
                "property bag payload is empty; zero-length bags are not packable".to_string(),
            ));
        }
        if bag.len() > PROP_BAG_MAX_BYTES {
            return Err(PageError::Format(format!(
                "property bag payload {} exceeds PROP_BAG_MAX_BYTES {PROP_BAG_MAX_BYTES}",
                bag.len()
            )));
        }
        self.insert_raw(bag)
    }

    /// Install a typed property block at an exact physiological redo target.
    ///
    /// Property blocks are append-only in the M3 live path. An existing
    /// target is accepted only when its length is unchanged, which permits
    /// deterministic overwrite replay without relocating slotted payloads.
    pub fn put_bag_at(&mut self, slot: SlotId, bag: &[u8]) -> Result<(), PageError> {
        self.require_page_type(PageType::PropSlotted)?;
        if bag.is_empty() || bag.len() > PROP_BAG_MAX_BYTES {
            return Err(PageError::Format(format!(
                "property bag payload length {} is outside 1..={PROP_BAG_MAX_BYTES}",
                bag.len()
            )));
        }
        let expected_len = u16::try_from(bag.len())
            .map_err(|_| PageError::Format(format!("record too large: {}", bag.len())))?;
        self.put_fixed_at(slot, bag, expected_len)
    }

    /// Install a fixed-size owner bag in the canonical direct-slot layout.
    /// Unlike [`Self::put_bag_at`], sparse high-water gaps reserve their fixed
    /// record bodies, making PRE-B.4's permanent tombstone offset arithmetic
    /// valid for every owner slot.
    pub(crate) fn put_fixed_bag_at_slot(
        &mut self,
        slot: SlotId,
        bag: &[u8],
        capacity: u16,
    ) -> Result<(), PageError> {
        self.require_page_type(PageType::PropSlotted)?;
        let expected_len = u16::try_from(bag.len())
            .map_err(|_| PageError::Format(format!("record too large: {}", bag.len())))?;
        if expected_len == 0 || usize::from(expected_len) > PROP_BAG_MAX_BYTES {
            return Err(PageError::Format(format!(
                "fixed bag payload length {} is outside 1..={PROP_BAG_MAX_BYTES}",
                bag.len()
            )));
        }
        if slot.0 < self.slot_count() && self.as_ref().is_permanent_tombstone(slot)? {
            return Err(PageError::SlotTombstoned(slot.0));
        }
        self.write_fixed_at_slot(slot, bag, expected_len, capacity)
    }

    /// Read a property-bag payload via the immutable view.
    ///
    /// # Errors
    /// See [`SlottedPageRef::read_bag`].
    pub fn read_bag(&self, slot: SlotId) -> Result<Option<&[u8]>, PageError> {
        self.as_ref().read_bag(slot)
    }

    // ---- internals ------------------------------------------------------

    fn validate_len(len: usize) -> Result<(), PageError> {
        if len != PAGE_SIZE {
            return Err(PageError::Format(format!(
                "page buffer must be {PAGE_SIZE} bytes, got {len}"
            )));
        }
        Ok(())
    }

    fn require_page_type(&self, expected: PageType) -> Result<(), PageError> {
        let got = self.bytes[5];
        if got != expected.as_byte() {
            return Err(PageError::WrongPageType {
                got,
                expected: expected.as_byte(),
            });
        }
        Ok(())
    }

    /// Core insert: writes `record` into the next free record-area
    /// slot and appends a new slot entry. Both the slot offset and
    /// `slot_count` / `free_space` in the header are updated.
    ///
    /// Returns [`PageError::Format`] if the (attacker/corruption-controllable)
    /// header carries an inconsistent record-area watermark that would
    /// underflow the offset back-computation — see the defensive guard below
    /// (#16; write-path sibling of the #592 read-path `validate_slot_count`).
    fn insert_raw(&mut self, record: &[u8]) -> Result<SlotId, PageError> {
        let rec_len = u16::try_from(record.len())
            .map_err(|_| PageError::Format(format!("record too large: {}", record.len())))?;
        let needed = rec_len + SLOT_SIZE as u16;

        let hdr = self.header();
        if hdr.free_space < needed {
            return Err(PageError::Full {
                needed,
                free: hdr.free_space,
            });
        }

        // record_area_start = PAGE_SIZE - bytes already consumed by records.
        // We track this implicitly by free_space: records start at
        // PAGE_SIZE - (used_record_bytes) and slots end at
        // SLOT_AREA_START + slot_count * SLOT_SIZE. The gap between
        // them is free_space, which is what we just checked.
        //
        // Defensive guard (#16; write-path sibling of #592). `free_space`
        // (header bytes 38..40) and `slot_count` (bytes 36..38) are read
        // verbatim from the page header, which lives *outside* the CRC body
        // (bytes 40..8192 — see `validate_slot_count` and `recompute_checksum`).
        // A crafted or bit-rotted page can therefore carry a self-consistent
        // CRC yet an inconsistent watermark — e.g. `free_space` up to
        // `u16::MAX` — passing both `open`'s checksum gate and the
        // `free_space < needed` check above. Unchecked, the back-computation
        // below underflows: it panics on the `usize` subtraction in debug and
        // wraps to a wild `record_start` in release, driving an out-of-bounds
        // (or in-bounds-but-wrong) slice write at the `copy_from_slice` below.
        // Mirror the #592 read-path guard: validate the untrusted header
        // invariant with checked arithmetic and reject before we trust it.
        // `record_start`'s subtraction cannot itself underflow once the
        // `free_space < needed` gate and `used_record_bytes` check both pass
        // (a valid `free_space >= needed` bounds `used_record_bytes` below
        // `PAGE_SIZE - rec_len`); the checked form is kept for defense in
        // depth so a future change to the gate cannot silently re-expose it.
        let used_record_bytes = PAGE_BODY_BYTES
            .checked_sub(hdr.free_space as usize)
            .and_then(|rem| rem.checked_sub((hdr.slot_count as usize) * SLOT_SIZE))
            .ok_or_else(|| {
                PageError::Format(format!(
                    "record-area watermark underflow: free_space={} slot_count={} exceeds PAGE_BODY_BYTES={PAGE_BODY_BYTES}",
                    hdr.free_space, hdr.slot_count
                ))
            })?;
        let record_start = PAGE_SIZE
            .checked_sub(used_record_bytes)
            .and_then(|start| start.checked_sub(rec_len as usize))
            .ok_or_else(|| {
                PageError::Format(format!(
                    "record-area watermark underflow: record_start underflow (used_record_bytes={used_record_bytes} rec_len={rec_len} exceeds PAGE_SIZE={PAGE_SIZE})"
                ))
            })?;

        // Write the record body (grows backward).
        self.bytes[record_start..record_start + (rec_len as usize)].copy_from_slice(record);

        // Append a slot entry (grows forward).
        let slot_idx = hdr.slot_count;
        let slot_off = SLOT_AREA_START + (slot_idx as usize) * SLOT_SIZE;
        let rec_off_u16 = u16::try_from(record_start).map_err(|_| {
            PageError::Format(format!("record offset overflows u16: {record_start}"))
        })?;
        self.bytes[slot_off..slot_off + 2].copy_from_slice(&rec_off_u16.to_le_bytes());
        self.bytes[slot_off + 2..slot_off + 4].copy_from_slice(&rec_len.to_le_bytes());

        // Publish new header.
        let mut new_hdr = hdr;
        new_hdr.slot_count = slot_idx + 1;
        new_hdr.free_space = hdr.free_space - needed;
        let hdr_bytes = new_hdr.to_bytes();
        self.bytes[..PageHeader::SIZE].copy_from_slice(&hdr_bytes);
        self.recompute_checksum();

        Ok(SlotId(slot_idx))
    }

    fn overwrite(
        &mut self,
        slot: SlotId,
        record: &[u8],
        expected_len: u16,
    ) -> Result<(), PageError> {
        let (off, len) = self
            .as_ref()
            .slot_entry(slot)?
            .ok_or(PageError::SlotTombstoned(slot.0))?;
        if len != expected_len {
            return Err(PageError::Format(format!(
                "slot {} has length {len}, expected {expected_len}",
                slot.0
            )));
        }
        self.bytes[off as usize..(off as usize) + (len as usize)].copy_from_slice(record);
        self.recompute_checksum();
        Ok(())
    }

    /// Canonical direct-address write for fixed-size record pages.
    ///
    /// Dense `insert_raw` writes record `k` at
    /// `PAGE_SIZE - (k + 1) * record_size`. Reserving that same region for
    /// every slot below the high-water mark makes the offset independent of
    /// arrival order while remaining byte-identical to dense v5 pages.
    fn write_fixed_at_slot(
        &mut self,
        slot: SlotId,
        record: &[u8],
        expected_len: u16,
        capacity: u16,
    ) -> Result<(), PageError> {
        if slot.0 >= capacity {
            return Err(PageError::SlotOutOfRange {
                slot: slot.0,
                count: capacity,
            });
        }
        if record.len() != usize::from(expected_len) {
            return Err(PageError::Format(format!(
                "fixed record length {} does not match expected {expected_len}",
                record.len()
            )));
        }

        let hdr = self.header();
        if hdr.slot_count > capacity {
            return Err(PageError::Format(format!(
                "fixed page high-water {} exceeds capacity {capacity}",
                hdr.slot_count
            )));
        }
        let old_used = usize::from(hdr.slot_count)
            .checked_mul(usize::from(expected_len) + SLOT_SIZE)
            .ok_or_else(|| PageError::Format("fixed page watermark overflow".to_owned()))?;
        let expected_old_free = PAGE_BODY_BYTES
            .checked_sub(old_used)
            .ok_or_else(|| PageError::Format("fixed page watermark underflow".to_owned()))?;
        if usize::from(hdr.free_space) != expected_old_free {
            return Err(PageError::Format(format!(
                "non-canonical fixed page watermark: count={} free={} expected={expected_old_free}",
                hdr.slot_count, hdr.free_space
            )));
        }

        let new_count = slot
            .0
            .checked_add(1)
            .ok_or_else(|| PageError::Format("fixed slot high-water overflow".to_owned()))?
            .max(hdr.slot_count);
        let new_used = usize::from(new_count)
            .checked_mul(usize::from(expected_len) + SLOT_SIZE)
            .ok_or_else(|| PageError::Format("fixed page high-water overflow".to_owned()))?;
        let new_free = PAGE_BODY_BYTES
            .checked_sub(new_used)
            .ok_or_else(|| PageError::Format("fixed page exceeds page body".to_owned()))?;
        let record_ordinal = usize::from(slot.0)
            .checked_add(1)
            .ok_or_else(|| PageError::Format("fixed record ordinal overflow".to_owned()))?;
        let record_end_distance = record_ordinal
            .checked_mul(usize::from(expected_len))
            .ok_or_else(|| PageError::Format("fixed record offset overflow".to_owned()))?;
        let record_start = PAGE_SIZE
            .checked_sub(record_end_distance)
            .ok_or_else(|| PageError::Format("fixed record offset underflow".to_owned()))?;
        let directory_end = SLOT_AREA_START
            .checked_add(usize::from(new_count) * SLOT_SIZE)
            .ok_or_else(|| PageError::Format("fixed directory offset overflow".to_owned()))?;
        if record_start < directory_end {
            return Err(PageError::Format(format!(
                "fixed slot {} overlaps directory: record_start={record_start} directory_end={directory_end}",
                slot.0
            )));
        }

        if slot.0 < hdr.slot_count
            && let Some((existing_offset, existing_len)) = self.as_ref().slot_entry(slot)?
            && (usize::from(existing_offset) != record_start || existing_len != expected_len)
        {
            return Err(PageError::Format(format!(
                "slot {} is not canonical: offset={existing_offset} len={existing_len}, expected offset={record_start} len={expected_len}",
                slot.0
            )));
        }

        if new_count > hdr.slot_count {
            let directory_start = SLOT_AREA_START + usize::from(hdr.slot_count) * SLOT_SIZE;
            self.bytes[directory_start..directory_end].fill(0);

            let old_record_start = PAGE_SIZE
                .checked_sub(usize::from(hdr.slot_count) * usize::from(expected_len))
                .ok_or_else(|| PageError::Format("old fixed record offset underflow".to_owned()))?;
            let new_record_start = PAGE_SIZE
                .checked_sub(usize::from(new_count) * usize::from(expected_len))
                .ok_or_else(|| PageError::Format("new fixed record offset underflow".to_owned()))?;
            self.bytes[new_record_start..old_record_start].fill(0);
        }

        self.bytes[record_start..record_start + usize::from(expected_len)].copy_from_slice(record);
        let slot_offset = SLOT_AREA_START + usize::from(slot.0) * SLOT_SIZE;
        let record_offset = u16::try_from(record_start).map_err(|_| {
            PageError::Format(format!("fixed record offset overflows u16: {record_start}"))
        })?;
        self.bytes[slot_offset..slot_offset + 2].copy_from_slice(&record_offset.to_le_bytes());
        self.bytes[slot_offset + 2..slot_offset + SLOT_SIZE]
            .copy_from_slice(&expected_len.to_le_bytes());

        let mut canonical = hdr;
        canonical.slot_count = new_count;
        canonical.free_space = u16::try_from(new_free)
            .map_err(|_| PageError::Format(format!("free space overflows u16: {new_free}")))?;
        self.bytes[..PageHeader::SIZE].copy_from_slice(&canonical.to_bytes());
        self.recompute_checksum();
        Ok(())
    }

    fn write_fixed_permanent_tombstone(
        &mut self,
        slot: SlotId,
        tombstone_lsn: Lsn,
        expected_len: u16,
        capacity: u16,
    ) -> Result<(), PageError> {
        if slot.0 < self.slot_count() && self.as_ref().is_permanent_tombstone(slot)? {
            let existing = self
                .as_ref()
                .permanent_tombstone_lsn(slot, expected_len)?
                .expect("permanent marker checked above");
            if tombstone_lsn.raw() <= existing {
                return Ok(());
            }
        }
        let mut marker_record = vec![0_u8; usize::from(expected_len)];
        marker_record[..size_of::<u64>()].copy_from_slice(&tombstone_lsn.raw().to_le_bytes());
        if slot.0 < self.slot_count() && self.as_ref().is_permanent_tombstone(slot)? {
            let record_ordinal = usize::from(slot.0) + 1;
            let record_start = PAGE_SIZE - record_ordinal * usize::from(expected_len);
            self.bytes[record_start..record_start + usize::from(expected_len)]
                .copy_from_slice(&marker_record);
            self.recompute_checksum();
            return Ok(());
        }
        self.write_fixed_at_slot(slot, &marker_record, expected_len, capacity)?;
        let slot_offset = SLOT_AREA_START + usize::from(slot.0) * SLOT_SIZE;
        self.bytes[slot_offset..slot_offset + 2].copy_from_slice(&0_u16.to_le_bytes());
        self.bytes[slot_offset + 2..slot_offset + SLOT_SIZE]
            .copy_from_slice(&PERMANENT_TOMBSTONE_LEN.to_le_bytes());
        self.recompute_checksum();
        Ok(())
    }

    fn put_fixed_at(
        &mut self,
        slot: SlotId,
        record: &[u8],
        expected_len: u16,
    ) -> Result<(), PageError> {
        let count = self.slot_count();
        if slot.0 == count {
            let installed = self.insert_raw(record)?;
            debug_assert_eq!(installed, slot);
            return Ok(());
        }
        if slot.0 > count {
            let gap_slots = slot.0 - count;
            let directory_bytes = gap_slots
                .checked_mul(SLOT_SIZE as u16)
                .ok_or_else(|| PageError::Format("sparse slot directory overflow".to_owned()))?;
            let hdr = self.header();
            if hdr.free_space < directory_bytes.saturating_add(expected_len + SLOT_SIZE as u16) {
                return Err(PageError::Full {
                    needed: directory_bytes.saturating_add(expected_len + SLOT_SIZE as u16),
                    free: hdr.free_space,
                });
            }
            let start = SLOT_AREA_START + count as usize * SLOT_SIZE;
            let end = SLOT_AREA_START + slot.0 as usize * SLOT_SIZE;
            self.bytes[start..end].fill(0);
            let mut sparse = hdr;
            sparse.slot_count = slot.0;
            sparse.free_space -= directory_bytes;
            self.bytes[..PageHeader::SIZE].copy_from_slice(&sparse.to_bytes());
            let installed = self.insert_raw(record)?;
            debug_assert_eq!(installed, slot);
            return Ok(());
        }
        if self.as_ref().slot_entry(slot)?.is_none() {
            let hdr = self.header();
            if hdr.free_space < expected_len {
                return Err(PageError::Full {
                    needed: expected_len,
                    free: hdr.free_space,
                });
            }
            let used_record_bytes = PAGE_BODY_BYTES
                .checked_sub(hdr.free_space as usize)
                .and_then(|used| used.checked_sub(hdr.slot_count as usize * SLOT_SIZE))
                .ok_or_else(|| {
                    PageError::Format("tombstone-revive watermark underflow".to_owned())
                })?;
            let record_start = PAGE_SIZE
                .checked_sub(used_record_bytes)
                .and_then(|start| start.checked_sub(expected_len as usize))
                .ok_or_else(|| {
                    PageError::Format("tombstone-revive record offset underflow".to_owned())
                })?;
            self.bytes[record_start..record_start + expected_len as usize].copy_from_slice(record);
            let slot_off = SLOT_AREA_START + slot.0 as usize * SLOT_SIZE;
            let record_offset = u16::try_from(record_start).map_err(|_| {
                PageError::Format(format!("record offset overflows u16: {record_start}"))
            })?;
            self.bytes[slot_off..slot_off + 2].copy_from_slice(&record_offset.to_le_bytes());
            self.bytes[slot_off + 2..slot_off + 4].copy_from_slice(&expected_len.to_le_bytes());
            let mut revived = hdr;
            revived.free_space -= expected_len;
            self.bytes[..PageHeader::SIZE].copy_from_slice(&revived.to_bytes());
            self.recompute_checksum();
            return Ok(());
        }
        self.overwrite(slot, record, expected_len)
    }

    /// Recompute CRC32C over the page body (bytes 40..PAGE_SIZE) and
    /// write it into `PageHeader::checksum`. Called at the end of every
    /// mutation so open-validated readers observe a consistent page.
    ///
    /// The checksum field itself lives at bytes 32..36 (inside the
    /// header), which is outside the body range, so rewriting the
    /// header does not invalidate the computed CRC.
    fn recompute_checksum(&mut self) {
        let crc = crc32c::crc32c(&self.bytes[PageHeader::SIZE..]);
        self.bytes[32..36].copy_from_slice(&crc.to_le_bytes());
    }
}

// ---- read-only slotted page --------------------------------------------

/// Read-only view. Holds only a shared borrow; cheap to construct.
#[derive(Debug)]
pub struct SlottedPageRef<'a> {
    bytes: &'a [u8],
}

impl<'a> SlottedPageRef<'a> {
    /// Attach to a page, validating the header.
    ///
    /// # Errors
    /// [`PageError::Format`] if `bytes.len() != PAGE_SIZE`, the header
    /// fails to decode, or `header.slot_count` exceeds [`MAX_SLOTS`].
    /// [`PageError::ChecksumMismatch`] if the CRC32C over bytes
    /// 40..PAGE_SIZE does not match `PageHeader::checksum`.
    pub fn open(bytes: &'a [u8]) -> Result<Self, PageError> {
        if bytes.len() != PAGE_SIZE {
            return Err(PageError::Format(format!(
                "page buffer must be {PAGE_SIZE} bytes, got {}",
                bytes.len()
            )));
        }
        let hdr_bytes: &[u8; PageHeader::SIZE] = (&bytes[..PageHeader::SIZE])
            .try_into()
            .expect("SLOT_AREA_START == PageHeader::SIZE");
        let hdr =
            PageHeader::from_bytes(hdr_bytes).map_err(|e| PageError::Format(e.to_string()))?;
        let computed = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
        if computed != hdr.checksum {
            return Err(PageError::ChecksumMismatch {
                stored: hdr.checksum,
                computed,
            });
        }
        // Reject a forged/corrupt `slot_count` before any reader can
        // address a slot past the page buffer (#592).
        validate_slot_count(hdr.slot_count)?;
        Ok(Self { bytes })
    }

    /// Skip header validation. Only callable inside this module — used
    /// by `SlottedPage::as_ref` where the header has already been validated
    /// on page init/open.
    fn from_bytes_unchecked(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Decoded header. Never fails because construction validated it.
    #[must_use]
    pub fn header(&self) -> PageHeader {
        let hdr_bytes: &[u8; PageHeader::SIZE] = (&self.bytes[..PageHeader::SIZE])
            .try_into()
            .expect("validated on open");
        PageHeader::from_bytes(hdr_bytes).expect("validated on open")
    }

    /// Full redo sub-LSN currently stamped on this page.
    #[must_use]
    pub fn page_lsn(&self) -> Lsn {
        Lsn::new(self.header().lsn)
    }

    /// Slot count from the header (includes tombstones).
    #[must_use]
    pub fn slot_count(&self) -> u16 {
        self.header().slot_count
    }

    /// Read a node record. Returns `Ok(None)` if the slot is tombstoned.
    ///
    /// # Errors
    /// [`PageError::SlotOutOfRange`] if `slot.0 >= slot_count`.
    /// [`PageError::Format`] if the slot length does not match
    /// `NodeRecord::SIZE`. [`PageError::RecordDecode`] on decode failure.
    pub fn read_node(&self, slot: SlotId) -> Result<Option<NodeRecord>, PageError> {
        let Some((off, len)) = self.slot_entry(slot)? else {
            return Ok(None);
        };
        if len as usize != NodeRecord::SIZE {
            return Err(PageError::Format(format!(
                "slot {} has length {len}, expected {}",
                slot.0,
                NodeRecord::SIZE
            )));
        }
        let buf: &[u8; NodeRecord::SIZE] = (&self.bytes
            [off as usize..(off as usize) + NodeRecord::SIZE])
            .try_into()
            .expect("length checked above");
        NodeRecord::from_bytes(buf)
            .map(Some)
            .map_err(|e| PageError::RecordDecode(e.to_string()))
    }

    /// Read a rel record. See [`Self::read_node`].
    ///
    /// # Errors
    /// As [`Self::read_node`], with `RelRecord::SIZE`.
    pub fn read_rel(&self, slot: SlotId) -> Result<Option<RelRecord>, PageError> {
        let Some((off, len)) = self.slot_entry(slot)? else {
            return Ok(None);
        };
        if len as usize != RelRecord::SIZE {
            return Err(PageError::Format(format!(
                "slot {} has length {len}, expected {}",
                slot.0,
                RelRecord::SIZE
            )));
        }
        let buf: &[u8; RelRecord::SIZE] = (&self.bytes
            [off as usize..(off as usize) + RelRecord::SIZE])
            .try_into()
            .expect("length checked above");
        RelRecord::from_bytes(buf)
            .map(Some)
            .map_err(|e| PageError::RecordDecode(e.to_string()))
    }

    /// Recover the preserved fixed node body beneath a reusable `(0, 0)`
    /// tombstone during the offline v5-to-v6 rewrite.
    ///
    /// This is intentionally separate from ordinary reads: runtime readers
    /// must continue to observe a tombstone as absent. An all-zero reserved
    /// body is a sparse hole and returns `None`.
    pub fn recover_tombstoned_node(&self, slot: SlotId) -> Result<Option<NodeRecord>, PageError> {
        self.require_ref_page_type(PageType::Node)?;
        self.recover_tombstoned_fixed::<NodeRecord, { NodeRecord::SIZE }>(
            slot,
            NodeRecord::from_bytes,
        )
    }

    /// Relationship counterpart of [`Self::recover_tombstoned_node`].
    pub fn recover_tombstoned_rel(&self, slot: SlotId) -> Result<Option<RelRecord>, PageError> {
        self.require_ref_page_type(PageType::Rel)?;
        self.recover_tombstoned_fixed::<RelRecord, { RelRecord::SIZE }>(slot, RelRecord::from_bytes)
    }

    fn recover_tombstoned_fixed<T, const N: usize>(
        &self,
        slot: SlotId,
        decode: impl FnOnce(&[u8; N]) -> arcgraph_core::Result<T>,
    ) -> Result<Option<T>, PageError> {
        if self.slot_entry(slot)?.is_some() || self.is_permanent_tombstone(slot)? {
            return Ok(None);
        }
        let ordinal = usize::from(slot.0) + 1;
        let start = PAGE_SIZE
            .checked_sub(ordinal.checked_mul(N).ok_or_else(|| {
                PageError::Format("tombstone recovery offset overflow".to_owned())
            })?)
            .ok_or_else(|| PageError::Format("tombstone recovery offset underflow".to_owned()))?;
        let raw = &self.bytes[start..start + N];
        if raw.iter().all(|byte| *byte == 0) {
            return Ok(None);
        }
        let fixed: &[u8; N] = raw
            .try_into()
            .map_err(|_| PageError::Format("unexpected fixed record length".to_owned()))?;
        decode(fixed)
            .map(Some)
            .map_err(|error| PageError::RecordDecode(error.to_string()))
    }

    fn require_ref_page_type(&self, expected: PageType) -> Result<(), PageError> {
        let got = self.header().page_type;
        if got != expected.as_byte() {
            return Err(PageError::WrongPageType {
                got,
                expected: expected.as_byte(),
            });
        }
        Ok(())
    }

    /// Iterate live node records. Skips tombstoned slots.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (SlotId, NodeRecord)> + '_ {
        let count = self.slot_count();
        (0..count).filter_map(move |i| {
            let s = SlotId(i);
            match self.read_node(s) {
                Ok(Some(r)) => Some((s, r)),
                _ => None,
            }
        })
    }

    /// Iterate live rel records. Skips tombstoned slots.
    pub fn iter_rels(&self) -> impl Iterator<Item = (SlotId, RelRecord)> + '_ {
        let count = self.slot_count();
        (0..count).filter_map(move |i| {
            let s = SlotId(i);
            match self.read_rel(s) {
                Ok(Some(r)) => Some((s, r)),
                _ => None,
            }
        })
    }

    // ---- property bags (v2 M1 — W-B1 slotted small-blob packing) --------

    /// Attach to an already-validated [`PageType::PropSlotted`] page
    /// WITHOUT recomputing the body CRC32C — the hot-read entry point
    /// for resident slotted pages whose bytes were checksum-validated at
    /// their trust boundary (WAL-replay install, checkpoint restore, or
    /// spill re-fault — all of which go through the full
    /// [`Self::open`]). Still validates length, header decode (magic /
    /// version / known page type), the `PropSlotted` type byte, and the
    /// #592 `slot_count` bound — every check that guards memory safety
    /// stays; only the 8 KiB CRC pass (which would dominate a ≤ 40 ns
    /// slot read, §4.4 budget) is skipped.
    ///
    /// # Errors
    /// [`PageError::Format`] on a bad length / header / `slot_count`;
    /// [`PageError::WrongPageType`] if the page is not `PropSlotted`.
    pub fn open_prop_trusted(bytes: &'a [u8]) -> Result<Self, PageError> {
        if bytes.len() != PAGE_SIZE {
            return Err(PageError::Format(format!(
                "page buffer must be {PAGE_SIZE} bytes, got {}",
                bytes.len()
            )));
        }
        let hdr_bytes: &[u8; PageHeader::SIZE] = (&bytes[..PageHeader::SIZE])
            .try_into()
            .expect("SLOT_AREA_START == PageHeader::SIZE");
        let hdr =
            PageHeader::from_bytes(hdr_bytes).map_err(|e| PageError::Format(e.to_string()))?;
        if hdr.page_type != PageType::PropSlotted.as_byte() {
            return Err(PageError::WrongPageType {
                got: hdr.page_type,
                expected: PageType::PropSlotted.as_byte(),
            });
        }
        validate_slot_count(hdr.slot_count)?;
        Ok(Self { bytes })
    }

    /// Read a variable-length property-bag payload, borrowed from the
    /// underlying page buffer. Returns `Ok(None)` if the slot is
    /// tombstoned.
    ///
    /// # Errors
    /// [`PageError::WrongPageType`] if the page is not `PropSlotted`;
    /// [`PageError::SlotOutOfRange`] / [`PageError::Format`] per
    /// `Self::slot_entry`'s bounds validation (offset inside the
    /// record area, end within the page).
    pub fn read_bag(&self, slot: SlotId) -> Result<Option<&'a [u8]>, PageError> {
        let got = self.bytes[5];
        if got != PageType::PropSlotted.as_byte() {
            return Err(PageError::WrongPageType {
                got,
                expected: PageType::PropSlotted.as_byte(),
            });
        }
        let Some((off, len)) = self.slot_entry(slot)? else {
            return Ok(None);
        };
        Ok(Some(
            &self.bytes[off as usize..(off as usize) + (len as usize)],
        ))
    }

    /// Return whether a slot carries durable pre-M4 deletion provenance.
    ///
    /// # Errors
    /// [`PageError::SlotOutOfRange`] if the slot is past the high-water mark.
    pub fn is_permanent_tombstone(&self, slot: SlotId) -> Result<bool, PageError> {
        let count = self.slot_count();
        if slot.0 >= count {
            return Err(PageError::SlotOutOfRange {
                slot: slot.0,
                count,
            });
        }
        let off = SLOT_AREA_START + usize::from(slot.0) * SLOT_SIZE;
        if off + SLOT_SIZE > PAGE_SIZE {
            return Err(PageError::Format(format!(
                "slot {} directory entry at offset {off} overruns page (count={count})",
                slot.0
            )));
        }
        let record_off = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
        let record_len = u16::from_le_bytes([self.bytes[off + 2], self.bytes[off + 3]]);
        Ok(record_off == 0 && record_len == PERMANENT_TOMBSTONE_LEN)
    }

    /// Read the deleting LSN embedded in a durable fixed-slot tombstone.
    /// Returns `None` for a live slot or reusable hole.
    pub fn permanent_tombstone_lsn(
        &self,
        slot: SlotId,
        fixed_record_len: u16,
    ) -> Result<Option<u64>, PageError> {
        if !self.is_permanent_tombstone(slot)? {
            return Ok(None);
        }
        if usize::from(fixed_record_len) < size_of::<u64>() {
            return Err(PageError::Format(
                "fixed tombstone body is too short for its LSN".to_owned(),
            ));
        }
        let ordinal = usize::from(slot.0)
            .checked_add(1)
            .ok_or_else(|| PageError::Format("fixed tombstone ordinal overflow".to_owned()))?;
        let distance = ordinal
            .checked_mul(usize::from(fixed_record_len))
            .ok_or_else(|| PageError::Format("fixed tombstone offset overflow".to_owned()))?;
        let start = PAGE_SIZE
            .checked_sub(distance)
            .ok_or_else(|| PageError::Format("fixed tombstone offset underflow".to_owned()))?;
        let end = start
            .checked_add(size_of::<u64>())
            .ok_or_else(|| PageError::Format("fixed tombstone LSN end overflow".to_owned()))?;
        let bytes: [u8; size_of::<u64>()] =
            self.bytes[start..end].try_into().expect("fixed-size slice");
        Ok(Some(u64::from_le_bytes(bytes)))
    }

    /// Decode `(offset, length)` for a slot. Returns `Ok(None)` for a
    /// reusable hole `(0, 0)` or durable tombstone `(0, u16::MAX)`.
    fn slot_entry(&self, slot: SlotId) -> Result<Option<(u16, u16)>, PageError> {
        let count = self.slot_count();
        if slot.0 >= count {
            return Err(PageError::SlotOutOfRange {
                slot: slot.0,
                count,
            });
        }
        let off = SLOT_AREA_START + (slot.0 as usize) * SLOT_SIZE;
        // Defense-in-depth (#592): `open` rejects an oversized `slot_count`,
        // but a caller reaching this via `from_bytes_unchecked` on an
        // unvalidated buffer must still never index past the page. The
        // directory entry spans `off..off + SLOT_SIZE`.
        if off + SLOT_SIZE > PAGE_SIZE {
            return Err(PageError::Format(format!(
                "slot {} directory entry at offset {off} overruns page (count={count})",
                slot.0
            )));
        }
        let record_off = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
        let record_len = u16::from_le_bytes([self.bytes[off + 2], self.bytes[off + 3]]);
        if record_off == 0 && (record_len == 0 || record_len == PERMANENT_TOMBSTONE_LEN) {
            return Ok(None);
        }
        // Invariant: offset must land inside the record area, and
        // record end must not overrun the page.
        let end = (record_off as usize)
            .checked_add(record_len as usize)
            .ok_or_else(|| {
                PageError::Format(format!(
                    "slot {} overflows addition: off={record_off} len={record_len}",
                    slot.0
                ))
            })?;
        // `record_off == directory end` is LEGAL: the record area may
        // start exactly where the slot directory ends (reachable only by
        // a payload that fills the whole body — e.g. one
        // PROP_BAG_MAX_BYTES bag, v2 M1; fixed-size Node/Rel records
        // never land there). Strictly BELOW the directory end is an
        // overlap with the directory = corruption.
        if (record_off as usize) < SLOT_AREA_START + (count as usize) * SLOT_SIZE || end > PAGE_SIZE
        {
            return Err(PageError::Format(format!(
                "slot {} points outside record area: off={record_off} len={record_len}",
                slot.0
            )));
        }
        Ok(Some((record_off, record_len)))
    }
}

// ---- tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    use arcgraph_core::ids::{LabelId, Lsn, NodeId, PageId, RelId, TenantId, TypeId};
    use proptest::prelude::*;

    use super::*;

    fn fresh_node_page(buf: &mut [u8]) -> SlottedPage<'_> {
        let hdr = PageHeader::new(PageId::new(7), PageType::Node, TenantId::DEFAULT);
        SlottedPage::init(buf, hdr).unwrap()
    }

    fn fresh_rel_page(buf: &mut [u8]) -> SlottedPage<'_> {
        let hdr = PageHeader::new(PageId::new(9), PageType::Rel, TenantId::DEFAULT);
        SlottedPage::init(buf, hdr).unwrap()
    }

    fn mk_node(id: u64) -> NodeRecord {
        NodeRecord::new(NodeId::new(id), LabelId::new(1), Lsn::new(42))
    }

    fn mk_rel(id: u64) -> RelRecord {
        RelRecord::new(
            RelId::new(id),
            TypeId::new(2),
            NodeId::new(10),
            NodeId::new(20),
            Lsn::new(42),
        )
    }

    // ---- capacity constants ----

    #[test]
    fn capacity_constants_match_math() {
        assert_eq!(NODE_CAPACITY, 119);
        assert_eq!(REL_CAPACITY, 81);
        assert!((NODE_CAPACITY as usize) * (SLOT_SIZE + NodeRecord::SIZE) <= PAGE_BODY_BYTES);
        assert!((REL_CAPACITY as usize) * (SLOT_SIZE + RelRecord::SIZE) <= PAGE_BODY_BYTES);
    }

    #[test]
    fn exact_redo_can_fill_sparse_and_released_reservation_slots() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        page.put_node_at(SlotId(2), &mk_node(3)).unwrap();
        assert_eq!(page.slot_count(), 3);
        assert!(page.read_node(SlotId(0)).unwrap().is_none());
        assert!(page.read_node(SlotId(1)).unwrap().is_none());
        assert_eq!(page.read_node(SlotId(2)).unwrap().unwrap().id, 3);

        page.put_node_at(SlotId(0), &mk_node(1)).unwrap();
        page.put_node_at(SlotId(1), &mk_node(2)).unwrap();
        assert_eq!(page.read_node(SlotId(0)).unwrap().unwrap().id, 1);
        assert_eq!(page.read_node(SlotId(1)).unwrap().unwrap().id, 2);
    }

    #[test]
    fn durable_tombstone_is_distinct_from_hole_and_rejects_publish_and_redo() {
        let mut buf = [0_u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        page.write_node_at_slot(SlotId(2), &mk_node(2)).unwrap();
        page.permanent_tombstone_node_at_slot(SlotId(2), Lsn::new(77))
            .unwrap();
        assert_eq!(page.read_node(SlotId(2)).unwrap(), None);
        assert!(page.as_ref().is_permanent_tombstone(SlotId(2)).unwrap());
        assert_eq!(
            page.as_ref()
                .permanent_tombstone_lsn(SlotId(2), NodeRecord::SIZE as u16)
                .unwrap(),
            Some(77)
        );
        assert!(matches!(
            page.write_node_at_slot(SlotId(2), &mk_node(22)),
            Err(PageError::SlotTombstoned(2))
        ));
        assert!(matches!(
            page.put_node_at(SlotId(2), &mk_node(22)),
            Err(PageError::SlotTombstoned(2))
        ));

        // A sparse reservation remains reusable, proving the marker does not
        // collapse ordinary `(0, 0)` holes into permanent deletion.
        page.put_node_at(SlotId(0), &mk_node(1)).unwrap();
        assert_eq!(page.read_node(SlotId(0)).unwrap().unwrap().id, 1);
    }

    // ---- init + open ----

    #[test]
    fn init_zeroes_body_and_sets_free_space() {
        let mut buf = [0xAAu8; PAGE_SIZE];
        let page = fresh_node_page(&mut buf);
        let h = page.header();
        assert_eq!(h.slot_count, 0);
        assert_eq!(h.free_space as usize, PAGE_BODY_BYTES);
        assert_eq!(h.tenant_id, TenantId::DEFAULT.raw());
        // Body is zeroed.
        assert!(buf[PageHeader::SIZE..].iter().all(|&b| b == 0));
    }

    #[test]
    fn open_validates_header() {
        let mut buf = [0u8; PAGE_SIZE];
        let _ = fresh_node_page(&mut buf);
        let reopened = SlottedPage::open(&mut buf);
        assert!(reopened.is_ok());

        // Corrupt magic.
        buf[0] = 0;
        let err = SlottedPage::open(&mut buf).unwrap_err();
        assert!(matches!(err, PageError::Format(_)));
    }

    // ---- insert + read ----

    #[test]
    fn node_insert_then_read_roundtrip() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let r1 = mk_node(1);
        let r2 = mk_node(2);
        let s1 = page.insert_node(&r1).unwrap();
        let s2 = page.insert_node(&r2).unwrap();
        assert_eq!(s1, SlotId(0));
        assert_eq!(s2, SlotId(1));
        assert_eq!(page.read_node(s1).unwrap(), Some(r1));
        assert_eq!(page.read_node(s2).unwrap(), Some(r2));
        assert_eq!(page.slot_count(), 2);
    }

    #[test]
    fn rel_insert_then_read_roundtrip() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_rel_page(&mut buf);
        let r = mk_rel(7);
        let s = page.insert_rel(&r).unwrap();
        assert_eq!(page.read_rel(s).unwrap(), Some(r));
    }

    #[test]
    fn insert_rejects_wrong_page_type() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let err = page.insert_rel(&mk_rel(1)).unwrap_err();
        assert!(matches!(err, PageError::WrongPageType { .. }));
    }

    #[test]
    fn read_rejects_out_of_range_slot() {
        let mut buf = [0u8; PAGE_SIZE];
        let page = fresh_node_page(&mut buf);
        let err = page.read_node(SlotId(0)).unwrap_err();
        assert!(matches!(
            err,
            PageError::SlotOutOfRange { slot: 0, count: 0 }
        ));
    }

    // ---- update in place ----

    #[test]
    fn update_node_overwrites_in_place_keeping_slot_id() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let r1 = mk_node(1);
        let s = page.insert_node(&r1).unwrap();

        let mut r2 = r1;
        r2.label_id = 99;
        page.update_node(s, &r2).unwrap();
        assert_eq!(page.read_node(s).unwrap(), Some(r2));
        assert_eq!(page.slot_count(), 1);
    }

    #[test]
    fn update_node_on_tombstoned_slot_fails() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let s = page.insert_node(&mk_node(1)).unwrap();
        page.tombstone(s).unwrap();
        let err = page.update_node(s, &mk_node(2)).unwrap_err();
        assert!(matches!(err, PageError::SlotTombstoned(0)));
    }

    // ---- tombstone ----

    #[test]
    fn tombstone_hides_record_from_reads() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let a = page.insert_node(&mk_node(1)).unwrap();
        let b = page.insert_node(&mk_node(2)).unwrap();
        page.tombstone(a).unwrap();
        assert_eq!(page.read_node(a).unwrap(), None);
        assert_eq!(page.read_node(b).unwrap(), Some(mk_node(2)));
        // slot_count retains the high-water mark.
        assert_eq!(page.slot_count(), 2);
    }

    #[test]
    fn tombstone_is_idempotent() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let s = page.insert_node(&mk_node(1)).unwrap();
        page.tombstone(s).unwrap();
        page.tombstone(s).unwrap();
        assert_eq!(page.read_node(s).unwrap(), None);
    }

    // ---- capacity + full ----

    #[test]
    fn insert_fills_page_to_capacity_then_reports_full() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        for i in 0..NODE_CAPACITY {
            let s = page.insert_node(&mk_node(i as u64)).unwrap();
            assert_eq!(s, SlotId(i));
        }
        let err = page.insert_node(&mk_node(9999)).unwrap_err();
        assert!(
            matches!(err, PageError::Full { .. }),
            "expected Full, got {err:?}"
        );
    }

    #[test]
    fn rel_insert_fills_to_capacity() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_rel_page(&mut buf);
        for i in 0..REL_CAPACITY {
            page.insert_rel(&mk_rel(i as u64)).unwrap();
        }
        let err = page.insert_rel(&mk_rel(9999)).unwrap_err();
        assert!(matches!(err, PageError::Full { .. }));
    }

    // ---- iteration ----

    #[test]
    fn iter_nodes_skips_tombstones_and_visits_in_slot_order() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let s0 = page.insert_node(&mk_node(10)).unwrap();
        let _s1 = page.insert_node(&mk_node(20)).unwrap();
        let _s2 = page.insert_node(&mk_node(30)).unwrap();
        page.tombstone(s0).unwrap();
        let got: Vec<(SlotId, u64)> = page.as_ref().iter_nodes().map(|(s, r)| (s, r.id)).collect();
        assert_eq!(got, vec![(SlotId(1), 20), (SlotId(2), 30)]);
    }

    // ---- structural invariants ----

    #[test]
    fn free_space_decreases_monotonically_by_slot_plus_record() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let mut prev = page.free_space();
        for i in 0..5 {
            page.insert_node(&mk_node(i)).unwrap();
            let now = page.free_space();
            assert_eq!(
                prev - now,
                (SLOT_SIZE + NodeRecord::SIZE) as u16,
                "free_space should drop by slot+record on every insert"
            );
            prev = now;
        }
    }

    #[test]
    fn records_grow_backward_from_page_end() {
        // After a single insert, the record should start at
        // PAGE_SIZE - NodeRecord::SIZE.
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        page.insert_node(&mk_node(42)).unwrap();
        let off = u16::from_le_bytes([buf[SLOT_AREA_START], buf[SLOT_AREA_START + 1]]) as usize;
        assert_eq!(off, PAGE_SIZE - NodeRecord::SIZE);
    }

    // ---- checksum (issue #15) ----

    #[test]
    fn fresh_page_opens_with_valid_checksum() {
        let mut buf = [0u8; PAGE_SIZE];
        let _ = fresh_node_page(&mut buf);
        // Both reopen paths must accept it.
        SlottedPage::open(&mut buf.clone()).unwrap();
        SlottedPageRef::open(&buf).unwrap();
    }

    #[test]
    fn checksum_is_nonzero_after_init() {
        let mut buf = [0u8; PAGE_SIZE];
        let page = fresh_node_page(&mut buf);
        // CRC32C of an all-zero 8152-byte buffer is 0x0 only for length 0;
        // for 8152 zeros it is a specific nonzero value. We just assert
        // that the header stored *some* value matching the body.
        let stored = page.header().checksum;
        let computed = crc32c::crc32c(&buf[PageHeader::SIZE..]);
        assert_eq!(stored, computed);
    }

    #[test]
    fn empty_page_checksum_is_stable_across_inits() {
        let hdr = PageHeader::new(PageId::new(7), PageType::Node, TenantId::DEFAULT);
        let mut a = [0u8; PAGE_SIZE];
        let mut b = [0u8; PAGE_SIZE];
        SlottedPage::init(&mut a, hdr).unwrap();
        SlottedPage::init(&mut b, hdr).unwrap();
        assert_eq!(a, b, "empty pages with equal header must be byte-identical");
    }

    #[test]
    fn mutation_updates_checksum() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let c0 = page.header().checksum;
        page.insert_node(&mk_node(1)).unwrap();
        let c1 = page.header().checksum;
        assert_ne!(c0, c1, "insert must change the body and thus the checksum");
        // Updated checksum must still match the body.
        assert_eq!(c1, crc32c::crc32c(&buf[PageHeader::SIZE..]));

        // Reopen must succeed.
        SlottedPageRef::open(&buf).unwrap();
    }

    #[test]
    fn tombstone_updates_checksum_and_reopens() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let s = page.insert_node(&mk_node(1)).unwrap();
        let before = page.header().checksum;
        page.tombstone(s).unwrap();
        let after = page.header().checksum;
        assert_ne!(before, after);
        SlottedPageRef::open(&buf).unwrap();
    }

    #[test]
    fn overwrite_updates_checksum_and_reopens() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = fresh_node_page(&mut buf);
        let s = page.insert_node(&mk_node(1)).unwrap();
        let before = page.header().checksum;
        let mut r2 = mk_node(1);
        r2.label_id = 99;
        page.update_node(s, &r2).unwrap();
        let after = page.header().checksum;
        assert_ne!(before, after);
        SlottedPageRef::open(&buf).unwrap();
    }

    #[test]
    fn corrupt_body_byte_fails_verification() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(42)).unwrap();
        }
        // Flip a body byte (outside the header).
        buf[PageHeader::SIZE + 17] ^= 0xFF;
        let err = SlottedPageRef::open(&buf).unwrap_err();
        assert!(
            matches!(err, PageError::ChecksumMismatch { .. }),
            "got {err:?}"
        );
        let err2 = SlottedPage::open(&mut buf).unwrap_err();
        assert!(
            matches!(err2, PageError::ChecksumMismatch { .. }),
            "got {err2:?}"
        );
    }

    #[test]
    fn corrupt_record_area_byte_fails_verification() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(42)).unwrap();
        }
        // Flip the last byte of the page (record area tail).
        buf[PAGE_SIZE - 1] ^= 0x01;
        let err = SlottedPageRef::open(&buf).unwrap_err();
        assert!(matches!(err, PageError::ChecksumMismatch { .. }));
    }

    #[test]
    fn checksum_field_change_alone_fails_verification() {
        // Directly corrupting the `checksum` field in the header must
        // also be detected (body unchanged, but stored != computed).
        let mut buf = [0u8; PAGE_SIZE];
        let _ = fresh_node_page(&mut buf);
        buf[32] ^= 0x55;
        let err = SlottedPageRef::open(&buf).unwrap_err();
        assert!(matches!(err, PageError::ChecksumMismatch { .. }));
    }

    // ---- #592: forged slot_count bound ----
    //
    // A crafted page can carry any `slot_count` up to `u16::MAX`. The
    // field lives at header bytes 36..38, OUTSIDE the CRC body range
    // (bytes 40..), so forging it leaves the body checksum valid and the
    // page passes `open`'s CRC gate. Before #592, the unbounded
    // `slot_count` let `slot_entry` index past the 8 KiB buffer and
    // panic. These tests pin the fix: reject at `open`, never panic on
    // read. Same class as #577/#594 (validate untrusted length-prefix).

    /// Forge `slot_count` to the smallest value (just over the cap) is
    /// covered by `slot_count_boundary_is_exact`; here we use the maximal
    /// adversarial value. Both public attach paths must reject it with a
    /// structured `Format` error — NOT a `ChecksumMismatch` (proving the
    /// CRC still passed) and NOT a panic.
    ///
    /// Pre-fix this test is RED: `open` returns `Ok`, so `unwrap_err`
    /// panics. Post-fix it is GREEN.
    #[test]
    fn open_rejects_forged_oversized_slot_count() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(1)).unwrap();
        }
        // Sanity: the well-formed page opens on both paths before forging.
        SlottedPageRef::open(&buf).unwrap();
        SlottedPage::open(&mut buf.clone()).unwrap();

        // Forge slot_count = u16::MAX (>> MAX_SLOTS). CRC untouched.
        buf[36..38].copy_from_slice(&u16::MAX.to_le_bytes());

        let err = SlottedPageRef::open(&buf).unwrap_err();
        assert!(
            matches!(err, PageError::Format(ref m) if m.contains("slot_count")),
            "ref open: expected slot_count Format reject, got {err:?}"
        );
        let err_mut = SlottedPage::open(&mut buf).unwrap_err();
        assert!(
            matches!(err_mut, PageError::Format(ref m) if m.contains("slot_count")),
            "mut open: expected slot_count Format reject, got {err_mut:?}"
        );
    }

    /// Strong-oracle boundary pin: `slot_count == MAX_SLOTS` is the
    /// largest value whose directory still fits, so it must be ACCEPTED;
    /// `MAX_SLOTS + 1` must be REJECTED. Guards against an off-by-one that
    /// would either over-reject a legitimate full page or leave the last
    /// out-of-bounds index reachable.
    #[test]
    fn slot_count_boundary_is_exact() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(1)).unwrap();
        }
        // Exactly MAX_SLOTS — accepted (CRC untouched by the forge).
        buf[36..38].copy_from_slice(&MAX_SLOTS.to_le_bytes());
        assert!(
            SlottedPageRef::open(&buf).is_ok(),
            "slot_count == MAX_SLOTS ({MAX_SLOTS}) must be accepted"
        );
        // One over the cap — rejected.
        buf[36..38].copy_from_slice(&(MAX_SLOTS + 1).to_le_bytes());
        let err = SlottedPageRef::open(&buf).unwrap_err();
        assert!(
            matches!(err, PageError::Format(ref m) if m.contains("slot_count")),
            "slot_count == MAX_SLOTS+1 must be a Format reject, got {err:?}"
        );
    }

    /// Defense-in-depth at the root-cause site: even bypassing `open`
    /// (a caller using the module-internal unchecked constructor on an
    /// unvalidated buffer), `slot_entry`/`read_node` over a forged
    /// `slot_count` must return a structured `Format` error rather than
    /// panic on a slice index.
    ///
    /// Pre-fix this test is RED with an out-of-bounds slice panic at the
    /// `MAX_SLOTS`-th slot (offset 8192 in an 8192-byte page). Post-fix it
    /// is GREEN.
    #[test]
    fn slot_entry_bounds_checks_forged_slot_count() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(1)).unwrap();
        }
        buf[36..38].copy_from_slice(&u16::MAX.to_le_bytes());

        // Bypass `open`'s validation deliberately.
        let view = SlottedPageRef::from_bytes_unchecked(&buf);
        // The first slot whose 4-byte directory entry would start at or
        // past PAGE_SIZE is index MAX_SLOTS (off = 40 + 2038*4 = 8192).
        let res = view.read_node(SlotId(MAX_SLOTS));
        assert!(
            matches!(res, Err(PageError::Format(_))),
            "expected a Format error at the out-of-bounds slot, got {res:?}"
        );
        // Iterating the entire forged slot_count must also not panic.
        let _ = view.iter_nodes().count();
        let _ = view.iter_rels().count();
    }

    // ---- #16: forged record-area watermark bound ----
    //
    // Write-path sibling of #592. `insert_raw` back-computes the record-area
    // watermark from `free_space` (header bytes 38..40) and `slot_count`
    // (bytes 36..38). Both fields live OUTSIDE the CRC body (bytes 40..), so a
    // crafted page can forge them while keeping the body checksum valid — it
    // passes `open`'s CRC gate AND its `slot_count <= MAX_SLOTS` bound. Before
    // #16 the unchecked `PAGE_BODY_BYTES - free_space - slot_count*SLOT_SIZE`
    // subtraction underflowed: a debug panic / a release out-of-bounds slice
    // write. These tests pin the fix: a structured `Format` reject, never a
    // panic. Same class as #577/#594 (validate an untrusted length-prefix).

    /// Forge `free_space` to the maximal adversarial value (`u16::MAX`) — the
    /// field is outside the CRC body, so the body checksum stays valid and the
    /// frame is still admitted by `open` (CRC matches, slot_count within cap).
    /// `insert_node` must then return a structured `Format` watermark-underflow
    /// error, NOT panic and NOT write out of bounds.
    ///
    /// RED-on-revert: with the `checked_sub` guard removed, `insert_raw`'s
    /// `PAGE_BODY_BYTES - (free_space as usize)` underflows and PANICS in debug
    /// ("attempt to subtract with overflow"); in release `record_start` wraps
    /// and the `copy_from_slice` slice-indexes out of bounds.
    #[test]
    fn insert_rejects_crafted_oversized_free_space() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(1)).unwrap();
        }
        // Sanity: the well-formed page reopens and accepts a second insert.
        {
            let mut tmp = buf;
            let mut page = SlottedPage::open(&mut tmp).unwrap();
            page.insert_node(&mk_node(2)).unwrap();
        }
        // Forge free_space = u16::MAX (>> PAGE_BODY_BYTES). CRC body untouched.
        buf[38..40].copy_from_slice(&u16::MAX.to_le_bytes());
        // `open` still admits it: CRC matches and slot_count (1) is in cap.
        let mut page = SlottedPage::open(&mut buf).unwrap();
        let err = page.insert_node(&mk_node(2)).unwrap_err();
        assert!(
            matches!(err, PageError::Format(ref m) if m.contains("watermark underflow")),
            "expected a Format watermark-underflow reject, got {err:?}"
        );
    }

    /// Boundary variant: forge `free_space` and `slot_count` so each is
    /// individually plausible (`free_space <= PAGE_BODY_BYTES`,
    /// `slot_count <= MAX_SLOTS`) but their SUM exceeds the record area —
    /// `free_space + slot_count*SLOT_SIZE > PAGE_BODY_BYTES`. Pins that the
    /// guard validates the *combined* watermark invariant, not merely an
    /// individually-oversized `free_space`. Also confirms the `free_space <
    /// needed` gate (8000 >= 68) does not mask the inconsistency.
    #[test]
    fn insert_rejects_crafted_inconsistent_watermark_sum() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            page.insert_node(&mk_node(1)).unwrap();
        }
        // free_space=8000 (<= PAGE_BODY_BYTES=8152), slot_count=100
        // (<= MAX_SLOTS=2038). 8000 + 100*SLOT_SIZE = 8400 > 8152 ⇒ inconsistent.
        buf[36..38].copy_from_slice(&100u16.to_le_bytes());
        buf[38..40].copy_from_slice(&8000u16.to_le_bytes());
        let mut page = SlottedPage::open(&mut buf).unwrap();
        let err = page.insert_node(&mk_node(2)).unwrap_err();
        assert!(
            matches!(err, PageError::Format(ref m) if m.contains("watermark underflow")),
            "expected a Format watermark-underflow reject, got {err:?}"
        );
    }

    /// Happy-path transparency (#16): the checked-subtraction guard must
    /// compute byte-identical record offsets for a well-formed page. Records
    /// grow backward from `PAGE_SIZE`, so record `k` (0-based) lands at
    /// `PAGE_SIZE - (k+1)*NodeRecord::SIZE` — exactly what the pre-guard
    /// formula produced. Strong-oracle pin on the literal offsets: if the
    /// guard ever perturbed the happy path this fails.
    #[test]
    fn insert_raw_guard_is_transparent_to_valid_pages() {
        let mut buf = [0u8; PAGE_SIZE];
        {
            let mut page = fresh_node_page(&mut buf);
            for i in 0..3u64 {
                let slot = page.insert_node(&mk_node(i)).unwrap();
                assert_eq!(page.read_node(slot).unwrap(), Some(mk_node(i)));
            }
        }
        // Slot directory: record k starts at PAGE_SIZE - (k+1)*NodeRecord::SIZE.
        for k in 0..3usize {
            let slot_off = SLOT_AREA_START + k * SLOT_SIZE;
            let off = u16::from_le_bytes([buf[slot_off], buf[slot_off + 1]]) as usize;
            let len = u16::from_le_bytes([buf[slot_off + 2], buf[slot_off + 3]]) as usize;
            assert_eq!(
                off,
                PAGE_SIZE - (k + 1) * NodeRecord::SIZE,
                "record {k} offset must be byte-identical to the pre-guard formula"
            );
            assert_eq!(len, NodeRecord::SIZE);
        }
        // Page reopens cleanly: the guard wrote nothing and kept the CRC valid.
        SlottedPageRef::open(&buf).unwrap();
    }

    // ---- property tests ----

    proptest! {
        #[test]
        fn prop_insert_then_read_any_count(ids in proptest::collection::vec(any::<u64>(), 0..NODE_CAPACITY as usize)) {
            let mut buf = [0u8; PAGE_SIZE];
            let mut page = fresh_node_page(&mut buf);
            let mut slots = Vec::with_capacity(ids.len());
            for id in &ids {
                slots.push(page.insert_node(&mk_node(*id)).unwrap());
            }
            for (slot, id) in slots.iter().zip(ids.iter()) {
                let got = page.read_node(*slot).unwrap().unwrap();
                prop_assert_eq!(got.id, *id);
            }
        }

        #[test]
        fn prop_tombstone_any_subset(
            ids in proptest::collection::vec(any::<u64>(), 1..20usize),
            mask in proptest::collection::vec(any::<bool>(), 1..20usize),
        ) {
            let n = ids.len().min(mask.len());
            let mut buf = [0u8; PAGE_SIZE];
            let mut page = fresh_node_page(&mut buf);
            let mut slots = Vec::with_capacity(n);
            for id in ids.iter().take(n) {
                slots.push(page.insert_node(&mk_node(*id)).unwrap());
            }
            for (i, s) in slots.iter().enumerate() {
                if mask[i] {
                    page.tombstone(*s).unwrap();
                }
            }
            for (i, s) in slots.iter().enumerate() {
                let got = page.read_node(*s).unwrap();
                if mask[i] {
                    prop_assert!(got.is_none());
                } else {
                    prop_assert_eq!(got.unwrap().id, ids[i]);
                }
            }
        }

        #[test]
        fn prop_update_preserves_other_slots(
            ids in proptest::collection::vec(any::<u64>(), 2..10usize),
            target_idx in any::<u8>(),
            new_label in any::<u32>(),
        ) {
            let mut buf = [0u8; PAGE_SIZE];
            let mut page = fresh_node_page(&mut buf);
            let mut slots = Vec::with_capacity(ids.len());
            for id in &ids {
                slots.push(page.insert_node(&mk_node(*id)).unwrap());
            }
            let t = (target_idx as usize) % ids.len();
            let mut r = mk_node(ids[t]);
            r.label_id = new_label;
            page.update_node(slots[t], &r).unwrap();
            for (i, s) in slots.iter().enumerate() {
                let got = page.read_node(*s).unwrap().unwrap();
                if i == t {
                    prop_assert_eq!(got.label_id, new_label);
                } else {
                    prop_assert_eq!(got.id, ids[i]);
                }
            }
        }

        /// #592 fault injection: build a real node page, then forge
        /// `slot_count` to an ARBITRARY `u16` (CRC stays valid — the field
        /// is outside the body range). `open` must never panic; if it
        /// admits the frame, reading every claimed slot must never panic
        /// and the admitted `slot_count` must be within capacity.
        ///
        /// Pre-fix this is RED (the read loop panics with an out-of-bounds
        /// slice index for any forged value `>= MAX_SLOTS + 1`; proptest
        /// shrinks the counterexample to 2039). Post-fix it is GREEN.
        #[test]
        fn prop_forged_slot_count_never_panics(
            forged in any::<u16>(),
            real in 0u64..50,
        ) {
            let mut buf = [0u8; PAGE_SIZE];
            {
                let mut page = fresh_node_page(&mut buf);
                for i in 0..real {
                    let _ = page.insert_node(&mk_node(i));
                }
            }
            buf[36..38].copy_from_slice(&forged.to_le_bytes());

            // A structured reject (oversized slot_count) is the intended
            // path and needs no assertion; only an admitted frame is read.
            if let Ok(view) = SlottedPageRef::open(&buf) {
                // Admitted ⇒ reading every claimed slot must not panic...
                for i in 0..view.slot_count() {
                    let _ = view.read_node(SlotId(i));
                }
                let _ = view.iter_nodes().count();
                // ...and the admitted slot_count is within capacity.
                prop_assert!(view.slot_count() <= MAX_SLOTS);
            }
        }

        /// #16 fault injection: build a real node page, forge `free_space`
        /// to an ARBITRARY `u16` (CRC stays valid — bytes 38..40 are outside
        /// the body range), reopen, and attempt an insert. `insert_node` must
        /// NEVER panic: it returns `Ok` (header consistent enough to fit),
        /// `PageError::Full` (forged small), or `PageError::Format` (watermark
        /// underflow). Pre-fix this is RED — a forged `free_space` past the
        /// record area panics on the unchecked `PAGE_BODY_BYTES - free_space`
        /// subtraction (proptest shrinks the counterexample to 8149, the
        /// smallest value that underflows with slot_count=1).
        #[test]
        fn prop_forged_free_space_insert_never_panics(forged in any::<u16>()) {
            let mut buf = [0u8; PAGE_SIZE];
            {
                let mut page = fresh_node_page(&mut buf);
                page.insert_node(&mk_node(1)).unwrap();
            }
            buf[38..40].copy_from_slice(&forged.to_le_bytes());
            // CRC + slot_count are intact, so `open` admits the frame; the
            // insert must resolve to a structured outcome, never a panic.
            if let Ok(mut page) = SlottedPage::open(&mut buf) {
                let res = page.insert_node(&mk_node(2));
                prop_assert!(
                    matches!(
                        res,
                        Ok(_) | Err(PageError::Full { .. }) | Err(PageError::Format(_))
                    ),
                    "insert over forged free_space={forged} must be a structured result, got {res:?}"
                );
            }
        }
    }

    // ─── v2 M1 — PropSlotted variable-length bag codec (ADR-230 / #1430) ───

    mod prop_bags {
        use super::*;

        fn fresh_prop_page(buf: &mut [u8]) -> SlottedPage<'_> {
            let header = PageHeader::new(PageId::new(77), PageType::PropSlotted, TenantId::DEFAULT);
            SlottedPage::init(buf, header).expect("init PropSlotted page")
        }

        #[test]
        fn bag_insert_then_read_roundtrip_variable_lengths() {
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut page = fresh_prop_page(&mut buf);
            let bags: Vec<Vec<u8>> = vec![
                b"a".to_vec(),
                vec![0x42; 60],
                vec![0x7F; 150],
                vec![0x01; 2048],
            ];
            let mut slots = Vec::new();
            for b in &bags {
                slots.push(page.insert_bag(b).expect("insert"));
            }
            for (slot, b) in slots.iter().zip(&bags) {
                assert_eq!(
                    page.read_bag(*slot).expect("read").expect("live"),
                    &b[..],
                    "bag must round-trip byte-identical"
                );
            }
        }

        #[test]
        fn bag_capacity_127_at_60_bytes() {
            // Design §M1.5 arithmetic: 8152 / (60 + 4) = 127.
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut page = fresh_prop_page(&mut buf);
            for i in 0..127 {
                page.insert_bag(&[(i & 0xFF) as u8; 60])
                    .unwrap_or_else(|e| panic!("bag {i} must fit: {e}"));
            }
            assert!(
                matches!(page.insert_bag(&[0u8; 60]), Err(PageError::Full { .. })),
                "the 128th 60 B bag must report Full"
            );
        }

        #[test]
        fn bag_max_size_fills_page_exactly_and_reads_back() {
            // PROP_BAG_MAX_BYTES = whole body minus one slot: offset ==
            // directory end — the boundary the slot_entry validation
            // deliberately permits.
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut page = fresh_prop_page(&mut buf);
            let bag = vec![0xEE; PROP_BAG_MAX_BYTES];
            let slot = page.insert_bag(&bag).expect("max-size bag must fit");
            assert_eq!(page.free_space(), 0, "page exactly full");
            assert_eq!(page.read_bag(slot).unwrap().unwrap(), &bag[..]);
            assert!(
                matches!(page.insert_bag(b"x"), Err(PageError::Full { .. })),
                "nothing further fits"
            );
        }

        #[test]
        fn bag_rejects_empty_and_oversize() {
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut page = fresh_prop_page(&mut buf);
            assert!(matches!(page.insert_bag(b""), Err(PageError::Format(_))));
            assert!(matches!(
                page.insert_bag(&vec![0u8; PROP_BAG_MAX_BYTES + 1]),
                Err(PageError::Format(_))
            ));
        }

        #[test]
        fn bag_apis_reject_wrong_page_type() {
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut node_page = fresh_node_page(&mut buf);
            assert!(matches!(
                node_page.insert_bag(b"bag"),
                Err(PageError::WrongPageType { .. })
            ));
            let node_ref = node_page.as_ref();
            assert!(matches!(
                node_ref.read_bag(SlotId(0)),
                Err(PageError::WrongPageType { .. })
            ));
            let mut buf2 = vec![0u8; PAGE_SIZE];
            let _ = fresh_prop_page(&mut buf2);
            assert!(matches!(
                SlottedPageRef::open_prop_trusted(&buf[..]),
                Err(PageError::WrongPageType { .. })
            ));
            assert!(SlottedPageRef::open_prop_trusted(&buf2[..]).is_ok());
        }

        #[test]
        fn bag_tombstone_reads_none_and_full_open_validates() {
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut page = fresh_prop_page(&mut buf);
            let s0 = page.insert_bag(b"first").unwrap();
            let s1 = page.insert_bag(b"second").unwrap();
            page.tombstone(s0).unwrap();
            assert!(page.read_bag(s0).unwrap().is_none(), "tombstoned → None");
            assert_eq!(page.read_bag(s1).unwrap().unwrap(), b"second");
            // The mutation-maintained CRC keeps the FULL open green.
            assert!(SlottedPageRef::open(&buf[..]).is_ok());
        }

        proptest! {
            /// Build-plan §2 M1 EXIT 2 — the packing proptest: pack N
            /// random small bags, read each back by slot, assert
            /// byte-equality; a bag that does not fit reports Full
            /// (the caller's open-a-fresh-page signal), never corrupts
            /// the page (all previously packed bags still read back).
            #[test]
            fn prop_pack_random_bags_roundtrip_and_full_is_clean(
                bags in proptest::collection::vec(
                    proptest::collection::vec(any::<u8>(), 1..=4096),
                    1..40
                ),
            ) {
                let mut buf = vec![0u8; PAGE_SIZE];
                let mut page = fresh_prop_page(&mut buf);
                let mut packed: Vec<(SlotId, Vec<u8>)> = Vec::new();
                for b in &bags {
                    match page.insert_bag(b) {
                        Ok(slot) => packed.push((slot, b.clone())),
                        Err(PageError::Full { .. }) => break,
                        Err(e) => prop_assert!(false, "unexpected: {e}"),
                    }
                }
                for (slot, b) in &packed {
                    prop_assert_eq!(
                        page.read_bag(*slot).expect("read").expect("live"),
                        &b[..]
                    );
                }
                // The page stays fully valid under the strict opener.
                prop_assert!(SlottedPageRef::open(&buf[..]).is_ok());
            }
        }
    }
}
