//! TEL (Transactional Edge Log) — LiveGraph-style adjacency blocks.
//!
//! Budget (design-v2 §3.3, ADR-010):
//! - Header: 32 B, cacheline-aligned. Single writer, many readers.
//! - Entry: 32 B ([`TelEntry`]), appended backwards inside the block.
//! - Block sizes: 64 B .. 64 KiB, doubling on overflow. Max 2 047
//!   entries per block (= (65 536 − 32) / 32). design-v2 §3.3 was
//!   amended post-M2.a to match this exact figure — see the
//!   "Why a ~2047-entry cap?" paragraph there and [`MAX_ENTRIES`].
//! - Overflow chain (M2-04): supernodes walk `prev_block_ptr`.
//! - Scan (M2-05): reader snapshots `entry_count` with Acquire at its
//!   MVCC LSN and reads [0..count) from a pinned page. LiveGraph
//!   Theorem 1: scans remain purely sequential under concurrent
//!   single-writer appends.
//!
//! This module defines the in-memory [`TelBlock`] that lives inside a
//! pinned buffer-pool page. M2-01 covers the header + layout; later
//! tasks add append (M2-02), doubling growth (M2-03), scan (M2-05),
//! and the overflow chain (M2-04).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, PageId, TelEntry, TenantId};

/// Size in bytes of the [`TelBlock`] header. 32 B = half a cache line.
pub const HEADER_SIZE: u32 = 32;

/// Size in bytes of one [`TelEntry`] written into the block.
pub const ENTRY_SIZE: u32 = TelEntry::SIZE as u32;

/// Smallest legal block: header + exactly one entry slot.
pub const MIN_BLOCK_BYTES: u32 = HEADER_SIZE + ENTRY_SIZE; // 64

/// Largest legal block: 64 KiB. Growth caps here before overflow
/// chaining kicks in (M2-04). design-v2 §3.3.
pub const MAX_BLOCK_BYTES: u32 = 65_536;

/// Maximum entries a single block can hold. Derived from
/// [`MAX_BLOCK_BYTES`] minus the header, divided by [`ENTRY_SIZE`].
/// design-v2 §3.3 was amended post-M2.a to state this figure
/// precisely (2 047, not the earlier rounded "2 048").
pub const MAX_ENTRIES: u32 = (MAX_BLOCK_BYTES - HEADER_SIZE) / ENTRY_SIZE;

/// Sentinel for "no predecessor block" in the overflow chain.
pub const NO_PREV_BLOCK: u64 = u64::MAX;

// Compile-time layout asserts.
const _: () = assert!(MIN_BLOCK_BYTES == 64);
const _: () = assert!(MAX_BLOCK_BYTES == 65_536);
const _: () = assert!(MAX_ENTRIES == 2047);
const _: () = assert!(ENTRY_SIZE == 32);

/// Errors returned from [`TelBlock`] operations. Kept local to the
/// storage crate: nothing else needs to pattern-match on TEL internals.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TelError {
    /// `block_size` was outside `[MIN_BLOCK_BYTES, MAX_BLOCK_BYTES]` or
    /// not `HEADER_SIZE + k * ENTRY_SIZE` for some `k ≥ 1`.
    #[error("invalid TEL block size: {got} (must be 32+k*32, in [64,65536])")]
    InvalidBlockSize {
        /// Bad size the caller asked for.
        got: u32,
    },

    /// Append attempted on a block at capacity. Caller must grow
    /// (M2-03) or link a new overflow block (M2-04).
    #[error("TEL block full: {count}/{capacity} entries")]
    Full {
        /// Current entry count.
        count: u32,
        /// Max entries this block can hold.
        capacity: u32,
    },

    /// `set_prev_block_ptr` attempted on a block that already has an
    /// overflow predecessor. Chain links are write-once.
    #[error("TEL block already linked to predecessor {existing:?}")]
    AlreadyLinked {
        /// The existing predecessor page id.
        existing: PageId,
    },
}

/// A single TEL block: header fields + backing entries region.
///
/// The block is the LiveGraph unit of adjacency storage for one source
/// vertex, optionally linked to older sibling blocks via
/// `prev_block_ptr` (M2-04). Entries are written from the high end of
/// the entries region toward the low end; the forward-iteration order
/// in `TelScan` (M2-05) visits them oldest-first (index 0) to
/// newest-last (index `entry_count - 1`).
///
/// # Concurrency model
///
/// The block supports **one writer and many readers concurrently**,
/// mediated by the single source of truth `entry_count`:
/// - Writers use Release ordering when publishing a new count.
/// - Readers use Acquire ordering when snapshotting.
/// - Bytes written into an entry slot are published by that Release
///   store, so the reader's Acquire load synchronizes with them.
///
/// Callers are responsible for serializing **writer against writer**;
/// in the buffer pool this is handled by the page latch. In unit tests
/// we enforce it structurally by spawning exactly one appender thread.
#[derive(Debug)]
pub struct TelBlock {
    src_vertex_id: u64,
    label: u32,
    tenant_id: u64,
    block_size: u32,
    capacity_entries: u32,
    entry_count: AtomicU32,
    prev_block_ptr: AtomicU64,
    // Accessors arrive with M2-02 (append) and M2-05 (scan); allowed
    // for M2-01 so the layout commit lands in isolation.
    #[allow(dead_code)]
    entries_buf: Box<[std::cell::UnsafeCell<u8>]>,

    /// Debug-only runtime check that `append` is not called
    /// concurrently. The production correctness story is still the
    /// buffer-pool page latch; this just catches a missing-latch bug
    /// at test time instead of producing torn entries silently.
    /// Compiled out entirely in release builds — zero cost.
    #[cfg(debug_assertions)]
    write_in_progress: std::sync::atomic::AtomicBool,
}

// SAFETY: `TelBlock` is `Sync` because:
// 1. `entry_count` / `prev_block_ptr` are `AtomicU*` — inherently `Sync`.
// 2. Immutable fields (`src_vertex_id`, `label`, `tenant_id`, `block_size`,
//    `capacity_entries`) are `Copy` primitives fixed at construction.
// 3. `entries_buf` wraps `UnsafeCell<u8>` (not `Sync` on its own), but
//    the single-writer / many-reader discipline is upheld by the
//    caller (buffer-pool page latch in production; one appender
//    thread in tests). The `AtomicU32` publication on `entry_count`
//    is the happens-before edge: any byte stored into
//    `entries_buf[slot..slot+32]` before the Release store of `N+1`
//    is visible to readers that Acquire-load `N+1` or higher
//    (LiveGraph Theorem 1).
// 4. In debug builds `write_in_progress` (an `AtomicBool`) adds a
//    runtime check that reinforces — but does not replace — the
//    single-writer discipline: `append` compare-exchanges it and
//    panics on loss. Release-mode builds omit the field entirely;
//    the page latch remains the sole production guarantor.
unsafe impl Sync for TelBlock {}

/// RAII guard: held for the duration of a single `append` call in
/// debug builds. Construction performs a `compare_exchange` that
/// panics on a lost race; `Drop` stores `false` so every exit path —
/// including panics in the middle of `append` — releases the guard.
#[cfg(debug_assertions)]
struct SingleWriterGuard<'a> {
    flag: &'a std::sync::atomic::AtomicBool,
}

#[cfg(debug_assertions)]
impl<'a> SingleWriterGuard<'a> {
    fn acquire(flag: &'a std::sync::atomic::AtomicBool) -> Self {
        use std::sync::atomic::Ordering;
        if flag
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            panic!(
                "TelBlock::append called concurrently — single-writer \
                 discipline violated (buffer-pool page latch missing?)"
            );
        }
        Self { flag }
    }
}

#[cfg(debug_assertions)]
impl Drop for SingleWriterGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Release);
    }
}

impl TelBlock {
    /// Allocate a fresh empty block sized to `block_size` bytes
    /// (header included). `block_size` must be `HEADER_SIZE + k * ENTRY_SIZE`
    /// with `k ∈ [1, MAX_ENTRIES]`.
    pub fn new(
        src: NodeId,
        label: LabelId,
        block_size: u32,
        tenant: TenantId,
    ) -> Result<Self, TelError> {
        if !(MIN_BLOCK_BYTES..=MAX_BLOCK_BYTES).contains(&block_size)
            || (block_size - HEADER_SIZE) % ENTRY_SIZE != 0
        {
            return Err(TelError::InvalidBlockSize { got: block_size });
        }
        let entries_bytes = (block_size - HEADER_SIZE) as usize;
        let capacity_entries = (entries_bytes / ENTRY_SIZE as usize) as u32;

        // Zero-initialize an `UnsafeCell<u8>` region. `UnsafeCell<T>`
        // has the same in-memory representation as `T` (documented
        // guarantee), so a zeroed `Vec<u8>` of the right length is
        // equivalent.
        let buf: Vec<u8> = vec![0u8; entries_bytes];
        let buf: Vec<std::cell::UnsafeCell<u8>> = {
            let mut manual = std::mem::ManuallyDrop::new(buf);
            let (ptr, len, cap) = (manual.as_mut_ptr(), manual.len(), manual.capacity());
            // SAFETY: `UnsafeCell<u8>` is `#[repr(transparent)]` over
            // `u8`; same size, same align, same valid bit patterns.
            // `capacity == len` because `vec![0u8; n]` allocates
            // exactly `n`. We move ownership of the allocation.
            unsafe { Vec::from_raw_parts(ptr.cast::<std::cell::UnsafeCell<u8>>(), len, cap) }
        };

        Ok(Self {
            src_vertex_id: src.raw(),
            label: label.raw(),
            tenant_id: tenant.raw(),
            block_size,
            capacity_entries,
            entry_count: AtomicU32::new(0),
            prev_block_ptr: AtomicU64::new(NO_PREV_BLOCK),
            entries_buf: buf.into_boxed_slice(),
            #[cfg(debug_assertions)]
            write_in_progress: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Source vertex whose outgoing (or incoming) edges this block stores.
    #[inline]
    #[must_use]
    pub fn src_vertex_id(&self) -> NodeId {
        NodeId::new(self.src_vertex_id)
    }

    /// Channel label / relationship-type id.
    #[inline]
    #[must_use]
    pub fn label(&self) -> LabelId {
        LabelId::new(self.label)
    }

    /// Tenant that owns this TEL block.
    #[inline]
    #[must_use]
    pub fn tenant_id(&self) -> TenantId {
        TenantId::new(self.tenant_id)
    }

    /// Total bytes (header + entries region).
    #[inline]
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Max entries this block can hold before `TelError::Full`.
    #[inline]
    #[must_use]
    pub fn capacity_entries(&self) -> u32 {
        self.capacity_entries
    }

    /// Currently-published entry count (Acquire).
    #[inline]
    #[must_use]
    pub fn entry_count(&self) -> u32 {
        self.entry_count.load(Ordering::Acquire)
    }

    /// Read the overflow-chain predecessor pointer. `None` at chain head.
    #[inline]
    #[must_use]
    pub fn prev_block_ptr(&self) -> Option<PageId> {
        match self.prev_block_ptr.load(Ordering::Acquire) {
            NO_PREV_BLOCK => None,
            raw => Some(PageId::new(raw)),
        }
    }

    /// Link this block to an older overflow predecessor.
    ///
    /// Supernodes (vertices with more outgoing edges than fit in a
    /// single [`MAX_BLOCK_BYTES`] block) are represented as a chain of
    /// blocks. The *newest* block is the chain head; each block's
    /// `prev_block_ptr` points at the next-older block. A reader walks
    /// newest→oldest via [`Self::prev_block_ptr`].
    ///
    /// # Contract
    ///
    /// - Called at most once per block, before the block is published
    ///   to concurrent readers, so the pointer is effectively immutable
    ///   after linking.
    /// - `prev` must name a block whose `(src_vertex_id, label)` match
    ///   this block's. The caller (buffer pool / MVCC layer) is
    ///   responsible for checking that; the block itself does not hold
    ///   a reference to `prev`.
    /// - Errors with [`TelError::AlreadyLinked`] if this block already
    ///   has a predecessor — overflow links are write-once.
    ///
    /// `Release` ordering ensures any reader that subsequently
    /// Acquires the pointer sees a fully-initialized predecessor.
    pub fn set_prev_block_ptr(&self, prev: PageId) -> Result<(), TelError> {
        self.prev_block_ptr
            .compare_exchange(
                NO_PREV_BLOCK,
                prev.raw(),
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map(|_| ())
            .map_err(|existing| TelError::AlreadyLinked {
                existing: PageId::new(existing),
            })
    }

    /// Append `entry` to the tail of the log. Entries are written into
    /// the backing buffer from the high end toward the low end: entry
    /// `i` (0-indexed, oldest) occupies
    /// `entries_buf[len - (i+1)*32 .. len - i*32]`.
    ///
    /// Returns the index the entry was stored at (i.e. the pre-append
    /// `entry_count`) on success, or [`TelError::Full`] when this
    /// block is at capacity (caller grows via M2-03 or chains via
    /// M2-04).
    ///
    /// # Concurrency
    ///
    /// `append` takes `&self` but **must be called by at most one
    /// thread at a time** for a given `TelBlock`. In production the
    /// buffer-pool page latch enforces this; in tests, the single
    /// appender thread does. Concurrent *readers* via
    /// [`Self::entry_count`] or the M2-05 `TelScan` are safe: the
    /// Release store on `entry_count` publishes both the new count
    /// and the freshly-written entry bytes (LiveGraph Theorem 1).
    pub fn append(&self, entry: TelEntry) -> Result<u32, TelError> {
        // Debug-only runtime check that the single-writer discipline
        // hasn't been violated (e.g. an MVCC commit path bypassing the
        // buffer-pool page latch). `_guard` holds the flag for the
        // body of this function and releases it on every exit path,
        // including the capacity-`Err` early return below.
        #[cfg(debug_assertions)]
        let _guard = SingleWriterGuard::acquire(&self.write_in_progress);

        // Writer-only load: no other writer is running (single-writer
        // discipline), so `Relaxed` is sufficient to read the count we
        // ourselves last stored.
        let count = self.entry_count.load(Ordering::Relaxed);
        if count >= self.capacity_entries {
            return Err(TelError::Full {
                count,
                capacity: self.capacity_entries,
            });
        }

        let buf_len = self.entries_buf.len();
        // Entry `count` lives at the end of the free region, growing
        // toward lower addresses. Offsets are in bytes.
        let slot_end = buf_len - (count as usize) * (ENTRY_SIZE as usize);
        let slot_start = slot_end - (ENTRY_SIZE as usize);

        let bytes = entry.to_bytes();
        // SAFETY: single-writer contract guarantees no other thread
        // writes into `entries_buf[slot_start..slot_end]` concurrently,
        // and no reader can observe these bytes until the Release
        // store below publishes the new `count`. `slot_start` /
        // `slot_end` are bounded by `buf_len` (checked via the
        // capacity check above), so every pointer write is in-bounds
        // for the allocation. `UnsafeCell::get()` yields a `*mut u8`
        // whose provenance covers the allocation's full range.
        unsafe {
            let base = self.entries_buf.as_ptr().cast::<u8>() as *mut u8;
            let dst = base.add(slot_start);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, ENTRY_SIZE as usize);
        }

        // Publish. Release pairs with Acquire in `entry_count()` and
        // in `TelScan::new`, establishing happens-before over every
        // byte written into the slot above.
        self.entry_count.store(count + 1, Ordering::Release);
        Ok(count)
    }

    /// Read the raw bytes of a committed entry slot. Intended for
    /// tests and the forthcoming `TelScan` (M2-05).
    ///
    /// Returns `None` when `index >= entry_count()` (not yet
    /// published, or out of capacity).
    #[must_use]
    pub fn entry_bytes(&self, index: u32) -> Option<[u8; TelEntry::SIZE]> {
        let count = self.entry_count.load(Ordering::Acquire);
        if index >= count {
            return None;
        }
        let buf_len = self.entries_buf.len();
        let slot_end = buf_len - (index as usize) * (ENTRY_SIZE as usize);
        let slot_start = slot_end - (ENTRY_SIZE as usize);
        let mut out = [0u8; TelEntry::SIZE];
        // SAFETY: `index < count` means the Release store of `count`
        // has already published these bytes; with our Acquire load
        // above, every write into the slot happens-before this read.
        // Slot bounds are in-range by the same arithmetic as `append`.
        unsafe {
            let base = self.entries_buf.as_ptr().cast::<u8>();
            let src = base.add(slot_start);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), ENTRY_SIZE as usize);
        }
        let _ = slot_end; // silence unused-read warning on release builds
        Some(out)
    }

    /// Read a committed entry, decoded into a `TelEntry`.
    #[inline]
    #[must_use]
    pub fn entry_at(&self, index: u32) -> Option<TelEntry> {
        self.entry_bytes(index).map(|b| TelEntry::from_bytes(&b))
    }

    /// Read a committed entry slot and apply the TEL MVCC visibility
    /// predicate used by [`Self::scan`].
    ///
    /// This keeps scan-style readers on one predicate:
    /// `created_lsn <= snapshot_lsn < expired_lsn`.
    #[inline]
    #[must_use]
    pub fn visible_entry_at(&self, index: u32, snapshot_lsn: Lsn) -> Option<TelEntry> {
        self.entry_at(index)
            .filter(|entry| entry.is_visible_at(snapshot_lsn))
    }

    /// Allocate a successor block with `2×` the capacity and copy every
    /// already-published entry into it, preserving insertion order.
    ///
    /// Intended as the "the block filled up" step: the caller swaps
    /// the new block in for the old one (the buffer pool will
    /// allocate a fresh page for it), then resumes appending. The
    /// returned block has a fresh `prev_block_ptr = NO_PREV_BLOCK`;
    /// if the caller is also introducing an overflow chain (M2-04),
    /// linking it is a separate step.
    ///
    /// Returns `None` when the current block is already at
    /// [`MAX_BLOCK_BYTES`] — the caller must then allocate a new
    /// overflow block and link it via `set_prev_block_ptr` (M2-04).
    #[must_use]
    pub fn grown(&self) -> Option<Self> {
        let next_size = next_block_size(self.block_size)?;
        let new_block = Self::new(
            NodeId::new(self.src_vertex_id),
            LabelId::new(self.label),
            next_size,
            TenantId::new(self.tenant_id),
        )
        .expect("next_block_size always returns a legal size");

        let count = self.entry_count.load(Ordering::Acquire);
        // Copy by-index so we preserve the "entry 0 is oldest" layout
        // in the new block's backward-growing slots.
        for i in 0..count {
            let bytes = self
                .entry_bytes(i)
                .expect("index < published count, bytes are visible after Acquire load");
            let entry = TelEntry::from_bytes(&bytes);
            new_block
                .append(entry)
                .expect("fresh block with 2x capacity cannot be full at count < current capacity");
        }
        Some(new_block)
    }
}

impl TelBlock {
    /// Return a [`TelScan`] over this block that filters entries by
    /// their MVCC visibility at `snapshot_lsn`.
    ///
    /// The scan snapshots `entry_count` with Acquire *at construction
    /// time* (LiveGraph: the reader captures the count at its
    /// snapshot LSN and then proceeds sequentially). Appends
    /// committed after construction are not observed — that is the
    /// theorem the concurrent-append proptest locks down.
    ///
    /// Yielded entries satisfy
    /// `created_lsn <= snapshot_lsn < expired_lsn` (same rule the
    /// `NodeRecord` / `RelRecord` visibility helpers use in
    /// `arcgraph-core`). Tombstoned and future edges are skipped.
    #[must_use]
    pub fn scan(&self, snapshot_lsn: Lsn) -> TelScan<'_> {
        TelScan {
            block: self,
            // Acquire: synchronizes with the append Release store,
            // giving us the frozen prefix of entries [0..snapshot).
            snapshot: self.entry_count.load(Ordering::Acquire),
            next: 0,
            snapshot_lsn,
        }
    }
}

/// Forward iterator over the MVCC-visible prefix of a [`TelBlock`] at
/// a fixed snapshot LSN.
///
/// The iterator captures the block's committed entry count at
/// construction (Acquire) and walks `[0..snapshot)` in insertion
/// order. LiveGraph Theorem 1: concurrent single-writer appends past
/// the snapshot do not perturb this walk, because the bytes in
/// `[0..snapshot)` were published before our Acquire load and the
/// writer's Release stores only ever extend toward higher indices.
///
/// `TelScan` is deliberately `!Send`: it borrows the block and is
/// scoped to a single query-executor task.
#[derive(Debug)]
pub struct TelScan<'a> {
    block: &'a TelBlock,
    snapshot: u32,
    next: u32,
    snapshot_lsn: Lsn,
}

impl<'a> TelScan<'a> {
    /// The entry count frozen for this scan, i.e. the upper exclusive
    /// bound on indices this iterator will ever visit.
    #[inline]
    #[must_use]
    pub fn snapshot_count(&self) -> u32 {
        self.snapshot
    }

    /// The MVCC LSN this scan filters visibility against.
    #[inline]
    #[must_use]
    pub fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }
}

impl Iterator for TelScan<'_> {
    type Item = TelEntry;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.snapshot {
            let i = self.next;
            self.next += 1;
            // `i < snapshot` means the bytes for this slot were
            // published before our Acquire load; `visible_entry_at`
            // re-Acquires defensively but keeps the visibility
            // predicate shared with cursor-style scans.
            if let Some(entry) = self.block.visible_entry_at(i, self.snapshot_lsn) {
                return Some(entry);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.snapshot - self.next) as usize;
        // Lower bound is 0 because MVCC filtering can skip every
        // remaining entry; upper bound is the unfiltered count.
        (0, Some(remaining))
    }
}

/// Return the next block size in the doubling schedule.
///
/// Sequence: `64 → 128 → 256 → … → 32 768 → 65 536`. `65 536` is the
/// cap; callers at that size must switch to overflow chaining
/// (M2-04).
#[must_use]
pub fn next_block_size(current: u32) -> Option<u32> {
    if current >= MAX_BLOCK_BYTES {
        return None;
    }
    // Only legal inputs belong in this sequence; anything else is a
    // programmer error. We clamp misuse to `None` rather than panic
    // so the growth step has a total function signature.
    if current < MIN_BLOCK_BYTES || (current - HEADER_SIZE) % ENTRY_SIZE != 0 {
        return None;
    }
    let doubled = current.saturating_mul(2);
    Some(doubled.min(MAX_BLOCK_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn constants_match_design_v2() {
        assert_eq!(HEADER_SIZE, 32);
        assert_eq!(ENTRY_SIZE, 32);
        assert_eq!(MIN_BLOCK_BYTES, 64);
        assert_eq!(MAX_BLOCK_BYTES, 65_536);
        assert_eq!(MAX_ENTRIES, 2_047);
        assert_eq!(NO_PREV_BLOCK, u64::MAX);
    }

    #[test]
    fn new_rejects_too_small() {
        let err =
            TelBlock::new(NodeId::new(1), LabelId::new(2), 32, TenantId::DEFAULT).unwrap_err();
        assert_eq!(err, TelError::InvalidBlockSize { got: 32 });
    }

    #[test]
    fn new_rejects_too_large() {
        let err = TelBlock::new(
            NodeId::new(1),
            LabelId::new(2),
            MAX_BLOCK_BYTES + ENTRY_SIZE,
            TenantId::DEFAULT,
        )
        .unwrap_err();
        assert!(matches!(err, TelError::InvalidBlockSize { .. }));
    }

    #[test]
    fn new_rejects_misaligned() {
        let err =
            TelBlock::new(NodeId::new(1), LabelId::new(2), 65, TenantId::DEFAULT).unwrap_err();
        assert_eq!(err, TelError::InvalidBlockSize { got: 65 });
    }

    #[test]
    fn new_accepts_minimum() {
        let b = TelBlock::new(
            NodeId::new(7),
            LabelId::new(3),
            MIN_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .unwrap();
        assert_eq!(b.block_size(), MIN_BLOCK_BYTES);
        assert_eq!(b.capacity_entries(), 1);
        assert_eq!(b.entry_count(), 0);
        assert_eq!(b.src_vertex_id(), NodeId::new(7));
        assert_eq!(b.label(), LabelId::new(3));
        assert_eq!(b.prev_block_ptr(), None);
        assert_eq!(b.tenant_id(), TenantId::DEFAULT);
    }

    #[test]
    fn new_accepts_maximum() {
        let b = TelBlock::new(
            NodeId::new(1),
            LabelId::new(2),
            MAX_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .unwrap();
        assert_eq!(b.capacity_entries(), MAX_ENTRIES);
    }

    // ---- append: basic behaviour ----

    use arcgraph_core::RelId;

    fn sample_entry(i: u64) -> TelEntry {
        TelEntry::new(NodeId::new(100 + i), RelId::new(200 + i), Lsn::new(i + 1))
    }

    #[test]
    fn append_returns_index_and_increments_count() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        assert_eq!(b.append(sample_entry(0)).unwrap(), 0);
        assert_eq!(b.entry_count(), 1);
        assert_eq!(b.append(sample_entry(1)).unwrap(), 1);
        assert_eq!(b.append(sample_entry(2)).unwrap(), 2);
        assert_eq!(b.entry_count(), 3);
    }

    #[test]
    fn append_reads_back_in_insertion_order() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        for i in 0..5u64 {
            b.append(sample_entry(i)).unwrap();
        }
        for i in 0..5u32 {
            let got = b.entry_at(i).unwrap();
            assert_eq!(got, sample_entry(u64::from(i)));
        }
    }

    #[test]
    fn append_fills_to_capacity_then_errors() {
        let b = TelBlock::new(
            NodeId::new(1),
            LabelId::new(1),
            MIN_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .unwrap();
        assert_eq!(b.capacity_entries(), 1);
        b.append(sample_entry(0)).unwrap();
        let err = b.append(sample_entry(1)).unwrap_err();
        assert_eq!(
            err,
            TelError::Full {
                count: 1,
                capacity: 1,
            }
        );
        // The published count must not have advanced past capacity.
        assert_eq!(b.entry_count(), 1);
    }

    #[test]
    fn entry_at_returns_none_beyond_published_count() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        assert!(b.entry_at(0).is_none());
        b.append(sample_entry(42)).unwrap();
        assert!(b.entry_at(0).is_some());
        assert!(b.entry_at(1).is_none());
        assert!(b.entry_at(u32::MAX).is_none());
    }

    #[test]
    fn entries_occupy_slots_growing_backward() {
        // Confirm the layout contract: entry 0 sits at the TAIL of
        // `entries_buf` and entry N sits `N*32` bytes closer to the
        // head. We verify indirectly by reading two distinct entries
        // back out in order — but also exercise the max block size
        // to make sure the arithmetic does not overflow.
        let b = TelBlock::new(
            NodeId::new(1),
            LabelId::new(1),
            MAX_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .unwrap();
        for i in 0..10u64 {
            b.append(sample_entry(i)).unwrap();
        }
        for i in 0..10u32 {
            assert_eq!(b.entry_at(i).unwrap(), sample_entry(u64::from(i)));
        }
    }

    // ---- M2-03: doubling growth ----

    #[test]
    fn next_block_size_follows_doubling_sequence() {
        let expected = [
            (64u32, Some(128u32)),
            (128, Some(256)),
            (256, Some(512)),
            (512, Some(1024)),
            (1024, Some(2048)),
            (2048, Some(4096)),
            (4096, Some(8192)),
            (8192, Some(16384)),
            (16384, Some(32768)),
            (32768, Some(65536)),
            (65536, None),
        ];
        for (cur, want) in expected {
            assert_eq!(next_block_size(cur), want, "current={cur}");
        }
    }

    #[test]
    fn next_block_size_rejects_invalid_inputs() {
        assert_eq!(next_block_size(0), None);
        assert_eq!(next_block_size(63), None);
        assert_eq!(next_block_size(65), None); // misaligned
        assert_eq!(next_block_size(u32::MAX), None); // > cap
    }

    #[test]
    fn grown_doubles_size_and_preserves_entries() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 64, TenantId::DEFAULT).unwrap();
        b.append(sample_entry(0)).unwrap();
        assert!(b.append(sample_entry(1)).is_err()); // 1-entry block full.

        let g = b.grown().expect("not at cap");
        assert_eq!(g.block_size(), 128);
        assert_eq!(g.capacity_entries(), 3); // (128-32)/32
        assert_eq!(g.entry_count(), 1);
        assert_eq!(g.entry_at(0).unwrap(), sample_entry(0));

        // The successor accepts further appends.
        g.append(sample_entry(1)).unwrap();
        g.append(sample_entry(2)).unwrap();
        assert_eq!(g.entry_count(), 3);
        assert!(g.append(sample_entry(3)).is_err());
    }

    #[test]
    fn grown_returns_none_at_max_size() {
        let b = TelBlock::new(
            NodeId::new(1),
            LabelId::new(1),
            MAX_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .unwrap();
        assert!(b.grown().is_none());
    }

    #[test]
    fn grown_preserves_src_and_label_and_unlinks_prev_ptr() {
        let b = TelBlock::new(NodeId::new(42), LabelId::new(7), 64, TenantId::DEFAULT).unwrap();
        b.append(sample_entry(0)).unwrap();
        let g = b.grown().unwrap();
        assert_eq!(g.src_vertex_id(), NodeId::new(42));
        assert_eq!(g.label(), LabelId::new(7));
        // A `grown` block is a *replacement* — the caller decides
        // whether/what to link into an overflow chain.
        assert_eq!(g.prev_block_ptr(), None);
    }

    proptest! {
        #[test]
        fn grown_preserves_all_entries_in_order(
            initial_capacity_exp in 0u32..=5, // 2^0..=2^5 = 1..=32 entries
            n_entries in 0u32..=32,
        ) {
            let initial_size = HEADER_SIZE + (1u32 << initial_capacity_exp) * ENTRY_SIZE;
            let b = TelBlock::new(NodeId::new(1), LabelId::new(1), initial_size, TenantId::DEFAULT).unwrap();
            let effective = n_entries.min(b.capacity_entries());
            for i in 0..effective {
                b.append(sample_entry(u64::from(i))).unwrap();
            }
            let Some(g) = b.grown() else {
                // Reached the max — acceptable; just skip.
                return Ok(());
            };
            prop_assert_eq!(g.entry_count(), effective);
            for i in 0..effective {
                prop_assert_eq!(g.entry_at(i).unwrap(), sample_entry(u64::from(i)));
            }
        }
    }

    proptest! {
        #[test]
        fn appended_entries_roundtrip_in_order(
            entries in prop::collection::vec(any::<u64>(), 0..=32),
        ) {
            // Sized to fit the largest legal batch (32 entries).
            let b = TelBlock::new(NodeId::new(1), LabelId::new(1), HEADER_SIZE + 32 * ENTRY_SIZE, TenantId::DEFAULT).unwrap();
            let mut sent = Vec::new();
            for (i, seed) in entries.iter().enumerate() {
                let e = TelEntry::new(
                    NodeId::new(*seed),
                    RelId::new(seed.wrapping_add(1)),
                    Lsn::new(i as u64 + 1),
                );
                let idx = b.append(e).unwrap();
                prop_assert_eq!(idx, i as u32);
                sent.push(e);
            }
            prop_assert_eq!(b.entry_count() as usize, sent.len());
            for (i, expected) in sent.iter().enumerate() {
                let got = b.entry_at(i as u32).unwrap();
                prop_assert_eq!(got, *expected);
            }
        }
    }

    proptest! {
        #[test]
        fn accepts_every_legal_size(
            k in 1u32..=MAX_ENTRIES,
            src in any::<u64>(),
            label in any::<u32>(),
        ) {
            let size = HEADER_SIZE + k * ENTRY_SIZE;
            let b = TelBlock::new(NodeId::new(src), LabelId::new(label), size, TenantId::DEFAULT).unwrap();
            prop_assert_eq!(b.block_size(), size);
            prop_assert_eq!(b.capacity_entries(), k);
            prop_assert_eq!(b.entry_count(), 0);
            prop_assert_eq!(b.src_vertex_id(), NodeId::new(src));
            prop_assert_eq!(b.label(), LabelId::new(label));
        }

        #[test]
        fn rejects_misaligned_sizes(
            size in (MIN_BLOCK_BYTES..=MAX_BLOCK_BYTES)
                .prop_filter("must be misaligned", |s| (s - HEADER_SIZE) % ENTRY_SIZE != 0),
        ) {
            let err = TelBlock::new(NodeId::new(1), LabelId::new(1), size, TenantId::DEFAULT).unwrap_err();
            prop_assert_eq!(err, TelError::InvalidBlockSize { got: size });
        }
    }

    // ---- M2-05: TelScan iterator ----

    fn entry_with_lsns(seed: u64, created: u64, expired: u64) -> TelEntry {
        TelEntry {
            dst_id: 100 + seed,
            rel_id: 200 + seed,
            created_lsn: created,
            expired_lsn: expired,
        }
    }

    #[test]
    fn scan_yields_all_visible_entries_in_order() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        for i in 0..5u64 {
            b.append(sample_entry(i)).unwrap();
        }
        // `sample_entry(i)` has created_lsn = i+1, expired_lsn = MAX.
        let seen: Vec<TelEntry> = b.scan(Lsn::new(1_000)).collect();
        assert_eq!(seen.len(), 5);
        for (i, e) in seen.iter().enumerate() {
            assert_eq!(*e, sample_entry(i as u64));
        }
    }

    #[test]
    fn scan_filters_entries_created_after_snapshot() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        // created_lsn = 1, 2, 3, 4, 5
        for i in 0..5u64 {
            b.append(sample_entry(i)).unwrap();
        }
        // Snapshot at LSN 3 must see entries with created_lsn <= 3.
        let seen: Vec<TelEntry> = b.scan(Lsn::new(3)).collect();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].created_lsn, 1);
        assert_eq!(seen[1].created_lsn, 2);
        assert_eq!(seen[2].created_lsn, 3);
    }

    #[test]
    fn scan_filters_expired_entries() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        // entry 0: created=1, expired=5  (visible at 1..5)
        // entry 1: created=2, expired=MAX (alive)
        // entry 2: created=3, expired=4  (visible at 3 only)
        b.append(entry_with_lsns(0, 1, 5)).unwrap();
        b.append(entry_with_lsns(1, 2, u64::MAX)).unwrap();
        b.append(entry_with_lsns(2, 3, 4)).unwrap();

        let at_3: Vec<_> = b.scan(Lsn::new(3)).collect();
        assert_eq!(at_3.len(), 3);
        let at_4: Vec<_> = b.scan(Lsn::new(4)).collect();
        // entry 2 expired at LSN 4 → filtered; entries 0 and 1 remain.
        assert_eq!(at_4.len(), 2);
        assert_eq!(at_4[0].rel_id, 200);
        assert_eq!(at_4[1].rel_id, 201);
        let at_6: Vec<_> = b.scan(Lsn::new(6)).collect();
        // entry 0 expired at LSN 5 → filtered. Only the alive entry
        // (entry 1) remains.
        assert_eq!(at_6.len(), 1);
        assert_eq!(at_6[0].rel_id, 201);
    }

    #[test]
    fn scan_freezes_snapshot_count_at_construction() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        b.append(sample_entry(0)).unwrap();
        b.append(sample_entry(1)).unwrap();

        let scan = b.scan(Lsn::new(1_000));
        assert_eq!(scan.snapshot_count(), 2);

        // Appends after scan construction must NOT be observed by
        // this iterator (LiveGraph theorem).
        b.append(sample_entry(2)).unwrap();
        b.append(sample_entry(3)).unwrap();

        let seen: Vec<_> = scan.collect();
        assert_eq!(seen.len(), 2, "scan must not see post-construction appends");
    }

    #[test]
    fn scan_size_hint_upper_bound_is_unfiltered_remaining() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 256, TenantId::DEFAULT).unwrap();
        for i in 0..3u64 {
            b.append(sample_entry(i)).unwrap();
        }
        let mut scan = b.scan(Lsn::new(1_000));
        assert_eq!(scan.size_hint(), (0, Some(3)));
        scan.next();
        assert_eq!(scan.size_hint(), (0, Some(2)));
    }

    #[test]
    fn scan_on_empty_block_yields_nothing() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        let seen: Vec<TelEntry> = b.scan(Lsn::new(1_000)).collect();
        assert!(seen.is_empty());
    }

    proptest! {
        #[test]
        fn scan_returns_consecutive_prefix_of_visible_entries(
            // Each entry: (seed, created_lsn in 1..100, expired_offset in 0..100).
            specs in prop::collection::vec((any::<u64>(), 1u64..100, 0u64..100), 0..=20),
            snapshot in 1u64..100,
        ) {
            let b = TelBlock::new(NodeId::new(1), LabelId::new(1), HEADER_SIZE + 32 * ENTRY_SIZE, TenantId::DEFAULT).unwrap();
            let mut sent: Vec<TelEntry> = Vec::new();
            for (seed, created, expired_offset) in &specs {
                let expired = if *expired_offset == 0 {
                    u64::MAX
                } else {
                    created.saturating_add(*expired_offset)
                };
                let e = entry_with_lsns(*seed, *created, expired);
                b.append(e).unwrap();
                sent.push(e);
            }
            let filter = |e: &TelEntry| e.is_visible_at(Lsn::new(snapshot));
            let want: Vec<TelEntry> = sent.iter().copied().filter(filter).collect();
            let got: Vec<TelEntry> = b.scan(Lsn::new(snapshot)).collect();
            prop_assert_eq!(got, want);
        }
    }

    // ---- M2-04: overflow chain ----

    #[test]
    fn set_prev_block_ptr_links_on_fresh_block() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        assert_eq!(b.prev_block_ptr(), None);
        b.set_prev_block_ptr(PageId::new(42)).unwrap();
        assert_eq!(b.prev_block_ptr(), Some(PageId::new(42)));
    }

    #[test]
    fn set_prev_block_ptr_is_write_once() {
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        b.set_prev_block_ptr(PageId::new(42)).unwrap();
        let err = b.set_prev_block_ptr(PageId::new(99)).unwrap_err();
        assert_eq!(
            err,
            TelError::AlreadyLinked {
                existing: PageId::new(42),
            }
        );
        // The published predecessor must not have been overwritten.
        assert_eq!(b.prev_block_ptr(), Some(PageId::new(42)));
    }

    #[test]
    fn grown_block_has_no_predecessor_even_if_source_is_linked() {
        // Linking is a caller concern: `grown()` returns a fresh block
        // at a new PageId, so the caller is responsible for attaching
        // the overflow pointer. We pin this behaviour so a future
        // "helpful" refactor doesn't silently inherit the link and
        // break the chain semantics (oldest→newest vs. newest→oldest).
        let b = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
        b.append(sample_entry(0)).unwrap();
        b.set_prev_block_ptr(PageId::new(7)).unwrap();
        let g = b.grown().unwrap();
        assert_eq!(g.prev_block_ptr(), None);
    }

    #[test]
    fn overflow_chain_walk_newest_to_oldest() {
        // Simulate a supernode: three blocks, each holding one entry,
        // chained newest→oldest. We do not have a real buffer pool in
        // this unit test, so we model the walk with a vector indexed by
        // PageId raw value.
        let mut pool: Vec<TelBlock> = Vec::new();
        for i in 0..3u64 {
            let b = TelBlock::new(
                NodeId::new(1),
                LabelId::new(1),
                MIN_BLOCK_BYTES,
                TenantId::DEFAULT,
            )
            .unwrap();
            b.append(sample_entry(i)).unwrap();
            pool.push(b);
        }
        // Chain head is pool[2] (newest); pool[2] → pool[1] → pool[0].
        pool[2].set_prev_block_ptr(PageId::new(1)).unwrap();
        pool[1].set_prev_block_ptr(PageId::new(0)).unwrap();

        let snap = Lsn::new(u64::MAX - 1);
        let mut walked: Vec<u64> = Vec::new();
        let mut cursor: Option<usize> = Some(2);
        while let Some(idx) = cursor {
            for e in pool[idx].scan(snap) {
                walked.push(e.rel_id);
            }
            cursor = pool[idx]
                .prev_block_ptr()
                .map(|p| usize::try_from(p.raw()).unwrap());
        }
        // Newest-first iteration yields entries in reverse append order.
        assert_eq!(walked, vec![200 + 2, 200 + 1, 200]);
    }

    // ---- single-writer guard (debug-only) ----

    /// Many threads hammering `append` on the same block in debug mode
    /// must trip the runtime guard on at least one of them. Runs only
    /// in debug mode; the guard compiles out in release.
    ///
    /// Determinism: the previous two-thread variant was flaky on
    /// ubuntu-latest (#11) because a 2-vCPU scheduler could run one
    /// thread's entire ~50 µs inner loop inside a single time slice
    /// before the other thread was dispatched, leaving zero overlap.
    /// Here we oversubscribe the CPU (`N_THREADS` > likely core count)
    /// and extend the per-thread wall time well past any OS time
    /// quantum, so at least two threads are guaranteed to be
    /// concurrently scheduled across the run — the `Barrier` only
    /// aligns the *start*; the guaranteed overlap comes from the
    /// oversubscription + duration. Threads short-circuit once any
    /// one of them has observed the panic, keeping wall time low on
    /// the happy path.
    #[cfg(debug_assertions)]
    #[test]
    fn append_rejects_concurrent_callers_in_debug_mode() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier};
        use std::thread;

        // 16 threads × 200 000 iterations = 3.2 M total append calls.
        // Even on a single-core CI runner this spans many millions of
        // cycles, far exceeding any scheduler time slice, so at least
        // one preemption is certain to land inside `append`.
        const N_THREADS: usize = 16;
        const ITERS: u32 = 200_000;

        let b = Arc::new(
            TelBlock::new(
                NodeId::new(1),
                LabelId::new(1),
                MAX_BLOCK_BYTES,
                TenantId::DEFAULT,
            )
            .unwrap(),
        );
        let barrier = Arc::new(Barrier::new(N_THREADS));
        let tripped = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..N_THREADS)
            .map(|tid| {
                let b = Arc::clone(&b);
                let bar = Arc::clone(&barrier);
                let tripped = Arc::clone(&tripped);
                thread::spawn(move || {
                    bar.wait();
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        for i in 0..ITERS {
                            if tripped.load(Ordering::Relaxed) {
                                return;
                            }
                            let seed = u64::from(tid as u32) * u64::from(ITERS) + u64::from(i);
                            let _ = b.append(sample_entry(seed));
                        }
                    }));
                    if result.is_err() {
                        tripped.store(true, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            let _ = h.join();
        }

        assert!(
            tripped.load(Ordering::Relaxed),
            "single-writer guard did not fire across {} threads × {} iterations",
            N_THREADS,
            ITERS
        );
    }
}
