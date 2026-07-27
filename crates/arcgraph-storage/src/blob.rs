//! Overflow BLOB property store (M2-31).
//!
//! Turns `PropertyData::Blob(Vec<u8>)` from a
//! [`crate::crud::PropError::OverflowNotYetImplemented`] rejection into
//! a round-trippable BLOB of 1 B .. 1 MiB under the M2.d budget
//! (design-v2 §3.2; M2-31 goal in the M2.d overnight prompt).
//!
//! # Design decisions
//!
//! - **DEC-4 (always-chain).** Every blob is encoded as a chain of
//!   fixed-size pages, regardless of length. No small-blob slotted
//!   optimization for alpha — the slotted variant is an M2.e
//!   optimisation. Single code path, single page format.
//! - **DEC-5 (head-carries-length).** The first page in the chain
//!   carries `total_len`; subsequent pages carry `total_len == 0`.
//!   Chain terminates when `next_page == 0`.
//! - **DEC-6 (WAL as `PutBlob`).** Every blob publish emits one
//!   [`crate::wal::WalRecordType::PutBlob`] record with the full
//!   payload inline. This pushes big payloads through the WAL but
//!   keeps replay trivially reconstructable; see DEC-6 for the cap
//!   analysis (1 MiB payload × expected create rate stays under the
//!   WAL-write budget of §4.4).
//!
//! # Page layout (in-memory for M2-31; on-disk in M2.e)
//!
//! For M2.d, blob chains live in a `DashMap<(TenantId, PageId),
//! Arc<BlobChunk>>` inside the store — the same
//! synthetic-page-id approach the TEL uses (design-v2 §3.3 note on
//! deferred slotting). A later sweep wires this onto
//! [`crate::buffer::BufferPool`] pages; the on-disk layout is kept
//! compatible by making the in-memory `BlobChunk` a direct mirror of
//! the intended page-body bytes:
//!
//! ```text
//!  0..8   next_page   (LE u64; 0 = last page in chain)
//!  8..12  total_len   (LE u32; nonzero on head, 0 elsewhere)
//! 12..16  chunk_len   (LE u32; <= BLOB_CHUNK_BYTES)
//! 16..    chunk bytes
//! ```
//!
//! `BLOB_CHUNK_BYTES` is `PAGE_SIZE - BLOB_PAGE_HEADER = 8176` so one
//! 1 MiB blob fits in `ceil(1_048_576 / 8176) = 129` pages. This is
//! well inside the `OVERFLOW_PAGE_MASK` 48-bit id space (DEC-2).
//!
//! # Tenancy (§A, ADR-011)
//!
//! Every blob page is keyed by `(TenantId, PageId)`. Cross-tenant
//! reads are impossible because the page map key includes the
//! tenant; [`BlobStore::get`] refuses a mismatched tenant with
//! [`BlobError::MissingHead`]. The `blob_tenant_isolated` test
//! proves it.
//!
//! # Garbage collection
//!
//! None in M2.d. Blob pages persist for the life of the process and
//! accumulate even as MVCC versions tombstone. A future M2.e sweep
//! walks live MVCC versions, collects reachable head-page ids, and
//! deletes everything else. Tracked: TODO(#M2-31-GC) — see the M2.d
//! handoff.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arcgraph_core::record::{PAGE_HEADER_VERSION, PAGE_MAGIC, PageHeader, PageType};
use arcgraph_core::{ArcGraphError, Lsn, PAGE_SIZE, PageId, TenantId};
use bytes::Bytes;
use dashmap::DashMap;
use thiserror::Error;

use crate::mutation_log::TxnMutationLog;
use crate::property::{BlobRef, OVERFLOW_PAGE_MASK};
use crate::records::{PROP_BAG_MAX_BYTES, SlotId, SlottedPage, SlottedPageRef};
use crate::wal::{WalHandle, WalRecordType};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/// Size of the per-page header (`next_page` + `total_len` + `chunk_len`).
pub const BLOB_PAGE_HEADER: usize = 16;

/// Bytes of blob payload in each chain page.
pub const BLOB_CHUNK_BYTES: usize = PAGE_SIZE - BLOB_PAGE_HEADER;

/// Hard cap on any single blob (§M2-31 exit criterion). 1 MiB is the
/// upper bound the proptest exercises.
pub const BLOB_MAX_BYTES: usize = 1 << 20;

const _: () = assert!(BLOB_CHUNK_BYTES >= 4096);

/// One chunk-page snapshot a staged blob contributes to the v3
/// `CommitBundle` under
/// [`crate::wal::bundle::BundlePageKind::Blob`] (N-2 / issue #81).
///
/// Pairs the allocator-assigned `PageId` with a full `PAGE_SIZE`
/// buffer encoded via `BlobChunk::encode_page`.
pub type BlobPageSnapshot = (PageId, Box<[u8; PAGE_SIZE]>);

/// Full tenant-qualified blob page image used by checkpoint and backup
/// snapshots.
pub type BlobPageImage = (TenantId, u64, Box<[u8; PAGE_SIZE]>);

/// SVC-1 / #849 / ADR-229 REQ-2 — return type of
/// [`BlobStore::iter_pages_resident_only`]: `(resident (tenant, page_id,
/// PAGE_SIZE bytes), evicted (tenant, page_id))`.
///
/// **#1404 M0 (bounded blob tier):** when the resident-tier bound is
/// engaged (a spill file is attached, [`BlobStore::with_bound`]), pages
/// whose durable image has been captured by a completed checkpoint may be
/// evicted from RAM and spilled to disk. The checkpoint producer's
/// resident-page capture (this iterator) then returns the still-resident
/// pages plus the ids of any pages that were evicted-to-spill, so the
/// producer backfills their durable disk images post-guard
/// (`read_evicted_page_images` → [`BlobStore::read_evicted_page`]). For an
/// unbounded store (no spill attached, the legacy default) nothing is ever
/// evicted and the evicted list is empty.
pub type BlobResidentPages = (Vec<BlobPageImage>, Vec<(TenantId, u64)>);

/// O(1) owner-5 checkpoint frontier captured under the global commit freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverflowCapture {
    count: u64,
    page_frontier: u64,
    epoch: u64,
}

impl OverflowCapture {
    #[must_use]
    pub const fn count(self) -> u64 {
        self.count
    }
}

// ─────────────────────────────────────────────────────────────────────
// #1404 M0 — bounded resident blob-page tier (bounded RSS on ingest)
// ─────────────────────────────────────────────────────────────────────
//
// BACK-OF-ENVELOPE (PD#5). The #1414 heaptrack attribution pinned the
// dominant #1404 resident term at ~8.2 KB/node = one full blob page per
// node's property bag, retained forever in `BlobStore.pages` (a pure,
// never-drained in-memory `DashMap`). At 2M nodes that is ~16 GB of blob
// pages alone; extrapolated it reaches the ~39 GB @2M OOM under a 40 GB
// cap. This tier bounds the RESIDENT blob-page set to a byte watermark
// (default 0.5 × cap, mirroring the #1405 drain design §3 watermark and
// the #1397 WAL-on-disk bound) so RSS is a function of the watermark, NOT
// of ingested node count. Each resident page costs one `Box<[u8; PAGE_SIZE]>`
// = 8 KB; a 4 GB high-watermark holds ~500k resident pages, and everything
// above spills to `blob-spill.db` and re-faults on read.
//
// INV-DURABLE (data-loss guard): a page is evict-eligible ONLY after a
// completed checkpoint has captured its durable image (the `checkpointed`
// bit is set by `iter_pages_resident_only` under the checkpoint freeze).
// So an evicted page's bytes are always durable ≤ `checkpoint_lsn` in the
// ADR-229 snapshot — evicting-before-durable (which would be data loss on
// crash) cannot happen. The spill file is an in-process runtime tier; on
// restart it is discarded and recovery rebuilds `pages` from WAL +
// checkpoint unchanged.
//
// INV-DRAIN (visibility): blob pages are single-image, content-addressed
// by `(tenant, page_id)` — NOT MVCC-snapshot-versioned (see module docs
// § Page layout + the `get` chain walk). A page id, once allocated for a
// chunk, is never rewritten with different content at the same id (the
// allocator is monotonic; a superseded blob is a NEW chain at NEW ids). So
// evict + re-fault of `(tenant, page_id)` returns the same bytes for every
// reader — there is no old-vs-new-image hazard the record page store faces.

/// Default share of the memory cap the resident blob tier may hold before
/// eviction engages, mirroring the #1405 drain design §3 high watermark
/// (`0.5 × cap`, leaving ~2× headroom for the checkpoint 2×-RAM spike +
/// fragmentation + the query working set).
pub const DEFAULT_BLOB_HIGH_WATERMARK_FRACTION: f64 = 0.5;

/// Default low-watermark share (`0.375 × cap`) — eviction drains down to
/// this before disengaging, giving hysteresis so the drain does not thrash
/// on/off at every install (#1405 drain design §3).
pub const DEFAULT_BLOB_LOW_WATERMARK_FRACTION: f64 = 0.375;

/// Operator knobs for the bounded resident blob tier (#1404 M0). Byte caps
/// on the resident blob-page set. `high` engages eviction; `low` is the
/// drain target (hysteresis). Config-strict under the code-quality policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobBoundConfig {
    /// Resident blob-page bytes above which eviction engages.
    pub high_watermark_bytes: u64,
    /// Resident blob-page bytes the drain targets before disengaging
    /// (must be `< high_watermark_bytes` for meaningful hysteresis).
    pub low_watermark_bytes: u64,
}

impl Default for BlobBoundConfig {
    /// The unbounded default: watermarks at `u64::MAX` so eviction never
    /// engages. Only meaningful when a spill file is attached; a store
    /// built via [`BlobStore::new`] carries this + `spill = None` and is
    /// the legacy pure-in-RAM store.
    fn default() -> Self {
        Self {
            high_watermark_bytes: u64::MAX,
            low_watermark_bytes: u64::MAX,
        }
    }
}

impl BlobBoundConfig {
    /// Environment variable naming the bounded blob tier's resident cap in
    /// BYTES. When set on the durable serve path, the BlobStore engages the
    /// bounded tier with `high = 0.5 × cap`, `low = 0.375 × cap`. Unset →
    /// the tier is engaged with [`Self::DEFAULT_RESIDENT_CAP_BYTES`] on the
    /// durable path (the fix is on-by-default) and disengaged on the
    /// in-memory/no-`--data` path.
    pub const ENV_RESIDENT_CAP_BYTES: &'static str = "ARCGRAPH_BLOB_RESIDENT_CAP_BYTES";

    /// Default resident blob-page cap when the env var is unset (4 GiB).
    /// Sized so the bounded tier's steady-state RSS contribution is a fixed
    /// budget (~500k resident 8 KB pages at the `0.5 ×` high watermark),
    /// independent of ingested node count — the #1404 fix. Operators with a
    /// tighter/looser memory budget override via [`Self::ENV_RESIDENT_CAP_BYTES`].
    pub const DEFAULT_RESIDENT_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;

    /// Derive watermarks from a memory cap using the design defaults
    /// (`0.5 × cap` high, `0.375 × cap` low).
    #[must_use]
    pub fn from_cap_bytes(cap_bytes: u64) -> Self {
        let high = (cap_bytes as f64 * DEFAULT_BLOB_HIGH_WATERMARK_FRACTION) as u64;
        let low = (cap_bytes as f64 * DEFAULT_BLOB_LOW_WATERMARK_FRACTION) as u64;
        Self {
            high_watermark_bytes: high.max(PAGE_SIZE as u64),
            low_watermark_bytes: low.max(PAGE_SIZE as u64).min(high.saturating_sub(1).max(1)),
        }
    }

    /// Read the resident cap from [`Self::ENV_RESIDENT_CAP_BYTES`], falling
    /// back to [`Self::DEFAULT_RESIDENT_CAP_BYTES`] when unset or unparsable
    /// (the safe default — a bounded tier is always better than unbounded).
    #[must_use]
    pub fn from_env() -> Self {
        let cap = std::env::var(Self::ENV_RESIDENT_CAP_BYTES)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&c| c >= PAGE_SIZE as u64 * 2)
            .unwrap_or(Self::DEFAULT_RESIDENT_CAP_BYTES);
        Self::from_cap_bytes(cap)
    }
}

/// Append-only durable spill file for evicted blob pages (#1404 M0).
///
/// The re-fault source: when a checkpoint-durable blob page is evicted
/// from the resident `DashMap` to bound RSS, its `PAGE_SIZE` bytes are
/// appended here (fsync-on-write is NOT required — INV-DURABLE is upheld
/// by the checkpoint capture that gated the eviction; the spill is a fast
/// runtime re-read tier, and any crash re-derives the page from WAL +
/// checkpoint on restart). The `(tenant, page_id) → byte-offset` index is
/// kept in RAM; at ~16 B/entry it is ~500× smaller than the 8 KB pages it
/// replaces, so it is bounded-enough (160 MB at 10M pages vs the ~80 GB of
/// pages it stands in for).
///
/// The append file itself is not size-bounded during one process lifetime:
/// re-fault/evict churn appends a new copy and leaves the prior copy as dead
/// space. Storage exhaustion is therefore an expected environmental failure,
/// and every spill read/write must propagate [`BlobError::SpillIo`] rather
/// than aborting the process.
#[derive(Debug)]
pub struct BlobSpill {
    /// The append-only page file. `Mutex` serializes appends + seeks.
    file: Mutex<File>,
    /// `(tenant, page_id) → byte offset of the page in `file``.
    offsets: DashMap<(TenantId, u64), SpillOffset>,
    /// Next append offset (also the current file length).
    write_offset: AtomicU64,
    /// Path (for diagnostics only).
    path: PathBuf,
    /// Deterministic integration-test seam proving checkpoint spill reads do
    /// not execute while the global commit freeze is held.
    #[cfg(debug_assertions)]
    debug_read_gate: Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Deterministic gate around the next checkpoint cursor scan.
    #[cfg(debug_assertions)]
    debug_capture_scan_gate: Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Deterministic gate after eviction samples the resident capture epoch
    /// and before it publishes the spill offset.
    #[cfg(debug_assertions)]
    debug_evict_epoch_gate: Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Deterministic gate after eviction publishes the spill offset epoch and
    /// before its fallback epoch handoff or resident removal.
    #[cfg(debug_assertions)]
    debug_after_evict_epoch_publish_gate:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Deterministic gate between the resident and spill passes of owner-5
    /// capture, used to force a resident-to-spill handoff in that window.
    #[cfg(debug_assertions)]
    debug_after_resident_capture_gate:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// One-shot production-type fault seam for the next spill write. The field
    /// is absent from release builds; debug integration tests use the real
    /// `BlobSpill`/`BlobStore` path rather than a fake I/O implementation.
    #[cfg(debug_assertions)]
    debug_fail_next_write: AtomicBool,
}

#[derive(Debug)]
struct SpillOffset {
    offset: u64,
    is_slotted: bool,
    last_overflow_capture_epoch: AtomicU64,
}

impl BlobSpill {
    /// Open (create/truncate) the spill file at `dir/blob-spill.db`. The
    /// file is process-local scratch — truncated on open, discarded on
    /// restart (recovery rebuilds the blob store from WAL + checkpoint).
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join("blob-spill.db");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self {
            file: Mutex::new(file),
            offsets: DashMap::new(),
            write_offset: AtomicU64::new(0),
            path,
            #[cfg(debug_assertions)]
            debug_read_gate: Mutex::new(None),
            #[cfg(debug_assertions)]
            debug_capture_scan_gate: Mutex::new(None),
            #[cfg(debug_assertions)]
            debug_evict_epoch_gate: Mutex::new(None),
            #[cfg(debug_assertions)]
            debug_after_evict_epoch_publish_gate: Mutex::new(None),
            #[cfg(debug_assertions)]
            debug_after_resident_capture_gate: Mutex::new(None),
            #[cfg(debug_assertions)]
            debug_fail_next_write: AtomicBool::new(false),
        })
    }

    /// Inject one ENOSPC/`StorageFull` failure into the next eviction spill
    /// write.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_fail_next_write_enospc(&self) {
        self.debug_fail_next_write.store(true, Ordering::Release);
    }

    /// Install a one-shot debug barrier around the next spill read.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_gate_next_read(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self
            .debug_read_gate
            .lock()
            .expect("blob spill debug gate mutex poisoned") = Some((entered, release));
    }

    /// Pause the next checkpoint spill-index scan at its first entry.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_gate_next_capture_scan(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self
            .debug_capture_scan_gate
            .lock()
            .expect("blob spill capture gate mutex poisoned") = Some((entered, release));
    }

    /// Pause the next eviction after it samples the resident capture epoch
    /// and before it publishes the spill offset.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_gate_next_evict_epoch_sample(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self
            .debug_evict_epoch_gate
            .lock()
            .expect("blob spill eviction epoch gate mutex poisoned") = Some((entered, release));
    }

    /// Pause the next eviction after it publishes the spill offset epoch and
    /// before its fallback epoch handoff or resident removal.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_gate_after_next_evict_epoch_publish(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self
            .debug_after_evict_epoch_publish_gate
            .lock()
            .expect("blob spill eviction epoch-publish gate mutex poisoned") =
            Some((entered, release));
    }

    /// Pause the next owner-5 capture after its resident pass and before its
    /// spill-index pass.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn __test_gate_after_next_resident_capture(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self
            .debug_after_resident_capture_gate
            .lock()
            .expect("blob spill resident-capture gate mutex poisoned") = Some((entered, release));
    }

    /// Append `page` for `(tenant, page_id)`, recording its offset. If the
    /// page was already spilled, overwrites the index to the newest copy
    /// (append-only file; stale bytes are dead space, reclaimed on the next
    /// process restart when the file is truncated).
    fn write_page(
        &self,
        tenant: TenantId,
        page_id: u64,
        page: &[u8; PAGE_SIZE],
        sampled_capture_epoch: u64,
        resident_capture_epoch: &AtomicU64,
    ) -> Result<(), BlobError> {
        #[cfg(debug_assertions)]
        if self.debug_fail_next_write.swap(false, Ordering::AcqRel) {
            // POSIX reserves errno 28 for ENOSPC on every Unix platform we
            // test. Preserve the raw errno so the Linux regression gate
            // exercises the same `StorageFull` diagnostic as the real fault.
            #[cfg(unix)]
            let error = std::io::Error::from_raw_os_error(28);
            #[cfg(not(unix))]
            let error = std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "injected blob spill storage-full failure",
            );
            return Err(self.io_error("write", tenant, page_id, error));
        }
        let off = self
            .write_offset
            .fetch_add(PAGE_SIZE as u64, Ordering::AcqRel);
        {
            let mut f = self.file.lock().map_err(|_| {
                self.io_error(
                    "write",
                    tenant,
                    page_id,
                    std::io::Error::other("blob spill file mutex poisoned"),
                )
            })?;
            f.seek(SeekFrom::Start(off))
                .map_err(|error| self.io_error("write", tenant, page_id, error))?;
            f.write_all(page.as_ref())
                .map_err(|error| self.io_error("write", tenant, page_id, error))?;
        }
        let key = (tenant, page_id);
        // Publish the new offset and its capture stamp as one entry-guarded
        // transition. The guard serializes against capture's `offsets.get`:
        // either we observe its prior published stamp, or capture observes
        // this offset and stamps it before reaching the spill pass. Loading
        // the resident stamp under the same guard closes the reverse handoff
        // when capture stamps the resident after eviction's initial sample.
        match self.offsets.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                let prior_published_epoch = entry
                    .get()
                    .last_overflow_capture_epoch
                    .load(Ordering::Acquire);
                let reconciled_epoch = sampled_capture_epoch
                    .max(prior_published_epoch)
                    .max(resident_capture_epoch.load(Ordering::Acquire));
                entry.insert(SpillOffset {
                    offset: off,
                    is_slotted: page_image_is_slotted(page),
                    last_overflow_capture_epoch: AtomicU64::new(reconciled_epoch),
                });
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let reconciled_epoch =
                    sampled_capture_epoch.max(resident_capture_epoch.load(Ordering::Acquire));
                entry.insert(SpillOffset {
                    offset: off,
                    is_slotted: page_image_is_slotted(page),
                    last_overflow_capture_epoch: AtomicU64::new(reconciled_epoch),
                });
            }
        }
        Ok(())
    }

    /// Read the spilled page for `(tenant, page_id)`, if present.
    fn read_page(
        &self,
        tenant: TenantId,
        page_id: u64,
    ) -> Result<Option<Box<[u8; PAGE_SIZE]>>, BlobError> {
        let Some(off) = self
            .offsets
            .get(&(tenant, page_id))
            .map(|e| e.value().offset)
        else {
            return Ok(None);
        };
        #[cfg(debug_assertions)]
        if let Some((entered, release)) = self
            .debug_read_gate
            .lock()
            .expect("blob spill debug gate mutex poisoned")
            .take()
        {
            entered.wait();
            release.wait();
        }
        let mut page: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        {
            let mut f = self.file.lock().map_err(|_| {
                self.io_error(
                    "read",
                    tenant,
                    page_id,
                    std::io::Error::other("blob spill file mutex poisoned"),
                )
            })?;
            f.seek(SeekFrom::Start(off))
                .map_err(|error| self.io_error("read", tenant, page_id, error))?;
            f.read_exact(page.as_mut())
                .map_err(|error| self.io_error("read", tenant, page_id, error))?;
        }
        Ok(Some(page))
    }

    fn io_error(
        &self,
        operation: &'static str,
        tenant: TenantId,
        page_id: u64,
        error: std::io::Error,
    ) -> BlobError {
        BlobError::SpillIo {
            operation,
            tenant,
            page_id,
            path: self.path.clone(),
            kind: error.kind(),
            raw_os_error: error.raw_os_error(),
            message: error.to_string(),
        }
    }

    fn missing_entry_error(&self, tenant: TenantId, page_id: u64) -> BlobError {
        BlobError::SpillEntryMissing {
            tenant,
            page_id,
            path: self.path.clone(),
        }
    }

    /// Path to the spill file (diagnostics/tests).
    #[doc(hidden)]
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// What a resident blob-store page IS (v2 M1 — the store is kind-aware).
///
/// - `Chain` — a DEC-4 chained-blob chunk page (the pre-M1 only kind;
///   still produced for payloads > [`PROP_BAG_MAX_BYTES`], the design
///   §M1.2 overflow tail, and read forever for pre-M1 stores).
/// - `Slotted` — a shared [`PageType::PropSlotted`] slotted heap page
///   packing MANY small property bags (v2 M1, ADR-230). The bytes are
///   the page image verbatim (`records.rs` slotted layout, CRC-valid);
///   encode-for-capture is the identity.
///
/// Page-id space is SHARED (one `next_page` allocator), so a given
/// `(tenant, page_id)` is exactly one kind for its lifetime — chains
/// and slotted pages never collide at an id (the allocator is monotone
/// and ids are never re-kinded; a superseded blob is a NEW chain at NEW
/// ids, see INV-DRAIN above; a slotted page only ever receives MORE
/// bags, never changes kind).
#[derive(Debug, Clone)]
enum ResidentKind {
    /// DEC-4 chained-blob chunk page.
    Chain(Arc<BlobChunk>),
    /// v2 M1 shared slotted property-bag heap page (image verbatim).
    Slotted(Arc<[u8; PAGE_SIZE]>),
}

impl ResidentKind {
    /// Encode this page's full `PAGE_SIZE` durable image — the byte
    /// layout the WAL bundle / checkpoint / spill tiers carry. For
    /// `Chain` this is `BlobChunk::encode_page` (unchanged from M0);
    /// for `Slotted` the resident bytes ARE the image (identity copy).
    fn encode_image(&self) -> Box<[u8; PAGE_SIZE]> {
        match self {
            Self::Chain(chunk) => chunk.encode_page(),
            Self::Slotted(bytes) => Box::new(**bytes),
        }
    }
}

/// A resident blob page + its eviction bookkeeping (#1404 M0).
///
/// `checkpointed` is set to `true` by [`BlobStore::iter_pages_resident_only`]
/// when a completed checkpoint captures the page's durable image; only
/// `checkpointed` pages are evict-eligible (INV-DURABLE). Cheap: one
/// `AtomicBool` per resident page, no side map.
#[derive(Debug)]
struct ResidentPage {
    kind: ResidentKind,
    checkpointed: AtomicBool,
    last_overflow_capture_epoch: AtomicU64,
}

impl ResidentPage {
    fn new(kind: ResidentKind) -> Self {
        Self {
            kind,
            checkpointed: AtomicBool::new(false),
            last_overflow_capture_epoch: AtomicU64::new(0),
        }
    }
}

/// Classify a raw `PAGE_SIZE` image as a slotted prop page vs a DEC-4
/// chain chunk (v2 M1). Used at the byte trust boundaries that carry
/// NO explicit kind tag: checkpoint restore (the snapshot blob section
/// is `(tenant, page_id, image)` with no kind byte) and spill re-fault.
/// WAL replay is explicit (`BundlePageKind::Blob` vs `::PropSlotted`)
/// but routes through the same kind-aware installer, so the classifier
/// is the single decode dispatch for all three.
///
/// Soundness: a slotted page starts `PAGE_MAGIC ("ARCG") | version |
/// page_type = PropSlotted`, while a chain page starts with its LE u64
/// `next_page`. Misclassifying a chain as slotted requires
/// `next_page & 0xFFFF_FFFF_FFFF == 0x0902_4743_5241` — i.e. a
/// `next_page` id ≥ ~9.9 × 10^12 (≈ 81 PiB of 8 KiB blob pages; the
/// monotone allocator debug-asserts ids stay under the 48-bit BlobRef
/// bound and no deployment is within three orders of magnitude of this
/// value) — AND the image must then survive the FULL slotted-page
/// validation (header decode + #592 slot-count bound + body CRC32C at
/// the install/re-fault boundary), a further ~2^-32 accidental-pass
/// screen. Misclassifying slotted as chain cannot happen (a slotted
/// page always carries the magic prefix by construction).
fn page_image_is_slotted(page: &[u8; PAGE_SIZE]) -> bool {
    page[0..4] == PAGE_MAGIC.to_le_bytes()
        && page[4] == PAGE_HEADER_VERSION
        && page[5] == PageType::PropSlotted.as_byte()
}

/// Outcome of an eviction attempt on one resident page (#1404 M0 drain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictOutcome {
    /// The page was spilled + dropped from the resident tier.
    Evicted,
    /// The page was already gone (concurrent rollback / GC).
    Gone,
    /// The page is not yet checkpoint-durable — kept resident (INV-DURABLE).
    NotDurable,
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Blob-specific faults surfaced at the BlobStore API boundary. Mapped
/// to [`crate::crud::CrudError`] when they cross into the crud layer
/// (see `From<BlobError> for CrudError`).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BlobError {
    /// Caller passed `b""`. Zero-length blobs collide with the
    /// inline-empty representation (`property_ref == 0`), so the store
    /// rejects them at the boundary.
    #[error("blob payload is empty; zero-length blobs are not supported")]
    Empty,

    /// Caller exceeded [`BLOB_MAX_BYTES`].
    #[error("blob length {0} exceeds M2-31 cap of {BLOB_MAX_BYTES} bytes")]
    TooLarge(usize),

    /// `get` was called with a head-page id that has no entry for the
    /// requested tenant. Either the ref is stale, the tenant is wrong,
    /// or the chain was GC'd.
    #[error("blob head-page {head} not found under tenant {tenant:?}")]
    MissingHead {
        /// Tenant the get was performed under.
        tenant: TenantId,
        /// Raw page id from the caller's BlobRef.
        head: u64,
    },

    /// Chain terminated early: a `next_page` pointer pointed at a page
    /// that does not exist. Indicates corruption or a partial write.
    #[error("blob chain broken at page {at}: {remaining} bytes outstanding")]
    BrokenChain {
        /// The next-page id that could not be resolved.
        at: u64,
        /// How many bytes of payload were still expected.
        remaining: usize,
    },

    /// Chain length disagreed with the head page's `total_len`.
    #[error(
        "blob chain length mismatch: head claims {claimed} bytes, \
         recovered {recovered}"
    )]
    LengthMismatch {
        /// `total_len` from the head page.
        claimed: usize,
        /// Bytes actually walked.
        recovered: usize,
    },

    /// WAL replay saw a `PutBlob` payload that did not decode. Raised
    /// by [`decode_put_blob_payload`].
    #[error("PutBlob wal record rejected: {0}")]
    WalDecode(String),

    /// v2 M1 — a packed-bag read (`slot_id >= 1`) failed: the page is
    /// not a valid `PropSlotted` page, the slot is out of range /
    /// tombstoned, or a chain walk resolved a slotted page. Loud
    /// corruption surface, never a silent empty bag.
    #[error("slotted prop-bag read failed at page {page} slot {slot}: {reason}")]
    SlotRead {
        /// Page id from the caller's BlobRef.
        page: u64,
        /// Raw (1-based) slot field from the caller's BlobRef.
        slot: u16,
        /// What went wrong.
        reason: String,
    },

    /// v2 M1 — the slotted staging path could not initialize or append
    /// to a scratch page (a `records.rs` codec-level failure that is
    /// not `Full`; `Full` is handled internally by opening a fresh
    /// page). Indicates a corrupt pooled image or a codec bug.
    #[error("slotted prop-bag staging failed: {0}")]
    SlotStage(String),

    /// The bounded blob tier could not read or write its process-local spill
    /// file. The resident page is retained (and re-queued on an eviction
    /// failure), so callers may surface/retry the operation without data loss.
    #[error(
        "blob spill {operation} failed for tenant {tenant:?} page {page_id} at {}: {message} ({kind:?}, raw_os_error={raw_os_error:?})",
        path.display()
    )]
    SpillIo {
        /// Spill operation that failed (`"read"` or `"write"`).
        operation: &'static str,
        /// Tenant-qualified spill key.
        tenant: TenantId,
        /// Raw blob page id in the spill key.
        page_id: u64,
        /// Process-local spill file path.
        path: PathBuf,
        /// Stable I/O category suitable for retry policy and tests.
        kind: std::io::ErrorKind,
        /// Original platform errno when the operating system supplied one.
        raw_os_error: Option<i32>,
        /// Operating-system diagnostic text.
        message: String,
    },

    /// A spill offset selected by a checkpoint capture vanished before its
    /// image could be read. This is a typed capture-establishment failure, not
    /// a process panic or a silently omitted page.
    #[error(
        "blob spill entry vanished for tenant {tenant:?} page {page_id} at {}",
        path.display()
    )]
    SpillEntryMissing {
        /// Tenant-qualified spill key.
        tenant: TenantId,
        /// Raw blob page id in the spill key.
        page_id: u64,
        /// Process-local spill file path.
        path: PathBuf,
    },

    /// A page read from spill failed its production page decoder. The spill
    /// file is a byte-trust boundary, so corruption is propagated explicitly.
    #[error(
        "blob spill page corrupt for tenant {tenant:?} page {page_id} at {}: {reason}",
        path.display()
    )]
    SpillCorrupt {
        /// Tenant-qualified spill key.
        tenant: TenantId,
        /// Raw blob page id in the spill key.
        page_id: u64,
        /// Process-local spill file path.
        path: PathBuf,
        /// Decoder/validation failure.
        reason: String,
    },
}

impl From<BlobError> for ArcGraphError {
    fn from(error: BlobError) -> Self {
        Self::Io(std::io::Error::other(error))
    }
}

// ─────────────────────────────────────────────────────────────────────
// BlobChunk (single-page body)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BlobChunk {
    /// Id of the next chain page, or `0` if this is the tail.
    next_page: u64,
    /// Total blob length in bytes. Nonzero only on the head page.
    total_len: u32,
    /// Actual payload bytes for this chunk. `bytes.len() <= BLOB_CHUNK_BYTES`.
    bytes: Bytes,
}

impl BlobChunk {
    /// Serialize this chunk into a full `PAGE_SIZE` buffer matching
    /// the on-disk layout documented at the top of the module:
    ///
    /// ```text
    ///  0..8   next_page   (u64 LE)
    ///  8..12  total_len   (u32 LE; nonzero on head only)
    /// 12..16  chunk_len   (u32 LE; <= BLOB_CHUNK_BYTES)
    /// 16..    chunk bytes (payload), zero-padded to PAGE_SIZE
    /// ```
    ///
    /// Used by the N-2 commit-builder integration to stage blob
    /// chain pages into the v3 `CommitBundle` alongside primary /
    /// secondary / record pages.
    fn encode_page(&self) -> Box<[u8; PAGE_SIZE]> {
        let mut page: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page[0..8].copy_from_slice(&self.next_page.to_le_bytes());
        page[8..12].copy_from_slice(&self.total_len.to_le_bytes());
        let chunk_len =
            u32::try_from(self.bytes.len()).expect("BLOB_CHUNK_BYTES fits in u32 by construction");
        page[12..16].copy_from_slice(&chunk_len.to_le_bytes());
        let payload_end = BLOB_PAGE_HEADER + self.bytes.len();
        debug_assert!(
            payload_end <= PAGE_SIZE,
            "blob chunk payload {} overruns PAGE_SIZE {PAGE_SIZE}",
            self.bytes.len(),
        );
        page[BLOB_PAGE_HEADER..payload_end].copy_from_slice(&self.bytes);
        page
    }

    /// Parse a `PAGE_SIZE` buffer back into a [`BlobChunk`].
    ///
    /// Rejects `chunk_len > BLOB_CHUNK_BYTES` as
    /// [`ArcGraphError::WalCorruption`]. Used by
    /// [`BlobStore::install_or_replace`] when the replay executor
    /// routes a `BundlePageKind::Blob` entry through the
    /// [`BlobStoreHandle`] trait.
    fn decode_page(page: &[u8; PAGE_SIZE]) -> arcgraph_core::Result<Self> {
        let next_page = u64::from_le_bytes(
            page[0..8]
                .try_into()
                .expect("slice of len 8 fits into [u8;8]"),
        );
        let total_len = u32::from_le_bytes(
            page[8..12]
                .try_into()
                .expect("slice of len 4 fits into [u8;4]"),
        );
        let chunk_len = u32::from_le_bytes(
            page[12..16]
                .try_into()
                .expect("slice of len 4 fits into [u8;4]"),
        ) as usize;
        if chunk_len > BLOB_CHUNK_BYTES {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!(
                    "BlobStoreHandle::install_or_replace: chunk_len {chunk_len} > \
                     BLOB_CHUNK_BYTES {BLOB_CHUNK_BYTES}"
                ),
            });
        }
        let payload_end = BLOB_PAGE_HEADER + chunk_len;
        let bytes = Bytes::copy_from_slice(&page[BLOB_PAGE_HEADER..payload_end]);
        Ok(Self {
            next_page,
            total_len,
            bytes,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// v2 M1 — txn-exclusive slotted staging + per-tenant open-page pool
// ─────────────────────────────────────────────────────────────────────
//
// CONCURRENCY DESIGN (the RULE-MT / M0.x atomic-capture lesson, baked
// in by construction rather than by locking discipline):
//
// A shared slotted page is owned by AT MOST ONE in-flight transaction
// at a time. A transaction CHECKS OUT the tenant's open page from
// `props_open_pool` (or initializes a fresh one) into its private
// `txn_slotted` scratch, appends its bags there over the txn's life,
// has its FINAL image captured once at commit-bundle build
// (`snapshot_txn_slotted_pages`), and CHECKS the page BACK IN only
// after the commit's WAL append assigned its LSN
// (`publish_txn_slotted`, called on commit success). For
// read-your-own-writes (the owning txn's scans deref its staged bags
// BEFORE commit — exactly the pre-M1 chain visibility timing), every
// append ALSO eagerly publishes an immutable SNAPSHOT of the scratch
// image to the resident tier; rollback restores the pre-checkout
// image / removes a fresh page. Consequences:
//
// - No writer ever mutates a page image another txn (or the
//   checkpoint capture, or a reader) can observe — capture is
//   atomic-by-exclusivity, not by lock choreography. The published
//   resident image only ever changes by whole-image supersession
//   (each snapshot is a fresh immutable Arc; committed slots' bytes
//   are identical across successive snapshots — appends only).
// - For any page P, the txns that touch P are strictly serialized by
//   the checkout, and check-in happens only after WAL-append: if t2
//   acquires P after t1, then LSN(t2) > LSN(t1) AND image(t2) ⊇
//   image(t1) (appends only — committed slots and their payload bytes
//   are never moved or rewritten). Replay applies whole-image
//   overwrites in LSN order, so it converges to the newest image and
//   every committed ref's bytes are present in it. No torn hybrid, no
//   capture-order/LSN-order inversion (the M0.x FIX-D class).
// - A concurrent txn that finds the tenant's pool slot empty (page
//   checked out) simply opens a FRESH page: fill efficiency degrades
//   under concurrency, correctness does not.
//
// RESIDENCY (OOM-guardrail census): `props_open_pool` holds ≤ 1 page
// image per ACTIVE tenant (8 KiB each) and `txn_slotted` holds the
// pages of IN-FLIGHT transactions only (drained on commit success,
// commit failure, and the explicit-abort discard path — the same
// lifecycle as `pending_blob_emits`). Neither grows with ingested
// data volume; both are O(active-tenants + in-flight-txns).

/// Minimum free bytes for a slotted page to be worth re-pooling as the
/// tenant's open page. Below this a typical incident-shape bag
/// (~60–150 B + 4 B slot) no longer fits, so the page is effectively
/// sealed and check-in would just thrash the pool slot.
const MIN_POOL_FREE_BYTES: u16 = 256;

/// TEST-ONLY escape hatch: `=1` routes EVERY bag through the DEC-4
/// chain path (the pre-M1 representation), bypassing slotted packing.
/// Two sanctioned uses, both test-side:
/// 1. **Fixture construction** — building a genuine pre-M1 chained
///    store for the migrate-on-open gates (the M1 binary otherwise
///    packs small bags, so a chained fixture is unreachable).
/// 2. **The RED-on-revert lever** (build-plan §2 M1 EXIT 5) — running
///    the batch WAL-B/node headline gate under this knob reproduces
///    the pre-M1 ~8,454 B/node and MUST fail the ≤ ~600 assertion,
///    proving the gate detects a coalescing revert.
///
/// NOT a supported production mode (chains remain a valid, readable
/// representation — this only changes what NEW writes produce). Read
/// once per process (`OnceLock`).
pub const ENV_M1_FORCE_CHAINED_BAGS: &str = "ARCGRAPH_M1_FORCE_CHAINED_BAGS";

/// Cached read of [`ENV_M1_FORCE_CHAINED_BAGS`].
fn force_chained_bags() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| std::env::var(ENV_M1_FORCE_CHAINED_BAGS).is_ok_and(|v| v == "1"))
}

/// The tenant's current open (partially-filled) slotted page, held by
/// the pool between transactions. The pool owns its OWN image copy —
/// independent of the resident tier, so eviction/re-fault of the
/// published copy never races the pool.
#[derive(Debug)]
struct PooledPropsPage {
    page_id: u64,
    image: Box<[u8; PAGE_SIZE]>,
    /// Header `free_space` at check-in — lets `stage_bag` decide
    /// whether the pooled page fits a bag without opening the image.
    free_space: u16,
}

/// Where a scratch page came from — drives rollback restoration.
#[derive(Debug)]
enum ScratchOrigin {
    /// Freshly allocated by this txn; on rollback nothing was ever
    /// published or pooled, the page id is simply burned (the
    /// allocator is monotone, ids are never reused — INV-DRAIN).
    Fresh,
    /// Checked out of the tenant pool; `pre_image` is the exact pooled
    /// state at checkout, restored to the pool on rollback so a
    /// rolled-back txn leaves the open page byte-identical.
    Pooled {
        pre_image: Box<[u8; PAGE_SIZE]>,
        pre_free_space: u16,
    },
}

/// One slotted page privately owned by an in-flight transaction.
#[derive(Debug)]
struct ScratchPage {
    tenant: TenantId,
    page_id: u64,
    image: Box<[u8; PAGE_SIZE]>,
    origin: ScratchOrigin,
    initial_slot_count: u16,
}

/// All slotted pages an in-flight transaction has staged bags into.
#[derive(Debug, Default)]
struct TxnSlottedScratch {
    pages: Vec<ScratchPage>,
}

// ─────────────────────────────────────────────────────────────────────
// BlobStore
// ─────────────────────────────────────────────────────────────────────

/// BLOB store. Publishes chains synchronously; not GC'd in M2.d (see
/// module docs on GC).
///
/// **#1404 M0 (bounded resident tier).** The `pages` map is the RESIDENT
/// hot tier. Without a spill file attached ([`BlobStore::new`], the legacy
/// default) it is an unbounded in-memory `DashMap` (unchanged behavior).
/// With a spill attached ([`BlobStore::with_bound`], the durable-serve
/// path), the resident set is bounded to `config.high_watermark_bytes`:
/// once a checkpoint has captured a page's durable image, that page is
/// evict-eligible and may be spilled to `blob-spill.db` to bound RSS,
/// re-faulting on read (see the tier design comment above `BlobBoundConfig`).
#[derive(Debug, Default)]
pub struct BlobStore {
    pages: DashMap<(TenantId, u64), Arc<ResidentPage>>,
    /// Synthetic page-id allocator (global across tenants). A fresh
    /// store starts it at 0 so the first `alloc_page_range` returns id
    /// `1`, keeping `0` reserved as the chain-terminator sentinel.
    ///
    /// **P0 #820 — durable across restart.** This counter is NOT carried
    /// in the `CommitBundle` `allocator_advances` section, so WAL replay
    /// re-seeds it from the recovered chains via
    /// [`BlobStoreHandle::install_or_replace`] (`fetch_max` per installed
    /// page). Without that seed a `--data` restart resets the allocator
    /// and post-restart property-blob writes reuse recovered page-ids,
    /// corrupting acked data on the next restart.
    next_page: AtomicU64,
    /// Logical DEC-4 overflow pages (resident + spilled), maintained on
    /// fresh publish/remove so checkpoint capture reads the owner count O(1).
    overflow_pages: AtomicU64,
    /// Monotone capture generation used to classify resident-vs-spilled pages
    /// without collecting the evicted owner under the commit freeze.
    overflow_capture_epoch: AtomicU64,

    // ── #1404 M0 bounded-tier state (inert unless `spill` is `Some`) ──
    /// Durable spill tier for evicted pages + the re-fault source. `None`
    /// = unbounded legacy behavior (nothing is ever evicted).
    spill: Option<Arc<BlobSpill>>,
    /// Resident-tier byte watermarks. Only consulted when `spill.is_some()`.
    config: BlobBoundConfig,
    /// Running count of resident page bytes (`pages.len() × PAGE_SIZE`),
    /// maintained as an `AtomicU64` so the trigger is a load, not a scan.
    resident_bytes: AtomicU64,
    /// FIFO eviction order: `(tenant, page_id)` in publish order. Cheap to
    /// maintain on the write path (one push); eviction pops from the front.
    /// If the front page is not yet checkpoint-durable, the drain restores it
    /// to the front and stops (FIFO-oldest-first ⟹ nothing behind a
    /// non-durable front is durable), so the FIFO evicts the oldest
    /// checkpoint-durable prefix and defers the rest to the next drain. See
    /// [`Self::drain_to_low_watermark`].
    evict_queue: Mutex<VecDeque<(TenantId, u64)>>,
    /// Count of evictions performed — test/observability only.
    evicted_count: AtomicU64,
    /// Count of re-faults from spill — test/observability only.
    refault_count: AtomicU64,
    /// Cumulative count of evict-queue probes across all drain passes — one
    /// per pop-front / evict-attempt in [`Self::drain_to_low_watermark`].
    /// Test/observability only: the #1404 M0 throughput-regression guard
    /// asserts per-publish drain probes stay O(evicted-this-pass) (O(1) in
    /// the all-NotDurable regime), NOT O(resident-page-count).
    drain_probe_count: AtomicU64,

    // ── v2 M1 — slotted small-blob packing state ──
    /// Per-in-flight-transaction private slotted scratch pages, keyed by
    /// txn id. Populated by [`Self::stage_bag`], captured by
    /// [`Self::snapshot_txn_slotted_pages`] at commit-bundle build,
    /// drained by [`Self::publish_txn_slotted`] (commit success) or
    /// [`Self::rollback_txn_slotted`] (commit failure / explicit-abort
    /// discard). O(in-flight txns) — see the module-level residency note.
    txn_slotted: DashMap<u64, TxnSlottedScratch>,
    /// Per-tenant open (partially-filled) slotted page pool — at most one
    /// entry per tenant, checked out exclusively by one txn at a time
    /// (see the v2 M1 concurrency design note above). O(active tenants).
    props_open_pool: Mutex<HashMap<TenantId, PooledPropsPage>>,
}

/// v2 M2 — a property-bag byte view returned by [`BlobStore::get_bag`]:
/// either a zero-copy range over a resident slotted page's immutable
/// `Arc` image, or the owned reassembly of a DEC-4 chain. Deref's to
/// the bag bytes either way.
#[derive(Debug, Clone)]
pub enum BagBytes {
    /// Zero-copy: a range over the page image (the `Arc` keeps the
    /// image alive + immutable — see [`BlobStore::get_bag`]'s
    /// soundness note).
    Paged {
        /// The shared page image.
        image: Arc<[u8; PAGE_SIZE]>,
        /// Bag start offset within the image.
        start: usize,
        /// Bag byte length.
        len: usize,
    },
    /// Owned bytes (chained bags reassemble across pages).
    Owned(Bytes),
}

impl std::ops::Deref for BagBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            BagBytes::Paged { image, start, len } => &image[*start..*start + *len],
            BagBytes::Owned(b) => b,
        }
    }
}

/// A blob chain produced by [`BlobStore::stage`] that has not yet
/// been installed into the store. Consumed by [`BlobStore::publish`].
///
/// The split between staging and publishing is the engine of review
/// block C-2 (ADR-022): the WAL `PutBlob` record MUST be fsynced
/// before the in-memory DashMap receives the entries, so that a
/// crash between WAL write and publish leaves no durable reference
/// pointing at phantom bytes.
#[derive(Debug)]
pub struct StagedBlob {
    blob_ref: BlobRef,
    pages: Vec<(u64, Arc<BlobChunk>)>,
}

impl StagedBlob {
    /// The head [`BlobRef`] the eventual `publish` will resolve to.
    #[must_use]
    pub fn blob_ref(&self) -> BlobRef {
        self.blob_ref
    }

    /// Serialize every chunk page in this staged chain to a
    /// `(page_id, PAGE_SIZE bytes)` pair suitable for inclusion in
    /// the v3 `CommitBundle`'s `staged_pages` section under
    /// [`crate::wal::bundle::BundlePageKind::Blob`].
    ///
    /// N-2 (issue #81) staging entry point: the CRUD commit builder
    /// drains these into the outer bundle so replay reconstructs
    /// the in-memory `BlobStore` chain pages via
    /// [`BlobStoreHandle::install_or_replace`] — closing the post-
    /// replay `BlobStoreError::MissingHead` gap that PR #79's X-2
    /// left open for `PropertyData::Blob`.
    ///
    /// Returns pages in chain order (head first). Each `page_id` is
    /// the DashMap key under `(tenant, page_id)`; the accompanying
    /// `PageId` on the bundle entry is just a newtype wrap of the
    /// same `u64`.
    #[must_use]
    pub fn page_bytes(&self) -> Vec<BlobPageSnapshot> {
        self.pages
            .iter()
            .map(|(page_id, chunk)| (PageId::new(*page_id), chunk.encode_page()))
            .collect()
    }
}

impl BlobStore {
    /// Install one checkpoint-durable page from an M3 physical base. The page
    /// is immediately eligible for bounded-tier eviction, so generation open
    /// never requires a whole `props.store` owner in RAM.
    pub fn install_m3_base_page(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<[u8; PAGE_SIZE]>,
    ) -> arcgraph_core::Result<()> {
        BlobStoreHandle::install_or_replace(self, tenant, page_id, page)?;
        let resident = self.pages.get(&(tenant, page_id.raw())).ok_or_else(|| {
            ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!("M3 base page {} vanished during install", page_id.raw()),
            }
        })?;
        resident.checkpointed.store(true, Ordering::Release);
        drop(resident);
        self.maybe_drain()?;
        Ok(())
    }

    /// Empty, UNBOUNDED store (the legacy default + the test/no-`--data`
    /// path). No spill file → nothing is ever evicted → identical behavior
    /// to the pre-#1404 store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// #1404 M0 — an empty store with the BOUNDED resident tier engaged.
    ///
    /// `spill` is the durable re-fault tier ([`BlobSpill::open`]); `config`
    /// sets the resident-byte watermarks. Once a checkpoint captures a
    /// page's durable image, that page is evict-eligible and may be spilled
    /// to bound RSS, re-faulting from `spill` on read. Used by the durable
    /// serve bootstrap; tests build it directly to exercise eviction.
    #[must_use]
    pub fn with_bound(spill: Arc<BlobSpill>, config: BlobBoundConfig) -> Self {
        Self {
            spill: Some(spill),
            config,
            ..Self::default()
        }
    }

    /// True iff the bounded resident tier is engaged (a spill file is
    /// attached). Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn is_bounded(&self) -> bool {
        self.spill.is_some()
    }

    /// Current resident blob-page bytes (the RSS this store contributes,
    /// modulo `DashMap`/`Arc` overhead). Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Acquire)
    }

    /// Number of pages evicted-to-spill since construction. Test only.
    #[doc(hidden)]
    #[must_use]
    pub fn evicted_count(&self) -> u64 {
        self.evicted_count.load(Ordering::Acquire)
    }

    /// Number of pages re-faulted from spill since construction. Test only.
    #[doc(hidden)]
    #[must_use]
    pub fn refault_count(&self) -> u64 {
        self.refault_count.load(Ordering::Acquire)
    }

    /// Cumulative evict-queue probes across all drain passes since
    /// construction — one per front-pop / evict-attempt in
    /// [`Self::drain_to_low_watermark`]. Test only: the #1404 M0
    /// throughput-regression guard uses this to assert per-publish drain cost
    /// is O(evicted-this-pass), not O(resident-page-count).
    #[doc(hidden)]
    #[must_use]
    pub fn drain_probe_count(&self) -> u64 {
        self.drain_probe_count.load(Ordering::Relaxed)
    }

    /// Force a drain pass (evict checkpoint-durable pages down to the low
    /// watermark). Exposed for integration tests + the durable serve
    /// bootstrap's explicit post-checkpoint drain hook; production also
    /// drains inline on `publish` (`maybe_drain`).
    #[doc(hidden)]
    pub fn force_drain_for_test(&self) -> Result<(), BlobError> {
        self.drain_to_low_watermark()
    }

    /// Snapshot every installed blob page as `(tenant, page_id,
    /// PAGE_SIZE bytes)`. Used by the ADR-229 checkpoint producer to
    /// capture the full blob page-image set into the durable checkpoint
    /// snapshot; restore feeds each page back through
    /// [`BlobStoreHandle::install_or_replace`] (the same replay entry
    /// point), which reconstructs the chain AND re-seeds `next_page`.
    /// Iteration order is arbitrary.
    pub fn iter_pages(&self) -> Result<Vec<BlobPageImage>, BlobError> {
        // Full-image capture: resident pages encode from RAM; if the
        // bounded tier is engaged, evicted pages are faulted from spill so
        // the returned set is complete (unbounded store never spills, so
        // this is a pure in-RAM walk there). Used by the eager `iter_pages`
        // callers (backup / non-freeze snapshot) — NOT the freeze-critical
        // `iter_pages_resident_only`, which stays non-faulting.
        let (mut resident, evicted) = self.iter_pages_resident_only();
        for (tenant, page_id) in evicted {
            if let Some(page) = self.read_evicted_page(tenant, page_id)? {
                resident.push((tenant, page_id, page));
            }
        }
        Ok(resident)
    }

    /// SVC-1 / #849 / ADR-229 REQ-2 — NON-FAULTING resident-page iterator
    /// for the checkpoint producer. Returns `(resident pages, evicted
    /// (tenant, page_id) ids)`.
    ///
    /// **#1404 M0.** For the UNBOUNDED store (no spill) nothing is ever
    /// evicted, so the evicted list is empty and the returned set is
    /// complete under the freeze (unchanged from pre-#1404). For the
    /// BOUNDED store this returns the RESIDENT pages (a non-faulting RAM
    /// byte-copy, so the checkpoint freeze never blocks on disk) and the
    /// ids of pages currently spilled; the producer reads those durable
    /// images post-guard via [`Self::read_evicted_page`].
    ///
    /// Crucially, capturing a resident page here MARKS it `checkpointed`
    /// (its durable image is about to land in the checkpoint snapshot),
    /// which makes it evict-eligible — this is the INV-DURABLE gate: a
    /// page is evictable ONLY after a completed checkpoint captured it.
    #[must_use]
    pub fn iter_pages_resident_only(&self) -> BlobResidentPages {
        // Sorted (tenant, page_id) order: DashMap iteration order is
        // nondeterministic, and the captured images land byte-for-byte in
        // the checkpoint metadata — a loader-built generation must be
        // byte-identical for any worker count / rerun (INV-M5.24), so the
        // capture order is pinned. Keys only (16 B each) are collected;
        // page images still stream one at a time.
        let keys = self.sorted_resident_keys();
        let mut resident = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(e) = self.pages.get(&key) else {
                continue;
            };
            // Mark checkpoint-durable → evict-eligible (INV-DURABLE gate).
            // Ordering::Release so the eviction path's Acquire load sees it.
            e.value().checkpointed.store(true, Ordering::Release);
            resident.push((key.0, key.1, e.value().kind.encode_image()));
        }
        (resident, self.evicted_page_ids())
    }

    /// Stable capture order for the checkpoint paths (see above).
    fn sorted_resident_keys(&self) -> Vec<(TenantId, u64)> {
        let mut keys: Vec<(TenantId, u64)> = self.pages.iter().map(|entry| *entry.key()).collect();
        keys.sort_unstable();
        keys
    }

    /// #1404 M0.x FIX-B — the STREAMING resident-page capture: emit each
    /// resident page `(tenant, page_id, &PAGE_SIZE-bytes)` through `f` ONE at a
    /// time (encode, emit, then DROP the 8 KB page before the next), so the
    /// checkpoint producer never pre-collects a whole `Vec` of 8 KB page copies
    /// under the freeze. That was the M0.5 whole-in-RAM shape AGAIN: a
    /// `Vec::with_capacity` populated with one `encode_page()` copy per resident
    /// page, O(cap) ≈ +2 GiB/checkpoint that STACKS toward the 40 GB budget.
    /// Returns the evicted page-ids (small: just `(tenant, page_id)` pairs) for
    /// the producer's post-guard supplement. Sets the INV-DURABLE `checkpointed`
    /// gate on each captured page exactly as [`Self::iter_pages_resident_only`]
    /// does. Wire-emitted bytes are byte-identical (same order, same
    /// `encode_page`), so the snapshot layout is unchanged.
    ///
    /// Cost: peak resident inside the capture is ONE `Box<[u8; PAGE_SIZE]>`
    /// (8 KB) + the sink's `BufWriter`, NOT O(resident-page-count) — the FIX-B
    /// bound.
    pub fn for_each_resident_page<F, E>(
        &self,
        mut f: F,
    ) -> std::result::Result<Vec<(TenantId, u64)>, E>
    where
        F: FnMut(TenantId, u64, &[u8; PAGE_SIZE]) -> std::result::Result<(), E>,
    {
        // Sorted capture order — see `iter_pages_resident_only` (INV-M5.24).
        for key in self.sorted_resident_keys() {
            let Some(e) = self.pages.get(&key) else {
                continue;
            };
            e.value().checkpointed.store(true, Ordering::Release);
            // Encode ONE page, emit it, then DROP it before the next iteration.
            let page = e.value().kind.encode_image();
            f(key.0, key.1, page.as_ref())?;
            // `page` (the ONE 8 KB copy) drops here.
        }
        Ok(self.evicted_page_ids())
    }

    /// Capture the total DEC-4 overflow count plus the stable evicted-page id
    /// set without faulting any spill image from disk. The checkpoint calls
    /// this under `TxnManager::checkpoint_freeze`, then performs spill reads
    /// only after releasing the freeze (REQ-2).
    #[must_use]
    pub fn overflow_page_count_and_evicted_ids(&self) -> (u64, Vec<(TenantId, u64)>) {
        let resident = self
            .pages
            .iter()
            .filter(|entry| matches!(&entry.value().kind, ResidentKind::Chain(_)))
            .count() as u64;
        let evicted = match &self.spill {
            None => Vec::new(),
            Some(spill) => spill
                .offsets
                .iter()
                .filter(|entry| !entry.value().is_slotted && !self.pages.contains_key(entry.key()))
                .map(|entry| *entry.key())
                .collect(),
        };
        (resident + evicted.len() as u64, evicted)
    }

    /// Capture the owner-5 logical count plus a stable page/epoch frontier in
    /// O(1). Callers take the brief global commit freeze around only this
    /// method; all page-id scans and image encoding happen later.
    #[must_use]
    pub fn capture_overflow_frontier(&self) -> OverflowCapture {
        OverflowCapture {
            count: self.overflow_pages.load(Ordering::Acquire),
            page_frontier: self.next_page.load(Ordering::Acquire),
            epoch: self
                .overflow_capture_epoch
                .fetch_add(1, Ordering::AcqRel)
                .checked_add(1)
                .expect("overflow capture epoch exhausted"),
        }
    }

    /// Stream the immutable owner-5 image set selected by `capture` with O(1)
    /// caller-owned memory. Resident pages carry the capture epoch into any
    /// racing eviction; the spill pass then skips those images and emits only
    /// pages not already observed resident.
    pub fn for_each_captured_overflow_page<F, E>(
        &self,
        capture: OverflowCapture,
        mut f: F,
    ) -> std::result::Result<u64, E>
    where
        F: FnMut(TenantId, u64, &[u8; PAGE_SIZE]) -> std::result::Result<(), E>,
        E: From<BlobError>,
    {
        let mut streamed = 0u64;
        // Sorted capture order — same INV-M5.24 determinism pin as
        // `iter_pages_resident_only`: these images land byte-for-byte in
        // the v9 incremental metadata.
        for key in self.sorted_resident_keys() {
            let Some(entry) = self.pages.get(&key) else {
                continue;
            };
            let (tenant, page_id) = key;
            if page_id > capture.page_frontier
                || !matches!(&entry.value().kind, ResidentKind::Chain(_))
            {
                continue;
            }
            entry
                .value()
                .last_overflow_capture_epoch
                .store(capture.epoch, Ordering::Release);
            entry.value().checkpointed.store(true, Ordering::Release);
            if let Some(spill) = &self.spill
                && let Some(offset) = spill.offsets.get(&(tenant, page_id))
            {
                offset
                    .last_overflow_capture_epoch
                    .fetch_max(capture.epoch, Ordering::AcqRel);
            }
            let page = entry.value().kind.encode_image();
            f(tenant, page_id, page.as_ref())?;
            streamed += 1;
        }

        if let Some(spill) = &self.spill {
            #[cfg(debug_assertions)]
            if let Some((entered, release)) = spill
                .debug_after_resident_capture_gate
                .lock()
                .expect("blob spill resident-capture gate mutex poisoned")
                .take()
            {
                entered.wait();
                release.wait();
            }
            #[cfg(debug_assertions)]
            let mut scan_gate = spill
                .debug_capture_scan_gate
                .lock()
                .expect("blob spill capture gate mutex poisoned")
                .take();
            for entry in spill.offsets.iter() {
                #[cfg(debug_assertions)]
                if let Some((entered, release)) = scan_gate.take() {
                    entered.wait();
                    release.wait();
                }
                let (tenant, page_id) = *entry.key();
                if entry.value().is_slotted
                    || page_id > capture.page_frontier
                    || entry
                        .value()
                        .last_overflow_capture_epoch
                        .load(Ordering::Acquire)
                        == capture.epoch
                {
                    continue;
                }
                drop(entry);
                let page = spill
                    .read_page(tenant, page_id)
                    .map_err(E::from)?
                    .ok_or_else(|| E::from(spill.missing_entry_error(tenant, page_id)))?;
                f(tenant, page_id, page.as_ref())?;
                streamed += 1;
            }
        }
        Ok(streamed)
    }

    /// Stream resident overflow pages only; property-slotted pages are
    /// excluded because M3 flushes them through store 0.
    pub fn for_each_resident_overflow_page<F, E>(&self, mut f: F) -> std::result::Result<u64, E>
    where
        F: FnMut(TenantId, u64, &[u8; PAGE_SIZE]) -> std::result::Result<(), E>,
    {
        let mut streamed = 0u64;
        for entry in self.pages.iter() {
            let ResidentKind::Chain(_) = &entry.value().kind else {
                continue;
            };
            let (tenant, page_id) = *entry.key();
            entry.value().checkpointed.store(true, Ordering::Release);
            let page = entry.value().kind.encode_image();
            f(tenant, page_id, page.as_ref())?;
            streamed += 1;
        }
        Ok(streamed)
    }

    /// Mark one store-0 home write checkpoint-durable for bounded-tier
    /// eviction, then enforce the resident watermark.
    pub fn mark_m3_page_checkpointed(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<(), BlobError> {
        if let Some(page) = self.pages.get(&(tenant, page_id.raw())) {
            page.checkpointed.store(true, Ordering::Release);
        }
        self.maybe_drain()
    }

    /// #1404 M0.x FIX-B — the evicted (spilled-not-resident) page-ids, shared by
    /// [`Self::iter_pages_resident_only`] + [`Self::for_each_resident_page`].
    /// Only the bounded store spills; the unbounded store returns an empty list.
    fn evicted_page_ids(&self) -> Vec<(TenantId, u64)> {
        match &self.spill {
            None => Vec::new(),
            Some(spill) => {
                let mut ids: Vec<(TenantId, u64)> = spill
                    .offsets
                    .iter()
                    .map(|e| *e.key())
                    .filter(|k| !self.pages.contains_key(k))
                    .collect();
                // Stable order — same INV-M5.24 determinism pin as the
                // resident capture above.
                ids.sort_unstable();
                ids
            }
        }
    }

    /// #1404 M0 — read an evicted blob page's durable image from the spill
    /// tier, for the checkpoint producer's post-guard evicted-supplement
    /// backfill (`read_evicted_page_images`). Returns `None` for an
    /// unbounded store or an id that is not spilled (resident-only).
    pub fn read_evicted_page(
        &self,
        tenant: TenantId,
        page_id: u64,
    ) -> Result<Option<Box<[u8; PAGE_SIZE]>>, BlobError> {
        let Some(spill) = self.spill.as_ref() else {
            return Ok(None);
        };
        spill.read_page(tenant, page_id)
    }

    /// Allocate page ids and construct the chunk chain for `bytes`
    /// **without** publishing it to the in-memory map. Paired with
    /// [`Self::publish`] (and [`Self::put_logged`] which sequences
    /// stage → WAL fsync → publish).
    ///
    /// Errors on empty / over-cap payloads just like [`Self::put`].
    pub fn stage(&self, tenant: TenantId, bytes: &[u8]) -> Result<StagedBlob, BlobError> {
        let _ = tenant;
        if bytes.is_empty() {
            return Err(BlobError::Empty);
        }
        if bytes.len() > BLOB_MAX_BYTES {
            return Err(BlobError::TooLarge(bytes.len()));
        }

        let chunk_count = bytes.len().div_ceil(BLOB_CHUNK_BYTES);
        // Reserve the full range of page ids up front so the chain is
        // contiguous in allocation order (easier to reason about in
        // tests; not required for correctness).
        let first_page = self.alloc_page_range(chunk_count);

        let total_len_u32 =
            u32::try_from(bytes.len()).expect("BLOB_MAX_BYTES fits in u32 by construction");

        let mut pages = Vec::with_capacity(chunk_count);
        for (i, chunk) in bytes.chunks(BLOB_CHUNK_BYTES).enumerate() {
            let page_id = first_page + i as u64;
            let next_page = if i + 1 == chunk_count { 0 } else { page_id + 1 };
            let total_len = if i == 0 { total_len_u32 } else { 0 };
            pages.push((
                page_id,
                Arc::new(BlobChunk {
                    next_page,
                    total_len,
                    bytes: Bytes::copy_from_slice(chunk),
                }),
            ));
        }

        debug_assert!(
            first_page <= OVERFLOW_PAGE_MASK,
            "blob page-id {first_page} overflows 48-bit BlobRef encoding"
        );
        Ok(StagedBlob {
            blob_ref: BlobRef::new(first_page, 0),
            pages,
        })
    }

    /// Install a [`StagedBlob`] into the resident map under `tenant`.
    ///
    /// After this call the chain is reachable via [`Self::get`]. For the
    /// bounded tier (#1404 M0), each newly resident page is enrolled in the
    /// eviction FIFO and, if the resident set now exceeds the high
    /// watermark, the checkpoint-durable oldest pages are drained to spill
    /// to bound RSS (see `Self::drain_to_low_watermark`).
    pub fn publish(&self, tenant: TenantId, staged: StagedBlob) -> Result<BlobRef, BlobError> {
        for (page_id, chunk) in staged.pages {
            self.insert_resident(tenant, page_id, ResidentKind::Chain(chunk));
        }
        // #1404 M0 — engage the drain if we crossed the high watermark.
        // No-op for the unbounded store (watermark = u64::MAX).
        self.maybe_drain()?;
        Ok(staged.blob_ref)
    }

    /// Insert a page into the resident tier, maintaining the resident-byte
    /// counter + the eviction FIFO. A fresh id adds `PAGE_SIZE` to the
    /// counter; overwriting an existing id (replay supersession, or a v2 M1
    /// slotted page re-published with more bags) leaves the count unchanged.
    fn insert_resident(&self, tenant: TenantId, page_id: u64, kind: ResidentKind) {
        let new_is_overflow = matches!(&kind, ResidentKind::Chain(_));
        let prev = self
            .pages
            .insert((tenant, page_id), Arc::new(ResidentPage::new(kind)));
        let old_is_overflow = prev
            .as_ref()
            .is_some_and(|page| matches!(&page.kind, ResidentKind::Chain(_)));
        match (old_is_overflow, new_is_overflow) {
            (false, true) => {
                self.overflow_pages.fetch_add(1, Ordering::AcqRel);
            }
            (true, false) => {
                self.overflow_pages.fetch_sub(1, Ordering::AcqRel);
            }
            _ => {}
        }
        if prev.is_none() {
            self.resident_bytes
                .fetch_add(PAGE_SIZE as u64, Ordering::AcqRel);
            if self.spill.is_some() {
                self.evict_queue
                    .lock()
                    .expect("blob evict_queue mutex poisoned")
                    .push_back((tenant, page_id));
            }
        }
    }

    /// Publish `bytes` as a blob under `tenant`. Returns a [`BlobRef`]
    /// whose `page_id` points at the head of the chain and whose
    /// `slot_id` is `0` (always-chained; see DEC-4).
    ///
    /// This is the no-WAL convenience path. Production callers on the
    /// CRUD write path MUST go through [`Self::put_logged`] so the
    /// blob is durable before the owning MVCC commit references it.
    pub fn put(&self, tenant: TenantId, bytes: &[u8]) -> Result<BlobRef, BlobError> {
        let staged = self.stage(tenant, bytes)?;
        self.publish(tenant, staged)
    }

    /// Allocate one page id from this store's SHARED slotted + chain
    /// page-id space for an offline producer that writes the page image
    /// somewhere else (the M5-D2 fresh loader's STORE_PROPS extent bag
    /// pages — `m4_migration::FreshV6Builder`). Allocations here are
    /// monotone with the chain allocations [`Self::put`] / stage perform,
    /// so loader-built slotted pages and loader-staged chains can never
    /// collide at an id; cold open then re-seeds `next_page` above every
    /// loaded id through `install_or_replace` / `install_m3_base_page`
    /// `fetch_max` (P0 #820), exactly like replay.
    #[must_use]
    pub fn allocate_shared_page_id(&self) -> u64 {
        self.alloc_page_range(1)
    }

    /// Publish `bytes` under `tenant`, appending a WAL `PutBlob`
    /// record BEFORE installing the chain in the in-memory map.
    ///
    /// Ordering (review block C-2 / ADR-022):
    /// 1. `stage` — allocate page ids and build the chunk chain.
    /// 2. `wal.append(PutBlob, …)` — blocks until the record is
    ///    fsynced. Pre-staging gives us the head page id that the
    ///    WAL payload embeds.
    /// 3. `publish` — install the chunks so readers can resolve them.
    ///
    /// A crash between (2) and (3) loses the in-memory publish but
    /// the WAL record has the bytes; recovery replays the record and
    /// reinstalls the chain. A crash between (1) and (2) loses the
    /// staged page ids entirely — but they were never visible to any
    /// caller, so nothing downstream references them.
    pub fn put_logged(
        &self,
        wal: &WalHandle,
        tenant: TenantId,
        bytes: &[u8],
    ) -> Result<BlobRef, BlobError> {
        let (blob_ref, _emits) = self.put_logged_and_stage(wal, tenant, bytes)?;
        Ok(blob_ref)
    }

    /// Same as [`Self::put_logged`] but ALSO returns the chain's
    /// per-page `(page_id, PAGE_SIZE bytes)` snapshots so the owning
    /// transaction's commit-builder can fold them into the v3
    /// `CommitBundle` under [`crate::wal::bundle::BundlePageKind::Blob`].
    ///
    /// N-2 (issue #81) staging entry point. Layered over
    /// `stage → WAL fsync → publish` such that the captured page
    /// bytes are the post-publish payload; replay's
    /// [`BlobStoreHandle::install_or_replace`] reconstructs the
    /// same in-memory chain on recovery, closing the
    /// `BlobStoreError::MissingHead` gap that PR #79's X-2 left
    /// open.
    ///
    /// The PutBlob WAL record is still emitted for pre-v4 back-
    /// compat (Lemma I2 permits double-apply — `install_or_replace`
    /// is an unconditional byte-overwrite). Deprecation is a
    /// follow-up issue per the N-2 prompt.
    pub fn put_logged_and_stage(
        &self,
        wal: &WalHandle,
        tenant: TenantId,
        bytes: &[u8],
    ) -> Result<(BlobRef, Vec<BlobPageSnapshot>), BlobError> {
        let staged = self.stage(tenant, bytes)?;
        let payload = encode_put_blob_payload(staged.blob_ref.page_id, bytes);
        wal.append(
            WalRecordType::PutBlob,
            /* txn_id = */ 0,
            now_millis(),
            tenant,
            payload,
        )
        .map_err(|e| BlobError::WalDecode(format!("wal append failed: {e}")))?;
        // Capture page snapshots BEFORE publish consumes the
        // StagedBlob. `page_bytes` takes `&self`, so ordering-wise
        // this is stage → wal.append (fsync) → capture → publish.
        // The pages it returns are the post-write bytes (they
        // mirror what publish installs into the DashMap).
        let pages = staged.page_bytes();
        let blob_ref = self.publish(tenant, staged)?;
        Ok((blob_ref, pages))
    }

    /// No-WAL companion to [`Self::put_logged_and_stage`]. Intended
    /// for tests that don't spawn a WAL writer but still want the
    /// captured chain pages to thread into a builder.
    pub fn put_and_stage(
        &self,
        tenant: TenantId,
        bytes: &[u8],
    ) -> Result<(BlobRef, Vec<BlobPageSnapshot>), BlobError> {
        let staged = self.stage(tenant, bytes)?;
        let pages = staged.page_bytes();
        let blob_ref = self.publish(tenant, staged)?;
        Ok((blob_ref, pages))
    }

    // ── v2 M1 — slotted small-blob staging (the DEC-4 lift) ──

    /// Stage a property bag for `txn_id` under `tenant` (v2 M1, ADR-230
    /// / design §M1.2 — the write-path entry that replaces per-bag
    /// [`Self::put_and_stage`] for SMALL bags).
    ///
    /// - `bytes.len() <= PROP_BAG_MAX_BYTES` → the bag is APPENDED into
    ///   the transaction's private slotted scratch page (checked out of
    ///   the tenant pool, or fresh — see the module concurrency note).
    ///   Returns a `BlobRef` whose slot field is LOAD-BEARING and
    ///   1-based (`slot_id = slot + 1`; 0 stays the chain discriminant)
    ///   and an EMPTY snapshot vec — the page image is captured ONCE
    ///   per bundle by [`Self::snapshot_txn_slotted_pages`], which is
    ///   the ~14× batch WAL amortization (one shared image per bundle,
    ///   not one dedicated 8 KiB chain per bag).
    /// - larger bags → the DEC-4 chain path, byte-for-byte unchanged
    ///   (stage + publish + per-page snapshots, `slot_id = 0`).
    ///
    /// Back-of-envelope (PD#5, design §M1.5): a 60 B bag + 4 B slot
    /// packs 127/page ⟹ a 100-bag batch commit stages ONE shared 8 KiB
    /// image instead of 100 dedicated ones (819 KB → 8 KB of page
    /// images; ~600 B/node WAL all-in vs 8,454). Single-record commits
    /// still stage the whole (one-bag) page image — no amortization
    /// until M3's delta WAL (the ADR-230 M1 honesty line).
    pub fn stage_bag(
        &self,
        tenant: TenantId,
        txn_id: u64,
        bytes: &[u8],
    ) -> Result<(BlobRef, Vec<BlobPageSnapshot>), BlobError> {
        if bytes.is_empty() {
            return Err(BlobError::Empty);
        }
        if bytes.len() > BLOB_MAX_BYTES {
            return Err(BlobError::TooLarge(bytes.len()));
        }
        if bytes.len() > PROP_BAG_MAX_BYTES || force_chained_bags() {
            // Overflow tail (design §M1.2): large payloads keep the
            // DEC-4 chain representation, path untouched. (The
            // `force_chained_bags` leg is the TEST-ONLY fixture /
            // RED-on-revert lever — see ENV_M1_FORCE_CHAINED_BAGS.)
            return self.put_and_stage(tenant, bytes);
        }

        let mut scratch = self.txn_slotted.entry(txn_id).or_default();

        // 1. Try the txn's current open scratch page (same tenant).
        if let Some(sp) = scratch.pages.last_mut() {
            if sp.tenant == tenant {
                match Self::append_bag_to_image(&mut sp.image, bytes) {
                    Ok(slot) => {
                        // Read-your-own-writes (see the eager-publish
                        // note below): refresh the resident snapshot so
                        // the OWNING txn's scans can deref this bag
                        // before commit — exactly the pre-M1 chain
                        // visibility timing.
                        self.insert_resident(
                            tenant,
                            sp.page_id,
                            ResidentKind::Slotted(Arc::from(sp.image.clone())),
                        );
                        return Ok((BlobRef::new(sp.page_id, slot.raw() + 1), Vec::new()));
                    }
                    Err(crate::records::PageError::Full { .. }) => {
                        // Fall through: seal this scratch page (it stays
                        // in the scratch for capture/publish) and open
                        // another below.
                    }
                    Err(e) => return Err(BlobError::SlotStage(e.to_string())),
                }
            }
        }

        // 2. Check the tenant's pooled open page out (only if this bag
        //    actually fits its remaining free space), else init fresh.
        let needed = bytes.len() as u16 + crate::records::SLOT_SIZE as u16;
        let pooled = {
            let mut pool = self
                .props_open_pool
                .lock()
                .expect("props_open_pool mutex poisoned");
            match pool.get(&tenant) {
                Some(p) if p.free_space >= needed => pool.remove(&tenant),
                _ => None,
            }
        };
        let mut sp = match pooled {
            Some(p) => {
                let pre_image = p.image.clone();
                let initial_slot_count = SlottedPageRef::open_prop_trusted(&p.image[..])
                    .map_err(|e| BlobError::SlotStage(e.to_string()))?
                    .slot_count();
                ScratchPage {
                    tenant,
                    page_id: p.page_id,
                    image: p.image,
                    origin: ScratchOrigin::Pooled {
                        pre_image,
                        pre_free_space: p.free_space,
                    },
                    initial_slot_count,
                }
            }
            None => {
                let page_id = self.alloc_page_range(1);
                debug_assert!(
                    page_id <= OVERFLOW_PAGE_MASK,
                    "slotted prop page-id {page_id} overflows 48-bit BlobRef encoding"
                );
                let mut image: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
                let header = PageHeader::new(PageId::new(page_id), PageType::PropSlotted, tenant);
                SlottedPage::init(&mut image[..], header)
                    .map_err(|e| BlobError::SlotStage(e.to_string()))?;
                ScratchPage {
                    tenant,
                    page_id,
                    image,
                    origin: ScratchOrigin::Fresh,
                    initial_slot_count: 0,
                }
            }
        };
        // Must fit: a fresh page holds up to PROP_BAG_MAX_BYTES by
        // construction, and a pooled page was fit-checked above.
        let slot = Self::append_bag_to_image(&mut sp.image, bytes)
            .map_err(|e| BlobError::SlotStage(e.to_string()))?;
        let blob_ref = BlobRef::new(sp.page_id, slot.raw() + 1);
        // EAGER PUBLISH (read-your-own-writes — the pre-M1 visibility
        // timing, which held-txn MERGE's match branch depends on: the
        // owning txn's scans read its OWN staged record and deref this
        // bag BEFORE commit; chains have always been resident from
        // stage time). The resident tier gets an immutable SNAPSHOT of
        // the scratch image per append (whole-Arc replacement — readers
        // never observe a mutating buffer, and committed slots' bytes
        // are identical in every successive snapshot). Records of OTHER
        // txns can't reference these slots (MVCC hides them), so the
        // only reader is the owner. Rollback restores the pre-checkout
        // image (pooled) or removes the entry (fresh) — see
        // `rollback_txn_slotted`. The commit-time BUNDLE capture
        // (`snapshot_txn_slotted_pages`) and its once-per-bundle
        // coalescing are unchanged by this in-process visibility
        // choice.
        self.insert_resident(
            tenant,
            sp.page_id,
            ResidentKind::Slotted(Arc::from(sp.image.clone())),
        );
        scratch.pages.push(sp);
        Ok((blob_ref, Vec::new()))
    }

    /// Append one bag into a txn-private page image. The image's CRC is
    /// valid by construction (init + every prior `insert_bag` recompute
    /// it), so the open skips the CRC pass (`open_prop_trusted` — see
    /// its trust-boundary note).
    fn append_bag_to_image(
        image: &mut [u8; PAGE_SIZE],
        bag: &[u8],
    ) -> Result<SlotId, crate::records::PageError> {
        let mut page = SlottedPage::open_prop_trusted(&mut image[..])?;
        page.insert_bag(bag)
    }

    /// Capture the FINAL image of every slotted page `txn_id` staged
    /// bags into — one `(page_id, PAGE_SIZE bytes)` snapshot per
    /// TOUCHED PAGE (not per bag), for the commit builder to fold into
    /// the `CommitBundle` under `BundlePageKind::PropSlotted`. This is
    /// the once-per-bundle coalescing (design §M1.3): reverting it to
    /// per-bag staging is exactly the RED-on-revert regression the M1
    /// headline gate pins.
    ///
    /// Does NOT drain the scratch — [`Self::publish_txn_slotted`]
    /// (commit success) or [`Self::rollback_txn_slotted`] (failure)
    /// finish the lifecycle.
    #[must_use]
    pub fn snapshot_txn_slotted_pages(&self, txn_id: u64) -> Vec<BlobPageSnapshot> {
        match self.txn_slotted.get(&txn_id) {
            None => Vec::new(),
            Some(s) => s
                .pages
                .iter()
                .map(|sp| (PageId::new(sp.page_id), sp.image.clone()))
                .collect(),
        }
    }

    /// Build M3 `props.store` intents for only the slots appended by this
    /// transaction. A pooled page may contain older committed slots; those
    /// are deliberately excluded so WAL volume remains proportional to this
    /// commit rather than to the page's lifetime contents.
    pub(crate) fn snapshot_txn_slotted_delta_intents(
        &self,
        txn_id: u64,
    ) -> Result<Vec<crate::wal::delta::DeltaIntent>, BlobError> {
        use crate::wal::delta::{DeltaIntent, DeltaOpKind, STORE_PROPS};

        let Some(scratch) = self.txn_slotted.get(&txn_id) else {
            return Ok(Vec::new());
        };
        let mut intents = Vec::new();
        for page in &scratch.pages {
            let view = SlottedPageRef::open_prop_trusted(&page.image[..])
                .map_err(|error| BlobError::SlotStage(error.to_string()))?;
            if matches!(page.origin, ScratchOrigin::Fresh) {
                intents.push(DeltaIntent::page_alloc(
                    STORE_PROPS,
                    page.tenant,
                    page.page_id,
                    PageType::PropSlotted,
                    page.page_id,
                ));
            }
            for slot in page.initial_slot_count..view.slot_count() {
                let payload = view
                    .read_bag(SlotId(slot))
                    .map_err(|error| BlobError::SlotStage(error.to_string()))?
                    .ok_or_else(|| {
                        BlobError::SlotStage(format!(
                            "new props.store slot {slot} is unexpectedly tombstoned"
                        ))
                    })?;
                intents.push(DeltaIntent {
                    kind: DeltaOpKind::PutPropBlock,
                    store_id: STORE_PROPS,
                    tenant_id: page.tenant,
                    page_no: page.page_id,
                    slot,
                    payload: Bytes::copy_from_slice(payload),
                });
            }
        }
        Ok(intents)
    }

    /// Apply this transaction's durable v9 props deltas to its private
    /// scratch images before publishing them to the resident tier.
    pub(crate) fn apply_txn_slotted_deltas(
        &self,
        txn_id: u64,
        deltas: &[crate::wal::delta::DeltaOp],
        commit_lsn: Lsn,
    ) -> arcgraph_core::Result<()> {
        use crate::wal::delta::{DeltaOpKind, STORE_PROPS};

        let props: Vec<_> = deltas
            .iter()
            .filter(|delta| {
                delta.store_id == STORE_PROPS
                    && matches!(
                        delta.kind,
                        DeltaOpKind::PageAlloc | DeltaOpKind::PutPropBlock
                    )
            })
            .collect();
        if props.is_empty() {
            return Ok(());
        }
        let mut scratch =
            self.txn_slotted
                .get_mut(&txn_id)
                .ok_or_else(|| ArcGraphError::WalCorruption {
                    lsn: commit_lsn,
                    reason: format!("missing props scratch for durable transaction {txn_id}"),
                })?;
        for delta in props {
            let page = scratch
                .pages
                .iter_mut()
                .find(|page| page.page_id == delta.page_no)
                .ok_or_else(|| ArcGraphError::WalCorruption {
                    lsn: delta.op_lsn,
                    reason: format!(
                        "props delta targets page {} outside transaction {txn_id} scratch",
                        delta.page_no
                    ),
                })?;
            if delta.kind == DeltaOpKind::PageAlloc {
                let mut slotted =
                    SlottedPage::open_prop_trusted(page.image.as_mut()).map_err(|error| {
                        ArcGraphError::WalCorruption {
                            lsn: delta.op_lsn,
                            reason: format!("cannot attach props scratch page: {error}"),
                        }
                    })?;
                slotted
                    .apply_redo_if_newer(delta.op_lsn, |_page| {
                        Ok::<(), std::convert::Infallible>(())
                    })
                    .expect("infallible PageAlloc scratch stamp");
            } else {
                crate::redo::apply_physical_delta(page.image.as_mut(), delta, commit_lsn)?;
            }
        }
        Ok(())
    }

    /// Commit-success hook: (re-)install every scratch page's FINAL
    /// image into the resident tier (idempotent w.r.t. the eager
    /// per-append snapshots — the final snapshot is content-identical
    /// to the last append's) and check still-useful pages back into
    /// the tenant pool. MUST be called only after the commit's WAL
    /// append succeeded — the POOL check-in ordering is what makes
    /// per-page bundle images monotone in LSN (see the module
    /// concurrency note).
    pub fn publish_txn_slotted(&self, txn_id: u64) -> Result<(), BlobError> {
        let Some((_, scratch)) = self.txn_slotted.remove(&txn_id) else {
            return Ok(());
        };
        for sp in scratch.pages {
            let free_space = match SlottedPageRef::open_prop_trusted(&sp.image[..]) {
                Ok(r) => r.header().free_space,
                // A scratch image that fails even the structural open is
                // a codec bug; publish the bytes anyway (they are what
                // the bundle carried) but never re-pool the page.
                Err(_) => 0,
            };
            let resident: Arc<[u8; PAGE_SIZE]> = Arc::from(sp.image.clone());
            self.insert_resident(sp.tenant, sp.page_id, ResidentKind::Slotted(resident));
            if free_space >= MIN_POOL_FREE_BYTES {
                self.pool_checkin(
                    sp.tenant,
                    PooledPropsPage {
                        page_id: sp.page_id,
                        image: sp.image,
                        free_space,
                    },
                );
            }
        }
        // #1404 M0 — same write-path drain hook as `publish`.
        self.maybe_drain()
    }

    /// Commit-failure / explicit-abort hook: discard the txn's scratch
    /// pages and UNDO their eager resident publishes (no reader outside
    /// the owning txn ever saw the aborted bags — their records never
    /// became visible). A page checked out of the tenant pool is
    /// restored to its exact checkout-time state BOTH in the pool and
    /// in the resident tier (the z1 unwind discipline, slotted leg); a
    /// fresh page is removed outright — its id is burned (monotone
    /// allocator, never reused).
    pub fn rollback_txn_slotted(&self, txn_id: u64) {
        let Some((_, scratch)) = self.txn_slotted.remove(&txn_id) else {
            return;
        };
        for sp in scratch.pages {
            match sp.origin {
                ScratchOrigin::Pooled {
                    pre_image,
                    pre_free_space,
                } => {
                    // Restore the resident tier to the pre-checkout
                    // committed image (only this txn's bags disappear;
                    // they were unreferenced).
                    self.insert_resident(
                        sp.tenant,
                        sp.page_id,
                        ResidentKind::Slotted(Arc::from(pre_image.clone())),
                    );
                    self.pool_checkin(
                        sp.tenant,
                        PooledPropsPage {
                            page_id: sp.page_id,
                            image: pre_image,
                            free_space: pre_free_space,
                        },
                    );
                }
                ScratchOrigin::Fresh => {
                    // The page existed ONLY for this txn — remove the
                    // eager-published resident entry + any spill copy
                    // (same bookkeeping as the chain rollback walk).
                    if self.pages.remove(&(sp.tenant, sp.page_id)).is_some() {
                        self.resident_bytes
                            .fetch_sub(PAGE_SIZE as u64, Ordering::AcqRel);
                    }
                    if let Some(spill) = &self.spill {
                        spill.offsets.remove(&(sp.tenant, sp.page_id));
                    }
                }
            }
        }
    }

    /// Return an open page to the tenant pool. Insert-if-vacant;
    /// when occupied, keep whichever page has MORE free space (the
    /// other page simply stops being pooled — its remaining free space
    /// is forgone, never a correctness concern).
    fn pool_checkin(&self, tenant: TenantId, page: PooledPropsPage) {
        let mut pool = self
            .props_open_pool
            .lock()
            .expect("props_open_pool mutex poisoned");
        match pool.entry(tenant) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(page);
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                if page.free_space > o.get().free_space {
                    o.insert(page);
                }
            }
        }
    }

    /// Test/observability: number of txns with live slotted scratch.
    #[doc(hidden)]
    #[must_use]
    pub fn txn_slotted_scratch_count(&self) -> usize {
        self.txn_slotted.len()
    }

    /// Retrieve the blob identified by `blob_ref` under `tenant`.
    ///
    /// v2 M1 dispatch (ADR-230 / design §M1.2): `slot_id == 0` is a
    /// DEC-4 chain head (exactly the pre-M1 invariant — every chained
    /// ref encodes slot 0, `BlobRef::new(first_page, 0)`); `slot_id >=
    /// 1` is a packed bag in a shared [`PageType::PropSlotted`] page at
    /// slot `slot_id - 1` (the 1-based encoding keeps 0 as the chain
    /// discriminant so pre-M1 refs read unchanged in a mixed store).
    ///
    /// #1404 M0 — pages are resolved via `Self::resolve_page`, which
    /// re-faults an evicted page back from the spill tier, so a bounded
    /// store returns byte-identical content whether or not the pages
    /// are currently resident.
    pub fn get(&self, tenant: TenantId, blob_ref: BlobRef) -> Result<Bytes, BlobError> {
        if blob_ref.slot_id == 0 {
            self.get_chain(tenant, blob_ref)
        } else {
            self.get_slotted(tenant, blob_ref)
        }
    }

    /// v2 M2 — ZERO-COPY bag read (design §4.2: "PropBlockView over
    /// the property block bytes [zero-copy, no allocation]").
    ///
    /// For a packed (slotted) bag this clones the resident page's
    /// `Arc` and returns a range view — NO byte copy, unlike
    /// [`Self::get`]'s `Bytes::copy_from_slice`. Sound because a
    /// published slotted image is IMMUTABLE: writers never mutate a
    /// published image in place — they publish a whole replacement
    /// `Arc` (the M1 atomic-by-exclusivity design, see the module
    /// concurrency note) — so a held `Arc` is a stable, consistent
    /// snapshot regardless of concurrent publishes or eviction (the
    /// Arc keeps the image alive even if the tier entry is replaced
    /// or spilled underneath).
    ///
    /// Chained bags fall back to the owned reassembly (a chain is
    /// multiple pages — there is no single buffer to borrow).
    ///
    /// Budget (PD#5): slotted hit = 1 Arc clone + the
    /// `open_prop_trusted` structural checks; zero heap bytes.
    pub fn get_bag(&self, tenant: TenantId, blob_ref: BlobRef) -> Result<BagBytes, BlobError> {
        if blob_ref.slot_id == 0 {
            return Ok(BagBytes::Owned(self.get_chain(tenant, blob_ref)?));
        }
        let kind = self
            .resolve_page(tenant, blob_ref.page_id)?
            .ok_or(BlobError::MissingHead {
                tenant,
                head: blob_ref.page_id,
            })?;
        let ResidentKind::Slotted(image) = kind else {
            return Err(BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: "ref addresses a slot but the page is a DEC-4 chain chunk".to_string(),
            });
        };
        let page_ref =
            SlottedPageRef::open_prop_trusted(&image[..]).map_err(|e| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: e.to_string(),
            })?;
        let bag = page_ref
            .read_bag(SlotId(blob_ref.slot_id - 1))
            .map_err(|e| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: e.to_string(),
            })?
            .ok_or_else(|| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: "slot is tombstoned".to_string(),
            })?;
        // Translate the borrowed bag slice into an (offset, len) range
        // over the SAME image the Arc owns, so the view outlives this
        // call without copying.
        let base = image.as_ptr() as usize;
        let start = bag.as_ptr() as usize - base;
        let len = bag.len();
        Ok(BagBytes::Paged { image, start, len })
    }

    /// DEC-4 chain read — the pre-M1 path, unchanged.
    fn get_chain(&self, tenant: TenantId, blob_ref: BlobRef) -> Result<Bytes, BlobError> {
        let head =
            self.resolve_chain_page(tenant, blob_ref.page_id)?
                .ok_or(BlobError::MissingHead {
                    tenant,
                    head: blob_ref.page_id,
                })?;

        let total_len = head.total_len as usize;
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(&head.bytes);
        let mut next = head.next_page;
        drop(head);

        while next != 0 {
            let remaining = total_len.saturating_sub(out.len());
            let page = self
                .resolve_chain_page(tenant, next)?
                .ok_or(BlobError::BrokenChain {
                    at: next,
                    remaining,
                })?;
            out.extend_from_slice(&page.bytes);
            next = page.next_page;
        }

        if out.len() != total_len {
            return Err(BlobError::LengthMismatch {
                claimed: total_len,
                recovered: out.len(),
            });
        }

        Ok(Bytes::from(out))
    }

    /// v2 M1 packed-bag read: resolve the shared slotted page, read the
    /// bag at `slot_id - 1`, copy it out. The resident page bytes were
    /// checksum-validated at their trust boundary (bundle install /
    /// checkpoint restore / spill re-fault), so the per-read validation
    /// is the cheap structural set (`open_prop_trusted`: header + type
    /// byte + #592 slot-count bound + slot-entry bounds) — the ≤ 40 ns
    /// §4.4 slot-read envelope, no 8 KiB CRC on the hot path.
    fn get_slotted(&self, tenant: TenantId, blob_ref: BlobRef) -> Result<Bytes, BlobError> {
        let kind = self
            .resolve_page(tenant, blob_ref.page_id)?
            .ok_or(BlobError::MissingHead {
                tenant,
                head: blob_ref.page_id,
            })?;
        let ResidentKind::Slotted(image) = kind else {
            // A slot-bearing ref (slot_id >= 1) can only legitimately
            // point at a PropSlotted page; hitting a chain chunk here
            // means the ref or the page is corrupt. Fail LOUD, never
            // silently mis-decode (the M2 "loud not silent" posture).
            return Err(BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: "ref addresses a slot but the page is a DEC-4 chain chunk".to_string(),
            });
        };
        let page_ref =
            SlottedPageRef::open_prop_trusted(&image[..]).map_err(|e| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: e.to_string(),
            })?;
        let bag = page_ref
            .read_bag(SlotId(blob_ref.slot_id - 1))
            .map_err(|e| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: e.to_string(),
            })?
            .ok_or_else(|| BlobError::SlotRead {
                page: blob_ref.page_id,
                slot: blob_ref.slot_id,
                reason: "slot is tombstoned".to_string(),
            })?;
        Ok(Bytes::copy_from_slice(bag))
    }

    /// Resolve `(tenant, page_id)` expecting a DEC-4 chain chunk.
    /// Returns `Ok(None)` on a genuine miss; a kind mismatch (the id
    /// resolves to a slotted page) is loud corruption, never a silent
    /// misread.
    fn resolve_chain_page(
        &self,
        tenant: TenantId,
        page_id: u64,
    ) -> Result<Option<Arc<BlobChunk>>, BlobError> {
        match self.resolve_page(tenant, page_id)? {
            None => Ok(None),
            Some(ResidentKind::Chain(chunk)) => Ok(Some(chunk)),
            Some(ResidentKind::Slotted(_)) => Err(BlobError::SlotRead {
                page: page_id,
                slot: 0,
                reason: "chain walk resolved a PropSlotted page".to_string(),
            }),
        }
    }

    /// Resolve `(tenant, page_id)` to its resident kind, re-faulting from
    /// the spill tier if the page was evicted from RAM (#1404 M0 re-fault
    /// path). A resident hit is the hot path (no I/O). A miss on both tiers
    /// returns `None` (the page is genuinely absent — stale ref, wrong
    /// tenant, or GC'd).
    ///
    /// The ordering hazard (evict-then-read = data corruption if there were
    /// no re-fault) is closed here: a bounded store's `get` NEVER trusts
    /// resident-absence as blob-absence; it consults the durable spill
    /// tier first, and eviction only ever moves a page whose durable image
    /// is already in spill (see [`Self::evict_one`]).
    ///
    /// v2 M1: the spilled image is re-classified via
    /// [`page_image_is_slotted`] and a slotted image is re-validated in
    /// FULL (header + slot-count + body CRC32C — the spill file is a
    /// byte trust boundary) before it re-enters the resident tier.
    fn resolve_page(
        &self,
        tenant: TenantId,
        page_id: u64,
    ) -> Result<Option<ResidentKind>, BlobError> {
        if let Some(e) = self.pages.get(&(tenant, page_id)) {
            return Ok(Some(e.value().kind.clone()));
        }
        // Resident miss — re-fault from spill (bounded store only).
        let Some(spill) = self.spill.as_ref() else {
            return Ok(None);
        };
        let Some(_page) = spill.read_page(tenant, page_id)? else {
            return Ok(None);
        };
        let decode = |page: Box<[u8; PAGE_SIZE]>| -> Result<ResidentKind, BlobError> {
            if page_image_is_slotted(page.as_ref()) {
                // Full validation incl. CRC at the trust boundary; the spill
                // image was written by `encode_image` (identity for slotted),
                // so a failure here is real corruption, not a format skew.
                SlottedPageRef::open(&page[..]).map_err(|error| BlobError::SpillCorrupt {
                    tenant,
                    page_id,
                    path: spill.path.clone(),
                    reason: error.to_string(),
                })?;
                Ok(ResidentKind::Slotted(Arc::from(page)))
            } else {
                let chunk = BlobChunk::decode_page(page.as_ref()).map_err(|error| {
                    BlobError::SpillCorrupt {
                        tenant,
                        page_id,
                        path: spill.path.clone(),
                        reason: error.to_string(),
                    }
                })?;
                Ok(ResidentKind::Chain(Arc::new(chunk)))
            }
        };
        // Warm the resident tier so a hot page does not thrash the spill.
        // The re-faulted page IS checkpoint-durable (it was evicted, which
        // requires it), so it is immediately evict-eligible again.
        let kind = match self.pages.entry((tenant, page_id)) {
            dashmap::mapref::entry::Entry::Occupied(entry) => entry.get().kind.clone(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                // The first spill read can be stale even though this entry is
                // vacant: N+1 may have been published and evicted after that
                // read. Re-read the current spill offset while the vacant
                // shard guard excludes a publish/remove ABA for this key.
                let current = spill.read_page(tenant, page_id)?;
                let Some(current) = current else {
                    // An aborting owner can remove spill.offsets without the
                    // pages guard; this key is legitimately no longer present.
                    return Ok(None);
                };
                let resident = Arc::new(ResidentPage::new(decode(current)?));
                resident.checkpointed.store(true, Ordering::Release);
                let kind = resident.kind.clone();
                entry.insert(resident);
                self.resident_bytes
                    .fetch_add(PAGE_SIZE as u64, Ordering::AcqRel);
                self.evict_queue
                    .lock()
                    .expect("blob evict_queue mutex poisoned")
                    .push_back((tenant, page_id));
                kind
            }
        };
        self.refault_count.fetch_add(1, Ordering::AcqRel);
        Ok(Some(kind))
    }

    // ── #1404 M0 — resident-tier drain (evict checkpoint-durable pages) ──

    /// Engage the drain if the resident set is over the high watermark.
    /// No-op for the unbounded store (`spill = None` → watermark
    /// `u64::MAX`). Called on the write path (`publish`) — a load + branch
    /// on the fast path, the drain loop only under pressure.
    fn maybe_drain(&self) -> Result<(), BlobError> {
        if self.spill.is_none() {
            return Ok(());
        }
        if self.resident_bytes.load(Ordering::Acquire) > self.config.high_watermark_bytes {
            self.drain_to_low_watermark()?;
        }
        Ok(())
    }

    /// Evict checkpoint-durable pages (oldest-first) to the spill tier
    /// until the resident set is at/below the low watermark (hysteresis) or
    /// no more evict-eligible pages remain.
    ///
    /// **Throttle-not-OOM (INV — writer admission, #1405 lever d):** this
    /// runs INLINE on the writer's `publish` call. When the drain cannot
    /// free enough pages (all resident pages are pending checkpoint
    /// durability), it returns with the resident set still bounded by the
    /// arrival rate — but the caller (`publish`) has already paid the drain
    /// cost synchronously, so ingest *slows* (back-pressure) rather than
    /// racing ahead and OOMing. Once a checkpoint marks more pages durable,
    /// the next `publish` drains them. This converts an OOM-kill into a
    /// throughput dip, the bounded-block guarantee of the design §5.
    ///
    /// **Cost — O(evicted-this-pass), not O(queue-length) (#1404 M0):** the
    /// evict queue is **FIFO oldest-first** (pages are `push_back`ed at
    /// publish / re-fault, popped from the front here). A page can only
    /// become checkpoint-durable AFTER it is enqueued, so the *oldest* queued
    /// page is always the *first* to become durable — durability of the queue
    /// is monotone front-to-back. Therefore if the FRONT page is not yet
    /// durable (`NotDurable`), NOTHING behind it is durable either; the pass
    /// restores the popped page to the front and `break`s immediately. Each
    /// drain thus costs O(pages actually evicted this pass); in the realistic
    /// all-NotDurable regime (checkpoints are rare — ~1 GiB-WAL / ~300s — so
    /// the front is almost always un-checkpointed) it is O(1) (pop front →
    /// NotDurable → push_front → break). The next `publish` retries the drain
    /// after a later checkpoint marks the front pages durable.
    ///
    /// This break-on-first-NotDurable replaces the prior `budget =
    /// queue.len()` + `push_back`-on-NotDurable, which walked the ENTIRE FIFO
    /// on EVERY publish and did O(queue-length) work per publish → O(N²)
    /// aggregate through the un-checkpointed window (the #1404 M0 throughput
    /// regression). Only the NotDurable *scheduling* changed — the INV-DURABLE
    /// gate and the evict-after-durable (spill-write-then-remove) ordering in
    /// [`Self::evict_one`] are UNTOUCHED, so this is durability-neutral.
    fn drain_to_low_watermark(&self) -> Result<(), BlobError> {
        let Some(spill) = self.spill.as_ref() else {
            return Ok(());
        };
        // Bounded scan: at most one full pass over the current queue, so a
        // drain call has bounded latency and cannot spin forever when every
        // page is pending durability (throttle, not livelock).
        let mut budget = self
            .evict_queue
            .lock()
            .expect("blob evict_queue mutex poisoned")
            .len();
        while budget > 0
            && self.resident_bytes.load(Ordering::Acquire) > self.config.low_watermark_bytes
        {
            budget -= 1;
            // One probe = one front-pop + evict-attempt. The #1404 M0
            // regression guard asserts this stays O(evicted-this-pass), not
            // O(queue-length), per publish (test/observability only).
            self.drain_probe_count.fetch_add(1, Ordering::Relaxed);
            let Some(key) = self
                .evict_queue
                .lock()
                .expect("blob evict_queue mutex poisoned")
                .pop_front()
            else {
                break;
            };
            let outcome = match self.evict_one(spill, key) {
                Ok(outcome) => outcome,
                Err(error) => {
                    // The durable spill write did not complete, so the page is
                    // still resident. Restore its FIFO position before
                    // propagating the typed error; a later drain may retry.
                    self.evict_queue
                        .lock()
                        .expect("blob evict_queue mutex poisoned")
                        .push_front(key);
                    return Err(error);
                }
            };
            match outcome {
                EvictOutcome::Evicted | EvictOutcome::Gone => {}
                EvictOutcome::NotDurable => {
                    // Not yet checkpoint-durable → restore it to the FRONT
                    // (it stays the oldest, retried first next drain) and
                    // BREAK the pass. FIFO-oldest-first ⟹ if the front page
                    // isn't durable, none of the newer pages behind it are
                    // either (they were enqueued later, so their durability
                    // can only lag) — scanning further would evict nothing and
                    // cost O(queue-length). Breaking keeps each drain
                    // O(evicted-this-pass) = O(1) in the all-NotDurable regime.
                    // INV-DURABLE holds: the page stays resident, un-evicted;
                    // the next publish retries after a later checkpoint marks
                    // it durable.
                    self.evict_queue
                        .lock()
                        .expect("blob evict_queue mutex poisoned")
                        .push_front(key);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Attempt to evict one resident page to spill. Returns the outcome so
    /// the drain loop can re-queue a not-yet-durable page.
    fn evict_one(
        &self,
        spill: &BlobSpill,
        key: (TenantId, u64),
    ) -> Result<EvictOutcome, BlobError> {
        // Snapshot the resident entry WITHOUT removing it — we only remove
        // after the durable spill write succeeds (evict-after-durable).
        let Some(entry) = self.pages.get(&key).map(|e| Arc::clone(e.value())) else {
            return Ok(EvictOutcome::Gone); // already removed (rollback/GC)
        };
        // INV-DURABLE gate: only evict a page a completed checkpoint has
        // captured. `iter_pages_resident_only` sets this under the freeze.
        if !entry.checkpointed.load(Ordering::Acquire) {
            return Ok(EvictOutcome::NotDurable);
        }
        let (tenant, page_id) = key;
        // Write the durable spill image FIRST, then drop the resident copy
        // (evict-then-read is safe: `resolve_page` faults from spill, which
        // is now populated). If it was already spilled (idempotent) the
        // index just refreshes to the latest offset. Kind-agnostic (v2 M1):
        // `encode_image` emits the chain chunk layout or the slotted page
        // verbatim; re-fault re-classifies via `page_image_is_slotted`.
        let sampled_capture_epoch = entry.last_overflow_capture_epoch.load(Ordering::Acquire);
        #[cfg(debug_assertions)]
        if let Some((entered, release)) = spill
            .debug_evict_epoch_gate
            .lock()
            .expect("blob spill eviction epoch gate mutex poisoned")
            .take()
        {
            entered.wait();
            release.wait();
        }
        spill.write_page(
            tenant,
            page_id,
            entry.kind.encode_image().as_ref(),
            sampled_capture_epoch,
            &entry.last_overflow_capture_epoch,
        )?;
        #[cfg(debug_assertions)]
        if let Some((entered, release)) = spill
            .debug_after_evict_epoch_publish_gate
            .lock()
            .expect("blob spill eviction epoch-publish gate mutex poisoned")
            .take()
        {
            entered.wait();
            release.wait();
        }
        // Capture may stamp the still-resident page after our sample but
        // before offset publication. Transfer the newest epoch into the
        // published offset before removing the resident copy so the spill
        // pass cannot emit an image the resident pass already emitted.
        if let Some(offset) = spill.offsets.get(&key) {
            offset.last_overflow_capture_epoch.fetch_max(
                entry.last_overflow_capture_epoch.load(Ordering::Acquire),
                Ordering::AcqRel,
            );
        }
        // Remove from resident AFTER the spill image is durable. A racing
        // `resolve_page` between the spill write and this remove sees the
        // resident copy (still present) → correct; after the remove it
        // faults from spill → correct. No window where the page is unreadable.
        if self
            .pages
            .remove_if(&key, |_, current| Arc::ptr_eq(current, &entry))
            .is_some()
        {
            self.resident_bytes
                .fetch_sub(PAGE_SIZE as u64, Ordering::AcqRel);
            self.evicted_count.fetch_add(1, Ordering::AcqRel);
            Ok(EvictOutcome::Evicted)
        } else {
            // A newer image superseded our snapshot while spill I/O was in
            // flight. It remains resident and must not count as evicted.
            Ok(EvictOutcome::Gone)
        }
    }

    /// Total RESIDENT page count across all tenants (test-only
    /// introspection). NOTE: for a bounded store this is the resident set,
    /// which may be less than the total blob-page count (evicted pages live
    /// in spill). Use [`Self::logical_page_count`] for the full logical
    /// count.
    #[doc(hidden)]
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// #1404 M0.x FIX-B — RESIDENT page count for the streaming checkpoint
    /// count header (`for_each_resident_page` streams exactly this many under
    /// the freeze). Matches the resident set the streaming capture walks (NOT
    /// the logical count — evicted pages are captured post-guard from spill).
    #[must_use]
    pub fn resident_page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total LOGICAL blob-page count = resident + evicted-to-spill. For an
    /// unbounded store this equals [`Self::page_count`]. Test-only.
    #[doc(hidden)]
    #[must_use]
    pub fn logical_page_count(&self) -> usize {
        match &self.spill {
            None => self.pages.len(),
            Some(spill) => {
                // Union of resident ids + spilled ids (a resident page may
                // also have a stale spill entry after re-fault warming).
                let resident_only = self
                    .pages
                    .iter()
                    .filter(|e| !spill.offsets.contains_key(e.key()))
                    .count();
                resident_only + spill.offsets.len()
            }
        }
    }

    fn alloc_page_range(&self, count: usize) -> u64 {
        let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
        let prev = self.next_page.fetch_add(count_u64, Ordering::AcqRel);
        prev.saturating_add(1)
    }

    /// ADR-033 Z-1 (b): register a blob chain head + page range as
    /// uncommitted under a transaction.
    ///
    /// Typically called immediately after [`Self::publish`] inside a
    /// builder-phase blob write, so that a subsequent WAL fsync
    /// failure can unwind the newly installed chain via
    /// [`Self::remove_uncommitted_chain`].
    ///
    /// Only the head page's `(tenant, page_id)` is recorded — walking
    /// the chain on rollback re-derives the page range via the
    /// `BlobChunk::next_page` linked-list pointers. The `_page_count`
    /// parameter is reserved for a future fast-path that stores the
    /// full range explicitly; see ADR-033 §4.
    pub fn register_uncommitted_chain(
        &self,
        log: &mut TxnMutationLog,
        tenant: TenantId,
        head_page: u64,
        _page_count: u64,
    ) {
        log.blob_heads.push((tenant, head_page));
    }

    /// ADR-033 Z-1 (b) rollback primitive: walk the chain starting
    /// at `head_page` and remove every page from the DashMap.
    ///
    /// Idempotent: removing a page that was never mapped (or already
    /// removed) stops the walk — chains are guaranteed contiguous at
    /// allocation time, so a broken chain at rollback indicates a
    /// prior partial rollback, which is the no-op we want.
    ///
    /// See ADR-033 §4 for the walk-to-remove rationale (simplicity
    /// over precomputed range storage).
    pub fn remove_uncommitted_chain(
        &self,
        tenant: TenantId,
        head_page: u64,
    ) -> Result<(), BlobError> {
        let mut cur = head_page;
        while cur != 0 {
            // The chain pointer may need re-faulting from spill if the page
            // was evicted before rollback (bounded tier). `resolve_page`
            // gives us `next_page` even for an evicted page; then we remove
            // both the resident and spill images. A slotted page terminates
            // the walk defensively (chains never link into slotted pages;
            // hitting one means the head was not a chain — stop, remove
            // nothing further).
            let next = match self.resolve_page(tenant, cur)? {
                Some(ResidentKind::Chain(c)) => Some(c.next_page),
                Some(ResidentKind::Slotted(_)) => break,
                None => None,
            };
            let removed_resident = self.pages.remove(&(tenant, cur)).is_some();
            if removed_resident {
                self.resident_bytes
                    .fetch_sub(PAGE_SIZE as u64, Ordering::AcqRel);
            }
            if let Some(spill) = &self.spill {
                spill.offsets.remove(&(tenant, cur));
            }
            self.overflow_pages.fetch_sub(1, Ordering::AcqRel);
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        Ok(())
    }

    /// Test/observability: does the store LOGICALLY map `(tenant,
    /// page_id)` — resident OR spilled? A bounded store may hold the page
    /// only in spill; `contains` reflects the logical presence a reader
    /// would see via [`Self::get`].
    #[doc(hidden)]
    #[must_use]
    pub fn contains(&self, tenant: TenantId, page_id: u64) -> bool {
        if self.pages.contains_key(&(tenant, page_id)) {
            return true;
        }
        match &self.spill {
            None => false,
            Some(spill) => spill.offsets.contains_key(&(tenant, page_id)),
        }
    }

    /// Test-only: does `(tenant, page_id)` currently occupy a RESIDENT RAM
    /// slot (as opposed to living only in spill)?
    #[doc(hidden)]
    #[must_use]
    pub fn is_resident(&self, tenant: TenantId, page_id: u64) -> bool {
        self.pages.contains_key(&(tenant, page_id))
    }
}

// ─────────────────────────────────────────────────────────────────────
// BlobStoreHandle — N-2 (issue #81) replay trait
// ─────────────────────────────────────────────────────────────────────

/// Routing target for a `BundlePageKind::Blob` staged page during
/// WAL replay (N-2 / issue #81).
///
/// Mirrors [`crate::wal::PrimaryPageStoreHandle`] /
/// [`crate::wal::RecordPageStoreHandle`] but carries the
/// `TenantId` on the method because blob chain pages are
/// physically keyed by `(tenant, page_id)` (see module docs §
/// Tenancy + [`BlobStore`]'s internal DashMap). Primary and
/// record page stores are single-tenant at the physical layer
/// (logical tenancy lives in the B-tree keys / slotted record
/// headers); blobs are multi-tenant at the physical layer.
///
/// The only implementor at v1.0 is [`BlobStore`]; the trait
/// exists to break the `wal/replay.rs` → `blob.rs` dependency
/// cycle the same way `PrimaryPageStoreHandle` does for the
/// primary index store.
///
/// # Idempotence contract (Lemma I2 — bundle-level)
///
/// `install_or_replace` is an **unconditional byte-copy**
/// overwrite — no byte-equality check against an existing entry.
/// This matches PR #79 X-1's fix for the primary and record
/// handles: idempotence is bundle-level (enforced upstream by
/// the executor's `applied_high_water ≥ bundle.commit_lsn`
/// skip + `apply_replay_mvcc_write` Lemma I1 check), so a later
/// bundle's entry for the same `(tenant, page_id)` is a
/// legitimate supersession.
pub trait BlobStoreHandle: Send + Sync {
    /// Install a blob chain page under `(tenant, page_id)`,
    /// overwriting any existing entry.
    ///
    /// `page` is a full `PAGE_SIZE`-byte buffer in the on-disk
    /// blob-page layout (see module docs § Page layout). The
    /// handle parses the header to reconstruct the in-memory
    /// `BlobChunk`; a malformed header (e.g. `chunk_len >
    /// BLOB_CHUNK_BYTES`) returns [`ArcGraphError::WalCorruption`].
    fn install_or_replace(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<[u8; PAGE_SIZE]>,
    ) -> arcgraph_core::Result<()>;
}

impl BlobStoreHandle for BlobStore {
    fn install_or_replace(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<[u8; PAGE_SIZE]>,
    ) -> arcgraph_core::Result<()> {
        // v2 M1 — kind-aware install. WAL replay tags the kind
        // (`BundlePageKind::Blob` vs `::PropSlotted`) but the checkpoint
        // snapshot's blob section carries raw `(tenant, page_id, image)`
        // entries with no kind byte, and both paths funnel here — so the
        // installer classifies the image itself (`page_image_is_slotted`,
        // see its soundness note) and then FULLY validates a slotted
        // image (header + #592 slot-count bound + body CRC32C) at this
        // byte trust boundary. A classified-slotted image that fails
        // validation is loud WalCorruption, never a silent chain decode.
        let kind = if page_image_is_slotted(page.as_ref()) {
            SlottedPageRef::open(&page[..]).map_err(|e| ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!(
                    "BlobStoreHandle::install_or_replace: PropSlotted page {} failed \
                     validation: {e}",
                    page_id.raw()
                ),
            })?;
            ResidentKind::Slotted(Arc::from(page))
        } else {
            ResidentKind::Chain(Arc::new(BlobChunk::decode_page(page.as_ref())?))
        };
        // #1404 M0 — install into the resident tier via the shared helper
        // so the resident-byte counter + eviction FIFO stay consistent on
        // the replay/restore path (identical semantics to the pre-#1404
        // `pages.insert` for the unbounded store: fresh id → resident, add
        // bytes; supersession → overwrite, no double-count).
        self.insert_resident(tenant, page_id.raw(), kind);
        // P0 #820 — durable blob page-id high-water recovery.
        //
        // `next_page` is a per-PROCESS allocator: a fresh `BlobStore`
        // starts it at 0 (first `alloc_page_range` returns id 1). Unlike
        // `NodeId` / `RelId` / `PageId`, the blob page-id high-water is
        // NOT carried in the v4 `CommitBundle` `allocator_advances`
        // section — so without seeding it here, a `--data` restart resets
        // the blob allocator to 0 while the recovered chains already
        // occupy ids 1..=N. The next process's property-blob writes then
        // REUSE recovered page-ids, and the FOLLOWING restart's replay
        // installs the colliding pages in commit_lsn order, overwriting
        // earlier nodes' property blobs with later ones — an acknowledged
        // (fsync'd) commit silently loses its property data on the 2nd
        // durable restart (CARDINAL durability bug #820: "count grows,
        // DISTINCT stuck at the last batch, earlier seqs gone").
        //
        // The page-id space is GLOBAL across tenants (`next_page` is one
        // counter; the DashMap key carries the tenant only for physical
        // partitioning), so the high-water is `max` over every installed
        // page-id regardless of tenant. `alloc_page_range` returns
        // `prev_high_water + 1`, so seeding `next_page` to the installed
        // id guarantees the next live allocation lands strictly above it.
        // Replay calls this for EVERY page of EVERY recovered chain (N-2 /
        // issue #81 staging), so the post-replay `next_page` reflects the
        // true physical end of the durable blob space. `fetch_max` is
        // monotonic + idempotent (Lemma I3): order-independent across
        // bundles and a no-op under double-replay.
        self.next_page.fetch_max(page_id.raw(), Ordering::AcqRel);
        Ok(())
    }
}

impl crate::redo::DeltaPageStore for BlobStore {
    fn read_page_for_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> arcgraph_core::Result<Option<Box<[u8; PAGE_SIZE]>>> {
        Ok(self
            .resolve_page(tenant, page_id.raw())?
            .map(|page| page.encode_image()))
    }

    fn install_page_from_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<[u8; PAGE_SIZE]>,
    ) -> arcgraph_core::Result<()> {
        BlobStoreHandle::install_or_replace(self, tenant, page_id, page)
    }
}

// ─────────────────────────────────────────────────────────────────────
// WAL integration
// ─────────────────────────────────────────────────────────────────────

/// Encode a `WalRecordType::PutBlob` payload.
///
/// Layout (little-endian):
/// ```text
/// 0..8   head page id (u64)
/// 8..12  total_len    (u32)
/// 12..   blob bytes   (total_len bytes)
/// ```
#[must_use]
pub fn encode_put_blob_payload(head_page: u64, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + bytes.len());
    out.extend_from_slice(&head_page.to_le_bytes());
    let len_u32 = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len_u32.to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// Decode a `WalRecordType::PutBlob` payload.
///
/// Returns `(head_page, bytes)`. Rejects payloads whose declared
/// length does not match the trailing tail length, or whose total
/// length exceeds [`BLOB_MAX_BYTES`].
pub fn decode_put_blob_payload(payload: &[u8]) -> arcgraph_core::Result<(u64, Vec<u8>)> {
    if payload.len() < 12 {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!(
                "PutBlob payload length {} < 12 bytes for header",
                payload.len()
            ),
        });
    }
    let mut head_bytes = [0u8; 8];
    head_bytes.copy_from_slice(&payload[..8]);
    let head_page = u64::from_le_bytes(head_bytes);

    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&payload[8..12]);
    let declared_len = u32::from_le_bytes(len_bytes) as usize;

    let tail = &payload[12..];
    if tail.len() != declared_len {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!(
                "PutBlob payload declares {declared_len} bytes but tail is {}",
                tail.len()
            ),
        });
    }
    if declared_len > BLOB_MAX_BYTES {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("PutBlob payload {declared_len} > cap {BLOB_MAX_BYTES}"),
        });
    }
    Ok((head_page, tail.to_vec()))
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn blob_put_then_get_roundtrips_small() {
        let store = BlobStore::new();
        let payload = b"hello world" as &[u8];
        let blob_ref = store.put(TenantId::DEFAULT, payload).unwrap();
        let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
        assert_eq!(out.as_ref(), payload);
    }

    #[test]
    fn blob_put_then_get_roundtrips_medium() {
        let store = BlobStore::new();
        let payload = vec![0x5Au8; 16_000];
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
        assert_eq!(out.as_ref(), &payload[..]);
    }

    #[test]
    fn blob_put_then_get_roundtrips_large() {
        let store = BlobStore::new();
        let payload: Vec<u8> = (0..BLOB_MAX_BYTES).map(|i| (i % 251) as u8).collect();
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
        assert_eq!(out.len(), BLOB_MAX_BYTES);
        assert_eq!(out.as_ref(), &payload[..]);
    }

    #[test]
    fn blob_empty_is_rejected() {
        let store = BlobStore::new();
        let err = store.put(TenantId::DEFAULT, b"").unwrap_err();
        assert!(matches!(err, BlobError::Empty));
    }

    #[test]
    fn blob_too_large_is_rejected() {
        let store = BlobStore::new();
        let payload = vec![0u8; BLOB_MAX_BYTES + 1];
        let err = store.put(TenantId::DEFAULT, &payload).unwrap_err();
        assert!(matches!(err, BlobError::TooLarge(_)));
    }

    #[test]
    fn blob_ref_encodes_in_property_ref_with_discriminator_bit_1() {
        use crate::property::OVERFLOW_BIT;
        let store = BlobStore::new();
        let blob_ref = store.put(TenantId::DEFAULT, b"x").unwrap();
        let raw = blob_ref.encode();
        assert_ne!(raw & OVERFLOW_BIT, 0);
    }

    #[test]
    fn blob_tenant_isolated() {
        let store = BlobStore::new();
        let payload = b"same bytes in two tenants" as &[u8];
        let t1 = TenantId::DEFAULT;
        let t2 = TenantId::new(42);
        let r1 = store.put(t1, payload).unwrap();
        let r2 = store.put(t2, payload).unwrap();
        // Different tenants use disjoint page-id allocations so the
        // refs differ.
        assert_ne!(r1.page_id, r2.page_id);
        // Cross-tenant reads must fail even though the page-id happens
        // to exist under another tenant.
        let err = store.get(t2, r1).unwrap_err();
        assert!(
            matches!(err, BlobError::MissingHead { .. }),
            "cross-tenant read must fail, got {err:?}"
        );
        // Same-tenant reads succeed.
        assert_eq!(store.get(t1, r1).unwrap().as_ref(), payload);
        assert_eq!(store.get(t2, r2).unwrap().as_ref(), payload);
    }

    #[test]
    fn blob_chain_walks_to_end_on_1mb_value() {
        let store = BlobStore::new();
        let payload = vec![0xA5u8; BLOB_MAX_BYTES];
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        let expected_chunks = BLOB_MAX_BYTES.div_ceil(BLOB_CHUNK_BYTES);
        assert_eq!(store.page_count(), expected_chunks);
        let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
        assert_eq!(out.len(), BLOB_MAX_BYTES);
        assert!(out.iter().all(|&b| b == 0xA5));
    }

    #[test]
    fn blob_missing_head_errors() {
        let store = BlobStore::new();
        let err = store
            .get(TenantId::DEFAULT, BlobRef::new(9999, 0))
            .unwrap_err();
        assert!(matches!(err, BlobError::MissingHead { head: 9999, .. }));
    }

    #[test]
    fn put_blob_payload_roundtrip() {
        let bytes = b"wal payload bytes";
        let enc = encode_put_blob_payload(42, bytes);
        let (head, out) = decode_put_blob_payload(&enc).unwrap();
        assert_eq!(head, 42);
        assert_eq!(out, bytes);
    }

    #[test]
    fn put_blob_payload_rejects_short_header() {
        let enc = vec![0u8; 8];
        assert!(decode_put_blob_payload(&enc).is_err());
    }

    #[test]
    fn put_blob_payload_rejects_length_mismatch() {
        let mut enc = encode_put_blob_payload(1, b"abc");
        enc.pop(); // drop a tail byte; declared-len no longer matches
        assert!(decode_put_blob_payload(&enc).is_err());
    }

    #[test]
    fn put_logged_wal_appears_before_dashmap_entry() {
        // Review block C-2 / ADR-022: a WAL append failure on
        // `put_logged` must leave the in-memory DashMap untouched.
        // We drive the failure by shutting the WAL writer down
        // before the append; the channel is closed and `append`
        // returns `WalUnavailable`.
        use crate::wal::writer::{WalConfig, WalWriter};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(WalConfig {
            dir: dir.path().to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: std::time::Duration::from_millis(2),
            group_commit_max_batch: 4,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        })
        .unwrap();
        let handle = writer.handle();
        writer.shutdown().unwrap();

        let store = BlobStore::new();
        let err = store
            .put_logged(&handle, TenantId::DEFAULT, b"wont-land")
            .unwrap_err();
        assert!(
            matches!(err, BlobError::WalDecode(_)),
            "expected WAL failure, got {err:?}"
        );
        assert_eq!(
            store.page_count(),
            0,
            "DashMap must not be populated if the WAL append failed"
        );
    }

    #[test]
    fn put_logged_record_is_fsynced_before_return() {
        // The real-WAL parallel: after `put_logged` returns, the
        // PutBlob record is already on disk. We observe this by
        // opening a fresh `WalRecoveryReader` while the writer is
        // still alive (so the group-commit batch has flushed) and
        // asserting the record is visible.
        use crate::wal::recovery::WalRecoveryReader;
        use crate::wal::writer::{WalConfig, WalWriter};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(WalConfig {
            dir: dir.path().to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: std::time::Duration::from_millis(1),
            group_commit_max_batch: 1,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        })
        .unwrap();
        let handle = writer.handle();

        let store = BlobStore::new();
        let payload = b"durable blob payload";
        let blob_ref = store
            .put_logged(&handle, TenantId::DEFAULT, payload)
            .unwrap();

        // Shutdown to guarantee the segment header is closed out.
        writer.shutdown().unwrap();

        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut saw_put_blob = false;
        for r in reader {
            let rec = r.unwrap();
            if rec.record_type == WalRecordType::PutBlob {
                let (head, bytes) = decode_put_blob_payload(&rec.payload).unwrap();
                assert_eq!(head, blob_ref.page_id);
                assert_eq!(bytes.as_slice(), payload);
                saw_put_blob = true;
            }
        }
        assert!(
            saw_put_blob,
            "PutBlob record must be on disk after put_logged return"
        );
        // And the in-memory store carries the chain.
        assert_eq!(
            store.get(TenantId::DEFAULT, blob_ref).unwrap().as_ref(),
            payload
        );
    }

    // ─── ADR-033 Z-1 (b): BlobStore rollback helpers ───

    #[test]
    fn register_uncommitted_chain_records_blob_head() {
        let store = BlobStore::new();
        let mut log = TxnMutationLog::new();
        let blob_ref = store.put(TenantId::DEFAULT, b"payload").unwrap();
        store.register_uncommitted_chain(&mut log, TenantId::DEFAULT, blob_ref.page_id, 1);
        assert_eq!(log.blob_heads.len(), 1);
        assert_eq!(log.blob_heads[0], (TenantId::DEFAULT, blob_ref.page_id));
    }

    #[test]
    fn remove_uncommitted_chain_single_page_blob() {
        let store = BlobStore::new();
        let blob_ref = store.put(TenantId::DEFAULT, b"one-page").unwrap();
        assert!(store.contains(TenantId::DEFAULT, blob_ref.page_id));
        store
            .remove_uncommitted_chain(TenantId::DEFAULT, blob_ref.page_id)
            .unwrap();
        assert!(!store.contains(TenantId::DEFAULT, blob_ref.page_id));
    }

    #[test]
    fn remove_uncommitted_chain_multi_page_blob() {
        let store = BlobStore::new();
        // Force multi-chunk by exceeding BLOB_CHUNK_BYTES.
        let payload = vec![0x5Au8; BLOB_CHUNK_BYTES + 100];
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        let expected_chunks = payload.len().div_ceil(BLOB_CHUNK_BYTES);
        assert_eq!(store.page_count(), expected_chunks);

        store
            .remove_uncommitted_chain(TenantId::DEFAULT, blob_ref.page_id)
            .unwrap();
        // Every page in the chain must be gone.
        assert_eq!(store.page_count(), 0);
    }

    #[test]
    fn remove_uncommitted_chain_is_idempotent() {
        let store = BlobStore::new();
        let blob_ref = store.put(TenantId::DEFAULT, b"hello").unwrap();
        store
            .remove_uncommitted_chain(TenantId::DEFAULT, blob_ref.page_id)
            .unwrap();
        // Second call on the now-absent chain must not panic.
        store
            .remove_uncommitted_chain(TenantId::DEFAULT, blob_ref.page_id)
            .unwrap();
    }

    #[test]
    fn remove_uncommitted_chain_honors_tenant() {
        let store = BlobStore::new();
        let t1 = TenantId::DEFAULT;
        let t2 = TenantId::new(42);
        let r1 = store.put(t1, b"same bytes").unwrap();
        let _r2 = store.put(t2, b"same bytes").unwrap();
        // Remove t1's chain only.
        store.remove_uncommitted_chain(t1, r1.page_id).unwrap();
        assert!(!store.contains(t1, r1.page_id));
        // t2's chain remains.
        assert!(store.page_count() > 0);
    }

    // ─── N-2 (issue #81): BlobStoreHandle + page encode/decode ───

    #[test]
    fn blob_chunk_encode_page_round_trips_through_decode() {
        let payload = Bytes::copy_from_slice(b"chunk-payload-bytes");
        let chunk = BlobChunk {
            next_page: 42,
            total_len: 1024,
            bytes: payload.clone(),
        };
        let page = chunk.encode_page();
        let decoded = BlobChunk::decode_page(page.as_ref()).unwrap();
        assert_eq!(decoded.next_page, 42);
        assert_eq!(decoded.total_len, 1024);
        assert_eq!(decoded.bytes, payload);
    }

    #[test]
    fn blob_chunk_decode_page_rejects_oversize_chunk_len() {
        // Forge a page whose chunk_len claims more than BLOB_CHUNK_BYTES.
        let mut page: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page[0..8].copy_from_slice(&0u64.to_le_bytes());
        page[8..12].copy_from_slice(&100u32.to_le_bytes());
        let bogus_chunk_len = (BLOB_CHUNK_BYTES + 1) as u32;
        page[12..16].copy_from_slice(&bogus_chunk_len.to_le_bytes());
        let err = BlobChunk::decode_page(page.as_ref()).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("chunk_len"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn blob_store_handle_install_or_replace_roundtrips_via_get() {
        // Round-trip: put → page_bytes() → drop store → fresh
        // store.install_or_replace(...) → get returns original
        // payload. This is the core N-2 invariant: replay can
        // reconstruct the BlobStore from bundle page bytes alone.
        let payload = vec![0xA7u8; 16_000]; // ~2 chunks
        let t = TenantId::DEFAULT;
        let staged = {
            let src = BlobStore::new();
            src.stage(t, &payload).unwrap()
        };
        let blob_ref = staged.blob_ref();
        let pages = staged.page_bytes();
        assert!(pages.len() >= 2, "payload must span at least 2 chunks");

        let fresh = BlobStore::new();
        for (page_id, page) in pages {
            <BlobStore as BlobStoreHandle>::install_or_replace(&fresh, t, page_id, page).unwrap();
        }
        let out = fresh.get(t, blob_ref).unwrap();
        assert_eq!(out.as_ref(), payload.as_slice());
    }

    #[test]
    fn blob_store_handle_install_or_replace_overwrites_without_byte_equality() {
        // PR #79 X-1 fix (Lemma I2 bundle-level): install_or_replace
        // is an unconditional overwrite — no byte-equality check.
        // A later bundle legitimately supersedes; same discipline as
        // PrimaryPageStoreHandle + RecordPageStoreHandle.
        let t = TenantId::DEFAULT;
        let staged_a = BlobStore::new().stage(t, b"first").unwrap();
        let staged_b = BlobStore::new()
            .stage(t, b"second-different-content")
            .unwrap();
        let (pid_a, page_a) = staged_a.page_bytes().into_iter().next().unwrap();
        let (_pid_b, page_b) = staged_b.page_bytes().into_iter().next().unwrap();

        let store = BlobStore::new();
        // First install under page_a.
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t, pid_a, page_a).unwrap();
        // Overwrite with page_b's bytes (different payload); must succeed.
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t, pid_a, page_b).unwrap();
    }

    #[test]
    fn blob_store_handle_install_or_replace_tenant_isolated() {
        let t1 = TenantId::DEFAULT;
        let t2 = TenantId::new(99);
        let staged = BlobStore::new().stage(t1, b"payload").unwrap();
        let (pid, page) = staged.page_bytes().into_iter().next().unwrap();

        let store = BlobStore::new();
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t1, pid, page).unwrap();
        // Same page_id under t2 must be absent.
        assert!(store.contains(t1, pid.raw()));
        assert!(!store.contains(t2, pid.raw()));
    }

    #[test]
    fn install_or_replace_seeds_next_page_so_restart_does_not_reuse_ids_820() {
        // P0 #820 — CARDINAL: replay-installing a recovered blob page MUST
        // advance the page-id high-water so a post-restart allocation never
        // reuses a recovered id. Models the durable restart sequence at the
        // BlobStore boundary: epoch-0 publishes a property blob → "restart"
        // (fresh store) replays it via install_or_replace → epoch-1 writes a
        // NEW property blob. Pre-fix, the fresh store's `next_page` stayed at
        // 0 and epoch-1 reused epoch-0's page-id; the next replay then
        // overwrote epoch-0's bytes (acked-data loss on the 2nd restart).
        let t = TenantId::DEFAULT;

        // Epoch 0: stage node-0's property blob (single chunk → head id 1).
        let staged0 = BlobStore::new().stage(t, b"seq=0").unwrap();
        let ref0 = staged0.blob_ref();
        let snaps0 = staged0.page_bytes();
        let head0 = snaps0[0].0.raw();

        // "Restart": a FRESH store replays epoch-0's blob page (the recovery
        // path — BlobStoreHandle::install_or_replace per recovered page).
        let recovered = BlobStore::new();
        for (pid, page) in snaps0 {
            <BlobStore as BlobStoreHandle>::install_or_replace(&recovered, t, pid, page).unwrap();
        }

        // Epoch 1 (post-restart): node-1's property blob MUST land at a fresh
        // page-id, NOT reuse epoch-0's recovered id.
        let staged1 = recovered.stage(t, b"seq=1").unwrap();
        let head1 = staged1.page_bytes()[0].0.raw();
        assert!(
            head1 > head0,
            "#820: post-restart blob alloc reused a recovered page-id \
             (head0={head0}, head1={head1}); install_or_replace did not seed next_page",
        );

        // End-to-end overwrite oracle: publishing epoch-1 must NOT clobber
        // epoch-0's recovered bytes (the acked property data must survive).
        recovered.publish(t, staged1).unwrap();
        let recovered0 = recovered.get(t, ref0).unwrap();
        assert_eq!(
            recovered0.as_ref(),
            b"seq=0",
            "#820: epoch-0 acked property blob was overwritten by epoch-1 (page-id reuse)",
        );
    }

    #[test]
    fn install_or_replace_next_page_seed_is_monotonic_and_idempotent_820() {
        // The #820 seed is `fetch_max`: order-independent across bundles and
        // a no-op under double-replay (Lemma I3). Installing a high id then a
        // low id must leave the high-water at the MAX, and re-installing is a
        // no-op on the allocator.
        let t = TenantId::DEFAULT;
        // Build two single-page blobs with distinct ids 1 and 2.
        let s_lo = BlobStore::new().stage(t, b"lo").unwrap(); // id 1
        let s_hi = {
            let src = BlobStore::new();
            let _ = src.stage(t, b"x").unwrap(); // burn id 1
            src.stage(t, b"hi").unwrap() // id 2
        };
        let (pid_lo, page_lo) = s_lo.page_bytes().into_iter().next().unwrap();
        let (pid_hi, page_hi) = s_hi.page_bytes().into_iter().next().unwrap();
        assert_eq!(pid_lo.raw(), 1);
        assert_eq!(pid_hi.raw(), 2);

        let store = BlobStore::new();
        // Install the HIGH id first, then the LOW id (out-of-order replay).
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t, pid_hi, page_hi.clone())
            .unwrap();
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t, pid_lo, page_lo).unwrap();
        // Double-replay the high id (idempotent under monotonic-max).
        <BlobStore as BlobStoreHandle>::install_or_replace(&store, t, pid_hi, page_hi).unwrap();

        // The next allocation must clear the MAX installed id (2) → id 3.
        let next = store.stage(t, b"next").unwrap();
        assert_eq!(
            next.page_bytes()[0].0.raw(),
            3,
            "#820: next_page seed must be max-over-installed (monotonic), regardless of order",
        );
    }

    #[test]
    fn staged_blob_page_bytes_has_one_entry_per_chunk() {
        let store = BlobStore::new();
        let payload = vec![0x5Au8; BLOB_CHUNK_BYTES + 100]; // 2 chunks
        let staged = store.stage(TenantId::DEFAULT, &payload).unwrap();
        let pages = staged.page_bytes();
        assert_eq!(pages.len(), 2);
        // Head page: total_len = payload.len(); tail page: total_len = 0.
        let head_decoded = BlobChunk::decode_page(pages[0].1.as_ref()).unwrap();
        assert_eq!(head_decoded.total_len as usize, payload.len());
        let tail_decoded = BlobChunk::decode_page(pages[1].1.as_ref()).unwrap();
        assert_eq!(tail_decoded.total_len, 0);
        assert_eq!(tail_decoded.next_page, 0);
    }

    // ─── #1404 M0 — bounded resident blob-page tier ───
    //
    // These exercise the four gates the fix must prove:
    //   1. RSS-bounded / drain-fires (resident bytes stay bounded).
    //   2. Re-fault correctness (evict → read = byte-identical).
    //   3. INV-DURABLE (a not-yet-checkpoint-durable page is NOT evicted).
    //   4. Throttle-not-OOM (over-drive keeps resident set bounded).
    // The crash-recovery-byte-equality INV-DURABLE oracle lives in
    // `tests/blob_bound_1404.rs` (needs the WAL + checkpoint harness).

    use tempfile::tempdir;

    /// Build a bounded store with a tiny cap so a handful of pages crosses
    /// the watermark. `high = cap`, `low = cap/2`.
    fn bounded_store(dir: &std::path::Path, cap_pages: u64) -> BlobStore {
        let spill = Arc::new(BlobSpill::open(dir).unwrap());
        let cfg = BlobBoundConfig {
            high_watermark_bytes: cap_pages * PAGE_SIZE as u64,
            low_watermark_bytes: (cap_pages / 2).max(1) * PAGE_SIZE as u64,
        };
        BlobStore::with_bound(spill, cfg)
    }

    /// Mark every currently-resident page checkpoint-durable (evict-eligible)
    /// — simulates a completed ADR-229 checkpoint capturing the resident set
    /// (which is exactly what `iter_pages_resident_only` does under the
    /// freeze). Returns the resident-page snapshot for symmetry.
    fn simulate_checkpoint(store: &BlobStore) -> BlobResidentPages {
        store.iter_pages_resident_only()
    }

    #[test]
    fn unbounded_store_never_evicts_legacy_behavior() {
        // The default (no spill) store must behave EXACTLY as pre-#1404:
        // nothing evicts, no re-fault, resident == logical.
        let store = BlobStore::new();
        assert!(!store.is_bounded());
        for i in 0..50 {
            store
                .put(TenantId::DEFAULT, format!("payload-{i}").as_bytes())
                .unwrap();
        }
        assert_eq!(store.evicted_count(), 0);
        assert_eq!(store.refault_count(), 0);
        assert_eq!(store.page_count(), 50);
        assert_eq!(store.logical_page_count(), 50);
        // iter_pages_resident_only reports NO evicted pages for the
        // unbounded store (the checkpoint-complete-under-freeze invariant).
        let (_resident, evicted) = store.iter_pages_resident_only();
        assert!(evicted.is_empty());
    }

    #[test]
    fn rss_bounded_drain_fires_under_sustained_put() {
        // HEADLINE #1404 TEST: sustained put under a tiny cap keeps the
        // RESIDENT byte set bounded (the drain fires), while the LOGICAL
        // page count grows — i.e. RSS is a function of the cap, not of the
        // number of blobs. RED-on-revert: an unbounded store (below) grows
        // resident bytes without bound.
        let dir = tempdir().unwrap();
        let cap_pages = 8;
        let store = bounded_store(dir.path(), cap_pages);
        let cap_bytes = cap_pages * PAGE_SIZE as u64;

        // Ingest far more single-page blobs than the cap; checkpoint
        // periodically so pages become evict-eligible (durable).
        for round in 0..20 {
            for i in 0..10 {
                store
                    .put(TenantId::DEFAULT, format!("r{round}-i{i}").as_bytes())
                    .unwrap();
            }
            // A checkpoint marks the resident set durable → drain can evict.
            simulate_checkpoint(&store);
            // Trigger the drain again post-checkpoint (a put would, but the
            // batch may have ended below the watermark).
            store.put(TenantId::DEFAULT, b"tick").unwrap();
        }

        // The resident byte set must stay bounded (at/near the cap), even
        // though we ingested ~200 blobs (~200 logical pages).
        assert!(
            store.resident_bytes() <= cap_bytes,
            "resident bytes {} exceeded cap {cap_bytes} — drain did not bound RSS",
            store.resident_bytes(),
        );
        assert!(
            store.evicted_count() > 0,
            "drain never fired — no pages were evicted",
        );
        // Logical page count reflects everything ingested (nothing lost).
        assert!(
            store.logical_page_count() >= 200,
            "logical page count {} lost pages",
            store.logical_page_count(),
        );
    }

    #[test]
    fn rss_bounded_red_on_revert_unbounded_grows() {
        // The RED-on-revert half of the headline: the SAME ingest against
        // an unbounded store grows resident bytes without bound (reproduces
        // the #1404 leak). This is what the bounded store fixes.
        let store = BlobStore::new(); // unbounded
        for round in 0..20 {
            for i in 0..10 {
                store
                    .put(TenantId::DEFAULT, format!("r{round}-i{i}").as_bytes())
                    .unwrap();
            }
        }
        // Unbounded: resident grows to ~200 pages (>> a small cap).
        assert!(
            store.resident_bytes() >= 200 * PAGE_SIZE as u64,
            "unbounded store unexpectedly bounded at {} bytes",
            store.resident_bytes(),
        );
        assert_eq!(store.evicted_count(), 0, "unbounded store must never evict");
    }

    #[test]
    fn refault_correctness_evict_then_read_byte_identical() {
        // RE-FAULT: put a multi-chunk blob → force its pages evicted →
        // read it back → byte-identical. Proves the durable spill image
        // re-faults the exact page. Uses a cap of 1 page so a >1-page blob
        // is guaranteed to have evicted chunks.
        let dir = tempdir().unwrap();
        let store = bounded_store(dir.path(), 1);

        let payload = vec![0xC3u8; BLOB_CHUNK_BYTES * 3 + 17]; // 4 chunks
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        // Make the pages durable, then drive the drain hard.
        simulate_checkpoint(&store);
        store.drain_to_low_watermark().unwrap();
        // At least some chunks must have been evicted (cap=1 page).
        assert!(
            store.evicted_count() > 0,
            "no eviction happened — cannot test re-fault",
        );

        // Read back — must be byte-identical (re-fault reconstructs).
        let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
        assert_eq!(out.as_ref(), &payload[..], "re-faulted blob differs");
        assert!(
            store.refault_count() > 0,
            "expected re-faults from spill on read",
        );
    }

    #[test]
    fn refault_red_on_revert_broken_fault_in_loses_data() {
        // RED-on-revert for re-fault: if the spill read path is bypassed
        // (simulated: remove the spill index entry so re-fault can't find
        // the page), an evicted-then-read returns MissingHead/BrokenChain
        // — i.e. WITHOUT the re-fault the fix corrupts reads. This is the
        // "break the fault-in" negative control.
        let dir = tempdir().unwrap();
        let store = bounded_store(dir.path(), 1);
        let payload = vec![0x9Au8; BLOB_CHUNK_BYTES + 100]; // 2 chunks
        let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
        simulate_checkpoint(&store);
        store.drain_to_low_watermark().unwrap();
        assert!(store.evicted_count() > 0);

        // Sabotage the re-fault source: clear the spill index. Now an
        // evicted page has no durable image to fault from.
        store.spill.as_ref().unwrap().offsets.clear();
        // Also drop any warmed resident copies so the read must fault.
        store.pages.clear();
        store.resident_bytes.store(0, Ordering::Release);

        let err = store.get(TenantId::DEFAULT, blob_ref).unwrap_err();
        assert!(
            matches!(
                err,
                BlobError::MissingHead { .. } | BlobError::BrokenChain { .. }
            ),
            "without the re-fault an evicted read must fail loud, got {err:?}",
        );
    }

    #[test]
    fn inv_durable_not_checkpointed_page_is_not_evicted() {
        // INV-DURABLE: a page whose durable image is NOT yet captured by a
        // checkpoint must NEVER be evicted (evicting-before-durable = data
        // loss on crash). We put pages OVER the cap but do NOT checkpoint;
        // the drain must evict NOTHING and keep them resident.
        let dir = tempdir().unwrap();
        let store = bounded_store(dir.path(), 2);
        for i in 0..10 {
            store
                .put(TenantId::DEFAULT, format!("nd-{i}").as_bytes())
                .unwrap();
        }
        // No checkpoint yet → no page is `checkpointed` → drain is a no-op.
        store.drain_to_low_watermark().unwrap();
        assert_eq!(
            store.evicted_count(),
            0,
            "INV-DURABLE VIOLATED: evicted a page before it was checkpoint-durable",
        );
        // All 10 pages are still resident + readable.
        assert_eq!(store.page_count(), 10);

        // After a checkpoint marks them durable, the drain evicts down to
        // the low watermark (proves the gate is the ONLY thing that held
        // eviction, not a bug).
        simulate_checkpoint(&store);
        store.drain_to_low_watermark().unwrap();
        assert!(
            store.evicted_count() > 0,
            "post-checkpoint drain should evict now that pages are durable",
        );
        assert!(store.resident_bytes() <= 2 * PAGE_SIZE as u64);
    }

    #[test]
    fn throttle_not_oom_overdrive_keeps_resident_bounded() {
        // THROTTLE-NOT-OOM: over-drive the writer far faster than any
        // checkpoint cadence. Because eviction runs INLINE on `publish`,
        // the writer pays the drain cost synchronously (back-pressure) and
        // the resident set never runs away — the OOM-kill is converted to a
        // bounded resident set (a throughput dip, not an OOM). We interleave
        // frequent checkpoints (durability) with a tight cap.
        let dir = tempdir().unwrap();
        let cap_pages = 4;
        let store = bounded_store(dir.path(), cap_pages);
        let cap_bytes = cap_pages * PAGE_SIZE as u64;

        for i in 0..500 {
            store
                .put(TenantId::DEFAULT, format!("od-{i}").as_bytes())
                .unwrap();
            // Checkpoint every few writes so the drain always has durable
            // pages to shed (models the interval checkpointer keeping pace).
            if i % 3 == 0 {
                simulate_checkpoint(&store);
            }
        }
        // Final checkpoint + drain to shed the tail.
        simulate_checkpoint(&store);
        store.drain_to_low_watermark().unwrap();

        assert!(
            store.resident_bytes() <= cap_bytes,
            "over-drive breached the cap: resident {} > cap {cap_bytes} (would OOM)",
            store.resident_bytes(),
        );
        // And no data was lost — every blob still reads back.
        assert_eq!(store.logical_page_count(), 500);
    }

    #[test]
    fn drain_cost_o1_when_all_pages_not_durable() {
        // #1404 M0 THROUGHPUT-REGRESSION GUARD — the regime `throttle_not_oom`
        // CANNOT catch (it checkpoints every 3 writes, so the drain always
        // finds a durable page at the front). Here we NEVER checkpoint, so
        // every resident page is `NotDurable` and the drain evicts nothing.
        //
        // In this all-NotDurable / over-watermark regime, per-publish drain
        // cost MUST be O(evicted-this-pass) = O(1) — NOT O(resident-page-count).
        // The old code (`budget = queue.len()` + `push_back`-on-NotDurable)
        // walked the ENTIRE FIFO every publish → O(queue-length) per publish →
        // O(N²) aggregate. That would make per-publish cost scale ~linearly
        // with resident-page-count, so the 40k/20k ratio would be ~2×.
        //
        // We measure the MARGINAL per-publish probe cost (deterministic op
        // counter, not wall-time) at two resident sizes and assert the ratio
        // stays ~1 (O(1)), catching any regression to O(queue-length) (~2×).

        // Marginal per-publish drain-probe cost at ~`target` resident pages:
        // fill to `target` un-checkpointed pages, then measure the probe delta
        // over a small window of further publishes (window << target so the
        // measured size is ~stable across the window).
        fn marginal_probe_cost(target: usize, window: usize) -> f64 {
            let dir = tempdir().unwrap();
            // high = low = 1 page → drain engages every publish once we are
            // over 1 resident page, but NOTHING evicts (all NotDurable), so
            // the resident set grows freely — exactly the regression regime.
            let store = bounded_store(dir.path(), 1);
            // Warm up to `target` resident pages. A 1-byte payload = exactly
            // one page (< BLOB_CHUNK_BYTES), so page_count == put count.
            for i in 0..target {
                store
                    .put(TenantId::DEFAULT, format!("nd-{i}").as_bytes())
                    .unwrap();
            }
            assert_eq!(
                store.page_count(),
                target,
                "each 1-page put must add exactly one resident page",
            );
            assert_eq!(
                store.evicted_count(),
                0,
                "no page is checkpoint-durable → nothing may evict (INV-DURABLE)",
            );
            // Measure the marginal probe cost over the window.
            let before = store.drain_probe_count();
            for i in 0..window {
                store
                    .put(TenantId::DEFAULT, format!("win-{target}-{i}").as_bytes())
                    .unwrap();
            }
            let delta = store.drain_probe_count() - before;
            // Still nothing evicted (the whole point of the regime).
            assert_eq!(store.evicted_count(), 0);
            delta as f64 / window as f64
        }

        let window = 200;
        let cost_20k = marginal_probe_cost(20_000, window);
        let cost_40k = marginal_probe_cost(40_000, window);

        // With the fix, each publish's drain probes ONCE (pop front →
        // NotDurable → push_front → break), so both costs are ≈1 and the
        // ratio is ≈1. Under the O(N²) regression each publish probes the
        // whole queue (~resident-page-count), so cost_40k/cost_20k ≈ 2.
        let ratio = cost_40k / cost_20k;
        assert!(
            cost_20k < 2.0 && cost_40k < 2.0,
            "per-publish drain cost must be O(1) in the all-NotDurable regime; \
             got cost_20k={cost_20k:.3} cost_40k={cost_40k:.3} probes/publish \
             (O(N²) regression would make these ~20000 / ~40000)",
        );
        assert!(
            ratio < 1.5,
            "per-publish drain cost must NOT scale with resident-page-count; \
             40k/20k ratio {ratio:.3} (fixed ≈1, O(N²) regression ≈2.0). \
             cost_20k={cost_20k:.3}, cost_40k={cost_40k:.3} probes/publish",
        );
    }

    #[test]
    fn concurrent_evict_vs_capture_completeness_no_page_lost() {
        // COMPLETENESS UNDER RACE (#1404 M0): a writer thread drives publishes
        // (→ inline drain → evict) while the main thread repeatedly captures
        // via `iter_pages_resident_only` (the checkpoint-producer's resident +
        // evicted snapshot). Every logical page id must appear in
        // resident ∪ evicted on EVERY capture — no page may fall through the
        // evict/capture race (that would be OQ-2 silent data loss: a page
        // removed from resident before it is reported evicted, or vice-versa).
        use std::collections::HashSet;
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let dir = tempdir().unwrap();
        // Tight cap so the writer's inline drain actively evicts during the
        // race (each capture marks pages durable → the next drain sheds them).
        let store = Arc::new(bounded_store(dir.path(), 4));
        let writers_done = Arc::new(AtomicBool::new(false));

        const N_WRITES: usize = 4_000;
        let writer_store = Arc::clone(&store);
        let writer_done = Arc::clone(&writers_done);
        let writer = thread::spawn(move || {
            for i in 0..N_WRITES {
                writer_store
                    .put(TenantId::DEFAULT, format!("race-{i}").as_bytes())
                    .expect("put must succeed");
            }
            writer_done.store(true, Ordering::Release);
        });

        // Main thread: hammer captures while the writer races. Each capture
        // returns the (resident, evicted) partition. The completeness property
        // is SET-based and readability-based (robust to the non-atomic two-
        // phase read inside `iter_pages_resident_only` under a live writer —
        // resident-count and evicted-count reflect slightly different instants,
        // so a raw union-COUNT can skew; a set of readable ids cannot):
        //   1. every id the capture REPORTS as evicted must be spill-readable
        //      AT THAT INSTANT (a dropped evicted id — the race hazard — points
        //      at nothing);
        //   2. accumulate every id ever observed (resident ∪ evicted) so we can
        //      prove at the end that NONE became permanently unreadable.
        let mut ever_seen: HashSet<(TenantId, u64)> = HashSet::new();
        let mut captures = 0usize;
        loop {
            let (resident, evicted) = store.iter_pages_resident_only();
            for (tenant, page_id, _bytes) in &resident {
                ever_seen.insert((*tenant, *page_id));
            }
            for key in &evicted {
                // A reported-evicted id must resolve to a durable spill image
                // right now — never dangle (that IS the "fell out of both"
                // silent-data-loss bug this guards).
                assert!(
                    store.read_evicted_page(key.0, key.1).unwrap().is_some(),
                    "evicted id ({:?},{}) reported but not spill-readable — lost in the race",
                    key.0,
                    key.1,
                );
                ever_seen.insert(*key);
            }
            captures += 1;
            if writers_done.load(Ordering::Acquire) && captures >= 8 {
                break;
            }
        }
        writer.join().expect("writer thread panicked");

        // Final settle (single-threaded now): drain, then assert the resident
        // ∪ evicted partition is EXACTLY the full logical set, every observed
        // id is still resolvable (nothing was lost across the whole race), and
        // no blob went missing.
        let _ = store.iter_pages_resident_only();
        store.force_drain_for_test().unwrap();
        let (resident, evicted) = store.iter_pages_resident_only();
        assert_eq!(
            resident.len() + evicted.len(),
            store.logical_page_count(),
            "post-race: resident ∪ evicted must equal the full logical set",
        );
        assert_eq!(
            store.logical_page_count(),
            N_WRITES,
            "no page lost across the entire race",
        );
        // Every id EVER observed during the race is still resolvable now
        // (resident hit or spill re-fault) — nothing fell permanently through
        // the evict/capture race.
        for (tenant, page_id) in &ever_seen {
            assert!(
                store.resolve_page(*tenant, *page_id).unwrap().is_some(),
                "page ({tenant:?},{page_id}) observed during the race is now unreadable (lost)",
            );
        }
        // And every settled-evicted page has its durable spill image.
        for (tenant, page_id) in &evicted {
            assert!(
                store
                    .read_evicted_page(*tenant, *page_id)
                    .unwrap()
                    .is_some(),
                "evicted page ({tenant:?},{page_id}) lost its spill image in the race",
            );
        }
    }

    #[test]
    fn checkpoint_reports_evicted_pages_for_backfill() {
        // The checkpoint-producer seam: after eviction, the NON-FAULTING
        // resident iterator must REPORT the evicted ids (so the producer
        // backfills their durable images from spill) — never silently drop
        // them (that would be OQ-2 data loss).
        let dir = tempdir().unwrap();
        let store = bounded_store(dir.path(), 1);
        for i in 0..6 {
            store
                .put(TenantId::DEFAULT, format!("ck-{i}").as_bytes())
                .unwrap();
        }
        simulate_checkpoint(&store);
        store.drain_to_low_watermark().unwrap();
        assert!(store.evicted_count() > 0);

        let (resident, evicted) = store.iter_pages_resident_only();
        assert!(!evicted.is_empty(), "evicted pages must be reported");
        // Every evicted id must be readable from spill (the backfill source).
        for (tenant, page_id) in &evicted {
            assert!(
                store
                    .read_evicted_page(*tenant, *page_id)
                    .unwrap()
                    .is_some(),
                "evicted page ({tenant:?},{page_id}) missing its spill image",
            );
        }
        // Resident + evicted together = the full logical set (nothing lost).
        assert_eq!(resident.len() + evicted.len(), store.logical_page_count());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(debug_assertions) { 100 } else { 500 },
            .. ProptestConfig::default()
        })]

        #[test]
        fn blob_roundtrip_any_size(
            seed in any::<u64>(),
            len in 1usize..=32_768,
        ) {
            // Deterministic pseudo-random bytes from seed + index.
            let payload: Vec<u8> = (0..len)
                .map(|i| ((seed.wrapping_add(i as u64)) & 0xFF) as u8)
                .collect();
            let store = BlobStore::new();
            let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
            let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
            prop_assert_eq!(out.as_ref(), &payload[..]);
        }

        #[test]
        fn put_blob_payload_roundtrip_any(
            head in any::<u64>(),
            bytes in prop::collection::vec(any::<u8>(), 0..=4096),
        ) {
            let enc = encode_put_blob_payload(head, &bytes);
            let (h, out) = decode_put_blob_payload(&enc).unwrap();
            prop_assert_eq!(h, head);
            prop_assert_eq!(out, bytes);
        }

        /// #1404 M0 — round-trip through the BOUNDED tier with FORCED
        /// eviction: any-size blob, put → checkpoint → drain (evict to
        /// spill) → read = byte-identical. The tiny cap (1 page) guarantees
        /// multi-chunk blobs have evicted pages, so this exercises re-fault
        /// across the whole size range (invariant 2, over any input).
        #[test]
        fn bounded_tier_roundtrip_survives_eviction_any_size(
            seed in any::<u64>(),
            len in 1usize..=32_768,
        ) {
            let payload: Vec<u8> = (0..len)
                .map(|i| ((seed.wrapping_add(i as u64)) & 0xFF) as u8)
                .collect();
            let dir = tempdir().unwrap();
            let store = bounded_store(dir.path(), 1); // cap = 1 page
            let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();
            // Force durability + eviction.
            let _ = store.iter_pages_resident_only();
            store.drain_to_low_watermark().unwrap();
            let out = store.get(TenantId::DEFAULT, blob_ref).unwrap();
            prop_assert_eq!(out.as_ref(), &payload[..]);
        }

        /// v2 M1 — slotted pack/unpack across the whole small-bag size
        /// range through the FULL txn lifecycle (stage → snapshot →
        /// publish → get), extending the records.rs codec proptest to
        /// the store integration: every bag reads back byte-identical
        /// and small bags carry a load-bearing 1-based slot.
        #[test]
        fn slotted_stage_publish_get_roundtrip_any_small_size(
            seed in any::<u64>(),
            lens in proptest::collection::vec(1usize..=PROP_BAG_MAX_BYTES, 1..12),
        ) {
            let store = BlobStore::new();
            let txn = seed | 1;
            let mut expect = Vec::new();
            for (i, len) in lens.iter().enumerate() {
                let payload: Vec<u8> = (0..*len)
                    .map(|j| ((seed ^ (i as u64) << 3).wrapping_add(j as u64) & 0xFF) as u8)
                    .collect();
                let (r, emits) = store
                    .stage_bag(TenantId::DEFAULT, txn, &payload)
                    .unwrap();
                prop_assert!(r.slot_id >= 1, "small bag must pack slotted");
                prop_assert!(emits.is_empty(), "slotted bags defer to snapshot");
                expect.push((r, payload));
            }
            prop_assert!(!store.snapshot_txn_slotted_pages(txn).is_empty());
            store.publish_txn_slotted(txn).unwrap();
            for (r, payload) in &expect {
                let got = store.get(TenantId::DEFAULT, *r).unwrap();
                prop_assert_eq!(got.as_ref(), &payload[..]);
            }
        }
    }

    // ─── v2 M1 — slotted small-blob packing (ADR-230 / #1430) ───

    #[test]
    fn m1_stage_bag_small_bags_share_one_page_and_snapshot_once() {
        // The headline mechanism: many small bags in ONE txn land in ONE
        // shared page, captured as ONE bundle snapshot (design §M1.3).
        let store = BlobStore::new();
        let txn = 7;
        let mut refs = Vec::new();
        for i in 0..100u8 {
            let bag = vec![i; 60];
            let (r, emits) = store.stage_bag(TenantId::DEFAULT, txn, &bag).unwrap();
            assert!(emits.is_empty(), "slotted bags emit nothing per-bag");
            refs.push((r, bag));
        }
        let first_page = refs[0].0.page_id;
        assert!(
            refs.iter().all(|(r, _)| r.page_id == first_page),
            "100 × 60 B bags fit one shared page (127-cap)"
        );
        // Slots are 1-based and dense.
        for (i, (r, _)) in refs.iter().enumerate() {
            assert_eq!(r.slot_id as usize, i + 1, "1-based load-bearing slots");
        }
        let snaps = store.snapshot_txn_slotted_pages(txn);
        assert_eq!(snaps.len(), 1, "ONE page image per bundle, not 100");
        store.publish_txn_slotted(txn).unwrap();
        for (r, bag) in &refs {
            assert_eq!(store.get(TenantId::DEFAULT, *r).unwrap().as_ref(), &bag[..]);
        }
        assert_eq!(store.txn_slotted_scratch_count(), 0, "scratch drained");
    }

    #[test]
    fn m1_bag_capacity_matches_design_arithmetic() {
        // Design §M1.5: 60 B bag + 4 B slot → 8152/64 = 127 bags/page.
        let store = BlobStore::new();
        let txn = 8;
        let mut pages = std::collections::HashSet::new();
        for i in 0..127u32 {
            let bag = vec![(i & 0xFF) as u8; 60];
            let (r, _) = store.stage_bag(TenantId::DEFAULT, txn, &bag).unwrap();
            pages.insert(r.page_id);
        }
        assert_eq!(pages.len(), 1, "127 × 60 B bags pack one page exactly");
        let (r128, _) = store.stage_bag(TenantId::DEFAULT, txn, &[0u8; 60]).unwrap();
        assert!(
            !pages.contains(&r128.page_id),
            "128th bag opens a fresh page"
        );
    }

    #[test]
    fn m1_large_bag_falls_back_to_chain_boundary_exact() {
        // Boundary: PROP_BAG_MAX_BYTES packs slotted; +1 chains.
        let store = BlobStore::new();
        let txn = 9;
        let (r_max, emits_max) = store
            .stage_bag(TenantId::DEFAULT, txn, &vec![0xAB; PROP_BAG_MAX_BYTES])
            .unwrap();
        assert!(r_max.slot_id >= 1, "PROP_BAG_MAX_BYTES still packs");
        assert!(emits_max.is_empty());

        let (r_over, emits_over) = store
            .stage_bag(TenantId::DEFAULT, txn, &vec![0xCD; PROP_BAG_MAX_BYTES + 1])
            .unwrap();
        assert_eq!(
            r_over.slot_id, 0,
            "oversize bag keeps the DEC-4 chain (slot 0)"
        );
        assert!(
            !emits_over.is_empty(),
            "chain pages emit per-bag snapshots exactly as pre-M1"
        );
        // Chain is immediately readable (published eagerly, pre-M1 shape).
        assert_eq!(
            store.get(TenantId::DEFAULT, r_over).unwrap().as_ref(),
            &vec![0xCD; PROP_BAG_MAX_BYTES + 1][..]
        );
        store.publish_txn_slotted(txn).unwrap();
        assert_eq!(
            store.get(TenantId::DEFAULT, r_max).unwrap().as_ref(),
            &vec![0xAB; PROP_BAG_MAX_BYTES][..]
        );
    }

    #[test]
    fn m1_read_your_own_staged_bag_before_commit() {
        // Read-your-own-writes (the held-txn MERGE regression, caught by
        // the mcp `held_txn_single_transaction_merge_twice_matches_second_nn4`
        // gate): a bag staged in an OPEN transaction must be
        // dereferenceable BEFORE publish/commit — exactly the pre-M1
        // chain visibility timing — so the owning txn's scans can match
        // on its own staged properties instead of degrading to an empty
        // bag and double-creating.
        let store = BlobStore::new();
        let (r, _) = store
            .stage_bag(TenantId::DEFAULT, 42, b"staged-not-yet-committed")
            .unwrap();
        assert_eq!(
            store.get(TenantId::DEFAULT, r).unwrap().as_ref(),
            b"staged-not-yet-committed",
            "a staged bag must deref before publish_txn_slotted"
        );
        // Multiple appends: every earlier staged bag stays readable as
        // later snapshots supersede.
        let (r2, _) = store.stage_bag(TenantId::DEFAULT, 42, b"second").unwrap();
        assert_eq!(
            store.get(TenantId::DEFAULT, r).unwrap().as_ref(),
            b"staged-not-yet-committed"
        );
        assert_eq!(
            store.get(TenantId::DEFAULT, r2).unwrap().as_ref(),
            b"second"
        );
        // And rollback of a FRESH page removes the eager publish.
        store.rollback_txn_slotted(42);
        assert!(
            store.get(TenantId::DEFAULT, r).is_err(),
            "rolled-back fresh page must not stay resident"
        );
    }

    #[test]
    fn m1_sequential_txns_share_the_pooled_open_page() {
        // Cross-txn packing over time: commit 1 puts a bag; commit 2
        // (a NEW txn) packs into the SAME page via the tenant pool.
        let store = BlobStore::new();
        let (r1, _) = store.stage_bag(TenantId::DEFAULT, 1, b"first").unwrap();
        store.publish_txn_slotted(1).unwrap();
        let (r2, _) = store.stage_bag(TenantId::DEFAULT, 2, b"second").unwrap();
        store.publish_txn_slotted(2).unwrap();
        assert_eq!(
            r1.page_id, r2.page_id,
            "sequential txns share the open page"
        );
        assert_eq!(r1.slot_id, 1);
        assert_eq!(r2.slot_id, 2);
        assert_eq!(store.get(TenantId::DEFAULT, r1).unwrap().as_ref(), b"first");
        assert_eq!(
            store.get(TenantId::DEFAULT, r2).unwrap().as_ref(),
            b"second"
        );
    }

    #[test]
    fn m1_rollback_restores_pool_exactly_and_committed_reads_unaffected() {
        // z1 discipline, slotted leg: txn A commits a bag; txn B stages
        // into the same pooled page then ROLLS BACK; txn C then stages
        // and gets EXACTLY the slot B would have used (pool restored to
        // the checkout-time image); A's committed bag reads unchanged.
        let store = BlobStore::new();
        let (ra, _) = store.stage_bag(TenantId::DEFAULT, 1, b"committed").unwrap();
        store.publish_txn_slotted(1).unwrap();

        let (rb, _) = store.stage_bag(TenantId::DEFAULT, 2, b"aborted").unwrap();
        assert_eq!(rb.page_id, ra.page_id);
        assert_eq!(rb.slot_id, 2);
        store.rollback_txn_slotted(2);
        assert_eq!(store.txn_slotted_scratch_count(), 0, "scratch dropped");

        // B's bag never published: the resident image still has ONLY A's.
        assert_eq!(
            store.get(TenantId::DEFAULT, ra).unwrap().as_ref(),
            b"committed"
        );
        assert!(
            store.get(TenantId::DEFAULT, rb).is_err(),
            "aborted bag's slot must NOT resolve from the published image"
        );

        let (rc, _) = store.stage_bag(TenantId::DEFAULT, 3, b"third").unwrap();
        store.publish_txn_slotted(3).unwrap();
        assert_eq!(rc.page_id, ra.page_id, "pool restored → same page");
        assert_eq!(rc.slot_id, 2, "pool restored → B's slot re-issued to C");
        assert_eq!(store.get(TenantId::DEFAULT, rc).unwrap().as_ref(), b"third");
    }

    #[test]
    fn m1_kind_mismatch_reads_fail_loud() {
        // slot ref → chain page, and chain ref → slotted page: both are
        // loud SlotRead corruption errors, never silent misreads.
        let store = BlobStore::new();
        let chain_ref = store.put(TenantId::DEFAULT, &vec![1u8; 20_000]).unwrap();
        let (slot_ref, _) = store.stage_bag(TenantId::DEFAULT, 1, b"bag").unwrap();
        store.publish_txn_slotted(1).unwrap();

        // A forged slot-bearing ref onto the chain's head page.
        let forged_slot_on_chain = BlobRef::new(chain_ref.page_id, 1);
        assert!(matches!(
            store.get(TenantId::DEFAULT, forged_slot_on_chain),
            Err(BlobError::SlotRead { .. })
        ));
        // A forged chain ref onto the slotted page.
        let forged_chain_on_slotted = BlobRef::new(slot_ref.page_id, 0);
        assert!(matches!(
            store.get(TenantId::DEFAULT, forged_chain_on_slotted),
            Err(BlobError::SlotRead { .. })
        ));
        // A tombstone-region / out-of-range slot.
        let forged_bad_slot = BlobRef::new(slot_ref.page_id, 99);
        assert!(matches!(
            store.get(TenantId::DEFAULT, forged_bad_slot),
            Err(BlobError::SlotRead { .. })
        ));
    }

    #[test]
    fn m1_bounded_tier_evicts_and_refaults_slotted_pages_byte_identical() {
        // M0-tier-over-slotted (EXIT 7 / the M0 co-existence contract):
        // a slotted page participates in INV-DURABLE eviction + spill
        // re-fault exactly like a chain page, byte-identical after the
        // round trip.
        let dir = tempdir().unwrap();
        let store = bounded_store(dir.path(), 1); // low == high == ONE page
        let mut refs = Vec::new();
        // 1000 B bags pack 8/page → 40 bags span ~5 slotted pages, so
        // the resident set genuinely exceeds the one-page watermark.
        for i in 0..40u8 {
            let bag = vec![i ^ 0x5A; 1000];
            let (r, _) = store
                .stage_bag(TenantId::DEFAULT, u64::from(i) + 1, &bag)
                .unwrap();
            store.publish_txn_slotted(u64::from(i) + 1).unwrap();
            refs.push((r, bag));
        }
        assert!(
            store.resident_bytes() > PAGE_SIZE as u64,
            "precondition: the slotted set must span multiple pages"
        );
        // INV-DURABLE: not evictable before checkpoint capture.
        store.drain_to_low_watermark().unwrap();
        assert_eq!(store.evicted_count(), 0, "un-checkpointed pages must stay");
        // Checkpoint capture marks durable → evictable.
        let _ = store.iter_pages_resident_only();
        store.drain_to_low_watermark().unwrap();
        assert!(store.evicted_count() > 0, "post-capture the drain evicts");
        // Every bag still reads byte-identical (re-fault path).
        for (r, bag) in &refs {
            assert_eq!(
                store.get(TenantId::DEFAULT, *r).unwrap().as_ref(),
                &bag[..],
                "slotted bag must survive evict/re-fault byte-identical"
            );
        }
        assert!(store.refault_count() > 0, "reads re-faulted from spill");
        // And the checkpoint's evicted-supplement source serves them.
        let (_, evicted) = store.iter_pages_resident_only();
        for (tenant, page_id) in evicted {
            let img = store.read_evicted_page(tenant, page_id).unwrap();
            assert!(img.is_some(), "evicted-supplement must serve spilled pages");
        }
    }

    #[test]
    fn m1_install_or_replace_rejects_corrupt_slotted_page() {
        // The trust boundary: a PropSlotted-classified image whose body
        // CRC is broken must be refused loudly (WalCorruption), never
        // silently decoded as a chain.
        let store = BlobStore::new();
        let (r, _) = store.stage_bag(TenantId::DEFAULT, 1, b"seed").unwrap();
        let snaps = store.snapshot_txn_slotted_pages(1);
        store.publish_txn_slotted(1).unwrap();
        let (page_id, mut image) = snaps.into_iter().next().unwrap();
        image[100] ^= 0xFF; // corrupt the body without touching the header
        let err = store
            .install_or_replace(TenantId::DEFAULT, page_id, image)
            .unwrap_err();
        assert!(
            matches!(err, ArcGraphError::WalCorruption { .. }),
            "corrupt slotted page must be refused loud, got {err:?}"
        );
        let _ = r;
    }

    #[test]
    fn m1_concurrent_txns_pack_disjoint_pages_and_all_roundtrip() {
        // RULE-MT (unit leg): ≥8 concurrent writer threads, each its own
        // txn, same tenant. The pool checkout guarantees no two ever
        // share an open page IMAGE concurrently; every bag round-trips.
        // (The bundle/replay concurrent leg lives in
        // tests/m1_slotted_packing.rs.)
        let store = Arc::new(BlobStore::new());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let txn = t + 1;
                let mut out = Vec::new();
                for i in 0..50u64 {
                    let bag = vec![((t * 50 + i) & 0xFF) as u8; 90];
                    let (r, _) = s.stage_bag(TenantId::DEFAULT, txn, &bag).unwrap();
                    out.push((r, bag));
                }
                s.publish_txn_slotted(txn).unwrap();
                out
            }));
        }
        let all: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer thread"))
            .collect();
        for (r, bag) in &all {
            assert_eq!(
                store.get(TenantId::DEFAULT, *r).unwrap().as_ref(),
                &bag[..],
                "every concurrently-staged bag must round-trip"
            );
        }
        // Distinct (page, slot) per bag — no two bags may alias.
        let mut seen = std::collections::HashSet::new();
        for (r, _) in &all {
            assert!(
                seen.insert((r.page_id, r.slot_id)),
                "no two bags may share (page, slot)"
            );
        }
    }
}
