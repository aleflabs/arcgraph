//! Slotted-record page store (M2-CUTOVER).
//!
//! Hosts the physical `PageType::Node` / `PageType::Rel` slotted pages
//! that CUTOVER's dual-write publishes into, mirroring
//! [`crate::primary_index::PrimaryPageStore`]'s in-memory-DashMap
//! pattern (DEC-17). The permanent home is `BufferPool` once WAL
//! replay lands with M2.e; at alpha, pages live behind a
//! `DashMap<PageId, Arc<RwLock<Box<[u8; PAGE_SIZE]>>>>` so readers
//! can hand-over-hand latch them without a `pin_read` warm-up path
//! for freshly allocated ids.
//!
//! The store is intentionally thin: allocation (via
//! [`crate::page_alloc::PageAllocator`]) and slotted-page codec work
//! (via [`crate::records::SlottedPage`]) remain the caller's concern.
//! The store just tracks `PageId → Latch` and guarantees single-
//! writer discipline per page.

use std::sync::Arc;

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{PageHeader, PageId, PageType, TenantId};
use dashmap::DashMap;
use parking_lot::RwLock;
use thiserror::Error;

use crate::mutation_log::{PageStoreKind, TxnMutationLog};
use crate::records::{PageError, SlottedPage};

/// Raw page buffer; `[u8; PAGE_SIZE]` lives on the heap behind a `Box`.
pub type PageBuf = [u8; PAGE_SIZE];

/// Per-page latch identical to the primary-index store's.
pub type RecordPageLatch = Arc<RwLock<Box<PageBuf>>>;

/// Error surface for the in-memory record store.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RecordStoreError {
    /// Page id not tracked (no prior `install`).
    #[error("record store: page {0:?} not mapped")]
    MissingPage(PageId),
    /// Attempted to install a page id that is already mapped.
    #[error("record store: page {0:?} already mapped")]
    DuplicatePage(PageId),
    /// Caller asked for a page type the record store doesn't host
    /// (only `Node` and `Rel` slotted pages are accepted).
    #[error("record store: unsupported page type {got}")]
    UnsupportedPageType {
        /// The rejected byte.
        got: u8,
    },
    /// Slotted-page codec failure during fresh-install initialization.
    #[error("record store: slotted-page codec error: {0}")]
    Codec(#[from] PageError),
}

/// In-memory record-page store (alpha; DEC-17 says this becomes a
/// `BufferPool` cache at M2.e).
#[derive(Default)]
pub struct RecordPageStore {
    pages: DashMap<PageId, RecordPageLatch>,
}

impl RecordPageStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pages: DashMap::new(),
        }
    }

    /// Number of tracked pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Is the store empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Return a cloneable latch for `page_id`.
    pub fn latch(&self, page_id: PageId) -> Result<RecordPageLatch, RecordStoreError> {
        self.pages
            .get(&page_id)
            .map(|r| Arc::clone(r.value()))
            .ok_or(RecordStoreError::MissingPage(page_id))
    }

    /// Snapshot every tracked `(PageId, latch)` pair into a `Vec`.
    /// Used by `CrudStore::bootstrap_primary_index` to walk the full
    /// set of installed pages — including pages that have been
    /// rotated out of the "currently open" slot-registry and whose
    /// contents are therefore invisible to the
    /// `CrudStore::open_pages` map. Iteration order is arbitrary.
    #[must_use]
    pub fn iter_pages(&self) -> Vec<(PageId, RecordPageLatch)> {
        self.pages
            .iter()
            .map(|e| (*e.key(), Arc::clone(e.value())))
            .collect()
    }

    /// SVC-1 / #849 / ADR-229 REQ-2 — NON-FAULTING resident-page iterator
    /// for the checkpoint producer. Returns `(resident latches,
    /// evicted-page-ids)` WITHOUT reading disk, so the commit-freeze WRITE
    /// guard held during the resident byte-copy never blocks foreground
    /// commits on synchronous fault-in (the 10M availability regression).
    /// `RecordPageStore` is a pure in-memory `DashMap` (no eviction), so
    /// every page is resident and the evicted list is ALWAYS empty. Mirror
    /// of [`crate::primary_index::PrimaryPageStore::iter_pages_resident_only`].
    #[must_use]
    pub fn iter_pages_resident_only(&self) -> (Vec<(PageId, RecordPageLatch)>, Vec<PageId>) {
        (self.iter_pages(), Vec::new())
    }

    /// #1404 M0.x FIX-B — STREAMING resident-page capture: emit each resident
    /// page `(PageId, &RecordPageLatch)` through `f` ONE at a time, so the
    /// producer never pre-collects a whole `Vec` of latches (or the per-page
    /// byte copies) under the freeze. Mirror of
    /// [`crate::primary_index::PrimaryPageStore::for_each_resident_page`]. Wire
    /// bytes byte-identical.
    pub fn for_each_resident_page<F, E>(&self, mut f: F) -> std::result::Result<Vec<PageId>, E>
    where
        F: FnMut(PageId, &RecordPageLatch) -> std::result::Result<(), E>,
    {
        for e in self.pages.iter() {
            f(*e.key(), e.value())?;
        }
        Ok(Vec::new())
    }

    /// #1404 M0.x FIX-B — resident page count for the streaming checkpoint
    /// count header. Cheap DashMap length read.
    #[must_use]
    pub fn resident_page_count(&self) -> usize {
        self.pages.len()
    }

    /// Install a freshly initialized slotted page under `page_id` with
    /// `page_type` (one of [`PageType::Node`] or [`PageType::Rel`]).
    /// Uses [`SlottedPage::init`] to stamp the header + zeroed body +
    /// recompute the body checksum so subsequent [`SlottedPage::open`]
    /// round-trips succeed.
    pub fn install_fresh(
        &self,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        match page_type {
            PageType::Node | PageType::Rel => {}
            other => {
                return Err(RecordStoreError::UnsupportedPageType {
                    got: other.as_byte(),
                });
            }
        }
        let mut buf: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
        let header = PageHeader::new(page_id, page_type, tenant);
        SlottedPage::init(buf.as_mut(), header)?;
        use dashmap::mapref::entry::Entry;
        match self.pages.entry(page_id) {
            Entry::Occupied(_) => Err(RecordStoreError::DuplicatePage(page_id)),
            Entry::Vacant(v) => {
                v.insert(Arc::new(RwLock::new(buf)));
                Ok(())
            }
        }
    }

    /// Transaction-aware [`Self::install_fresh`] sibling that records
    /// the new page in the transaction's mutation log so rollback can
    /// remove it.
    ///
    /// Behaves exactly like [`Self::install_fresh`] plus the
    /// `log.new_pages.push((PageStoreKind::Record, page_id))` entry.
    pub fn install_fresh_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        self.install_fresh(page_id, page_type, tenant)?;
        log.new_pages.push((PageStoreKind::Record, page_id));
        Ok(())
    }

    /// ADR-033 Z-1 (b): capture `page_id`'s pre-mutation bytes into
    /// `log` (if not already captured) and return a write-guard for
    /// in-place mutation. Parallel to
    /// [`crate::primary_index::PrimaryPageStore::capture_and_latch`].
    ///
    /// **Concurrency.** Callers must hold whatever external
    /// serialization primitive the record store relies on
    /// (`CrudStore::open_pages` slot-open serialization at v1.0);
    /// the helper does not add its own serialization beyond the
    /// page's `RwLock`.
    ///
    /// Idempotent within a transaction: first call captures pre-W
    /// bytes; subsequent calls on the same page are no-ops.
    ///
    /// Returns `None` if the page is not mapped. Callers that
    /// allocate pages via [`Self::install_fresh_for_txn`] should
    /// call `install_fresh_for_txn` before the first
    /// `capture_and_write` on the new page — a freshly installed
    /// page has nothing pre-W to capture, so the `has_captured`
    /// dedup tracks only the identity, not the fresh-install origin.
    pub fn capture_and_write(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        let latch = self.latch(page_id)?;
        // Y-2: dedup on (kind, page_id). Previously keyed on page_id
        // alone, which collided with PrimaryPageStore captures on the
        // shared numeric PageId — producing a silent no-op that left
        // the record-page ghost intact on rollback.
        if !log.has_captured(PageStoreKind::Record, page_id) {
            let mut snapshot: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
            {
                let read = latch.read();
                snapshot.copy_from_slice(read.as_ref().as_ref());
            }
            log.page_mutations
                .push((PageStoreKind::Record, page_id, snapshot));
        }
        Ok(latch)
    }

    /// ADR-033 Z-1 (b) rollback primitive: remove a record page
    /// that was installed during a rolled-back transaction. Returns
    /// the removed latch or `None` if not mapped. See
    /// [`crate::primary_index::PrimaryPageStore::remove_page`] for the
    /// hazard-note on non-rollback callers.
    pub fn remove_page(&self, page_id: PageId) -> Option<RecordPageLatch> {
        self.pages.remove(&page_id).map(|(_, latch)| latch)
    }

    /// ADR-033 Z-1 (b) rollback primitive: restore a record page's
    /// in-memory bytes to a pre-captured snapshot. Parallel to
    /// [`crate::primary_index::PrimaryPageStore::restore_page_bytes`].
    pub fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError> {
        let latch = self.latch(page_id)?;
        let mut guard = latch.write();
        let bytes: &mut PageBuf = guard.as_mut();
        bytes.copy_from_slice(pre_bytes);
        Ok(())
    }

    /// ADR-032 Slice 3 + PR #79 X-1 / X-2 review fold-in:
    /// unconditional byte-copy install. Overwrites any existing
    /// page-id mapping; installs fresh if none. Called by the
    /// replay executor's
    /// [`crate::wal::RecordPageStoreHandle::install_or_replace`]
    /// impl.
    ///
    /// Lemma I2 is **bundle-level** — a later bundle's entry for
    /// the same page_id is a legitimate supersession, NOT a
    /// corruption. Byte-equality comparison at the entry level
    /// (the pre-fold-in behaviour) was the X-1 bug.
    pub fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError> {
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

    /// Test/observability: does the store currently map `page_id`?
    #[doc(hidden)]
    #[must_use]
    pub fn contains(&self, page_id: PageId) -> bool {
        self.pages.contains_key(&page_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::SlottedPage;

    #[test]
    fn install_and_latch_roundtrip() {
        let store = RecordPageStore::new();
        let pid = PageId::new(42);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let latch = store.latch(pid).unwrap();
        // Page can be opened by the slotted-page codec after install_fresh.
        let mut g = latch.write();
        let _page = SlottedPage::open(g.as_mut().as_mut()).unwrap();
    }

    #[test]
    fn duplicate_install_rejected() {
        let store = RecordPageStore::new();
        let pid = PageId::new(1);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let err = store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap_err();
        assert!(matches!(err, RecordStoreError::DuplicatePage(_)));
    }

    #[test]
    fn missing_page_errors() {
        let store = RecordPageStore::new();
        let err = store.latch(PageId::new(999)).unwrap_err();
        assert!(matches!(err, RecordStoreError::MissingPage(_)));
    }

    #[test]
    fn unsupported_page_type_rejected() {
        let store = RecordPageStore::new();
        let err = store
            .install_fresh(PageId::new(1), PageType::IndexLeaf, TenantId::DEFAULT)
            .unwrap_err();
        assert!(matches!(err, RecordStoreError::UnsupportedPageType { .. }));
    }

    // ─── ADR-033 Z-1 (b): RecordPageStore rollback helpers ───

    #[test]
    fn install_fresh_for_txn_records_new_pages_entry() {
        let store = RecordPageStore::new();
        let mut log = TxnMutationLog::new();
        store
            .install_fresh_for_txn(&mut log, PageId::new(55), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        assert_eq!(log.new_pages.len(), 1);
        assert_eq!(log.new_pages[0], (PageStoreKind::Record, PageId::new(55)));
        assert!(store.contains(PageId::new(55)));
    }

    #[test]
    fn capture_and_write_captures_pre_mutation_bytes_once() {
        let store = RecordPageStore::new();
        store
            .install_fresh(PageId::new(60), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let mut log = TxnMutationLog::new();

        // First capture snapshots pre-W bytes.
        {
            let latch = store.capture_and_write(&mut log, PageId::new(60)).unwrap();
            let mut g = latch.write();
            g.as_mut().as_mut()[16] = 0xAA;
        }
        assert_eq!(log.page_mutations.len(), 1);
        assert_eq!(log.page_mutations[0].0, PageStoreKind::Record);
        assert_eq!(log.page_mutations[0].1, PageId::new(60));

        // Second capture is idempotent.
        {
            let latch = store.capture_and_write(&mut log, PageId::new(60)).unwrap();
            let mut g = latch.write();
            g.as_mut().as_mut()[17] = 0xBB;
        }
        assert_eq!(log.page_mutations.len(), 1);

        // Captured snapshot reflects pre-W bytes at offset 16 (= 0).
        let snap = &log.page_mutations[0].2;
        assert_eq!(snap.as_ref()[16], 0);
        assert_eq!(snap.as_ref()[17], 0);
    }

    #[test]
    fn capture_and_write_missing_page_errors() {
        let store = RecordPageStore::new();
        let mut log = TxnMutationLog::new();
        let err = store
            .capture_and_write(&mut log, PageId::new(999))
            .unwrap_err();
        assert!(matches!(err, RecordStoreError::MissingPage(_)));
    }

    #[test]
    fn remove_page_and_restore_page_bytes_roundtrip() {
        let store = RecordPageStore::new();
        store
            .install_fresh(PageId::new(70), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        // Capture pre-W bytes (a freshly initialized slotted node page).
        let pristine: Box<PageBuf> = {
            let latch = store.latch(PageId::new(70)).unwrap();
            let g = latch.read();
            let mut copy: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
            copy.copy_from_slice(g.as_ref().as_ref());
            copy
        };
        // Mutate + assert bytes changed.
        {
            let latch = store.latch(PageId::new(70)).unwrap();
            let mut g = latch.write();
            g.as_mut().as_mut()[16] = 0x5A;
        }
        // Restore pristine bytes.
        store
            .restore_page_bytes(PageId::new(70), pristine.as_ref())
            .unwrap();
        // Assert restoration succeeded.
        {
            let latch = store.latch(PageId::new(70)).unwrap();
            let g = latch.read();
            assert_eq!(g.as_ref().as_ref()[16], 0);
        }
        // Remove page.
        let removed = store.remove_page(PageId::new(70));
        assert!(removed.is_some());
        assert!(!store.contains(PageId::new(70)));
    }
}
