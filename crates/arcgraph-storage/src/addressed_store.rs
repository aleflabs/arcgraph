//! Production direct-addressed node and relationship page stores.
//!
//! The M4 slice keeps this store optional while the primary B-tree remains
//! available. Each `(tenant, record kind)` owns a distinct page space, which
//! enforces ADR-230 amendment-02's node/relationship store split without a
//! second arithmetic namespace. Every id mapping calls the single
//! [`RecordKind::address`] derivation from design §M4.1.

use std::sync::Arc;

use arcgraph_core::record::{NodeRecord, RelRecord};
use arcgraph_core::{Lsn, NodeId, PageId, PageType, RelId, TenantId};
use dashmap::DashMap;
use thiserror::Error;

use crate::address::AddressError;
use crate::primary_index::RecordKind;
use crate::record_store::{RecordPageLatch, RecordPageStore, RecordStoreError};
use crate::records::{PageError, SlotId, SlottedPage, SlottedPageRef};

/// Direct-addressed record-store failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AddressedStoreError {
    /// The raw logical id cannot be addressed.
    #[error(transparent)]
    Address(#[from] AddressError),
    /// The underlying production page store rejected an operation.
    #[error(transparent)]
    Store(#[from] RecordStoreError),
    /// The slotted-page codec rejected an operation or page image.
    #[error(transparent)]
    Page(#[from] PageError),
    /// A page was tagged for another tenant.
    #[error("direct-address page {page_id:?} belongs to {got:?}, expected {expected:?}")]
    TenantMismatch {
        /// Logical page whose header was inconsistent.
        page_id: PageId,
        /// Tenant decoded from the page header.
        got: TenantId,
        /// Tenant used for the lookup.
        expected: TenantId,
    },
    /// A page in a kind-specific store carried the wrong page type.
    #[error("direct-address page {page_id:?} has type {got}, expected {expected}")]
    PageTypeMismatch {
        /// Logical page whose type was inconsistent.
        page_id: PageId,
        /// Page-type byte decoded from its header.
        got: u8,
        /// Page-type byte required by the kind-specific store.
        expected: u8,
    },
    /// A direct-address id was deleted and may never be occupied again.
    #[error(
        "direct-address {kind:?} id {id} for {tenant:?} is permanently tombstoned at LSN {tombstone_lsn}"
    )]
    PermanentTombstone {
        /// Tenant that owns the retired id.
        tenant: TenantId,
        /// Kind-specific store containing the retired id.
        kind: RecordKind,
        /// Raw record id that cannot be reused.
        id: u64,
        /// Commit LSN that made the tombstone permanent.
        tombstone_lsn: u64,
    },
}

/// Apply the P1-b address-read error taxonomy.
///
/// Only a slot beyond an existing page's high-water mark is an address gap.
/// Every format, decode, checksum, tenant, type, and store failure remains a
/// hard error; masking any of those as `None` would turn corruption into
/// silent data loss.
pub fn address_read_disposition<T>(
    result: Result<Option<T>, AddressedStoreError>,
) -> Result<Option<T>, AddressedStoreError> {
    match result {
        Err(AddressedStoreError::Page(PageError::SlotOutOfRange { .. })) => Ok(None),
        other => other,
    }
}

/// Node/relationship direct-address page stores, isolated by tenant and kind.
///
/// Reads use only `DashMap::get` + [`RecordPageStore::latch`]; neither missing
/// tenant stores nor missing pages are created on lookup. Writes are the sole
/// creation path and use the production [`SlottedPage`] codec.
#[derive(Default)]
pub struct AddressedRecordStore {
    stores: DashMap<(TenantId, RecordKind), Arc<RecordPageStore>>,
}

impl AddressedRecordStore {
    /// Construct an empty direct-addressed store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn page_type(kind: RecordKind) -> PageType {
        match kind {
            RecordKind::Node => PageType::Node,
            RecordKind::Rel => PageType::Rel,
        }
    }

    fn store_for_write(&self, tenant: TenantId, kind: RecordKind) -> Arc<RecordPageStore> {
        Arc::clone(
            self.stores
                .entry((tenant, kind))
                .or_insert_with(|| Arc::new(RecordPageStore::new()))
                .value(),
        )
    }

    fn store_for_read(&self, tenant: TenantId, kind: RecordKind) -> Option<Arc<RecordPageStore>> {
        self.stores
            .get(&(tenant, kind))
            .map(|store| Arc::clone(store.value()))
    }

    fn writable_page(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        page_id: PageId,
    ) -> Result<RecordPageLatch, AddressedStoreError> {
        let store = self.store_for_write(tenant, kind);
        match store.latch(page_id) {
            Ok(latch) => Ok(latch),
            Err(RecordStoreError::MissingPage(_)) => {
                match store.install_fresh(page_id, Self::page_type(kind), tenant) {
                    Ok(()) | Err(RecordStoreError::DuplicatePage(_)) => {}
                    Err(error) => return Err(error.into()),
                }
                store.latch(page_id).map_err(Into::into)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn readable_page(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        page_id: PageId,
    ) -> Result<Option<RecordPageLatch>, AddressedStoreError> {
        let Some(store) = self.store_for_read(tenant, kind) else {
            return Ok(None);
        };
        match store.latch(page_id) {
            Ok(latch) => Ok(Some(latch)),
            Err(RecordStoreError::MissingPage(_)) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn validate_page(
        page: &SlottedPageRef<'_>,
        tenant: TenantId,
        kind: RecordKind,
        page_id: PageId,
    ) -> Result<(), AddressedStoreError> {
        let header = page.header();
        let got_tenant = TenantId::new(header.tenant_id);
        if got_tenant != tenant {
            return Err(AddressedStoreError::TenantMismatch {
                page_id,
                got: got_tenant,
                expected: tenant,
            });
        }
        let expected_type = Self::page_type(kind).as_byte();
        if header.page_type != expected_type {
            return Err(AddressedStoreError::PageTypeMismatch {
                page_id,
                got: header.page_type,
                expected: expected_type,
            });
        }
        Ok(())
    }

    /// Write a node at the one address derived from its raw id.
    pub fn write_node(
        &self,
        tenant: TenantId,
        record: &NodeRecord,
    ) -> Result<(), AddressedStoreError> {
        let (page_no, slot) = RecordKind::Node.address(record.id)?;
        let page_id = PageId::new(page_no);
        let latch = self.writable_page(tenant, RecordKind::Node, page_id)?;
        let mut guard = latch.write();
        let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
        Self::validate_page(&page.as_ref(), tenant, RecordKind::Node, page_id)?;
        if slot < page.slot_count()
            && let Some(tombstone_lsn) = page
                .as_ref()
                .permanent_tombstone_lsn(SlotId(slot), NodeRecord::SIZE as u16)?
        {
            return Err(AddressedStoreError::PermanentTombstone {
                tenant,
                kind: RecordKind::Node,
                id: record.id,
                tombstone_lsn,
            });
        }
        page.write_node_at_slot(SlotId(slot), record)?;
        Ok(())
    }

    /// Write a relationship at the one address derived from its raw id.
    pub fn write_rel(
        &self,
        tenant: TenantId,
        record: &RelRecord,
    ) -> Result<(), AddressedStoreError> {
        let (page_no, slot) = RecordKind::Rel.address(record.id)?;
        let page_id = PageId::new(page_no);
        let latch = self.writable_page(tenant, RecordKind::Rel, page_id)?;
        let mut guard = latch.write();
        let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
        Self::validate_page(&page.as_ref(), tenant, RecordKind::Rel, page_id)?;
        if slot < page.slot_count()
            && let Some(tombstone_lsn) = page
                .as_ref()
                .permanent_tombstone_lsn(SlotId(slot), RelRecord::SIZE as u16)?
        {
            return Err(AddressedStoreError::PermanentTombstone {
                tenant,
                kind: RecordKind::Rel,
                id: record.id,
                tombstone_lsn,
            });
        }
        page.write_rel_at_slot(SlotId(slot), record)?;
        Ok(())
    }

    /// Read a node without creating a tenant store or page on a miss.
    pub fn read_node(
        &self,
        tenant: TenantId,
        id: NodeId,
    ) -> Result<Option<NodeRecord>, AddressedStoreError> {
        let (page_no, slot) = RecordKind::Node.address(id.raw())?;
        let page_id = PageId::new(page_no);
        let Some(latch) = self.readable_page(tenant, RecordKind::Node, page_id)? else {
            return Ok(None);
        };
        let guard = latch.read();
        let page = SlottedPageRef::open(guard.as_ref().as_ref())?;
        Self::validate_page(&page, tenant, RecordKind::Node, page_id)?;
        address_read_disposition(page.read_node(SlotId(slot)).map_err(Into::into))
    }

    /// Read a relationship without creating a tenant store or page on a miss.
    pub fn read_rel(
        &self,
        tenant: TenantId,
        id: RelId,
    ) -> Result<Option<RelRecord>, AddressedStoreError> {
        let (page_no, slot) = RecordKind::Rel.address(id.raw())?;
        let page_id = PageId::new(page_no);
        let Some(latch) = self.readable_page(tenant, RecordKind::Rel, page_id)? else {
            return Ok(None);
        };
        let guard = latch.read();
        let page = SlottedPageRef::open(guard.as_ref().as_ref())?;
        Self::validate_page(&page, tenant, RecordKind::Rel, page_id)?;
        address_read_disposition(page.read_rel(SlotId(slot)).map_err(Into::into))
    }

    fn tombstone(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        id: u64,
        tombstone_lsn: Lsn,
    ) -> Result<bool, AddressedStoreError> {
        let (page_no, slot) = kind.address(id)?;
        let page_id = PageId::new(page_no);
        let latch = self.writable_page(tenant, kind, page_id)?;
        let mut guard = latch.write();
        let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
        Self::validate_page(&page.as_ref(), tenant, kind, page_id)?;
        let was_live = if slot < page.slot_count() {
            match kind {
                RecordKind::Node => page.read_node(SlotId(slot))?.is_some(),
                RecordKind::Rel => page.read_rel(SlotId(slot))?.is_some(),
            }
        } else {
            false
        };
        match kind {
            RecordKind::Node => {
                page.permanent_tombstone_node_at_slot(SlotId(slot), tombstone_lsn)?;
            }
            RecordKind::Rel => {
                page.permanent_tombstone_rel_at_slot(SlotId(slot), tombstone_lsn)?;
            }
        }
        Ok(was_live)
    }

    /// Permanently tombstone an existing post-swap node slot.
    pub fn tombstone_node(
        &self,
        tenant: TenantId,
        id: NodeId,
    ) -> Result<bool, AddressedStoreError> {
        self.tombstone_node_at_lsn(tenant, id, Lsn::MAX)
    }

    /// Permanently tombstone a node slot at its deleting commit LSN.
    pub fn tombstone_node_at_lsn(
        &self,
        tenant: TenantId,
        id: NodeId,
        tombstone_lsn: Lsn,
    ) -> Result<bool, AddressedStoreError> {
        self.tombstone(tenant, RecordKind::Node, id.raw(), tombstone_lsn)
    }

    /// Permanently tombstone an existing post-swap relationship slot.
    pub fn tombstone_rel(&self, tenant: TenantId, id: RelId) -> Result<bool, AddressedStoreError> {
        self.tombstone_rel_at_lsn(tenant, id, Lsn::MAX)
    }

    /// Permanently tombstone a relationship slot at its deleting commit LSN.
    pub fn tombstone_rel_at_lsn(
        &self,
        tenant: TenantId,
        id: RelId,
        tombstone_lsn: Lsn,
    ) -> Result<bool, AddressedStoreError> {
        self.tombstone(tenant, RecordKind::Rel, id.raw(), tombstone_lsn)
    }

    /// Number of allocated pages in one tenant/kind store.
    ///
    /// This is also the no-create-on-read oracle used by the M4 lifecycle
    /// gate. Querying the count does not create the store.
    #[must_use]
    pub fn page_count(&self, tenant: TenantId, kind: RecordKind) -> usize {
        self.store_for_read(tenant, kind)
            .map_or(0, |store| store.len())
    }
}
