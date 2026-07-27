//! Correctness-first v5 -> v6 offline store rewriter.
//!
//! The source generation is read-only. Every v6 extent mapping is emitted in
//! dense append order after the fixed directory head, and every rewritten
//! slotted page is stamped at the migration frontier. The caller owns the
//! generation fsync/publish protocol; this module only builds and verifies an
//! invisible generation.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use arcgraph_core::{Lsn, PAGE_SIZE, PageHeader, PageId, PageType, TenantId};

use crate::address::AddressError;
use crate::addressed_store::AddressedRecordStore;
use crate::blob::BlobStore;
use crate::crud::{node_mvcc_key, rel_mvcc_key};
use crate::extent::{
    DIRECTORY_ENTRIES_PER_PAGE, DIRECTORY_ENTRY_BYTES, DIRECTORY_HEAD_BYTES, EXTENT_BYTES,
    EXTENT_PAGES, MAX_EXTENTS_PER_STORE, PRODUCTION_EXTENT_SUBDIR, production_extent_store_path,
};
use crate::m3_migration::{M3_PROPS_STORE_FILE, M3_RECORD_STORE_FILE, M3_TENANTS_DIR};
use crate::owner_budget::OwnerBulkBudgets;
use crate::owner_index::{OWNER_INDEX_DISK_CAP_BYTES, OwnerForwardIndex, str_hash_56};
use crate::owner_payload::{OWNER_PAYLOAD_DISK_CAP_BYTES, OwnerPayloadStore};
use crate::owner_rewrite::{
    BoundedOwnerSorter, OWNER_REWRITE_RUN_BUFFER_BYTES, OWNER_REWRITE_SCRATCH_CAP_BYTES,
    OwnerRewriteError, OwnerRewriteScratchBudget,
};
use crate::owner_row::{
    BindingOwnerValue, ClassOwnerValue, GrantOwnerValue, InternOwnerValue,
    OWNER_ALLOCATOR_MARKER_ID, OWNER_ROWS_PER_PAGE, OwnerAllocatorMarker, OwnerRow, OwnerRowClass,
    owner_forward_index_path, owner_payload_path,
};
use crate::primary_index::RecordKind;
use crate::records::{
    NODE_CAPACITY, PROP_BAG_MAX_BYTES, REL_CAPACITY, SlotId, SlottedPage, SlottedPageRef,
};
use crate::transaction::TxnManager;
use crate::wal::{
    STORE_BLOB_OVERFLOW, STORE_GRANTS, STORE_INTERN, STORE_NODE_BINDINGS, STORE_PROPS,
    STORE_RECORD, STORE_REL_BINDINGS, STORE_RELS, STORE_SECONDARY_INDEX, STORE_TEL,
};

/// Complete extent-backed v6 store-id set. The primary B-tree (store 3) is
/// deliberately absent; secondary index pages remain generation artifacts.
pub const M4_EXTENT_STORE_IDS: &[u16] = &[
    STORE_PROPS,
    STORE_RECORD,
    STORE_TEL,
    STORE_SECONDARY_INDEX,
    STORE_BLOB_OVERFLOW,
    STORE_RELS,
    STORE_NODE_BINDINGS,
    STORE_REL_BINDINGS,
    STORE_INTERN,
    STORE_GRANTS,
];

/// One physical mapping emitted by the rewriter, exposed for verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewrittenExtent {
    pub tenant: TenantId,
    pub store_id: u16,
    pub logical_extent: u64,
    pub physical_offset: u64,
}

/// Counts and the complete physical-offset ledger for one rewrite.
#[derive(Debug, Default)]
pub struct M4RewriteReport {
    pub nodes: u64,
    pub rels: u64,
    pub prop_pages: u64,
    pub tenants: BTreeSet<TenantId>,
    pub extents: Vec<RewrittenExtent>,
    /// Owner 6 intern rows migrated.
    pub intern_rows: u64,
    /// Owner 7 binding rows migrated.
    pub binding_rows: u64,
    /// Owner 8 document grant rows migrated.
    pub grant_rows: u64,
    /// Durable ACL classes emitted after canonical grant grouping.
    pub class_rows: u64,
    /// Peak shared external-sort scratch bytes (hard-capped).
    pub owner_scratch_peak_bytes: u64,
}

/// Counts observed while opening the authoritative v6 extent base.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct M4BaseLoadReport {
    pub nodes: u64,
    pub rels: u64,
    pub record_pages: u64,
    pub prop_pages: u64,
}

/// The single LSN frontier owned by one loader invocation.
///
/// Loader pages are stamped at `migration_lsn`; the first live-WAL commit is
/// allocated from `next_lsn`.  Keeping the pair in one release-validated value
/// prevents an attach caller from independently rebasing either side of the
/// handoff and suppressing redo for the first post-attach commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoaderMigrationFrontier {
    migration_lsn: Lsn,
    next_lsn: u64,
}

/// Complete set of final-layout producers covered by the M5 encoding gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderTarget {
    /// New tenant generation, attached by the fresh-load slice.
    Fresh,
    /// Beside-build of an already attached tenant generation.
    ExistingRecluster,
    /// Independent M4-lite reference producer; never used for M5 attach.
    M4LiteReference,
}

impl LoaderMigrationFrontier {
    pub fn new(migration_lsn: Lsn) -> Result<Self> {
        ensure!(
            migration_lsn != Lsn::ZERO && migration_lsn != Lsn::MAX,
            "loader migration LSN must be non-zero and below u64::MAX"
        );
        Ok(Self {
            migration_lsn,
            next_lsn: migration_lsn.raw() + 1,
        })
    }

    #[must_use]
    pub const fn migration_lsn(self) -> Lsn {
        self.migration_lsn
    }

    #[must_use]
    pub const fn next_lsn(self) -> u64 {
        self.next_lsn
    }
}

struct ExtentStoreWriter {
    tenant: TenantId,
    store_id: u16,
    file: File,
    mapping_count: u64,
    last_logical_extent: Option<u64>,
    current_directory_page: Option<(u64, Box<[u8; PAGE_SIZE]>)>,
    extents: Vec<RewrittenExtent>,
}

impl ExtentStoreWriter {
    fn create(generation: &Path, tenant: TenantId, store_id: u16) -> Result<Self> {
        let path = production_extent_store_path(generation, tenant, store_id)
            .context("M4 store id has no production extent path")?;
        fs::create_dir_all(path.parent().expect("production store has a parent"))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("create v6 extent store {}", path.display()))?;
        Ok(Self {
            tenant,
            store_id,
            file,
            mapping_count: 0,
            last_logical_extent: None,
            current_directory_page: None,
            extents: Vec::new(),
        })
    }

    fn ensure_extent(&mut self, logical_extent: u64, migration_lsn: Lsn) -> Result<u64> {
        if self.last_logical_extent == Some(logical_extent) {
            return Ok(self.extents.last().expect("mapping exists").physical_offset);
        }
        if let Some(last) = self.last_logical_extent {
            ensure!(
                logical_extent > last,
                "v6 logical extents must be emitted in increasing order: {logical_extent} after {last}"
            );
        }
        ensure!(
            logical_extent < MAX_EXTENTS_PER_STORE,
            "logical extent {logical_extent} exceeds MAX_EXTENTS_PER_STORE"
        );
        ensure!(
            self.mapping_count < MAX_EXTENTS_PER_STORE,
            "extent count exceeds MAX_EXTENTS_PER_STORE"
        );
        let physical_offset = DIRECTORY_HEAD_BYTES
            .checked_add(
                self.mapping_count
                    .checked_mul(EXTENT_BYTES)
                    .context("v6 dense extent offset overflow")?,
            )
            .context("v6 physical extent offset overflow")?;
        let directory_page_index = logical_extent / DIRECTORY_ENTRIES_PER_PAGE;
        let directory_slot = logical_extent % DIRECTORY_ENTRIES_PER_PAGE;
        if self
            .current_directory_page
            .as_ref()
            .is_some_and(|(page, _)| *page != directory_page_index)
        {
            self.flush_directory_page(migration_lsn)?;
        }
        if self.current_directory_page.is_none() {
            self.current_directory_page = Some((directory_page_index, Box::new([0_u8; PAGE_SIZE])));
        }
        let (_, bytes) = self.current_directory_page.as_mut().expect("created above");
        let offset = PageHeader::SIZE + directory_slot as usize * DIRECTORY_ENTRY_BYTES;
        bytes[offset..offset + 8].copy_from_slice(&logical_extent.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&physical_offset.to_le_bytes());
        bytes[offset + 16..offset + 20].copy_from_slice(&(logical_extent as u32).to_le_bytes());
        bytes[offset + 20..offset + 24].copy_from_slice(&1_u32.to_le_bytes());
        self.mapping_count += 1;
        self.last_logical_extent = Some(logical_extent);
        self.extents.push(RewrittenExtent {
            tenant: self.tenant,
            store_id: self.store_id,
            logical_extent,
            physical_offset,
        });
        Ok(physical_offset)
    }

    fn write_page(
        &mut self,
        page_no: u64,
        bytes: &[u8; PAGE_SIZE],
        migration_lsn: Lsn,
    ) -> Result<()> {
        let view = SlottedPageRef::open(bytes).context("validate rewritten v6 page")?;
        ensure!(
            view.header().page_id == page_no,
            "rewritten page header id {} != logical page {page_no}",
            view.header().page_id
        );
        ensure!(
            view.header().tenant_id == self.tenant.raw(),
            "rewritten page tenant does not match its extent store"
        );
        ensure!(
            view.page_lsn() == migration_lsn,
            "rewritten page_lsn {} != migration_lsn {}",
            view.page_lsn().raw(),
            migration_lsn.raw()
        );
        let logical_extent = page_no / EXTENT_PAGES;
        let within_extent = page_no % EXTENT_PAGES;
        let physical = self.ensure_extent(logical_extent, migration_lsn)?;
        let offset = physical
            .checked_add(within_extent * PAGE_SIZE as u64)
            .context("v6 data page offset overflow")?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(bytes)?;
        Ok(())
    }

    fn flush_directory_page(&mut self, migration_lsn: Lsn) -> Result<()> {
        let Some((page_index, mut bytes)) = self.current_directory_page.take() else {
            return Ok(());
        };
        let entries = bytes[PageHeader::SIZE..]
            .chunks_exact(DIRECTORY_ENTRY_BYTES)
            .filter(|entry| entry.iter().any(|byte| *byte != 0))
            .count();
        let tagged_page = crate::extent::directory_page_no(page_index)?;
        let mut header = PageHeader::new(PageId::new(tagged_page), PageType::Free, self.tenant);
        header.flags = self.store_id;
        header.lsn = migration_lsn.raw();
        header.slot_count =
            u16::try_from(entries).context("directory entry count overflows u16")?;
        header.free_space =
            u16::try_from(PAGE_SIZE - PageHeader::SIZE - entries * DIRECTORY_ENTRY_BYTES)
                .context("directory free-space overflows u16")?;
        header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
        bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        self.file
            .seek(SeekFrom::Start(page_index * PAGE_SIZE as u64))?;
        self.file.write_all(bytes.as_ref())?;
        Ok(())
    }

    /// Write one non-slotted [`PageType::Tel`] page image (M5-D2,
    /// amendment §3.2). TEL pages are not slotted, so the slotted-view
    /// validation in [`Self::write_page`] cannot apply; this variant
    /// enforces the same identity/frontier contract through the raw
    /// [`PageHeader`] instead — exactly the fields the M3 delta codec's
    /// TEL leg validates on redo (`redo.rs` `apply_recovery_delta`,
    /// `STORE_TEL` + `PageAlloc`): page id, tenant, `PageType::Tel`,
    /// body CRC32C, and `page_lsn == migration_lsn`.
    fn write_tel_page(
        &mut self,
        page_no: u64,
        bytes: &[u8; PAGE_SIZE],
        migration_lsn: Lsn,
    ) -> Result<()> {
        ensure!(
            self.store_id == STORE_TEL,
            "TEL page routed to non-TEL extent store {}",
            self.store_id
        );
        let header = PageHeader::from_bytes(
            bytes[..PageHeader::SIZE]
                .try_into()
                .expect("fixed page header"),
        )
        .context("validate fresh TEL page header")?;
        ensure!(
            header.page_id == page_no
                && header.tenant_id == self.tenant.raw()
                && header.page_type == PageType::Tel.as_byte()
                && header.lsn == migration_lsn.raw()
                && crc32c::crc32c(&bytes[PageHeader::SIZE..]) == header.checksum,
            "fresh TEL page identity/frontier/checksum mismatch at page {page_no}"
        );
        let logical_extent = page_no / EXTENT_PAGES;
        let within_extent = page_no % EXTENT_PAGES;
        let physical = self.ensure_extent(logical_extent, migration_lsn)?;
        let offset = physical
            .checked_add(within_extent * PAGE_SIZE as u64)
            .context("v6 TEL page offset overflow")?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(bytes)?;
        Ok(())
    }

    fn finish(mut self, migration_lsn: Lsn) -> Result<Vec<RewrittenExtent>> {
        self.flush_directory_page(migration_lsn)?;
        self.file.sync_all()?;
        verify_dense_offsets(&self.extents)?;
        Ok(self.extents)
    }
}

struct DirectRecordWriter {
    kind: RecordKind,
    store: ExtentStoreWriter,
    current: Option<(u64, Box<[u8; PAGE_SIZE]>)>,
}

/// Loader link to the one total id-address derivation. This function contains
/// no arithmetic of its own: node, relationship, and tombstone emission all
/// route through [`RecordKind::address`], including its sentinel/tag bounds.
pub fn loader_record_address(kind: RecordKind, id: u64) -> Result<(u64, u16), AddressError> {
    kind.address(id)
}

impl DirectRecordWriter {
    fn new(generation: &Path, tenant: TenantId, kind: RecordKind) -> Result<Self> {
        let store_id = match kind {
            RecordKind::Node => STORE_RECORD,
            RecordKind::Rel => STORE_RELS,
        };
        Ok(Self {
            kind,
            store: ExtentStoreWriter::create(generation, tenant, store_id)?,
            current: None,
        })
    }

    fn write_node(&mut self, record: &arcgraph_core::record::NodeRecord, lsn: Lsn) -> Result<()> {
        ensure!(self.kind == RecordKind::Node, "node routed to rel writer");
        let (page_no, slot) = loader_record_address(self.kind, record.id)?;
        self.prepare_page(page_no, lsn)?;
        let (_, bytes) = self.current.as_mut().expect("prepared page");
        SlottedPage::open(bytes.as_mut())?
            .write_node_at_slot(crate::records::SlotId(slot), record)?;
        Ok(())
    }

    fn write_rel(&mut self, record: &arcgraph_core::record::RelRecord, lsn: Lsn) -> Result<()> {
        ensure!(self.kind == RecordKind::Rel, "rel routed to node writer");
        let (page_no, slot) = loader_record_address(self.kind, record.id)?;
        self.prepare_page(page_no, lsn)?;
        let (_, bytes) = self.current.as_mut().expect("prepared page");
        SlottedPage::open(bytes.as_mut())?
            .write_rel_at_slot(crate::records::SlotId(slot), record)?;
        Ok(())
    }

    fn tombstone(&mut self, id: u64, lsn: Lsn) -> Result<()> {
        let (page_no, slot) = loader_record_address(self.kind, id)?;
        self.prepare_page(page_no, lsn)?;
        let (_, bytes) = self.current.as_mut().expect("prepared page");
        let mut page = SlottedPage::open(bytes.as_mut())?;
        match self.kind {
            RecordKind::Node => page.permanent_tombstone_node_at_slot(SlotId(slot), lsn)?,
            RecordKind::Rel => page.permanent_tombstone_rel_at_slot(SlotId(slot), lsn)?,
        }
        Ok(())
    }

    fn prepare_page(&mut self, page_no: u64, lsn: Lsn) -> Result<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|(page, _)| *page == page_no)
        {
            return Ok(());
        }
        if let Some((current, _)) = self.current.as_ref() {
            ensure!(page_no > *current, "v5 record scan is not in id order");
        }
        self.flush_page(lsn)?;
        let page_type = match self.kind {
            RecordKind::Node => PageType::Node,
            RecordKind::Rel => PageType::Rel,
        };
        let mut bytes = Box::new([0_u8; PAGE_SIZE]);
        let mut header = PageHeader::new(PageId::new(page_no), page_type, self.store.tenant);
        header.lsn = lsn.raw();
        SlottedPage::init(bytes.as_mut(), header)?;
        self.current = Some((page_no, bytes));
        Ok(())
    }

    fn flush_page(&mut self, lsn: Lsn) -> Result<()> {
        if let Some((page_no, bytes)) = self.current.take() {
            self.store.write_page(page_no, bytes.as_ref(), lsn)?;
        }
        Ok(())
    }

    fn finish(mut self, lsn: Lsn) -> Result<Vec<RewrittenExtent>> {
        self.flush_page(lsn)?;
        self.store.finish(lsn)
    }
}

struct OwnerDirectWriter {
    store: ExtentStoreWriter,
    current: Option<(u64, Box<[u8; PAGE_SIZE]>)>,
    last_target: Option<(u64, u16)>,
}

impl OwnerDirectWriter {
    fn new(generation: &Path, tenant: TenantId, store_id: u16) -> Result<Self> {
        Ok(Self {
            store: ExtentStoreWriter::create(generation, tenant, store_id)?,
            current: None,
            last_target: None,
        })
    }

    fn write(&mut self, row: &OwnerRow, lsn: Lsn) -> Result<()> {
        ensure!(
            row.class().store_id() == self.store.store_id,
            "owner row routed to wrong extent store"
        );
        let address = row.class().address(row.id())?;
        let target = (address.page_no, address.slot.raw());
        if let Some(previous) = self.last_target {
            ensure!(
                target > previous,
                "owner direct-row stream is not strictly address-sorted"
            );
        }
        if self
            .current
            .as_ref()
            .is_none_or(|(page_no, _)| *page_no != address.page_no)
        {
            self.flush(lsn)?;
            let mut bytes = Box::new([0_u8; PAGE_SIZE]);
            let mut header = PageHeader::new(
                PageId::new(address.page_no),
                PageType::PropSlotted,
                self.store.tenant,
            );
            header.flags = self.store.store_id;
            header.lsn = lsn.raw();
            SlottedPage::init(bytes.as_mut(), header)?;
            self.current = Some((address.page_no, bytes));
        }
        let (_, bytes) = self.current.as_mut().expect("owner page prepared");
        SlottedPage::open(bytes.as_mut())?.put_fixed_bag_at_slot(
            address.slot,
            &row.encode(),
            OWNER_ROWS_PER_PAGE as u16,
        )?;
        self.last_target = Some(target);
        Ok(())
    }

    fn flush(&mut self, lsn: Lsn) -> Result<()> {
        if let Some((page_no, bytes)) = self.current.take() {
            self.store.write_page(page_no, bytes.as_ref(), lsn)?;
        }
        Ok(())
    }

    fn finish(mut self, lsn: Lsn) -> Result<Vec<RewrittenExtent>> {
        self.flush(lsn)?;
        self.store.finish(lsn)
    }
}

const OWNER_INDEX_BUILD_BATCH: usize = 65_536;

struct OwnerClassCompanion {
    payload: OwnerPayloadStore,
    index: Option<OwnerForwardIndex>,
    pending_index: Vec<(u64, u64)>,
}

struct OwnerWriters {
    direct: BTreeMap<u16, OwnerDirectWriter>,
    companions: BTreeMap<OwnerRowClass, OwnerClassCompanion>,
}

impl OwnerWriters {
    fn create(
        generation: &Path,
        tenant: TenantId,
        budgets: Option<&OwnerBulkBudgets>,
    ) -> Result<Self> {
        let mut direct = BTreeMap::new();
        for store_id in [
            STORE_NODE_BINDINGS,
            STORE_REL_BINDINGS,
            STORE_INTERN,
            STORE_GRANTS,
        ] {
            direct.insert(
                store_id,
                OwnerDirectWriter::new(generation, tenant, store_id)?,
            );
        }
        let mut companions = BTreeMap::new();
        for class in OwnerRowClass::ALL {
            // M5-D3 (amendment §5): the bulk loader passes census-derived
            // caps for the two binding classes; every other caller (and
            // every other class) keeps the fixed incremental defaults.
            let (index_cap, payload_cap) = match (class, budgets) {
                (OwnerRowClass::NodeBinding, Some(budgets)) => (
                    budgets.node_bindings.index_cap_bytes,
                    budgets.node_bindings.payload_cap_bytes,
                ),
                (OwnerRowClass::RelBinding, Some(budgets)) => (
                    budgets.rel_bindings.index_cap_bytes,
                    budgets.rel_bindings.payload_cap_bytes,
                ),
                _ => (OWNER_INDEX_DISK_CAP_BYTES, OWNER_PAYLOAD_DISK_CAP_BYTES),
            };
            let payload = OwnerPayloadStore::create(
                &owner_payload_path(generation, tenant, class),
                payload_cap,
            )?;
            let index = owner_forward_index_path(generation, tenant, class)
                .map(|path| OwnerForwardIndex::create(&path, index_cap))
                .transpose()?;
            companions.insert(
                class,
                OwnerClassCompanion {
                    payload,
                    index,
                    pending_index: Vec::with_capacity(OWNER_INDEX_BUILD_BATCH),
                },
            );
        }
        Ok(Self { direct, companions })
    }

    fn write_logical(
        &mut self,
        class: OwnerRowClass,
        id: u64,
        logical: &[u8],
        forward_hash: Option<u64>,
        lsn: Lsn,
    ) -> Result<()> {
        let companion = self
            .companions
            .get_mut(&class)
            .context("owner class companion is missing")?;
        let row = OwnerRow::new(class, id, companion.payload.encode(logical)?)?;
        self.direct
            .get_mut(&class.store_id())
            .context("owner direct writer is missing")?
            .write(&row, lsn)?;
        if let Some(hash) = forward_hash {
            ensure!(companion.index.is_some(), "forward owner has no index");
            companion.pending_index.push((hash, id));
            if companion.pending_index.len() >= OWNER_INDEX_BUILD_BATCH {
                Self::flush_index(companion)?;
            }
        }
        Ok(())
    }

    fn flush_index(companion: &mut OwnerClassCompanion) -> Result<()> {
        if companion.pending_index.is_empty() {
            return Ok(());
        }
        let entries = std::mem::take(&mut companion.pending_index);
        companion
            .index
            .as_ref()
            .context("owner forward index is missing")?
            .insert_batch(entries)?;
        companion.pending_index = Vec::with_capacity(OWNER_INDEX_BUILD_BATCH);
        Ok(())
    }

    fn finish(mut self, lsn: Lsn) -> Result<Vec<RewrittenExtent>> {
        for companion in self.companions.values_mut() {
            Self::flush_index(companion)?;
        }
        let mut extents = Vec::new();
        for (_, writer) in self.direct {
            extents.extend(writer.finish(lsn)?);
        }
        Ok(extents)
    }
}

struct TenantWriters {
    nodes: DirectRecordWriter,
    rels: DirectRecordWriter,
    props: ExtentStoreWriter,
    /// STORE_TEL writer. The v5→v6 migration producers never write into
    /// it (the v5 format has no on-disk TEL; adjacency is rebuilt at open
    /// pre-M6), so it finishes complete-but-empty there — byte-identical
    /// to the pre-D2 `empty` handling. The M5-D2 fresh loader streams
    /// both TEL directions through it (amendment §3.2, INV-M5.20).
    tel: ExtentStoreWriter,
    empty: Vec<ExtentStoreWriter>,
    owners: OwnerWriters,
}

impl TenantWriters {
    fn create(generation: &Path, tenant: TenantId) -> Result<Self> {
        Self::create_with_budgets(generation, tenant, None)
    }

    fn create_with_budgets(
        generation: &Path,
        tenant: TenantId,
        budgets: Option<&OwnerBulkBudgets>,
    ) -> Result<Self> {
        let mut empty = Vec::new();
        for store_id in [STORE_SECONDARY_INDEX, STORE_BLOB_OVERFLOW] {
            empty.push(ExtentStoreWriter::create(generation, tenant, store_id)?);
        }
        Ok(Self {
            nodes: DirectRecordWriter::new(generation, tenant, RecordKind::Node)?,
            rels: DirectRecordWriter::new(generation, tenant, RecordKind::Rel)?,
            props: ExtentStoreWriter::create(generation, tenant, STORE_PROPS)?,
            tel: ExtentStoreWriter::create(generation, tenant, STORE_TEL)?,
            empty,
            owners: OwnerWriters::create(generation, tenant, budgets)?,
        })
    }

    fn finish(self, lsn: Lsn) -> Result<Vec<RewrittenExtent>> {
        let mut extents = self.nodes.finish(lsn)?;
        extents.extend(self.rels.finish(lsn)?);
        extents.extend(self.props.finish(lsn)?);
        extents.extend(self.tel.finish(lsn)?);
        extents.extend(self.owners.finish(lsn)?);
        for store in self.empty {
            extents.extend(store.finish(lsn)?);
        }
        Ok(extents)
    }
}

/// TEL chain direction being streamed into [`FreshV6Builder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FreshTelDirection {
    /// Outgoing adjacency — entries' `dst_id` is the relationship target.
    Out,
    /// Incoming adjacency — entries' `dst_id` is the relationship SOURCE
    /// (the reverse-TEL convention: the block header's owner is the
    /// destination vertex, mirroring the in-memory reverse chains).
    In,
}

/// Maximum TEL entries in one loader-built on-disk block, and the largest a
/// single BLOCK may ever grow to regardless of the packed-vs-supernode page
/// layout below: body = `PAGE_SIZE - PageHeader::SIZE` bytes, block = 32 B
/// TEL header + entries (design-v2 §3.3 layout). A block never exceeds one
/// page's worth of bytes — supernode blocks fill a whole page; packed
/// (small) blocks are one of several sharing a page. `append_tel_entry`
/// force-flushes a block once it reaches this cap, exactly as before #1519.
pub const FRESH_TEL_ENTRIES_PER_PAGE: u64 = ((PAGE_SIZE - PageHeader::SIZE) as u64
    - crate::tel::HEADER_SIZE as u64)
    / crate::tel::ENTRY_SIZE as u64;

/// #1519 DENSIFY: supernode threshold. A flushed (owner, type) block with
/// `entry_count >= TEL_SUPERNODE_THRESHOLD_ENTRIES` gets its own dedicated
/// `PageType::Tel` page (the pre-#1519 page-per-block layout, chained via
/// `prev_block_ptr` exactly as before); a block below the threshold is
/// small enough that packing SEVERAL per page recovers the ~200× blowup a
/// low-average-degree fixture would otherwise pay (5.9 TB @ 100M measured
/// by the M5-D3 100M rung, avg out-degree 5 / 7 types => 1-2 entries/block
/// against an 8 KiB page).
///
/// Rationale for the exact cutoff (half of [`FRESH_TEL_ENTRIES_PER_PAGE`],
/// = 126 of 253): a block at or above half a page's entry capacity can
/// share its page with AT MOST one similarly-sized sibling — the packing
/// win collapses to <=2x right as the directory bookkeeping and the loss
/// of "this owner's whole adjacency is one contiguous page" locality start
/// to cost something. Below half, many small blocks (the common case at
/// low average degree) pack N-per-page with N large, which is exactly the
/// regime the #1519 regression pin (`tel_disk_size_is_dense`) targets.
pub const TEL_SUPERNODE_THRESHOLD_ENTRIES: u32 = (FRESH_TEL_ENTRIES_PER_PAGE / 2) as u32;

/// `PageHeader::flags` value for a densified PACKED `PageType::Tel` page:
/// body starts with a [`TEL_PACKED_DIR_HEADER_BYTES`]-byte directory count
/// header, followed by N [`TEL_PACKED_DIR_SLOT_BYTES`]-byte directory
/// slots, followed by the N packed blocks themselves (unchanged per-block
/// byte layout). Distinct from `flags = 0` (the supernode/chain page —
/// single block, no directory, byte-identical to the pre-#1519 format) so
/// a reader can dispatch on ONE header field without guessing from body
/// contents.
pub const TEL_PAGE_FLAG_PACKED: u16 = 1;

/// Packed-page directory header: `block_count: u32 LE` + 4 reserved bytes
/// (always zero on write; a reader must not assume any particular value on
/// old bytes because THIS page format never shipped before #1519 — the
/// reserved word exists purely so the header stays 8-byte aligned for the
/// directory slots that follow).
pub const TEL_PACKED_DIR_HEADER_BYTES: usize = 8;

/// Packed-page directory slot: `owner_id: u64 LE (8) | type_id: u32 LE (4)
/// | offset: u16 LE (2) | len: u16 LE (2)` = 16 bytes. `offset` is the
/// byte offset from the page BODY start (i.e. relative to the first byte
/// after `PageHeader::SIZE`) where this block's 32-byte TEL header begins;
/// `len` is the block's total byte length (TEL header + entries), i.e.
/// exactly the `block_size` field inside that block's own header.
pub const TEL_PACKED_DIR_SLOT_BYTES: usize = 16;

/// Encode a resolved TEL block location as the opaque ref stamped into
/// `NodeRecord::out_tel_ref` / `in_tel_ref` and into a block's
/// `prev_block_ptr` chain-link field (#1519 read-path contract).
///
/// `page_no` is the physical `PageType::Tel` page; `slot` is this block's
/// index into that page's packed directory (always `0` for a supernode /
/// chain page, which holds exactly one block and carries no directory).
/// Packing `(page_no, slot)` into one u64 keeps every existing u64
/// ref/pointer field byte-compatible: `page_no` occupies the high 48 bits
/// (comfortably covers any realistic extent count) and `slot` the low 16
/// bits (a page can never pack more than a few hundred minimal blocks, so
/// 16 bits is far more than enough headroom). `0` remains the "no chain"
/// sentinel at the `NodeRecord` level because valid page ids start at 1.
#[must_use]
pub const fn encode_tel_ref(page_no: u64, slot: u16) -> u64 {
    (page_no << 16) | slot as u64
}

/// Inverse of [`encode_tel_ref`]: `(page_no, slot)`.
#[must_use]
pub const fn decode_tel_ref(reference: u64) -> (u64, u16) {
    (reference >> 16, (reference & 0xFFFF) as u16)
}

/// One fresh node for [`FreshV6Builder::push_node`] (M5-D2): identity +
/// crud-grain property bag + both TEL chain-head refs (0 = no chain).
#[derive(Debug, Clone, Copy)]
pub struct FreshNode<'a> {
    /// Dense loader-assigned node id (> 0, ascending across pushes).
    pub id: u64,
    /// Label id.
    pub label: u32,
    /// External binding id (UTF-8).
    pub external_id: &'a str,
    /// Crud-grain property payload; empty = no properties.
    pub bag: &'a [u8],
    /// Outgoing TEL chain-head page id (0 = none).
    pub out_tel_ref: u64,
    /// Incoming TEL chain-head page id (0 = none).
    pub in_tel_ref: u64,
}

/// One fresh relationship for [`FreshV6Builder::push_relationship`].
#[derive(Debug, Clone, Copy)]
pub struct FreshRel<'a> {
    /// Dense loader-assigned relationship id (> 0, ascending).
    pub id: u64,
    /// Relationship type id.
    pub type_id: u32,
    /// Resolved source node id.
    pub source_id: u64,
    /// Resolved target node id.
    pub target_id: u64,
    /// External binding id (UTF-8).
    pub external_id: &'a str,
    /// Crud-grain property payload; empty = no properties.
    pub bag: &'a [u8],
}

/// #1519 DENSIFY: the ONE resident shared packing buffer that small
/// (sub-[`TEL_SUPERNODE_THRESHOLD_ENTRIES`]) TEL blocks accumulate into
/// before a page-grain flush — the packed-page counterpart of the
/// pre-#1519 "one page-grain TEL block" residency budget (module docs,
/// `m5_load.rs` header comment): O(1) resident regardless of owner count,
/// bounded by one 8 KiB page's directory + block bytes at any time.
///
/// Blocks are appended in STREAM order (id-ordered per the `s6` merge —
/// `merge_tel_runs`), so the directory this buffer builds is already
/// `(owner_id, type_id)`-ascending; a page is flushed either when the next
/// block would not fit, or explicitly at chain-stream end
/// ([`FreshV6Builder::finish`]). Determinism (INV-M5.24): the flush
/// decision depends ONLY on accumulated byte size vs. page capacity, both
/// pure functions of stream content — no worker-count-dependent state.
#[derive(Default)]
struct PendingTelPackPage {
    /// `(owner_id, type_id, block_bytes)` in the order blocks were pushed
    /// (== id-ascending, per the caller's streaming contract).
    blocks: Vec<(u64, u32, Vec<u8>)>,
    /// Running total of already-buffered block bytes (NOT including the
    /// directory header/slots — recomputed at flush time from `blocks.len()`).
    body_bytes: usize,
    /// The page number reserved for the CURRENT buffer contents, claimed
    /// from `next_tel_page` the moment the first block lands in an empty
    /// buffer (not at flush time). This is what makes a just-pushed
    /// block's [`encode_tel_ref`] ref valid immediately, before the page
    /// is actually written to disk — and keeps `next_tel_page` advancing
    /// in the same deterministic, stream-order-only sequence regardless
    /// of when the accumulated page happens to fill up.
    page_no: Option<u64>,
}

impl PendingTelPackPage {
    /// Bytes a page with `n` directory slots has left for block bytes.
    fn capacity_for(n: usize) -> usize {
        (PAGE_SIZE - PageHeader::SIZE)
            .saturating_sub(TEL_PACKED_DIR_HEADER_BYTES + n * TEL_PACKED_DIR_SLOT_BYTES)
    }

    /// Push one small block into the buffer, flushing the current page
    /// first if it doesn't have room (directory growth included). Returns
    /// the [`encode_tel_ref`] ref for the pushed block's final resting
    /// page/slot.
    fn push(
        &mut self,
        ctx: &mut TelWriteCtx<'_>,
        owner: u64,
        type_id: u32,
        block_bytes: Vec<u8>,
    ) -> Result<u64> {
        let would_be_slots = self.blocks.len() + 1;
        if !self.blocks.is_empty()
            && self.body_bytes + block_bytes.len() > Self::capacity_for(would_be_slots)
        {
            self.flush(ctx)?;
        }
        // A single block must fit alone even on an otherwise-empty page;
        // `flush_tel_block`'s page-bound check above already guarantees
        // `block_bytes.len() <= PAGE_SIZE - PageHeader::SIZE`, and the
        // supernode threshold keeps packed blocks well under half a page,
        // so one directory slot's overhead never pushes a lone block over.
        let page_no = *self.page_no.get_or_insert_with(|| {
            let claimed = *ctx.next_tel_page;
            *ctx.next_tel_page = ctx
                .next_tel_page
                .checked_add(1)
                .expect("TEL page id overflow (checked at flush time too)");
            claimed
        });
        let slot = self.blocks.len() as u16;
        self.body_bytes += block_bytes.len();
        self.blocks.push((owner, type_id, block_bytes));
        Ok(encode_tel_ref(page_no, slot))
    }

    /// Flush the accumulated blocks as one densified packed page (`flags =
    /// `[`TEL_PAGE_FLAG_PACKED`]`) and reset the buffer. A no-op when empty
    /// (nothing accumulated since the last flush). The page number was
    /// already reserved from `next_tel_page` at first-push time (see
    /// [`Self::push`]), so this method needs no further page-id counter.
    fn flush(&mut self, ctx: &mut TelWriteCtx<'_>) -> Result<()> {
        if self.blocks.is_empty() {
            return Ok(());
        }
        let blocks = std::mem::take(&mut self.blocks);
        self.body_bytes = 0;
        let page_no = self
            .page_no
            .take()
            .expect("non-empty buffer always has a reserved page_no");
        let block_count = u32::try_from(blocks.len()).expect("page-bounded block count");
        let dir_bytes = TEL_PACKED_DIR_HEADER_BYTES + blocks.len() * TEL_PACKED_DIR_SLOT_BYTES;
        let mut bytes = Box::new([0_u8; PAGE_SIZE]);
        let body = &mut bytes[PageHeader::SIZE..];
        body[0..4].copy_from_slice(&block_count.to_le_bytes());
        body[4..8].copy_from_slice(&0_u32.to_le_bytes());
        let mut cursor = dir_bytes;
        for (index, (owner, type_id, block)) in blocks.iter().enumerate() {
            let slot_offset = TEL_PACKED_DIR_HEADER_BYTES + index * TEL_PACKED_DIR_SLOT_BYTES;
            let block_offset = u16::try_from(cursor).context("TEL packed block offset overflow")?;
            let block_len = u16::try_from(block.len()).context("TEL packed block len overflow")?;
            body[slot_offset..slot_offset + 8].copy_from_slice(&owner.to_le_bytes());
            body[slot_offset + 8..slot_offset + 12].copy_from_slice(&type_id.to_le_bytes());
            body[slot_offset + 12..slot_offset + 14].copy_from_slice(&block_offset.to_le_bytes());
            body[slot_offset + 14..slot_offset + 16].copy_from_slice(&block_len.to_le_bytes());
            body[cursor..cursor + block.len()].copy_from_slice(block);
            cursor += block.len();
        }
        ensure!(
            cursor <= PAGE_SIZE - PageHeader::SIZE,
            "packed TEL page overflow"
        );
        let mut header = PageHeader::new(PageId::new(page_no), PageType::Tel, ctx.tenant);
        header.lsn = ctx.frontier.raw();
        header.flags = TEL_PAGE_FLAG_PACKED;
        header.slot_count = u16::try_from(block_count).expect("block count fits a page");
        header.free_space =
            u16::try_from(PAGE_SIZE - PageHeader::SIZE - cursor).expect("packed page bounds");
        header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
        bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        ctx.tel
            .write_tel_page(page_no, bytes.as_ref(), ctx.frontier)?;
        Ok(())
    }
}

/// Borrowed write context threaded through the TEL flush helpers
/// ([`PendingTelPackPage::push`]/`flush`, `FreshV6Builder::flush_tel_block`)
/// — bundles the extent writer, the shared page-id counter, and the
/// per-generation tenant/frontier stamp so those functions stay under
/// clippy's argument-count lint instead of taking four+ loose parameters.
struct TelWriteCtx<'a> {
    tel: &'a mut ExtentStoreWriter,
    next_tel_page: &'a mut u64,
    tenant: TenantId,
    frontier: Lsn,
}

/// In-progress single-owner TEL chain (one direction) being streamed.
struct FreshTelChain {
    owner: u64,
    direction: FreshTelDirection,
    /// Channel (relationship type) of the block currently accumulating.
    current_type: Option<u32>,
    /// `(neighbor, rel_id)` entries of the current single-channel block,
    /// bounded by [`FRESH_TEL_ENTRIES_PER_PAGE`] (flushed page-grain —
    /// O(1) resident regardless of owner degree).
    entries: Vec<(u64, u64)>,
    /// Opaque [`encode_tel_ref`] ref of the newest block flushed so far
    /// (the chain walks newest → oldest through `prev_block_ptr`, which
    /// carries the SAME encoding). `None` until the first flush.
    newest_ref: Option<u64>,
    /// Strictly-ascending `(type_id, rel_id)` enforcement.
    last_key: Option<(u32, u64)>,
}

/// Streaming producer for a fresh M5 tenant targeting the same production v6
/// extent layout as the M4 rewriter (salvaged sound from PR #1504 per
/// `docs/design/M5D-REDESIGN-AMENDMENT.md` §9). It retains at most one
/// direct-addressed page per store, one open property-bag page, one
/// page-grain TEL block, plus the bounded owner-index batch.
///
/// # M5-D2 served-store completeness (amendment §3, INV-M5.20)
///
/// - **Typed properties:** every `push_node` / `push_relationship` carries
///   its crud-grain property bag. Bags ≤ [`PROP_BAG_MAX_BYTES`] pack into
///   shared [`PageType::PropSlotted`] pages written into the STORE_PROPS
///   extents (the SAME page codec — [`SlottedPage::insert_bag`] — and the
///   SAME 1-based-slot [`crate::property::BlobRef`] convention as the live
///   `BlobStore::stage_bag` slotted path), and the record's `property_ref`
///   is stamped via the production `encode_overflow_*` encoders. Cold open
///   restores them through `load_v6_physical_base` →
///   `BlobStore::install_m3_base_page`, the proven M4 doctrine.
/// - **Oversized bags:** > [`PROP_BAG_MAX_BYTES`] chains through the
///   production [`BlobStore::put`] DEC-4 path into the caller-supplied
///   [`BlobStore`]; the loader's first checkpoint
///   ([`establish_fresh_v6_checkpoint`]) captures those chain pages as
///   page-images in the v9 incremental metadata — the LANDED blob layout
///   (blob.overflow store 5 remains page-image per the Director ruling
///   recorded in `checkpoint/snapshot.rs`), restored at cold open by
///   `read_incremental_metadata`. Page ids for slotted and chain pages
///   come from the ONE shared `BlobStore` allocator, so they never
///   collide, and cold-open installs `fetch_max` the allocator above
///   every loaded id.
/// - **TEL (both directions):** the loader's source-sorted and
///   destination-sorted runs stream through
///   [`Self::begin_tel_chain`] / [`Self::append_tel_entry`] /
///   [`Self::finish_tel_chain`] into STORE_TEL extents. Each (owner, type)
///   segment is still exactly one design-v2 §3.3 [`crate::tel`] BLOCK
///   (32 B TEL header + entries written BACKWARDS, entry 0 = oldest — the
///   exact `TelBlock::append`/`entry_bytes` offset arithmetic, UNCHANGED
///   by #1519); what #1519 changes is how blocks are PACKED INTO PAGES:
///
///   ```text
///   PageHeader (PageHeader::SIZE B): page_id, PageType::Tel, tenant,
///       lsn = migration_lsn, checksum = crc32c(body); flags
///       distinguishes the two page shapes below.
///
///   flags = 0 (SUPERNODE / chain page — pre-#1519 shape, unchanged):
///     one block fills the body directly at offset 0:
///       0..8    src_vertex_id (owner)      u64 LE
///       8..12   block_size (= 32 + n*32)   u32 LE
///       12..16  entry_count (= n)          u32 LE
///       16..24  prev_block_ptr             u64 LE ([`encode_tel_ref`];
///                                          u64::MAX = none)
///       24..28  label (relationship type)  u32 LE
///       28..32  lock (always 0 on disk)    u32 LE
///       entries as above.
///     Used for blocks with `entry_count >=
///     `[`TEL_SUPERNODE_THRESHOLD_ENTRIES`]` — a supernode's block(s) get
///     their own page(s), chained via `prev_block_ptr` exactly as before.
///
///   flags = `[`TEL_PAGE_FLAG_PACKED`]` (DENSIFIED — #1519):
///     body[0..4]  block_count            u32 LE
///     body[4..8]  reserved (0)           u32 LE
///     directory[0..block_count]: `[`TEL_PACKED_DIR_SLOT_BYTES`]`-byte
///       slots starting at body offset `[`TEL_PACKED_DIR_HEADER_BYTES`]`:
///         owner_id  u64 LE (8) | type_id  u32 LE (4)
///         offset    u16 LE (2) | len      u16 LE (2)
///       `offset`/`len` locate one block (the SAME byte shape as the
///       flags=0 case above) within the body: block bytes are
///       `body[offset..offset+len]`.
///     blocks: `block_count` blocks packed back-to-back immediately after
///       the directory, id-ordered by `(owner_id, type_id)` (determinism —
///       INV-M5.24) — each in the unchanged per-block byte format.
///     Used for blocks with `entry_count <
///     `[`TEL_SUPERNODE_THRESHOLD_ENTRIES`]`; multiple owners' small
///     blocks share one page instead of each wasting a whole 8 KiB page
///     (the #1519 fix for the ~200× blowup the M5-D3 100M rung measured
///     at low average degree).
///   ```
///
///   Blocks are single-channel (one relationship type), per the
///   production `TelBlock` grain; an owner's per-type segments link into
///   one chain via `prev_block_ptr`, which (like `NodeRecord.out_tel_ref` /
///   `in_tel_ref`) carries an [`encode_tel_ref`] `(page_no, slot)` pair —
///   `0` = no chain at the `NodeRecord` level (page ids start at 1); a
///   reader resolves a ref by opening `page_no`, checking `flags`, and
///   either reading the sole block directly (`flags = 0`) or looking up
///   `slot` in the packed directory (`flags = `[`TEL_PAGE_FLAG_PACKED`]`).
///   Nothing serves these pages before M6 (bootstrap still performs the
///   #780 in-RAM rebuild, and no live path emits STORE_TEL deltas, so
///   post-load appends cannot collide with loaded pages on disk);
///   INV-M5.20 is the M5→M6 contract that the data IS on disk.
pub struct FreshV6Builder {
    tenant: TenantId,
    frontier: Lsn,
    writers: TenantWriters,
    /// Shared page-id space + chain staging for property bags. The caller
    /// hands the SAME store to [`establish_fresh_v6_checkpoint`] so
    /// oversized-bag chain pages ride the first checkpoint's metadata.
    blob: Arc<BlobStore>,
    /// Open shared slotted property-bag page: `(page_id, image)`.
    open_prop_page: Option<(u64, Box<[u8; PAGE_SIZE]>)>,
    /// Next STORE_TEL page id (1-based; 0 = the "no chain" record ref).
    next_tel_page: u64,
    /// #1519 DENSIFY: the ONE resident shared packing buffer for
    /// sub-threshold blocks, spanning across owner chains. Flushed when
    /// full (mid-stream) or explicitly at [`Self::finish`] (any leftover).
    tel_pack: PendingTelPackPage,
    /// In-flight TEL chain, if any.
    tel_chain: Option<FreshTelChain>,
    /// Once `In`-direction streaming starts, `Out` is sealed (single
    /// monotone page sequence through the shared extent writer).
    tel_in_started: bool,
    /// Once record emission starts, TEL streaming is sealed (refs must
    /// exist before the records that carry them).
    records_started: bool,
    last_node: Option<u64>,
    last_rel: Option<u64>,
    /// Out-/in-TEL entries written (report surface).
    pub out_tel_entries: u64,
    /// In-TEL entries written (report surface).
    pub in_tel_entries: u64,
    /// Oversized property bags chained through the blob store.
    pub chained_bags: u64,
    report: M4RewriteReport,
}

impl FreshV6Builder {
    pub fn create(
        generation: &Path,
        tenant: TenantId,
        frontier: Lsn,
        blob: Arc<BlobStore>,
    ) -> Result<Self> {
        Self::create_with_budgets(generation, tenant, frontier, blob, None)
    }

    /// [`Self::create`] with census-derived owner-substrate caps for the
    /// bulk path (M5-D3, amendment §5). `None` keeps the fixed incremental
    /// defaults — the migration legs and every non-bulk caller pass `None`.
    pub fn create_with_budgets(
        generation: &Path,
        tenant: TenantId,
        frontier: Lsn,
        blob: Arc<BlobStore>,
        budgets: Option<&OwnerBulkBudgets>,
    ) -> Result<Self> {
        ensure!(
            frontier != Lsn::ZERO && frontier != Lsn::MAX,
            "fresh v6 frontier is invalid"
        );
        let mut report = M4RewriteReport::default();
        report.tenants.insert(tenant);
        Ok(Self {
            tenant,
            frontier,
            writers: TenantWriters::create_with_budgets(generation, tenant, budgets)?,
            blob,
            open_prop_page: None,
            next_tel_page: 1,
            tel_pack: PendingTelPackPage::default(),
            tel_chain: None,
            tel_in_started: false,
            records_started: false,
            last_node: None,
            last_rel: None,
            out_tel_entries: 0,
            in_tel_entries: 0,
            chained_bags: 0,
            report,
        })
    }

    /// Begin streaming one owner's TEL chain. Owners must be strictly
    /// ascending per direction, and every `Out` chain must complete
    /// before the first `In` chain (one monotone page sequence).
    pub fn begin_tel_chain(&mut self, direction: FreshTelDirection, owner: u64) -> Result<()> {
        ensure!(
            !self.records_started,
            "TEL chains must be streamed before record emission"
        );
        ensure!(self.tel_chain.is_none(), "previous TEL chain is unfinished");
        match direction {
            FreshTelDirection::Out => {
                ensure!(
                    !self.tel_in_started,
                    "out-TEL chain after in-TEL streaming started"
                );
            }
            FreshTelDirection::In => self.tel_in_started = true,
        }
        self.tel_chain = Some(FreshTelChain {
            owner,
            direction,
            current_type: None,
            entries: Vec::new(),
            newest_ref: None,
            last_key: None,
        });
        Ok(())
    }

    /// Append one adjacency entry to the in-flight chain. `(type_id,
    /// rel_id)` must be strictly ascending within the chain; `neighbor`
    /// is the far endpoint (target for `Out`, source for `In`).
    pub fn append_tel_entry(&mut self, type_id: u32, neighbor: u64, rel_id: u64) -> Result<()> {
        let chain = self
            .tel_chain
            .as_mut()
            .context("append_tel_entry without an open TEL chain")?;
        ensure!(
            chain.last_key.is_none_or(|last| (type_id, rel_id) > last),
            "TEL entries are not (type, rel) sorted"
        );
        chain.last_key = Some((type_id, rel_id));
        let type_changed = chain.current_type.is_some_and(|current| current != type_id);
        let block_full = chain.entries.len() as u64 >= FRESH_TEL_ENTRIES_PER_PAGE;
        if type_changed || block_full {
            let mut ctx = TelWriteCtx {
                tel: &mut self.writers.tel,
                next_tel_page: &mut self.next_tel_page,
                tenant: self.tenant,
                frontier: self.frontier,
            };
            Self::flush_tel_block(&mut ctx, &mut self.tel_pack, chain)?;
        }
        chain.current_type = Some(type_id);
        chain.entries.push((neighbor, rel_id));
        match chain.direction {
            FreshTelDirection::Out => self.out_tel_entries += 1,
            FreshTelDirection::In => self.in_tel_entries += 1,
        }
        Ok(())
    }

    /// Complete the in-flight chain, returning the head ref
    /// ([`encode_tel_ref`]) to stamp into the owner's `out_tel_ref` /
    /// `in_tel_ref` (never 0). NOTE: the pending packed page (if any small
    /// block from this chain is sitting in it) is deliberately NOT flushed
    /// here — packing spans MULTIPLE owners' chains by design, so the
    /// buffer only flushes when full or at [`Self::finish`].
    pub fn finish_tel_chain(&mut self) -> Result<u64> {
        let mut chain = self
            .tel_chain
            .take()
            .context("finish_tel_chain without an open TEL chain")?;
        if !chain.entries.is_empty() {
            let mut ctx = TelWriteCtx {
                tel: &mut self.writers.tel,
                next_tel_page: &mut self.next_tel_page,
                tenant: self.tenant,
                frontier: self.frontier,
            };
            Self::flush_tel_block(&mut ctx, &mut self.tel_pack, &mut chain)?;
        }
        chain
            .newest_ref
            .context("TEL chain finished with zero entries")
    }

    /// Serialize the current single-channel block's bytes (the unchanged
    /// per-block layout: 32 B TEL header + entries written BACKWARDS).
    /// Drains `chain.entries`. Returns `(owner, label, block_size,
    /// entry_count, bytes)`.
    fn encode_tel_block(
        frontier: Lsn,
        chain: &mut FreshTelChain,
        prev_ref: u64,
    ) -> (u64, u32, u32, Vec<u8>) {
        let label = chain.current_type.expect("non-empty block has a channel");
        let entry_count = u32::try_from(chain.entries.len()).expect("page-bounded entry count");
        let block_size = crate::tel::HEADER_SIZE + entry_count * crate::tel::ENTRY_SIZE;
        let mut bytes = vec![0_u8; block_size as usize];
        bytes[0..8].copy_from_slice(&chain.owner.to_le_bytes());
        bytes[8..12].copy_from_slice(&block_size.to_le_bytes());
        bytes[12..16].copy_from_slice(&entry_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&prev_ref.to_le_bytes());
        bytes[24..28].copy_from_slice(&label.to_le_bytes());
        bytes[28..32].copy_from_slice(&0_u32.to_le_bytes());
        for (index, (neighbor, rel_id)) in chain.entries.drain(..).enumerate() {
            let entry = arcgraph_core::TelEntry::new(
                arcgraph_core::NodeId::new(neighbor),
                arcgraph_core::RelId::new(rel_id),
                frontier,
            );
            let end = block_size as usize - index * crate::tel::ENTRY_SIZE as usize;
            let start = end - crate::tel::ENTRY_SIZE as usize;
            bytes[start..end].copy_from_slice(&entry.to_bytes());
        }
        (chain.owner, label, entry_count, bytes)
    }

    /// Flush the current single-channel block. #1519 DENSIFY dispatch:
    /// blocks at/above [`TEL_SUPERNODE_THRESHOLD_ENTRIES`] get a dedicated
    /// chain page (`flags = 0`, byte-identical to the pre-#1519 shape);
    /// smaller blocks go through the shared [`PendingTelPackPage`] so
    /// several owners' blocks share one 8 KiB page.
    fn flush_tel_block(
        ctx: &mut TelWriteCtx<'_>,
        pack: &mut PendingTelPackPage,
        chain: &mut FreshTelChain,
    ) -> Result<()> {
        if chain.entries.is_empty() {
            return Ok(());
        }
        let entry_count = chain.entries.len() as u32;
        ensure!(
            PageHeader::SIZE
                + (crate::tel::HEADER_SIZE + entry_count * crate::tel::ENTRY_SIZE) as usize
                <= PAGE_SIZE,
            "TEL block exceeds one page"
        );
        let frontier = ctx.frontier;
        let prev_ref = chain.newest_ref.unwrap_or(crate::tel::NO_PREV_BLOCK);
        // #1519 RED-on-revert seam (`tel_disk_size_is_dense`, the #1519
        // regression pin): force every block down the pre-#1519
        // page-per-block path regardless of size. cfg-gated + bounded per
        // the standing test-hook rule; production builds compile this
        // check out entirely (the `false` branch below is a compile-time
        // constant, so the packing path is never even reachable-checked
        // for cost in release builds).
        #[cfg(feature = "fault-injection")]
        let force_page_per_block = std::env::var_os("ARCGRAPH_M5_TEL_PAGE_PER_BLOCK").is_some();
        #[cfg(not(feature = "fault-injection"))]
        let force_page_per_block = false;
        if force_page_per_block || entry_count >= TEL_SUPERNODE_THRESHOLD_ENTRIES {
            // Supernode: force out any pending packed page FIRST so page
            // numbers stay allocated in the deterministic id-ordered
            // stream sequence (packing never reorders across a supernode
            // boundary — INV-M5.24 byte-identity depends on a total order
            // over page allocation that depends only on stream content).
            pack.flush(ctx)?;
            let (_owner, _label, block_entry_count, block_bytes) =
                Self::encode_tel_block(frontier, chain, prev_ref);
            let page_no = *ctx.next_tel_page;
            *ctx.next_tel_page = ctx
                .next_tel_page
                .checked_add(1)
                .context("TEL page id overflow")?;
            let mut bytes = Box::new([0_u8; PAGE_SIZE]);
            bytes[PageHeader::SIZE..PageHeader::SIZE + block_bytes.len()]
                .copy_from_slice(&block_bytes);
            let mut header = PageHeader::new(PageId::new(page_no), PageType::Tel, ctx.tenant);
            header.lsn = frontier.raw();
            header.flags = 0;
            header.slot_count = u16::try_from(block_entry_count).expect("entry count fits a page");
            header.free_space = u16::try_from(PAGE_SIZE - PageHeader::SIZE - block_bytes.len())
                .expect("block fits the page body");
            header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
            bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
            ctx.tel.write_tel_page(page_no, bytes.as_ref(), frontier)?;
            chain.newest_ref = Some(encode_tel_ref(page_no, 0));
        } else {
            let (owner, label, _block_entry_count, block_bytes) =
                Self::encode_tel_block(frontier, chain, prev_ref);
            let reference = pack.push(ctx, owner, label, block_bytes)?;
            chain.newest_ref = Some(reference);
        }
        chain.current_type = None;
        Ok(())
    }

    /// Materialize one crud-grain property bag: ≤ [`PROP_BAG_MAX_BYTES`]
    /// packs into the open shared slotted page (STORE_PROPS extents);
    /// larger chains through the production [`BlobStore::put`] DEC-4
    /// path (first-checkpoint page-images). Empty bag = no properties.
    fn stage_props(&mut self, bag: &[u8]) -> Result<Option<crate::property::BlobRef>> {
        if bag.is_empty() {
            return Ok(None);
        }
        if bag.len() > PROP_BAG_MAX_BYTES {
            let blob_ref = self
                .blob
                .put(self.tenant, bag)
                .context("chain oversized fresh property bag")?;
            self.chained_bags += 1;
            return Ok(Some(blob_ref));
        }
        if let Some((page_id, image)) = self.open_prop_page.as_mut() {
            match SlottedPage::open_prop_trusted(&mut image[..])
                .map_err(|error| anyhow::anyhow!("reopen fresh prop page: {error}"))?
                .insert_bag(bag)
            {
                Ok(slot) => {
                    return Ok(Some(crate::property::BlobRef::new(*page_id, slot.0 + 1)));
                }
                Err(crate::records::PageError::Full { .. }) => self.flush_prop_page()?,
                Err(error) => bail!("insert fresh property bag: {error}"),
            }
        }
        let page_id = self.blob.allocate_shared_page_id();
        let mut image: Box<[u8; PAGE_SIZE]> = Box::new([0_u8; PAGE_SIZE]);
        let mut header = PageHeader::new(PageId::new(page_id), PageType::PropSlotted, self.tenant);
        header.lsn = self.frontier.raw();
        let mut page = SlottedPage::init(&mut image[..], header)
            .map_err(|error| anyhow::anyhow!("init fresh prop page: {error}"))?;
        let slot = page
            .insert_bag(bag)
            .map_err(|error| anyhow::anyhow!("insert fresh property bag: {error}"))?;
        self.open_prop_page = Some((page_id, image));
        Ok(Some(crate::property::BlobRef::new(page_id, slot.0 + 1)))
    }

    /// Flush the open shared slotted bag page into the STORE_PROPS
    /// extents. Page ids from the shared blob allocator are monotone, so
    /// successive flushes satisfy the extent writer's ordering contract.
    fn flush_prop_page(&mut self) -> Result<()> {
        if let Some((page_id, image)) = self.open_prop_page.take() {
            self.writers
                .props
                .write_page(page_id, image.as_ref(), self.frontier)?;
            self.report.prop_pages += 1;
        }
        Ok(())
    }

    pub fn push_node(&mut self, node: FreshNode<'_>) -> Result<()> {
        let FreshNode {
            id,
            label,
            external_id,
            bag,
            out_tel_ref,
            in_tel_ref,
        } = node;
        ensure!(
            self.last_node.is_none_or(|last| id > last),
            "fresh nodes are not id-sorted"
        );
        ensure!(
            id > 0 && id < OWNER_ALLOCATOR_MARKER_ID,
            "fresh node id is out of range"
        );
        ensure!(
            self.tel_chain.is_none(),
            "record emission with an unfinished TEL chain"
        );
        self.records_started = true;
        // #1519: `out_tel_ref`/`in_tel_ref` are opaque `encode_tel_ref`
        // (page_no, slot) pairs, not bare page numbers — decode before
        // bounds-checking against the page-id counter. `0` means "no
        // chain" and always passes (page ids are 1-based).
        let (out_page, _) = decode_tel_ref(out_tel_ref);
        let (in_page, _) = decode_tel_ref(in_tel_ref);
        ensure!(
            (out_tel_ref == 0 || out_page < self.next_tel_page)
                && (in_tel_ref == 0 || in_page < self.next_tel_page),
            "fresh node TEL ref names an unwritten TEL page"
        );
        let mut record = arcgraph_core::record::NodeRecord::new(
            arcgraph_core::NodeId::new(id),
            arcgraph_core::LabelId::new(label),
            self.frontier,
        );
        if let Some(blob_ref) = self.stage_props(bag)? {
            crate::property::encode_overflow_node(blob_ref, &mut record);
        }
        record.out_tel_ref = out_tel_ref;
        record.in_tel_ref = in_tel_ref;
        self.writers.nodes.write_node(&record, self.frontier)?;
        let logical = BindingOwnerValue {
            kind: 0,
            external_id: external_id.to_owned(),
            payload_hash: None,
            active: true,
        }
        .encode()?;
        self.writers.owners.write_logical(
            OwnerRowClass::NodeBinding,
            id,
            &logical,
            Some(str_hash_56(external_id)),
            self.frontier,
        )?;
        self.last_node = Some(id);
        self.report.nodes += 1;
        self.report.binding_rows += 1;
        Ok(())
    }

    pub fn push_relationship(&mut self, rel: FreshRel<'_>) -> Result<()> {
        let FreshRel {
            id,
            type_id,
            source_id,
            target_id,
            external_id,
            bag,
        } = rel;
        ensure!(
            self.last_rel.is_none_or(|last| id > last),
            "fresh relationships are not id-sorted"
        );
        ensure!(
            id > 0 && id < OWNER_ALLOCATOR_MARKER_ID,
            "fresh relationship id is out of range"
        );
        ensure!(
            self.tel_chain.is_none(),
            "record emission with an unfinished TEL chain"
        );
        self.records_started = true;
        let mut record = arcgraph_core::record::RelRecord::new(
            arcgraph_core::RelId::new(id),
            arcgraph_core::TypeId::new(type_id),
            arcgraph_core::NodeId::new(source_id),
            arcgraph_core::NodeId::new(target_id),
            self.frontier,
        );
        if let Some(blob_ref) = self.stage_props(bag)? {
            crate::property::encode_overflow_rel(blob_ref, &mut record);
        }
        self.writers.rels.write_rel(&record, self.frontier)?;
        let logical = BindingOwnerValue {
            kind: 1,
            external_id: external_id.to_owned(),
            payload_hash: None,
            active: true,
        }
        .encode()?;
        self.writers.owners.write_logical(
            OwnerRowClass::RelBinding,
            id,
            &logical,
            Some(str_hash_56(external_id)),
            self.frontier,
        )?;
        self.last_rel = Some(id);
        self.report.rels += 1;
        self.report.binding_rows += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<M4RewriteReport> {
        ensure!(
            self.tel_chain.is_none(),
            "fresh build finished with an unfinished TEL chain"
        );
        // #1519: any small blocks still sitting in the shared packing
        // buffer (the last page hadn't filled up) must reach disk — the
        // buffer spans across owner chains by design, so nothing else
        // flushes it once TEL streaming for both directions is done.
        let mut ctx = TelWriteCtx {
            tel: &mut self.writers.tel,
            next_tel_page: &mut self.next_tel_page,
            tenant: self.tenant,
            frontier: self.frontier,
        };
        self.tel_pack.flush(&mut ctx)?;
        self.flush_prop_page()?;
        self.report.extents = self.writers.finish(self.frontier)?;
        verify_dense_offsets_by_store(&self.report.extents)?;
        ensure!(
            self.report.tenants == BTreeSet::from([self.tenant]),
            "fresh tenant census drifted"
        );
        Ok(self.report)
    }
}

struct MigrationOwnerVisitor {
    interns: BoundedOwnerSorter,
    bindings: BoundedOwnerSorter,
    permissions: BoundedOwnerSorter,
}

impl MigrationOwnerVisitor {
    fn new(root: &Path, budget: Arc<OwnerRewriteScratchBudget>) -> Result<Self> {
        Ok(Self {
            interns: BoundedOwnerSorter::new(
                root,
                "interns",
                OWNER_REWRITE_RUN_BUFFER_BYTES,
                Arc::clone(&budget),
            )?,
            bindings: BoundedOwnerSorter::new(
                root,
                "bindings",
                OWNER_REWRITE_RUN_BUFFER_BYTES,
                Arc::clone(&budget),
            )?,
            permissions: BoundedOwnerSorter::new(
                root,
                "permissions",
                OWNER_REWRITE_RUN_BUFFER_BYTES,
                budget,
            )?,
        })
    }
}

impl crate::checkpoint::IncrementalOwnerVisitor for MigrationOwnerVisitor {
    fn intern(
        &mut self,
        tenant: TenantId,
        id: arcgraph_core::StringId,
        name: String,
    ) -> Result<(), crate::checkpoint::CheckpointError> {
        let name_len =
            u32::try_from(name.len()).map_err(|_| crate::checkpoint::CheckpointError::Corrupt {
                reason: "intern name exceeds u32 during M4 migration".to_owned(),
            })?;
        let mut record = Vec::with_capacity(20 + name.len());
        record.extend_from_slice(&tenant.raw().to_be_bytes());
        record.extend_from_slice(&u64::from(id.raw()).to_be_bytes());
        record.extend_from_slice(&name_len.to_be_bytes());
        record.extend_from_slice(name.as_bytes());
        self.interns
            .push(record)
            .map_err(checkpoint_owner_rewrite)?;
        Ok(())
    }

    fn idempotency(
        &mut self,
        tenant: TenantId,
        kind: u8,
        external_id: String,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) -> Result<(), crate::checkpoint::CheckpointError> {
        let class = match kind {
            0 => OwnerRowClass::NodeBinding,
            1 => OwnerRowClass::RelBinding,
            other => {
                return Err(crate::checkpoint::CheckpointError::Corrupt {
                    reason: format!(
                        "M4 migration cannot route idempotency kind {other} to the node/rel split"
                    ),
                });
            }
        };
        let external_len = u32::try_from(external_id.len()).map_err(|_| {
            crate::checkpoint::CheckpointError::Corrupt {
                reason: "idempotency external id exceeds u32 during M4 migration".to_owned(),
            }
        })?;
        let mut record = Vec::with_capacity(31 + external_id.len());
        record.extend_from_slice(&tenant.raw().to_be_bytes());
        record.push(class as u8);
        record.extend_from_slice(&internal_id.to_be_bytes());
        record.push(kind);
        record.push(u8::from(payload_hash.is_some()));
        record.extend_from_slice(&payload_hash.unwrap_or(0).to_be_bytes());
        record.extend_from_slice(&external_len.to_be_bytes());
        record.extend_from_slice(external_id.as_bytes());
        self.bindings
            .push(record)
            .map_err(checkpoint_owner_rewrite)?;
        Ok(())
    }

    fn permission(
        &mut self,
        tenant: TenantId,
        doc: arcgraph_core::NodeId,
        grants: BTreeSet<String>,
    ) -> Result<(), crate::checkpoint::CheckpointError> {
        let value = ClassOwnerValue { grants };
        let encoded =
            value
                .encode()
                .map_err(|error| crate::checkpoint::CheckpointError::Corrupt {
                    reason: format!("encode ACL class during M4 migration: {error}"),
                })?;
        let hash = value
            .hash()
            .map_err(|error| crate::checkpoint::CheckpointError::Corrupt {
                reason: format!("hash ACL class during M4 migration: {error}"),
            })?;
        let encoded_len = u32::try_from(encoded.len()).map_err(|_| {
            crate::checkpoint::CheckpointError::Corrupt {
                reason: "ACL class encoding exceeds u32 during M4 migration".to_owned(),
            }
        })?;
        let mut record = Vec::with_capacity(28 + encoded.len());
        record.extend_from_slice(&tenant.raw().to_be_bytes());
        record.extend_from_slice(&hash.to_be_bytes());
        record.extend_from_slice(&encoded_len.to_be_bytes());
        record.extend_from_slice(&encoded);
        record.extend_from_slice(&doc.raw().to_be_bytes());
        self.permissions
            .push(record)
            .map_err(checkpoint_owner_rewrite)?;
        Ok(())
    }
}

fn ensure_tenant_writers<'a>(
    writers: &'a mut BTreeMap<TenantId, TenantWriters>,
    destination: &Path,
    tenant: TenantId,
) -> Result<&'a mut TenantWriters> {
    match writers.entry(tenant) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => Ok(entry.insert(TenantWriters::create(destination, tenant)?)),
    }
}

fn rewrite_owner_metadata(
    source: &Path,
    destination: &Path,
    migration_lsn: Lsn,
    writers: &mut BTreeMap<TenantId, TenantWriters>,
    report: &mut M4RewriteReport,
) -> Result<bool> {
    let sidecar = crate::checkpoint::read_latest_sidecar(source)?
        .context("final v5 checkpoint sidecar is absent during owner rewrite")?;
    ensure!(
        sidecar.incremental_metadata
            && !sidecar.full_state_snapshot
            && sidecar.checkpoint_lsn == migration_lsn,
        "M4 owner rewrite source is not the final v5 incremental checkpoint"
    );
    let metadata_path = crate::checkpoint::incremental_metadata_path(
        source,
        migration_lsn,
        sidecar.metadata_generation,
    );
    // Compatibility for the repository's pre-M4 synthetic fixture: its
    // checkpoint consists only of the 48-byte header and explicitly denotes
    // an empty owner set. Real incremental checkpoints always carry the fixed
    // owner tags + CRC and take the fully validated streaming path below.
    if metadata_path.metadata()?.len() == 48 {
        report.owner_scratch_peak_bytes = 0;
        return Ok(false);
    }
    let scratch_root = destination.join(".owner-rewrite.tmp");
    fs::create_dir(&scratch_root)?;
    File::open(destination)?.sync_all()?;
    let budget = Arc::new(OwnerRewriteScratchBudget::new(
        OWNER_REWRITE_SCRATCH_CAP_BYTES,
    ));
    let mut visitor = MigrationOwnerVisitor::new(&scratch_root, Arc::clone(&budget))?;
    let metadata = crate::checkpoint::visit_incremental_metadata_owners(
        source,
        migration_lsn,
        sidecar.metadata_generation,
        &mut visitor,
    )?;
    ensure!(
        metadata.dpt.is_empty(),
        "final v5 owner rewrite DPT is not empty"
    );

    let MigrationOwnerVisitor {
        interns,
        bindings,
        permissions,
    } = visitor;

    let mut last_intern: Option<(TenantId, u64)> = None;
    let mut intern_high = BTreeMap::<TenantId, u64>::new();
    interns.finish_visit(|record| {
        let mut cursor = 0_usize;
        let tenant = TenantId::new(take_be_u64(record, &mut cursor, "intern tenant")?);
        let id = take_be_u64(record, &mut cursor, "intern id")?;
        let name_len = take_be_u32(record, &mut cursor, "intern name length")? as usize;
        let name = take_bytes(record, &mut cursor, name_len, "intern name")?;
        ensure_rewrite(cursor == record.len(), "intern run carries trailing bytes")?;
        ensure_rewrite(
            id > 0 && id < OWNER_ALLOCATOR_MARKER_ID,
            "intern id is out of range",
        )?;
        if last_intern == Some((tenant, id)) {
            return Err(OwnerRewriteError::Corrupt(
                "two intern rows share one direct id".to_owned(),
            ));
        }
        let name = std::str::from_utf8(name)
            .map_err(|_| OwnerRewriteError::Corrupt("intern run name is not UTF-8".to_owned()))?;
        let logical = InternOwnerValue {
            name: name.to_owned(),
        }
        .encode()
        .map_err(owner_rewrite_codec)?;
        ensure_tenant_writers(writers, destination, tenant)
            .map_err(owner_rewrite_anyhow)?
            .owners
            .write_logical(
                OwnerRowClass::InternedString,
                id,
                &logical,
                Some(str_hash_56(name)),
                migration_lsn,
            )
            .map_err(owner_rewrite_anyhow)?;
        intern_high
            .entry(tenant)
            .and_modify(|high| *high = (*high).max(id))
            .or_insert(id);
        last_intern = Some((tenant, id));
        report.intern_rows += 1;
        Ok(())
    })?;
    for (tenant, high_water) in intern_high {
        let marker = OwnerAllocatorMarker {
            kind: crate::wal::AllocatorKind::InternString.as_byte(),
            high_water,
        }
        .encode();
        ensure_tenant_writers(writers, destination, tenant)?
            .owners
            .write_logical(
                OwnerRowClass::InternedString,
                OWNER_ALLOCATOR_MARKER_ID,
                &marker,
                None,
                migration_lsn,
            )?;
    }

    let mut last_binding: Option<(TenantId, u8, u64)> = None;
    bindings.finish_visit(|record| {
        let mut cursor = 0_usize;
        let tenant = TenantId::new(take_be_u64(record, &mut cursor, "binding tenant")?);
        let class_byte = take_u8(record, &mut cursor, "binding class")?;
        let class = match class_byte {
            1 => OwnerRowClass::NodeBinding,
            2 => OwnerRowClass::RelBinding,
            _ => {
                return Err(OwnerRewriteError::Corrupt(
                    "binding run carries invalid owner class".to_owned(),
                ));
            }
        };
        let id = take_be_u64(record, &mut cursor, "binding id")?;
        let kind = take_u8(record, &mut cursor, "binding kind")?;
        let has_hash = take_u8(record, &mut cursor, "binding hash flag")?;
        let payload_hash = take_be_u64(record, &mut cursor, "binding payload hash")?;
        let external_len = take_be_u32(record, &mut cursor, "binding external length")? as usize;
        let external = take_bytes(record, &mut cursor, external_len, "binding external id")?;
        ensure_rewrite(cursor == record.len(), "binding run carries trailing bytes")?;
        ensure_rewrite(has_hash <= 1, "binding payload-hash flag is not boolean")?;
        let external = std::str::from_utf8(external).map_err(|_| {
            OwnerRewriteError::Corrupt("binding external id is not UTF-8".to_owned())
        })?;
        let key = (tenant, class_byte, id);
        if last_binding == Some(key) {
            return Err(OwnerRewriteError::Corrupt(
                "two bindings share one direct reverse id".to_owned(),
            ));
        }
        let logical = BindingOwnerValue {
            kind,
            external_id: external.to_owned(),
            payload_hash: (has_hash == 1).then_some(payload_hash),
            active: true,
        }
        .encode()
        .map_err(owner_rewrite_codec)?;
        ensure_tenant_writers(writers, destination, tenant)
            .map_err(owner_rewrite_anyhow)?
            .owners
            .write_logical(
                class,
                id,
                &logical,
                Some(str_hash_56(external)),
                migration_lsn,
            )
            .map_err(owner_rewrite_anyhow)?;
        last_binding = Some(key);
        report.binding_rows += 1;
        Ok(())
    })?;

    let mut grant_sorter = BoundedOwnerSorter::new(
        &scratch_root,
        "doc-grants",
        OWNER_REWRITE_RUN_BUFFER_BYTES,
        Arc::clone(&budget),
    )?;
    let mut current_class: Option<(TenantId, u64, Vec<u8>, u32)> = None;
    let mut next_class = BTreeMap::<TenantId, u32>::new();
    permissions.finish_visit(|record| {
        let mut cursor = 0_usize;
        let tenant = TenantId::new(take_be_u64(record, &mut cursor, "permission tenant")?);
        let hash = take_be_u64(record, &mut cursor, "permission class hash")?;
        let class_len = take_be_u32(record, &mut cursor, "permission class length")? as usize;
        let canonical = take_bytes(record, &mut cursor, class_len, "permission class")?.to_vec();
        let doc = take_be_u64(record, &mut cursor, "permission doc")?;
        ensure_rewrite(
            cursor == record.len(),
            "permission run carries trailing bytes",
        )?;
        let class_id = match &current_class {
            Some((current_tenant, current_hash, current_bytes, class_id))
                if *current_tenant == tenant
                    && *current_hash == hash
                    && *current_bytes == canonical =>
            {
                *class_id
            }
            _ => {
                let allocator = next_class.entry(tenant).or_insert(0);
                let id = *allocator;
                *allocator = allocator.checked_add(1).ok_or_else(|| {
                    OwnerRewriteError::Corrupt("ACL class id allocator overflow".to_owned())
                })?;
                ensure_rewrite(
                    u64::from(id) < OWNER_ALLOCATOR_MARKER_ID,
                    "ACL class id exceeds direct-row capacity",
                )?;
                ensure_tenant_writers(writers, destination, tenant)
                    .map_err(owner_rewrite_anyhow)?
                    .owners
                    .write_logical(
                        OwnerRowClass::ClassId,
                        u64::from(id),
                        &canonical,
                        Some(hash),
                        migration_lsn,
                    )
                    .map_err(owner_rewrite_anyhow)?;
                current_class = Some((tenant, hash, canonical.clone(), id));
                report.class_rows += 1;
                id
            }
        };
        let mut grant = Vec::with_capacity(20);
        grant.extend_from_slice(&tenant.raw().to_be_bytes());
        grant.extend_from_slice(&doc.to_be_bytes());
        grant.extend_from_slice(&class_id.to_be_bytes());
        grant_sorter.push(grant)?;
        Ok(())
    })?;
    for (tenant, next) in next_class {
        if next == 0 {
            continue;
        }
        let marker = OwnerAllocatorMarker {
            kind: crate::wal::AllocatorKind::AclClass.as_byte(),
            high_water: u64::from(next - 1),
        }
        .encode();
        ensure_tenant_writers(writers, destination, tenant)?
            .owners
            .write_logical(
                OwnerRowClass::ClassId,
                OWNER_ALLOCATOR_MARKER_ID,
                &marker,
                None,
                migration_lsn,
            )?;
    }

    let mut last_grant: Option<(TenantId, u64)> = None;
    grant_sorter.finish_visit(|record| {
        let mut cursor = 0_usize;
        let tenant = TenantId::new(take_be_u64(record, &mut cursor, "grant tenant")?);
        let doc = take_be_u64(record, &mut cursor, "grant doc")?;
        let class_id = take_be_u32(record, &mut cursor, "grant class id")?;
        ensure_rewrite(cursor == record.len(), "grant run carries trailing bytes")?;
        if last_grant == Some((tenant, doc)) {
            return Err(OwnerRewriteError::Corrupt(
                "two grant rows share one document id".to_owned(),
            ));
        }
        let logical = GrantOwnerValue {
            class_id,
            active: true,
        }
        .encode();
        ensure_tenant_writers(writers, destination, tenant)
            .map_err(owner_rewrite_anyhow)?
            .owners
            .write_logical(OwnerRowClass::Grant, doc, &logical, None, migration_lsn)
            .map_err(owner_rewrite_anyhow)?;
        last_grant = Some((tenant, doc));
        report.grant_rows += 1;
        Ok(())
    })?;

    fs::remove_dir(&scratch_root)?;
    File::open(destination)?.sync_all()?;
    report.owner_scratch_peak_bytes = budget.peak();
    ensure!(
        report.owner_scratch_peak_bytes <= OWNER_REWRITE_SCRATCH_CAP_BYTES,
        "owner rewrite exceeded its scratch cap"
    );
    Ok(true)
}

fn owner_rewrite_codec(error: crate::owner_row::OwnerRowError) -> OwnerRewriteError {
    OwnerRewriteError::Corrupt(format!("owner logical codec: {error}"))
}

fn checkpoint_owner_rewrite(error: OwnerRewriteError) -> crate::checkpoint::CheckpointError {
    crate::checkpoint::CheckpointError::Corrupt {
        reason: format!("bounded owner rewrite: {error}"),
    }
}

fn owner_rewrite_anyhow(error: anyhow::Error) -> OwnerRewriteError {
    OwnerRewriteError::Corrupt(error.to_string())
}

fn ensure_rewrite(condition: bool, reason: &str) -> Result<(), OwnerRewriteError> {
    if condition {
        Ok(())
    } else {
        Err(OwnerRewriteError::Corrupt(reason.to_owned()))
    }
}

fn take_u8(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u8, OwnerRewriteError> {
    let value = *bytes
        .get(*cursor)
        .ok_or_else(|| OwnerRewriteError::Corrupt(format!("{what} overruns run record")))?;
    *cursor += 1;
    Ok(value)
}

fn take_be_u32(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u32, OwnerRewriteError> {
    let field = take_bytes(bytes, cursor, 4, what)?;
    Ok(u32::from_be_bytes(field.try_into().map_err(|_| {
        OwnerRewriteError::Corrupt(format!("{what} is malformed"))
    })?))
}

fn take_be_u64(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u64, OwnerRewriteError> {
    let field = take_bytes(bytes, cursor, 8, what)?;
    Ok(u64::from_be_bytes(field.try_into().map_err(|_| {
        OwnerRewriteError::Corrupt(format!("{what} is malformed"))
    })?))
}

fn take_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8], OwnerRewriteError> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| OwnerRewriteError::Corrupt(format!("{what} length wraps")))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| OwnerRewriteError::Corrupt(format!("{what} overruns run record")))?;
    *cursor = end;
    Ok(value)
}

/// Rewrite the immutable final v5 base into a complete extent-backed v6 base.
pub fn rewrite_v5_generation(
    source: &Path,
    destination: &Path,
    migration_lsn: Lsn,
) -> Result<M4RewriteReport> {
    rewrite_v5_generation_at_frontier(
        source,
        destination,
        LoaderMigrationFrontier::new(migration_lsn)?,
    )
}

/// Independent M4-lite reference entry point. It consumes the same validated
/// frontier contract as M5 but retains its own source scan for the
/// `loader_vs_m4lite_differential` oracle.
pub fn rewrite_v5_generation_at_frontier(
    source: &Path,
    destination: &Path,
    frontier: LoaderMigrationFrontier,
) -> Result<M4RewriteReport> {
    let migration_lsn = frontier.migration_lsn();
    preserve_generation_artifacts(source, destination)?;
    let mut report = M4RewriteReport::default();
    let mut writers = BTreeMap::<TenantId, TenantWriters>::new();

    let tenants_root = source.join(M3_TENANTS_DIR);
    let tenant_entries = match fs::read_dir(&tenants_root) {
        Ok(entries) => entries.collect::<std::io::Result<Vec<_>>>()?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error).context("enumerate v5 tenant record stores"),
    };
    for entry in tenant_entries {
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let tenant_raw: u64 = entry
            .file_name()
            .to_str()
            .context("v5 tenant directory is not UTF-8")?
            .parse()
            .context("v5 tenant directory is not a numeric tenant id")?;
        let tenant = TenantId::new(tenant_raw);
        let record_path = entry.path().join(M3_RECORD_STORE_FILE);
        if !record_path.is_file() {
            continue;
        }
        report.tenants.insert(tenant);
        let tenant_writers = match writers.entry(tenant) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(TenantWriters::create(destination, tenant)?),
        };
        scan_page_file(&record_path, |_, bytes| {
            let view = SlottedPageRef::open(bytes.as_ref())?;
            ensure!(
                view.header().tenant_id == tenant.raw(),
                "v5 record page tenant differs from its tenant directory"
            );
            match PageType::from_byte(view.header().page_type)? {
                PageType::Node => {
                    for raw_slot in 0..view.slot_count() {
                        let slot = SlotId(raw_slot);
                        match view.read_node(slot)? {
                            Some(record) if record.is_visible_at(migration_lsn) => {
                                tenant_writers.nodes.write_node(&record, migration_lsn)?;
                                report.nodes += 1;
                            }
                            Some(record) => {
                                tenant_writers.nodes.tombstone(record.id, migration_lsn)?;
                            }
                            None => {
                                if let Some(record) = view.recover_tombstoned_node(slot)? {
                                    // Directory provenance outranks the stale body: a
                                    // v5 tombstone preserves bytes only for recovery and
                                    // must never be re-evaluated as a live MVCC record.
                                    tenant_writers.nodes.tombstone(record.id, migration_lsn)?;
                                }
                            }
                        }
                    }
                }
                PageType::Rel => {
                    for raw_slot in 0..view.slot_count() {
                        let slot = SlotId(raw_slot);
                        match view.read_rel(slot)? {
                            Some(record) if record.is_visible_at(migration_lsn) => {
                                tenant_writers.rels.write_rel(&record, migration_lsn)?;
                                report.rels += 1;
                            }
                            Some(record) => {
                                tenant_writers.rels.tombstone(record.id, migration_lsn)?;
                            }
                            None => {
                                if let Some(record) = view.recover_tombstoned_rel(slot)? {
                                    tenant_writers.rels.tombstone(record.id, migration_lsn)?;
                                }
                            }
                        }
                    }
                }
                other => bail!("v5 record.store contains non-record page {other:?}"),
            }
            Ok(())
        })?;
    }

    let props_path = source.join(M3_PROPS_STORE_FILE);
    if props_path.is_file() {
        scan_page_file(&props_path, |page_no, mut bytes| {
            let tenant = {
                let view = SlottedPageRef::open(bytes.as_ref())?;
                ensure!(
                    view.header().page_type == PageType::PropSlotted.as_byte(),
                    "v5 props.store contains a non-property page"
                );
                TenantId::new(view.header().tenant_id)
            };
            stamp_page_lsn(bytes.as_mut(), migration_lsn)?;
            report.tenants.insert(tenant);
            let tenant_writers = match writers.entry(tenant) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(TenantWriters::create(destination, tenant)?),
            };
            tenant_writers
                .props
                .write_page(page_no, bytes.as_ref(), migration_lsn)?;
            report.prop_pages += 1;
            Ok(())
        })?;
    }

    // ADR-229 owners 6–8: stream the final v5 metadata through bounded
    // external runs, build every direct owner row + forward candidate index,
    // and only then permit metadata capture retirement at the generation
    // boundary. No complete binding/name/grant map is materialized here.
    let owner_metadata_present = rewrite_owner_metadata(
        source,
        destination,
        migration_lsn,
        &mut writers,
        &mut report,
    )?;
    report.tenants.extend(writers.keys().copied());

    // Production bootstrap always registers the DEFAULT tenant's extent
    // owners, even when the migrated corpus contains only non-default
    // tenants. Make those empty artifacts part of the invisible build so the
    // first post-swap open never creates files inside the selected generation.
    report.tenants.insert(TenantId::DEFAULT);
    if let Entry::Vacant(entry) = writers.entry(TenantId::DEFAULT) {
        entry.insert(TenantWriters::create(destination, TenantId::DEFAULT)?);
    }
    for (_, tenant_writers) in writers {
        report.extents.extend(tenant_writers.finish(migration_lsn)?);
    }
    // INV-S3.11 simultaneity: only after every owner store, companion
    // payload, and forward index is durable do we publish a metadata anchor
    // with owners 6–8 retired (stable tags, zero counts).
    let sidecar = crate::checkpoint::read_latest_sidecar(source)?
        .context("v5 sidecar disappeared before owner capture retirement")?;
    if owner_metadata_present {
        let retired = crate::checkpoint::retire_incremental_lookup_owner_sections(
            source,
            destination,
            migration_lsn,
            sidecar.metadata_generation,
        )?;
        ensure!(
            retired.intern_names == report.intern_rows
                && retired.idempotency_bindings == report.binding_rows
                && retired.permission_docs == report.grant_rows,
            "retired owner counts disagree with page-backed rewrite counts"
        );
    }
    normalize_first_v6_checkpoint_generation(destination, migration_lsn)?;
    verify_dense_offsets_by_store(&report.extents)?;
    verify_complete_store_set(destination, &report.tenants)?;
    Ok(report)
}

/// M5 final-layout producer. `M4LiteReference` deliberately dispatches to the
/// pre-existing M4-lite producer above; the M5 arms use the independent
/// streaming scan below and meet only at the canonical page writers.
pub fn load_v5_generation(
    source: &Path,
    destination: &Path,
    frontier: LoaderMigrationFrontier,
    target: LoaderTarget,
) -> Result<M4RewriteReport> {
    match target {
        LoaderTarget::M4LiteReference => {
            rewrite_v5_generation_at_frontier(source, destination, frontier)
        }
        LoaderTarget::Fresh | LoaderTarget::ExistingRecluster => {
            load_v5_generation_m5(source, destination, frontier)
        }
    }
}

/// Independent M5 producer. Do not collapse this scan into
/// `rewrite_v5_generation_at_frontier`: that function is the differential's
/// M4-lite oracle. Both producers intentionally converge only at
/// `TenantWriters`, which owns the one canonical on-disk encoding.
fn load_v5_generation_m5(
    source: &Path,
    destination: &Path,
    frontier: LoaderMigrationFrontier,
) -> Result<M4RewriteReport> {
    let migration_lsn = frontier.migration_lsn();
    preserve_generation_artifacts(source, destination)?;
    let mut report = M4RewriteReport::default();
    let mut writers = BTreeMap::<TenantId, TenantWriters>::new();

    let tenants_root = source.join(M3_TENANTS_DIR);
    let tenant_entries = match fs::read_dir(&tenants_root) {
        Ok(entries) => entries.collect::<std::io::Result<Vec<_>>>()?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error).context("enumerate M5 tenant record stores"),
    };
    for entry in tenant_entries {
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let tenant = TenantId::new(
            entry
                .file_name()
                .to_str()
                .context("M5 tenant directory is not UTF-8")?
                .parse()
                .context("M5 tenant directory is not a numeric tenant id")?,
        );
        let record_path = entry.path().join(M3_RECORD_STORE_FILE);
        if !record_path.is_file() {
            continue;
        }
        report.tenants.insert(tenant);
        let tenant_writers = match writers.entry(tenant) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(TenantWriters::create(destination, tenant)?),
        };
        scan_page_file(&record_path, |_, bytes| {
            let view = SlottedPageRef::open(bytes.as_ref())?;
            ensure!(
                view.header().tenant_id == tenant.raw(),
                "M5 record page tenant differs from its tenant directory"
            );
            match PageType::from_byte(view.header().page_type)? {
                PageType::Node => {
                    for raw_slot in 0..view.slot_count() {
                        let slot = SlotId(raw_slot);
                        match view.read_node(slot)? {
                            Some(record) if record.is_visible_at(migration_lsn) => {
                                tenant_writers.nodes.write_node(&record, migration_lsn)?;
                                report.nodes += 1;
                            }
                            Some(record) => {
                                tenant_writers.nodes.tombstone(record.id, migration_lsn)?;
                            }
                            None => {
                                if let Some(record) = view.recover_tombstoned_node(slot)? {
                                    tenant_writers.nodes.tombstone(record.id, migration_lsn)?;
                                }
                            }
                        }
                    }
                }
                PageType::Rel => {
                    for raw_slot in 0..view.slot_count() {
                        let slot = SlotId(raw_slot);
                        match view.read_rel(slot)? {
                            Some(record) if record.is_visible_at(migration_lsn) => {
                                tenant_writers.rels.write_rel(&record, migration_lsn)?;
                                report.rels += 1;
                            }
                            Some(record) => {
                                tenant_writers.rels.tombstone(record.id, migration_lsn)?;
                            }
                            None => {
                                if let Some(record) = view.recover_tombstoned_rel(slot)? {
                                    tenant_writers.rels.tombstone(record.id, migration_lsn)?;
                                }
                            }
                        }
                    }
                }
                other => bail!("M5 record.store contains non-record page {other:?}"),
            }
            Ok(())
        })?;
    }

    // INV-M5.8 RED-on-revert control (M5-D2 gate 3): a loader-only
    // encoding divergence — here modeled as the M5 producer silently
    // dropping the property pages — MUST redden the byte differential
    // against the independent M4-lite producer. cfg-gated + bounded per
    // the standing test-hook rule; production builds compile it out.
    #[cfg(feature = "fault-injection")]
    let drop_loader_props = std::env::var_os("ARCGRAPH_M5_DROP_LOADER_PROPS").is_some();
    #[cfg(not(feature = "fault-injection"))]
    let drop_loader_props = false;
    let props_path = source.join(M3_PROPS_STORE_FILE);
    if props_path.is_file() && !drop_loader_props {
        scan_page_file(&props_path, |page_no, mut bytes| {
            let tenant = {
                let view = SlottedPageRef::open(bytes.as_ref())?;
                ensure!(
                    view.header().page_type == PageType::PropSlotted.as_byte(),
                    "M5 props.store contains a non-property page"
                );
                TenantId::new(view.header().tenant_id)
            };
            stamp_page_lsn(bytes.as_mut(), migration_lsn)?;
            report.tenants.insert(tenant);
            let tenant_writers = match writers.entry(tenant) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(TenantWriters::create(destination, tenant)?),
            };
            tenant_writers
                .props
                .write_page(page_no, bytes.as_ref(), migration_lsn)?;
            report.prop_pages += 1;
            Ok(())
        })?;
    }

    let owner_metadata_present = rewrite_owner_metadata(
        source,
        destination,
        migration_lsn,
        &mut writers,
        &mut report,
    )?;
    report.tenants.extend(writers.keys().copied());
    report.tenants.insert(TenantId::DEFAULT);
    if let Entry::Vacant(entry) = writers.entry(TenantId::DEFAULT) {
        entry.insert(TenantWriters::create(destination, TenantId::DEFAULT)?);
    }
    for (_, tenant_writers) in writers {
        report.extents.extend(tenant_writers.finish(migration_lsn)?);
    }
    let sidecar = crate::checkpoint::read_latest_sidecar(source)?
        .context("v5 sidecar disappeared before M5 owner capture retirement")?;
    if owner_metadata_present {
        let retired = crate::checkpoint::retire_incremental_lookup_owner_sections(
            source,
            destination,
            migration_lsn,
            sidecar.metadata_generation,
        )?;
        ensure!(
            retired.intern_names == report.intern_rows
                && retired.idempotency_bindings == report.binding_rows
                && retired.permission_docs == report.grant_rows,
            "M5 retired owner counts disagree with page-backed rewrite counts"
        );
    }
    normalize_first_v6_checkpoint_generation(destination, migration_lsn)?;
    verify_dense_offsets_by_store(&report.extents)?;
    verify_complete_store_set(destination, &report.tenants)?;
    Ok(report)
}

/// INV-M5.5 / M4 Slice-3a seam: pin the successor's first checkpoint at
/// metadata generation 1.
///
/// The invisible build inherits the source's final v5 sidecar and metadata
/// artifact by byte copy, so its `metadata_generation` is wherever the v5
/// store's per-directory counter happened to stop (the quiesce checkpoint
/// alone makes it ≥ 2 on any store that ever checkpointed before upgrade).
/// Deferred predecessor cleanup proves "the live v6 store has checkpointed
/// AFTER the swap" as `metadata_generation > 1`; that proof is sound only if
/// the migration checkpoint itself is always generation 1. Rename the
/// retained artifact, republish the sidecar, and prune strays — all inside
/// the not-yet-committed building directory, so a crash anywhere here is
/// discarded with the scratch tree on the next upgrade attempt.
fn normalize_first_v6_checkpoint_generation(destination: &Path, migration_lsn: Lsn) -> Result<()> {
    let sidecar = crate::checkpoint::read_latest_sidecar(destination)?
        .context("invisible v6 build lost its migration checkpoint sidecar")?;
    ensure!(
        sidecar.incremental_metadata
            && !sidecar.full_state_snapshot
            && sidecar.checkpoint_lsn == migration_lsn,
        "invisible v6 build sidecar is not the migration checkpoint"
    );
    if sidecar.metadata_generation != 1 {
        let from = crate::checkpoint::incremental_metadata_path(
            destination,
            migration_lsn,
            sidecar.metadata_generation,
        );
        let to = crate::checkpoint::incremental_metadata_path(destination, migration_lsn, 1);
        fs::rename(&from, &to).with_context(|| {
            format!("pin migration metadata {} at generation 1", from.display())
        })?;
        crate::checkpoint::write_sidecar_atomic(
            destination,
            &crate::checkpoint::CheckpointSidecar::incremental(
                migration_lsn,
                sidecar.snapshot_last_wal_lsn,
                sidecar.created_unix_ms,
                1,
            ),
        )?;
    }
    crate::checkpoint::prune_incremental_metadata(destination, migration_lsn, 1)?;
    Ok(())
}

fn stamp_page_lsn(bytes: &mut [u8; PAGE_SIZE], migration_lsn: Lsn) -> Result<()> {
    let mut page = SlottedPage::open(bytes)?;
    ensure!(
        page.page_lsn().raw() <= migration_lsn.raw(),
        "v5 page LSN exceeds migration frontier"
    );
    page.apply_redo_if_newer(migration_lsn, |_| Ok::<(), std::convert::Infallible>(()))
        .expect("infallible page-LSN stamp");
    Ok(())
}

fn scan_page_file(
    path: &Path,
    mut visit: impl FnMut(u64, Box<[u8; PAGE_SIZE]>) -> Result<()>,
) -> Result<()> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    ensure!(
        len % PAGE_SIZE as u64 == 0,
        "v5 page file is not page aligned"
    );
    for page_no in 0..len / PAGE_SIZE as u64 {
        let mut bytes = Box::new([0_u8; PAGE_SIZE]);
        file.read_exact(bytes.as_mut())?;
        if bytes.iter().all(|byte| *byte == 0) {
            continue;
        }
        visit(page_no, bytes)?;
    }
    Ok(())
}

fn preserve_generation_artifacts(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(
                "VERSION"
                    | "MANIFEST"
                    | "LSN_SEED"
                    | "wal"
                    | M3_PROPS_STORE_FILE
                    | M3_TENANTS_DIR
                    | PRODUCTION_EXTENT_SUBDIR
            )
        ) {
            continue;
        }
        copy_entry(&entry.path(), &destination.join(&name))?;
    }
    ensure!(
        destination.join("pages.db").is_file(),
        "v5 catalog root pages.db is missing"
    );
    Ok(())
}

fn copy_entry(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing migration source symlink"
    );
    if metadata.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            copy_entry(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else if metadata.is_file() {
        fs::copy(source, destination)?;
        File::open(destination)?.sync_all()?;
    } else {
        bail!("unsupported v5 generation artifact {}", source.display());
    }
    Ok(())
}

fn verify_dense_offsets(extents: &[RewrittenExtent]) -> Result<()> {
    let mut offsets = BTreeSet::new();
    for (index, extent) in extents.iter().enumerate() {
        ensure!(
            extent.physical_offset >= DIRECTORY_HEAD_BYTES,
            "v6 extent overlaps the reserved directory head"
        );
        ensure!(
            offsets.insert(extent.physical_offset),
            "two logical extents share physical offset {}",
            extent.physical_offset
        );
        ensure!(
            extent.physical_offset == DIRECTORY_HEAD_BYTES + index as u64 * EXTENT_BYTES,
            "v6 physical extent offsets are not dense append order"
        );
    }
    Ok(())
}

fn verify_dense_offsets_by_store(extents: &[RewrittenExtent]) -> Result<()> {
    let mut stores = BTreeMap::<(TenantId, u16), Vec<RewrittenExtent>>::new();
    for extent in extents {
        stores
            .entry((extent.tenant, extent.store_id))
            .or_default()
            .push(*extent);
    }
    for store_extents in stores.values() {
        verify_dense_offsets(store_extents)?;
    }
    Ok(())
}

/// Verify that every tenant has every production v6 store artifact.
pub fn verify_complete_store_set(generation: &Path, tenants: &BTreeSet<TenantId>) -> Result<()> {
    ensure!(
        generation.join("pages.db").is_file(),
        "v6 catalog root is missing"
    );
    for tenant in tenants {
        for store_id in M4_EXTENT_STORE_IDS {
            let path = production_extent_store_path(generation, *tenant, *store_id)
                .context("complete-set store id has no production path")?;
            ensure!(
                path.is_file(),
                "v6 store artifact is missing: {}",
                path.display()
            );
            read_extent_ledger(&path, *tenant, *store_id)
                .with_context(|| format!("verify v6 extent directory {}", path.display()))?;
        }
        for class in OwnerRowClass::ALL {
            let payload = owner_payload_path(generation, *tenant, class);
            ensure!(
                payload.is_file(),
                "v6 owner payload companion is missing: {}",
                payload.display()
            );
            // Bulk-only: `verify_complete_store_set` runs immediately after
            // a fresh v6 generation is built (migration / M5-D3 bulk
            // load — every call site is a post-build verification, never
            // the general serve path), so the companion may legally carry
            // a census-derived budget exceeding the incremental default.
            // `open_bulk` scopes the growth-above-published semantics to
            // exactly this path (FIX 2, M5-D3 / #1518 skeptic review); the
            // serve/recovery boot path (`OwnerRowRegistry::open_logical`)
            // keeps calling the absolute-ceiling `open`.
            OwnerPayloadStore::open_bulk(&payload, OWNER_PAYLOAD_DISK_CAP_BYTES)
                .with_context(|| format!("verify owner payload {}", payload.display()))?;
            if let Some(index) = owner_forward_index_path(generation, *tenant, class) {
                ensure!(
                    index.is_dir(),
                    "v6 owner forward index is missing: {}",
                    index.display()
                );
                OwnerForwardIndex::open_bulk(&index, OWNER_INDEX_DISK_CAP_BYTES)
                    .with_context(|| format!("verify owner index {}", index.display()))?;
            }
        }
    }
    Ok(())
}

/// Establish the first production incremental checkpoint for a fresh v6
/// generation (salvaged sound from PR #1504 per
/// `docs/design/M5D-REDESIGN-AMENDMENT.md` §9/§2.6 — it is the leg-(c)
/// loader's "first checkpoint seed"). This is the non-test counterpart of the
/// M4 40M RSS rung's checkpoint producer and uses the same
/// extent/owner/checkpointer types that normal bootstrap restores.
///
/// `blob` is the loader's shared property-bag store (M5-D2, amendment
/// §3.2): oversized-bag DEC-4 chain pages staged there by
/// [`FreshV6Builder::push_node`] / [`FreshV6Builder::push_relationship`]
/// are captured into this checkpoint's v9 incremental metadata as
/// page-images — the LANDED blob layout (blob.overflow store 5 is
/// page-image at the M4 format per the Director ruling recorded in
/// `checkpoint/snapshot.rs`) — and restored by cold open through
/// `read_incremental_metadata`, exactly like any incremental store's
/// chained blobs. A build with no oversized bags passes an empty store
/// and the section serializes zero pages (byte-identical to the D1
/// shape).
pub fn establish_fresh_v6_checkpoint(
    root: &Path,
    tenant: TenantId,
    frontier: Lsn,
    blob: &BlobStore,
) -> Result<crate::checkpoint::CheckpointSidecar> {
    use crate::checkpoint::{CheckpointSnapshot, incremental_checkpoint, read_latest_sidecar};
    use crate::io::{PageIo, PosixPageIo};

    let dpt = Arc::new(crate::redo::DirtyPageTable::new());
    let mut checkpointer =
        crate::checkpoint::WriteBehindCheckpointer::new_extent_only(Arc::clone(&dpt))
            .with_doublewrite_area(Arc::new(crate::checkpoint::DoublewriteArea::new(root)));
    let mut owner_stores = Vec::new();
    for store_id in M4_EXTENT_STORE_IDS {
        let path = production_extent_store_path(root, tenant, *store_id)
            .context("fresh checkpoint extent path")?;
        let physical: Arc<dyn PageIo> = Arc::new(PosixPageIo::open(path)?);
        let directory = Arc::new(crate::extent::ExtentDirectory::new(
            tenant, *store_id, physical, 16,
        ));
        let data = Arc::new(crate::extent::ExtentDataPageStore::new(
            Arc::clone(&directory),
            32,
        ));
        checkpointer = checkpointer
            .with_extent_directory_target(directory)
            .with_extent_data_target(Arc::clone(&data));
        owner_stores.push(Arc::new(crate::owner_row::OwnerRowStore::new(data)));
    }
    let owner = Arc::new(crate::owner_row::OwnerRowRegistry::open_logical(
        root,
        owner_stores,
        Arc::clone(&dpt),
    )?);
    let intern = crate::intern::InternTable::page_backed(Arc::clone(&owner))?;
    let idempotency = Arc::new(crate::idempotency::IdempotencyStore::page_backed(
        Arc::clone(&owner),
    ));
    let permissions = crate::permissions::PermissionIndex::page_backed(
        Arc::clone(&owner),
        Arc::clone(&idempotency),
        tenant,
    )?;
    let txn = Arc::new(TxnManager::new());
    txn.seed_after_replay(frontier);
    let primary = crate::primary_index::PrimaryPageStore::new();
    let records = crate::record_store::RecordPageStore::new();
    let allocator = Arc::new(crate::page_alloc::PageAllocator::new());
    let crud = Arc::new(crate::crud::CrudStore::new());
    let allocator_seed = crate::crud::crud_allocator_seed_handle(crud, allocator);
    let snapshot = CheckpointSnapshot {
        txn: &txn,
        primary_pages: &primary,
        record_pages: &records,
        blob,
        allocator_seed: allocator_seed.as_ref(),
        intern: &intern,
        idempotency: idempotency.as_ref(),
        permissions: &permissions,
        permissions_tenant: tenant,
    };
    let catalog_io = Arc::new(PosixPageIo::open_or_create(root.join("pages.db"))?);
    let report = incremental_checkpoint(
        root,
        &crate::buffer::BufferPool::new(1, catalog_io),
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )?;
    ensure!(
        report.checkpoint_lsn == frontier,
        "fresh checkpoint frontier changed"
    );
    let sidecar = read_latest_sidecar(root)?.context("fresh v6 checkpoint sidecar is absent")?;
    ensure!(
        sidecar.incremental_metadata
            && !sidecar.full_state_snapshot
            && sidecar.checkpoint_lsn == frontier,
        "fresh v6 checkpoint is not the production incremental shape"
    );
    Ok(sidecar)
}

/// Return the physical offset ledger decoded from one production extent file.
/// This is intentionally directory-derived; callers use it to recover the
/// next-physical counter after restart rather than resetting to the head base.
pub fn read_extent_ledger(
    path: &Path,
    tenant: TenantId,
    store_id: u16,
) -> Result<Vec<RewrittenExtent>> {
    let mut file = File::open(path)?;
    let mut extents = Vec::new();
    let head_pages =
        (file.metadata()?.len() / PAGE_SIZE as u64).min(DIRECTORY_HEAD_BYTES / PAGE_SIZE as u64);
    for page_index in 0..head_pages {
        let mut bytes = [0_u8; PAGE_SIZE];
        file.seek(SeekFrom::Start(page_index * PAGE_SIZE as u64))?;
        file.read_exact(&mut bytes)?;
        if bytes.iter().all(|byte| *byte == 0) {
            continue;
        }
        let header = PageHeader::from_bytes(
            bytes[..PageHeader::SIZE]
                .try_into()
                .expect("fixed directory header"),
        )?;
        ensure!(
            header.tenant_id == tenant.raw() && header.flags == store_id,
            "extent directory identity mismatch"
        );
        ensure!(
            crc32c::crc32c(&bytes[PageHeader::SIZE..]) == header.checksum,
            "extent directory checksum mismatch"
        );
        for slot in 0..DIRECTORY_ENTRIES_PER_PAGE {
            let offset = PageHeader::SIZE + slot as usize * DIRECTORY_ENTRY_BYTES;
            let entry = &bytes[offset..offset + DIRECTORY_ENTRY_BYTES];
            if entry.iter().all(|byte| *byte == 0) {
                continue;
            }
            let logical_extent = u64::from_le_bytes(entry[..8].try_into().unwrap());
            let physical_offset = u64::from_le_bytes(entry[8..16].try_into().unwrap());
            ensure!(
                u32::from_le_bytes(entry[20..24].try_into().unwrap()) == 1,
                "unsupported extent generation"
            );
            extents.push(RewrittenExtent {
                tenant,
                store_id,
                logical_extent,
                physical_offset,
            });
        }
    }
    extents.sort_by_key(|extent| extent.physical_offset);
    verify_dense_offsets(&extents)?;
    Ok(extents)
}

/// Recover the next dense physical offset from durable directory entries.
pub fn recover_next_physical_offset(path: &Path, tenant: TenantId, store_id: u16) -> Result<u64> {
    let extents = read_extent_ledger(path, tenant, store_id)?;
    Ok(extents.last().map_or(DIRECTORY_HEAD_BYTES, |extent| {
        extent.physical_offset + EXTENT_BYTES
    }))
}

/// Load the direct-addressed v6 base without consulting or rebuilding the
/// retired primary B-tree. WAL replay begins strictly above this frontier.
pub fn load_v6_physical_base(
    generation: &Path,
    checkpoint_lsn: Lsn,
    txn: &TxnManager,
    addressed: &AddressedRecordStore,
    blob: &BlobStore,
) -> Result<M4BaseLoadReport> {
    let mut report = M4BaseLoadReport::default();
    let tenants_root = generation.join(M3_TENANTS_DIR);
    for entry in fs::read_dir(&tenants_root).context("enumerate v6 tenants")? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(raw) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u64>().ok())
        else {
            continue;
        };
        let tenant = TenantId::new(raw);
        for (store_id, kind) in [
            (STORE_RECORD, Some(RecordKind::Node)),
            (STORE_RELS, Some(RecordKind::Rel)),
            (STORE_PROPS, None),
        ] {
            let path = production_extent_store_path(generation, tenant, store_id)
                .context("v6 base store has no production path")?;
            let ledger = read_extent_ledger(&path, tenant, store_id)?;
            let mut file = File::open(&path)?;
            let file_len = file.metadata()?.len();
            for extent in ledger {
                for within in 0..EXTENT_PAGES {
                    let page_no = extent.logical_extent * EXTENT_PAGES + within;
                    let mut bytes = Box::new([0_u8; PAGE_SIZE]);
                    let physical_page = extent.physical_offset + within * PAGE_SIZE as u64;
                    if physical_page + PAGE_SIZE as u64 > file_len {
                        continue;
                    }
                    file.seek(SeekFrom::Start(physical_page))?;
                    file.read_exact(bytes.as_mut())?;
                    if bytes.iter().all(|byte| *byte == 0) {
                        continue;
                    }
                    let view = SlottedPageRef::open(bytes.as_ref())?;
                    let page_lsn = view.page_lsn();
                    // M4 homes are physiological write-through stores. A
                    // strict commit may therefore reach its home page after
                    // WAL fsync but before the next checkpoint sidecar. That
                    // page is a valid redo-covered image, even when its LSN
                    // is above `checkpoint_lsn`; incremental replay starts at
                    // the DPT redo floor and treats an already-newer home as
                    // idempotent coverage. Reject only malformed identity or
                    // an unstamped non-zero page here.
                    ensure!(
                        view.header().page_id == page_no
                            && view.header().tenant_id == tenant.raw()
                            && page_lsn != Lsn::ZERO,
                        "v6 base page identity/LSN mismatch"
                    );
                    match kind {
                        Some(RecordKind::Node) => {
                            ensure!(
                                view.header().page_type == PageType::Node.as_byte(),
                                "nodes.store contains a non-node page"
                            );
                            for (_, record) in view.iter_nodes() {
                                addressed.write_node(tenant, &record)?;
                                txn.apply_replay_mvcc_write(
                                    Lsn::new(record.created_lsn),
                                    tenant,
                                    node_mvcc_key(arcgraph_core::NodeId::new(record.id)),
                                    Some(bytes::Bytes::copy_from_slice(&record.to_bytes())),
                                );
                                report.nodes += 1;
                            }
                            for raw_slot in 0..view.slot_count() {
                                let slot = SlotId(raw_slot);
                                if !view.is_permanent_tombstone(slot)? {
                                    continue;
                                }
                                let id = page_no
                                    .checked_mul(u64::from(NODE_CAPACITY))
                                    .and_then(|base| base.checked_add(u64::from(raw_slot)))
                                    .context("node tombstone id overflow")?;
                                if id != 0 {
                                    addressed.tombstone_node_at_lsn(
                                        tenant,
                                        arcgraph_core::NodeId::new(id),
                                        page_lsn,
                                    )?;
                                }
                            }
                            report.record_pages += 1;
                        }
                        Some(RecordKind::Rel) => {
                            ensure!(
                                view.header().page_type == PageType::Rel.as_byte(),
                                "rels.store contains a non-relationship page"
                            );
                            for (_, record) in view.iter_rels() {
                                addressed.write_rel(tenant, &record)?;
                                txn.apply_replay_mvcc_write(
                                    Lsn::new(record.created_lsn),
                                    tenant,
                                    rel_mvcc_key(arcgraph_core::RelId::new(record.id)),
                                    Some(bytes::Bytes::copy_from_slice(&record.to_bytes())),
                                );
                                report.rels += 1;
                            }
                            for raw_slot in 0..view.slot_count() {
                                let slot = SlotId(raw_slot);
                                if !view.is_permanent_tombstone(slot)? {
                                    continue;
                                }
                                let id = page_no
                                    .checked_mul(u64::from(REL_CAPACITY))
                                    .and_then(|base| base.checked_add(u64::from(raw_slot)))
                                    .context("relationship tombstone id overflow")?;
                                if id != 0 {
                                    addressed.tombstone_rel_at_lsn(
                                        tenant,
                                        arcgraph_core::RelId::new(id),
                                        page_lsn,
                                    )?;
                                }
                            }
                            report.record_pages += 1;
                        }
                        None => {
                            ensure!(
                                view.header().page_type == PageType::PropSlotted.as_byte(),
                                "props.store contains a non-property page"
                            );
                            blob.install_m3_base_page(tenant, PageId::new(page_no), bytes)?;
                            report.prop_pages += 1;
                        }
                    }
                }
            }
        }
    }
    txn.seed_after_replay(checkpoint_lsn);
    Ok(report)
}

#[cfg(all(test, unix))]
mod owner_rss_gate {
    use std::os::unix::fs::MetadataExt;
    use std::path::PathBuf;

    use arcgraph_core::{NodeId, RelId, TypeId};

    use super::*;
    use crate::DirtyPageTable;
    use crate::buffer::BufferPool;
    use crate::checkpoint::{
        CheckpointSnapshot, DoublewriteArea, WriteBehindCheckpointer, incremental_checkpoint,
        read_latest_sidecar,
    };
    use crate::crud::{CrudStore, crud_allocator_seed_handle};
    use crate::extent::{ExtentDataPageStore, ExtentDirectory};
    use crate::idempotency::IdempotencyStore;
    use crate::intern::InternTable;
    use crate::io::{PageIo, PosixPageIo};
    use crate::owner_row::{OwnerRowRegistry, OwnerRowStore};
    use crate::page_alloc::PageAllocator;
    use crate::permissions::PermissionIndex;
    use crate::primary_index::PrimaryPageStore;
    use crate::record_store::RecordPageStore;
    use crate::redo::DeltaPageStore;
    use crate::wal::{STORE_GRANTS, STORE_INTERN, STORE_NODE_BINDINGS, STORE_REL_BINDINGS};

    const SAMPLE_EVERY: u64 = 1_000_000;
    const RSS_CAP_BYTES: u64 = 40 * 1024 * 1024 * 1024;
    const DISK_CAP_BYTES: u64 = 20 * 1024 * 1024 * 1024;
    const PLATEAU_RANGE_BYTES: u64 = 256 * 1024 * 1024;
    const RESIDENT_REVERT_ENV: &str = "ARCGRAPH_OWNER_RSS_REVERT_RESIDENT_MAP";

    fn rss_bytes() -> std::io::Result<u64> {
        let status = fs::read_to_string("/proc/self/status")?;
        let line = status
            .lines()
            .find(|line| line.starts_with("VmRSS:"))
            .ok_or_else(|| std::io::Error::other("/proc/self/status has no VmRSS"))?;
        let kib = line
            .split_ascii_whitespace()
            .nth(1)
            .ok_or_else(|| std::io::Error::other("VmRSS has no numeric field"))?
            .parse::<u64>()
            .map_err(|error| std::io::Error::other(format!("parse VmRSS: {error}")))?;
        kib.checked_mul(1024)
            .ok_or_else(|| std::io::Error::other("VmRSS byte count overflow"))
    }

    fn allocated_bytes(path: &Path) -> std::io::Result<u64> {
        let mut total = 0_u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                total = total.saturating_add(allocated_bytes(&entry.path())?);
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.blocks().saturating_mul(512));
            }
        }
        Ok(total)
    }

    fn relationship(id: u64) -> arcgraph_core::record::RelRecord {
        arcgraph_core::record::RelRecord::new(
            RelId::new(id),
            TypeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            Lsn::new(100),
        )
    }

    fn open_owner_registry(root: &Path, tenant: TenantId) -> Result<Arc<OwnerRowRegistry>> {
        let stores = [
            STORE_NODE_BINDINGS,
            STORE_REL_BINDINGS,
            STORE_INTERN,
            STORE_GRANTS,
        ]
        .into_iter()
        .map(|store_id| -> Result<Arc<OwnerRowStore>> {
            let path = production_extent_store_path(root, tenant, store_id)
                .context("owner RSS gate store path")?;
            let physical: Arc<dyn PageIo> = Arc::new(PosixPageIo::open(path)?);
            let directory = Arc::new(ExtentDirectory::new(tenant, store_id, physical, 16));
            Ok(Arc::new(OwnerRowStore::new(Arc::new(
                ExtentDataPageStore::new(directory, 32),
            ))))
        })
        .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(OwnerRowRegistry::open_logical(
            root,
            stores,
            Arc::new(DirtyPageTable::new()),
        )?))
    }

    fn establish_production_checkpoint(
        root: &Path,
        tenant: TenantId,
        frontier: Lsn,
    ) -> Result<Lsn> {
        let dpt = Arc::new(DirtyPageTable::new());
        let mut checkpointer = WriteBehindCheckpointer::new_extent_only(Arc::clone(&dpt))
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(root)));
        let mut owner_stores = Vec::new();
        for store_id in M4_EXTENT_STORE_IDS {
            let path = production_extent_store_path(root, tenant, *store_id)
                .context("checkpoint RSS gate store path")?;
            let physical: Arc<dyn PageIo> = Arc::new(PosixPageIo::open(path)?);
            let directory = Arc::new(ExtentDirectory::new(tenant, *store_id, physical, 16));
            let data = Arc::new(ExtentDataPageStore::new(Arc::clone(&directory), 32));
            checkpointer = checkpointer
                .with_extent_directory_target(directory)
                .with_extent_data_target(Arc::clone(&data));
            owner_stores.push(Arc::new(OwnerRowStore::new(data)));
        }
        let owner = Arc::new(OwnerRowRegistry::open_logical(
            root,
            owner_stores,
            Arc::clone(&dpt),
        )?);
        let intern = InternTable::page_backed(Arc::clone(&owner))?;
        let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&owner)));
        let permissions =
            PermissionIndex::page_backed(Arc::clone(&owner), Arc::clone(&idempotency), tenant)?;

        let txn = Arc::new(TxnManager::new());
        txn.seed_after_replay(frontier);
        let primary = PrimaryPageStore::new();
        let records = RecordPageStore::new();
        let blob = BlobStore::new();
        let allocator = Arc::new(PageAllocator::new());
        let crud = Arc::new(CrudStore::new());
        let allocator_seed = crud_allocator_seed_handle(crud, allocator);
        let snapshot = CheckpointSnapshot {
            txn: &txn,
            primary_pages: &primary,
            record_pages: &records,
            blob: &blob,
            allocator_seed: allocator_seed.as_ref(),
            intern: &intern,
            idempotency: idempotency.as_ref(),
            permissions: &permissions,
            permissions_tenant: tenant,
        };
        let catalog_io = Arc::new(PosixPageIo::open_or_create(root.join("pages.db"))?);
        let report = incremental_checkpoint(
            root,
            &BufferPool::new(1, catalog_io),
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        )?;
        ensure!(
            report.checkpoint_lsn == frontier,
            "RSS checkpoint frontier changed"
        );
        let sidecar = read_latest_sidecar(root)?
            .context("production incremental checkpoint did not establish its sidecar")?;
        ensure!(
            sidecar.incremental_metadata
                && !sidecar.full_state_snapshot
                && sidecar.checkpoint_lsn == report.checkpoint_lsn,
            "RSS checkpoint sidecar does not name the physical frontier"
        );
        Ok(report.checkpoint_lsn)
    }

    fn finish_empty_production_stores(root: &Path, tenant: TenantId, frontier: Lsn) -> Result<()> {
        for store_id in [
            STORE_PROPS,
            STORE_RECORD,
            STORE_TEL,
            STORE_SECONDARY_INDEX,
            STORE_BLOB_OVERFLOW,
        ] {
            ExtentStoreWriter::create(root, tenant, store_id)?.finish(frontier)?;
        }
        Ok(())
    }

    fn served_bytes(root: &Path, tenant: TenantId, count: u64) -> Result<Vec<u8>> {
        let registry = open_owner_registry(root, tenant)?;
        let idempotency = IdempotencyStore::page_backed(Arc::clone(&registry));
        let rel_path = production_extent_store_path(root, tenant, STORE_RELS)
            .context("relationship RSS gate store path")?;
        let physical: Arc<dyn PageIo> = Arc::new(PosixPageIo::open(rel_path)?);
        let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RELS, physical, 16));
        let rel_store = ExtentDataPageStore::new(directory, 32);
        let mut served = Vec::new();
        let ids = [1, count / 3, count / 2, count.saturating_sub(1), count];
        for id in ids {
            let external_id = format!("rel-{id:010}");
            let binding = idempotency
                .try_get(tenant, 1, &external_id)?
                .with_context(|| format!("binding {external_id} was not served"))?;
            ensure!(
                binding.internal_id == id,
                "binding resolved to wrong relation"
            );
            let reverse = idempotency
                .try_external_id_for(tenant, 1, id)?
                .with_context(|| format!("reverse binding {id} was not served"))?;
            ensure!(reverse == external_id, "reverse binding bytes changed");

            let (page_no, slot) = RecordKind::Rel.address(id)?;
            let page = rel_store
                .read_page_for_redo(tenant, PageId::new(page_no))?
                .with_context(|| format!("relationship page {page_no} was not served"))?;
            let record = SlottedPageRef::open(page.as_ref())?
                .read_rel(SlotId(slot))?
                .with_context(|| format!("relationship {id} was not served"))?;
            ensure!(record == relationship(id), "relationship bytes changed");
            served.extend_from_slice(&record.to_bytes());
            served.extend_from_slice(reverse.as_bytes());
        }
        ensure!(
            idempotency.resident_len() == 0,
            "forward DashMap became resident"
        );
        ensure!(
            idempotency.resident_reverse_len() == 0,
            "reverse DashMap became resident"
        );
        Ok(served)
    }

    /// Linux proving ground for the #1404 owner-residency close. The caller
    /// must set `ARCGRAPH_OWNER_RSS_RELATIONSHIPS` to exactly 10M or 40M and
    /// `ARCGRAPH_OWNER_RSS_ROOT` to gate-owned scratch. The permanent store
    /// and all index merge intermediates share a measured 20 GiB disk cap.
    #[test]
    #[ignore = "OCI VM only: real 10M/40M relationship owner RSS rung"]
    fn owner_residency_linux_rss_rung() -> Result<()> {
        ensure!(
            cfg!(target_os = "linux"),
            "owner RSS rung is admissible only on Linux"
        );
        let count = std::env::var("ARCGRAPH_OWNER_RSS_RELATIONSHIPS")
            .context("ARCGRAPH_OWNER_RSS_RELATIONSHIPS is required")?
            .parse::<u64>()
            .context("parse ARCGRAPH_OWNER_RSS_RELATIONSHIPS")?;
        ensure!(
            matches!(count, 10_000_000 | 40_000_000),
            "RSS rung must be exactly 10M or 40M relationships"
        );
        let parent = PathBuf::from(
            std::env::var_os("ARCGRAPH_OWNER_RSS_ROOT")
                .context("ARCGRAPH_OWNER_RSS_ROOT is required")?,
        );
        fs::create_dir_all(&parent)?;
        let root = parent.join(format!("rung-{count}"));
        fs::create_dir(&root)
            .with_context(|| format!("create clean rung at {}", root.display()))?;
        let tenant = TenantId::new(0x5a17);
        let migration_lsn = Lsn::new(100);
        let mut rels = DirectRecordWriter::new(&root, tenant, RecordKind::Rel)?;
        let mut owners = OwnerWriters::create(&root, tenant, None)?;
        let mut curve = Vec::new();
        // Test-only negative control. Read once before the loop (never per
        // row); the workload itself is bounded to the exact 10M/40M rung.
        // Keeping the production-shaped resident DashMap alive through the
        // plateau census must make the RSS slope oracle bite.
        let resident_revert = std::env::var_os(RESIDENT_REVERT_ENV).is_some();
        let reverted_owner = resident_revert.then(dashmap::DashMap::<String, u64>::new);

        for id in 1..=count {
            rels.write_rel(&relationship(id), migration_lsn)?;
            let external_id = format!("rel-{id:010}");
            if let Some(resident) = &reverted_owner {
                resident.insert(external_id.clone(), id);
            }
            let logical = BindingOwnerValue {
                kind: 1,
                external_id: external_id.clone(),
                payload_hash: Some(id),
                active: true,
            }
            .encode()?;
            owners.write_logical(
                OwnerRowClass::RelBinding,
                id,
                &logical,
                Some(str_hash_56(&external_id)),
                migration_lsn,
            )?;
            if id % SAMPLE_EVERY == 0 {
                let rss = rss_bytes()?;
                let disk = allocated_bytes(&root)?;
                ensure!(rss <= RSS_CAP_BYTES, "40 GiB RSS cap exceeded at N={id}");
                ensure!(disk <= DISK_CAP_BYTES, "20 GiB disk cap exceeded at N={id}");
                println!(
                    "OWNER_RSS_SAMPLE platform=OCI_VM n={id} rss_bytes={rss} allocated_bytes={disk} resident_revert={resident_revert}"
                );
                curve.push((id, rss));
            }
        }
        rels.finish(migration_lsn)?;
        owners.finish(migration_lsn)?;
        finish_empty_production_stores(&root, tenant, migration_lsn)?;
        let checkpoint_lsn = establish_production_checkpoint(&root, tenant, migration_lsn)?;
        let disk = allocated_bytes(&root)?;
        ensure!(disk <= DISK_CAP_BYTES, "20 GiB final disk cap exceeded");
        println!(
            "OWNER_RSS_CHECKPOINT platform=OCI_VM n={count} checkpoint_lsn={} allocated_bytes={disk} cap_bytes={DISK_CAP_BYTES}",
            checkpoint_lsn.raw()
        );

        let steady_floor = count / 2;
        let steady: Vec<_> = curve.iter().filter(|(n, _)| *n >= steady_floor).collect();
        ensure!(
            steady.len() >= 5,
            "RSS curve has too few steady-state samples"
        );
        let minimum = steady.iter().map(|(_, rss)| *rss).min().unwrap_or(0);
        let maximum = steady.iter().map(|(_, rss)| *rss).max().unwrap_or(u64::MAX);
        ensure!(
            maximum.saturating_sub(minimum) <= PLATEAU_RANGE_BYTES,
            "second-half RSS range exceeds 256 MiB plateau bound"
        );

        let before_restart = served_bytes(&root, tenant, count)?;
        let after_restart = served_bytes(&root, tenant, count)?;
        ensure!(
            before_restart == after_restart,
            "served relationship/binding bytes changed across cold restart"
        );
        let final_rss = rss_bytes()?;
        ensure!(
            final_rss <= RSS_CAP_BYTES,
            "40 GiB restart RSS cap exceeded"
        );
        println!(
            "OWNER_RSS_RESULT platform=OCI_VM n={count} plateau_min_bytes={minimum} plateau_max_bytes={maximum} final_rss_bytes={final_rss} byte_restart=identical no_oom=true"
        );
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn owner_rss_checkpoint_smoke_uses_real_store_producer() -> Result<()> {
        const COUNT: u64 = 1_200;
        let parent = tempfile::tempdir()?;
        let root = parent.path().join("checkpoint-smoke");
        fs::create_dir(&root)?;
        let tenant = TenantId::new(0x5a17);
        let frontier = Lsn::new(100);
        let mut rels = DirectRecordWriter::new(&root, tenant, RecordKind::Rel)?;
        let mut owners = OwnerWriters::create(&root, tenant, None)?;
        for id in 1..=COUNT {
            rels.write_rel(&relationship(id), frontier)?;
            let external_id = format!("rel-{id:010}");
            owners.write_logical(
                OwnerRowClass::RelBinding,
                id,
                &BindingOwnerValue {
                    kind: 1,
                    external_id: external_id.clone(),
                    payload_hash: Some(id),
                    active: true,
                }
                .encode()?,
                Some(str_hash_56(&external_id)),
                frontier,
            )?;
        }
        rels.finish(frontier)?;
        owners.finish(frontier)?;
        finish_empty_production_stores(&root, tenant, frontier)?;
        assert_eq!(
            establish_production_checkpoint(&root, tenant, frontier)?,
            frontier
        );
        assert_eq!(
            served_bytes(&root, tenant, COUNT)?,
            served_bytes(&root, tenant, COUNT)?
        );
        Ok(())
    }
}

/// Re-publish WAL-recovered post-migration MVCC heads into their arithmetic
/// slots. Group A deliberately retains the first v6 checkpoint and fresh WAL
/// until Group B lands the v6 incremental producer, so this bounded recovery
/// pass closes the crash window without consulting the retired primary tree.
///
/// The version's `created_lsn` is authoritative here. On the v8 /
/// non-delta path, a live record decoded from the MVCC payload can carry
/// the canonical `Lsn::ZERO` placeholder even though its version is
/// committed at a non-zero LSN. Refuse a zero version LSN and stamp the
/// live record from `state_lsn`, matching tombstone handling and preventing
/// the #1616 visibility/replay defect from being reintroduced through this
/// sibling MVCC-to-physical rebuild.
pub fn rebuild_addressed_from_mvcc(
    txn: &TxnManager,
    addressed: &AddressedRecordStore,
    frontier: Lsn,
) -> Result<M4BaseLoadReport> {
    let mut report = M4BaseLoadReport::default();
    let mut first_error = None;
    for tenant in txn.tenants_with_chains() {
        txn.for_each_visible_record_state(
            tenant,
            frontier,
            |key, value, state_lsn, previous_value| {
                if first_error.is_some() {
                    return;
                }
                let tombstone = value.is_none();
                let Some(bytes) = value.or(previous_value) else {
                    return;
                };
                if state_lsn == Lsn::ZERO {
                    first_error = Some(anyhow::anyhow!(
                        "tenant {} key {key} has a zero MVCC created_lsn; refusing to publish a \
                         direct-addressed row with a fabricated visibility LSN (issue #1616)",
                        tenant.raw(),
                    ));
                    return;
                }
                let result = if key & crate::crud::REL_TAG_BIT == 0
                    && bytes.len() == arcgraph_core::record::NodeRecord::SIZE
                {
                    let bytes: &[u8; arcgraph_core::record::NodeRecord::SIZE] =
                        bytes.try_into().expect("length checked above");
                    arcgraph_core::record::NodeRecord::from_bytes(bytes)
                        .map_err(anyhow::Error::from)
                        .and_then(|mut record| {
                            let id = arcgraph_core::NodeId::new(record.id);
                            ensure!(
                                key == node_mvcc_key(id),
                                "node MVCC key differs from recovered record id"
                            );
                            if tombstone {
                                addressed.tombstone_node_at_lsn(tenant, id, state_lsn)?;
                            } else {
                                record.created_lsn = state_lsn.raw();
                                addressed.write_node(tenant, &record)?;
                                report.nodes += 1;
                            }
                            Ok(())
                        })
                } else if key & crate::crud::REL_TAG_BIT != 0
                    && bytes.len() == arcgraph_core::record::RelRecord::SIZE
                {
                    let bytes: &[u8; arcgraph_core::record::RelRecord::SIZE] =
                        bytes.try_into().expect("length checked above");
                    arcgraph_core::record::RelRecord::from_bytes(bytes)
                        .map_err(anyhow::Error::from)
                        .and_then(|mut record| {
                            let id = arcgraph_core::RelId::new(record.id);
                            ensure!(
                                key == rel_mvcc_key(id),
                                "relationship MVCC key differs from recovered record id"
                            );
                            if tombstone {
                                addressed.tombstone_rel_at_lsn(tenant, id, state_lsn)?;
                            } else {
                                record.created_lsn = state_lsn.raw();
                                addressed.write_rel(tenant, &record)?;
                                report.rels += 1;
                            }
                            Ok(())
                        })
                } else {
                    Ok(())
                };
                if let Err(error) = result {
                    first_error = Some(error);
                }
            },
        );
    }
    if let Some(error) = first_error {
        return Err(error.context("rebuild direct-address authority from v6 WAL"));
    }
    Ok(report)
}
