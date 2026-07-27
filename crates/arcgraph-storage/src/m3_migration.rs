//! One-time v4 full-checkpoint -> v5 physical-generation translator.
//!
//! This decoder is migration-only: normal v9 recovery consumes the physical
//! `props.store` / `record.store` base plus incremental metadata and never
//! learns the v8 page-image WAL format.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};

use crate::WriteBehindCheckpointer;
use crate::blob::{BlobStore, BlobStoreHandle};
use crate::buffer::BufferPool;
use crate::checkpoint::{
    CheckpointError, CheckpointSnapshot, DoublewriteArea, incremental_checkpoint,
    read_latest_sidecar, restore_latest_checkpoint,
};
use crate::crud::{CrudStore, crud_allocator_seed_handle, node_mvcc_key, rel_mvcc_key};
use crate::idempotency::IdempotencyStore;
use crate::intern::InternTable;
use crate::io::InMemoryPageIo;
use crate::page_alloc::PageAllocator;
use crate::page_store::TenantFilePageIo;
use crate::page_store::{BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig};
use crate::permissions::PermissionIndex;
use crate::primary_index::{
    BootstrapStats, PageSlot, PrimaryIndex, PrimaryKey, PrimaryPageStore, RecordKind,
};
use crate::record_store::RecordPageStore;
use crate::records::{SlottedPage, SlottedPageRef};
use crate::redo::DirtyPageTable;
use crate::transaction::TxnManager;
use crate::wal::AllocatorSeedHandle;

/// Physical M3 property-page file.
pub const M3_PROPS_STORE_FILE: &str = "props.store";
/// Physical M3 record-page file.
pub const M3_RECORD_STORE_FILE: &str = "record.store";
/// Directory containing tenant-qualified M3 record homes.
pub const M3_TENANTS_DIR: &str = "tenants";

#[must_use]
pub fn m3_record_store_path(generation: &Path, tenant: TenantId) -> std::path::PathBuf {
    generation
        .join(M3_TENANTS_DIR)
        .join(tenant.raw().to_string())
        .join(M3_RECORD_STORE_FILE)
}

/// Counts from the one-time translator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M3TranslationReport {
    pub migration_lsn: Lsn,
    pub prop_pages: u64,
    pub record_pages: u64,
    pub overflow_pages: u64,
}

/// Counts observed while opening an M3 physical base without whole-owner
/// capture. Only one 8 KiB page is resident in the loader itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M3BaseLoadReport {
    pub prop_pages: u64,
    pub record_pages: u64,
    pub nodes: u64,
    pub rels: u64,
}

/// Register the checkpointed record base in the bounded page tier, rebuild
/// the post-restart MVCC head set from the authoritative record bytes, and
/// load checkpointed property pages through the bounded blob tier.
pub fn load_v9_physical_base(
    generation: &Path,
    checkpoint_lsn: Lsn,
    txn: &TxnManager,
    records: &BufferedRecordPageStore,
    blob: &BlobStore,
) -> Result<M3BaseLoadReport, CheckpointError> {
    let mut report = M3BaseLoadReport {
        prop_pages: 0,
        record_pages: 0,
        nodes: 0,
        rels: 0,
    };
    let record_files = TenantFilePageIo::new(generation, M3_RECORD_STORE_FILE);
    let tenants = record_files
        .existing_tenants()
        .map_err(|error| corrupt(format!("enumerate tenant record stores: {error}")))?;
    for home_tenant in tenants {
        let mut record_file = std::fs::File::open(record_files.path_for(home_tenant))?;
        scan_page_file(&mut record_file, |page_id, page| {
            let view = SlottedPageRef::open(page.as_ref()).map_err(|error| {
                corrupt(format!(
                    "record.store page {} is invalid: {error}",
                    page_id.raw()
                ))
            })?;
            let tenant = TenantId::new(view.header().tenant_id);
            if tenant != home_tenant {
                return Err(corrupt(format!(
                    "record.store tenant {} contains page owned by tenant {}",
                    home_tenant.raw(),
                    tenant.raw()
                )));
            }
            records.register_home_page(page_id, tenant);
            match PageType::from_byte(view.header().page_type)
                .map_err(|error| corrupt(format!("record.store page type is invalid: {error}")))?
            {
                PageType::Node => {
                    for (_, record) in view.iter_nodes() {
                        txn.apply_replay_mvcc_write(
                            Lsn::new(record.created_lsn),
                            tenant,
                            node_mvcc_key(arcgraph_core::NodeId::new(record.id)),
                            Some(bytes::Bytes::copy_from_slice(&record.to_bytes())),
                        );
                        report.nodes += 1;
                    }
                }
                PageType::Rel => {
                    for (_, record) in view.iter_rels() {
                        let key = rel_mvcc_key(arcgraph_core::RelId::new(record.id));
                        txn.apply_replay_mvcc_write(
                            Lsn::new(record.created_lsn),
                            tenant,
                            key,
                            Some(bytes::Bytes::copy_from_slice(&record.to_bytes())),
                        );
                        if record.expired_lsn != Lsn::MAX.raw() {
                            txn.apply_replay_mvcc_write(
                                Lsn::new(record.expired_lsn),
                                tenant,
                                key,
                                None,
                            );
                        }
                        report.rels += 1;
                    }
                }
                other => {
                    return Err(corrupt(format!(
                        "record.store page {} has non-record type {other:?}",
                        page_id.raw()
                    )));
                }
            }
            report.record_pages += 1;
            Ok(())
        })?;
    }

    let mut props_file = std::fs::File::open(generation.join(M3_PROPS_STORE_FILE))?;
    scan_page_file(&mut props_file, |page_id, page| {
        let view = SlottedPageRef::open(page.as_ref()).map_err(|error| {
            corrupt(format!(
                "props.store page {} is invalid: {error}",
                page_id.raw()
            ))
        })?;
        if view.header().page_type != PageType::PropSlotted.as_byte() {
            return Err(corrupt(format!(
                "props.store page {} has non-property type {}",
                page_id.raw(),
                view.header().page_type
            )));
        }
        blob.install_m3_base_page(TenantId::new(view.header().tenant_id), page_id, page)
            .map_err(|error| corrupt(format!("props.store base install failed: {error}")))?;
        report.prop_pages += 1;
        Ok(())
    })?;
    txn.seed_after_replay(checkpoint_lsn);
    Ok(report)
}

/// Reconcile the retained M3 primary index against exact physical record
/// coordinates. This streams `record.store`; it never materializes the owner
/// as a `Vec` and is idempotent when checkpoint metadata already carried the
/// index pages.
pub fn bootstrap_primary_from_v9_base(
    records: &BufferedRecordPageStore,
    primary: &PrimaryIndex,
    recovered_frontier: Lsn,
) -> Result<BootstrapStats, CheckpointError> {
    let mut total = BootstrapStats::default();
    records.for_each_tracked_page(|page_id, registered_tenant| {
        let page = records
            .copy_tracked_page_for_bootstrap(page_id, registered_tenant)
            .map_err(|error| {
                corrupt(format!(
                    "record cache page {} unavailable during primary bootstrap: {error}",
                    page_id.raw()
                ))
            })?;
        let view = SlottedPageRef::open(page.as_ref()).map_err(|error| {
            corrupt(format!(
                "redone record page {} is invalid: {error}",
                page_id.raw()
            ))
        })?;
        let tenant = TenantId::new(view.header().tenant_id);
        if tenant != registered_tenant {
            return Err(corrupt(format!(
                "record page {} tenant {} != registered tenant {}",
                page_id.raw(),
                tenant.raw(),
                registered_tenant.raw()
            )));
        }
        match PageType::from_byte(view.header().page_type)
            .map_err(|error| corrupt(format!("record.store page type is invalid: {error}")))?
        {
            PageType::Node => {
                for (slot, record) in view.iter_nodes() {
                    let stats = primary
                        .bootstrap_from_mvcc(std::iter::once((
                            PrimaryKey::new(tenant, RecordKind::Node, record.id),
                            PageSlot::new(page_id, slot),
                        )))
                        .map_err(|error| {
                            corrupt(format!("M3 primary bootstrap failed: {error}"))
                        })?;
                    total.indexed += stats.indexed;
                    total.skipped += stats.skipped;
                }
            }
            PageType::Rel => {
                for (slot, record) in view.iter_rels() {
                    let key = PrimaryKey::new(tenant, RecordKind::Rel, record.id);
                    if !record.is_visible_at(recovered_frontier) {
                        primary.remove(key).map_err(|error| {
                            corrupt(format!("M3 primary expiry reconcile failed: {error}"))
                        })?;
                        continue;
                    }
                    let stats = primary
                        .bootstrap_from_mvcc(std::iter::once((key, PageSlot::new(page_id, slot))))
                        .map_err(|error| {
                            corrupt(format!("M3 primary bootstrap failed: {error}"))
                        })?;
                    total.indexed += stats.indexed;
                    total.skipped += stats.skipped;
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(total)
}

fn scan_page_file(
    file: &mut std::fs::File,
    mut visit: impl FnMut(PageId, Box<[u8; PAGE_SIZE]>) -> Result<(), CheckpointError>,
) -> Result<(), CheckpointError> {
    let len = file.metadata()?.len();
    if len % PAGE_SIZE as u64 != 0 {
        return Err(corrupt("M3 physical store length is not page-aligned"));
    }
    file.seek(SeekFrom::Start(0))?;
    for raw in 0..len / PAGE_SIZE as u64 {
        let mut page = Box::new([0u8; PAGE_SIZE]);
        file.read_exact(page.as_mut())?;
        if page.iter().all(|byte| *byte == 0) {
            continue;
        }
        visit(PageId::new(raw), page)?;
    }
    Ok(())
}

fn corrupt(reason: impl Into<String>) -> CheckpointError {
    CheckpointError::Corrupt {
        reason: reason.into(),
    }
}

/// Decode the final v4 full checkpoint and write a complete v5 base into an
/// otherwise-new generation directory. Peak translation memory is one page
/// plus the existing streaming metadata buffer.
pub fn translate_v4_checkpoint(
    source: &Path,
    destination: &Path,
) -> Result<M3TranslationReport, CheckpointError> {
    let txn = Arc::new(TxnManager::new());
    let primary = Arc::new(PrimaryPageStore::new());
    let records = Arc::new(RecordPageStore::new());
    let blob = Arc::new(BlobStore::new());
    let allocator = Arc::new(PageAllocator::new());
    let crud = Arc::new(CrudStore::new_with_existing_page_stores(
        None,
        None,
        Arc::clone(&allocator),
        Arc::clone(&records),
        Arc::clone(&blob),
    ));
    let intern = Arc::new(InternTable::new());
    let idempotency = Arc::new(IdempotencyStore::new());
    let permissions = Arc::new(PermissionIndex::new());
    let seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&crud), Arc::clone(&allocator));
    let source_snapshot = CheckpointSnapshot {
        txn: &txn,
        primary_pages: &primary,
        record_pages: &records,
        blob: &blob,
        allocator_seed: seed.as_ref(),
        intern: &intern,
        idempotency: &idempotency,
        permissions: &permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let restored = restore_latest_checkpoint(source, &source_snapshot)?
        .ok_or_else(|| corrupt("v4 migration source has no valid full checkpoint"))?;
    let source_sidecar = read_latest_sidecar(source)?
        .ok_or_else(|| corrupt("v4 migration source has no checkpoint sidecar"))?;
    if !source_sidecar.full_state_snapshot
        || source_sidecar.incremental_metadata
        || source_sidecar.checkpoint_lsn != restored.checkpoint_lsn
    {
        return Err(corrupt(
            "v4 migration source checkpoint is not a full-state anchor",
        ));
    }
    let migration_lsn = restored.checkpoint_lsn;

    let mut record_files = std::collections::BTreeMap::new();
    let mut record_pages = 0u64;
    records.for_each_resident_page(|page_id, latch| {
        let guard = latch.read();
        let mut bytes = Box::new(**guard);
        stamp_page_lsn(bytes.as_mut(), migration_lsn)?;
        let header: &[u8; arcgraph_core::PageHeader::SIZE] = bytes
            [..arcgraph_core::PageHeader::SIZE]
            .try_into()
            .expect("fixed page header slice");
        let tenant = TenantId::new(
            arcgraph_core::PageHeader::from_bytes(header)
                .map_err(|error| corrupt(format!("migration record page header: {error}")))?
                .tenant_id,
        );
        let file = match record_files.entry(tenant) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let path = m3_record_store_path(destination, tenant);
                std::fs::create_dir_all(path.parent().expect("tenant store path has parent"))?;
                entry.insert(OpenOptions::new().write(true).create_new(true).open(path)?)
            }
        };
        write_page(file, page_id, bytes.as_ref())?;
        record_pages += 1;
        Ok::<(), CheckpointError>(())
    })?;
    for file in record_files.values() {
        file.sync_all()?;
    }

    let mut props_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join(M3_PROPS_STORE_FILE))?;
    let overflow = Arc::new(BlobStore::new());
    let mut prop_pages = 0u64;
    let mut overflow_pages = 0u64;
    let evicted = blob.for_each_resident_page(|tenant, page_no, page| {
        let is_props = SlottedPageRef::open(&page[..])
            .is_ok_and(|view| view.header().page_type == PageType::PropSlotted.as_byte());
        if is_props {
            let mut bytes = Box::new(*page);
            stamp_page_lsn(bytes.as_mut(), migration_lsn)?;
            write_page(&mut props_file, PageId::new(page_no), bytes.as_ref())?;
            prop_pages += 1;
        } else {
            overflow
                .install_or_replace(tenant, PageId::new(page_no), Box::new(*page))
                .map_err(|error| corrupt(format!("restore overflow page {page_no}: {error}")))?;
            overflow_pages += 1;
        }
        Ok::<(), CheckpointError>(())
    })?;
    if !evicted.is_empty() {
        return Err(corrupt(
            "unbounded migration restore unexpectedly evicted blob pages",
        ));
    }
    props_file.sync_all()?;

    let translated_snapshot = CheckpointSnapshot {
        txn: &txn,
        primary_pages: &primary,
        record_pages: &records,
        blob: &overflow,
        allocator_seed: seed.as_ref(),
        intern: &intern,
        idempotency: &idempotency,
        permissions: &permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let io = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 1,
            write_fraction: 0.0,
        },
    ));
    let props = Arc::new(BufferedRecordPageStore::with_cache_cap(
        Arc::clone(&pools),
        1,
    ));
    let records_home = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 1));
    let write_behind =
        WriteBehindCheckpointer::new(Arc::new(DirtyPageTable::new()), props, records_home)
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(destination)));
    let catalog_pool = BufferPool::new(1, Arc::new(InMemoryPageIo::new()));
    let allocator_for_capture = Arc::clone(&allocator);
    let crud_for_capture = Arc::clone(&crud);
    let report = incremental_checkpoint(
        destination,
        &catalog_pool,
        &translated_snapshot,
        &write_behind,
        move || {
            let mut advances = allocator_for_capture.snapshot_advances();
            advances.extend(crud_for_capture.snapshot_allocator_advances());
            (advances, None)
        },
        Ok,
    )?;
    if report.checkpoint_lsn != migration_lsn || report.redo_lsn != migration_lsn {
        return Err(corrupt(
            "translated v9 checkpoint did not preserve the migration frontier",
        ));
    }
    Ok(M3TranslationReport {
        migration_lsn,
        prop_pages,
        record_pages,
        overflow_pages,
    })
}

fn stamp_page_lsn(bytes: &mut [u8; PAGE_SIZE], migration_lsn: Lsn) -> Result<(), CheckpointError> {
    let mut page = SlottedPage::open(bytes)
        .map_err(|error| corrupt(format!("migration page validation failed: {error}")))?;
    if page.page_lsn().raw() > migration_lsn.raw() {
        return Err(corrupt("page LSN exceeds the final checkpoint frontier"));
    }
    page.apply_redo_if_newer(migration_lsn, |_| Ok::<(), std::convert::Infallible>(()))
        .expect("infallible migration page stamp");
    Ok(())
}

fn write_page(
    file: &mut std::fs::File,
    page_id: PageId,
    bytes: &[u8; PAGE_SIZE],
) -> Result<(), CheckpointError> {
    let offset = page_id
        .raw()
        .checked_mul(PAGE_SIZE as u64)
        .ok_or_else(|| corrupt("migration page offset overflow"))?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    Ok(())
}
