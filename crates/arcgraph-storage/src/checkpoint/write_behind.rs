//! M3 write-behind page checkpointer.
//!
//! Dirty pages are copied through the ADR-140 pin-coupled surface and
//! written home in bounded batches. A page leaves the DPT only when the
//! generation captured before its copy still matches after the durable
//! home write. A concurrent re-dirty therefore retains the original
//! `recLSN`, which in turn holds `redo_lsn = min(recLSN)` down.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use arcgraph_core::{ArcGraphError, Lsn, PageId, Result, TenantId};

use crate::blob::BlobStore;
use crate::checkpoint::doublewrite::{DoublewriteArea, DoublewriteKey};
use crate::io::{PageBuf, PageIo};
use crate::page_store::BufferedRecordPageStore;
use crate::redo::{DirtyPageSnapshot, DirtyPageTable};
use crate::wal::{STORE_PROPS, STORE_RECORD};

/// Default maximum resident page images per home-write batch (2 MiB).
pub const DEFAULT_WRITE_BEHIND_BATCH_PAGES: usize = 256;

/// Pin-coupled copy plus durable home-write surface used by the checkpointer.
pub trait PageFlushTarget: Send + Sync {
    /// Copy one self-consistent page image under its pin-coupled write latch.
    fn copy_page_pinned(&self, tenant: TenantId, page_id: PageId) -> Result<Option<Box<PageBuf>>>;

    /// Durably write a batch to home locations. DWB ordering is layered on
    /// this method by phase 3; phase 2 establishes the write-behind/DPT seam.
    fn write_pages_home(&self, images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()>;
}

impl PageFlushTarget for BufferedRecordPageStore {
    fn copy_page_pinned(&self, tenant: TenantId, page_id: PageId) -> Result<Option<Box<PageBuf>>> {
        BufferedRecordPageStore::copy_page_pinned_for_tenant(self, tenant, page_id).map_err(
            |error| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "write-behind copy_page_pinned({tenant:?}, {page_id:?}) failed: {error}"
                )))
            },
        )
    }

    fn write_pages_home(&self, images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()> {
        BufferedRecordPageStore::write_pages_home_qualified(self, images)
    }
}

/// Store-0 flush target: copies the live immutable BlobStore page and writes
/// it to the M3 physical props home before making it eviction-eligible.
pub struct BlobPageFlushTarget {
    blob: Arc<BlobStore>,
    home: Arc<dyn PageIo>,
}

impl BlobPageFlushTarget {
    #[must_use]
    pub fn new(blob: Arc<BlobStore>, home: Arc<dyn PageIo>) -> Self {
        Self { blob, home }
    }
}

impl PageFlushTarget for BlobPageFlushTarget {
    fn copy_page_pinned(&self, tenant: TenantId, page_id: PageId) -> Result<Option<Box<PageBuf>>> {
        crate::redo::DeltaPageStore::read_page_for_redo(self.blob.as_ref(), tenant, page_id)
    }

    fn write_pages_home(&self, images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()> {
        for (_, page_id, page) in images {
            self.home.write_page(*page_id, page.as_ref())?;
        }
        self.home.flush()?;
        for (tenant, page_id, _) in images {
            self.blob.mark_m3_page_checkpointed(*tenant, *page_id)?;
        }
        Ok(())
    }
}

/// Observability from one completed write-behind pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteBehindReport {
    /// DPT entries observed at pass start.
    pub snapshot_pages: usize,
    /// Page images durably written home.
    pub flushed_pages: usize,
    /// Flushed entries retained because their dirty generation changed.
    pub retained_redirties: usize,
    /// ARIES recovery/prune anchor after generation-checked removals.
    pub redo_lsn: Lsn,
}

/// Bounded write-behind checkpointer for M3's two delta stores.
pub struct WriteBehindCheckpointer {
    /// Serializes admission before any DPT snapshot and remains held through
    /// DWB staging, home fsync, and generation-matched completion. The guard
    /// carries no mutable protocol state, so poison does not invalidate the
    /// admission ordering itself.
    pass_admission: Mutex<()>,
    dpt: Arc<DirtyPageTable>,
    props: Option<Arc<dyn PageFlushTarget>>,
    records: Option<Arc<dyn PageFlushTarget>>,
    batch_pages: usize,
    doublewrite: Option<Arc<DoublewriteArea>>,
    data_targets: BTreeMap<u16, Arc<dyn PageFlushTarget>>,
    directory_targets: BTreeMap<u16, Arc<dyn PageFlushTarget>>,
    extent_data_targets: BTreeMap<(TenantId, u16), Arc<crate::extent::ExtentDataPageStore>>,
    extent_directory_targets: BTreeMap<(TenantId, u16), Arc<crate::extent::ExtentDirectory>>,
    /// INV-M5.10 negative control (fault-injection only): synthetic routes
    /// injected by the build-isolation gate to prove its route-census
    /// assertion bites. Never consulted by any flush pass.
    #[cfg(feature = "fault-injection")]
    injected_routes: std::sync::Mutex<BTreeSet<(Option<TenantId>, u16)>>,
}

#[derive(Default)]
struct PassCounts {
    flushed_pages: usize,
    retained_redirties: usize,
    /// Priority passes need exact completion identities for eviction. Full
    /// passes omit the set so their bookkeeping remains O(1) beyond the DPT
    /// snapshot they already own.
    completed_keys: Option<BTreeSet<crate::redo::DirtyPageKey>>,
}

impl WriteBehindCheckpointer {
    fn admit_pass(&self) -> std::sync::MutexGuard<'_, ()> {
        self.pass_admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[must_use]
    pub fn new(
        dpt: Arc<DirtyPageTable>,
        props: Arc<dyn PageFlushTarget>,
        records: Arc<dyn PageFlushTarget>,
    ) -> Self {
        Self::with_batch_pages(dpt, props, records, DEFAULT_WRITE_BEHIND_BATCH_PAGES)
    }

    #[must_use]
    pub fn with_batch_pages(
        dpt: Arc<DirtyPageTable>,
        props: Arc<dyn PageFlushTarget>,
        records: Arc<dyn PageFlushTarget>,
        batch_pages: usize,
    ) -> Self {
        assert!(
            batch_pages > 0,
            "write-behind batch must contain at least one page"
        );
        Self {
            pass_admission: Mutex::new(()),
            dpt,
            props: Some(props),
            records: Some(records),
            batch_pages,
            doublewrite: None,
            data_targets: BTreeMap::new(),
            directory_targets: BTreeMap::new(),
            extent_data_targets: BTreeMap::new(),
            extent_directory_targets: BTreeMap::new(),
            #[cfg(feature = "fault-injection")]
            injected_routes: std::sync::Mutex::new(BTreeSet::new()),
        }
    }

    /// Construct an extent-only checkpointer for the M4 generation.
    ///
    /// Every M4 physical store is registered through
    /// [`Self::with_extent_data_target`] and
    /// [`Self::with_extent_directory_target`]. Keeping the retired M3
    /// props/record targets absent makes it impossible for a v10 checkpoint
    /// to flush a direct-address page through the legacy home by mistake.
    #[must_use]
    pub fn new_extent_only(dpt: Arc<DirtyPageTable>) -> Self {
        Self {
            pass_admission: Mutex::new(()),
            dpt,
            props: None,
            records: None,
            batch_pages: DEFAULT_WRITE_BEHIND_BATCH_PAGES,
            doublewrite: None,
            data_targets: BTreeMap::new(),
            directory_targets: BTreeMap::new(),
            extent_data_targets: BTreeMap::new(),
            extent_directory_targets: BTreeMap::new(),
            #[cfg(feature = "fault-injection")]
            injected_routes: std::sync::Mutex::new(BTreeSet::new()),
        }
    }

    /// Layer a checksummed DWB fsync before every home-write batch.
    #[must_use]
    pub fn with_doublewrite_area(mut self, doublewrite: Arc<DoublewriteArea>) -> Self {
        self.doublewrite = Some(doublewrite);
        self
    }

    /// Route tagged extent-directory pages for `store_id` through their
    /// fixed-head flush target. Data pages for the same store keep using the
    /// ordinary props/records target.
    #[must_use]
    pub fn with_directory_target(
        mut self,
        store_id: u16,
        target: Arc<dyn PageFlushTarget>,
    ) -> Self {
        self.directory_targets.insert(store_id, target);
        self
    }

    /// Register an additional extent-backed data store (for example TEL).
    /// M3's built-in props/record routes remain unchanged.
    #[must_use]
    pub fn with_data_target(mut self, store_id: u16, target: Arc<dyn PageFlushTarget>) -> Self {
        self.data_targets.insert(store_id, target);
        self
    }

    /// Register one tenant-qualified production extent data target. It
    /// supersedes the legacy M3 route only for that exact owner.
    #[must_use]
    pub fn with_extent_data_target(
        mut self,
        target: Arc<crate::extent::ExtentDataPageStore>,
    ) -> Self {
        self.extent_data_targets.insert(
            (target.directory().tenant(), target.directory().store_id()),
            target,
        );
        self
    }

    /// Register one tenant-qualified fixed-head directory flush target.
    #[must_use]
    pub fn with_extent_directory_target(
        mut self,
        target: Arc<crate::extent::ExtentDirectory>,
    ) -> Self {
        self.extent_directory_targets
            .insert((target.tenant(), target.store_id()), target);
        self
    }

    /// Flush a pass for a v9/v10 checkpoint, refusing to establish incremental
    /// metadata unless the DWB durability barrier is wired. The legacy
    /// `flush_pass` remains useful to unit-test the DPT seam in isolation;
    /// production v9 establishment must call this stricter entry point.
    pub fn flush_pass_with_doublewrite(&self, checkpoint_lsn: Lsn) -> Result<WriteBehindReport> {
        if self.doublewrite.is_none() {
            return Err(ArcGraphError::WalCorruption {
                lsn: checkpoint_lsn,
                reason: "incremental checkpoint requires a DoublewriteArea before home writes"
                    .to_owned(),
            });
        }
        self.flush_pass(checkpoint_lsn)
    }

    /// Capture the post-flush DPT written into incremental metadata.
    /// Callers take the brief commit-freeze while invoking this method so the
    /// checkpoint frontier and DPT describe one page-apply boundary.
    #[must_use]
    pub fn metadata_dpt_snapshot(&self) -> Vec<DirtyPageSnapshot> {
        self.dpt.snapshot()
    }

    /// Complete configured routing census. `None` denotes the legacy
    /// any-tenant M3 route; `Some(tenant)` denotes an exact v6 extent owner.
    #[must_use]
    pub fn route_census(&self) -> BTreeSet<(Option<TenantId>, u16)> {
        let mut routes = BTreeSet::new();
        if self.props.is_some() {
            routes.insert((None, STORE_PROPS));
        }
        if self.records.is_some() {
            routes.insert((None, STORE_RECORD));
        }
        routes.extend(
            self.data_targets
                .keys()
                .chain(self.directory_targets.keys())
                .map(|store| (None, *store)),
        );
        routes.extend(
            self.extent_data_targets
                .keys()
                .chain(self.extent_directory_targets.keys())
                .map(|(tenant, store)| (Some(*tenant), *store)),
        );
        #[cfg(feature = "fault-injection")]
        routes.extend(
            self.injected_routes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .copied(),
        );
        routes
    }

    /// Record one synthetic route as if the invisible build had registered a
    /// flush target on the LIVE checkpointer. Gate-only negative control for
    /// the INV-M5.10 route census; no flush pass ever consults this set.
    #[cfg(feature = "fault-injection")]
    pub fn inject_route_for_build_isolation_gate(&self, route: (Option<TenantId>, u16)) {
        self.injected_routes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(route);
    }

    /// Synchronous admission-state probe for the #1528 release gate.
    #[cfg(feature = "fault-injection")]
    #[doc(hidden)]
    #[must_use]
    pub fn pass_admission_is_held_for_gate(&self) -> bool {
        match self.pass_admission.try_lock() {
            Ok(_guard) => false,
            Err(std::sync::TryLockError::WouldBlock) => true,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                panic!("write-behind admission mutex poisoned during #1528 gate")
            }
        }
    }

    /// Flush one stable DPT observation without a global commit freeze.
    ///
    /// Earlier successful batches remain valid if a later batch errors; only
    /// entries whose home write completed are eligible for DPT removal.
    pub fn flush_pass(&self, checkpoint_lsn: Lsn) -> Result<WriteBehindReport> {
        // Admission precedes the snapshot and spans the complete stage -> home
        // -> completion lifecycle. Thus a later pass can observe either this
        // pass's retained generation or its completed absence, but can never
        // publish a newer home image before this pass publishes an older one.
        let _admission = self.admit_pass();
        let snapshot = self.dpt.snapshot();
        let counts = self.dispatch_snapshot_to_targets(&snapshot, false)?;

        Ok(WriteBehindReport {
            snapshot_pages: snapshot.len(),
            flushed_pages: counts.flushed_pages,
            retained_redirties: counts.retained_redirties,
            redo_lsn: self.dpt.redo_lsn(checkpoint_lsn),
        })
    }

    /// M6.1 MECH-E2 — the eviction driver's handshake target. Flushes ONLY
    /// the caller-supplied dirty-page keys (NOT a full DPT sweep) through
    /// the SAME `PageFlushTarget` pipeline `flush_pass` uses — same
    /// routing, same batching, same doublewrite staging, same
    /// generation-matching DPT removal. This is what makes MECH-E2 true:
    /// the evictor never calls `write_pages_home` itself; it hands the
    /// victim's qualified key to THIS checkpointer method, which remains
    /// the single writer.
    ///
    /// Keys not currently dirty in the DPT (already flushed by a
    /// concurrent pass, or never dirty) are silently skipped — the caller
    /// (the eviction driver) treats "not in the DPT" as "already covered
    /// by a durable home", which is exactly INV-M6.2's disjunction.
    ///
    /// Returns the set of keys whose durable home write completed AND
    /// whose DPT entry was removed (i.e. `dirty_gen` did not advance
    /// between the caller's copy and this flush) — the eviction driver
    /// uses membership in this set to decide whether a frame is now safe
    /// to reclaim (MECH-E3/E4). A key present in the input but ABSENT from
    /// the returned set means either (a) it was not dirty (no home write
    /// needed — the driver may still reclaim), or (b) it was re-dirtied
    /// after a concurrent snapshot raced this flush (the driver must NOT
    /// reclaim; the frame stays resident until a later pass observes a
    /// clean generation-matched flush).
    pub fn flush_priority_keys(
        &self,
        keys: &[crate::redo::DirtyPageKey],
    ) -> Result<BTreeSet<crate::redo::DirtyPageKey>> {
        let _admission = self.admit_pass();
        let snapshot: Vec<DirtyPageSnapshot> = keys
            .iter()
            .filter_map(|key| self.dpt.snapshot_key(*key))
            .collect();
        if snapshot.is_empty() {
            return Ok(BTreeSet::new());
        }
        let counts = self.dispatch_snapshot_to_targets(&snapshot, true)?;
        // Return the exact generation-matched compare-and-remove outcomes.
        // Inferring completion from a later absent-key lookup is unsafe: an
        // overlapping pass may have removed a different generation/epoch.
        Ok(counts
            .completed_keys
            .expect("priority flush must track completed keys"))
    }

    /// Shared target-dispatch loop: routes a snapshot's entries to each
    /// registered `PageFlushTarget` exactly as `flush_pass` always has —
    /// extracted so [`Self::flush_priority_keys`] reuses the identical
    /// routing/batching/DWB/generation-matched-removal pipeline (MECH-E2:
    /// one writer, one dispatch path, whether the trigger is periodic
    /// checkpoint cadence or eviction pressure).
    fn dispatch_snapshot_to_targets(
        &self,
        snapshot: &[DirtyPageSnapshot],
        track_completed_keys: bool,
    ) -> Result<PassCounts> {
        for dirty in snapshot {
            let is_directory = crate::extent::is_directory_page(dirty.key.page_no);
            let extent_routed = self.routed_to_extent(*dirty, is_directory)?;
            let supported = if is_directory {
                self.directory_targets.contains_key(&dirty.key.store_id) || extent_routed
            } else {
                matches!(dirty.key.store_id, STORE_PROPS | STORE_RECORD)
                    || self.data_targets.contains_key(&dirty.key.store_id)
                    || extent_routed
            };
            if !supported {
                return Err(ArcGraphError::WalCorruption {
                    lsn: dirty.rec_lsn,
                    reason: format!(
                        "DPT contains store_id {} outside the M3 delta set",
                        dirty.key.store_id
                    ),
                });
            }
        }

        let mut counts = PassCounts {
            completed_keys: track_completed_keys.then(BTreeSet::new),
            ..PassCounts::default()
        };
        if let Some(props) = &self.props {
            self.flush_store(
                STORE_PROPS,
                props.as_ref(),
                snapshot,
                false,
                None,
                false,
                &mut counts,
            )?;
        }
        if let Some(records) = &self.records {
            self.flush_store(
                STORE_RECORD,
                records.as_ref(),
                snapshot,
                false,
                None,
                false,
                &mut counts,
            )?;
        }
        for (&store_id, target) in &self.data_targets {
            self.flush_store(
                store_id,
                target.as_ref(),
                snapshot,
                false,
                None,
                false,
                &mut counts,
            )?;
        }
        for (&store_id, target) in &self.directory_targets {
            self.flush_store(
                store_id,
                target.as_ref(),
                snapshot,
                true,
                None,
                false,
                &mut counts,
            )?;
        }
        // A data-page DWB key is logical. Its directory mapping must already
        // be durable before any such batch can replace the DWB file; otherwise
        // bootstrap cannot resolve a torn data home before WAL replay starts.
        for (&(tenant, store_id), target) in &self.extent_directory_targets {
            self.flush_store(
                store_id,
                target.as_ref(),
                snapshot,
                true,
                Some(tenant),
                true,
                &mut counts,
            )?;
        }
        for (&(tenant, store_id), target) in &self.extent_data_targets {
            self.flush_store(
                store_id,
                target.as_ref(),
                snapshot,
                false,
                Some(tenant),
                true,
                &mut counts,
            )?;
        }

        Ok(counts)
    }

    #[allow(clippy::too_many_arguments)] // M4 Slice-2: recovery fn threads directory/data/dpt/doublewrite
    fn flush_store(
        &self,
        store_id: u16,
        target: &dyn PageFlushTarget,
        snapshot: &[DirtyPageSnapshot],
        directory_namespace: bool,
        owner_tenant: Option<TenantId>,
        extent_only: bool,
        counts: &mut PassCounts,
    ) -> Result<()> {
        let mut images = Vec::with_capacity(self.batch_pages);
        let mut tokens = Vec::with_capacity(self.batch_pages);
        for dirty in snapshot.iter().copied() {
            if dirty.key.store_id != store_id
                || crate::extent::is_directory_page(dirty.key.page_no) != directory_namespace
                || owner_tenant.is_some_and(|tenant| dirty.key.tenant_id != tenant)
            {
                continue;
            }
            let extent_routed = self.routed_to_extent(dirty, directory_namespace)?;
            if extent_routed != extent_only {
                continue;
            }
            let page_id = PageId::new(dirty.key.page_no);
            let image = target
                .copy_page_pinned(dirty.key.tenant_id, page_id)
                .map_err(|error| ArcGraphError::WalCorruption {
                    lsn: dirty.rec_lsn,
                    reason: format!(
                        "incremental checkpoint copy failed for tenant {} store {} page {}: {error}",
                        dirty.key.tenant_id.raw(),
                        store_id,
                        dirty.key.page_no,
                    ),
                })?
                .ok_or_else(|| ArcGraphError::PageCorruption {
                    page_id,
                    reason: format!(
                        "DPT references missing store_id {store_id} page at recLSN {}",
                        dirty.rec_lsn.raw()
                    ),
                })?;
            images.push((dirty.key.tenant_id, page_id, image));
            tokens.push(dirty);
            if images.len() == self.batch_pages {
                self.finish_batch(store_id, target, &mut images, &mut tokens, counts)?;
            }
        }
        self.finish_batch(store_id, target, &mut images, &mut tokens, counts)
    }

    fn routed_to_extent(
        &self,
        dirty: DirtyPageSnapshot,
        directory_namespace: bool,
    ) -> Result<bool> {
        let owner = (dirty.key.tenant_id, dirty.key.store_id);
        if directory_namespace {
            return Ok(self.extent_directory_targets.contains_key(&owner));
        }
        let Some(target) = self.extent_data_targets.get(&owner) else {
            return Ok(false);
        };
        target
            .directory()
            .mapping(dirty.key.page_no / crate::extent::EXTENT_PAGES)
            .map(|mapping| mapping.is_some())
    }

    fn finish_batch(
        &self,
        store_id: u16,
        target: &dyn PageFlushTarget,
        images: &mut Vec<(TenantId, PageId, Box<PageBuf>)>,
        tokens: &mut Vec<DirtyPageSnapshot>,
        counts: &mut PassCounts,
    ) -> Result<()> {
        if images.is_empty() {
            return Ok(());
        }
        if let Some(doublewrite) = &self.doublewrite {
            let staged: Vec<_> = tokens
                .iter()
                .zip(images.iter())
                .map(|(dirty, (_, _, page))| {
                    (
                        DoublewriteKey {
                            tenant_id: dirty.key.tenant_id,
                            store_id,
                            page_no: dirty.key.page_no,
                        },
                        page.as_ref(),
                    )
                })
                .collect();
            doublewrite
                .stage_batch(&staged)
                .map_err(|error| ArcGraphError::WalCorruption {
                    lsn: tokens.first().map_or(Lsn::ZERO, |dirty| dirty.rec_lsn),
                    reason: format!(
                        "incremental checkpoint doublewrite stage failed for store {store_id}: {error}"
                    ),
                })?;
        }
        target
            .write_pages_home(images)
            .map_err(|error| ArcGraphError::WalCorruption {
                lsn: tokens.first().map_or(Lsn::ZERO, |dirty| dirty.rec_lsn),
                reason: format!(
                    "incremental checkpoint home write failed for store {store_id}: {error}"
                ),
            })?;
        counts.flushed_pages += images.len();
        for dirty in tokens.drain(..) {
            if self.dpt.complete_flush(dirty) {
                if let Some(completed_keys) = &mut counts.completed_keys {
                    completed_keys.insert(dirty.key);
                }
            } else {
                counts.retained_redirties += 1;
            }
        }
        images.clear();
        Ok(())
    }
}
