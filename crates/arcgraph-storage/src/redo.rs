//! M3 redo ordering and dirty-page-table primitives.
//!
//! The redo clock is a first-class object: every commit owns one
//! contiguous [`RedoLsnRange`], every physiological op owns one unique
//! LSN inside that range, recovery orders bundles by range base, and a
//! page stamps the exact op LSN it has applied. Legal allocation gaps
//! are tolerated; overlapping non-duplicate ranges are corruption.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord};
use arcgraph_core::{ArcGraphError, Lsn, PageId, Result as ArcGraphResult, TenantId};
use dashmap::DashMap;

use crate::io::PageBuf;
use crate::owner_row::{OWNER_ROW_BYTES, OWNER_ROWS_PER_PAGE, owner_delta_is_retirement};
use crate::page_store::{BufferedRecordPageStore, RecordPageBackend};
use crate::records::{SlotId, SlottedPage};
use crate::wal::{
    DeltaOp, DeltaOpKind, STORE_BLOB_OVERFLOW, STORE_GRANTS, STORE_INTERN, STORE_NODE_BINDINGS,
    STORE_PROPS, STORE_RECORD, STORE_REL_BINDINGS, STORE_RELS, STORE_TEL,
};

/// One commit's contiguous allocation in the global redo order.
///
/// `end` is also the commit's MVCC LSN. A commit with no physiological
/// ops still consumes a singleton range so visibility remains strictly
/// monotone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedoLsnRange {
    base: Lsn,
    end: Lsn,
}

impl RedoLsnRange {
    /// Build a range from its inclusive endpoints.
    #[must_use]
    pub const fn new(base: Lsn, end: Lsn) -> Option<Self> {
        if base.raw() == 0 || base.raw() > end.raw() {
            return None;
        }
        Some(Self { base, end })
    }

    /// Reconstruct a range from the bundle's commit LSN (range end)
    /// and emitted op count. Zero-op commits consume width one.
    #[must_use]
    pub fn ending_at(commit_lsn: Lsn, op_count: usize) -> Option<Self> {
        let width = u64::try_from(op_count.max(1)).ok()?;
        let base = commit_lsn.raw().checked_sub(width - 1)?;
        Self::new(Lsn::new(base), commit_lsn)
    }

    /// Inclusive first op LSN; recovery's bundle sort key.
    #[must_use]
    pub const fn base(self) -> Lsn {
        self.base
    }

    /// Inclusive range end; also the commit/MVCC LSN.
    #[must_use]
    pub const fn end(self) -> Lsn {
        self.end
    }

    /// Alias documenting the end's MVCC role.
    #[must_use]
    pub const fn commit_lsn(self) -> Lsn {
        self.end
    }

    /// LSN immediately before this range. `install_order` waits for
    /// this value, not `commit_lsn - 1`.
    #[must_use]
    pub const fn predecessor(self) -> Lsn {
        Lsn::new(self.base.raw() - 1)
    }

    /// Width of the inclusive range.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end.raw() - self.base.raw() + 1
    }

    /// Ranges are never empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// LSN for op `index` in build order.
    #[must_use]
    pub fn op_lsn(self, index: usize) -> Option<Lsn> {
        let index = u64::try_from(index).ok()?;
        if index >= self.len() {
            return None;
        }
        Some(Lsn::new(self.base.raw() + index))
    }

    /// Whether `lsn` lies in this inclusive range.
    #[must_use]
    pub const fn contains(self, lsn: Lsn) -> bool {
        self.base.raw() <= lsn.raw() && lsn.raw() <= self.end.raw()
    }
}

/// Recovery-order validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RedoOrderError {
    /// Earlier unique range in sorted order.
    pub previous: RedoLsnRange,
    /// Range that illegally overlaps it.
    pub overlapping: RedoLsnRange,
}

impl fmt::Display for RedoOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "overlapping redo ranges {:?} and {:?}",
            self.previous, self.overlapping
        )
    }
}

impl std::error::Error for RedoOrderError {}

/// Observability from sorting a physical WAL stream into redo order.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RedoOrderStats {
    /// Exact duplicate ranges (retry/duplicate WAL frames). Legal and
    /// replayed idempotently through page LSNs.
    pub duplicate_ranges: u64,
    /// Number of gaps between unique ranges. Legal: allocation can win
    /// and its commit can die before append.
    pub gaps: u64,
}

/// Sort bundles by the full redo range and validate the total order.
///
/// Physical WAL append order is deliberately not trusted. Exact
/// duplicate ranges remain in the output so page-LSN idempotence is
/// exercised; non-identical overlaps are rejected as corruption.
pub fn sort_by_redo_range<T>(
    bundles: &mut [T],
    range_of: impl Fn(&T) -> RedoLsnRange,
) -> Result<RedoOrderStats, RedoOrderError> {
    bundles.sort_by_key(|bundle| {
        let range = range_of(bundle);
        (range.base().raw(), range.end().raw())
    });

    let mut stats = RedoOrderStats::default();
    let mut previous_unique: Option<RedoLsnRange> = None;
    for bundle in bundles {
        let current = range_of(bundle);
        let Some(previous) = previous_unique else {
            previous_unique = Some(current);
            continue;
        };
        if current == previous {
            stats.duplicate_ranges += 1;
            continue;
        }
        if current.base().raw() <= previous.end().raw() {
            return Err(RedoOrderError {
                previous,
                overlapping: current,
            });
        }
        if current.base().raw() > previous.end().raw().saturating_add(1) {
            stats.gaps += 1;
        }
        previous_unique = Some(current);
    }
    Ok(stats)
}

/// Apply one physiological redo op iff its full sub-LSN is newer than
/// the page's stamp. The stamp advances only after `apply` succeeds.
pub fn apply_redo_if_newer<E>(
    page_lsn: &mut Lsn,
    op_lsn: Lsn,
    apply: impl FnOnce() -> Result<(), E>,
) -> Result<bool, E> {
    if op_lsn.raw() <= page_lsn.raw() {
        return Ok(false);
    }
    apply()?;
    *page_lsn = op_lsn;
    Ok(true)
}

/// Apply one M3 physical data delta to an already-present slotted page.
///
/// This is the shared live/recovery mutation primitive for `PutRecord`,
/// `PutPropBlock`, and `TombstoneRecord`. It validates that the WAL target
/// names this exact tenant/page/store, applies only when the op's full
/// sub-LSN is newer than the page LSN, and stamps the page only after the
/// physiological mutation succeeds. `PageAlloc` owns the missing-page
/// lifecycle and is intentionally handled by v9 recovery rather than here.
pub fn apply_physical_delta(
    page_bytes: &mut [u8; PAGE_SIZE],
    op: &DeltaOp,
    commit_lsn: Lsn,
) -> ArcGraphResult<bool> {
    op.validate_shape()?;
    if !matches!(
        op.kind,
        DeltaOpKind::PutRecord
            | DeltaOpKind::PutPropBlock
            | DeltaOpKind::TombstoneRecord
            | DeltaOpKind::InternBind
            | DeltaOpKind::AclGrant
    ) {
        return Err(delta_corruption(
            op,
            format!("DeltaOp kind {:?} is not a physical data mutation", op.kind),
        ));
    }
    if commit_lsn.raw() < op.op_lsn.raw() {
        return Err(delta_corruption(
            op,
            format!(
                "bundle commit_lsn {} precedes op_lsn {}",
                commit_lsn.raw(),
                op.op_lsn.raw()
            ),
        ));
    }

    let mut page = SlottedPage::open(page_bytes)
        .map_err(|error| delta_corruption(op, format!("cannot open target page: {error}")))?;
    let header = page.header();
    if header.page_id != op.page_no || header.tenant_id != op.tenant_id.raw() {
        return Err(delta_corruption(
            op,
            format!(
                "delta target tenant/page ({}, {}) does not match page header ({}, {})",
                op.tenant_id.raw(),
                op.page_no,
                header.tenant_id,
                header.page_id
            ),
        ));
    }
    let page_type = PageType::from_byte(header.page_type)
        .map_err(|error| delta_corruption(op, format!("invalid target page type: {error}")))?;
    match op.store_id {
        STORE_PROPS if page_type == PageType::PropSlotted => {}
        STORE_RECORD if matches!(page_type, PageType::Node | PageType::Rel) => {}
        STORE_RELS if page_type == PageType::Rel => {}
        STORE_NODE_BINDINGS | STORE_REL_BINDINGS | STORE_INTERN | STORE_GRANTS
            if page_type == PageType::PropSlotted && header.flags == op.store_id => {}
        STORE_PROPS | STORE_RECORD | STORE_RELS | STORE_NODE_BINDINGS | STORE_REL_BINDINGS
        | STORE_INTERN | STORE_GRANTS => {
            return Err(delta_corruption(
                op,
                format!(
                    "page type {page_type:?} does not belong to store_id {}",
                    op.store_id
                ),
            ));
        }
        _ => {
            return Err(delta_corruption(
                op,
                format!("store_id {} is not page-LSN-governed at M3", op.store_id),
            ));
        }
    }

    let slot = SlotId(op.slot);
    page.apply_redo_if_newer(op.op_lsn, |page| match op.kind {
        DeltaOpKind::PutRecord if page_type == PageType::Node => {
            let bytes: &[u8; NodeRecord::SIZE] = op.payload.as_ref().try_into().map_err(|_| {
                delta_corruption(op, "PutRecord node payload is not exactly 64 bytes")
            })?;
            let record = NodeRecord::from_bytes(bytes)
                .map_err(|error| delta_corruption(op, format!("invalid node payload: {error}")))?;
            page.put_node_at(slot, &record)
                .map_err(|error| delta_corruption(op, format!("PutRecord apply failed: {error}")))
        }
        DeltaOpKind::PutRecord if page_type == PageType::Rel => {
            let bytes: &[u8; RelRecord::SIZE] = op.payload.as_ref().try_into().map_err(|_| {
                delta_corruption(op, "PutRecord relationship payload is not exactly 96 bytes")
            })?;
            let record = RelRecord::from_bytes(bytes).map_err(|error| {
                delta_corruption(op, format!("invalid relationship payload: {error}"))
            })?;
            page.put_rel_at(slot, &record)
                .map_err(|error| delta_corruption(op, format!("PutRecord apply failed: {error}")))
        }
        DeltaOpKind::PutRecord => Err(delta_corruption(
            op,
            format!("PutRecord cannot target page type {page_type:?}"),
        )),
        DeltaOpKind::PutPropBlock if page_type == PageType::PropSlotted => page
            .put_bag_at(slot, &op.payload)
            .map_err(|error| delta_corruption(op, format!("PutPropBlock apply failed: {error}"))),
        DeltaOpKind::PutPropBlock => Err(delta_corruption(
            op,
            format!("PutPropBlock cannot target page type {page_type:?}"),
        )),
        DeltaOpKind::InternBind | DeltaOpKind::AclGrant if page_type == PageType::PropSlotted => {
            if owner_delta_is_retirement(&op.payload) {
                page.permanent_tombstone_fixed_bag_at_slot(
                    slot,
                    op.op_lsn,
                    OWNER_ROW_BYTES as u16,
                    OWNER_ROWS_PER_PAGE as u16,
                )
                .map_err(|error| {
                    delta_corruption(op, format!("owner-row retirement apply failed: {error}"))
                })
            } else {
                page.put_fixed_bag_at_slot(slot, &op.payload, OWNER_ROWS_PER_PAGE as u16)
                    .map_err(|error| {
                        delta_corruption(op, format!("owner-row put apply failed: {error}"))
                    })
            }
        }
        DeltaOpKind::InternBind | DeltaOpKind::AclGrant => Err(delta_corruption(
            op,
            format!("owner-row delta cannot target page type {page_type:?}"),
        )),
        DeltaOpKind::TombstoneRecord if page_type == PageType::Node => page
            .tombstone(slot)
            .map_err(|error| delta_corruption(op, format!("node tombstone apply failed: {error}"))),
        DeltaOpKind::TombstoneRecord if page_type == PageType::Rel => {
            let mut record = page
                .read_rel(slot)
                .map_err(|error| {
                    delta_corruption(op, format!("relationship tombstone read failed: {error}"))
                })?
                .ok_or_else(|| {
                    delta_corruption(op, "relationship tombstone targets an empty slot")
                })?;
            record.expired_lsn = commit_lsn.raw();
            page.update_rel(slot, &record).map_err(|error| {
                delta_corruption(op, format!("relationship tombstone apply failed: {error}"))
            })
        }
        DeltaOpKind::TombstoneRecord => Err(delta_corruption(
            op,
            format!("TombstoneRecord cannot target page type {page_type:?}"),
        )),
        _ => unreachable!("physical data kinds checked before page attach"),
    })
}

fn delta_corruption(op: &DeltaOp, reason: impl Into<String>) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn: op.op_lsn,
        reason: reason.into(),
    }
}

/// Physical store seam used by v9 redo. A missing return is distinct from a
/// present-but-torn page: present bytes are checksum-validated by the shared
/// slotted-page attach before any page-LSN comparison.
pub trait DeltaPageStore: Send + Sync {
    fn read_page_for_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> ArcGraphResult<Option<Box<PageBuf>>>;
    fn install_page_from_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> ArcGraphResult<()>;
}

impl DeltaPageStore for BufferedRecordPageStore {
    fn read_page_for_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> ArcGraphResult<Option<Box<PageBuf>>> {
        self.copy_page_pinned_for_tenant(tenant, page_id)
            .map_err(|error| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "v9 redo read_page_for_redo({tenant:?}, {page_id:?}) failed: {error}"
                )))
            })
    }

    fn install_page_from_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> ArcGraphResult<()> {
        self.register_home_page(page_id, tenant);
        RecordPageBackend::install_or_replace_for_tenant(self, tenant, page_id, page).map_err(
            |error| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "v9 redo install_page_from_redo({tenant:?}, {page_id:?}) failed: {error}"
                )))
            },
        )
    }
}

/// Missing→Formatted→Live lifecycle outcome for one physical redo op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryDeltaOutcome {
    /// Page LSN already covered the op.
    Idempotent,
    /// `PageAlloc` formatted a previously-missing page.
    Formatted,
    /// A data mutation advanced a live page.
    Applied,
}

/// Apply one v9 physical op through the explicit page lifecycle state
/// machine. DWB restore must have completed before callers invoke this: a
/// checksum failure here is a surviving torn page and therefore corruption.
pub fn apply_recovery_delta(
    props: &dyn DeltaPageStore,
    records: &dyn DeltaPageStore,
    dpt: &DirtyPageTable,
    op: &DeltaOp,
    commit_lsn: Lsn,
) -> ArcGraphResult<RecoveryDeltaOutcome> {
    op.validate_shape()?;
    if op.store_id == STORE_BLOB_OVERFLOW {
        return Err(delta_corruption(
            op,
            "blob.overflow store_id 5 is PAGE-IMAGE at M3; physical delta replay is reserved",
        ));
    }
    if !op.kind.is_physical() {
        return Err(delta_corruption(
            op,
            format!("logical op {:?} passed to physical redo", op.kind),
        ));
    }
    let store: &dyn DeltaPageStore = match op.store_id {
        STORE_PROPS => props,
        STORE_RECORD | STORE_TEL | STORE_RELS | STORE_NODE_BINDINGS | STORE_REL_BINDINGS
        | STORE_INTERN | STORE_GRANTS => records,
        other => {
            return Err(delta_corruption(
                op,
                format!("store_id {other} is outside the M3 delta set"),
            ));
        }
    };
    let page_id = PageId::new(op.page_no);
    let present = store.read_page_for_redo(op.tenant_id, page_id)?;
    let missing = present
        .as_ref()
        .is_none_or(|page| page.iter().all(|byte| *byte == 0));

    if missing {
        if op.kind != DeltaOpKind::PageAlloc {
            return Err(delta_corruption(
                op,
                format!(
                    "delta {:?} references absent store {} page {} tenant {:?} with no preceding PageAlloc",
                    op.kind, op.store_id, op.page_no, op.tenant_id
                ),
            ));
        }
        let page_type = PageType::from_byte(op.payload[0]).map_err(|error| {
            delta_corruption(op, format!("PageAlloc carries invalid page type: {error}"))
        })?;
        let mut bytes: Box<PageBuf> = Box::new([0; PAGE_SIZE]);
        if page_type == PageType::Tel {
            let mut header = PageHeader::new(page_id, page_type, op.tenant_id);
            header.lsn = op.op_lsn.raw();
            header.free_space = (PAGE_SIZE - PageHeader::SIZE) as u16;
            header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
            bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        } else {
            let mut header = PageHeader::new(page_id, page_type, op.tenant_id);
            if crate::owner_row::is_owner_store_id(op.store_id) {
                header.flags = op.store_id;
            }
            let mut page = SlottedPage::init(bytes.as_mut(), header).map_err(|error| {
                delta_corruption(op, format!("PageAlloc format failed: {error}"))
            })?;
            page.apply_redo_if_newer(op.op_lsn, |_page| Ok::<(), std::convert::Infallible>(()))
                .expect("infallible PageAlloc stamp");
        }
        store.install_page_from_redo(op.tenant_id, page_id, bytes)?;
        dpt.mark_dirty(
            DirtyPageKey {
                tenant_id: op.tenant_id,
                store_id: op.store_id,
                page_no: op.page_no,
            },
            op.op_lsn,
        );
        return Ok(RecoveryDeltaOutcome::Formatted);
    }

    let mut bytes = present.expect("missing handled above");
    let page_lsn = if op.store_id == STORE_TEL && op.kind == DeltaOpKind::PageAlloc {
        let header_bytes: &[u8; PageHeader::SIZE] = bytes[..PageHeader::SIZE]
            .try_into()
            .expect("page header has fixed size");
        let header = PageHeader::from_bytes(header_bytes).map_err(|error| {
            delta_corruption(
                op,
                format!("torn/invalid TEL page survived DWB restore: {error}"),
            )
        })?;
        if header.page_id != op.page_no
            || header.tenant_id != op.tenant_id.raw()
            || header.page_type != PageType::Tel.as_byte()
            || crc32c::crc32c(&bytes[PageHeader::SIZE..]) != header.checksum
        {
            return Err(delta_corruption(
                op,
                "torn/invalid TEL page survived DWB restore",
            ));
        }
        Lsn::new(header.lsn)
    } else {
        SlottedPage::open(bytes.as_mut())
            .map_err(|error| {
                delta_corruption(
                    op,
                    format!("torn/invalid page survived DWB restore: {error}"),
                )
            })?
            .page_lsn()
    };
    if op.kind == DeltaOpKind::PageAlloc {
        if page_lsn.raw() >= op.op_lsn.raw() {
            return Ok(RecoveryDeltaOutcome::Idempotent);
        }
        return Err(delta_corruption(
            op,
            "PageAlloc encountered an already-formatted page with an older page_lsn",
        ));
    }

    if !apply_physical_delta(bytes.as_mut(), op, commit_lsn)? {
        return Ok(RecoveryDeltaOutcome::Idempotent);
    }
    store.install_page_from_redo(op.tenant_id, page_id, bytes)?;
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: op.tenant_id,
            store_id: op.store_id,
            page_no: op.page_no,
        },
        op.op_lsn,
    );
    Ok(RecoveryDeltaOutcome::Applied)
}

/// Physical page identity used by the M3 DPT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirtyPageKey {
    pub tenant_id: TenantId,
    pub store_id: u16,
    pub page_no: u64,
}

/// A stable DPT observation captured for a flush pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyPageSnapshot {
    pub key: DirtyPageKey,
    /// First redo LSN that dirtied the page since its last successful
    /// home flush (ARIES recLSN).
    pub rec_lsn: Lsn,
    /// Table-wide monotonic dirty generation/stamp. Changes on every dirty
    /// mark and is never reused during this DPT's lifetime, including after a
    /// key is removed and reinserted. Together with `key`, this is the full
    /// compare-and-remove identity.
    pub dirty_gen: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirtyPageEntry {
    rec_lsn: Lsn,
    dirty_gen: u64,
}

/// Concurrent dirty-page table. Memory is O(currently dirty pages),
/// never O(pages ever touched).
#[derive(Debug, Default)]
pub struct DirtyPageTable {
    entries: DashMap<DirtyPageKey, DirtyPageEntry>,
    /// Last allocated dirty generation. This clock is table-wide so a vacant
    /// insert cannot reset a removed key to a previously completed token.
    dirty_gen_clock: AtomicU64,
}

impl DirtyPageTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn next_dirty_gen(&self) -> u64 {
        let previous = self
            .dirty_gen_clock
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("dirty generation exhausted");
        previous + 1
    }

    /// Insert a first-dirty entry or bump its generation on re-dirty.
    /// The original recLSN is retained until a generation-matching
    /// flush removes the entry.
    pub fn mark_dirty(&self, key: DirtyPageKey, op_lsn: Lsn) -> DirtyPageSnapshot {
        use dashmap::mapref::entry::Entry;

        let entry = match self.entries.entry(key) {
            Entry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                debug_assert!(
                    op_lsn.raw() >= entry.rec_lsn.raw(),
                    "redo apply regressed below the page's recLSN"
                );
                entry.dirty_gen = self.next_dirty_gen();
                *entry
            }
            Entry::Vacant(vacant) => {
                let entry = DirtyPageEntry {
                    rec_lsn: op_lsn,
                    dirty_gen: self.next_dirty_gen(),
                };
                vacant.insert(entry);
                entry
            }
        };
        DirtyPageSnapshot {
            key,
            rec_lsn: entry.rec_lsn,
            dirty_gen: entry.dirty_gen,
        }
    }

    /// Mark a page dirty by an op whose LSN may be BELOW the page's current
    /// recLSN — the ARIES `rec_lsn` must cover the EARLIEST unflushed change, so
    /// it is LOWERED to `min(existing, op_lsn)`. Used for the empty-slot extent
    /// directory install (a committed op-100 into a page first-dirtied at 200):
    /// without the lowering a checkpoint captures recLSN=200, redo starts at 200,
    /// and the durable op-100 install is a redo-hole = silent data loss. Redo
    /// idempotence on replay is by ENTRY-PRESENCE (see extent apply), so replaying
    /// the already-applied op from the lowered window is a clean Idempotent skip.
    pub fn mark_dirty_covering(&self, key: DirtyPageKey, op_lsn: Lsn) -> DirtyPageSnapshot {
        use dashmap::mapref::entry::Entry;
        let entry = match self.entries.entry(key) {
            Entry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                entry.rec_lsn = Lsn::new(entry.rec_lsn.raw().min(op_lsn.raw()));
                entry.dirty_gen = self.next_dirty_gen();
                *entry
            }
            Entry::Vacant(vacant) => {
                let entry = DirtyPageEntry {
                    rec_lsn: op_lsn,
                    dirty_gen: self.next_dirty_gen(),
                };
                vacant.insert(entry);
                entry
            }
        };
        DirtyPageSnapshot {
            key,
            rec_lsn: entry.rec_lsn,
            dirty_gen: entry.dirty_gen,
        }
    }

    /// Restore an ARIES DPT captured by an incremental checkpoint. The
    /// metadata decoder has already validated key uniqueness, recLSNs, and
    /// generations; this method preserves the generation tokens exactly.
    pub fn restore(&self, snapshots: &[DirtyPageSnapshot]) {
        // A restored token may be the largest value this process has seen.
        // Advance, never reset, the runtime allocator before publishing the
        // entries so every later dirty mark receives a distinct token. Restore
        // itself is an exclusive bootstrap operation, as it was before this
        // clock existed.
        let restored_high_water = snapshots
            .iter()
            .map(|snapshot| snapshot.dirty_gen)
            .max()
            .unwrap_or(0);
        self.dirty_gen_clock
            .fetch_max(restored_high_water, Ordering::Relaxed);
        self.entries.clear();
        for snapshot in snapshots {
            self.entries.insert(
                snapshot.key,
                DirtyPageEntry {
                    rec_lsn: snapshot.rec_lsn,
                    dirty_gen: snapshot.dirty_gen,
                },
            );
        }
    }

    /// Snapshot current entries for one flush pass. Ordering is stable
    /// for deterministic tests and checkpoint metadata.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DirtyPageSnapshot> {
        let mut snapshots: Vec<_> = self
            .entries
            .iter()
            .map(|entry| DirtyPageSnapshot {
                key: *entry.key(),
                rec_lsn: entry.value().rec_lsn,
                dirty_gen: entry.value().dirty_gen,
            })
            .collect();
        snapshots.sort_by_key(|entry| entry.key);
        snapshots
    }

    /// M6.1 — point lookup for one key, used by the eviction driver's
    /// MECH-E2 handshake (enqueue the victim's qualified key on the
    /// write-behind checkpointer's priority flush): the driver needs a
    /// single key's current `(rec_lsn, dirty_gen)` observation, not a
    /// whole-table snapshot. Returns `None` if the key is not (or no
    /// longer) dirty — the eviction driver treats that as "already
    /// covered by a durable home", i.e. immediately reclaimable.
    #[must_use]
    pub fn snapshot_key(&self, key: DirtyPageKey) -> Option<DirtyPageSnapshot> {
        self.entries.get(&key).map(|entry| DirtyPageSnapshot {
            key,
            rec_lsn: entry.rec_lsn,
            dirty_gen: entry.dirty_gen,
        })
    }

    /// Clear a flushed page only if its key and monotonic dirty stamp still
    /// match the caller's snapshot. Returns true iff that exact entry was
    /// removed. Because stamps never reset, a remove/reinsert ABA cannot match.
    pub fn complete_flush(&self, flushed: DirtyPageSnapshot) -> bool {
        self.entries
            .remove_if(&flushed.key, |_, current| {
                current.dirty_gen == flushed.dirty_gen
            })
            .is_some()
    }

    /// ARIES redo/prune anchor: minimum recLSN still in the DPT, or
    /// `checkpoint_lsn` when every page is clean.
    #[must_use]
    pub fn redo_lsn(&self, checkpoint_lsn: Lsn) -> Lsn {
        self.entries
            .iter()
            .map(|entry| entry.value().rec_lsn)
            .min()
            .unwrap_or(checkpoint_lsn)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
