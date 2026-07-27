//! CRUD surface (M2.c) — node/relationship mutations layered on top of
//! the MVCC kernel (`transaction.rs`), the TEL (`tel.rs`), and the
//! buffer pool (`buffer.rs`).
//!
//! This module owns:
//!
//! - Tenant-scoped ID allocation (`CrudStore`).
//! - Property discriminator encoding for the inline-fit case (design-v2
//!   §3.2).
//! - Translation of `records::PageError` into the public
//!   [`CrudError`] at the CRUD boundary (so the core error enum stays
//!   frozen — see ADR-011 §"Core error freeze" and the M2.c kickoff).
//!
//! The canonical MVCC payload for a node is its `NodeRecord::to_bytes()`
//! serialization, written through [`Transaction::write`]. Readers
//! consult the MVCC version chain at their snapshot LSN; page
//! materialization lands in M2-23 (`read_node`) which pins a buffer-pool
//! page and decodes via `records::SlottedPage`. M2-21 deliberately stays
//! MVCC-only so snapshot visibility is the single observable surface and
//! the page codec can be wired independently.
//!
//! Latency / memory budget (§4.4, 5 K TPS envelope, per create_node):
//!
//! - `CrudStore::alloc_node`: one DashMap get-or-insert (amortized ≤ 50
//!   ns) + one `AtomicU64::fetch_add` (≤ 10 ns on x86_64). Steady-state
//!   hits the DashMap hit path (no insert) and the atomic alone.
//! - `create_node`: one 64-byte stack serialization + one
//!   `Bytes::copy_from_slice` (64 B heap alloc) + `Transaction::write`
//!   (one HashMap insert). ≤ 200 ns warm-cache is a comfortable upper
//!   bound; 20 K create_node ops/s (§4.4 target) is ≈ 50 μs amortized
//!   per op, i.e. ~250× over budget — room for the page write coming
//!   in M2-23 plus the WAL append coming in M2-WAL.

use arcgraph_core::{
    ArcGraphError, LabelId, Lsn, NodeId, NodeRecord, PAGE_SIZE, PageId, RelId, RelRecord, StringId,
    TelEntry, TenantId, TypeId,
};
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use crate::addressed_store::{AddressedRecordStore, AddressedStoreError};
use crate::blob::{BlobError, BlobStore};
use crate::mutation_log::{IndexHandle, PageStoreKind, TxnMutationLog};
use crate::page_alloc::PageAllocator;
use crate::page_store::{RecordPageBackend, RecordPageKey};
use crate::primary_index::{
    BootstrapStats, IndexError, PageSlot, PrimaryIndex, PrimaryKey, RecordKind,
};
use crate::property::{
    InlineShape, encode_inline_node, encode_inline_rel, encode_overflow_node, encode_overflow_rel,
};
use crate::record_store::{RecordPageStore, RecordStoreError};
use crate::records::{NODE_CAPACITY, REL_CAPACITY, SlotId, SlottedPage, SlottedPageRef};
use crate::redo::{DirtyPageKey, DirtyPageTable};
use crate::secondary_handle::{SecondaryIndexHandle, SecondaryIndexValue};
use crate::tel::{MAX_BLOCK_BYTES, MIN_BLOCK_BYTES, TelBlock, TelError, next_block_size};
use crate::transaction::{MvccKey, Transaction, TxnManager};
use crate::vector_store::VectorPageStoreHandle;
use crate::wal::WalHandle;
use crate::wal::bundle::{AclGrantEntry, AclGrantOp};
use crate::wal::bundle::{
    AllocatorAdvance, AllocatorKind, SideChannelWrite, StagedEmit, VectorPageEntry,
};
use crate::wal::delta::{DeltaIntent, DeltaOp, DeltaOpKind, STORE_PROPS, STORE_RECORD};
use arcgraph_core::PageType;

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Property-encoding failures surfaced at the CRUD boundary.
///
/// Kept local to `crud.rs` rather than being added to `ArcGraphError`
/// so the core error taxonomy stays frozen for M2.c. Callers translate
/// via `From<PropError> for CrudError`.
///
/// M2-31 delivers the BLOB store, so `OverflowNotYetImplemented` is now
/// unreachable from the CRUD path (create_* routes `PropertyData::Blob`
/// through [`BlobStore`]). The variant is retained for one milestone
/// to keep the error taxonomy compatible with M2.c consumers; it will
/// be removed in M2.e after the cutover lands.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PropError {
    /// Historic M2.c rejection for out-of-line payloads. Retained but
    /// no longer produced by the CRUD path.
    #[error(
        "property payload did not fit inline; out-of-line storage should route through BlobStore"
    )]
    OverflowNotYetImplemented,
}

/// Public CRUD error type. Every fallible CRUD call returns this.
///
/// The variants intentionally do not wrap `ArcGraphError` directly —
/// the MVCC kernel returns `ArcGraphError` from `commit()`, and that
/// flows through [`Transaction::commit`] unchanged; CRUD-layer faults
/// (property encoding, page codec, id-space exhaustion) surface here.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CrudError {
    /// Property encoding failed (see [`PropError`]).
    #[error(transparent)]
    Property(#[from] PropError),

    /// BlobStore operation failed during [`PropertyData::Blob`] publish
    /// or read-back.
    #[error(transparent)]
    Blob(#[from] BlobError),

    /// Tenant's 63-bit node-id space is exhausted. At 1 M allocations/s
    /// this takes ~292 000 years, so this is a pure safety net.
    #[error("node id space exhausted for tenant {tenant:?}")]
    NodeIdExhausted {
        /// Affected tenant.
        tenant: TenantId,
    },

    /// TEL block ran out of room and could neither grow nor accept an
    /// overflow link. This is fatal for the caller's `create_rel` and
    /// indicates an invariant violation inside the TEL chain manager.
    #[error("TEL append failed: {0}")]
    Tel(#[from] TelError),

    /// MVCC kernel rejected the commit (write-write conflict or similar).
    /// Re-exported so CRUD callers have a single error type to match on.
    #[error("MVCC commit failed: {0}")]
    Mvcc(#[from] ArcGraphError),

    /// Mutating operation (update_*, delete_*) targeted an id that has no
    /// version visible at the transaction's snapshot. Silent upsert on
    /// update or silent success on delete are both footguns; we surface
    /// the mismatch explicitly so callers can treat it as a logic error.
    #[error("{kind} id {id} not found in tenant {tenant:?}")]
    NotFound {
        /// Record kind: `"node"` or `"rel"`.
        kind: &'static str,
        /// Raw id value (node id or rel id).
        id: u64,
        /// Affected tenant.
        tenant: TenantId,
    },

    /// Slotted-record store failure during dual-write install / read.
    #[error("record store: {0}")]
    RecordStore(#[from] RecordStoreError),

    /// Primary-index failure during dual-write publish / lookup.
    #[error("primary index: {0}")]
    Index(#[from] IndexError),

    /// Slotted-page codec failure during dual-write install / read.
    #[error("record page codec: {0}")]
    RecordPage(#[from] crate::records::PageError),

    /// Direct-addressed record-store failure during alternate publish/read.
    #[error("direct-addressed record store: {0}")]
    AddressedStore(#[from] AddressedStoreError),

    /// M4 logical-owner encoding/index/planning failure. Filesystem failures
    /// remain typed all the way to the ingest caller.
    #[error("M4 owner store: {0}")]
    OwnerRow(#[from] crate::owner_row::OwnerRowError),

    /// A page pinned through the index fast path was tagged with a
    /// tenant that does not match the transaction's tenant. Defensive
    /// — this should never fire in correct dual-write code.
    #[error("tenant mismatch on page {page_id:?}: got {got:?}, expected {expected:?}")]
    TenantMismatch {
        /// The page whose header disagrees with the reader's tenant.
        page_id: PageId,
        /// Tenant read from the page header.
        got: TenantId,
        /// Tenant carried by the reading transaction.
        expected: TenantId,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Property data
// ─────────────────────────────────────────────────────────────────────

/// Property payload presented by callers.
///
/// Encoding (design-v2 §3.2):
///
/// - [`PropertyData::Empty`] → `property_ref = 0`, inline fields zeroed.
/// - [`PropertyData::InlineU32Pair`] → stored in
///   `inline_u32a` / `inline_u32b`, `property_ref = 0`,
///   `HAS_EXTENDED = 0`.
/// - [`PropertyData::Blob`] → published to the crate-local
///   [`BlobStore`] as an overflow chain; the record carries a
///   `BlobRef` in its `property_ref` slot (M2-31).
/// - [`PropertyData::TypedBlock`] → v2 M2 (ADR-230 row M2): the typed
///   inline property block. Two-phase staged inside the owning
///   transaction — overflow payload first (its `BlobRef` patches the
///   block tail via [`crate::prop_block::patch_overflow_tail`]), then
///   the block — both through the SAME `stage_bag` machinery as
///   `Blob`, so atomicity, WAL amortization, read-your-own-writes and
///   the abort unwind are inherited unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyData {
    /// No properties attached.
    Empty,
    /// Two inline u32s (e.g. a tiny numeric property set cached on the
    /// record).
    InlineU32Pair(u32, u32),
    /// Arbitrary opaque bytes (the M1 JSON bag payload; still readable
    /// forever — the mixed-store dispatch — and still the migration
    /// fixtures' write shape under `ARCGRAPH_M1_FORCE_CHAINED_BAGS`).
    Blob(Vec<u8>),
    /// v2 M2 typed property block + optional overflow payload (design
    /// §M2.1; built by the mcp encode bridge).
    TypedBlock(crate::prop_block::TypedBagParts),
}

impl PropertyData {
    /// Apply this payload to a fresh `NodeRecord`. Blob payloads are
    /// published through `blobs` and stored by reference; inline
    /// payloads land in the record directly.
    ///
    /// Blob publishes go through [`BlobStore::put_and_stage`], which
    /// stages + publishes the chain and returns each page's `PAGE_SIZE`
    /// snapshot for folding into the owning transaction's v3+
    /// `CommitBundle` (N-2 / issue #81). The caller buffers the returned
    /// emits via [`CrudStore::buffer_blob_emits`]; `crud::commit` drains
    /// them through `take_blob_emits` so the blob pages ride the **same
    /// single fsync** as the MVCC + index writes, and WAL replay
    /// reconstructs the chain from the bundle's
    /// [`crate::wal::bundle::BundlePageKind::Blob`] entries.
    ///
    /// # #810 — no per-record `PutBlob` fsync
    ///
    /// The pre-#810 durable path emitted a standalone *synchronous*
    /// [`WalRecordType::PutBlob`](crate::wal::WalRecordType) record here
    /// (`put_logged_and_stage`), which BLOCKS until fsync on the Strict
    /// tier — once PER record. Because every property-bearing record
    /// encodes as a `Blob` (`property_data_for_json_map`), a batched
    /// `graph.ingest` of N records fsynced N+1 times (one PutBlob per
    /// record + the CommitBundle), collapsing durable bulk-load to
    /// ~170 rec/s. That PutBlob record was redundant: the bundle already
    /// carries (and replay already reconstructs from) the same blob
    /// pages. Dropping it folds the N per-record fsyncs into the single
    /// commit fsync with byte-identical durable state, and tightens
    /// atomicity — a record and its blob pages now land in one
    /// all-or-nothing bundle rather than across a PutBlob fsync + a later
    /// commit fsync.
    ///
    /// Returns the chain pages' `StagedEmit`s (all tagged
    /// [`crate::wal::bundle::BundlePageKind::Blob`]); empty for
    /// inline payloads.
    fn apply_to_node(
        &self,
        rec: &mut NodeRecord,
        tenant: TenantId,
        blobs: &BlobStore,
        txn_id: u64,
    ) -> Result<Vec<StagedEmit>, CrudError> {
        match self {
            PropertyData::Empty => {
                encode_inline_node(InlineShape::U32Pair(0, 0), rec);
                Ok(Vec::new())
            }
            PropertyData::InlineU32Pair(a, b) => {
                encode_inline_node(InlineShape::U32Pair(*a, *b), rec);
                Ok(Vec::new())
            }
            PropertyData::Blob(bytes) => {
                // v2 M1 (ADR-230 / design §M1.2): small bags
                // (≤ PROP_BAG_MAX_BYTES) pack into the transaction's
                // private shared slotted page — the returned emits are
                // EMPTY and the page image is captured ONCE per bundle
                // by `crud::commit` (`snapshot_txn_slotted_pages`), the
                // ~14× batch WAL amortization. Larger bags keep the
                // DEC-4 chain path unchanged (#810 stage-only: the
                // returned chain pages ride the owning transaction's
                // single CommitBundle fsync; no per-record PutBlob).
                let (blob_ref, pages) = blobs.stage_bag(tenant, txn_id, bytes)?;
                encode_overflow_node(blob_ref, rec);
                Ok(blob_pages_to_emits(pages))
            }
            PropertyData::TypedBlock(parts) => {
                let (blob_ref, emits) = stage_typed_block(parts, tenant, blobs, txn_id)?;
                encode_overflow_node(blob_ref, rec);
                Ok(emits)
            }
        }
    }

    /// Apply to a fresh `RelRecord`. Mirror of `apply_to_node` — see
    /// that method's rustdoc for the #810 stage-only durability contract
    /// (blob pages ride the owning transaction's single `CommitBundle`
    /// fsync; no per-record `PutBlob` fsync).
    fn apply_to_rel(
        &self,
        rec: &mut RelRecord,
        tenant: TenantId,
        blobs: &BlobStore,
        txn_id: u64,
    ) -> Result<Vec<StagedEmit>, CrudError> {
        match self {
            PropertyData::Empty => {
                encode_inline_rel(InlineShape::U32Pair(0, 0), rec);
                Ok(Vec::new())
            }
            PropertyData::InlineU32Pair(a, b) => {
                encode_inline_rel(InlineShape::U32Pair(*a, *b), rec);
                Ok(Vec::new())
            }
            PropertyData::Blob(bytes) => {
                // v2 M1: small bags pack slotted; larger bags keep the
                // DEC-4 chain (#810 stage-only). See `apply_to_node`.
                let (blob_ref, pages) = blobs.stage_bag(tenant, txn_id, bytes)?;
                encode_overflow_rel(blob_ref, rec);
                Ok(blob_pages_to_emits(pages))
            }
            PropertyData::TypedBlock(parts) => {
                let (blob_ref, emits) = stage_typed_block(parts, tenant, blobs, txn_id)?;
                encode_overflow_rel(blob_ref, rec);
                Ok(emits)
            }
        }
    }
}

/// v2 M2 — stage one typed property block (+ its optional overflow
/// payload) inside `txn_id`, returning the BLOCK's [`BlobRef`] (what
/// the record's `property_ref` carries) and the combined chain-page
/// emits.
///
/// Ordering is load-bearing: the OVERFLOW payload stages FIRST so its
/// `BlobRef` can be patched into the block's tail before the block
/// itself stages — both inside the SAME transaction scratch, so the
/// pair commits (one bundle, one fsync) or unwinds (rollback restores
/// the scratch pre-images) atomically. Design §M2.1's "overflow tail";
/// the two-phase flow mirrors the mcp encode bridge's
/// `EncodedPropBlock::into_block_bytes_deferred` contract.
///
/// Budget (PD#5): identical to two `Blob` stagings of the same byte
/// volume — no extra copies beyond the 8-byte tail patch.
fn stage_typed_block(
    parts: &crate::prop_block::TypedBagParts,
    tenant: TenantId,
    blobs: &BlobStore,
    txn_id: u64,
) -> Result<(crate::property::BlobRef, Vec<StagedEmit>), CrudError> {
    let mut emits: Vec<StagedEmit> = Vec::new();
    let block_bytes: std::borrow::Cow<'_, [u8]> = match &parts.overflow {
        None => std::borrow::Cow::Borrowed(&parts.block),
        Some(overflow) => {
            let (oref, opages) = blobs.stage_bag(tenant, txn_id, overflow)?;
            emits.extend(blob_pages_to_emits(opages));
            let mut block = parts.block.clone();
            crate::prop_block::patch_overflow_tail(&mut block, oref).map_err(|e| {
                CrudError::Blob(crate::blob::BlobError::SlotStage(format!(
                    "typed-block tail patch failed: {e}"
                )))
            })?;
            std::borrow::Cow::Owned(block)
        }
    };
    let (blob_ref, pages) = blobs.stage_bag(tenant, txn_id, &block_bytes)?;
    emits.extend(blob_pages_to_emits(pages));
    Ok((blob_ref, emits))
}

/// N-2 (issue #81) helper: wrap raw blob chain page bytes into
/// `StagedEmit`s tagged [`crate::wal::bundle::BundlePageKind::Blob`]
/// so the v3 commit bundle codec routes them through
/// [`crate::wal::PageStoreTarget`]'s blob leg on replay.
fn blob_pages_to_emits(pages: Vec<crate::blob::BlobPageSnapshot>) -> Vec<StagedEmit> {
    pages
        .into_iter()
        .map(|(page_id, bytes)| StagedEmit {
            kind: crate::wal::bundle::BundlePageKind::Blob,
            page_id,
            bytes,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// MVCC key namespace
// ─────────────────────────────────────────────────────────────────────

/// Bit set on the `MvccKey` to tag a relationship-id key. Nodes live in
/// the low half (bit 63 = 0), rels in the high half (bit 63 = 1). This
/// keeps node and rel MVCC chains in disjoint key spaces without
/// requiring a second version map.
///
/// Introduced in M2-21 (node-only writer) so M2-22's `create_rel` can
/// use it without a retroactive key-space rewrite.
pub const REL_TAG_BIT: u64 = 1u64 << 63;

/// Project a `NodeId` into the MVCC key namespace.
///
/// `pub` so the recovery-reconcile integration tests (#1380) can install
/// a node record directly into MVCC (modelling the warn-and-continue
/// dual-write degrade where the MVCC side committed but the index install
/// was skipped) from crates outside `arcgraph-storage`.
#[inline]
#[must_use]
pub fn node_mvcc_key(id: NodeId) -> MvccKey {
    debug_assert!(
        id.raw() & REL_TAG_BIT == 0,
        "node ids with the top bit set collide with the rel tag bit",
    );
    id.raw()
}

/// Project a `RelId` into the MVCC key namespace. Sets [`REL_TAG_BIT`]
/// so rel chains never collide with node chains.
///
/// `pub` for the same reason as [`node_mvcc_key`] (the #1380
/// recovery-reconcile integration tests).
#[inline]
#[must_use]
pub fn rel_mvcc_key(id: RelId) -> MvccKey {
    debug_assert!(
        id.raw() & REL_TAG_BIT == 0,
        "rel ids must be allocated with top bit clear; allocator sets the tag",
    );
    id.raw() | REL_TAG_BIT
}

/// Issue #129 P0 fix: monotonic-max raise of a per-tenant
/// `AtomicU64` counter inside a `DashMap<TenantId, AtomicU64>`.
/// Get-or-create the counter, then `compare_exchange_weak` until
/// `counter >= target` (idempotent; double-replay is a no-op).
///
/// Used by [`CrudStore::seed_node_from_advance`] /
/// [`CrudStore::seed_rel_from_advance`] during WAL replay so post-
/// recovery `alloc_node` / `alloc_rel` cannot reuse an id a
/// pre-fault commit consumed.
fn seed_atomic_counter_max(map: &DashMap<TenantId, AtomicU64>, tenant: TenantId, target: u64) {
    // Cold path: get-or-create the counter at `target`.
    if !map.contains_key(&tenant) {
        // Race-tolerant insert: a concurrent `alloc_*` may insert
        // first, in which case the entry below sees the existing
        // counter and we cmpxchg-up below.
        map.entry(tenant).or_insert_with(|| AtomicU64::new(target));
    }
    if let Some(entry) = map.get(&tenant) {
        let counter = entry.value();
        let mut cur = counter.load(Ordering::Acquire);
        while cur < target {
            match counter.compare_exchange_weak(cur, target, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// TEL chain side-store
// ─────────────────────────────────────────────────────────────────────

/// One TEL adjacency chain: the head block (newest) followed by a
/// linked list of older overflow blocks. Each block has a unique
/// synthetic `PageId` ([`CrudStore::next_virtual_page`]); the
/// [`TelBlock::set_prev_block_ptr`] link encodes the chain in the
/// blocks themselves and the store holds `Arc`s keyed by that
/// synthetic id so scans can walk backward.
///
/// Per ADR-018, blocks are in-memory `Arc<TelBlock>` for the v1.0
/// alpha; M2.d moves them into buffer-pool pages alongside record
/// page flushing. `TelBlock`'s single-writer discipline is honored
/// externally by the `Mutex<TelChain>` guarding the chain; the
/// debug-only `write_in_progress` guard inside `TelBlock::append`
/// will still trip if anyone bypasses the mutex.
#[derive(Debug)]
struct TelChain {
    /// Newest block, into which appends target.
    head: Arc<TelBlock>,
    /// Synthetic page id of the head block. Stored here so we can
    /// thread it into `set_prev_block_ptr` on the successor when the
    /// head rolls over into an overflow predecessor.
    head_page: PageId,
}

/// Buffered `create_rel` operation. The TEL append is deferred until
/// `crud::commit` runs, at which point the MVCC install has stamped
/// a real `commit_lsn` and we can write the TEL entry with that LSN.
#[derive(Debug, Clone)]
struct PendingTelAppend {
    tenant: TenantId,
    src: NodeId,
    dst: NodeId,
    rel: RelId,
    /// Channel id for the TEL chain. We key chains by
    /// `(tenant, src, channel)`; M2-22 uses the relationship's
    /// [`TypeId`] as the channel so scan_out can filter by type
    /// without a second pass.
    channel: LabelId,
}

// ─────────────────────────────────────────────────────────────────────
// CrudAllocatorSeedHandle — issue #129 P0 fix replay wiring
// ─────────────────────────────────────────────────────────────────────

/// Wrapper that implements [`crate::wal::AllocatorSeedHandle`] by
/// dispatching `AllocatorAdvance` entries from a v4 `CommitBundle`
/// to the matching CRUD-layer allocator: `Node` / `Rel` route to
/// the wrapped [`CrudStore`]; `Page*` variants route to the wrapped
/// [`PageAllocator`].
///
/// Use [`crud_allocator_seed_handle`] to build one from `Arc<...>`
/// and pass it via [`crate::wal::PageStoreTarget::with_allocator_seed`]
/// during recovery so post-replay `alloc_node` / `alloc_rel` /
/// fresh-page allocations cannot reuse an id a pre-fault commit
/// consumed (ADR-034 D-1 restored).
pub struct CrudAllocatorSeedHandle {
    store: Arc<CrudStore>,
    allocator: Arc<PageAllocator>,
    /// M4 page-backed string allocator. `None` on the resident (pre-M4) arm,
    /// whose intern ids are rebuilt from the checkpoint rather than durably
    /// allocated.
    intern: Option<Arc<crate::intern::InternTable>>,
    /// M4 page-backed ACL class allocator. `None` on the resident arm.
    permissions: Option<Arc<crate::permissions::PermissionIndex>>,
}

impl crate::wal::AllocatorSeedHandle for CrudAllocatorSeedHandle {
    fn seed_from_advance(&self, advance: AllocatorAdvance) {
        match advance.kind {
            AllocatorKind::Node | AllocatorKind::Rel => {
                self.store.apply_allocator_advance(advance);
            }
            AllocatorKind::PageFree
            | AllocatorKind::PageNode
            | AllocatorKind::PageRel
            | AllocatorKind::PageTel
            | AllocatorKind::PageIndexInternal
            | AllocatorKind::PageIndexLeaf
            | AllocatorKind::PageVectorNeighbor
            | AllocatorKind::PageWalBuffer
            | AllocatorKind::PageIndexOverflow
            | AllocatorKind::PagePropSlotted => {
                self.allocator.apply_allocator_advance(advance);
            }
            // M4: these were DROPPED pre-fix. The in-RAM counters are seeded at
            // bootstrap from the last CHECKPOINTED allocator marker, before WAL
            // replay — so without re-driving the replayed advance they stay
            // pinned at the checkpoint high-water and a crash-restart reissues
            // StringIds / AclClassIds that a post-checkpoint commit already
            // committed durably (a never-reissue violation, on the #1404-close
            // surface itself). Seed them exactly as node/rel are seeded.
            AllocatorKind::InternString => {
                if let Some(intern) = &self.intern {
                    intern.seed_string_allocator(advance.tenant, advance.new_high_water);
                }
            }
            AllocatorKind::AclClass => {
                if let Some(permissions) = &self.permissions {
                    permissions.seed_class_allocator(advance.tenant, advance.new_high_water);
                }
            }
        }
    }
}

/// Build a [`CrudAllocatorSeedHandle`] suitable for
/// [`crate::wal::PageStoreTarget::with_allocator_seed`] during WAL
/// recovery. Both `Arc<CrudStore>` and `Arc<PageAllocator>` are
/// the live counters the runtime will consult post-replay.
#[must_use]
pub fn crud_allocator_seed_handle(
    store: Arc<CrudStore>,
    allocator: Arc<PageAllocator>,
) -> Arc<CrudAllocatorSeedHandle> {
    Arc::new(CrudAllocatorSeedHandle {
        store,
        allocator,
        intern: None,
        permissions: None,
    })
}

/// [`crud_allocator_seed_handle`] plus the M4 page-backed owner allocators.
///
/// The durable M4 bootstrap MUST use this variant: without the `intern` /
/// `permissions` handles, replayed `AllocatorAdvance{InternString | AclClass}`
/// entries have nowhere to land, the in-RAM counters stay pinned at the last
/// checkpoint's high-water, and a crash-restart reissues `StringId`s /
/// `AclClassId`s that post-checkpoint commits already durably consumed.
///
/// Pass `None` for either allocator that is not page-backed (the resident,
/// pre-M4 arm rebuilds those ids from the checkpoint instead).
pub fn crud_allocator_seed_handle_with_owners(
    store: Arc<CrudStore>,
    allocator: Arc<PageAllocator>,
    intern: Option<Arc<crate::intern::InternTable>>,
    permissions: Option<Arc<crate::permissions::PermissionIndex>>,
) -> Arc<CrudAllocatorSeedHandle> {
    Arc::new(CrudAllocatorSeedHandle {
        store,
        allocator,
        intern,
        permissions,
    })
}

// ─────────────────────────────────────────────────────────────────────
// CrudStore — tenant-scoped id allocator + TEL chain map
// ─────────────────────────────────────────────────────────────────────

/// Per-database CRUD side state. One instance alongside [`crate::transaction::TxnManager`].
///
/// Holds:
///
/// - Per-tenant node/rel id allocators.
/// - TEL adjacency chains keyed by `(tenant, src, channel)` (M2-22).
///   For v1.0 alpha these are `Arc<TelBlock>` in memory; M2.d moves
///   them into buffer-pool pages (ADR-018).
/// - A per-transaction "pending TEL appends" buffer (M2-22). Entries
///   are drained by [`commit`] after MVCC has stamped the real
///   `commit_lsn`.
///
/// It deliberately does NOT own page allocation — that stays with the
/// buffer pool.
pub struct CrudStore {
    next_node: DashMap<TenantId, AtomicU64>,
    next_rel: DashMap<TenantId, AtomicU64>,
    /// Keyed by `(tenant, src, channel)`. The outer Mutex enforces
    /// single-writer discipline for TEL append (respecting the
    /// debug-only guard inside `TelBlock::append`).
    tel_chains: DashMap<(TenantId, NodeId, LabelId), Arc<Mutex<TelChain>>>,
    /// V11-S-02 / K2 B-3: per `(tenant, src)` TEL channel index for
    /// untyped expands. `tel_heads_for_src` is O(channels_of_src)
    /// instead of O(total_chains); populate is O(channels_of_src)
    /// for the bounded SmallVec dedup scan on first chain creation.
    tel_channels_by_src: DashMap<(TenantId, NodeId), SmallVec<[LabelId; 4]>>,
    /// W26-β-2 / ADR-131 — reverse adjacency index. Keyed by
    /// `(tenant, dst, channel)`. For an edge `(src)-[r:ty]->(dst)`,
    /// this chain at `(tenant, dst, ty)` stores a `TelEntry` where
    /// the `dst_id` field semantically holds the ORIGINAL SRC
    /// (i.e., the neighbor of `dst` on the other end of the edge).
    /// Reuses the [`TelBlock`] / [`TelChain`] primitive shape;
    /// distinct map for structural separation per ADR-131 §D-1
    /// option-2 (forward TEL chain stays canonical).
    ///
    /// Maintained when `reverse_index_enabled` is `true` (the v1.1
    /// default). The commit drain (`crud::commit`) calls
    /// [`Self::tel_append_reverse`] after the forward [`Self::tel_append`].
    /// Scanned by [`scan_in`] for `Direction::RightToLeft` +
    /// `Direction::Undirected` substrate-side expand.
    reverse_tel_chains: DashMap<(TenantId, NodeId, LabelId), Arc<Mutex<TelChain>>>,
    /// Reverse sister of `tel_channels_by_src`, keyed by `(tenant, dst)`,
    /// so untyped inbound expands avoid scanning all reverse chains.
    reverse_tel_channels_by_dst: DashMap<(TenantId, NodeId), SmallVec<[LabelId; 4]>>,
    /// All blocks ever installed, keyed by synthetic page id. Scans
    /// follow `prev_block_ptr` links into this map; heads live here
    /// too for uniform access.
    tel_blocks: DashMap<(TenantId, PageId), Arc<TelBlock>>,
    /// Test/bench instrumentation for proving lazy overflow walks.
    tel_block_fetches: AtomicU64,
    /// W26-β-2 / ADR-131 — reverse-chain block map, parallel to
    /// [`Self::tel_blocks`]. Distinct page-id namespace via the
    /// shared `next_virtual_page` allocator (each reverse block
    /// burns one tick); registry separation keeps overflow walks
    /// strictly within their own direction (a forward `prev_block_ptr`
    /// lookup never resolves a reverse block, and vice versa).
    reverse_tel_blocks: DashMap<(TenantId, PageId), Arc<TelBlock>>,
    /// Test-only instrumentation for proving lazy reverse overflow walks.
    #[cfg(test)]
    reverse_tel_block_fetches: AtomicU64,
    /// Synthetic page-id allocator for TEL blocks. Starts at 1 to
    /// keep `PageId(0)` free as a sentinel.
    next_virtual_page: AtomicU64,
    /// W26-β-2 / ADR-131 — global enable flag for the reverse
    /// adjacency index. `true` at construction (the v1.1 default);
    /// flipping it `false` via [`Self::set_reverse_index_enabled`]
    /// causes both [`Self::tel_append_reverse`] and [`scan_in`] to
    /// short-circuit. Used by AC-4 fault-injection tests to simulate
    /// "post-recovery, reverse index unbuilt" and verify the
    /// substrate surfaces a structured error rather than silent
    /// empty results (per `feedback_load_bearing_pr_requires_fault_injection_tests.md`).
    reverse_index_enabled: std::sync::atomic::AtomicBool,
    /// Per-txn pending TEL appends, drained by `commit`.
    pending_tel: DashMap<u64, Vec<PendingTelAppend>>,
    /// Overflow BLOB store (M2-31). Publishes are synchronous; any
    /// `PropertyData::Blob` routed through `apply_to_*` lands here and
    /// the record carries the returned [`crate::property::BlobRef`].
    ///
    /// Wrapped in `Arc` since N-2 (issue #81) so the WAL replay
    /// target can hold a shared handle to the same store the CRUD
    /// layer writes into. Pre-N-2 the BlobStore was owned inline;
    /// recovery now wires
    /// [`crate::wal::PageStoreTarget::with_blob_store`] against
    /// `Arc::clone(store.blob_store())` so a post-replay
    /// dereference via `BlobStore::get` resolves the freshly
    /// reinstalled chain pages.
    blobs: Arc<BlobStore>,
    /// Optional WAL producer for durable blob publishes. When `Some`,
    /// the CRUD path routes `PropertyData::Blob` through
    /// [`BlobStore::put_logged`] so the `PutBlob` record is fsynced
    /// before the owning MVCC commit (review block C-1 / ADR-022).
    /// When `None`, the no-WAL [`BlobStore::put`] is used — the
    /// intended mode for tests that don't spawn a WAL writer.
    wal: Option<WalHandle>,

    // ---- M2-CUTOVER dual-write wiring ----
    /// Shared page allocator for slotted record pages + primary-index
    /// pages. `None` means dual-write is disabled (the M2.c behavior).
    allocator: Option<Arc<PageAllocator>>,
    /// In-memory slotted-record store hosting `PageType::Node` /
    /// `PageType::Rel` pages. `None` when dual-write is disabled.
    ///
    /// W26-ε-2 / ADR-140 wire-through: when [`Self::new_with_page_store`]
    /// is used, this field carries the inner hot cache of the
    /// [`BufferedRecordPageStore`] — keeping existing CRUD call sites
    /// that read via `self.records` operational while the
    /// `buffered_records` handle exposes the cache+spill substrate for
    /// recovery + ops eviction. See ADR-140 D-3 §"The
    /// `RecordPageBackend` adapter trait".
    records: Option<Arc<RecordPageStore>>,
    /// W26-ε-2 / ADR-140 — cache + spill wrapper around `records`. When
    /// `Some`, WAL replay routes through this handle (via the
    /// [`crate::wal::replay::RecordPageStoreHandle`] impl) and ops can
    /// drive `evict_lru` to bound RSS. When `None` (the legacy
    /// constructors), the in-memory `records` DashMap is the sole
    /// page store — RSS scales linearly with ingested page count
    /// (W22-DB-α-1-cap known issue, closed at v1.0-GA by callers
    /// migrating to the new constructor).
    buffered_records: Option<Arc<crate::page_store::BufferedRecordPageStore>>,
    /// Primary B-tree index for `(tenant, kind, id) → (page, slot)`
    /// lookups. `None` when dual-write is disabled.
    primary: Option<Arc<PrimaryIndex>>,
    /// Optional M4 direct-address target. This slice keeps it alongside the
    /// primary B-tree; the migration swap that makes it authoritative lands
    /// later. Node/rel and tenant page spaces are separated inside the store.
    addressed_records: Option<Arc<AddressedRecordStore>>,
    /// True only after the offline v6 generation swap retires the primary
    /// B-tree. Slice-1 alternate/differential callers leave this false.
    addressed_authoritative: bool,
    /// Secondary B-tree handle — property → NodeId reverse index (M2-34).
    /// `None` when dual-write is enabled but no secondary was
    /// configured (existing callers of `new_with_index` land here).
    /// When `Some`, the CUTOVER drain publishes node-property entries
    /// into the secondary on every `Create` / `Update` / `Delete`
    /// install for `RecordKind::Node`. Rels are not indexed in M2.d.
    secondary: Option<Arc<dyn SecondaryIndexHandle>>,
    /// Per-`(tenant, kind)` "open" page — destination for the next
    /// dual-written record. Advances to a fresh page when current page
    /// is full.
    open_pages: DashMap<(TenantId, RecordKind), PageId>,
    /// M3 allocation shadow for built-but-not-yet-applied record deltas.
    /// Page bytes remain unchanged until WAL durability.
    record_reservations: Mutex<RecordReservationTable>,
    /// Periodic-tier v9 page applies waiting for exact fsync proof, together
    /// with the record coordinates whose authoritative bytes still live in
    /// the MVCC chain rather than the physical page store.
    deferred_v9_applies: Mutex<DeferredV9ApplyQueue>,
    /// One-shot deterministic gate after a deferred front has exact durable
    /// proof and immediately before its physical deltas apply.
    #[cfg(any(debug_assertions, feature = "fault-injection"))]
    debug_deferred_v9_apply_gate: Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Multi-caller test rendezvous immediately before deferred writers
    /// attempt to drain/enqueue, used to hold RULE-MT peers in flight.
    #[cfg(any(debug_assertions, feature = "fault-injection"))]
    debug_deferred_v9_caller_gate: Mutex<Option<DebugDeferredV9CallerGate>>,
    #[cfg(test)]
    fail_durable_v9_apply: std::sync::atomic::AtomicBool,
    m3_dpt: parking_lot::RwLock<Option<Arc<DirtyPageTable>>>,
    /// Per-txn pending record-page installs, drained by `commit`
    /// after `tx.commit()` returns the real `commit_lsn`.
    pending_installs: DashMap<u64, Vec<PendingInstall>>,
    /// Per-txn pending blob chain page snapshots, drained by
    /// `commit` and folded into the v3 `CommitBundle`'s
    /// `staged_pages` section under
    /// [`crate::wal::bundle::BundlePageKind::Blob`]. N-2
    /// (issue #81).
    ///
    /// Populated by `apply_to_node` / `apply_to_rel` when the
    /// caller is a `PropertyData::Blob`: the CRUD layer publishes
    /// the chain into `self.blobs` AND stages the per-page bytes
    /// here so replay can reconstruct the in-memory
    /// [`BlobStore`] from the bundle alone. Without this staging,
    /// a post-replay dereference of `NodeRecord.property_ref` hits
    /// [`BlobError::MissingHead`] because replay never installed
    /// the chain pages into the fresh `BlobStore` instance.
    pending_blob_emits: DashMap<u64, Vec<StagedEmit>>,
    /// M3.a Slice G.4 (commit-bundle vector page staging): per-txn
    /// pending vector arena page snapshots, drained by `commit` and
    /// folded into the v5 `CommitBundle`'s `vector_pages` section.
    ///
    /// Populated by future M3.b vector writers (Slice G.5 / G.7) via
    /// [`Self::stage_vector_page`]: the producer captures the page
    /// bytes under the arena's write latch and pushes a
    /// `PendingVectorEmit` here so replay can reinstall the bytes
    /// via
    /// [`crate::vector_store::VectorPageStoreHandle::install_or_replace`]
    /// in commit_lsn order. Mirrors the `pending_blob_emits` pattern
    /// (DashMap keyed by `txn_id`, drained at commit, discarded on
    /// rollback). Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
    pending_vector_emits: DashMap<u64, Vec<PendingVectorEmit>>,
    /// #352 Part 2 (ADR-199): per-txn pending `external_id →
    /// internal_id` idempotency bindings, drained by `commit` and folded
    /// into the v6 `CommitBundle`'s `idempotency_bindings` section so the
    /// binding is durified atomically with the node/rel write that
    /// allocated `internal_id`. Populated by `arcgraph-mcp`'s ingest path
    /// via [`Self::stage_idempotency_binding`] after `create_node` /
    /// `create_rel` returns the id. Mirrors the `pending_vector_emits`
    /// pattern (DashMap keyed by `txn_id`, drained at commit, discarded
    /// on rollback). The stored entries are
    /// [`crate::wal::bundle::IdempotencyBindingEntry`] directly (no
    /// separate pending type — the wire shape carries no commit_lsn).
    pending_idempotency_bindings: DashMap<u64, Vec<PendingIdempotencyBinding>>,
    /// #1221 (ADR-218): per-txn pending document-level ACL grant/revoke
    /// ops, drained by `commit` and folded into the v8 `CommitBundle`'s
    /// `acl_grants` section so the `PermissionIndex` mutation is durified
    /// atomically with the commit that carries it. Populated by the ACL
    /// write-through (the `AclWalSink` issuing a dedicated single-op
    /// commit; see [`crate::permissions`]) via [`Self::stage_acl_grant`].
    /// Mirrors the `pending_idempotency_bindings` pattern (DashMap keyed
    /// by `txn_id`, drained at commit, discarded on rollback). The stored
    /// entries are [`crate::wal::bundle::AclGrantEntry`] directly (the
    /// wire shape carries no commit_lsn). **Per-txn `Vec` order is
    /// preserved through to the encoder** — the `acl_grants` encoder must
    /// NOT re-sort (ADR-218 last-writer-wins invariant).
    pending_acl_grants: DashMap<u64, Vec<crate::wal::bundle::AclGrantEntry>>,
    /// Shared idempotency store used by delete paths to resolve
    /// `(tenant, kind, internal_id) -> external_id` before staging a
    /// release into the same commit bundle. `None` for legacy/unit
    /// callers that do not use ingest idempotency.
    idempotency_store: parking_lot::RwLock<Option<Arc<crate::IdempotencyStore>>>,
    /// M4 physical metadata owner. Present only after a v10 generation has
    /// opened all owner companions. Its planning latch spans owner intent
    /// construction through Phase-3 apply, preventing two concurrent commits
    /// from assigning conflicting physical extents.
    owner_rows: parking_lot::RwLock<Option<Arc<crate::owner_row::OwnerRowRegistry>>>,
    /// M3.a Slice G.5 (Z-1 (b) rollback dispatch). When `Some`, the
    /// rollback closure restores pre-W vector arena bytes via
    /// [`VectorPageStoreHandle::restore_page_bytes`] on WAL fsync
    /// failure. When `None`, the rollback arm warns-and-skips
    /// (matches the no-wiring-as-pre-M3.a-deployment posture set by
    /// `wal::replay`'s no-store dispatch). Per ADR-033 §6 +
    /// ADR-035 §7.5.
    vector_store: Option<Arc<dyn VectorPageStoreHandle>>,
    /// M3.b ADR-039 §D-6 / §D-7 — BM25 commit-side hook. When `Some`,
    /// the rollback closure drains `log.bm25_pending` and dispatches
    /// [`crate::mutation_log::Bm25IndexStoreHandle::rollback_pending`]
    /// per tenant on WAL fsync failure. When `None`, the rollback arm
    /// warn-and-skips (mirrors the `vector_store` opt-in posture).
    /// Wired via [`Self::with_bm25_store`].
    ///
    /// At v1.0 the `commit_pending` invocation is dormant — the
    /// kernel commit closure in `transaction.rs` is frozen by the
    /// parallel M3.b session boundary, so [`Self::commit_bm25_pending`]
    /// exists as a `pub(crate)` helper that future slices wire into
    /// the kernel. The rollback (load-bearing safety) IS wired in
    /// the closure below.
    bm25_store: Option<Arc<dyn crate::mutation_log::Bm25IndexStoreHandle>>,

    /// M4-41 (M4-04a) — per-tenant catalog stats. Lazily created on
    /// first commit that touches a given tenant; accessible
    /// post-commit via [`Self::catalog_stats`]. Each tenant's
    /// [`CatalogStats`] instance is independent — multi-tenant
    /// isolation is structural per ADR-038 §2 D-25.
    ///
    /// The map itself is keyed by `TenantId`, so DashMap's sharded
    /// locking gives concurrent commits across tenants a contention-
    /// free fast path. Within a single tenant, commits are
    /// serialized by the MVCC kernel (per ADR-031 / ADR-034) so the
    /// inner `CatalogStats` updates also see no contention in
    /// practice; the atomics inside `CatalogStats` are belt-and-
    /// braces against future pipelined-commit work.
    ///
    /// Stats only fire for the dual-write commit path (i.e., when
    /// `primary.is_some()`). The non-dual-write path (used by
    /// pre-M2.d unit tests) does not buffer `PendingInstall`s and
    /// therefore does not trigger stats updates. Production
    /// deployments enable dual-write; this is a known and
    /// acceptable v1.0 limitation.
    catalog_stats: DashMap<TenantId, Arc<crate::catalog::CatalogStats>>,

    /// W28 Feature #582 (ADR-045) — optional observability sink for the
    /// `arcgraph_hot_vertex_warnings_total{tenant}` counter (design-v2
    /// §10.2 **line 721**).
    ///
    /// When `Some`, [`Self::tel_append`] + [`Self::tel_append_reverse`]
    /// fire [`crate::metrics::MetricsSink::record_hot_vertex_warning`]
    /// at the SAME overflow-block-allocation site that already emits
    /// the `tracing::warn!` HOT_VERTEX_WARNING (design-v2 §3.3). When
    /// `None` (the default + every legacy caller), the emit path is a
    /// single nullable-ptr check — no behavior change, zero-overhead
    /// (PD-5). The concrete impl is
    /// `arcgraph-mcp::transport::metrics::MetricsRegistry`; the dep
    /// edge `mcp → storage` (never the inverse) keeps PD-7 bounded
    /// contexts satisfied (the trait is storage-resident).
    ///
    /// This closes the W17δ #313 no-op trampoline: the metric was
    /// *registered* (and the `MetricsRegistry::record_hot_vertex_warning`
    /// method existed) but had NO producer caller — `tel_append` only
    /// logged via `tracing::warn!`. See
    /// `feedback_noop_trampoline_anti_pattern.md`.
    metrics_sink: Option<Arc<dyn crate::metrics::MetricsSink>>,

    /// **RC-1 (secondary property index, #1366)** — snapshot-horizon
    /// queue of secondary-index old-value removals.
    ///
    /// # The false-negative cliff this closes
    ///
    /// The pre-RC-1 commit drain eagerly zeroed the old-value NodeId
    /// slot at commit-builder time (`remove_property_deferred`). A
    /// reader on a snapshot predating the writer's commit — a
    /// concurrent reader, or a long-running txn arbitrarily later —
    /// would look up `email = a` after a writer did `a → b`, find
    /// nothing, and MISS a node visible in its own snapshot. The
    /// candidate-then-verify contract (ADR-023) filters false
    /// positives (ghosts) but is structurally blind to a MISSING
    /// entry: "no entry" is indistinguishable from "no match".
    ///
    /// # The insert-only fix
    ///
    /// Commit-path maintenance is now **insert-only**. Every old-value
    /// removal (from a `SET` value change, `REMOVE`, or `DELETE`) is
    /// enqueued here, stamped with the removing commit's `Lsn`, and
    /// applied to the B-tree only once
    /// [`TxnManager::oldest_active_snapshot`](crate::transaction::TxnManager::oldest_active_snapshot)
    /// has passed that LSN — i.e. once no live snapshot can still
    /// observe the superseded value. Until applied, the superseded
    /// entry is a **ghost**: read-safe, because the mandatory verify
    /// step hydrates the node through the reader's snapshot and drops
    /// the candidate on the `= a` recheck (the node now reads `b`).
    ///
    /// A removal is additionally guarded by the latest successful
    /// re-assertion generation for the exact `(tenant, label, property,
    /// value, node)` entry. If a later commit re-inserted that entry, an
    /// older queued removal is a no-op. In-flight re-assertions are
    /// registered under the same mutex as the check-and-remove step, so
    /// removal cannot race between a newer insert and publication of its
    /// final commit LSN (#1464).
    ///
    /// # Durability posture (intentionally lossy)
    ///
    /// This queue is in-memory only. Losing it on crash strands ghosts
    /// (extra index entries that verify-filter to nothing) — reclaimed
    /// by the #1386-pattern reconcile — but **never** a correctness
    /// violation: a stranded ghost is a false positive, not a false
    /// negative. Per the design memo §Index-class RC-1 amendment we do
    /// NOT add durability to this queue for Phase 0.
    ///
    /// Budget: one `Mutex<DeferredRemovalState>`; drained opportunistically
    /// at the end of `commit` (amortized O(applied) work, gated by the
    /// snapshot horizon so the common case drains everything a long-running
    /// reader is not pinning). `DeferredRemoval` is 40 bytes; a burst of
    /// overwrites while one long reader is pinned bounds the queue by
    /// (writes × declared-index-count) for that reader's lifetime. The
    /// generation map holds the latest update-side assertion for an entry
    /// until a same-or-newer removal supersedes it; it is in-memory and
    /// intentionally shares the queue's lossy-on-crash posture.
    deferred_removals: Mutex<DeferredRemovalState>,
}

/// Exact secondary entry identity used by the #1464 generation guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SecondaryEntry {
    tenant: TenantId,
    label: LabelId,
    property_key: StringId,
    value: SecondaryIndexValue,
    node: NodeId,
}

/// RC-1 removal queue plus the #1464 re-assertion generations that guard it.
///
/// `inflight_reassertions` closes the insert/publication race: an update-side
/// insert and its marker are installed while this mutex is held. A remover
/// either runs first (then the later insert restores the entry) or observes
/// the marker and leaves the removal queued until commit success/failure is
/// known. Successful commits replace the marker with their final LSN;
/// aborted commits simply discard it after secondary-page rollback.
#[derive(Debug, Default)]
struct DeferredRemovalState {
    removals: Vec<DeferredRemoval>,
    latest_reassertions: HashMap<SecondaryEntry, Lsn>,
    inflight_reassertions: HashSet<(u64, SecondaryEntry)>,
}

impl DeferredRemovalState {
    fn register_inflight(&mut self, txn_id: u64, entries: &[SecondaryEntry]) {
        for &entry in entries {
            self.inflight_reassertions.insert((txn_id, entry));
        }
    }

    fn finish_inflight(&mut self, txn_id: u64, commit_lsn: Lsn, entries: &[SecondaryEntry]) {
        for &entry in entries {
            self.inflight_reassertions.remove(&(txn_id, entry));
            self.latest_reassertions
                .entry(entry)
                .and_modify(|latest| {
                    if commit_lsn.raw() > latest.raw() {
                        *latest = commit_lsn;
                    }
                })
                .or_insert(commit_lsn);
        }
    }

    fn discard_inflight(&mut self, txn_id: u64, entries: &[SecondaryEntry]) {
        for &entry in entries {
            self.inflight_reassertions.remove(&(txn_id, entry));
        }
    }

    fn has_inflight(&self, entry: SecondaryEntry) -> bool {
        self.inflight_reassertions
            .iter()
            .any(|(_, candidate)| *candidate == entry)
    }
}

/// **RC-1** — one enqueued secondary-index old-value removal, stamped
/// with the removing commit's `Lsn`. Applied to the B-tree only after
/// `oldest_active_snapshot()` passes `commit_lsn` (no live snapshot can
/// still observe the superseded value). See
/// [`CrudStore::deferred_removals`].
#[derive(Debug, Clone, Copy)]
struct DeferredRemoval {
    /// The commit whose maintenance superseded this entry. The removal
    /// applies only once `oldest_active_snapshot() > commit_lsn`.
    commit_lsn: Lsn,
    tenant: TenantId,
    label: LabelId,
    property_key: StringId,
    value: SecondaryIndexValue,
    node: NodeId,
}

impl DeferredRemoval {
    fn entry(self) -> SecondaryEntry {
        SecondaryEntry {
            tenant: self.tenant,
            label: self.label,
            property_key: self.property_key,
            value: self.value,
            node: self.node,
        }
    }
}

impl Default for CrudStore {
    fn default() -> Self {
        // W26-β-2 / ADR-131 — `reverse_index_enabled` defaults to
        // `true` (v1.1+ posture). The hand-rolled Default impl exists
        // because `AtomicBool::default()` is `false`; the previous
        // `#[derive(Default)]` would have shipped a silently-disabled
        // reverse index. Every other field uses its derived default.
        Self {
            next_node: DashMap::default(),
            next_rel: DashMap::default(),
            tel_chains: DashMap::default(),
            tel_channels_by_src: DashMap::default(),
            reverse_tel_chains: DashMap::default(),
            reverse_tel_channels_by_dst: DashMap::default(),
            tel_blocks: DashMap::default(),
            tel_block_fetches: AtomicU64::default(),
            reverse_tel_blocks: DashMap::default(),
            #[cfg(test)]
            reverse_tel_block_fetches: AtomicU64::default(),
            next_virtual_page: AtomicU64::default(),
            reverse_index_enabled: std::sync::atomic::AtomicBool::new(true),
            pending_tel: DashMap::default(),
            blobs: Arc::default(),
            wal: None,
            allocator: None,
            records: None,
            buffered_records: None,
            primary: None,
            addressed_records: None,
            addressed_authoritative: false,
            secondary: None,
            open_pages: DashMap::default(),
            record_reservations: Mutex::new(RecordReservationTable::default()),
            deferred_v9_applies: Mutex::new(DeferredV9ApplyQueue::default()),
            #[cfg(any(debug_assertions, feature = "fault-injection"))]
            debug_deferred_v9_apply_gate: Mutex::new(None),
            #[cfg(any(debug_assertions, feature = "fault-injection"))]
            debug_deferred_v9_caller_gate: Mutex::new(None),
            #[cfg(test)]
            fail_durable_v9_apply: std::sync::atomic::AtomicBool::new(false),
            m3_dpt: parking_lot::RwLock::new(None),
            pending_installs: DashMap::default(),
            pending_blob_emits: DashMap::default(),
            pending_vector_emits: DashMap::default(),
            pending_idempotency_bindings: DashMap::default(),
            pending_acl_grants: DashMap::default(),
            idempotency_store: parking_lot::RwLock::new(None),
            owner_rows: parking_lot::RwLock::new(None),
            vector_store: None,
            bm25_store: None,
            catalog_stats: DashMap::default(),
            metrics_sink: None,
            deferred_removals: Mutex::new(DeferredRemovalState::default()),
        }
    }
}

/// M3.a Slice G.4: a vector arena page mutation staged by a
/// producer (vector writer in `arcgraph-vector`, post-G.5/G.7),
/// awaiting drain into the v5 `CommitBundle`'s `vector_pages`
/// section by [`commit`]. Carries everything the bundle codec needs
/// EXCEPT the `commit_lsn` (stamped at drain time inside the
/// builder closure when the MVCC commit_lsn is allocated).
///
/// Mirrors [`StagedEmit`] for the blob path. The `partition` and
/// `index_id` fields are reserved for v1.1 (always
/// `PartitionId::ZERO` / `0` at v1.0). See
/// [`crate::wal::bundle::VectorPageEntry`] for the on-wire shape.
#[derive(Debug, Clone)]
pub struct PendingVectorEmit {
    /// Tenant whose arena holds this page.
    pub tenant: TenantId,
    /// Partition slot. v1.0 invariant: always `PartitionId::ZERO`.
    pub partition: arcgraph_core::PartitionId,
    /// Reserved for v1.1 multi-index lift; always 0 at v1.0.
    pub index_id: u64,
    /// The page that was mutated (vector-arena allocator-assigned).
    pub page_id: PageId,
    /// A heap-allocated copy of the page's post-mutation bytes
    /// captured under the arena's write latch.
    pub bytes: Box<[u8; PAGE_SIZE]>,
}

#[derive(Debug, Clone)]
struct PendingIdempotencyBinding {
    entry: crate::wal::bundle::IdempotencyBindingEntry,
    payload_hash: Option<u64>,
}

/// Buffered record-page mutation waiting on its owning txn's `commit_lsn`.
///
/// `Create` allocates a new slot. `Update` rewrites an existing slot
/// in place (primary index entry stays pinned to the same coordinates).
/// `Delete` tombstones the slot AND the primary-index entry.
#[derive(Debug, Clone)]
enum PendingInstall {
    Create {
        tenant: TenantId,
        kind: RecordKind,
        id: u64,
        bytes: Vec<u8>,
        /// ADR-025 §5 — source node label for relationship creates,
        /// captured before commit so the stats hook can maintain
        /// `max_out_degree_sketch[label, rel_type]` without re-reading
        /// storage after the commit. `None` for node creates.
        src_label_raw: Option<u32>,
    },
    Update {
        tenant: TenantId,
        kind: RecordKind,
        id: u64,
        bytes: Vec<u8>,
    },
    Delete {
        tenant: TenantId,
        kind: RecordKind,
        id: u64,
        /// M4-41 — label_id (Node) / type_id (Rel) of the version
        /// being tombstoned, captured pre-delete so the post-commit
        /// stats hook can decrement the right counter without
        /// re-reading the MVCC chain. `None` when the dual-write
        /// caller did not capture it (e.g., a future delete path
        /// that bypasses [`delete_node_with_store`] /
        /// [`delete_rel_with_store`]); the stats hook then
        /// conservatively skips the per-label/per-type decrement
        /// while still updating the tenant-wide totals.
        prior_topology_raw: Option<u32>,
    },
}

impl PendingInstall {
    fn tenant(&self) -> TenantId {
        match self {
            PendingInstall::Create { tenant, .. }
            | PendingInstall::Update { tenant, .. }
            | PendingInstall::Delete { tenant, .. } => *tenant,
        }
    }
    fn kind(&self) -> RecordKind {
        match self {
            PendingInstall::Create { kind, .. }
            | PendingInstall::Update { kind, .. }
            | PendingInstall::Delete { kind, .. } => *kind,
        }
    }
    fn id(&self) -> u64 {
        match self {
            PendingInstall::Create { id, .. }
            | PendingInstall::Update { id, .. }
            | PendingInstall::Delete { id, .. } => *id,
        }
    }
}

#[derive(Debug)]
struct RecordSlotReservation {
    tenant: TenantId,
    page_id: PageId,
    slot: SlotId,
}

#[derive(Debug)]
struct RecordPageShadow {
    tenant: TenantId,
    kind: RecordKind,
    next_slot: u16,
    released_slots: std::collections::BTreeSet<u16>,
    pending_new_owner: Option<u64>,
}

#[derive(Debug, Default)]
struct RecordReservationTable {
    pages: std::collections::HashMap<RecordPageKey, RecordPageShadow>,
    by_txn: std::collections::HashMap<u64, Vec<RecordSlotReservation>>,
}

#[derive(Debug)]
struct DeferredV9Apply {
    txn_id: u64,
    commit_lsn: Lsn,
    deltas: Vec<DeltaOp>,
}

#[cfg(any(debug_assertions, feature = "fault-injection"))]
#[derive(Debug)]
struct DebugDeferredV9CallerGate {
    remaining: usize,
    entered: Arc<std::sync::Barrier>,
    release: Arc<std::sync::Barrier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DeferredRecordSlot {
    page: RecordPageKey,
    slot: SlotId,
}

#[derive(Debug, Default)]
struct DeferredV9ApplyQueue {
    entries: std::collections::VecDeque<DeferredV9Apply>,
    /// A count, rather than a set, because consecutive deferred updates may
    /// touch the same slot. Applying the older bundle must not unmark bytes
    /// that are still superseded by a later queued bundle.
    pending_record_slots: std::collections::HashMap<DeferredRecordSlot, usize>,
}

impl DeferredV9ApplyQueue {
    fn record_slots(deltas: &[DeltaOp]) -> impl Iterator<Item = DeferredRecordSlot> + '_ {
        deltas
            .iter()
            .filter(|delta| {
                delta.store_id == STORE_RECORD
                    && matches!(
                        delta.kind,
                        DeltaOpKind::PutRecord | DeltaOpKind::TombstoneRecord
                    )
            })
            .map(|delta| DeferredRecordSlot {
                page: RecordPageKey::new(delta.tenant_id, PageId::new(delta.page_no)),
                slot: SlotId(delta.slot),
            })
    }

    fn push_back(&mut self, entry: DeferredV9Apply) {
        let touched: std::collections::HashSet<_> = Self::record_slots(&entry.deltas).collect();
        for slot in touched {
            *self.pending_record_slots.entry(slot).or_default() += 1;
        }
        self.entries.push_back(entry);
    }

    fn clear_front_markers(&mut self) {
        let front = self
            .entries
            .front()
            .expect("front applied before marker clear");
        let touched: std::collections::HashSet<_> = Self::record_slots(&front.deltas).collect();
        for slot in touched {
            let count = self
                .pending_record_slots
                .get_mut(&slot)
                .expect("queued record delta must own a pending marker");
            *count -= 1;
            if *count == 0 {
                self.pending_record_slots.remove(&slot);
            }
        }
    }

    fn record_slot_is_pending(&self, tenant: TenantId, page: PageId, slot: SlotId) -> bool {
        self.pending_record_slots.contains_key(&DeferredRecordSlot {
            page: RecordPageKey::new(tenant, page),
            slot,
        })
    }
}

/// Oldest WAL range whose Periodic v9 page apply is still queued.
///
/// Checkpointing samples this under `TxnManager::checkpoint_freeze`: the
/// logical frontier must remain before `commit_lsn`, while physical replay
/// must begin no later than `redo_lsn` so the bundle's PageAlloc is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeferredV9Boundary {
    pub commit_lsn: Lsn,
    pub redo_lsn: Lsn,
}

impl CrudStore {
    /// Empty store, no WAL. Allocators + chains are created lazily on
    /// first use. Use [`Self::with_wal`] to wire a WAL writer for
    /// durable blob publishes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty store with a WAL handle for durable blob publishes.
    /// Production callers that spawn a [`crate::wal::WalWriter`]
    /// should prefer this constructor so every `PropertyData::Blob`
    /// on the CRUD write path emits a `PutBlob` record.
    #[must_use]
    pub fn with_wal(wal: WalHandle) -> Self {
        Self {
            wal: Some(wal),
            ..Self::default()
        }
    }

    /// Attach a WAL handle after durable bootstrap has replayed the existing WAL.
    ///
    /// This preserves the store's page stores, BM25 service, and metrics
    /// sink while enabling WAL-backed blob and intern publishes for
    /// subsequent commits.
    pub fn attach_wal(&mut self, wal: WalHandle) {
        self.wal = Some(wal);
    }

    /// Attach the M3 dirty-page table used by incremental checkpointing.
    ///
    /// v9 commit deltas enter this table only after their exact WAL record has
    /// completed fsync and the corresponding page mutation has been installed.
    /// This preserves the no-steal boundary when the Periodic durability tier
    /// acknowledges a commit before that fsync completes.
    pub fn attach_m3_dirty_page_table(&self, dpt: Arc<DirtyPageTable>) {
        *self.m3_dpt.write() = Some(dpt);
    }

    /// M3.a Slice G.5 — wire a [`VectorPageStoreHandle`] for Z-1 (b)
    /// rollback dispatch. Returns `self` so the call chains onto the
    /// other constructors (`new`, `new_with_index`, `new_with_indices`,
    /// `with_wal`) without duplicating their ad-hoc field shapes.
    ///
    /// Production wiring: every callsite that constructs a
    /// `CrudStore` for a deployment with a vector arena MUST chain
    /// `.with_vector_store(handle)` so the rollback closure has a
    /// dispatch target on WAL fsync failure. When omitted, the
    /// `PageStoreKind::Vector` arm of the rollback closure
    /// warns-and-skips (mirrors the pre-M3.a posture set by
    /// `wal::replay`'s no-store dispatch).
    ///
    /// Per ADR-033 §6 + ADR-035 §7.5.
    #[must_use]
    pub fn with_vector_store(mut self, handle: Arc<dyn VectorPageStoreHandle>) -> Self {
        self.vector_store = Some(handle);
        self
    }

    /// Attach the optional M4 direct-address record store.
    ///
    /// While this slice is active, committed CRUD mutations may publish to
    /// both this target and the primary-B-tree path. The migration slice owns
    /// the later authority swap; attaching this target does not retire or
    /// mutate the primary index.
    #[must_use]
    pub fn with_addressed_record_store(mut self, store: Arc<AddressedRecordStore>) -> Self {
        self.addressed_records = Some(store);
        self
    }

    /// Attach the post-migration M4 authority. This is intentionally a
    /// separate constructor from the pre-swap alternate so on-open code cannot
    /// flip authority without selecting a v6 generation.
    #[must_use]
    pub fn with_authoritative_addressed_record_store(
        mut self,
        store: Arc<AddressedRecordStore>,
    ) -> Self {
        self.addressed_records = Some(store);
        self.addressed_authoritative = true;
        self
    }

    /// Shared direct-address store, if this alternate path was attached.
    #[must_use]
    pub fn addressed_record_store(&self) -> Option<&Arc<AddressedRecordStore>> {
        self.addressed_records.as_ref()
    }

    fn has_physical_record_target(&self) -> bool {
        self.primary.is_some() || self.addressed_records.is_some()
    }

    /// Wire the shared idempotency store so delete paths can stage
    /// durable release entries for records created through ingest.
    #[must_use]
    pub fn with_idempotency_store(mut self, store: Arc<crate::IdempotencyStore>) -> Self {
        self.idempotency_store = parking_lot::RwLock::new(Some(store));
        self
    }

    /// Attach a shared idempotency store after construction.
    pub fn set_idempotency_store(&self, store: Arc<crate::IdempotencyStore>) {
        *self.idempotency_store.write() = Some(store);
    }

    /// Attach the M4 page-backed logical owner after recovery and before the
    /// server accepts writes.
    pub fn set_owner_rows(&self, owner: Arc<crate::owner_row::OwnerRowRegistry>) {
        *self.owner_rows.write() = Some(owner);
    }

    /// Shared vector page store handle (`None` unless wired via
    /// [`Self::with_vector_store`]). Slice G.5 dispatch surface.
    #[must_use]
    pub fn vector_store(&self) -> Option<&Arc<dyn VectorPageStoreHandle>> {
        self.vector_store.as_ref()
    }

    /// M3.b ADR-039 §D-7 — wire a [`crate::mutation_log::Bm25IndexStoreHandle`]
    /// for Z-1 (b) BM25 rollback dispatch. Mirrors
    /// [`Self::with_vector_store`]'s opt-in posture.
    ///
    /// Production wiring: every callsite that constructs a
    /// `CrudStore` for a deployment with BM25 text search MUST chain
    /// `.with_bm25_store(handle)` so the rollback closure has a
    /// dispatch target on WAL fsync failure. When omitted, the
    /// `bm25_pending` drain in the rollback closure warn-and-skips
    /// (matches the `with_vector_store` posture).
    #[must_use]
    pub fn with_bm25_store(
        mut self,
        handle: Arc<dyn crate::mutation_log::Bm25IndexStoreHandle>,
    ) -> Self {
        self.bm25_store = Some(handle);
        self
    }

    /// Shared BM25 commit-side handle (`None` unless wired via
    /// [`Self::with_bm25_store`]).
    #[must_use]
    pub fn bm25_store(&self) -> Option<&Arc<dyn crate::mutation_log::Bm25IndexStoreHandle>> {
        self.bm25_store.as_ref()
    }

    /// W28 Feature #582 (ADR-045) — wire a
    /// [`crate::metrics::MetricsSink`] so the CRUD TEL overflow path
    /// fires the `arcgraph_hot_vertex_warnings_total{tenant}` counter
    /// (design-v2 §10.2 line 721).
    ///
    /// Production wiring: the `arcgraph` server binary's
    /// `bootstrap_storage_backend` chains `.with_metrics_sink(registry)`
    /// when the operator passes `--metrics-http <addr>`. When omitted,
    /// hot-vertex emission is a no-op (every overflow path
    /// short-circuits on `self.metrics_sink.is_none()`) — the legacy
    /// zero-overhead default.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn crate::metrics::MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// Shared observability sink (`None` unless wired via
    /// [`Self::with_metrics_sink`]).
    #[must_use]
    pub fn metrics_sink(&self) -> Option<&Arc<dyn crate::metrics::MetricsSink>> {
        self.metrics_sink.as_ref()
    }

    /// Emit one `arcgraph_hot_vertex_warnings_total{tenant}` increment
    /// (design-v2 §10.2 line 721) if a sink is wired.
    ///
    /// Budget (PD-5): the TEL overflow path is NOT the per-edge hot
    /// path — it fires once per `MAX_BLOCK_BYTES`-sized overflow-block
    /// allocation (a hot vertex accrues ~10-50 of these over its
    /// lifetime per design-v2 §3.3), so the `Some` branch's vtable +
    /// atomic increment is negligible against the block-allocation +
    /// chain-walk work that precedes it. The `None` branch is a single
    /// nullable-ptr check.
    #[inline]
    fn record_hot_vertex_warning(&self, tenant: TenantId) {
        if let Some(sink) = self.metrics_sink.as_ref() {
            sink.record_hot_vertex_warning(tenant);
        }
    }

    /// M3.b ADR-039 §D-5 — commit-side BM25 drain helper.
    ///
    /// Drains `log.bm25_pending` and calls
    /// [`crate::mutation_log::Bm25IndexStoreHandle::commit_pending`]
    /// per tenant. Intended to be called AFTER WAL fsync success and
    /// BEFORE Phase 3 publish, mirroring the post-fsync sequencing
    /// described in ADR-039 §D-5.
    ///
    /// **Dormant at v1.0.** The kernel commit closure in
    /// `transaction.rs` is frozen by the parallel M3.b session
    /// boundary; this helper exists so future slices (M4 query layer
    /// commit-context wiring or a follow-up PR that lifts the
    /// `transaction.rs` boundary) can invoke it without re-deriving
    /// the drain shape. v1.0 tests exercise commit_pending directly
    /// against `Bm25Service` without going through this helper.
    ///
    /// Errors from `commit_pending` surface as `tracing::warn!` and
    /// do NOT abort the drain — matches the rollback closure's
    /// posture so a transient Tantivy error on one tenant does not
    /// strand a sibling tenant's commit.
    #[allow(dead_code)]
    pub(crate) fn commit_bm25_pending(&self, log: &mut crate::mutation_log::TxnMutationLog) {
        let drained: Vec<_> = log.bm25_pending.drain(..).collect();
        let Some(bm25) = self.bm25_store.as_ref() else {
            if !drained.is_empty() {
                tracing::warn!(
                    "ADR-039 commit-pending: no Bm25IndexStoreHandle wired into \
                     CrudStore; skipping bm25_pending drain ({} tenants). Wire \
                     one via CrudStore::with_bm25_store.",
                    drained.len(),
                );
            }
            return;
        };
        for tenant in drained {
            if let Err(e) = bm25.commit_pending(tenant) {
                tracing::warn!(
                    "ADR-039 commit-pending: bm25 commit_pending failed for \
                     tenant {:?}: {}",
                    tenant,
                    e,
                );
            }
        }
    }

    /// Dual-write-enabled constructor — primary index only (M2-CUTOVER).
    ///
    /// The returned store publishes every `create_*` / `update_*` /
    /// `delete_*` into both the MVCC chain (authoritative visibility
    /// per ADR-023) AND the slotted record-page store + the primary
    /// B-tree index (the read-accelerator side of the dual write).
    /// The `records` store is constructed internally (alpha-only,
    /// DashMap-hosted per DEC-17); the `primary` index and `allocator`
    /// are shared with other subsystems through `Arc`.
    ///
    /// Preserved as the pre-M2-34 entry point. For the full
    /// property→NodeId reverse index, use
    /// [`Self::new_with_indices`].
    #[must_use]
    pub fn new_with_index(
        wal: Option<WalHandle>,
        primary: Arc<PrimaryIndex>,
        allocator: Arc<PageAllocator>,
    ) -> Self {
        Self::new_with_indices(wal, primary, None, allocator)
    }

    /// Dual-write constructor that reuses already-created page stores.
    ///
    /// Durable bootstrap recovers into raw page-store handles before the WAL
    /// writer attaches, then wraps those same handles in the served
    /// [`CrudStore`] so replayed pages are not discarded.
    #[must_use]
    pub fn new_with_existing_stores(
        wal: Option<WalHandle>,
        primary: Arc<PrimaryIndex>,
        allocator: Arc<PageAllocator>,
        records: Arc<RecordPageStore>,
        blobs: Arc<BlobStore>,
    ) -> Self {
        Self::new_with_existing_page_stores(Some(primary), wal, allocator, records, blobs)
    }

    /// Constructor for durable recovery before the primary-index wrapper exists.
    ///
    /// The raw record/blob stores are replay targets. After recovery, bootstrap
    /// wraps the same primary page store in a [`PrimaryIndex`] and calls
    /// [`Self::attach_primary_index`] before serving.
    #[must_use]
    pub fn new_with_existing_page_stores(
        primary: Option<Arc<PrimaryIndex>>,
        wal: Option<WalHandle>,
        allocator: Arc<PageAllocator>,
        records: Arc<RecordPageStore>,
        blobs: Arc<BlobStore>,
    ) -> Self {
        Self {
            wal,
            allocator: Some(allocator),
            records: Some(records),
            blobs,
            primary,
            ..Self::default()
        }
    }

    /// Constructor for a recovered M3 generation whose record pages begin in
    /// the bounded disk-backed tier rather than a whole-owner RAM capture.
    #[must_use]
    pub fn new_with_existing_buffered_page_store(
        primary: Option<Arc<PrimaryIndex>>,
        wal: Option<WalHandle>,
        allocator: Arc<PageAllocator>,
        records: Arc<crate::page_store::BufferedRecordPageStore>,
        blobs: Arc<BlobStore>,
    ) -> Self {
        Self {
            wal,
            allocator: Some(allocator),
            records: None,
            buffered_records: Some(records),
            blobs,
            primary,
            ..Self::default()
        }
    }

    /// Attach the served primary index after durable recovery wraps the replayed
    /// primary page store.
    pub fn attach_primary_index(&mut self, primary: Arc<PrimaryIndex>) {
        self.primary = Some(primary);
    }

    /// Dual-write-enabled constructor with an optional secondary
    /// property index (M2-34).
    ///
    /// When `secondary` is `Some`, the commit drain also publishes
    /// every `Create` / `Update` / `Delete` of a `RecordKind::Node`
    /// into the secondary's `(tenant, label, property_key, value) →
    /// node` entries. Updates diff the pre-image (read from the
    /// primary-index + record-page store) against the new bytes to
    /// avoid spurious index churn on unchanged properties.
    ///
    /// When `secondary` is `None`, the store behaves identically to
    /// [`Self::new_with_index`].
    #[must_use]
    pub fn new_with_indices(
        wal: Option<WalHandle>,
        primary: Arc<PrimaryIndex>,
        secondary: Option<Arc<dyn SecondaryIndexHandle>>,
        allocator: Arc<PageAllocator>,
    ) -> Self {
        Self {
            wal,
            allocator: Some(allocator),
            records: Some(Arc::new(RecordPageStore::new())),
            primary: Some(primary),
            secondary,
            ..Self::default()
        }
    }

    /// W26-ε-2 / ADR-140 — wire-through constructor that backs the
    /// slotted record page store with a
    /// [`crate::page_store::BufferedRecordPageStore`] (cache + spill).
    ///
    /// Production deployments that need RSS-bounded ingest (e.g.,
    /// LDBC SF-100 fixture load larger than RAM) MUST use this
    /// constructor; the legacy [`Self::new_with_indices`] keeps the
    /// in-memory DashMap path for small-deployment ramps + tests.
    ///
    /// Wiring contract:
    ///
    /// - The buffered store's hot cache (`crate::page_store::BufferedRecordPageStore::cache`)
    ///   is registered as `self.records` so existing CRUD call sites
    ///   that read via `self.records.as_ref()` continue to work
    ///   against the buffered store's hot tier (cache-hit-only path
    ///   for now; ADR-140 §Forward-deferred lifts implicit fault-in
    ///   into the `latch` site in a v1.1 follow-up).
    /// - The buffered store handle is exposed via
    ///   [`Self::buffered_records`] for ops eviction + the
    ///   [`crate::wal::replay::RecordPageStoreHandle`] cast used by
    ///   recovery.
    ///
    /// Per ADR-140 D-3 §"The `RecordPageBackend` adapter trait".
    #[must_use]
    pub fn new_with_page_store(
        wal: Option<WalHandle>,
        primary: Arc<PrimaryIndex>,
        secondary: Option<Arc<dyn SecondaryIndexHandle>>,
        allocator: Arc<PageAllocator>,
        page_store: Arc<crate::page_store::BufferedRecordPageStore>,
    ) -> Self {
        Self {
            wal,
            allocator: Some(allocator),
            records: None,
            buffered_records: Some(page_store),
            primary: Some(primary),
            secondary,
            ..Self::default()
        }
    }

    /// W26-ε-2 / ADR-140 — buffered record page store handle (`None`
    /// unless the store was constructed via
    /// [`Self::new_with_page_store`]).
    ///
    /// Exposed so ops can drive `evict_lru` to bound RSS and so the
    /// recovery wiring can cast to
    /// [`crate::wal::replay::RecordPageStoreHandle`] without
    /// re-fetching the inner cache.
    #[must_use]
    pub fn buffered_records(&self) -> Option<&Arc<crate::page_store::BufferedRecordPageStore>> {
        self.buffered_records.as_ref()
    }

    /// Shared secondary-index handle (`None` unless configured via
    /// [`Self::new_with_indices`]).
    #[must_use]
    pub fn secondary(&self) -> Option<&Arc<dyn SecondaryIndexHandle>> {
        self.secondary.as_ref()
    }

    /// **RC-1 (#1366)** — enqueue a batch of secondary-index old-value
    /// removals, stamped with the removing commit's `Lsn`. Called by
    /// [`commit`] on commit success. See [`Self::deferred_removals`].
    ///
    /// Empty batches are a no-op (the common case: creates, and updates
    /// whose declared-index property value did not change). The queue is
    /// in-memory only (intentionally lossy — a lost entry strands a
    /// read-safe ghost, never a missing entry).
    fn enqueue_deferred_removals(
        &self,
        commit_lsn: Lsn,
        tenant: TenantId,
        batch: &[(LabelId, StringId, SecondaryIndexValue, NodeId)],
    ) {
        if batch.is_empty() {
            return;
        }
        let mut state = self.deferred_removals.lock();
        for &(label, property_key, value, node) in batch {
            state.removals.push(DeferredRemoval {
                commit_lsn,
                tenant,
                label,
                property_key,
                value,
                node,
            });
        }
    }

    /// Publish the final commit generation for update-side secondary entries.
    /// The insert itself registered each entry as in-flight while holding this
    /// same mutex, so a ready remover cannot pass the guard in between.
    fn finish_secondary_reassertions(
        &self,
        txn_id: u64,
        commit_lsn: Lsn,
        entries: &[SecondaryEntry],
    ) {
        self.deferred_removals
            .lock()
            .finish_inflight(txn_id, commit_lsn, entries);
    }

    /// Drop in-flight assertion markers after the enclosing commit rolled its
    /// secondary inserts back. No generation is published for an aborted txn.
    fn discard_secondary_reassertions(&self, txn_id: u64, entries: &[SecondaryEntry]) {
        self.deferred_removals
            .lock()
            .discard_inflight(txn_id, entries);
    }

    /// **RC-1 (#1366)** — apply every enqueued secondary-index deferred
    /// removal whose removing-commit LSN the snapshot horizon has
    /// reached (`horizon >= commit_lsn`), i.e. no live snapshot can
    /// still observe the superseded value. Removals not yet cleared
    /// stay queued.
    ///
    /// Before deleting, the exact entry's latest successful re-assertion
    /// generation is compared with the removal generation. `latest > removal`
    /// makes the removal a no-op (#1464); `latest <= removal` is the legitimate
    /// stale-entry case and still deletes. A matching in-flight assertion
    /// leaves the removal queued until that transaction commits or aborts.
    ///
    /// # Why `>=` is the exact MVCC boundary (not `>`)
    ///
    /// A version created at `commit_lsn = L` is visible to a snapshot
    /// `S` iff `S >= L`. A reader with `S < L` still sees the OLD value
    /// and needs the old-value index entry; a reader with `S >= L` sees
    /// the NEW value and verify-filters the old entry as a ghost. So the
    /// removal of the old value is safe once NO active reader has
    /// `S < L`, i.e. once `oldest_active_snapshot() >= L`. Using `>`
    /// would be one LSN too conservative (it would strand a ghost even
    /// after every reader can no longer observe the old value) but is
    /// NOT a correctness bug — it only delays reclamation.
    ///
    /// Best-effort per ADR-023: a backend remove failure logs and moves
    /// on — a stranded ghost is read-safe. Applying a removal here is
    /// the ONLY place the secondary B-tree slot for an old value is
    /// zeroed; the commit path never does it eagerly (that was the
    /// false-negative cliff). The removals apply through the standalone
    /// `remove_property` path (own crash-atomic bundle), NOT the
    /// bundle-folded `_deferred` path, because they run outside any
    /// enclosing commit builder.
    fn apply_ready_deferred_removals(&self, horizon: Lsn) {
        let Some(secondary) = self.secondary.as_ref() else {
            // No secondary configured — drop any queued entries so the
            // queue cannot grow unbounded on a store that will never
            // apply them.
            let mut state = self.deferred_removals.lock();
            state.removals.clear();
            state.latest_reassertions.clear();
            state.inflight_reassertions.clear();
            return;
        };
        // The state mutex is also the secondary-maintenance ordering gate.
        // Update-side inserts take it while mutating the backend and registering
        // their in-flight marker. Holding it through check + remove gives two
        // safe orders: removal first then newer insert, or newer insert/marker
        // first then guarded no-op. `remove_property` commits a standalone
        // SYSTEM bundle but never re-enters CrudStore, so this lock order is
        // acyclic with the backend write gate.
        let mut state = self.deferred_removals.lock();
        let mut index = 0;
        while index < state.removals.len() {
            let removal = state.removals[index];
            if horizon.raw() < removal.commit_lsn.raw() {
                index += 1;
                continue;
            }

            let entry = removal.entry();
            let reasserted_by_newer_commit = state
                .latest_reassertions
                .get(&entry)
                .is_some_and(|latest| latest.raw() > removal.commit_lsn.raw());
            if reasserted_by_newer_commit {
                state.removals.swap_remove(index);
                continue;
            }
            if state.has_inflight(entry) {
                index += 1;
                continue;
            }

            let removal = state.removals.swap_remove(index);
            let assertion_is_superseded = state
                .latest_reassertions
                .get(&entry)
                .is_some_and(|latest| latest.raw() <= removal.commit_lsn.raw());
            if assertion_is_superseded {
                state.latest_reassertions.remove(&entry);
            }
            if let Err(e) = secondary.remove_property(
                removal.tenant,
                removal.label,
                removal.property_key,
                removal.value,
                removal.node,
            ) {
                tracing::warn!(
                    "RC-1 deferred removal apply failed (label={:?}, pk={:?}, val={:?}, \
                     node={:?}, commit_lsn={:?}): {}; entry stranded as a read-safe ghost",
                    removal.label,
                    removal.property_key,
                    removal.value,
                    removal.node,
                    removal.commit_lsn,
                    e,
                );
            }
        }
    }

    /// **RC-1 test/observability** — number of secondary-index deferred
    /// removals currently queued (awaiting the snapshot horizon).
    #[doc(hidden)]
    #[must_use]
    pub fn deferred_removal_queue_len(&self) -> usize {
        self.deferred_removals.lock().removals.len()
    }

    /// Shared slotted-record store (`None` on the no-index constructor).
    #[must_use]
    pub fn records(&self) -> Option<&Arc<RecordPageStore>> {
        self.records.as_ref()
    }

    /// Publish committed logical mutations into the optional M4 arithmetic
    /// store. This runs only after the transaction commit boundary returns
    /// success, so an aborted transaction cannot materialize a direct slot.
    ///
    /// The store remains an alternate accelerator in this slice: a publish
    /// failure is logged after the already-successful commit and the MVCC /
    /// primary paths remain authoritative. The migration slice owns the
    /// authority flip and its stronger open/recovery gate.
    fn publish_addressed_installs(&self, installs: &[PendingInstall], commit_lsn: Lsn) {
        let Some(addressed) = self.addressed_records.as_ref() else {
            return;
        };
        for install in installs {
            let tenant = install.tenant();
            let kind = install.kind();
            let id = install.id();
            let result: Result<(), CrudError> = (|| match install {
                PendingInstall::Create { bytes, .. } | PendingInstall::Update { bytes, .. } => {
                    let mut physical = bytes.clone();
                    fixup_created_lsn(&mut physical, kind, commit_lsn);
                    match kind {
                        RecordKind::Node => {
                            let record = decode_node_bytes(&physical)?;
                            addressed.write_node(tenant, &record)?;
                        }
                        RecordKind::Rel => {
                            let record = decode_rel_bytes(&physical)?;
                            addressed.write_rel(tenant, &record)?;
                        }
                    }
                    Ok(())
                }
                PendingInstall::Delete { .. } => {
                    let existed = match kind {
                        RecordKind::Node => {
                            addressed.tombstone_node_at_lsn(tenant, NodeId::new(id), commit_lsn)?
                        }
                        RecordKind::Rel => {
                            addressed.tombstone_rel_at_lsn(tenant, RelId::new(id), commit_lsn)?
                        }
                    };
                    if !existed {
                        tracing::warn!(
                            ?tenant,
                            ?kind,
                            id,
                            "direct-address tombstone target was absent; MVCC remains authoritative"
                        );
                    }
                    Ok(())
                }
            })();
            if let Err(error) = result {
                tracing::warn!(
                    ?tenant,
                    ?kind,
                    id,
                    %error,
                    "direct-address alternate publish failed after logical commit"
                );
            }
        }
    }

    fn record_backend(&self) -> Option<&dyn RecordPageBackend> {
        self.buffered_records
            .as_deref()
            .map(|records| records as &dyn RecordPageBackend)
            .or_else(|| {
                self.records
                    .as_deref()
                    .map(|records| records as &dyn RecordPageBackend)
            })
    }

    /// Shared overflow BLOB store. N-2 (issue #81): the WAL replay
    /// target clones this to route `BundlePageKind::Blob` entries
    /// into the same store the CRUD layer publishes into, closing
    /// the post-replay `MissingHead` gap on `PropertyData::Blob`.
    #[must_use]
    pub fn blob_store(&self) -> &Arc<BlobStore> {
        &self.blobs
    }

    /// Shared primary-index handle (`None` on the no-index constructor).
    #[must_use]
    pub fn primary(&self) -> Option<&Arc<PrimaryIndex>> {
        self.primary.as_ref()
    }

    /// Shared WAL handle (`None` for the ephemeral / no-WAL stores used
    /// by tests + `--in-memory`). P0 #776: the MCP write path reads this
    /// to WAL-log new label / rel-type interns (via
    /// [`crate::intern::intern_label_logged`] /
    /// [`crate::intern::intern_type_logged`]) so names survive a durable
    /// restart. `Some` exactly when the store was built durable
    /// (`new_with_index(Some(handle), ..)`), matching the existing
    /// [`BlobStore::put_logged`] gating.
    #[must_use]
    pub fn wal(&self) -> Option<&WalHandle> {
        self.wal.as_ref()
    }

    /// Shared page allocator (`None` on the no-index constructor).
    #[must_use]
    pub fn allocator(&self) -> Option<&Arc<PageAllocator>> {
        self.allocator.as_ref()
    }

    /// Allocate the next [`NodeId`] for `tenant`. Ids start at 1 per
    /// tenant; `NodeId(0)` is reserved as an unused sentinel.
    ///
    /// Returns [`CrudError::NodeIdExhausted`] if the 63-bit node-id
    /// space is exhausted for this tenant — bit 63 is reserved as the
    /// MVCC rel tag (see [`REL_TAG_BIT`]).
    pub fn alloc_node(&self, tenant: TenantId) -> Result<NodeId, CrudError> {
        let entry = self
            .next_node
            .entry(tenant)
            .or_insert_with(|| AtomicU64::new(0));
        let prev = entry.fetch_add(1, Ordering::AcqRel);
        let id = prev.saturating_add(1);
        if id & REL_TAG_BIT != 0 {
            return Err(CrudError::NodeIdExhausted { tenant });
        }
        Ok(NodeId::new(id))
    }

    /// Allocate the next [`RelId`] for `tenant`. Same budget rules as
    /// [`Self::alloc_node`]; bit 63 stays clear so
    /// [`rel_mvcc_key`] can set it as the MVCC key tag.
    pub fn alloc_rel(&self, tenant: TenantId) -> Result<RelId, CrudError> {
        let entry = self
            .next_rel
            .entry(tenant)
            .or_insert_with(|| AtomicU64::new(0));
        let prev = entry.fetch_add(1, Ordering::AcqRel);
        let id = prev.saturating_add(1);
        if id & REL_TAG_BIT != 0 {
            return Err(CrudError::NodeIdExhausted { tenant });
        }
        Ok(RelId::new(id))
    }

    /// Last allocated `NodeId` for `tenant`, or `0` if no
    /// allocations have been made yet for this tenant. Issue #129
    /// P0 fix — drained at commit time into the v4 `CommitBundle`'s
    /// `allocator_advances` section so post-recovery `alloc_node`
    /// cannot reuse a `NodeId` a pre-fault commit consumed.
    ///
    /// Counter convention: `next_node` stores the last allocated
    /// id, advanced by `fetch_add(1)` returning prev=last_allocated;
    /// the next id is `prev + 1`. Pristine tenants have no entry in
    /// the DashMap → `0`.
    #[must_use]
    pub fn node_high_water(&self, tenant: TenantId) -> u64 {
        self.next_node
            .get(&tenant)
            .map_or(0, |e| e.load(Ordering::Acquire))
    }

    /// Last allocated `RelId` for `tenant`. Symmetric with
    /// [`Self::node_high_water`].
    #[must_use]
    pub fn rel_high_water(&self, tenant: TenantId) -> u64 {
        self.next_rel
            .get(&tenant)
            .map_or(0, |e| e.load(Ordering::Acquire))
    }

    /// Idempotent monotonic seed: ensures the next `alloc_node` for
    /// `tenant` returns at least `high_water + 1`. Replays in
    /// commit_lsn order from the v4 `CommitBundle`'s
    /// `allocator_advances` section. Lemma I3 — applying the same
    /// advance twice (or applying an older advance after a newer
    /// one) is a no-op (issue #129 P0 fix).
    pub fn seed_node_from_advance(&self, tenant: TenantId, high_water: u64) {
        seed_atomic_counter_max(&self.next_node, tenant, high_water);
    }

    /// Idempotent monotonic seed for `alloc_rel`. Symmetric with
    /// [`Self::seed_node_from_advance`].
    pub fn seed_rel_from_advance(&self, tenant: TenantId, high_water: u64) {
        seed_atomic_counter_max(&self.next_rel, tenant, high_water);
    }

    /// Issue #129 P0 fix: dispatch a single
    /// [`AllocatorAdvance`] to the right CRUD-layer counter
    /// (Node → `next_node`, Rel → `next_rel`). `Page*` variants
    /// belong to [`crate::page_alloc::PageAllocator`] and are
    /// silently ignored here — the combined replay handle
    /// [`crud_allocator_seed_handle`] dispatches Page* into the
    /// `PageAllocator`. Idempotent monotonic-max (Lemma I3).
    pub fn apply_allocator_advance(&self, advance: AllocatorAdvance) {
        match advance.kind {
            AllocatorKind::Node => {
                self.seed_node_from_advance(advance.tenant, advance.new_high_water)
            }
            AllocatorKind::Rel => {
                self.seed_rel_from_advance(advance.tenant, advance.new_high_water)
            }
            // Page* variants are not owned by CrudStore — they
            // route to PageAllocator. The unified
            // `CrudAllocatorSeedHandle` wrapper takes care of the
            // cross-store dispatch; this method is the
            // CrudStore-only leg.
            AllocatorKind::PageFree
            | AllocatorKind::PageNode
            | AllocatorKind::PageRel
            | AllocatorKind::PageTel
            | AllocatorKind::PageIndexInternal
            | AllocatorKind::PageIndexLeaf
            | AllocatorKind::PageVectorNeighbor
            | AllocatorKind::PageWalBuffer
            | AllocatorKind::PageIndexOverflow
            | AllocatorKind::PagePropSlotted
            | AllocatorKind::InternString
            | AllocatorKind::AclClass => {}
        }
    }

    /// Snapshot per-tenant `next_node` + `next_rel` high-water as a
    /// vec of [`AllocatorAdvance`] entries. Drained at commit time
    /// by [`crate::crud::commit`] into the v4 `CommitBundle`'s
    /// `allocator_advances` section.
    ///
    /// Pristine tenants (no `alloc_node` / `alloc_rel` calls) are
    /// omitted to keep the wire payload tight. The drain is over
    /// the GLOBAL state — all tenants — because the encode point is
    /// per-commit and the cost is negligible (≤ N_tenants × 2
    /// entries × 17 B). On replay only the matching `(tenant, kind)`
    /// counters are seeded; over-counting is harmless under
    /// monotonic-max.
    #[must_use]
    pub fn snapshot_allocator_advances(&self) -> Vec<AllocatorAdvance> {
        let mut out: Vec<AllocatorAdvance> = Vec::new();
        for entry in self.next_node.iter() {
            let high = entry.value().load(Ordering::Acquire);
            if high > 0 {
                out.push(AllocatorAdvance {
                    tenant: *entry.key(),
                    kind: AllocatorKind::Node,
                    new_high_water: high,
                });
            }
        }
        for entry in self.next_rel.iter() {
            let high = entry.value().load(Ordering::Acquire);
            if high > 0 {
                out.push(AllocatorAdvance {
                    tenant: *entry.key(),
                    kind: AllocatorKind::Rel,
                    new_high_water: high,
                });
            }
        }
        out
    }

    /// Allocate a fresh synthetic page id for a TEL block.
    fn alloc_virtual_page(&self) -> PageId {
        PageId::new(self.next_virtual_page.fetch_add(1, Ordering::AcqRel) + 1)
    }

    fn note_tel_channel_for_src(&self, tenant: TenantId, src: NodeId, channel: LabelId) {
        let mut channels = self.tel_channels_by_src.entry((tenant, src)).or_default();
        if !channels.contains(&channel) {
            channels.push(channel);
        }
    }

    fn note_reverse_tel_channel_for_dst(&self, tenant: TenantId, dst: NodeId, channel: LabelId) {
        let mut channels = self
            .reverse_tel_channels_by_dst
            .entry((tenant, dst))
            .or_default();
        if !channels.contains(&channel) {
            channels.push(channel);
        }
    }

    /// Get-or-create the TEL chain for `(tenant, src, channel)`. First
    /// block in a fresh chain is [`MIN_BLOCK_BYTES`] (one entry slot);
    /// it doubles via `grown()` as it fills.
    fn tel_chain_for(
        &self,
        tenant: TenantId,
        src: NodeId,
        channel: LabelId,
    ) -> Result<Arc<Mutex<TelChain>>, CrudError> {
        if let Some(existing) = self.tel_chains.get(&(tenant, src, channel)) {
            return Ok(Arc::clone(existing.value()));
        }
        // First-time initialization race (#27). DashMap's
        // `entry().or_insert_with` serializes on the bucket, so
        // exactly one concurrent initializer wins the chain. The
        // losers also constructed a block and burned a page id —
        // previously both were published into `tel_blocks` *before*
        // the winner was decided, orphaning the losers' blocks for
        // the life of the process.
        //
        // Fix: publish the block into `tel_blocks` only on the
        // winning path. The losers still burn a `u64` page-id tick
        // (cheap, no heap allocation escapes the frame), but their
        // `Arc<TelBlock>` is dropped on function exit.
        let page = self.alloc_virtual_page();
        let block = Arc::new(
            TelBlock::new(src, channel, MIN_BLOCK_BYTES, tenant).map_err(CrudError::from)?,
        );
        let mut winner = false;
        let chain = self
            .tel_chains
            .entry((tenant, src, channel))
            .or_insert_with(|| {
                winner = true;
                Arc::new(Mutex::new(TelChain {
                    head: Arc::clone(&block),
                    head_page: page,
                }))
            })
            .clone();
        if winner {
            // Safe to publish: this thread owns the chain entry the
            // block is referenced from, so a concurrent
            // `tel_block(tenant, page)` lookup that follows a chain
            // read we produced will find the block.
            self.tel_blocks.insert((tenant, page), block);
            self.note_tel_channel_for_src(tenant, src, channel);
        }
        // Loser path: `block` is dropped at scope exit; `page` is
        // leaked as a `u64` tick and never referenced.
        Ok(chain)
    }

    /// Test-only accessor for counting the blocks registered under
    /// `tenant`. Used by the #27 race regression test.
    #[cfg(test)]
    pub(crate) fn tel_blocks_len_for_tenant(&self, tenant: TenantId) -> usize {
        self.tel_blocks
            .iter()
            .filter(|e| e.key().0 == tenant)
            .count()
    }

    /// Perform a TEL append for `(tenant, src, channel)` with the
    /// commit LSN stamped on the entry. Handles growth (via
    /// `TelBlock::grown`) and overflow chaining (fresh MIN-sized block
    /// linking the full old one).
    ///
    /// Returns `(head_page_id, slot_index)` on success — head_page_id
    /// is the synthetic page id of the block the entry ultimately
    /// landed in.
    #[allow(clippy::too_many_arguments)]
    fn tel_append(
        &self,
        tenant: TenantId,
        src: NodeId,
        channel: LabelId,
        dst: NodeId,
        rel: RelId,
        commit_lsn: Lsn,
    ) -> Result<(PageId, u32), CrudError> {
        let chain_lock = self.tel_chain_for(tenant, src, channel)?;
        let mut chain = chain_lock.lock();

        let entry = TelEntry::new(dst, rel, commit_lsn);

        // Fast path: current head has room.
        match chain.head.append(entry) {
            Ok(slot) => Ok((chain.head_page, slot)),
            Err(TelError::Full { .. }) => {
                // Try growth (double the size) before overflow.
                if chain.head.block_size() < MAX_BLOCK_BYTES
                    && next_block_size(chain.head.block_size()).is_some()
                {
                    // P0 #812: if the block we are growing is itself an
                    // overflow successor (i.e. it already links a
                    // predecessor via `prev_block_ptr`), the grown
                    // REPLACEMENT must inherit that link. `grown()`
                    // returns a fresh block with `prev_block_ptr =
                    // NO_PREV_BLOCK` (it is a replacement, not a
                    // sibling), so without re-linking, the entire
                    // predecessor chain is orphaned — every edge before
                    // this block silently vanishes from `scan_out`.
                    // This is the root cause of the "~2048-edge cap":
                    // the first MIN-sized overflow head (1 entry) grows
                    // on the very next insert, dropping the link to the
                    // full 2047-entry predecessor (the 2049-th insert).
                    let inherited_prev = chain.head.prev_block_ptr();
                    let grown = chain
                        .head
                        .grown()
                        .expect("grown() returns Some below MAX_BLOCK_BYTES");
                    if let Some(prev) = inherited_prev {
                        grown.set_prev_block_ptr(prev).expect(
                            "freshly grown block has no predecessor link yet (write-once invariant)",
                        );
                    }
                    let slot = grown
                        .append(entry)
                        .expect("freshly grown block has room for at least one entry");
                    let new_block = Arc::new(grown);
                    let new_page = self.alloc_virtual_page();
                    self.tel_blocks
                        .insert((tenant, new_page), Arc::clone(&new_block));
                    // The old head is superseded; readers holding an
                    // Arc to it still see a frozen prefix (LiveGraph
                    // Theorem 1). We don't evict `tel_blocks[old]`
                    // because in-flight scans may still reach it via
                    // a previously-captured head_page snapshot.
                    chain.head = new_block;
                    chain.head_page = new_page;
                    Ok((new_page, slot))
                } else {
                    // At MAX_BLOCK_BYTES: link a fresh overflow head.
                    //
                    // Hot-vertex warning (#30). Every overflow
                    // allocation is a signal that this vertex's
                    // adjacency is approaching the supernode
                    // threshold (~10-50 overflow blocks per
                    // design-v2 §3.3's 2-hop-latency budget). We
                    // walk back through `prev_block_ptr` under the
                    // held chain lock to compute the existing chain
                    // depth — cheap (one DashMap lookup per hop;
                    // chain depth is by definition bounded by the
                    // supernode soft ceiling) — and emit a single
                    // `tracing::warn!` so operators can see hot
                    // vertices before they hit the cliff.
                    let mut chain_depth: u64 = 1;
                    let mut walker: Arc<TelBlock> = Arc::clone(&chain.head);
                    while let Some(pid) = walker.prev_block_ptr() {
                        match self.tel_block(tenant, pid) {
                            Some(older) => {
                                walker = older;
                                chain_depth += 1;
                            }
                            None => break,
                        }
                    }
                    tracing::warn!(
                        tenant_id = tenant.raw(),
                        src_id = src.raw(),
                        channel = channel.raw(),
                        chain_depth = chain_depth,
                        block_bytes = chain.head.block_size(),
                        "TEL overflow block allocated — vertex approaching supernode threshold \
                         (design-v2 §3.3; HOT_VERTEX_WARNING, issue #30)"
                    );
                    // W28 Feature #582 (ADR-045) — fire the §10.2 line
                    // 721 `arcgraph_hot_vertex_warnings_total{tenant}`
                    // counter at the SAME site as the warn. No-op when
                    // no sink is wired. Closes the W17δ #313 no-op
                    // trampoline (`feedback_noop_trampoline_anti_pattern`).
                    self.record_hot_vertex_warning(tenant);

                    let old_head_page = chain.head_page;
                    let fresh = Arc::new(
                        TelBlock::new(src, channel, MIN_BLOCK_BYTES, tenant)
                            .map_err(CrudError::from)?,
                    );
                    fresh.set_prev_block_ptr(old_head_page)?;
                    let slot = fresh
                        .append(entry)
                        .expect("fresh MIN-sized block has room for exactly one entry");
                    let new_page = self.alloc_virtual_page();
                    self.tel_blocks
                        .insert((tenant, new_page), Arc::clone(&fresh));
                    chain.head = fresh;
                    chain.head_page = new_page;
                    Ok((new_page, slot))
                }
            }
            Err(other) => Err(CrudError::from(other)),
        }
    }

    /// Fetch a block by synthetic id. Used by scan_out in M2-24.
    #[doc(hidden)]
    #[must_use]
    pub fn tel_block(&self, tenant: TenantId, page: PageId) -> Option<Arc<TelBlock>> {
        self.tel_block_fetches.fetch_add(1, Ordering::Relaxed);
        self.tel_blocks.get(&(tenant, page)).map(|e| Arc::clone(&e))
    }

    /// Test/bench instrumentation: count overflow predecessor fetches.
    #[doc(hidden)]
    #[must_use]
    pub fn tel_block_fetches_for_test(&self) -> u64 {
        self.tel_block_fetches.load(Ordering::Relaxed)
    }

    /// Test/bench instrumentation: reset overflow predecessor fetches.
    #[doc(hidden)]
    pub fn reset_tel_block_fetches_for_test(&self) {
        self.tel_block_fetches.store(0, Ordering::Relaxed);
    }

    /// Fetch the current chain head for `(tenant, src, channel)`.
    #[doc(hidden)]
    #[must_use]
    pub fn tel_head(
        &self,
        tenant: TenantId,
        src: NodeId,
        channel: LabelId,
    ) -> Option<(PageId, Arc<TelBlock>)> {
        self.tel_chains.get(&(tenant, src, channel)).map(|chain| {
            let guard = chain.lock();
            (guard.head_page, Arc::clone(&guard.head))
        })
    }

    /// Snapshot the TEL chain heads for every channel under
    /// `(tenant, src)`, returning `(channel, head_block, head_page_id)`
    /// sorted by `channel.raw()` ascending so scan_out's union iteration
    /// order is deterministic.
    ///
    /// DashMap iteration is not ordered; snapshot-and-sort gives callers
    /// a stable view. Concurrent channel creations after this returns
    /// are not observed — that is the "snapshot chains at iterator
    /// construction" contract called out in M2-24.
    fn tel_heads_for_src(
        &self,
        tenant: TenantId,
        src: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        let mut out: Vec<(LabelId, Arc<TelBlock>, PageId)> = self
            .tel_channels_by_src
            .get(&(tenant, src))
            .map(|channels| {
                channels
                    .iter()
                    .filter_map(|&ch| {
                        self.tel_chains.get(&(tenant, src, ch)).map(|chain| {
                            let g = chain.value().lock();
                            (ch, Arc::clone(&g.head), g.head_page)
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by_key(|(ch, _, _)| ch.raw());
        out
    }

    /// Benchmark hook for V11-S-02: snapshot untyped forward TEL heads
    /// without the MVCC visibility probes performed by [`scan_out`].
    #[doc(hidden)]
    #[must_use]
    pub fn tel_heads_for_src_for_bench(
        &self,
        tenant: TenantId,
        src: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        self.tel_heads_for_src(tenant, src)
    }

    /// Benchmark hook for the pre-V11-S-02 full-scan shape.
    #[doc(hidden)]
    #[must_use]
    pub fn tel_heads_for_src_legacy_scan_for_bench(
        &self,
        tenant: TenantId,
        src: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        let mut out: Vec<(LabelId, Arc<TelBlock>, PageId)> = self
            .tel_chains
            .iter()
            .filter_map(|e| {
                let (t, s, ch) = *e.key();
                if t == tenant && s == src {
                    let g = e.value().lock();
                    Some((ch, Arc::clone(&g.head), g.head_page))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by_key(|(ch, _, _)| ch.raw());
        out
    }

    // ─────────────────────────────────────────────────────────────────
    // W26-β-2 / ADR-131 — reverse adjacency (inbound TEL expand)
    // ─────────────────────────────────────────────────────────────────

    /// Is the reverse adjacency index enabled for this store?
    ///
    /// W26-β-2 / ADR-131 — defaults to `true` (v1.1+ posture). When
    /// `false`, both `Self::tel_append_reverse` and the [`scan_in`]
    /// helper short-circuit and `CrudExecutorSubstrate::expand` is
    /// expected to surface a structured
    /// `crate::SubstrateAccessError::IndexUnavailable` for
    /// `Direction::RightToLeft` + `Direction::Undirected` — never
    /// silent-empty results.
    #[must_use]
    pub fn reverse_index_enabled(&self) -> bool {
        self.reverse_index_enabled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Test-only / operational override of the reverse-index enable
    /// flag.
    ///
    /// W26-β-2 / ADR-131 — used by AC-4 fault-injection tests to
    /// simulate "post-recovery, reverse index unbuilt" so the
    /// substrate's structured-error path can be exercised. In
    /// production this stays `true` (the v1.1 default); the operator-
    /// facing knob lands at v1.2 when persisted-on-disk reverse index
    /// format is ratified.
    pub fn set_reverse_index_enabled(&self, enabled: bool) {
        self.reverse_index_enabled
            .store(enabled, std::sync::atomic::Ordering::Release);
    }

    /// Get-or-create the REVERSE TEL chain for `(tenant, dst, channel)`.
    ///
    /// Mirrors [`Self::tel_chain_for`] but writes into the
    /// [`Self::reverse_tel_chains`] map + [`Self::reverse_tel_blocks`]
    /// page registry. The same #27-style race discipline applies:
    /// only the winning initializer publishes its block into the
    /// reverse page map.
    fn reverse_tel_chain_for(
        &self,
        tenant: TenantId,
        dst: NodeId,
        channel: LabelId,
    ) -> Result<Arc<Mutex<TelChain>>, CrudError> {
        if let Some(existing) = self.reverse_tel_chains.get(&(tenant, dst, channel)) {
            return Ok(Arc::clone(existing.value()));
        }
        let page = self.alloc_virtual_page();
        let block = Arc::new(
            // The TelBlock header's `src_vertex_id` carries `dst` here
            // — the reverse chain's "anchor vertex" IS the dst from
            // the forward edge's perspective. The semantic overload
            // is intentional: a TelBlock's header is opaque to its
            // owner map, and the `(tenant, dst, channel)` key on the
            // reverse_tel_chains map is the authoritative routing
            // discriminator.
            TelBlock::new(dst, channel, MIN_BLOCK_BYTES, tenant).map_err(CrudError::from)?,
        );
        let mut winner = false;
        let chain = self
            .reverse_tel_chains
            .entry((tenant, dst, channel))
            .or_insert_with(|| {
                winner = true;
                Arc::new(Mutex::new(TelChain {
                    head: Arc::clone(&block),
                    head_page: page,
                }))
            })
            .clone();
        if winner {
            self.reverse_tel_blocks.insert((tenant, page), block);
            self.note_reverse_tel_channel_for_dst(tenant, dst, channel);
        }
        Ok(chain)
    }

    /// Append a reverse TEL entry for `(tenant, dst, channel)` with
    /// `created_lsn` stamped from the commit LSN. The entry's
    /// `dst_id` field semantically holds the ORIGINAL SRC (the
    /// neighbor of `dst` on the other end of the forward edge).
    ///
    /// Short-circuits to `Ok((PageId::ZERO, 0))` when
    /// [`Self::reverse_index_enabled`] is `false` — used by the
    /// fault-injection harness; production deployments keep the
    /// flag `true` and incur the ~200ns per-append cost (DashMap
    /// lookup + chain mutex + TelBlock::append; in-memory work
    /// only, well within the §4.4 5K TPS write budget per
    /// ADR-131 §D-3).
    #[allow(clippy::too_many_arguments)]
    fn tel_append_reverse(
        &self,
        tenant: TenantId,
        dst: NodeId,
        channel: LabelId,
        src: NodeId,
        rel: RelId,
        commit_lsn: Lsn,
    ) -> Result<(PageId, u32), CrudError> {
        if !self.reverse_index_enabled() {
            return Ok((PageId::ZERO, 0));
        }
        let chain_lock = self.reverse_tel_chain_for(tenant, dst, channel)?;
        let mut chain = chain_lock.lock();

        // The reverse entry stores `src` in the `dst_id` field
        // (semantic re-purposing — see `reverse_tel_chains` rustdoc).
        let entry = TelEntry::new(src, rel, commit_lsn);

        // Fast path: current head has room.
        match chain.head.append(entry) {
            Ok(slot) => Ok((chain.head_page, slot)),
            Err(TelError::Full { .. }) => {
                // Growth before overflow — mirrors `tel_append`'s
                // fast path (`grown()` doubles the block size).
                if chain.head.block_size() < MAX_BLOCK_BYTES
                    && next_block_size(chain.head.block_size()).is_some()
                {
                    // P0 #812: re-link the predecessor on the grown
                    // replacement — identical hazard to the forward
                    // `tel_append` path. The reverse chain overflows
                    // when a single `dst` accrues >2047 inbound edges
                    // (an inbound supernode); without re-linking, the
                    // grown overflow head orphans the predecessor chain
                    // and `scan_in` silently drops every earlier inbound
                    // edge. See the forward-path comment for the full
                    // mechanism.
                    let inherited_prev = chain.head.prev_block_ptr();
                    let grown = chain
                        .head
                        .grown()
                        .expect("grown() returns Some below MAX_BLOCK_BYTES");
                    if let Some(prev) = inherited_prev {
                        grown.set_prev_block_ptr(prev).expect(
                            "freshly grown block has no predecessor link yet (write-once invariant)",
                        );
                    }
                    let slot = grown
                        .append(entry)
                        .expect("freshly grown block has room for at least one entry");
                    let new_block = Arc::new(grown);
                    let new_page = self.alloc_virtual_page();
                    self.reverse_tel_blocks
                        .insert((tenant, new_page), Arc::clone(&new_block));
                    chain.head = new_block;
                    chain.head_page = new_page;
                    Ok((new_page, slot))
                } else {
                    // Overflow chain — hot-vertex warning fires on
                    // the reverse path too (per ADR-131 §D-3 +
                    // design-v2 §3.3 supernode discipline).
                    let mut chain_depth: u64 = 1;
                    let mut walker: Arc<TelBlock> = Arc::clone(&chain.head);
                    while let Some(pid) = walker.prev_block_ptr() {
                        match self.reverse_tel_block(tenant, pid) {
                            Some(older) => {
                                walker = older;
                                chain_depth += 1;
                            }
                            None => break,
                        }
                    }
                    tracing::warn!(
                        tenant_id = tenant.raw(),
                        dst_id = dst.raw(),
                        channel = channel.raw(),
                        chain_depth = chain_depth,
                        block_bytes = chain.head.block_size(),
                        "REVERSE TEL overflow block allocated — inbound \
                         fan-in approaching supernode threshold \
                         (design-v2 §3.3; ADR-131 §D-3 HOT_VERTEX_WARNING)"
                    );
                    // W28 Feature #582 (ADR-045) — inbound fan-in is
                    // the same per-tenant hot-vertex signal as the
                    // forward path; fire the §10.2 line 721 counter
                    // here too. No-op when no sink is wired.
                    self.record_hot_vertex_warning(tenant);

                    let old_head_page = chain.head_page;
                    let fresh = Arc::new(
                        TelBlock::new(dst, channel, MIN_BLOCK_BYTES, tenant)
                            .map_err(CrudError::from)?,
                    );
                    fresh.set_prev_block_ptr(old_head_page)?;
                    let slot = fresh
                        .append(entry)
                        .expect("fresh MIN-sized block has room for exactly one entry");
                    let new_page = self.alloc_virtual_page();
                    self.reverse_tel_blocks
                        .insert((tenant, new_page), Arc::clone(&fresh));
                    chain.head = fresh;
                    chain.head_page = new_page;
                    Ok((new_page, slot))
                }
            }
            Err(other) => Err(CrudError::from(other)),
        }
    }

    /// Fetch a REVERSE TEL block by synthetic page id. Used by
    /// [`scan_in`] for overflow-chain traversal.
    #[doc(hidden)]
    #[must_use]
    pub fn reverse_tel_block(&self, tenant: TenantId, page: PageId) -> Option<Arc<TelBlock>> {
        #[cfg(test)]
        self.reverse_tel_block_fetches
            .fetch_add(1, Ordering::Relaxed);
        self.reverse_tel_blocks
            .get(&(tenant, page))
            .map(|e| Arc::clone(&e))
    }

    #[cfg(test)]
    fn reverse_tel_block_fetches_for_test(&self) -> u64 {
        self.reverse_tel_block_fetches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn reset_reverse_tel_block_fetches_for_test(&self) {
        self.reverse_tel_block_fetches.store(0, Ordering::Relaxed);
    }

    /// Fetch the current REVERSE chain head for `(tenant, dst, channel)`.
    #[doc(hidden)]
    #[must_use]
    pub fn reverse_tel_head(
        &self,
        tenant: TenantId,
        dst: NodeId,
        channel: LabelId,
    ) -> Option<(PageId, Arc<TelBlock>)> {
        self.reverse_tel_chains
            .get(&(tenant, dst, channel))
            .map(|chain| {
                let guard = chain.lock();
                (guard.head_page, Arc::clone(&guard.head))
            })
    }

    /// Snapshot the REVERSE TEL chain heads for every channel under
    /// `(tenant, dst)`. Mirrors [`Self::tel_heads_for_src`] for the
    /// reverse map; deterministic order by `channel.raw()`.
    fn reverse_tel_heads_for_dst(
        &self,
        tenant: TenantId,
        dst: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        let mut out: Vec<(LabelId, Arc<TelBlock>, PageId)> = self
            .reverse_tel_channels_by_dst
            .get(&(tenant, dst))
            .map(|channels| {
                channels
                    .iter()
                    .filter_map(|&ch| {
                        self.reverse_tel_chains
                            .get(&(tenant, dst, ch))
                            .map(|chain| {
                                let g = chain.value().lock();
                                (ch, Arc::clone(&g.head), g.head_page)
                            })
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by_key(|(ch, _, _)| ch.raw());
        out
    }

    /// P0 #780 — recovery hook: reinstate the in-memory TEL forward +
    /// reverse adjacency for ONE recovered relationship.
    ///
    /// The rel RECORD survives a durable restart (WAL replay reinstates it
    /// into the MVCC + record stores via the CommitBundle), but the TEL
    /// adjacency chains do NOT participate in the CommitBundle (the
    /// MVCC↔TEL atomicity gap, issue #20) and [`Self::tel_append`] had no
    /// replay caller — so after a restart `scan_out` / `scan_in` walk
    /// empty chains and `MATCH ()-[r]->()` counts read 0 (#780).
    ///
    /// This is the recovery-time analogue of the live commit drain
    /// (`commit` §"Drain TEL appends AFTER commit", lines ~3593): for a
    /// single edge it performs the identical forward + reverse appends
    /// with the identical channel projection (`channel = LabelId::new(ty.raw())`,
    /// see [`create_rel`]). `tel_append_reverse` short-circuits to
    /// a no-op when the reverse index is disabled, exactly as in the drain.
    ///
    /// `commit_lsn` is the visibility LSN stamped into the reinstated
    /// [`TelEntry`]. The caller ([`crate::recovery::rebuild_all_tenant_adjacency`])
    /// passes the recovered watermark (`applied_commit_lsn`); every
    /// post-recovery reader snapshot is `>= applied_commit_lsn`, so an
    /// entry stamped at the watermark is visible at every reachable
    /// snapshot (the MVCC kernel probe in `scan_out` remains the
    /// authoritative tombstone filter).
    ///
    /// `rec` is the recovered [`RelRecord`] (decoded from the MVCC-visible
    /// rel bytes); its `src_id` / `dst_id` / `type_id` / `id` carry the
    /// topology this rebuilds. The record's own `created_lsn` field is NOT
    /// used for the TEL entry (it is `Lsn::ZERO` as written by the create
    /// path); `commit_lsn` above is the authoritative visibility LSN.
    pub(crate) fn reinstate_rel_adjacency(
        &self,
        tenant: TenantId,
        rec: &RelRecord,
        commit_lsn: Lsn,
    ) -> Result<(), CrudError> {
        let src = NodeId::new(rec.src_id);
        let dst = NodeId::new(rec.dst_id);
        let rel = RelId::new(rec.id);
        let channel = LabelId::new(rec.type_id);
        self.tel_append(tenant, src, channel, dst, rel, commit_lsn)?;
        self.tel_append_reverse(tenant, dst, channel, src, rel, commit_lsn)?;
        Ok(())
    }

    /// #1380 — recovery-time reconciliation of a single MVCC-visible
    /// record's PRIMARY (id → `(page, slot)`) and, for nodes, SECONDARY
    /// (property → node) index entries.
    ///
    /// # The bug this heals (#1380 — dual-write split-brain)
    ///
    /// The live commit drain (`crud::commit`) commits the MVCC record in
    /// Phase 1, then attempts the primary/secondary index install as a
    /// dual write. Per ADR-023 an index-install FAILURE **degrades but
    /// does not fail** the commit — the drain logs a `tracing::warn!` and
    /// `continue`s (see the `Err(e) =>` arm ~line 3766). The result: the
    /// MVCC record is committed and durable, but its primary-id (and any
    /// secondary-label) index entry is MISSING. The node is then
    /// SCAN-visible (MVCC is authoritative) yet PERMANENTLY absent from
    /// `read_node_with_store`'s primary fast-path / secondary label lookup
    /// — and, because the primary/secondary indices were NOT rebuilt at
    /// recovery (only stats + TEL were), the split-brain SURVIVES restart.
    ///
    /// # The fix (mirrors [`Self::reinstate_rel_adjacency`] + `stats`/`tel`
    /// rebuild)
    ///
    /// This is the recovery-time analogue of the live commit's index
    /// install. MVCC is authoritative (ADR-023 / ADR-030), so the caller
    /// ([`crate::recovery::rebuild_all_tenant_index`]) walks every
    /// MVCC-visible record at the recovered watermark via
    /// [`TxnManager::for_each_visible_record_with_created_lsn`] — the
    /// SAME per-tenant walk `stats`/`tel` rebuild use, carrying the
    /// version's authoritative visibility LSN — and calls this for each.
    /// For a record whose primary entry is ALREADY present (the normal
    /// case, and the re-run case) this is a NO-OP (see idempotency below);
    /// for a split-brained record it re-installs the record page, points
    /// the primary index at it, and re-publishes the node's secondary
    /// property entries.
    ///
    /// # Idempotency
    ///
    /// The primary-lookup gate makes this idempotent: if `primary.lookup`
    /// finds the key we return `Ok(false)` without touching any store, so
    /// (a) a normally-committed record is untouched, (b) a second recovery
    /// pass re-installs nothing, and (c) re-installing a present entry is
    /// never a duplicate.
    ///
    /// # Durability posture (same as `tel`/`stats` rebuild)
    ///
    /// The reinstalled record page + index entries are derivative state
    /// re-derived from the authoritative MVCC store on every restart, so —
    /// exactly like the TEL adjacency rebuild (#780) and stats rebuild
    /// (M4-41) — this pass restores the in-memory index for the current
    /// process. The standalone `primary.upsert` folds its own IndexPage
    /// snapshot into a `CommitBundle` when a WAL is attached (durifying the
    /// heal), and re-derives it from MVCC if not; either way the next
    /// restart reconciles again, so an existing corrupt data-dir is
    /// healed on first recovery and stays healed.
    ///
    /// Returns `Ok(true)` if the record's index entry was MISSING and got
    /// reinstated, `Ok(false)` if it was already present (no-op). Errors
    /// from the underlying install/upsert are propagated so the caller can
    /// count + surface per-record failures without failing the tenant.
    pub(crate) fn reinstate_record_index(
        &self,
        tenant: TenantId,
        key: MvccKey,
        bytes: &[u8],
        created_lsn: Lsn,
    ) -> Result<bool, CrudError> {
        // Dual-write disabled (no primary index): nothing to reconcile.
        let Some(primary) = self.primary.as_ref() else {
            return Ok(false);
        };

        // MvccKey namespace split (per `REL_TAG_BIT`): bit 63 clear ⇒
        // Node, set ⇒ Rel. Disjoint by construction. Decode once to learn
        // the record's kind + logical id.
        //
        // #1616: the record payload's own `created_lsn` is not a reliable
        // visibility LSN. The v8 / non-delta path stores the payload with
        // the canonical `Lsn::ZERO` placeholder, while the MVCC VERSION
        // always carries the authoritative value. The caller threads that
        // version LSN in from
        // `TxnManager::for_each_visible_record_with_created_lsn`.
        let (kind, id) = if key & REL_TAG_BIT == 0 {
            let rec = decode_node_bytes(bytes)?;
            (RecordKind::Node, rec.id)
        } else {
            let rec = decode_rel_bytes(bytes)?;
            (RecordKind::Rel, rec.id)
        };

        let pk = PrimaryKey::new(tenant, kind, id);
        // Idempotency gate: a present primary entry means the dual-write
        // succeeded (or a prior reconcile pass already healed it). No-op.
        if primary.lookup(pk)?.is_some() {
            return Ok(false);
        }

        // Split-brain: the MVCC record is visible but the primary index
        // has no entry. Re-install the record page from the MVCC bytes at
        // the record's own committed LSN and point the primary index at
        // the new slot. `install_create` re-stamps the created_lsn from
        // the value we pass, so the reinstalled slot carries the same
        // visibility LSN it committed at (`read_node_with_store`'s
        // snapshot gate then behaves identically to a normal commit).
        //
        // `Lsn::ZERO` is the payload placeholder / pre-first-commit clock
        // value, never a committed version LSN. Refuse it rather than
        // publishing a slot visible to every snapshot and violating the
        // v9 base loader's ascending-LSN replay contract.
        if created_lsn == Lsn::ZERO {
            return Err(CrudError::Mvcc(ArcGraphError::TransactionAborted {
                reason: format!(
                    "#1380 index reconcile: record (tenant {}, kind {kind:?}, id {id}) has a \
                     zero MVCC created_lsn; refusing to install a record slot with a \
                     fabricated visibility LSN (issue #1616)",
                    tenant.raw(),
                ),
            }));
        }
        let mut record_emits: Vec<StagedEmit> = Vec::new();
        let mut mutation_log = TxnMutationLog::new();
        let (page_id, slot_id) = install_create(
            self,
            0,
            tenant,
            kind,
            bytes,
            created_lsn,
            &mut mutation_log,
            &mut record_emits,
        )?;
        primary
            .upsert(pk, PageSlot::new(page_id, slot_id))
            .map_err(CrudError::from)?;

        // Secondary (property → node) reinstall for nodes only. Rels are
        // not indexed in the secondary (M2.d), mirroring the live drain's
        // `RecordKind::Node` guard. Best-effort: a secondary publish
        // failure logs + does not fail the primary heal (per ADR-023 —
        // the primary lookup is already restored, which is the correctness
        // fix; the secondary is a read accelerator).
        if kind == RecordKind::Node
            && let Some(secondary) = self.secondary.as_ref()
        {
            let rec = decode_node_bytes(bytes)?;
            let label = LabelId::new(rec.label_id);
            let node_id = NodeId::new(rec.id);
            for (prop_key, val) in node_properties(&rec) {
                if let Err(e) = secondary.insert_property(tenant, label, prop_key, val, node_id) {
                    tracing::warn!(
                        tenant_raw = tenant.raw(),
                        node_id = rec.id,
                        error = %e,
                        "#1380 index reconcile: secondary property reinstall failed; \
                         primary lookup restored, label lookup will miss until next write",
                    );
                }
            }
        }
        Ok(true)
    }

    /// Benchmark hook for V11-S-02 TEL-only fixture construction.
    #[doc(hidden)]
    pub fn reinstate_rel_adjacency_for_bench(
        &self,
        tenant: TenantId,
        rec: &RelRecord,
        commit_lsn: Lsn,
    ) -> Result<(), CrudError> {
        self.reinstate_rel_adjacency(tenant, rec, commit_lsn)
    }

    /// Drain the pending-TEL buffer for an aborted or committed txn.
    fn take_pending(&self, txn_id: u64) -> Vec<PendingTelAppend> {
        self.pending_tel
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    fn buffer_pending(&self, txn_id: u64, p: PendingTelAppend) {
        self.pending_tel.entry(txn_id).or_default().push(p);
    }

    /// Discard pending TEL appends and blob emits for `txn_id`. Call on
    /// abort paths where `crud::commit` is not invoked.
    pub fn discard_pending(&self, txn_id: u64) {
        let _ = self.pending_tel.remove(&txn_id);
        self.discard_pending_blob_emits(txn_id);
        // A failed v10 commit left these staged: `take_idempotency_bindings` /
        // `take_acl_grants` only drain on the SUCCESS path, so an abort used to
        // leak the per-txn entries in the DashMap for the process lifetime.
        // Both discards are idempotent no-ops after a successful commit.
        self.discard_pending_idempotency_bindings(txn_id);
        self.discard_pending_acl_grants(txn_id);
    }

    // ---- M2-CUTOVER: pending record-page installs ----

    fn buffer_install(&self, txn_id: u64, inst: PendingInstall) {
        self.pending_installs.entry(txn_id).or_default().push(inst);
    }

    fn take_installs(&self, txn_id: u64) -> Vec<PendingInstall> {
        self.pending_installs
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    /// Discard any pending record-page installs for `txn_id`. Call on
    /// abort paths where `crud::commit` is not invoked.
    pub fn discard_pending_installs(&self, txn_id: u64) {
        let _ = self.pending_installs.remove(&txn_id);
    }

    // ---- M4-41: per-tenant catalog stats access ----

    /// Get-or-create the [`CatalogStats`](crate::catalog::CatalogStats)
    /// instance for `tenant`. Internal helper used by the post-commit
    /// stats hook: the first commit that touches a tenant materialises
    /// its stats; subsequent commits hit the cached `Arc`.
    fn tenant_catalog_stats(&self, tenant: TenantId) -> Arc<crate::catalog::CatalogStats> {
        Arc::clone(
            self.catalog_stats
                .entry(tenant)
                .or_insert_with(|| Arc::new(crate::catalog::CatalogStats::new()))
                .value(),
        )
    }

    /// Public read-side accessor for a tenant's catalog stats. Returns
    /// `None` until the first commit for that tenant has fired (matches
    /// the `CatalogProvider::label_cardinality` documented "no-stats"
    /// sentinel).
    ///
    /// Consumed by the future production `CatalogProvider` impl in
    /// `arcgraph-query`'s executor wiring (M4-08+); also exercised
    /// by integration tests in `tests/catalog_stats_integration.rs`.
    #[must_use]
    pub fn catalog_stats(&self, tenant: TenantId) -> Option<Arc<crate::catalog::CatalogStats>> {
        self.catalog_stats
            .get(&tenant)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// M4-41 cold-start rebuild surface (per ADR-038 amendment-06
    /// §D-25.1): get-or-create the per-tenant `CatalogStats` handle.
    ///
    /// Used by [`crate::recovery::stats_rebuild::rebuild_catalog_stats_for_tenant`]
    /// to obtain an `Arc<CatalogStats>` it can call
    /// `begin_commit_observation()` / `observe_commit()` on during the
    /// per-tenant cold-start MVCC walk. The read-only [`Self::catalog_stats`]
    /// returns `None` for un-materialised tenants and therefore cannot
    /// initialise the substrate; this method does.
    ///
    /// The live commit-pipeline path (`crud::commit`) uses the private
    /// `tenant_catalog_stats` for the same get-or-create, but recovery
    /// runs from outside the crate's commit hot path and needs an
    /// explicit public entry point.
    #[must_use]
    pub fn init_catalog_stats(&self, tenant: TenantId) -> Arc<crate::catalog::CatalogStats> {
        self.tenant_catalog_stats(tenant)
    }

    // ---- N-2 (issue #81): pending blob chain page emits ----

    /// Buffer a batch of blob chain-page `StagedEmit`s produced by
    /// an `apply_to_*` call against `PropertyData::Blob`. Appends
    /// to the per-txn queue; an empty batch is a no-op.
    fn buffer_blob_emits(&self, txn_id: u64, emits: Vec<StagedEmit>) {
        if emits.is_empty() {
            return;
        }
        self.pending_blob_emits
            .entry(txn_id)
            .or_default()
            .extend(emits);
    }

    /// Remove and return the transaction's blob chain-page emits
    /// accumulated during the builder phase. Drained once by
    /// [`commit`] and folded into the v3 `CommitBundle`'s
    /// `staged_pages` section.
    ///
    /// Applies a `(tenant, page_id) → last-writer-wins` dedupe: a
    /// single blob write allocates distinct page ids per chunk
    /// (see `BlobStore::alloc_page_range`) so the common case is a
    /// no-op, but a pathological commit that drives two writes
    /// colliding on `(tenant, page_id)` (out-of-band staging) would
    /// otherwise emit duplicate bundle entries. Inline here per the
    /// N-2 prompt; N-5 / issue #84 tracks generalising this.
    fn take_blob_emits(&self, txn_id: u64) -> Vec<StagedEmit> {
        let raw: Vec<StagedEmit> = self
            .pending_blob_emits
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default();
        if raw.len() <= 1 {
            return raw;
        }
        use std::collections::HashMap;
        // Keep the last emit for each page_id while preserving the
        // original order for deterministic bundle encoding.
        //
        // (We don't need a tenant in the dedup key because the
        // store-level allocator assigns unique page ids across
        // the whole BlobStore; a single-tenant bundle would never
        // see a collision.)
        let mut last_seen: HashMap<PageId, usize> = HashMap::with_capacity(raw.len());
        for (i, emit) in raw.iter().enumerate() {
            last_seen.insert(emit.page_id, i);
        }
        let mut out: Vec<StagedEmit> = Vec::with_capacity(last_seen.len());
        for (i, emit) in raw.into_iter().enumerate() {
            // Keep only the latest occurrence of each page_id;
            // preserves chain-order for the unique case.
            if last_seen.get(&emit.page_id) == Some(&i) {
                out.push(emit);
            }
        }
        out
    }

    /// Discard any pending blob chain-page emits for `txn_id`. Call
    /// on abort paths where `crud::commit` is not invoked.
    ///
    /// v2 M1: ALSO discards the txn's private slotted prop-page scratch
    /// (restoring any pool-checked-out page to its checkout-time state)
    /// so an explicit abort leaks neither scratch entries nor pool
    /// capacity — the slotted sibling of the chain-emit discard.
    pub fn discard_pending_blob_emits(&self, txn_id: u64) {
        let _ = self.pending_blob_emits.remove(&txn_id);
        self.blobs.rollback_txn_slotted(txn_id);
    }

    // ---- M3.a Slice G.4: pending vector arena page emits ----

    /// Stage a vector arena page mutation for the v5 `CommitBundle`
    /// drain at commit time.
    ///
    /// Producers (M3.b vector writers in `arcgraph-vector`,
    /// post-Slice G.5 / G.7) call this AFTER capturing the page's
    /// post-mutation bytes under the arena's write latch. The bytes
    /// are buffered per-txn; [`commit`] drains the buffer into the
    /// bundle's `vector_pages` section, where the bytes ride the
    /// same group-commit fsync as primary writes.
    ///
    /// `partition` MUST be `PartitionId::ZERO`; `index_id` MUST be
    /// 0 (single index per tenant).
    /// `bytes.len()` MUST equal [`PAGE_SIZE`] (enforced by the
    /// `Box<[u8; PAGE_SIZE]>` shape).
    ///
    /// Mirrors `Self::buffer_blob_emits` for the vector path. Per
    /// ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
    #[allow(clippy::too_many_arguments)] // staging API mirrors the v5 wire shape.
    pub fn stage_vector_page(
        &self,
        txn_id: u64,
        tenant: TenantId,
        partition: arcgraph_core::PartitionId,
        index_id: u64,
        page_id: PageId,
        bytes: Box<[u8; PAGE_SIZE]>,
    ) {
        debug_assert_eq!(
            partition,
            arcgraph_core::PartitionId::ZERO,
            "M3.a Slice G.4: partition MUST be PartitionId::ZERO"
        );
        debug_assert_eq!(
            index_id, 0,
            "M3.a Slice G.4: index_id MUST be 0 at v1.0 (single-index per tenant; \
             multi-index lift is v1.1)"
        );
        self.pending_vector_emits
            .entry(txn_id)
            .or_default()
            .push(PendingVectorEmit {
                tenant,
                partition,
                index_id,
                page_id,
                bytes,
            });
    }

    /// Remove and return the transaction's vector arena page emits
    /// accumulated during the builder phase. Drained once by
    /// [`commit`] and folded into the v5 `CommitBundle`'s
    /// `vector_pages` section.
    ///
    /// Mirrors [`Self::take_blob_emits`]. Unlike the blob path we
    /// do NOT dedupe here at v1.0 — vector writers stage exactly
    /// one emit per (tenant, page_id) per commit by construction
    /// (the arena's write latch serializes per-page mutations).
    /// G.5 / G.7 may add a debug-only assertion when their writers
    /// land.
    pub(crate) fn take_vector_emits(&self, txn_id: u64) -> Vec<PendingVectorEmit> {
        self.pending_vector_emits
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    /// Discard any pending vector arena-page emits for `txn_id`.
    /// Call on abort paths where `crud::commit` is not invoked.
    pub fn discard_pending_vector_emits(&self, txn_id: u64) {
        let _ = self.pending_vector_emits.remove(&txn_id);
    }

    // ─── #352 Part 2 — idempotency binding staging (ADR-199 v6 fold) ──

    /// #352 Part 2 (ADR-199): stage one `external_id → internal_id`
    /// idempotency binding for `txn_id`. Called by `arcgraph-mcp`'s
    /// ingest path AFTER `create_node` / `create_rel` returns the
    /// `internal_id`, so the binding can ride the same transaction's v6
    /// `CommitBundle` (`idempotency_bindings` section) and be durified
    /// **atomically** with the node/rel write.
    ///
    /// `kind` is an opaque discriminator (mcp maps `IdempotencyKind`:
    /// `0 = Node`, `1 = Rel`); storage attaches no meaning to it. The
    /// binding is drained at commit by `Self::take_idempotency_bindings`
    /// and discarded on abort by
    /// [`Self::discard_pending_idempotency_bindings`]. Mirrors
    /// [`Self::stage_vector_page`].
    #[allow(clippy::too_many_arguments)] // one explicit durable binding tuple plus txn owner.
    pub fn stage_idempotency_binding(
        &self,
        txn_id: u64,
        tenant: TenantId,
        kind: u8,
        external_id: String,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) {
        self.pending_idempotency_bindings
            .entry(txn_id)
            .or_default()
            .push(PendingIdempotencyBinding {
                entry: crate::wal::bundle::IdempotencyBindingEntry {
                    op: crate::wal::bundle::IdempotencyBindingOp::Install,
                    tenant,
                    kind,
                    internal_id,
                    external_id,
                },
                payload_hash,
            });
    }

    /// #1010 / ADR-199 amendment: stage release of one
    /// `external_id -> internal_id` idempotency binding for `txn_id`.
    /// Drained into the same v7 `CommitBundle` section as installs so a
    /// delete and its binding release are durable atomically.
    pub fn stage_idempotency_release(
        &self,
        txn_id: u64,
        tenant: TenantId,
        kind: u8,
        internal_id: u64,
        external_id: String,
    ) {
        self.pending_idempotency_bindings
            .entry(txn_id)
            .or_default()
            .push(PendingIdempotencyBinding {
                entry: crate::wal::bundle::IdempotencyBindingEntry {
                    op: crate::wal::bundle::IdempotencyBindingOp::Release,
                    tenant,
                    kind,
                    internal_id,
                    external_id,
                },
                payload_hash: None,
            });
    }

    /// Remove and return the transaction's staged idempotency bindings.
    /// Drained once by [`commit`] and folded into the v6 `CommitBundle`'s
    /// `idempotency_bindings` section. Mirrors [`Self::take_vector_emits`].
    fn take_idempotency_bindings(&self, txn_id: u64) -> Vec<PendingIdempotencyBinding> {
        self.pending_idempotency_bindings
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    /// Discard any pending idempotency bindings for `txn_id`. Call on
    /// abort paths where [`commit`] is not invoked, mirroring
    /// [`Self::discard_pending_vector_emits`].
    pub fn discard_pending_idempotency_bindings(&self, txn_id: u64) {
        let _ = self.pending_idempotency_bindings.remove(&txn_id);
    }

    /// #1221 (ADR-218): stage one document-level ACL grant/revoke op for
    /// `txn_id`. Called by the ACL write-through (`AclWalSink`) so the op
    /// rides the same transaction's v8 `CommitBundle` (`acl_grants`
    /// section) and is durified **atomically** with the commit. The op is
    /// drained at commit by `Self::take_acl_grants` and discarded on
    /// abort by [`Self::discard_pending_acl_grants`]. Mirrors
    /// [`Self::stage_idempotency_binding`].
    ///
    /// **Order is preserved** (`push`): the per-txn `Vec` carries the
    /// staging order through to the `acl_grants` encoder, which must NOT
    /// re-sort (ADR-218 last-writer-wins invariant).
    pub fn stage_acl_grant(&self, txn_id: u64, entry: crate::wal::bundle::AclGrantEntry) {
        self.pending_acl_grants
            .entry(txn_id)
            .or_default()
            .push(entry);
    }

    /// Remove and return the transaction's staged ACL grant/revoke ops,
    /// in staging (append) order. Drained once by [`commit`] and folded
    /// into the v8 `CommitBundle`'s `acl_grants` section. Mirrors
    /// [`Self::take_idempotency_bindings`].
    pub(crate) fn take_acl_grants(&self, txn_id: u64) -> Vec<crate::wal::bundle::AclGrantEntry> {
        self.pending_acl_grants
            .remove(&txn_id)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    /// Discard any pending ACL grant/revoke ops for `txn_id`. Call on
    /// abort paths where [`commit`] is not invoked, mirroring
    /// [`Self::discard_pending_idempotency_bindings`].
    pub fn discard_pending_acl_grants(&self, txn_id: u64) {
        let _ = self.pending_acl_grants.remove(&txn_id);
    }

    /// M3.a Slice G.5 — capture-and-stage helper for vector arena
    /// page mutations. Mirrors the `capture_and_write` (record store)
    /// and `capture_and_latch` (primary index) pattern at the
    /// CRUD-API boundary so future vector writers (G.7+) get a
    /// single-call surface for the dual capture (pre-W into mutation
    /// log, post-W into commit-bundle staging) that ADR-033 §3
    /// requires.
    ///
    /// At v1.0 the production callsite is the test surface — vector
    /// CRUD operations land in `arcgraph-vector` (F.4/F.5 frozen) and
    /// reach this helper through the public seam below. The helper
    /// itself is callable today; the production wirings come at
    /// G.7+.
    ///
    /// Idempotent within a transaction: a second call for the same
    /// `(tenant, page_id)` no-ops on the capture leg (Y-2 compound
    /// dedup via [`TxnMutationLog::has_captured`]) but still pushes
    /// a new staging emit (post-W bytes layer onto each other; the
    /// bundle drains the LATEST emit at commit time).
    ///
    /// Per ADR-031 amendment-02 + ADR-033 §3 + ADR-035 §7.5.
    #[allow(clippy::too_many_arguments)] // staging-API mirrors the v5 wire shape.
    pub fn capture_and_stage_vector_page(
        &self,
        log: &mut TxnMutationLog,
        txn_id: u64,
        tenant: TenantId,
        partition: arcgraph_core::PartitionId,
        index_id: u64,
        page_id: PageId,
        pre_w_bytes: &[u8; PAGE_SIZE],
        post_w_bytes: Box<[u8; PAGE_SIZE]>,
    ) {
        debug_assert_eq!(
            partition,
            arcgraph_core::PartitionId::ZERO,
            "M3.a Slice G.5: partition MUST be PartitionId::ZERO at v1.0"
        );
        debug_assert_eq!(index_id, 0, "M3.a Slice G.5: index_id MUST be 0 at v1.0");
        if !log.has_captured(PageStoreKind::Vector, page_id) {
            log.page_mutations
                .push((PageStoreKind::Vector, page_id, Box::new(*pre_w_bytes)));
        }
        self.stage_vector_page(txn_id, tenant, partition, index_id, page_id, post_w_bytes);
    }

    /// One-shot migration: walk the current record-page state and
    /// populate the primary index from it (DEC-14).
    ///
    /// Idempotent — calls [`PrimaryIndex::bootstrap_from_mvcc`] with an
    /// iterator collected from the record store, so already-indexed
    /// keys end up in `stats.skipped`. Accepts `_txn_mgr` for signature
    /// symmetry with the M2.e full-MVCC-walk variant; the alpha
    /// implementation only relies on state already materialized by
    /// CUTOVER drains. Returns `Ok(default)` when dual-write is
    /// disabled (no primary / no records), matching the "no-op on
    /// fresh database" guarantee.
    pub fn bootstrap_primary_index(
        &self,
        _txn_mgr: &Transaction<'_>,
    ) -> Result<BootstrapStats, CrudError> {
        let (primary, records) = match (self.primary.as_ref(), self.record_backend()) {
            (Some(p), Some(r)) => (p, r),
            _ => return Ok(BootstrapStats::default()),
        };
        let mut entries: Vec<(PrimaryKey, PageSlot)> = Vec::new();
        // Walk EVERY page the record store tracks — not just the
        // "currently open" per-(tenant, kind) page. `rotate_open_page`
        // removes the previously-open entry from `open_pages` before
        // inserting its successor, so iterating `open_pages` alone
        // misses every page that has ever rotated out (any tenant
        // with more than NODE_CAPACITY records). Walk via
        // `RecordPageStore::iter_pages` instead, determining tenant
        // and record kind from the page header on each visit.
        for (_tenant, page_id, latch) in records.iter_pages_qualified() {
            let g = latch.read();
            let Ok(page) = SlottedPageRef::open(g.as_ref().as_ref()) else {
                continue;
            };
            let hdr = page.header();
            let tenant = TenantId::new(hdr.tenant_id);
            let page_type = match arcgraph_core::PageType::from_byte(hdr.page_type) {
                Ok(pt) => pt,
                Err(_) => continue,
            };
            match page_type {
                arcgraph_core::PageType::Node => {
                    for (slot, rec) in page.iter_nodes() {
                        entries.push((
                            PrimaryKey::new(tenant, RecordKind::Node, rec.id),
                            PageSlot::new(page_id, slot),
                        ));
                    }
                }
                arcgraph_core::PageType::Rel => {
                    for (slot, rec) in page.iter_rels() {
                        entries.push((
                            PrimaryKey::new(tenant, RecordKind::Rel, rec.id),
                            PageSlot::new(page_id, slot),
                        ));
                    }
                }
                // Index pages etc. are not record pages; skip.
                _ => continue,
            }
        }
        primary
            .bootstrap_from_mvcc(entries)
            .map_err(CrudError::from)
    }

    /// Get-or-create the "open" page id for `(tenant, kind)`. Allocates
    /// a fresh page and installs it in the record store when called
    /// for the first time on a given pair, or when the currently-open
    /// page is full.
    #[cfg_attr(not(test), allow(dead_code))]
    fn open_or_fresh_page(&self, tenant: TenantId, kind: RecordKind) -> Result<PageId, CrudError> {
        self.open_or_fresh_page_inner(tenant, kind, None)
    }

    fn open_or_fresh_page_for_txn(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        mutation_log: &mut TxnMutationLog,
    ) -> Result<PageId, CrudError> {
        self.open_or_fresh_page_inner(tenant, kind, Some(mutation_log))
    }

    fn open_or_fresh_page_inner(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        mutation_log: Option<&mut TxnMutationLog>,
    ) -> Result<PageId, CrudError> {
        let (allocator, records) = match (&self.allocator, self.record_backend()) {
            (Some(a), Some(r)) => (a, r),
            _ => {
                return Err(CrudError::Index(IndexError::CorruptPage {
                    page_id: PageId::ZERO,
                    reason: "dual-write store missing allocator / record store".to_owned(),
                }));
            }
        };
        if let Some(pid) = self.open_pages.get(&(tenant, kind)) {
            return Ok(*pid);
        }
        let page_type = match kind {
            RecordKind::Node => PageType::Node,
            RecordKind::Rel => PageType::Rel,
        };
        // #811: allocate from the record store's SINGLE flat page-id
        // domain (Node + Rel slotted pages share `RecordPageStore`'s
        // `PageId` keyspace), NOT the per-`(tenant, page_type)` counter.
        // Independent Node/Rel counters each start at `PageId(1)`, so the
        // first Rel record page collided with the first Node record page
        // in the shared store — the silent dual-write divergence #811
        // (and the collateral-corruption half of #812). The page is still
        // installed with its REAL `page_type` stamped in the header; only
        // the allocation domain is unified. See
        // `PageAllocator::RECORD_PAGE_DOMAIN`.
        let pid = allocator.alloc_record_page(tenant);
        if let Some(log) = mutation_log {
            records.install_fresh_for_txn(log, pid, page_type, tenant)?;
        } else {
            records.install_fresh(pid, page_type, tenant)?;
        }
        self.open_pages.insert((tenant, kind), pid);
        Ok(pid)
    }

    fn rotate_open_page_for_txn(
        &self,
        tenant: TenantId,
        kind: RecordKind,
        mutation_log: &mut TxnMutationLog,
    ) -> Result<PageId, CrudError> {
        self.open_pages.remove(&(tenant, kind));
        self.open_or_fresh_page_for_txn(tenant, kind, mutation_log)
    }

    fn reserve_record_slot(
        &self,
        txn_id: u64,
        tenant: TenantId,
        kind: RecordKind,
    ) -> Result<(PageId, SlotId, bool), CrudError> {
        let allocator = self
            .allocator
            .as_ref()
            .expect("dual-write store missing allocator");
        let records = self
            .record_backend()
            .expect("dual-write store missing records store");
        let capacity = match kind {
            RecordKind::Node => NODE_CAPACITY,
            RecordKind::Rel => REL_CAPACITY,
        };
        let mut table = self.record_reservations.lock();

        let mut page_id = self.open_pages.get(&(tenant, kind)).map(|entry| *entry);
        if let Some(existing) = page_id {
            let key = RecordPageKey::new(tenant, existing);
            if let std::collections::hash_map::Entry::Vacant(vacant) = table.pages.entry(key) {
                let latch = records.latch_for_tenant(tenant, existing)?;
                let guard = latch.read();
                let page = SlottedPageRef::open(guard.as_ref().as_ref())?;
                vacant.insert(RecordPageShadow {
                    tenant,
                    kind,
                    next_slot: page.slot_count(),
                    released_slots: std::collections::BTreeSet::new(),
                    pending_new_owner: None,
                });
            }
            let shadow = table.pages.get(&key).expect("inserted above");
            if shadow
                .pending_new_owner
                .is_some_and(|owner| owner != txn_id)
                || (shadow.next_slot >= capacity && shadow.released_slots.is_empty())
            {
                page_id = None;
            }
        }

        let page_id = match page_id {
            Some(page_id) => page_id,
            None => {
                let page_id = allocator.alloc_record_page(tenant);
                table.pages.insert(
                    RecordPageKey::new(tenant, page_id),
                    RecordPageShadow {
                        tenant,
                        kind,
                        next_slot: 0,
                        released_slots: std::collections::BTreeSet::new(),
                        pending_new_owner: Some(txn_id),
                    },
                );
                self.open_pages.insert((tenant, kind), page_id);
                page_id
            }
        };

        let (slot, page_is_pending_for_txn) = {
            let shadow = table
                .pages
                .get_mut(&RecordPageKey::new(tenant, page_id))
                .expect("shadow exists");
            debug_assert_eq!((shadow.tenant, shadow.kind), (tenant, kind));
            let slot = match shadow.released_slots.pop_first() {
                Some(slot) => slot,
                None => {
                    let slot = shadow.next_slot;
                    shadow.next_slot = shadow
                        .next_slot
                        .checked_add(1)
                        .expect("record slot shadow exhausted");
                    slot
                }
            };
            (SlotId(slot), shadow.pending_new_owner == Some(txn_id))
        };
        table
            .by_txn
            .entry(txn_id)
            .or_default()
            .push(RecordSlotReservation {
                tenant,
                page_id,
                slot,
            });
        Ok((page_id, slot, page_is_pending_for_txn))
    }

    fn release_record_reservations(&self, txn_id: u64) {
        let mut table = self.record_reservations.lock();
        let Some(reservations) = table.by_txn.remove(&txn_id) else {
            return;
        };
        let mut remove_pages = std::collections::BTreeSet::new();
        for reservation in reservations {
            let key = RecordPageKey::new(reservation.tenant, reservation.page_id);
            if let Some(shadow) = table.pages.get_mut(&key) {
                if shadow.pending_new_owner == Some(txn_id) {
                    remove_pages.insert(key);
                } else {
                    shadow.released_slots.insert(reservation.slot.raw());
                }
            }
        }
        for key in remove_pages {
            table.pages.remove(&key);
            self.open_pages.retain(|tenant_kind, current_page| {
                tenant_kind.0 != key.tenant_id || *current_page != key.page_id
            });
        }
    }

    fn complete_record_reservations(&self, txn_id: u64) {
        let mut table = self.record_reservations.lock();
        let Some(reservations) = table.by_txn.remove(&txn_id) else {
            return;
        };
        for reservation in reservations {
            if let Some(shadow) = table
                .pages
                .get_mut(&RecordPageKey::new(reservation.tenant, reservation.page_id))
                && shadow.pending_new_owner == Some(txn_id)
            {
                shadow.pending_new_owner = None;
            }
        }
    }

    fn apply_durable_v9_deltas(
        &self,
        txn_id: u64,
        deltas: &[DeltaOp],
        commit_lsn: Lsn,
    ) -> arcgraph_core::Result<()> {
        #[cfg(test)]
        if self.fail_durable_v9_apply.load(Ordering::Acquire) {
            return Err(ArcGraphError::Io(std::io::Error::other(
                "injected durable v9 page-apply failure",
            )));
        }
        let physical_extents = self.owner_rows.read().clone();
        if physical_extents.is_none() {
            let records = self
                .record_backend()
                .ok_or_else(|| ArcGraphError::WalCorruption {
                    lsn: commit_lsn,
                    reason: "v9 delta apply requires the record page store".to_owned(),
                })?;
            for delta in deltas.iter().filter(|delta| delta.store_id == STORE_RECORD) {
                let page_id = PageId::new(delta.page_no);
                if delta.kind == DeltaOpKind::PageAlloc {
                    let page_type = PageType::from_byte(delta.payload[0]).map_err(|error| {
                        ArcGraphError::WalCorruption {
                            lsn: delta.op_lsn,
                            reason: format!("invalid live PageAlloc type: {error}"),
                        }
                    })?;
                    // #1457 MF5(b) — the `MissingPage` arm's
                    // `install_fresh` MUST happen only after we hold a
                    // pin covering this exact page_id, not before. The
                    // OLD ordering (`install_fresh` here, THEN
                    // `latch_pinned_for_tenant` below) opened a window
                    // where the fresh page is cache-resident with NO
                    // DPT entry yet (dirty-marking happens only after
                    // the pin, below) — MECH-E1's clean arm reads that
                    // as "durable image current" (false: this page has
                    // never been homed at all) and can reclaim the
                    // sole in-memory copy of a never-written-home page
                    // in exactly that window, so the pinned fault-in a
                    // few lines down fails with a genuine `MissingPage`
                    // (a fail-stop that only recovers on restart/WAL
                    // replay, not a silent loss, but still an
                    // availability regression this reorder closes).
                    // Reordering to pin-THEN-install closes it: once
                    // `latch_pinned_for_tenant` returns, the page is
                    // BOTH pinned and resident (or install_fresh made
                    // it so immediately after, still under the pin's
                    // exclusion), so no clean-arm claim can land in
                    // between.
                    //
                    // `latch_pinned_for_tenant` on a not-yet-installed
                    // page_id fails with `MissingPage` (there is nothing
                    // to fault in yet) — that failure itself carries no
                    // pin, so it's safe to fall through to
                    // `install_fresh` on that specific error and retry
                    // the pinned acquisition once the page exists.
                    let pinned = match records.latch_pinned_for_tenant(delta.tenant_id, page_id) {
                        Ok(pinned) => pinned,
                        Err(RecordStoreError::MissingPage(_)) => {
                            records
                                .install_fresh(page_id, page_type, delta.tenant_id)
                                .map_err(|error| ArcGraphError::WalCorruption {
                                    lsn: delta.op_lsn,
                                    reason: format!("live PageAlloc failed: {error}"),
                                })?;
                            // Defense-in-depth for the residual
                            // install-to-pin gap (`install_fresh` and
                            // the retry below are two separate calls;
                            // nothing but CPU scheduling separates
                            // them): mark the DPT dirty for this page
                            // BEFORE the pin retry below, so that even
                            // if a concurrent evictor's clean-arm
                            // observes the page in exactly that narrow
                            // window, `is_clean` now reports live-dirty
                            // (not "no entry = clean") and the reclaim
                            // routes through the dirty-arm's
                            // checkpointer handshake instead of an
                            // immediate no-I/O clean-arm removal — this
                            // never-homed page fails the checkpointer's
                            // flush (nothing to flush FROM yet) and is
                            // therefore retained. The authoritative mark
                            // (with the real stamped LSN) still happens
                            // below after the redo-stamp for the
                            // production dirty-generation bookkeeping;
                            // this one is a defensive pre-mark.
                            self.mark_m3_dirty(delta);
                            records
                                .latch_pinned_for_tenant(delta.tenant_id, page_id)
                                .map_err(|error| ArcGraphError::WalCorruption {
                                    lsn: delta.op_lsn,
                                    reason: format!("live PageAlloc attach failed: {error}"),
                                })?
                        }
                        Err(error) => {
                            return Err(ArcGraphError::WalCorruption {
                                lsn: delta.op_lsn,
                                reason: format!("live PageAlloc lookup failed: {error}"),
                            });
                        }
                    };
                    let mut guard = pinned.latch().write();
                    let mut page = SlottedPage::open(guard.as_mut().as_mut()).map_err(|error| {
                        ArcGraphError::WalCorruption {
                            lsn: delta.op_lsn,
                            reason: format!("live PageAlloc page invalid: {error}"),
                        }
                    })?;
                    page.apply_redo_if_newer(delta.op_lsn, |_page| {
                        Ok::<(), std::convert::Infallible>(())
                    })
                    .expect("infallible live PageAlloc stamp");
                    self.mark_m3_dirty(delta);
                    drop(guard);
                    drop(pinned);
                    continue;
                }
                // #1521 M6.1 P0-1 — same pin-coupled requirement as the
                // PageAlloc arm above: mutate-then-mark-dirty under a
                // pin held across both, closing the bare-latch
                // revalidate-to-removal race for the physical-delta
                // path (this is the exact site the skeptic's PoC and
                // `crud.rs:3701-3710` cite).
                let pinned = records
                    .latch_pinned_for_tenant(delta.tenant_id, page_id)
                    .map_err(|error| ArcGraphError::WalCorruption {
                        lsn: delta.op_lsn,
                        reason: format!("live delta target missing: {error}"),
                    })?;
                let mut guard = pinned.latch().write();
                crate::redo::apply_physical_delta(guard.as_mut(), delta, commit_lsn)?;
                self.mark_m3_dirty(delta);
                drop(guard);
                drop(pinned);
            }
        }
        self.blobs
            .apply_txn_slotted_deltas(txn_id, deltas, commit_lsn)?;
        for delta in deltas.iter().filter(|delta| delta.store_id == STORE_PROPS) {
            self.mark_m3_dirty(delta);
        }
        self.blobs.publish_txn_slotted(txn_id)?;
        if let Some(extents) = physical_extents {
            extents.apply_committed(deltas, commit_lsn)?;
        }
        self.complete_record_reservations(txn_id);
        Ok(())
    }

    fn mark_m3_dirty(&self, delta: &DeltaOp) {
        if let Some(dpt) = self.m3_dpt.read().as_ref() {
            dpt.mark_dirty(
                DirtyPageKey {
                    tenant_id: delta.tenant_id,
                    store_id: delta.store_id,
                    page_no: delta.page_no,
                },
                delta.op_lsn,
            );
        }
    }

    fn apply_or_defer_v9_deltas(
        &self,
        txn_id: u64,
        deltas: &[DeltaOp],
        commit_lsn: Lsn,
    ) -> arcgraph_core::Result<()> {
        #[cfg(any(debug_assertions, feature = "fault-injection"))]
        let debug_caller_gate = {
            let mut gate = self.debug_deferred_v9_caller_gate.lock();
            gate.as_mut().and_then(|gate| {
                if gate.remaining == 0 {
                    None
                } else {
                    gate.remaining -= 1;
                    Some((Arc::clone(&gate.entered), Arc::clone(&gate.release)))
                }
            })
        };
        #[cfg(any(debug_assertions, feature = "fault-injection"))]
        if let Some((entered, release)) = debug_caller_gate {
            entered.wait();
            release.wait();
        }
        self.drain_deferred_v9_applies()?;
        let wal = self.wal.as_ref().ok_or(ArcGraphError::WalUnavailable)?;
        let mut queue = self.deferred_v9_applies.lock();
        if !queue.entries.is_empty() {
            queue.push_back(DeferredV9Apply {
                txn_id,
                commit_lsn,
                deltas: deltas.to_vec(),
            });
            return Ok(());
        }
        if wal.take_exact_durable(commit_lsn) {
            // Serialize the empty-queue direct path against a concurrent
            // enqueue. Once an older commit is queued, no later commit may
            // jump the per-page LSN before that queue entry applies.
            self.apply_durable_v9_deltas(txn_id, deltas, commit_lsn)
        } else {
            queue.push_back(DeferredV9Apply {
                txn_id,
                commit_lsn,
                deltas: deltas.to_vec(),
            });
            Ok(())
        }
    }

    /// Drain Periodic v9 page applies whose exact WAL records have completed
    /// fsync. The queue is commit-ordered; a non-durable front stops the pass.
    pub fn drain_deferred_v9_applies(&self) -> arcgraph_core::Result<usize> {
        let Some(wal) = self.wal.as_ref() else {
            return Ok(0);
        };
        let mut applied = 0;
        let mut queue = self.deferred_v9_applies.lock();
        while let Some(entry) = queue.entries.front() {
            if !wal.has_exact_durable(entry.commit_lsn) {
                break;
            }
            #[cfg(any(debug_assertions, feature = "fault-injection"))]
            let debug_gate = { self.debug_deferred_v9_apply_gate.lock().take() };
            #[cfg(any(debug_assertions, feature = "fault-injection"))]
            if let Some((entered, release)) = debug_gate {
                // The seam must not retain the queue mutex: the interleaving
                // under test is the next commit snapshotting this front's
                // pending-record marker before any physical delta applies.
                // Re-check the front and its proof after release because a
                // concurrent drain may have completed it in the meantime.
                drop(queue);
                entered.wait();
                release.wait();
                queue = self.deferred_v9_applies.lock();
                continue;
            }
            // Apply while both the queue entry and its exact durability proof
            // remain owned. On error, returning leaves the commit at the
            // front with its proof intact, so every later drain/checkpoint
            // fail-stops behind the same commit instead of reclaiming it.
            self.apply_durable_v9_deltas(entry.txn_id, &entry.deltas, entry.commit_lsn)?;
            let commit_lsn = entry.commit_lsn;
            queue.clear_front_markers();
            queue.entries.pop_front().expect("front applied above");
            let consumed = wal.take_exact_durable(commit_lsn);
            debug_assert!(consumed);
            applied += 1;
        }
        Ok(applied)
    }

    /// Pause the next proven-durable deferred v9 front immediately before
    /// physical apply. Test-only deterministic integration-test seam.
    #[cfg(any(debug_assertions, feature = "fault-injection"))]
    #[doc(hidden)]
    pub fn __test_gate_next_deferred_v9_apply(
        &self,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self.debug_deferred_v9_apply_gate.lock() = Some((entered, release));
    }

    /// Hold the next `callers` deferred-v9 writer calls before they can
    /// drain or enqueue. Test-only RULE-MT rendezvous; callers provide
    /// barriers with themselves plus one controlling test participant.
    #[cfg(any(debug_assertions, feature = "fault-injection"))]
    #[doc(hidden)]
    pub fn __test_gate_deferred_v9_callers(
        &self,
        callers: usize,
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        assert!(callers > 0, "deferred-v9 caller gate needs a participant");
        *self.debug_deferred_v9_caller_gate.lock() = Some(DebugDeferredV9CallerGate {
            remaining: callers,
            entered,
            release,
        });
    }

    /// Return the oldest deferred commit and the first physical op it owns.
    /// The queue is commit ordered, so its front is the only checkpoint clamp
    /// needed. An empty physical set falls back to the commit LSN.
    #[must_use]
    pub fn deferred_v9_boundary(&self) -> Option<DeferredV9Boundary> {
        let queue = self.deferred_v9_applies.lock();
        let front = queue.entries.front()?;
        let redo_lsn = front
            .deltas
            .iter()
            .filter(|delta| delta.kind.is_physical())
            .map(|delta| delta.op_lsn)
            .min()
            .unwrap_or(front.commit_lsn);
        Some(DeferredV9Boundary {
            commit_lsn: front.commit_lsn,
            redo_lsn,
        })
    }

    /// Whether an immediately-published addressed record is at or beyond the
    /// oldest Periodic-v9 physical install still in flight.
    ///
    /// The primary store has an exact `(page, slot)` pending marker. The M4
    /// alternate uses a disjoint arithmetic page space, so commit LSN is the
    /// common identity: because the deferred queue is commit ordered, every
    /// addressed version at or after its front must fall back to MVCC until
    /// the physical queue drains.
    fn addressed_record_install_is_pending(&self, created_lsn: u64) -> bool {
        self.deferred_v9_applies
            .lock()
            .entries
            .front()
            .is_some_and(|front| created_lsn >= front.commit_lsn.raw())
    }
}

// ─────────────────────────────────────────────────────────────────────
// M2-34 secondary-index property wiring
// ─────────────────────────────────────────────────────────────────────

/// Reserved `StringId` used as the property-key stand-in for
/// `NodeRecord::inline_u32a`.
///
/// The current node schema carries two positional inline `u32`
/// properties (see `NodeRecord::inline_u32a` / `inline_u32b`); the
/// secondary index's key demands a `StringId`-typed property key. We
/// reserve two well-known ids for the positional slots; named-property
/// schemas (where callers provide the property-key name) are an M3
/// task once the intern table grows user-facing property names. Ids 1
/// and 2 are below the intern table's first-allocated id (M2-32 starts
/// interning at a freshly-bumped `AtomicU32`), so they never collide
/// with a user-interned name.
pub const INLINE_U32A_PROPERTY_KEY: StringId = StringId::new(1);

/// Reserved `StringId` for `NodeRecord::inline_u32b`. See
/// [`INLINE_U32A_PROPERTY_KEY`].
pub const INLINE_U32B_PROPERTY_KEY: StringId = StringId::new(2);

/// Extract the positional-property set that the secondary index
/// publishes for `rec`. Returns the two `(property_key, value)` pairs
/// corresponding to `inline_u32a` and `inline_u32b`.
#[must_use]
fn node_properties(rec: &NodeRecord) -> [(StringId, SecondaryIndexValue); 2] {
    [
        (
            INLINE_U32A_PROPERTY_KEY,
            SecondaryIndexValue::U32(rec.inline_u32a),
        ),
        (
            INLINE_U32B_PROPERTY_KEY,
            SecondaryIndexValue::U32(rec.inline_u32b),
        ),
    ]
}

/// Fetch `id`'s prior `NodeRecord` through the primary index. Physical page
/// bytes are normally the pre-image because this runs before the current
/// update/delete is installed. When an older Periodic commit still owns the
/// slot's pending marker, those bytes lag the committed MVCC chain, so the
/// caller-threaded MVCC pre-image is authoritative.
///
/// `pending_record_slots` is an owned snapshot taken by the caller. This
/// helper must not acquire `deferred_v9_applies`: commit/drain callers may
/// already own that non-reentrant mutex.
fn read_prior_node_from_store(
    store: &CrudStore,
    tenant: TenantId,
    id: NodeId,
    pending_record_slots: &std::collections::HashSet<DeferredRecordSlot>,
    prior_mvcc: Option<&NodeRecord>,
) -> Option<NodeRecord> {
    let primary = store.primary.as_ref()?;
    let records = store.record_backend()?;
    let key = PrimaryKey::new(tenant, RecordKind::Node, id.raw());
    let slot = primary.lookup(key).ok().flatten()?;
    if pending_record_slots.contains(&DeferredRecordSlot {
        page: RecordPageKey::new(tenant, slot.page),
        slot: slot.slot,
    }) {
        return prior_mvcc.cloned();
    }
    let latch = records.latch_for_tenant(tenant, slot.page).ok()?;
    let g = latch.read();
    let page = SlottedPageRef::open(g.as_ref().as_ref()).ok()?;
    page.read_node(slot.slot).ok().flatten()
}

/// Bundle-aware version of `publish_node_properties_insert` — ADR-031.
/// Returns the staged `IndexPage` snapshots from the secondary
/// writes so the caller can fold them into the outer `CommitBundle`.
/// Errors are logged and skipped (same `tracing::warn!` policy as
/// the pre-ADR-031 shape).
fn publish_node_properties_insert_deferred(
    secondary: &Arc<dyn SecondaryIndexHandle>,
    tenant: TenantId,
    rec: &NodeRecord,
    node_id: NodeId,
    log: &mut TxnMutationLog,
) -> Vec<StagedEmit> {
    let label = LabelId::new(rec.label_id);
    let mut staged: Vec<StagedEmit> = Vec::new();
    for (pk, val) in node_properties(rec) {
        match secondary.insert_property_deferred(tenant, label, pk, val, node_id, log) {
            Ok(emits) => staged.extend(emits),
            Err(e) => tracing::warn!(
                "secondary insert_property_deferred({:?}, label={}, pk={:?}, val={:?}, node={:?}) failed: {}",
                tenant,
                rec.label_id,
                pk,
                val,
                node_id,
                e
            ),
        }
    }
    staged
}

/// **RC-1 (#1366)** — old-value removals collected during the commit
/// drain, to be enqueued on [`CrudStore::deferred_removals`] stamped
/// with the commit's `Lsn`. NOT applied to the B-tree at commit-builder
/// time — that eager path was the false-negative cliff. See
/// [`CrudStore::deferred_removals`].
fn enqueue_node_properties_remove(
    rec: &NodeRecord,
    node_id: NodeId,
    out: &mut Vec<(LabelId, StringId, SecondaryIndexValue, NodeId)>,
) {
    // Tenant is stamped later at enqueue time from the commit's txn
    // tenant (single-tenant per ADR-011), so it is not needed here.
    let label = LabelId::new(rec.label_id);
    for (pk, val) in node_properties(rec) {
        out.push((label, pk, val, node_id));
    }
}

/// **RC-1 (#1366)** — bundle-aware UPDATE maintenance under insert-only
/// commit-path semantics.
///
/// The NEW property values are inserted synchronously into the B-tree
/// now (staged into the outer `CommitBundle`), so a reader on a
/// snapshot at-or-after this commit finds the new value. Every OLD
/// property value whose `(label, value)` changed is pushed into `out`
/// for **deferred** removal — the superseded `(email=a) → n` entry
/// becomes a ghost, kept live until `oldest_active_snapshot()` passes
/// this commit's LSN so no pre-commit snapshot reader can miss `n`.
// Eight params (secondary + tenant + prior + new_rec + node_id + log +
// removals + reassertions): the log threads Z-1 F-1 rollback capture,
// `out` collects RC-1 deferred removals, and `reassertions` collects the
// successful new-side inserts for the #1464 generation guard. These are
// load-bearing MVCC/crash-consistency plumbing, not a refactor smell.
#[allow(clippy::too_many_arguments)]
fn diff_and_publish_node_properties_deferred(
    secondary: &Arc<dyn SecondaryIndexHandle>,
    tenant: TenantId,
    prior: &NodeRecord,
    new_rec: &NodeRecord,
    node_id: NodeId,
    log: &mut TxnMutationLog,
    out: &mut Vec<(LabelId, StringId, SecondaryIndexValue, NodeId)>,
    reassertions: &mut Vec<SecondaryEntry>,
) -> Vec<StagedEmit> {
    let old_props = node_properties(prior);
    let new_props = node_properties(new_rec);
    let old_label = LabelId::new(prior.label_id);
    let new_label = LabelId::new(new_rec.label_id);
    let mut staged: Vec<StagedEmit> = Vec::new();

    // NEW side: insert synchronously (insert-only commit path).
    for (i, (pk, new_val)) in new_props.iter().enumerate() {
        let (_, old_val) = old_props[i];
        let label_changed = old_label != new_label;
        if label_changed || &old_val != new_val {
            match secondary.insert_property_deferred(tenant, new_label, *pk, *new_val, node_id, log)
            {
                Ok(emits) => {
                    staged.extend(emits);
                    reassertions.push(SecondaryEntry {
                        tenant,
                        label: new_label,
                        property_key: *pk,
                        value: *new_val,
                        node: node_id,
                    });
                }
                Err(e) => tracing::warn!(
                    "secondary insert_property_deferred on update (new side) failed: {}",
                    e
                ),
            }
        }
    }
    // OLD side: DEFER the removal past the snapshot horizon (RC-1). Do
    // NOT eagerly zero the slot — that is the exact false-negative the
    // deferred queue exists to prevent.
    for (i, (pk, old_val)) in old_props.iter().enumerate() {
        let (_, new_val) = new_props[i];
        let label_changed = old_label != new_label;
        if label_changed || old_val != &new_val {
            out.push((old_label, *pk, *old_val, node_id));
        }
    }
    staged
}

/// Rewrite the `created_lsn` field in the serialized `NodeRecord`
/// or `RelRecord` bytes to `commit_lsn`. Offsets:
/// - `NodeRecord::created_lsn` lives at bytes 56..64 (see
///   `arcgraph_core::record::NodeRecord::to_bytes`).
/// - `RelRecord::created_lsn` lives at bytes 48..56.
fn fixup_created_lsn(bytes: &mut [u8], kind: RecordKind, commit_lsn: Lsn) {
    let off = match kind {
        RecordKind::Node => 56,
        RecordKind::Rel => 48,
    };
    bytes[off..off + 8].copy_from_slice(&commit_lsn.raw().to_le_bytes());
}

#[allow(clippy::too_many_arguments)] // mirrors one physical target plus payload.
fn stage_record_delta_intent(
    mutation_log: &mut TxnMutationLog,
    tenant: TenantId,
    kind: RecordKind,
    page_id: PageId,
    slot: SlotId,
    op_kind: DeltaOpKind,
    payload: Bytes,
    reserved_fresh_page: bool,
) {
    if !mutation_log.delta_mode {
        return;
    }
    let is_new_page = mutation_log
        .new_pages
        .iter()
        .any(|(store, id)| *store == PageStoreKind::Record && *id == page_id);
    let alloc_already_staged = mutation_log.delta_intents.iter().any(|intent| {
        intent.kind == DeltaOpKind::PageAlloc
            && intent.store_id == STORE_RECORD
            && intent.page_no == page_id.raw()
    });
    if (is_new_page || reserved_fresh_page) && !alloc_already_staged {
        let page_type = match kind {
            RecordKind::Node => PageType::Node,
            RecordKind::Rel => PageType::Rel,
        };
        mutation_log.delta_intents.push(DeltaIntent::page_alloc(
            STORE_RECORD,
            tenant,
            page_id.raw(),
            page_type,
            page_id.raw(),
        ));
    }
    mutation_log.delta_intents.push(DeltaIntent {
        kind: op_kind,
        store_id: STORE_RECORD,
        tenant_id: tenant,
        page_no: page_id.raw(),
        slot: slot.raw(),
        payload,
    });
}

/// Install one pending record into its destination page, rotating the
/// "open" page if it fills up mid-install. Returns the `(page_id,
/// slot_id)` pair that identifies the new slot.
#[allow(clippy::too_many_arguments)] // builder state is explicit at this durability seam.
fn install_create(
    store: &CrudStore,
    txn_id: u64,
    tenant: TenantId,
    kind: RecordKind,
    bytes: &[u8],
    commit_lsn: Lsn,
    mutation_log: &mut TxnMutationLog,
    record_emits: &mut Vec<StagedEmit>,
) -> Result<(PageId, SlotId), CrudError> {
    let records = store
        .record_backend()
        .expect("dual-write store missing records store");
    let mut bytes_local = bytes.to_vec();
    fixup_created_lsn(&mut bytes_local, kind, commit_lsn);
    if mutation_log.delta_mode {
        let (page_id, slot, fresh_page) = store.reserve_record_slot(txn_id, tenant, kind)?;
        stage_record_delta_intent(
            mutation_log,
            tenant,
            kind,
            page_id,
            slot,
            DeltaOpKind::PutRecord,
            Bytes::from(bytes_local),
            fresh_page,
        );
        return Ok((page_id, slot));
    }
    let cap: u16 = match kind {
        RecordKind::Node => NODE_CAPACITY,
        RecordKind::Rel => REL_CAPACITY,
    };
    let mut page_id = store.open_or_fresh_page_for_txn(tenant, kind, mutation_log)?;

    loop {
        // #1457 MF5(a) — pin across the mutate: this legacy (non-delta)
        // path mutates the record page's RAM copy and stages the intent
        // for the outer commit bundle, but does NOT mark the page dirty
        // in the M3 DPT itself (that only happens for the v9 delta path
        // — `apply_durable_v9_deltas`'s `mark_m3_dirty`). Without a pin
        // live across the mutate, a concurrent `evict_for_capacity`
        // sees no DPT entry for this page (`is_clean` returns
        // `Some(true)`) and reclaims it via the CLEAN arm — no I/O,
        // immediate — discarding the sole RAM copy of a mutation that
        // has not yet reached the WAL (the outer commit's bundle build
        // has not fsynced yet). `latch_pinned_for_tenant` closes this:
        // the pin excludes the clean-arm's removal claim for the whole
        // mutate, matching the v9 delta path's own pin discipline.
        let pinned = records.latch_pinned_for_tenant(tenant, page_id)?;
        let mut guard = pinned.latch().write();
        let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
        if page.slot_count() >= cap {
            drop(guard);
            drop(pinned);
            page_id = store.rotate_open_page_for_txn(tenant, kind, mutation_log)?;
            continue;
        }
        let slot = match kind {
            RecordKind::Node => {
                let rec = decode_node_bytes(&bytes_local)?;
                page.insert_node(&rec)?
            }
            RecordKind::Rel => {
                let rec = decode_rel_bytes(&bytes_local)?;
                page.insert_rel(&rec)?
            }
        };
        // PR #79 X-2 fold-in: stage the post-mutation record
        // page bytes into the commit bundle so replay can
        // reconstruct `RecordPageStore`. Without this, post-
        // replay `read_node_with_store` hits `MissingPage`.
        record_emits.push(snapshot_record_page(page_id, guard.as_ref()));
        stage_record_delta_intent(
            mutation_log,
            tenant,
            kind,
            page_id,
            slot,
            DeltaOpKind::PutRecord,
            Bytes::from(bytes_local),
            false,
        );
        drop(guard);
        drop(pinned);
        return Ok((page_id, slot));
    }
}

/// PR #79 X-2 fold-in helper: snapshot a slotted record page as a
/// `StagedEmit` with `kind = BundlePageKind::Record` so the
/// post-mutation bytes ride the outer CommitBundle + land in
/// `RecordPageStore` on replay.
fn snapshot_record_page(page_id: PageId, bytes: &[u8; PAGE_SIZE]) -> StagedEmit {
    let mut copy: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    copy.copy_from_slice(bytes);
    StagedEmit {
        kind: crate::wal::bundle::BundlePageKind::Record,
        page_id,
        bytes: copy,
    }
}

/// Bundle-aware variant of `install_update` — ADR-031. Returns any
/// staged `IndexPage` snapshots produced by the primary-index path
/// (non-zero only when the update fell back to create — i.e. the
/// primary had no prior entry). In-place slot rewrites do not touch
/// the primary index, so they return an empty vector.
#[allow(clippy::too_many_arguments)]
fn install_update_deferred(
    store: &CrudStore,
    primary: &PrimaryIndex,
    txn_id: u64,
    tenant: TenantId,
    kind: RecordKind,
    id: u64,
    bytes: &[u8],
    commit_lsn: Lsn,
    sc_writes: &mut Vec<SideChannelWrite>,
    mutation_log: &mut TxnMutationLog,
    record_emits: &mut Vec<StagedEmit>,
) -> Result<Vec<StagedEmit>, CrudError> {
    let records = store
        .record_backend()
        .expect("dual-write store missing records store");
    let key = PrimaryKey::new(tenant, kind, id);
    let Some(slot) = primary.lookup(key)? else {
        let (page_id, slot_id) = install_create(
            store,
            txn_id,
            tenant,
            kind,
            bytes,
            commit_lsn,
            mutation_log,
            record_emits,
        )?;
        let (_prev, staged) = primary.upsert_deferred(
            key,
            PageSlot::new(page_id, slot_id),
            sc_writes,
            mutation_log,
        )?;
        return Ok(staged);
    };
    let mut bytes_local = bytes.to_vec();
    fixup_created_lsn(&mut bytes_local, kind, commit_lsn);
    if mutation_log.delta_mode {
        stage_record_delta_intent(
            mutation_log,
            tenant,
            kind,
            slot.page,
            slot.slot,
            DeltaOpKind::PutRecord,
            Bytes::from(bytes_local),
            false,
        );
        return Ok(Vec::new());
    }
    // ADR-033 Y-1 fix (2026-04-24): capture pre-W bytes of the
    // record page BEFORE the in-place slot rewrite. Pre-fix the
    // UPDATE path mutated record page bytes with `created_lsn=W`
    // stamped in the record header but never captured. On WAL
    // fsync failure the MVCC version was popped while the record
    // page retained the W-stamped ghost; a reader at a snapshot
    // advanced past W by a subsequent successful commit walked
    // primary → `(page, slot)` → read ghost bytes → passed MVCC
    // visibility (`created_lsn=W ≤ snapshot`) → returned data
    // MVCC never held. This violated ADR-023. Capturing here
    // makes Z-1 (b) rollback restore the pre-W slot bytes.
    // #1457 MF5(c) — the v8/non-delta path does not mark the M3 DPT,
    // so pin BEFORE capturing the rollback pre-image and retain that pin
    // across the in-place rewrite. Otherwise the clean reclaim arm can
    // discard the sole pre-WAL RAM copy between this mutation and the
    // outer commit bundle reaching durable storage.
    let pinned = records.latch_pinned_for_tenant(tenant, slot.page)?;
    let capture_latch = records.capture_and_write_for_tenant(mutation_log, tenant, slot.page)?;
    drop(capture_latch);
    let mut guard = pinned.latch().write();
    let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
    match kind {
        RecordKind::Node => {
            let rec = decode_node_bytes(&bytes_local)?;
            page.update_node(slot.slot, &rec)?;
        }
        RecordKind::Rel => {
            let rec = decode_rel_bytes(&bytes_local)?;
            page.update_rel(slot.slot, &rec)?;
        }
    }
    // PR #79 X-2 fold-in: stage post-mutation record page bytes
    // so replay reconstructs the slot rewrite (not just the
    // pre-mutation bytes that Z-1 (b) capture keeps for
    // rollback).
    record_emits.push(snapshot_record_page(slot.page, guard.as_ref()));
    stage_record_delta_intent(
        mutation_log,
        tenant,
        kind,
        slot.page,
        slot.slot,
        DeltaOpKind::PutRecord,
        Bytes::from(bytes_local),
        false,
    );
    drop(guard);
    drop(pinned);
    // In-place slot rewrite: primary-index entry is pinned to the
    // same (page_id, slot_id), so no IndexPage mutation occurs.
    Ok(Vec::new())
}

/// Bundle-aware variant of `install_delete` — ADR-031. Returns any
/// staged `IndexPage` snapshots produced by the tombstone + primary-
/// index mark-only remove. Empty when the primary had no prior
/// entry.
#[allow(clippy::too_many_arguments)]
fn install_delete_deferred(
    store: &CrudStore,
    primary: &PrimaryIndex,
    tenant: TenantId,
    kind: RecordKind,
    id: u64,
    sc_writes: &mut Vec<SideChannelWrite>,
    mutation_log: &mut TxnMutationLog,
    record_emits: &mut Vec<StagedEmit>,
) -> Result<Vec<StagedEmit>, CrudError> {
    let records = store
        .record_backend()
        .expect("dual-write store missing records store");
    let key = PrimaryKey::new(tenant, kind, id);
    let Some(slot) = primary.lookup(key)? else {
        return Ok(Vec::new());
    };
    if mutation_log.delta_mode {
        stage_record_delta_intent(
            mutation_log,
            tenant,
            kind,
            slot.page,
            slot.slot,
            DeltaOpKind::TombstoneRecord,
            Bytes::new(),
            false,
        );
        let (_prev, staged) = primary.remove_deferred(key, sc_writes, mutation_log)?;
        return Ok(staged);
    }
    // ADR-033 Y-1 fix (2026-04-24): capture pre-W bytes of the
    // record page BEFORE the in-place tombstone. Symmetric to
    // the UPDATE path — see the rationale in
    // `install_update_deferred`. The tombstone overwrites the
    // slot's header with a W-stamped tombstone marker; without
    // capture, that marker survives WAL rollback and a later
    // successful commit's snapshot advance exposes it.
    // #1457 MF5(d) — symmetric with the non-delta update path above:
    // pin before capture and keep the pin live across the tombstone so
    // the DPT-clean reclaim arm cannot drop the pre-WAL page image.
    let pinned = records.latch_pinned_for_tenant(tenant, slot.page)?;
    let capture_latch = records.capture_and_write_for_tenant(mutation_log, tenant, slot.page)?;
    drop(capture_latch);
    let mut guard = pinned.latch().write();
    let mut page = SlottedPage::open(guard.as_mut().as_mut())?;
    page.tombstone(slot.slot)?;
    // PR #79 X-2 fold-in: stage post-tombstone record page bytes
    // so replay reconstructs the tombstone marker.
    record_emits.push(snapshot_record_page(slot.page, guard.as_ref()));
    stage_record_delta_intent(
        mutation_log,
        tenant,
        kind,
        slot.page,
        slot.slot,
        DeltaOpKind::TombstoneRecord,
        Bytes::new(),
        false,
    );
    drop(guard);
    drop(pinned);
    let (_prev, staged) = primary.remove_deferred(key, sc_writes, mutation_log)?;
    Ok(staged)
}

// ─────────────────────────────────────────────────────────────────────
// M2-21 — create_node
// ─────────────────────────────────────────────────────────────────────

/// Allocate a fresh node and buffer its write into the transaction.
///
/// The write is installed by MVCC at commit, at which point the
/// transaction's `commit_lsn` becomes the node's
/// [`crate::transaction::Version::created_lsn`]. This function does NOT
/// commit — the caller drives the transaction lifecycle.
///
/// **Tenancy.** `tenant` is passed explicitly and must match the
/// transaction's tenant. MVCC keys are `(TenantId, MvccKey)` pairs
/// (ADR-011), so a mismatch would silently write into an unrelated
/// tenant's id space. `tx.snapshot()` does not expose the tenant —
/// adding an accessor is outside the M2-WAL-only modification budget
/// for `transaction.rs` — so higher layers are trusted to plumb the
/// same tenant into both.
///
/// **NodeRecord.created_lsn.** Stamped `Lsn::ZERO` here because the
/// commit LSN is not yet known. MVCC reads consult
/// [`crate::transaction::Version::created_lsn`] (the canonical
/// visibility anchor); the record's own field is rewritten during
/// page materialization in M2-23. ADR-007 keeps this pair MVCC-only —
/// no temporal semantics.
///
/// Returns the allocated [`NodeId`] on success; no on-disk state
/// changes until the transaction commits.
pub fn create_node(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    tenant: TenantId,
    label: LabelId,
    props: &PropertyData,
) -> Result<NodeId, CrudError> {
    let node_id = store.alloc_node(tenant)?;
    let mut rec = NodeRecord::new(node_id, label, Lsn::ZERO);
    let blob_emits = props.apply_to_node(&mut rec, tenant, &store.blobs, tx.id())?;
    store.buffer_blob_emits(tx.id(), blob_emits);
    let record_bytes = rec.to_bytes();
    tx.write(
        node_mvcc_key(node_id),
        Bytes::copy_from_slice(&record_bytes),
    );
    // Dual-write buffering (M2-CUTOVER): drained by `commit` after
    // `tx.commit()` stamps the real `commit_lsn`.
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Create {
                tenant,
                kind: RecordKind::Node,
                id: node_id.raw(),
                bytes: record_bytes.to_vec(),
                src_label_raw: None,
            },
        );
    }
    Ok(node_id)
}

// ─────────────────────────────────────────────────────────────────────
// M2-24 — scan_out
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BlockCursor {
    block: Arc<TelBlock>,
    idx: u32,
    frozen: u32,
}

impl BlockCursor {
    #[inline]
    fn new(block: Arc<TelBlock>) -> Self {
        let frozen = block.entry_count();
        Self {
            block,
            idx: 0,
            frozen,
        }
    }
}

/// Lazy outgoing TEL walk for one `(tenant, src, type_filter)` scan.
///
/// # Performance budget (PD #5)
///
/// Open is `O(channels_of_src)` via the V11-S-02 `tel_heads_for_src`
/// index and snapshots only the current chain heads. Each yielded
/// entry is `O(1)` amortized plus the same one MVCC tombstone probe
/// the eager implementation paid (`tx.read(rel_mvcc_key(rel_id))`);
/// overflow predecessor blocks are fetched only when the cursor crosses
/// a block boundary. Memory is `O(channels_of_src)` for the captured
/// head list plus one `Arc<TelBlock>` for the active block, replacing
/// the pre-V11-S-03 `Vec<TelEntry>` sized to the full result.
///
/// Visibility is fixed by `tenant` + `snapshot` captured at
/// construction. `Transaction::read(&self)` resolves append-only MVCC
/// version chains at that fixed snapshot LSN, so probing tombstones at
/// yield time is equivalent to probing at construction time. TEL entry
/// bytes are likewise time-invariant for the frozen per-block prefix:
/// entries appended after construction carry `created_lsn > snapshot`
/// and are filtered by [`TelBlock::visible_entry_at`].
#[derive(Debug)]
pub struct ScanOutCursor {
    tenant: TenantId,
    snapshot: Lsn,
    chain_heads: std::vec::IntoIter<(LabelId, Arc<TelBlock>, PageId)>,
    cur: Option<BlockCursor>,
}

/// Lazy incoming TEL walk for one `(tenant, dst, type_filter)` scan.
///
/// This is the reverse-index counterpart to [`ScanOutCursor`]. Opening
/// snapshots only the matching reverse-chain heads; predecessor blocks are
/// fetched as the cursor crosses block boundaries and no adjacency-sized
/// `Vec` is built. The reverse-index availability check is deliberately made
/// at construction so executor cursors retain the existing fail-at-open
/// contract.
#[derive(Debug)]
pub struct ScanInCursor {
    tenant: TenantId,
    snapshot: Lsn,
    chain_heads: std::vec::IntoIter<(LabelId, Arc<TelBlock>, PageId)>,
    cur: Option<BlockCursor>,
}

impl ScanInCursor {
    /// Capture the reverse chain-head snapshot for an incoming scan.
    pub fn new(
        store: &CrudStore,
        tx: &Transaction<'_>,
        dst: NodeId,
        type_filter: Option<TypeId>,
    ) -> Result<Self, ScanInError> {
        if !store.reverse_index_enabled() {
            return Err(ScanInError::ReverseIndexDisabled);
        }
        let tenant = tx.tenant();
        let chain_heads: Vec<(LabelId, Arc<TelBlock>, PageId)> = match type_filter {
            Some(ty) => {
                let channel = LabelId::new(ty.raw());
                store
                    .reverse_tel_head(tenant, dst, channel)
                    .map(|(page, head)| vec![(channel, head, page)])
                    .unwrap_or_default()
            }
            None => store.reverse_tel_heads_for_dst(tenant, dst),
        };
        Ok(Self {
            tenant,
            snapshot: tx.snapshot(),
            chain_heads: chain_heads.into_iter(),
            cur: None,
        })
    }

    /// Advance by one MVCC-visible, non-tombstoned reverse TEL entry.
    #[inline]
    pub fn next_entry(&mut self, store: &CrudStore, tx: &Transaction<'_>) -> Option<TelEntry> {
        debug_assert_eq!(tx.tenant(), self.tenant);
        debug_assert_eq!(tx.snapshot(), self.snapshot);
        loop {
            if self.cur.is_none() {
                let (_channel, head, _head_page) = self.chain_heads.next()?;
                self.cur = Some(BlockCursor::new(head));
            }

            if let Some(cur) = self.cur.as_mut() {
                while cur.idx < cur.frozen {
                    let idx = cur.idx;
                    cur.idx += 1;
                    let Some(entry) = cur.block.visible_entry_at(idx, self.snapshot) else {
                        continue;
                    };
                    if tx.read(rel_mvcc_key(RelId::new(entry.rel_id))).is_some() {
                        return Some(entry);
                    }
                }

                self.cur = cur
                    .block
                    .prev_block_ptr()
                    .and_then(|pid| store.reverse_tel_block(self.tenant, pid))
                    .map(BlockCursor::new);
            }
        }
    }
}

impl ScanOutCursor {
    /// Capture the chain-head snapshot for an outgoing scan.
    #[must_use]
    pub fn new(
        store: &CrudStore,
        tx: &Transaction<'_>,
        src: NodeId,
        type_filter: Option<TypeId>,
    ) -> Self {
        let tenant = tx.tenant();
        let chain_heads: Vec<(LabelId, Arc<TelBlock>, PageId)> = match type_filter {
            Some(ty) => {
                let channel = LabelId::new(ty.raw());
                store
                    .tel_head(tenant, src, channel)
                    .map(|(page, head)| vec![(channel, head, page)])
                    .unwrap_or_default()
            }
            None => store.tel_heads_for_src(tenant, src),
        };
        Self {
            tenant,
            snapshot: tx.snapshot(),
            chain_heads: chain_heads.into_iter(),
            cur: None,
        }
    }

    /// Advance by one MVCC-visible, non-tombstoned TEL entry.
    ///
    /// `store` + `tx` are passed per call so owning production cursors
    /// can drive this state machine without self-referential borrows.
    #[inline]
    pub fn next_entry(&mut self, store: &CrudStore, tx: &Transaction<'_>) -> Option<TelEntry> {
        debug_assert_eq!(tx.tenant(), self.tenant);
        debug_assert_eq!(tx.snapshot(), self.snapshot);
        loop {
            if self.cur.is_none() {
                let (_channel, head, _head_page) = self.chain_heads.next()?;
                self.cur = Some(BlockCursor::new(head));
            }

            if let Some(cur) = self.cur.as_mut() {
                while cur.idx < cur.frozen {
                    let idx = cur.idx;
                    cur.idx += 1;
                    let Some(entry) = cur.block.visible_entry_at(idx, self.snapshot) else {
                        continue;
                    };
                    if tx.read(rel_mvcc_key(RelId::new(entry.rel_id))).is_some() {
                        return Some(entry);
                    }
                }

                self.cur = cur
                    .block
                    .prev_block_ptr()
                    .and_then(|pid| store.tel_block(self.tenant, pid))
                    .map(BlockCursor::new);
            }
        }
    }
}

/// Lazy out-edge cursor for `src` at `tx`'s snapshot LSN.
#[must_use]
pub fn scan_out_cursor(
    store: &CrudStore,
    tx: &Transaction<'_>,
    src: NodeId,
    type_filter: Option<TypeId>,
) -> ScanOutCursor {
    ScanOutCursor::new(store, tx, src, type_filter)
}

/// Lazy in-edge cursor for `dst` at `tx`'s snapshot LSN.
///
/// Returns [`ScanInError::ReverseIndexDisabled`] at open rather than
/// deferring the operational error until iteration.
pub fn scan_in_cursor(
    store: &CrudStore,
    tx: &Transaction<'_>,
    dst: NodeId,
    type_filter: Option<TypeId>,
) -> Result<ScanInCursor, ScanInError> {
    ScanInCursor::new(store, tx, dst, type_filter)
}

/// Out-edge scan for `src` at `tx`'s snapshot LSN.
///
/// Behavior:
///
/// - `type_filter = Some(ty)`: fetch the single chain keyed by
///   `(tx.tenant(), src, LabelId::new(ty.raw()))`. Yields nothing if
///   no such chain exists.
/// - `type_filter = None`: union across all channels under
///   `(tx.tenant(), src)`. Chains are snapshotted at iterator
///   construction (the DashMap view is captured into a `Vec` sorted
///   by `channel.raw()` ascending); channels added after construction
///   are not observed.
///
/// **Walk order.** Per chain, newest→oldest: head block entries first
/// (insertion order), then follow `TelBlock::prev_block_ptr` through
/// the overflow chain. Within a block, entries are yielded in
/// insertion order (the same order they were appended).
///
/// **Visibility.** Two filters are applied, both honoring
/// `tx.snapshot()`:
/// 1. In-block TEL-entry filter: `created_lsn ≤ tx.snapshot()
///    < expired_lsn` via `TelBlock::scan`, the same rule `read_node`
///    applies to `NodeRecord`. `expired_lsn` is reserved and always
///    `Lsn::MAX` in v1.0 alpha (#19, design-v2 §3.2 errata), so this
///    filter currently only hides future edges.
/// 2. MVCC tombstone filter: each surviving entry is probed via
///    `tx.read(rel_mvcc_key(e.rel_id))`; entries whose rel has been
///    tombstoned by a `delete_rel` visible at `tx.snapshot()` are
///    dropped. Per ADR-023 / ADR-030 MVCC is authoritative; the TEL
///    is a scan-side view that `scan_out` reconciles against the
///    kernel. Fixed in #22 (previously the TEL entry survived and
///    produced phantom edges).
///
/// **Arc discipline.** Each chain's head block is captured as
/// `Arc<TelBlock>` at construction. Overflow predecessors are fetched
/// lazily when reached; they are sealed predecessor blocks, while the
/// active head's frozen prefix is fixed by the entry-count snapshot.
/// Grown/overflow rotations that happen concurrently on another thread
/// leave this scan's frozen prefix untouched (LiveGraph Theorem 1).
///
/// **Tenancy.** `tx.tenant()` is the only tenant consulted; all
/// `CrudStore::tel_block` lookups for overflow predecessors are keyed
/// by it, so no cross-tenant entry can leak.
///
/// Returns an eager `Vec::IntoIter` fast path for the materialized
/// one-hop and two-hop surfaces. The explicit [`scan_out_cursor`]
/// sibling is the lazy surface used by streaming traversal callers.
pub fn scan_out<'a>(
    store: &'a CrudStore,
    tx: &'a Transaction<'_>,
    src: NodeId,
    type_filter: Option<TypeId>,
) -> impl Iterator<Item = TelEntry> + 'a {
    let mut cursor = ScanOutCursor::new(store, tx, src, type_filter);
    let mut out = Vec::new();
    while let Some(entry) = cursor.next_entry(store, tx) {
        out.push(entry);
    }
    out.into_iter()
}

// ─────────────────────────────────────────────────────────────────────
// W26-β-2 / ADR-131 — scan_in (reverse adjacency walk)
// ─────────────────────────────────────────────────────────────────────

/// Errors returned from [`scan_in`].
///
/// `#[non_exhaustive]` under the code-quality policy convention; the W26-β-2 surface
/// only ships [`Self::ReverseIndexDisabled`] at v1.1. Forward-pinned
/// variants land as additive evolution (e.g., a v1.2 persisted-index
/// version-mismatch error per ADR-131 §"Forward-deferred to v1.2").
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScanInError {
    /// The reverse-adjacency index is disabled for this store (the
    /// AC-4 fault-injection harness flips this; production v1.1
    /// keeps it enabled). Callers MUST surface a structured error
    /// (not silent-empty results) per
    /// `feedback_load_bearing_pr_requires_fault_injection_tests.md`.
    #[error(
        "reverse adjacency index disabled or not yet built; \
         RightToLeft/Undirected expand cannot be served (ADR-131 §D-5)"
    )]
    ReverseIndexDisabled,
}

/// In-edge scan for `dst` at `tx`'s snapshot LSN.
///
/// W26-β-2 / ADR-131 — symmetric counterpart to [`scan_out`]: walks
/// the REVERSE TEL chain at `(tx.tenant(), dst, channel)` and yields
/// the entries that are visible at `tx.snapshot()`.
///
/// **Entry shape.** Each yielded [`TelEntry`] has its `dst_id` field
/// semantically holding the ORIGINAL SRC of the edge — i.e., the
/// neighbor of `dst` on the OTHER end of the forward edge. Consumers
/// constructing a `BoundEdge` for `Direction::RightToLeft` use this
/// `dst_id` as the rel's `src` and the `dst` argument (this `dst`)
/// as the rel's `dst`.
///
/// **Reverse-index-disabled posture.** Returns
/// `Err(ScanInError::ReverseIndexDisabled)` when
/// [`CrudStore::reverse_index_enabled`] is `false` — used by AC-4
/// fault-injection tests. Production deployments at v1.1 leave the
/// flag `true` so this surface always returns `Ok`.
///
/// **Visibility + tenancy + Arc discipline.** Identical to
/// [`scan_out`] — MVCC tombstone filter via `tx.read(rel_mvcc_key)`,
/// LiveGraph Theorem 1 frozen-prefix walk, `tx.tenant()` is the
/// only tenant consulted.
///
/// Returns a `Result<Vec<TelEntry>, ScanInError>` rather than an
/// iterator because the `Err` short-circuit needs to surface before
/// the lazy `into_iter()` resolves — callers pattern-matching on
/// "disabled vs empty vs filled" need the eager surface.
pub fn scan_in(
    store: &CrudStore,
    tx: &Transaction<'_>,
    dst: NodeId,
    type_filter: Option<TypeId>,
) -> Result<Vec<TelEntry>, ScanInError> {
    let mut entries: Vec<TelEntry> = Vec::new();
    let mut cursor = ScanInCursor::new(store, tx, dst, type_filter)?;
    while let Some(entry) = cursor.next_entry(store, tx) {
        entries.push(entry);
    }
    Ok(entries)
}

// ─────────────────────────────────────────────────────────────────────
// M2-23 — read_node via MVCC lookup
// ─────────────────────────────────────────────────────────────────────

/// Read a node at `tx`'s snapshot LSN.
///
/// Per ADR-018 §Decision, node records live only in the MVCC chain
/// for the v1.0 alpha; there is no page to pin and no slot directory
/// to consult. This is a single `tx.read(node_mvcc_key(id))`, decoded
/// via `NodeRecord::from_bytes`.
///
/// Returns:
///
/// - `Ok(Some(rec))` if a version of `id` is visible at `tx.snapshot()`.
/// - `Ok(None)` if `id` does not exist at that snapshot, or has been
///   tombstoned by a committed delete ≤ `tx.snapshot()`.
/// - `Err(CrudError::Mvcc(_))` only if the on-chain bytes fail to
///   decode (corrupted record — should not happen for records written
///   through `create_node`; this variant guards the decode boundary).
///
/// Tenancy is carried by `tx` (ADR-011): the `(tenant, key)` key used
/// by the MVCC map includes `tx.tenant_id()` implicitly through
/// `tx.read`, so no explicit `tenant` parameter is needed here.
pub fn read_node(tx: &Transaction<'_>, id: NodeId) -> Result<Option<NodeRecord>, CrudError> {
    let Some(bytes) = tx.read(node_mvcc_key(id)) else {
        return Ok(None);
    };
    let arr: &[u8; NodeRecord::SIZE] = bytes.as_ref().try_into().map_err(|_| {
        CrudError::Mvcc(ArcGraphError::InvalidRecordLength {
            got: bytes.len(),
            expected: NodeRecord::SIZE,
        })
    })?;
    let rec = NodeRecord::from_bytes(arr).map_err(CrudError::Mvcc)?;
    Ok(Some(rec))
}

/// Dual-write read path (M2-CUTOVER): primary-index fast path with MVCC
/// fallback per ADR-023.
///
/// Sequence:
/// 1. If `store` carries a primary index, look up
///    `(tx.tenant(), Node, id)`. On hit, pin the record page, defensively
///    check its tenant tag against `tx.tenant()`, and read the slot.
/// 2. **Snapshot-isolation gate.** The slot storage is non-MVCC —
///    `install_update` rewrites the bytes in place, so an
///    update-after-our-snapshot overwrites the live slot with bytes
///    from a future version. We apply [`NodeRecord::is_visible_at`]
///    (the full created-LSN + deleted-flag verdict) at `tx.snapshot()`.
///    If the slot is not visible, the fast-path bytes are not the version
///    we're supposed to see, so we fall through to MVCC.
///    This is the concrete mechanism that implements ADR-023's
///    "MVCC is authoritative on disagreement" contract.
/// 3. On index miss or on a "slot too new" gate miss, fall through
///    to the MVCC path identical to [`read_node`].
pub fn read_node_with_store(
    store: &CrudStore,
    tx: &Transaction<'_>,
    id: NodeId,
) -> Result<Option<NodeRecord>, CrudError> {
    if store.addressed_authoritative {
        return read_node_with_address(store, tx, id);
    }
    if let (Some(primary), Some(records)) = (store.primary.as_ref(), store.record_backend()) {
        let key = PrimaryKey::new(tx.tenant(), RecordKind::Node, id.raw());
        if let Some(slot) = primary.lookup(key)? {
            if store.deferred_v9_applies.lock().record_slot_is_pending(
                tx.tenant(),
                slot.page,
                slot.slot,
            ) {
                return read_node(tx, id);
            }
            let latch = records.latch_for_tenant(tx.tenant(), slot.page)?;
            let g = latch.read();
            let page = SlottedPageRef::open(g.as_ref().as_ref())?;
            let hdr = page.header();
            let page_tenant = TenantId::new(hdr.tenant_id);
            if page_tenant != tx.tenant() {
                return Err(CrudError::TenantMismatch {
                    page_id: slot.page,
                    got: page_tenant,
                    expected: tx.tenant(),
                });
            }
            match page.read_node(slot.slot) {
                Ok(Some(rec)) => {
                    if rec.is_visible_at(tx.snapshot()) {
                        return Ok(Some(rec));
                    }
                    // Slot holds bytes newer than our snapshot; MVCC chain
                    // still carries the correct prior Version — fall through.
                }
                // Tombstoned slot — MVCC decides (fall through).
                Ok(None) => {}
                // Accelerator-lag class (v2 M1 RULE-MT gate finding,
                // PRE-EXISTING): after replaying CONCURRENTLY-committed
                // bundles, a record page's whole-image reconstruction can
                // lag the primary index (staged page images of one page
                // captured by racing builders can land in the bundle
                // stream in an order replay's last-LSN-wins overwrite
                // resolves to an older tail), so the index may reference
                // a slot past the reconstructed image's slot_count. The
                // MVCC chain is the AUTHORITY and holds the record
                // (verified: the skew never loses data) — treat
                // out-of-range exactly like the two sibling lag classes
                // above and fall through, instead of surfacing a hard
                // error for a record that exists. Genuine corruption
                // classes (checksum mismatch, format violations) still
                // propagate loudly below. The underlying capture-order
                // skew is tracked as a follow-up on the record-page
                // staging path (superseded structurally by M3's delta
                // WAL, which deletes whole-page staging).
                Err(crate::records::PageError::SlotOutOfRange { .. }) => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    // Slow path: MVCC version chain.
    read_node(tx, id)
}

/// M4 arithmetic-address alternate for [`read_node_with_store`].
///
/// This slice deliberately keeps both selectors: the existing function maps
/// id through the primary B-tree, while this function maps the same raw id
/// through the single [`RecordKind::address`] derivation owned by the attached
/// [`AddressedRecordStore`]. The record bytes and visibility rule are
/// unchanged. A missing page, gap, tombstone, Periodic-v9 install still in
/// flight, or physical version not visible at `tx.snapshot()` falls through
/// to the same authoritative MVCC chain as the B-tree path.
pub fn read_node_with_address(
    store: &CrudStore,
    tx: &Transaction<'_>,
    id: NodeId,
) -> Result<Option<NodeRecord>, CrudError> {
    if let Some(records) = store.addressed_records.as_ref()
        && let Some(record) = records.read_node(tx.tenant(), id)?
        && !store.addressed_record_install_is_pending(record.created_lsn)
        && record.is_visible_at(tx.snapshot())
    {
        return Ok(Some(record));
    }
    read_node(tx, id)
}

/// Dual-write counterpart of [`read_rel`]. See [`read_node_with_store`]
/// for the snapshot-isolation contract that both functions share.
pub fn read_rel_with_store(
    store: &CrudStore,
    tx: &Transaction<'_>,
    id: RelId,
) -> Result<Option<RelRecord>, CrudError> {
    if store.addressed_authoritative {
        return read_rel_with_address(store, tx, id);
    }
    if let (Some(primary), Some(records)) = (store.primary.as_ref(), store.record_backend()) {
        let key = PrimaryKey::new(tx.tenant(), RecordKind::Rel, id.raw());
        if let Some(slot) = primary.lookup(key)? {
            if store.deferred_v9_applies.lock().record_slot_is_pending(
                tx.tenant(),
                slot.page,
                slot.slot,
            ) {
                return read_rel(tx, id);
            }
            let latch = records.latch_for_tenant(tx.tenant(), slot.page)?;
            let g = latch.read();
            let page = SlottedPageRef::open(g.as_ref().as_ref())?;
            let hdr = page.header();
            let page_tenant = TenantId::new(hdr.tenant_id);
            if page_tenant != tx.tenant() {
                return Err(CrudError::TenantMismatch {
                    page_id: slot.page,
                    got: page_tenant,
                    expected: tx.tenant(),
                });
            }
            match page.read_rel(slot.slot) {
                Ok(Some(rec)) => {
                    if rec.is_visible_at(tx.snapshot()) {
                        return Ok(Some(rec));
                    }
                }
                Ok(None) => {}
                // Accelerator-lag fall-through — the rel-side mirror of
                // the `read_node_with_store` SlotOutOfRange class (see
                // that arm's comment for the replay capture-order skew
                // analysis; MVCC below is the authority).
                Err(crate::records::PageError::SlotOutOfRange { .. }) => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    read_rel(tx, id)
}

/// Relationship counterpart of [`read_node_with_address`].
///
/// The arithmetic path uses the same [`RelRecord::is_visible_at`] verdict as
/// [`read_rel_with_store`] and falls through to the unchanged MVCC chain on an
/// accelerator miss, an in-flight Periodic-v9 install, or a visibility
/// disagreement.
pub fn read_rel_with_address(
    store: &CrudStore,
    tx: &Transaction<'_>,
    id: RelId,
) -> Result<Option<RelRecord>, CrudError> {
    if let Some(records) = store.addressed_records.as_ref()
        && let Some(record) = records.read_rel(tx.tenant(), id)?
        && !store.addressed_record_install_is_pending(record.created_lsn)
        && record.is_visible_at(tx.snapshot())
    {
        return Ok(Some(record));
    }
    read_rel(tx, id)
}

// ─────────────────────────────────────────────────────────────────────
// M2-22 — create_rel + deferred TEL append
// ─────────────────────────────────────────────────────────────────────

/// Allocate a fresh relationship and buffer its write into the
/// transaction. The TEL adjacency append is **deferred** to
/// [`commit`]: we cannot stamp `TelEntry.created_lsn` at call time
/// because the commit LSN is assigned by the MVCC install.
///
/// Asymmetry vs. `create_node` (per ADR-018):
///
/// - `RelRecord` bytes go into the MVCC chain under
///   `rel_mvcc_key(rel_id)` exactly like node records.
/// - A `PendingTelAppend` is buffered on the store keyed by
///   `tx.id()`. [`commit`] drains the buffer after `tx.commit()`
///   returns the real `commit_lsn`, and appends a `TelEntry` with
///   that LSN into the `(tenant, src, channel=ty.into())` chain
///   under the chain's mutex.
///
/// **MVCC↔TEL atomicity window — issue #20, OPEN.** The TEL append is
/// NOT folded into the commit. [`commit`] publishes the rel's MVCC
/// visibility in Phase 3 (`visible.store(commit_lsn)` at
/// `transaction.rs::commit_with_bundle_writes`) and only THEN drains the
/// staged `PendingTelAppend`s (`for p in pending_tel` → `Self::tel_append`),
/// AFTER `commit_with_bundle_and_rollback` has returned. Because a reader
/// sources its snapshot from `visible` (`begin_inner`), a concurrent txn
/// that begins in that gap reads its snapshot at `commit_lsn`, sees this
/// `RelRecord` via MVCC ([`read_rel`]), yet [`scan_out`] misses the
/// adjacency (the TEL entry is unappended) — a snapshot-isolation
/// violation across the MVCC+TEL composite store.
///
/// The earlier note here ("acceptable … the txn is not the publisher of
/// its own visibility; M2-WAL closes it") was stale: the three-phase
/// commit (ADR-031/032/033) made the committing thread the publisher
/// (Phase 3), inverting the ADR-018 §Decision order (TEL-append THEN
/// MVCC write under the commit gate); and M2-WAL did NOT close it (TEL
/// does not ride the `CommitBundle` — the crash-recovery analogue is
/// rebuilt by #780's `reinstate_rel_adjacency`, not by the bundle).
/// Closing the live window is a commit-path reorder (drain TEL before
/// Phase-3 `visible.store`) and is ADR-gated; see the
/// `mvcc_tel_window_20_*` regression tests.
///
/// Returns the allocated [`RelId`]. Does NOT commit.
#[allow(clippy::too_many_arguments)]
pub fn create_rel(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    tenant: TenantId,
    src: NodeId,
    dst: NodeId,
    ty: TypeId,
    props: &PropertyData,
) -> Result<RelId, CrudError> {
    let rel_id = store.alloc_rel(tenant)?;
    let mut rec = RelRecord::new(rel_id, ty, src, dst, Lsn::ZERO);
    let blob_emits = props.apply_to_rel(&mut rec, tenant, &store.blobs, tx.id())?;
    store.buffer_blob_emits(tx.id(), blob_emits);
    let record_bytes = rec.to_bytes();
    tx.write(rel_mvcc_key(rel_id), Bytes::copy_from_slice(&record_bytes));
    let src_label_raw = read_node(tx, src)?.map(|node| node.label_id);
    // Key TEL chains by (tenant, src, channel) where channel =
    // LabelId(ty.raw()). TypeId and LabelId are disjoint u32
    // namespaces in arcgraph-core but TelBlock's header stores a
    // `label_id: u32`, so we project TypeId into LabelId for the
    // chain key and block header without loss.
    let channel = LabelId::new(ty.raw());
    store.buffer_pending(
        tx.id(),
        PendingTelAppend {
            tenant,
            src,
            dst,
            rel: rel_id,
            channel,
        },
    );
    // Dual-write buffering (M2-CUTOVER): drained by `commit`.
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Create {
                tenant,
                kind: RecordKind::Rel,
                id: rel_id.raw(),
                bytes: record_bytes.to_vec(),
                src_label_raw,
            },
        );
    }
    Ok(rel_id)
}

/// Commit a transaction and drain its deferred TEL appends.
///
/// Sequencing:
///
/// 1. Capture `tx.id()` (needed to drain the pending buffer after
///    `tx.commit()` consumes `tx`).
/// 2. `tx.commit()` runs OCC validation + MVCC install, returning
///    the transaction's `commit_lsn`. On conflict the pending
///    buffer is discarded and the error is propagated.
/// 3. For each buffered `PendingTelAppend`, call
///    `CrudStore::tel_append` with the real commit LSN. The
///    TEL append is single-writer-per-chain (chain mutex) and
///    idempotent under retry by construction (a fresh append
///    would allocate a new slot — by M2-WAL, the WAL record
///    carries slot coords, so replay is exact).
///
/// Aborts the transaction implicitly on MVCC failure (commit
/// consumes `tx`). Call [`CrudStore::discard_pending`] before
/// calling `tx.abort()` directly if you bypass this wrapper.
pub fn commit(mut tx: Transaction<'_>, store: &CrudStore) -> core::result::Result<Lsn, CrudError> {
    let txn_id = tx.id();
    // M3.a Slice G.5: capture the txn's tenant BEFORE
    // `commit_with_bundle_and_rollback` consumes `tx`. The rollback
    // closure needs the tenant to dispatch
    // `VectorPageStoreHandle::restore_page_bytes` calls (the
    // mutation-log entry shape is `(kind, page_id, pre_w_bytes)` —
    // tenant is NOT carried per-entry because at v1.0 a transaction
    // is single-tenant per ADR-011 + Transaction::tenant_id; every
    // `(Vector, page_id)` entry in this log belongs to `txn_tenant`).
    let txn_tenant = tx.tenant();
    let writes_delta = crate::wal::is_delta_bundle_format(tx.wal_format_version());
    // RC-1 (#1366): capture the MVCC GC anchor while `tx` is still live
    // (it is consumed by `commit_with_bundle_and_rollback` below). This
    // txn is in the active set, so the value is a conservative floor:
    // a previously-enqueued secondary removal stamped `L` is safe to
    // apply only once `oldest_active_snapshot() > L`, and this txn's own
    // snapshot keeps the floor from advancing past any removal a reader
    // (including this writer's own read set) could still observe. This
    // commit's own removals (stamped with the fresh `commit_lsn`, the
    // newest LSN) are therefore never applied in this same call — they
    // correctly defer to a later commit whose horizon has cleared them.
    let snapshot_horizon = tx.oldest_active_snapshot();

    // ADR-031: fold every WAL emission for this commit into a single
    // `CommitBundle` fire. The bundle builder runs AFTER MVCC Phase 1
    // (commit_gate-held silent install) and BEFORE Phase 2 (single
    // `wal.append(CommitBundle)`), gets the allocated `commit_lsn`,
    // performs slotted-page installs + primary/secondary index
    // upserts via the `*_deferred` siblings, and returns the staged
    // `IndexPage` byte snapshots to ride the same fsync. Target:
    // `records/commit == 1` on the E2 workload (pre-fix 2.02).
    //
    // Pre-decoding note: `take_installs` moves the pending buffer
    // OUT before the builder runs so we avoid taking a second
    // borrow on `store` inside the closure. TEL appends still drain
    // AFTER the commit returns — they are not part of the bundle
    // (issue #20 MVCC↔TEL atomicity remains open).
    let installs = store.take_installs(txn_id);
    let pending_tel = store.take_pending(txn_id);
    // N-2 (issue #81): drain blob chain-page emits BEFORE the
    // bundle builder closure so they participate in the same v3
    // `CommitBundle` fsync as primary / record / secondary staged
    // pages. An empty pending_blob_emits entry is the normal-case
    // (all-inline) shape.
    let blob_emits = store.take_blob_emits(txn_id);
    // v2 M1 (ADR-230 / design §M1.3): capture the FINAL image of every
    // slotted prop page this txn packed bags into — ONE snapshot per
    // TOUCHED PAGE per bundle (not per bag), which is the ~14× batch
    // WAL amortization. Captured here (pre-builder) like `blob_emits`;
    // the tenant pool check-in happens only on commit SUCCESS
    // (`publish_txn_slotted` below); rollback restores the pre-txn
    // resident/pool state (`rollback_txn_slotted`).
    let slotted_emits: Vec<StagedEmit> = store
        .blobs
        .snapshot_txn_slotted_pages(txn_id)
        .into_iter()
        .map(|(page_id, bytes)| StagedEmit {
            kind: crate::wal::bundle::BundlePageKind::PropSlotted,
            page_id,
            bytes,
        })
        .collect();
    let mut physical_prop_pages = Vec::new();
    if crate::wal::is_delta_bundle_format(tx.wal_format_version()) {
        let prop_intents = store.blobs.snapshot_txn_slotted_delta_intents(txn_id)?;
        physical_prop_pages.extend(
            prop_intents
                .iter()
                .filter(|intent| {
                    intent.store_id == STORE_PROPS && intent.kind == DeltaOpKind::PageAlloc
                })
                .map(|intent| intent.page_no),
        );
        tx.mutation_log_mut().delta_intents.extend(prop_intents);
    }
    // M3.a Slice G.4 (commit-bundle vector page staging): drain
    // vector arena page emits BEFORE the bundle builder closure
    // so they participate in the same v5 `CommitBundle` fsync as
    // primary / record / secondary / blob staged pages. Mirrors
    // the `take_blob_emits` pattern. An empty
    // `pending_vector_emits` entry is the normal-case shape for
    // pre-G.5/G.7 deployments where no vector writers are wired
    // yet. Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
    let vector_emits = store.take_vector_emits(txn_id);
    // Snapshot the pending-slot view once and thread it through the commit
    // builder. Re-locking `deferred_v9_applies` from the pre-image helper can
    // self-deadlock when a drain already owns the non-reentrant mutex.
    let needs_node_prior = store.secondary.is_some()
        && installs.iter().any(|inst| {
            matches!(
                inst,
                PendingInstall::Update {
                    kind: RecordKind::Node,
                    ..
                } | PendingInstall::Delete {
                    kind: RecordKind::Node,
                    ..
                }
            )
        });
    let pending_record_slots: std::collections::HashSet<_> = if needs_node_prior {
        store
            .deferred_v9_applies
            .lock()
            .pending_record_slots
            .keys()
            .copied()
            .collect()
    } else {
        std::collections::HashSet::new()
    };
    // `Transaction::read` would return this transaction's buffered NEW value.
    // Capture the committed snapshot version instead so a pending physical
    // slot gets the exact prior image used for the secondary-index diff.
    let prior_mvcc_nodes: std::collections::HashMap<u64, NodeRecord> =
        if pending_record_slots.is_empty() {
            std::collections::HashMap::new()
        } else {
            installs
                .iter()
                .filter_map(|inst| match inst {
                    PendingInstall::Update {
                        kind: RecordKind::Node,
                        id,
                        ..
                    }
                    | PendingInstall::Delete {
                        kind: RecordKind::Node,
                        id,
                        ..
                    } => tx
                        .read_snapshot(node_mvcc_key(NodeId::new(*id)))
                        .and_then(|bytes| decode_node_bytes(bytes.as_ref()).ok())
                        .map(|record| (*id, record)),
                    _ => None,
                })
                .collect()
        };
    // #352 Part 2 (ADR-199): drain this txn's idempotency bindings and
    // stage them on the transaction so they ride this commit's v6
    // CommitBundle (`idempotency_bindings` section) atomically with the
    // node/rel writes that allocated the internal ids. Unlike vector
    // emits (folded INSIDE the builder closure because each entry needs
    // the commit_lsn stamp), bindings carry no commit_lsn, so they are
    // staged on `tx` directly BEFORE the commit call. On a commit-time
    // failure the bundle is never written, so the bindings fall away
    // with the aborted transaction (no in-memory publish either — mcp
    // installs to the IdempotencyStore only on commit success).
    let idempotency_bindings = store.take_idempotency_bindings(txn_id);
    let idempotency_releases_to_publish: Vec<_> = idempotency_bindings
        .iter()
        .filter(|pending| {
            matches!(
                pending.entry.op,
                crate::wal::bundle::IdempotencyBindingOp::Release
            )
        })
        .map(|pending| pending.entry.clone())
        .collect();
    // INV-S3.11: a v10 generation has no logical idempotency tail. Translate
    // the drained owner operations into authoritative direct rows in the SAME
    // transaction. The owner planning guard stays live through Phase 3, so a
    // concurrent first touch of the same logical extent cannot allocate a
    // conflicting physical extent. v9 retains the mature logical section.
    let physical_owner_registry = if tx.wal_format_version() == crate::wal::BUNDLE_FORMAT_V10 {
        Some(store.owner_rows.read().clone().ok_or_else(|| {
            CrudError::Mvcc(ArcGraphError::TransactionAborted {
                reason: "v10 idempotency mutation has no M4 owner registry".to_owned(),
            })
        })?)
    } else {
        None
    };
    let mut physical_owner_rows = Vec::new();
    if let Some(owner) = physical_owner_registry.as_ref() {
        for pending in &idempotency_bindings {
            let entry = &pending.entry;
            let class = match entry.kind {
                0 => crate::owner_row::OwnerRowClass::NodeBinding,
                1 => crate::owner_row::OwnerRowClass::RelBinding,
                other => {
                    return Err(CrudError::Mvcc(ArcGraphError::TransactionAborted {
                        reason: format!("unsupported v10 idempotency kind {other}"),
                    }));
                }
            };
            let row = if matches!(entry.op, crate::wal::bundle::IdempotencyBindingOp::Install) {
                let logical = crate::owner_row::BindingOwnerValue {
                    kind: entry.kind,
                    external_id: entry.external_id.clone(),
                    payload_hash: pending.payload_hash,
                    active: true,
                }
                .encode()?;
                owner.prepare_indexed_logical_row(
                    entry.tenant,
                    class,
                    entry.internal_id,
                    crate::owner_index::str_hash_56(&entry.external_id),
                    &logical,
                )?
            } else {
                owner.prepare_retired_binding_row(entry.tenant, class, entry.internal_id)?
            };
            physical_owner_rows.push(row);
        }
    }
    let physical_record_rows = if physical_owner_registry.is_some()
        && store.primary.is_none()
        && store.addressed_authoritative
    {
        installs
            .iter()
            .map(|install| match install {
                PendingInstall::Create {
                    kind, id, bytes, ..
                }
                | PendingInstall::Update {
                    kind, id, bytes, ..
                } => (*kind, *id, Some(Bytes::copy_from_slice(bytes))),
                PendingInstall::Delete { kind, id, .. } => (*kind, *id, None),
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let _physical_owner_plan = physical_owner_registry
        .as_ref()
        .filter(|_| {
            !physical_prop_pages.is_empty()
                || !physical_owner_rows.is_empty()
                || !physical_record_rows.is_empty()
        })
        .map(|owner| owner.planning_guard());
    if let Some(owner) = physical_owner_registry.as_ref() {
        let owner_tenant = tx.tenant();
        if !physical_prop_pages.is_empty() {
            // Property PageAlloc intents were captured before the shared M4
            // planner.  Put their missing ExtentAlloc intents first so both
            // live apply and sequential WAL replay resolve every page through
            // a durable directory mapping.
            let extent_intents =
                owner.plan_page_extents_locked(owner_tenant, STORE_PROPS, &physical_prop_pages)?;
            tx.mutation_log_mut()
                .delta_intents
                .splice(0..0, extent_intents);
        }
        if !physical_record_rows.is_empty() {
            let record_intents =
                owner.plan_direct_records_locked(owner_tenant, &physical_record_rows)?;
            tx.mutation_log_mut().delta_intents.extend(record_intents);
        }
        if !physical_owner_rows.is_empty() {
            let owner_intents = owner.plan_rows_locked(owner_tenant, &physical_owner_rows)?;
            tx.mutation_log_mut().delta_intents.extend(owner_intents);
        }
    }
    if physical_owner_registry.is_none() {
        tx.stage_idempotency_bindings(
            idempotency_bindings
                .into_iter()
                .map(|pending| pending.entry)
                .collect(),
        );
    }

    // #1221 (ADR-218): drain this txn's staged ACL grant/revoke ops and
    // stage them on the transaction so they ride this commit's v8
    // CommitBundle (`acl_grants` section) atomically with the commit.
    // Same discipline as `idempotency_bindings` — the entries carry no
    // commit_lsn, so they are staged on `tx` directly BEFORE the commit
    // call; on a commit-time failure the bundle is never written, so the
    // ops fall away with the aborted transaction. Order is PRESERVED
    // through `take_acl_grants` (append order) → `stage_acl_grants`
    // (append-extend) → the v8 encoder (no re-sort), so replay is
    // last-writer-wins per doc (ADR-218 invariant).
    let acl_grants = store.take_acl_grants(txn_id);
    tx.stage_acl_grants(acl_grants);

    // RC-1 (#1366): old-value secondary removals collected during the
    // bundle-builder drain below. Declared OUTSIDE the closure so it
    // survives the commit and can be enqueued (stamped with the real
    // `commit_lsn`) onto `store.deferred_removals` ONLY on commit
    // success. On a commit failure the batch is dropped — the aborted
    // txn never superseded anything, so there is nothing to defer.
    let mut deferred_removals_batch: Vec<(LabelId, StringId, SecondaryIndexValue, NodeId)> =
        Vec::new();
    // #1464: successful update-side secondary inserts. Each insert is
    // registered as in-flight under `deferred_removals`' maintenance gate
    // inside the builder; commit success replaces that marker with the final
    // `commit_lsn`, while rollback discards it.
    let mut secondary_reassertions_batch: Vec<SecondaryEntry> = Vec::new();

    let commit_result = tx.commit_with_bundle_apply_and_rollback(
        |commit_lsn, sc_writes, allocator_advances, vector_pages, mutation_log| {
            // Fold drained vector emits into the bundle's vector_pages
            // section. The closure receives a pre-existing `&mut Vec`
            // that may carry pre-registered entries (none today, hook
            // for v1.1 additional staging surfaces); we extend it.
            // Stamp `commit_lsn` on every entry — at v1.0 commit_lsn ==
            // bundle.commit_lsn, but storing per-entry leaves room for
            // v1.1 batched bundles where multiple commit_lsns share a
            // frame.
            for emit in vector_emits {
                vector_pages.push(VectorPageEntry {
                    tenant: emit.tenant,
                    partition: emit.partition,
                    index_id: emit.index_id,
                    page_id: emit.page_id,
                    commit_lsn,
                    bytes: emit.bytes,
                });
            }
            let mut staged: Vec<StagedEmit> = Vec::new();
            // Fold blob chain pages first so they land ahead of
            // record/primary entries in the bundle's staged_pages
            // list. Ordering is decoupled from correctness (replay
            // dispatches by kind) but makes the bundle's layout
            // predictable for test assertions and operator log
            // inspection.
            staged.extend(blob_emits);
            // v2 M1: shared slotted prop pages ride the same bundle —
            // one image per touched page (see the capture above).
            staged.extend(slotted_emits);
            let Some(primary) = store.primary.as_ref() else {
                // No primary-index configured: installs were buffered
                // but there is nothing to drain. Still snapshot
                // allocator state — the no-primary path can still have
                // run `alloc_node` etc. via the same store.
                allocator_advances.extend(store.snapshot_allocator_advances());
                if let Some(alloc) = store.allocator.as_ref() {
                    allocator_advances.extend(alloc.snapshot_advances());
                }
                return Ok(staged);
            };
            for inst in &installs {
                let (tenant, kind, id) = (inst.tenant(), inst.kind(), inst.id());

                // Capture the pre-image BEFORE install_update_deferred /
                // install_delete_deferred rewrites / tombstones the
                // slot — only needed for nodes when a secondary index is
                // configured. Tracks the ADR-030 pre-fix behavior
                // faithfully.
                let prior_node = if matches!(kind, RecordKind::Node)
                    && store.secondary.is_some()
                    && matches!(
                        inst,
                        PendingInstall::Update { .. } | PendingInstall::Delete { .. }
                    ) {
                    read_prior_node_from_store(
                        store,
                        tenant,
                        NodeId::new(id),
                        &pending_record_slots,
                        prior_mvcc_nodes.get(&id),
                    )
                } else {
                    None
                };

                let primary_emits = match inst {
                    PendingInstall::Create { bytes, .. } => {
                        match install_create(
                            store,
                            txn_id,
                            tenant,
                            kind,
                            bytes,
                            commit_lsn,
                            mutation_log,
                            &mut staged,
                        ) {
                            Ok((pid, sid)) => {
                                match primary.upsert_deferred(
                                    PrimaryKey::new(tenant, kind, id),
                                    PageSlot::new(pid, sid),
                                    sc_writes,
                                    mutation_log,
                                ) {
                                    Ok((_prev, emits)) => Ok(emits),
                                    Err(e) => Err(CrudError::from(e)),
                                }
                            }
                            Err(e) => Err(e),
                        }
                    }
                    PendingInstall::Update { bytes, .. } => install_update_deferred(
                        store,
                        primary,
                        txn_id,
                        tenant,
                        kind,
                        id,
                        bytes,
                        commit_lsn,
                        sc_writes,
                        mutation_log,
                        &mut staged,
                    ),
                    PendingInstall::Delete { .. } => install_delete_deferred(
                        store,
                        primary,
                        tenant,
                        kind,
                        id,
                        sc_writes,
                        mutation_log,
                        &mut staged,
                    ),
                };
                match primary_emits {
                    Ok(emits) => staged.extend(emits),
                    Err(e) => {
                        // Per ADR-023: index install failure degrades
                        // lookup performance but not correctness. Log
                        // and move to the next install. We intentionally
                        // do NOT fail the bundle — the MVCC side already
                        // committed in Phase 1, and the bundle will
                        // still fsync even with fewer staged emits.
                        tracing::warn!(
                            "dual-write {} for {:?} id {} failed: {}",
                            match inst {
                                PendingInstall::Create { .. } => "create",
                                PendingInstall::Update { .. } => "update",
                                PendingInstall::Delete { .. } => "delete",
                            },
                            kind,
                            id,
                            e
                        );
                        continue;
                    }
                }

                // Secondary wiring (M2-34). Rels are node-label scoped
                // in the secondary key, so they're skipped.
                //
                // RC-2 (#1366) write-follows-declare: apply synchronous
                // maintenance whenever the index is maintenance-active,
                // which is TRUE for BOTH `Building` and `Online`. Gating on
                // `Online` only is the false-negative RC-2 closes — a node
                // written while `Building` would be absent once the index
                // goes `Online`. `maintenance_active()` is the single gate;
                // the drop/absent case (Phase 1) returns false here.
                if matches!(kind, RecordKind::Node)
                    && let Some(secondary) = store.secondary.as_ref()
                    && secondary.maintenance_active()
                {
                    let sec_emits = match inst {
                        PendingInstall::Create { bytes, .. } => {
                            decode_node_bytes(bytes).ok().map(|rec| {
                                publish_node_properties_insert_deferred(
                                    secondary,
                                    tenant,
                                    &rec,
                                    NodeId::new(id),
                                    mutation_log,
                                )
                            })
                        }
                        PendingInstall::Update { bytes, .. } => {
                            let new_rec_opt = decode_node_bytes(bytes).ok();
                            match (prior_node.as_ref(), new_rec_opt) {
                                (Some(prior), Some(new_rec)) => {
                                    // RC-1: NEW values insert now; OLD values
                                    // are DEFERRED past the snapshot horizon.
                                    // Hold the generation-state mutex through
                                    // insert + in-flight registration so a
                                    // ready old removal cannot interleave.
                                    let mut removal_state = store.deferred_removals.lock();
                                    let reassertion_start = secondary_reassertions_batch.len();
                                    let emits = diff_and_publish_node_properties_deferred(
                                        secondary,
                                        tenant,
                                        prior,
                                        &new_rec,
                                        NodeId::new(id),
                                        mutation_log,
                                        &mut deferred_removals_batch,
                                        &mut secondary_reassertions_batch,
                                    );
                                    removal_state.register_inflight(
                                        txn_id,
                                        &secondary_reassertions_batch[reassertion_start..],
                                    );
                                    Some(emits)
                                }
                                (None, Some(new_rec)) => {
                                    // No prior state (update-as-create):
                                    // publish every property as a fresh
                                    // insert.
                                    Some(publish_node_properties_insert_deferred(
                                        secondary,
                                        tenant,
                                        &new_rec,
                                        NodeId::new(id),
                                        mutation_log,
                                    ))
                                }
                                _ => None,
                            }
                        }
                        PendingInstall::Delete { .. } => {
                            // RC-1: DELETE removes every indexed value — all
                            // DEFERRED past the snapshot horizon so a
                            // pre-delete snapshot reader still finds the node
                            // (it fails the MVCC visibility recheck on the
                            // tombstone, but never a false negative from a
                            // missing entry).
                            if let Some(prior) = prior_node.as_ref() {
                                enqueue_node_properties_remove(
                                    prior,
                                    NodeId::new(id),
                                    &mut deferred_removals_batch,
                                );
                            }
                            None
                        }
                    };
                    if let Some(emits) = sec_emits {
                        staged.extend(emits);
                    }
                }
            }
            // Issue #129 P0 fix: snapshot allocator high-water for
            // every (tenant, kind) at builder-end so the v4 bundle
            // durifies advances atomically with this commit. After all
            // PageAllocator.alloc / CrudStore.alloc_node / alloc_rel
            // calls inside the builder have completed (record-store
            // PageId allocations included), the snapshot reflects the
            // post-builder state. Replay seeds counters via
            // `seed_from_advance` in commit_lsn order; over-counting
            // (other tenants' tenants whose allocator state hasn't
            // moved this commit) is harmless under monotonic-max.
            // ADR-034 D-1 restored.
            allocator_advances.extend(store.snapshot_allocator_advances());
            if let Some(alloc) = store.allocator.as_ref() {
                allocator_advances.extend(alloc.snapshot_advances());
            }
            Ok(staged)
        },
        |deltas, commit_lsn| store.apply_or_defer_v9_deltas(txn_id, deltas, commit_lsn),
        // ADR-033 Z-1 (b) rollback closure. Runs under commit_gate
        // AFTER MVCC version unwind and BEFORE install_order
        // advances, on WAL fsync failure or builder error. The
        // ordering (root_changes → new_pages → page_mutations →
        // blob_heads) is specified in ADR-033 §5/§6: root_changes
        // first so an in-flight reader that captures new_root from
        // root_cache finds it still mapped in page_store when the
        // subsequent new_pages removal runs.
        //
        // Scope: primary-index B-tree pages, record-store pages
        // that the builder explicitly captured via
        // install_fresh_for_txn / capture_and_write, blob chain
        // heads, AND (Z-1 F-1, #1366) secondary-index pages — the
        // deferred F-1 gap is now closed: the secondary write path
        // captures pre-W bytes + records fresh pages into the log, and
        // the Secondary arms below dispatch through the published
        // `SecondaryIndexHandle` rollback methods (PD#7; the concrete
        // `SecondaryPageStore` lives in `arcgraph-index`).
        |log| {
            store.release_record_reservations(txn_id);
            // Step 1: restore root_cache atomics BEFORE removing new
            // pages, so readers that captured new_root_id from
            // root_cache.load find the page still mapped. Drains ALL
            // root_changes regardless of which index configured — a
            // Secondary root_change must not be left in the log.
            for (handle, old_root_id) in log.root_changes.drain(..) {
                match handle {
                    IndexHandle::PRIMARY => {
                        if let Some(primary) = store.primary.as_ref() {
                            primary.restore_root_cache(old_root_id);
                        }
                    }
                    IndexHandle::SECONDARY => {
                        // Z-1 F-1: restore the secondary index's cached
                        // root + clear its pending grow-root stash so
                        // the aborted new root is never persisted.
                        if let Some(secondary) = store.secondary.as_ref() {
                            secondary.rollback_restore_root(old_root_id);
                        }
                    }
                    _ => {
                        tracing::warn!(
                            "ADR-033 rollback: unhandled root_change handle {:?} (old_root={:?})",
                            handle,
                            old_root_id,
                        );
                    }
                }
            }
            // Step 2: remove newly-installed pages from each store.
            let drained_new_pages: Vec<_> = log.new_pages.drain(..).collect();
            for (kind, page_id) in drained_new_pages {
                match kind {
                    PageStoreKind::Primary => {
                        if let Some(primary) = store.primary.as_ref() {
                            let _ = primary.page_store().remove_page(page_id);
                        }
                    }
                    PageStoreKind::Record => {
                        if let Some(records) = store.record_backend() {
                            // ADR-033 Z-1(b): remove the page for the ABORTING TENANT,
                            // not (DEFAULT, page) — a tenant-blind `remove_page` routes
                            // to TenantId::DEFAULT and leaves the aborting tenant's page
                            // (and its MVCC ghost) live.
                            let _ = records.remove_page_for_tenant(txn_tenant, page_id);
                            store
                                .open_pages
                                .retain(|_, current_page| *current_page != page_id);
                        }
                    }
                    PageStoreKind::Secondary => {
                        // Z-1 F-1 (#1366): remove the fresh secondary
                        // page (split / overflow / grow-root) via the
                        // published rollback dispatch. Root_changes
                        // already restored the cached root above, so a
                        // reader that captured the aborted new root
                        // still finds it mapped until this removal.
                        if let Some(secondary) = store.secondary.as_ref() {
                            secondary.rollback_remove_page(page_id);
                        }
                    }
                    PageStoreKind::Vector => {
                        // M3.a Slice G.5 (ADR-033 §6). At v1.0, no
                        // production path pushes Vector entries into
                        // `log.new_pages` — vector arena allocations
                        // are serialized under the arena's per-tenant
                        // write latch and reuse pages on retry rather
                        // than tracking new-page allocations through
                        // the txn mutation log. The arm exists for
                        // structural symmetry with Primary / Record.
                        // Current production drainage is expected to
                        // be empty.
                        tracing::warn!(
                            "ADR-033 Z-1 rollback: unexpected Vector new_pages entry \
                             for page {:?}; v1.0 vector arenas do not push into \
                             log.new_pages. This is a no-op but signals an upstream \
                             capture-discipline bug.",
                            page_id,
                        );
                    }
                    PageStoreKind::Bm25 => {
                        // ADR-039 §D-6 — symbolic at v1.0. BM25
                        // rollback drains `bm25_pending` (per-tenant)
                        // rather than `new_pages` (per-page). No
                        // production path pushes Bm25 entries here at
                        // v1.0; the arm exists for exhaustive match
                        // coverage so v1.1+ segment-page restoration
                        // can populate the body without touching the
                        // dispatch shape.
                        tracing::warn!(
                            "ADR-039 Z-1 rollback: unexpected Bm25 new_pages entry \
                             for page {:?}; v1.0 BM25 rollback uses bm25_pending \
                             (per-tenant), not new_pages (per-page). This is a \
                             no-op but signals an upstream capture-discipline bug.",
                            page_id,
                        );
                    }
                }
            }
            // Step 3: restore pre-W bytes into existing pages.
            //
            // Y-2: each entry carries its `PageStoreKind` so rollback
            // dispatches EXACTLY to the store that captured it. No
            // cross-store fallthrough — the pre-Y-2 "try primary, then
            // records" fallthrough was latent-unsafe because numeric
            // PageId ranges overlap across stores (primary uses
            // SYSTEM-tenant allocator; record uses per-tenant
            // allocator; both can produce PageId(1) simultaneously).
            // A fallthrough on mis-captured kind would restore
            // primary bytes to a record page or vice versa.
            let drained_page_mutations: Vec<_> = log.page_mutations.drain(..).collect();
            for (kind, page_id, pre_bytes) in drained_page_mutations {
                match kind {
                    PageStoreKind::Primary => {
                        if let Some(primary) = store.primary.as_ref() {
                            let _ = primary
                                .page_store()
                                .restore_page_bytes(page_id, pre_bytes.as_ref());
                        }
                    }
                    PageStoreKind::Record => {
                        if let Some(records) = store.record_backend() {
                            // ADR-033 Z-1(b): restore the pre-image to the ABORTING
                            // TENANT's page, not (DEFAULT, page). A tenant-blind
                            // `restore_page_bytes` routes to TenantId::DEFAULT (see the
                            // page_store default impl), leaving a live MVCC ghost at
                            // (txn_tenant, page) on a non-DEFAULT abort. Mirror the
                            // Vector arm below, already tenant-qualified.
                            let _ = records.restore_page_bytes_for_tenant(
                                txn_tenant,
                                page_id,
                                pre_bytes.as_ref(),
                            );
                        }
                    }
                    PageStoreKind::Secondary => {
                        // Z-1 F-1 (#1366): restore the secondary page's
                        // pre-W bytes via the published rollback
                        // dispatch (PD#7 — the concrete
                        // `SecondaryPageStore` lives in `arcgraph-index`;
                        // storage never reaches across the boundary).
                        if let Some(secondary) = store.secondary.as_ref()
                            && let Err(e) =
                                secondary.rollback_restore_page(page_id, pre_bytes.as_ref())
                        {
                            tracing::warn!(
                                "ADR-033 Z-1 rollback: secondary restore_page_bytes failed \
                                 for page {:?}: {}",
                                page_id,
                                e,
                            );
                        }
                    }
                    PageStoreKind::Vector => {
                        // M3.a Slice G.5 (ADR-033 Z-1 (b) +
                        // ADR-035 §7.5). Restore the pre-W bytes
                        // captured by
                        // [`CrudStore::capture_and_stage_vector_page`]
                        // into the tenant's vector arena via the
                        // wired `VectorPageStoreHandle`. When no
                        // handle is wired, the arm warns-and-skips
                        // — the same posture the WAL replay
                        // executor uses for un-wired Vector entries
                        // (see `wal::replay::PageStoreTarget`).
                        //
                        // `txn_tenant` is captured from the outer
                        // scope: [`Transaction::tenant`] is fixed for
                        // the lifetime of this transaction
                        // (ADR-011 — every MVCC key is qualified by
                        // its txn's tenant), so a single-tenant
                        // rollback dispatch is sound at v1.0.
                        if let Some(vector) = store.vector_store.as_ref() {
                            if let Err(e) =
                                vector.restore_page_bytes(txn_tenant, page_id, pre_bytes.as_ref())
                            {
                                tracing::warn!(
                                    "ADR-033 Z-1 rollback: vector restore_page_bytes \
                                     failed for tenant {:?} page {:?}: {}",
                                    txn_tenant,
                                    page_id,
                                    e
                                );
                            }
                        } else {
                            tracing::warn!(
                                "ADR-033 Z-1 rollback: no VectorPageStoreHandle wired \
                                 into CrudStore; skipping Vector page_mutation entry \
                                 for page {:?}. Wire one via \
                                 CrudStore::with_vector_store to enable rollback.",
                                page_id,
                            );
                        }
                    }
                    PageStoreKind::Bm25 => {
                        // ADR-039 §D-6 — symbolic at v1.0. BM25
                        // rollback drains `bm25_pending` (per-tenant);
                        // there is no per-page pre-W byte snapshot
                        // for BM25 at v1.0 because Tantivy's
                        // `IndexWriter` buffer is the rollback
                        // granularity. The arm exists for exhaustive
                        // match coverage; production drainage is
                        // expected to be empty.
                        tracing::warn!(
                            "ADR-039 Z-1 rollback: unexpected Bm25 page_mutation entry \
                             for page {:?}; v1.0 BM25 rollback uses bm25_pending \
                             (per-tenant), not page_mutations (per-page). This is a \
                             no-op but signals an upstream capture-discipline bug.",
                            page_id,
                        );
                    }
                }
            }
            // Step 4: remove uncommitted blob chains.
            let drained_blob_heads: Vec<_> = log.blob_heads.drain(..).collect();
            for (tenant, head) in drained_blob_heads {
                if let Err(error) = store.blobs.remove_uncommitted_chain(tenant, head) {
                    // Rollback must continue draining the remaining mutation
                    // log. The typed spill error is reported explicitly; the
                    // failed page stays resident/spilled and is at worst an
                    // unreachable allocation until restart.
                    tracing::error!(
                        tenant = ?tenant,
                        head,
                        %error,
                        "ADR-033 rollback could not remove an uncommitted blob chain",
                    );
                }
            }

            // Step 5 (ADR-039 §D-6): drain bm25_pending and dispatch
            // rollback per tenant.
            //
            // The bm25 drain runs AFTER page_mutations and blob_heads
            // — Tantivy's IndexWriter buffer is the rollback granularity
            // (no pre-W bytes for BM25 at v1.0). The drain is per-tenant
            // (one rollback_pending call per touched tenant) because
            // Tantivy serializes the writer per index (per-tenant at
            // v1.0). When `bm25_store` is `None`, warn-and-skip mirrors
            // the Vector arm posture so deployments without BM25 wiring
            // do not surface a hard error.
            let drained_bm25: Vec<_> = log.bm25_pending.drain(..).collect();
            for tenant in drained_bm25 {
                if let Some(bm25) = store.bm25_store.as_ref() {
                    if let Err(e) = bm25.rollback_pending(tenant) {
                        tracing::warn!(
                            "ADR-039 Z-1 rollback: bm25 rollback_pending failed \
                             for tenant {:?}: {}",
                            tenant,
                            e,
                        );
                    }
                } else {
                    tracing::warn!(
                        "ADR-039 Z-1 rollback: no Bm25IndexStoreHandle wired \
                         into CrudStore; skipping bm25_pending entry for \
                         tenant {:?}. Wire one via CrudStore::with_bm25_store.",
                        tenant,
                    );
                }
            }
        },
    );

    let commit_lsn = match commit_result {
        Ok(lsn) => lsn,
        Err(e) => {
            store.discard_pending(txn_id);
            store.discard_pending_installs(txn_id);
            // v2 M1 — rollback path: discard the txn's private slotted
            // scratch pages; a page checked out of the tenant pool is
            // restored to its exact checkout-time state (nothing was
            // published — readers never saw the aborted bags).
            store.blobs.rollback_txn_slotted(txn_id);
            // RC-1: the aborted txn superseded nothing durably, so its
            // collected old-value removals fall away with it (the batch
            // is dropped here — never enqueued). Its secondary page inserts
            // were rolled back by the closure above, so discard their
            // in-flight #1464 markers without publishing a generation.
            store.discard_secondary_reassertions(txn_id, &secondary_reassertions_batch);
            return Err(CrudError::from(e));
        }
    };

    // M4 S1 addressing alternate: the logical commit succeeded, so mirror
    // its exact record bytes/tombstones into the optional arithmetic store.
    // This is post-commit derivative state in this slice; failures warn and
    // leave MVCC + the still-present primary B-tree authoritative.
    store.publish_addressed_installs(&installs, commit_lsn);

    // v2 M1 — the commit is durable: install the txn's slotted prop
    // pages into the resident tier (their owning records became visible
    // with this commit) and check still-open pages back into the tenant
    // pool. Check-in strictly AFTER the WAL append is what serializes
    // per-page image states in LSN order (see the blob.rs v2 M1
    // concurrency note).
    if !writes_delta {
        store.blobs.publish_txn_slotted(txn_id)?;
    }

    // RC-1 (#1366) + #1464: the commit is durable. First replace every
    // in-flight update-side insert marker with the real `commit_lsn`; then
    // enqueue old-value removals and opportunistically apply everything the
    // snapshot horizon cleared. Publishing the generation first also covers
    // an older commit whose post-commit enqueue was delayed behind this one.
    // Enqueue-then-drain (not drain-only) keeps the no-reader common case
    // eager while a long reader still pins its required ghosts.
    store.finish_secondary_reassertions(txn_id, commit_lsn, &secondary_reassertions_batch);
    store.enqueue_deferred_removals(commit_lsn, txn_tenant, &deferred_removals_batch);
    store.apply_ready_deferred_removals(snapshot_horizon);

    // ADR-032 Slice 2: the primary index's grow_root root-pointer
    // persist is folded into the outer CommitBundle's sidechannel
    // list (see `primary.upsert_deferred(..., sc_writes)` inside the
    // builder above). No post-commit drain is required — the
    // CommitBundle v2 codec + Phase-3 apply_sidechannel_mvcc_write
    // make "MVCC has root-pointer R ⟹ page_store has R installed"
    // an invariant of every durable commit, closing #66 by
    // construction.
    //
    // The secondary index still uses the pre-Slice-2 stash +
    // post-commit drain shape; the parallel refactor remains deferred.
    // Leaving the secondary drain in place keeps
    // secondary grow_root working identically to pre-Slice-2. This
    // note is scoped to the COMMIT (durable-persist) path only: the
    // ABORT-path crash hazard was closed by Z-1 F-1 (#1366) — the
    // rollback closure above now drains the secondary's aborted
    // `new_pages` / `root_changes` / `page_mutations` (see Step 1/2
    // and their `IndexHandle::SECONDARY` / `PageStoreKind::Secondary`
    // arms), so an aborted secondary grow_root no longer leaks pages
    // or leaves a stale root. What remains pre-existing is the
    // COMMIT-side #66-class hazard: unlike the primary (whose
    // root-pointer persist folds into the CommitBundle above), the
    // secondary's root-pointer persist is a SEPARATE post-commit
    // `persist_pending_root_update`, so a crash BETWEEN the commit
    // fsync and this persist can still leave the durable root pointer
    // one grow_root behind. Closing that is the deferred Slice-2
    // follow-up.
    if let Some(secondary) = store.secondary.as_ref()
        && let Err(e) = secondary.persist_pending_root_update()
    {
        tracing::warn!(
            "secondary-index deferred root-pointer persist failed: {}",
            e
        );
    }

    if physical_owner_registry.is_none()
        && !idempotency_releases_to_publish.is_empty()
        && let Some(idempotency) = store.idempotency_store.read().clone()
    {
        for entry in idempotency_releases_to_publish {
            idempotency.release(entry.tenant, entry.kind, &entry.external_id);
        }
    }

    // Drain TEL appends AFTER commit (unchanged). TEL does not
    // participate in the CommitBundle; its MVCC↔TEL atomicity gap is
    // issue #20.
    //
    // W26-β-2 / ADR-131 — for every pending append, ALSO publish the
    // reverse-direction entry keyed by `(tenant, dst, channel)`. The
    // reverse path is in-memory work (~200ns) and inherits the same
    // MVCC↔TEL atomicity gap (issue #20) as the forward path; if it
    // fires inside this loop they share the same crash window.
    // `tel_append_reverse` short-circuits to a no-op when
    // `store.reverse_index_enabled()` is `false` (used by the AC-4
    // fault-injection harness; production deployments leave it
    // enabled at v1.1).
    for p in pending_tel {
        store.tel_append(p.tenant, p.src, p.channel, p.dst, p.rel, commit_lsn)?;
        store.tel_append_reverse(p.tenant, p.dst, p.channel, p.src, p.rel, commit_lsn)?;
    }

    // M4-41 (M4-04a) — catalog stats hook per ADR-038 §2 D-25.
    //
    // After the WAL fsync has succeeded and the MVCC commit has
    // landed (every byte above this line is either a successful
    // dual-write or a TEL append the kernel has already accepted),
    // walk the same `installs` vector the bundle builder consumed
    // and update the per-tenant stats counters. The vector is the
    // committed delta — every entry corresponds to a record that
    // is now durably present (Create), durably overwritten (Update,
    // a no-op for cardinality), or durably tombstoned (Delete) at
    // `commit_lsn`.
    //
    // Updates are scoped per tenant because each `PendingInstall`
    // carries its own `tenant` field. Multi-tenant isolation is
    // structural — we get-or-create the right `CatalogStats` per
    // entry; cross-tenant pollution is impossible by construction.
    //
    // `Update` does not change `label_id` or `type_id` (those are
    // topology, not mutable properties — see `update_node` and
    // `update_rel` rustdoc), so the stats hook skips it.
    //
    // `Delete` decrements iff the prior label/type was captured in
    // the install (the `delete_*_with_store` API does this); if
    // not, the per-class counter is conservatively skipped while
    // the tenant-wide total is still decremented. v1.0 production
    // routes deletes through `*_with_store`, so the conservative
    // skip is dead code in production but defensive against any
    // future delete path that bypasses store-aware capture.
    //
    // Per-tenant `observe_commit()` runs once at the end of the
    // walk so the `total_*_count()` boundary translation switches
    // from `None` to `Some(count)` exactly when the first commit
    // for that tenant lands. Without this, totals would surface as
    // `Some(0)` for a fresh tenant whose first commit only modified
    // a non-stats-tracked path; the boundary contract is "None
    // until any commit observed."
    //
    // M4-04e (issue #210) — `begin_commit_observation()` runs
    // ONCE per touched tenant BEFORE the increment walk fires;
    // `observe_commit()` at the END pairs with it. The two markers
    // bracket the per-counter Relaxed writes so that a concurrent
    // `CatalogStats::snapshot()` can detect mid-commit interleaving
    // (`commits_started > commits_observed` ⟹ commit in flight ⟹
    // snapshot retries). Without the begin marker the snapshot's
    // SeqLock retry can't catch torn cross-key reads. The end marker
    // is invoked UNCONDITIONALLY (outside `catch_unwind`) so a
    // partial-walk panic still rebalances `commits_started ==
    // commits_observed`, restoring snapshot liveness for subsequent
    // readers.
    //
    // PR #170 reviewer Finding 2 — `catch_unwind` wrap. The WAL has
    // already fsynced and `commit_lsn` is final by the time we get
    // here; a panic anywhere inside the stats-update block (e.g.,
    // a future `CatalogStats` refactor that allocates and OOMs, or
    // a corrupted `installs` entry) MUST NOT bubble up as a commit
    // failure. We log the divergence and return `Ok(commit_lsn)`
    // because the commit IS durable. Stats inconsistency is
    // observable via the log; correctness of the commit is
    // unaffected.
    //
    // `AssertUnwindSafe` is sound here because the only mutated
    // state is the `&CrudStore` (DashMap + AtomicU64 increments,
    // both panic-safe — atomics never panic, DashMap entry-or-insert
    // panics only on allocator failure which is unwind-safe). No
    // shared lock is held across the boundary; a panic mid-walk
    // leaves `CatalogStats` in a possibly-inconsistent-but-not-
    // corrupted state (some increments applied, others not), which
    // is precisely the divergence we log.
    //
    // PR #170 reviewer Finding 3 — `tracing::warn!` on decode
    // failure (per the `match` arms below) so a future bytes-shape
    // regression in NodeRecord / RelRecord codecs surfaces as a
    // log line rather than a silent stats-skip.
    if !installs.is_empty() {
        // M4-04e: pre-compute the unique set of tenants this commit
        // touches. The iteration uses only `inst.tenant()` (no
        // allocation), so it cannot panic; we hoist it out of the
        // `catch_unwind` so the begin/end markers can run
        // unconditionally around the increment walk.
        let mut touched_tenants: Vec<TenantId> = Vec::new();
        for inst in &installs {
            let tenant = inst.tenant();
            if !touched_tenants.contains(&tenant) {
                touched_tenants.push(tenant);
            }
        }

        // M4-04e begin markers. Bump `commits_started` Release per
        // tenant BEFORE any per-counter increment fires; this is the
        // SeqLock front fence the snapshot reader detects to retry on
        // mid-commit interleaving.
        for tenant in &touched_tenants {
            store
                .tenant_catalog_stats(*tenant)
                .begin_commit_observation();
        }

        let stats_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for inst in &installs {
                let tenant = inst.tenant();
                let stats = store.tenant_catalog_stats(tenant);
                match inst {
                    PendingInstall::Create {
                        kind,
                        bytes,
                        src_label_raw,
                        ..
                    } => match kind {
                        RecordKind::Node => match decode_node_bytes(bytes) {
                            Ok(rec) => {
                                stats.increment_label(LabelId::new(rec.label_id));
                                stats.increment_total_nodes();
                            }
                            Err(e) => {
                                tracing::warn!(
                                    ?tenant,
                                    error = %e,
                                    "CatalogStats: decode failure for node entry; \
                                     stats not updated for this entry",
                                );
                            }
                        },
                        RecordKind::Rel => match decode_rel_bytes(bytes) {
                            Ok(rec) => {
                                let rel_type = TypeId::new(rec.type_id);
                                stats.increment_rel_type(rel_type);
                                stats.increment_total_rels();
                                if let Some(label) = src_label_raw {
                                    stats.record_out_degree(
                                        LabelId::new(*label),
                                        rel_type,
                                        NodeId::new(rec.src_id),
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    ?tenant,
                                    error = %e,
                                    "CatalogStats: decode failure for rel entry; \
                                     stats not updated for this entry",
                                );
                            }
                        },
                    },
                    PendingInstall::Update { .. } => {
                        // Cardinality unchanged — properties may have
                        // changed but label_id / type_id are immutable
                        // through the update_* surface (PR #170
                        // reviewer Finding 4 enforces this via a
                        // debug_assert! in update_node / update_rel).
                    }
                    PendingInstall::Delete {
                        kind,
                        prior_topology_raw,
                        ..
                    } => match kind {
                        RecordKind::Node => {
                            if let Some(raw) = prior_topology_raw {
                                stats.decrement_label(LabelId::new(*raw));
                            }
                            stats.decrement_total_nodes();
                        }
                        RecordKind::Rel => {
                            // The max-out-degree sketch is increment-only:
                            // deletes may leave an overestimate, which is
                            // safe for a supernode firewall because it can
                            // only make the planner more conservative.
                            if let Some(raw) = prior_topology_raw {
                                stats.decrement_rel_type(TypeId::new(*raw));
                            }
                            stats.decrement_total_rels();
                        }
                    },
                }
            }
        }));

        // M4-04e end markers. UNCONDITIONAL — runs even if the
        // increment walk panicked above. Bumps `commits_observed`
        // Release per tenant, restoring the SeqLock invariant
        // `commits_started == commits_observed` so subsequent
        // `snapshot()` callers don't spin retrying. Stats counters
        // may be partially updated if a panic occurred mid-walk
        // (logged below), but the SeqLock invariant is preserved so
        // the planner can read whatever state IS consistent.
        for tenant in &touched_tenants {
            store.tenant_catalog_stats(*tenant).observe_commit();
        }

        if let Err(panic_payload) = stats_result {
            // PR #170 reviewer Finding 2: WAL is durable, commit
            // succeeds. Log the divergence; do not propagate the
            // panic to the caller (which would surface as a commit
            // failure even though the data IS persisted).
            let panic_msg = panic_payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic_payload.downcast_ref::<&'static str>().copied())
                .unwrap_or("<non-string panic payload>");
            tracing::error!(
                commit_lsn = ?commit_lsn,
                panic_message = panic_msg,
                "CatalogStats: stats-update panic post-commit; \
                 WAL is durable, stats may diverge from on-disk state. \
                 See ADR-038 §2 D-25 'Persistence/recovery (M4-31+ wiring \
                 requirement)' — M4-31+ rebuild-on-cold-start path will \
                 reconcile any divergence on next process restart.",
            );
        }
    }

    Ok(commit_lsn)
}

// ─────────────────────────────────────────────────────────────────────
// #1221 (ADR-218) — CrudAclWalSink: durable WAL sink for PermissionIndex
// ─────────────────────────────────────────────────────────────────────

/// Durable [`crate::permissions::AclWalSink`] backed by the CRUD/WAL
/// layer (#1221 — ADR-218).
///
/// The `PermissionIndex` write-through (`apply_doc_acl` / `revoke_doc`)
/// is called OUTSIDE any open transaction (the seed + live `graph.ingest`
/// paths commit the content/provenance graph first, THEN apply the ACL).
/// So each ACL op rides its OWN dedicated single-op v8 commit: begin a
/// transaction, stage exactly one `AclGrantEntry`, and `commit` — which
/// folds it into the v8 bundle's `acl_grants` section and fsyncs (the
/// txn is the SYSTEM/strict-tier path; the empty MVCC write-set is legal,
/// the bundle carries only the `acl_grants` tail). Both-or-neither: the
/// op is durable iff its commit is. On commit failure the in-memory index
/// already holds the op (applied first by `apply_doc_acl_inner`), so a
/// torn write degrades to "lost on the next restart" — fail-closed (the
/// doc reverts to UNCLASSIFIED), never a widen.
///
/// The op is **always one entry per commit**, so the per-doc-per-bundle
/// collision the ADR-218 append-order invariant guards against never
/// arises on this path — but the encoder preserves append order
/// regardless (the invariant is enforced unconditionally in the codec).
pub struct CrudAclWalSink {
    txn_mgr: Arc<TxnManager>,
    store: Arc<CrudStore>,
    tenant: TenantId,
}

// `CrudStore` / `TxnManager` are not `Debug` (large internal state), so
// the `AclWalSink: Debug` bound is satisfied with a placeholder that
// surfaces only the tenant — mirrors the `TenantHandle` manual-Debug
// pattern in `router.rs`.
impl std::fmt::Debug for CrudAclWalSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrudAclWalSink")
            .field("tenant", &self.tenant)
            .finish_non_exhaustive()
    }
}

impl CrudAclWalSink {
    /// Wire a sink for `tenant` over the shared `txn_mgr` + `store`.
    #[must_use]
    pub fn new(txn_mgr: Arc<TxnManager>, store: Arc<CrudStore>, tenant: TenantId) -> Self {
        Self {
            txn_mgr,
            store,
            tenant,
        }
    }

    /// Durify exactly one ACL op via a dedicated single-op v8 commit.
    /// Logs (does NOT panic) on commit failure — the caller already
    /// mutated the in-memory index; a failed durify is a fail-closed
    /// "lost on next restart", never a widen.
    fn durify_one(&self, entry: AclGrantEntry) {
        let tx = self.txn_mgr.begin(self.tenant);
        let txn_id = tx.id();
        self.store.stage_acl_grant(txn_id, entry);
        // `commit` drains `pending_acl_grants[txn_id]` → stages on `tx` →
        // encodes the v8 bundle. An empty MVCC write-set still fires a
        // CommitBundle record carrying the `acl_grants` tail.
        if let Err(e) = commit(tx, &self.store) {
            // Discard the staged op if commit bailed before draining
            // (defensive — `commit` drains on its happy path; this covers
            // an early-return variant). `txn_id` is unique-per-txn so this
            // never touches another txn's buffer.
            self.store.discard_pending_acl_grants(txn_id);
            tracing::warn!(
                target: "arcgraph_storage::permissions",
                tenant = ?self.tenant,
                error = %e,
                "ACL WAL durify failed; op applied in-memory but NOT durable \
                 (fail-closed — reverts to UNCLASSIFIED on next restart)"
            );
        }
    }
}

impl crate::permissions::AclWalSink for CrudAclWalSink {
    fn durify_apply(&self, doc: NodeId, grants: &std::collections::BTreeSet<String>) {
        self.durify_one(AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant: self.tenant,
            doc,
            grants: grants.clone(),
        });
    }

    fn durify_revoke(&self, doc: NodeId) {
        self.durify_one(AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant: self.tenant,
            doc,
            grants: std::collections::BTreeSet::new(),
        });
    }
}

// ─────────────────────────────────────────────────────────────────────
// M2-25 — update_node / update_rel via MVCC (and read_rel symmetry)
// ─────────────────────────────────────────────────────────────────────

pub(crate) fn decode_node_bytes(bytes: &[u8]) -> Result<NodeRecord, CrudError> {
    let arr: &[u8; NodeRecord::SIZE] = bytes.try_into().map_err(|_| {
        CrudError::Mvcc(ArcGraphError::InvalidRecordLength {
            got: bytes.len(),
            expected: NodeRecord::SIZE,
        })
    })?;
    NodeRecord::from_bytes(arr).map_err(CrudError::Mvcc)
}

pub(crate) fn decode_rel_bytes(bytes: &[u8]) -> Result<RelRecord, CrudError> {
    let arr: &[u8; RelRecord::SIZE] = bytes.try_into().map_err(|_| {
        CrudError::Mvcc(ArcGraphError::InvalidRecordLength {
            got: bytes.len(),
            expected: RelRecord::SIZE,
        })
    })?;
    RelRecord::from_bytes(arr).map_err(CrudError::Mvcc)
}

/// Read `id`'s relationship record at `tx`'s snapshot LSN.
///
/// Per ADR-018, rel record bytes live only in the MVCC chain; this
/// is a single `tx.read(rel_mvcc_key(id))` with no page pin, symmetric
/// with [`read_node`].
pub fn read_rel(tx: &Transaction<'_>, id: RelId) -> Result<Option<RelRecord>, CrudError> {
    let Some(bytes) = tx.read(rel_mvcc_key(id)) else {
        return Ok(None);
    };
    decode_rel_bytes(bytes.as_ref()).map(Some)
}

/// Update `id`'s properties, staging a new MVCC version at commit.
///
/// Only properties are mutable through this call. `label_id` is
/// preserved from the version visible at `tx.snapshot()`; callers that
/// need a label change must do it through a schema migration (out of
/// M2.c scope).
///
/// Existence check: if no version of `id` is visible, returns
/// [`CrudError::NotFound`]. Silent upsert on update is a footgun — a
/// caller that wants "create or update" composes `create_node` and
/// `update_node` explicitly.
///
/// Conflict handling is standard OCC: a write–write race with another
/// in-flight updater surfaces as [`ArcGraphError::MvccConflict`] on
/// `tx.commit()`, not here; this call only buffers the new bytes.
///
/// `created_lsn` on the rebuilt record stays at `Lsn::ZERO` — the
/// authoritative LSN lives on the MVCC `Version`; the in-record field
/// is a placeholder per ADR-018 until page materialization lands at
/// M2.d.
pub fn update_node(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    id: NodeId,
    props: &PropertyData,
) -> Result<(), CrudError> {
    let key = node_mvcc_key(id);
    let Some(bytes) = tx.read(key) else {
        return Err(CrudError::NotFound {
            kind: "node",
            id: id.raw(),
            tenant: tx.tenant(),
        });
    };
    let current = decode_node_bytes(bytes.as_ref())?;
    let mut rec = NodeRecord::new(id, LabelId::new(current.label_id), Lsn::ZERO);
    let blob_emits = props.apply_to_node(&mut rec, tx.tenant(), &store.blobs, tx.id())?;
    store.buffer_blob_emits(tx.id(), blob_emits);
    let record_bytes = rec.to_bytes();
    // PR #170 reviewer Finding 4 — `label_id` is immutable through
    // `update_node`. Structurally guaranteed (rec is constructed with
    // `current.label_id`; `apply_to_node` only mutates property
    // fields), but debug-asserting the round-trip catches future
    // API regressions (e.g., a refactor that adds a `label`
    // parameter to `update_node`) and codec drift in `NodeRecord::
    // to_bytes` / `from_bytes`. The CatalogStats commit hook
    // (`PendingInstall::Update => no-op`) relies on this
    // invariant.
    debug_assert_eq!(
        decode_node_bytes(&record_bytes).map(|r| r.label_id).ok(),
        Some(current.label_id),
        "label_id is immutable through update_node; CatalogStats hook \
         (PR #170 reviewer Finding 4) relies on this invariant",
    );
    tx.write(key, Bytes::copy_from_slice(&record_bytes));
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Update {
                tenant: tx.tenant(),
                kind: RecordKind::Node,
                id: id.raw(),
                bytes: record_bytes.to_vec(),
            },
        );
    }
    Ok(())
}

/// Update `id`'s relationship properties, staging a new MVCC version
/// at commit.
///
/// Endpoints (`src_id`, `dst_id`) and `type_id` are preserved from the
/// currently visible version — those are topology, not mutable
/// properties. The TEL chain is **not** touched: the entry keyed by
/// `(dst_id, rel_id, created_lsn)` remains valid because none of those
/// fields change. Readers at past snapshots see the old props via the
/// MVCC chain; readers at the new snapshot see the new props through
/// the same `TelEntry` → `read_rel` path.
///
/// Same existence/OCC semantics as [`update_node`].
pub fn update_rel(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    id: RelId,
    props: &PropertyData,
) -> Result<(), CrudError> {
    let key = rel_mvcc_key(id);
    let Some(bytes) = tx.read(key) else {
        return Err(CrudError::NotFound {
            kind: "rel",
            id: id.raw(),
            tenant: tx.tenant(),
        });
    };
    let current = decode_rel_bytes(bytes.as_ref())?;
    let mut rec = RelRecord::new(
        id,
        TypeId::new(current.type_id),
        NodeId::new(current.src_id),
        NodeId::new(current.dst_id),
        Lsn::ZERO,
    );
    let blob_emits = props.apply_to_rel(&mut rec, tx.tenant(), &store.blobs, tx.id())?;
    store.buffer_blob_emits(tx.id(), blob_emits);
    let record_bytes = rec.to_bytes();
    // PR #170 reviewer Finding 4 — `type_id` is immutable through
    // `update_rel`. Same rationale as `update_node`'s assertion:
    // structurally guaranteed by construction, debug-asserted to
    // catch future API regression / codec drift. The CatalogStats
    // commit hook (`PendingInstall::Update => no-op`) relies on
    // this invariant.
    debug_assert_eq!(
        decode_rel_bytes(&record_bytes).map(|r| r.type_id).ok(),
        Some(current.type_id),
        "type_id is immutable through update_rel; CatalogStats hook \
         (PR #170 reviewer Finding 4) relies on this invariant",
    );
    tx.write(key, Bytes::copy_from_slice(&record_bytes));
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Update {
                tenant: tx.tenant(),
                kind: RecordKind::Rel,
                id: id.raw(),
                bytes: record_bytes.to_vec(),
            },
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// M2-26 — delete_node / delete_rel via MVCC tombstone
// ─────────────────────────────────────────────────────────────────────

/// Tombstone `id`, staging the deletion as an MVCC `Version {
/// value: None }` at commit.
///
/// Per ADR-018, deletes are a pure MVCC operation in v1.0 alpha: the
/// kernel closes the previous version's `expired_lsn` at install and
/// the tombstone version becomes the one visible at later snapshots.
/// No page is touched; no slot is freed. Pre-delete snapshots continue
/// to read the prior version through the MVCC chain.
///
/// Existence check: if `id` has no version visible at `tx.snapshot()`,
/// returns [`CrudError::NotFound`]. Silent success on a missing id
/// hides caller bugs.
pub fn delete_node(tx: &mut Transaction<'_>, id: NodeId) -> Result<(), CrudError> {
    let key = node_mvcc_key(id);
    if tx.read(key).is_none() {
        return Err(CrudError::NotFound {
            kind: "node",
            id: id.raw(),
            tenant: tx.tenant(),
        });
    }
    tx.delete(key);
    Ok(())
}

/// Tombstone a relationship.
///
/// Symmetric with [`delete_node`]: stages `tx.delete(rel_mvcc_key(id))`;
/// `read_rel` at later snapshots returns `Ok(None)`.
///
/// **v1.0 alpha limitation.** The TEL entry for the rel is *not*
/// removed and `scan_out` continues to yield it. The caller must
/// pair the rel id against `read_rel` if a dead-entry filter is
/// required. Proper TEL-side tombstoning lands with the M2.d primary
/// index. Tracked as GitHub issue #22
/// ("scan_out does not filter out TelEntries for MVCC-tombstoned rels").
pub fn delete_rel(tx: &mut Transaction<'_>, id: RelId) -> Result<(), CrudError> {
    let key = rel_mvcc_key(id);
    if tx.read(key).is_none() {
        return Err(CrudError::NotFound {
            kind: "rel",
            id: id.raw(),
            tenant: tx.tenant(),
        });
    }
    tx.delete(key);
    Ok(())
}

/// Dual-write-aware `delete_node` (M2-CUTOVER): tombstones MVCC AND
/// the slotted record page AND the primary index entry. Falls back to
/// pure MVCC when the store has no primary configured.
pub fn delete_node_with_store(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    id: NodeId,
) -> Result<(), CrudError> {
    // M4-41: capture the prior label BEFORE staging the delete so
    // the post-commit stats hook can decrement the right
    // per-label counter without re-reading the (now-tombstoned)
    // MVCC chain. `delete_node` below also re-reads the same key
    // for its existence check, but the duplicate is cheap (one
    // DashMap probe under tx.snapshot()) and keeps the stats
    // capture co-located with the delete API rather than
    // sprinkling label-extraction logic inside `delete_node`.
    let prior_node = tx
        .read(node_mvcc_key(id))
        .and_then(|bytes| decode_node_bytes(bytes.as_ref()).ok())
        .map(|rec| (rec.label_id, store.idempotency_store.read().clone()));
    let prior_topology_raw = prior_node.as_ref().map(|(label, _)| *label);
    delete_node(tx, id)?;
    if let Some((_, Some(idempotency))) = prior_node {
        if let Some(external_id) = idempotency.try_external_id_for(tx.tenant(), 0, id.raw())? {
            store.stage_idempotency_release(tx.id(), tx.tenant(), 0, id.raw(), external_id);
        }
    }
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Delete {
                tenant: tx.tenant(),
                kind: RecordKind::Node,
                id: id.raw(),
                prior_topology_raw,
            },
        );
    }
    Ok(())
}

/// Delete a node AND revoke its doc-ACL, atomically from the caller's
/// perspective (#1379, MUST-CON-04).
///
/// # The leak this closes
///
/// [`delete_node_with_store`] tombstones the MVCC record (and the
/// primary-index slot) but does NOT touch the tenant's
/// [`crate::permissions::PermissionIndex`]. So a
/// deleted node's `doc_class` mapping SURVIVES the delete:
/// `is_visible(id, P)` stays `true` for every principal `P` the node was
/// granted to, and — because the BM25 / vector substrates key their
/// tombstoning off a SEPARATE `mark_*_node_deleted` seam that the served
/// path drives independently — a delete that skipped the ACL revoke
/// leaves the deleted node retrievable AND is_visible-true forever. That
/// is a live data-leak: a sensitive node stays readable after it was
/// deleted.
///
/// # The symmetric revoke (mirrors the ingest write-through)
///
/// The ingest path applies a doc-ACL via
/// [`crate::permissions::PermissionIndex::apply_doc_acl`] (the served
/// wiring lives in `arcgraph-mcp`'s `apply_live_acl_grants`, reaching
/// `permissions()` off the routed [`crate::router::TenantHandle`]).
/// DELETE is the symmetric op, so it drives the symmetric
/// [`crate::permissions::PermissionIndex::revoke_doc`] — which drops the
/// doc→class mapping (→ UNCLASSIFIED → invisible to every principal,
/// fail-closed) AND, when a durable [`crate::permissions::AclWalSink`] is
/// wired (durable `serve --data`), durifies a `Revoke` op into the WAL's
/// `acl_grants` section (#1221 / ADR-218) so the revoke SURVIVES A BARE
/// RESTART. On replay the WAL re-drives `revoke_doc_replayed` and the doc
/// stays UNCLASSIFIED.
///
/// # Bounded-context note
///
/// [`crate::permissions::PermissionIndex`] lives in THIS crate
/// (`arcgraph-storage::permissions`), so taking `&PermissionIndex` here
/// crosses no crate boundary (PD-7). The served caller (`arcgraph-mcp`'s
/// `delete_node` substrate op) reaches the per-tenant index via
/// `TenantHandle::permissions()` exactly as the ingest write-through
/// does, then calls this — `crud` never reaches "up" into the router
/// itself.
///
/// The order is delete-then-revoke: the tombstone is staged on `tx`
/// (committed by the caller) and the ACL revoke rides its own dedicated
/// single-op v8 commit inside `revoke_doc` (both-or-neither with its own
/// commit, ADR-218). A revoke that races the tombstone commit only ever
/// UNDER-grants (the doc reverts to invisible early) — never a widen.
pub fn delete_node_with_store_and_revoke(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    id: NodeId,
    permissions: &crate::permissions::PermissionIndex,
) -> Result<(), CrudError> {
    delete_node_with_store(store, tx, id)?;
    // #1379 — the symmetric revoke. `revoke_doc` durifies via the wired
    // `AclWalSink` (durable restart-survivable Revoke) when a durable
    // backend is in play, and is an in-memory-only drop otherwise.
    permissions.revoke_doc_checked(id)?;
    Ok(())
}

/// Dual-write-aware `delete_rel` symmetric with [`delete_node_with_store`].
pub fn delete_rel_with_store(
    store: &CrudStore,
    tx: &mut Transaction<'_>,
    id: RelId,
) -> Result<(), CrudError> {
    // M4-41: symmetric prior-type capture. See
    // `delete_node_with_store` for rationale.
    let prior_topology_raw = tx
        .read(rel_mvcc_key(id))
        .and_then(|bytes| decode_rel_bytes(bytes.as_ref()).ok())
        .map(|rec| rec.type_id);
    delete_rel(tx, id)?;
    if store.has_physical_record_target() {
        store.buffer_install(
            tx.id(),
            PendingInstall::Delete {
                tenant: tx.tenant(),
                kind: RecordKind::Rel,
                id: id.raw(),
                prior_topology_raw,
            },
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::TxnManager;

    #[test]
    fn deferred_v9_applies_preserve_commit_order_when_later_exact_proof_arrives_first() {
        use std::path::PathBuf;
        use std::time::Duration;

        use crate::wal::segment::{SegmentHeader, segment_filename};
        use crate::wal::{BUNDLE_FORMAT_V9, WalConfig, WalWriter};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join(segment_filename(0)),
            SegmentHeader {
                format_version: BUNDLE_FORMAT_V9,
            }
            .encode(),
        )
        .unwrap();
        let config = WalConfig {
            dir: PathBuf::from(dir.path()),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_secs(60),
            group_commit_max_batch: 16,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        };
        let writer = WalWriter::spawn_from(config, Lsn::ZERO).unwrap();
        let wal = writer.handle();

        // Phase-2 may reach the WAL out of commit order. Inject the exact
        // completion edge directly: later is durable, earlier is pending.
        wal.__test_mark_exact_durable(Lsn::new(10));

        let (_txn, mut store, _primary) = build_dual_write_store();
        store.attach_wal(wal.clone());
        let store = Arc::new(store);
        let page_id = PageId::new(77);
        let earlier = vec![
            DeltaOp::new(
                DeltaOpKind::PageAlloc,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(4),
                {
                    let mut payload = vec![PageType::Node.as_byte()];
                    payload.extend_from_slice(&1u64.to_le_bytes());
                    payload
                },
            )
            .unwrap(),
            DeltaOp::new(
                DeltaOpKind::PutRecord,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(5),
                NodeRecord::new(NodeId::new(1), LabelId::new(1), Lsn::new(5))
                    .to_bytes()
                    .to_vec(),
            )
            .unwrap(),
        ];
        let later = vec![
            DeltaOp::new(
                DeltaOpKind::PageAlloc,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(9),
                {
                    let mut payload = vec![PageType::Node.as_byte()];
                    payload.extend_from_slice(&1u64.to_le_bytes());
                    payload
                },
            )
            .unwrap(),
            DeltaOp::new(
                DeltaOpKind::PutRecord,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                1,
                Lsn::new(10),
                NodeRecord::new(NodeId::new(2), LabelId::new(2), Lsn::new(10))
                    .to_bytes()
                    .to_vec(),
            )
            .unwrap(),
        ];

        let first = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || store.apply_or_defer_v9_deltas(5, &earlier, Lsn::new(5)))
        };
        first.join().unwrap().unwrap();
        let second = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || store.apply_or_defer_v9_deltas(10, &later, Lsn::new(10)))
        };
        second.join().unwrap().unwrap();

        assert_eq!(
            store.deferred_v9_boundary(),
            Some(DeferredV9Boundary {
                commit_lsn: Lsn::new(5),
                redo_lsn: Lsn::new(4),
            }),
            "checkpoint must retain the oldest queued bundle including PageAlloc"
        );

        wal.__test_mark_exact_durable(Lsn::new(5));
        assert_eq!(store.drain_deferred_v9_applies().unwrap(), 2);
        let latch = store.records().unwrap().latch(page_id).unwrap();
        let guard = latch.read();
        let page = SlottedPageRef::open(guard.as_ref()).unwrap();
        assert_eq!(
            page.read_node(SlotId(0)).unwrap().unwrap().id,
            1,
            "the earlier same-page commit must not be skipped after a later page-LSN install"
        );
        assert_eq!(page.read_node(SlotId(1)).unwrap().unwrap().id, 2);
        drop(guard);
        writer.shutdown().unwrap();
    }

    #[test]
    fn failed_deferred_v9_apply_keeps_entry_and_proof_and_blocks_reclamation() {
        use std::path::PathBuf;
        use std::time::Duration;

        use crate::buffer::BufferPool;
        use crate::checkpoint::{
            CheckpointError, CheckpointSnapshot, DoublewriteArea, WriteBehindCheckpointer,
            incremental_checkpoint, read_latest_sidecar,
        };
        use crate::idempotency::IdempotencyStore;
        use crate::intern::InternTable;
        use crate::io::{InMemoryPageIo, PageIo};
        use crate::page_store::{
            BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig,
        };
        use crate::permissions::PermissionIndex;
        use crate::primary_index::PrimaryPageStore;
        use crate::redo::DirtyPageTable;
        use crate::wal::segment::{SegmentHeader, segment_filename};
        use crate::wal::{
            BUNDLE_FORMAT_V9, PageStoreTarget, WalConfig, WalRecord, WalRecordType, WalWriter,
            encode_commit_bundle_v9, reclaim_segments_below,
            recover_from_wal_encrypted_incremental,
        };
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let page_id = PageId::new(77);
        let node_id = NodeId::new(700);
        let mut alloc_payload = vec![PageType::Node.as_byte()];
        alloc_payload.extend_from_slice(&1u64.to_le_bytes());
        let deltas = vec![
            DeltaOp::new(
                DeltaOpKind::PageAlloc,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(9),
                alloc_payload,
            )
            .unwrap(),
            DeltaOp::new(
                DeltaOpKind::PutRecord,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(10),
                NodeRecord::new(node_id, LabelId::new(7), Lsn::new(10))
                    .to_bytes()
                    .to_vec(),
            )
            .unwrap(),
        ];
        let payload = encode_commit_bundle_v9(
            Lsn::new(10),
            TenantId::DEFAULT,
            &std::collections::HashMap::new(),
            &[],
            &deltas,
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        let mut closed = SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode()
        .to_vec();
        WalRecord {
            record_type: WalRecordType::CommitBundle,
            txn_id: 10,
            lsn: Lsn::new(10),
            timestamp_ms: 0,
            tenant_id: TenantId::DEFAULT,
            payload,
        }
        .encode(&mut closed)
        .unwrap();
        std::fs::write(dir.path().join(segment_filename(0)), closed).unwrap();
        std::fs::write(
            dir.path().join(segment_filename(1)),
            SegmentHeader {
                format_version: BUNDLE_FORMAT_V9,
            }
            .encode(),
        )
        .unwrap();

        let config = WalConfig {
            dir: PathBuf::from(dir.path()),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_secs(60),
            group_commit_max_batch: 16,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        };
        let writer = WalWriter::spawn_from(config, Lsn::new(10)).unwrap();
        let wal = writer.handle();
        wal.__test_mark_exact_durable(Lsn::new(10));

        let (txn, mut store, primary) = build_dual_write_store();
        txn.seed_after_replay(Lsn::new(20));
        store.attach_wal(wal.clone());
        let store = Arc::new(store);
        store.deferred_v9_applies.lock().push_back(DeferredV9Apply {
            txn_id: 10,
            commit_lsn: Lsn::new(10),
            deltas,
        });
        store.fail_durable_v9_apply.store(true, Ordering::Release);
        let first_error = store.drain_deferred_v9_applies().unwrap_err();
        assert!(
            first_error
                .to_string()
                .contains("injected durable v9 page-apply failure")
        );

        let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
        let pools = Arc::new(PerTenantBufferPool::with_config(
            io,
            PerTenantBufferPoolConfig {
                frames_per_tenant: 8,
                write_fraction: 0.0,
            },
        ));
        let flush_store = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 16));
        let checkpointer = WriteBehindCheckpointer::new(
            Arc::new(DirtyPageTable::new()),
            flush_store.clone(),
            flush_store,
        )
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
        let allocator = Arc::new(PageAllocator::new());
        let allocator_seed = crud_allocator_seed_handle(Arc::clone(&store), allocator);
        let record_pages = store.records.as_ref().unwrap();
        let intern = InternTable::new();
        let idempotency = IdempotencyStore::new();
        let permissions = PermissionIndex::new();
        let snapshot = CheckpointSnapshot {
            txn: &txn,
            primary_pages: primary.page_store(),
            record_pages,
            blob: &store.blobs,
            allocator_seed: allocator_seed.as_ref(),
            intern: &intern,
            idempotency: &idempotency,
            permissions: &permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        let catalog = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
        let result = incremental_checkpoint(
            dir.path(),
            &catalog,
            &snapshot,
            &checkpointer,
            || (Vec::new(), store.deferred_v9_boundary()),
            |_| {
                store.drain_deferred_v9_applies().map_err(|error| {
                    CheckpointError::Io(std::io::Error::other(error.to_string()))
                })?;
                Ok(Lsn::new(20))
            },
        );

        // On the buggy pop/consume-before-apply order, the first error loses
        // the queue entry and proof. This retry then establishes at 20,
        // reclaims segment 0, and an anchored restart cannot recover commit
        // 10. Keep this branch as the strong RED-on-revert oracle.
        if let Ok(report) = result.as_ref() {
            let reclaimed = reclaim_segments_below(dir.path(), report.redo_lsn).unwrap();
            assert_eq!(reclaimed.deleted_segments, vec![0]);
            let recovered = Arc::new(TxnManager::new());
            recover_from_wal_encrypted_incremental(
                dir.path(),
                Arc::clone(&recovered),
                PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new())),
                None,
                None,
                report.checkpoint_lsn,
                report.redo_lsn,
            )
            .unwrap();
            assert_eq!(
                recovered.read_at(TenantId::DEFAULT, node_mvcc_key(node_id), Lsn::new(20)),
                Some(Bytes::copy_from_slice(
                    &NodeRecord::new(node_id, LabelId::new(7), Lsn::new(10)).to_bytes()
                )),
                "checkpoint reclaimed through a lost durable deferred commit"
            );
        }

        assert!(result.is_err(), "checkpoint advanced past the failed apply");
        assert!(read_latest_sidecar(dir.path()).unwrap().is_none());
        assert_eq!(
            store.deferred_v9_boundary(),
            Some(DeferredV9Boundary {
                commit_lsn: Lsn::new(10),
                redo_lsn: Lsn::new(9),
            })
        );
        assert!(wal.has_exact_durable(Lsn::new(10)));
        assert!(dir.path().join(segment_filename(0)).exists());
        assert!(dir.path().join(segment_filename(1)).exists());
        writer.shutdown().unwrap();
    }

    fn decode_node(bytes: &[u8]) -> NodeRecord {
        let arr: &[u8; NodeRecord::SIZE] = bytes.try_into().expect("node-record-sized payload");
        NodeRecord::from_bytes(arr).expect("valid NodeRecord bytes")
    }

    fn build_dual_write_store() -> (Arc<TxnManager>, CrudStore, Arc<PrimaryIndex>) {
        let txn_mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let primary =
            Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
        let store = CrudStore::new_with_index(None, Arc::clone(&primary), Arc::clone(&alloc));
        (txn_mgr, store, primary)
    }

    fn head_fingerprints(heads: Vec<(LabelId, Arc<TelBlock>, PageId)>) -> Vec<(u32, u64, u32)> {
        heads
            .into_iter()
            .map(|(ch, head, page)| (ch.raw(), page.raw(), head.entry_count()))
            .collect()
    }

    #[test]
    fn create_rel_commit_updates_max_out_degree_snapshot() {
        let (mgr, store, _primary) = build_dual_write_store();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(7);
        let rel_type = TypeId::new(3);

        let mut tx = mgr.begin(tenant);
        let hub = create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
        let other_src = create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
        let dsts = [
            create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap(),
            create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap(),
            create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap(),
            create_node(&store, &mut tx, tenant, label, &PropertyData::Empty).unwrap(),
        ];
        for dst in &dsts[..3] {
            create_rel(
                &store,
                &mut tx,
                tenant,
                hub,
                *dst,
                rel_type,
                &PropertyData::Empty,
            )
            .unwrap();
        }
        create_rel(
            &store,
            &mut tx,
            tenant,
            other_src,
            dsts[3],
            rel_type,
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();

        let snapshot = store.catalog_stats(tenant).unwrap().snapshot();
        let entries = snapshot.max_out_degree_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries
                .iter()
                .map(|entry| (entry.label, entry.rel_type, entry.vertex, entry.degree))
                .collect::<Vec<_>>(),
            vec![(label, rel_type, hub, 3), (label, rel_type, other_src, 1)]
        );
    }

    fn full_scan_tel_heads_for_src(
        store: &CrudStore,
        tenant: TenantId,
        src: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        let mut out: Vec<_> = store
            .tel_chains
            .iter()
            .filter_map(|e| {
                let (t, s, ch) = *e.key();
                if t == tenant && s == src {
                    let g = e.value().lock();
                    Some((ch, Arc::clone(&g.head), g.head_page))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by_key(|(ch, _, _)| ch.raw());
        out
    }

    fn full_scan_reverse_tel_heads_for_dst(
        store: &CrudStore,
        tenant: TenantId,
        dst: NodeId,
    ) -> Vec<(LabelId, Arc<TelBlock>, PageId)> {
        let mut out: Vec<_> = store
            .reverse_tel_chains
            .iter()
            .filter_map(|e| {
                let (t, d, ch) = *e.key();
                if t == tenant && d == dst {
                    let g = e.value().lock();
                    Some((ch, Arc::clone(&g.head), g.head_page))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by_key(|(ch, _, _)| ch.raw());
        out
    }

    #[test]
    fn dual_write_populates_mvcc_and_index_and_page() {
        let (mgr, store, primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(42),
            &PropertyData::InlineU32Pair(7, 8),
        )
        .unwrap();
        let commit_lsn = commit(tx, &store).unwrap();
        assert!(commit_lsn.raw() > 0);

        // Primary index now carries (TenantId::DEFAULT, Node, id).
        let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
        let slot = primary.lookup(key).unwrap().expect("index hit");

        // Record page has the node at that slot, with created_lsn fixed up.
        let records = store.records().expect("dual-write configured");
        let latch = records.latch(slot.page).unwrap();
        let g = latch.read();
        let page = crate::records::SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
        let rec = page.read_node(slot.slot).unwrap().expect("live slot");
        assert_eq!(rec.id, id.raw());
        assert_eq!(rec.label_id, 42);
        assert_eq!(
            rec.created_lsn,
            commit_lsn.raw(),
            "created_lsn must be fixed up to commit_lsn"
        );
    }

    #[test]
    fn dual_write_disabled_constructor_is_index_free() {
        // Existing M2.c behavior: no index, no record store.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let _id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();
        assert!(store.primary().is_none());
        assert!(store.records().is_none());
        assert!(store.allocator().is_none());
    }

    #[test]
    fn read_node_with_store_takes_index_fast_path_after_commit() {
        let (mgr, store, _primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(5, 9),
        )
        .unwrap();
        let commit_lsn = commit(tx, &store).unwrap();

        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node_with_store(&store, &reader, id).unwrap().unwrap();
        assert_eq!(rec.id, id.raw());
        // Fast-path `created_lsn` is the page-materialized commit LSN,
        // not the MVCC `Lsn::ZERO` placeholder.
        assert_eq!(rec.created_lsn, commit_lsn.raw());
    }

    #[test]
    fn read_node_with_store_falls_back_to_mvcc_on_index_miss() {
        // A CrudStore without an index dispatches straight to the MVCC
        // path even when called through the fast-path API.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(3, 3),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node_with_store(&store, &reader, id).unwrap().unwrap();
        assert_eq!(rec.id, id.raw());
    }

    #[test]
    fn read_with_store_respects_snapshot_isolation_on_update() {
        // Regression for ADR-023 §Consequences: the fast path is
        // authoritative only when the slot's `created_lsn` is `≤
        // tx.snapshot()`. An update rewrites the slot in place, so a
        // pre-update snapshot reader must NOT see the post-update bytes
        // through the fast path — it should fall through to MVCC.
        let (mgr, store, _primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(11, 22),
        )
        .unwrap();
        commit(tx, &store).unwrap();

        // Reader captures a snapshot BEFORE the update commits.
        let pre_update_reader = mgr.begin(TenantId::DEFAULT);

        let mut tx = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut tx, id, &PropertyData::InlineU32Pair(999, 888)).unwrap();
        commit(tx, &store).unwrap();

        let rec = read_node_with_store(&store, &pre_update_reader, id)
            .unwrap()
            .unwrap();
        assert_eq!(
            rec.inline_u32a, 11,
            "pre-update snapshot reader must see OLD value via MVCC fallback"
        );
        assert_eq!(rec.inline_u32b, 22);

        // A fresh reader at the post-update snapshot still sees the new value.
        let post_update_reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node_with_store(&store, &post_update_reader, id)
            .unwrap()
            .unwrap();
        assert_eq!(rec.inline_u32a, 999);
        assert_eq!(rec.inline_u32b, 888);
    }

    #[test]
    fn dual_write_update_rewrites_slot_in_place() {
        let (mgr, store, primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
        let before_slot = primary.lookup(key).unwrap().unwrap();

        let mut tx = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut tx, id, &PropertyData::InlineU32Pair(99, 100)).unwrap();
        commit(tx, &store).unwrap();

        // Primary-index slot coordinates must NOT change under a pure
        // property update — update_node rewrites in place.
        let after_slot = primary.lookup(key).unwrap().unwrap();
        assert_eq!(before_slot, after_slot);

        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node_with_store(&store, &reader, id).unwrap().unwrap();
        assert_eq!(rec.inline_u32a, 99);
        assert_eq!(rec.inline_u32b, 100);
    }

    #[test]
    fn dual_write_delete_tombstones_page_and_index() {
        let (mgr, store, primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();

        let mut tx = mgr.begin(TenantId::DEFAULT);
        delete_node_with_store(&store, &mut tx, id).unwrap();
        commit(tx, &store).unwrap();

        // Index marks the key tombstoned (lookup returns None).
        let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
        assert_eq!(primary.lookup(key).unwrap(), None);
        // Fast-path read falls back to MVCC which also sees the tombstone.
        let reader = mgr.begin(TenantId::DEFAULT);
        assert!(read_node_with_store(&store, &reader, id).unwrap().is_none());
    }

    #[test]
    fn bootstrap_primary_index_walks_rotated_pages() {
        // Regression for the review's C-B finding. Creating more than
        // `NODE_CAPACITY` nodes forces at least one `rotate_open_page`
        // call, which evicts the older full page from `open_pages`.
        // `bootstrap_primary_index` must still observe that page
        // through `RecordPageStore::iter_pages`, not just whatever
        // page is currently "open" per (tenant, kind).
        let (mgr, store, _primary) = build_dual_write_store();
        // NODE_CAPACITY ≈ 121 at 8 KiB pages; request double so at
        // least one rotation is guaranteed.
        let target_count = (NODE_CAPACITY as u32) * 2 + 3;
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..target_count {
            create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(i),
                &PropertyData::Empty,
            )
            .unwrap();
        }
        commit(tx, &store).unwrap();

        // All rows are already indexed by the dual-write drain, so a
        // correct bootstrap visits all of them and skips all of them.
        let reader = mgr.begin(TenantId::DEFAULT);
        let stats = store.bootstrap_primary_index(&reader).unwrap();
        assert_eq!(
            stats.skipped, target_count as usize,
            "bootstrap must visit every installed record, including those on rotated-out pages"
        );
        assert_eq!(stats.indexed, 0);
    }

    #[test]
    fn bootstrap_primary_index_is_idempotent() {
        let (mgr, store, primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..5u32 {
            create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(i),
                &PropertyData::Empty,
            )
            .unwrap();
        }
        commit(tx, &store).unwrap();

        // Entries are already indexed (normal dual-write drain);
        // bootstrap should skip them.
        let reader = mgr.begin(TenantId::DEFAULT);
        let stats = store.bootstrap_primary_index(&reader).unwrap();
        assert_eq!(stats.indexed, 0, "already-indexed rows skipped");
        assert!(stats.skipped > 0, "skipped count = # of rows seen");
        let _ = primary; // kept alive for clarity
    }

    #[test]
    fn dual_write_packs_multiple_nodes_on_one_page() {
        // NODE_CAPACITY is large (~121/page at 8 KiB); 10 creates in
        // one txn should land on the same page id.
        let (mgr, store, primary) = build_dual_write_store();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let mut ids = Vec::new();
        for i in 0..10u32 {
            ids.push(
                create_node(
                    &store,
                    &mut tx,
                    TenantId::DEFAULT,
                    LabelId::new(i),
                    &PropertyData::InlineU32Pair(i, i ^ 0xAA),
                )
                .unwrap(),
            );
        }
        commit(tx, &store).unwrap();

        let mut seen_pages = std::collections::HashSet::new();
        for id in ids {
            let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
            let slot = primary.lookup(key).unwrap().unwrap();
            seen_pages.insert(slot.page);
        }
        assert_eq!(
            seen_pages.len(),
            1,
            "10 nodes must share one page (NODE_CAPACITY well above 10)"
        );
    }

    #[test]
    fn allocates_monotonic_ids_per_tenant() {
        let store = CrudStore::new();
        let a = store.alloc_node(TenantId::DEFAULT).unwrap();
        let b = store.alloc_node(TenantId::DEFAULT).unwrap();
        let c = store.alloc_node(TenantId::DEFAULT).unwrap();
        assert_eq!(a.raw(), 1);
        assert_eq!(b.raw(), 2);
        assert_eq!(c.raw(), 3);
    }

    #[test]
    fn alloc_is_per_tenant() {
        let store = CrudStore::new();
        let t_a = TenantId::new(100);
        let t_b = TenantId::new(101);
        assert_eq!(store.alloc_node(t_a).unwrap().raw(), 1);
        assert_eq!(store.alloc_node(t_b).unwrap().raw(), 1);
        assert_eq!(store.alloc_node(t_a).unwrap().raw(), 2);
        assert_eq!(store.alloc_node(t_b).unwrap().raw(), 2);
    }

    #[test]
    fn create_node_roundtrips_via_mvcc() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::InlineU32Pair(11, 22),
        )
        .unwrap();
        // Read-your-writes: visible to the same txn before commit.
        let bytes = tx.read(node_mvcc_key(id)).expect("RYW read");
        let rec = decode_node(&bytes);
        assert_eq!(rec.id, id.raw());
        assert_eq!(rec.label_id, 7);
        assert_eq!(rec.inline_u32a, 11);
        assert_eq!(rec.inline_u32b, 22);
        assert!(!rec.is_deleted());

        let commit_lsn = tx.commit().unwrap();
        // Visible to a fresh reader at the committed snapshot.
        let reader = mgr.begin(TenantId::DEFAULT);
        assert!(reader.snapshot().raw() >= commit_lsn.raw());
        let back = reader.read(node_mvcc_key(id)).expect("post-commit read");
        assert_eq!(back, bytes);
    }

    #[test]
    fn create_node_empty_props_zero_inline_fields() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let bytes = tx.read(node_mvcc_key(id)).unwrap();
        let rec = decode_node(&bytes);
        assert_eq!(rec.property_ref, 0);
        assert_eq!(rec.inline_u32a, 0);
        assert_eq!(rec.inline_u32b, 0);
        assert_eq!(
            rec.flags & arcgraph_core::record::node_flags::HAS_EXTENDED,
            0
        );
    }

    #[test]
    fn create_node_blob_props_accepted_via_overflow() {
        // Small blob path — M2-31 routes PropertyData::Blob through
        // BlobStore and sets the overflow bit on property_ref.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Blob(vec![1, 2, 3]),
        )
        .unwrap();
        tx.commit().unwrap();
        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node(&reader, id).unwrap().unwrap();
        assert_ne!(
            rec.flags & arcgraph_core::record::node_flags::HAS_EXTENDED,
            0
        );
    }

    #[test]
    fn create_node_tenant_isolated() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let t_a = TenantId::new(100);
        let t_b = TenantId::new(101);

        let mut tx_a = mgr.begin(t_a);
        let id_a = create_node(
            &store,
            &mut tx_a,
            t_a,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx_a.commit().unwrap();

        let mut tx_b = mgr.begin(t_b);
        let id_b = create_node(
            &store,
            &mut tx_b,
            t_b,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx_b.commit().unwrap();

        // Both got id 1, but the (tenant, key) tuples in the MVCC map
        // are disjoint.
        assert_eq!(id_a.raw(), 1);
        assert_eq!(id_b.raw(), 1);

        // Tenant A cannot observe tenant B's node, and vice versa.
        let reader_a = mgr.begin(t_a);
        let reader_b = mgr.begin(t_b);
        let read_a = reader_a.read(node_mvcc_key(id_a)).unwrap();
        let read_b = reader_b.read(node_mvcc_key(id_b)).unwrap();
        // Both records encode id=1 in their bytes, so the raw payload
        // is equal — the isolation claim is about the (tenant, key)
        // chain, which is tested by chain_len below.
        assert_eq!(decode_node(&read_a).id, 1);
        assert_eq!(decode_node(&read_b).id, 1);
        assert_eq!(mgr.chain_len(t_a, node_mvcc_key(id_a)), 1);
        assert_eq!(mgr.chain_len(t_b, node_mvcc_key(id_b)), 1);
    }

    #[test]
    fn snapshot_before_create_does_not_see_node() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        // Reader begins before any write.
        let reader = mgr.begin(TenantId::DEFAULT);
        let mut writer = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut writer,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        writer.commit().unwrap();
        assert!(reader.read(node_mvcc_key(id)).is_none());
    }

    fn decode_rel(bytes: &[u8]) -> RelRecord {
        let arr: &[u8; RelRecord::SIZE] = bytes.try_into().expect("rel-record-sized payload");
        RelRecord::from_bytes(arr).expect("valid RelRecord bytes")
    }

    #[test]
    fn create_rel_roundtrips_via_mvcc_and_tel() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let src = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let rel = create_rel(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            src,
            dst,
            TypeId::new(9),
            &PropertyData::InlineU32Pair(3, 4),
        )
        .unwrap();
        // RYW: rel bytes visible on the same txn.
        let ryw = tx.read(rel_mvcc_key(rel)).expect("RYW rel read");
        let rec = decode_rel(&ryw);
        assert_eq!(rec.id, rel.raw());
        assert_eq!(rec.type_id, 9);
        assert_eq!(rec.src_id, src.raw());
        assert_eq!(rec.dst_id, dst.raw());
        assert_eq!(rec.inline_u32a, 3);
        assert_eq!(rec.inline_u32b, 4);
        // TEL chain not yet populated — append is deferred to commit.
        assert!(
            store
                .tel_head(TenantId::DEFAULT, src, LabelId::new(9))
                .is_none()
        );
        let commit_lsn = commit(tx, &store).unwrap();
        // TEL entry landed with commit_lsn.
        let (_page, head) = store
            .tel_head(TenantId::DEFAULT, src, LabelId::new(9))
            .expect("chain exists post-commit");
        assert_eq!(head.entry_count(), 1);
        let entry = head.entry_at(0).unwrap();
        assert_eq!(entry.dst_id, dst.raw());
        assert_eq!(entry.rel_id, rel.raw());
        assert_eq!(entry.created_lsn, commit_lsn.raw());
        // Rel key is tagged; node key is not — disjoint spaces.
        assert_ne!(rel_mvcc_key(rel), node_mvcc_key(src));
        assert_eq!(rel_mvcc_key(rel) & REL_TAG_BIT, REL_TAG_BIT);
    }

    #[test]
    fn tel_append_grows_across_block_boundary() {
        // MIN_BLOCK_BYTES = 64 = 32 header + 1 entry slot.
        // First append fills head (1/1); second triggers grown()
        // doubling to capacity 3, etc. Push enough entries to force
        // at least two growth events and confirm head_page updates.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let src = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        // 10 rels on the same channel → forces multiple growth steps
        // from MIN_BLOCK_BYTES.
        let mut rels = Vec::new();
        for _ in 0..10 {
            let r = create_rel(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                src,
                dst,
                TypeId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            rels.push(r);
        }
        let commit_lsn = commit(tx, &store).unwrap();
        let (head_page, head) = store
            .tel_head(TenantId::DEFAULT, src, LabelId::new(1))
            .unwrap();
        // All 10 entries landed in the head after growth (no
        // overflow: 10 entries × 32 B + 32 B header = 352 B ≪ MAX).
        assert_eq!(head.entry_count(), 10);
        assert!(head.block_size() > MIN_BLOCK_BYTES);
        // head_page must be some later synthetic id than the first
        // block we ever allocated.
        assert!(head_page.raw() >= 1);
        // Entries are ordered by append; all carry the same commit_lsn.
        for (i, r) in rels.iter().enumerate() {
            let e = head.entry_at(i as u32).unwrap();
            assert_eq!(e.rel_id, r.raw());
            assert_eq!(e.created_lsn, commit_lsn.raw());
        }
    }

    /// Regression (#27): concurrent first-time `tel_chain_for` on the
    /// same `(tenant, src, channel)` must leave exactly one block in
    /// `tel_blocks`, not one per thread. Before the fix every
    /// participant published its candidate block *before* the
    /// `or_insert_with` winner was decided, leaking the losers'
    /// blocks for the life of the process.
    #[test]
    fn tel_chain_for_losers_drop_orphan_blocks() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 16;

        let store = Arc::new(CrudStore::new());
        let tenant = TenantId::new(7);
        let src = NodeId::new(42);
        let channel = LabelId::new(99);

        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                // Align all threads on the first-time slow path so
                // the race window is exercised.
                barrier.wait();
                store.tel_chain_for(tenant, src, channel).unwrap()
            }));
        }
        let chains: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads observe the same chain (Arc equality via
        // pointer identity of the Mutex).
        let first = Arc::as_ptr(&chains[0]);
        for c in &chains[1..] {
            assert!(
                std::ptr::eq(Arc::as_ptr(c), first),
                "all concurrent callers must resolve to the same TelChain instance"
            );
        }

        // Exactly one block registered under the tenant — no
        // orphans from losing initializers.
        assert_eq!(
            store.tel_blocks_len_for_tenant(tenant),
            1,
            "expected 1 block for the chain head; got orphan(s) from the race"
        );
    }

    #[test]
    fn tel_chains_are_tenant_and_channel_keyed() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let t_a = TenantId::new(100);
        let t_b = TenantId::new(101);

        let mut tx_a = mgr.begin(t_a);
        let s_a = create_node(
            &store,
            &mut tx_a,
            t_a,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let d_a = create_node(
            &store,
            &mut tx_a,
            t_a,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let _r_a = create_rel(
            &store,
            &mut tx_a,
            t_a,
            s_a,
            d_a,
            TypeId::new(7),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx_a, &store).unwrap();

        let mut tx_b = mgr.begin(t_b);
        let s_b = create_node(
            &store,
            &mut tx_b,
            t_b,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let d_b = create_node(
            &store,
            &mut tx_b,
            t_b,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let _r_b = create_rel(
            &store,
            &mut tx_b,
            t_b,
            s_b,
            d_b,
            TypeId::new(7),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx_b, &store).unwrap();

        // Even with identical src id (1) + channel (7), chains are
        // distinct per tenant.
        let (_, head_a) = store.tel_head(t_a, s_a, LabelId::new(7)).unwrap();
        let (_, head_b) = store.tel_head(t_b, s_b, LabelId::new(7)).unwrap();
        assert_eq!(head_a.entry_count(), 1);
        assert_eq!(head_b.entry_count(), 1);
        // Cross-tenant chains are distinct Arcs even though NodeId
        // raw values collide (both tenants start at 1).
        let (page_a, _) = store.tel_head(t_a, s_a, LabelId::new(7)).unwrap();
        let (page_b, _) = store.tel_head(t_b, s_b, LabelId::new(7)).unwrap();
        assert_ne!(page_a, page_b);
        // Different channel on the same tenant+src is also distinct.
        assert!(store.tel_head(t_a, s_a, LabelId::new(8)).is_none());
    }

    #[test]
    fn tel_channel_indexes_match_full_scan_oracle() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenants = [TenantId::new(21), TenantId::new(22)];

        for tenant in tenants {
            let mut tx = mgr.begin(tenant);
            let src_a = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let src_b = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let dst_a = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let dst_b = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();

            for ty in [
                TypeId::new(17),
                TypeId::new(3),
                TypeId::new(9),
                TypeId::new(3),
            ] {
                create_rel(
                    &store,
                    &mut tx,
                    tenant,
                    src_a,
                    dst_a,
                    ty,
                    &PropertyData::Empty,
                )
                .unwrap();
            }
            for ty in [TypeId::new(5), TypeId::new(2)] {
                create_rel(
                    &store,
                    &mut tx,
                    tenant,
                    src_b,
                    dst_b,
                    ty,
                    &PropertyData::Empty,
                )
                .unwrap();
            }
            create_rel(
                &store,
                &mut tx,
                tenant,
                src_a,
                dst_b,
                TypeId::new(21),
                &PropertyData::Empty,
            )
            .unwrap();
            commit(tx, &store).unwrap();

            for src in [src_a, src_b] {
                assert_eq!(
                    head_fingerprints(store.tel_heads_for_src(tenant, src)),
                    head_fingerprints(full_scan_tel_heads_for_src(&store, tenant, src)),
                    "forward index must match the old full-scan result for tenant/src"
                );
            }
            for dst in [dst_a, dst_b] {
                assert_eq!(
                    head_fingerprints(store.reverse_tel_heads_for_dst(tenant, dst)),
                    head_fingerprints(full_scan_reverse_tel_heads_for_dst(&store, tenant, dst)),
                    "reverse index must match the old full-scan result for tenant/dst"
                );
            }
        }
    }

    #[test]
    fn recovered_rels_populate_untyped_tel_channel_indexes_in_sorted_order() {
        let store = CrudStore::new();
        let tenant = TenantId::new(33);
        let src = NodeId::new(700);
        let dst = NodeId::new(800);
        let commit_lsn = Lsn::new(99);

        for (rel, ty) in [
            (RelId::new(1), TypeId::new(30)),
            (RelId::new(2), TypeId::new(10)),
            (RelId::new(3), TypeId::new(20)),
        ] {
            let rec = RelRecord::new(rel, ty, src, dst, Lsn::ZERO);
            store
                .reinstate_rel_adjacency(tenant, &rec, commit_lsn)
                .unwrap();
        }

        let forward_channels: Vec<_> = store
            .tel_heads_for_src(tenant, src)
            .into_iter()
            .map(|(ch, _, _)| ch.raw())
            .collect();
        assert_eq!(forward_channels, vec![10, 20, 30]);
        assert_eq!(
            forward_channels,
            full_scan_tel_heads_for_src(&store, tenant, src)
                .into_iter()
                .map(|(ch, _, _)| ch.raw())
                .collect::<Vec<_>>(),
            "recovered forward channels must be visible to untyped snapshots"
        );

        let reverse_channels: Vec<_> = store
            .reverse_tel_heads_for_dst(tenant, dst)
            .into_iter()
            .map(|(ch, _, _)| ch.raw())
            .collect();
        assert_eq!(reverse_channels, vec![10, 20, 30]);
        assert_eq!(
            reverse_channels,
            full_scan_reverse_tel_heads_for_dst(&store, tenant, dst)
                .into_iter()
                .map(|(ch, _, _)| ch.raw())
                .collect::<Vec<_>>(),
            "recovered reverse channels must be visible to untyped snapshots"
        );
    }

    #[test]
    fn aborted_rel_leaves_no_tel_entry() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let src = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let _rel = create_rel(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            src,
            dst,
            TypeId::new(3),
            &PropertyData::Empty,
        )
        .unwrap();
        let txn_id = tx.id();
        // User chooses to abort: they MUST discard pending before
        // Transaction::abort, per crud::commit docs.
        store.discard_pending(txn_id);
        tx.abort();
        // No TEL chain ever materialized.
        assert!(
            store
                .tel_head(TenantId::DEFAULT, src, LabelId::new(3))
                .is_none()
        );
    }

    #[test]
    fn read_node_at_current_snapshot_returns_latest() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(5),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .unwrap();
        tx.commit().unwrap();
        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node(&reader, id).unwrap().expect("node visible");
        assert_eq!(rec.id, id.raw());
        assert_eq!(rec.label_id, 5);
        assert_eq!(rec.inline_u32a, 1);
        assert_eq!(rec.inline_u32b, 2);
    }

    #[test]
    fn read_node_pre_create_snapshot_returns_none() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        // Snapshot captured before the writer exists.
        let reader = mgr.begin(TenantId::DEFAULT);
        let mut writer = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut writer,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        writer.commit().unwrap();
        assert!(read_node(&reader, id).unwrap().is_none());
    }

    #[test]
    fn read_node_after_tombstone_returns_none() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx.commit().unwrap();
        // Tombstone via Transaction::delete (public MVCC surface).
        let mut del = mgr.begin(TenantId::DEFAULT);
        del.delete(node_mvcc_key(id));
        del.commit().unwrap();
        let reader = mgr.begin(TenantId::DEFAULT);
        assert!(read_node(&reader, id).unwrap().is_none());
    }

    #[test]
    fn read_node_across_version_chain_returns_latest_visible() {
        // Write v1, capture a snapshot, then write v2. Old snapshot
        // sees v1; fresh snapshot sees v2. Confirms read_node follows
        // the MVCC visibility rule, not the chain head.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx1 = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx1,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(10, 20),
        )
        .unwrap();
        tx1.commit().unwrap();

        let mid_reader = mgr.begin(TenantId::DEFAULT);

        // Overwrite with a new version (property update via raw write).
        let mut tx2 = mgr.begin(TenantId::DEFAULT);
        let mut new_rec = NodeRecord::new(id, LabelId::new(1), Lsn::ZERO);
        PropertyData::InlineU32Pair(30, 40)
            .apply_to_node(&mut new_rec, TenantId::DEFAULT, &store.blobs, tx2.id())
            .unwrap();
        tx2.write(
            node_mvcc_key(id),
            Bytes::copy_from_slice(&new_rec.to_bytes()),
        );
        tx2.commit().unwrap();

        // mid_reader's snapshot predates tx2's commit → sees v1.
        let r1 = read_node(&mid_reader, id).unwrap().unwrap();
        assert_eq!(r1.inline_u32a, 10);
        assert_eq!(r1.inline_u32b, 20);

        // Fresh reader sees v2.
        let fresh = mgr.begin(TenantId::DEFAULT);
        let r2 = read_node(&fresh, id).unwrap().unwrap();
        assert_eq!(r2.inline_u32a, 30);
        assert_eq!(r2.inline_u32b, 40);
    }

    /// Helper: create `src` + `n` distinct dst nodes + one rel per dst
    /// with `ty`, commit, return (src, rels).
    fn setup_rels(
        mgr: &TxnManager,
        store: &CrudStore,
        tenant: TenantId,
        ty: TypeId,
        n: usize,
    ) -> (NodeId, Vec<RelId>) {
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let mut rels = Vec::with_capacity(n);
        for _ in 0..n {
            let dst = create_node(
                store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let r = create_rel(store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
            rels.push(r);
        }
        commit(tx, store).unwrap();
        (src, rels)
    }

    #[test]
    fn scan_out_yields_all_entries_for_src_union() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        // Three chains under the same src: types 10, 20, 30, with
        // 4, 3, 5 rels respectively. Build everything in one txn so
        // the (src, dst_i) pairs share a src.
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let mut expected_rels: Vec<RelId> = Vec::new();
        for &(ty, count) in &[(10u32, 4usize), (20, 3), (30, 5)] {
            for _ in 0..count {
                let dst = create_node(
                    &store,
                    &mut tx,
                    tenant,
                    LabelId::new(1),
                    &PropertyData::Empty,
                )
                .unwrap();
                let r = create_rel(
                    &store,
                    &mut tx,
                    tenant,
                    src,
                    dst,
                    TypeId::new(ty),
                    &PropertyData::Empty,
                )
                .unwrap();
                expected_rels.push(r);
            }
        }
        commit(tx, &store).unwrap();

        let reader = mgr.begin(tenant);
        let out: Vec<TelEntry> = scan_out(&store, &reader, src, None).collect();
        assert_eq!(out.len(), 4 + 3 + 5);
        let mut rels_seen: Vec<u64> = out.iter().map(|e| e.rel_id).collect();
        rels_seen.sort_unstable();
        let mut expected_ids: Vec<u64> = expected_rels.iter().map(|r| r.raw()).collect();
        expected_ids.sort_unstable();
        assert_eq!(rels_seen, expected_ids);
    }

    #[test]
    fn scan_out_type_filter_narrows_to_single_chain() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let mut rels_10: Vec<RelId> = Vec::new();
        for _ in 0..3 {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            rels_10.push(
                create_rel(
                    &store,
                    &mut tx,
                    tenant,
                    src,
                    dst,
                    TypeId::new(10),
                    &PropertyData::Empty,
                )
                .unwrap(),
            );
        }
        // Different type on same src — must NOT appear in the filtered scan.
        for _ in 0..4 {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            create_rel(
                &store,
                &mut tx,
                tenant,
                src,
                dst,
                TypeId::new(99),
                &PropertyData::Empty,
            )
            .unwrap();
        }
        commit(tx, &store).unwrap();

        let reader = mgr.begin(tenant);
        let out: Vec<TelEntry> = scan_out(&store, &reader, src, Some(TypeId::new(10))).collect();
        assert_eq!(out.len(), 3);
        let mut seen: Vec<u64> = out.iter().map(|e| e.rel_id).collect();
        seen.sort_unstable();
        let mut expect: Vec<u64> = rels_10.iter().map(|r| r.raw()).collect();
        expect.sort_unstable();
        assert_eq!(seen, expect);

        // Absent type yields empty.
        let empty: Vec<TelEntry> =
            scan_out(&store, &reader, src, Some(TypeId::new(1234))).collect();
        assert!(empty.is_empty());
    }

    #[test]
    fn scan_out_respects_snapshot_visibility() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(7);

        // Batch 1: 5 rels, committed.
        let (src, first) = setup_rels(&mgr, &store, tenant, ty, 5);

        // Capture a snapshot after batch 1.
        let mid_reader = mgr.begin(tenant);

        // Batch 2: 5 more rels on the same src, committed after mid_reader.
        let mut tx = mgr.begin(tenant);
        for _ in 0..5 {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        }
        commit(tx, &store).unwrap();

        let mid_out: Vec<TelEntry> = scan_out(&store, &mid_reader, src, Some(ty)).collect();
        assert_eq!(mid_out.len(), 5);
        let mut seen: Vec<u64> = mid_out.iter().map(|e| e.rel_id).collect();
        seen.sort_unstable();
        let mut expect: Vec<u64> = first.iter().map(|r| r.raw()).collect();
        expect.sort_unstable();
        assert_eq!(seen, expect);

        let fresh = mgr.begin(tenant);
        let all: Vec<TelEntry> = scan_out(&store, &fresh, src, Some(ty)).collect();
        assert_eq!(all.len(), 10);
    }

    /// Closes tracking issue #21: verify scan walks the overflow chain
    /// link after MAX_BLOCK_BYTES is reached.
    #[test]
    fn scan_out_walks_overflow_chain() {
        use crate::tel::MAX_ENTRIES;
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        // MAX_ENTRIES=2047 fills a MAX_BLOCK_BYTES block; one more
        // forces the overflow-link path in tel_append.
        let n = MAX_ENTRIES as usize + 1;
        let (src, rels) = setup_rels(&mgr, &store, tenant, ty, n);

        // The head chain must now point at a fresh MIN-sized block
        // whose prev_block_ptr links the full old head.
        let (_head_page, head) = store.tel_head(tenant, src, LabelId::new(1)).unwrap();
        assert_eq!(head.entry_count(), 1, "overflow head holds the one extra");
        let prev_pid = head.prev_block_ptr().expect("overflow link must be set");
        let prev = store
            .tel_block(tenant, prev_pid)
            .expect("predecessor in store");
        assert_eq!(prev.entry_count(), MAX_ENTRIES, "predecessor full");

        let reader = mgr.begin(tenant);
        let out: Vec<TelEntry> = scan_out(&store, &reader, src, Some(ty)).collect();
        assert_eq!(out.len(), n);
        let mut seen: Vec<u64> = out.iter().map(|e| e.rel_id).collect();
        seen.sort_unstable();
        let mut expect: Vec<u64> = rels.iter().map(|r| r.raw()).collect();
        expect.sort_unstable();
        assert_eq!(seen, expect);
    }

    #[test]
    fn scan_out_fetches_overflow_predecessors_lazily() {
        use crate::tel::MAX_ENTRIES;
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        // Two full MAX-sized predecessors plus a one-entry head gives a
        // >=3-block chain. The cursor must not fetch either predecessor
        // until iteration crosses the current block boundary.
        let n = (MAX_ENTRIES as usize * 2) + 1;
        let (src, _rels) = setup_rels(&mgr, &store, tenant, ty, n);
        let (_head_page, head) = store.tel_head(tenant, src, LabelId::new(1)).unwrap();
        let head_entries = head.entry_count() as usize;
        assert_eq!(head_entries, 1, "fixture should leave a one-entry head");

        let reader = mgr.begin(tenant);
        store.reset_tel_block_fetches_for_test();
        let mut cursor = scan_out_cursor(&store, &reader, src, Some(ty));

        for _ in 0..head_entries {
            assert!(cursor.next_entry(&store, &reader).is_some());
        }
        assert_eq!(
            store.tel_block_fetches_for_test(),
            0,
            "constructing and draining the head block must not fetch predecessors"
        );

        assert!(
            cursor.next_entry(&store, &reader).is_some(),
            "crossing into predecessor yields"
        );
        assert_eq!(
            store.tel_block_fetches_for_test(),
            1,
            "first predecessor is fetched only when the cursor crosses the head boundary"
        );
    }

    #[test]
    fn scan_out_cursor_snapshot_ignores_mid_iteration_appends_and_new_channels() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        let (src, first_rels) = setup_rels(&mgr, &store, tenant, ty, 3);
        let reader = mgr.begin(tenant);
        let mut cursor = scan_out_cursor(&store, &reader, src, None);

        let first = cursor.next_entry(&store, &reader).expect("prefix row");
        assert!(
            first_rels.iter().any(|r| r.raw() == first.rel_id),
            "prefix row comes from the pre-snapshot fixture"
        );

        let mut writer = mgr.begin(tenant);
        let dst_same_channel = create_node(
            &store,
            &mut writer,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let appended_same_channel = create_rel(
            &store,
            &mut writer,
            tenant,
            src,
            dst_same_channel,
            ty,
            &PropertyData::Empty,
        )
        .unwrap();
        let dst_new_channel = create_node(
            &store,
            &mut writer,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let appended_new_channel = create_rel(
            &store,
            &mut writer,
            tenant,
            src,
            dst_new_channel,
            TypeId::new(2),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(writer, &store).unwrap();

        let mut rest = Vec::new();
        while let Some(entry) = cursor.next_entry(&store, &reader) {
            rest.push(entry);
        }
        let mut seen: Vec<u64> = std::iter::once(first.rel_id)
            .chain(rest.iter().map(|e| e.rel_id))
            .collect();
        seen.sort_unstable();
        let mut expected: Vec<u64> = first_rels.iter().map(|r| r.raw()).collect();
        expected.sort_unstable();
        assert_eq!(
            seen, expected,
            "cursor snapshot must ignore same-head appends and channels created after construction"
        );
        assert!(!seen.contains(&appended_same_channel.raw()));
        assert!(!seen.contains(&appended_new_channel.raw()));

        let fresh = mgr.begin(tenant);
        let fresh_seen: Vec<u64> = scan_out(&store, &fresh, src, None)
            .map(|e| e.rel_id)
            .collect();
        assert!(fresh_seen.contains(&appended_same_channel.raw()));
        assert!(fresh_seen.contains(&appended_new_channel.raw()));
    }

    #[test]
    fn scan_out_reader_transaction_pins_gc_anchor_until_drop() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);
        let (src, _rels) = setup_rels(&mgr, &store, tenant, ty, 2);

        let reader = mgr.begin(tenant);
        let cursor_snapshot = reader.snapshot();
        let mut cursor = scan_out_cursor(&store, &reader, src, Some(ty));
        assert!(
            cursor.next_entry(&store, &reader).is_some(),
            "cursor owns a visible prefix"
        );

        for _ in 0..3 {
            let mut writer = mgr.begin(tenant);
            let dst = create_node(
                &store,
                &mut writer,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            create_rel(
                &store,
                &mut writer,
                tenant,
                src,
                dst,
                ty,
                &PropertyData::Empty,
            )
            .unwrap();
            commit(writer, &store).unwrap();
        }

        let pinned = mgr.gc();
        assert!(
            pinned.anchor <= cursor_snapshot,
            "active scan transaction must pin GC anchor: stats={pinned:?}, snapshot={cursor_snapshot:?}"
        );

        drop(cursor);
        drop(reader);
        let released = mgr.gc();
        assert!(
            released.anchor > cursor_snapshot,
            "after cursor transaction drops, GC anchor advances: stats={released:?}, snapshot={cursor_snapshot:?}"
        );
    }

    // ── P0 #812 — supernode fan-out past the overflow boundary ────────
    //
    // The sibling `scan_out_walks_overflow_chain` stops at MAX_ENTRIES+1
    // (= 2048), the exact boundary where the FIRST overflow block holds
    // its single entry and still links the full predecessor. The bug
    // lives one insert LATER: the MIN-sized overflow head (capacity 1)
    // grows on the 2049-th insert, and pre-fix `grown()` returned a
    // replacement with `prev_block_ptr = NO_PREV_BLOCK`, orphaning the
    // 2047-entry predecessor. `scan_out` then walked only the new head
    // → silent loss of ~2047 edges with `inserted_count` unchanged.
    // These tests fan out FAR past the boundary (5000 = ~2.4 blocks) so
    // the fix is exercised across MULTIPLE overflow→grow cycles.

    /// Number of edges in the supernode tests. 5000 > 2 × MAX_ENTRIES
    /// (4094) so the chain spans three blocks and exercises two distinct
    /// overflow events plus every intervening grow-after-overflow.
    const SUPERNODE_FANOUT: usize = 5000;

    /// AC-1 (forward): a HUB with `SUPERNODE_FANOUT` out-edges of one
    /// type must have EVERY edge queryable via `scan_out`. Strong oracle:
    /// the exact set of rel ids AND dst ids round-trips — not merely the
    /// count. RED pre-fix: `scan_out` collapses to ~906 (only the
    /// still-growing newest block survives).
    #[test]
    fn supernode_forward_fanout_beyond_cap_all_queryable() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        let mut tx = mgr.begin(tenant);
        let hub = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        // Map rel_id -> dst so the oracle can verify topology, not just
        // cardinality.
        let mut expected: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
        for _ in 0..SUPERNODE_FANOUT {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let r =
                create_rel(&store, &mut tx, tenant, hub, dst, ty, &PropertyData::Empty).unwrap();
            assert!(
                expected.insert(r.raw(), dst.raw()).is_none(),
                "rel ids must be unique"
            );
        }
        let commit_lsn = commit(tx, &store).unwrap();
        assert!(commit_lsn.raw() > 0);

        let reader = mgr.begin(tenant);
        let out: Vec<TelEntry> = scan_out(&store, &reader, hub, Some(ty)).collect();
        assert_eq!(
            out.len(),
            SUPERNODE_FANOUT,
            "every supernode out-edge must be queryable (no silent overflow drop)"
        );
        // Exhaustive oracle: exact (rel_id -> dst) topology, every edge.
        let got: std::collections::BTreeMap<u64, u64> =
            out.iter().map(|e| (e.rel_id, e.dst_id)).collect();
        assert_eq!(got, expected, "every rel id maps to its original dst");
    }

    /// AC-1 (inbound / reverse path): `SUPERNODE_FANOUT` distinct sources
    /// all pointing at ONE `dst` (an inbound supernode) must all be
    /// queryable via `scan_in`. Pins the `tel_append_reverse` growth
    /// branch — the symmetric `grown()`-drops-prev hazard. RED pre-fix:
    /// `scan_in` collapses far below the inserted count.
    #[test]
    fn supernode_inbound_fanin_beyond_cap_all_queryable() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        assert!(
            store.reverse_index_enabled(),
            "reverse index must be on for the inbound supernode oracle"
        );
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        let mut tx = mgr.begin(tenant);
        let sink = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        // Reverse entries store the ORIGINAL SRC in `dst_id` (ADR-131).
        let mut expected_srcs: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for _ in 0..SUPERNODE_FANOUT {
            let src = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            create_rel(&store, &mut tx, tenant, src, sink, ty, &PropertyData::Empty).unwrap();
            expected_srcs.insert(src.raw());
        }
        commit(tx, &store).unwrap();

        let reader = mgr.begin(tenant);
        let in_edges = scan_in(&store, &reader, sink, Some(ty)).expect("reverse index enabled");
        assert_eq!(
            in_edges.len(),
            SUPERNODE_FANOUT,
            "every inbound edge of the sink must be queryable (no reverse overflow drop)"
        );
        let got_srcs: std::collections::BTreeSet<u64> = in_edges.iter().map(|e| e.dst_id).collect();
        assert_eq!(got_srcs, expected_srcs, "every inbound source preserved");
    }

    /// TEL per-chain structural-isolation invariant: node A overflows
    /// (`SUPERNODE_FANOUT` edges) while a separately-keyed node B holds
    /// a handful. After the re-link fix, A recovers its full fan-out
    /// AND B's chain is byte-identical.
    ///
    /// ⚠️ SCOPE — read before trusting this as a "collateral" guard.
    /// TEL adjacency chains are keyed by `(tenant, src, channel)` over a
    /// MONOTONIC `alloc_virtual_page` counter (never reused), so node
    /// A's chain growth is *structurally incapable* of touching node
    /// B's chain. The B-assertion below therefore holds REGARDLESS of
    /// the silent-drop bug — it is GREEN-BOTH-WAYS and does NOT RED-flip
    /// on the #812 mechanism (only the A-assertion does: A collapses to
    /// ~906 pre-fix). This test pins that the re-link fix preserves the
    /// per-chain keying; it is **not** a guard for the collateral
    /// corruption #812 reported.
    ///
    /// #812's reported "an overflowing node corrupts unrelated
    /// relationships elsewhere" is a SEPARATE root cause — the
    /// dual-write PageId-collision #811 — which this in-memory,
    /// dual-write-OFF `CrudStore::new()` cannot exhibit (it has no
    /// `RecordPageStore`). That corruption is now fixed and pinned by
    /// the `collateral_corruption_dual_write_page_collision_is_811`
    /// regression below (un-ignored when the #811 allocation fix landed).
    #[test]
    fn supernode_overflow_isolated_to_own_tel_chain() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        let mut tx = mgr.begin(tenant);
        let a = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        for _ in 0..SUPERNODE_FANOUT {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            create_rel(&store, &mut tx, tenant, a, dst, ty, &PropertyData::Empty).unwrap();
        }
        // Node B, created AFTER A overflows, with a small known edge set.
        let b = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let mut b_expected: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for _ in 0..5 {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let r = create_rel(&store, &mut tx, tenant, b, dst, ty, &PropertyData::Empty).unwrap();
            b_expected.insert(r.raw());
        }
        commit(tx, &store).unwrap();

        let reader = mgr.begin(tenant);
        // A recovers its full fan-out (RED pre-fix: collapses to ~906).
        assert_eq!(
            scan_out(&store, &reader, a, Some(ty)).count(),
            SUPERNODE_FANOUT,
            "overflowing node A must retain all its edges"
        );
        // B's separately-keyed chain is intact. GREEN-both-ways — a
        // structural-keying invariant, NOT a #812 collateral-corruption
        // guard (see the scope note above).
        let b_got: std::collections::BTreeSet<u64> = scan_out(&store, &reader, b, Some(ty))
            .map(|e| e.rel_id)
            .collect();
        assert_eq!(
            b_got, b_expected,
            "node B's separately-keyed TEL chain must be unperturbed by A's overflow"
        );
    }

    /// #811 REGRESSION — the collateral-corruption half of #812 (was a
    /// `#[ignore]`'d forward-pin until the #811 fix landed; now active).
    ///
    /// #812 reported two P0 symptoms: (1) the silent ~2048-edge cap
    /// (fixed by the TEL re-link above), and (2) an overflowing node
    /// *collaterally corrupts unrelated relationships elsewhere*.
    /// Symptom (2) was a SEPARATE, pre-existing root cause filed as
    /// #811: [`PageAllocator`] handed out PageIds partitioned by
    /// `(TenantId, PageType)` (the Node and Rel sequences EACH started
    /// at `PageId(1)`), but [`RecordPageStore`] keys its page map by
    /// `PageId` ALONE. So the first Rel record page collided with the
    /// first Node record page in the shared dual-write store — from
    /// cold, *overflow-independent* (the first rel, not the 2049-th).
    /// Live, `install_fresh` returned `DuplicatePage` ("page already
    /// mapped") and the commit drain WARN-swallowed it (crud.rs ~3295,
    /// "we intentionally do NOT fail the bundle"); on WAL replay
    /// `install_or_replace` would silently OVERWRITE the Node page bytes
    /// with the Rel page's — cross-kind corruption. THIS is the
    /// "collateral corruption" #812 reported, catastrophic at scale (it
    /// fired from the first rel, not only on supernodes).
    ///
    /// **The fix** (no ADR, no on-disk format change, no store re-key):
    /// `open_or_fresh_page` now allocates BOTH record page types from a
    /// single per-tenant sequence via
    /// [`PageAllocator::alloc_record_page`] (canonically keyed
    /// `RECORD_PAGE_DOMAIN = Node`), matching the `RecordPageStore`'s
    /// one flat `PageId` keyspace. The page is still installed with its
    /// REAL type in the header, so on-disk identity / commit-bundle
    /// `Record` entries / replay are unchanged. The #826 author had
    /// speculated #811 was ADR-gated (a store/handle re-key); unifying
    /// the *allocation domain* instead avoids that surface entirely.
    ///
    /// The #812 TEL re-link fix CANNOT exhibit this: TEL chains are
    /// keyed by `(tenant, src, channel)` over a MONOTONIC virtual-page
    /// counter, never the record-store PageId space — which is exactly
    /// why `supernode_overflow_isolated_to_own_tel_chain` is
    /// green-both-ways for node B.
    ///
    /// This pin drives the REAL dual-write routing
    /// (`open_or_fresh_page`) and asserts the property the fix
    /// establishes: a same-tenant Rel record page installs at an id
    /// that does NOT collide with the Node record page, and the Node
    /// page survives. Pre-fix it RED-flipped (the Rel install returned
    /// `DuplicatePage`).
    #[test]
    fn collateral_corruption_dual_write_page_collision_is_811() {
        let (_mgr, store, _primary) = build_dual_write_store();
        let tenant = TenantId::DEFAULT;

        // The first Node record page for this tenant installs cleanly.
        let node_pid = store
            .open_or_fresh_page(tenant, RecordKind::Node)
            .expect("first node page installs");

        // The first Rel record page for the SAME tenant. Pre-#811 the
        // allocator handed out `PageId(1)` from a separate Rel counter
        // and the flat-keyed record store rejected it as `DuplicatePage`
        // (collision with the node page). Post-#811 it draws the next id
        // from the unified record-page domain and installs cleanly.
        let rel_pid = store.open_or_fresh_page(tenant, RecordKind::Rel).expect(
            "#811: a same-tenant Rel page must install without colliding with the Node page",
        );

        // STRONG oracle: the two record pages occupy DISTINCT ids in the
        // shared keyspace — no collision, so neither can overwrite the
        // other on live install or WAL replay.
        assert_ne!(
            node_pid, rel_pid,
            "#811: Node and Rel record pages must not share a PageId in the flat record keyspace"
        );

        // Both pages are mapped and survive — the Node page was not
        // clobbered by the Rel install (the cross-kind corruption path).
        let records = store.records().expect("dual-write configured");
        assert!(
            records.contains(node_pid),
            "#811: the Node record page must survive the Rel install"
        );
        assert!(
            records.contains(rel_pid),
            "#811: the Rel record page must be mapped after install"
        );
    }

    /// #811 acceptance (1) — first edge from cold dual-writes cleanly
    /// with ZERO store divergence (in-memory leg).
    ///
    /// A fresh dual-write store: two nodes + one relationship in a single
    /// commit. The node record page takes the first id in the record
    /// store's flat keyspace; pre-#811 the rel record page drew
    /// `PageId(1)` from an independent Rel counter and collided with the
    /// node page — `install_create` returned `DuplicatePage`, the commit
    /// drain WARN-swallowed it, and the rel was SILENTLY absent from the
    /// primary index + record store (the dual-write target) while
    /// remaining visible via MVCC/TEL. `read_rel_with_store` falls back
    /// to MVCC on a primary miss, so the divergence is invisible to
    /// readers — which is why the durable `round_trip_rel_post_replay` is
    /// green-both-ways and does NOT catch this.
    ///
    /// STRONG no-divergence oracle (the two stores AGREE, not merely "no
    /// WARN appeared"): the rel is present in BOTH the primary index AND
    /// the record store, at consistent coordinates, on a page DISTINCT
    /// from the node page; the node page survives. Pre-fix the primary
    /// lookup returns `None` → RED.
    #[test]
    fn dual_write_first_edge_from_cold_no_divergence() {
        let (mgr, store, primary) = build_dual_write_store();
        let tenant = TenantId::DEFAULT;
        let mut tx = mgr.begin(tenant);
        let a = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let b = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let r = create_rel(
            &store,
            &mut tx,
            tenant,
            a,
            b,
            TypeId::new(7),
            &PropertyData::InlineU32Pair(11, 22),
        )
        .unwrap();
        commit(tx, &store).unwrap();

        let records = store.records().expect("dual-write configured");

        // The rel dual-wrote into the PRIMARY INDEX (pre-#811: `None` —
        // the collision made `install_create` fail and the entry was
        // never published).
        let rel_key = PrimaryKey::new(tenant, RecordKind::Rel, r.raw());
        let rel_slot = primary.lookup(rel_key).unwrap().expect(
            "#811: the first edge must be dual-written into the primary index (pre-fix: absent)",
        );

        // ...and the RECORD STORE maps that page and holds the rel at the
        // indexed slot — both stores AGREE on the rel-page mapping.
        assert!(
            records.contains(rel_slot.page),
            "#811: the record store must map the rel page the primary index points at"
        );
        {
            let latch = records.latch(rel_slot.page).unwrap();
            let g = latch.read();
            let page = crate::records::SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
            assert_eq!(
                page.header().page_type,
                PageType::Rel.as_byte(),
                "#811: the rel page must be stamped Rel in its header"
            );
            let rec = page
                .read_rel(rel_slot.slot)
                .unwrap()
                .expect("rel slot live");
            assert_eq!(rec.id, r.raw());
            assert_eq!(rec.src_id, a.raw());
            assert_eq!(rec.dst_id, b.raw());
            assert_eq!(rec.type_id, 7);
        }

        // The rel page is DISTINCT from the node page (no collision), and
        // the node page survives intact in the shared keyspace.
        let node_key = PrimaryKey::new(tenant, RecordKind::Node, a.raw());
        let node_slot = primary.lookup(node_key).unwrap().expect("node indexed");
        assert_ne!(
            rel_slot.page, node_slot.page,
            "#811: the rel record page must not collide with the node record page"
        );
        assert!(
            records.contains(node_slot.page),
            "#811: the node record page must survive the rel install"
        );
        {
            let latch = records.latch(node_slot.page).unwrap();
            let g = latch.read();
            let page = crate::records::SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
            assert_eq!(
                page.header().page_type,
                PageType::Node.as_byte(),
                "#811: the node page must remain stamped Node (not clobbered by the rel page)"
            );
        }
    }

    /// #811 acceptance — the dual-write commit drain emits NO
    /// "page already mapped" collision WARN for the first edge from cold.
    /// This is the explicit log-level oracle complementing the stronger
    /// data-agreement oracle in
    /// `dual_write_first_edge_from_cold_no_divergence`: a silenced WARN
    /// over diverged stores would be the worst outcome (doctrine §3), so
    /// we pin BOTH that the stores agree AND that the WARN never fired.
    /// Pre-fix the `tracing::warn!("dual-write create for ... failed:
    /// ... already mapped")` line fires → RED.
    #[tracing_test::traced_test]
    #[test]
    fn dual_write_first_edge_emits_no_collision_warn() {
        let (mgr, store, _primary) = build_dual_write_store();
        let tenant = TenantId::DEFAULT;
        let mut tx = mgr.begin(tenant);
        let a = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let b = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let _r = create_rel(
            &store,
            &mut tx,
            tenant,
            a,
            b,
            TypeId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();

        logs_assert(|lines: &[&str]| {
            for line in lines {
                if line.contains("already mapped")
                    || (line.contains("dual-write") && line.contains("failed"))
                {
                    return Err(format!(
                        "#811: unexpected dual-write collision WARN: {line}"
                    ));
                }
            }
            Ok(())
        });
    }

    /// #811 acceptance (2) — sustained create+delete churn shows ZERO
    /// dual-write divergence across EVERY cycle.
    ///
    /// Pre-#811 the Node and Rel record-page sequences were independent
    /// and overlapped in the low-id range, so as node pages and rel pages
    /// grew they collided intermittently (the issue reported "6 WARNs /
    /// 200 create+delete"). Here a node pool spanning ≥3 record pages
    /// (`NODE_CAPACITY == 119`) is created first so the rel-page ids the
    /// churn allocates land squarely on already-mapped node-page ids
    /// pre-fix; post-#811 the unified record-page domain hands every page
    /// a distinct id, so NONE collide.
    ///
    /// EXHAUSTIVE oracle (every cycle checked, not sampled): after each
    /// rel create+commit the rel is in BOTH the primary index and the
    /// record store on a page that is NOT a node page; after each
    /// delete+commit it is gone. `divergence` accumulates any violation
    /// and MUST be empty. Pre-fix it is non-empty (the first rel alone
    /// collides with node page 1).
    #[test]
    fn dual_write_churn_create_delete_no_divergence() {
        let (mgr, store, primary) = build_dual_write_store();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(3);
        let records = store.records().expect("dual-write configured");

        // Node pool spanning ≥3 record pages (NODE_CAPACITY = 119).
        const POOL: usize = 300;
        let mut nodes = Vec::with_capacity(POOL);
        let mut tx = mgr.begin(tenant);
        for _ in 0..POOL {
            nodes.push(
                create_node(
                    &store,
                    &mut tx,
                    tenant,
                    LabelId::new(1),
                    &PropertyData::Empty,
                )
                .unwrap(),
            );
        }
        commit(tx, &store).unwrap();

        // Snapshot the node record pages — at this point every page the
        // record store maps is a node page.
        let node_pages: std::collections::BTreeSet<u64> = records
            .iter_pages()
            .iter()
            .map(|(pid, _)| pid.raw())
            .collect();
        assert!(
            node_pages.len() >= 3,
            "node pool must span ≥3 record pages to overlap the rel-page id range; got {}",
            node_pages.len()
        );

        let mut divergence: Vec<String> = Vec::new();
        const CYCLES: usize = 200;
        for i in 0..CYCLES {
            let a = nodes[i % POOL];
            let b = nodes[(i + 1) % POOL];

            // CREATE the rel.
            let mut tx = mgr.begin(tenant);
            let r = create_rel(
                &store,
                &mut tx,
                tenant,
                a,
                b,
                ty,
                &PropertyData::InlineU32Pair(i as u32, 0),
            )
            .unwrap();
            commit(tx, &store).unwrap();

            let rel_key = PrimaryKey::new(tenant, RecordKind::Rel, r.raw());
            match primary.lookup(rel_key).unwrap() {
                None => divergence.push(format!(
                    "cycle {i}: rel {} absent from primary index (dual-write collided)",
                    r.raw()
                )),
                Some(slot) => {
                    if !records.contains(slot.page) {
                        divergence.push(format!(
                            "cycle {i}: rel page {:?} not mapped in record store",
                            slot.page
                        ));
                    } else if node_pages.contains(&slot.page.raw()) {
                        divergence.push(format!(
                            "cycle {i}: rel landed on NODE page {:?} (cross-kind collision)",
                            slot.page
                        ));
                    } else {
                        let latch = records.latch(slot.page).unwrap();
                        let g = latch.read();
                        let page =
                            crate::records::SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
                        match page.read_rel(slot.slot).unwrap() {
                            Some(rec) if rec.id == r.raw() => {}
                            _ => divergence.push(format!(
                                "cycle {i}: record store slot does not hold rel {}",
                                r.raw()
                            )),
                        }
                    }
                }
            }

            // DELETE the rel (the churn half — frees nothing on disk; the
            // allocator never recycles, so ids keep climbing).
            let mut tx = mgr.begin(tenant);
            delete_rel_with_store(&store, &mut tx, r).unwrap();
            commit(tx, &store).unwrap();
            if primary.lookup(rel_key).unwrap().is_some() {
                divergence.push(format!(
                    "cycle {i}: rel {} still in primary index after delete",
                    r.raw()
                ));
            }
        }

        assert!(
            divergence.is_empty(),
            "#811: dual-write divergence across {CYCLES} create+delete cycles must be ZERO; \
             got {} violation(s):\n{}",
            divergence.len(),
            divergence.join("\n")
        );

        // Every node page survived all of the churn — no rel install ever
        // overwrote one.
        for np in &node_pages {
            assert!(
                records.contains(PageId::new(*np)),
                "#811: node page {np} must survive all {CYCLES} churn cycles"
            );
        }
    }

    // ── W28 Feature #582 (ADR-045) — hot-vertex metrics wire ──────────

    /// **Founding no-op-trampoline regression guard** (the #1 fix in
    /// W28 #582). Before this slice, the `arcgraph_hot_vertex_warnings_total`
    /// metric was *registered* in `MetricsRegistry` but had NO producer
    /// caller — `tel_append`'s overflow site only emitted `tracing::warn!`.
    /// This test injects a recording `MetricsSink` into a real
    /// `CrudStore`, drives the REAL producer path (commit → `tel_append`
    /// overflow) via the public CRUD API, and asserts the sink observed
    /// the call. A regression that drops the
    /// `self.record_hot_vertex_warning(tenant)` call (re-introducing the
    /// trampoline) fails here even though the `tracing::warn!` + the
    /// overflow chain still work — exactly the bug class
    /// `feedback_noop_trampoline_anti_pattern.md` catches.
    ///
    /// Strong oracle: MAX_ENTRIES edges fill a MAX_BLOCK_BYTES block;
    /// the (MAX_ENTRIES+1)-th forces EXACTLY ONE forward overflow (the
    /// sibling `scan_out_walks_overflow_chain` pins the "1 entry in the
    /// fresh head, MAX_ENTRIES in the predecessor" shape). Each `dst` is
    /// distinct, so no reverse chain overflows → the forward path is the
    /// sole emitter → exactly `== 1`.
    #[test]
    fn tel_overflow_fires_hot_vertex_warning_through_sink() {
        use crate::metrics::{CountingMetricsSink, MetricsSink};
        use crate::tel::MAX_ENTRIES;
        let mgr = TxnManager::new();
        let sink = Arc::new(CountingMetricsSink::new());
        let store = CrudStore::new().with_metrics_sink(sink.clone() as Arc<dyn MetricsSink>);
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);

        // Baseline: no overflow yet → zero emissions (the producer must
        // not fire on non-overflow appends).
        assert_eq!(sink.hot_vertex_warning_count(), 0);

        let n = MAX_ENTRIES as usize + 1;
        let _ = setup_rels(&mgr, &store, tenant, ty, n);

        assert_eq!(
            sink.hot_vertex_warning_count(),
            1,
            "exactly one forward TEL overflow must fire record_hot_vertex_warning \
             through the sink (the no-op-trampoline guard)"
        );
    }

    /// Fault-injection per failure mode: the unwired (`metrics_sink:
    /// None`) path must be a zero-cost no-op — no panic, and the
    /// overflow chain behavior is byte-for-byte identical to a store
    /// without a sink. Pins the PD-5 `Option::is_none()` early-out.
    #[test]
    fn tel_overflow_without_sink_is_noop_and_preserves_behavior() {
        use crate::tel::MAX_ENTRIES;
        let mgr = TxnManager::new();
        // Default CrudStore — metrics_sink is None.
        let store = CrudStore::new();
        assert!(
            store.metrics_sink().is_none(),
            "default CrudStore must have no metrics sink wired"
        );
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);
        let n = MAX_ENTRIES as usize + 1;

        // Must not panic; overflow chain must scan back the full set.
        let (src, rels) = setup_rels(&mgr, &store, tenant, ty, n);
        assert_eq!(rels.len(), n);
        let reader = mgr.begin(tenant);
        let out_count = scan_out(&store, &reader, src, Some(ty)).count();
        assert_eq!(
            out_count, n,
            "None-sink overflow path must not change CRUD behavior"
        );
    }

    #[test]
    fn scan_out_tenant_isolated() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let t_a = TenantId::new(100);
        let t_b = TenantId::new(101);
        let ty = TypeId::new(3);

        let (src_a, _rels_a) = setup_rels(&mgr, &store, t_a, ty, 3);
        let (src_b, _rels_b) = setup_rels(&mgr, &store, t_b, ty, 4);
        // Both tenants' src ids collide numerically (both 1).
        assert_eq!(src_a.raw(), src_b.raw());

        let reader_a = mgr.begin(t_a);
        let reader_b = mgr.begin(t_b);

        // Per-tenant rel ids also collide numerically (both allocators
        // start at 1), so isolation is proven by cardinality: a leak
        // between chains would merge both tenants' rels into 7 under
        // either scan, not 3 / 4.
        let a_count = scan_out(&store, &reader_a, src_a, Some(ty)).count();
        let b_count = scan_out(&store, &reader_b, src_b, Some(ty)).count();
        assert_eq!(a_count, 3);
        assert_eq!(b_count, 4);
        // Same property via the None (union) path.
        assert_eq!(scan_out(&store, &reader_a, src_a, None).count(), 3);
        assert_eq!(scan_out(&store, &reader_b, src_b, None).count(), 4);
    }

    // ── M2-25 update_node / update_rel ────────────────────────────

    #[test]
    fn update_node_new_version_visible_to_later_snapshot() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .unwrap();
        tx.commit().unwrap();

        let mut upd = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut upd, id, &PropertyData::InlineU32Pair(5, 6)).unwrap();
        upd.commit().unwrap();

        let reader = mgr.begin(TenantId::DEFAULT);
        let rec = read_node(&reader, id).unwrap().unwrap();
        assert_eq!(rec.inline_u32a, 5);
        assert_eq!(rec.inline_u32b, 6);
        // Label preserved across update.
        assert_eq!(rec.label_id, 7);
    }

    #[test]
    fn update_node_old_version_visible_to_pre_update_snapshot() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(10, 20),
        )
        .unwrap();
        tx.commit().unwrap();

        // Reader begun before the update should keep seeing the old
        // inline pair through the MVCC chain.
        let pre = mgr.begin(TenantId::DEFAULT);

        let mut upd = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut upd, id, &PropertyData::InlineU32Pair(99, 99)).unwrap();
        upd.commit().unwrap();

        let rec = read_node(&pre, id).unwrap().unwrap();
        assert_eq!(rec.inline_u32a, 10);
        assert_eq!(rec.inline_u32b, 20);
    }

    #[test]
    fn update_node_on_nonexistent_id_returns_not_found() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let err = update_node(&store, &mut tx, NodeId::new(4242), &PropertyData::Empty)
            .expect_err("update on missing id must fail");
        assert!(matches!(
            err,
            CrudError::NotFound {
                kind: "node",
                id: 4242,
                ..
            }
        ));
    }

    #[test]
    fn update_node_concurrent_writers_detect_conflict() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx.commit().unwrap();

        // Two overlapping updaters race on the same key; OCC must
        // reject the second.
        let mut a = mgr.begin(TenantId::DEFAULT);
        let mut b = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut a, id, &PropertyData::InlineU32Pair(1, 1)).unwrap();
        update_node(&store, &mut b, id, &PropertyData::InlineU32Pair(2, 2)).unwrap();
        a.commit().unwrap();
        let err = b.commit().expect_err("b must lose OCC race");
        assert!(matches!(err, ArcGraphError::MvccConflict { .. }));
    }

    #[test]
    fn update_rel_preserves_tel_entry_identity() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(11);

        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let rel = create_rel(
            &store,
            &mut tx,
            tenant,
            src,
            dst,
            ty,
            &PropertyData::InlineU32Pair(1, 1),
        )
        .unwrap();
        commit(tx, &store).unwrap();

        let before: Vec<TelEntry> = {
            let r = mgr.begin(tenant);
            scan_out(&store, &r, src, Some(ty)).collect()
        };
        assert_eq!(before.len(), 1);
        let before_entry = before[0];

        let mut upd = mgr.begin(tenant);
        update_rel(&store, &mut upd, rel, &PropertyData::InlineU32Pair(9, 9)).unwrap();
        upd.commit().unwrap();

        let after: Vec<TelEntry> = {
            let r = mgr.begin(tenant);
            scan_out(&store, &r, src, Some(ty)).collect()
        };
        assert_eq!(after.len(), 1);
        // Same dst/rel/created_lsn — TEL entry untouched.
        assert_eq!(after[0].dst_id, before_entry.dst_id);
        assert_eq!(after[0].rel_id, before_entry.rel_id);
        assert_eq!(after[0].created_lsn, before_entry.created_lsn);
        // Yet the MVCC-side bytes reflect the new props.
        let r = mgr.begin(tenant);
        let rec = read_rel(&r, rel).unwrap().unwrap();
        assert_eq!(rec.inline_u32a, 9);
        assert_eq!(rec.inline_u32b, 9);
    }

    // ── M2-31 PropertyData::Blob end-to-end ───────────────────────

    #[test]
    fn create_node_with_blob_roundtrips_through_overflow() {
        use crate::property::{PropertyReadout, decode_node as decode_property_node};
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let payload = vec![0xAB_u8; 20_000]; // > BLOB_CHUNK_BYTES, forces chain

        let mut tx = mgr.begin(tenant);
        let id = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(3),
            &PropertyData::Blob(payload.clone()),
        )
        .unwrap();
        tx.commit().unwrap();

        let reader = mgr.begin(tenant);
        let rec = read_node(&reader, id).unwrap().unwrap();
        match decode_property_node(&rec) {
            PropertyReadout::Overflow(blob_ref) => {
                let round = store.blobs.get(tenant, blob_ref).unwrap();
                assert_eq!(round.as_ref(), payload.as_slice());
            }
            other => panic!("expected overflow readout, got {other:?}"),
        }
    }

    #[test]
    fn aborted_blob_writes_do_not_leak_pending_emits_across_transactions() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        for cycle in 0..2 {
            let mut tx = mgr.begin(tenant);
            let txn_id = tx.id();
            create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(3),
                // > BLOB_CHUNK_BYTES forces a chain-page emit — keeps the
                // test pinned to the pending_blob_emits path even if a
                // future small-blob packing step (M1) inlines tiny blobs.
                &PropertyData::Blob(vec![cycle; 20_000]),
            )
            .unwrap();
            assert!(
                store.pending_blob_emits.contains_key(&txn_id),
                "blob write must populate the per-txn pending emit buffer"
            );

            store.discard_pending(txn_id);
            tx.abort();

            assert!(
                !store.pending_blob_emits.contains_key(&txn_id),
                "aborting transaction {txn_id} must discard its pending blob emits"
            );
            assert!(
                store.pending_blob_emits.is_empty(),
                "repeated aborts must not grow the pending blob emit buffer"
            );
        }
    }

    #[test]
    fn update_node_blob_visible_to_later_snapshot() {
        use crate::property::{PropertyReadout, decode_node as decode_property_node};
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let id = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .unwrap();
        tx.commit().unwrap();

        let payload = vec![0x42_u8; 9000];
        let mut upd = mgr.begin(tenant);
        update_node(&store, &mut upd, id, &PropertyData::Blob(payload.clone())).unwrap();
        upd.commit().unwrap();

        let reader = mgr.begin(tenant);
        let rec = read_node(&reader, id).unwrap().unwrap();
        match decode_property_node(&rec) {
            PropertyReadout::Overflow(blob_ref) => {
                let round = store.blobs.get(tenant, blob_ref).unwrap();
                assert_eq!(round.as_ref(), payload.as_slice());
            }
            other => panic!("expected overflow readout, got {other:?}"),
        }
    }

    // ── M2-26 delete_node / delete_rel ────────────────────────────

    #[test]
    fn delete_node_commits_tombstone_then_read_returns_none() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx.commit().unwrap();

        let mut del = mgr.begin(TenantId::DEFAULT);
        delete_node(&mut del, id).unwrap();
        del.commit().unwrap();

        let reader = mgr.begin(TenantId::DEFAULT);
        assert!(read_node(&reader, id).unwrap().is_none());
    }

    #[test]
    fn delete_node_pre_delete_snapshot_still_sees_record() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        tx.commit().unwrap();

        let mid = mgr.begin(TenantId::DEFAULT);

        let mut del = mgr.begin(TenantId::DEFAULT);
        delete_node(&mut del, id).unwrap();
        del.commit().unwrap();

        assert!(read_node(&mid, id).unwrap().is_some());
    }

    #[test]
    fn delete_node_on_nonexistent_id_returns_not_found() {
        let mgr = TxnManager::new();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let err = delete_node(&mut tx, NodeId::new(42)).expect_err("not found");
        assert!(matches!(
            err,
            CrudError::NotFound {
                kind: "node",
                id: 42,
                ..
            }
        ));
    }

    #[test]
    fn delete_rel_commits_tombstone_then_read_rel_returns_none() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let rel = create_rel(
            &store,
            &mut tx,
            tenant,
            src,
            dst,
            TypeId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();

        let mut del = mgr.begin(tenant);
        delete_rel(&mut del, rel).unwrap();
        del.commit().unwrap();

        let reader = mgr.begin(tenant);
        assert!(read_rel(&reader, rel).unwrap().is_none());
    }

    /// Post-#22 invariant: a `delete_rel` tombstone is hidden from
    /// `scan_out` at the caller's snapshot. The TEL entry itself is
    /// still present in the chain (the TEL is append-only; block
    /// rewrite is M2.e scope), but the MVCC probe in `scan_out`
    /// filters the dead entry out.
    #[test]
    fn delete_rel_tombstone_hides_from_scan_out() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(1);
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let rel = create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        commit(tx, &store).unwrap();

        let before = {
            let r = mgr.begin(tenant);
            scan_out(&store, &r, src, Some(ty)).count()
        };
        assert_eq!(before, 1);

        let mut del = mgr.begin(tenant);
        delete_rel(&mut del, rel).unwrap();
        del.commit().unwrap();

        let r = mgr.begin(tenant);
        assert!(read_rel(&r, rel).unwrap().is_none());
        let after = scan_out(&store, &r, src, Some(ty)).count();
        assert_eq!(
            after, 0,
            "scan_out filters MVCC-tombstoned rels via tx.read (#22)"
        );
    }

    /// Regression test (#22): with 10 rels created and 5 deleted,
    /// `scan_out` yields exactly the 5 survivors at a post-delete
    /// snapshot. Exercises the MVCC probe across multiple entries in
    /// one chain.
    #[test]
    fn scan_out_skips_mvcc_tombstoned_rels() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(42);

        // Phase 1 — create src and 10 rels src→dst_i all under `ty`.
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let mut rels: Vec<RelId> = Vec::with_capacity(10);
        for _ in 0..10 {
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            rels.push(
                create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap(),
            );
        }
        commit(tx, &store).unwrap();

        // Pre-delete sanity: scan_out yields all 10.
        {
            let r = mgr.begin(tenant);
            let seen: Vec<TelEntry> = scan_out(&store, &r, src, Some(ty)).collect();
            assert_eq!(seen.len(), 10, "all 10 rels visible before deletes");
        }

        // Phase 2 — delete the first 5 rels.
        let to_delete: Vec<RelId> = rels.iter().take(5).copied().collect();
        let to_keep: Vec<RelId> = rels.iter().skip(5).copied().collect();
        let mut del_tx = mgr.begin(tenant);
        for r in &to_delete {
            delete_rel(&mut del_tx, *r).unwrap();
        }
        del_tx.commit().unwrap();

        // Phase 3 — post-delete snapshot: scan_out yields exactly the
        // 5 survivors and none of the tombstoned ids.
        let reader = mgr.begin(tenant);
        let seen: Vec<TelEntry> = scan_out(&store, &reader, src, Some(ty)).collect();
        assert_eq!(seen.len(), 5, "exactly the 5 non-deleted rels visible");

        let seen_ids: std::collections::HashSet<u64> = seen.iter().map(|e| e.rel_id).collect();
        for kept in &to_keep {
            assert!(
                seen_ids.contains(&kept.raw()),
                "kept rel {:?} must be visible",
                kept
            );
        }
        for gone in &to_delete {
            assert!(
                !seen_ids.contains(&gone.raw()),
                "tombstoned rel {:?} must not surface",
                gone
            );
        }

        // Also verify the `type_filter = None` union path applies the
        // same filter.
        let all: Vec<TelEntry> = scan_out(&store, &reader, src, None).collect();
        assert_eq!(
            all.len(),
            5,
            "union scan also filters tombstoned rels under `None` filter"
        );
    }

    #[test]
    fn node_mvcc_key_keeps_top_bit_clear() {
        // Sanity: fresh allocations stay in the low half of the
        // MvccKey namespace so they never collide with rel keys.
        let store = CrudStore::new();
        for _ in 0..1024 {
            let id = store.alloc_node(TenantId::DEFAULT).unwrap();
            assert_eq!(node_mvcc_key(id) & REL_TAG_BIT, 0);
        }
    }

    // ── issue #20: MVCC↔TEL composite snapshot-isolation window ───
    //
    // `create_rel` buffers the rel's `RelRecord` into the MVCC chain
    // (`tx.write(rel_mvcc_key(..))`, crud.rs ~3144) and stages a
    // `PendingTelAppend` (crud.rs ~3151). `crud::commit` publishes the
    // rel's visibility in MVCC Phase 3 — `visible.store(commit_lsn)` at
    // `transaction.rs::commit_with_bundle_writes` (~1323) — and only
    // THEN drains the staged TEL appends (`for p in pending_tel`,
    // crud.rs ~3778), AFTER `commit_with_bundle_and_rollback` has
    // returned. Because a reader sources its snapshot from `visible`
    // (`begin_inner`, transaction.rs ~617), there is a window in which a
    // concurrent txn sees the rel via MVCC (`read_rel`) but NOT via the
    // TEL adjacency (`scan_out`) — a snapshot-isolation violation across
    // the MVCC+TEL composite store. The two tests below pin it.

    /// Deterministic characterization of the #20 failure mode (the
    /// intermediate state the live window exposes): a rel whose
    /// `RelRecord` is MVCC-committed (visible to a snapshot ≥ its
    /// `commit_lsn`) but whose TEL adjacency is not yet appended is
    /// visible via `read_rel` yet ABSENT from `scan_out`.
    ///
    /// We freeze that state on purpose: stage with the real `create_rel`
    /// (which buffers the MVCC write AND a `PendingTelAppend`), then
    /// commit ONLY the MVCC versions via the raw `Transaction::commit`,
    /// which — unlike `crud::commit` — does not drain `pending_tel`.
    /// That is exactly the state a concurrent reader can observe in the
    /// gap between Phase-3 `visible.store(commit_lsn)` and the
    /// post-commit `pending_tel` drain on the live path.
    ///
    /// This pin stays GREEN before and after the eventual commit-path
    /// fix (it constructs the pre-drain state by hand); it documents the
    /// read-side mechanism, it is NOT the ordering regression gate. The
    /// gate is the `#[ignore]`d concurrent test below, which exercises
    /// the real `crud::commit` ordering.
    #[test]
    fn mvcc_tel_window_20_scan_out_misses_mvcc_committed_rel_until_tel_appended() {
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(7);

        // src + dst committed normally.
        let mut tx0 = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx0,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx0,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx0, &store).unwrap();

        // Stage the rel with the REAL create_rel, then commit ONLY the
        // MVCC versions (raw Transaction::commit) — leaving the TEL
        // append undrained, freezing the #20 window's intermediate state.
        let mut tx = mgr.begin(tenant);
        let rel = create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        let txn_id = tx.id();
        let commit_lsn = tx.commit().unwrap();

        // The rel IS visible via MVCC at a fresh snapshot...
        let r1 = mgr.begin(tenant);
        assert!(
            read_rel(&r1, rel).unwrap().is_some(),
            "rel must be MVCC-visible after commit (visible.store published commit_lsn)"
        );
        // ...but scan_out MISSES it, because the TEL entry is unappended.
        let visible_via_tel = scan_out(&store, &r1, src, Some(ty)).any(|e| e.rel_id == rel.raw());
        assert!(
            !visible_via_tel,
            "issue #20 window: a rel visible via MVCC (read_rel) is absent from scan_out \
             while its TEL adjacency is unappended — a composite-store snapshot-isolation \
             inconsistency. commit_lsn={commit_lsn:?}"
        );
        drop(r1);

        // Appending the TEL entry (what crud::commit's post-commit drain
        // does) closes the inconsistency: a fresh reader now sees the rel
        // via BOTH paths.
        let channel = LabelId::new(ty.raw());
        store
            .tel_append(tenant, src, channel, dst, rel, commit_lsn)
            .unwrap();
        store.discard_pending(txn_id); // drop the now-applied staged append

        let r2 = mgr.begin(tenant);
        assert!(
            read_rel(&r2, rel).unwrap().is_some(),
            "rel still MVCC-visible"
        );
        let n = scan_out(&store, &r2, src, Some(ty))
            .filter(|e| e.rel_id == rel.raw())
            .count();
        assert_eq!(
            n, 1,
            "after the TEL append, scan_out sees the rel — both representations agree"
        );
    }

    /// Real-path regression gate for issue #20: a writer thread commits
    /// `create_rel`s through `crud::commit`; reader threads concurrently
    /// `begin()` and assert the composite invariant
    ///
    ///   `read_rel(rel).is_some()`  ⟹  `scan_out(src)` contains `rel`.
    ///
    /// Each rel fans out from its own fresh `src`, so `scan_out` walks a
    /// one-entry chain (O(1) per probe). The writer publishes the just-
    /// allocated `rel_id` (Release) before calling `crud::commit`; a
    /// reader that sees `read_rel(rel).is_some()` has a snapshot ≥
    /// `commit_lsn` (i.e. it began after Phase-3 `visible.store`) and so
    /// MUST also see the adjacency. A violation means the reader landed
    /// in the gap between `visible.store` (transaction.rs ~1323) and the
    /// `pending_tel` drain (crud.rs ~3778).
    ///
    /// No false positives: the writer is serial, so `tel_append_i`
    /// happens-before `visible.store_{i+1}`; a reader observing
    /// `visible ≥ commit_lsn_{i+1}` necessarily sees `tel_append_i`. The
    /// only violation case is `snapshot == commit_lsn_i` with
    /// `tel_append_i` not yet run — the #20 window precisely.
    ///
    /// `#[ignore]`d because (a) it is RED while #20 is open and (b) it is
    /// a race, so it is kept off the default CI run. Empirically the
    /// window recurs on every commit: a multi-core `--ignored` run
    /// reliably observes tens of thousands of violations (measured
    /// ~62k–80k of ~171k–202k MVCC-visible probes, 3/3 runs) — it is not
    /// a rare-interleaving lottery. The commit-path fix (TEL drain before
    /// Phase-3 `visible.store`, ADR-gated) flips this GREEN and un-ignores
    /// it; a follow-up should add a forced commit-path seam so the gate is
    /// deterministic rather than load-dependent.
    #[test]
    #[ignore = "issue #20 OPEN: demonstrates the MVCC↔TEL commit-ordering window; run \
                with --ignored. Un-ignore once the commit-path reorder lands."]
    fn mvcc_tel_window_20_concurrent_committer_scanner_invariant() {
        use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        let mgr = Arc::new(TxnManager::new());
        let store = Arc::new(CrudStore::new());
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(7);

        const N: usize = 20_000;
        const READERS: usize = 3;

        // Pre-create N (src_i, dst_i) node pairs, committed up front, so
        // the writer's per-iteration commit does ONLY the create_rel
        // (keeping the window tight + the chains one-entry).
        let mut pairs: Vec<(NodeId, NodeId)> = Vec::with_capacity(N);
        {
            let mut i = 0;
            while i < N {
                let mut tx = mgr.begin(tenant);
                let upper = (i + 500).min(N);
                for _ in i..upper {
                    let s = create_node(
                        &store,
                        &mut tx,
                        tenant,
                        LabelId::new(1),
                        &PropertyData::Empty,
                    )
                    .unwrap();
                    let d = create_node(
                        &store,
                        &mut tx,
                        tenant,
                        LabelId::new(1),
                        &PropertyData::Empty,
                    )
                    .unwrap();
                    pairs.push((s, d));
                }
                commit(tx, &store).unwrap();
                i = upper;
            }
        }
        let pairs = Arc::new(pairs);
        let rel_ids: Arc<Vec<AtomicU64>> = Arc::new((0..N).map(|_| AtomicU64::new(0)).collect());
        let cursor = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let violations = Arc::new(AtomicU64::new(0));
        let observed_visible = Arc::new(AtomicU64::new(0));
        let start = Arc::new(Barrier::new(READERS + 1));

        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let mgr = Arc::clone(&mgr);
                let store = Arc::clone(&store);
                let pairs = Arc::clone(&pairs);
                let rel_ids = Arc::clone(&rel_ids);
                let cursor = Arc::clone(&cursor);
                let done = Arc::clone(&done);
                let violations = Arc::clone(&violations);
                let observed_visible = Arc::clone(&observed_visible);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    while !done.load(Ordering::Acquire) {
                        let i = cursor.load(Ordering::Acquire);
                        if i >= N {
                            continue;
                        }
                        let rid = rel_ids[i].load(Ordering::Acquire);
                        if rid == 0 {
                            continue;
                        }
                        let (src, _dst) = pairs[i];
                        let tx = mgr.begin(tenant);
                        if read_rel(&tx, RelId::new(rid)).unwrap().is_some() {
                            // Visible via MVCC at this snapshot (≥ commit_lsn_i,
                            // i.e. after Phase-3 visible.store). Snapshot
                            // isolation requires the adjacency to be visible too.
                            observed_visible.fetch_add(1, Ordering::Relaxed);
                            let present =
                                scan_out(&store, &tx, src, Some(ty)).any(|e| e.rel_id == rid);
                            if !present {
                                violations.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();

        let writer = {
            let mgr = Arc::clone(&mgr);
            let store = Arc::clone(&store);
            let pairs = Arc::clone(&pairs);
            let rel_ids = Arc::clone(&rel_ids);
            let cursor = Arc::clone(&cursor);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for (i, &(src, dst)) in pairs.iter().enumerate() {
                    let mut tx = mgr.begin(tenant);
                    let rel =
                        create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty)
                            .unwrap();
                    rel_ids[i].store(rel.raw(), Ordering::Release);
                    cursor.store(i, Ordering::Release);
                    // crud::commit publishes Phase-3 visibility (visible.store)
                    // and THEN drains pending_tel (tel_append) — the #20 window.
                    commit(tx, &store).unwrap();
                }
            })
        };

        writer.join().unwrap();
        done.store(true, Ordering::Release);
        for r in readers {
            r.join().unwrap();
        }

        let v = violations.load(Ordering::Relaxed);
        let seen = observed_visible.load(Ordering::Relaxed);
        eprintln!(
            "issue #20 demonstration: readers observed the just-committed rel via MVCC \
             {seen} times; {v} of those fell inside the MVCC↔TEL window (rel visible via \
             read_rel but absent from scan_out)."
        );
        assert_eq!(
            v, 0,
            "issue #20 OPEN: observed {v} MVCC↔TEL snapshot-isolation violations — a \
             create_rel was visible via MVCC read_rel but its adjacency was missing from \
             scan_out. visible.store (transaction.rs Phase 3, ~1323) is published before \
             the post-commit pending_tel drain (crud.rs ~3778). Closing the window \
             (commit-path reorder, ADR-gated) makes this assertion hold."
        );
    }

    // ── M2-WAL: CRUD-level durability integration ─────────────────

    mod wal {
        use std::path::Path;
        use std::time::Duration;

        use tempfile::tempdir;

        use super::*;
        use crate::wal::{WalConfig, WalRecord, WalRecordType, WalWriter};

        fn fast_config(dir: &Path) -> WalConfig {
            WalConfig {
                dir: dir.to_path_buf(),
                segment_size_bytes: 16 * 1024 * 1024,
                group_commit_window: Duration::from_millis(2),
                group_commit_max_batch: 4,
                metrics_sink: None,
                encryption: None,
                inflight_budget_bytes: None,
            }
        }

        fn drain_segments(dir: &Path) -> Vec<WalRecord> {
            let mut out = Vec::new();
            let segs = crate::wal::segment::list_segments(dir).unwrap();
            for seg in segs {
                let bytes =
                    std::fs::read(dir.join(crate::wal::segment::segment_filename(seg))).unwrap();
                // Skip the 8-byte segment header (issue #39 format
                // versioning); records start at SegmentHeader::SIZE.
                let mut cursor = crate::wal::segment::SegmentHeader::SIZE;
                while cursor < bytes.len() {
                    let (r, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
                    out.push(r);
                    cursor += consumed;
                }
            }
            out
        }

        #[test]
        fn create_node_then_commit_lands_in_wal() {
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let mgr = TxnManager::with_wal(writer.handle());
            let store = CrudStore::new();

            let mut tx = mgr.begin(TenantId::DEFAULT);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(3),
                &PropertyData::InlineU32Pair(7, 8),
            )
            .unwrap();
            // Issue #129 P0 fix: route through `crud::commit` so the
            // builder closure runs and drains allocator advances
            // (`tx.commit()` is the kernel-only path with a no-op
            // builder; it has no access to the CrudStore allocator
            // state).
            let commit_lsn = commit(tx, &store).unwrap();
            writer.shutdown().unwrap();

            let records = drain_segments(dir.path());
            assert_eq!(records.len(), 1);
            let r = &records[0];
            // ADR-031: all MVCC commits emit `CommitBundle`. The
            // bundle's MVCC writes section is byte-compatible with
            // the legacy `Commit = 2` payload body, and the
            // `n_index_pages` field at the tail is `0` when no
            // builder plumbed staged emits (this call uses raw
            // `tx.commit()`, not `crud::commit`).
            assert_eq!(r.record_type, WalRecordType::CommitBundle);
            // M3.a Slice G.4: commit path emits v5 bundles (extends
            // v4 with vector_pages tail).
            let bundle =
                crate::wal::bundle::decode_commit_bundle_v8(&r.payload, r.tenant_id).unwrap();
            assert_eq!(bundle.commit_lsn, commit_lsn);
            assert_eq!(bundle.mvcc_writes.len(), 1);
            // Key is the node MVCC key — low-half of the namespace.
            let key = *bundle.mvcc_writes.keys().next().expect("single MVCC write");
            assert_eq!(key, node_mvcc_key(id));
            assert_eq!(key & REL_TAG_BIT, 0);
            assert!(bundle.staged_pages.is_empty());
            // The CRUD store allocated NodeId(1) for this tenant;
            // the v4 bundle MUST carry the corresponding
            // AllocatorAdvance so post-recovery alloc_node cannot
            // reuse it. Without it the canary
            // `t1_strict_byte_identical_after_fault_recovery` fails.
            assert!(
                bundle
                    .allocator_advances
                    .iter()
                    .any(|a| a.tenant == TenantId::DEFAULT
                        && a.kind == crate::wal::AllocatorKind::Node
                        && a.new_high_water == id.raw()),
                "v4 bundle must record Node allocator advance for the \
                 freshly-allocated NodeId; got {:?}",
                bundle.allocator_advances,
            );
        }

        #[test]
        fn create_rel_commit_logs_one_record_with_both_writes() {
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let mgr = TxnManager::with_wal(writer.handle());
            let store = CrudStore::new();
            let tenant = TenantId::DEFAULT;

            let mut tx = mgr.begin(tenant);
            let src = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let dst = create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap();
            let _rel = create_rel(
                &store,
                &mut tx,
                tenant,
                src,
                dst,
                TypeId::new(5),
                &PropertyData::Empty,
            )
            .unwrap();
            commit(tx, &store).unwrap();
            writer.shutdown().unwrap();

            let records = drain_segments(dir.path());
            // Exactly one WAL record per MVCC commit regardless of
            // how many node/rel writes it aggregates. The TEL append
            // happens after commit; it does not emit its own WAL
            // record (see issue #20 — M2-WAL does NOT close the
            // MVCC↔TEL atomicity gap).
            assert_eq!(records.len(), 1);
            // ADR-031: `CommitBundle` carries the aggregated MVCC
            // writes. This call site commits via `crud::commit` so
            // staged IndexPage emits would be non-zero if a primary
            // index were configured; here the store has no primary
            // index, so `index_pages` must be empty.
            assert_eq!(records[0].record_type, WalRecordType::CommitBundle);
            // M3.a Slice G.4: commit path emits v5 bundles (extends
            // v4 with vector_pages tail).
            let bundle = crate::wal::bundle::decode_commit_bundle_v8(
                &records[0].payload,
                records[0].tenant_id,
            )
            .unwrap();
            assert_eq!(bundle.mvcc_writes.len(), 3);
            assert!(bundle.staged_pages.is_empty());
            // The CRUD store allocated 2 NodeIds + 1 RelId; v4
            // bundle must record advances for both kinds.
            assert!(
                bundle
                    .allocator_advances
                    .iter()
                    .any(|a| a.tenant == tenant && a.kind == crate::wal::AllocatorKind::Node)
            );
            assert!(
                bundle
                    .allocator_advances
                    .iter()
                    .any(|a| a.tenant == tenant && a.kind == crate::wal::AllocatorKind::Rel)
            );
        }

        #[test]
        fn blob_property_rides_commit_bundle_no_standalone_putblob() {
            // #810: a `PropertyData::Blob` on the durable CRUD write
            // path must NOT emit a standalone synchronous `PutBlob` WAL
            // record. That record fsynced once PER record (it went out
            // through the blocking `WalHandle::append` at blob-stage
            // time), and because every property-bearing record encodes
            // as a Blob, a batched ingest fsynced N+1 times — the
            // ~170 rec/s durable-bulk-load regression.
            //
            // The blob chain pages instead ride the owning transaction's
            // single `CommitBundle` as `BundlePageKind::Blob` staged
            // pages (drained by `crud::commit` via `take_blob_emits`);
            // WAL replay reconstructs the chain from those bundle entries
            // (`wal/replay.rs`), so the blob is durable with ONE fsync —
            // and atomically with its node record.
            //
            // Supersedes the pre-#810 `blob_put_emits_wal_record_before_commit`,
            // which pinned the now-removed PutBlob-before-Commit ordering
            // (review block C-1 / ADR-022). The durability that ordering
            // protected is now provided by the in-bundle Blob page —
            // asserted here (the node's blob head page rides the bundle)
            // and end-to-end in `tests/wal_replay_round_trip.rs`
            // (`round_trip_node_with_blob_property_post_replay`).
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let mgr = TxnManager::with_wal(writer.handle());
            let store = CrudStore::with_wal(writer.handle());

            let payload = b"hello blob durable world";
            let mut tx = mgr.begin(TenantId::DEFAULT);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(9),
                &PropertyData::Blob(payload.to_vec()),
            )
            .unwrap();
            // The durable commit path is `crud::commit` — it folds the
            // staged blob pages into the bundle. Raw `Transaction::commit`
            // is MVCC-only and is not the sanctioned blob-durability path.
            let commit_lsn = commit(tx, &store).unwrap();
            writer.shutdown().unwrap();

            let records = drain_segments(dir.path());

            // #810 contract: NO standalone PutBlob fsync on the durable path.
            assert!(
                records
                    .iter()
                    .all(|r| r.record_type != WalRecordType::PutBlob),
                "post-#810 the durable blob write path must NOT emit a \
                 per-record PutBlob fsync; the chain pages ride the CommitBundle"
            );

            // Exactly one CommitBundle carries the whole commit.
            let bundles: Vec<&WalRecord> = records
                .iter()
                .filter(|r| r.record_type == WalRecordType::CommitBundle)
                .collect();
            assert_eq!(bundles.len(), 1, "exactly one CommitBundle per commit");
            let bundle = crate::wal::bundle::decode_commit_bundle_v8(
                &bundles[0].payload,
                bundles[0].tenant_id,
            )
            .unwrap();
            assert!(commit_lsn.raw() > 0);

            // v2 M1 (ADR-230): a SMALL bag packs into a shared slotted
            // page and rides the bundle as a PropSlotted-kind staged
            // page — the same single-fsync durability the removed
            // PutBlob record provided (#810), now with the ~14× batch
            // amortization. (A > PROP_BAG_MAX_BYTES payload would still
            // ride as Blob-kind chain pages — pinned by the dedicated
            // M1 overflow tests.)
            let slotted_pages: Vec<&crate::wal::bundle::DecodedStagedPage> = bundle
                .staged_pages
                .iter()
                .filter(|p| p.kind == crate::wal::bundle::BundlePageKind::PropSlotted)
                .collect();
            assert!(
                !slotted_pages.is_empty(),
                "the CommitBundle must carry the packed slotted prop page (#810 durability, \
                 v2 M1 packing)"
            );

            // The node record's overflow page id must be one of the
            // PropSlotted pages that ride the bundle — proving the
            // durable bytes and the in-record reference agree.
            let reader_tx = mgr.begin(TenantId::DEFAULT);
            let bytes = reader_tx.read(node_mvcc_key(id)).unwrap();
            let rec = decode_node(&bytes);
            assert_ne!(
                rec.property_ref & crate::property::OVERFLOW_BIT,
                0,
                "node record must carry an overflow property_ref"
            );
            // property_ref = OVERFLOW_BIT | (page_id << OVERFLOW_SLOT_BITS) | slot_id
            let head_from_rec = (rec.property_ref & !crate::property::OVERFLOW_BIT)
                >> crate::property::OVERFLOW_SLOT_BITS;
            let slot_from_rec = rec.property_ref & crate::property::OVERFLOW_SLOT_MASK;
            assert!(
                slot_from_rec >= 1,
                "v2 M1: a packed bag's ref slot field is 1-based load-bearing (got 0 = chain)"
            );
            assert!(
                slotted_pages
                    .iter()
                    .any(|p| p.page_id == PageId::new(head_from_rec)),
                "the node's slotted prop page {head_from_rec} must ride the CommitBundle"
            );
            // And the bag round-trips byte-identical through the packed
            // representation (the M1 consistency anchor).
            let got = store
                .blobs
                .get(
                    TenantId::DEFAULT,
                    crate::property::BlobRef::decode(rec.property_ref).unwrap(),
                )
                .unwrap();
            assert_eq!(
                got.as_ref(),
                payload,
                "packed bag must round-trip byte-identical"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // W26-β-2 / ADR-131 — reverse adjacency (`scan_in`) tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn scan_in_default_reverse_index_enabled() {
        // The default-constructed CrudStore has the reverse index
        // enabled (the v1.1 posture). The hand-rolled `Default`
        // impl on CrudStore exists exactly to override
        // `AtomicBool::default()` (which would be `false`).
        let store = CrudStore::new();
        assert!(
            store.reverse_index_enabled(),
            "v1.1 default: reverse index MUST be enabled (AtomicBool::default() trap closed by hand-rolled Default impl)"
        );
    }

    #[test]
    fn scan_in_yields_entries_matching_forward_chain() {
        // For an edge src→dst, scan_out at src yields a TelEntry
        // whose dst_id = dst. scan_in at dst yields a TelEntry
        // whose dst_id = src (semantic re-purposing per ADR-131
        // §"reverse_tel_chains" rustdoc).
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let ty = TypeId::new(7);
        let rel = create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        commit(tx, &store).unwrap();

        let reader = mgr.begin(tenant);
        // Forward: scan_out at src yields TelEntry{ dst_id = dst, rel_id = rel }
        let out: Vec<TelEntry> = scan_out(&store, &reader, src, Some(ty)).collect();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dst_id, dst.raw());
        assert_eq!(out[0].rel_id, rel.raw());

        // Reverse: scan_in at dst yields TelEntry{ dst_id = src, rel_id = rel }
        let in_entries: Vec<TelEntry> =
            scan_in(&store, &reader, dst, Some(ty)).expect("scan_in succeeds");
        assert_eq!(in_entries.len(), 1);
        assert_eq!(
            in_entries[0].dst_id,
            src.raw(),
            "reverse entry's dst_id field semantically holds original src"
        );
        assert_eq!(in_entries[0].rel_id, rel.raw());
    }

    #[test]
    fn scan_in_cursor_fetches_overflow_predecessors_lazily() {
        use crate::tel::MAX_ENTRIES;

        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;
        let ty = TypeId::new(7);
        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let rel_count = (MAX_ENTRIES as usize * 2) + 1;
        for _ in 0..rel_count {
            create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        }
        commit(tx, &store).unwrap();

        let (_head_page, head) = store
            .reverse_tel_head(tenant, dst, LabelId::new(ty.raw()))
            .unwrap();
        let head_entries = head.entry_count() as usize;
        assert_eq!(head_entries, 1, "fixture should leave a one-entry head");

        let reader = mgr.begin(tenant);
        store.reset_reverse_tel_block_fetches_for_test();
        let mut cursor = scan_in_cursor(&store, &reader, dst, Some(ty)).unwrap();
        for _ in 0..head_entries {
            assert!(cursor.next_entry(&store, &reader).is_some());
        }
        assert_eq!(
            store.reverse_tel_block_fetches_for_test(),
            0,
            "opening and draining the reverse head must not fetch predecessors"
        );
        assert!(cursor.next_entry(&store, &reader).is_some());
        assert_eq!(
            store.reverse_tel_block_fetches_for_test(),
            1,
            "the first predecessor is fetched only at the head boundary"
        );
    }

    #[test]
    fn scan_in_surfaces_structured_error_when_reverse_index_disabled() {
        // W26-β-2 / ADR-131 AC-4: when the reverse index is disabled
        // (post-recovery or fault-injection), `scan_in` MUST return
        // `Err(ScanInError::ReverseIndexDisabled)` — never silent
        // empty. Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`.
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let ty = TypeId::new(7);
        let _rel = create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        commit(tx, &store).unwrap();

        // Disable the reverse index AFTER population.
        store.set_reverse_index_enabled(false);

        let reader = mgr.begin(tenant);
        let r = scan_in(&store, &reader, dst, Some(ty));
        assert_eq!(
            r,
            Err(ScanInError::ReverseIndexDisabled),
            "AC-4: disabled reverse index MUST surface structured error, not silent empty"
        );

        // Forward path remains operative.
        let out: Vec<TelEntry> = scan_out(&store, &reader, src, Some(ty)).collect();
        assert_eq!(
            out.len(),
            1,
            "LeftToRight path MUST be unaffected by reverse-index disable flag"
        );
    }

    #[test]
    fn scan_in_no_reverse_entries_written_when_disabled_at_commit_time() {
        // Sub-pattern of AC-4: when the reverse index is DISABLED
        // at the time of `create_rel` + `commit`, the reverse
        // append short-circuits and NO reverse chain is written.
        // Re-enabling the flag later does NOT retroactively
        // populate the reverse chain — the chain stays empty for
        // those rels.
        //
        // This pins the operational shape: the v1.1 toggle is
        // global-per-store + no-rebuild-on-flip; persisted index +
        // rebuild semantics is a v1.2+ extension per ADR-131
        // §"Forward-deferred to v1.2".
        let mgr = TxnManager::new();
        let store = CrudStore::new();
        store.set_reverse_index_enabled(false);
        let tenant = TenantId::DEFAULT;

        let mut tx = mgr.begin(tenant);
        let src = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let dst = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let ty = TypeId::new(7);
        let _rel = create_rel(&store, &mut tx, tenant, src, dst, ty, &PropertyData::Empty).unwrap();
        commit(tx, &store).unwrap();

        // The reverse chain at (dst, ty) was never created because
        // tel_append_reverse short-circuited. Re-enabling reads.
        store.set_reverse_index_enabled(true);
        let reader = mgr.begin(tenant);
        let in_entries: Vec<TelEntry> =
            scan_in(&store, &reader, dst, Some(ty)).expect("scan_in succeeds when enabled");
        assert!(
            in_entries.is_empty(),
            "reverse entries NOT retroactively populated by re-enabling the flag (v1.1 no-rebuild semantics)"
        );
    }
}
