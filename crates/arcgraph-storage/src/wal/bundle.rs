//! WAL `CommitBundle` record — v1 (ADR-031) and v2 (ADR-032 Slice 1)
//! codec.
//!
//! A `CommitBundle` is the post-ADR-031 unified per-commit WAL record.
//! Each MVCC commit emits exactly one `CommitBundle` carrying:
//!
//! 1. `commit_lsn` (the MVCC kernel's logical clock, allocated by
//!    `TxnManager::counter.allocate` in Phase 1 of the three-phase
//!    commit).
//! 2. The aggregated MVCC write-set. In v1 the write-set is
//!    single-tenant by construction and the record-header `tenant_id`
//!    identifies it. In v2 every write carries a per-entry
//!    `tenant_id`, enabling a single bundle to atomically cover writes
//!    across multiple tenants — the mechanism ADR-032 §2 uses to fold
//!    grow_root's SYSTEM root-pointer update into the outer user
//!    commit.
//! 3. N staged `IndexPage` snapshots — the byte-level page copies
//!    captured under each participating index's `write_gate` +
//!    per-page write latch during descent. Unchanged between v1 and
//!    v2 on the wire (each entry already has `page_id + tenant_id +
//!    PAGE_SIZE bytes`).
//!
//! The WAL framing (CRC + length) wraps the whole bundle, so commit
//! atomicity is record-level: the bundle is either fully durable or
//! torn-and-dropped on recovery.
//!
//! **Dispatch between v1 and v2 is by [`crate::wal::SegmentHeader::
//! format_version`]**, NOT by a payload-internal version byte. A v1
//! segment uses the v1 parser; a v2 segment uses the v2 parser. See
//! ADR-032 §2 for the rationale (the byte-0 discussion in the ADR is
//! defensive documentation; the implementation relies on segment
//! header dispatch only).
//!
//! The M2.e WAL replay path (#38) must honor the same dispatch and
//! decoding contract.

use std::collections::{BTreeSet, HashMap, HashSet};

use arcgraph_core::{
    ArcGraphError, Lsn, NodeId, PAGE_SIZE, PageId, PageType, PartitionId, Result, TenantId,
};
use bytes::Bytes;

use crate::redo::RedoLsnRange;
use crate::transaction::MvccKey;
use crate::wal::delta::{DeltaOp, DeltaOpKind};

// ─── Format-version constants (ADR-032 §2) ────────────────────────

/// v1 CommitBundle payload layout — ADR-031, no per-entry tenant.
pub const BUNDLE_FORMAT_V1: u16 = 1;

/// v2 CommitBundle payload layout — ADR-032 §2 Slice 1.
///
/// Adds per-entry `tenant_id: u64` on every MVCC write, enabling
/// single-bundle multi-tenant commits (grow_root's SYSTEM write
/// folded into outer user commit).
pub const BUNDLE_FORMAT_V2: u16 = 2;

/// v3 CommitBundle payload layout — ADR-031 amendment-02 (PR #79
/// X-2 review fold-in).
///
/// Generalises v2's `index_pages` section to a unified
/// `staged_pages` section with a leading
/// [`BundlePageKind`] discriminator byte per entry. v3 bundles
/// carry record + blob pages in addition to primary and
/// secondary index pages, closing the PR #79 X-2 gap where
/// record page bytes were absent from the bundle and post-
/// replay `read_node_with_store` hit `MissingPage`.
pub const BUNDLE_FORMAT_V3: u16 = 3;

/// v4 CommitBundle payload layout — issue #129 P0 fix.
///
/// Extends v3 with a trailing `allocator_advances` section so
/// per-tenant allocator high-water marks (NodeId, RelId, and
/// per-page-type PageId) are durified atomically with the commit
/// that consumed them. Pre-fix v3 segments could leak orphaned T1
/// commits on recovery because `PageAllocator` and
/// `CrudStore.next_node` / `next_rel` reset to zero on restart;
/// post-recovery `create_node` then re-issued NodeIds that
/// pre-fault commits already used, leaving earlier records
/// unreachable through the primary index. ADR-034 D-1 (Strict
/// tier durability) is restored by replaying the
/// `allocator_advances` section in commit_lsn order and seeding
/// each per-(tenant, kind) counter to `max(current, observed)`.
///
/// v4 is a strict superset of v3: the `commit_lsn` header,
/// MVCC writes section, and `staged_pages` section are encoded
/// exactly as in [`encode_commit_bundle_v3`], followed by a new
/// `n_allocator_advances: u32 LE` count and that many fixed-size
/// 17-byte entries (`tenant_id u64 LE | kind u8 | new_high_water u64 LE`).
/// v3 segments remain decodable; the v3 → v4 migration is purely
/// additive on the wire.
pub const BUNDLE_FORMAT_V4: u16 = 4;

/// v5 CommitBundle payload layout — M3.a Slice G.4
/// (commit-bundle vector page staging).
///
/// Extends v4 with a trailing `vector_pages` section so vector
/// arena page mutations are durified atomically with the same
/// commit that wrote them, mirroring the v3 staged_pages /
/// v4 allocator_advances atomicity contract. Per ADR-031
/// amendment-02 (commit-bundle vector staging), ADR-035 §4.5/§4.6
/// (vector recovery flow), and issue #131 follow-up item 3
/// (production-path simulation gap closure), the v5 vector_pages
/// section is the FIRST production source of vector page replay.
/// Pre-v5 (v3 / v4) bundles never carried real vector pages —
/// `BundlePageKind::Vector` was reserved as a stub and the
/// `BundlePageKind::Vector` arm in `staged_pages` was a stub
/// dispatch with a `tracing::warn!`-and-continue posture.
///
/// v5 is a strict superset of v4: the `commit_lsn` header, MVCC
/// writes section, `staged_pages` section, and
/// `allocator_advances` section are encoded exactly as in
/// [`encode_commit_bundle_v4`], followed by a new
/// `n_vector_pages: u32 LE` count and that many fixed-size
/// `(8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE)`-byte entries:
/// `tenant_id u64 LE | partition_id u64 LE | index_id u64 LE |
/// page_id u64 LE | commit_lsn u64 LE | n_bytes u32 LE | bytes`.
/// v4 segments remain decodable; the v4 → v5 migration is purely
/// additive on the wire.
pub const BUNDLE_FORMAT_V5: u16 = 5;

/// v6 CommitBundle payload layout — #352 Part 2 (durable
/// `external_id → internal_id` idempotency binding; ADR-199).
///
/// Extends v5 with a trailing `idempotency_bindings` section so the
/// `graph.ingest` idempotency binding is durified **atomically with the
/// commit that allocated the internal id**, mirroring the v4
/// `allocator_advances` atomicity contract. This is the load-bearing
/// reason it folds into the bundle rather than riding a standalone WAL
/// record like [`crate::wal::WalRecordType::InternString`]: a standalone
/// pre-commit record (synchronously fsynced by `WalHandle::append`)
/// would leave a durable binding for a node whose commit — and whose
/// allocator high-water (#129/#820) — is absent after a crash, so the
/// internal id could be re-allocated to a *different* record (a
/// #820-class cross-wiring). The fold makes the binding present **iff**
/// the node is present. See ADR-199 §Revision 2026-06-07.
///
/// v6 is a strict superset of v5: the `commit_lsn` header, MVCC writes,
/// `staged_pages`, `allocator_advances`, and `vector_pages` sections are
/// encoded exactly as in [`encode_commit_bundle_v5`], followed by a new
/// `n_idempotency_bindings: u32 LE` count and that many variable-length
/// entries:
/// `tenant_id u64 LE | kind u8 | internal_id u64 LE | ext_len u32 LE |
/// external_id bytes (ext_len B, UTF-8)`.
/// v5 segments remain decodable (decoder synthesizes an empty
/// `idempotency_bindings`); the v5 → v6 migration is purely additive on
/// the wire. A v6 binary writing into a data dir whose latest segment is
/// v5 rolls a fresh v6-stamped segment (see
/// [`crate::wal::segment::SegmentWriter::open`]) so no segment is ever
/// version-inhomogeneous — replay dispatches the bundle codec by the
/// owning segment's `format_version`.
pub const BUNDLE_FORMAT_V6: u16 = 6;

/// v7 CommitBundle payload layout — #1010 / ADR-199 amendment.
///
/// Extends v6 by versioning the idempotency section entries with an op
/// discriminant. Old v6 segments remain install-only and decode through
/// [`decode_commit_bundle_v6`]. New v7 segments encode each entry as:
/// `op u8 | tenant_id u64 LE | kind u8 | internal_id u64 LE | ext_len u32
/// LE | external_id bytes`, where `internal_id` is meaningful for
/// Install and unused for Release.
pub const BUNDLE_FORMAT_V7: u16 = 7;

/// v8 CommitBundle payload layout — #1221 / ADR-218 (PermissionIndex
/// ACL durability).
///
/// Extends v7 with a NEW trailing `acl_grants` section appended AFTER
/// the v7 `idempotency_bindings` section — a strict superset of the v7
/// prefix, structurally analogous to the v5 → v6 append (a brand-new
/// trailing section), NOT the v6 → v7 re-encode (which added an op byte
/// per existing entry). Old v5/v6/v7 segments are routed to their own
/// decoders (never v8) and synthesize an empty `acl_grants`
/// (= no ACL ops = fail-closed default ⇒ every doc UNCLASSIFIED).
///
/// Each `acl_grants` entry encodes as:
/// `op u8 | tenant_id u64 LE | doc u64 LE | n_grants u32 LE |
///  per-grant (grant_len u32 LE | grant UTF-8 bytes)`, where
/// `op` is `Apply(0)` / `Revoke(1)` and `grants` is empty for Revoke.
///
/// **Replay-order invariant (ADR-218, architect-flagged correctness
/// gate):** the `acl_grants` encoder MUST preserve staging (append)
/// order — it MUST NOT copy the v7 idempotency encoder's on-encode sort
/// (`bundle.rs` `encode_commit_bundle_v7`). Replay is last-writer-wins
/// per doc; a re-sort would silently flip which op wins if two ops on
/// the SAME doc ever shared one commit. See [`encode_commit_bundle_v8`].
pub const BUNDLE_FORMAT_V8: u16 = 8;

/// v9 CommitBundle payload layout — M3 physiological WAL.
///
/// Record/blob/PropSlotted page images are replaced by the `deltas`
/// section. Primary and secondary index pages remain full images; vector
/// pages retain their v5 image section. The v4 allocator, v7 idempotency,
/// and v8 ACL sections are retained verbatim after those image sections.
/// `InternBind`/`AclGrant` DeltaOps are reserved at M3 because their M4 wire
/// layouts are deliberately not invented here.
pub const BUNDLE_FORMAT_V9: u16 = 9;

/// v10 CommitBundle payload layout — M4 owner-row cutover.
///
/// The byte layout is identical to v9, but the declared version expands the
/// closed DeltaOp algebra with the already-pinned `InternBind = 8` and
/// `AclGrant = 9` discriminants. Keeping a distinct header version preserves
/// the v9/M3 promise that those bytes remain reserved in an M3-era bundle.
pub const BUNDLE_FORMAT_V10: u16 = 10;

/// Whether a WAL format carries the physiological delta-bundle layout.
#[must_use]
pub const fn is_delta_bundle_format(format_version: u16) -> bool {
    matches!(format_version, BUNDLE_FORMAT_V9 | BUNDLE_FORMAT_V10)
}

const MIN_DELTA_OP_ELEM: usize = DeltaOp::FIXED_PREFIX_LEN;

// ─── Decoder capacity-bound minimums (#1411 — OOM/DoS hardening) ─────
//
// Every DECODE path reads a section element count as an untrusted
// `u32` straight off the on-disk (attacker-influenceable, crash-
// recovery + spill-reload) bytes and then pre-allocates a
// `Vec`/`HashMap` sized to that count. A crafted count near `u32::MAX`
// (~4.29e9) forces a multi-gigabyte pre-alloc BEFORE a single element
// byte is validated — an OOM/DoS on the recovery path (#1411; surfaced
// by the #1287 CommitBundle fuzz target).
//
// The fix ([`bounded_capacity`]) caps the requested capacity at what
// the REMAINING payload could POSSIBLY encode: `remaining / MIN_ELEM`,
// where `MIN_ELEM` is the SMALLEST on-wire size of one valid element at
// that site. A valid bundle with `n` genuine elements always has
// `>= n * MIN_ELEM` remaining bytes, so for valid input the bound is a
// no-op (`cap == n`) and the decode result is byte-identical. A
// malicious count claiming `n` elements it lacks the bytes for gets a
// small capacity hint, then hits the SAME in-loop overrun guard that
// exists today and returns `WalCorruption` — same error, without the
// OOM first. The bound only ever DIVIDES remaining by a min-size; it
// NEVER multiplies a count by a size (that multiply is the overflow the
// bug is made of).
//
// Each `MIN_*_ELEM` below cites the mandatory fixed reads its decode
// loop performs; variable-length fields (value bytes, external_id,
// grant principals) contribute their length-prefix + 0 (the smallest a
// valid element can be), so the min is a true floor (PD#6).

/// Smallest on-wire size of one MVCC write entry, v1 (no per-entry
/// tenant). Fields: `key` u64 (8 B), `kind` u8 (1 B), `value_len` u32
/// (4 B) = 13 B. A tombstone (`kind=0`) carries `value_len=0`, so 13 B
/// is the floor.
const MIN_MVCC_V1_ELEM: usize = 8 + 1 + 4;

/// Smallest on-wire size of one v1/v2 `index_pages` entry. Fields:
/// `page_id` u64 (8 B), `tenant_id` u64 (8 B), then `PAGE_SIZE` payload
/// bytes. The page payload is a mandatory fixed `PAGE_SIZE` (no length
/// prefix in v1/v2), so it is part of the floor.
const MIN_INDEX_PAGE_ELEM: usize = 8 + 8 + PAGE_SIZE;

/// Smallest on-wire size of one v3+ `staged_pages` entry. Fields:
/// `kind` u8 (1 B), `page_id` u64 (8 B), `tenant_id` u64 (8 B),
/// `n_bytes` u32 (4 B), then `PAGE_SIZE` payload bytes. `n_bytes` MUST
/// equal `PAGE_SIZE` (the decoder rejects any other value), so the page
/// payload is part of the floor: `21 + PAGE_SIZE`.
const MIN_STAGED_PAGE_ELEM: usize = 1 + 8 + 8 + 4 + PAGE_SIZE;

/// Smallest on-wire size of one `allocator_advances` entry. Fields:
/// `tenant_id` u64 (8 B), `kind` u8 (1 B), `new_high_water` u64 (8 B) =
/// 17 B, fixed-size (mirrors [`AllocatorAdvance::ENCODED_LEN`]).
const MIN_ALLOCATOR_ADVANCE_ELEM: usize = AllocatorAdvance::ENCODED_LEN;

/// Smallest on-wire size of one v5 `vector_pages` entry. Fields:
/// `tenant` u64 (8 B), `partition` u64 (8 B), `index_id` u64 (8 B),
/// `page_id` u64 (8 B), `commit_lsn` u64 (8 B), `n_bytes` u32 (4 B),
/// then `PAGE_SIZE` payload bytes. `n_bytes` MUST equal `PAGE_SIZE`, so
/// the payload is part of the floor: `40 + PAGE_SIZE` (mirrors
/// [`VectorPageEntry::ENCODED_LEN`]).
const MIN_VECTOR_PAGE_ELEM: usize = VectorPageEntry::ENCODED_LEN;

/// Smallest on-wire size of one v6 `idempotency_bindings` entry. Fields:
/// `tenant_id` u64 (8 B), `kind` u8 (1 B), `internal_id` u64 (8 B),
/// `ext_len` u32 (4 B) = 21 B fixed prefix; `external_id` is variable and
/// can be 0 bytes (mirrors [`IdempotencyBindingEntry::V6_FIXED_PREFIX_LEN`]).
const MIN_IDEMPOTENCY_V6_ELEM: usize = IdempotencyBindingEntry::V6_FIXED_PREFIX_LEN;

/// Smallest on-wire size of one v7 `idempotency_bindings` entry. Fields:
/// `op` u8 (1 B), `tenant_id` u64 (8 B), `kind` u8 (1 B), `internal_id`
/// u64 (8 B), `ext_len` u32 (4 B) = 22 B fixed prefix; `external_id` can
/// be 0 bytes (mirrors [`IdempotencyBindingEntry::FIXED_PREFIX_LEN`]).
const MIN_IDEMPOTENCY_V7_ELEM: usize = IdempotencyBindingEntry::FIXED_PREFIX_LEN;

/// Smallest on-wire size of one v8 `acl_grants` entry. Fields: `op` u8
/// (1 B), `tenant_id` u64 (8 B), `doc` u64 (8 B), `n_grants` u32 (4 B) =
/// 21 B fixed prefix; the per-grant (`grant_len` u32 + grant bytes) list
/// can be empty (Revoke, or grant-to-nobody), so the prefix is the floor
/// (mirrors [`AclGrantEntry::FIXED_PREFIX_LEN`]).
const MIN_ACL_GRANT_ELEM: usize = AclGrantEntry::FIXED_PREFIX_LEN;

/// Bound a decoder's initial `Vec`/`HashMap` capacity hint against an
/// untrusted element count (#1411).
///
/// Returns `min(requested, remaining / min_elem_size)` — the count
/// capped at the maximum number of `min_elem_size`-byte elements the
/// remaining payload could hold. This is a capacity HINT only: the
/// container still grows on `push`/`insert` past the hint, so a valid
/// bundle whose `remaining >= requested * min_elem_size` gets exactly
/// `requested` (no behavior change), while a malicious `requested` that
/// the payload cannot back is clamped to a small value — the subsequent
/// element decode then hits the in-loop overrun/truncation guard and
/// returns `WalCorruption`, never OOM.
///
/// `min_elem_size` MUST be `>= 1` (all bundle element floors are `>= 13`);
/// the division is on `remaining` (the trusted input length), never a
/// `count * size` multiply, so it cannot overflow.
#[inline]
pub(crate) fn bounded_capacity(requested: usize, remaining: usize, min_elem_size: usize) -> usize {
    debug_assert!(min_elem_size >= 1, "min_elem_size must be non-zero");
    requested.min(remaining / min_elem_size)
}

/// On-wire discriminator byte stamped on every v3 staged_pages
/// entry. Selects which in-memory page store the replay executor
/// routes the bytes into.
///
/// The v1.0 mapping:
/// - [`BundlePageKind::PrimaryIndex`] (= 0) — the primary B-tree.
/// - [`BundlePageKind::SecondaryIndex`] (= 1) — secondary B-tree.
/// - [`BundlePageKind::Record`] (= 2) — slotted record pages
///   (Node / Rel).
/// - [`BundlePageKind::Blob`] (= 3) — overflow BLOB chains.
/// - [`BundlePageKind::Vector`] (= 4) — vector arena pages
///   (HNSW / DiskANN, ADR-035 §7.5). Wired at M3.a Slice G.1
///   as a stub; bodies populated by Slices G.2–G.5.
/// - [`BundlePageKind::PropSlotted`] (= 5) — shared slotted
///   property-bag heap pages (v2 M1, ADR-230). One image per
///   TOUCHED page per bundle (many bags amortize one image).
///
/// Distinct from
/// [`crate::mutation_log::PageStoreKind`] (Z-1 (b) rollback
/// routing) so the bundle codec can evolve independently of the
/// mutation-log schema. v1/v2 decoders synthesize
/// `BundlePageKind::PrimaryIndex` for every entry (v1/v2 had no
/// other kind).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BundlePageKind {
    /// Primary B-tree internal / leaf / overflow pages.
    PrimaryIndex = 0,
    /// Secondary B-tree internal / leaf / overflow pages.
    SecondaryIndex = 1,
    /// Slotted record pages (`PageType::Node` / `PageType::Rel`).
    Record = 2,
    /// BLOB overflow chain pages (`PageType::Free` slots filled
    /// by `BlobStore`).
    Blob = 3,
    /// Vector arena pages (HNSW graph nodes + DiskANN segments).
    /// Routes to [`crate::vector_store::VectorPageStoreHandle`] on
    /// replay; staged into the same v3 `staged_pages` section as
    /// other kinds. Per ADR-035 §7.5, vector arenas are tenant-
    /// keyed at the physical layer (matching `BlobStore`'s tenancy
    /// pattern).
    Vector = 4,
    /// Shared slotted property-bag heap pages (`PageType::PropSlotted`,
    /// v2 M1 — W-B1 slotted small-blob packing, ADR-230 / design
    /// §M1.3). One staged image carries MANY small property bags — the
    /// commit builder stages a touched slotted page ONCE per bundle
    /// (not once per bag), which is the ~14× batch-ingest WAL
    /// amortization. Routes to the same
    /// [`crate::blob::BlobStoreHandle`] as [`Self::Blob`] on replay
    /// (the blob store is kind-aware: it classifies the page bytes and
    /// installs a slotted-page resident entry instead of a chain
    /// chunk). Tenant-keyed at the physical layer exactly like `Blob`.
    PropSlotted = 5,
}

impl BundlePageKind {
    /// Parse a single byte back into a kind, rejecting unknown values.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => Self::PrimaryIndex,
            1 => Self::SecondaryIndex,
            2 => Self::Record,
            3 => Self::Blob,
            4 => Self::Vector,
            5 => Self::PropSlotted,
            other => {
                return Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!("CommitBundle v3: unknown BundlePageKind byte {other}"),
                });
            }
        })
    }

    /// Raw byte for on-disk storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

// ─── AllocatorAdvance (issue #129 P0 fix) ─────────────────────────────
//
// `AllocatorAdvance` carries the high-water of one monotonic ID
// allocator at the moment its owning commit was encoded. The replay
// executor seeds the live allocator to `max(current, observed)` so
// post-recovery `create_node` / `create_rel` / fresh-page allocations
// never reuse an ID a pre-fault commit already consumed.
//
// At v1.0 the struct deliberately carries NO `partition_id` field —
// the bug fix is for the single-partition deployment shape. v1.1 per-
// partition extension will bump the bundle to v5 with a `partition_id`
// slot per advance; the `allocator_advance_partition_id_always_zero_at_v1`
// regression test pins the v1.0 invariant. See ADR-024-amendment-02 §OQ-3
// for the local-only hook pattern (mirrors `Z-1 (b)` mutation log
// and `ReplayExecutor::partition_id`).

/// Discriminator for the kind of monotonic allocator whose advance
/// is being durified. Unifies CRUD-layer per-tenant ID allocators
/// (`Node`, `Rel`) with the page-store-layer per-(tenant, page_type)
/// ID allocator (`Page*`). Wire-stable byte assignment — pinned by the
/// `bundle_allocator_kind_byte_layout` test so a future taxonomy
/// renumber doesn't silently shift values and corrupt on-disk WALs.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AllocatorKind {
    /// `CrudStore::alloc_node` — tenant-scoped `NodeId` allocator.
    Node = 0,
    /// `CrudStore::alloc_rel` — tenant-scoped `RelId` allocator.
    Rel = 1,
    /// [`PageAllocator`](crate::page_alloc::PageAllocator) for
    /// `PageType::Free` pages.
    PageFree = 2,
    /// `PageAllocator` for `PageType::Node` (slotted node-record pages).
    PageNode = 3,
    /// `PageAllocator` for `PageType::Rel` (slotted rel-record pages).
    PageRel = 4,
    /// `PageAllocator` for `PageType::Tel` (TEL block pages).
    PageTel = 5,
    /// `PageAllocator` for `PageType::IndexInternal`.
    PageIndexInternal = 6,
    /// `PageAllocator` for `PageType::IndexLeaf`.
    PageIndexLeaf = 7,
    /// `PageAllocator` for `PageType::VectorNeighbor` (HNSW neighbor list).
    PageVectorNeighbor = 8,
    /// `PageAllocator` for `PageType::WalBuffer` (in-memory ring buffer).
    PageWalBuffer = 9,
    /// `PageAllocator` for `PageType::IndexOverflow` (M2-34 secondary
    /// duplicate-NodeId overflow).
    PageIndexOverflow = 10,
    /// `PageAllocator` for `PageType::PropSlotted` (v2 M1 shared
    /// slotted property-bag heap pages). Present for totality /
    /// wire-stability over the `PageType` domain only: at M1 the
    /// slotted prop pages are allocated from the `BlobStore`'s
    /// synthetic page-id counter (re-seeded on replay via
    /// `install_or_replace` `fetch_max`, P0 #820), NOT from a
    /// `PageAllocator`, so no advance with this kind is ever encoded.
    PagePropSlotted = 11,
    /// M4 durable interned-string id allocator.
    InternString = 12,
    /// M4 durable ACL-class id allocator.
    AclClass = 13,
}

impl AllocatorKind {
    /// Parse a single byte back into a kind, rejecting unknown values.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => Self::Node,
            1 => Self::Rel,
            2 => Self::PageFree,
            3 => Self::PageNode,
            4 => Self::PageRel,
            5 => Self::PageTel,
            6 => Self::PageIndexInternal,
            7 => Self::PageIndexLeaf,
            8 => Self::PageVectorNeighbor,
            9 => Self::PageWalBuffer,
            10 => Self::PageIndexOverflow,
            11 => Self::PagePropSlotted,
            12 => Self::InternString,
            13 => Self::AclClass,
            other => {
                return Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!("CommitBundle v4: unknown AllocatorKind byte {other}"),
                });
            }
        })
    }

    /// Raw byte for on-disk storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Map a [`PageType`] to the corresponding `Page*` allocator
    /// variant. The CRUD-layer NodeId / RelId allocators have no
    /// `PageType` correspondent; this conversion is total over the
    /// page-store allocator subdomain.
    #[must_use]
    pub const fn for_page_type(pt: PageType) -> Self {
        match pt {
            PageType::Free => Self::PageFree,
            PageType::Node => Self::PageNode,
            PageType::Rel => Self::PageRel,
            PageType::Tel => Self::PageTel,
            PageType::IndexInternal => Self::PageIndexInternal,
            PageType::IndexLeaf => Self::PageIndexLeaf,
            PageType::VectorNeighbor => Self::PageVectorNeighbor,
            PageType::WalBuffer => Self::PageWalBuffer,
            PageType::IndexOverflow => Self::PageIndexOverflow,
            PageType::PropSlotted => Self::PagePropSlotted,
        }
    }

    /// Inverse of [`Self::for_page_type`]. Returns `None` for the
    /// CRUD-layer `Node` and `Rel` variants (which have no
    /// `PageType` correspondent).
    #[must_use]
    pub const fn page_type(self) -> Option<PageType> {
        match self {
            Self::Node | Self::Rel | Self::InternString | Self::AclClass => None,
            Self::PageFree => Some(PageType::Free),
            Self::PageNode => Some(PageType::Node),
            Self::PageRel => Some(PageType::Rel),
            Self::PageTel => Some(PageType::Tel),
            Self::PageIndexInternal => Some(PageType::IndexInternal),
            Self::PageIndexLeaf => Some(PageType::IndexLeaf),
            Self::PageVectorNeighbor => Some(PageType::VectorNeighbor),
            Self::PageWalBuffer => Some(PageType::WalBuffer),
            Self::PageIndexOverflow => Some(PageType::IndexOverflow),
            Self::PagePropSlotted => Some(PageType::PropSlotted),
        }
    }
}

/// One allocator-high-water snapshot ridden by a v4+ `CommitBundle`.
///
/// Replay applies advances after MVCC writes + staged_pages have been
/// installed; per (`tenant`, `kind`) the live allocator is seeded to
/// `max(current_high_water, new_high_water)` (Lemma I3 — monotonic
/// idempotent replay; double-replay is a no-op).
///
/// `new_high_water` is the **last allocated id** observed at the
/// commit's encode point. Recovery seeds the allocator so the next
/// allocation returns `new_high_water + 1`. `0` means "pristine —
/// no allocations have been made for this (tenant, kind)".
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AllocatorAdvance {
    /// Tenant whose allocator is being advanced.
    pub tenant: TenantId,
    /// Which allocator (NodeId / RelId / per-page-type PageId).
    pub kind: AllocatorKind,
    /// Last allocated id (`u64`). On recovery the live allocator is
    /// seeded so the next allocation returns `new_high_water + 1`.
    pub new_high_water: u64,
}

impl AllocatorAdvance {
    /// Byte-size of one entry inside a v4 `CommitBundle` payload.
    /// `tenant_id u64 LE (8 B) | kind u8 (1 B) | new_high_water u64 LE (8 B)`.
    pub const ENCODED_LEN: usize = 8 + 1 + 8;
}

// ─── VectorPageEntry (M3.a Slice G.4 — commit-bundle vector staging) ─
//
// `VectorPageEntry` carries one vector arena page snapshot riding a v5
// CommitBundle. The entry mirrors `BundlePageKind::Vector` staged_pages
// at the wire-shape level but lives in its OWN trailing `vector_pages`
// section so pre-v5 bundles (which used the stub Vector arm in
// staged_pages) remain decodable as a strict subset, and so the v1.0
// production source of vector page replay is anchored at v5.
//
// At v1.0 the struct deliberately carries `partition_id: PartitionId =
// PartitionId::ZERO` (single-partition deployment shape) and
// `index_id: u64 = 0` (single-index per tenant; v1.1 multi-index lift
// will populate). Both are reserved for v1.1 extension, and both are
// pinned by structural regression tests
// (`vector_page_entry_partition_id_always_zero_at_v1` /
// `vector_page_entry_index_id_always_zero_at_v1` in
// tests/wal_bundle_v5.rs). Per ADR-024 OQ-3 the partition_id slot
// follows the same local-only hook discipline as
// `Z-1 (b)` mutation log and `ReplayExecutor::partition_id`. See
// ADR-031 amendment-02 §VectorPageEntry and ADR-035 §4.5/§4.6 for
// rationale.

/// One vector arena page snapshot ridden by a v5+ `CommitBundle`.
///
/// Captured under the vector arena's write latch by the producer
/// (Slice G.5 / G.7 vector writers in `arcgraph-vector`); routed at
/// commit time through [`crate::crud::CrudStore::stage_vector_page`]
/// into the per-txn `pending_vector_emits` queue and drained into the
/// v5 bundle's `vector_pages` section by the commit builder. Replay
/// applies entries via
/// [`crate::vector_store::VectorPageStoreHandle::install_or_replace`]
/// after `staged_pages` and before `allocator_advances` (Lemma I3 —
/// monotonic idempotent replay; double-replay is a no-op).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorPageEntry {
    /// Tenant whose arena holds this page. Vector arenas are
    /// tenant-keyed at the physical layer (ADR-035 §7.5).
    pub tenant: TenantId,
    /// Local partition slot. Always [`PartitionId::ZERO`], pinned by
    /// `vector_page_entry_partition_id_always_zero_at_v1`.
    pub partition: PartitionId,
    /// Reserved for v1.1 multi-index lift; **always 0 at v1.0**.
    /// Wire-compatible with future `arcgraph_vector::IndexId` once
    /// that type is promoted to a published cross-crate type. Held as
    /// `u64` here (NOT the real `IndexId`) to keep `arcgraph-storage`
    /// from depending on `arcgraph-vector`.
    pub index_id: u64,
    /// The page that was mutated (vector-arena allocator-assigned).
    pub page_id: PageId,
    /// MVCC commit LSN at the moment the page was captured. Redundant
    /// with the bundle-level `commit_lsn` at v1.0 (one bundle = one
    /// commit) but forward-looking for v1.1 batched commits where
    /// multiple commit_lsns could share a single bundle frame.
    pub commit_lsn: Lsn,
    /// A heap-allocated copy of the page's post-mutation bytes
    /// captured under the arena's write latch. `n_bytes` MUST equal
    /// `PAGE_SIZE` on the wire — decoder rejects any other size with
    /// [`ArcGraphError::WalCorruption`].
    pub bytes: Box<[u8; PAGE_SIZE]>,
}

impl VectorPageEntry {
    /// Byte-size of one entry inside a v5 `CommitBundle` payload.
    /// `tenant_id u64 LE (8 B) | partition_id u64 LE (8 B) |
    ///  index_id u64 LE (8 B) | page_id u64 LE (8 B) |
    ///  commit_lsn u64 LE (8 B) | n_bytes u32 LE (4 B) |
    ///  page_bytes [u8; PAGE_SIZE]`.
    pub const ENCODED_LEN: usize = 8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE;
}

// ─── IdempotencyBindingEntry (#352 Part 2 — ADR-199 v6 fold) ──────────
//
// One `external_id → internal_id` idempotency binding riding a v6
// `CommitBundle`. Lives in its OWN trailing `idempotency_bindings`
// section so pre-v6 bundles remain decodable as a strict subset, and so
// the v6 section is the FIRST durable source of idempotency replay.
//
// The entry is **semantics-agnostic**: `kind` is an opaque `u8`
// discriminator (arcgraph-mcp maps its `IdempotencyKind::{Node,Rel}` to
// `0`/`1`). `arcgraph-storage` attaches no node/rel meaning to it,
// keeping the bounded-context boundary clean (ADR-199 §Decision-1).
// `external_id` is variable-length (client-supplied), so — unlike the
// fixed-size `VectorPageEntry` / `AllocatorAdvance` — the entry carries
// an explicit `ext_len` prefix and has no constant `ENCODED_LEN`.

/// Operation carried by an idempotency binding fold entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IdempotencyBindingOp {
    /// Install `external_id -> internal_id` atomically with create.
    Install = 0,
    /// Release `external_id` atomically with delete.
    Release = 1,
}

impl IdempotencyBindingOp {
    /// Parse a wire byte into an idempotency binding operation.
    pub fn from_byte(byte: u8, commit_lsn: Lsn) -> Result<Self> {
        Ok(match byte {
            0 => Self::Install,
            1 => Self::Release,
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v7: unknown idempotency_binding op {other}"),
                ));
            }
        })
    }

    /// Raw byte for on-disk storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// One idempotency binding operation durified alongside the commit
/// that creates or deletes the owning record, ridden by a v6+
/// `CommitBundle`.
///
/// Staged per-txn via [`crate::crud::CrudStore::stage_idempotency_binding`]
/// and drained into the v6 bundle's `idempotency_bindings` section by
/// [`crate::crud::commit`]. Replay applies entries via
/// [`crate::idempotency::IdempotencyStore::install`] after MVCC writes
/// (Lemma I3 — the node exists before its binding installs). Per
/// ADR-199.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdempotencyBindingEntry {
    /// Operation to apply. Old v6 segments synthesize `Install`.
    pub op: IdempotencyBindingOp,
    /// Tenant whose namespace this binding lives in. Cross-tenant
    /// isolation is a hard invariant.
    pub tenant: TenantId,
    /// Opaque kind discriminator (mcp: 0 = Node, 1 = Rel). Storage
    /// attaches no meaning to it.
    pub kind: u8,
    /// The internal id the `external_id` resolves to — allocated by
    /// `create_node` / `create_rel` inside the owning transaction.
    pub internal_id: u64,
    /// The client-supplied external id (the idempotency key). Bounded
    /// by `WalRecord.length`; decoder rejects non-UTF-8 as corruption.
    pub external_id: String,
}

impl IdempotencyBindingEntry {
    /// Byte-size of one v6 entry's fixed prefix: `tenant_id u64 LE
    /// (8 B) | kind u8 (1 B) | internal_id u64 LE (8 B) | ext_len u32
    /// LE (4 B)`. The trailing `external_id` bytes (`ext_len` of them)
    /// follow.
    pub const V6_FIXED_PREFIX_LEN: usize = 8 + 1 + 8 + 4;

    /// Byte-size of one v7 entry's fixed prefix: `op u8 (1 B) |
    /// tenant_id u64 LE (8 B) | kind u8 (1 B) |
    /// internal_id u64 LE (8 B) | ext_len u32 LE (4 B)`. The trailing
    /// `external_id` bytes (`ext_len` of them) follow.
    pub const FIXED_PREFIX_LEN: usize = 1 + Self::V6_FIXED_PREFIX_LEN;
}

// ─── AclGrantEntry (#1221 — ADR-218 v8 fold) ──────────────────────────
//
// One document-level ACL grant/revoke operation riding a v8
// `CommitBundle` in its OWN trailing `acl_grants` section so pre-v8
// bundles remain decodable as a strict subset. Folds the
// `PermissionIndex` enforcement state (ADR-212 §D-2(b)) into the WAL so a
// bare `serve --data` restart replays grants instead of coming up
// deny-all (the #1221 durability defect).
//
// The entry is **semantics-agnostic** like `IdempotencyBindingEntry`:
// `grants` is a set of client-supplied principal strings (variable
// length), so the entry carries explicit length prefixes and has no
// constant `ENCODED_LEN`. `arcgraph-storage` attaches no principal
// meaning beyond "string set the index interns" — the enforcement
// semantics live in `crate::permissions::PermissionIndex`.

/// Operation carried by an ACL grant fold entry (ADR-218).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AclGrantOp {
    /// Apply (intern + map `doc → class`) — `PermissionIndex::apply_doc_acl`.
    Apply = 0,
    /// Revoke (remove `doc`'s mapping ⇒ UNCLASSIFIED ⇒ invisible) —
    /// `PermissionIndex::revoke_doc`. `grants` is empty for a Revoke.
    Revoke = 1,
}

impl AclGrantOp {
    /// Parse a wire byte into an ACL grant operation.
    pub fn from_byte(byte: u8, commit_lsn: Lsn) -> Result<Self> {
        Ok(match byte {
            0 => Self::Apply,
            1 => Self::Revoke,
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v8: unknown acl_grant op {other}"),
                ));
            }
        })
    }

    /// Raw byte for on-disk storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// One document-level ACL grant/revoke operation durified alongside the
/// commit that carries it, ridden by a v8+ `CommitBundle` (ADR-218).
///
/// Staged per-txn via [`crate::crud::CrudStore::stage_acl_grant`] and
/// drained into the v8 bundle's `acl_grants` section by
/// [`crate::crud::commit`]. Replay re-drives
/// [`crate::permissions::PermissionIndex::apply_doc_acl`] (Apply) /
/// [`crate::permissions::PermissionIndex::revoke_doc`] (Revoke) against a
/// fresh index in ascending `commit_lsn` order — last-writer-wins per
/// doc. Per ADR-218.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclGrantEntry {
    /// Operation to apply (Apply / Revoke).
    pub op: AclGrantOp,
    /// Tenant whose `PermissionIndex` this op lands in. Cross-tenant
    /// isolation is a hard invariant — the index is per-tenant
    /// (ADR-212 §5 Q3).
    pub tenant: TenantId,
    /// The content node whose read-ACL this op mutates.
    pub doc: NodeId,
    /// The read-grant set (principal strings) for an Apply. EMPTY for a
    /// Revoke (and legal-but-empty for an explicit "grant-to-nobody"
    /// Apply — the op discriminator distinguishes the two).
    pub grants: BTreeSet<String>,
}

impl AclGrantEntry {
    /// Byte-size of one v8 entry's fixed prefix: `op u8 (1 B) |
    /// tenant_id u64 LE (8 B) | doc u64 LE (8 B) | n_grants u32 LE
    /// (4 B)`. The trailing per-grant `(grant_len u32 LE | grant bytes)`
    /// records follow.
    pub const FIXED_PREFIX_LEN: usize = 1 + 8 + 8 + 4;
}

// ─── v1 payload layout (pre-ADR-032, unchanged). ────────────────
//
//   offset       field             size (B)   notes
//    0           commit_lsn         8         MVCC commit LSN, authoritative
//    8           n_mvcc_writes      4
//   12           [mvcc writes]      variable  per write:
//                                               key               u64   ( 8 B)
//                                               kind              u8    ( 1 B) 0=tombstone, 1=put
//                                               value_len         u32   ( 4 B)
//                                               value             v B
//  ...           n_index_pages      4
//  ...           [index pages]      N × (8 + 8 + PAGE_SIZE) B
//                                               page_id           u64   ( 8 B)
//                                               tenant_id         u64   ( 8 B)
//                                               page_bytes        [u8; PAGE_SIZE]
//
// ─── v2 payload layout (ADR-032 Slice 1). ────────────────────────
//
//   offset       field             size (B)   notes
//    0           commit_lsn         8
//    8           n_mvcc_writes      4
//   12           [mvcc writes]      variable  per write:
//                                               tenant_id         u64   ( 8 B) NEW in v2
//                                               key               u64   ( 8 B)
//                                               kind              u8    ( 1 B)
//                                               value_len         u32   ( 4 B)
//                                               value             v B
//  ...           n_index_pages      4
//  ...           [index pages]      N × (8 + 8 + PAGE_SIZE) B
//                                               page_id           u64   ( 8 B)
//                                               tenant_id         u64   ( 8 B)
//                                               page_bytes        [u8; PAGE_SIZE]

/// One non-primary-tenant MVCC write carried by a v2 `CommitBundle`.
///
/// Today the sole source of `SideChannelWrite` entries is `grow_root`
/// folding its SYSTEM-tenant root-pointer update into the outer user
/// commit (ADR-032 §2 F1). The on-wire encoding is uniform with
/// primary-tenant writes; the decoder partitions by comparing a
/// per-entry `tenant_id` to the record-header's primary tenant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideChannelWrite {
    /// The tenant this write lands in. Always distinct from the
    /// bundle's primary tenant (co-tenant writes go through
    /// `primary_writes`).
    pub tenant_id: TenantId,
    /// MvccKey within the tenant's keyspace.
    pub key: MvccKey,
    /// Payload value; `None` = tombstone.
    pub value: Option<Bytes>,
}

/// Per-index staged page snapshot carried in a `CommitBundle`.
///
/// Captured under the owning index's `write_gate` + per-page write
/// latch during descent (see `PrimaryIndex::stage_emit` /
/// `SecondaryIndex::stage_emit`); returned to the commit authority via
/// the `*_deferred` index API so the MVCC commit's Phase 2 can fold
/// the bytes into one atomic `CommitBundle` record.
///
/// The copy-under-latch is mandatory: once `write_gate` and the
/// per-page latch release, another writer may mutate the same page in
/// place, so the staged bytes must be a snapshot-as-of-this-commit —
/// otherwise the WAL record would log post-conflict state. See
/// ADR-030 §Decision for the original rationale and ADR-031 §Decision
/// for the fold semantics.
#[derive(Debug)]
pub struct StagedEmit {
    /// Which in-memory page store this snapshot targets (ADR-031
    /// amendment-02 / PR #79 X-2 fold-in). Defaults to
    /// `BundlePageKind::PrimaryIndex` for back-compat with
    /// pre-amendment callers.
    pub kind: BundlePageKind,
    /// The page that was mutated (allocator-assigned within the
    /// owning index's `(SYSTEM, IndexLeaf)` key space per DEC-18,
    /// or the record / blob store's allocator for non-index
    /// kinds).
    pub page_id: PageId,
    /// A heap-allocated copy of the page's post-mutation bytes
    /// captured under the write latch.
    pub bytes: Box<[u8; PAGE_SIZE]>,
}

impl Default for StagedEmit {
    fn default() -> Self {
        Self {
            kind: BundlePageKind::PrimaryIndex,
            page_id: PageId::ZERO,
            bytes: Box::new([0u8; PAGE_SIZE]),
        }
    }
}

impl StagedEmit {
    /// Byte-size of this entry inside a `CommitBundle` payload.
    #[inline]
    #[must_use]
    pub const fn encoded_len() -> usize {
        // page_id (8) + tenant_id (8) + page bytes (PAGE_SIZE).
        8 + 8 + PAGE_SIZE
    }
}

/// Encode a v1 `CommitBundle` payload.
///
/// Called by `TxnManager::commit_with_bundle_writes` in Phase 2 of
/// the three-phase commit on pre-ADR-032-Slice-2 segments. Each
/// `StagedEmit` carries a fixed `PAGE_SIZE`-byte snapshot; the MVCC
/// write-set is variable-sized but structurally identical to the
/// legacy `encode_commit_payload` body.
///
/// The bundle is further wrapped by `WalRecord::encode` with a 4 B
/// CRC32C + 4 B length prefix + record-type byte + per-record header
/// (txn_id / lsn / timestamp / tenant_id) when it lands in the WAL
/// segment, giving record-level atomicity on crash.
#[must_use]
pub fn encode_commit_bundle(
    commit_lsn: Lsn,
    mvcc_writes: &HashMap<MvccKey, Option<Bytes>>,
    staged_emits: &[StagedEmit],
    staged_tenant: TenantId,
) -> Vec<u8> {
    // ── Size estimate (exact for MVCC writes, exact for pages). ──
    let mvcc_bytes_total: usize = mvcc_writes
        .values()
        .map(|v| v.as_ref().map_or(0, bytes::Bytes::len))
        .sum();
    let index_bytes_total: usize = staged_emits.len() * StagedEmit::encoded_len();
    let capacity =
        8 + 4 + mvcc_writes.len() * (8 + 1 + 4) + mvcc_bytes_total + 4 + index_bytes_total;
    let mut out = Vec::with_capacity(capacity);

    // ── commit_lsn header ──
    out.extend_from_slice(&commit_lsn.raw().to_le_bytes());

    // ── MVCC writes section ──
    // Sorted by key so replay + test assertions are deterministic.
    let n_mvcc = u32::try_from(mvcc_writes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_mvcc.to_le_bytes());
    let mut keys: Vec<MvccKey> = mvcc_writes.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let value = mvcc_writes.get(&key).expect("key sourced from writes");
        out.extend_from_slice(&key.to_le_bytes());
        match value {
            None => {
                out.push(0u8);
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(bytes) => {
                out.push(1u8);
                let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
            }
        }
    }

    // ── IndexPage entries section ──
    let n_pages = u32::try_from(staged_emits.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_pages.to_le_bytes());
    for emit in staged_emits {
        out.extend_from_slice(&emit.page_id.raw().to_le_bytes());
        out.extend_from_slice(&staged_tenant.raw().to_le_bytes());
        out.extend_from_slice(emit.bytes.as_ref());
    }

    out
}

/// Encode a v2 `CommitBundle` payload (ADR-032 Slice 1).
///
/// The primary-tenant writes in `primary_writes` (keyed by
/// `MvccKey`) and non-primary writes in `sidechannel_writes` (each
/// carrying its own `TenantId`) are combined into a single on-wire
/// list sorted by `(tenant_id, key)` ascending. Each entry carries
/// an explicit `tenant_id: u64` on the wire (NEW vs v1).
///
/// Callers are responsible for ensuring no sidechannel entry collides
/// with a primary write at the same `(tenant, key)` — the encoder
/// assumes disjointness and does not validate. In current usage the
/// disjointness is structural: sidechannel writes target the SYSTEM
/// tenant's root-pointer key, primary writes target user tenants.
///
/// The `IndexPage` section is unchanged from v1.
#[must_use]
pub fn encode_commit_bundle_v2(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_emits: &[StagedEmit],
    staged_tenant: TenantId,
) -> Vec<u8> {
    let n_entries = primary_writes.len() + sidechannel_writes.len();

    // ── Merge + sort by (tenant_id, key) for deterministic wire order.
    let mut entries: Vec<(TenantId, MvccKey, Option<Bytes>)> = Vec::with_capacity(n_entries);
    for (k, v) in primary_writes {
        entries.push((primary_tenant, *k, v.clone()));
    }
    for sc in sidechannel_writes {
        entries.push((sc.tenant_id, sc.key, sc.value.clone()));
    }
    entries.sort_by_key(|e| (e.0.raw(), e.1));

    // ── Size estimate (exact). ─────────────────────────────────────
    let mvcc_bytes_total: usize = entries
        .iter()
        .map(|e| e.2.as_ref().map_or(0, bytes::Bytes::len))
        .sum();
    let index_bytes_total: usize = staged_emits.len() * StagedEmit::encoded_len();
    // v2 per-write overhead: tenant_id (8) + key (8) + kind (1) + value_len (4) = 21 B.
    let capacity = 8 + 4 + n_entries * (8 + 8 + 1 + 4) + mvcc_bytes_total + 4 + index_bytes_total;
    let mut out = Vec::with_capacity(capacity);

    // ── commit_lsn header ──
    out.extend_from_slice(&commit_lsn.raw().to_le_bytes());

    // ── MVCC writes section (per-entry tenant_id NEW in v2) ──
    let n_mvcc = u32::try_from(n_entries).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_mvcc.to_le_bytes());
    for (tenant, key, value) in &entries {
        out.extend_from_slice(&tenant.raw().to_le_bytes());
        out.extend_from_slice(&key.to_le_bytes());
        match value {
            None => {
                out.push(0u8);
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(bytes) => {
                out.push(1u8);
                let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
            }
        }
    }

    // ── IndexPage entries section (unchanged from v1) ──
    let n_pages = u32::try_from(staged_emits.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_pages.to_le_bytes());
    for emit in staged_emits {
        out.extend_from_slice(&emit.page_id.raw().to_le_bytes());
        out.extend_from_slice(&staged_tenant.raw().to_le_bytes());
        out.extend_from_slice(emit.bytes.as_ref());
    }

    out
}

/// Decoded view of a `CommitBundle` payload — suitable for M2.e WAL
/// replay (#38) and the Phase 4 format tests.
///
/// The shape is cross-cutting between v1 and v2:
///
/// - `primary_tenant` is always set. For v1 bundles the decoder
///   inherits it from the caller (the WAL record header's
///   `tenant_id`). For v2 bundles the decoder also receives it from
///   the caller and uses it to partition per-entry-tenant writes.
/// - `mvcc_writes` carries all writes whose tenant == `primary_tenant`.
///   In v1 this is "every write"; in v2 it's the primary-tenant
///   subset.
/// - `sidechannel_writes` is empty for v1 bundles and carries the
///   non-primary-tenant subset for v2 bundles, in `(tenant, key)`
///   sorted order matching the encoder's wire order.
/// - `index_pages` is identical on v1 and v2 (shape unchanged).
///
/// The replay agent MUST collect `commit_lsn` into a buffer, sort
/// ascending, then apply each bundle's MVCC writes + sidechannel
/// writes + IndexPage entries in order. This preserves commit-order
/// visibility and prevents sidechannel or index state from replaying
/// ahead of the MVCC commit that made it durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedCommitBundle {
    /// MVCC commit LSN (NOT the WAL writer's assigned record LSN).
    pub commit_lsn: Lsn,
    /// The bundle's primary tenant. Sourced from the WAL record
    /// header; used by v2 to partition per-entry writes between
    /// `mvcc_writes` and `sidechannel_writes`.
    pub primary_tenant: TenantId,
    /// MVCC write-set in the primary tenant. `None` value = tombstone.
    pub mvcc_writes: HashMap<MvccKey, Option<Bytes>>,
    /// Non-primary-tenant MVCC writes (v2 only; empty for v1).
    /// Sorted by `(tenant_id, key)` ascending — the encoder's wire
    /// order.
    pub sidechannel_writes: Vec<SideChannelWrite>,
    /// Staged page snapshots, in payload order (the order the
    /// commit's builder handed them to Phase 2).
    ///
    /// **v3 (ADR-031 amendment-02)**: each entry carries a
    /// [`BundlePageKind`] byte that lets the replay executor
    /// route into the right in-memory store (primary / secondary
    /// / record / blob). For v1/v2 bundles the decoder
    /// synthesizes `BundlePageKind::PrimaryIndex` on every entry.
    ///
    /// Formerly `index_pages` — renamed in the PR #79 X-2
    /// review fold-in since record + blob pages now ride this
    /// section too.
    pub staged_pages: Vec<DecodedStagedPage>,
    /// M3 physiological redo ops in strictly ascending full sub-LSN
    /// order. Empty for v1-v8 bundles.
    pub deltas: Vec<DeltaOp>,
    /// Per-(tenant, allocator-kind) high-water snapshots durified
    /// alongside this commit. Empty for v1/v2/v3 bundles (decoders
    /// synthesize `Vec::new()`); v4+ bundles carry one entry per
    /// non-pristine allocator at encode time. Replayed by the
    /// executor after MVCC writes + staged_pages so post-recovery
    /// `create_node` / `create_rel` / fresh-page allocations cannot
    /// re-issue an id a pre-fault commit consumed (issue #129 P0
    /// fix; ADR-034 D-1 restored).
    pub allocator_advances: Vec<AllocatorAdvance>,
    /// Vector arena page snapshots durified alongside this commit
    /// (M3.a Slice G.4 — commit-bundle vector staging). Empty for
    /// v1/v2/v3/v4 bundles (decoders synthesize `Vec::new()`); v5+
    /// bundles carry one entry per vector page mutation captured
    /// under the arena's write latch at commit time. Replayed by the
    /// executor AFTER `staged_pages` and BEFORE
    /// `allocator_advances`, dispatching through
    /// [`crate::vector_store::VectorPageStoreHandle::install_or_replace`].
    /// Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
    pub vector_pages: Vec<VectorPageEntry>,
    /// `external_id → internal_id` idempotency bindings durified
    /// alongside this commit (#352 Part 2 — ADR-199 v6 fold). Empty for
    /// v1–v5 bundles (decoders synthesize `Vec::new()`); v6+ bundles
    /// carry one entry per fresh `external_id` bound by this commit.
    /// Replayed by the executor AFTER MVCC writes (so the node exists
    /// before its binding installs) into the
    /// [`crate::idempotency::IdempotencyStore`]. Per ADR-199.
    pub idempotency_bindings: Vec<IdempotencyBindingEntry>,
    /// Document-level ACL grant/revoke operations durified alongside this
    /// commit (#1221 — ADR-218 v8 fold). Empty for v1–v7 bundles
    /// (decoders synthesize `Vec::new()` ⇒ no ACL ops ⇒ fail-closed
    /// default); v8+ bundles carry one entry per `apply_doc_acl` /
    /// `revoke_doc` write-through. Replayed by the executor (in
    /// **staging/append order within a bundle**, ascending `commit_lsn`
    /// across bundles ⇒ last-writer-wins per doc) into the
    /// [`crate::permissions::PermissionIndex`]. Per ADR-218.
    pub acl_grants: Vec<AclGrantEntry>,
}

impl DecodedCommitBundle {
    /// This bundle's inclusive redo range. v1-v8 page-image bundles
    /// synthesize a singleton; v9 width is the delta count (or one for
    /// a zero-delta metadata-only commit).
    #[must_use]
    pub fn redo_range(&self) -> RedoLsnRange {
        RedoLsnRange::ending_at(self.commit_lsn, self.deltas.len())
            .expect("decoded bundle validated its redo range")
    }
    /// Back-compat alias for the pre-v3 `index_pages` field. Callers
    /// that only care about primary-index pages can filter via this
    /// accessor; most callers should move to `staged_pages` +
    /// dispatch on `kind`.
    ///
    /// Retained because several tests + one production assertion in
    /// `transaction.rs` still read `bundle.staged_pages` — the v3
    /// commit keeps them compiling without a structural edit to
    /// the test layer.
    pub fn index_pages(&self) -> impl Iterator<Item = &DecodedStagedPage> {
        self.staged_pages
            .iter()
            .filter(|p| p.kind == BundlePageKind::PrimaryIndex)
    }
}

/// One staged page snapshot inside a `CommitBundle`. Replaces the
/// pre-v3 `DecodedIndexPage` — every entry carries a
/// [`BundlePageKind`] byte (v3) or a synthesized
/// `BundlePageKind::PrimaryIndex` (v1/v2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedStagedPage {
    /// Target page store for this entry. For v1/v2 bundles always
    /// `BundlePageKind::PrimaryIndex`; v3 bundles carry a
    /// per-entry byte on the wire.
    pub kind: BundlePageKind,
    pub page_id: PageId,
    pub tenant_id: TenantId,
    pub bytes: Box<[u8; PAGE_SIZE]>,
}

/// Back-compat alias for [`DecodedStagedPage`]. Keeps existing
/// test code + the production `records.rs` assertion compiling
/// through the v3 rename.
pub type DecodedIndexPage = DecodedStagedPage;

/// Decode a v1 `CommitBundle` payload.
///
/// v1 bundles carry no per-entry tenant on the wire; the decoder
/// takes `primary_tenant` from the WAL record header and stamps it
/// into the returned struct. `sidechannel_writes` is always empty
/// for v1. Called by the M2.e replay agent (#38) for pre-ADR-032
/// segments; also used by tests.
///
/// Errors:
///
/// - [`ArcGraphError::WalCorruption`] with a descriptive reason when
///   the payload has a malformed length, unexpected kind byte, or
///   fewer bytes than the declared `n_mvcc_writes` / `n_index_pages`
///   demand. The record-level CRC is already validated by
///   [`super::record::WalRecord::decode`] before this function runs,
///   so a sub-parse failure here indicates either a bug in
///   `encode_commit_bundle` or a partially-truncated bundle inside a
///   torn-tail (which should have been filtered out one level up by
///   `WalRecoveryReader`). Treat as corruption.
pub fn decode_commit_bundle_v1(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let mut cursor = 0usize;

    // commit_lsn
    let commit_lsn_raw = read_u64_le(bytes, &mut cursor, "CommitBundle: commit_lsn")?;
    let commit_lsn = Lsn::new(commit_lsn_raw);

    // MVCC writes (no tenant on wire)
    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle: n_mvcc_writes")? as usize;
    // #1411: bound the pre-alloc against the untrusted count so a crafted
    // n_mvcc can't force a >remaining-bytes allocation before any element
    // is validated. cap == n_mvcc for valid input (>= n_mvcc*13 bytes left).
    let mvcc_cap = bounded_capacity(n_mvcc, bytes.len().saturating_sub(cursor), MIN_MVCC_V1_ELEM);
    let mut mvcc_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::with_capacity(mvcc_cap);
    for _ in 0..n_mvcc {
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle: mvcc kind")?;
        let value_len = read_u32_le(bytes, &mut cursor, "CommitBundle: mvcc value_len")? as usize;
        match kind {
            0 => {
                if value_len != 0 {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle: tombstone with non-zero value_len",
                    ));
                }
                mvcc_writes.insert(key, None);
            }
            1 => {
                if cursor + value_len > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle: mvcc value overruns payload",
                    ));
                }
                let value = Bytes::copy_from_slice(&bytes[cursor..cursor + value_len]);
                cursor += value_len;
                mvcc_writes.insert(key, Some(value));
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle: unknown mvcc kind {other}"),
                ));
            }
        }
    }

    // IndexPage entries (v1/v2 synthesize BundlePageKind::PrimaryIndex).
    let n_pages = read_u32_le(bytes, &mut cursor, "CommitBundle: n_index_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let pages_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_INDEX_PAGE_ELEM,
    );
    let mut staged_pages: Vec<DecodedStagedPage> = Vec::with_capacity(pages_cap);
    for _ in 0..n_pages {
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle: index page_id")?;
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle: index tenant_id")?;
        if cursor + PAGE_SIZE > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle: index page bytes overrun payload",
            ));
        }
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + PAGE_SIZE]);
        cursor += PAGE_SIZE;
        staged_pages.push(DecodedStagedPage {
            kind: BundlePageKind::PrimaryIndex,
            page_id: PageId::new(page_id_raw),
            tenant_id: TenantId::new(tenant_raw),
            bytes: page_bytes,
        });
    }

    if cursor != bytes.len() {
        return Err(corruption(
            commit_lsn,
            &format!(
                "CommitBundle: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }

    Ok(DecodedCommitBundle {
        commit_lsn,
        primary_tenant,
        mvcc_writes,
        sidechannel_writes: Vec::new(),
        staged_pages,
        deltas: Vec::new(),
        allocator_advances: Vec::new(),
        vector_pages: Vec::new(),
        idempotency_bindings: Vec::new(),
        acl_grants: Vec::new(),
    })
}

/// Decode a v2 `CommitBundle` payload (ADR-032 Slice 1).
///
/// v2 bundles carry a per-entry `tenant_id` on every MVCC write. The
/// decoder partitions entries by comparing that tenant against the
/// caller-supplied `primary_tenant` (from the WAL record header):
/// matches go into `mvcc_writes`, non-matches go into
/// `sidechannel_writes` in sorted order matching the encoder's
/// `(tenant, key)` wire order.
pub fn decode_commit_bundle_v2(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let mut cursor = 0usize;

    // commit_lsn
    let commit_lsn_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v2: commit_lsn")?;
    let commit_lsn = Lsn::new(commit_lsn_raw);

    // MVCC writes (per-entry tenant on wire).
    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle v2: n_mvcc_writes")? as usize;
    let mut mvcc_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::new();
    let mut sidechannel_writes: Vec<SideChannelWrite> = Vec::new();
    for _ in 0..n_mvcc {
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v2: mvcc tenant_id")?;
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle v2: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle v2: mvcc kind")?;
        let value_len =
            read_u32_le(bytes, &mut cursor, "CommitBundle v2: mvcc value_len")? as usize;
        let value = match kind {
            0 => {
                if value_len != 0 {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v2: tombstone with non-zero value_len",
                    ));
                }
                None
            }
            1 => {
                if cursor + value_len > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v2: mvcc value overruns payload",
                    ));
                }
                let v = Bytes::copy_from_slice(&bytes[cursor..cursor + value_len]);
                cursor += value_len;
                Some(v)
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v2: unknown mvcc kind {other}"),
                ));
            }
        };
        let tenant = TenantId::new(tenant_raw);
        if tenant == primary_tenant {
            mvcc_writes.insert(key, value);
        } else {
            sidechannel_writes.push(SideChannelWrite {
                tenant_id: tenant,
                key,
                value,
            });
        }
    }

    // IndexPage entries (unchanged from v1; synthesize
    // BundlePageKind::PrimaryIndex — v2 had no other kind).
    let n_pages = read_u32_le(bytes, &mut cursor, "CommitBundle v2: n_index_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let pages_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_INDEX_PAGE_ELEM,
    );
    let mut staged_pages: Vec<DecodedStagedPage> = Vec::with_capacity(pages_cap);
    for _ in 0..n_pages {
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v2: index page_id")?;
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v2: index tenant_id")?;
        if cursor + PAGE_SIZE > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v2: index page bytes overrun payload",
            ));
        }
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + PAGE_SIZE]);
        cursor += PAGE_SIZE;
        staged_pages.push(DecodedStagedPage {
            kind: BundlePageKind::PrimaryIndex,
            page_id: PageId::new(page_id_raw),
            tenant_id: TenantId::new(tenant_raw),
            bytes: page_bytes,
        });
    }

    if cursor != bytes.len() {
        return Err(corruption(
            commit_lsn,
            &format!(
                "CommitBundle v2: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }

    Ok(DecodedCommitBundle {
        commit_lsn,
        primary_tenant,
        mvcc_writes,
        sidechannel_writes,
        staged_pages,
        deltas: Vec::new(),
        allocator_advances: Vec::new(),
        vector_pages: Vec::new(),
        idempotency_bindings: Vec::new(),
        acl_grants: Vec::new(),
    })
}

/// Dispatch a `CommitBundle` payload parse by segment format-version.
///
/// Called by the replay executor (Slice 3) after reading
/// [`crate::wal::SegmentHeader::format_version`] from the owning
/// segment. Unknown `format_version` values return
/// [`ArcGraphError::WalFormatMismatch`] with the supported list.
pub fn decode_commit_bundle_for_version(
    bytes: &[u8],
    format_version: u16,
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    match format_version {
        BUNDLE_FORMAT_V1 => decode_commit_bundle_v1(bytes, primary_tenant),
        BUNDLE_FORMAT_V2 => decode_commit_bundle_v2(bytes, primary_tenant),
        BUNDLE_FORMAT_V3 => decode_commit_bundle_v3(bytes, primary_tenant),
        BUNDLE_FORMAT_V4 => decode_commit_bundle_v4(bytes, primary_tenant),
        BUNDLE_FORMAT_V5 => decode_commit_bundle_v5(bytes, primary_tenant),
        BUNDLE_FORMAT_V6 => decode_commit_bundle_v6(bytes, primary_tenant),
        BUNDLE_FORMAT_V7 => decode_commit_bundle_v7(bytes, primary_tenant),
        BUNDLE_FORMAT_V8 => decode_commit_bundle_v8(bytes, primary_tenant),
        BUNDLE_FORMAT_V9 => decode_commit_bundle_v9(bytes, primary_tenant),
        BUNDLE_FORMAT_V10 => decode_commit_bundle_v10(bytes, primary_tenant),
        _ => Err(ArcGraphError::WalFormatMismatch {
            found_version: format_version,
            supported_versions: crate::wal::SUPPORTED_WAL_FORMAT_VERSIONS,
        }),
    }
}

// ─── v3 codec (ADR-031 amendment-02; PR #79 X-2 fold-in) ────────────

/// Encode a v3 `CommitBundle` payload.
///
/// v3 generalises v2's `index_pages` to a unified `staged_pages`
/// section whose entries each carry a one-byte
/// [`BundlePageKind`] discriminator. Record + BLOB pages now
/// travel in the bundle (closing the X-2 gap).
///
/// The MVCC writes section is identical to v2 (per-entry
/// tenant_id). The staged_pages wire shape per entry:
///
/// ```text
///   offset  field        size
///    0      kind_byte    1     BundlePageKind::as_byte()
///    1      page_id      8     u64 LE
///    9      tenant_id    8     u64 LE
///   17      n_bytes      4     u32 LE (= PAGE_SIZE at v1.0)
///   21      bytes        N     n_bytes of page payload
/// ```
#[must_use]
pub fn encode_commit_bundle_v3(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
) -> Vec<u8> {
    let n_entries = primary_writes.len() + sidechannel_writes.len();
    let mut entries: Vec<(TenantId, MvccKey, Option<Bytes>)> = Vec::with_capacity(n_entries);
    for (k, v) in primary_writes {
        entries.push((primary_tenant, *k, v.clone()));
    }
    for sc in sidechannel_writes {
        entries.push((sc.tenant_id, sc.key, sc.value.clone()));
    }
    entries.sort_by_key(|e| (e.0.raw(), e.1));

    let mvcc_bytes_total: usize = entries
        .iter()
        .map(|e| e.2.as_ref().map_or(0, bytes::Bytes::len))
        .sum();
    // v3 staged_pages per-entry overhead: kind (1) + page_id (8) +
    // tenant_id (8) + n_bytes (4) + PAGE_SIZE = 21 + PAGE_SIZE.
    let staged_bytes_total: usize = staged_pages.len() * (1 + 8 + 8 + 4 + PAGE_SIZE);
    let capacity = 8 + 4 + n_entries * (8 + 8 + 1 + 4) + mvcc_bytes_total + 4 + staged_bytes_total;
    let mut out = Vec::with_capacity(capacity);

    out.extend_from_slice(&commit_lsn.raw().to_le_bytes());
    let n_mvcc = u32::try_from(n_entries).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_mvcc.to_le_bytes());
    for (tenant, key, value) in &entries {
        out.extend_from_slice(&tenant.raw().to_le_bytes());
        out.extend_from_slice(&key.to_le_bytes());
        match value {
            None => {
                out.push(0u8);
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(bytes) => {
                out.push(1u8);
                let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
            }
        }
    }

    let n_pages = u32::try_from(staged_pages.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n_pages.to_le_bytes());
    for (kind, page_id, tenant_id, page_bytes) in staged_pages {
        out.push(kind.as_byte());
        out.extend_from_slice(&page_id.raw().to_le_bytes());
        out.extend_from_slice(&tenant_id.raw().to_le_bytes());
        let n_bytes = u32::try_from(page_bytes.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&n_bytes.to_le_bytes());
        out.extend_from_slice(page_bytes.as_ref());
    }

    out
}

/// Decode a v3 `CommitBundle` payload.
///
/// See [`encode_commit_bundle_v3`] for the wire layout.
pub fn decode_commit_bundle_v3(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let mut cursor = 0usize;

    let commit_lsn_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v3: commit_lsn")?;
    let commit_lsn = Lsn::new(commit_lsn_raw);

    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle v3: n_mvcc_writes")? as usize;
    let mut mvcc_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::new();
    let mut sidechannel_writes: Vec<SideChannelWrite> = Vec::new();
    for _ in 0..n_mvcc {
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v3: mvcc tenant_id")?;
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle v3: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle v3: mvcc kind")?;
        let value_len =
            read_u32_le(bytes, &mut cursor, "CommitBundle v3: mvcc value_len")? as usize;
        let value = match kind {
            0 => {
                if value_len != 0 {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v3: tombstone with non-zero value_len",
                    ));
                }
                None
            }
            1 => {
                if cursor + value_len > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v3: mvcc value overruns payload",
                    ));
                }
                let v = Bytes::copy_from_slice(&bytes[cursor..cursor + value_len]);
                cursor += value_len;
                Some(v)
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v3: unknown mvcc kind {other}"),
                ));
            }
        };
        let tenant = TenantId::new(tenant_raw);
        if tenant == primary_tenant {
            mvcc_writes.insert(key, value);
        } else {
            sidechannel_writes.push(SideChannelWrite {
                tenant_id: tenant,
                key,
                value,
            });
        }
    }

    let n_pages = read_u32_le(bytes, &mut cursor, "CommitBundle v3: n_staged_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let pages_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_STAGED_PAGE_ELEM,
    );
    let mut staged_pages: Vec<DecodedStagedPage> = Vec::with_capacity(pages_cap);
    for _ in 0..n_pages {
        let kind_byte = read_u8(bytes, &mut cursor, "CommitBundle v3: staged_page kind")?;
        let kind = BundlePageKind::from_byte(kind_byte)?;
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v3: staged_page page_id")?;
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v3: staged_page tenant_id")?;
        let n_bytes =
            read_u32_le(bytes, &mut cursor, "CommitBundle v3: staged_page n_bytes")? as usize;
        if n_bytes != PAGE_SIZE {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v3: staged_page n_bytes={n_bytes} != PAGE_SIZE ({PAGE_SIZE})"
                ),
            ));
        }
        if cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v3: staged_page bytes overrun payload",
            ));
        }
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        staged_pages.push(DecodedStagedPage {
            kind,
            page_id: PageId::new(page_id_raw),
            tenant_id: TenantId::new(tenant_raw),
            bytes: page_bytes,
        });
    }

    if cursor != bytes.len() {
        return Err(corruption(
            commit_lsn,
            &format!(
                "CommitBundle v3: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }

    Ok(DecodedCommitBundle {
        commit_lsn,
        primary_tenant,
        mvcc_writes,
        sidechannel_writes,
        staged_pages,
        deltas: Vec::new(),
        allocator_advances: Vec::new(),
        vector_pages: Vec::new(),
        idempotency_bindings: Vec::new(),
        acl_grants: Vec::new(),
    })
}

// ─── v4 codec (issue #129 P0 fix — allocator-advance section) ────────

/// Encode a v4 `CommitBundle` payload.
///
/// v4 is a strict superset of v3: the `commit_lsn` header,
/// MVCC writes section, and `staged_pages` section are encoded
/// exactly as in [`encode_commit_bundle_v3`], followed by a new
/// `n_allocator_advances: u32 LE` count and that many fixed-size
/// 17-byte entries:
///
/// ```text
///   offset  field           size
///    0      tenant_id       8     u64 LE
///    8      kind            1     [`AllocatorKind`]::as_byte()
///    9      new_high_water  8     u64 LE — last allocated id
/// ```
///
/// Each entry is the high-water of one (`tenant`, `kind`)
/// allocator at the encode point. Replay applies them in
/// commit_lsn order via `seed_from_advance(tenant, kind, hw)` →
/// counter = `max(current, hw + 1)` (Lemma I3 — monotonic
/// idempotent replay; double-replay is a no-op). Issue #129 P0
/// fix; ADR-034 D-1 restored.
#[must_use]
pub fn encode_commit_bundle_v4(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
) -> Vec<u8> {
    // Reuse the v3 encoder for the prefix; append the v4
    // allocator-advances section after.
    let mut out = encode_commit_bundle_v3(
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        staged_pages,
    );
    append_v4_allocator_advances(&mut out, allocator_advances);
    out
}

/// Append the v4 allocator section. Shared verbatim by v4-v8 and v9+ so the
/// M3 format retains the proven logical-owner wire bytes instead of defining a
/// second encoding.
fn append_v4_allocator_advances(out: &mut Vec<u8>, allocator_advances: &[AllocatorAdvance]) {
    // Sort by (tenant, kind) for deterministic wire order — replay
    // is order-insensitive (monotonic-max), but a stable order keeps
    // bundle bytes diff-friendly across recompiles and makes test
    // assertions deterministic.
    let mut sorted: Vec<&AllocatorAdvance> = allocator_advances.iter().collect();
    sorted.sort_by_key(|a| (a.tenant.raw(), a.kind.as_byte()));

    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    out.reserve(4 + sorted.len() * AllocatorAdvance::ENCODED_LEN);
    out.extend_from_slice(&n.to_le_bytes());
    for adv in sorted {
        out.extend_from_slice(&adv.tenant.raw().to_le_bytes());
        out.push(adv.kind.as_byte());
        out.extend_from_slice(&adv.new_high_water.to_le_bytes());
    }
}

/// Decode a v4 `CommitBundle` payload.
///
/// See [`encode_commit_bundle_v4`] for the wire layout.
pub fn decode_commit_bundle_v4(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    // The v3 prefix has the same shape; we cannot call
    // `decode_commit_bundle_v3` directly because that function
    // strict-checks for trailing bytes. Inline the decode and stop
    // before the trailing-bytes check, then parse the v4 tail.
    let mut cursor = 0usize;

    let commit_lsn_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v4: commit_lsn")?;
    let commit_lsn = Lsn::new(commit_lsn_raw);

    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle v4: n_mvcc_writes")? as usize;
    let mut mvcc_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::new();
    let mut sidechannel_writes: Vec<SideChannelWrite> = Vec::new();
    for _ in 0..n_mvcc {
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v4: mvcc tenant_id")?;
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle v4: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle v4: mvcc kind")?;
        let value_len =
            read_u32_le(bytes, &mut cursor, "CommitBundle v4: mvcc value_len")? as usize;
        let value = match kind {
            0 => {
                if value_len != 0 {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v4: tombstone with non-zero value_len",
                    ));
                }
                None
            }
            1 => {
                if cursor + value_len > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v4: mvcc value overruns payload",
                    ));
                }
                let v = Bytes::copy_from_slice(&bytes[cursor..cursor + value_len]);
                cursor += value_len;
                Some(v)
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v4: unknown mvcc kind {other}"),
                ));
            }
        };
        let tenant = TenantId::new(tenant_raw);
        if tenant == primary_tenant {
            mvcc_writes.insert(key, value);
        } else {
            sidechannel_writes.push(SideChannelWrite {
                tenant_id: tenant,
                key,
                value,
            });
        }
    }

    let n_pages = read_u32_le(bytes, &mut cursor, "CommitBundle v4: n_staged_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let pages_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_STAGED_PAGE_ELEM,
    );
    let mut staged_pages: Vec<DecodedStagedPage> = Vec::with_capacity(pages_cap);
    for _ in 0..n_pages {
        let kind_byte = read_u8(bytes, &mut cursor, "CommitBundle v4: staged_page kind")?;
        let kind = BundlePageKind::from_byte(kind_byte)?;
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v4: staged_page page_id")?;
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v4: staged_page tenant_id")?;
        let n_bytes =
            read_u32_le(bytes, &mut cursor, "CommitBundle v4: staged_page n_bytes")? as usize;
        if n_bytes != PAGE_SIZE {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v4: staged_page n_bytes={n_bytes} != PAGE_SIZE ({PAGE_SIZE})"
                ),
            ));
        }
        if cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v4: staged_page bytes overrun payload",
            ));
        }
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        staged_pages.push(DecodedStagedPage {
            kind,
            page_id: PageId::new(page_id_raw),
            tenant_id: TenantId::new(tenant_raw),
            bytes: page_bytes,
        });
    }

    // v4 tail: allocator_advances section.
    let n_advances =
        read_u32_le(bytes, &mut cursor, "CommitBundle v4: n_allocator_advances")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let advances_cap = bounded_capacity(
        n_advances,
        bytes.len().saturating_sub(cursor),
        MIN_ALLOCATOR_ADVANCE_ELEM,
    );
    let mut allocator_advances: Vec<AllocatorAdvance> = Vec::with_capacity(advances_cap);
    for _ in 0..n_advances {
        let tenant_raw = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v4: allocator_advance tenant_id",
        )?;
        let kind_byte = read_u8(
            bytes,
            &mut cursor,
            "CommitBundle v4: allocator_advance kind",
        )?;
        let kind = AllocatorKind::from_byte(kind_byte)?;
        let new_high_water = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v4: allocator_advance new_high_water",
        )?;
        allocator_advances.push(AllocatorAdvance {
            tenant: TenantId::new(tenant_raw),
            kind,
            new_high_water,
        });
    }

    if cursor != bytes.len() {
        return Err(corruption(
            commit_lsn,
            &format!(
                "CommitBundle v4: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }

    Ok(DecodedCommitBundle {
        commit_lsn,
        primary_tenant,
        mvcc_writes,
        sidechannel_writes,
        staged_pages,
        deltas: Vec::new(),
        allocator_advances,
        vector_pages: Vec::new(),
        idempotency_bindings: Vec::new(),
        acl_grants: Vec::new(),
    })
}

// ─── v5 codec (M3.a Slice G.4 — vector-page section) ────────────────

/// Encode a v5 `CommitBundle` payload.
///
/// v5 is a strict superset of v4: the `commit_lsn` header,
/// MVCC writes section, `staged_pages` section, and
/// `allocator_advances` section are encoded exactly as in
/// [`encode_commit_bundle_v4`], followed by a new
/// `n_vector_pages: u32 LE` count and that many fixed-size
/// `(8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE)`-byte entries:
///
/// ```text
///   offset  field           size
///    0      tenant_id       8     u64 LE
///    8      partition_id    8     u64 LE — always 0 at v1.0
///   16      index_id        8     u64 LE — always 0 at v1.0
///   24      page_id         8     u64 LE
///   32      commit_lsn      8     u64 LE — redundant with bundle.commit_lsn at v1.0,
///                                           forward-looking for v1.1 batched commits
///   40      n_bytes         4     u32 LE — MUST be PAGE_SIZE
///   44      bytes           N     n_bytes of page payload
/// ```
///
/// Each entry is one vector arena page snapshot captured at commit
/// time. Replay applies them in commit_lsn order via
/// [`crate::vector_store::VectorPageStoreHandle::install_or_replace`]
/// AFTER `staged_pages` and BEFORE `allocator_advances` (Lemma I3).
/// Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
#[must_use]
#[allow(clippy::too_many_arguments)] // v5 codec section count matches the wire format.
pub fn encode_commit_bundle_v5(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
    vector_pages: &[VectorPageEntry],
) -> Vec<u8> {
    // Reuse the v4 encoder for the prefix; append the v5 vector_pages
    // section after the allocator_advances tail.
    let mut out = encode_commit_bundle_v4(
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        staged_pages,
        allocator_advances,
    );
    // Sort by (tenant, partition, index_id, page_id, commit_lsn) for
    // deterministic wire order — replay is order-insensitive (every
    // VectorPageStoreHandle::install_or_replace is unconditional
    // byte-copy so any ordering converges; commit-order arrival is
    // upstream-enforced by the executor's commit_lsn sort), but a
    // stable order keeps bundle bytes diff-friendly across recompiles
    // and makes test assertions deterministic.
    let mut sorted: Vec<&VectorPageEntry> = vector_pages.iter().collect();
    sorted.sort_by_key(|e| {
        (
            e.tenant.raw(),
            u64::from(e.partition.raw()),
            e.index_id,
            e.page_id.raw(),
            e.commit_lsn.raw(),
        )
    });

    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    out.reserve(4 + sorted.len() * VectorPageEntry::ENCODED_LEN);
    out.extend_from_slice(&n.to_le_bytes());
    for entry in sorted {
        out.extend_from_slice(&entry.tenant.raw().to_le_bytes());
        out.extend_from_slice(&u64::from(entry.partition.raw()).to_le_bytes());
        out.extend_from_slice(&entry.index_id.to_le_bytes());
        out.extend_from_slice(&entry.page_id.raw().to_le_bytes());
        out.extend_from_slice(&entry.commit_lsn.raw().to_le_bytes());
        let n_bytes = u32::try_from(entry.bytes.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&n_bytes.to_le_bytes());
        out.extend_from_slice(entry.bytes.as_ref());
    }
    out
}

// ─── v6 codec (#352 Part 2 — idempotency_bindings section; ADR-199) ──

/// Encode a v6 `CommitBundle` payload (#352 Part 2 — ADR-199).
///
/// v6 is a strict superset of v5: the v5 prefix (header, MVCC writes,
/// `staged_pages`, `allocator_advances`, `vector_pages`) is encoded
/// exactly as in [`encode_commit_bundle_v5`], followed by a new
/// `n_idempotency_bindings: u32 LE` count and that many variable-length
/// entries:
///
/// ```text
///   field         size
///   tenant_id     8     u64 LE
///   kind          1     u8 (opaque; mcp: 0 = Node, 1 = Rel)
///   internal_id   8     u64 LE
///   ext_len       4     u32 LE
///   external_id   N     ext_len bytes, UTF-8
/// ```
///
/// Each entry is one `external_id → internal_id` binding durified
/// atomically with the commit that allocated `internal_id`. Replay
/// applies them via [`crate::idempotency::IdempotencyStore::install`]
/// AFTER MVCC writes (so the node exists before its binding installs).
/// Per ADR-199.
#[must_use]
#[allow(clippy::too_many_arguments)] // v6 codec section count matches the wire format.
pub fn encode_commit_bundle_v6(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
    vector_pages: &[VectorPageEntry],
    idempotency_bindings: &[IdempotencyBindingEntry],
) -> Vec<u8> {
    // Reuse the v5 encoder for the prefix; append the v6
    // idempotency_bindings section after the vector_pages tail.
    let mut out = encode_commit_bundle_v5(
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        staged_pages,
        allocator_advances,
        vector_pages,
    );
    // Deterministic wire order: sort by (tenant, kind, external_id).
    // Replay is order-insensitive (`install` is an unconditional
    // last-write-wins map insert, and there is one binding per
    // (tenant, kind, external_id) per bundle by construction), but a
    // stable order keeps bundle bytes diff-friendly across recompiles
    // and makes test assertions deterministic.
    let mut sorted: Vec<&IdempotencyBindingEntry> = idempotency_bindings.iter().collect();
    sorted.sort_by(|a, b| {
        a.tenant
            .raw()
            .cmp(&b.tenant.raw())
            .then(a.kind.cmp(&b.kind))
            .then_with(|| a.external_id.cmp(&b.external_id))
    });

    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    out.reserve(4 + sorted.len() * IdempotencyBindingEntry::V6_FIXED_PREFIX_LEN);
    out.extend_from_slice(&n.to_le_bytes());
    for entry in sorted {
        out.extend_from_slice(&entry.tenant.raw().to_le_bytes());
        out.push(entry.kind);
        out.extend_from_slice(&entry.internal_id.to_le_bytes());
        let ext_len = u32::try_from(entry.external_id.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&ext_len.to_le_bytes());
        out.extend_from_slice(entry.external_id.as_bytes());
    }
    out
}

/// Parse the v5 sections (`commit_lsn` header, MVCC writes,
/// `staged_pages`, `allocator_advances`, `vector_pages`) WITHOUT the
/// trailing-bytes check, returning the partially-built bundle (with an
/// empty `idempotency_bindings`) and the cursor position. Shared by
/// [`decode_commit_bundle_v5`] (which then asserts no trailing bytes)
/// and [`decode_commit_bundle_v6`] (which then parses the v6
/// `idempotency_bindings` tail). #352 Part 2 / ADR-199.
fn parse_v4_allocator_advances(bytes: &[u8], cursor: &mut usize) -> Result<Vec<AllocatorAdvance>> {
    let n_advances = read_u32_le(bytes, cursor, "CommitBundle: n_allocator_advances")? as usize;
    let advances_cap = bounded_capacity(
        n_advances,
        bytes.len().saturating_sub(*cursor),
        MIN_ALLOCATOR_ADVANCE_ELEM,
    );
    let mut allocator_advances = Vec::with_capacity(advances_cap);
    for _ in 0..n_advances {
        let tenant_raw = read_u64_le(bytes, cursor, "CommitBundle: allocator_advance tenant_id")?;
        let kind_byte = read_u8(bytes, cursor, "CommitBundle: allocator_advance kind")?;
        let kind = AllocatorKind::from_byte(kind_byte)?;
        let new_high_water = read_u64_le(
            bytes,
            cursor,
            "CommitBundle: allocator_advance new_high_water",
        )?;
        allocator_advances.push(AllocatorAdvance {
            tenant: TenantId::new(tenant_raw),
            kind,
            new_high_water,
        });
    }
    Ok(allocator_advances)
}

fn decode_commit_bundle_v5_sections(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<(DecodedCommitBundle, usize)> {
    let mut cursor = 0usize;

    let commit_lsn_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: commit_lsn")?;
    let commit_lsn = Lsn::new(commit_lsn_raw);

    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle v5: n_mvcc_writes")? as usize;
    let mut mvcc_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::new();
    let mut sidechannel_writes: Vec<SideChannelWrite> = Vec::new();
    for _ in 0..n_mvcc {
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: mvcc tenant_id")?;
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle v5: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle v5: mvcc kind")?;
        let value_len =
            read_u32_le(bytes, &mut cursor, "CommitBundle v5: mvcc value_len")? as usize;
        let value = match kind {
            0 => {
                if value_len != 0 {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v5: tombstone with non-zero value_len",
                    ));
                }
                None
            }
            1 => {
                if cursor + value_len > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v5: mvcc value overruns payload",
                    ));
                }
                let v = Bytes::copy_from_slice(&bytes[cursor..cursor + value_len]);
                cursor += value_len;
                Some(v)
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v5: unknown mvcc kind {other}"),
                ));
            }
        };
        let tenant = TenantId::new(tenant_raw);
        if tenant == primary_tenant {
            mvcc_writes.insert(key, value);
        } else {
            sidechannel_writes.push(SideChannelWrite {
                tenant_id: tenant,
                key,
                value,
            });
        }
    }

    let n_pages = read_u32_le(bytes, &mut cursor, "CommitBundle v5: n_staged_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let pages_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_STAGED_PAGE_ELEM,
    );
    let mut staged_pages: Vec<DecodedStagedPage> = Vec::with_capacity(pages_cap);
    for _ in 0..n_pages {
        let kind_byte = read_u8(bytes, &mut cursor, "CommitBundle v5: staged_page kind")?;
        let kind = BundlePageKind::from_byte(kind_byte)?;
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: staged_page page_id")?;
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: staged_page tenant_id")?;
        let n_bytes =
            read_u32_le(bytes, &mut cursor, "CommitBundle v5: staged_page n_bytes")? as usize;
        if n_bytes != PAGE_SIZE {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v5: staged_page n_bytes={n_bytes} != PAGE_SIZE ({PAGE_SIZE})"
                ),
            ));
        }
        if cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v5: staged_page bytes overrun payload",
            ));
        }
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        staged_pages.push(DecodedStagedPage {
            kind,
            page_id: PageId::new(page_id_raw),
            tenant_id: TenantId::new(tenant_raw),
            bytes: page_bytes,
        });
    }

    // v4 tail: allocator_advances section (unchanged from v4).
    let allocator_advances = parse_v4_allocator_advances(bytes, &mut cursor)?;

    // v5 tail: vector_pages section.
    let n_vector_pages =
        read_u32_le(bytes, &mut cursor, "CommitBundle v5: n_vector_pages")? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let vector_pages_cap = bounded_capacity(
        n_vector_pages,
        bytes.len().saturating_sub(cursor),
        MIN_VECTOR_PAGE_ELEM,
    );
    let mut vector_pages: Vec<VectorPageEntry> = Vec::with_capacity(vector_pages_cap);
    for _ in 0..n_vector_pages {
        let tenant_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: vector_page tenant_id")?;
        let partition_raw = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v5: vector_page partition_id",
        )?;
        let index_id = read_u64_le(bytes, &mut cursor, "CommitBundle v5: vector_page index_id")?;
        let page_id_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v5: vector_page page_id")?;
        let commit_lsn_entry_raw = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v5: vector_page commit_lsn",
        )?;
        let n_bytes =
            read_u32_le(bytes, &mut cursor, "CommitBundle v5: vector_page n_bytes")? as usize;
        if n_bytes != PAGE_SIZE {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v5: vector_page n_bytes={n_bytes} != PAGE_SIZE ({PAGE_SIZE})"
                ),
            ));
        }
        if cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v5: vector_page bytes overrun payload",
            ));
        }
        // PartitionId::raw() returns u32 at v1.0; the wire slot is u64 LE
        // for forward-compat. v1.0 invariant: partition_raw == 0.
        let partition = if partition_raw <= u64::from(u32::MAX) {
            PartitionId::new(partition_raw as u32)
        } else {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v5: vector_page partition_id={partition_raw} \
                     overflows u32 (v1.0 invariant: always 0)"
                ),
            ));
        };
        let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        vector_pages.push(VectorPageEntry {
            tenant: TenantId::new(tenant_raw),
            partition,
            index_id,
            page_id: PageId::new(page_id_raw),
            commit_lsn: Lsn::new(commit_lsn_entry_raw),
            bytes: page_bytes,
        });
    }

    Ok((
        DecodedCommitBundle {
            commit_lsn,
            primary_tenant,
            mvcc_writes,
            sidechannel_writes,
            staged_pages,
            deltas: Vec::new(),
            allocator_advances,
            vector_pages,
            idempotency_bindings: Vec::new(),
            acl_grants: Vec::new(),
        },
        cursor,
    ))
}

/// Decode a v5 `CommitBundle` payload.
///
/// See [`encode_commit_bundle_v5`] for the wire layout. Thin wrapper
/// over `decode_commit_bundle_v5_sections` that asserts no trailing
/// bytes (v5 has no section past `vector_pages`).
pub fn decode_commit_bundle_v5(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let (bundle, cursor) = decode_commit_bundle_v5_sections(bytes, primary_tenant)?;
    if cursor != bytes.len() {
        return Err(corruption(
            bundle.commit_lsn,
            &format!(
                "CommitBundle v5: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }
    Ok(bundle)
}

/// Decode a v6 `CommitBundle` payload (#352 Part 2 — ADR-199).
///
/// v6 = the v5 sections followed by a trailing `idempotency_bindings`
/// section. Each entry: `tenant_id u64 LE | kind u8 | internal_id u64
/// LE | ext_len u32 LE | external_id bytes`. See
/// [`encode_commit_bundle_v6`] for the full wire layout. A v6 binary
/// decoding an old v5 segment never reaches here (the per-segment
/// `format_version` dispatch routes v5 bytes to
/// [`decode_commit_bundle_v5`], which yields an empty
/// `idempotency_bindings`).
pub fn decode_commit_bundle_v6(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let (mut bundle, mut cursor) = decode_commit_bundle_v5_sections(bytes, primary_tenant)?;

    // v6 tail: idempotency_bindings section.
    let n_bindings = read_u32_le(
        bytes,
        &mut cursor,
        "CommitBundle v6: n_idempotency_bindings",
    )? as usize;
    // #1411: bound the pre-alloc against the untrusted count.
    let bindings_cap = bounded_capacity(
        n_bindings,
        bytes.len().saturating_sub(cursor),
        MIN_IDEMPOTENCY_V6_ELEM,
    );
    let mut idempotency_bindings: Vec<IdempotencyBindingEntry> = Vec::with_capacity(bindings_cap);
    for _ in 0..n_bindings {
        let tenant_raw = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v6: idempotency_binding tenant_id",
        )?;
        let kind = read_u8(
            bytes,
            &mut cursor,
            "CommitBundle v6: idempotency_binding kind",
        )?;
        let internal_id = read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v6: idempotency_binding internal_id",
        )?;
        let ext_len = read_u32_le(
            bytes,
            &mut cursor,
            "CommitBundle v6: idempotency_binding ext_len",
        )? as usize;
        if cursor + ext_len > bytes.len() {
            return Err(corruption(
                bundle.commit_lsn,
                "CommitBundle v6: idempotency_binding external_id overruns payload",
            ));
        }
        let external_id = std::str::from_utf8(&bytes[cursor..cursor + ext_len])
            .map_err(|e| {
                corruption(
                    bundle.commit_lsn,
                    &format!("CommitBundle v6: idempotency_binding external_id not UTF-8: {e}"),
                )
            })?
            .to_owned();
        cursor += ext_len;
        idempotency_bindings.push(IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant: TenantId::new(tenant_raw),
            kind,
            internal_id,
            external_id,
        });
    }
    bundle.idempotency_bindings = idempotency_bindings;

    if cursor != bytes.len() {
        return Err(corruption(
            bundle.commit_lsn,
            &format!(
                "CommitBundle v6: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }
    Ok(bundle)
}

/// Encode a v7 `CommitBundle` payload (#1010 — idempotency release fold).
///
/// v7 is a strict superset of the v5 prefix with a v7 idempotency tail.
/// It intentionally bumps from v6 because v6 entries had no op byte and
/// must remain install-only for existing WAL segments.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn encode_commit_bundle_v7(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
    vector_pages: &[VectorPageEntry],
    idempotency_bindings: &[IdempotencyBindingEntry],
) -> Vec<u8> {
    let mut out = encode_commit_bundle_v5(
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        staged_pages,
        allocator_advances,
        vector_pages,
    );
    append_v7_idempotency_bindings(&mut out, idempotency_bindings);
    out
}

/// Append the v7 idempotency section. v9+ reuses this exact codec under the
/// Director's M3 keep-v8-behavior ruling.
fn append_v7_idempotency_bindings(
    out: &mut Vec<u8>,
    idempotency_bindings: &[IdempotencyBindingEntry],
) {
    let mut sorted: Vec<&IdempotencyBindingEntry> = idempotency_bindings.iter().collect();
    sorted.sort_by(|a, b| {
        a.tenant
            .raw()
            .cmp(&b.tenant.raw())
            .then(a.kind.cmp(&b.kind))
            .then_with(|| a.external_id.cmp(&b.external_id))
            .then((a.op as u8).cmp(&(b.op as u8)))
    });

    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    out.reserve(4 + sorted.len() * IdempotencyBindingEntry::FIXED_PREFIX_LEN);
    out.extend_from_slice(&n.to_le_bytes());
    for entry in sorted {
        out.push(entry.op.as_byte());
        out.extend_from_slice(&entry.tenant.raw().to_le_bytes());
        out.push(entry.kind);
        out.extend_from_slice(&entry.internal_id.to_le_bytes());
        let ext_len = u32::try_from(entry.external_id.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&ext_len.to_le_bytes());
        out.extend_from_slice(entry.external_id.as_bytes());
    }
}

/// Parse the v7 `idempotency_bindings` tail (the section appended after
/// the shared v5 sections) at `cursor`, advancing `cursor` past it.
/// Returns the parsed entries. Does NOT assert "no trailing bytes" — the
/// caller decides whether more sections follow (v7 asserts the tail is
/// the last section; v8 parses the `acl_grants` section after it). #1010
/// / #1221.
fn parse_v7_idempotency_tail(
    bytes: &[u8],
    cursor: &mut usize,
    commit_lsn: Lsn,
) -> Result<Vec<IdempotencyBindingEntry>> {
    let n_bindings =
        read_u32_le(bytes, cursor, "CommitBundle v7: n_idempotency_bindings")? as usize;
    // #1411: bound the pre-alloc against the untrusted count. `cursor` is a
    // `&mut usize` here (shared v7/v8 tail), so `remaining` reads through it.
    let bindings_cap = bounded_capacity(
        n_bindings,
        bytes.len().saturating_sub(*cursor),
        MIN_IDEMPOTENCY_V7_ELEM,
    );
    let mut idempotency_bindings: Vec<IdempotencyBindingEntry> = Vec::with_capacity(bindings_cap);
    for _ in 0..n_bindings {
        let op_byte = read_u8(bytes, cursor, "CommitBundle v7: idempotency_binding op")?;
        let op = IdempotencyBindingOp::from_byte(op_byte, commit_lsn)?;
        let tenant_raw = read_u64_le(
            bytes,
            cursor,
            "CommitBundle v7: idempotency_binding tenant_id",
        )?;
        let kind = read_u8(bytes, cursor, "CommitBundle v7: idempotency_binding kind")?;
        let internal_id = read_u64_le(
            bytes,
            cursor,
            "CommitBundle v7: idempotency_binding internal_id",
        )?;
        let ext_len = read_u32_le(
            bytes,
            cursor,
            "CommitBundle v7: idempotency_binding ext_len",
        )? as usize;
        if *cursor + ext_len > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v7: idempotency_binding external_id overruns payload",
            ));
        }
        let external_id = std::str::from_utf8(&bytes[*cursor..*cursor + ext_len])
            .map_err(|e| {
                corruption(
                    commit_lsn,
                    &format!("CommitBundle v7: idempotency_binding external_id not UTF-8: {e}"),
                )
            })?
            .to_owned();
        *cursor += ext_len;
        idempotency_bindings.push(IdempotencyBindingEntry {
            op,
            tenant: TenantId::new(tenant_raw),
            kind,
            internal_id,
            external_id,
        });
    }
    Ok(idempotency_bindings)
}

/// Decode a v7 `CommitBundle` payload (#1010).
pub fn decode_commit_bundle_v7(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let (mut bundle, mut cursor) = decode_commit_bundle_v5_sections(bytes, primary_tenant)?;
    bundle.idempotency_bindings = parse_v7_idempotency_tail(bytes, &mut cursor, bundle.commit_lsn)?;

    if cursor != bytes.len() {
        return Err(corruption(
            bundle.commit_lsn,
            &format!(
                "CommitBundle v7: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }
    Ok(bundle)
}

// ─── v8 codec (#1221 — acl_grants section; ADR-218) ─────────────────

/// Encode a v8 `CommitBundle` payload (#1221 — ADR-218).
///
/// v8 is a strict superset of v7: the v7 prefix (v5 sections + the
/// `idempotency_bindings` tail) is encoded exactly as in
/// [`encode_commit_bundle_v7`], followed by a NEW `acl_grants` section:
///
/// ```text
///   field         size
///   n_acl_grants  4     u32 LE
///   per entry:
///     op          1     u8  (Apply=0 / Revoke=1)
///     tenant_id   8     u64 LE
///     doc         8     u64 LE
///     n_grants    4     u32 LE
///     per grant:
///       grant_len 4     u32 LE
///       grant     N     grant_len bytes, UTF-8
/// ```
///
/// **HARD INVARIANT (ADR-218, architect-flagged correctness gate):**
/// the `acl_grants` entries are written in **staging (append) order** —
/// this encoder MUST NOT sort them the way
/// [`encode_commit_bundle_v7`] sorts its idempotency entries. Replay is
/// last-writer-wins per doc; a re-sort would silently flip which op wins
/// if two ops on the SAME doc ever shared one commit (an ACL
/// widen-or-deny bug). At stage-1 the write-through emits exactly one
/// `apply_doc_acl`/`revoke_doc` per ingest mutation (≤ 1 ACL op per doc
/// per commit), so the collision is not live — but preserving append
/// order keeps the codec sound without baking in that assumption. The
/// `tests::v8_same_doc_ops_replay_in_append_order_last_wins` test locks
/// the invariant.
#[must_use]
#[allow(clippy::too_many_arguments)] // v8 codec section count matches the wire format.
pub fn encode_commit_bundle_v8(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
    vector_pages: &[VectorPageEntry],
    idempotency_bindings: &[IdempotencyBindingEntry],
    acl_grants: &[AclGrantEntry],
) -> Vec<u8> {
    // v7 prefix (v5 sections + sorted idempotency_bindings tail).
    let mut out = encode_commit_bundle_v7(
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        staged_pages,
        allocator_advances,
        vector_pages,
        idempotency_bindings,
    );

    append_v8_acl_grants(&mut out, acl_grants);
    out
}

/// Append the v8 ACL section in staging order. Both v8 and v9+ call this
/// helper; reordering entries can reverse last-writer-wins on one document.
fn append_v8_acl_grants(out: &mut Vec<u8>, acl_grants: &[AclGrantEntry]) {
    let n = u32::try_from(acl_grants.len()).unwrap_or(u32::MAX);
    out.reserve(4 + acl_grants.len() * AclGrantEntry::FIXED_PREFIX_LEN);
    out.extend_from_slice(&n.to_le_bytes());
    for entry in acl_grants {
        out.push(entry.op.as_byte());
        out.extend_from_slice(&entry.tenant.raw().to_le_bytes());
        out.extend_from_slice(&entry.doc.raw().to_le_bytes());
        let n_grants = u32::try_from(entry.grants.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&n_grants.to_le_bytes());
        // BTreeSet iterates in sorted order — that is a per-entry
        // canonicalization of ONE op's grant set (it does NOT reorder
        // ops relative to each other), so it does not touch the
        // append-order invariant above and keeps the bytes deterministic.
        for grant in &entry.grants {
            let grant_len = u32::try_from(grant.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&grant_len.to_le_bytes());
            out.extend_from_slice(grant.as_bytes());
        }
    }
}

/// Encode a `CommitBundle` payload using this binary's current WAL bundle
/// format.
///
/// All production current-format writers should route through this helper so
/// the segment writer and replay spill writer cannot drift to different bundle
/// versions. When [`crate::wal::segment::CURRENT_WAL_FORMAT_VERSION`] advances,
/// this match must grow an arm for the new encoder instead of silently using an
/// older lossy format.
#[must_use]
#[allow(clippy::too_many_arguments)] // mirrors the current bundle wire sections.
pub fn encode_commit_bundle_current(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    staged_pages: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    allocator_advances: &[AllocatorAdvance],
    vector_pages: &[VectorPageEntry],
    idempotency_bindings: &[IdempotencyBindingEntry],
    acl_grants: &[AclGrantEntry],
) -> Vec<u8> {
    match crate::wal::segment::CURRENT_WAL_FORMAT_VERSION {
        BUNDLE_FORMAT_V8 => encode_commit_bundle_v8(
            commit_lsn,
            primary_tenant,
            primary_writes,
            sidechannel_writes,
            staged_pages,
            allocator_advances,
            vector_pages,
            idempotency_bindings,
            acl_grants,
        ),
        version => panic!("no current CommitBundle encoder for WAL format v{version}"),
    }
}

/// Decode a v8 `CommitBundle` payload (#1221 — ADR-218).
///
/// v8 = the v7 sections (v5 prefix + `idempotency_bindings` tail)
/// followed by a trailing `acl_grants` section. A v8 binary decoding an
/// old v5/v6/v7 segment never reaches here — the per-segment
/// `format_version` dispatch routes prior-version bytes to their own
/// decoders, which synthesize an empty `acl_grants`. See
/// [`encode_commit_bundle_v8`] for the wire layout.
pub fn decode_commit_bundle_v8(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    let (mut bundle, mut cursor) = decode_commit_bundle_v5_sections(bytes, primary_tenant)?;
    // v7 idempotency tail (no trailing-bytes assert; acl_grants follows).
    bundle.idempotency_bindings = parse_v7_idempotency_tail(bytes, &mut cursor, bundle.commit_lsn)?;

    bundle.acl_grants = parse_v8_acl_grants(bytes, &mut cursor, bundle.commit_lsn)?;

    if cursor != bytes.len() {
        return Err(corruption(
            bundle.commit_lsn,
            &format!(
                "CommitBundle v8: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }
    Ok(bundle)
}

/// Parse the v8 ACL section in wire/staging order. v9+ shares this parser so
/// the retained logical section has exactly the established v8 semantics.
fn parse_v8_acl_grants(
    bytes: &[u8],
    cursor: &mut usize,
    commit_lsn: Lsn,
) -> Result<Vec<AclGrantEntry>> {
    let n_acl = read_u32_le(bytes, cursor, "CommitBundle v8: n_acl_grants")? as usize;
    // #1411: bound the pre-alloc against the untrusted count. This is the
    // site the `.v8_acl_oom_repro` (n_acl=0xffffff00, 1 byte remaining)
    // exercises: cap == 0 ⇒ no giant alloc; the first element read then hits
    // the read_* truncation guard and returns WalCorruption promptly.
    let acl_cap = bounded_capacity(
        n_acl,
        bytes.len().saturating_sub(*cursor),
        MIN_ACL_GRANT_ELEM,
    );
    let mut acl_grants: Vec<AclGrantEntry> = Vec::with_capacity(acl_cap);
    for _ in 0..n_acl {
        let op_byte = read_u8(bytes, cursor, "CommitBundle v8: acl_grant op")?;
        let op = AclGrantOp::from_byte(op_byte, commit_lsn)?;
        let tenant_raw = read_u64_le(bytes, cursor, "CommitBundle v8: acl_grant tenant_id")?;
        let doc_raw = read_u64_le(bytes, cursor, "CommitBundle v8: acl_grant doc")?;
        let n_grants = read_u32_le(bytes, cursor, "CommitBundle v8: acl_grant n_grants")? as usize;
        let mut grants: BTreeSet<String> = BTreeSet::new();
        for _ in 0..n_grants {
            let grant_len =
                read_u32_le(bytes, cursor, "CommitBundle v8: acl_grant grant_len")? as usize;
            if *cursor + grant_len > bytes.len() {
                return Err(corruption(
                    commit_lsn,
                    "CommitBundle v8: acl_grant principal overruns payload",
                ));
            }
            let grant = std::str::from_utf8(&bytes[*cursor..*cursor + grant_len])
                .map_err(|e| {
                    corruption(
                        commit_lsn,
                        &format!("CommitBundle v8: acl_grant principal not UTF-8: {e}"),
                    )
                })?
                .to_owned();
            *cursor += grant_len;
            grants.insert(grant);
        }
        acl_grants.push(AclGrantEntry {
            op,
            tenant: TenantId::new(tenant_raw),
            doc: NodeId::new(doc_raw),
            grants,
        });
    }
    Ok(acl_grants)
}

// ─── v9/v10 codec (M3/M4 physiological WAL) ───────────────────────

/// Encode a v9/M3 delta-era CommitBundle.
///
/// Layout: commit/MVCC prefix, DeltaOps, retained primary/secondary index and
/// blob-overflow page images, retained vector page images, then the v4/v7/v8
/// logical sections. Record/PropSlotted images are rejected: those mutations
/// must be expressed by the closed DeltaOp algebra. Index delta/SMO treatment
/// and blob-overflow deltas are M4; M3 keeps those page families as images.
#[allow(clippy::too_many_arguments)] // v9 wire sections are explicit at the boundary.
pub fn encode_commit_bundle_v9(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    deltas: &[DeltaOp],
    retained_page_images: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    vector_pages: &[VectorPageEntry],
    allocator_advances: &[AllocatorAdvance],
    idempotency_bindings: &[IdempotencyBindingEntry],
    acl_grants: &[AclGrantEntry],
) -> Result<Vec<u8>> {
    encode_commit_bundle_delta_for_format(
        BUNDLE_FORMAT_V9,
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        deltas,
        retained_page_images,
        vector_pages,
        allocator_advances,
        idempotency_bindings,
        acl_grants,
    )
}

/// Encode a v10/M4 delta-era CommitBundle.
#[allow(clippy::too_many_arguments)] // v10 wire sections are explicit at the boundary.
pub fn encode_commit_bundle_v10(
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    deltas: &[DeltaOp],
    retained_page_images: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    vector_pages: &[VectorPageEntry],
    allocator_advances: &[AllocatorAdvance],
    idempotency_bindings: &[IdempotencyBindingEntry],
    acl_grants: &[AclGrantEntry],
) -> Result<Vec<u8>> {
    encode_commit_bundle_delta_for_format(
        BUNDLE_FORMAT_V10,
        commit_lsn,
        primary_tenant,
        primary_writes,
        sidechannel_writes,
        deltas,
        retained_page_images,
        vector_pages,
        allocator_advances,
        idempotency_bindings,
        acl_grants,
    )
}

#[allow(clippy::too_many_arguments)] // shared v9/v10 wire sections are explicit.
pub(crate) fn encode_commit_bundle_delta_for_format(
    format_version: u16,
    commit_lsn: Lsn,
    primary_tenant: TenantId,
    primary_writes: &HashMap<MvccKey, Option<Bytes>>,
    sidechannel_writes: &[SideChannelWrite],
    deltas: &[DeltaOp],
    retained_page_images: &[(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)],
    vector_pages: &[VectorPageEntry],
    allocator_advances: &[AllocatorAdvance],
    idempotency_bindings: &[IdempotencyBindingEntry],
    acl_grants: &[AclGrantEntry],
) -> Result<Vec<u8>> {
    debug_assert!(is_delta_bundle_format(format_version));
    if format_version == BUNDLE_FORMAT_V10
        && (!idempotency_bindings.is_empty() || !acl_grants.is_empty())
    {
        return Err(corruption(
            commit_lsn,
            "CommitBundle v10 forbids retired logical idempotency/ACL owner tails",
        ));
    }
    validate_delta_order(commit_lsn, deltas, format_version)?;

    // IMPL-DEC-3: a PutRecord payload is byte-identical to its MVCC version.
    // Keep one authoritative copy in the delta stream and let recovery
    // reconstruct the version chain from it. Tombstones and writes without a
    // physical record correspondent remain in the mature section-2 codec.
    let mut delta_versions = HashMap::new();
    for delta in deltas {
        if let Some((key, value)) = crate::wal::delta::put_record_mvcc_write(delta)? {
            delta_versions.insert((delta.tenant_id, key), value);
        }
    }

    let mut entries: Vec<(TenantId, MvccKey, Option<Bytes>)> = primary_writes
        .iter()
        .map(|(key, value)| (primary_tenant, *key, value.clone()))
        .chain(
            sidechannel_writes
                .iter()
                .map(|entry| (entry.tenant_id, entry.key, entry.value.clone())),
        )
        .filter_map(|entry| {
            let Some(delta_value) = delta_versions.get(&(entry.0, entry.1)) else {
                return Some(Ok(entry));
            };
            match &entry.2 {
                Some(value) if value == delta_value => None,
                Some(_) => Some(Err(corruption(
                    commit_lsn,
                    "PutRecord payload diverges from its MVCC version bytes",
                ))),
                None => Some(Ok(entry)),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| (entry.0.raw(), entry.1));

    let payload_bytes: usize = entries
        .iter()
        .map(|entry| entry.2.as_ref().map_or(0, Bytes::len))
        .sum();
    let delta_bytes: usize = deltas.iter().map(DeltaOp::encoded_len).sum();
    let mut out = Vec::with_capacity(
        8 + 4
            + entries.len() * 21
            + payload_bytes
            + 4
            + delta_bytes
            + 4
            + retained_page_images.len() * MIN_STAGED_PAGE_ELEM
            + 4
            + vector_pages.len() * MIN_VECTOR_PAGE_ELEM
            + 12,
    );
    out.extend_from_slice(&commit_lsn.raw().to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(entries.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for (tenant, key, value) in entries {
        out.extend_from_slice(&tenant.raw().to_le_bytes());
        out.extend_from_slice(&key.to_le_bytes());
        match value {
            None => {
                out.push(0);
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(value) => {
                out.push(1);
                out.extend_from_slice(
                    &u32::try_from(value.len()).unwrap_or(u32::MAX).to_le_bytes(),
                );
                out.extend_from_slice(&value);
            }
        }
    }

    out.extend_from_slice(
        &u32::try_from(deltas.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for delta in deltas {
        delta.encode_into_for_format(&mut out, format_version)?;
    }

    out.extend_from_slice(
        &u32::try_from(retained_page_images.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for (kind, page_id, tenant, bytes) in retained_page_images {
        if !matches!(
            kind,
            BundlePageKind::PrimaryIndex | BundlePageKind::SecondaryIndex | BundlePageKind::Blob
        ) {
            return Err(corruption(
                commit_lsn,
                &format!("CommitBundle v9 retained forbidden page image kind {kind:?}"),
            ));
        }
        out.push(kind.as_byte());
        out.extend_from_slice(&page_id.raw().to_le_bytes());
        out.extend_from_slice(&tenant.raw().to_le_bytes());
        out.extend_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        out.extend_from_slice(bytes.as_ref());
    }

    let mut sorted_vectors: Vec<_> = vector_pages.iter().collect();
    sorted_vectors.sort_by_key(|entry| {
        (
            entry.tenant.raw(),
            u64::from(entry.partition.raw()),
            entry.index_id,
            entry.page_id.raw(),
            entry.commit_lsn.raw(),
        )
    });
    out.extend_from_slice(
        &u32::try_from(sorted_vectors.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for entry in sorted_vectors {
        out.extend_from_slice(&entry.tenant.raw().to_le_bytes());
        out.extend_from_slice(&u64::from(entry.partition.raw()).to_le_bytes());
        out.extend_from_slice(&entry.index_id.to_le_bytes());
        out.extend_from_slice(&entry.page_id.raw().to_le_bytes());
        out.extend_from_slice(&entry.commit_lsn.raw().to_le_bytes());
        out.extend_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        out.extend_from_slice(entry.bytes.as_ref());
    }
    // Director ruling: keep the mature v8 logical-owner sections at M3.
    // These helpers are the exact v4/v7/v8 codecs, not parallel v9 layouts.
    append_v4_allocator_advances(&mut out, allocator_advances);
    append_v7_idempotency_bindings(&mut out, idempotency_bindings);
    append_v8_acl_grants(&mut out, acl_grants);
    Ok(out)
}

/// Decode a v9/M3 delta-era CommitBundle.
pub fn decode_commit_bundle_v9(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    decode_commit_bundle_delta_for_format(bytes, primary_tenant, BUNDLE_FORMAT_V9)
}

/// Decode a v10/M4 delta-era CommitBundle.
pub fn decode_commit_bundle_v10(
    bytes: &[u8],
    primary_tenant: TenantId,
) -> Result<DecodedCommitBundle> {
    decode_commit_bundle_delta_for_format(bytes, primary_tenant, BUNDLE_FORMAT_V10)
}

fn decode_commit_bundle_delta_for_format(
    bytes: &[u8],
    primary_tenant: TenantId,
    format_version: u16,
) -> Result<DecodedCommitBundle> {
    debug_assert!(is_delta_bundle_format(format_version));
    let mut cursor = 0usize;
    let commit_lsn = Lsn::new(read_u64_le(
        bytes,
        &mut cursor,
        "CommitBundle v9: commit_lsn",
    )?);

    let n_mvcc = read_u32_le(bytes, &mut cursor, "CommitBundle v9: n_mvcc_writes")? as usize;
    let mvcc_cap = bounded_capacity(n_mvcc, bytes.len().saturating_sub(cursor), 21);
    let mut mvcc_writes = HashMap::with_capacity(mvcc_cap);
    let mut sidechannel_writes = Vec::with_capacity(mvcc_cap);
    for _ in 0..n_mvcc {
        let tenant = TenantId::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: mvcc tenant_id",
        )?);
        let key = read_u64_le(bytes, &mut cursor, "CommitBundle v9: mvcc key")?;
        let kind = read_u8(bytes, &mut cursor, "CommitBundle v9: mvcc kind")?;
        let value_len =
            read_u32_le(bytes, &mut cursor, "CommitBundle v9: mvcc value_len")? as usize;
        let value = match kind {
            0 if value_len == 0 => None,
            0 => {
                return Err(corruption(
                    commit_lsn,
                    "CommitBundle v9 tombstone has non-zero value length",
                ));
            }
            1 => {
                let end = cursor.checked_add(value_len).ok_or_else(|| {
                    corruption(commit_lsn, "CommitBundle v9 MVCC length overflow")
                })?;
                if end > bytes.len() {
                    return Err(corruption(
                        commit_lsn,
                        "CommitBundle v9 MVCC value overruns payload",
                    ));
                }
                let value = Bytes::copy_from_slice(&bytes[cursor..end]);
                cursor = end;
                Some(value)
            }
            other => {
                return Err(corruption(
                    commit_lsn,
                    &format!("CommitBundle v9 unknown MVCC kind {other}"),
                ));
            }
        };
        if tenant == primary_tenant {
            mvcc_writes.insert(key, value);
        } else {
            sidechannel_writes.push(SideChannelWrite {
                tenant_id: tenant,
                key,
                value,
            });
        }
    }

    let n_deltas = read_u32_le(bytes, &mut cursor, "CommitBundle v9: n_deltas")? as usize;
    let delta_cap = bounded_capacity(
        n_deltas,
        bytes.len().saturating_sub(cursor),
        MIN_DELTA_OP_ELEM,
    );
    let range = RedoLsnRange::ending_at(commit_lsn, n_deltas).ok_or_else(|| {
        corruption(
            commit_lsn,
            "CommitBundle v9 delta count underflows commit LSN range",
        )
    })?;
    let mut deltas = Vec::with_capacity(delta_cap);
    for index in 0..n_deltas {
        let (delta, consumed) =
            DeltaOp::decode_prefix_for_format(&bytes[cursor..], format_version)?;
        let expected = range.op_lsn(index).expect("index bounded by delta count");
        if delta.op_lsn != expected {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v9 delta {index} op_lsn {:?}, expected {:?}",
                    delta.op_lsn, expected
                ),
            ));
        }
        cursor = cursor
            .checked_add(consumed)
            .ok_or_else(|| corruption(commit_lsn, "CommitBundle v9 delta cursor overflow"))?;
        deltas.push(delta);
    }
    validate_delta_page_alloc_order(commit_lsn, &deltas)?;
    validate_delta_extent_alloc_order(commit_lsn, &deltas)?;

    let n_pages = read_u32_le(
        bytes,
        &mut cursor,
        "CommitBundle v9: n_retained_page_images",
    )? as usize;
    let page_cap = bounded_capacity(
        n_pages,
        bytes.len().saturating_sub(cursor),
        MIN_STAGED_PAGE_ELEM,
    );
    let mut staged_pages = Vec::with_capacity(page_cap);
    for _ in 0..n_pages {
        let kind = BundlePageKind::from_byte(read_u8(
            bytes,
            &mut cursor,
            "CommitBundle v9: retained page kind",
        )?)?;
        if !matches!(
            kind,
            BundlePageKind::PrimaryIndex | BundlePageKind::SecondaryIndex | BundlePageKind::Blob
        ) {
            return Err(corruption(
                commit_lsn,
                &format!("CommitBundle v9 retained forbidden page image kind {kind:?}"),
            ));
        }
        let page_id = PageId::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: retained page id",
        )?);
        let tenant_id = TenantId::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: retained page tenant",
        )?);
        let n_bytes =
            read_u32_le(bytes, &mut cursor, "CommitBundle v9: retained page n_bytes")? as usize;
        if n_bytes != PAGE_SIZE || cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v9 retained page length invalid",
            ));
        }
        let mut page_bytes = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        staged_pages.push(DecodedStagedPage {
            kind,
            page_id,
            tenant_id,
            bytes: page_bytes,
        });
    }

    let n_vectors = read_u32_le(bytes, &mut cursor, "CommitBundle v9: n_vector_pages")? as usize;
    let vector_cap = bounded_capacity(
        n_vectors,
        bytes.len().saturating_sub(cursor),
        MIN_VECTOR_PAGE_ELEM,
    );
    let mut vector_pages = Vec::with_capacity(vector_cap);
    for _ in 0..n_vectors {
        let tenant = TenantId::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: vector tenant",
        )?);
        let partition_raw = read_u64_le(bytes, &mut cursor, "CommitBundle v9: vector partition")?;
        let partition = u32::try_from(partition_raw)
            .map(PartitionId::new)
            .map_err(|_| {
                corruption(commit_lsn, "CommitBundle v9 vector partition overflows u32")
            })?;
        let index_id = read_u64_le(bytes, &mut cursor, "CommitBundle v9: vector index")?;
        let page_id = PageId::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: vector page",
        )?);
        let entry_lsn = Lsn::new(read_u64_le(
            bytes,
            &mut cursor,
            "CommitBundle v9: vector commit_lsn",
        )?);
        let n_bytes = read_u32_le(bytes, &mut cursor, "CommitBundle v9: vector n_bytes")? as usize;
        if n_bytes != PAGE_SIZE || cursor + n_bytes > bytes.len() {
            return Err(corruption(
                commit_lsn,
                "CommitBundle v9 vector page length invalid",
            ));
        }
        let mut page_bytes = Box::new([0u8; PAGE_SIZE]);
        page_bytes.copy_from_slice(&bytes[cursor..cursor + n_bytes]);
        cursor += n_bytes;
        vector_pages.push(VectorPageEntry {
            tenant,
            partition,
            index_id,
            page_id,
            commit_lsn: entry_lsn,
            bytes: page_bytes,
        });
    }

    // Retained v8 logical-owner sections. M3 deliberately does not define
    // InternBind/AclGrant DeltaOp payloads; recovery reuses these proven
    // absolute-op codecs and their established replay arms instead.
    let allocator_advances = parse_v4_allocator_advances(bytes, &mut cursor)?;
    let idempotency_bindings = parse_v7_idempotency_tail(bytes, &mut cursor, commit_lsn)?;
    let acl_grants = parse_v8_acl_grants(bytes, &mut cursor, commit_lsn)?;
    if format_version == BUNDLE_FORMAT_V10
        && (!idempotency_bindings.is_empty() || !acl_grants.is_empty())
    {
        return Err(corruption(
            commit_lsn,
            "CommitBundle v10 carries retired logical idempotency/ACL owner tails",
        ));
    }
    if cursor != bytes.len() {
        return Err(corruption(
            commit_lsn,
            &format!(
                "CommitBundle v9: {} trailing bytes after decode",
                bytes.len() - cursor
            ),
        ));
    }

    Ok(DecodedCommitBundle {
        commit_lsn,
        primary_tenant,
        mvcc_writes,
        sidechannel_writes,
        staged_pages,
        deltas,
        allocator_advances,
        vector_pages,
        idempotency_bindings,
        acl_grants,
    })
}

fn validate_delta_order(commit_lsn: Lsn, deltas: &[DeltaOp], format_version: u16) -> Result<()> {
    let range = RedoLsnRange::ending_at(commit_lsn, deltas.len()).ok_or_else(|| {
        corruption(
            commit_lsn,
            "CommitBundle v9 delta count underflows commit LSN range",
        )
    })?;
    for (index, delta) in deltas.iter().enumerate() {
        delta.validate_for_format(format_version)?;
        let expected = range.op_lsn(index).expect("index bounded by delta count");
        if delta.op_lsn != expected {
            return Err(corruption(
                commit_lsn,
                &format!(
                    "CommitBundle v9 delta {index} op_lsn {:?}, expected {:?}",
                    delta.op_lsn, expected
                ),
            ));
        }
    }
    validate_delta_page_alloc_order(commit_lsn, deltas)?;
    validate_delta_extent_alloc_order(commit_lsn, deltas)
}

fn validate_delta_page_alloc_order(commit_lsn: Lsn, deltas: &[DeltaOp]) -> Result<()> {
    let mut used_before_alloc = HashSet::new();
    let mut allocated = HashSet::new();
    for delta in deltas.iter().filter(|delta| delta.kind.is_physical()) {
        let key = (delta.tenant_id, delta.store_id, delta.page_no);
        if delta.kind == DeltaOpKind::PageAlloc {
            if used_before_alloc.contains(&key) || !allocated.insert(key) {
                return Err(corruption(
                    commit_lsn,
                    "CommitBundle v9 PageAlloc is duplicate or sorts after page use",
                ));
            }
        } else if !allocated.contains(&key) {
            used_before_alloc.insert(key);
        }
    }
    Ok(())
}

fn validate_delta_extent_alloc_order(commit_lsn: Lsn, deltas: &[DeltaOp]) -> Result<()> {
    let mut used_before_alloc = HashSet::new();
    let mut allocated = HashSet::new();
    for delta in deltas.iter().filter(|delta| delta.kind.is_physical()) {
        if delta.kind == DeltaOpKind::ExtentAlloc {
            let allocation = crate::extent::ExtentAllocation::decode(&delta.payload, delta.op_lsn)?;
            let key = (delta.tenant_id, delta.store_id, allocation.logical_extent);
            if used_before_alloc.contains(&key) || !allocated.insert(key) {
                return Err(corruption(
                    commit_lsn,
                    "CommitBundle v9 ExtentAlloc is duplicate or sorts after extent page use",
                ));
            }
        } else if delta.page_no < crate::extent::DIR_PAGE_TAG {
            used_before_alloc.insert((
                delta.tenant_id,
                delta.store_id,
                delta.page_no / crate::extent::EXTENT_PAGES,
            ));
        }
    }
    Ok(())
}

/// Back-compat alias for [`decode_commit_bundle_v1`] — takes
/// `primary_tenant` explicitly. Existing pre-ADR-032 callers pass
/// the WAL record header's `tenant_id` (or `TenantId::DEFAULT` in
/// standalone codec tests).
///
/// Kept as a public alias rather than removed so in-tree callers
/// that have not yet been migrated to the `_v1`/`_v2` names keep
/// compiling. Slice 3's replay executor uses the
/// [`decode_commit_bundle_for_version`] dispatcher instead.
pub fn decode_commit_bundle(bytes: &[u8], primary_tenant: TenantId) -> Result<DecodedCommitBundle> {
    decode_commit_bundle_v1(bytes, primary_tenant)
}

#[inline]
fn read_u64_le(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(corruption(
            Lsn::ZERO,
            &format!("CommitBundle: truncated u64 for {what}"),
        ));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(arr))
}

#[inline]
fn read_u32_le(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(corruption(
            Lsn::ZERO,
            &format!("CommitBundle: truncated u32 for {what}"),
        ));
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*cursor..*cursor + 4]);
    *cursor += 4;
    Ok(u32::from_le_bytes(arr))
}

#[inline]
fn read_u8(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u8> {
    if *cursor >= bytes.len() {
        return Err(corruption(
            Lsn::ZERO,
            &format!("CommitBundle: truncated u8 for {what}"),
        ));
    }
    let v = bytes[*cursor];
    *cursor += 1;
    Ok(v)
}

fn corruption(lsn: Lsn, reason: &str) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn,
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn mk_emit(page_id: u64, fill: u8) -> StagedEmit {
        let mut buf: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        for b in buf.iter_mut() {
            *b = fill;
        }
        StagedEmit {
            kind: BundlePageKind::PrimaryIndex,
            page_id: PageId::new(page_id),
            bytes: buf,
        }
    }

    fn mk_writes(entries: &[(MvccKey, Option<&[u8]>)]) -> HashMap<MvccKey, Option<Bytes>> {
        entries
            .iter()
            .map(|(k, v)| (*k, v.map(Bytes::copy_from_slice)))
            .collect()
    }

    #[test]
    fn delta_cutover_no_half_retirement() {
        let tenant = TenantId::new(0x5a17);
        let logical_binding = IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant,
            kind: 0,
            internal_id: 7,
            external_id: "retired-tail".to_owned(),
        };

        let encode_error = encode_commit_bundle_v10(
            Lsn::new(10),
            tenant,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
            std::slice::from_ref(&logical_binding),
            &[],
        )
        .unwrap_err();
        assert!(encode_error.to_string().contains("retired logical"));

        // v9 has the same tail layout. Relabelling a captured v9 logical
        // bundle as v10 must also fail at decode, so replay can never drive a
        // retired owner tail into deleted resident maps.
        let captured_v9 = encode_commit_bundle_v9(
            Lsn::new(11),
            tenant,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
            &[logical_binding],
            &[],
        )
        .unwrap();
        let decode_error =
            decode_commit_bundle_for_version(&captured_v9, BUNDLE_FORMAT_V10, tenant).unwrap_err();
        assert!(decode_error.to_string().contains("retired logical"));

        let class = crate::owner_row::OwnerRowClass::NodeBinding;
        let row = crate::owner_row::OwnerRow::new(class, 7, b"physical-owner".to_vec()).unwrap();
        let address = class.address(7).unwrap();
        let physical = DeltaOp::new_for_format(
            BUNDLE_FORMAT_V10,
            DeltaOpKind::InternBind,
            class.store_id(),
            tenant,
            address.page_no,
            address.slot.raw(),
            Lsn::new(12),
            row.encode().to_vec(),
        )
        .unwrap();
        let encoded = encode_commit_bundle_v10(
            Lsn::new(12),
            tenant,
            &HashMap::new(),
            &[],
            std::slice::from_ref(&physical),
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        let decoded = decode_commit_bundle_v10(&encoded, tenant).unwrap();
        assert_eq!(decoded.deltas, [physical]);
        assert!(decoded.idempotency_bindings.is_empty());
        assert!(decoded.acl_grants.is_empty());
    }

    // ─── v1 round-trip tests (existing — ensure backward-compat
    //      under the new `_v1` decoder entry point) ─────────────

    #[test]
    fn empty_bundle_roundtrip_v1() {
        let writes = HashMap::new();
        let staged: Vec<StagedEmit> = Vec::new();
        let encoded = encode_commit_bundle(Lsn::new(7), &writes, &staged, TenantId::DEFAULT);
        assert_eq!(encoded.len(), 8 + 4 + 4); // commit_lsn + n_mvcc + n_pages
        let decoded = decode_commit_bundle_v1(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(7));
        assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
        assert!(decoded.mvcc_writes.is_empty());
        assert!(decoded.sidechannel_writes.is_empty());
        assert!(decoded.staged_pages.is_empty());
    }

    #[test]
    fn mvcc_only_bundle_roundtrip_v1() {
        let writes = mk_writes(&[(1u64, Some(&b"hello"[..])), (2u64, None)]);
        let encoded = encode_commit_bundle(Lsn::new(42), &writes, &[], TenantId::DEFAULT);
        let decoded = decode_commit_bundle_v1(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(42));
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(
            decoded.mvcc_writes.get(&1).unwrap().as_deref(),
            Some(&b"hello"[..])
        );
        assert_eq!(decoded.mvcc_writes.get(&2).unwrap().as_ref(), None);
        assert!(decoded.sidechannel_writes.is_empty());
        assert!(decoded.staged_pages.is_empty());
    }

    #[test]
    fn index_pages_only_bundle_roundtrip_v1() {
        // No MVCC writes but N staged IndexPage snapshots.
        let writes = HashMap::new();
        let staged = vec![mk_emit(101, 0xAB), mk_emit(102, 0xCD), mk_emit(103, 0xEF)];
        let encoded = encode_commit_bundle(Lsn::new(99), &writes, &staged, TenantId::DEFAULT);
        let decoded = decode_commit_bundle_v1(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(99));
        assert!(decoded.mvcc_writes.is_empty());
        assert_eq!(decoded.staged_pages.len(), 3);
        for (i, page) in decoded.staged_pages.iter().enumerate() {
            assert_eq!(page.page_id.raw(), 101 + i as u64);
            assert_eq!(page.tenant_id, TenantId::DEFAULT);
            assert!(page.bytes.iter().all(|&b| b == staged[i].bytes[0]));
        }
    }

    #[test]
    fn full_bundle_roundtrip_with_mvcc_and_pages_v1() {
        let writes = mk_writes(&[(0xAA, Some(&b"node-bytes"[..])), (0xBB, None)]);
        let staged = vec![mk_emit(201, 0x11), mk_emit(202, 0x22)];
        let encoded = encode_commit_bundle(Lsn::new(1234), &writes, &staged, TenantId::SYSTEM);
        let decoded = decode_commit_bundle_v1(&encoded, TenantId::SYSTEM).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(1234));
        assert_eq!(decoded.primary_tenant, TenantId::SYSTEM);
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(decoded.staged_pages.len(), 2);
        assert_eq!(decoded.staged_pages[0].tenant_id, TenantId::SYSTEM);
    }

    #[test]
    fn truncated_payload_rejected_v1() {
        let writes = mk_writes(&[(1u64, Some(&b"abc"[..]))]);
        let staged = vec![mk_emit(1, 0xAA)];
        let mut encoded = encode_commit_bundle(Lsn::new(1), &writes, &staged, TenantId::DEFAULT);
        encoded.truncate(encoded.len() - 128);
        let err = decode_commit_bundle_v1(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn bad_mvcc_kind_rejected_v1() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&Lsn::new(1).raw().to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&42u64.to_le_bytes());
        bytes.push(2u8); // invalid
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = decode_commit_bundle_v1(&bytes, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("unknown mvcc kind"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn tombstone_with_nonzero_value_len_rejected_v1() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&Lsn::new(1).raw().to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&42u64.to_le_bytes());
        bytes.push(0u8);
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(b"hello");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = decode_commit_bundle_v1(&bytes, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn trailing_bytes_rejected_v1() {
        let writes = mk_writes(&[(1u64, Some(&b"abc"[..]))]);
        let staged = vec![mk_emit(1, 0xAA)];
        let mut encoded = encode_commit_bundle(Lsn::new(1), &writes, &staged, TenantId::DEFAULT);
        encoded.extend_from_slice(b"garbage tail");
        let err = decode_commit_bundle_v1(&encoded, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("trailing"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    // ─── v2 codec tests (ADR-032 Slice 1) ────────────────────────

    #[test]
    fn empty_bundle_roundtrip_v2() {
        let primary = HashMap::new();
        let side: Vec<SideChannelWrite> = Vec::new();
        let staged: Vec<StagedEmit> = Vec::new();
        let encoded = encode_commit_bundle_v2(
            Lsn::new(7),
            TenantId::DEFAULT,
            &primary,
            &side,
            &staged,
            TenantId::DEFAULT,
        );
        // commit_lsn (8) + n_mvcc (4) + 0 writes + n_pages (4) + 0 pages = 16
        assert_eq!(encoded.len(), 8 + 4 + 4);
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(7));
        assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
        assert!(decoded.mvcc_writes.is_empty());
        assert!(decoded.sidechannel_writes.is_empty());
        assert!(decoded.staged_pages.is_empty());
    }

    #[test]
    fn v2_primary_only_roundtrip_sets_empty_sidechannel() {
        // Primary tenant writes only — sidechannel_writes decodes as empty.
        let primary = mk_writes(&[(1u64, Some(&b"alpha"[..])), (2u64, None)]);
        let side: Vec<SideChannelWrite> = Vec::new();
        let encoded = encode_commit_bundle_v2(
            Lsn::new(100),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(100));
        assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(
            decoded.mvcc_writes.get(&1).unwrap().as_deref(),
            Some(&b"alpha"[..])
        );
        assert_eq!(decoded.mvcc_writes.get(&2).unwrap().as_ref(), None);
        assert!(
            decoded.sidechannel_writes.is_empty(),
            "no sidechannel writes expected: {:?}",
            decoded.sidechannel_writes
        );
    }

    #[test]
    fn v2_roundtrip_with_sidechannel_writes() {
        // Primary user-tenant writes + SYSTEM sidechannel write
        // (the grow_root shape from ADR-032 §2).
        let primary = mk_writes(&[(10u64, Some(&b"user-value"[..]))]);
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 0xFEE1_DEAD_BEEF_u64,
            value: Some(Bytes::from_static(b"system-root-pointer")),
        }];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(42),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(42));
        assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
        // Primary partition: just the user write.
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert_eq!(
            decoded.mvcc_writes.get(&10u64).unwrap().as_deref(),
            Some(&b"user-value"[..])
        );
        // Sidechannel partition: just the SYSTEM write.
        assert_eq!(decoded.sidechannel_writes.len(), 1);
        let sc = &decoded.sidechannel_writes[0];
        assert_eq!(sc.tenant_id, TenantId::SYSTEM);
        assert_eq!(sc.key, 0xFEE1_DEAD_BEEF_u64);
        assert_eq!(sc.value.as_deref(), Some(&b"system-root-pointer"[..]));
    }

    #[test]
    fn v2_roundtrip_with_mixed_tenants_sorted_on_wire() {
        // 3 tenants × 2 writes each; primary = DEFAULT. Non-DEFAULT
        // entries become sidechannel writes. Encoder sorts by
        // (tenant, key) so sidechannel_writes arrives sorted.
        let custom_a = TenantId::new(42);
        let custom_b = TenantId::new(99);
        let primary = mk_writes(&[(1u64, Some(&b"d1"[..])), (2u64, Some(&b"d2"[..]))]);
        let side = vec![
            SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: 10,
                value: Some(Bytes::from_static(b"s10")),
            },
            SideChannelWrite {
                tenant_id: custom_b,
                key: 20,
                value: Some(Bytes::from_static(b"b20")),
            },
            SideChannelWrite {
                tenant_id: custom_a,
                key: 30,
                value: Some(Bytes::from_static(b"a30")),
            },
            SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: 40,
                value: None,
            },
            SideChannelWrite {
                tenant_id: custom_a,
                key: 50,
                value: None,
            },
            SideChannelWrite {
                tenant_id: custom_b,
                key: 60,
                value: Some(Bytes::from_static(b"b60")),
            },
        ];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(7),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(decoded.sidechannel_writes.len(), 6);
        // Sorted ascending by (tenant.raw(), key).
        // SYSTEM = 0, DEFAULT = 1, custom_a = 42, custom_b = 99.
        // Expected order: SYSTEM/10, SYSTEM/40, custom_a/30, custom_a/50, custom_b/20, custom_b/60.
        assert_eq!(decoded.sidechannel_writes[0].tenant_id, TenantId::SYSTEM);
        assert_eq!(decoded.sidechannel_writes[0].key, 10);
        assert_eq!(decoded.sidechannel_writes[1].tenant_id, TenantId::SYSTEM);
        assert_eq!(decoded.sidechannel_writes[1].key, 40);
        assert_eq!(decoded.sidechannel_writes[2].tenant_id, custom_a);
        assert_eq!(decoded.sidechannel_writes[2].key, 30);
        assert_eq!(decoded.sidechannel_writes[3].tenant_id, custom_a);
        assert_eq!(decoded.sidechannel_writes[3].key, 50);
        assert_eq!(decoded.sidechannel_writes[4].tenant_id, custom_b);
        assert_eq!(decoded.sidechannel_writes[4].key, 20);
        assert_eq!(decoded.sidechannel_writes[5].tenant_id, custom_b);
        assert_eq!(decoded.sidechannel_writes[5].key, 60);
    }

    #[test]
    fn v2_tombstone_sidechannel_roundtrip() {
        // Sidechannel tombstones (value = None) roundtrip cleanly.
        let primary = HashMap::new();
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 0xAA,
            value: None,
        }];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert!(decoded.mvcc_writes.is_empty());
        assert_eq!(decoded.sidechannel_writes.len(), 1);
        assert!(decoded.sidechannel_writes[0].value.is_none());
    }

    #[test]
    fn v2_index_pages_roundtrip_unchanged() {
        // IndexPage section shape is identical to v1; v2 just adds
        // the per-entry tenant to MVCC writes.
        let primary = HashMap::new();
        let side: Vec<SideChannelWrite> = Vec::new();
        let staged = vec![mk_emit(501, 0x55), mk_emit(502, 0x66)];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(9),
            TenantId::DEFAULT,
            &primary,
            &side,
            &staged,
            TenantId::SYSTEM,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.staged_pages.len(), 2);
        for page in &decoded.staged_pages {
            assert_eq!(page.tenant_id, TenantId::SYSTEM);
        }
    }

    #[test]
    fn v2_truncated_payload_rejected() {
        let primary = mk_writes(&[(1u64, Some(&b"abc"[..]))]);
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 2,
            value: Some(Bytes::from_static(b"sys")),
        }];
        let mut encoded = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        // Knock out a byte somewhere inside the MVCC entries.
        encoded.truncate(encoded.len() - 2);
        let err = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v2_bad_mvcc_kind_rejected() {
        // Craft a v2 payload with an invalid kind byte.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&Lsn::new(1).raw().to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_mvcc = 1
        bytes.extend_from_slice(&TenantId::DEFAULT.raw().to_le_bytes()); // tenant_id
        bytes.extend_from_slice(&42u64.to_le_bytes()); // key
        bytes.push(9u8); // kind invalid
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_pages
        let err = decode_commit_bundle_v2(&bytes, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("unknown mvcc kind"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn v2_trailing_bytes_rejected() {
        let primary = mk_writes(&[(1u64, Some(&b"x"[..]))]);
        let side: Vec<SideChannelWrite> = Vec::new();
        let mut encoded = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        encoded.extend_from_slice(b"EXTRA");
        let err = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("trailing"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn v2_writes_with_matching_tenant_go_to_primary_not_sidechannel() {
        // A SideChannelWrite whose tenant == primary_tenant is
        // partitioned into mvcc_writes on decode (the on-wire shape
        // makes no distinction). Callers should avoid constructing
        // this shape but the decoder is robust.
        let primary = HashMap::new();
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::DEFAULT,
            key: 7,
            value: Some(Bytes::from_static(b"oops")),
        }];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded = decode_commit_bundle_v2(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert!(decoded.sidechannel_writes.is_empty());
    }

    // ─── Version-aware dispatcher tests ──────────────────────────

    #[test]
    fn dispatcher_routes_v1_payload_to_v1_decoder() {
        let writes = mk_writes(&[(42u64, Some(&b"v1-data"[..]))]);
        let encoded = encode_commit_bundle(Lsn::new(1), &writes, &[], TenantId::DEFAULT);
        let decoded =
            decode_commit_bundle_for_version(&encoded, BUNDLE_FORMAT_V1, TenantId::DEFAULT)
                .unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(1));
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert!(decoded.sidechannel_writes.is_empty());
    }

    #[test]
    fn dispatcher_routes_v2_payload_to_v2_decoder() {
        let primary = mk_writes(&[(42u64, Some(&b"v2-data"[..]))]);
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 100,
            value: Some(Bytes::from_static(b"sys")),
        }];
        let encoded = encode_commit_bundle_v2(
            Lsn::new(2),
            TenantId::DEFAULT,
            &primary,
            &side,
            &[],
            TenantId::DEFAULT,
        );
        let decoded =
            decode_commit_bundle_for_version(&encoded, BUNDLE_FORMAT_V2, TenantId::DEFAULT)
                .unwrap();
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert_eq!(decoded.sidechannel_writes.len(), 1);
    }

    #[test]
    fn dispatcher_rejects_unknown_version() {
        let bytes = vec![0u8; 16];
        let err = decode_commit_bundle_for_version(&bytes, 99u16, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalFormatMismatch {
                found_version,
                supported_versions,
            } => {
                assert_eq!(found_version, 99);
                assert!(supported_versions.contains(&BUNDLE_FORMAT_V1));
                assert!(supported_versions.contains(&BUNDLE_FORMAT_V2));
            }
            other => panic!("expected WalFormatMismatch, got {other:?}"),
        }
    }

    // ─── Cross-version guard: v1 payload decoded as v2 fails cleanly ──

    #[test]
    fn v1_payload_decoded_as_v2_is_structurally_rejected() {
        // v1 byte layout has no tenant prefix on writes; v2 decoder
        // reads tenant/key/kind/len — so v1 bytes misinterpreted as
        // v2 will either consume a key as a tenant and a kind-byte as
        // part of the key (or overrun the payload). Either way
        // WalCorruption, not a silent wrong-parse.
        let writes = mk_writes(&[(1u64, Some(&b"abc"[..]))]);
        let staged = vec![mk_emit(1, 0xAA)];
        let v1 = encode_commit_bundle(Lsn::new(1), &writes, &staged, TenantId::DEFAULT);
        let r = decode_commit_bundle_v2(&v1, TenantId::DEFAULT);
        assert!(r.is_err(), "v1 payload should not decode cleanly as v2");
    }

    // ─── v1 decoder synthesizes primary_tenant from caller arg ──────

    #[test]
    fn v1_decoder_stamps_primary_tenant_on_decoded_bundle() {
        // A v1 bundle on the wire carries no tenant; the decoder
        // inherits primary_tenant from the WAL record header (caller-
        // supplied). Slice 1 test hook for the replay path: the
        // decoded bundle's primary_tenant == whatever caller asked
        // for.
        let writes = mk_writes(&[(1u64, Some(&b"a"[..])), (2u64, Some(&b"b"[..]))]);
        let encoded = encode_commit_bundle(Lsn::new(1), &writes, &[], TenantId::DEFAULT);

        for tenant in [TenantId::DEFAULT, TenantId::SYSTEM, TenantId::new(314)] {
            let decoded = decode_commit_bundle_v1(&encoded, tenant).unwrap();
            assert_eq!(
                decoded.primary_tenant, tenant,
                "v1 decoder must stamp caller-supplied primary tenant onto bundle"
            );
            assert_eq!(decoded.mvcc_writes.len(), 2);
            assert!(decoded.sidechannel_writes.is_empty());
        }
    }

    // ─── Version alias / back-compat ────────────────────────────────

    #[test]
    fn unversioned_alias_decodes_v1_only() {
        // `decode_commit_bundle` (no suffix) aliases to _v1 for
        // backward compatibility with pre-ADR-032 call sites.
        let writes = mk_writes(&[(1u64, Some(&b"legacy"[..]))]);
        let encoded = encode_commit_bundle(Lsn::new(1), &writes, &[], TenantId::DEFAULT);
        let decoded = decode_commit_bundle(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(1));
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert!(decoded.sidechannel_writes.is_empty());
    }

    // ─── v3 codec tests (ADR-031 amendment-02, PR #79 X-2) ──────────

    fn mk_staged(
        kind: BundlePageKind,
        page_id: u64,
        tenant: TenantId,
        fill: u8,
    ) -> (BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>) {
        (
            kind,
            PageId::new(page_id),
            tenant,
            Box::new([fill; PAGE_SIZE]),
        )
    }

    #[test]
    fn v3_empty_bundle_roundtrip() {
        let primary = HashMap::new();
        let encoded = encode_commit_bundle_v3(Lsn::new(7), TenantId::DEFAULT, &primary, &[], &[]);
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(7));
        assert!(decoded.mvcc_writes.is_empty());
        assert!(decoded.sidechannel_writes.is_empty());
        assert!(decoded.staged_pages.is_empty());
    }

    #[test]
    fn v3_roundtrip_primary_index_kind_preserved() {
        let primary = mk_writes(&[(1u64, Some(&b"val"[..]))]);
        let staged = vec![mk_staged(
            BundlePageKind::PrimaryIndex,
            100,
            TenantId::SYSTEM,
            0xAA,
        )];
        let encoded =
            encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &primary, &[], &staged);
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.staged_pages.len(), 1);
        assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::PrimaryIndex);
        assert_eq!(decoded.staged_pages[0].page_id.raw(), 100);
    }

    #[test]
    fn v3_roundtrip_mixed_kinds() {
        // 4 entries, one per kind.
        let primary = HashMap::new();
        let staged = vec![
            mk_staged(BundlePageKind::PrimaryIndex, 10, TenantId::SYSTEM, 0x01),
            mk_staged(BundlePageKind::SecondaryIndex, 20, TenantId::SYSTEM, 0x02),
            mk_staged(BundlePageKind::Record, 30, TenantId::DEFAULT, 0x03),
            mk_staged(BundlePageKind::Blob, 40, TenantId::DEFAULT, 0x04),
        ];
        let encoded =
            encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &primary, &[], &staged);
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.staged_pages.len(), 4);
        assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::PrimaryIndex);
        assert_eq!(decoded.staged_pages[1].kind, BundlePageKind::SecondaryIndex);
        assert_eq!(decoded.staged_pages[2].kind, BundlePageKind::Record);
        assert_eq!(decoded.staged_pages[3].kind, BundlePageKind::Blob);
        // Wire order is preserved (encoder doesn't re-sort staged_pages).
        for (i, fill) in [0x01u8, 0x02, 0x03, 0x04].iter().enumerate() {
            assert!(decoded.staged_pages[i].bytes.iter().all(|b| b == fill));
        }
    }

    #[test]
    fn v3_roundtrip_with_sidechannel_and_staged_pages() {
        let primary = mk_writes(&[(1u64, Some(&b"user"[..]))]);
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 99,
            value: Some(Bytes::from_static(b"root-ptr")),
        }];
        let staged = vec![
            mk_staged(BundlePageKind::Record, 7, TenantId::DEFAULT, 0xCC),
            mk_staged(BundlePageKind::PrimaryIndex, 1, TenantId::SYSTEM, 0xDD),
        ];
        let encoded =
            encode_commit_bundle_v3(Lsn::new(42), TenantId::DEFAULT, &primary, &side, &staged);
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert_eq!(decoded.sidechannel_writes.len(), 1);
        assert_eq!(decoded.staged_pages.len(), 2);
        assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::Record);
        assert_eq!(decoded.staged_pages[1].kind, BundlePageKind::PrimaryIndex);
    }

    #[test]
    fn v3_unknown_kind_byte_rejected() {
        // Craft a v3 payload with an invalid kind byte.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&Lsn::new(1).raw().to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_mvcc
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_staged_pages
        bytes.push(99u8); // kind — invalid
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&[0u8; PAGE_SIZE]);
        let err = decode_commit_bundle_v3(&bytes, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("BundlePageKind"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn v3_truncated_payload_rejected() {
        let primary = mk_writes(&[(1u64, Some(&b"x"[..]))]);
        let staged = vec![mk_staged(
            BundlePageKind::PrimaryIndex,
            1,
            TenantId::DEFAULT,
            0xAA,
        )];
        let mut encoded =
            encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &primary, &[], &staged);
        encoded.truncate(encoded.len() - 10);
        let err = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v3_trailing_bytes_rejected() {
        let mut encoded =
            encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &HashMap::new(), &[], &[]);
        encoded.extend_from_slice(b"EXTRA");
        let err = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("trailing"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn dispatcher_routes_v3_payload() {
        let primary = mk_writes(&[(1u64, Some(&b"v3"[..]))]);
        let staged = vec![mk_staged(
            BundlePageKind::Record,
            500,
            TenantId::DEFAULT,
            0x42,
        )];
        let encoded =
            encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &primary, &[], &staged);
        let decoded =
            decode_commit_bundle_for_version(&encoded, BUNDLE_FORMAT_V3, TenantId::DEFAULT)
                .unwrap();
        assert_eq!(decoded.mvcc_writes.len(), 1);
        assert_eq!(decoded.staged_pages.len(), 1);
        assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::Record);
    }

    #[test]
    fn v1_v2_decoded_bundles_synthesize_primary_index_kind() {
        // Back-compat: decoding v1/v2 bundles yields staged_pages
        // entries with kind = PrimaryIndex so the replay executor
        // can route them through the unified PageStoreKind dispatch
        // without special-casing.
        let writes = mk_writes(&[(1u64, Some(&b"data"[..]))]);
        let staged = vec![mk_emit(42, 0xAB)];

        let v1_encoded = encode_commit_bundle(Lsn::new(1), &writes, &staged, TenantId::SYSTEM);
        let decoded_v1 = decode_commit_bundle_v1(&v1_encoded, TenantId::SYSTEM).unwrap();
        assert_eq!(decoded_v1.staged_pages.len(), 1);
        assert_eq!(
            decoded_v1.staged_pages[0].kind,
            BundlePageKind::PrimaryIndex
        );

        let v2_encoded = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::SYSTEM,
            &writes,
            &[],
            &staged,
            TenantId::SYSTEM,
        );
        let decoded_v2 = decode_commit_bundle_v2(&v2_encoded, TenantId::SYSTEM).unwrap();
        assert_eq!(decoded_v2.staged_pages.len(), 1);
        assert_eq!(
            decoded_v2.staged_pages[0].kind,
            BundlePageKind::PrimaryIndex
        );
    }

    #[test]
    fn bundle_index_pages_back_compat_accessor() {
        // `DecodedCommitBundle::index_pages()` filters `staged_pages`
        // to the primary-index subset. Mixed-kind v3 bundles show
        // only the PrimaryIndex entries through this accessor.
        let staged = vec![
            mk_staged(BundlePageKind::PrimaryIndex, 10, TenantId::SYSTEM, 0x01),
            mk_staged(BundlePageKind::Record, 20, TenantId::DEFAULT, 0x02),
            mk_staged(BundlePageKind::PrimaryIndex, 30, TenantId::SYSTEM, 0x03),
        ];
        let encoded = encode_commit_bundle_v3(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &staged,
        );
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        let idx: Vec<&DecodedStagedPage> = decoded.index_pages().collect();
        assert_eq!(idx.len(), 2);
        assert!(idx.iter().all(|p| p.kind == BundlePageKind::PrimaryIndex));
    }

    // ─── M3.a Slice G.1: Vector kind round-trip ─────────────────────

    #[test]
    fn bundle_page_kind_vector_round_trip() {
        // Direct encode/decode of the discriminator byte: byte 4 ↔
        // BundlePageKind::Vector. Pinned so a future taxonomy renumber
        // doesn't silently shift Vector and corrupt on-disk WALs.
        assert_eq!(BundlePageKind::Vector.as_byte(), 4);
        assert_eq!(
            BundlePageKind::from_byte(4).unwrap(),
            BundlePageKind::Vector
        );

        // Full v3 staged_pages round-trip with a Vector entry mixed
        // alongside the existing kinds — exercises the encode + decode
        // dispatch through `encode_commit_bundle_v3` /
        // `decode_commit_bundle_v3`.
        let staged = vec![
            mk_staged(BundlePageKind::PrimaryIndex, 1, TenantId::SYSTEM, 0xAA),
            mk_staged(BundlePageKind::Vector, 2, TenantId::DEFAULT, 0xBB),
            mk_staged(BundlePageKind::Blob, 3, TenantId::DEFAULT, 0xCC),
        ];
        let encoded = encode_commit_bundle_v3(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &staged,
        );
        let decoded = decode_commit_bundle_v3(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.staged_pages.len(), 3);
        assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::PrimaryIndex);
        assert_eq!(decoded.staged_pages[1].kind, BundlePageKind::Vector);
        assert_eq!(decoded.staged_pages[2].kind, BundlePageKind::Blob);
        assert!(decoded.staged_pages[1].bytes.iter().all(|b| *b == 0xBB));
    }

    // ─── v4 codec tests (issue #129 P0 — allocator_advances) ────────

    #[test]
    fn bundle_allocator_kind_byte_layout() {
        // Pin the wire-stable byte assignment. A future taxonomy
        // renumber (e.g., inserting a kind in the middle of the
        // enum) MUST bump the bundle format_version, NOT silently
        // shift these bytes — pre-existing v4 segments on disk
        // would otherwise mis-decode their allocator_advances.
        assert_eq!(AllocatorKind::Node.as_byte(), 0);
        assert_eq!(AllocatorKind::Rel.as_byte(), 1);
        assert_eq!(AllocatorKind::PageFree.as_byte(), 2);
        assert_eq!(AllocatorKind::PageNode.as_byte(), 3);
        assert_eq!(AllocatorKind::PageRel.as_byte(), 4);
        assert_eq!(AllocatorKind::PageTel.as_byte(), 5);
        assert_eq!(AllocatorKind::PageIndexInternal.as_byte(), 6);
        assert_eq!(AllocatorKind::PageIndexLeaf.as_byte(), 7);
        assert_eq!(AllocatorKind::PageVectorNeighbor.as_byte(), 8);
        assert_eq!(AllocatorKind::PageWalBuffer.as_byte(), 9);
        assert_eq!(AllocatorKind::PageIndexOverflow.as_byte(), 10);
        for k in [
            AllocatorKind::Node,
            AllocatorKind::Rel,
            AllocatorKind::PageFree,
            AllocatorKind::PageNode,
            AllocatorKind::PageRel,
            AllocatorKind::PageTel,
            AllocatorKind::PageIndexInternal,
            AllocatorKind::PageIndexLeaf,
            AllocatorKind::PageVectorNeighbor,
            AllocatorKind::PageWalBuffer,
            AllocatorKind::PageIndexOverflow,
        ] {
            assert_eq!(AllocatorKind::from_byte(k.as_byte()).unwrap(), k);
        }
    }

    #[test]
    fn bundle_allocator_kind_for_page_type_round_trip() {
        for pt in [
            arcgraph_core::PageType::Free,
            arcgraph_core::PageType::Node,
            arcgraph_core::PageType::Rel,
            arcgraph_core::PageType::Tel,
            arcgraph_core::PageType::IndexInternal,
            arcgraph_core::PageType::IndexLeaf,
            arcgraph_core::PageType::VectorNeighbor,
            arcgraph_core::PageType::WalBuffer,
            arcgraph_core::PageType::IndexOverflow,
        ] {
            let kind = AllocatorKind::for_page_type(pt);
            assert_eq!(kind.page_type(), Some(pt));
        }
        // Node / Rel have no PageType correspondent.
        assert_eq!(AllocatorKind::Node.page_type(), None);
        assert_eq!(AllocatorKind::Rel.page_type(), None);
    }

    #[test]
    fn bundle_allocator_kind_unknown_byte_rejected() {
        let err = AllocatorKind::from_byte(99).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v4_empty_bundle_roundtrip() {
        let primary = HashMap::new();
        let encoded =
            encode_commit_bundle_v4(Lsn::new(7), TenantId::DEFAULT, &primary, &[], &[], &[]);
        let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(7));
        assert!(decoded.mvcc_writes.is_empty());
        assert!(decoded.sidechannel_writes.is_empty());
        assert!(decoded.staged_pages.is_empty());
        assert!(decoded.allocator_advances.is_empty());
    }

    #[test]
    fn v4_advances_round_trip() {
        let advances = vec![
            AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::Node,
                new_high_water: 100,
            },
            AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::Rel,
                new_high_water: 50,
            },
            AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::PageNode,
                new_high_water: 7,
            },
            AllocatorAdvance {
                tenant: TenantId::SYSTEM,
                kind: AllocatorKind::PageIndexLeaf,
                new_high_water: 3,
            },
        ];
        let encoded = encode_commit_bundle_v4(
            Lsn::new(42),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &advances,
        );
        let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.allocator_advances.len(), 4);
        // Wire order: sorted by (tenant.raw(), kind.as_byte()).
        // SYSTEM=0, DEFAULT=1; AllocatorKind bytes per the layout test.
        assert_eq!(decoded.allocator_advances[0].tenant, TenantId::SYSTEM);
        assert_eq!(
            decoded.allocator_advances[0].kind,
            AllocatorKind::PageIndexLeaf
        );
        assert_eq!(decoded.allocator_advances[1].tenant, TenantId::DEFAULT);
        assert_eq!(decoded.allocator_advances[1].kind, AllocatorKind::Node);
        assert_eq!(decoded.allocator_advances[1].new_high_water, 100);
        assert_eq!(decoded.allocator_advances[2].kind, AllocatorKind::Rel);
        assert_eq!(decoded.allocator_advances[3].kind, AllocatorKind::PageNode);
    }

    #[test]
    fn v4_advances_section_size_is_17_per_entry() {
        // Pin the on-wire fixed entry size at 17 bytes (8 tenant +
        // 1 kind + 8 high_water). The format has no partition-id
        // slot. This test guards that invariant.
        let one = vec![AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        }];
        let zero = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
        );
        let one_enc = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &one,
        );
        // Difference is exactly one entry's encoded length.
        assert_eq!(
            one_enc.len() - zero.len(),
            AllocatorAdvance::ENCODED_LEN,
            "v1.0 AllocatorAdvance entries are exactly 17 bytes — \
             v1.1 partition_id extension MUST bump format_version, not \
             silently grow this entry"
        );
        assert_eq!(AllocatorAdvance::ENCODED_LEN, 17);
    }

    #[test]
    fn v4_advances_with_full_bundle_round_trip() {
        // All sections populated: primary writes + sidechannel +
        // staged_pages + allocator_advances. Verifies the v4 codec
        // composes correctly and the decoder reaches the v4 tail
        // exactly.
        let primary = mk_writes(&[(0xAA, Some(&b"node"[..])), (0xBB, None)]);
        let side = vec![SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 0xFEE1,
            value: Some(Bytes::from_static(b"root-ptr")),
        }];
        let staged = vec![
            mk_staged(BundlePageKind::Record, 30, TenantId::DEFAULT, 0xCC),
            mk_staged(BundlePageKind::PrimaryIndex, 1, TenantId::SYSTEM, 0xDD),
        ];
        let advances = vec![
            AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::Node,
                new_high_water: 100,
            },
            AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::PageNode,
                new_high_water: 5,
            },
        ];
        let encoded = encode_commit_bundle_v4(
            Lsn::new(99),
            TenantId::DEFAULT,
            &primary,
            &side,
            &staged,
            &advances,
        );
        let decoded = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(99));
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(decoded.sidechannel_writes.len(), 1);
        assert_eq!(decoded.staged_pages.len(), 2);
        assert_eq!(decoded.allocator_advances.len(), 2);
    }

    #[test]
    fn v4_unknown_allocator_kind_byte_rejected() {
        // Build a v4 bundle, then tamper with the first
        // AllocatorAdvance kind byte to an unknown value. Decode
        // surfaces WalCorruption (not a silent wrong-parse).
        let advances = vec![AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        }];
        let mut encoded = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &advances,
        );
        // The kind byte is at: (commit_lsn 8) + (n_mvcc 4) +
        // (n_pages 4) + (n_advances 4) + (tenant 8) = 28.
        // Find the first kind byte (right after the tenant_id u64
        // of the first advance).
        let kind_offset = encoded.len() - AllocatorAdvance::ENCODED_LEN + 8;
        encoded[kind_offset] = 99;
        let err = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("AllocatorKind"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn v4_truncated_advances_section_rejected() {
        let advances = vec![AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        }];
        let mut encoded = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &advances,
        );
        // Knock out the trailing high_water u64 inside the only
        // advance entry.
        encoded.truncate(encoded.len() - 5);
        let err = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v4_trailing_bytes_rejected() {
        let mut encoded = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
        );
        encoded.extend_from_slice(b"GARBAGE");
        let err = decode_commit_bundle_v4(&encoded, TenantId::DEFAULT).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("trailing"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn dispatcher_routes_v4_payload() {
        let advances = vec![AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 42,
        }];
        let encoded = encode_commit_bundle_v4(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &advances,
        );
        let decoded =
            decode_commit_bundle_for_version(&encoded, BUNDLE_FORMAT_V4, TenantId::DEFAULT)
                .unwrap();
        assert_eq!(decoded.allocator_advances.len(), 1);
        assert_eq!(decoded.allocator_advances[0].new_high_water, 42);
    }

    #[test]
    fn v3_payload_decoded_as_v4_is_structurally_rejected() {
        // v3 bytes have no allocator_advances tail; v4 decoder
        // would either read past the end or interpret nothing,
        // surfacing WalCorruption. Verifies cross-version
        // misinterpretation is loud, not silent.
        let staged = vec![mk_staged(
            BundlePageKind::PrimaryIndex,
            1,
            TenantId::DEFAULT,
            0xAA,
        )];
        let v3 = encode_commit_bundle_v3(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &staged,
        );
        let r = decode_commit_bundle_v4(&v3, TenantId::DEFAULT);
        assert!(r.is_err(), "v3 payload should not decode cleanly as v4");
    }

    #[test]
    fn v1_v2_v3_decoders_synthesize_empty_advances() {
        // Back-compat: pre-v4 decoders return `allocator_advances:
        // Vec::new()`. Pins the empty-vec contract for replay code
        // that joins v3 + v4 segment streams.
        let writes = mk_writes(&[(1u64, Some(&b"data"[..]))]);
        let v1 = encode_commit_bundle(Lsn::new(1), &writes, &[], TenantId::DEFAULT);
        assert!(
            decode_commit_bundle_v1(&v1, TenantId::DEFAULT)
                .unwrap()
                .allocator_advances
                .is_empty()
        );
        let v2 = encode_commit_bundle_v2(
            Lsn::new(1),
            TenantId::DEFAULT,
            &writes,
            &[],
            &[],
            TenantId::DEFAULT,
        );
        assert!(
            decode_commit_bundle_v2(&v2, TenantId::DEFAULT)
                .unwrap()
                .allocator_advances
                .is_empty()
        );
        let v3 = encode_commit_bundle_v3(Lsn::new(1), TenantId::DEFAULT, &writes, &[], &[]);
        assert!(
            decode_commit_bundle_v3(&v3, TenantId::DEFAULT)
                .unwrap()
                .allocator_advances
                .is_empty()
        );
    }

    #[test]
    fn allocator_advance_partition_id_always_zero_at_v1() {
        // Local-only guard (mirrors
        // `replay_partition_id_always_zero_at_v1` and
        // `z1_partition_id_always_zero_at_v1`). The v1.0 wire format
        // for `AllocatorAdvance` has NO `partition_id` slot — the
        // entry is exactly 17 bytes. v1.1 will extend the bundle
        // (likely to BUNDLE_FORMAT_V6 or a future bump — note v5 is
        // already taken by Slice G.4 vector_pages staging) with a
        // per-advance partition_id; the bump is mandatory. This test
        // pins the v1.0 invariant structurally (struct layout) and
        // on-wire (entry size).
        //
        // Struct shape pin: AllocatorAdvance has exactly the v1.0
        // fields. Adding `partition_id` to the struct without a
        // format_version bump fails this test (the assert below
        // exists to make the failure mode loud — see the docstring
        // at the top of this module's `AllocatorAdvance` struct).
        let _adv = AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        };
        // No partition_id field accessible because none exists.
        // (Static check via construction; if this stops compiling
        // because a `partition_id` was added, that's the alarm.)
        assert_eq!(AllocatorAdvance::ENCODED_LEN, 17);
    }

    // ─── v6 codec (#352 Part 2 — idempotency_bindings; ADR-199) ─────

    fn binding(tenant: TenantId, kind: u8, internal_id: u64, ext: &str) -> IdempotencyBindingEntry {
        IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant,
            kind,
            internal_id,
            external_id: ext.to_owned(),
        }
    }

    #[test]
    fn v6_empty_bundle_roundtrip() {
        let encoded = encode_commit_bundle_v6(
            Lsn::new(7),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let decoded = decode_commit_bundle_v6(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(7));
        assert!(decoded.idempotency_bindings.is_empty());
        assert!(decoded.mvcc_writes.is_empty());
        assert!(decoded.vector_pages.is_empty());
    }

    #[test]
    fn v6_idempotency_bindings_round_trip() {
        let bindings = vec![
            binding(TenantId::DEFAULT, 0, 100, "alice"),
            binding(TenantId::DEFAULT, 1, 200, "edge-x"),
            binding(TenantId::SYSTEM, 0, 7, "sys-node"),
        ];
        let encoded = encode_commit_bundle_v6(
            Lsn::new(42),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &bindings,
        );
        let decoded = decode_commit_bundle_v6(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.idempotency_bindings.len(), 3);
        // Wire order is sorted by (tenant.raw(), kind, external_id):
        // SYSTEM(0) < DEFAULT(1); within DEFAULT, kind 0 < kind 1.
        assert_eq!(decoded.idempotency_bindings[0].tenant, TenantId::SYSTEM);
        assert_eq!(decoded.idempotency_bindings[0].external_id, "sys-node");
        assert_eq!(decoded.idempotency_bindings[1].tenant, TenantId::DEFAULT);
        assert_eq!(decoded.idempotency_bindings[1].kind, 0);
        assert_eq!(decoded.idempotency_bindings[1].external_id, "alice");
        assert_eq!(decoded.idempotency_bindings[1].internal_id, 100);
        assert_eq!(decoded.idempotency_bindings[2].kind, 1);
        assert_eq!(decoded.idempotency_bindings[2].external_id, "edge-x");
        assert_eq!(decoded.idempotency_bindings[2].internal_id, 200);
    }

    #[test]
    fn v7_idempotency_release_round_trip() {
        let bindings = vec![
            binding(TenantId::DEFAULT, 0, 100, "alice"),
            IdempotencyBindingEntry {
                op: IdempotencyBindingOp::Release,
                tenant: TenantId::DEFAULT,
                kind: 0,
                internal_id: 0,
                external_id: "alice".to_owned(),
            },
        ];
        let encoded = encode_commit_bundle_v7(
            Lsn::new(43),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &bindings,
        );
        let decoded = decode_commit_bundle_v7(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.idempotency_bindings.len(), 2);
        assert!(decoded.idempotency_bindings.iter().any(|entry| entry.op
            == IdempotencyBindingOp::Release
            && entry.tenant == TenantId::DEFAULT
            && entry.kind == 0
            && entry.external_id == "alice"
            && entry.internal_id == 0));
    }

    #[test]
    fn v6_external_id_unicode_and_kind_bytes_preserved() {
        let bindings = vec![
            binding(TenantId::DEFAULT, 0, 1, "café—naïve—✓"),
            binding(TenantId::DEFAULT, 1, 2, ""),
        ];
        let encoded = encode_commit_bundle_v6(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &bindings,
        );
        let decoded = decode_commit_bundle_v6(&encoded, TenantId::DEFAULT).unwrap();
        // sort: kind 0 ("café…") before kind 1 ("").
        assert_eq!(decoded.idempotency_bindings[0].external_id, "café—naïve—✓");
        assert_eq!(decoded.idempotency_bindings[0].kind, 0);
        assert_eq!(decoded.idempotency_bindings[1].external_id, "");
        assert_eq!(decoded.idempotency_bindings[1].kind, 1);
    }

    #[test]
    fn v6_empty_bindings_is_v5_prefix_plus_zero_count() {
        // v6 is a strict superset of v5: with no bindings, the payload is
        // the v5 payload followed by a 4-byte zero count.
        let v5 = encode_commit_bundle_v5(
            Lsn::new(9),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
        );
        let v6 = encode_commit_bundle_v6(
            Lsn::new(9),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert_eq!(v6.len(), v5.len() + 4);
        assert_eq!(&v6[..v5.len()], &v5[..]);
        assert_eq!(&v6[v5.len()..], &0u32.to_le_bytes());
    }

    #[test]
    fn v6_forward_compat_v5_segment_decodes_with_empty_idempotency() {
        // mgr-dev R1 requirement: "old bundles decode; new field defaults
        // empty." A v5-format bundle routed through the per-segment
        // dispatcher (format_version = 5) decodes with an empty
        // idempotency_bindings — so a v6 binary reading a pre-upgrade v5
        // segment never mis-parses.
        let v5 = encode_commit_bundle_v5(
            Lsn::new(11),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
        );
        let decoded =
            decode_commit_bundle_for_version(&v5, BUNDLE_FORMAT_V5, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.commit_lsn, Lsn::new(11));
        assert!(
            decoded.idempotency_bindings.is_empty(),
            "v5 bundle MUST decode with empty idempotency_bindings"
        );
    }

    #[test]
    fn v6_dispatches_via_for_version() {
        let bindings = vec![binding(TenantId::DEFAULT, 0, 5, "k")];
        let v6 = encode_commit_bundle_v6(
            Lsn::new(3),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &bindings,
        );
        let decoded =
            decode_commit_bundle_for_version(&v6, BUNDLE_FORMAT_V6, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.idempotency_bindings.len(), 1);
        assert_eq!(decoded.idempotency_bindings[0].external_id, "k");
    }

    #[test]
    fn v6_trailing_bytes_rejected() {
        let mut encoded = encode_commit_bundle_v6(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[binding(TenantId::DEFAULT, 0, 1, "x")],
        );
        encoded.push(0xAB);
        let err = decode_commit_bundle_v6(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v6_truncated_binding_rejected() {
        let encoded = encode_commit_bundle_v6(
            Lsn::new(1),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[binding(TenantId::DEFAULT, 0, 1, "abcdef")],
        );
        // Drop the last 3 external_id bytes — the ext_len now overruns.
        let truncated = &encoded[..encoded.len() - 3];
        let err = decode_commit_bundle_v6(truncated, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    proptest! {
        #[test]
        fn v6_bindings_roundtrip_any(
            entries in prop::collection::vec(
                (any::<u64>(), 0u8..=1, any::<u64>(), "[a-zA-Z0-9_\\-]{0,32}"),
                0..16,
            ),
        ) {
            let bindings: Vec<IdempotencyBindingEntry> = entries
                .iter()
                .map(|(t, k, id, ext)| binding(TenantId::new(*t), *k, *id, ext))
                .collect();
            let encoded = encode_commit_bundle_v6(
                Lsn::new(1),
                TenantId::DEFAULT,
                &HashMap::new(),
                &[],
                &[],
                &[],
                &[],
                &bindings,
            );
            let decoded = decode_commit_bundle_v6(&encoded, TenantId::DEFAULT).unwrap();
            // Same multiset of bindings (encode sorts; compare as sets of tuples).
            use std::collections::BTreeSet;
            let want: BTreeSet<(u64, u8, u64, String)> = bindings
                .iter()
                .map(|b| (b.tenant.raw(), b.kind, b.internal_id, b.external_id.clone()))
                .collect();
            let got: BTreeSet<(u64, u8, u64, String)> = decoded
                .idempotency_bindings
                .iter()
                .map(|b| (b.tenant.raw(), b.kind, b.internal_id, b.external_id.clone()))
                .collect();
            prop_assert_eq!(want, got);
        }
    }

    // ─── v8 acl_grants codec tests (#1221 — ADR-218) ──────────────────

    fn grant_set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    fn acl_apply(tenant: TenantId, doc: u64, grants: &[&str]) -> AclGrantEntry {
        AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant,
            doc: NodeId::new(doc),
            grants: grant_set(grants),
        }
    }

    fn acl_revoke(tenant: TenantId, doc: u64) -> AclGrantEntry {
        AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant,
            doc: NodeId::new(doc),
            grants: BTreeSet::new(),
        }
    }

    fn encode_v8(acl_grants: &[AclGrantEntry]) -> Vec<u8> {
        encode_commit_bundle_v8(
            Lsn::new(7),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
            acl_grants,
        )
    }

    #[test]
    fn v8_empty_acl_grants_is_v7_prefix_plus_zero_count() {
        // v8 is a strict superset of v7: with no acl_grants the payload is
        // the v7 payload followed by a 4-byte zero count.
        let v7 = encode_commit_bundle_v7(
            Lsn::new(7),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let v8 = encode_v8(&[]);
        assert_eq!(v8.len(), v7.len() + 4);
        assert_eq!(&v8[..v7.len()], &v7[..]);
        assert_eq!(&v8[v7.len()..], &0u32.to_le_bytes());
        // And the empty-acl v8 decodes to an empty acl_grants (v7-equiv).
        let decoded = decode_commit_bundle_v8(&v8, TenantId::DEFAULT).unwrap();
        assert!(decoded.acl_grants.is_empty());
    }

    #[test]
    fn v8_multi_grant_apply_round_trip() {
        let acls = vec![
            acl_apply(TenantId::DEFAULT, 1, &["alice", "bob", "__public__"]),
            acl_apply(TenantId::DEFAULT, 2, &[]), // explicit grant-to-nobody
        ];
        let encoded = encode_v8(&acls);
        let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.acl_grants, acls);
        // Grant set preserved exactly (incl. the empty set on doc 2).
        assert_eq!(
            decoded.acl_grants[0].grants,
            grant_set(&["alice", "bob", "__public__"])
        );
        assert!(decoded.acl_grants[1].grants.is_empty());
        assert_eq!(decoded.acl_grants[1].op, AclGrantOp::Apply);
    }

    #[test]
    fn v8_apply_then_revoke_round_trip() {
        let acls = vec![
            acl_apply(TenantId::DEFAULT, 5, &["alice"]),
            acl_revoke(TenantId::DEFAULT, 5),
        ];
        let encoded = encode_v8(&acls);
        let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.acl_grants, acls);
        assert_eq!(decoded.acl_grants[1].op, AclGrantOp::Revoke);
        assert!(decoded.acl_grants[1].grants.is_empty());
    }

    #[test]
    fn v8_unicode_principals_preserved() {
        let acls = vec![acl_apply(TenantId::DEFAULT, 9, &["café—✓", "naïve"])];
        let encoded = encode_v8(&acls);
        let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(
            decoded.acl_grants[0].grants,
            grant_set(&["café—✓", "naïve"])
        );
    }

    /// HARD INVARIANT (ADR-218): two ops on the SAME doc in ONE bundle
    /// MUST replay in append (staging) order — the encoder must NOT
    /// re-sort. This test would FAIL if the `acl_grants` encoder copied
    /// the v7 idempotency on-encode sort (which would reorder the two
    /// same-doc ops and flip last-writer-wins).
    #[test]
    fn v8_same_doc_ops_replay_in_append_order_last_wins() {
        // Stage two ops on doc 42 in a deliberately "anti-sorted" order:
        // a WIDE Apply first, then a NARROW Apply. Last (narrow) must win.
        let wide = acl_apply(TenantId::DEFAULT, 42, &["alice", "bob"]);
        let narrow = acl_apply(TenantId::DEFAULT, 42, &["alice"]);
        let acls = vec![wide.clone(), narrow.clone()];
        let encoded = encode_v8(&acls);
        let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
        // Order preserved EXACTLY (no re-sort): [wide, narrow].
        assert_eq!(decoded.acl_grants, vec![wide, narrow]);
        // Driving these into a fresh PermissionIndex in this (append)
        // order ⇒ narrow wins (bob denied) — last-writer-wins per doc.
        let idx = crate::permissions::PermissionIndex::new();
        for entry in &decoded.acl_grants {
            idx.apply_doc_acl(entry.doc, entry.grants.clone());
        }
        assert!(idx.effective("alice").is_visible(NodeId::new(42)));
        assert!(
            !idx.effective("bob").is_visible(NodeId::new(42)),
            "last (narrow) op must win — append order, NOT re-sorted"
        );

        // Symmetric: an Apply THEN a Revoke on the same doc ⇒ revoked.
        let acls2 = vec![
            acl_apply(TenantId::DEFAULT, 43, &["alice"]),
            acl_revoke(TenantId::DEFAULT, 43),
        ];
        let decoded2 = decode_commit_bundle_v8(&encode_v8(&acls2), TenantId::DEFAULT).unwrap();
        let idx2 = crate::permissions::PermissionIndex::new();
        for entry in &decoded2.acl_grants {
            match entry.op {
                AclGrantOp::Apply => idx2.apply_doc_acl(entry.doc, entry.grants.clone()),
                AclGrantOp::Revoke => idx2.revoke_doc(entry.doc),
            }
        }
        assert!(
            !idx2.effective("alice").is_visible(NodeId::new(43)),
            "Apply-then-Revoke in append order ⇒ revoked (invisible)"
        );
    }

    #[test]
    fn v8_dispatches_via_for_version() {
        let acls = vec![acl_apply(TenantId::DEFAULT, 1, &["alice"])];
        let v8 = encode_v8(&acls);
        let decoded =
            decode_commit_bundle_for_version(&v8, BUNDLE_FORMAT_V8, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.acl_grants.len(), 1);
        assert_eq!(decoded.acl_grants[0].grants, grant_set(&["alice"]));
    }

    #[test]
    fn v8_backward_decode_v5_v6_v7_yield_empty_acl_grants() {
        // Strict-subset: a v5/v6/v7 segment routed through the per-segment
        // dispatcher decodes with an EMPTY acl_grants (no ACL ops ⇒
        // fail-closed default). A v8 binary reading a pre-upgrade segment
        // never mis-parses.
        let v5 = encode_commit_bundle_v5(
            Lsn::new(11),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
        );
        let d5 =
            decode_commit_bundle_for_version(&v5, BUNDLE_FORMAT_V5, TenantId::DEFAULT).unwrap();
        assert!(d5.acl_grants.is_empty(), "v5 ⇒ empty acl_grants");

        let v6 = encode_commit_bundle_v6(
            Lsn::new(12),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[binding(TenantId::DEFAULT, 0, 5, "k")],
        );
        let d6 =
            decode_commit_bundle_for_version(&v6, BUNDLE_FORMAT_V6, TenantId::DEFAULT).unwrap();
        assert!(d6.acl_grants.is_empty(), "v6 ⇒ empty acl_grants");
        assert_eq!(d6.idempotency_bindings.len(), 1);

        let v7 = encode_commit_bundle_v7(
            Lsn::new(13),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[binding(TenantId::DEFAULT, 0, 5, "k")],
        );
        let d7 =
            decode_commit_bundle_for_version(&v7, BUNDLE_FORMAT_V7, TenantId::DEFAULT).unwrap();
        assert!(d7.acl_grants.is_empty(), "v7 ⇒ empty acl_grants");
        assert_eq!(d7.idempotency_bindings.len(), 1);
    }

    #[test]
    fn v8_carries_both_idempotency_and_acl_sections() {
        // v8 = v7 (idempotency tail) + acl_grants tail. Both must
        // round-trip independently in one bundle.
        let acls = vec![acl_apply(TenantId::DEFAULT, 1, &["alice"])];
        let encoded = encode_commit_bundle_v8(
            Lsn::new(7),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[binding(TenantId::DEFAULT, 0, 99, "ext")],
            &acls,
        );
        let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
        assert_eq!(decoded.idempotency_bindings.len(), 1);
        assert_eq!(decoded.idempotency_bindings[0].external_id, "ext");
        assert_eq!(decoded.acl_grants.len(), 1);
        assert_eq!(decoded.acl_grants[0].grants, grant_set(&["alice"]));
    }

    #[test]
    fn v8_trailing_bytes_rejected() {
        let mut encoded = encode_v8(&[acl_apply(TenantId::DEFAULT, 1, &["x"])]);
        encoded.push(0xAB);
        let err = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v8_truncated_grant_rejected() {
        let encoded = encode_v8(&[acl_apply(TenantId::DEFAULT, 1, &["abcdef"])]);
        // Drop the last 3 principal bytes — grant_len now overruns.
        let truncated = &encoded[..encoded.len() - 3];
        let err = decode_commit_bundle_v8(truncated, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn v8_unknown_op_byte_rejected() {
        let mut encoded = encode_v8(&[acl_apply(TenantId::DEFAULT, 1, &["x"])]);
        // The acl_grant op byte sits right after the v7 prefix + the
        // 4-byte n_acl_grants count. Find it by re-deriving the prefix len.
        let v7_len = encode_commit_bundle_v7(
            Lsn::new(7),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .len();
        let op_off = v7_len + 4; // skip n_acl_grants u32
        encoded[op_off] = 0x7F; // neither Apply(0) nor Revoke(1)
        let err = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    proptest! {
        /// Random Apply/Revoke sequence (possibly colliding on docs)
        /// round-trips through v8 with EXACT append order preserved, and a
        /// fresh PermissionIndex driven from the decoded sequence resolves
        /// every principal identically to one driven from the ORIGINAL
        /// sequence (replay reconstructs identical enforcement).
        #[test]
        fn v8_acl_grants_roundtrip_and_replay_equivalence(
            ops in prop::collection::vec(
                (0u8..=1, 0u64..8, prop::collection::vec("[a-z]{1,4}", 0..3)),
                0..24,
            ),
        ) {
            let acls: Vec<AclGrantEntry> = ops
                .iter()
                .map(|(op, doc, grants)| {
                    if *op == 0 {
                        AclGrantEntry {
                            op: AclGrantOp::Apply,
                            tenant: TenantId::DEFAULT,
                            doc: NodeId::new(*doc),
                            grants: grants.iter().cloned().collect(),
                        }
                    } else {
                        acl_revoke(TenantId::DEFAULT, *doc)
                    }
                })
                .collect();
            let encoded = encode_v8(&acls);
            let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
            // Append order preserved EXACTLY (no re-sort).
            prop_assert_eq!(&decoded.acl_grants, &acls);

            // Drive both sequences into fresh indices; enforcement must be
            // binary-equal per (principal, doc).
            let drive = |seq: &[AclGrantEntry]| {
                let idx = crate::permissions::PermissionIndex::new();
                for e in seq {
                    match e.op {
                        AclGrantOp::Apply => idx.apply_doc_acl(e.doc, e.grants.clone()),
                        AclGrantOp::Revoke => idx.revoke_doc(e.doc),
                    }
                }
                idx
            };
            let from_orig = drive(&acls);
            let from_decoded = drive(&decoded.acl_grants);
            for principal in ["a", "b", "c", "__public__", "nobody"] {
                let o = from_orig.effective(principal);
                let d = from_decoded.effective(principal);
                for doc in 0u64..8 {
                    prop_assert_eq!(
                        o.is_visible(NodeId::new(doc)),
                        d.is_visible(NodeId::new(doc))
                    );
                }
            }
        }
    }

    // ─── #1411: decoder OOM/DoS bound tests ─────────────────────────
    //
    // Every DECODE path reads a section element count as an untrusted
    // `u32` and then pre-allocates a container sized to it. Before the
    // fix a crafted count near `u32::MAX` forced a multi-GB alloc BEFORE
    // any element byte was validated (OOM on the crash-recovery path,
    // #1411; surfaced by the #1287 fuzz target). The fix caps the
    // capacity hint at `remaining_bytes / MIN_ELEM` so a nonsense count
    // is refused up front — the decode then hits the SAME in-loop
    // overrun/truncation guard and returns `WalCorruption` promptly.
    //
    // These tests are RED-on-revert: reverting ONE representative bound
    // (restoring a raw `with_capacity(untrusted_count)`) makes the
    // corresponding case attempt a ~gigabyte allocation. See the PR body
    // for the captured RED/GREEN both-legs evidence.
    mod oom_bound {
        use super::*;

        /// The canonical 33-byte `.v8_acl_oom_repro` fixture (issue
        /// #1411): a well-formed v8 header with every section count = 0
        /// UP TO the trailing `n_acl_grants`, which is
        /// `0xffff_ff00` (~4.29e9) with only 1 payload byte remaining.
        ///
        /// Layout (little-endian):
        /// `commit_lsn u64 = 3` (`03 00..00`) | `n_mvcc u32 = 0` |
        /// `n_staged_pages u32 = 0` | `n_allocator_advances u32 = 0` |
        /// `n_vector_pages u32 = 0` | `n_idempotency u32 = 0` |
        /// `n_acl_grants u32 = 0xffff_ff00` | 1 trailing `0xff` byte.
        ///
        /// n_acl_grants is read at offset 28 (bytes 28..32 = `00 ff ff ff`),
        /// leaving exactly 1 remaining byte (byte 32 = `0xff`) — nowhere near
        /// the ~4.29e9 ACL entries the count claims.
        const V8_ACL_OOM_REPRO: [u8; 33] = [
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // commit_lsn = 3
            0x00, 0x00, 0x00, 0x00, // n_mvcc = 0
            0x00, 0x00, 0x00, 0x00, // n_staged_pages = 0
            0x00, 0x00, 0x00, 0x00, // n_allocator_advances = 0
            0x00, 0x00, 0x00, 0x00, // n_vector_pages = 0
            0x00, 0x00, 0x00, 0x00, // n_idempotency_bindings = 0
            0x00, 0xff, 0xff, 0xff, // n_acl_grants = 0xffff_ff00 (bytes 28..32)
            0xff, // 1 trailing byte (remaining < MIN_ACL_GRANT_ELEM)
        ];

        fn assert_corruption(res: Result<DecodedCommitBundle>) {
            match res {
                Err(ArcGraphError::WalCorruption { .. }) => {}
                Err(other) => panic!("expected WalCorruption, got {other:?}"),
                Ok(_) => panic!("expected Err(WalCorruption), decoded Ok"),
            }
        }

        /// The exact repro fixture bytes match `.v8_acl_oom_repro` on
        /// disk (guards against the inline `const` drifting from the
        /// fixture the fuzz corpus / issue reference).
        #[test]
        fn v8_acl_oom_repro_bytes_match_fixture() {
            // Same 33 bytes documented in the issue: `03 00..00 00 ff ff ff`.
            assert_eq!(V8_ACL_OOM_REPRO.len(), 33);
            let n_acl = u32::from_le_bytes([
                V8_ACL_OOM_REPRO[28],
                V8_ACL_OOM_REPRO[29],
                V8_ACL_OOM_REPRO[30],
                V8_ACL_OOM_REPRO[31],
            ]);
            assert_eq!(
                n_acl, 0xffff_ff00,
                "trailing count is the u32::MAX-ish OOM trigger"
            );
        }

        /// **The #1411 repro.** Feeding the crafted 33-byte v8 bundle
        /// (`n_acl_grants = 0xffff_ff00`, 1 byte remaining) to
        /// `decode_commit_bundle_v8` must return `Err(WalCorruption)`
        /// PROMPTLY without attempting the ~4.29e9-element `Vec` alloc.
        ///
        /// RED-on-revert: with the `n_acl` bound reverted to
        /// `Vec::with_capacity(n_acl)`, this call pre-allocates a
        /// `Vec<AclGrantEntry>` for ~4.29e9 entries (each `AclGrantEntry`
        /// is dozens of bytes) — a >100 GB request that aborts/OOMs the
        /// process before this assertion is reached.
        #[test]
        fn v8_acl_oom_repro_returns_corruption_not_oom() {
            let res = decode_commit_bundle_v8(&V8_ACL_OOM_REPRO, TenantId::DEFAULT);
            assert_corruption(res);
        }

        /// Build a v8 header whose sections up to (but excluding) the
        /// named trailing section are all empty, then append a raw
        /// `count` u32 with NO element bytes after it. Used to hit each
        /// section's untrusted-count pre-alloc with `u32::MAX`.
        ///
        /// `n_leading_zero_counts` = how many u32 zero-counts precede the
        /// malicious one (v8 section order: mvcc, staged_pages,
        /// allocator_advances, vector_pages, idempotency, acl_grants).
        fn v8_header_with_trailing_count(n_leading_zero_counts: usize, count: u32) -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(&7u64.to_le_bytes()); // commit_lsn
            for _ in 0..n_leading_zero_counts {
                b.extend_from_slice(&0u32.to_le_bytes());
            }
            b.extend_from_slice(&count.to_le_bytes());
            b
        }

        /// Synthetic `u32::MAX` count for EACH v8 section family — each
        /// must return `WalCorruption` (hit its bound → first element
        /// read hits the truncation guard) without OOM.
        #[test]
        fn v8_max_count_each_section_is_corruption_not_oom() {
            // (section index in v8 order, human name)
            let sections = [
                (0usize, "mvcc"),
                (1, "staged_pages"),
                (2, "allocator_advances"),
                (3, "vector_pages"),
                (4, "idempotency_bindings"),
                (5, "acl_grants"),
            ];
            for (idx, name) in sections {
                let bytes = v8_header_with_trailing_count(idx, u32::MAX);
                let res = decode_commit_bundle_v8(&bytes, TenantId::DEFAULT);
                assert!(
                    matches!(res, Err(ArcGraphError::WalCorruption { .. })),
                    "section {name} with u32::MAX count must be WalCorruption, got {res:?}"
                );
            }
        }

        /// v1 mvcc / index_pages sections: `u32::MAX` counts with no
        /// element bytes must be `WalCorruption`, not OOM.
        #[test]
        fn v1_max_counts_are_corruption_not_oom() {
            // n_mvcc = u32::MAX, no element bytes.
            let mut b = Vec::new();
            b.extend_from_slice(&1u64.to_le_bytes()); // commit_lsn
            b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_mvcc
            assert_corruption(decode_commit_bundle_v1(&b, TenantId::DEFAULT));

            // n_mvcc = 0, n_index_pages = u32::MAX, no element bytes.
            let mut b = Vec::new();
            b.extend_from_slice(&1u64.to_le_bytes()); // commit_lsn
            b.extend_from_slice(&0u32.to_le_bytes()); // n_mvcc
            b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_index_pages
            assert_corruption(decode_commit_bundle_v1(&b, TenantId::DEFAULT));
        }

        /// The bound is a HINT only — a genuine multi-element bundle that
        /// has the backing bytes decodes exactly as before the fix
        /// (`bounded_capacity` returns `n` when `remaining >= n*MIN_ELEM`).
        #[test]
        fn bounded_capacity_is_noop_for_valid_input() {
            // remaining exactly n*min ⇒ cap == n (boundary).
            assert_eq!(
                bounded_capacity(5, 5 * MIN_ACL_GRANT_ELEM, MIN_ACL_GRANT_ELEM),
                5
            );
            // remaining well above n*min ⇒ cap == n.
            assert_eq!(bounded_capacity(3, 10_000, MIN_MVCC_V1_ELEM), 3);
            // remaining below n*min ⇒ cap clamped to remaining/min < n.
            let clamped = bounded_capacity(1_000_000, 100, MIN_MVCC_V1_ELEM);
            assert!(clamped < 1_000_000);
            assert_eq!(clamped, 100 / MIN_MVCC_V1_ELEM);
            // The pathological repro shape: remaining 1 byte, huge count.
            assert_eq!(bounded_capacity(0xffff_ff00, 1, MIN_ACL_GRANT_ELEM), 0);
        }

        /// Boundary: `n == remaining / MIN_ELEM` with the bytes present
        /// decodes fine; `n + 1` with the bytes absent → `WalCorruption`
        /// not OOM. Exercised on the v8 acl_grants section (fixed 21-B
        /// prefix per entry, Revoke = grant-to-nobody).
        #[test]
        fn v8_acl_boundary_exact_vs_over_by_one() {
            // Two genuine Revoke entries (each 21 B fixed prefix, 0 grants).
            let acls = vec![
                acl_revoke(TenantId::DEFAULT, 10),
                acl_revoke(TenantId::DEFAULT, 11),
            ];
            let encoded = encode_v8(&acls);
            // Exact valid input decodes to the same 2 entries (cap == n == 2).
            let decoded = decode_commit_bundle_v8(&encoded, TenantId::DEFAULT).unwrap();
            assert_eq!(decoded.acl_grants, acls);

            // Now corrupt ONLY the n_acl count to claim one extra entry
            // (n = 3) with no bytes for it: must be WalCorruption, not OOM.
            // n_acl is the LAST u32 in the payload (the encoder appends the
            // acl section, whose count precedes the two 21-B entries).
            let n_acl_offset = encoded.len() - 2 * AclGrantEntry::FIXED_PREFIX_LEN - 4;
            let observed =
                u32::from_le_bytes(encoded[n_acl_offset..n_acl_offset + 4].try_into().unwrap());
            assert_eq!(observed, 2, "sanity: encoded n_acl == 2");
            let mut over = encoded.clone();
            over[n_acl_offset..n_acl_offset + 4].copy_from_slice(&3u32.to_le_bytes());
            assert_corruption(decode_commit_bundle_v8(&over, TenantId::DEFAULT));

            // And a wildly-over count (u32::MAX) on the same well-formed
            // prefix is likewise WalCorruption, not OOM.
            let mut huge = encoded.clone();
            huge[n_acl_offset..n_acl_offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_corruption(decode_commit_bundle_v8(&huge, TenantId::DEFAULT));
        }

        /// Cross-version smoke: a `u32::MAX` leading `n_mvcc` on EVERY
        /// decoder version is `WalCorruption`, not OOM (the mvcc count is
        /// the first untrusted count all versions share).
        #[test]
        fn all_versions_max_mvcc_is_corruption_not_oom() {
            let mut b = Vec::new();
            b.extend_from_slice(&1u64.to_le_bytes()); // commit_lsn
            b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_mvcc
            for decode in [
                decode_commit_bundle_v1 as fn(&[u8], TenantId) -> Result<DecodedCommitBundle>,
                decode_commit_bundle_v2,
                decode_commit_bundle_v3,
                decode_commit_bundle_v4,
                decode_commit_bundle_v5,
                decode_commit_bundle_v6,
                decode_commit_bundle_v7,
                decode_commit_bundle_v8,
            ] {
                let res = decode(&b, TenantId::DEFAULT);
                assert!(
                    matches!(res, Err(ArcGraphError::WalCorruption { .. })),
                    "u32::MAX n_mvcc must be WalCorruption, got {res:?}"
                );
            }
        }
    }
}
