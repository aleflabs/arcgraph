//! #352 Part 2 — durable `external_id → internal_id` idempotency store.
//!
//! `graph.ingest` (W17α PR #349) gives each ingested record an
//! at-least-once identity: when a record carries `external_id =
//! Some(s)`, a re-submission of `(tenant, s)` must resolve to the SAME
//! internal id. The binding `(TenantId, kind, external_id) →
//! internal_id` used to live in a process-side in-memory map in
//! `arcgraph-mcp`, capped per-tenant and **lost on every restart**
//! (issue #352). This module makes the binding **durable**: it is the
//! storage-resident, replay-rebuilt lookup table, and the durable
//! source of truth is the WAL.
//!
//! # Where the durability comes from (ADR-199, v6 CommitBundle fold)
//!
//! Unlike the [`crate::intern::InternTable`] (whose bindings ride a
//! standalone [`crate::wal::WalRecordType::InternString`] record), the
//! idempotency binding rides **inside the owning commit's
//! `CommitBundle`** as a new `idempotency_bindings` section (bundle
//! format v6). This is deliberate and load-bearing:
//!
//! - A standalone pre-commit record would be made durable by
//!   `WalHandle::append`'s **synchronous** fsync *before* the owning
//!   `crud::commit`. A crash in that window leaves a durable binding for
//!   a node whose commit (and whose allocator high-water, #129/#820) is
//!   absent → the `internal_id` is re-allocated to a *different* record
//!   on the next live `create_node` → cross-wiring (a #820-class durable
//!   inconsistency, strictly worse than the duplicate-mint #352 fixes).
//! - For interning an orphan `name → id` is benign (0-row queries); for
//!   idempotency it is not. So the binding MUST be atomic with the
//!   commit. Folding it into the `CommitBundle` makes it present **iff**
//!   the node is present — exactly the `allocator_advances` (v4)
//!   precedent for "per-commit metadata that must be atomic with the
//!   MVCC writes." See ADR-199 §Revision 2026-06-07.
//!
//! The store itself is therefore a pure `(key → existing-id)` lookup
//! with no allocator to seed — internal ids come from the node/rel
//! allocators already seeded by MVCC replay (#820/#824).
//!
//! # Tenancy + semantics-agnosticism
//!
//! Every lookup is keyed by `(TenantId, kind, external_id)`. Two
//! tenants may use the same `external_id` and bind to **different**
//! internal ids; cross-tenant isolation is a hard invariant. The
//! `kind` byte is an **opaque** discriminator: `arcgraph-storage`
//! attaches no meaning to it. `arcgraph-mcp` owns the semantic
//! `IdempotencyKind` (node vs rel) and maps it to a `u8` at the
//! published boundary, keeping the two namespaces disjoint inside one
//! store (a re-submitted node never resolves to a rel's id, or
//! vice-versa) without leaking node/rel semantics into storage.
//!
//! # Budget
//!
//! Lookup (`get`): one DashMap read on a `(TenantId, u8, String)` key.
//! Constructing the key allocates the `String` once (~50 ns) — DashMap
//! keys are owned, not `Borrow`-polymorphic, identical to `InternTable`.
//! At a ~5 K TPS ingest budget this is far inside the envelope. No
//! global lock — DashMap shards serialize per-bucket; distinct
//! external_ids proceed in parallel.
//!
//! # Residency (ADR-199 open question; v1.1 decision)
//!
//! v1.1 ships the in-memory DashMap rebuilt on replay (this module):
//! removes the silent-loss + restart-loss, bounded memory ≈ O(distinct
//! external_ids/tenant).
//!
//! # #1404 M0.x — bounded resident tier + durable QUERYABLE spill (RE-2)
//!
//! The pure in-RAM DashMap above is O(N-distinct-external-ids). At the
//! #1404 acceptance (measured ~1 binding/node, ~9M @9M, both node AND rel
//! side) the `forward`/`reverse` maps grow with ingested count and, worse,
//! `IdempotencyStore::iter_all` (the ADR-229 checkpoint capture,
//! `snapshot.rs:808`) materializes the WHOLE binding set — owned `String`s
//! and all — into a `Vec` UNDER `checkpoint_freeze` (`producer.rs:132`).
//! That is the RE-2 **freeze-capture** term that OOM'd the
//! 10M-nodes+20M-rels acceptance at `producer.rs:139` (burst
//! 9.54 → 17.30 → 30.52 GB vs a 40 GB cap).
//!
//! The M0.x fix bounds the RESIDENT binding set to a byte watermark and
//! **spills the rest to a durable, QUERYABLE-by-key store**
//! ([`IdempotencySpill`]) — mirroring the #1404 M0 [`crate::blob::BlobStore`]
//! bounded-tier + [`crate::blob::BlobSpill`] shape, but the spill is keyed by
//! the **binding key `(TenantId, u8, String)`** (not a page id), because
//! bindings are **lookup-load-bearing**: an at-least-once ingest identity.
//! Dropping ONE binding = a **duplicate** on re-ingest of that external_id.
//! So bounding here is **SPILL-to-durable-queryable, NEVER
//! evict-to-nowhere** (fable RE-2). [`IdempotencyStore::get`] faults the
//! binding back in from the spill on a resident miss → correctness is
//! IDENTICAL, latency-tiered.
//!
//! **INV-DURABLE** (the data-loss guard, mirrors the blob tier): a binding
//! is spill-eligible ONLY after a completed checkpoint has captured its
//! durable image (`iter_all` sets the `checkpointed` gate under the freeze).
//! Its bytes are then durable ≤ `checkpoint_lsn` in the ADR-229 snapshot AND
//! in the spill file; evicting-before-durable (which would lose identity on
//! crash) cannot happen. The bindings are ALSO WAL-logged in the owning
//! `CommitBundle` (module docs above), so the spill file is process-local
//! scratch — truncated on open, rebuilt from WAL + checkpoint on restart
//! (like [`crate::blob::BlobSpill`]).
//!
//! Without a spill attached ([`IdempotencyStore::new`], the legacy default +
//! the in-memory / no-`--data` path) nothing is ever evicted and behavior is
//! byte-identical to the pre-#1404 store.
//!
//! True unbounded structural scale (SF-1000 / web-scale) is an on-disk
//! page-store-backed B-tree index — the M4/M6 record-native step (ADR-230
//! OQ-G); M0.x is the rc-track interim that bounds the freeze-capture RSS.

use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arcgraph_core::TenantId;
use dashmap::DashMap;

use crate::owner_index::str_hash_56;
use crate::owner_row::{BindingOwnerValue, OwnerRowClass, OwnerRowError, OwnerRowRegistry};

/// Value stored for one idempotency binding.
///
/// `payload_hash = None` means the binding came from the durable v6
/// `CommitBundle` replay path, whose wire format carries only
/// `external_id -> internal_id`. Live-process ingest installs use
/// `Some(hash)` so the MCP adapter can distinguish a true retry from a
/// same-external_id/different-payload conflict without changing WAL
/// format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdempotencyBinding {
    /// Previously committed node/relationship id.
    pub internal_id: u64,
    /// Hash of the original ingest payload when known in this process.
    pub payload_hash: Option<u64>,
}

// ─────────────────────────────────────────────────────────────────────
// #1404 M0.x — bounded resident binding tier (bounded freeze-capture RSS)
// ─────────────────────────────────────────────────────────────────────
//
// BACK-OF-ENVELOPE (PD#5). #1404 measured ~1 binding/node with external-id
// ingest (2.08M @2M, ~9M @9M, both node AND rel side). Each resident forward
// entry is `(TenantId=8, u8, String) → IdempotencyBinding{u64, Option<u64>}`;
// with a ~32 B external_id that is ~64 B forward + ~48 B reverse + DashMap
// bucket overhead ≈ ~150 B/binding resident. At 9M that is ~1.3 GB resident —
// small vs the blob term ALONE, BUT `iter_all()` under `checkpoint_freeze`
// (`producer.rs:132`) materializes ALL of it into a fresh `Vec` (owned
// Strings) — a transient ~2× spike ON TOP of the resident set, inside the
// freeze, which is the RE-2 freeze-capture burst term (9.54→30.52 GB). This
// tier bounds the RESIDENT binding set to a byte watermark so the resident
// working set (and the `iter_all` capture that walks it) is a function of the
// watermark, NOT of ingested binding count. A 256 MiB high-watermark holds
// ~1.7M resident bindings; everything above spills to `idempotency-spill.db`
// and re-faults on lookup.
//
// INV-DURABLE (identity-loss guard): a binding is evict-eligible ONLY after a
// completed checkpoint has captured its durable image (the `checkpointed` bit
// is set by `iter_all` under the checkpoint freeze). So an evicted binding is
// always durable in BOTH the ADR-229 snapshot AND the spill file — spilling
// (moving to a durable, queryable tier) NEVER drops identity. This is the
// hard difference from an LRU/evict-to-nowhere: a re-ingest of a spilled
// external_id STILL de-dupes because `get()` faults it back in.
//
// INV-DRAIN is not applicable to bindings the way it is to MVCC versions:
// a binding is a single content-addressed `(key) → id` fact, not a
// snapshot-versioned chain. A spilled binding faults back identically for
// every reader (no old-vs-new-image hazard) — same as the blob single-image
// argument (`blob.rs` INV-DRAIN comment).

/// Default share of the memory cap the resident binding tier may hold before
/// eviction engages, mirroring the #1404 M0 blob tier + #1405 drain design §3
/// high watermark (`0.5 × cap`).
pub const DEFAULT_IDEMPOTENCY_HIGH_WATERMARK_FRACTION: f64 = 0.5;

/// Default low-watermark share (`0.375 × cap`) — eviction drains down to this
/// before disengaging (hysteresis), mirroring the blob tier.
pub const DEFAULT_IDEMPOTENCY_LOW_WATERMARK_FRACTION: f64 = 0.375;

/// Fixed per-binding accounting weight in bytes (both maps + overhead). The
/// resident-byte counter is `bindings × IDEMPOTENCY_BINDING_WEIGHT_BYTES`, so
/// the watermark trigger is a load, not a heap-size scan. Sized from the
/// back-of-envelope above (~150 B) rounded to a stable constant.
pub const IDEMPOTENCY_BINDING_WEIGHT_BYTES: u64 = 160;

/// #1404 M0.x round-3 FIX-4 — the evict FIFO's ENTRY cap, as a multiple of
/// the resident-binding cap the high watermark implies. `release()` removes a
/// binding from both tiers but NOT from the FIFO (a queue sweep would be
/// O(queue)); its stale entry is reclaimed as a cheap `Gone` pop by the next
/// drain pass. Pre-fix that reclaim ran ONLY under resident-BYTE pressure, so
/// an ingest-then-delete workload (TTL expiry / re-sync / ephemeral incident
/// entities — live set under the watermark) never drained and the FIFO leaked
/// one `(TenantId, u8, String)` entry per released binding (round-3 skeptic 4:
/// 50,000 stale entries for a 0-binding store; ~2-4 GB at 40M-rel churn — the
/// OOM class #1404 exists to kill). With this cap, `maybe_drain` also engages
/// when the FIFO exceeds `factor × (high_watermark_bytes /
/// IDEMPOTENCY_BINDING_WEIGHT_BYTES)` entries, bounding queue memory to the
/// same order as the resident tier it mirrors. Factor 2: one live entry per
/// resident binding + an equal allowance of Gone/duplicate backlog before a
/// reclaim pass engages.
pub const IDEMPOTENCY_EVICT_QUEUE_CAP_FACTOR: u64 = 2;

/// Operator knobs for the bounded resident binding tier (#1404 M0.x). Byte
/// caps on the resident binding set. `high` engages eviction; `low` is the
/// drain target (hysteresis). Config-strict under the code-quality policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdempotencyBoundConfig {
    /// Resident binding bytes above which eviction engages.
    pub high_watermark_bytes: u64,
    /// Resident binding bytes the drain targets before disengaging (must be
    /// `< high_watermark_bytes` for meaningful hysteresis).
    pub low_watermark_bytes: u64,
}

impl Default for IdempotencyBoundConfig {
    /// The unbounded default: watermarks at `u64::MAX` so eviction never
    /// engages. Only meaningful when a spill file is attached; a store built
    /// via [`IdempotencyStore::new`] carries this + `spill = None` and is the
    /// legacy pure-in-RAM store.
    fn default() -> Self {
        Self {
            high_watermark_bytes: u64::MAX,
            low_watermark_bytes: u64::MAX,
        }
    }
}

impl IdempotencyBoundConfig {
    /// Environment variable naming the bounded binding tier's resident cap in
    /// BYTES. When set on the durable serve path, the store engages the
    /// bounded tier with `high = 0.5 × cap`, `low = 0.375 × cap`.
    pub const ENV_RESIDENT_CAP_BYTES: &'static str = "ARCGRAPH_IDEMPOTENCY_RESIDENT_CAP_BYTES";

    /// Default resident binding cap when the env var is unset (256 MiB).
    /// Sized so the bounded tier's steady-state RSS contribution + its
    /// `iter_all` capture spike are a fixed budget (~1.7M resident bindings
    /// at the `0.5 ×` high watermark), independent of ingested binding count —
    /// the #1404 M0.x fix.
    pub const DEFAULT_RESIDENT_CAP_BYTES: u64 = 256 * 1024 * 1024;

    /// Derive watermarks from a memory cap using the design defaults
    /// (`0.5 × cap` high, `0.375 × cap` low).
    #[must_use]
    pub fn from_cap_bytes(cap_bytes: u64) -> Self {
        let high = (cap_bytes as f64 * DEFAULT_IDEMPOTENCY_HIGH_WATERMARK_FRACTION) as u64;
        let low = (cap_bytes as f64 * DEFAULT_IDEMPOTENCY_LOW_WATERMARK_FRACTION) as u64;
        Self {
            high_watermark_bytes: high.max(IDEMPOTENCY_BINDING_WEIGHT_BYTES),
            low_watermark_bytes: low
                .max(IDEMPOTENCY_BINDING_WEIGHT_BYTES)
                .min(high.saturating_sub(1).max(1)),
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
            .filter(|&c| c >= IDEMPOTENCY_BINDING_WEIGHT_BYTES * 2)
            .unwrap_or(Self::DEFAULT_RESIDENT_CAP_BYTES);
        Self::from_cap_bytes(cap)
    }
}

/// One spilled binding image as it lives on disk: enough to reconstruct the
/// forward AND reverse entries on re-fault + on `iter_all` capture. The
/// external_id and internal_id are both present so a re-fault restores the
/// full `(forward, reverse)` pair, and `iter_all` can enumerate the spilled
/// set completely for the ADR-229 snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpilledBinding {
    external_id: String,
    internal_id: u64,
    payload_hash: Option<u64>,
}

/// Append-only durable spill file for evicted idempotency bindings
/// (#1404 M0.x), QUERYABLE by the binding key `(TenantId, u8, String)` —
/// indexed by a **bounded-RAM on-disk hash-chain index** (#1404 M0.x round-2
/// FIX-3).
///
/// Mirrors [`crate::blob::BlobSpill`] (a durable file + a re-fault on read),
/// but the key is the BINDING key, not a page id — the "queryable-by-key"
/// requirement (fable RE-2): a lookup must resolve a spilled `external_id`
/// back to its `internal_id`, not just a page image.
///
/// # FIX-3 — why the index is on disk (round-2 REJECT, skeptic 6)
///
/// The pre-fix index was a pair of in-RAM `DashMap`s (`(tenant,kind,ext)→off`
/// + `(tenant,kind,id)→off`), written on every spill and never trimmed: the
///   spill BOUNDED the resident tier but RELOCATED the O(N-bindings) term into
///   the spill's own index (measured 1:1 growth — 20K bindings → ~19,950
///   entries; 200K → ~199,950; ≈4-8+ GB undrainable at 20-40M rels — the same
///   OOM class the spill was introduced to kill). The census-class resident
///   owner, OOM-guardrail Rule (e).
///
/// The index is now a classic external separate-chaining hash table (Knuth,
/// TAOCP vol. 3 §6.4; the dbm/ndbm bucket-chain lineage) over TWO
/// process-local scratch files:
///
/// - **Record file** (`idempotency-spill.db`) — append-only binding images,
///   self-describing (tenant + kind + both ids in the record), so a chain
///   candidate is verified against the FULL key on disk — a 64-bit hash
///   match alone is never trusted.
/// - **Index file** (`idempotency-spill.idx`) — fixed 32-byte chain nodes
///   `(key_hash, record_off, prev, flags)`. Each bucket is a singly-linked
///   NEWEST-FIRST chain threaded through `prev`. A remove appends a
///   TOMBSTONE node: the first full-key match in a chain is the
///   authoritative newest verdict (live image or dead key).
///
/// **In-RAM footprint: O(buckets), NOT O(N)** — two `Vec<u64>` bucket-head
/// arrays (2 × [`SPILL_INDEX_DEFAULT_BUCKETS`] × 8 B = 32 MiB at the
/// production default) + O(1) counters, flat as N grows. Everything else
/// lives in the two scratch files.
///
/// # Budget (B = buckets, N = live spilled bindings; per-op, page-cached)
///
/// - Lookup / contains / guarded-retire: one chain walk = O(N/B) 32-byte
///   node reads (~19 at N=40M, B=2^21 ≈ tens of µs), + ONE record read per
///   full-hash candidate (a false candidate needs a 64-bit collision inside
///   one bucket — negligible, and it is then rejected by the full-key
///   compare, never silently believed).
/// - Insert (evict): record append + one chain walk (exact `live_forward`
///   accounting under overwrites) + two 32-byte node appends.
/// - Enumeration (capture walk) / per-tenant count: O(total nodes)
///   node+record reads — the SAME O(N)-disk class as the pre-fix
///   `offsets.iter()` + per-entry `read_binding_at`, unchanged.
/// - `binding_count`'s resident-side dedup filter: one chain walk per
///   RESIDENT binding (watermark-bounded), under the capture write guard.
///
/// # Concurrency
///
/// ALL index reads/mutations serialize under the `index` mutex; the record
/// file keeps its own mutex; lock order is strictly `index → file`, never
/// reversed. Every compound op (find-then-tombstone, insert-with-liveness-
/// check) is therefore at least as atomic as the pre-fix per-entry DashMap
/// ops, so the round-2 FIX-1 offset-guarded retire semantics and the FIX-2
/// capture-exclusion framing carry over unchanged.
///
/// Recovery: process-local scratch — truncated on open, discarded on restart.
/// Bindings recover from WAL + checkpoint (module docs; `idempotency.rs:62`),
/// so neither file needs to survive a crash.
#[derive(Debug)]
pub struct IdempotencySpill {
    /// The append-only record file. `Mutex` serializes appends + seeks.
    file: Mutex<File>,
    /// FIX-3 — the bounded-RAM index state (chain-node file + the bucket-head
    /// arrays, the ONLY in-RAM index residue). See the type docs.
    index: Mutex<SpillIndex>,
    /// Bucket count, fixed at open. [`Self::open_with_buckets`] shrinks it in
    /// tests to force long chains (collision/tombstone paths) at small N.
    buckets: u64,
    /// Per-instance SipHash seed for bucket hashing. The files are scratch,
    /// so the seed only needs to be stable for the process lifetime.
    hasher: RandomState,
    /// Next record-file append offset (also the current file length).
    write_offset: AtomicU64,
    /// Path (for diagnostics only).
    path: PathBuf,
}

/// FIX-3 — mutable spill-index state: the fixed-size-node chain file plus the
/// O(buckets) in-RAM bucket heads and O(1) counters. Guarded by
/// [`IdempotencySpill::index`]; every field access happens under that lock.
#[derive(Debug)]
struct SpillIndex {
    /// The chain-node file (`idempotency-spill.idx`, scratch).
    file: File,
    /// Next node append offset.
    write_off: u64,
    /// Forward bucket heads: newest node offset per bucket; `IDX_NIL` = empty.
    heads_fwd: Vec<u64>,
    /// Reverse bucket heads (the `(tenant, kind, internal_id)` direction).
    heads_rev: Vec<u64>,
    /// Exact count of DISTINCT LIVE forward keys, maintained on every insert /
    /// tombstone under the index lock — replaces the pre-fix `offsets.len()`.
    live_forward: u64,
}

/// One fixed-size chain node in the index file (32 B, little-endian):
/// `0..8` key_hash · `8..16` record_off · `16..24` prev · `24` flags · pad.
#[derive(Debug, Clone, Copy)]
struct IdxNode {
    /// Full 64-bit key hash (pre-filter; full key verified from the record).
    key_hash: u64,
    /// Record-file offset of the binding image this node indexes — or, for a
    /// tombstone, of the image being killed (so the tombstone's key is
    /// full-key verifiable too).
    record_off: u64,
    /// Next-older node in this bucket's chain; `IDX_NIL` terminates.
    prev: u64,
    /// [`IDX_FLAG_TOMBSTONE`] or 0.
    flags: u8,
}

/// A full-key-verified chain hit: the NEWEST node for the key.
#[derive(Debug)]
struct ChainHit {
    /// Record-file offset the newest node points at.
    record_off: u64,
    /// `false` = the newest verdict is a tombstone (key absent from spill).
    live: bool,
    /// The parsed record (valid image for a live hit; the killed image for a
    /// tombstone).
    binding: SpilledBinding,
}

/// FIX-3 — production bucket count for the spill index: two head arrays at
/// 2 × 2^21 × 8 B = 32 MiB flat RSS (allocated only for BOUNDED stores, which
/// exist to trade a fixed budget for O(N) growth), ~19-node chains at 40M
/// live bindings.
pub const SPILL_INDEX_DEFAULT_BUCKETS: u64 = 1 << 21;
/// Fixed index-node length in `idempotency-spill.idx`.
const IDX_NODE_LEN: u64 = 32;
/// Empty-bucket / end-of-chain sentinel.
const IDX_NIL: u64 = u64::MAX;
/// Node flag: this node KILLS its key (newest verdict = absent).
const IDX_FLAG_TOMBSTONE: u8 = 1;

impl IdempotencySpill {
    /// Open (create/truncate) the spill files at `dir/idempotency-spill.{db,idx}`.
    /// Process-local scratch — truncated on open, discarded on restart
    /// (recovery rebuilds the bindings from WAL + checkpoint).
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        Self::open_with_buckets(dir, SPILL_INDEX_DEFAULT_BUCKETS)
    }

    /// FIX-3 test hook — open with a caller-chosen bucket count so tests can
    /// force LONG chains (shadowed copies, tombstones, collisions) at small N.
    /// Production uses [`Self::open`]'s default.
    #[doc(hidden)]
    pub fn open_with_buckets(dir: &Path, buckets: u64) -> std::io::Result<Self> {
        let buckets = buckets.max(1);
        let path = dir.join("idempotency-spill.db");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let idx_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(dir.join("idempotency-spill.idx"))?;
        Ok(Self {
            file: Mutex::new(file),
            index: Mutex::new(SpillIndex {
                file: idx_file,
                write_off: 0,
                heads_fwd: vec![
                    IDX_NIL;
                    usize::try_from(buckets).expect("bucket count fits usize")
                ],
                heads_rev: vec![
                    IDX_NIL;
                    usize::try_from(buckets).expect("bucket count fits usize")
                ],
                live_forward: 0,
            }),
            buckets,
            hasher: RandomState::new(),
            write_offset: AtomicU64::new(0),
            path,
        })
    }

    /// Bucket hash of a FORWARD key. Domain-tagged so the two directions can
    /// never alias even for byte-identical tuples.
    fn hash_fwd(&self, tenant: TenantId, kind: u8, external_id: &str) -> u64 {
        let mut h = self.hasher.build_hasher();
        0u8.hash(&mut h);
        tenant.hash(&mut h);
        kind.hash(&mut h);
        external_id.hash(&mut h);
        h.finish()
    }

    /// Bucket hash of a REVERSE key.
    fn hash_rev(&self, tenant: TenantId, kind: u8, internal_id: u64) -> u64 {
        let mut h = self.hasher.build_hasher();
        1u8.hash(&mut h);
        tenant.hash(&mut h);
        kind.hash(&mut h);
        internal_id.hash(&mut h);
        h.finish()
    }

    /// Append one chain node to the index file; returns its offset. Caller
    /// holds the index lock.
    fn idx_append(idx: &mut SpillIndex, node: &IdxNode) -> std::io::Result<u64> {
        let mut buf = [0u8; IDX_NODE_LEN as usize];
        buf[0..8].copy_from_slice(&node.key_hash.to_le_bytes());
        buf[8..16].copy_from_slice(&node.record_off.to_le_bytes());
        buf[16..24].copy_from_slice(&node.prev.to_le_bytes());
        buf[24] = node.flags;
        let off = idx.write_off;
        idx.file.seek(SeekFrom::Start(off))?;
        idx.file.write_all(&buf)?;
        idx.write_off += IDX_NODE_LEN;
        Ok(off)
    }

    /// Read one chain node from the index file. Caller holds the index lock.
    fn idx_read(idx: &mut SpillIndex, off: u64) -> std::io::Result<IdxNode> {
        let mut buf = [0u8; IDX_NODE_LEN as usize];
        idx.file.seek(SeekFrom::Start(off))?;
        idx.file.read_exact(&mut buf)?;
        Ok(IdxNode {
            key_hash: u64::from_le_bytes(buf[0..8].try_into().expect("slice of len 8")),
            record_off: u64::from_le_bytes(buf[8..16].try_into().expect("slice of len 8")),
            prev: u64::from_le_bytes(buf[16..24].try_into().expect("slice of len 8")),
            flags: buf[24],
        })
    }

    /// Append a tombstone node for `key_hash` killing the image at
    /// `record_off`, prepended to the forward (`reverse = false`) or reverse
    /// chain. Caller holds the index lock (find + tombstone is atomic).
    fn tombstone_node(
        &self,
        idx: &mut SpillIndex,
        key_hash: u64,
        record_off: u64,
        reverse: bool,
    ) -> std::io::Result<()> {
        let b = (key_hash % self.buckets) as usize;
        let prev = if reverse {
            idx.heads_rev[b]
        } else {
            idx.heads_fwd[b]
        };
        let off = Self::idx_append(
            idx,
            &IdxNode {
                key_hash,
                record_off,
                prev,
                flags: IDX_FLAG_TOMBSTONE,
            },
        )?;
        if reverse {
            idx.heads_rev[b] = off;
        } else {
            idx.heads_fwd[b] = off;
        }
        // NOTE: compaction is triggered by the CALLERS after their
        // `live_forward` accounting settles — compacting here would race the
        // pending decrement and skew the survivor tripwire.
        Ok(())
    }

    /// FIX-3 — amortized chain compaction. Shadowed copies (re-spills) and
    /// tombstones are APPEND-only, so without reclamation the index file and
    /// every enumeration walk grow with TOTAL CHURN, not live keys — the
    /// O(N)-class term merely relocated to disk + capture-time (exactly the
    /// "moved the term, didn't kill it" failure the round-2 verdict names).
    /// When the file exceeds 8× the live-node bound, rewrite every bucket
    /// keeping only the NEWEST node per distinct key, dropping dead weight:
    /// enumeration + file size return to O(live); amortized O(1) per append.
    fn maybe_compact(&self, idx: &mut SpillIndex) -> std::io::Result<()> {
        // Live nodes ≈ one forward + one reverse per live binding. The
        // 4096-NODE (128 KiB) floor keeps a tiny-live/high-churn store (hot
        // keys re-spilled continuously) from re-compacting every few dozen
        // appends — each compaction scans ALL bucket heads, so it must stay
        // amortized against thousands of appends, not tens.
        let live_bound = idx
            .live_forward
            .saturating_mul(2)
            .saturating_mul(IDX_NODE_LEN);
        if idx.write_off <= live_bound.saturating_mul(8).max(4096 * IDX_NODE_LEN) {
            return Ok(());
        }
        self.compact(idx)
    }

    /// Rewrite the index file keeping only the newest LIVE node per distinct
    /// key (a dead-newest key keeps nothing — "no node" and "tombstone-newest"
    /// are equivalent verdicts). Processes ONE bucket-chain at a time, so the
    /// transient RAM is the per-chain seen-set — O(N/buckets), never O(N).
    /// Caller holds the index lock for the whole rewrite (compaction is atomic
    /// w.r.t. every other index op).
    fn compact(&self, idx: &mut SpillIndex) -> std::io::Result<()> {
        let idx_path = self.path.with_file_name("idempotency-spill.idx");
        let tmp_path = self.path.with_file_name("idempotency-spill.idx.compact");
        let mut new_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut new_off: u64 = 0;
        let buckets = idx.heads_fwd.len();
        let mut new_heads_fwd = vec![IDX_NIL; buckets];
        let mut new_heads_rev = vec![IDX_NIL; buckets];
        let mut new_live_fwd: u64 = 0;
        let mut write_survivor =
            |new_heads: &mut [u64], b: usize, node: &IdxNode| -> std::io::Result<()> {
                let mut buf = [0u8; IDX_NODE_LEN as usize];
                buf[0..8].copy_from_slice(&node.key_hash.to_le_bytes());
                buf[8..16].copy_from_slice(&node.record_off.to_le_bytes());
                buf[16..24].copy_from_slice(&new_heads[b].to_le_bytes());
                buf[24] = node.flags;
                new_file.write_all(&buf)?;
                new_heads[b] = new_off;
                new_off += IDX_NODE_LEN;
                Ok(())
            };
        for b in 0..buckets {
            // Forward chain: newest-first, first full-key occurrence wins.
            let mut node_off = idx.heads_fwd[b];
            if node_off != IDX_NIL {
                let mut seen: HashSet<(TenantId, u8, String)> = HashSet::new();
                while node_off != IDX_NIL {
                    let node = Self::idx_read(idx, node_off)?;
                    let (t, k, sb) = self.read_record_at(node.record_off)?;
                    if seen.insert((t, k, sb.external_id)) && node.flags & IDX_FLAG_TOMBSTONE == 0 {
                        write_survivor(&mut new_heads_fwd, b, &node)?;
                        new_live_fwd += 1;
                    }
                    node_off = node.prev;
                }
            }
            // Reverse chain: same rule keyed by internal_id.
            let mut node_off = idx.heads_rev[b];
            if node_off != IDX_NIL {
                let mut seen: HashSet<(TenantId, u8, u64)> = HashSet::new();
                while node_off != IDX_NIL {
                    let node = Self::idx_read(idx, node_off)?;
                    let (t, k, sb) = self.read_record_at(node.record_off)?;
                    if seen.insert((t, k, sb.internal_id)) && node.flags & IDX_FLAG_TOMBSTONE == 0 {
                        write_survivor(&mut new_heads_rev, b, &node)?;
                    }
                    node_off = node.prev;
                }
            }
        }
        // The survivor walk and the maintained counter compute the same
        // quantity two independent ways — a mismatch is an accounting bug
        // (it would skew `binding_count`, which the producer's CountSkew
        // hard-abort catches downstream; here it is a debug tripwire).
        debug_assert_eq!(
            new_live_fwd, idx.live_forward,
            "compaction survivor count diverged from live_forward",
        );
        // Atomically adopt the compacted file (scratch — no fsync needed).
        std::fs::rename(&tmp_path, &idx_path)?;
        idx.file = new_file;
        idx.write_off = new_off;
        idx.heads_fwd = new_heads_fwd;
        idx.heads_rev = new_heads_rev;
        Ok(())
    }

    /// Walk the FORWARD chain for `(tenant, kind, external_id)`: the newest
    /// full-key-verified verdict, or `None` if the key was never spilled.
    /// Caller holds the index lock.
    fn find_fwd(
        &self,
        idx: &mut SpillIndex,
        key_hash: u64,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> std::io::Result<Option<ChainHit>> {
        let mut node_off = idx.heads_fwd[(key_hash % self.buckets) as usize];
        while node_off != IDX_NIL {
            let node = Self::idx_read(idx, node_off)?;
            if node.key_hash == key_hash {
                let (t, k, sb) = self.read_record_at(node.record_off)?;
                if t == tenant && k == kind && sb.external_id == external_id {
                    return Ok(Some(ChainHit {
                        record_off: node.record_off,
                        live: node.flags & IDX_FLAG_TOMBSTONE == 0,
                        binding: sb,
                    }));
                }
            }
            node_off = node.prev;
        }
        Ok(None)
    }

    /// Walk the REVERSE chain for `(tenant, kind, internal_id)`. Caller holds
    /// the index lock.
    fn find_rev(
        &self,
        idx: &mut SpillIndex,
        key_hash: u64,
        tenant: TenantId,
        kind: u8,
        internal_id: u64,
    ) -> std::io::Result<Option<ChainHit>> {
        let mut node_off = idx.heads_rev[(key_hash % self.buckets) as usize];
        while node_off != IDX_NIL {
            let node = Self::idx_read(idx, node_off)?;
            if node.key_hash == key_hash {
                let (t, k, sb) = self.read_record_at(node.record_off)?;
                if t == tenant && k == kind && sb.internal_id == internal_id {
                    return Ok(Some(ChainHit {
                        record_off: node.record_off,
                        live: node.flags & IDX_FLAG_TOMBSTONE == 0,
                        binding: sb,
                    }));
                }
            }
            node_off = node.prev;
        }
        Ok(None)
    }

    /// Append the spilled binding image for `(tenant, kind, external_id)`,
    /// recording its offset in BOTH the forward and reverse indices. Re-spilling
    /// a key overwrites both indices to the newest copy (append-only file; stale
    /// bytes are dead space reclaimed on the next restart's truncate).
    ///
    /// Returns the byte offset of the appended record, so the caller
    /// (`evict_one`) can ROLL BACK exactly this image (offset-guarded) if the
    /// resident value changed under it (#1404 M0.x round-2 FIX-1).
    ///
    /// On-disk record layout v2 (little-endian; FIX-3 made records
    /// self-describing so a chain candidate is full-key verifiable):
    /// ```text
    ///  0..8   tenant       (u64, TenantId::raw)
    ///  8..9   kind         (u8)
    ///  9..17  internal_id  (u64)
    /// 17..18  hash_present (u8: 0/1)
    /// 18..26  payload_hash (u64; 0 when absent)
    /// 26..30  ext_len      (u32)
    /// 30..    external_id  (ext_len UTF-8 bytes)
    /// ```
    fn write_binding(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) -> std::io::Result<u64> {
        let ext_bytes = external_id.as_bytes();
        let mut rec = Vec::with_capacity(30 + ext_bytes.len());
        rec.extend_from_slice(&tenant.raw().to_le_bytes());
        rec.push(kind);
        rec.extend_from_slice(&internal_id.to_le_bytes());
        rec.push(u8::from(payload_hash.is_some()));
        rec.extend_from_slice(&payload_hash.unwrap_or(0).to_le_bytes());
        rec.extend_from_slice(
            &u32::try_from(ext_bytes.len())
                .expect("external_id length fits in u32")
                .to_le_bytes(),
        );
        rec.extend_from_slice(ext_bytes);
        let off = self
            .write_offset
            .fetch_add(rec.len() as u64, Ordering::AcqRel);
        {
            let mut f = self
                .file
                .lock()
                .expect("idempotency spill file mutex poisoned");
            f.seek(SeekFrom::Start(off))?;
            f.write_all(&rec)?;
        }
        // Index BOTH directions under the index lock (#1404 M0.x FIX-A — one
        // append feeds forward + reverse, so `external_id_for` faults from
        // here too). The liveness pre-check keeps `live_forward` EXACT under
        // overwrites: an idempotent re-spill of a still-live key must not
        // double-count (pre-fix `offsets.insert` overwrote in place).
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h_f = self.hash_fwd(tenant, kind, external_id);
        let was_live = self
            .find_fwd(&mut idx, h_f, tenant, kind, external_id)?
            .is_some_and(|hit| hit.live);
        let b = (h_f % self.buckets) as usize;
        let prev = idx.heads_fwd[b];
        let node_off = Self::idx_append(
            &mut idx,
            &IdxNode {
                key_hash: h_f,
                record_off: off,
                prev,
                flags: 0,
            },
        )?;
        idx.heads_fwd[b] = node_off;
        if !was_live {
            idx.live_forward += 1;
        }
        let h_r = self.hash_rev(tenant, kind, internal_id);
        let b_r = (h_r % self.buckets) as usize;
        let prev = idx.heads_rev[b_r];
        let node_off = Self::idx_append(
            &mut idx,
            &IdxNode {
                key_hash: h_r,
                record_off: off,
                prev,
                flags: 0,
            },
        )?;
        idx.heads_rev[b_r] = node_off;
        self.maybe_compact(&mut idx)?;
        Ok(off)
    }

    /// Read the LIVE spilled binding for `(tenant, kind, external_id)`, if
    /// present.
    fn read_binding(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> std::io::Result<Option<SpilledBinding>> {
        Ok(self
            .lookup_forward(tenant, kind, external_id)?
            .map(|(_, sb)| sb))
    }

    /// Read the LIVE spilled binding + the record offset it lives at.
    /// The offset feeds the round-2 FIX-1 offset-guarded retires.
    fn lookup_forward(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> std::io::Result<Option<(u64, SpilledBinding)>> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_fwd(tenant, kind, external_id);
        Ok(self
            .find_fwd(&mut idx, h, tenant, kind, external_id)?
            .filter(|hit| hit.live)
            .map(|hit| (hit.record_off, hit.binding)))
    }

    /// #1404 M0.x FIX-A — read the LIVE spilled binding by INTERNAL id (the
    /// reverse direction), if present. Backs `external_id_for`'s spill
    /// fault-in on the delete path so an evicted binding's external_id is
    /// always recoverable.
    fn read_binding_by_internal(
        &self,
        tenant: TenantId,
        kind: u8,
        internal_id: u64,
    ) -> std::io::Result<Option<SpilledBinding>> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_rev(tenant, kind, internal_id);
        Ok(self
            .find_rev(&mut idx, h, tenant, kind, internal_id)?
            .filter(|hit| hit.live)
            .map(|hit| hit.binding))
    }

    /// Read + parse the self-describing record at `off` from the record file.
    fn read_record_at(&self, off: u64) -> std::io::Result<(TenantId, u8, SpilledBinding)> {
        let mut header = [0u8; 30];
        let mut ext_buf;
        {
            let mut f = self
                .file
                .lock()
                .expect("idempotency spill file mutex poisoned");
            f.seek(SeekFrom::Start(off))?;
            f.read_exact(&mut header)?;
            let ext_len = u32::from_le_bytes(
                header[26..30]
                    .try_into()
                    .expect("slice of len 4 fits into [u8;4]"),
            ) as usize;
            ext_buf = vec![0u8; ext_len];
            f.read_exact(&mut ext_buf)?;
        }
        let tenant = TenantId::new(u64::from_le_bytes(
            header[0..8]
                .try_into()
                .expect("slice of len 8 fits into [u8;8]"),
        ));
        let kind = header[8];
        let internal_id = u64::from_le_bytes(
            header[9..17]
                .try_into()
                .expect("slice of len 8 fits into [u8;8]"),
        );
        let has_hash = header[17] != 0;
        let raw_hash = u64::from_le_bytes(
            header[18..26]
                .try_into()
                .expect("slice of len 8 fits into [u8;8]"),
        );
        let external_id = String::from_utf8(ext_buf).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("spilled idempotency external_id not utf-8: {e}"),
            )
        })?;
        Ok((
            tenant,
            kind,
            SpilledBinding {
                external_id,
                internal_id,
                payload_hash: has_hash.then_some(raw_hash),
            },
        ))
    }

    /// Tombstone the LIVE forward entry for the key (release path). No-op if
    /// absent or already dead. Pre-fix equivalent: `offsets.remove(&key)`.
    fn remove_forward(&self, tenant: TenantId, kind: u8, external_id: &str) -> std::io::Result<()> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_fwd(tenant, kind, external_id);
        if let Some(hit) = self.find_fwd(&mut idx, h, tenant, kind, external_id)? {
            if hit.live {
                self.tombstone_node(&mut idx, h, hit.record_off, false)?;
                idx.live_forward -= 1;
                self.maybe_compact(&mut idx)?;
            }
        }
        Ok(())
    }

    /// Round-2 FIX-1 — offset-GUARDED forward retire: tombstone ONLY if the
    /// newest live image for the key is the record at `guard_off`, never a
    /// newer legitimate image written by a concurrent eviction. Pre-fix
    /// equivalent: `offsets.remove_if(&key, |_, o| *o == guard_off)`.
    fn retire_forward_if(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        guard_off: u64,
    ) -> std::io::Result<()> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_fwd(tenant, kind, external_id);
        if let Some(hit) = self.find_fwd(&mut idx, h, tenant, kind, external_id)? {
            if hit.live && hit.record_off == guard_off {
                self.tombstone_node(&mut idx, h, hit.record_off, false)?;
                idx.live_forward -= 1;
                self.maybe_compact(&mut idx)?;
            }
        }
        Ok(())
    }

    /// Tombstone the LIVE reverse entry for `(tenant, kind, internal_id)`.
    /// Pre-fix equivalent: `reverse_offsets.remove(&key)`.
    fn remove_reverse(&self, tenant: TenantId, kind: u8, internal_id: u64) -> std::io::Result<()> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_rev(tenant, kind, internal_id);
        if let Some(hit) = self.find_rev(&mut idx, h, tenant, kind, internal_id)? {
            if hit.live {
                self.tombstone_node(&mut idx, h, hit.record_off, true)?;
                self.maybe_compact(&mut idx)?;
            }
        }
        Ok(())
    }

    /// Round-2 FIX-1 — offset-GUARDED reverse retire. Pre-fix equivalent:
    /// `reverse_offsets.remove_if(&key, |_, o| *o == guard_off)`.
    fn retire_reverse_if(
        &self,
        tenant: TenantId,
        kind: u8,
        internal_id: u64,
        guard_off: u64,
    ) -> std::io::Result<()> {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        let h = self.hash_rev(tenant, kind, internal_id);
        if let Some(hit) = self.find_rev(&mut idx, h, tenant, kind, internal_id)? {
            if hit.live && hit.record_off == guard_off {
                self.tombstone_node(&mut idx, h, hit.record_off, true)?;
                self.maybe_compact(&mut idx)?;
            }
        }
        Ok(())
    }

    /// Whether the key has a LIVE spilled image. Pre-fix equivalent:
    /// `offsets.contains_key(&key)`.
    fn contains_forward(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> std::io::Result<bool> {
        Ok(self.lookup_forward(tenant, kind, external_id)?.is_some())
    }

    /// Exact count of DISTINCT LIVE forward keys — O(1) (the counter is
    /// maintained under the index lock). Pre-fix equivalent: `offsets.len()`.
    fn live_forward_len(&self) -> u64 {
        self.index
            .lock()
            .expect("idempotency spill index mutex poisoned")
            .live_forward
    }

    /// FIX-3 — enumerate every DISTINCT LIVE spilled binding, one at a time
    /// (the capture walk; pre-fix: `offsets.iter()` + per-entry
    /// `read_binding_at`). Chains are newest-first, so the FIRST full-key
    /// occurrence in a chain is the authoritative verdict; older shadowed
    /// copies and tombstoned keys are skipped via a per-chain seen-set — the
    /// only per-call RAM, O(chain length) = O(N/buckets) transient, dropped
    /// per bucket, never O(N).
    fn for_each_live<E, F>(&self, mut f: F) -> Result<(), E>
    where
        F: FnMut(TenantId, u8, SpilledBinding) -> Result<(), E>,
    {
        let mut idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        for b in 0..idx.heads_fwd.len() {
            let mut node_off = idx.heads_fwd[b];
            if node_off == IDX_NIL {
                continue;
            }
            let mut seen: HashSet<(TenantId, u8, String)> = HashSet::new();
            while node_off != IDX_NIL {
                let node = Self::idx_read(&mut idx, node_off)
                    .expect("idempotency spill index read for enumeration failed");
                let (t, k, sb) = self
                    .read_record_at(node.record_off)
                    .expect("idempotency spill read for enumeration failed");
                let dead = node.flags & IDX_FLAG_TOMBSTONE != 0;
                let key = (t, k, sb.external_id.clone());
                if !seen.contains(&key) {
                    seen.insert(key);
                    if !dead {
                        f(t, k, sb)?;
                    }
                }
                node_off = node.prev;
            }
        }
        Ok(())
    }

    /// Count of DISTINCT LIVE spilled keys for `tenant` (test/introspection —
    /// a full index walk). Pre-fix: `offsets.iter().filter(tenant)`.
    fn count_forward_for_tenant(&self, tenant: TenantId) -> usize {
        let mut n = 0usize;
        self.for_each_live::<std::convert::Infallible, _>(|t, _, _| {
            if t == tenant {
                n += 1;
            }
            Ok(())
        })
        .expect("infallible");
        n
    }

    /// FIX-3 gate oracle — the spill index's RESIDENT in-RAM entry-slot count:
    /// the two bucket-head arrays (chain nodes live on disk; nothing else is
    /// resident). CONSTANT in N post-fix; the pre-fix DashMap pair grew 1:1.
    #[doc(hidden)]
    #[must_use]
    pub fn index_resident_entries(&self) -> u64 {
        let idx = self
            .index
            .lock()
            .expect("idempotency spill index mutex poisoned");
        (idx.heads_fwd.len() + idx.heads_rev.len()) as u64
    }

    /// Path to the spill file (diagnostics/tests).
    #[doc(hidden)]
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// A resident binding + its eviction bookkeeping (#1404 M0.x).
///
/// `checkpointed` is set to `true` by [`IdempotencyStore::iter_all`] when a
/// completed checkpoint captures the binding's durable image; only
/// `checkpointed` bindings are evict-eligible (INV-DURABLE). One `AtomicBool`
/// per resident binding, no side map.
/// `install_gen` (#1404 M0.x round-3 FIX-5) is a store-unique INSTALL
/// GENERATION stamped on EVERY install/overwrite (including the re-fault
/// warm-insert). `evict_one` compares-and-removes on `(internal_id,
/// install_gen)`: a plain `internal_id` compare cannot distinguish a
/// same-`internal_id` FRESH re-publish (WAL replay / idempotent re-publish —
/// see [`IdempotencyStore::install`] docs) from the evictor's own stale
/// snapshot, so the pre-fix evictor could drop the fresh binding from the
/// resident tier while the overwriter retired the evictor's spill image →
/// both tiers absent for a continuously-live binding → `get()==None` → a
/// duplicate on re-ingest (round-3 skeptic 1: 35 misses / 23 stable).
#[derive(Debug)]
struct ResidentBinding {
    binding: IdempotencyBinding,
    checkpointed: AtomicBool,
    install_gen: u64,
}

/// #1500 FIX-5(f) — sharded per-key SLOT generations (the vacant-ABA stamp).
///
/// `insert_resident_warm_if_vacant` (FIX-5(e)) rejects a stale warm-insert
/// only while the fresher same-key install is STILL RESIDENT. It cannot see
/// the full ABA: vacant → fresh install → checkpoint-capture → fresh evict →
/// vacant again, all between a re-faulting reader's spill read and its
/// vacant-check — the reader then installs its OLDER spill image as the
/// resident value, marked checkpoint-durable, so the regression sticks and
/// re-spills (fix5b gate: a committed same-id re-publish followed by reads
/// serving the pre-publish `payload_hash` — an at-least-once violation
/// upstream, the payload-dedup verdict is computed from that hash).
///
/// Every REMOVAL of a resident forward entry (evict compare-and-remove,
/// release) bumps its key's slot generation BEFORE the remove. A re-faulting
/// reader snapshots the generation BEFORE its two-tier read; the warm-insert
/// re-loads it under the vacant entry's shard lock and REJECTS (caller
/// re-faults fresh) if it advanced: an observed-vacant slot whose generation
/// moved has completed at least one install+removal cycle since the snapshot,
/// so the reader's spill image may be superseded. Visibility: the bump is
/// sequenced before the remove in the remover, and observing the slot VACANT
/// through the shard lock synchronizes-with that remove, so the re-load sees
/// the bump.
///
/// Sharded by key hash (fixed 1024 slots — O(1) resident memory, never
/// per-key growth; this store's OOM class is #1404): a collision only causes
/// a spurious re-fault retry, never a missed rejection. NOT a lock: the hot
/// resident-hit read path never touches this, and writers only `fetch_add`.
struct SlotGenerations([AtomicU64; SLOT_GENERATION_SHARDS]);

const SLOT_GENERATION_SHARDS: usize = 1024;

impl std::fmt::Debug for SlotGenerations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotGenerations").finish_non_exhaustive()
    }
}

impl Default for SlotGenerations {
    fn default() -> Self {
        Self(std::array::from_fn(|_| AtomicU64::new(0)))
    }
}

impl SlotGenerations {
    fn shard(&self, key: &(TenantId, u8, String)) -> &AtomicU64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        &self.0[(hasher.finish() as usize) % SLOT_GENERATION_SHARDS]
    }

    /// Snapshot the key's slot generation (reader side, before the two-tier
    /// read).
    fn load(&self, key: &(TenantId, u8, String)) -> u64 {
        self.shard(key).load(Ordering::Acquire)
    }

    /// Advance the key's slot generation. MUST be called BEFORE the resident
    /// forward entry is removed (the vacant observation is what publishes it).
    fn bump(&self, key: &(TenantId, u8, String)) {
        self.shard(key).fetch_add(1, Ordering::AcqRel);
    }
}

impl ResidentBinding {
    fn new(binding: IdempotencyBinding, install_gen: u64) -> Self {
        Self {
            binding,
            checkpointed: AtomicBool::new(false),
            install_gen,
        }
    }
}

/// Outcome of an eviction attempt on one resident binding (#1404 M0.x drain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictOutcome {
    /// The binding was spilled + dropped from the resident tier.
    Evicted,
    /// The binding was already gone (concurrent release / overwrite).
    Gone,
    /// The binding is not yet checkpoint-durable — kept resident
    /// (INV-DURABLE).
    NotDurable,
}

/// Per-tenant, semantics-agnostic `(kind, external_id) → internal_id`
/// store. Rebuilt on WAL replay from the `CommitBundle`
/// `idempotency_bindings` section; the durable source of truth is the
/// WAL, so an entry, once committed, can never be silently lost.
///
/// **#1404 M0.x (bounded resident tier).** The `forward`/`reverse` maps are
/// the RESIDENT hot tier. Without a spill file attached ([`Self::new`], the
/// legacy default) they are unbounded in-memory `DashMap`s (unchanged
/// behavior). With a spill attached ([`Self::with_bound`], the durable-serve
/// path), the resident set is bounded to `config.high_watermark_bytes`: once a
/// checkpoint has captured a binding's durable image, that binding is
/// evict-eligible and may be spilled to `idempotency-spill.db` to bound RSS
/// (+ its `iter_all` freeze-capture spike), faulting back in on lookup.
#[derive(Debug, Default)]
struct ResidentIdempotencyOwners {
    /// Forward map: `(tenant, kind, external_id)` → resident binding + evict
    /// bookkeeping.
    forward: DashMap<(TenantId, u8, String), ResidentBinding>,
    /// Reverse map: `(tenant, kind, internal_id)` → external_id.
    reverse: DashMap<(TenantId, u8, u64), String>,

    // ── #1404 M0.x bounded-tier state (inert unless `spill` is `Some`) ──
    /// Durable QUERYABLE spill tier for evicted bindings + the re-fault
    /// source. `None` = unbounded legacy behavior (nothing is ever evicted).
    spill: Option<Arc<IdempotencySpill>>,
    /// Resident-tier byte watermarks. Only consulted when `spill.is_some()`.
    config: IdempotencyBoundConfig,
    /// Running resident binding bytes (`forward.len() ×
    /// IDEMPOTENCY_BINDING_WEIGHT_BYTES`), an `AtomicU64` so the drain trigger
    /// is a load, not a scan.
    resident_bytes: AtomicU64,
    /// FIFO eviction order: `(tenant, kind, external_id)` in install order.
    /// Cheap to maintain on the write path (one push); eviction pops from the
    /// front. If the front binding is not yet checkpoint-durable, the drain
    /// restores it to the front and stops (FIFO-oldest-first ⟹ nothing behind
    /// a non-durable front is durable), so the drain is O(evicted-this-pass).
    /// `release` does NOT sweep its entry (that would be an O(queue) scan on
    /// the delete path); the stale entry resolves as a cheap `Gone` pop.
    /// Round-3 FIX-4 bounds the FIFO itself: `maybe_drain` also engages past
    /// [`Self::evict_queue_cap`] entries, so an ingest-then-delete workload
    /// reclaims the Gone backlog even with zero resident-byte pressure.
    /// Round-3 FIX-4b: while over the entry cap AND stale entries exist
    /// (`evict_queue_stale_hint`), the drain scans PAST a not-yet-durable
    /// front (retained + restored, duplicates dropped) — same-key churn with
    /// no checkpoint parks a permanently-`NotDurable` entry at the front,
    /// and the front-break alone reclaimed nothing.
    evict_queue: Mutex<VecDeque<(TenantId, u8, String)>>,
    /// Round-3 FIX-4b — upper bound on the number of RECLAIMABLE
    /// (`Gone`-or-duplicate) entries in `evict_queue`. `release` of a
    /// resident key strands that key's live FIFO entry (it will probe `Gone`,
    /// or `NotDurable`-duplicate if the key is re-installed) — the ONLY way
    /// reclaimable backlog is created — so release increments; the drain
    /// decrements per `Gone` pop / duplicate drop; the evict-rollback
    /// re-enqueue compensates its own `Gone` decrement (+1). The counter may
    /// OVERCOUNT (a release racing an in-flight evictor that consumes the
    /// entry) — that direction is safe, costing at most a bounded wasted
    /// scan; it never UNDERCOUNTS, so a zero reading proves there is nothing
    /// past a `NotDurable` front worth scanning for and the drain keeps the
    /// FIX-4 one-probe break (the drain-cost guard's O(installs) bound).
    evict_queue_stale_hint: AtomicU64,
    /// Count of evictions performed — test/observability only.
    evicted_count: AtomicU64,
    /// Count of re-faults from spill — test/observability only.
    refault_count: AtomicU64,
    /// Cumulative evict-queue probes across all drain passes — test only.
    drain_probe_count: AtomicU64,
    /// #1404 M0.x — the CAPTURE peak-in-flight record count from the LAST
    /// [`Self::for_each_binding`] pass: the max number of binding records the
    /// capture held resident SIMULTANEOUSLY. The streaming capture holds ≤1
    /// (read → emit → drop), so this stays a small constant INDEPENDENT of N —
    /// the freeze-capture RSS bound. A reverted whole-`Vec` capture would set
    /// this to N. Test/observability only (the gatex O(1)-in-N oracle reads it);
    /// updated even for the unbounded store (peak 1 there too, resident walk).
    capture_peak_in_flight: AtomicU64,
    /// #1404 M0.x FIX-D — capture-vs-install exclusion (the racy-two-pass /
    /// silent-data-loss fix). The ADR-229 checkpoint capture snapshots the
    /// section COUNT (`binding_count`) then STREAMS the records
    /// (`for_each_binding`) in TWO passes over this live map. A post-commit
    /// `install`/`release` interleaved BETWEEN the two passes (it runs OUTSIDE
    /// `checkpoint_freeze`) would skew header≠stream → a mis-framed section the
    /// producer writes `Ok` → #1365 WAL reclaim below the frontier → next-boot
    /// snapshot reject → from-zero replay missing the reclaimed prefix =
    /// PERMANENT DATA LOSS. This `RwLock` closes it: `install`/`release` take
    /// the READ guard (concurrent with EACH OTHER — no ingest serialization),
    /// and the CAPTURE takes the WRITE guard for the whole count+stream span
    /// (exclusive → no install can interleave, so count == streamed
    /// deterministically). The HOT READ path (`get`/`resolve_binding`/
    /// `external_id_for`) NEVER touches this lock — reads stay lock-free (the
    /// spec's "do NOT plumb checkpoint_lock into the hot read path"). Wire
    /// layout is byte-untouched (count-first, same records). Defense-in-depth:
    /// the producer ALSO hard-checks streamed==header before finalize.
    capture_lock: parking_lot::RwLock<()>,
    /// #1404 M0.x round-2 FIX-1 — spill→resident MOVE epoch (seqlock-style).
    /// The two-tier lookup (`resolve_binding`/`external_id_for`) checks
    /// resident-then-spill WITHOUT a lock; the one transition that can hide a
    /// LIVE binding from that read order is a spill→resident move (an
    /// overwrite-`install` inserting the fresh resident copy and then retiring
    /// the stale spill-index entry): a reader that missed resident BEFORE the
    /// insert and missed spill AFTER the retire would see a false
    /// double-miss → `get()==None` for a continuously-live binding → duplicate
    /// on re-ingest. Every such mover increments this AFTER the resident
    /// insert and BEFORE the spill-index removal; a reader that double-misses
    /// re-checks the epoch and retries iff it moved (a STABLE double-miss is
    /// authoritative absence). The resident→spill direction (evict) needs no
    /// epoch: it populates spill BEFORE removing resident, so the
    /// resident-then-spill read order always finds it.
    move_epoch: AtomicU64,
    /// #1404 M0.x round-3 FIX-5 — monotone install-generation source. Every
    /// resident insert ([`Self::insert_resident_warm`]) stamps the new
    /// [`ResidentBinding`] with a fresh generation, so `evict_one`'s
    /// compare-and-remove can distinguish a same-`internal_id` fresh
    /// re-publish from the exact snapshot it spilled (see the
    /// [`ResidentBinding`] docs for the double-miss this closes).
    install_gen_counter: AtomicU64,
    /// #1404 M0.x round-3 FIX-5(c) — at most ONE in-flight evictor per key.
    /// The eviction FIFO can hold DUPLICATE entries for a key (the Gone-branch
    /// re-enqueue admits this as benign), so two drain passes can run
    /// `evict_one` on the SAME key concurrently. Each writes its own spill
    /// image; the loser's rollback (`retire_forward_if` on its OWN newest
    /// node) then tombstones a node that SHADOWS the winner's older LIVE
    /// image — spill lookups honor the NEWEST node's verdict, so the
    /// spilled-only binding turns invisible → a stable `get()==None` for a
    /// continuously-live binding. This map is the interlock: an evictor
    /// `insert`s its key here before snapshotting and skips (returns `Gone`)
    /// if a sibling already owns the key's transition. Bounded: ≤ one entry
    /// per concurrently-draining key, removed on every exit.
    evict_inflight: DashMap<(TenantId, u8, String), ()>,
    /// #1500 FIX-5(f) — per-key-shard slot generations for the re-fault
    /// warm-insert's vacant-ABA rejection (see [`SlotGenerations`]).
    slot_generations: SlotGenerations,
}

#[derive(Debug)]
pub struct IdempotencyStore {
    /// M4 authoritative owner.
    physical: Option<Arc<OwnerRowRegistry>>,
    /// Entire pre-M4 owner bundle. This is absent, not merely empty, for a
    /// page-backed facade, so neither direction's DashMap can scale with N.
    resident: Option<Box<ResidentIdempotencyOwners>>,
}

impl Default for IdempotencyStore {
    fn default() -> Self {
        Self {
            physical: None,
            resident: Some(Box::default()),
        }
    }
}

impl IdempotencyStore {
    /// Empty, UNBOUNDED store (the legacy default + the test/no-`--data`
    /// path). No spill file → nothing is ever evicted → identical behavior to
    /// the pre-#1404 store. Per-tenant state is created lazily on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// #1404 M0.x — an empty store with the BOUNDED resident tier engaged.
    ///
    /// `spill` is the durable QUERYABLE re-fault tier ([`IdempotencySpill::open`]);
    /// `config` sets the resident-byte watermarks. Once a checkpoint captures a
    /// binding's durable image, that binding is evict-eligible and may be
    /// spilled to bound RSS, faulting back in from `spill` on [`Self::get`].
    /// Used by the durable serve bootstrap; tests build it directly to
    /// exercise eviction.
    #[must_use]
    pub fn with_bound(spill: Arc<IdempotencySpill>, config: IdempotencyBoundConfig) -> Self {
        Self {
            physical: None,
            resident: Some(Box::new(ResidentIdempotencyOwners {
                spill: Some(spill),
                config,
                ..ResidentIdempotencyOwners::default()
            })),
        }
    }

    /// M4 page-backed binding facade. It has neither a spill file nor any
    /// resident forward/reverse entries.
    #[must_use]
    pub fn page_backed(owner: Arc<OwnerRowRegistry>) -> Self {
        Self {
            physical: Some(owner),
            resident: None,
        }
    }

    /// True only for the post-M4 owner facade.
    #[must_use]
    pub fn is_page_backed(&self) -> bool {
        self.physical.is_some()
    }

    /// Structural census: the post-swap binding owner has no resident-map or
    /// spill bundle that could be repopulated by replay.
    #[must_use]
    pub fn has_resident_owner_maps(&self) -> bool {
        self.resident.is_some()
    }

    fn resident(&self) -> &ResidentIdempotencyOwners {
        match self.resident.as_deref() {
            Some(resident) => resident,
            None => unreachable!("resident idempotency path selected for page-backed facade"),
        }
    }

    /// True iff the bounded resident tier is engaged (a spill file is
    /// attached). Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn is_bounded(&self) -> bool {
        self.resident
            .as_deref()
            .is_some_and(|resident| resident.spill.is_some())
    }

    /// Current resident binding bytes (the RSS this store's binding maps
    /// contribute, modulo DashMap overhead). Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.resident.as_deref().map_or(0, |resident| {
            resident.resident_bytes.load(Ordering::Acquire)
        })
    }

    /// Number of resident bindings (may be < the logical binding count when
    /// bindings are evicted-to-spill). Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn resident_len(&self) -> usize {
        self.resident
            .as_deref()
            .map_or(0, |resident| resident.forward.len())
    }

    /// #1404 M0.x FIX-A — number of RESIDENT reverse entries. The whole-store
    /// bound requires this to stay bounded near the forward resident set (both
    /// tiers evict together); pre-FIX-A it grew O(N) (the 6th OOM sibling).
    /// Test/observability.
    #[doc(hidden)]
    #[must_use]
    pub fn resident_reverse_len(&self) -> usize {
        self.resident
            .as_deref()
            .map_or(0, |resident| resident.reverse.len())
    }

    /// Number of bindings evicted-to-spill since construction. Test only.
    #[doc(hidden)]
    #[must_use]
    pub fn evicted_count(&self) -> u64 {
        self.resident
            .as_deref()
            .map_or(0, |resident| resident.evicted_count.load(Ordering::Acquire))
    }

    /// Number of bindings re-faulted from spill since construction. Test only.
    #[doc(hidden)]
    #[must_use]
    pub fn refault_count(&self) -> u64 {
        self.resident
            .as_deref()
            .map_or(0, |resident| resident.refault_count.load(Ordering::Acquire))
    }

    /// Cumulative evict-queue probes across all drain passes since
    /// construction. Test only: the drain-cost regression guard asserts
    /// per-install drain cost is O(evicted-this-pass), not O(resident-count).
    #[doc(hidden)]
    #[must_use]
    pub fn drain_probe_count(&self) -> u64 {
        self.resident.as_deref().map_or(0, |resident| {
            resident.drain_probe_count.load(Ordering::Relaxed)
        })
    }

    /// #1404 M0.x round-3 FIX-4 gate oracle — current evict-FIFO entry count.
    /// `release` leaves its FIFO entry behind (reclaimed as a cheap `Gone` pop
    /// by a drain pass); pre-fix an ingest-then-delete workload leaked one
    /// entry per released binding (50,000 for a 0-binding store). Post-fix
    /// this stays O([`Self::evict_queue_cap`]), independent of total
    /// ever-released. Test/observability only.
    #[doc(hidden)]
    #[must_use]
    pub fn evict_queue_len(&self) -> usize {
        self.resident()
            .evict_queue
            .lock()
            .expect("idempotency evict_queue mutex poisoned")
            .len()
    }

    /// Force a drain pass (evict checkpoint-durable bindings down to the low
    /// watermark). Exposed for integration tests + the durable serve
    /// bootstrap's explicit post-checkpoint drain hook; production also drains
    /// inline on `install` (`maybe_drain`).
    #[doc(hidden)]
    pub fn force_drain_for_test(&self) {
        self.drain_to_low_watermark();
    }

    /// #1404 M0.x round-2 FIX-3 gate oracle — the resident in-RAM entry-slot
    /// count of the spill's index (0 for the unbounded store). Post-fix this
    /// is the two O(buckets) head arrays — CONSTANT in ingested binding count;
    /// the pre-fix `offsets`/`reverse_offsets` DashMap pair grew 1:1 with N
    /// (the relocated O(N-rels) OOM the FIX-3 gate red-lines on revert).
    #[doc(hidden)]
    #[must_use]
    pub fn spill_index_resident_entries(&self) -> u64 {
        self.resident()
            .spill
            .as_ref()
            .map_or(0, |s| s.index_resident_entries())
    }

    /// Test hook — mark every RESIDENT binding checkpoint-durable
    /// (evict-eligible, the INV-DURABLE gate) WITHOUT the full capture walk:
    /// the at-scale FIX-3 gate's marker loop must not pay `for_each_binding`'s
    /// O(spilled) enumeration per round just to flip resident flags.
    #[doc(hidden)]
    pub fn mark_resident_durable_for_test(&self) {
        for e in self.resident().forward.iter() {
            e.value().checkpointed.store(true, Ordering::Release);
        }
    }

    /// Look up the binding previously recorded for
    /// `(tenant, kind, external_id)`, or `None` if no binding exists.
    ///
    /// Read-only w.r.t. identity: a lookup never DROPS a binding, so a
    /// long-running tenant's bindings stay resolvable for the life of the
    /// process (and across restarts, via replay).
    ///
    /// **#1404 M0.x — spill fault-in (the load-bearing at-least-once leg).**
    /// A bounded store's lookup NEVER trusts resident-absence as
    /// binding-absence: on a resident miss it consults the durable, queryable
    /// spill tier (`Self::resolve_binding`), which faults the binding back
    /// into the resident maps. So an idempotent re-ingest of an external_id
    /// whose binding was spilled STILL de-dupes — correctness is IDENTICAL to
    /// the unbounded store, only latency-tiered. This is why bounding is
    /// spill-to-durable-queryable, NOT evict-to-nowhere (fable RE-2): dropping
    /// a binding to nowhere would return `None` here → a DUPLICATE on
    /// re-ingest.
    #[must_use]
    pub fn get(&self, tenant: TenantId, kind: u8, external_id: &str) -> Option<IdempotencyBinding> {
        self.try_get(tenant, kind, external_id)
            .unwrap_or_else(|error| {
                tracing::error!(%error, "M4 binding lookup failed closed");
                None
            })
    }

    /// Typed durable lookup. A 56-bit hash match is only a candidate: the
    /// complete external id and kind are rechecked against the authoritative
    /// direct row before an internal id can be returned.
    pub fn try_get(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> Result<Option<IdempotencyBinding>, OwnerRowError> {
        let Some(owner) = self.physical.as_ref() else {
            return Ok(self.resolve_binding(tenant, kind, external_id));
        };
        let class = binding_class(kind)?;
        let match_row =
            owner.find_verified(tenant, class, str_hash_56(external_id), |_id, logical| {
                BindingOwnerValue::decode(logical).is_ok_and(|value| {
                    value.active && value.kind == kind && value.external_id == external_id
                })
            })?;
        match match_row {
            Some((internal_id, logical)) => {
                let value = BindingOwnerValue::decode(&logical)?;
                Ok(Some(IdempotencyBinding {
                    internal_id,
                    payload_hash: value.payload_hash,
                }))
            }
            None => Ok(None),
        }
    }

    /// Resolve `(tenant, kind, external_id)` to its [`IdempotencyBinding`],
    /// re-faulting from the spill tier if it was evicted from RAM (#1404 M0.x
    /// re-fault path). A resident hit is the hot path (no I/O). A miss on both
    /// tiers returns `None` (genuinely absent — never seen, or released).
    ///
    /// The ordering hazard (evict-then-lookup = a missed identity → duplicate
    /// if there were no re-fault) is closed here: a bounded store's lookup
    /// consults the durable spill tier on a resident miss, and eviction only
    /// ever moves a binding whose durable image is already in spill (see
    /// [`Self::evict_one`]).
    fn resolve_binding(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> Option<IdempotencyBinding> {
        let key = (tenant, kind, external_id.to_owned());
        // #1404 M0.x round-2 FIX-1 — seqlock-style two-tier read. The lookup
        // stays LOCK-FREE (the spec's "do NOT plumb checkpoint_lock into the
        // hot read path"); instead of a lock it retries iff `move_epoch`
        // advanced across a double-miss (a concurrent spill→resident move
        // raced our two checks — see the field docs for the proof). A STABLE
        // double-miss is authoritative absence. Termination: every retry
        // requires another concurrent move of some binding; finite in any
        // real execution.
        loop {
            let epoch = self.resident().move_epoch.load(Ordering::Acquire);
            // #1500 FIX-5(f) — snapshot the key's slot generation BEFORE the
            // two-tier read: the warm-insert below rejects (and we re-fault
            // fresh) if a removal advanced it, closing the vacant-ABA the
            // vacant-check alone cannot see (vacant → fresh install → fresh
            // evict → vacant, between our spill read and our vacant-check).
            let slot_gen = self.resident().slot_generations.load(&key);
            if let Some(e) = self.resident().forward.get(&key) {
                return Some(e.value().binding);
            }
            // Resident miss — re-fault from spill (bounded store only).
            let spill = self.resident().spill.as_ref()?;
            if let Some(spilled) = spill
                .read_binding(tenant, kind, external_id)
                .expect("idempotency spill read on re-fault failed")
            {
                let binding = IdempotencyBinding {
                    internal_id: spilled.internal_id,
                    payload_hash: spilled.payload_hash,
                };
                let served = {
                    // #1404 M0.x round-2 FIX-2 (skeptics 4+5) — the re-fault
                    // WARM-INSERT is a writer to the section the capture
                    // frames, exactly like install/release, so it takes the
                    // same capture-exclusion READ guard for the insert pair.
                    // Unguarded (pre-fix) it interleaved the producer's
                    // write-guarded count+stream → header≠streamed →
                    // CountSkew aborted ~62% of checkpoints under read load →
                    // the WAL frontier never advanced → the unbounded-WAL
                    // regression (#1404/#1365) returned. The hot RESIDENT-HIT
                    // path above stays lock-free; only the (I/O-bound anyway)
                    // re-fault slow path pays an uncontended read-lock.
                    //
                    // SAFETY(no-self-deadlock): parking_lot's RwLock is
                    // non-reentrant, but the capture thread — the only
                    // write()-holder (`capture_guard`) — never calls
                    // `get()`/`resolve_binding`/`external_id_for` during the
                    // count+stream (it walks `forward`/the spill index
                    // directly), and no caller of this method already holds
                    // `capture_lock` (install/release take it but never look
                    // up). So this read() can block on a capture, never on
                    // itself.
                    let _warm = self.resident().capture_lock.read();
                    // Warm the resident tier so a hot key does not thrash the
                    // spill. The re-faulted binding IS checkpoint-durable (it
                    // was evicted, which requires it), so it is immediately
                    // evict-eligible again.
                    //
                    // Round-3 FIX-5(e) — insert-if-VACANT, never a blind
                    // overwrite: a fresh same-key `install` can land between
                    // our spill read and this warm-insert, and a blind
                    // `forward.insert` would CLOBBER the fresher binding with
                    // the stale spill image (same id, stale `payload_hash` —
                    // marked `checkpointed` to boot, so the clobbered value is
                    // immediately evictable and the staleness sticks). If an
                    // installer won the race, serve ITS value — it is newer by
                    // definition. Reproduced (fix5b gate, hash-freshness
                    // oracle): 35,689 stale serves/run with the blind insert,
                    // 0 with insert-if-vacant.
                    // #1500 FIX-5(f) — `None` means the slot generation
                    // advanced since our snapshot while the slot is now
                    // VACANT: at least one install+removal cycle completed
                    // under us (the vacant-ABA), so the spill image we hold
                    // may be superseded. Retry the whole two-tier read.
                    let Some((served, inserted)) = self.insert_resident_warm_if_vacant(
                        tenant,
                        kind,
                        external_id,
                        binding,
                        slot_gen,
                    ) else {
                        continue;
                    };
                    if inserted {
                        // #1404 M0.x FIX-A — re-populate the resident REVERSE
                        // entry too, so a later `external_id_for` resolves it
                        // resident (and the reverse tier stays consistent with
                        // the forward tier after a fault-in). Only when OUR
                        // warm-insert landed: a racing installer already wrote
                        // the reverse entry for its own (fresher) value.
                        self.resident()
                            .reverse
                            .insert((tenant, kind, binding.internal_id), external_id.to_owned());
                    }
                    served
                };
                self.resident().refault_count.fetch_add(1, Ordering::AcqRel);
                // Round-2 writer-class census: the re-fault warm-insert GROWS
                // the resident tier like install, so it engages the same
                // drain — a sustained read-refault regime (no installs) must
                // not balloon the resident set past the watermark. Called
                // AFTER the warm guard drops: the drain re-acquires the
                // capture read guard itself (non-reentrant lock).
                self.maybe_drain();
                return Some(served);
            }
            if self.resident().move_epoch.load(Ordering::Acquire) == epoch {
                return None; // stable double-miss — genuinely absent
            }
        }
    }

    /// Install (or overwrite) the `(tenant, kind, external_id) →
    /// internal_id` binding.
    ///
    /// Used on two paths, both of which must agree:
    /// - **Replay**: [`crate::wal::replay::ReplayExecutor`] re-installs
    ///   every `idempotency_bindings` entry from each committed
    ///   `CommitBundle` so the store recovers the durable set after a
    ///   restart.
    /// - **Live publish**: `arcgraph-mcp` installs the binding AFTER
    ///   `crud::commit` succeeds (mirroring the R1 HIGH-2 post-commit
    ///   publish discipline), so a commit-time failure never leaves a
    ///   binding for a rolled-back id.
    ///
    /// Idempotent under double-install / double-replay: re-installing
    /// the same `(tenant, kind, external_id, internal_id)` overwrites
    /// with an identical value (Lemma I2 parity with the MVCC chain).
    /// A binding is expected to be stable for a given key, so an
    /// overwrite with a *different* id would indicate upstream id reuse
    /// — which the v6 atomic fold prevents by construction.
    pub fn install(&self, tenant: TenantId, kind: u8, external_id: &str, internal_id: u64) {
        self.install_with_payload_hash(tenant, kind, external_id, internal_id, None);
    }

    /// Install (or overwrite) a live-process binding with a remembered
    /// payload hash.
    ///
    /// **#1404 M0.x.** A fresh binding is enrolled in the eviction FIFO with
    /// its resident-byte weight; a bounded store drains checkpoint-durable
    /// oldest bindings to spill if the resident set now exceeds the high
    /// watermark. An overwrite of a currently-spilled key clears the stale
    /// spill entry (the resident copy is now authoritative) so a later
    /// re-fault never resurrects the superseded id.
    pub fn install_with_payload_hash(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) {
        if self.physical.is_some() {
            if let Err(error) = self.try_install_with_payload_hash(
                tenant,
                kind,
                external_id,
                internal_id,
                payload_hash,
            ) {
                tracing::error!(%error, "M4 binding publish failed");
            }
            return;
        }
        // #1404 M0.x FIX-D — take the capture-exclusion READ guard for the
        // full install (concurrent with other installs; blocks ONLY during a
        // capture's WRITE-guarded count+stream, so the capture sees a
        // consistent binding set). Cheap: an uncontended parking_lot read.
        // Round-2 FIX-2: EXPLICITLY scoped so it is dropped BEFORE the
        // `maybe_drain` at the end — the drain now takes its own capture read
        // guard (the evict is a mover of the framed set too), and parking_lot
        // read() is non-reentrant (a recursive read with a queued writer would
        // self-deadlock).
        let _install = self.resident().capture_lock.read();
        let key = (tenant, kind, external_id.to_owned());
        // #1404 M0.x round-2 FIX-1 — snapshot any STALE spill image (its index
        // offset + the internal_id recorded IN it) BEFORE inserting the fresh
        // binding. The retire below is guarded on this exact offset, so it can
        // never delete a NEWER legitimate image written by a concurrent
        // eviction of the fresh binding (the pre-fix blind `offsets.remove`
        // could → fresh binding lost from BOTH tiers → `get()==None` →
        // duplicate on re-ingest).
        let stale_spill: Option<(u64, u64)> = self.resident().spill.as_ref().and_then(|spill| {
            let (off, old) = spill
                .lookup_forward(tenant, kind, external_id)
                .expect("idempotency spill read on install-overwrite failed")?;
            Some((off, old.internal_id))
        });
        // Drop the reverse entry of any prior binding at this key (resident OR
        // spilled) so `external_id_for` never resolves a superseded id — but
        // NOT for a same-id re-install (replay / idempotent re-publish): a
        // remove-then-reinsert of the SAME reverse key is a pure visibility
        // hazard; the insert below overwrites in place.
        let prior_internal: Option<u64> = self
            .resident()
            .forward
            .get(&key)
            .map(|old| old.value().binding.internal_id)
            .or(stale_spill.map(|(_, id)| id));
        if let Some(old_id) = prior_internal {
            if old_id != internal_id {
                self.resident().reverse.remove(&(tenant, kind, old_id));
            }
        }
        let binding = IdempotencyBinding {
            internal_id,
            payload_hash,
        };
        // Round-2 FIX-1 — insert the fresh resident binding FIRST, then retire
        // the stale spill image. The pre-fix order (remove spill entry, THEN
        // insert resident) opened a both-tiers-miss window in which a
        // concurrent `get()` of a continuously-live binding returned `None`.
        // A live install is NOT yet checkpoint-durable → not evict-eligible
        // until a checkpoint captures it (INV-DURABLE).
        self.insert_resident_warm(
            tenant,
            kind,
            external_id,
            binding,
            /* checkpointed = */ false,
        );
        self.resident()
            .reverse
            .insert((tenant, kind, internal_id), external_id.to_owned());
        // Retire the superseded spill image (BOTH indices, FIX-A), guarded on
        // the snapshotted offset. `move_epoch` is bumped AFTER the resident
        // insert and BEFORE the index removal (the seqlock contract the
        // two-tier readers rely on — see the `move_epoch` field docs).
        //
        // Round-3 FIX-5 — the retire interlock vs a CONCURRENT evictor:
        // - An image the evictor writes AFTER the entry snapshot (l. `stale_spill`)
        //   can never be tombstoned here: record offsets are monotone-unique
        //   (append-only data file; compaction rewrites only the index,
        //   preserving `record_off`), and `retire_forward_if` fires only when
        //   the NEWEST live image sits at exactly the snapshotted offset.
        // - An image the evictor wrote BEFORE the entry snapshot (the snapshot
        //   read the evictor's own fresh image) is retired ONLY on an
        //   id-CHANGING overwrite. For a SAME-id re-publish the retire is
        //   SKIPPED: the image maps the same `(key → internal_id)` truth (only
        //   a pre-durability `payload_hash` can lag, and WAL replay re-installs
        //   it on recovery), it may BE the evictor's authoritative sole copy of
        //   a just-evicted binding, and tombstoning it was the second half of
        //   the round-3 double-miss (FIX-5). Left live it is either shadowed by
        //   a newer re-spill (newest-wins lookup — a tombstoned/shadowed newest
        //   never resurfaces older nodes) or removed by `release`; a
        //   resident+spill-live key is an already-normal state (re-fault leaves
        //   one) that `binding_count`/`for_each_binding` dedup once.
        if let Some((off, old_id)) = stale_spill.filter(|(_, old_id)| *old_id != internal_id) {
            let spill = self
                .resident()
                .spill
                .as_ref()
                .expect("stale_spill implies a spill tier");
            self.resident().move_epoch.fetch_add(1, Ordering::AcqRel);
            spill
                .retire_forward_if(tenant, kind, external_id, off)
                .expect("idempotency spill forward retire on install-overwrite failed");
            spill
                .retire_reverse_if(tenant, kind, old_id, off)
                .expect("idempotency spill reverse retire on install-overwrite failed");
        }
        // #1404 M0.x — engage the drain if we crossed the high watermark.
        // No-op for the unbounded store (watermark = u64::MAX). Round-2
        // FIX-2: drop the install guard FIRST — the drain re-acquires the
        // capture read guard itself (non-reentrant lock; see above).
        drop(_install);
        self.maybe_drain();
    }

    /// Typed physical binding publish. Mainline v10 CRUD installs ride the
    /// record transaction; this entry point is for standalone owner writes
    /// and durability gates.
    pub fn try_install_with_payload_hash(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) -> Result<(), OwnerRowError> {
        let Some(owner) = self.physical.as_ref() else {
            self.install_with_payload_hash(tenant, kind, external_id, internal_id, payload_hash);
            return Ok(());
        };
        let class = binding_class(kind)?;
        let logical = BindingOwnerValue {
            kind,
            external_id: external_id.to_owned(),
            payload_hash,
            active: true,
        }
        .encode()?;
        let row = owner.encode_logical_row(tenant, class, internal_id, &logical)?;
        owner.commit_indexed_logical_row(tenant, row, str_hash_56(external_id))?;
        Ok(())
    }

    /// Insert `binding` into the resident forward map for
    /// `(tenant, kind, external_id)`, maintaining the resident-byte counter +
    /// the eviction FIFO. A fresh key adds one binding-weight to the counter
    /// and enqueues it; overwriting an existing resident key leaves the count
    /// unchanged (it is already enqueued). `checkpointed` seeds the
    /// INV-DURABLE gate — `true` on a spill re-fault (already durable), `false`
    /// on a live install.
    fn insert_resident_warm(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        binding: IdempotencyBinding,
        checkpointed: bool,
    ) {
        let key = (tenant, kind, external_id.to_owned());
        // Round-3 FIX-5 — every insert (fresh, overwrite, re-fault warm) gets
        // a store-unique install generation; `evict_one` keys its
        // compare-and-remove on it.
        let install_gen = self
            .resident()
            .install_gen_counter
            .fetch_add(1, Ordering::AcqRel);
        let resident = ResidentBinding::new(binding, install_gen);
        if checkpointed {
            resident.checkpointed.store(true, Ordering::Release);
        }
        let prev = self.resident().forward.insert(key.clone(), resident);
        if prev.is_none() {
            self.resident()
                .resident_bytes
                .fetch_add(IDEMPOTENCY_BINDING_WEIGHT_BYTES, Ordering::AcqRel);
            if self.resident().spill.is_some() {
                self.resident()
                    .evict_queue
                    .lock()
                    .expect("idempotency evict_queue mutex poisoned")
                    .push_back(key);
            }
        }
    }

    /// Round-3 FIX-5(e) — the RE-FAULT warm-insert: insert `binding` (a
    /// checkpoint-durable spill image) ONLY if the key is vacant. A blind
    /// overwrite here would clobber a FRESHER binding a racing same-key
    /// `install` just published (the spill image is by construction older
    /// than any concurrent install), silently regressing its
    /// `payload_hash` — and, marked durable, making the stale value
    /// immediately evictable so the regression sticks (fix5b gate: 35,689
    /// stale serves/run with the blind insert).
    ///
    /// Returns `Some((served, inserted))`: the binding the reader should
    /// serve — the given `binding` if our insert landed, else the fresher
    /// resident value that beat us — and whether our insert landed (the
    /// caller re-populates the reverse entry only then). Returns `None`
    /// (#1500 FIX-5(f), the vacant-ABA rejection) when the slot is vacant but
    /// its generation advanced past `expected_slot_gen`: an install+removal
    /// cycle (fresh install → fresh evict, or a release) completed between
    /// the caller's spill read and this vacant-check, so `binding` may be
    /// superseded — the caller must re-fault fresh instead of installing it.
    fn insert_resident_warm_if_vacant(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
        binding: IdempotencyBinding,
        expected_slot_gen: u64,
    ) -> Option<(IdempotencyBinding, bool)> {
        use dashmap::mapref::entry::Entry;
        let key = (tenant, kind, external_id.to_owned());
        let install_gen = self
            .resident()
            .install_gen_counter
            .fetch_add(1, Ordering::AcqRel);
        let (served, inserted) = match self.resident().forward.entry(key.clone()) {
            Entry::Occupied(e) => (e.get().binding, false),
            Entry::Vacant(v) => {
                // #1500 FIX-5(f) — re-load the slot generation UNDER the
                // vacant entry's shard lock: observing the slot vacant
                // synchronizes-with the removal that emptied it, and every
                // removal bumps BEFORE removing, so an ABA cycle since our
                // snapshot is visible here. Marking the stale image durable
                // below is what made the pre-fix regression stick — reject
                // instead.
                if self.resident().slot_generations.load(&key) != expected_slot_gen {
                    return None;
                }
                let resident = ResidentBinding::new(binding, install_gen);
                // The re-faulted binding IS checkpoint-durable (it was
                // evicted, which requires it) → immediately evict-eligible.
                resident.checkpointed.store(true, Ordering::Release);
                v.insert(resident);
                (binding, true)
            }
        };
        if inserted {
            self.resident()
                .resident_bytes
                .fetch_add(IDEMPOTENCY_BINDING_WEIGHT_BYTES, Ordering::AcqRel);
            if self.resident().spill.is_some() {
                self.resident()
                    .evict_queue
                    .lock()
                    .expect("idempotency evict_queue mutex poisoned")
                    .push_back(key);
            }
        }
        Some((served, inserted))
    }

    /// Release the `(tenant, kind, external_id)` binding.
    ///
    /// Used by delete replay and live delete publish so an external id
    /// becomes reusable once its owning node/relationship is tombstoned.
    /// Missing entries are harmless: release is idempotent under
    /// double-replay and tolerant of deletes for records that never had an
    /// external id.
    ///
    /// **#1404 M0.x.** Release removes BOTH tiers: the resident forward entry
    /// (decrementing the resident-byte counter) AND any spilled image, so a
    /// released external_id truly becomes free (a later re-fault cannot
    /// resurrect it). The reverse entry is dropped for the id it pointed at,
    /// whether that id was learned resident or via a spill fault.
    pub fn release(&self, tenant: TenantId, kind: u8, external_id: &str) {
        if self.physical.is_some() {
            if let Err(error) = self.try_release(tenant, kind, external_id) {
                tracing::error!(%error, "M4 binding release failed");
            }
            return;
        }
        // #1404 M0.x FIX-D — capture-exclusion read guard (a release also
        // mutates the section the capture frames; exclude it during a capture's
        // count+stream so the header can't disagree with the streamed set).
        let _release = self.resident().capture_lock.read();
        let key = (tenant, kind, external_id.to_owned());
        // #1500 FIX-5(f) — advance the key's slot generation BEFORE removing
        // either tier: a re-faulting reader holding a pre-release spill image
        // must reject its warm-insert rather than resurrect a released
        // binding. Unconditional (even for a spilled-only release): the
        // reader's hazard window is [spill read → vacant-check], and only the
        // bump makes a concurrent release visible inside it.
        self.resident().slot_generations.bump(&key);
        // Learn the internal_id so we can clear BOTH spill indices (forward +
        // reverse, FIX-A) and the resident reverse entry.
        let mut internal_id: Option<u64> = None;
        if let Some((_, resident)) = self.resident().forward.remove(&key) {
            internal_id = Some(resident.binding.internal_id);
            self.resident()
                .resident_bytes
                .fetch_sub(IDEMPOTENCY_BINDING_WEIGHT_BYTES, Ordering::AcqRel);
            // Round-3 FIX-4b — releasing a RESIDENT key strands its live FIFO
            // entry (Gone-or-duplicate from here on): hint the drain that the
            // queue holds one more reclaimable entry.
            if self.resident().spill.is_some() {
                self.resident()
                    .evict_queue_stale_hint
                    .fetch_add(1, Ordering::AcqRel);
            }
        } else if let Some(spill) = self.resident().spill.as_ref() {
            // Not resident but possibly spilled — fault the id from the spill.
            if let Ok(Some(old)) = spill.read_binding(tenant, kind, external_id) {
                internal_id = Some(old.internal_id);
            }
        }
        if let Some(id) = internal_id {
            self.resident().reverse.remove(&(tenant, kind, id));
            if let Some(spill) = self.resident().spill.as_ref() {
                spill
                    .remove_reverse(tenant, kind, id)
                    .expect("idempotency spill reverse remove on release failed");
            }
        }
        if let Some(spill) = self.resident().spill.as_ref() {
            spill
                .remove_forward(tenant, kind, external_id)
                .expect("idempotency spill forward remove on release failed");
        }
    }

    /// Typed standalone physical release.
    pub fn try_release(
        &self,
        tenant: TenantId,
        kind: u8,
        external_id: &str,
    ) -> Result<(), OwnerRowError> {
        let Some(owner) = self.physical.as_ref() else {
            self.release(tenant, kind, external_id);
            return Ok(());
        };
        let Some(binding) = self.try_get(tenant, kind, external_id)? else {
            return Ok(());
        };
        let row =
            owner.prepare_retired_binding_row(tenant, binding_class(kind)?, binding.internal_id)?;
        owner.commit_row(tenant, row)?;
        Ok(())
    }

    /// Resolve the external id bound to an internal id, if present.
    ///
    /// **#1404 M0.x FIX-A — reverse fault-in (NOT a benign miss).** The reverse
    /// map is now TIERED like the forward map: `evict_one` drops the resident
    /// reverse entry, and this method faults it back from the spill's reverse
    /// index on a resident miss. This is load-bearing on the DELETE path:
    /// `external_id_for` is called by `crud.rs:5234 stage_idempotency_release`
    /// to learn the external_id to release when a node/rel is tombstoned. A
    /// reverse MISS there would leave the external_id un-released → a future
    /// re-ingest of that external_id de-dupes to the now-DELETED internal id
    /// (a wrong-result). So a resident miss MUST consult the durable spill
    /// (never return `None` for an evicted-but-live binding). Faulting also
    /// re-warms the resident reverse entry.
    #[must_use]
    pub fn external_id_for(&self, tenant: TenantId, kind: u8, internal_id: u64) -> Option<String> {
        if self.physical.is_some() {
            return self
                .try_external_id_for(tenant, kind, internal_id)
                .unwrap_or_else(|error| {
                    tracing::error!(%error, "M4 binding reverse lookup failed closed");
                    None
                });
        }
        // #1404 M0.x round-2 FIX-1 — same seqlock-style two-tier read as
        // `resolve_binding` (the reverse maps have the same resident-then-
        // spill order and the same spill→resident movers).
        loop {
            let epoch = self.resident().move_epoch.load(Ordering::Acquire);
            if let Some(e) = self.resident().reverse.get(&(tenant, kind, internal_id)) {
                return Some(e.value().clone());
            }
            // Resident-reverse miss — fault from the spill reverse index
            // (bounded store only). An evicted-but-live binding is recoverable
            // here; a genuinely-released binding is absent from BOTH tiers →
            // `None`.
            let spill = self.resident().spill.as_ref()?;
            if let Some(spilled) = spill
                .read_binding_by_internal(tenant, kind, internal_id)
                .expect("idempotency spill reverse read failed")
            {
                // Re-warm the resident reverse entry so a repeated lookup is
                // hot. #1404 M0.x round-2 FIX-2 — the re-warm participates in
                // capture exclusion like install/release (same SAFETY
                // argument as `resolve_binding`: the capture thread never
                // calls this method, and no caller already holds
                // `capture_lock`, so the read() cannot self-deadlock).
                {
                    let _warm = self.resident().capture_lock.read();
                    self.resident()
                        .reverse
                        .insert((tenant, kind, internal_id), spilled.external_id.clone());
                }
                return Some(spilled.external_id);
            }
            if self.resident().move_epoch.load(Ordering::Acquire) == epoch {
                return None; // stable double-miss — genuinely absent
            }
        }
    }

    /// Typed direct-address reverse lookup. No forward index is touched, so a
    /// mapped address requires exactly one owner data-page fault.
    pub fn try_external_id_for(
        &self,
        tenant: TenantId,
        kind: u8,
        internal_id: u64,
    ) -> Result<Option<String>, OwnerRowError> {
        let Some(owner) = self.physical.as_ref() else {
            return Ok(self.external_id_for(tenant, kind, internal_id));
        };
        let Some(logical) = owner.read_logical(tenant, binding_class(kind)?, internal_id)? else {
            return Ok(None);
        };
        let value = BindingOwnerValue::decode(&logical)?;
        Ok((value.active && value.kind == kind).then_some(value.external_id))
    }

    /// Total number of LOGICAL bindings across all tenants + kinds =
    /// resident + evicted-to-spill. For an unbounded store this equals the
    /// resident forward count. Introspection / metrics only.
    #[must_use]
    pub fn total_len(&self) -> usize {
        if let Some(owner) = self.physical.as_ref() {
            return owner
                .tenants()
                .flat_map(|tenant| {
                    [OwnerRowClass::NodeBinding, OwnerRowClass::RelBinding]
                        .into_iter()
                        .map(move |class| (tenant, class))
                })
                .filter_map(|(tenant, class)| owner.candidate_count(tenant, class).ok())
                .sum::<u64>() as usize;
        }
        match &self.resident().spill {
            None => self.resident().forward.len(),
            Some(spill) => {
                // Union of resident keys + spilled keys (a resident key may
                // also carry a stale spill entry after re-fault warming; count
                // it once).
                let resident_only = self
                    .resident()
                    .forward
                    .iter()
                    .filter(|e| {
                        let (t, k, ext) = e.key();
                        !spill
                            .contains_forward(*t, *k, ext)
                            .expect("idempotency spill contains read failed")
                    })
                    .count();
                resident_only
                    + usize::try_from(spill.live_forward_len()).expect("live count fits usize")
            }
        }
    }

    /// Whether the store holds no bindings at all (resident or spilled).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        if self.physical.is_some() {
            return self.total_len() == 0;
        }
        self.resident().forward.is_empty()
            && self
                .resident()
                .spill
                .as_ref()
                .is_none_or(|s| s.live_forward_len() == 0)
    }

    /// Number of LOGICAL bindings for `tenant` across both kinds
    /// (resident + spilled). O(N) filtered scan — test / introspection only.
    #[doc(hidden)]
    #[must_use]
    pub fn len_for_tenant(&self, tenant: TenantId) -> usize {
        if let Some(owner) = self.physical.as_ref() {
            return [OwnerRowClass::NodeBinding, OwnerRowClass::RelBinding]
                .into_iter()
                .filter_map(|class| owner.candidate_count(tenant, class).ok())
                .sum::<u64>() as usize;
        }
        match &self.resident().spill {
            None => self
                .resident()
                .forward
                .iter()
                .filter(|e| e.key().0 == tenant)
                .count(),
            Some(spill) => {
                let resident_only = self
                    .resident()
                    .forward
                    .iter()
                    .filter(|e| {
                        let (t, k, ext) = e.key();
                        *t == tenant
                            && !spill
                                .contains_forward(*t, *k, ext)
                                .expect("idempotency spill contains read failed")
                    })
                    .count();
                let spilled = spill.count_forward_for_tenant(tenant);
                resident_only + spilled
            }
        }
    }

    /// The number of LOGICAL bindings the capture will emit = resident (not
    /// also-spilled) + spilled. This is the count [`Self::for_each_binding`]
    /// enumerates; the ADR-229 producer writes it as the section header BEFORE
    /// streaming the bindings (so the wire count precedes the records, exactly
    /// as the pre-M0.x `iter_all().len()` did). Cheap: two DashMap length
    /// reads + one filtered scan of the (bounded) resident set.
    ///
    /// This MUST equal the number of callbacks [`Self::for_each_binding`]
    /// fires, or the snapshot's declared count and streamed records diverge.
    /// The two use the SAME resident-vs-spilled dedup rule (a resident key that
    /// is ALSO spilled is counted / emitted once, from the resident walk).
    #[must_use]
    pub fn binding_count(&self) -> u64 {
        if self.physical.is_some() {
            return 0;
        }
        match &self.resident().spill {
            None => self.resident().forward.len() as u64,
            Some(spill) => {
                // Resident keys that are NOT also spilled + all spilled keys.
                // (A resident key that is also spilled — a stale spill entry
                // after a re-fault — is emitted once from the resident walk and
                // skipped in the spill walk, so it is counted once here.)
                let resident_only = self
                    .resident()
                    .forward
                    .iter()
                    .filter(|e| {
                        let (t, k, ext) = e.key();
                        !spill
                            .contains_forward(*t, *k, ext)
                            .expect("idempotency spill contains read failed")
                    })
                    .count() as u64;
                resident_only + spill.live_forward_len()
            }
        }
    }

    /// ADR-229 checkpoint producer — STREAM every logical binding as
    /// `(tenant, kind, external_id, internal_id, payload_hash)` through `f`,
    /// ONE at a time, NEVER building a whole-`Vec`. Restore feeds each back
    /// through [`Self::install_with_payload_hash`] (the same WAL-replay entry
    /// point). Iteration order is arbitrary. Stops + returns on the first `f`
    /// error (a sink write failure).
    ///
    /// **#1404 M0.x — the freeze-capture bound + INV-DURABLE gate.** This runs
    /// UNDER `checkpoint_freeze` (`producer.rs:132`). It is the streaming twin
    /// of the (now `#[cfg(test)]`-only) whole-`Vec` `iter_all`, mirroring the
    /// M0.5 snapshot streaming: the RESIDENT forward map is walked emitting each
    /// binding then dropping it, then the durable spill index is walked reading
    /// each spilled record → emit → DROP before the next (≤ ONE record resident
    /// at a time, NOT a `Vec`). This is what actually bounds the freeze-capture
    /// RSS: without it, the capture re-collects the entire ~9M-binding set into
    /// one `Vec` under the freeze (the whole-in-RAM sibling M0.x exists to kill,
    /// exactly like the M0.5 evicted-supplement whole-`Vec` REJECT).
    ///
    /// Capturing a resident binding here MARKS it `checkpointed` (its durable
    /// image is about to land in the snapshot), which makes it evict-eligible —
    /// the INV-DURABLE gate: a binding is evictable ONLY after a completed
    /// checkpoint captured it.
    ///
    /// The number of callbacks fired equals [`Self::binding_count`] (same
    /// resident-vs-spilled dedup), so the producer can write the count header
    /// first and then stream exactly that many records.
    ///
    /// **#1404 M0.x FIX-D — returns the ACTUAL streamed count** so the producer
    /// can hard-check it against the header (`binding_count`) it already wrote:
    /// a mismatch aborts the checkpoint (never a corrupt-but-Ok snapshot).
    /// Under the capture WRITE guard ([`Self::capture_guard`]) the two are
    /// deterministically equal (no install can interleave), so the check is
    /// defense-in-depth; without the guard (e.g. a raw test call) it still
    /// counts truthfully.
    pub fn for_each_binding<F, E>(&self, mut f: F) -> Result<u64, E>
    where
        F: FnMut(TenantId, u8, &str, u64, Option<u64>) -> Result<(), E>,
    {
        if self.physical.is_some() {
            return Ok(0);
        }
        // #1404 M0.x — track the peak records the capture holds SIMULTANEOUSLY,
        // so the gatex O(1)-in-N oracle can prove the freeze-capture is bounded.
        // The streaming path holds exactly ONE record in-flight (read → emit →
        // drop), so the peak stays 1 regardless of N. A reverted whole-`Vec`
        // capture (pre-collect all then replay) would push the peak to N.
        let mut in_flight: u64 = 0;
        let mut peak: u64 = 0;
        let mut streamed: u64 = 0;
        // Resident bindings: mark checkpoint-durable → evict-eligible
        // (INV-DURABLE gate), then emit. Release ordering so the eviction
        // path's Acquire load sees it. Each binding is emitted from the
        // borrowed DashMap entry — no owned `String` is retained across the
        // callback (the key is borrowed as `&str`).
        //
        // M5-D3 FIX 4 (#1518 skeptic review) — sorted `(tenant, kind,
        // external_id)` capture order. `DashMap` iteration order is
        // nondeterministic, and these bindings land byte-for-byte in
        // checkpoint metadata (same INV-M5.24 class as the blob page-image
        // capture, see `blob.rs::sorted_resident_keys`). Only the KEYS are
        // collected into a `Vec` — the streaming discipline above still
        // holds for the actual binding payloads.
        let forward = &self.resident().forward;
        let mut keys: Vec<(TenantId, u8, String)> =
            forward.iter().map(|e| e.key().clone()).collect();
        keys.sort_unstable();
        for key in &keys {
            let Some(entry) = forward.get(key) else {
                continue;
            };
            let (tenant, kind, ext) = key;
            entry.value().checkpointed.store(true, Ordering::Release);
            let b = entry.value().binding;
            in_flight += 1;
            peak = peak.max(in_flight);
            f(*tenant, *kind, ext, b.internal_id, b.payload_hash)?;
            in_flight -= 1;
            streamed += 1;
        }
        // Spilled bindings not currently resident: read their durable image
        // ONE at a time (bounded working set) → emit → DROP before the next, so
        // the snapshot is complete WITHOUT holding all spilled records resident.
        // Only the bounded store spills; the unbounded store's `spill` is
        // `None` → no extra work.
        if let Some(spill) = &self.resident().spill {
            spill.for_each_live(|tenant, kind, sb| {
                if self
                    .resident()
                    .forward
                    .contains_key(&(tenant, kind, sb.external_id.clone()))
                {
                    // Already emitted from the resident walk above (stale spill
                    // entry after a re-fault) — skip to keep the set a set.
                    return Ok(());
                }
                in_flight += 1;
                peak = peak.max(in_flight);
                f(
                    tenant,
                    kind,
                    &sb.external_id,
                    sb.internal_id,
                    sb.payload_hash,
                )?;
                in_flight -= 1;
                streamed += 1;
                // `sb` (the ONE record) drops here before the next callback.
                Ok(())
            })?;
        }
        self.resident()
            .capture_peak_in_flight
            .store(peak, Ordering::Release);
        Ok(streamed)
    }

    /// #1404 M0.x FIX-D — acquire the capture WRITE guard. The ADR-229 producer
    /// holds this across BOTH `binding_count()` (the count header) AND
    /// `for_each_binding()` (the stream), so no concurrent `install`/`release`
    /// (which take the READ guard) can interleave and skew header≠stream. The
    /// hot READ path is untouched. Returns a RAII guard; drop it after the
    /// section is streamed. For the unbounded store it is equally valid (an
    /// install can race the two-pass there too).
    pub fn capture_guard(&self) -> Option<parking_lot::RwLockWriteGuard<'_, ()>> {
        self.resident
            .as_deref()
            .map(|resident| resident.capture_lock.write())
    }

    /// #1404 M0.x — the CAPTURE peak-in-flight record count from the last
    /// [`Self::for_each_binding`] pass (the max records the capture held
    /// resident at once). O(1) in N for the streaming path (≤1). The gatex
    /// oracle asserts this stays flat across a 16× size difference. Test /
    /// observability.
    #[doc(hidden)]
    #[must_use]
    pub fn capture_peak_in_flight(&self) -> u64 {
        self.resident.as_deref().map_or(0, |resident| {
            resident.capture_peak_in_flight.load(Ordering::Acquire)
        })
    }

    /// #1404 M0.x — the whole-`Vec` capture, retained as a `#[cfg(test)]`
    /// ORACLE ONLY (mirroring the M0.5 `append_evicted_supplement` whole-`Vec`
    /// oracle). The PRODUCTION capture is [`Self::for_each_binding`], which
    /// streams one record at a time (never a `Vec`). This oracle exists so the
    /// streaming-vs-whole-`Vec` byte-identity differential + the older tests
    /// can compare against a materialized set; it is NOT on any freeze-critical
    /// path.
    #[cfg(test)]
    #[must_use]
    pub fn iter_all(&self) -> Vec<(TenantId, u8, String, u64, Option<u64>)> {
        let mut out = Vec::new();
        let _ = self.for_each_binding::<_, std::convert::Infallible>(|t, k, ext, id, h| {
            out.push((t, k, ext.to_owned(), id, h));
            Ok(())
        });
        out
    }

    // ── #1404 M0.x — resident-tier drain (evict checkpoint-durable bindings) ──

    /// #1404 M0.x round-3 FIX-4 — the evict FIFO's entry cap.
    ///
    /// Budget: at the 256 MiB default cap this is ~3.4M entries (2 × the
    /// ~1.7M-binding resident cap); a queue entry is one key clone (tens of
    /// bytes), so worst-case FIFO RSS stays the same order as the resident
    /// tier it mirrors — instead of O(total-ever-installed) under
    /// ingest-then-delete churn (the pre-fix leak: one stale entry per
    /// released binding, forever).
    fn evict_queue_cap(&self) -> usize {
        let resident_cap =
            self.resident().config.high_watermark_bytes / IDEMPOTENCY_BINDING_WEIGHT_BYTES;
        usize::try_from(resident_cap.saturating_mul(IDEMPOTENCY_EVICT_QUEUE_CAP_FACTOR))
            .unwrap_or(usize::MAX)
            .max(1)
    }

    /// Engage the drain if the resident set is over the high watermark OR the
    /// evict FIFO is over its entry cap (round-3 FIX-4 — `release` leaves its
    /// FIFO entry behind as a cheap `Gone`, so an ingest-then-delete workload
    /// whose live set stays under the watermark builds a Gone backlog with
    /// ZERO resident-byte pressure; pre-fix it never drained → one leaked
    /// entry per released binding). No-op for the unbounded store (`spill =
    /// None` → watermark `u64::MAX`, and its FIFO is never pushed). Called on
    /// the install path — the resident check stays a load + branch; the
    /// FIFO-cap check costs one uncontended mutex lock ONLY when the resident
    /// check says no (short-circuit `||`), noise next to install's map insert
    /// + key clone. The drain loop itself runs only under pressure.
    fn maybe_drain(&self) {
        if self.resident().spill.is_none() {
            return;
        }
        if self.resident().resident_bytes.load(Ordering::Acquire)
            > self.resident().config.high_watermark_bytes
            || self
                .resident()
                .evict_queue
                .lock()
                .expect("idempotency evict_queue mutex poisoned")
                .len()
                > self.evict_queue_cap()
        {
            self.drain_to_low_watermark();
        }
    }

    /// Evict checkpoint-durable bindings (oldest-first) to the spill tier
    /// until the resident set is at/below the low watermark (hysteresis) or no
    /// more evict-eligible bindings remain.
    ///
    /// **Throttle-not-OOM (mirrors the M0 blob drain):** runs INLINE on the
    /// writer's `install` call. When the drain cannot free enough bindings
    /// (all resident bindings pending checkpoint durability), it returns with
    /// the resident set still bounded by the arrival rate; ingest *slows*
    /// (back-pressure) rather than racing ahead and OOMing.
    ///
    /// **Cost — O(evicted-this-pass), not O(resident-count):** the evict queue
    /// is FIFO oldest-first. A binding can only become checkpoint-durable
    /// AFTER it is enqueued, so the oldest queued binding is always the first
    /// to become durable — durability of the queue is monotone front-to-back.
    /// If the FRONT is not yet durable (`NotDurable`), nothing behind it is
    /// either; restore it to the front and `break`. Identical scheduling
    /// argument to [`crate::blob::BlobStore::drain_to_low_watermark`].
    ///
    /// **Round-3 FIX-4b — the break applies to the RESIDENT axis only.** The
    /// monotone argument says nothing newer is *durable*; it does NOT say
    /// nothing newer is *reclaimable*: a released key's stale FIFO entry is a
    /// free `Gone` pop, and same-key install→release→install churn with no
    /// intervening checkpoint parks a permanently-`NotDurable` entry at the
    /// front of a duplicate backlog that grows O(cycles) — pre-4b the break
    /// fired on the first pop and reclaimed NOTHING (the fix4 cap trigger
    /// engaged every install, futilely). While the FIFO is over its entry
    /// cap AND the stale hint says reclaimable backlog exists
    /// (`evict_queue_stale_hint` — releases of resident keys strand exactly
    /// one entry each), a `NotDurable` pop is retained aside (restored to
    /// the front after the pass, order preserved; same-key duplicates
    /// dropped) and the scan continues. Cost stays honest twice over: a
    /// zero hint keeps the one-probe break (the drain-cost guard's
    /// 1000-distinct-installs-no-checkpoint regime — an irreducible queue
    /// must NOT be rescanned per install), and the retained prefix shrinks
    /// the LIVE queue toward the cap, bounding a productive scan by the
    /// over-cap excess plus entries actually reclaimed.
    fn drain_to_low_watermark(&self) {
        let Some(spill) = self.resident().spill.as_ref() else {
            return;
        };
        // #1404 M0.x round-2 FIX-2 — the EVICT is a mover of the framed set
        // (forward.remove + spill-index insert): interleaved with the producer's
        // count+stream it double-counts the moving binding between
        // `binding_count`'s two reads (header=N+1 vs streamed=N → CountSkew →
        // checkpoint abort → WAL frontier stall). Every drain caller (install
        // — guard dropped there first —, the read-refault warm path, the
        // bootstrap post-checkpoint hook, tests) participates in capture
        // exclusion via THIS read guard. Held for the whole pass: the pass is
        // O(evicted-this-pass) bounded, and a capture blocks it rather than
        // interleaving it.
        //
        // SAFETY(no-self-deadlock): non-reentrant parking_lot read — no drain
        // caller holds `capture_lock` at this point (install/resolve_binding
        // drop their guards before calling), and the capture write-holder
        // never drains.
        let _drain = self.resident().capture_lock.read();
        // Round-3 FIX-4 — the pass continues while EITHER pressure axis is
        // high: resident bytes over the low watermark (round-2 hysteresis) OR
        // the FIFO over its entry cap. The latter reclaims the cheap `Gone`
        // backlog `release` leaves behind even when the resident tier is
        // already at rest (an ingest-then-delete workload produces ONLY that
        // axis). Reclaiming continuously at the cap is also what keeps the
        // O(evicted-this-pass) bound above honest — pre-fix the backlog
        // accumulated unboundedly and the next resident-pressure pass walked
        // ALL of it (an O(total-released) stall on one writer's install).
        let queue_cap = self.evict_queue_cap();
        let mut budget = self
            .resident()
            .evict_queue
            .lock()
            .expect("idempotency evict_queue mutex poisoned")
            .len();
        // Round-3 FIX-4b — `NotDurable` entries popped while the FIFO is over
        // its entry cap are RETAINED (held aside, restored to the FRONT after
        // the pass in original order) instead of breaking the pass, so the
        // scan reaches the reclaimable backlog BEHIND a not-yet-durable
        // front. `retained_keys` dedups within the pass: a second `NotDurable`
        // pop of an already-retained key is a stale DUPLICATE entry (same-key
        // install→release→install churn leaves one per re-install) and is
        // DROPPED — the retained sibling keeps the key represented, the same
        // argument `evict_one`'s in-flight-claim skip rests on.
        let mut retained: Vec<(TenantId, u8, String)> = Vec::new();
        let mut retained_keys: HashSet<(TenantId, u8, String)> = HashSet::new();
        while budget > 0
            && (self.resident().resident_bytes.load(Ordering::Acquire)
                > self.resident().config.low_watermark_bytes
                || self
                    .resident()
                    .evict_queue
                    .lock()
                    .expect("idempotency evict_queue mutex poisoned")
                    .len()
                    > queue_cap)
        {
            budget -= 1;
            self.resident()
                .drain_probe_count
                .fetch_add(1, Ordering::Relaxed);
            let Some(key) = self
                .resident()
                .evict_queue
                .lock()
                .expect("idempotency evict_queue mutex poisoned")
                .pop_front()
            else {
                break;
            };
            match self.evict_one(spill, key.clone()) {
                EvictOutcome::Evicted => {}
                EvictOutcome::Gone => {
                    // Reclaimed a stale (released/duplicate) entry — burn one
                    // unit of the stale hint. The evict-rollback `Gone` (the
                    // one non-stale `Gone`, its live entry re-enqueued)
                    // compensates this with its own +1.
                    let _ = self.resident().evict_queue_stale_hint.fetch_update(
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        |v| Some(v.saturating_sub(1)),
                    );
                }
                EvictOutcome::NotDurable => {
                    if retained_keys.contains(&key) {
                        // Stale duplicate of a key retained THIS pass — drop
                        // it (the retained entry represents the key; keeping
                        // both is what let the same-key-churn backlog grow
                        // O(cycles) past the break below).
                        let _ = self.resident().evict_queue_stale_hint.fetch_update(
                            Ordering::AcqRel,
                            Ordering::Acquire,
                            |v| Some(v.saturating_sub(1)),
                        );
                        continue;
                    }
                    if self
                        .resident()
                        .evict_queue_stale_hint
                        .load(Ordering::Acquire)
                        > 0
                        && self
                            .resident()
                            .evict_queue
                            .lock()
                            .expect("idempotency evict_queue mutex poisoned")
                            .len()
                            > queue_cap
                    {
                        // FIFO-cap axis still active AND the stale hint says
                        // reclaimable (`Gone`/duplicate) backlog exists behind
                        // this front — a released key's entry must be
                        // reclaimable REGARDLESS of a not-yet-durable entry
                        // ahead of it (FIX-4b invariant). Retain + continue.
                        // Both conjuncts bound the scan: a zero hint (nothing
                        // was ever released — the drain-cost guard's 1000
                        // distinct-install regime) breaks below on the FIRST
                        // probe, and the LIVE length (retained entries
                        // excluded) shrinking to the cap self-limits an
                        // irreducible all-`NotDurable` prefix to the over-cap
                        // excess.
                        retained_keys.insert(key.clone());
                        retained.push(key);
                        continue;
                    }
                    // Resident-pressure axis only → restore to the FRONT
                    // (still the oldest, retried first next drain) and BREAK
                    // the pass (FIFO-oldest-first ⟹ nothing newer is durable
                    // either) — the round-3 FIX-4 O(evicted-this-pass) bound,
                    // unchanged on this axis. INV-DURABLE holds: the binding
                    // stays resident, un-evicted; the next install retries
                    // after a later checkpoint marks it durable.
                    self.resident()
                        .evict_queue
                        .lock()
                        .expect("idempotency evict_queue mutex poisoned")
                        .push_front(key);
                    break;
                }
            }
        }
        // Restore retained `NotDurable` entries to the FRONT in original
        // relative order (they were popped oldest-first; reverse push_front
        // re-fronts the oldest). They precede any break-arm restore + the
        // unpopped remainder — FIFO install order is preserved.
        if !retained.is_empty() {
            let mut queue = self
                .resident()
                .evict_queue
                .lock()
                .expect("idempotency evict_queue mutex poisoned");
            for key in retained.into_iter().rev() {
                queue.push_front(key);
            }
        }
    }

    /// Attempt to evict one resident binding to spill. Returns the outcome so
    /// the drain loop can re-queue a not-yet-durable binding.
    fn evict_one(&self, spill: &IdempotencySpill, key: (TenantId, u8, String)) -> EvictOutcome {
        // Round-3 FIX-5(c) — claim the key's transition. A sibling in-flight
        // evictor (duplicate FIFO entries) would otherwise race us
        // write-vs-rollback on the same key and its rollback tombstone would
        // SHADOW our live image (see the `evict_inflight` field docs). The
        // sibling's queue entry keeps the key represented, so skipping here
        // loses nothing.
        if self
            .resident()
            .evict_inflight
            .insert(key.clone(), ())
            .is_some()
        {
            return EvictOutcome::Gone;
        }
        let outcome = self.evict_one_claimed(spill, &key);
        self.resident().evict_inflight.remove(&key);
        outcome
    }

    /// [`Self::evict_one`] body, run while holding the key's
    /// `evict_inflight` claim.
    fn evict_one_claimed(
        &self,
        spill: &IdempotencySpill,
        key: &(TenantId, u8, String),
    ) -> EvictOutcome {
        // Snapshot the resident entry WITHOUT removing it — we only remove
        // after the durable spill write succeeds (evict-after-durable).
        // Round-3 FIX-5: the snapshot includes the INSTALL GENERATION so the
        // compare-and-remove below can never mistake a same-`internal_id`
        // fresh re-publish for this exact snapshot.
        let (binding, checkpointed, install_gen) = match self.resident().forward.get(key) {
            Some(e) => (
                e.value().binding,
                e.value().checkpointed.load(Ordering::Acquire),
                e.value().install_gen,
            ),
            None => return EvictOutcome::Gone, // already removed (release/overwrite)
        };
        // INV-DURABLE gate: only evict a binding a completed checkpoint has
        // captured. `iter_all` sets this under the freeze.
        if !checkpointed {
            return EvictOutcome::NotDurable;
        }
        let (tenant, kind, ext) = key;
        // Write the durable, QUERYABLE spill image FIRST (BOTH forward + reverse
        // indices, FIX-A), then drop the resident copies (evict-then-lookup is
        // safe: `resolve_binding`/`external_id_for` fault from spill, which is
        // now populated). Idempotent re-spill just refreshes the index offsets.
        let spilled_off = spill
            .write_binding(
                *tenant,
                *kind,
                ext,
                binding.internal_id,
                binding.payload_hash,
            )
            .expect("idempotency spill write on eviction failed");
        // #1404 M0.x round-2 FIX-1 (skeptic 1 — silent data loss): remove from
        // resident AFTER the spill image is durable, and ONLY if the resident
        // value is still the one we snapshotted+spilled. A blind `remove(&key)`
        // races a concurrent same-key `install` that replaced the value AFTER
        // our snapshot: install cleared the spill index for the key (the fresh
        // binding supersedes it), so the blind remove drops the FRESH binding
        // from BOTH tiers → `get()==None` right after its own successful
        // install → duplicate on re-ingest (reproduced: 1/1/3 None-losses).
        //
        // Round-3 FIX-5 — the compare-and-remove keys on `(internal_id,
        // install_gen)`. An `internal_id`-only compare is NOT unique per
        // binding: a same-id fresh re-publish (WAL replay / idempotent
        // re-publish, see `install`'s docs) carries the SAME id as our stale
        // snapshot, so the pre-fix remove dropped the FRESH binding while the
        // overwriter's install-entry retire tombstoned OUR freshly-written
        // spill image → both tiers absent, `move_epoch` stable → a stable
        // `get()==None` for a continuously-live binding (round-3 skeptic 1:
        // 35 misses / 23 stable after 50 retries). The generation is bumped
        // on EVERY insert, so a match proves the resident value is the exact
        // snapshot we just spilled.
        // #1500 FIX-5(f) — advance the key's slot generation BEFORE the
        // compare-and-remove: a re-faulting reader that later observes the
        // slot VACANT must see this bump and reject the warm-insert of the
        // spill image it read BEFORE this eviction cycle (the vacant-ABA:
        // vacant → fresh install → this evict → vacant would otherwise let
        // the reader install its stale image as checkpoint-durable). Bumping
        // on the failed-compare path too is a benign spurious re-fault.
        self.resident().slot_generations.bump(key);
        let removed = self.resident().forward.remove_if(key, |_, v| {
            v.binding.internal_id == binding.internal_id && v.install_gen == install_gen
        });
        if removed.is_some() {
            self.resident()
                .resident_bytes
                .fetch_sub(IDEMPOTENCY_BINDING_WEIGHT_BYTES, Ordering::AcqRel);
            self.resident().evicted_count.fetch_add(1, Ordering::AcqRel);
            // #1404 M0.x FIX-A — ALSO drop the resident REVERSE entry (this is
            // the 6th OOM sibling the ultracode found: pre-fix `evict_one` left
            // reverse resident → O(N) growth). It is reconstructible from the
            // spill reverse index (`read_binding_by_internal`), so
            // `external_id_for` faults it back in on the delete path. Round-2
            // FIX-1(c): guarded — dropped ONLY on the successful-evict branch
            // and ONLY if it still maps this internal_id to OUR external_id
            // (never a concurrent installer's fresh reverse entry).
            self.resident()
                .reverse
                .remove_if(&(*tenant, *kind, binding.internal_id), |_, v| v == ext);
            EvictOutcome::Evicted
        } else {
            // Round-2 FIX-1(b): a concurrent `install` (or `release`) changed
            // the resident value under us between the snapshot and the
            // compare-and-remove. The spill image we JUST wrote is a STALE
            // superseded copy — roll it back so a later resident-miss re-fault
            // can never resurrect the old id (`remove_if` alone leaves this
            // stale-id residual, verdict §17). OFFSET-guarded: remove the index
            // entries only if they still point at OUR record, never a newer
            // legitimate spill image written by a subsequent eviction.
            //
            // Round-3 FIX-5(b) — this rollback is a spill-visibility-NEGATIVE
            // mover: the tombstone flips the key's spill verdict live→dead
            // (the tombstoned newest node shadows every older node), so a
            // two-tier reader whose forward check preceded a concurrent
            // re-install and whose spill check lands after this tombstone
            // would double-miss a continuously-live binding. Bump `move_epoch`
            // BEFORE the tombstone — exactly the seqlock contract every other
            // hiding mover follows — so that reader retries and finds the
            // fresh resident insert (round-3 skeptic 1's transient-miss leg;
            // reproduced: 5-10 reader misses per gate run with the bump
            // absent, epoch pinned at 0 the whole run).
            self.resident().move_epoch.fetch_add(1, Ordering::AcqRel);
            spill
                .retire_forward_if(*tenant, *kind, ext, spilled_off)
                .expect("idempotency spill forward retire on evict-rollback failed");
            spill
                .retire_reverse_if(*tenant, *kind, binding.internal_id, spilled_off)
                .expect("idempotency spill reverse retire on evict-rollback failed");
            // The key was popped off the eviction FIFO by this drain pass, and
            // a concurrent OVERWRITE-install does not re-enqueue (it sees the
            // key already resident). Without a re-enqueue the still-resident
            // fresh binding would be un-evictable forever — a permanent
            // resident leak. Re-enqueue iff still resident (a `release` race
            // leaves nothing to evict; a benign duplicate queue entry from an
            // adjacent re-install race just resolves as `Gone` later).
            if self.resident().forward.contains_key(key) {
                self.resident()
                    .evict_queue
                    .lock()
                    .expect("idempotency evict_queue mutex poisoned")
                    .push_back(key.clone());
                // Round-3 FIX-4b — this `Gone` consumed the key's LIVE entry
                // (not a stale one) and re-enqueued it; compensate the drain
                // loop's per-`Gone` stale-hint decrement so the hint cannot
                // UNDERCOUNT (an undercount would break the drain past a
                // `NotDurable` front with reclaimable backlog still behind
                // it — the FIX-4b leak, reintroduced).
                self.resident()
                    .evict_queue_stale_hint
                    .fetch_add(1, Ordering::AcqRel);
            }
            EvictOutcome::Gone
        }
    }
}

fn binding_class(kind: u8) -> Result<OwnerRowClass, OwnerRowError> {
    match kind {
        0 => Ok(OwnerRowClass::NodeBinding),
        1 => Ok(OwnerRowClass::RelBinding),
        other => Err(OwnerRowError::InvalidEnvelope(format!(
            "unsupported binding kind {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODE: u8 = 0;
    const REL: u8 = 1;

    #[test]
    fn get_miss_on_empty_store() {
        let store = IdempotencyStore::new();
        assert_eq!(store.get(TenantId::DEFAULT, NODE, "x"), None);
        assert!(store.is_empty());
        assert_eq!(store.total_len(), 0);
    }

    #[test]
    fn install_then_get_roundtrips() {
        let store = IdempotencyStore::new();
        store.install(TenantId::DEFAULT, NODE, "alice", 42);
        assert_eq!(
            store.get(TenantId::DEFAULT, NODE, "alice"),
            Some(IdempotencyBinding {
                internal_id: 42,
                payload_hash: None,
            })
        );
        assert_eq!(store.total_len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn install_with_payload_hash_then_get_roundtrips_hash() {
        let store = IdempotencyStore::new();
        store.install_with_payload_hash(TenantId::DEFAULT, NODE, "alice", 42, Some(99));
        let binding = store
            .get(TenantId::DEFAULT, NODE, "alice")
            .expect("binding");
        assert_eq!(binding.internal_id, 42);
        assert_eq!(binding.payload_hash, Some(99));
        assert_ne!(binding.payload_hash, Some(100));
    }

    #[test]
    fn release_removes_installed_binding() {
        let store = IdempotencyStore::new();
        store.install(TenantId::DEFAULT, NODE, "alice", 42);
        assert!(store.get(TenantId::DEFAULT, NODE, "alice").is_some());

        store.release(TenantId::DEFAULT, NODE, "alice");

        assert_eq!(store.get(TenantId::DEFAULT, NODE, "alice"), None);
        assert!(store.is_empty());
    }

    #[test]
    fn payload_hash_mismatch_is_observable_to_callers() {
        let store = IdempotencyStore::new();
        store.install_with_payload_hash(TenantId::DEFAULT, NODE, "doc1", 42, Some(1_001));
        let binding = store.get(TenantId::DEFAULT, NODE, "doc1").expect("binding");
        let incoming_hash = 9_999;
        assert_ne!(binding.payload_hash, Some(incoming_hash));
        assert_eq!(binding.internal_id, 42);
    }

    #[test]
    fn kind_namespaces_are_disjoint() {
        // Same (tenant, external_id) under two kinds binds independently:
        // a re-submitted node never resolves to a rel's id.
        let store = IdempotencyStore::new();
        store.install(TenantId::DEFAULT, NODE, "shared", 1);
        store.install(TenantId::DEFAULT, REL, "shared", 2);
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, "shared")
                .map(|b| b.internal_id),
            Some(1)
        );
        assert_eq!(
            store
                .get(TenantId::DEFAULT, REL, "shared")
                .map(|b| b.internal_id),
            Some(2)
        );
    }

    #[test]
    fn tenants_are_isolated() {
        // Cross-tenant isolation is a hard invariant: the same
        // external_id under two tenants binds to different ids and a
        // lookup is tenant-scoped.
        let store = IdempotencyStore::new();
        let t1 = TenantId::new(100);
        let t2 = TenantId::new(101);
        store.install(t1, NODE, "dup", 7);
        store.install(t2, NODE, "dup", 9);
        assert_eq!(store.get(t1, NODE, "dup").map(|b| b.internal_id), Some(7));
        assert_eq!(store.get(t2, NODE, "dup").map(|b| b.internal_id), Some(9));
        // A tenant with no binding for the key sees None.
        assert_eq!(store.get(TenantId::new(102), NODE, "dup"), None);
    }

    #[test]
    fn install_is_idempotent_under_double_replay() {
        // Re-installing the same binding (the double-replay / Lemma I2
        // case) is a no-op overwrite — same value, count unchanged.
        let store = IdempotencyStore::new();
        store.install(TenantId::DEFAULT, NODE, "k", 5);
        store.install(TenantId::DEFAULT, NODE, "k", 5);
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, "k")
                .map(|b| b.internal_id),
            Some(5)
        );
        assert_eq!(store.total_len(), 1);
    }

    #[test]
    fn len_for_tenant_counts_only_that_tenant() {
        let store = IdempotencyStore::new();
        let t1 = TenantId::new(1);
        let t2 = TenantId::new(2);
        store.install(t1, NODE, "a", 1);
        store.install(t1, REL, "b", 2);
        store.install(t2, NODE, "a", 3);
        assert_eq!(store.len_for_tenant(t1), 2);
        assert_eq!(store.len_for_tenant(t2), 1);
        assert_eq!(store.total_len(), 3);
    }

    #[test]
    fn no_cap_holds_beyond_the_old_100k_ceiling() {
        // #352 acceptance (cap removal): the durable store has NO
        // per-tenant ceiling. Insert past the old 100K cap and confirm
        // every binding still resolves. (Bounded above for test speed;
        // 100_001 is enough to cross the former MAX_IDEMPOTENCY cap.)
        let store = IdempotencyStore::new();
        let tenant = TenantId::DEFAULT;
        let n: u64 = 100_001;
        for i in 0..n {
            store.install(tenant, NODE, &format!("ext-{i}"), i);
        }
        assert_eq!(store.total_len(), n as usize);
        assert_eq!(
            store.get(tenant, NODE, "ext-0").map(|b| b.internal_id),
            Some(0)
        );
        assert_eq!(
            store
                .get(tenant, NODE, &format!("ext-{}", n - 1))
                .map(|b| b.internal_id),
            Some(n - 1)
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // #1404 M0.x — bounded resident binding tier (spill / re-fault /
    // INV-DURABLE / drain-cost)
    // ─────────────────────────────────────────────────────────────────

    use std::sync::Arc;
    use tempfile::tempdir;

    /// A bounded store with a tiny cap so a handful of installs force
    /// eviction, plus its spill dir kept alive for the store's lifetime.
    fn bounded_store(cap_bindings: u64) -> (IdempotencyStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let spill = Arc::new(IdempotencySpill::open(dir.path()).unwrap());
        let cfg = IdempotencyBoundConfig {
            high_watermark_bytes: cap_bindings * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
            low_watermark_bytes: (cap_bindings / 2).max(1) * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        };
        (IdempotencyStore::with_bound(spill, cfg), dir)
    }

    /// Simulate a completed ADR-229 checkpoint: `iter_all` marks every
    /// resident binding checkpoint-durable (the INV-DURABLE gate the
    /// producer sets under the freeze), returning the captured set.
    fn checkpoint_capture(store: &IdempotencyStore) -> usize {
        store.iter_all().len()
    }

    #[test]
    fn bounded_store_nothing_evicts_before_a_checkpoint() {
        // INV-DURABLE: a binding is evict-eligible ONLY after a checkpoint
        // captured its durable image. Before any checkpoint, the drain must
        // reclaim nothing even under memory pressure.
        let (store, _dir) = bounded_store(2);
        for i in 0..20u64 {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        store.force_drain_for_test();
        assert_eq!(
            store.evicted_count(),
            0,
            "INV-DURABLE: nothing may evict before the first checkpoint",
        );
        // Every binding still resolves (all resident).
        for i in 0..20u64 {
            assert_eq!(
                store
                    .get(TenantId::DEFAULT, NODE, &format!("ext-{i}"))
                    .map(|b| b.internal_id),
                Some(i)
            );
        }
    }

    #[test]
    fn spilled_binding_still_resolves_via_refault() {
        // The load-bearing leg: after a checkpoint + drain evicts bindings to
        // spill, a `get` of a spilled external_id STILL returns its binding
        // (faulted in from the durable, queryable spill). This is what makes a
        // re-ingest de-dupe instead of duplicating.
        let (store, _dir) = bounded_store(2);
        let n = 30u64;
        for i in 0..n {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        // Checkpoint marks resident bindings durable, then drain evicts.
        checkpoint_capture(&store);
        store.force_drain_for_test();
        assert!(
            store.evicted_count() > 0,
            "eviction did not fire post-checkpoint — cannot test the re-fault",
        );
        // EVERY external_id — resident or spilled — still resolves to the
        // right id. A miss here would be a lost identity → a duplicate on
        // re-ingest.
        for i in 0..n {
            let got = store
                .get(TenantId::DEFAULT, NODE, &format!("ext-{i}"))
                .unwrap_or_else(|| panic!("ext-{i} unresolvable after spill — lost identity"));
            assert_eq!(got.internal_id, i, "ext-{i} faulted to the WRONG id");
        }
        assert!(store.refault_count() > 0, "no re-faults happened");
        // The logical set is complete (resident + spilled).
        assert_eq!(store.total_len(), n as usize);
    }

    #[test]
    fn payload_hash_survives_spill_roundtrip() {
        // The spill record must preserve the payload_hash (a retry-vs-conflict
        // discriminator), not just the internal_id.
        let (store, _dir) = bounded_store(1);
        for i in 0..10u64 {
            store.install_with_payload_hash(
                TenantId::DEFAULT,
                NODE,
                &format!("ext-{i}"),
                i,
                Some(1000 + i),
            );
        }
        checkpoint_capture(&store);
        store.force_drain_for_test();
        assert!(store.evicted_count() > 0);
        for i in 0..10u64 {
            let b = store
                .get(TenantId::DEFAULT, NODE, &format!("ext-{i}"))
                .expect("spilled binding");
            assert_eq!(b.internal_id, i);
            assert_eq!(b.payload_hash, Some(1000 + i));
        }
    }

    #[test]
    fn iter_all_captures_resident_plus_spilled_completely() {
        // The freeze-capture (`iter_all`) must enumerate the FULL logical set
        // — resident AND spilled — so the ADR-229 snapshot loses nothing.
        let (store, _dir) = bounded_store(2);
        let n = 25u64;
        for i in 0..n {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        checkpoint_capture(&store);
        store.force_drain_for_test();
        assert!(store.evicted_count() > 0);
        // A SECOND capture now spans resident + spilled.
        let captured = store.iter_all();
        assert_eq!(
            captured.len(),
            n as usize,
            "iter_all lost a binding across the resident/spill boundary",
        );
        // Every id present exactly once.
        let mut ids: Vec<u64> = captured.iter().map(|(_, _, _, id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..n).collect::<Vec<_>>());
    }

    /// M5-D3 FIX 4 (#1518 skeptic review) — `for_each_binding`'s RESIDENT
    /// capture must stream in sorted `(tenant, kind, external_id)` order, a
    /// pure function of the key set — not raw `DashMap` iteration order
    /// (which is a function of insertion history via shard/bucket layout;
    /// same nondeterminism class the blob page-image capture was pinned
    /// for, see `blob.rs::sorted_resident_keys`). Two unbounded stores
    /// holding the IDENTICAL bindings, installed in DIFFERENT orders, must
    /// stream in IDENTICAL order.
    ///
    /// RED-on-revert: replace the sorted-key capture with a raw
    /// `self.resident().forward.iter()` walk (the pre-fix code) — this
    /// test then fails intermittently (most runs, on a large enough key
    /// set), since two stores built from the same bindings in different
    /// insertion order generally land in different DashMap bucket order.
    #[test]
    fn for_each_binding_capture_order_is_sorted_not_insertion_or_shard_order() {
        let forward_store = IdempotencyStore::new();
        let reverse_store = IdempotencyStore::new();
        let tenants = [TenantId::new(11), TenantId::new(4), TenantId::new(97)];
        let bindings: Vec<(TenantId, u8, String, u64)> = tenants
            .iter()
            .flat_map(|tenant| {
                (0..200_u64).map(move |i| {
                    let kind = if i % 2 == 0 { NODE } else { REL };
                    (*tenant, kind, format!("ext-{tenant:?}-{i:04}"), i)
                })
            })
            .collect();

        for (tenant, kind, ext, id) in &bindings {
            forward_store.install(*tenant, *kind, ext, *id);
        }
        for (tenant, kind, ext, id) in bindings.iter().rev() {
            reverse_store.install(*tenant, *kind, ext, *id);
        }

        let mut forward_stream = Vec::new();
        forward_store
            .for_each_binding(
                |tenant,
                 kind,
                 ext,
                 internal_id,
                 _hash|
                 -> std::result::Result<(), std::convert::Infallible> {
                    forward_stream.push((tenant, kind, ext.to_owned(), internal_id));
                    Ok(())
                },
            )
            .unwrap();
        let mut reverse_stream = Vec::new();
        reverse_store
            .for_each_binding(
                |tenant,
                 kind,
                 ext,
                 internal_id,
                 _hash|
                 -> std::result::Result<(), std::convert::Infallible> {
                    reverse_stream.push((tenant, kind, ext.to_owned(), internal_id));
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            forward_stream.len(),
            bindings.len(),
            "sanity: every binding streamed"
        );
        assert_eq!(
            forward_stream, reverse_stream,
            "DEFECT: for_each_binding's capture order depends on insertion \
             history (DashMap shard/bucket layout), not a sort — two stores \
             with identical bindings streamed in DIFFERENT order"
        );
        let mut sorted = forward_stream.clone();
        sorted.sort_unstable_by(|left, right| {
            (left.0, left.1, &left.2).cmp(&(right.0, right.1, &right.2))
        });
        assert_eq!(
            forward_stream, sorted,
            "capture order must be exactly sorted (tenant, kind, external_id)"
        );
    }

    #[test]
    fn overwrite_of_spilled_key_supersedes_the_stale_image() {
        // If a spilled key is re-installed with a NEW id, a later `get` must
        // return the NEW id (the stale spill image must not resurrect).
        let (store, _dir) = bounded_store(2);
        for i in 0..20u64 {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        checkpoint_capture(&store);
        store.force_drain_for_test();
        assert!(store.evicted_count() > 0);
        // Overwrite an evicted key with a new id.
        store.install(TenantId::DEFAULT, NODE, "ext-0", 999);
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, "ext-0")
                .map(|b| b.internal_id),
            Some(999),
            "overwrite of a spilled key must supersede the stale spill image",
        );
    }

    #[test]
    fn release_of_spilled_key_frees_it() {
        // Releasing a spilled external_id truly frees it (a later get misses,
        // and total_len drops) — release removes both tiers.
        let (store, _dir) = bounded_store(2);
        for i in 0..20u64 {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        checkpoint_capture(&store);
        store.force_drain_for_test();
        let before = store.total_len();
        store.release(TenantId::DEFAULT, NODE, "ext-1");
        assert_eq!(store.get(TenantId::DEFAULT, NODE, "ext-1"), None);
        assert_eq!(store.total_len(), before - 1);
    }

    #[test]
    fn drain_cost_is_bounded_per_install_not_resident_count() {
        // Drain-cost regression guard (mirrors the M0 blob guard): under the
        // realistic all-NotDurable regime (installs between checkpoints), each
        // install's drain does O(1) probing (pop front → NotDurable → break),
        // NOT O(resident-count).
        let (store, _dir) = bounded_store(4);
        // Install many WITHOUT a checkpoint in between → all NotDurable.
        for i in 0..1000u64 {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        // Nothing durable → nothing evicted, and total probes are bounded
        // (one break-probe per over-watermark install), NOT ~O(N²).
        assert_eq!(store.evicted_count(), 0);
        assert!(
            store.drain_probe_count() <= 1000,
            "drain probed {} times — should be O(installs), not O(N × resident)",
            store.drain_probe_count(),
        );
    }

    #[test]
    fn unbounded_store_never_spills_or_evicts() {
        // The legacy default: no spill attached → nothing evicts, behavior is
        // the pre-#1404 pure-in-RAM store.
        let store = IdempotencyStore::new();
        assert!(!store.is_bounded());
        for i in 0..500u64 {
            store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), i);
        }
        checkpoint_capture(&store);
        store.force_drain_for_test();
        assert_eq!(store.evicted_count(), 0);
        assert_eq!(store.refault_count(), 0);
        assert_eq!(store.total_len(), 500);
    }

    /// GATEX (capture-peak, the manager-verify required gate) — the CAPTURE's
    /// peak-resident is O(1) in binding-count. This kills the whole-in-RAM
    /// sibling: the freeze-capture must NOT re-collect the entire binding set
    /// into one `Vec` under `checkpoint_freeze`. Measured at 2 sizes (64 vs
    /// 1024, all spilled) via the PRODUCTION `for_each_binding`; RED-on-revert
    /// = the whole-`Vec` `iter_all` oracle (peak grows ~16× with N).
    ///
    /// This lives as a UNIT test (not integration) because it compares against
    /// the `#[cfg(test)]`-only whole-`Vec` `iter_all` oracle — exactly the M0.5
    /// pattern (the whole-`Vec` supplement kept as a `#[cfg(test)]` differential
    /// oracle).
    #[test]
    fn gatex_capture_peak_resident_is_o1_in_binding_count() {
        // Build a bounded store of size `n` with EVERY binding spilled.
        fn all_spilled(n: u64) -> (IdempotencyStore, tempfile::TempDir) {
            let dir = tempdir().unwrap();
            let spill = Arc::new(IdempotencySpill::open(dir.path()).unwrap());
            // Tiny cap → the drain evicts everything to spill.
            let cfg = IdempotencyBoundConfig {
                high_watermark_bytes: 2 * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
                low_watermark_bytes: IDEMPOTENCY_BINDING_WEIGHT_BYTES,
            };
            let store = IdempotencyStore::with_bound(spill, cfg);
            for i in 0..n {
                store.install(TenantId::DEFAULT, NODE, &format!("ext-{i:08}"), i);
            }
            // Mark durable (freeze capture) then drain to spill.
            let _ = store.for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()));
            store.force_drain_for_test();
            assert!(store.evicted_count() > 0, "nothing spilled at n={n}");
            (store, dir)
        }

        // STREAMING peak = the store's OWN `capture_peak_in_flight` counter
        // (the max records the PRODUCTION capture held resident SIMULTANEOUSLY).
        // The streaming path reads → emits → DROPS each record, so the counter
        // stays 1 regardless of N. A reverted whole-`Vec` `for_each_binding`
        // (pre-collect all then replay) would drive this counter to N → the
        // gate flips RED. This is the revert-sensitive PRODUCTION-metric probe.
        fn streaming_peak(store: &IdempotencyStore) -> u64 {
            let mut count = 0u64;
            store
                .for_each_binding::<_, std::convert::Infallible>(|_, _, _ext, _, _| {
                    count += 1;
                    Ok(())
                })
                .expect("infallible");
            assert_eq!(count, store.binding_count());
            store.capture_peak_in_flight()
        }

        // WHOLE-`Vec` peak (the reverted term) = ALL N records held at once.
        fn whole_vec_peak(store: &IdempotencyStore) -> u64 {
            store.iter_all().len() as u64
        }

        let small_n = 64u64;
        let large_n = 1024u64; // 16× larger
        let (small, _d1) = all_spilled(small_n);
        let (large, _d2) = all_spilled(large_n);
        assert_eq!(small.binding_count(), small_n);
        assert_eq!(large.binding_count(), large_n);

        // ── PRODUCTION streaming capture: FLAT peak (O(1)) across 16× size ──
        let s_small = streaming_peak(&small);
        let s_large = streaming_peak(&large);
        assert_eq!(
            s_small, 1,
            "streaming capture peak must be 1 record (O(1)), got {s_small} at n={small_n}",
        );
        assert_eq!(
            s_large, 1,
            "streaming capture peak must be 1 record (O(1)), got {s_large} at n={large_n}",
        );
        let s_ratio = s_large / s_small.max(1);
        assert_eq!(
            s_ratio, 1,
            "streaming peak must be FLAT (ratio 1), got {s_ratio}"
        );

        // ── RED-on-revert: whole-`Vec` capture peak GROWS ~16× with N ──
        let w_small = whole_vec_peak(&small);
        let w_large = whole_vec_peak(&large);
        assert_eq!(w_small, small_n);
        assert_eq!(w_large, large_n);
        let w_ratio = w_large / w_small.max(1);
        assert_eq!(
            w_ratio, 16,
            "whole-Vec capture peak ratio must be ~16× (the reverted unbounded term), got {w_ratio}",
        );

        // The load-bearing contrast, printed for the record: streaming is FLAT
        // (1, 1) while the reverted whole-`Vec` scales (64, 1024).
        println!(
            "GATEX capture-peak records — streaming(PROD): n={small_n}→{s_small}, n={large_n}→{s_large} (ratio {s_ratio}×, O(1)); \
             whole-Vec[REVERTED]: n={small_n}→{w_small}, n={large_n}→{w_large} (ratio {w_ratio}×, O(N))",
        );
    }

    /// #1500 FIX-5(f) RED-on-revert — the DETERMINISTIC vacant-ABA schedule
    /// the probabilistic `fix5b` gate (48 writers + 16 readers + drain
    /// pressure) only hits by luck. The stale reader's interleaving is driven
    /// step-by-step:
    ///
    /// 1. install H(1) → checkpoint-capture → evict: slot VACANT, spill H(1).
    /// 2. Stale reader snapshots the slot generation and "reads" H(1) from
    ///    spill (exactly `resolve_binding`'s state before its warm-insert).
    /// 3. The ABA under it: fresh same-id install H(2) → capture → evict —
    ///    the slot is VACANT AGAIN and the newest spill image is H(2).
    /// 4. The reader's warm-insert of stale H(1) must be REJECTED (`None`):
    ///    pre-fix (no slot-generation check) the vacant-check accepted it,
    ///    installed H(1) as checkpoint-durable resident truth, and every
    ///    subsequent read served the pre-publish hash (fix5b's "stale
    ///    payload_hash for a committed same-id re-publish").
    #[test]
    fn refault_warm_insert_rejects_stale_image_after_vacant_aba() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Arc::new(IdempotencySpill::open(dir.path()).unwrap());
        let store = IdempotencyStore::with_bound(
            spill,
            IdempotencyBoundConfig {
                high_watermark_bytes: IDEMPOTENCY_BINDING_WEIGHT_BYTES,
                // A single resident binding sits above the low watermark, so
                // every forced drain evicts it (deterministic vacancy).
                low_watermark_bytes: 1,
            },
        );
        let key = (TenantId::DEFAULT, NODE, "aba-key".to_owned());

        // 1. Publish H(1), make it durable, evict it: vacant + spill H(1).
        store.install_with_payload_hash(TenantId::DEFAULT, NODE, "aba-key", 42, Some(1));
        store
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        store.force_drain_for_test();
        assert_eq!(store.resident_len(), 0, "H(1) must be evicted to spill");

        // 2. The stale reader: slot-generation snapshot + the spill image it
        //    would have read (H(1)) before being preempted.
        let stale_snapshot = store.resident().slot_generations.load(&key);
        let stale_image = IdempotencyBinding {
            internal_id: 42,
            payload_hash: Some(1),
        };

        // 3. The ABA completes under the preempted reader: fresh same-id
        //    re-publish H(2) → capture → evict. Vacant again; newest spill
        //    image is H(2).
        store.install_with_payload_hash(TenantId::DEFAULT, NODE, "aba-key", 42, Some(2));
        store
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        store.force_drain_for_test();
        assert_eq!(store.resident_len(), 0, "H(2) must be evicted to spill");

        // 4. The reader resumes: its warm-insert of stale H(1) into the
        //    (vacant) slot MUST be rejected — the slot generation advanced.
        let outcome = store.insert_resident_warm_if_vacant(
            TenantId::DEFAULT,
            NODE,
            "aba-key",
            stale_image,
            stale_snapshot,
        );
        assert!(
            outcome.is_none(),
            "FIX-5(f) FAIL: the vacant-ABA warm-insert of a stale spill image \
             was ACCEPTED — a committed same-id re-publish can now serve its \
             pre-publish payload_hash (fix5b regression)",
        );

        // The rejected reader re-faults fresh: the served hash is H(2).
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, "aba-key")
                .map(|b| b.payload_hash),
            Some(Some(2)),
            "post-rejection re-fault must serve the newest committed hash",
        );

        // Positive control: with a CURRENT snapshot (no removal since), the
        // warm-insert path still lands — the rejection is ABA-specific, not a
        // blanket re-fault disable.
        store.force_drain_for_test();
        assert_eq!(store.resident_len(), 0);
        let fresh_snapshot = store.resident().slot_generations.load(&key);
        let fresh_image = IdempotencyBinding {
            internal_id: 42,
            payload_hash: Some(2),
        };
        assert_eq!(
            store.insert_resident_warm_if_vacant(
                TenantId::DEFAULT,
                NODE,
                "aba-key",
                fresh_image,
                fresh_snapshot,
            ),
            Some((fresh_image, true)),
            "a warm-insert with an unmoved slot generation must land",
        );
    }
}
