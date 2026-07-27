//! MVCC transaction manager (M2.b tasks M2-10 … M2-15).
//!
//! See design-v2 §4.3, ADR-007 (MVCC uses `created_lsn`/`expired_lsn`
//! for visibility only — temporal queries remain v1.1), and ADR-010
//! (sync `PageIo` is sufficient; MVCC introduces no async I/O).
//!
//! This module is the pure MVCC kernel: an LSN allocator, an
//! OCC-validated version store keyed by an abstract `u64`, a per-txn
//! write buffer, snapshot-isolated reads, and a background GC. CRUD
//! (M2.c) layers node/rel/page integration on top; M2.b deliberately
//! stays below the record layer to land testable isolation semantics
//! independently of the disk format.
//!
//! Latency / memory budget (design-v2 §4.4, "5 K TPS"):
//!
//! - `LsnCounter::allocate_range`: one atomic range reservation —
//!   tens-of-nanoseconds class on the commit path.
//! - `Transaction::read`: `DashMap::get` + reverse-linear-scan of a
//!   version chain. For typical chains (≤ 16 versions, given GC) this
//!   is ≤ 100 ns warm-cache.
//! - `Transaction::commit`: three-phase. Phase 1 (validate, allocate,
//!   install) under `commit_gate`, ~1–2 µs per write. Phase 2
//!   (`wal.append`) OUTSIDE the gate so 8 concurrent writers
//!   pipeline into a single group-commit fsync batch. Phase 3
//!   (ordered `visible.publish`) serialized by `install_order` with
//!   a condvar, ~µs per writer. Holding the gate across the WAL
//!   round-trip was the M2-E2 TPS blocker.
//! - Active-txn table: one `(u64, Lsn)` entry per in-flight txn. At 30
//!   concurrent commits in flight (design-v2 §4.4 budget) ≈ 480 B.
//!
//! Invariants (tested in `tests/mvcc_*.rs`, each at 5 K proptest cases
//! in release mode):
//!
//! 1. *Snapshot isolation* — a reader sees exactly the committed
//!    prefix at its snapshot LSN; writes committed after its snapshot
//!    are invisible.
//! 2. *No-lost-updates* — of two concurrent writers to the same key,
//!    at most one commits; the loser returns `MvccConflict`.
//! 3. *Read-your-writes* — a transaction reads its own buffered writes
//!    before consulting the committed version store.
//! 4. *Write-write conflict detection* — any commit whose write-set
//!    intersects a version installed after its snapshot is aborted
//!    with `MvccConflict`.
//! 5. *GC safety* — no version still reachable by an active snapshot
//!    is reclaimed.
//! 7. *Commit atomicity* — the commit_lsn watermark visible to
//!    readers (`visible`) advances only AFTER every write of the
//!    committing transaction is installed AND the WAL record for
//!    that commit is durable on disk. The allocator's atomic reservation
//!    happens first (to stamp `created_lsn`) and install happens
//!    silently under `commit_gate`, but no reader can observe the
//!    new LSN as their snapshot until Phase 3 has published it via
//!    `visible.store`. Corollary: a reader's snapshot either sees
//!    all of a commit's writes or none.
//! 6. *Begin/gc serialization* — `begin` publishes a sentinel
//!    (`Lsn::MAX`) into `active` BEFORE reading `counter.current()`,
//!    then upgrades the entry to the captured snapshot. A concurrent
//!    `gc()` either sees the sentinel (and is conservative — anchors
//!    to `Lsn::ZERO` if every active entry is pending) or sees the
//!    finalized snapshot (and anchors to it). No window exists in
//!    which a captured-but-unpublished snapshot can be outrun by a
//!    counter advance.
//! 8. *Commit-path pipelining* — `commit_gate` is released BEFORE
//!    the `wal.append` round-trip, so 8 concurrent writers batch
//!    into a single group-commit fsync rather than serializing. A
//!    secondary serialization primitive — `install_order`
//!    (advancing strictly by commit_lsn) gated by
//!    `install_cv` — orders the post-fsync `visible.store` so the
//!    watermark advances monotonically. On WAL failure, the silent
//!    Phase-1 install is rolled back (versions popped, predecessor
//!    `expired_lsn` restored) under `commit_gate` so concurrent
//!    validators never observe a half-popped chain. `visible`
//!    stays unchanged on WAL failure (preserving invariant 4); only
//!    `install_order` advances, so successor commits can progress
//!    without deadlocking on a skipped LSN.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{ArcGraphError, DurabilityTier, Lsn, Result, TenantDurabilityLookup, TenantId};
use bytes::Bytes;
use dashmap::{DashMap, DashSet};
use parking_lot::{Condvar, Mutex};

use crate::config::WalErrorPolicy;
use crate::mutation_log::TxnMutationLog;
use crate::redo::RedoLsnRange;
use crate::wal::bundle::{
    AclGrantEntry, AllocatorAdvance, IdempotencyBindingEntry, SideChannelWrite, StagedEmit,
    VectorPageEntry, encode_commit_bundle_current, encode_commit_bundle_delta_for_format,
};
use crate::wal::{DeltaOp, WalHandle, WalRecordType, is_delta_bundle_format};

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Abstract MVCC key. M2.b uses a plain `u64` so the kernel is
/// independent of record encoding. M2.c (CRUD) will wrap node/rel ids
/// behind a newtype above this layer.
pub type MvccKey = u64;

/// A single MVCC version of a key. Layout mirrors the `created_lsn` /
/// `expired_lsn` pair reserved in every node and relationship record
/// by design-v2 §3.2 and ADR-007.
///
/// - `created_lsn` is the commit LSN of the transaction that installed
///   this version.
/// - `expired_lsn` is `Lsn::MAX` while the version is live; on
///   overwrite or delete, it is set to the commit LSN of the
///   overwriting transaction.
/// - `value = None` encodes a tombstone (delete). We keep tombstones
///   in the chain until GC to preserve the "this key was visible at
///   snapshot S but is gone at snapshot S'" read semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Version {
    /// Commit LSN that installed this version.
    pub created_lsn: Lsn,
    /// Commit LSN that superseded this version, or `Lsn::MAX` while live.
    pub expired_lsn: Lsn,
    /// `Some(bytes)` for a live value, `None` for a tombstone.
    pub value: Option<Bytes>,
}

impl Version {
    /// Is this version live (not superseded) at the time of inspection?
    #[inline]
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.expired_lsn == Lsn::MAX
    }

    /// Visibility predicate (design-v2 §4.3, MVCC snapshot rule).
    ///
    /// A version is visible to `snapshot` iff
    /// `created_lsn <= snapshot < expired_lsn`. `expired_lsn == MAX`
    /// means "never expired yet", which is strictly greater than any
    /// concrete snapshot.
    #[inline]
    #[must_use]
    pub fn visible_to(&self, snapshot: Lsn) -> bool {
        self.created_lsn.raw() <= snapshot.raw() && snapshot.raw() < self.expired_lsn.raw()
    }
}

// ─────────────────────────────────────────────────────────────────────
// M2-10: logical LSN counter
// ─────────────────────────────────────────────────────────────────────

/// Monotonic 64-bit logical LSN allocator. Cache-line-isolated so it
/// does not false-share with neighbouring atomics (design-v2 §4.1).
///
/// LSN 0 is reserved as the "never seen" floor per
/// `arcgraph_core::Lsn::ZERO`; allocation begins at 1.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct LsnCounter {
    inner: AtomicU64,
    _pad: [u8; 56],
}

impl LsnCounter {
    /// First LSN returned by [`allocate`](Self::allocate).
    pub const INITIAL: u64 = 1;

    /// Construct a counter whose first `allocate()` returns
    /// [`Self::INITIAL`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: AtomicU64::new(Self::INITIAL - 1),
            _pad: [0; 56],
        }
    }

    /// Construct a counter seeded so that the *next* LSN returned by
    /// `allocate` is `start`. Used by WAL recovery to resume LSN
    /// allocation above the persisted watermark.
    #[must_use]
    pub const fn with_floor(start: u64) -> Self {
        Self {
            inner: AtomicU64::new(start.saturating_sub(1)),
            _pad: [0; 56],
        }
    }

    /// Read the last allocated LSN without advancing.
    #[inline]
    #[must_use]
    pub fn current(&self) -> Lsn {
        Lsn::new(self.inner.load(Ordering::Acquire))
    }

    /// Allocate the next LSN. Strictly greater than every previously
    /// returned value across threads.
    #[inline]
    #[must_use]
    pub fn allocate(&self) -> Lsn {
        self.allocate_range(1).commit_lsn()
    }

    /// Allocate one contiguous range for a commit's physiological redo
    /// ops (M3 IMPL-DEC-2). `op_count == 0` still consumes width one so
    /// MVCC-only commits retain a unique commit LSN.
    #[must_use]
    pub fn allocate_range(&self, op_count: usize) -> RedoLsnRange {
        let width = u64::try_from(op_count.max(1)).expect("redo op count exceeds u64");
        let previous = self
            .inner
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |last| {
                last.checked_add(width)
            })
            .expect("redo LSN space exhausted");
        let base = previous.checked_add(1).expect("redo LSN base overflow");
        let end = previous.checked_add(width).expect("redo LSN end overflow");
        RedoLsnRange::new(Lsn::new(base), Lsn::new(end))
            .expect("allocated redo range must be non-zero and ordered")
    }

    /// ADR-032 replay seed: monotonically advance the counter so the
    /// next `allocate()` returns `floor + 1`. A no-op if `floor` is
    /// not greater than the current value. Idempotent across
    /// repeated calls. Called from
    /// [`TxnManager::seed_after_replay`].
    #[inline]
    pub fn advance_to(&self, floor: Lsn) {
        self.inner.fetch_max(floor.raw(), Ordering::AcqRel);
    }
}

impl Default for LsnCounter {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// M2-11 + M2-13 + M2-14 + M2-15: transaction manager
// ─────────────────────────────────────────────────────────────────────

type VersionChain = Vec<Version>;

/// Composite key into the version store. Each (tenant, abstract key)
/// pair has an independent version chain so version visibility is
/// tenant-scoped (ADR-011). The global LSN allocator and GC anchor
/// are shared across all tenants per ADR-011 §"MVCC LSN decision".
type VersionKey = (TenantId, MvccKey);

/// Outcome of [`TxnManager::apply_replay_mvcc_write`] (ADR-032 §R2
/// + PR #79 Y-3 fold-in).
///
/// Three mutually-exclusive states; the replay executor routes each
/// variant to its own counter so operators can tell legitimate
/// idempotent skips (good, common on double-replay) from out-of-order
/// rejections (bad, always a bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayApplyOutcome {
    /// New `Version` pushed onto the chain. Common case on first
    /// replay; counted by `wal_replay_mvcc_versions_installed`.
    Applied,
    /// Chain already had a `Version` with the same `created_lsn`.
    /// Lemma I1 idempotent skip; counted by
    /// `wal_replay_bundles_skipped_idempotent`.
    Idempotent,
    /// Chain's last `Version` has `created_lsn` strictly greater
    /// than the bundle's. The executor sorts bundles before apply,
    /// so this path signals an upstream bug (e.g., unsynced buffer
    /// ordering). Counted by
    /// `wal_replay_out_of_order_apply_rejected`; `tracing::error!`
    /// fires so the regression is loud in logs.
    OutOfOrder,
}

impl ReplayApplyOutcome {
    /// True iff a new Version was pushed.
    #[inline]
    #[must_use]
    pub fn is_applied(self) -> bool {
        matches!(self, Self::Applied)
    }
}

/// Transaction-manager state. One instance per database; CRUD and
/// query layers take `&TxnManager`.
///
/// Two-counter pattern (invariant 7): `counter` is the LSN allocator
/// — it advances eagerly at `commit_writes` entry so each committing
/// txn can stamp its own `created_lsn`. `visible` is a monotonic
/// watermark that advances ONLY after the install loop of a commit
/// has finished AND its WAL record is durable (invariant 4). New
/// transactions source their snapshot from `visible`, not from
/// `counter`, so a reader can never capture a snapshot in the middle
/// of another commit's install phase or before its WAL is fsynced.
///
/// Three-phase commit (invariant 8): `commit_gate` now covers only
/// Phase 1 (validate + allocate + silent install); Phase 2
/// (`wal.append`) runs OUTSIDE the gate so concurrent writers
/// pipeline into one fsync batch; Phase 3 (`visible.store` +
/// rollback-on-WAL-failure) is serialized by `install_order` +
/// `install_cv` so the watermark advances strictly by commit_lsn
/// order.
pub struct TxnManager {
    counter: LsnCounter,
    /// Monotonic commit watermark. Readers use this for their
    /// snapshot LSN; a commit publishes its `commit_lsn` here only
    /// in Phase 3 on WAL success (invariants 4, 6, 7). On WAL
    /// failure this value is unchanged, so no reader snapshot ever
    /// covers a non-durable LSN.
    visible: AtomicU64,
    versions: DashMap<VersionKey, VersionChain>,
    /// Per-tenant chain index (issue #238 — closes O(K² × N) DashMap
    /// scan from PR #236 MED-1). For each tenant `T`, holds the set of
    /// `MvccKey`s for which a version chain has been pushed into
    /// `versions` keyed by `(T, key)`. Maintained on every push path
    /// (`commit_with_bundle_writes` Phase 1, sidechannel apply,
    /// `apply_replay_mvcc_write`, the test-only
    /// `commit_with_barriers_raw`) — **insert-only**.
    ///
    /// Allows [`Self::for_each_visible_record`] and
    /// [`Self::tenants_with_chains`] to iterate per-tenant in
    /// `O(N_tenant)` rather than scanning every shard of `versions`
    /// (`O(N_total)`). Per ADR-038 amendment-06 §D-25.2 (1)/(2)/(3),
    /// the aggregate cost of `rebuild_all_tenant_stats` drops from
    /// `O(K² × N_per_tenant)` to `O(K × N_per_tenant) = O(N_total)`.
    ///
    /// **Monotone-growing (PR #243 round-2 MED-1 closure).** This
    /// index is **never shrunk**. The original PR removed entries on
    /// GC `remove_if` success, but a GC vs. concurrent commit
    /// interleave (`gc()` `versions.remove_if` succeeds → concurrent
    /// commit pushes a new chain + `register_chain_key` →
    /// `unregister_chain_key` REMOVES the just-registered key) could
    /// leave a key present in `versions` but absent from
    /// `tenant_chain_keys` — a false-negative that would make the
    /// committing thread's record invisible to a subsequent
    /// `for_each_visible_record` walk. Dropping the unregister path
    /// makes the false-negative window vacuously empty: the index is
    /// a strict SUPERSET of "tenants × keys with currently-non-empty
    /// chains" and `for_each_visible_record`'s per-chain visibility
    /// filter handles the empty-chain (false-positive) case
    /// correctly. Memory is bounded by `K_active × max_keys_ever_committed_per_tenant`,
    /// which v1.0-alpha tenant-size constraints (`docs/arcgraph-design-v2.md`
    /// §10) already bound.
    ///
    /// **Tenant entries are also retained on emptiness.** A tenant
    /// with sporadic commits would thrash its DashMap entry between
    /// the GC of its last live key and the next commit. The empty-set
    /// case has no observable behaviour (iteration produces zero
    /// callbacks) and the allocation footprint is bounded by
    /// `K_active`.
    tenant_chain_keys: DashMap<TenantId, DashSet<MvccKey>>,
    /// txn_id → snapshot LSN. Entries live for the duration of the
    /// transaction and anchor GC (see `oldest_active_snapshot`).
    active: DashMap<u64, Lsn>,
    next_txn_id: AtomicU64,
    /// #1404 M0.x FIX-C — begin-generation FENCE against the begin-vs-gc race
    /// the default-on M0.x gc DRIVER activates. `begin_inner` bumps this at the
    /// START of its two-phase publish (BEFORE inserting the pending sentinel and
    /// BEFORE reading `visible`); `oldest_active_snapshot` snapshots it before +
    /// after its `active` scan and, if it advanced (a begin published — or is
    /// mid-publish — during the scan, so its sentinel may not have propagated to
    /// the shard the scan already visited), CLAMPS the anchor to `Lsn::ZERO`
    /// (protect everything this pass). This closes the shard-late-insert
    /// residual that a clamp-on-observed-pending alone leaves open: a begin that
    /// inserted its sentinel into a shard the scan already passed is invisible
    /// to the scan, but its generation bump is NOT (the scan re-reads the gen
    /// after). gc reclaiming nothing for one pass is harmless (the next pass
    /// retries once the begin has published its concrete snapshot). The hot
    /// commit/read path never reads this counter; only `begin` (write) bumps it.
    begin_generation: AtomicU64,
    /// Commit-gate. Serializes Phase 1 (validate + allocate + silent
    /// install) and the Phase-3 rollback window (on WAL failure), so
    /// OCC is linearizable w.r.t. the version store and rollback
    /// never races a concurrent validator's chain scan.
    ///
    /// It is NOT held across `wal.append` (invariant 8). That was the
    /// M2-E2 TPS blocker.
    commit_gate: Mutex<()>,
    /// Phase-3 ordering. `install_order` is the highest commit_lsn
    /// whose Phase 3 has completed (success OR failure). Waiters
    /// block on `install_cv` until it reaches the range predecessor.
    /// Every completed Phase 3 advances it; in particular, a WAL
    /// failure advances `install_order` (so successors can proceed)
    /// WITHOUT advancing `visible` (so readers never see the
    /// non-durable LSN).
    install_order: Mutex<u64>,
    install_cv: Condvar,
    /// Optional WAL handle. When set, `commit_writes` fsyncs a single
    /// aggregated WAL record for the committing transaction in
    /// Phase 2, outside the commit gate. A WAL I/O failure rolls
    /// back the Phase-1 silent install so readers never observe the
    /// non-durable LSN (invariant 4). M2-WAL.
    wal: parking_lot::RwLock<Option<WalHandle>>,
    /// ADR-034 §Slice D — optional per-tenant durability tier
    /// resolver. Queried at commit TIME (Phase 2) to decide whether
    /// the commit uses [`WalHandle::append`] (T1 / Strict) or
    /// [`WalHandle::append_async`] (T3 / Periodic). When `None`, all
    /// commits default to Strict — preserves pre-ADR-034 behaviour
    /// for `TxnManager` instances constructed without a catalog.
    durability_lookup: parking_lot::RwLock<Option<Arc<dyn TenantDurabilityLookup>>>,
    /// SVC-1 / #849 / ADR-229 — checkpoint/commit serialization lock.
    ///
    /// Every three-phase commit takes a **READ** guard for the FULL span
    /// (Phase 1 alloc + silent install → builder page-write → Phase 2 WAL
    /// fsync → Phase 3 `visible.store`), so concurrent commits still
    /// pipeline (readers don't exclude readers). The checkpoint producer
    /// takes the **WRITE** guard (`checkpoint_freeze`) around the frontier
    /// read + the full-state capture, which blocks until every in-flight
    /// commit has finished Phase 3 (WAL-durable + visible) and prevents a
    /// new commit from starting mid-capture.
    ///
    /// This is the missing primitive the ULTRACODE verdict identified: it
    /// makes the whole checkpoint capture **point-in-time-consistent with
    /// the commit frontier**, closing BLOCK-1 (allocator captured in a
    /// commit skew window → live-id reuse) and BLOCK-2 (page image of a
    /// not-yet-WAL-durable commit → phantom record). While the write guard
    /// is held: no commit is between its `counter.allocate()` and its
    /// `visible.store`, so `current_lsn()` (the frontier), the allocator
    /// high-water, `for_each_visible_record`, and every page-store latch
    /// are all captured against ONE quiescent instant.
    ///
    /// Budget (PD#5): the read guard adds one uncontended `RwLock::read`
    /// to the commit path (~tens of ns, no cross-core cache-line bounce in
    /// the common no-checkpoint-in-flight case). The write guard is held
    /// only for the checkpoint capture (background / shutdown, NOT the hot
    /// path), bounded by the capture cost, and only stalls foreground
    /// commits for that window — the pacing the ADR-229 trigger threshold
    /// already sizes.
    checkpoint_lock: parking_lot::RwLock<()>,

    // ── #1404 M0.x — MVCC version-row drain driver (bounded resident set) ──
    //
    // The reclaimer `gc()` (`transaction.rs:766`) exists but was DRIVEN only at
    // the ADR-229 checkpoint trigger (`bootstrap.rs:798`) — rare (~1 GiB-WAL /
    // ~300 s). Between checkpoints, superseded MVCC versions (update/delete
    // churn — and, load-bearingly, the REL-side adjacency updates that the
    // #1404 acceptance OOM'd on) accumulate resident with NOTHING driving their
    // reclamation, contributing to the freeze-capture working set the
    // checkpoint then pins (`producer.rs:132`). This driver runs `gc()` on the
    // FRONTIER ADVANCE (watermark-triggered, mirroring the M0 blob drain on
    // `publish`) so the resident superseded-version set stays bounded BETWEEN
    // checkpoints. INV-DRAIN is UNCHANGED: `gc()` still reclaims a version iff
    // `expired_lsn ≤ oldest_active_snapshot` — it never touches a
    // snapshot-visible version. This changes only WHEN/HOW-OFTEN `gc()` runs,
    // not its correctness (design §3 "we only change the trigger cadence").
    //
    // NOTE ON INSERT-ONLY LIVE VERSIONS: under pure insert-only ingest every
    // version is live (`expired_lsn = MAX`), so `gc()` reclaims nothing — the
    // design (§2.1 / OQ-2) correctly scopes *live-version RAM-eviction* as the
    // harder, single-image-page-store problem deferred to the record-native
    // M4/M6 steps. M0.x's MVCC leg bounds the SUPERSEDED set (the churn/rel
    // adjacency term) by driving `gc()`; the live-insert term is bounded
    // structurally by M1–M6 (ADR-230). Driving `gc()` here is the reclaim that
    // was defined but not driven during sustained ingest.
    /// Commits since the last driver-initiated `gc()`. A relaxed counter bumped
    /// once per completed commit (outside the commit locks). When it crosses
    /// `gc_drive_interval`, the next commit drives a `gc()` pass.
    commits_since_gc: AtomicU64,
    /// Watermark: drive a `gc()` pass every this-many commits. `0` disables
    /// the driver (the legacy behavior — `gc()` only on the checkpoint
    /// trigger). Set by [`Self::with_gc_drive_interval`].
    gc_drive_interval: u64,
    /// Count of driver-initiated `gc()` passes — test/observability only.
    driven_gc_passes: AtomicU64,
    /// Cumulative versions reclaimed by driver-initiated passes —
    /// test/observability only.
    driven_gc_reclaimed: AtomicU64,
}

impl Default for TxnManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics returned by [`TxnManager::gc`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcStats {
    /// Number of expired versions physically removed.
    pub reclaimed: u64,
    /// Number of keys inspected.
    pub scanned_keys: u64,
    /// Number of empty chains pruned from the version map.
    pub pruned_keys: u64,
    /// The anchor LSN used for this GC pass (floor of active snapshots
    /// at the moment of the pass).
    pub anchor: Lsn,
}

impl Default for GcStats {
    fn default() -> Self {
        Self {
            reclaimed: 0,
            scanned_keys: 0,
            pruned_keys: 0,
            anchor: Lsn::ZERO,
        }
    }
}

impl TxnManager {
    /// Construct an empty manager. LSN starts at
    /// [`LsnCounter::INITIAL`]. No WAL, no durability resolver —
    /// every commit behaves as Strict / T1 (pre-ADR-034 semantics).
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: LsnCounter::new(),
            visible: AtomicU64::new(Lsn::ZERO.raw()),
            versions: DashMap::new(),
            tenant_chain_keys: DashMap::new(),
            active: DashMap::new(),
            next_txn_id: AtomicU64::new(1),
            begin_generation: AtomicU64::new(0),
            commit_gate: Mutex::new(()),
            install_order: Mutex::new(Lsn::ZERO.raw()),
            install_cv: Condvar::new(),
            wal: parking_lot::RwLock::new(None),
            durability_lookup: parking_lot::RwLock::new(None),
            checkpoint_lock: parking_lot::RwLock::new(()),
            commits_since_gc: AtomicU64::new(0),
            gc_drive_interval: 0,
            driven_gc_passes: AtomicU64::new(0),
            driven_gc_reclaimed: AtomicU64::new(0),
        }
    }

    /// #1404 M0.x — enable the frontier-advance-triggered MVCC drain: drive a
    /// `gc()` pass every `interval` commits so the resident superseded-version
    /// set stays bounded between checkpoints. `0` leaves the driver disabled
    /// (legacy — `gc()` only on the checkpoint trigger). Builder-style; call
    /// once at bootstrap. See the `commits_since_gc` field docs + [`Self::gc`].
    #[must_use]
    pub fn with_gc_drive_interval(mut self, interval: u64) -> Self {
        self.gc_drive_interval = interval;
        self
    }

    /// Number of driver-initiated `gc()` passes so far (#1404 M0.x). Test /
    /// observability.
    #[doc(hidden)]
    #[must_use]
    pub fn driven_gc_passes(&self) -> u64 {
        self.driven_gc_passes.load(Ordering::Acquire)
    }

    /// Cumulative versions reclaimed by driver-initiated `gc()` passes
    /// (#1404 M0.x). Test / observability.
    #[doc(hidden)]
    #[must_use]
    pub fn driven_gc_reclaimed(&self) -> u64 {
        self.driven_gc_reclaimed.load(Ordering::Acquire)
    }

    /// Number of resident version-chain keys currently in `versions`
    /// (#1404 M0.x). Test / observability — the resident MVCC working-set
    /// proxy the drain bounds.
    #[doc(hidden)]
    #[must_use]
    pub fn resident_version_keys(&self) -> usize {
        self.versions.len()
    }

    /// #1404 M0.x — the frontier-advance drain hook. Bumps the commit counter
    /// and, when it crosses `gc_drive_interval`, drives ONE `gc()` pass (which
    /// reclaims superseded versions ≤ `oldest_active_snapshot`, INV-DRAIN
    /// preserved). Called on the commit path AFTER Phase 3 releases (the
    /// frontier is advanced + the commit locks are dropped), so the drain never
    /// runs inside a commit lock. No-op when the driver is disabled
    /// (`gc_drive_interval == 0`). Cheap on the fast path: one relaxed
    /// fetch_add + one compare; the `gc()` scan runs only on the ~1-in-interval
    /// commit that crosses the watermark.
    fn maybe_drive_gc(&self) {
        if self.gc_drive_interval == 0 {
            return;
        }
        let n = self.commits_since_gc.fetch_add(1, Ordering::Relaxed) + 1;
        if n < self.gc_drive_interval {
            return;
        }
        // Reset the counter BEFORE the pass so a concurrent committer that
        // also crosses the watermark doesn't double-drive; a small race that
        // drives one extra/fewer pass is harmless (gc() is idempotent w.r.t.
        // correctness — it only reclaims already-unreachable versions).
        self.commits_since_gc.store(0, Ordering::Relaxed);
        let stats = self.gc();
        self.driven_gc_passes.fetch_add(1, Ordering::AcqRel);
        self.driven_gc_reclaimed
            .fetch_add(stats.reclaimed, Ordering::AcqRel);
    }

    /// Construct a manager that durably logs every committing
    /// transaction's write-set through `wal` before installing.
    ///
    /// Payload format for the emitted `WalRecordType::Commit` record
    /// (little-endian, append-only so recovery can stream it):
    ///
    /// ```text
    ///   commit_lsn  u64
    ///   n_writes    u32
    ///   for each write:
    ///     key       u64       // MvccKey (tagged with REL_TAG_BIT for rels)
    ///     kind      u8        // 0 = tombstone, 1 = put
    ///     value_len u32       // 0 if kind == tombstone
    ///     value     [u8; value_len]
    /// ```
    ///
    /// The record's `txn_id` / `tenant_id` / `lsn` (WAL LSN — distinct
    /// from the MVCC commit LSN) are carried in the WAL header.
    /// Crash recovery (M2.e) reads this payload back into MVCC chains
    /// per ADR-018.
    #[must_use]
    pub fn with_wal(wal: WalHandle) -> Self {
        Self {
            counter: LsnCounter::new(),
            visible: AtomicU64::new(Lsn::ZERO.raw()),
            versions: DashMap::new(),
            tenant_chain_keys: DashMap::new(),
            active: DashMap::new(),
            next_txn_id: AtomicU64::new(1),
            begin_generation: AtomicU64::new(0),
            commit_gate: Mutex::new(()),
            install_order: Mutex::new(Lsn::ZERO.raw()),
            install_cv: Condvar::new(),
            wal: parking_lot::RwLock::new(Some(wal)),
            durability_lookup: parking_lot::RwLock::new(None),
            checkpoint_lock: parking_lot::RwLock::new(()),
            commits_since_gc: AtomicU64::new(0),
            gc_drive_interval: 0,
            driven_gc_passes: AtomicU64::new(0),
            driven_gc_reclaimed: AtomicU64::new(0),
        }
    }

    /// Attach a WAL handle after crash recovery has replayed into this manager.
    ///
    /// Durable bootstrap must recover and truncate any torn tail before a writer
    /// opens the WAL directory. The manager's MVCC state is already populated by
    /// then, so the handle is installed in-place for subsequent commits.
    pub fn attach_wal(&self, wal: WalHandle) {
        *self.wal.write() = Some(wal);
    }

    /// ADR-034 §Slice D — attach a per-tenant [`TenantDurabilityLookup`]
    /// (typically the `SystemCatalog`) so the commit path can dispatch
    /// on tier at commit time.
    ///
    /// When unset (the default; see [`Self::new`] and
    /// [`Self::with_wal`]), every commit runs under Strict / T1 —
    /// the pre-ADR-034 default. Setting this resolver post-construction
    /// is the expected v1.0 wiring path: construct TxnManager → open
    /// WAL → construct & bootstrap catalog → attach the catalog as the
    /// tier resolver.
    ///
    /// Uses `Arc<dyn ...>` rather than generics so the resolver can
    /// be swapped at runtime (e.g., tests that want to inject a
    /// mock resolver). The trait object overhead is one vtable
    /// pointer dispatch per commit — amortised by the WAL
    /// microsecond-scale cost.
    pub fn set_durability_lookup(&mut self, lookup: Arc<dyn TenantDurabilityLookup>) {
        self.attach_durability_lookup(lookup);
    }

    /// Attach a durability resolver through a shared manager handle.
    ///
    /// Durable bootstrap constructs the manager before WAL recovery and holds it
    /// in an `Arc`, then installs the catalog resolver after recovery and
    /// writer attachment.
    pub fn attach_durability_lookup(&self, lookup: Arc<dyn TenantDurabilityLookup>) {
        *self.durability_lookup.write() = Some(lookup);
    }

    /// ADR-034 §Slice D — resolve the effective durability tier for
    /// `tenant` at commit time.
    ///
    /// Short-circuits `TenantId::SYSTEM` to Strict regardless of the
    /// resolver's answer (I-D7: SYSTEM is T1-enforced, non-configurable).
    /// Falls back to Strict if no resolver is set, so test harnesses
    /// that bypass the catalog get the safe default.
    #[inline]
    fn tier_for_commit(&self, tenant: TenantId) -> DurabilityTier {
        if tenant == TenantId::SYSTEM {
            return DurabilityTier::Strict;
        }
        match self.durability_lookup.read().as_ref() {
            Some(lookup) => lookup.durability_tier(tenant),
            None => DurabilityTier::Strict,
        }
    }

    fn wal_format_version(&self) -> u16 {
        self.wal.read().as_ref().map_or(
            crate::wal::segment::CURRENT_WAL_FORMAT_VERSION,
            WalHandle::format_version,
        )
    }

    /// Current *visible* LSN watermark — the highest commit_lsn whose
    /// install loop has finished. This is what readers source their
    /// snapshot from; it is NOT the allocator's value (see invariant
    /// 7). For observability, tests, and benches.
    #[inline]
    #[must_use]
    pub fn current_lsn(&self) -> Lsn {
        Lsn::new(self.visible.load(Ordering::Acquire))
    }

    /// SVC-1 / #849 / ADR-229 — freeze the commit path for a consistent
    /// checkpoint capture. Acquires the checkpoint/commit WRITE guard:
    /// blocks until every in-flight commit has completed Phase 3
    /// (`visible.store`, WAL-durable) and prevents any new commit from
    /// entering Phase 1 while the returned guard is held.
    ///
    /// The checkpoint producer holds this guard across BOTH the frontier
    /// read (`current_lsn`) AND the full-state capture (MVCC, page images,
    /// allocator advances), so all owners are captured against ONE
    /// quiescent instant. No commit can allocate an id absent from the
    /// snapshot (the BLOCK-1 skew) nor leave a not-yet-WAL-durable page
    /// image in the snapshot (the BLOCK-2 phantom). Drop the guard to
    /// resume commits.
    ///
    /// NOT for the hot path: only the background / shutdown checkpoint
    /// producer calls this (design-v2 §4.1). Held for the capture window
    /// only.
    pub fn checkpoint_freeze(&self) -> parking_lot::RwLockWriteGuard<'_, ()> {
        self.checkpoint_lock.write()
    }

    /// Test-only: acquire the checkpoint/commit READ guard (the side a
    /// commit holds for its full span). Exposed so a test can prove the
    /// mutual exclusion with `checkpoint_freeze` (BLOCK-2) without driving
    /// a full concurrent commit. Production code takes this guard
    /// implicitly inside `commit_with_bundle_writes`.
    #[doc(hidden)]
    pub fn __test_commit_read_guard(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.checkpoint_lock.read()
    }

    /// Test hook: the allocator's current value, which leads
    /// `current_lsn` during an in-flight commit. Use sparingly —
    /// tests that observe `allocator_lsn` advancing ahead of
    /// `current_lsn` during a commit are observing an implementation
    /// detail, not a contract.
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn allocator_lsn(&self) -> Lsn {
        self.counter.current()
    }

    /// Begin a new transaction at the current LSN snapshot.
    ///
    /// Two-phase publish (invariant 6): we insert a `Lsn::MAX`
    /// sentinel into `active` BEFORE reading `counter.current()`, then
    /// upgrade the entry to the captured snapshot. A concurrent
    /// `gc()` either observes the sentinel (and is conservative) or
    /// observes the finalized snapshot (and anchors to it). Either
    /// way, a captured snapshot cannot be outrun by a counter advance
    /// in the window between read and publish.
    ///
    /// Reusing `Lsn::MAX` as the pending-begin marker is safe because
    /// `active` holds snapshot LSNs, not version expiration LSNs: a
    /// real snapshot can never equal `Lsn::MAX` (that would mean "I
    /// can see every future commit"), so the sentinel is unambiguous.
    pub fn begin(&self, tenant: TenantId) -> Transaction<'_> {
        let (txn_id, snapshot) = self.begin_inner();
        Transaction {
            manager: ManagerRef::Borrowed(self),
            txn_id,
            tenant_id: tenant,
            snapshot,
            writes: HashMap::new(),
            sidechannel_writes: Vec::new(),
            allocator_advances: Vec::new(),
            vector_pages: Vec::new(),
            idempotency_bindings: Vec::new(),
            acl_grants: Vec::new(),
            mutation_log: TxnMutationLog::new(),
            state: TxnState::Active,
        }
    }

    /// ADR-197 §Decision layer (1): begin a transaction that OWNS its
    /// manager via an `Arc` clone, yielding an [`OwnedTxn`]
    /// (`Transaction<'static>`) the caller can hold across `await`
    /// points and move between threads (`Send`). The Bolt explicit-
    /// transaction handler (`arcgraph-mcp`) holds one of these for the
    /// lifetime of a BEGIN…COMMIT/ROLLBACK.
    ///
    /// Semantically identical to [`Self::begin`] — same snapshot
    /// capture (invariant 6 two-phase publish), same
    /// `active`-set anchoring, same commit/abort machinery. The ONLY
    /// difference is `ManagerRef::Owned` vs `Borrowed`, so the
    /// transaction can outlive a borrow of `self`.
    ///
    /// Takes `&Arc<Self>` (not `self`) so the caller's existing
    /// `Arc<TxnManager>` is cheaply cloned into the transaction; the
    /// returned [`OwnedTxn`] keeps the manager alive for its lifetime.
    pub fn begin_owned(self: &Arc<Self>, tenant: TenantId) -> OwnedTxn {
        let (txn_id, snapshot) = self.begin_inner();
        OwnedTxn {
            inner: Transaction {
                manager: ManagerRef::Owned(Arc::clone(self)),
                txn_id,
                tenant_id: tenant,
                snapshot,
                writes: HashMap::new(),
                sidechannel_writes: Vec::new(),
                allocator_advances: Vec::new(),
                vector_pages: Vec::new(),
                idempotency_bindings: Vec::new(),
                acl_grants: Vec::new(),
                mutation_log: TxnMutationLog::new(),
                state: TxnState::Active,
            },
        }
    }

    /// Shared begin-bookkeeping for [`Self::begin`] +
    /// [`Self::begin_owned`]: allocate a txn id, publish the
    /// `Lsn::MAX` pending sentinel, capture the snapshot, and upgrade
    /// the `active` entry (invariant 6 two-phase publish). Returns
    /// `(txn_id, snapshot)`.
    #[inline]
    fn begin_inner(&self) -> (u64, Lsn) {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::AcqRel);
        // #1404 M0.x FIX-C — begin-generation FENCE: bump BEFORE the two-phase
        // publish (before the sentinel insert AND the `visible` read). A
        // concurrent `oldest_active_snapshot` reads the gen before + after its
        // `active` scan; if it advanced (this begin was in-flight during the
        // scan), it clamps to `Lsn::ZERO` — protecting the version this begin's
        // about-to-be-published snapshot needs, EVEN IF the sentinel below
        // landed in a shard the scan had already passed (the shard-late-insert
        // residual a clamp-on-observed-pending alone leaves open). AcqRel so
        // this bump is ordered before the inserts that follow.
        self.begin_generation.fetch_add(1, Ordering::AcqRel);
        self.active.insert(txn_id, Lsn::MAX);
        let snapshot = Lsn::new(self.visible.load(Ordering::Acquire));
        self.active.insert(txn_id, snapshot);
        (txn_id, snapshot)
    }

    /// Floor of snapshots held by transactions currently live.
    ///
    /// Entries equal to `Lsn::MAX` are pending begins (see `begin`
    /// invariant 6) — they have not yet resolved a concrete snapshot
    /// but MUST still protect versions from reclamation.
    ///
    /// # #1404 M0.x FIX-C — begin-vs-gc race fence (the default-on gc DRIVER)
    ///
    /// The M0.x gc DRIVER runs `gc()` (hence this) CONCURRENTLY with `begin`s
    /// on the ingest path. A begin publishes its snapshot in two phases
    /// (`begin_inner`: sentinel insert → read `visible` → snapshot insert), so
    /// there is a window where the begin's eventual snapshot is not yet in
    /// `active`. The pre-M0.x code IGNORED a pending sentinel whenever ANY
    /// concrete floor existed (`any_pending && floor.is_none()` only) — so a
    /// begin whose eventual snapshot is BELOW the concrete floor was NOT
    /// protected → gc reclaimed a version it needs → SILENT WRONG-READ
    /// (reproduced: 415 None-reads/2.38M at interval=1). Two fences close it:
    ///
    /// 1. **Clamp to `Lsn::ZERO` on ANY observed pending sentinel** (not only
    ///    when all are pending). A pending begin's eventual snapshot can be
    ///    arbitrarily low, so gc must reclaim nothing this pass and retry once
    ///    the begin has published its concrete snapshot.
    /// 2. **begin-generation fence for the shard-late-insert residual:** a
    ///    begin that inserted its sentinel into a shard this scan already
    ///    passed is invisible to the scan (DashMap shard-local visibility), so
    ///    fence (1) alone leaves a residual (~15/run in the repro). We snapshot
    ///    `begin_generation` BEFORE the scan and re-read it AFTER; if it
    ///    advanced, a begin was in-flight during the scan → clamp to `Lsn::ZERO`
    ///    regardless of what the scan observed. `begin_inner` bumps the gen
    ///    BEFORE its inserts, so any concurrent begin is caught here.
    ///
    /// Returns `counter.current()`/`visible` only when `active` is empty AND no
    /// begin raced (no live txns, nothing to protect).
    pub fn oldest_active_snapshot(&self) -> Lsn {
        // Fence (2), part A — the generation before the scan.
        let gen_before = self.begin_generation.load(Ordering::Acquire);
        let mut floor: Option<u64> = None;
        let mut any_pending = false;
        for entry in self.active.iter() {
            let s = entry.value().raw();
            if s == Lsn::MAX.raw() {
                any_pending = true;
                continue;
            }
            floor = Some(floor.map_or(s, |f| f.min(s)));
        }
        // Fence (2), part B — a begin published (or is mid-publish) during the
        // scan; its sentinel may have landed in a shard we already passed.
        // Protect everything this pass.
        let gen_after = self.begin_generation.load(Ordering::Acquire);
        if gen_before != gen_after {
            return Lsn::ZERO;
        }
        // Fence (1) — ANY observed pending sentinel: a pending begin's eventual
        // snapshot can be below the concrete floor, so protect everything.
        if any_pending {
            return Lsn::ZERO;
        }
        Lsn::new(floor.unwrap_or_else(|| self.visible.load(Ordering::Acquire)))
    }

    /// Register `(tenant, key)` in the per-tenant chain index
    /// (`tenant_chain_keys`). Idempotent (safe to call repeatedly for
    /// the same `(tenant, key)`); the underlying `DashSet::insert` is
    /// a no-op when the key is already present.
    ///
    /// Issue #238 (PR #236 MED-1 closure): every push path on
    /// `versions` MUST follow with a `register_chain_key` so
    /// [`Self::for_each_visible_record`] and
    /// [`Self::tenants_with_chains`] iterate per-tenant rather than
    /// scanning every shard.
    ///
    /// Cost: one `DashMap::entry` lookup + one `DashSet::insert`
    /// (under the same shard), both lock-free in the common
    /// already-present case.
    #[inline]
    fn register_chain_key(&self, tenant: TenantId, key: MvccKey) {
        self.tenant_chain_keys
            .entry(tenant)
            .or_default()
            .insert(key);
    }

    /// M2-15: Reclaim versions that no active snapshot can observe.
    ///
    /// Safety: a version `V` is reachable by an active snapshot `S`
    /// iff `V.created_lsn ≤ S < V.expired_lsn`. With
    /// `A = oldest_active_snapshot`, any `V` with
    /// `V.expired_lsn ≤ A` is unreachable by every active `S ≥ A`.
    /// Live versions have `expired_lsn = MAX` and are preserved.
    pub fn gc(&self) -> GcStats {
        let anchor = self.oldest_active_snapshot();
        let anchor_raw = anchor.raw();
        let mut stats = GcStats {
            anchor,
            ..GcStats::default()
        };
        let keys: Vec<VersionKey> = self.versions.iter().map(|e| *e.key()).collect();
        for key in keys {
            stats.scanned_keys += 1;
            let empty_after = {
                let mut entry = match self.versions.get_mut(&key) {
                    Some(e) => e,
                    None => continue,
                };
                let before = entry.len();
                entry.retain(|v| v.expired_lsn.raw() > anchor_raw);
                stats.reclaimed += (before - entry.len()) as u64;
                entry.is_empty()
            };
            if empty_after
                && self
                    .versions
                    .remove_if(&key, |_, chain| chain.is_empty())
                    .is_some()
            {
                // Count only the keys that were actually removed.
                // A racing commit may have re-populated the chain
                // between `drop(entry)` and `remove_if`, in which
                // case the predicate returns false and nothing is
                // pruned — so nothing should be counted.
                stats.pruned_keys += 1;
                // Issue #238 / PR #243 MED-1 (round-2 closure): the
                // per-tenant chain index is monotone-growing — see
                // the `tenant_chain_keys` field rustdoc for the
                // false-negative race rationale. We deliberately
                // do NOT drop the index entry here.
            }
        }
        stats
    }

    /// Test-only hook mirroring `gc` with a barrier pause between
    /// the `empty_after` check and the `remove_if` prune call. Lets
    /// a test racing a concurrent commit force a predicate-false
    /// outcome deterministically and assert `stats.pruned_keys`
    /// reflects actual removals.
    #[doc(hidden)]
    pub fn gc_with_prune_barrier(
        &self,
        before_remove_if: &std::sync::Barrier,
        after_repopulate: &std::sync::Barrier,
    ) -> GcStats {
        let anchor = self.oldest_active_snapshot();
        let anchor_raw = anchor.raw();
        let mut stats = GcStats {
            anchor,
            ..GcStats::default()
        };
        let keys: Vec<VersionKey> = self.versions.iter().map(|e| *e.key()).collect();
        for key in keys {
            stats.scanned_keys += 1;
            let empty_after = {
                let mut entry = match self.versions.get_mut(&key) {
                    Some(e) => e,
                    None => continue,
                };
                let before = entry.len();
                entry.retain(|v| v.expired_lsn.raw() > anchor_raw);
                stats.reclaimed += (before - entry.len()) as u64;
                entry.is_empty()
            };
            if empty_after {
                before_remove_if.wait();
                after_repopulate.wait();
                if self
                    .versions
                    .remove_if(&key, |_, chain| chain.is_empty())
                    .is_some()
                {
                    stats.pruned_keys += 1;
                    // Issue #238 / PR #243 MED-1 (round-2 closure):
                    // symmetric with `gc()` — index is monotone-growing.
                    // See `tenant_chain_keys` field rustdoc.
                }
            }
        }
        stats
    }

    /// Test/introspection: number of committed versions for a key,
    /// including tombstones that GC has not yet reclaimed.
    #[doc(hidden)]
    pub fn chain_len(&self, tenant: TenantId, key: MvccKey) -> usize {
        self.versions.get(&(tenant, key)).map_or(0, |c| c.len())
    }

    /// Test/introspection (issue #238): number of `MvccKey`s recorded
    /// in the per-tenant chain index for `tenant`. May be larger than
    /// the count of currently-non-empty chains by the index's
    /// staleness contract (see `tenant_chain_keys` field rustdoc).
    #[doc(hidden)]
    pub fn tenant_chain_key_count(&self, tenant: TenantId) -> usize {
        self.tenant_chain_keys.get(&tenant).map_or(0, |s| s.len())
    }

    /// Test/introspection (issue #238): does the per-tenant chain
    /// index contain `(tenant, key)`?
    #[doc(hidden)]
    pub fn tenant_chain_index_contains(&self, tenant: TenantId, key: MvccKey) -> bool {
        self.tenant_chain_keys
            .get(&tenant)
            .is_some_and(|s| s.contains(&key))
    }

    /// Test/benchmark hook (issue #238): the **pre-#238 legacy**
    /// `for_each_visible_record` shape — `DashMap::iter()` walks every
    /// shard then `.filter()`s by `TenantId`. Used by the Tier-1
    /// stress test (`tests/m4_41_chain_index_stress.rs`) to compare
    /// wall-time against the post-#238 path under the SAME memory
    /// load (avoiding the cost of a separate process-revert cycle).
    ///
    /// Strictly equivalent in observable behaviour to
    /// [`Self::for_each_visible_record`]; the only difference is
    /// algorithmic complexity (`O(N_total)` per call vs. `O(N_tenant)`
    /// per call). Production code MUST NOT call this method;
    /// `#[doc(hidden)]` keeps it off the public API surface.
    ///
    /// **TODO(#246): retire after M4-41-impl wires the production
    /// rebuild path** — once a steady-state perf-pin lands on the
    /// indexed path with measured numbers, the legacy comparator is
    /// no longer load-bearing for any test gate.
    #[doc(hidden)]
    pub fn for_each_visible_record_legacy_for_test(
        &self,
        tenant: TenantId,
        snapshot: Lsn,
        mut callback: impl FnMut(MvccKey, &[u8]),
    ) {
        let keys: Vec<VersionKey> = self
            .versions
            .iter()
            .filter(|e| e.key().0 == tenant)
            .map(|e| *e.key())
            .collect();
        for vk in keys {
            if let Some(chain) = self.versions.get(&vk) {
                for v in chain.iter().rev() {
                    if v.visible_to(snapshot) {
                        if let Some(bytes) = &v.value {
                            callback(vk.1, bytes.as_ref());
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Test/introspection: number of currently active transactions.
    #[doc(hidden)]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// M4-41 cold-start rebuild support (per ADR-038 amendment-06
    /// §D-25.1). Returns the unique set of tenants that have at least
    /// one MVCC chain in this manager, sorted by raw `TenantId` for
    /// deterministic iteration.
    ///
    /// Used by [`crate::recovery::stats_rebuild::rebuild_all_tenant_stats`]
    /// to enumerate tenants for per-tenant rebuild after WAL replay.
    /// Per-tenant rebuild is the per-amendment-06 §2.5.3 cross-tenant
    /// query-during-recovery shape: each tenant's stats are independent;
    /// the rebuild walks per-tenant slices without taking a global
    /// barrier.
    ///
    /// # PERF
    ///
    /// `O(K)` where `K` is the number of tenants with at least one
    /// chain (issue #238 — closes O(N_total) DashMap-shard scan from
    /// PR #236 MED-1). Reads from the per-tenant chain index
    /// (`tenant_chain_keys`).
    ///
    /// Per the index's monotone-growing contract (PR #243 MED-1
    /// closure — see `tenant_chain_keys` field rustdoc), tenants are
    /// **never removed** from the index. A tenant whose chains have
    /// all been reclaimed by GC, or all became empty via
    /// `rollback_writes`, still appears here. Both cases are benign —
    /// the rebuild driver's per-tenant `for_each_visible_record` walk
    /// is a no-op for tenants with only empty (or fully-reclaimed)
    /// chains.
    #[must_use]
    pub fn tenants_with_chains(&self) -> Vec<TenantId> {
        let mut out: Vec<TenantId> = self.tenant_chain_keys.iter().map(|e| *e.key()).collect();
        out.sort_by_key(|t| t.raw());
        out
    }

    /// M4-41 cold-start rebuild support (per ADR-038 amendment-06
    /// §D-25.1). Invoke `callback(mvcc_key, &bytes)` for every MVCC key
    /// in `tenant` whose **latest** version visible at `snapshot` is a
    /// live (non-tombstone) value.
    ///
    /// MVCC visibility filter mirrors [`Self::read_at`]:
    /// `created_lsn ≤ snapshot < expired_lsn`. Within a chain, the
    /// FIRST visible version (walking back-to-front, i.e., the most
    /// recently committed visible version) is the "latest visible";
    /// older visible versions are skipped. If the latest visible
    /// version is a tombstone (`value == None`) the callback is not
    /// invoked for that key — the entity was deleted by `snapshot`.
    ///
    /// Iteration is per-tenant (`(tenant, _)` keys only); cross-tenant
    /// pollution is structurally impossible because the version map's
    /// outer key includes `TenantId`.
    ///
    /// The keys are collected up-front so the DashMap shard read guard
    /// is not held across user-controlled `callback` invocations.
    /// Memory cost: `O(visible_keys_in_tenant)` (one `(TenantId, u64)`
    /// per chain). At v1.0 alpha tenant sizes (~1M nodes per tenant per
    /// amendment-06 §D-25.2 watermark justification) this is ~16 MB
    /// peak — well inside the recovery budget.
    ///
    /// # PERF
    ///
    /// `O(N_tenant)` per call — one `DashMap::get` lookup on
    /// `tenant_chain_keys` (the per-tenant chain index) followed by an
    /// iteration over that tenant's `MvccKey` set. Closes the
    /// `O(K² × N_per_tenant)` aggregate cost from PR #236 MED-1: the
    /// aggregate cost when called per-tenant inside
    /// [`crate::recovery::stats_rebuild::rebuild_all_tenant_stats`] is
    /// now `O(K × N_per_tenant) = O(N_total)`, restoring the
    /// per-tenant independence assumed by ADR-038 amendment-06 §D-25.2
    /// process-restart budget.
    ///
    /// **What's pinned at v1.0-alpha:** the Tier-1 relative
    /// algorithmic gate (`tests/m4_41_chain_index_stress.rs::
    /// rebuild_aggregate_walk_at_K50_N50K_index_strictly_faster_than_legacy`
    /// — index path is ≥ 1.5× faster than the pre-#238
    /// DashMap-scan-and-filter shape, cache-warm best-of-10, on
    /// dev hardware). Phase 4.3 reverse-test on the
    /// amendment-spec'd Tier-2 shape (K=50 N=200K — 200K per tenant
    /// sub-watermark vs §D-25.2's per-tenant 10M trigger; 10M
    /// aggregate is a memory-fit choice, NOT the watermark)
    /// pins the budget regression: reverting `for_each_visible_record`
    /// to the legacy shape on the SAME host yields a `K×`-slower
    /// distribution than the indexed path.
    ///
    /// **Absolute 5 s p99 budget (post-#247):** the amendment-06
    /// §D-25.2 per-tenant 10M-node watermark IS PINNED at v1.0-alpha
    /// via the parallel acceptance gate
    /// `tests/m4_41_chain_index_stress_K50_parallel.rs::
    /// rebuild_all_tenant_stats_at_K50_N200K_parallel_within_5s_p99`
    /// (issue #247 closure, PR #253). The amendment's multi-tenant
    /// scaling paragraph specified that per-tenant rebuild be
    /// parallelised across `min(num_tenants, num_cpus)` threads;
    /// `recovery::stats_rebuild::rebuild_all_tenant_stats` now drives
    /// the per-tenant fan-out via `rayon::iter::IntoParallelIterator`,
    /// closing the 5 s budget on commodity dev hardware (clean-host
    /// re-verification follow-up at issue #251 also closed by PR
    /// #253 — see amendment-06 §7.1). The test-internal threshold is
    /// strictly tighter than the amendment's 5 s budget (2.5 s) to
    /// load-bear the parallelism property under the Phase 4.3
    /// reverse-test discipline (PR #253 round-2 reviewer M-1
    /// closure; serial driver max_of_10 ≈ 4.4-4.8 s on uncongested
    /// commodity hardware passes the 5 s budget but fails the 2.5 s
    /// gate). The K=50 N=1M (1M per tenant — still under the
    /// per-tenant 10M watermark; 50M aggregate above the comfortable
    /// sub-watermark serial-rebuild zone) shape is a v1.0-GA
    /// characterisation point tracked at issue #249.
    ///
    /// The index may carry stale empty-chain entries (post-rollback
    /// or post-GC of a fully-reclaimed chain — the index is
    /// monotone-growing per PR #243 MED-1 closure); the visibility
    /// filter inside the inner loop turns those into zero callback
    /// invocations, preserving correctness (per `tenant_chain_keys`
    /// field rustdoc).
    pub fn for_each_visible_record(
        &self,
        tenant: TenantId,
        snapshot: Lsn,
        mut callback: impl FnMut(MvccKey, &[u8]),
    ) {
        self.for_each_visible_record_with_created_lsn(tenant, snapshot, |key, bytes, _| {
            callback(key, bytes);
        });
    }

    /// [`Self::for_each_visible_record`] carrying each visible version's
    /// authoritative `created_lsn`.
    ///
    /// # Why this exists (issue #1616)
    ///
    /// An MVCC value's *payload* is not a reliable carrier of
    /// `created_lsn`. On the v8 / non-delta commit path, the CRUD codecs
    /// store record bytes with that field in its canonical `Lsn::ZERO`
    /// placeholder form. The commit LSN is stamped into the record-page
    /// slot at install time and onto the MVCC version, not into that
    /// stored payload.
    ///
    /// This is not a universal invariant for payload bytes: delta-mode
    /// commit and the v6/v9 physical base loaders can seed chains with
    /// stamped payloads. The rule is directional: the version is always
    /// authoritative, while the payload is only sometimes right.
    ///
    /// A recovery pass that re-derives physical state from MVCC must
    /// therefore take the visibility LSN from the version. On paths where
    /// the two disagree, using the payload yields `Lsn::ZERO`, making the
    /// derived slot visible to snapshots that predate its creation and
    /// feeding a descending LSN into the v9 base loader's ordered replay.
    pub fn for_each_visible_record_with_created_lsn(
        &self,
        tenant: TenantId,
        snapshot: Lsn,
        mut callback: impl FnMut(MvccKey, &[u8], Lsn),
    ) {
        self.for_each_visible_record_state(
            tenant,
            snapshot,
            |key, value, created_lsn, _previous_value| {
                if let Some(bytes) = value {
                    callback(key, bytes, created_lsn);
                }
            },
        );
    }

    /// Walk each key's latest state visible at `snapshot`, including deletes.
    ///
    /// `previous_value` is populated only for a tombstone and names the most
    /// recent earlier value in the same chain. The v6 authority rebuild uses
    /// that value to distinguish record keys from other tenant-local MVCC
    /// namespaces before it permanently clears an arithmetic record slot.
    pub(crate) fn for_each_visible_record_state(
        &self,
        tenant: TenantId,
        snapshot: Lsn,
        mut callback: impl FnMut(MvccKey, Option<&[u8]>, Lsn, Option<&[u8]>),
    ) {
        // Look up the tenant's chain-key set in O(1). Tenants with no
        // chains return early — no allocation, no iteration.
        let Some(key_set) = self.tenant_chain_keys.get(&tenant) else {
            return;
        };
        // Collect keys up-front so neither the outer DashMap shard
        // guard nor the inner DashSet shard guard is held across the
        // user-controlled `callback` invocations (mirrors the
        // pre-#238 contract: callbacks never run while a shard guard
        // is held).
        //
        // Memory cost: one `MvccKey` (`u64`) per chain in the tenant.
        // At v1.0-alpha tenant ceiling (~1M chains) this is ~8 MB
        // peak per call — well inside the recovery budget.
        let keys: Vec<MvccKey> = key_set.iter().map(|e| *e.key()).collect();
        drop(key_set);
        for key in keys {
            let vk = (tenant, key);
            if let Some(chain) = self.versions.get(&vk) {
                for (index, v) in chain.iter().enumerate().rev() {
                    if v.visible_to(snapshot) {
                        let previous_value = v.value.is_none().then(|| {
                            chain[..index]
                                .iter()
                                .rev()
                                .find_map(|prior| prior.value.as_deref())
                        });
                        callback(
                            key,
                            v.value.as_deref(),
                            v.created_lsn,
                            previous_value.flatten(),
                        );
                        break;
                    }
                }
            }
        }
    }

    /// #849 B3(a) — coalesce a commit's staged page snapshots by
    /// `(kind, page_id)`, keeping the LAST snapshot of each page
    /// (last-write-wins within the atomic bundle).
    ///
    /// `install_create` (`crud.rs::snapshot_record_page`) and the
    /// primary index's `upsert_deferred` push a FULL `PAGE_SIZE` page
    /// snapshot per record / key mutation, so a B-record commit whose
    /// records land on one slotted page previously staged that page B
    /// times — `B × 8 KiB` of WAL for `~8 KiB` of final state (measured
    /// ~16.7 KiB of WAL **per record**, constant across batch sizes 1
    /// → 1000, in `tests/durable_ingest_throughput_849.rs`; the
    /// `#849` 2.5 GiB-for-140 K-records observation). Every snapshot of
    /// a given page in one commit carries the same `commit_lsn` and the
    /// LAST is the cumulative post-image (slotted inserts only grow a
    /// page; an in-commit update / delete leaves the final slot state
    /// in the last snapshot), and replay overwrites the page store by
    /// `(kind, page_id)` (`wal/replay.rs`) — so the last snapshot alone
    /// reconstructs identical state. The earlier snapshots are pure
    /// write amplification. After coalescing, WAL volume scales with
    /// PAGES TOUCHED, not RECORDS WRITTEN, with NO durability change:
    /// the bundle is still ONE atomic append + fsync, and acked ==
    /// fsync'd-before-ack still holds.
    fn coalesce_staged_pages(
        staged: &[StagedEmit],
        tenant_id: TenantId,
    ) -> Vec<(
        crate::wal::bundle::BundlePageKind,
        arcgraph_core::PageId,
        TenantId,
        Box<[u8; arcgraph_core::PAGE_SIZE]>,
    )> {
        // Index of the LAST snapshot for each (kind, page_id).
        let mut last_idx: HashMap<
            (crate::wal::bundle::BundlePageKind, arcgraph_core::PageId),
            usize,
        > = HashMap::with_capacity(staged.len());
        for (i, e) in staged.iter().enumerate() {
            last_idx.insert((e.kind, e.page_id), i);
        }
        // Emit kept entries in ascending original-index order for a
        // deterministic bundle layout (replay is order-independent
        // across distinct page_ids; this keeps test / operator-log
        // output stable + mirrors the pre-coalesce ordering for the
        // common single-snapshot-per-page case).
        let mut keep: Vec<usize> = last_idx.into_values().collect();
        keep.sort_unstable();
        keep.into_iter()
            .map(|i| {
                let e = &staged[i];
                (e.kind, e.page_id, tenant_id, e.bytes.clone())
            })
            .collect()
    }

    /// #1200: collapse duplicate sidechannel writes last-write-wins by
    /// `(tenant_id, key)`, in place. Surviving entries are emitted in
    /// ascending last-occurrence-index order (mirrors
    /// [`Self::coalesce_staged_pages`] for a deterministic bundle
    /// layout; replay is order-independent across distinct keys).
    ///
    /// This is the sidechannel sibling of [`Self::coalesce_staged_pages`]
    /// (which coalesces staged PAGES last-wins by `(kind, page_id)`):
    /// `coalesce_staged_pages` conspicuously did NOT touch
    /// `sidechannel_writes`, and that gap IS the #1200 defect.
    ///
    /// **Why this is required.** A single B-tree commit that crosses
    /// the height-2→3 boundary (>51,765 keys, leaf cap 203 / internal
    /// cap 254 — `primary_index.rs`) fires
    /// [`crate::primary_index::PrimaryIndex::grow_root`] ≥ 2 times. Each
    /// grow_root pushes ONE `(SYSTEM, PRIMARY_INDEX_ROOT_KEY)`
    /// `SideChannelWrite` onto the shared vec (no dedup — ADR-032
    /// Slice-2 retired the `pending_root: AtomicU64` slot that used to
    /// coalesce these by construction). The whole commit allocates ONE
    /// `commit_lsn`, so ≥ 2 root-pointer writes land at the SAME
    /// commit_lsn. That breaks two things, both fixed here by collapsing
    /// to the LAST write per `(tenant, key)`:
    ///
    /// 1. **Debug-assert (DEBUG).** The 2nd
    ///    [`Self::apply_sidechannel_mvcc_write`] at the same commit_lsn
    ///    trips the `created_lsn < commit_lsn` invariant (the invariant
    ///    is CORRECT and load-bearing — we do NOT relax it; we remove
    ///    the duplicate that violates it).
    /// 2. **Latent replay root-corruption (RELEASE).** On crash + WAL
    ///    replay, [`Self::apply_replay_mvcc_write`]'s idempotency skip
    ///    keys ONLY on `last.created_lsn == commit_lsn` (value-blind,
    ///    "Lemma I1"). sc[0] (the INTERMEDIATE root) applies; sc[1] (the
    ///    FINAL root) matches the created_lsn → `Idempotent` → SKIPPED.
    ///    The durable MVCC root is stranded at the intermediate root and
    ///    ~78% of index keys become unreachable post-recovery (#1200).
    ///
    /// **Last-wins is the correct semantics.** `grow_root` appends to
    /// `sc_writes` in split-cascade order AFTER publishing the new root
    /// to `root_cache` (`primary_index.rs`), so within one commit the
    /// LAST `(SYSTEM, ROOT_KEY)` push is the outermost/final root — the
    /// one live reads and a correct replay must converge on. Keeping the
    /// last entry (and dropping the earlier intermediate-root writes)
    /// restores exactly the last-wins invariant the retired AtomicU64
    /// slot used to provide, without resurrecting the slot (which would
    /// regress the ADR-032 Slice-2 bundle-folded design).
    ///
    /// Distinct `(tenant, key)` writes are preserved (no collapse).
    /// This runs once per commit on a vec that is typically 0–2 entries
    /// (one per grow_root); the HashMap pass is O(n) and off the
    /// per-record hot path (grow_root fires at most once per insert, and
    /// the 2nd only past 51,765 keys), so the cost is negligible.
    fn coalesce_sidechannel_writes(sidechannel_writes: &mut Vec<SideChannelWrite>) {
        if sidechannel_writes.len() < 2 {
            // Common case: 0 or 1 sidechannel write — nothing to
            // collapse. Avoids a HashMap allocation on every commit.
            return;
        }
        // Index of the LAST write for each (tenant_id, key).
        let mut last_idx: HashMap<(TenantId, MvccKey), usize> =
            HashMap::with_capacity(sidechannel_writes.len());
        for (i, sc) in sidechannel_writes.iter().enumerate() {
            last_idx.insert((sc.tenant_id, sc.key), i);
        }
        if last_idx.len() == sidechannel_writes.len() {
            // No duplicates — leave the vec untouched (preserves the
            // existing ordering for the no-grow_root-collision path).
            return;
        }
        // Keep only the last-occurrence index per key, emitted in
        // ascending original-index order for a deterministic bundle
        // layout (mirrors `coalesce_staged_pages`). Replay is
        // order-independent across distinct keys; this keeps test /
        // operator-log output stable.
        let mut keep: Vec<usize> = last_idx.into_values().collect();
        keep.sort_unstable();
        let mut kept_iter = keep.iter().copied().peekable();
        let mut write_pos = 0usize;
        for read_pos in 0..sidechannel_writes.len() {
            if kept_iter.peek() == Some(&read_pos) {
                kept_iter.next();
                sidechannel_writes.swap(write_pos, read_pos);
                write_pos += 1;
            }
        }
        sidechannel_writes.truncate(write_pos);
    }

    /// Bundle-aware commit. The MVCC kernel's core three-phase commit
    /// with an injection point between Phase 1 (commit_gate-held
    /// silent install) and Phase 2 (`wal.append(CommitBundle)`) — the
    /// `builder` closure runs with the allocated `commit_lsn`,
    /// performs caller-side work that depends on that LSN (slotted-
    /// page installs, index upserts via `*_deferred` siblings), and
    /// returns any staged `IndexPage` snapshots to fold into the
    /// single WAL record.
    ///
    /// **ADR-032 Slice 2.** The builder now also receives a
    /// `&mut Vec<SideChannelWrite>` into which it (or callees like
    /// [`crate::primary_index::PrimaryIndex::grow_root`]) pushes
    /// non-primary-tenant MVCC writes that must ride the outer
    /// commit's CommitBundle atomically. Any writes pre-registered
    /// via [`Transaction::register_sidechannel_mvcc_write`] are
    /// passed in from [`Transaction::commit_with_bundle`]. Phase 2
    /// encodes the bundle with [`encode_commit_bundle_v2`]
    /// (per-entry tenant-id on the wire). Phase 3 applies each
    /// sidechannel write to the MVCC chain via
    /// [`Self::apply_sidechannel_mvcc_write`], AFTER the WAL append
    /// succeeds — on WAL failure the sidechannel writes are NOT
    /// applied (symmetric with the primary rollback path).
    ///
    /// Preserves invariants 1–10 and introduces the "MVCC has root-pointer R
    /// implies page_store has R installed" invariant by construction.
    #[allow(clippy::too_many_arguments)]
    fn commit_with_bundle_writes<F, A, R>(
        &self,
        txn_id: u64,
        tenant_id: TenantId,
        snapshot: Lsn,
        writes: &HashMap<MvccKey, Option<Bytes>>,
        sidechannel_writes: &mut Vec<SideChannelWrite>,
        allocator_advances: &mut Vec<AllocatorAdvance>,
        vector_pages: &mut Vec<VectorPageEntry>,
        // #352 Part 2 (ADR-199): read-only — the bindings are populated
        // by `crud::commit` on the `Transaction` BEFORE this call (they
        // carry no commit_lsn, so unlike `vector_pages` they need no
        // builder-closure access). Encoded into the v6 bundle below.
        idempotency_bindings: &[IdempotencyBindingEntry],
        // #1221 (ADR-218): read-only — the ACL grant/revoke ops are
        // populated by `crud::commit` on the `Transaction` BEFORE this
        // call (like `idempotency_bindings`, they carry no commit_lsn).
        // Encoded into the v8 bundle's `acl_grants` section below in
        // staging (append) order. Encoded into the v8 bundle below.
        acl_grants: &[AclGrantEntry],
        mutation_log: &mut TxnMutationLog,
        builder: F,
        apply: A,
        rollback: R,
    ) -> Result<Lsn>
    where
        F: FnOnce(
            Lsn,
            &mut Vec<SideChannelWrite>,
            &mut Vec<AllocatorAdvance>,
            &mut Vec<VectorPageEntry>,
            &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>>,
        A: FnOnce(&[DeltaOp], Lsn) -> Result<()>,
        R: FnOnce(&mut TxnMutationLog),
    {
        // SVC-1 / #849 / ADR-229 — checkpoint/commit serialization (READ
        // side). Held for the ENTIRE three-phase commit span: Phase 1
        // (alloc + silent install), the builder (page byte-writes), Phase
        // 2 (WAL fsync), and Phase 3 (`visible.store`). A concurrent
        // checkpoint takes the WRITE guard (`checkpoint_freeze`) and thus
        // cannot capture the frontier / allocator / pages while any commit
        // is between its `counter.allocate()` and its `visible.store` —
        // closing the BLOCK-1 allocator-skew id-reuse and BLOCK-2
        // phantom-page-image corruption paths. Read guards do not exclude
        // each other, so commit pipelining (invariant 8) is preserved: the
        // guard only ever blocks a commit while a checkpoint capture holds
        // the write guard (a bounded, paced background window).
        let _checkpoint_read = self.checkpoint_lock.read();
        if writes.is_empty()
            && sidechannel_writes.is_empty()
            && acl_grants.is_empty()
            && mutation_log.delta_intents.is_empty()
        {
            // Read-only / no-op commit: no MVCC install, no WAL
            // record. The builder is NOT invoked because a no-op
            // commit has no allocated commit_lsn to pass and no
            // semantic dependency on post-alloc work — callers that
            // need a builder side-effect should issue a write first.
            // Sidechannel writes pre-registered via
            // `register_sidechannel_mvcc_write` also count as "work";
            // if ANY are pending we fall through and run a normal
            // commit with an empty primary write-set.
            //
            // #1221 (ADR-218): an `acl_grants`-only commit IS work — the
            // ACL write-through (CrudAclWalSink) fires a dedicated
            // single-op commit with an EMPTY MVCC write-set whose sole
            // payload is the v8 `acl_grants` tail. Without this carve-out
            // the ACL op would be silently dropped (never durified) and a
            // bare restart would lose it — the #1221 defect. (Idempotency
            // bindings never reach here: they always ride a node/rel
            // write, so `writes` is non-empty when they are staged.)
            self.active.remove(&txn_id);
            return Ok(Lsn::new(self.visible.load(Ordering::Acquire)));
        }

        let mut apply = Some(apply);
        let mut rollback = Some(rollback);
        let wal_handle = self.wal.read().clone();
        let delta_format = wal_handle
            .as_ref()
            .map(WalHandle::format_version)
            .filter(|version| is_delta_bundle_format(*version));
        let writes_delta = delta_format.is_some();
        mutation_log.delta_mode = writes_delta;

        // Delta bundles must know the exact physical-op count before allocating their
        // contiguous range. Its builder therefore runs inside Phase 1 with a
        // provisional record LSN; DeltaIntent::assign stamps the final
        // commit LSN after allocation. Legacy v8 keeps its established
        // allocate-then-build pipeline.
        let (commit_range, staged_emits) = if writes_delta {
            let _gate = self.commit_gate.lock();
            for key in writes.keys() {
                let vk = (tenant_id, *key);
                if let Some(chain) = self.versions.get(&vk)
                    && let Some(last) = chain.last()
                    && last.created_lsn.raw() > snapshot.raw()
                {
                    return Err(ArcGraphError::MvccConflict {
                        target: format!("key:{key}"),
                    });
                }
            }
            let staged = match builder(
                Lsn::ZERO,
                sidechannel_writes,
                allocator_advances,
                vector_pages,
                mutation_log,
            ) {
                Ok(staged) => staged,
                Err(error) => {
                    rollback.take().expect("rollback closure consumed once")(mutation_log);
                    self.active.remove(&txn_id);
                    return Err(error);
                }
            };
            let range = self
                .counter
                .allocate_range(mutation_log.delta_intents.len());
            let commit_lsn = range.commit_lsn();
            for (key, value) in writes {
                let vk = (tenant_id, *key);
                let mut chain = self.versions.entry(vk).or_default();
                if let Some(last) = chain.last_mut()
                    && last.is_live()
                {
                    last.expired_lsn = commit_lsn;
                }
                chain.push(Version {
                    created_lsn: commit_lsn,
                    expired_lsn: Lsn::MAX,
                    value: value.clone(),
                });
                drop(chain);
                self.register_chain_key(tenant_id, *key);
            }
            (range, staged)
        } else {
            let commit_range = {
                let _gate = self.commit_gate.lock();
                for key in writes.keys() {
                    let vk = (tenant_id, *key);
                    if let Some(chain) = self.versions.get(&vk)
                        && let Some(last) = chain.last()
                        && last.created_lsn.raw() > snapshot.raw()
                    {
                        return Err(ArcGraphError::MvccConflict {
                            target: format!("key:{key}"),
                        });
                    }
                }
                let range = self.counter.allocate_range(1);
                let commit_lsn = range.commit_lsn();
                for (key, value) in writes {
                    let vk = (tenant_id, *key);
                    let mut chain = self.versions.entry(vk).or_default();
                    if let Some(last) = chain.last_mut()
                        && last.is_live()
                    {
                        last.expired_lsn = commit_lsn;
                    }
                    chain.push(Version {
                        created_lsn: commit_lsn,
                        expired_lsn: Lsn::MAX,
                        value: value.clone(),
                    });
                    drop(chain);
                    self.register_chain_key(tenant_id, *key);
                }
                range
            };
            let commit_lsn = commit_range.commit_lsn();
            let staged = match builder(
                commit_lsn,
                sidechannel_writes,
                allocator_advances,
                vector_pages,
                mutation_log,
            ) {
                Ok(staged) => staged,
                Err(error) => {
                    self.wait_for_install_turn(commit_range);
                    {
                        let _gate = self.commit_gate.lock();
                        self.rollback_writes(tenant_id, commit_lsn, writes);
                        rollback.take().expect("rollback closure consumed once")(mutation_log);
                    }
                    self.advance_install_order_and_notify(commit_lsn);
                    self.active.remove(&txn_id);
                    return Err(error);
                }
            };
            (commit_range, staged)
        };
        let commit_lsn = commit_range.commit_lsn();
        for vector in vector_pages.iter_mut() {
            if vector.commit_lsn == Lsn::ZERO {
                vector.commit_lsn = commit_lsn;
            }
        }
        let deltas: Vec<DeltaOp> = if writes_delta {
            std::mem::take(&mut mutation_log.delta_intents)
                .into_iter()
                .enumerate()
                .map(|(index, intent)| {
                    let op_lsn = commit_range
                        .op_lsn(index)
                        .expect("delta intent index is bounded by allocated range");
                    intent
                        .assign_for_format(
                            op_lsn,
                            commit_lsn,
                            delta_format.expect("delta format is present in delta mode"),
                        )
                        .expect("CRUD produced an invalid delta intent")
                })
                .collect()
        } else {
            Vec::new()
        };
        // M4 owner ids can be returned to later graph commits immediately.
        // Their physical rows therefore always use the strict durability arm,
        // even when the tenant's ordinary record RPO is Periodic. Phase 3
        // consumes the exact fsync proof before publishing the row.
        let contains_owner_deltas = deltas.iter().any(|delta| {
            matches!(
                delta.kind,
                crate::wal::DeltaOpKind::InternBind | crate::wal::DeltaOpKind::AclGrant
            )
        });
        let delta_primary_writes = if writes_delta {
            let mut normalized = writes.clone();
            let _gate = self.commit_gate.lock();
            for delta in &deltas {
                let Some((key, delta_value)) = crate::wal::delta::put_record_mvcc_write(delta)?
                else {
                    continue;
                };
                let Some(Some(original)) = normalized.get(&key) else {
                    continue;
                };
                let created_lsn_offset = match delta_value.len() {
                    64 => 56,
                    96 => 48,
                    _ => unreachable!("PutRecord length validated"),
                };
                let mut expected = delta_value.to_vec();
                expected[created_lsn_offset..created_lsn_offset + 8]
                    .copy_from_slice(&0u64.to_le_bytes());
                if original.as_ref() != expected.as_slice() {
                    continue;
                }
                normalized.insert(key, Some(delta_value.clone()));
                if let Some(mut chain) = self.versions.get_mut(&(tenant_id, key))
                    && let Some(version) = chain
                        .iter_mut()
                        .find(|version| version.created_lsn == commit_lsn)
                {
                    version.value = Some(delta_value);
                }
            }
            Some(normalized)
        } else {
            None
        };

        // #1200: coalesce duplicate sidechannel writes last-write-wins
        // by (tenant_id, key) BEFORE both consumers below — the Phase-2
        // `encode_commit_bundle_v7` (so the durable bundle carries ONE
        // entry per key) AND the Phase-3 `apply_sidechannel_mvcc_write`
        // loop (so no 2nd write fires at the same commit_lsn). A single
        // >51,765-key commit fires ≥ 2 grow_roots, each pushing a
        // `(SYSTEM, PRIMARY_INDEX_ROOT_KEY)` write; un-coalesced, the
        // duplicate trips the `apply_sidechannel_mvcc_write` debug-assert
        // AND, on crash + replay, the value-blind `Idempotent` skip
        // (`apply_replay_mvcc_write`) drops the FINAL root, stranding the
        // durable root at the intermediate one (~78% of index keys
        // unreachable post-recovery). Keeping the LAST write per key
        // (= the final/outermost root, since grow_root appends in split-
        // cascade order) restores the last-wins invariant the retired
        // ADR-032 Slice-2 `pending_root` slot used to provide. See
        // `Self::coalesce_sidechannel_writes` for the full rationale.
        // Natural completion of #849 (which coalesced staged PAGES on
        // this exact bulk-load path but not the sidechannel writes).
        Self::coalesce_sidechannel_writes(sidechannel_writes);

        // ─── Phase 2: single CommitBundle append OUTSIDE the gate ──
        //
        // `wal.append` blocks until the group-commit `fdatasync`
        // returns. With the gate released, 8 concurrent writers
        // pipeline into one fsync batch (invariant 8, ADR-031 §3.5).
        //
        // ADR-032 Slice 2 cutover: encode with v2 codec so the per-
        // entry tenant-id survives to replay. Both primary writes
        // and sidechannel writes ride one atomic bundle.
        //
        // **ADR-031 amendment-02 / PR #79 X-2 fold-in**: cutover
        // to v3 codec — `staged_emits` entries now carry a
        // `BundlePageKind` byte so record / blob pages travel in
        // the bundle alongside primary + secondary index pages.
        // The `StagedEmit::kind` field defaults to `PrimaryIndex`
        // so pre-amendment callers that build StagedEmits without
        // specifying `kind` keep producing v2-equivalent-shape
        // bundles (modulo the extra byte per entry).
        //
        // Failure is carried to Phase 3 (not returned early) so
        // successors waiting on `install_order == range_base - 1`
        // observe our Phase 3 run regardless of success/failure. A
        // bare early-return would leak the silent install AND
        // deadlock successors.
        let wal_err = if let Some(wal) = wal_handle.as_ref() {
            // #849 B3(a): coalesce duplicate per-record page snapshots
            // (last-write-wins by (kind, page_id)) so the bundle logs
            // each touched page ONCE instead of once-per-record.
            let staged_v4 = Self::coalesce_staged_pages(&staged_emits, tenant_id);
            // Issue #129 P0 fix: v4 codec carries the
            // allocator_advances tail so per-tenant NodeId / RelId /
            // PageId high-water marks survive WAL recovery. Without
            // this, post-fault `create_node` re-issues NodeIds that
            // pre-fault commits already consumed, orphaning earlier
            // T1 commits through the primary index (ADR-034 D-1
            // violation).
            //
            // **M3.a Slice G.4 (commit-bundle vector page staging)**:
            // cutover to v5 codec. v5 extends v4 with a trailing
            // `vector_pages` section so vector arena page mutations
            // are durified atomically with the commit that wrote
            // them. Producers stage pages via
            // `CrudStore::stage_vector_page`; the drain happens
            // before the bundle builder closure runs (see
            // `crud::commit`) so the closure receives a populated
            // `vector_pages` slice. Per ADR-031 amendment-02 +
            // ADR-035 §4.5/§4.6.
            // #1221 (ADR-218): cutover to the v8 codec. v8 extends v7 with
            // a trailing `acl_grants` section so `PermissionIndex`
            // grant/revoke ops are durified atomically with the commit
            // that carries them. Producers stage ops via
            // `CrudStore::stage_acl_grant`; the drain happens in
            // `crud::commit` and stages them on the `Transaction` BEFORE
            // this call (like `idempotency_bindings`, they carry no
            // commit_lsn). Append-order preserved — see
            // `encode_commit_bundle_v8`'s invariant.
            let payload = if writes_delta {
                let retained_pages: Vec<_> = staged_v4
                    .into_iter()
                    .filter(|(kind, _, _, _)| {
                        matches!(
                            kind,
                            crate::wal::BundlePageKind::PrimaryIndex
                                | crate::wal::BundlePageKind::SecondaryIndex
                                | crate::wal::BundlePageKind::Blob
                        )
                    })
                    .collect();
                encode_commit_bundle_delta_for_format(
                    delta_format.expect("delta format is present in delta mode"),
                    commit_lsn,
                    tenant_id,
                    delta_primary_writes
                        .as_ref()
                        .expect("delta normalized writes exist"),
                    sidechannel_writes,
                    &deltas,
                    &retained_pages,
                    vector_pages,
                    allocator_advances,
                    idempotency_bindings,
                    acl_grants,
                )
            } else {
                Ok(encode_commit_bundle_current(
                    commit_lsn,
                    tenant_id,
                    writes,
                    sidechannel_writes,
                    &staged_v4,
                    allocator_advances,
                    vector_pages,
                    idempotency_bindings,
                    acl_grants,
                ))
            };
            // ADR-034 §Slice D: dispatch on tier at commit time.
            // SYSTEM is always Strict (I-D7). Unknown / no-resolver
            // also falls back to Strict (pre-ADR-034 behaviour).
            //
            // Periodic / T3: the async enqueue returns as soon as
            // bytes are accepted by the writer's pending buffer.
            // Phase 3 advances `visible` immediately per D-4 — the
            // caller observes Ok(commit_lsn) before the fsync
            // completes. Durability is provided by the scheduler
            // (rpo_ms bound), by a piggybacking T1 commit (I-D3),
            // or by a shutdown drain.
            //
            // Strict / T1: the sync append blocks on the group-
            // commit fsync (pre-ADR-034 path, unchanged).
            match payload {
                Err(error) => Some(error),
                Ok(payload) => match if contains_owner_deltas {
                    DurabilityTier::Strict
                } else {
                    self.tier_for_commit(tenant_id)
                } {
                    DurabilityTier::Strict if writes_delta => wal
                        .append_at(
                            commit_lsn,
                            WalRecordType::CommitBundle,
                            txn_id,
                            now_millis(),
                            tenant_id,
                            payload,
                        )
                        .err(),
                    DurabilityTier::Strict => wal
                        .append(
                            WalRecordType::CommitBundle,
                            txn_id,
                            now_millis(),
                            tenant_id,
                            payload,
                        )
                        .err(),
                    DurabilityTier::Periodic { .. } if writes_delta => wal
                        .append_async_at(
                            commit_lsn,
                            WalRecordType::CommitBundle,
                            txn_id,
                            now_millis(),
                            tenant_id,
                            payload,
                        )
                        .err(),
                    DurabilityTier::Periodic { .. } => wal
                        .append_async(
                            WalRecordType::CommitBundle,
                            txn_id,
                            now_millis(),
                            tenant_id,
                            payload,
                        )
                        .err(),
                },
            }
        } else {
            None
        };

        // ─── Phase 3: ordered install-order advance ────────────────
        self.wait_for_install_turn(commit_range);
        match wal_err {
            None => {
                if writes_delta
                    && let Err(error) = apply.take().expect("delta apply closure consumed once")(
                        &deltas, commit_lsn,
                    )
                {
                    tracing::error!(
                        ?commit_lsn,
                        %error,
                        "durable delta commit failed during install-after-durability; aborting"
                    );
                    std::process::abort();
                }
                // ADR-032 §2: apply sidechannel MVCC writes to their
                // non-primary tenant chains. Skips OCC.
                //
                // #1200 correction: the PrimaryIndex `write_gate` only
                // serializes CONCURRENT grow_roots across commits — it
                // does NOTHING about SEQUENTIAL grow_roots WITHIN one
                // commit. A single >51,765-key commit fires ≥ 2
                // grow_roots, each pushing a `(SYSTEM, ROOT_KEY)` write
                // at the SAME commit_lsn. What now guarantees AT MOST
                // ONE sidechannel write per `(tenant, key)` per commit
                // is `coalesce_sidechannel_writes` (called above, before
                // both this loop and the bundle encode), NOT write_gate.
                // So no second `apply_sidechannel_mvcc_write` fires at
                // the same commit_lsn (the debug-assert at
                // `apply_sidechannel_mvcc_write` stays correct + unmet),
                // and replay applies exactly the final root.
                for sc in sidechannel_writes.iter() {
                    self.apply_sidechannel_mvcc_write(
                        commit_lsn,
                        sc.tenant_id,
                        sc.key,
                        sc.value.clone(),
                    );
                }
                self.visible.store(commit_lsn.raw(), Ordering::Release);
                self.advance_install_order_and_notify(commit_lsn);
                self.active.remove(&txn_id);
                // #1404 M0.x — frontier advanced + this txn's snapshot pin
                // dropped (`active.remove` above); drive the watermark-
                // triggered MVCC drain so the resident superseded-version set
                // stays bounded between checkpoints. Runs OUTSIDE every commit
                // lock (Phase 3 already released), no-op when disabled.
                // INV-DRAIN preserved: `gc()` reclaims only versions
                // `expired_lsn ≤ oldest_active_snapshot`.
                self.maybe_drive_gc();
                Ok(commit_lsn)
            }
            Some(e) => {
                // ADR-033 §8: check the WAL error policy BEFORE
                // running any rollback. The `abort` policy is a
                // fail-fast replacement for Z-1 (b); it shortcuts
                // out of this function by killing the process.
                // No rollback runs (ADR-032's WAL replay rebuilds
                // state from the durable prefix on restart).
                //
                // The policy read pays one env-var lookup on first
                // call and uses the cached OnceLock value thereafter.
                if matches!(WalErrorPolicy::global(), WalErrorPolicy::Abort) {
                    tracing::error!(
                        "ADR-033 §8 abort policy: WAL fsync failed at commit_lsn {:?}; aborting process. error: {}",
                        commit_lsn,
                        e,
                    );
                    std::process::abort();
                }

                // Sidechannel writes were NOT installed in Phase 1 —
                // no rollback needed for them. Primary writes are
                // rolled back under commit_gate below.
                //
                // ADR-033 Z-1 (b): the caller's rollback closure runs
                // under commit_gate AFTER the MVCC unwind and BEFORE
                // install_order advances. Order matters (ADR-033 §5
                // root-ordering, §6 sequence). The MVCC + in-memory
                // rollback pair is the complete undo of the failed
                // commit's effects.
                {
                    let _gate = self.commit_gate.lock();
                    self.rollback_writes(tenant_id, commit_lsn, writes);
                    rollback.take().expect("rollback closure consumed once")(mutation_log);
                }
                self.advance_install_order_and_notify(commit_lsn);
                self.active.remove(&txn_id);

                // ADR-033 §3c: wrap the original WAL error in
                // `WalErrorRolledBack` to signal to the caller that
                // the transaction's in-memory state is fully unwound
                // and the operation is retryable by construction.
                // The inner `e` is preserved for diagnostics via
                // `std::error::Error::source`.
                Err(ArcGraphError::WalErrorRolledBack {
                    source: Box::new(e),
                })
            }
        }
    }

    /// #37 [A-1] crash-atomicity — emit the staged index-page snapshots
    /// of ONE logical **standalone** index operation, plus any
    /// SYSTEM-tenant root-pointer MVCC writes, in a SINGLE
    /// `CommitBundle` WAL record.
    ///
    /// This replaces the legacy per-page `IndexPage` drain on the
    /// standalone (non-bundle-aware) index-write path
    /// (`PrimaryIndex::write` / `SecondaryIndex::write` and their
    /// `insert` / `upsert` / `remove` public wrappers +
    /// `bootstrap_from_mvcc`). The legacy drain emitted one
    /// `WalRecordType::IndexPage` record per staged page, so a crash
    /// between two such records during a leaf split (`apply_leaf_op`)
    /// or an overflow-successor allocation (`append_at_or_past_tail`)
    /// left an orphan page on replay — one sibling durable, the other
    /// not. Folding every page of the logical op into one CRC-framed
    /// bundle makes the op crash-atomic: replay applies ALL pages or
    /// NONE (a torn bundle fails CRC and is dropped, ADR-031 §R5).
    ///
    /// Realizes ADR-031 (CommitBundle) for the standalone path; the
    /// CommitBundle *hot* path (`Self::commit_with_bundle_writes`)
    /// is unchanged. Index pages are SYSTEM-tenant per DEC-18, so the
    /// bundle's `primary_tenant` is SYSTEM and the root-pointer writes
    /// (grow_root) ride the bundle's primary MVCC-writes section —
    /// replay's `apply_bundle` restores them onto the SYSTEM chain
    /// atomically with the page installs.
    ///
    /// Mirrors `commit_with_bundle_writes`'s three-phase shape minus
    /// OCC (there are no user primary writes to validate — the index
    /// `write_gate` already serialized the page mutations, and
    /// root-pointer writes skip OCC by construction per ADR-032 §2) and
    /// minus the builder (the index mutation already ran under the
    /// index's `write_gate`; `staged` + `sidechannel_writes` are its
    /// captured output). It participates in the Phase-3 `install_order`
    /// protocol so it stays correct if it ever races a hot-path commit.
    ///
    /// **Failure semantics (unchanged from the legacy drain).** On WAL
    /// failure the in-memory index pages already installed by the
    /// caller remain ahead of the durable prefix; replay rebuilds from
    /// the durable prefix on restart. The root-pointer MVCC write is
    /// NOT applied on WAL failure, so MVCC and the (non-durable) page
    /// install agree that the op did not durably happen.
    ///
    /// `sidechannel_writes` MUST be SYSTEM-tenant (the only producer is
    /// grow_root's root-pointer update); this is debug-asserted.
    ///
    /// `wal` is the WAL the staged index pages emit into — the caller
    /// passes the **index's own** [`WalHandle`]
    /// (`PrimaryIndex::wal` / `SecondaryIndex::wal`), which is the
    /// historical emit target for index pages (pre-bundle, the index
    /// drained directly to it). In production it is the same handle as
    /// `self.wal` (both wired from one `WalWriter` per
    /// `TxnManager::with_wal` + `*Index::new(.., Some(wal))`); some unit
    /// tests deliberately give the index a WAL while leaving the
    /// `TxnManager` WAL-less, so the WAL target is taken from the
    /// parameter rather than `self.wal`. `None` (no WAL anywhere) keeps
    /// the in-memory installs + applies the MVCC writes, matching the
    /// legacy drain's no-WAL early-return behaviour.
    pub fn commit_index_pages_atomic(
        &self,
        wal: Option<&WalHandle>,
        staged: &[StagedEmit],
        sidechannel_writes: &[SideChannelWrite],
    ) -> Result<()> {
        if staged.is_empty() && sidechannel_writes.is_empty() {
            return Ok(());
        }

        // SVC-1 / #849 / ADR-229 — checkpoint/commit serialization (READ
        // side), FIRST statement, function-scoped RAII: spans Phase 1
        // (`counter.allocate` below), the `IndexPage` staging into the
        // captured `record_pages` owner, Phase 2 (WAL fsync), and Phase 3
        // (`visible.store` at the bottom). This is the SECOND live commit
        // path (`commit_index_pages_atomic`, the standalone SYSTEM index /
        // grow_root root-pointer path); WITHOUT this guard a concurrent
        // checkpoint could (a) capture an `IndexPage` whose SYSTEM/IndexLeaf
        // allocator high-water it did NOT capture → post-restart re-hand-out
        // of that page id → B-TREE PAGE ALIASING (the ULTRACODE re-verify
        // BLOCK-1 index-page residual; latent on x86-64 TSO, reachable on
        // aarch64), and (b) capture a torn `IndexPage` mid-stage (BLOCK-2
        // variant). Taking the read guard here mirrors
        // `commit_with_bundle_writes` and makes the ADR-229 §Consequences
        // "every three-phase commit takes a read guard" invariant TRUE for
        // BOTH paths. Lock order is preserved: `checkpoint_read →
        // commit_gate` (same as the other path); this path never takes
        // `producer_mutex` → no inversion, no deadlock.
        let _checkpoint_read = self.checkpoint_lock.read();

        // ─── Phase 1: allocate commit_lsn (under commit_gate) ───────
        // No OCC + no silent install: there are no user primary writes.
        // The SYSTEM root-pointer writes skip OCC by construction (the
        // caller holds the index `write_gate`, ADR-032 §2).
        let commit_range = {
            let _gate = self.commit_gate.lock();
            self.counter.allocate_range(1)
        };
        let commit_lsn = commit_range.commit_lsn();

        // The standalone index path is SYSTEM-tenant (DEC-18). Fold the
        // SYSTEM root-pointer writes into the bundle's primary
        // MVCC-writes section so replay restores them onto the SYSTEM
        // chain atomically with the staged page installs.
        let primary_tenant = TenantId::SYSTEM;
        let mut sys_writes: HashMap<MvccKey, Option<Bytes>> = HashMap::new();
        for sc in sidechannel_writes {
            // #769 R1 NIT #5: harden the SYSTEM-tenant invariant from a
            // debug-only `debug_assert_eq!` (a no-op in release) to a hard
            // error. The sole producer is grow_root's root-pointer update
            // (SYSTEM-tenant by construction via
            // `SecondaryIndex::take_pending_root_sidechannel`); a
            // non-SYSTEM write would otherwise silently fold into the
            // SYSTEM primary MVCC chain on replay. The closed caller set
            // makes this unreachable today — the guard is defense-in-depth
            // on a crash-durability path.
            if sc.tenant_id != primary_tenant {
                return Err(ArcGraphError::TransactionAborted {
                    reason: format!(
                        "commit_index_pages_atomic: standalone index sidechannel writes must be \
                         SYSTEM-tenant (got {:?}); sole producer is grow_root's root-pointer update",
                        sc.tenant_id
                    ),
                });
            }
            sys_writes.insert(sc.key, sc.value.clone());
        }

        // ─── Phase 2: single CommitBundle append OUTSIDE the gate ───
        let delta_wal = wal.filter(|wal| is_delta_bundle_format(wal.format_version()));
        let wal_err = if let Some(wal) = wal {
            // #849 B3(a): coalesce duplicate page snapshots
            // (last-write-wins by (kind, page_id)) — symmetric with the
            // main CRUD commit path.
            let staged_pages = Self::coalesce_staged_pages(staged, primary_tenant);
            let payload = if is_delta_bundle_format(wal.format_version()) {
                encode_commit_bundle_delta_for_format(
                    wal.format_version(),
                    commit_lsn,
                    primary_tenant,
                    &sys_writes,
                    &[], // SYSTEM writes ride the primary section
                    &[], // index-only commits have no physical data deltas
                    &staged_pages,
                    &[],
                    &[],
                    &[],
                    &[],
                )
            } else {
                Ok(encode_commit_bundle_current(
                    commit_lsn,
                    primary_tenant,
                    &sys_writes,
                    &[], // sidechannel section unused: SYSTEM writes ride `primary`
                    &staged_pages,
                    &[], // no allocator advances on the standalone index path
                    &[], // no vector pages on the standalone index path
                    &[], // no idempotency bindings on the standalone index path (#352)
                    &[], // no acl_grants on the standalone index path (#1221)
                ))
            };
            // SYSTEM is always Strict (I-D7): synchronous append that
            // blocks on the group-commit fsync.
            payload
                .and_then(|payload| {
                    if is_delta_bundle_format(wal.format_version()) {
                        wal.append_at(
                            commit_lsn,
                            WalRecordType::CommitBundle,
                            /* txn_id = */ 0,
                            now_millis(),
                            primary_tenant,
                            payload,
                        )
                    } else {
                        wal.append(
                            WalRecordType::CommitBundle,
                            /* txn_id = */ 0,
                            now_millis(),
                            primary_tenant,
                            payload,
                        )
                    }
                    .map(|_| ())
                })
                .err()
        } else {
            None
        };
        if wal_err.is_none()
            && let Some(wal) = delta_wal
        {
            let consumed = wal.take_exact_durable(commit_lsn);
            debug_assert!(consumed);
        }

        // ─── Phase 3: ordered install ───────────────────────────────
        self.wait_for_install_turn(commit_range);
        let result = match wal_err {
            None => {
                // NOTE (#1200): this apply loop reads the RAW
                // `sidechannel_writes` vec, NOT the `sys_writes` HashMap
                // above (which de-dups only the ENCODE side). It does NOT
                // need the bundle-folded path's `coalesce_sidechannel_writes`
                // — and is safe — because this STANDALONE path commits ONE
                // logical index operation per call, which triggers AT MOST
                // ONE `grow_root` (≤1 SYSTEM root-pointer write). With ≤1
                // write per (tenant,key), the second `apply_sidechannel_mvcc_write`
                // at the same `commit_lsn` (the #1200 debug-assert trip /
                // replay-corruption condition) never arises here. The safety
                // is the single-insert-per-call invariant, NOT de-dup: a
                // future reader must NOT "fix" this to read `sys_writes` or
                // add coalescing on the belief the HashMap protects it.
                for sc in sidechannel_writes {
                    self.apply_sidechannel_mvcc_write(
                        commit_lsn,
                        sc.tenant_id,
                        sc.key,
                        sc.value.clone(),
                    );
                }
                self.visible.store(commit_lsn.raw(), Ordering::Release);
                Ok(())
            }
            Some(e) => Err(e),
        };
        // Advance install_order in BOTH outcomes so successors waiting
        // on `install_order == range_base - 1` always make progress
        // (mirrors `commit_with_bundle_writes` Phase 3).
        self.advance_install_order_and_notify(commit_lsn);
        result
    }

    /// ADR-032 §2 sidechannel MVCC write primitive.
    ///
    /// Pushes a new `Version` onto the `(tenant, key)` chain with
    /// `created_lsn = commit_lsn` and expires any prior live Version
    /// at that commit_lsn — identical to the Phase-1 install of a
    /// normal user write. **Skips OCC** (no write-set validation, no
    /// `MvccConflict` return path) because the caller is required to
    /// hold an external serialization gate on the target `(tenant,
    /// key)` pair.
    ///
    /// The sole in-tree caller today is grow_root (via
    /// `Self::commit_with_bundle_writes` Phase 3). grow_root holds
    /// the `crate::primary_index::PrimaryIndex::write_gate` mutex,
    /// which serializes all grow_root callers and therefore
    /// serializes sidechannel writes to the SYSTEM root-pointer key.
    /// No two callers can sidechannel-write the same `(SYSTEM,
    /// PRIMARY_INDEX_ROOT_KEY)` pair concurrently, so skipping OCC is
    /// safe by construction.
    ///
    /// **Debug assertion.** The chain's last version must have
    /// `created_lsn < commit_lsn` OR the chain must be empty. A
    /// caller that violates this — attempting to apply a sidechannel
    /// write at an LSN already present on the chain — is a logic bug
    /// (it would mask an out-of-order install and corrupt visibility
    /// ordering) and panics in debug builds.
    pub fn apply_sidechannel_mvcc_write(
        &self,
        commit_lsn: Lsn,
        tenant: TenantId,
        key: MvccKey,
        value: Option<Bytes>,
    ) {
        let vk = (tenant, key);
        let mut chain = self.versions.entry(vk).or_default();
        debug_assert!(
            chain
                .last()
                .is_none_or(|v| v.created_lsn.raw() < commit_lsn.raw()),
            "apply_sidechannel_mvcc_write to key ({tenant:?}, {key}) at commit_lsn \
             {commit_lsn:?}: last version has created_lsn >= commit_lsn (logic bug — caller \
             must hold the external write_gate)",
        );
        if let Some(last) = chain.last_mut()
            && last.is_live()
        {
            last.expired_lsn = commit_lsn;
        }
        chain.push(Version {
            created_lsn: commit_lsn,
            expired_lsn: Lsn::MAX,
            value,
        });
        // Issue #238: register the (tenant, key) in the per-tenant
        // chain index. Sidechannel writes typically land on
        // SYSTEM-tenant root-pointer keys; populating here keeps the
        // SYSTEM tenant visible to `tenants_with_chains` for cold-start
        // rebuilds without requiring a workaround in the rebuild driver.
        drop(chain);
        self.register_chain_key(tenant, key);
    }

    /// ADR-032 §R2 Step 3 replay primitive. Idempotent apply of an
    /// MVCC write at `commit_lsn`.
    ///
    /// Distinct from [`Self::apply_sidechannel_mvcc_write`] in two
    /// ways:
    ///
    /// 1. **Idempotent (Lemma I1).** If the chain already ends with a
    ///    Version whose `created_lsn == commit_lsn` AND the stored
    ///    value matches `value`, this is a no-op — the bundle has
    ///    already been applied (double-replay case). Returns `false`
    ///    to signal "skipped".
    /// 2. **Gap-tolerant (§R7).** If the chain's last version has
    ///    `created_lsn > commit_lsn`, the replay executor encountered
    ///    an out-of-order bundle (per §R2 step 2 the executor sorts
    ///    bundles before apply, so this path indicates an upstream
    ///    bug). Panics in debug, silently no-ops in release to
    ///    preserve replay determinism.
    ///
    /// Not for production commits — `commit_with_bundle_writes` is
    /// the primary path. Callers: replay executor only. See ADR-032
    /// §R2 and §Invariant 11 (Recovery idempotence).
    ///
    /// Returns a [`ReplayApplyOutcome`] distinguishing:
    ///
    /// - `Applied` — new Version pushed.
    /// - `Idempotent` — chain's last `created_lsn == commit_lsn`;
    ///   bundle was already applied (Lemma I1). Double-replay is
    ///   a no-op.
    /// - `OutOfOrder` — chain's last `created_lsn > commit_lsn`.
    ///   The executor sorts bundles before apply so this path
    ///   indicates an upstream bug. Debug-asserts; release
    ///   logs `tracing::error!` and returns without push.
    ///
    /// **PR #79 Y-3 fold-in**: formerly returned `bool`, which
    /// conflated legitimate idempotent skip (outcome `Idempotent`,
    /// good) with OOO rejection (outcome `OutOfOrder`, bad). The
    /// conflation hid a latent regression signal behind the
    /// `bundles_skipped_idempotent` counter. Splitting the two
    /// outcomes gives the executor a separate
    /// `wal_replay_out_of_order_apply_rejected` counter to surface
    /// OOO occurrences.
    pub fn apply_replay_mvcc_write(
        &self,
        commit_lsn: Lsn,
        tenant: TenantId,
        key: MvccKey,
        value: Option<Bytes>,
    ) -> ReplayApplyOutcome {
        self.apply_replay_mvcc_write_inner(commit_lsn, tenant, key, value, false)
    }

    /// Reconcile a pre-frontier WAL row with an MVCC head reconstructed from
    /// an incremental checkpoint's home page. A home page may already carry
    /// a later version than the first DPT redo record; that is coverage, not
    /// executor mis-ordering, so the older record is an idempotent no-op.
    pub(crate) fn apply_incremental_replay_mvcc_write(
        &self,
        commit_lsn: Lsn,
        tenant: TenantId,
        key: MvccKey,
        value: Option<Bytes>,
    ) -> ReplayApplyOutcome {
        self.apply_replay_mvcc_write_inner(commit_lsn, tenant, key, value, true)
    }

    fn apply_replay_mvcc_write_inner(
        &self,
        commit_lsn: Lsn,
        tenant: TenantId,
        key: MvccKey,
        value: Option<Bytes>,
        incremental_base_may_cover_commit: bool,
    ) -> ReplayApplyOutcome {
        let vk = (tenant, key);
        let mut chain = self.versions.entry(vk).or_default();

        // Lemma I1: if the chain's last version has the same
        // `created_lsn`, this bundle was already applied. Skip.
        if let Some(last) = chain.last()
            && last.created_lsn.raw() == commit_lsn.raw()
        {
            return ReplayApplyOutcome::Idempotent;
        }

        // §R7 gap tolerance: the executor applies bundles in
        // commit_lsn-ascending order, so a chain.last() with a
        // strictly greater created_lsn is an upstream bug (out-of-
        // order apply). Debug-assert; release-noop. The executor's
        // OOO counter (§7 surface) picks this up.
        if let Some(last) = chain.last()
            && last.created_lsn.raw() > commit_lsn.raw()
        {
            if incremental_base_may_cover_commit {
                return ReplayApplyOutcome::Idempotent;
            }
            debug_assert!(
                false,
                "apply_replay_mvcc_write to key ({tenant:?}, {key}) at commit_lsn \
                 {commit_lsn:?}: chain last has created_lsn {:?} > commit_lsn (out-of-order \
                 replay — executor bug)",
                last.created_lsn,
            );
            return ReplayApplyOutcome::OutOfOrder;
        }

        // Chain is either empty or its last version has created_lsn
        // strictly less than commit_lsn: safe to push.
        if let Some(last) = chain.last_mut()
            && last.is_live()
        {
            last.expired_lsn = commit_lsn;
        }
        chain.push(Version {
            created_lsn: commit_lsn,
            expired_lsn: Lsn::MAX,
            value,
        });
        // Issue #238: register the (tenant, key) in the per-tenant
        // chain index. Replay populates the index alongside the chain
        // it rebuilds, so post-replay `for_each_visible_record`
        // (called by the cold-start rebuild driver) walks per-tenant
        // in O(N_tenant) rather than scanning every shard.
        drop(chain);
        self.register_chain_key(tenant, key);
        ReplayApplyOutcome::Applied
    }

    /// ADR-032 §R2 Step 3d replay finalization. Advance the MVCC
    /// counter, visible watermark, and install_order to
    /// `max_commit_lsn`.
    ///
    /// Called by the replay executor after every bundle has been
    /// applied. Semantically:
    ///
    /// - `counter` = `max_commit_lsn + 1` so the next `allocate()`
    ///   returns `max_commit_lsn + 2`. (The counter's internal
    ///   representation is "last allocated LSN"; we set it to
    ///   `max_commit_lsn` so the next reservation returns
    ///   `max_commit_lsn + 1`.)
    /// - `visible` = `max_commit_lsn` so new transactions see every
    ///   replayed write from their snapshot (§R2 Step 3d +
    ///   Invariant 14).
    /// - `install_order` = `max_commit_lsn` so the next committing
    ///   transaction's range-predecessor wait
    ///   returns immediately without blocking on a non-existent
    ///   predecessor.
    ///
    /// Idempotent via `max`: a second replay over the same WAL that
    /// yields the same `max_commit_lsn` is a no-op on each atomic.
    /// A lower `max_commit_lsn` (shouldn't happen, but safe) is
    /// ignored.
    ///
    /// Not for production — the commit path advances these fields
    /// per-commit.
    pub fn seed_after_replay(&self, max_commit_lsn: Lsn) {
        if max_commit_lsn == Lsn::ZERO {
            return;
        }
        // Counter: last allocated LSN = max_commit_lsn. Uses
        // fetch_max for monotone behaviour across idempotent calls.
        self.counter.advance_to(max_commit_lsn);
        // Visible watermark — monotone via fetch_max.
        self.visible
            .fetch_max(max_commit_lsn.raw(), Ordering::AcqRel);
        // Install-order — also monotone. Notify waiters in case any
        // are parked (should be none at replay time, but cheap).
        {
            let mut guard = self.install_order.lock();
            if *guard < max_commit_lsn.raw() {
                *guard = max_commit_lsn.raw();
            }
        }
        self.install_cv.notify_all();
    }

    /// Wait until `install_order` reaches the predecessor of this
    /// commit's contiguous redo range.
    ///
    /// Called in Phase 3. Blocks on `install_cv` until the preceding
    /// range has completed its Phase 3 (success or failure).
    /// For the first-ever range (`base == 1`) the target is
    /// `Lsn::ZERO.raw() == 0` and the initial `install_order` value
    /// is also 0, so the loop exits immediately.
    fn wait_for_install_turn(&self, commit_range: RedoLsnRange) {
        let target = commit_range.predecessor().raw();
        let mut guard = self.install_order.lock();
        while *guard < target {
            self.install_cv.wait(&mut guard);
        }
    }

    /// Advance `install_order` to `to` and wake every waiter.
    ///
    /// Called at the end of Phase 3. `notify_all` is cheap at 8-way
    /// concurrency; the successor wakes, finds its target met, and
    /// proceeds. Others re-sleep. `install_order` is only ever
    /// advanced — never regressed — so callers with stale `to`
    /// values are a no-op.
    fn advance_install_order_and_notify(&self, to: Lsn) {
        {
            let mut guard = self.install_order.lock();
            if *guard < to.raw() {
                *guard = to.raw();
            }
        }
        self.install_cv.notify_all();
    }

    /// Roll back a Phase-1 silent install.
    ///
    /// Invariant on entry (§4 of WAL-COMMIT-GATE-DESIGN.md): our
    /// `Version` is the last entry in each of `writes`'s key chains,
    /// because no concurrent commit could stack on top (they would
    /// have seen `chain.last.created_lsn > their_snapshot` and
    /// conflicted during Phase-1 validation).
    ///
    /// Called under `commit_gate` so concurrent Phase-1 validators
    /// never observe a half-popped chain.
    fn rollback_writes(
        &self,
        tenant_id: TenantId,
        commit_lsn: Lsn,
        writes: &HashMap<MvccKey, Option<Bytes>>,
    ) {
        for key in writes.keys() {
            let vk = (tenant_id, *key);
            let Some(mut chain) = self.versions.get_mut(&vk) else {
                continue;
            };
            // Defensive: only pop if the last entry truly belongs to
            // us. Matches the §4 "our version is last" invariant but
            // avoids corruption if the invariant ever breaks.
            let our_at_end = chain
                .last()
                .map(|v| v.created_lsn == commit_lsn)
                .unwrap_or(false);
            if !our_at_end {
                continue;
            }
            chain.pop();
            // Restore predecessor's `expired_lsn` from our commit_lsn
            // back to MAX (the Phase-1 install set it to our LSN
            // when it expired the predecessor).
            if let Some(prev) = chain.last_mut()
                && prev.expired_lsn == commit_lsn
            {
                prev.expired_lsn = Lsn::MAX;
            }
        }
    }

    /// Mirrors `commit_writes` with explicit key ordering and two
    /// barrier pauses (post-allocate, post-first-install). Used only
    /// by `Transaction::commit_with_barriers`.
    ///
    /// Updated for the three-phase commit path (invariant 8): Phase 1
    /// (validate + allocate + silent install) holds `commit_gate`;
    /// the two barriers remain inside Phase 1, after allocation and
    /// between installs. No WAL — this hook is used only by tests
    /// that construct `TxnManager::new()` (no WAL handle), so
    /// Phase 2 is a no-op. Phase 3 publishes `visible` and advances
    /// `install_order` exactly as the real `commit_writes` would.
    #[allow(clippy::too_many_arguments)]
    fn commit_with_barriers_raw(
        &self,
        txn_id: u64,
        tenant_id: TenantId,
        snapshot: Lsn,
        key_order: &[MvccKey],
        writes: &HashMap<MvccKey, Option<Bytes>>,
        between_alloc_and_install: &std::sync::Barrier,
        between_first_and_second: &std::sync::Barrier,
    ) -> Result<Lsn> {
        if key_order.is_empty() {
            self.active.remove(&txn_id);
            return Ok(Lsn::new(self.visible.load(Ordering::Acquire)));
        }
        // ─── Phase 1: validate + allocate + silent install ──────────
        let commit_range = {
            let _gate = self.commit_gate.lock();
            for key in key_order {
                let vk = (tenant_id, *key);
                if let Some(chain) = self.versions.get(&vk)
                    && let Some(last) = chain.last()
                    && last.created_lsn.raw() > snapshot.raw()
                {
                    return Err(ArcGraphError::MvccConflict {
                        target: format!("key:{key}"),
                    });
                }
            }
            let commit_range = self.counter.allocate_range(1);
            let commit_lsn = commit_range.commit_lsn();
            between_alloc_and_install.wait();
            for (i, key) in key_order.iter().enumerate() {
                let value = writes.get(key).expect("key_order references buffered key");
                let vk = (tenant_id, *key);
                let mut chain = self.versions.entry(vk).or_default();
                if let Some(last) = chain.last_mut()
                    && last.is_live()
                {
                    last.expired_lsn = commit_lsn;
                }
                chain.push(Version {
                    created_lsn: commit_lsn,
                    expired_lsn: Lsn::MAX,
                    value: value.clone(),
                });
                drop(chain);
                // Issue #238: keep the per-tenant chain index in sync
                // on the test-only barrier path so tests that exercise
                // `for_each_visible_record` or `tenants_with_chains`
                // see the same shape as the production commit path.
                self.register_chain_key(tenant_id, *key);
                if i == 0 {
                    between_first_and_second.wait();
                }
            }
            commit_range
        };
        let commit_lsn = commit_range.commit_lsn();

        // ─── Phase 2: no-op (no WAL in this test hook). ────────────

        // ─── Phase 3: ordered install-order advance. ───────────────
        self.wait_for_install_turn(commit_range);
        self.visible.store(commit_lsn.raw(), Ordering::Release);
        self.advance_install_order_and_notify(commit_lsn);
        self.active.remove(&txn_id);
        Ok(commit_lsn)
    }

    fn abort_txn(&self, txn_id: u64) {
        self.active.remove(&txn_id);
    }

    /// Test-only hook mirroring the real `begin()` code path (the
    /// fixed two-phase publish) with two barrier pauses wedged
    /// between the sentinel insert and the counter read, and between
    /// the counter read and the snapshot upgrade.
    ///
    /// `after_snapshot_read` fires immediately after the snapshot is
    /// captured (lets the driver thread know "B has read counter").
    /// `before_publish` fires just before the active-txn entry is
    /// upgraded from the `Lsn::MAX` sentinel to the concrete snapshot
    /// (lets the driver run `gc()` in the window where the fix relies
    /// on the sentinel to anchor GC).
    ///
    /// Two barriers (not one) are required because a single barrier
    /// cannot both gate the driver's counter advance (which must run
    /// *after* B reads) and the final publish (which must run *after*
    /// the driver's gc).
    #[doc(hidden)]
    pub fn begin_with_barrier(
        &self,
        tenant: TenantId,
        after_snapshot_read: &std::sync::Barrier,
        before_publish: &std::sync::Barrier,
    ) -> Transaction<'_> {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::AcqRel);
        self.active.insert(txn_id, Lsn::MAX);
        let snapshot = Lsn::new(self.visible.load(Ordering::Acquire));
        after_snapshot_read.wait();
        before_publish.wait();
        self.active.insert(txn_id, snapshot);
        Transaction {
            manager: ManagerRef::Borrowed(self),
            txn_id,
            tenant_id: tenant,
            snapshot,
            writes: HashMap::new(),
            sidechannel_writes: Vec::new(),
            allocator_advances: Vec::new(),
            vector_pages: Vec::new(),
            idempotency_bindings: Vec::new(),
            acl_grants: Vec::new(),
            mutation_log: TxnMutationLog::new(),
            state: TxnState::Active,
        }
    }

    /// Test helper: visible value at snapshot (bypasses txn buffer).
    #[doc(hidden)]
    pub fn read_at(&self, tenant: TenantId, key: MvccKey, snapshot: Lsn) -> Option<Bytes> {
        let chain = self.versions.get(&(tenant, key))?;
        for v in chain.iter().rev() {
            if v.visible_to(snapshot) {
                return v.value.clone();
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────
// M2-11 + M2-12 + M2-14: Transaction handle
// ─────────────────────────────────────────────────────────────────────

/// Lifecycle state of a [`Transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    /// Begun, not yet committed or aborted.
    Active,
    /// `commit` succeeded.
    Committed,
    /// `abort` was called or a conflict aborted this txn.
    Aborted,
}

/// How a [`Transaction`] holds its owning [`TxnManager`].
///
/// ADR-197 §Decision layer (1): the v1.0-α transaction is always
/// constructed by borrowing the manager ([`Self::Borrowed`]) — the
/// existing [`TxnManager::begin`] path. Bolt explicit transactions
/// (ADR-197) need a transaction that survives across the `await`
/// points of a connection's async message loop, which a borrow
/// cannot. [`Self::Owned`] carries an `Arc<TxnManager>` clone instead,
/// so the transaction can be held with a `'static` lifetime
/// ([`OwnedTxn`]) and is `Send`. Both variants `Deref` to the same
/// `&TxnManager`, so every `Transaction` method (`read` / `commit` /
/// `abort` / …) is byte-for-byte identical regardless of variant —
/// this is an ownership wrapper, NOT a second transaction engine.
enum ManagerRef<'m> {
    /// The classic borrow — `TxnManager::begin(&self)`.
    Borrowed(&'m TxnManager),
    /// An `Arc`-owned manager — `TxnManager::begin_owned(&Arc<Self>)`.
    /// Enables a `Transaction<'static>` ([`OwnedTxn`]) the Bolt
    /// handler holds across `await` points.
    Owned(Arc<TxnManager>),
}

impl std::ops::Deref for ManagerRef<'_> {
    type Target = TxnManager;

    #[inline]
    fn deref(&self) -> &TxnManager {
        match self {
            ManagerRef::Borrowed(m) => m,
            ManagerRef::Owned(m) => m,
        }
    }
}

/// A snapshot-isolated transaction. Writes are buffered in a per-txn
/// delta (M2-12); reads first consult the delta (read-your-writes,
/// M2-14) and fall through to the committed version store.
pub struct Transaction<'m> {
    manager: ManagerRef<'m>,
    txn_id: u64,
    tenant_id: TenantId,
    snapshot: Lsn,
    writes: HashMap<MvccKey, Option<Bytes>>,
    /// ADR-032 Slice 2: non-primary-tenant MVCC writes that must ride
    /// the outer commit's CommitBundle as atomically as the primary
    /// writes. Populated via [`Self::register_sidechannel_mvcc_write`]
    /// or by the builder threading a `&mut Vec<SideChannelWrite>` from
    /// [`Self::commit_with_bundle`] down to `grow_root`. Phase 2
    /// encodes them into the v2 bundle payload; Phase 3 applies them
    /// via [`TxnManager::apply_sidechannel_mvcc_write`].
    sidechannel_writes: Vec<SideChannelWrite>,
    /// Issue #129 P0 fix: per-(tenant, allocator-kind) high-water
    /// snapshots that must ride this transaction's outer
    /// `CommitBundle` atomically with primary writes. Populated by
    /// the builder closure (typically by snapshotting the live
    /// allocators at builder-end via
    /// [`crate::page_alloc::PageAllocator::snapshot_advances`] +
    /// [`crate::crud::CrudStore::snapshot_allocator_advances`]).
    /// Phase 2 encodes the registered advances into the v4
    /// `CommitBundle` payload's `allocator_advances` section; replay
    /// applies them via `seed_from_advance` in commit_lsn order.
    allocator_advances: Vec<AllocatorAdvance>,
    /// M3.a Slice G.4 (commit-bundle vector page staging): vector
    /// arena page snapshots that must ride this transaction's outer
    /// `CommitBundle` atomically with primary writes. Populated by
    /// the builder closure (typically by draining the per-txn
    /// `pending_vector_emits` queue staged via
    /// [`crate::crud::CrudStore::stage_vector_page`]). Phase 2
    /// encodes the entries into the v5 `CommitBundle` payload's
    /// `vector_pages` section; replay applies them via
    /// [`crate::vector_store::VectorPageStoreHandle::install_or_replace`]
    /// AFTER `staged_pages` and BEFORE `allocator_advances`
    /// (Lemma I3 — monotonic idempotent replay). Per ADR-031
    /// amendment-02 + ADR-035 §4.5/§4.6.
    vector_pages: Vec<VectorPageEntry>,
    /// #352 Part 2 (ADR-199): `external_id → internal_id` idempotency
    /// bindings that must ride this transaction's `CommitBundle`
    /// atomically with the node/rel writes that allocated the internal
    /// ids. Populated by `crud::commit` (draining the per-txn
    /// `pending_idempotency_bindings` queue staged via
    /// [`crate::crud::CrudStore::stage_idempotency_binding`]) BEFORE the
    /// commit call — unlike `vector_pages`, the entries carry no
    /// `commit_lsn`, so they need no builder-closure access. Phase 2
    /// encodes them into the v6 `CommitBundle` payload's
    /// `idempotency_bindings` section; replay applies them via
    /// [`crate::idempotency::IdempotencyStore::install`] AFTER MVCC
    /// writes. Folding into the bundle (rather than a standalone
    /// pre-commit WAL record) is what makes the binding present iff the
    /// commit is — see ADR-199 §Revision 2026-06-07.
    idempotency_bindings: Vec<IdempotencyBindingEntry>,
    /// #1221 (ADR-218): document-level ACL grant/revoke operations that
    /// must ride this transaction's `CommitBundle` (v8 `acl_grants`
    /// section) atomically with the commit that carries them. Populated
    /// by `crud::commit` (draining the per-txn `pending_acl_grants` queue
    /// staged via [`crate::crud::CrudStore::stage_acl_grant`]) BEFORE the
    /// commit call — like `idempotency_bindings`, the entries carry no
    /// `commit_lsn`, so they need no builder-closure access. Phase 2
    /// encodes them into the v8 `CommitBundle` payload's `acl_grants`
    /// section in **staging (append) order**; replay re-drives
    /// `PermissionIndex::apply_doc_acl` / `revoke_doc` (ascending
    /// `commit_lsn` ⇒ last-writer-wins per doc). Per ADR-218.
    acl_grants: Vec<AclGrantEntry>,
    /// ADR-033 Z-1 (b): per-transaction in-memory mutation log.
    /// Populated during the builder phase via the page-store
    /// `capture_and_latch` / `install_for_txn` helpers (Phase 2b) and
    /// the blob-store `register_uncommitted_chain` helper; drained by
    /// [`TxnManager::rollback_wal_failure`] on WAL fsync failure.
    ///
    /// Always present (even for read-only transactions), because the
    /// log is small and always-allocated keeps the builder-closure
    /// signature uniform across commit variants.
    mutation_log: TxnMutationLog,
    state: TxnState,
}

impl<'m> Transaction<'m> {
    /// The snapshot LSN at which this txn sees the committed store.
    #[inline]
    #[must_use]
    pub fn snapshot(&self) -> Lsn {
        self.snapshot
    }

    /// Opaque per-instance id. Useful for logging and GC anchoring.
    #[inline]
    #[must_use]
    pub fn id(&self) -> u64 {
        self.txn_id
    }

    /// Tenant this transaction operates within. Every MVCC key is
    /// qualified by this tenant (ADR-011); CRUD callers (e.g.
    /// `crud::scan_out`) use it to key into per-tenant side stores
    /// like the TEL chain map.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant_id
    }

    /// Active bundle format selected by the attached WAL generation.
    #[must_use]
    pub(crate) fn wal_format_version(&self) -> u16 {
        self.manager.wal_format_version()
    }

    /// **RC-1 (#1366)** — read the manager's MVCC GC anchor
    /// (`oldest_active_snapshot`) through this transaction's manager
    /// handle. `crud::commit` reads it while `self` is still live (this
    /// txn is in the active set, so the value is a conservative floor)
    /// to decide which previously-enqueued secondary-index deferred
    /// removals the snapshot horizon has cleared. See
    /// [`crate::crud::CrudStore::apply_ready_deferred_removals`].
    #[inline]
    #[must_use]
    pub(crate) fn oldest_active_snapshot(&self) -> Lsn {
        self.manager.oldest_active_snapshot()
    }

    /// #352 Part 2 (ADR-199): stage the `external_id → internal_id`
    /// idempotency bindings that must ride this transaction's
    /// `CommitBundle` (v6 `idempotency_bindings` section), atomically
    /// with the node/rel writes that allocated the internal ids.
    ///
    /// Called by [`crate::crud::commit`] after draining the per-txn
    /// `pending_idempotency_bindings` queue, BEFORE the commit. Unlike
    /// `vector_pages`, the bindings carry no `commit_lsn`, so they are
    /// staged here rather than in the builder closure. Append-extends
    /// any already-staged set (a no-op for the common empty case).
    pub(crate) fn stage_idempotency_bindings(&mut self, bindings: Vec<IdempotencyBindingEntry>) {
        self.idempotency_bindings.extend(bindings);
    }

    /// #1221 (ADR-218): stage the document-level ACL grant/revoke ops
    /// that must ride this transaction's `CommitBundle` (v8 `acl_grants`
    /// section), atomically with the commit that carries them.
    ///
    /// Called by [`crate::crud::commit`] after draining the per-txn
    /// `pending_acl_grants` queue, BEFORE the commit. Like
    /// `idempotency_bindings`, the ops carry no `commit_lsn`, so they are
    /// staged here rather than in the builder closure. **Append-extends**
    /// any already-staged set — preserving the staging order the
    /// `acl_grants` encoder relies on for last-writer-wins replay (must
    /// NOT be re-sorted — ADR-218 invariant).
    pub(crate) fn stage_acl_grants(&mut self, grants: Vec<AclGrantEntry>) {
        self.acl_grants.extend(grants);
    }

    /// Lifecycle state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> TxnState {
        self.state
    }

    /// True iff this txn has buffered any write to `key`.
    #[inline]
    #[must_use]
    pub fn has_pending_write(&self, key: MvccKey) -> bool {
        self.writes.contains_key(&key)
    }

    /// Snapshot-isolated read with read-your-writes semantics.
    pub fn read(&self, key: MvccKey) -> Option<Bytes> {
        debug_assert_eq!(self.state, TxnState::Active);
        if let Some(pending) = self.writes.get(&key) {
            return pending.clone();
        }
        self.manager.read_at(self.tenant_id, key, self.snapshot)
    }

    /// Read the committed version visible at this transaction's snapshot,
    /// bypassing its own buffered writes. Commit drains use this for durable
    /// accelerator pre-images when the physical page is still deferred.
    pub(crate) fn read_snapshot(&self, key: MvccKey) -> Option<Bytes> {
        debug_assert_eq!(self.state, TxnState::Active);
        self.manager.read_at(self.tenant_id, key, self.snapshot)
    }

    /// Buffer a write. Installs happen at commit; aborts discard.
    pub fn write(&mut self, key: MvccKey, value: Bytes) {
        debug_assert_eq!(self.state, TxnState::Active);
        self.writes.insert(key, Some(value));
    }

    /// Buffer a delete (tombstone). Installs at commit.
    pub fn delete(&mut self, key: MvccKey) {
        debug_assert_eq!(self.state, TxnState::Active);
        self.writes.insert(key, None);
    }

    /// Number of buffered write-set entries.
    #[inline]
    #[must_use]
    pub fn write_set_len(&self) -> usize {
        self.writes.len()
    }

    /// ADR-033 Z-1 (b): read-only access to this transaction's
    /// in-memory mutation log. Intended for tests and observability
    /// — callers on the hot path populate the log via the
    /// page-store and blob-store helpers (Phase 2b), not by
    /// touching this reference.
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn mutation_log(&self) -> &TxnMutationLog {
        &self.mutation_log
    }

    /// M3.a Slice G.5 — mutable access to this transaction's
    /// in-memory mutation log. The production builder closure
    /// receives `&mut TxnMutationLog` directly via
    /// [`Self::commit_with_bundle_and_rollback`]; this `#[doc(hidden)]`
    /// seam exists so tests that exercise the rollback closure
    /// end-to-end via [`crate::crud::commit`] can populate
    /// `(PageStoreKind::Vector, page_id, pre_w_bytes)` entries from
    /// outside the builder closure (the entries are otherwise only
    /// pushed by the future G.7+ vector writers that thread through
    /// the closure's `&mut TxnMutationLog` argument).
    ///
    /// Production callers continue to use the builder closure's log
    /// reference. ADR-033 §3 (capture-and-latch helpers) treats this
    /// helper as test-only.
    #[doc(hidden)]
    #[inline]
    pub fn mutation_log_mut(&mut self) -> &mut TxnMutationLog {
        &mut self.mutation_log
    }

    /// OCC commit (M2-13). On conflict returns
    /// [`ArcGraphError::MvccConflict`] and the txn becomes `Aborted`.
    ///
    /// Post-ADR-031: emits a single `CommitBundle` WAL record with
    /// zero staged IndexPage entries (the legacy pre-ADR-031
    /// `Commit = 2` emission is retired on the hot path). Callers
    /// that need to fold index-layer staged emits into the same
    /// fsync should use [`Self::commit_with_bundle`].
    ///
    /// Post-ADR-032 Slice 2: bundles are encoded with the v2 codec
    /// and may carry sidechannel writes registered via
    /// [`Self::register_sidechannel_mvcc_write`].
    pub fn commit(self) -> Result<Lsn> {
        self.commit_with_bundle(|_, _, _, _, _| Ok(Vec::new()))
    }

    /// ADR-032 Slice 2: register a non-primary-tenant MVCC write to
    /// ride this transaction's outer CommitBundle atomically.
    ///
    /// Appends `(tenant, key, value)` to the transaction's
    /// `sidechannel_writes` vec. [`Self::commit_with_bundle`]
    /// passes a `&mut Vec<SideChannelWrite>` handle down to the
    /// builder so grow_root and friends can register lazily during
    /// the builder phase; this public method is for callers that
    /// know the sidechannel shape BEFORE committing (e.g., test
    /// harnesses or future admin tools).
    ///
    /// Phase 2 of `commit_with_bundle` encodes the registered writes
    /// into the v2 CommitBundle payload; Phase 3 applies each via
    /// [`TxnManager::apply_sidechannel_mvcc_write`] after WAL fsync
    /// succeeds. On WAL failure the registrations are discarded (no
    /// Phase-1 install happened; nothing to roll back).
    pub fn register_sidechannel_mvcc_write(
        &mut self,
        tenant: TenantId,
        key: MvccKey,
        value: Option<Bytes>,
    ) {
        debug_assert_eq!(self.state, TxnState::Active);
        self.sidechannel_writes.push(SideChannelWrite {
            tenant_id: tenant,
            key,
            value,
        });
    }

    /// Bundle-aware commit — ADR-031 single-fire per commit, extended
    /// by ADR-032 Slice 2 to also atomically carry non-primary-tenant
    /// sidechannel MVCC writes.
    ///
    /// The `builder` closure runs OUTSIDE `commit_gate`, between
    /// Phase 1 (silent MVCC install) and Phase 2 (single
    /// `wal.append(CommitBundle)`). It receives:
    ///
    /// - the allocated `commit_lsn`, and
    /// - a `&mut Vec<SideChannelWrite>` into which it (or a callee
    ///   like `crate::primary_index::PrimaryIndex::grow_root`) can
    ///   push sidechannel writes that must ride this commit's bundle
    ///   atomically. The vec is pre-populated with any writes
    ///   pre-registered via
    ///   [`Self::register_sidechannel_mvcc_write`].
    ///
    /// The builder returns a `Vec<StagedEmit>` containing the post-
    /// mutation byte snapshots of every index page the commit touched.
    /// Phase 2 encodes `staged_emits` + primary writes + sidechannel
    /// writes into a single CommitBundle v2 payload — one group-commit
    /// fire per commit regardless of how many tenants it touches.
    ///
    /// On builder error: identical recovery path to a WAL-failed
    /// commit. Phase-1 silent install (primary writes only;
    /// sidechannel writes never touched Phase 1) is rolled back under
    /// `commit_gate`; `install_order` advances so successors can
    /// proceed; `visible` stays unchanged.
    ///
    /// On builder success + WAL success: `visible` advances to
    /// `commit_lsn`; sidechannel writes are applied via
    /// [`TxnManager::apply_sidechannel_mvcc_write`]; the txn is
    /// `Committed`.
    ///
    /// On builder success + WAL failure: Phase 3 rolls back primary
    /// writes and returns `Err`. Sidechannel writes are NOT applied
    /// (they were never installed in Phase 1).
    pub fn commit_with_bundle<F>(self, builder: F) -> Result<Lsn>
    where
        F: FnOnce(
            Lsn,
            &mut Vec<SideChannelWrite>,
            &mut Vec<AllocatorAdvance>,
            &mut Vec<VectorPageEntry>,
            &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>>,
    {
        // ADR-033 Z-1 (b): this convenience wrapper supplies a no-op
        // rollback closure. Callers that populate the mutation_log
        // via the builder MUST use [`Self::commit_with_bundle_and_rollback`]
        // instead so the log drains on WAL fsync failure. A non-empty
        // mutation_log reaching the no-op rollback is a logic bug —
        // the closure's debug_assert catches it.
        self.commit_with_bundle_and_rollback(builder, |log| {
            debug_assert!(
                log.is_empty(),
                "commit_with_bundle called with non-empty mutation_log; \
                 use commit_with_bundle_and_rollback to supply a Z-1 rollback \
                 closure. Log: {} page_mutations, {} new_pages, {} root_changes, \
                 {} blob_heads",
                log.page_mutations.len(),
                log.new_pages.len(),
                log.root_changes.len(),
                log.blob_heads.len(),
            );
        })
    }

    /// ADR-033 Z-1 (b): bundle-aware commit with an explicit
    /// in-memory rollback closure. On WAL fsync failure or builder
    /// error, the `rollback` closure runs under `commit_gate` (after
    /// MVCC version unwind, before `install_order` advances) with
    /// `&mut TxnMutationLog` — drain the log to restore page stores
    /// to their pre-W state per ADR-033 §5 (root-ordering) and §6
    /// (sequence).
    ///
    /// For callers that do not mutate in-memory page state (pure
    /// MVCC commits without index/blob/record-page side-effects),
    /// [`Self::commit_with_bundle`] supplies a no-op rollback.
    pub fn commit_with_bundle_and_rollback<F, R>(self, builder: F, rollback: R) -> Result<Lsn>
    where
        F: FnOnce(
            Lsn,
            &mut Vec<SideChannelWrite>,
            &mut Vec<AllocatorAdvance>,
            &mut Vec<VectorPageEntry>,
            &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>>,
        R: FnOnce(&mut TxnMutationLog),
    {
        let wal = self.manager.wal.read().clone();
        self.commit_with_bundle_apply_and_rollback(
            builder,
            move |_, commit_lsn| {
                if let Some(wal) = wal {
                    wal.take_exact_durable(commit_lsn);
                }
                Ok(())
            },
            rollback,
        )
    }

    pub(crate) fn commit_with_bundle_apply_and_rollback<F, A, R>(
        mut self,
        builder: F,
        apply: A,
        rollback: R,
    ) -> Result<Lsn>
    where
        F: FnOnce(
            Lsn,
            &mut Vec<SideChannelWrite>,
            &mut Vec<AllocatorAdvance>,
            &mut Vec<VectorPageEntry>,
            &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>>,
        A: FnOnce(&[DeltaOp], Lsn) -> Result<()>,
        R: FnOnce(&mut TxnMutationLog),
    {
        debug_assert_eq!(self.state, TxnState::Active);
        match self.manager.commit_with_bundle_writes(
            self.txn_id,
            self.tenant_id,
            self.snapshot,
            &self.writes,
            &mut self.sidechannel_writes,
            &mut self.allocator_advances,
            &mut self.vector_pages,
            &self.idempotency_bindings,
            &self.acl_grants,
            &mut self.mutation_log,
            builder,
            apply,
            rollback,
        ) {
            Ok(commit_lsn) => {
                self.state = TxnState::Committed;
                Ok(commit_lsn)
            }
            Err(e) => {
                self.manager.abort_txn(self.txn_id);
                self.state = TxnState::Aborted;
                Err(e)
            }
        }
    }

    /// Test-only hook mirroring the *current* (buggy, pre-Bug-2-fix)
    /// `commit()` code path with two barrier pauses: one immediately
    /// after `counter.allocate()` advances (lets the driver confirm
    /// the mid-install window has opened), and one after the first
    /// key in `key_order` is installed (lets the driver observe the
    /// half-applied state).
    ///
    /// The buffered write set is installed in `key_order` rather than
    /// HashMap iteration order so the test is deterministic.
    ///
    /// Commit 4 (the Bug 2 fix) updates this hook to match the new
    /// two-counter commit path so the reproducer continues to stress
    /// real code.
    #[doc(hidden)]
    pub fn commit_with_barriers(
        mut self,
        key_order: &[MvccKey],
        between_alloc_and_install: &std::sync::Barrier,
        between_first_and_second: &std::sync::Barrier,
    ) -> Result<Lsn> {
        debug_assert_eq!(self.state, TxnState::Active);
        assert_eq!(
            key_order.len(),
            self.writes.len(),
            "key_order must enumerate every buffered write exactly once"
        );
        let result = self.manager.commit_with_barriers_raw(
            self.txn_id,
            self.tenant_id,
            self.snapshot,
            key_order,
            &self.writes,
            between_alloc_and_install,
            between_first_and_second,
        );
        match result {
            Ok(lsn) => {
                self.state = TxnState::Committed;
                Ok(lsn)
            }
            Err(e) => {
                self.manager.abort_txn(self.txn_id);
                self.state = TxnState::Aborted;
                Err(e)
            }
        }
    }

    /// Discard buffered writes and release the active-txn slot.
    pub fn abort(mut self) {
        if self.state == TxnState::Active {
            self.manager.abort_txn(self.txn_id);
            self.state = TxnState::Aborted;
        }
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if self.state == TxnState::Active {
            self.manager.abort_txn(self.txn_id);
            self.state = TxnState::Aborted;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ADR-197 §Decision layer (1): OwnedTxn — an owning transaction handle
// the Bolt explicit-transaction layer holds across async `await`
// points.
// ─────────────────────────────────────────────────────────────────────

/// A [`Transaction`] that OWNS its [`TxnManager`] (via an `Arc` clone),
/// so it has a `'static` lifetime and is `Send` — holdable across the
/// `await` points of a Bolt connection's async message loop and
/// movable between threads.
///
/// # Why this exists (ADR-197)
///
/// Bolt 5 explicit transactions (BEGIN…COMMIT/ROLLBACK) span multiple
/// RUN messages, each awaited separately on the connection task. The
/// borrowed [`Transaction<'m>`] cannot survive across those awaits
/// (its `&'m TxnManager` borrow does not outlive a single call).
/// `OwnedTxn` carries an `Arc<TxnManager>` instead (see
/// `ManagerRef::Owned`), keeping the manager alive for the
/// transaction's whole lifetime.
///
/// # It is an ownership wrapper, NOT a new transaction engine
///
/// Every operation delegates to the SAME inner [`Transaction`] — the
/// same buffered write-set, the same snapshot, the same ADR-031/033
/// commit path, the same abort/Drop semantics. `OwnedTxn::commit` is
/// `Transaction::commit`; `OwnedTxn::abort` is `Transaction::abort`.
/// MVCC snapshot isolation (ADR-047) is preserved because the inner
/// transaction holds one snapshot LSN for its whole lifetime.
///
/// # Staging writes + the COMMIT seam (ADR-197 #802 R1 finding #1)
///
/// CRUD callers stage reads/writes through the inner transaction
/// borrowed via [`Self::txn`] / [`Self::txn_mut`] (the same
/// `crud::*(&tx)` / `crud::*(&mut tx)` free functions the auto-commit
/// path uses). Those `crud::*` writes buffer their primary-index
/// installs, CDC events, and TEL appends in the
/// [`crate::crud::CrudStore`] keyed by [`Self::id`] — exactly as the
/// auto-commit path does. At ROLLBACK / abort / Drop the MVCC writes
/// are discarded.
///
/// At BEGIN…COMMIT the held writes MUST commit through the SAME
/// [`crate::crud::commit`] machinery the auto-commit path uses —
/// `crud::commit(owned.into_inner(), &store)` (see [`Self::into_inner`])
/// — so the primary-index dual-write + WAL bundle + CDC flush + TEL
/// drain all fire under one CommitBundle. The bare [`Self::commit`]
/// commits the MVCC version store ONLY and is reserved for pure-MVCC
/// held txns (raw `tx.write` with no CrudStore side effects); using it
/// for a CrudStore-staged tx is the finding-#1 silent half-commit.
pub struct OwnedTxn {
    inner: Transaction<'static>,
}

impl std::fmt::Debug for OwnedTxn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Transaction` is not `Debug` (it holds the manager handle);
        // surface the safe scalar fields for diagnostics.
        f.debug_struct("OwnedTxn")
            .field("txn_id", &self.inner.txn_id)
            .field("tenant", &self.inner.tenant_id)
            .field("snapshot", &self.inner.snapshot)
            .field("state", &self.inner.state)
            .field("write_set_len", &self.inner.writes.len())
            .finish()
    }
}

impl OwnedTxn {
    /// Borrow the inner transaction for read-only CRUD ops
    /// (`crud::read_node(&tx, …)`, `tx.read(key)`, `tx.snapshot()`).
    #[inline]
    #[must_use]
    pub fn txn(&self) -> &Transaction<'static> {
        &self.inner
    }

    /// Borrow the inner transaction mutably for staging writes
    /// (`crud::create_node(&crud, &mut tx, …)`, `tx.write(key, val)`).
    /// Writes buffer in the held write-set and only become durable at
    /// [`Self::commit`].
    #[inline]
    pub fn txn_mut(&mut self) -> &mut Transaction<'static> {
        &mut self.inner
    }

    /// The snapshot LSN this transaction reads at. Held constant for
    /// the transaction's lifetime (snapshot isolation, ADR-047).
    #[inline]
    #[must_use]
    pub fn snapshot(&self) -> Lsn {
        self.inner.snapshot()
    }

    /// Opaque per-instance transaction id.
    #[inline]
    #[must_use]
    pub fn id(&self) -> u64 {
        self.inner.id()
    }

    /// Tenant this transaction is scoped to (ADR-011).
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.inner.tenant()
    }

    /// Lifecycle state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> TxnState {
        self.inner.state()
    }

    /// Commit the held transaction at the **MVCC-version-store layer
    /// only** — the SAME code as [`Transaction::commit`]. On MVCC
    /// write-write conflict returns [`ArcGraphError::MvccConflict`] and
    /// the transaction is aborted.
    ///
    /// # ⚠️ This does NOT drain CrudStore-buffered side effects
    ///
    /// A held tx whose writes were staged through the `crud::*` free
    /// functions (`crud::create_node`, `crud::create_rel`, …) buffers
    /// its primary-index installs, CDC events, and TEL appends in the
    /// [`crate::crud::CrudStore`] keyed by [`Self::id`]. Those are
    /// drained ONLY by [`crate::crud::commit`]. Committing such a tx via
    /// this method installs the MVCC versions but SKIPS the
    /// primary-index dual-write + WAL bundle + CDC flush — a silent
    /// half-commit (ADR-197 #802 R1 finding #1). The Bolt explicit-tx
    /// COMMIT path therefore MUST commit through
    /// `crate::crud::commit(owned.into_inner(), store)` (see
    /// [`Self::into_inner`]); this method is reserved for **pure-MVCC**
    /// held txns that stage raw `tx.write(key, val)` only (no CrudStore
    /// side effects) — e.g. the storage-layer unit tests + the primary
    /// index's own root-pointer persist.
    pub fn commit(self) -> Result<Lsn> {
        self.inner.commit()
    }

    /// ADR-197 #802 R1 finding #1 — move the inner
    /// [`Transaction<'static>`](Transaction) OUT so a CrudStore-aware
    /// caller can commit it through the FULL
    /// [`crate::crud::commit`] machinery (primary-index dual-write +
    /// WAL bundle + CDC flush + TEL drain), converging the explicit-tx
    /// COMMIT with the auto-commit path. The inner transaction carries
    /// the same buffered MVCC write-set + the same txn id the CrudStore
    /// keyed its pending installs/CDC/TEL by, so
    /// `crud::commit(owned.into_inner(), &store)` drains them all under
    /// one CommitBundle fsync.
    ///
    /// `OwnedTxn` has no `Drop` of its own (it relies on the inner
    /// `Transaction`'s Drop-aborts-if-Active), so this move is the
    /// MSRV-safe handoff: the wrapper is consumed and the returned
    /// `Transaction` either commits (consumed by `crud::commit`) or, if
    /// dropped without commit, aborts via `Transaction::drop`.
    #[must_use = "the moved-out Transaction must be committed (crud::commit) or it Drop-aborts"]
    pub fn into_inner(self) -> Transaction<'static> {
        self.inner
    }

    /// Abort the held transaction — discards all buffered writes (the
    /// SAME code as [`Transaction::abort`]). This is the real ROLLBACK:
    /// nothing the transaction staged becomes visible. Also fires
    /// implicitly on Drop if neither `commit` nor `abort` was called
    /// (Bolt connection-drop / RESET-mid-tx → no leaked partial
    /// commit).
    pub fn abort(self) {
        self.inner.abort();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests — small-case invariants. Heavy property tests live in
// `tests/mvcc_*.rs` integration files.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ADR-032 Slice 2: sidechannel MVCC write primitives ──────

    #[test]
    fn apply_sidechannel_mvcc_write_no_occ_visibility() {
        // The sidechannel primitive must:
        // (1) push a Version onto the (tenant, key) chain with
        //     created_lsn = commit_lsn,
        // (2) expire any prior live Version at commit_lsn,
        // (3) skip OCC (no MvccConflict on concurrent-but-
        //     write_gate-serialized callers),
        // (4) make the new value visible to readers at snapshot ==
        //     commit_lsn.
        let m = TxnManager::new();

        // Seed: a prior SYSTEM write at LSN 1.
        let mut t1 = m.begin(TenantId::SYSTEM);
        t1.write(42u64, Bytes::from_static(b"initial"));
        let lsn1 = t1.commit().unwrap();
        assert_eq!(lsn1.raw(), 1);

        // The allocator advanced past lsn1; a sidechannel write at
        // a fresh commit_lsn = lsn1+1 is the normal shape grow_root
        // produces when it rides inside an outer user commit.
        let sc_lsn = Lsn::new(lsn1.raw() + 1);
        // Manually advance counter so future reads don't see stale
        // visible watermark. (In real usage this is handled by the
        // outer commit's Phase 3 visible.store.)
        m.apply_sidechannel_mvcc_write(
            sc_lsn,
            TenantId::SYSTEM,
            42u64,
            Some(Bytes::from_static(b"sidechannel")),
        );
        m.visible.store(sc_lsn.raw(), Ordering::Release);

        // (1) + (2): the chain now has 2 versions; the seed is
        //     expired at sc_lsn.
        let chain_len = m.chain_len(TenantId::SYSTEM, 42u64);
        assert_eq!(chain_len, 2, "seed + sidechannel write");

        // (4): a reader at snapshot = sc_lsn sees the new value.
        let reader_new = m.begin(TenantId::SYSTEM);
        assert_eq!(reader_new.read(42u64).as_deref(), Some(&b"sidechannel"[..]));

        // A reader at the old snapshot (lsn1) still sees the seed —
        // visibility is unchanged for pre-sidechannel snapshots.
        let old_value = m.read_at(TenantId::SYSTEM, 42u64, lsn1);
        assert_eq!(old_value.as_deref(), Some(&b"initial"[..]));

        // (3): no OCC validation — a concurrent user-tenant write
        //     to the same key would conflict under OCC, but that's
        //     outside the sidechannel primitive's contract (callers
        //     hold external write_gate).
    }

    // ─── #1200: coalesce_sidechannel_writes (last-wins by (tenant,key))

    fn sc(tenant: u64, key: MvccKey, val: &'static [u8]) -> SideChannelWrite {
        SideChannelWrite {
            tenant_id: TenantId::new(tenant),
            key,
            value: Some(Bytes::from_static(val)),
        }
    }

    #[test]
    fn coalesce_sidechannel_writes_collapses_duplicate_root_key_last_wins() {
        // The #1200 shape: two grow_roots in one commit push two
        // (SYSTEM, ROOT_KEY) writes — sc[0] = intermediate root,
        // sc[1] = final root. Coalescing must keep ONLY the last (final
        // root); without this, replay's value-blind idempotency skip
        // strands the durable root at the intermediate one.
        let root_key = crate::primary_index::PRIMARY_INDEX_ROOT_KEY;
        let intermediate = 3u64.to_le_bytes();
        let final_root = 259u64.to_le_bytes();
        let mut writes = vec![
            SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: root_key,
                value: Some(Bytes::copy_from_slice(&intermediate)),
            },
            SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: root_key,
                value: Some(Bytes::copy_from_slice(&final_root)),
            },
        ];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert_eq!(writes.len(), 1, "two same-key writes collapse to one");
        assert_eq!(writes[0].tenant_id, TenantId::SYSTEM);
        assert_eq!(writes[0].key, root_key);
        assert_eq!(
            writes[0].value.as_deref(),
            Some(&final_root[..]),
            "last-wins: the FINAL root (259) survives, not the intermediate (3)"
        );
    }

    #[test]
    fn coalesce_sidechannel_writes_single_entry_unchanged() {
        let mut writes = vec![sc(0, 1, b"only")];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].value.as_deref(), Some(&b"only"[..]));
    }

    #[test]
    fn coalesce_sidechannel_writes_empty_unchanged() {
        let mut writes: Vec<SideChannelWrite> = vec![];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert!(writes.is_empty());
    }

    #[test]
    fn coalesce_sidechannel_writes_distinct_keys_not_collapsed() {
        // Distinct (tenant, key) writes must all survive — only exact
        // (tenant, key) duplicates collapse. Order is preserved.
        let mut writes = vec![sc(0, 1, b"a"), sc(0, 2, b"b"), sc(7, 1, b"c")];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert_eq!(writes.len(), 3, "distinct keys are not collapsed");
        assert_eq!(writes[0].key, 1);
        assert_eq!(writes[0].tenant_id, TenantId::new(0));
        assert_eq!(writes[0].value.as_deref(), Some(&b"a"[..]));
        assert_eq!(writes[1].key, 2);
        assert_eq!(writes[1].value.as_deref(), Some(&b"b"[..]));
        assert_eq!(writes[2].key, 1);
        assert_eq!(writes[2].tenant_id, TenantId::new(7));
        assert_eq!(writes[2].value.as_deref(), Some(&b"c"[..]));
    }

    #[test]
    fn coalesce_sidechannel_writes_keeps_last_among_three_duplicates() {
        // Three writes to the same key (3 grow_roots, the >111k-key
        // height-3→4 case) → only the LAST VALUE survives. Surviving
        // entries are placed at their last-occurrence index (matching
        // `coalesce_staged_pages`): key 1's last occurrence is index 3,
        // key 9's is index 1, so the deterministic output is [k9, k1=v3].
        let mut writes = vec![
            sc(0, 1, b"v1"),
            sc(0, 9, b"other"),
            sc(0, 1, b"v2"),
            sc(0, 1, b"v3"),
        ];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert_eq!(writes.len(), 2);
        // key 9 (last-occurrence index 1) precedes key 1 (last-occurrence
        // index 3, value v3 = last-wins).
        assert_eq!(writes[0].key, 9);
        assert_eq!(writes[0].value.as_deref(), Some(&b"other"[..]));
        assert_eq!(writes[1].key, 1);
        assert_eq!(
            writes[1].value.as_deref(),
            Some(&b"v3"[..]),
            "last value (v3) wins among the three same-key writes"
        );
    }

    #[test]
    fn coalesce_sidechannel_writes_tombstone_last_wins() {
        // A tombstone (value=None) as the last write must win over an
        // earlier value at the same key.
        let mut writes = vec![
            sc(0, 5, b"alive"),
            SideChannelWrite {
                tenant_id: TenantId::new(0),
                key: 5,
                value: None,
            },
        ];
        TxnManager::coalesce_sidechannel_writes(&mut writes);
        assert_eq!(writes.len(), 1);
        assert!(
            writes[0].value.is_none(),
            "last-wins tombstone survives over earlier value"
        );
    }

    #[test]
    fn apply_sidechannel_mvcc_write_on_empty_chain_creates_first_version() {
        let m = TxnManager::new();
        let sc_lsn = m.counter.allocate();
        m.apply_sidechannel_mvcc_write(
            sc_lsn,
            TenantId::SYSTEM,
            0xDEAD_BEEFu64,
            Some(Bytes::from_static(b"fresh")),
        );
        m.visible.store(sc_lsn.raw(), Ordering::Release);

        assert_eq!(m.chain_len(TenantId::SYSTEM, 0xDEAD_BEEFu64), 1);
        let reader = m.begin(TenantId::SYSTEM);
        assert_eq!(reader.read(0xDEAD_BEEFu64).as_deref(), Some(&b"fresh"[..]));
    }

    #[test]
    fn apply_sidechannel_mvcc_write_tombstone_roundtrip() {
        let m = TxnManager::new();
        // Seed.
        let mut t1 = m.begin(TenantId::SYSTEM);
        t1.write(7u64, Bytes::from_static(b"live"));
        t1.commit().unwrap();

        // Sidechannel tombstone.
        let sc_lsn = m.counter.allocate();
        m.apply_sidechannel_mvcc_write(sc_lsn, TenantId::SYSTEM, 7u64, None);
        m.visible.store(sc_lsn.raw(), Ordering::Release);

        let reader = m.begin(TenantId::SYSTEM);
        assert_eq!(
            reader.read(7u64),
            None,
            "sidechannel tombstone masks prior live version at new snapshot"
        );
    }

    #[test]
    fn register_sidechannel_mvcc_write_accumulates_on_transaction() {
        // ADR-032 Slice 2: the public register API on Transaction
        // pushes to an internal Vec that Phase 2 reads for bundle
        // encoding + Phase 3 reads for apply.
        let m = TxnManager::new();
        let mut t = m.begin(TenantId::DEFAULT);

        // Some user writes (so the commit isn't a no-op).
        t.write(1u64, Bytes::from_static(b"user"));

        t.register_sidechannel_mvcc_write(
            TenantId::SYSTEM,
            0xA5A5u64,
            Some(Bytes::from_static(b"sys-a")),
        );
        t.register_sidechannel_mvcc_write(TenantId::SYSTEM, 0x5A5Au64, None);
        assert_eq!(t.sidechannel_writes.len(), 2);

        let _lsn = t.commit().unwrap();

        // Post-commit both sidechannel writes are visible on the
        // SYSTEM tenant at the current visible watermark.
        let reader = m.begin(TenantId::SYSTEM);
        assert_eq!(reader.read(0xA5A5u64).as_deref(), Some(&b"sys-a"[..]));
        assert_eq!(reader.read(0x5A5Au64), None); // tombstone
    }

    #[test]
    fn lsn_counter_is_monotonic() {
        let c = LsnCounter::new();
        let a = c.allocate();
        let b = c.allocate();
        let d = c.allocate();
        assert!(a.raw() < b.raw());
        assert!(b.raw() < d.raw());
        assert_eq!(c.current(), d);
    }

    #[test]
    fn lsn_counter_starts_at_initial() {
        let c = LsnCounter::new();
        assert_eq!(c.allocate().raw(), LsnCounter::INITIAL);
    }

    #[test]
    fn lsn_counter_with_floor_resumes() {
        let c = LsnCounter::with_floor(42);
        assert_eq!(c.allocate().raw(), 42);
        assert_eq!(c.allocate().raw(), 43);
    }

    #[test]
    fn install_order_wait_keys_on_range_predecessor() {
        use std::sync::mpsc;
        use std::time::Duration;

        let manager = Arc::new(TxnManager::new());
        let range = RedoLsnRange::new(Lsn::new(5), Lsn::new(9)).unwrap();
        *manager.install_order.lock() = range.predecessor().raw();

        let (done_tx, done_rx) = mpsc::channel();
        let waiter_manager = Arc::clone(&manager);
        let waiter = std::thread::spawn(move || {
            waiter_manager.wait_for_install_turn(range);
            done_tx.send(()).unwrap();
        });
        if done_rx.recv_timeout(Duration::from_secs(1)).is_err() {
            // Unstick a regressed `commit_lsn - 1` waiter before
            // failing, so the test never leaks a parked thread.
            manager.advance_install_order_and_notify(range.end());
            waiter.join().unwrap();
            panic!("install_order waited for range end - 1 instead of range base - 1");
        }
        waiter.join().unwrap();
    }

    #[test]
    fn successor_range_waits_until_predecessor_range_completes() {
        use std::sync::mpsc;
        use std::time::Duration;

        let manager = Arc::new(TxnManager::new());
        let first = manager.counter.allocate_range(3);
        let second = manager.counter.allocate_range(2);
        assert_eq!(second.predecessor(), first.end());

        let (done_tx, done_rx) = mpsc::channel();
        let waiter_manager = Arc::clone(&manager);
        let waiter = std::thread::spawn(move || {
            waiter_manager.wait_for_install_turn(second);
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(25)).is_err(),
            "successor passed install_order before predecessor completed"
        );
        manager.advance_install_order_and_notify(first.end());
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("successor did not wake after predecessor range completed");
        waiter.join().unwrap();
    }

    #[test]
    fn visibility_basic() {
        let v = Version {
            created_lsn: Lsn::new(10),
            expired_lsn: Lsn::new(20),
            value: Some(Bytes::from_static(b"x")),
        };
        assert!(!v.visible_to(Lsn::new(9)));
        assert!(v.visible_to(Lsn::new(10)));
        assert!(v.visible_to(Lsn::new(19)));
        assert!(!v.visible_to(Lsn::new(20)));
    }

    #[test]
    fn visibility_live() {
        let v = Version {
            created_lsn: Lsn::new(10),
            expired_lsn: Lsn::MAX,
            value: Some(Bytes::from_static(b"x")),
        };
        assert!(v.visible_to(Lsn::new(10)));
        assert!(v.visible_to(Lsn::new(1_000_000)));
        assert!(!v.visible_to(Lsn::new(9)));
    }

    #[test]
    fn commit_installs_version() {
        let m = TxnManager::new();
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(7, Bytes::from_static(b"hello"));
        let lsn = t.commit().unwrap();
        assert_eq!(m.chain_len(TenantId::DEFAULT, 7), 1);
        assert_eq!(
            m.read_at(TenantId::DEFAULT, 7, lsn).as_deref(),
            Some(&b"hello"[..])
        );
    }

    #[test]
    fn ryw_shadows_committed() {
        let m = TxnManager::new();
        let mut t0 = m.begin(TenantId::DEFAULT);
        t0.write(1, Bytes::from_static(b"v0"));
        t0.commit().unwrap();
        let mut t = m.begin(TenantId::DEFAULT);
        assert_eq!(t.read(1).as_deref(), Some(&b"v0"[..]));
        t.write(1, Bytes::from_static(b"v1"));
        assert_eq!(t.read(1).as_deref(), Some(&b"v1"[..]));
        t.delete(1);
        assert_eq!(t.read(1), None);
    }

    #[test]
    fn ww_conflict_detected() {
        let m = TxnManager::new();
        let mut a = m.begin(TenantId::DEFAULT);
        let mut b = m.begin(TenantId::DEFAULT);
        a.write(9, Bytes::from_static(b"a"));
        b.write(9, Bytes::from_static(b"b"));
        a.commit().unwrap();
        let err = b.commit().unwrap_err();
        assert!(matches!(err, ArcGraphError::MvccConflict { .. }));
    }

    #[test]
    fn disjoint_writes_both_commit() {
        let m = TxnManager::new();
        let mut a = m.begin(TenantId::DEFAULT);
        let mut b = m.begin(TenantId::DEFAULT);
        a.write(1, Bytes::from_static(b"a"));
        b.write(2, Bytes::from_static(b"b"));
        assert!(a.commit().is_ok());
        assert!(b.commit().is_ok());
        // O-C (W28-S3): read the writes back through a fresh snapshot.
        // The prior `is_ok()`-only oracle proved both commits *returned*
        // Ok but never that the disjoint writes were actually installed
        // / visible — a commit path that dropped the write on the floor
        // while returning Ok would have passed.
        let t = m.begin(TenantId::DEFAULT);
        assert_eq!(t.read(1).as_deref(), Some(&b"a"[..]));
        assert_eq!(t.read(2).as_deref(), Some(&b"b"[..]));
    }

    #[test]
    fn snapshot_isolation_ignores_later_writes() {
        let m = TxnManager::new();
        let mut t0 = m.begin(TenantId::DEFAULT);
        t0.write(1, Bytes::from_static(b"v0"));
        t0.commit().unwrap();
        let reader = m.begin(TenantId::DEFAULT);
        let mut writer = m.begin(TenantId::DEFAULT);
        writer.write(1, Bytes::from_static(b"v1"));
        writer.commit().unwrap();
        assert_eq!(reader.read(1).as_deref(), Some(&b"v0"[..]));
    }

    #[test]
    fn abort_drops_writes() {
        let m = TxnManager::new();
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(3, Bytes::from_static(b"x"));
        t.abort();
        assert_eq!(m.chain_len(TenantId::DEFAULT, 3), 0);
    }

    #[test]
    fn drop_without_commit_aborts() {
        let m = TxnManager::new();
        {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(4, Bytes::from_static(b"y"));
        }
        assert_eq!(m.chain_len(TenantId::DEFAULT, 4), 0);
        assert_eq!(m.active_count(), 0);
    }

    // ── ADR-197 layer (1): OwnedTxn (begin_owned) invariants ──────────

    #[test]
    fn owned_txn_commit_persists_writes() {
        // The held-tx COMMIT path is the SAME commit machinery as the
        // borrowed Transaction: a write staged in an OwnedTxn becomes
        // visible after commit().
        let m = Arc::new(TxnManager::new());
        let mut t = m.begin_owned(TenantId::DEFAULT);
        t.txn_mut().write(10, Bytes::from_static(b"owned-v1"));
        let lsn = t.commit().unwrap();
        assert_eq!(lsn.raw(), 1);
        // A fresh reader sees the committed write.
        let r = m.begin(TenantId::DEFAULT);
        assert_eq!(r.read(10).as_deref(), Some(&b"owned-v1"[..]));
        assert_eq!(m.chain_len(TenantId::DEFAULT, 10), 1);
    }

    #[test]
    fn owned_txn_abort_discards_writes() {
        // The load-bearing ROLLBACK discriminator at the storage layer:
        // abort() on a held tx leaves NO version on the chain.
        let m = Arc::new(TxnManager::new());
        let mut t = m.begin_owned(TenantId::DEFAULT);
        t.txn_mut().write(11, Bytes::from_static(b"rolled-back"));
        t.abort();
        assert_eq!(
            m.chain_len(TenantId::DEFAULT, 11),
            0,
            "abort() must discard the staged write — no version installed"
        );
        assert_eq!(m.active_count(), 0, "abort releases the active-set anchor");
    }

    #[test]
    fn owned_txn_drop_without_commit_aborts() {
        // Connection-drop / RESET-mid-tx safety: dropping an OwnedTxn
        // without commit/abort aborts it (no leaked partial commit, no
        // leaked active-set anchor).
        let m = Arc::new(TxnManager::new());
        {
            let mut t = m.begin_owned(TenantId::DEFAULT);
            t.txn_mut().write(12, Bytes::from_static(b"leaked?"));
        }
        assert_eq!(m.chain_len(TenantId::DEFAULT, 12), 0);
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn owned_txn_read_your_writes() {
        // Read-your-writes within the held tx (the multi-statement
        // BEGIN→RUN(read-back-own-write) case): a write staged in the
        // OwnedTxn is visible to a read on the SAME tx before commit.
        let m = Arc::new(TxnManager::new());
        let mut t = m.begin_owned(TenantId::DEFAULT);
        t.txn_mut().write(13, Bytes::from_static(b"mine"));
        assert_eq!(t.txn().read(13).as_deref(), Some(&b"mine"[..]));
    }

    #[test]
    fn owned_txn_snapshot_isolation_uncommitted_invisible_to_others() {
        // Snapshot isolation (ADR-047): a held-but-uncommitted write is
        // NOT visible to a concurrent transaction on a different
        // snapshot.
        let m = Arc::new(TxnManager::new());
        let mut writer = m.begin_owned(TenantId::DEFAULT);
        writer
            .txn_mut()
            .write(14, Bytes::from_static(b"uncommitted"));
        // A concurrent reader (separate snapshot) does NOT see the
        // uncommitted write.
        let reader = m.begin(TenantId::DEFAULT);
        assert_eq!(
            reader.read(14),
            None,
            "uncommitted held-tx write must be invisible"
        );
        // After the writer commits, a NEW reader sees it (the old
        // reader keeps its snapshot).
        writer.commit().unwrap();
        let reader2 = m.begin(TenantId::DEFAULT);
        assert_eq!(reader2.read(14).as_deref(), Some(&b"uncommitted"[..]));
        assert_eq!(
            reader.read(14),
            None,
            "the pre-commit snapshot is unchanged"
        );
    }

    #[test]
    fn owned_txn_is_send() {
        // The whole point of OwnedTxn: it must be `Send` so the Bolt
        // connection task can hold it across `await`. Compile-time
        // assertion.
        fn assert_send<T: Send>() {}
        assert_send::<OwnedTxn>();
        // And holdable across a thread move (Arc keeps the manager
        // alive).
        let m = Arc::new(TxnManager::new());
        let mut t = m.begin_owned(TenantId::DEFAULT);
        t.txn_mut().write(15, Bytes::from_static(b"threaded"));
        let lsn = std::thread::spawn(move || t.commit().unwrap())
            .join()
            .unwrap();
        assert_eq!(lsn.raw(), 1);
        assert_eq!(m.chain_len(TenantId::DEFAULT, 15), 1);
    }

    #[test]
    fn gc_reclaims_old_versions() {
        let m = TxnManager::new();
        for v in 0..5u8 {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::copy_from_slice(&[v]));
            t.commit().unwrap();
        }
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 5);
        let stats = m.gc();
        // O-M (W28-S3): deterministic fixture — 5 committed versions, no
        // active snapshot, GC keeps the latest 1 and reclaims exactly 4.
        // Was `>= 4`, which a version-leaking GC (reclaiming too few via
        // an inflated count, or too many) could not be caught by; the
        // exact `== 4` is consistent with the `chain_len == 1` below.
        assert_eq!(stats.reclaimed, 4, "stats={stats:?}");
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 1);
    }

    #[test]
    fn gc_respects_active_snapshot() {
        let m = TxnManager::new();
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(1, Bytes::from_static(b"v0"));
        t.commit().unwrap();
        let reader = m.begin(TenantId::DEFAULT);
        for v in 1..5u8 {
            let mut w = m.begin(TenantId::DEFAULT);
            w.write(1, Bytes::copy_from_slice(&[v]));
            w.commit().unwrap();
        }
        let _ = m.gc();
        assert_eq!(reader.read(1).as_deref(), Some(&b"v0"[..]));
    }

    #[test]
    fn read_only_txn_does_not_bump_lsn() {
        let m = TxnManager::new();
        let before = m.current_lsn();
        let t = m.begin(TenantId::DEFAULT);
        let _ = t.read(1);
        t.commit().unwrap();
        assert_eq!(m.current_lsn(), before);
    }

    #[test]
    fn tombstone_version_survives_gc_while_live() {
        let m = TxnManager::new();
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(1, Bytes::from_static(b"v"));
        t.commit().unwrap();
        let mut t2 = m.begin(TenantId::DEFAULT);
        t2.delete(1);
        t2.commit().unwrap();
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 2);
        let _ = m.gc();
        // Old version reclaimed; live tombstone retained.
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 1);
    }

    #[test]
    fn gc_pruned_keys_count_excludes_racing_repopulations() {
        use std::sync::{Arc, Barrier};

        // We want to force `gc()` to observe an empty chain for key
        // K (so `empty_after = true`), then a racing commit
        // repopulates K before `remove_if` fires. `remove_if`'s
        // predicate returns false; stats.pruned_keys must NOT be
        // incremented for K.
        //
        // Setup: install a live version at K=A, then overwrite it
        // with a tombstone at K (tombstones with expired=MAX stay
        // live — but we want the chain to go empty, so we instead
        // hand-craft the version chain via private access to
        // simulate the post-retain empty state. This is cleaner
        // than orchestrating the write pattern needed to produce
        // a chain whose retain drops every entry.
        let m = Arc::new(TxnManager::new());
        // Key K1: chain with only an already-expired version.
        // retain under anchor=max will drop it, producing
        // empty_after=true.
        m.versions.insert(
            (TenantId::DEFAULT, 1),
            vec![Version {
                created_lsn: Lsn::new(1),
                expired_lsn: Lsn::new(2),
                value: Some(Bytes::from_static(b"x")),
            }],
        );
        // Advance `visible` so anchor (no active txns) >= 2.
        m.visible.store(10, Ordering::Release);

        let before_remove_if = Arc::new(Barrier::new(2));
        let after_repopulate = Arc::new(Barrier::new(2));

        let gc_handle = {
            let m = Arc::clone(&m);
            let b1 = Arc::clone(&before_remove_if);
            let b2 = Arc::clone(&after_repopulate);
            std::thread::spawn(move || m.gc_with_prune_barrier(&b1, &b2))
        };

        // Wait for gc to reach the prune check with empty_after=true.
        before_remove_if.wait();
        // Racing commit: repopulate K1 before remove_if runs.
        m.versions
            .get_mut(&(TenantId::DEFAULT, 1))
            .unwrap()
            .push(Version {
                created_lsn: Lsn::new(11),
                expired_lsn: Lsn::MAX,
                value: Some(Bytes::from_static(b"new")),
            });
        // Signal gc to proceed with remove_if now that chain is
        // repopulated.
        after_repopulate.wait();

        let stats = gc_handle.join().unwrap();
        // K1 was repopulated → predicate false → NOT pruned.
        assert_eq!(
            stats.pruned_keys, 0,
            "pruned_keys should exclude racing repopulations; stats={stats:?}"
        );
        // Racing-installed version survives.
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 1);
    }

    #[test]
    fn gc_pruned_keys_counts_truly_empty_chains() {
        let m = TxnManager::new();
        // Chain K=1 with only already-expired entries — will be
        // fully drained and pruned.
        m.versions.insert(
            (TenantId::DEFAULT, 1),
            vec![Version {
                created_lsn: Lsn::new(1),
                expired_lsn: Lsn::new(2),
                value: Some(Bytes::from_static(b"x")),
            }],
        );
        m.visible.store(10, Ordering::Release);
        let stats = m.gc();
        assert_eq!(stats.pruned_keys, 1, "stats={stats:?}");
        assert_eq!(m.chain_len(TenantId::DEFAULT, 1), 0);
    }

    /// PR #243 round-2 MED-1 regression test — GC + concurrent
    /// commit interleave does not drop a still-visible record from
    /// the per-tenant chain index.
    ///
    /// Round-1 reviewer surfaced a window where:
    ///
    ///   T_gc:    versions.remove_if(K, chain_empty) succeeds.
    ///   T_commit: chain.push(v) + register_chain_key(K).
    ///   T_gc:    (would-have-been) unregister_chain_key(K).
    ///
    /// Final state pre-fix-up: `versions[(t, K)]` has T_commit's
    /// live version but `tenant_chain_keys[t]` does NOT contain `K`
    /// — a false-negative that violates the documented superset
    /// invariant. `for_each_visible_record(t, ...)` would miss the
    /// freshly-committed record.
    ///
    /// Post-fix-up (this PR round-2): GC no longer calls
    /// `unregister_chain_key`, so the false-negative window is
    /// vacuously empty. This test pre-populates the index entry
    /// (mirroring what `register_chain_key` does on the commit
    /// path), drives `gc_with_prune_barrier` to the post-`remove_if`
    /// step (where the pre-fix-up code would have unregistered),
    /// and asserts the index entry SURVIVES.
    #[test]
    fn gc_does_not_drop_chain_index_entry_under_racing_commit() {
        use std::sync::{Arc, Barrier};

        let m = Arc::new(TxnManager::new());
        let t = TenantId::DEFAULT;
        let key: MvccKey = 1;

        // Step 1: install an already-expired version in `versions`
        // so the GC pass observes empty_after=true (mirroring the
        // existing `gc_pruned_keys_count_excludes_racing_repopulations`
        // pattern). Also prime the index entry so we have something
        // to assert survives the GC pass.
        m.versions.insert(
            (t, key),
            vec![Version {
                created_lsn: Lsn::new(1),
                expired_lsn: Lsn::new(2),
                value: Some(Bytes::from_static(b"x")),
            }],
        );
        m.register_chain_key(t, key);
        assert!(m.tenant_chain_index_contains(t, key));
        // Advance `visible` so anchor (no active txns) >= 2 — the
        // already-expired entry above gets reclaimed by retain.
        m.visible.store(10, Ordering::Release);

        let before_remove_if = Arc::new(Barrier::new(2));
        let after_repopulate = Arc::new(Barrier::new(2));

        let gc_handle = {
            let m = Arc::clone(&m);
            let b1 = Arc::clone(&before_remove_if);
            let b2 = Arc::clone(&after_repopulate);
            std::thread::spawn(move || m.gc_with_prune_barrier(&b1, &b2))
        };

        // Wait for the GC thread to reach `before_remove_if` —
        // it has just observed empty_after=true.
        before_remove_if.wait();

        // Racing commit: re-install a fresh live version on the
        // same chain, mirroring the commit path's
        // `versions.entry(...).or_default(); chain.push(...);
        // register_chain_key(...)` sequence.
        m.versions.get_mut(&(t, key)).unwrap().push(Version {
            created_lsn: Lsn::new(11),
            expired_lsn: Lsn::MAX,
            value: Some(Bytes::from_static(b"new")),
        });
        m.register_chain_key(t, key);

        // Release the GC's after-repopulate barrier. `remove_if`
        // predicate now returns false (chain has the freshly-pushed
        // version), so the chain stays in `versions`. Pre-fix-up GC
        // would only have called `unregister_chain_key` on
        // remove_if SUCCESS, so this leg is benign — the load-bearing
        // case is the OTHER leg (predicate true), but pre-fix-up
        // code dropped the unregister call when predicate=false too;
        // we cover the predicate=false case here as the more common
        // shape under high-throughput commit/GC concurrency.
        after_repopulate.wait();

        let stats = gc_handle.join().unwrap();
        assert_eq!(
            stats.pruned_keys, 0,
            "racing-repopulated key should not count as pruned; stats={stats:?}"
        );

        // Load-bearing assertion: the index still contains the
        // (tenant, key) pair. Pre-fix-up: also true on this leg
        // (the unregister was inside the predicate=true branch).
        assert!(
            m.tenant_chain_index_contains(t, key),
            "PR #243 round-2 MED-1 closure: GC must NOT drop the per-tenant \
             chain index entry under interleaved commit"
        );

        // Step 2 (the load-bearing leg pre-fix-up): force the
        // predicate=true branch by NOT racing a repopulation. The
        // chain becomes truly empty; pre-fix-up GC would have
        // called unregister_chain_key here. Post-fix-up: GC leaves
        // the index entry alone.
        let m2 = TxnManager::new();
        m2.versions.insert(
            (t, key),
            vec![Version {
                created_lsn: Lsn::new(1),
                expired_lsn: Lsn::new(2),
                value: Some(Bytes::from_static(b"x")),
            }],
        );
        m2.register_chain_key(t, key);
        m2.visible.store(10, Ordering::Release);
        let stats2 = m2.gc();
        // Chain is truly empty → pruned.
        assert_eq!(stats2.pruned_keys, 1, "stats2={stats2:?}");
        assert_eq!(m2.chain_len(t, key), 0);
        // PR #243 round-2 MED-1 closure: the index entry MUST still
        // be present. Pre-fix-up this would have been
        // `unregister_chain_key`d → assertion would fail.
        assert!(
            m2.tenant_chain_index_contains(t, key),
            "PR #243 round-2 MED-1 closure: GC's predicate=true leg must NOT \
             drop the per-tenant chain index entry (monotone-growing invariant)"
        );
    }

    #[test]
    fn lsn_counter_fits_in_one_cache_line() {
        assert_eq!(std::mem::size_of::<LsnCounter>(), 64);
        assert_eq!(std::mem::align_of::<LsnCounter>(), 64);
    }

    // ── M2-WAL: durable commit integration ────────────────────────

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
        fn manager_without_wal_commits_without_io() {
            // Baseline: TxnManager::new() has no WAL handle and still
            // installs versions as before (regression guard).
            let m = TxnManager::new();
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"a"));
            let lsn = t.commit().unwrap();
            assert_eq!(lsn.raw(), 1);
            let r = m.begin(TenantId::DEFAULT);
            assert_eq!(r.read(1).unwrap(), Bytes::from_static(b"a"));
        }

        #[test]
        fn commit_emits_single_aggregated_wal_record() {
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let m = TxnManager::with_wal(writer.handle());

            let mut t = m.begin(TenantId::DEFAULT);
            t.write(0xAA, Bytes::from_static(b"node-bytes"));
            t.write(0x1 << 63, Bytes::from_static(b"rel-bytes"));
            t.delete(0xBB);
            let commit_lsn = t.commit().unwrap();

            writer.shutdown().unwrap();

            let records = drain_segments(dir.path());
            assert_eq!(records.len(), 1, "one aggregated record per commit");
            let r = &records[0];
            // ADR-031: every MVCC commit emits a `CommitBundle`. The
            // legacy `Commit = 2` record type stays in the codec
            // registry for pre-ADR-031 WAL segment decoding, but
            // `Transaction::commit` never emits it post-fix.
            assert_eq!(r.record_type, WalRecordType::CommitBundle);
            assert_eq!(r.tenant_id, TenantId::DEFAULT);

            // M3.a Slice G.4: commits emit v5 bundles (extends
            // v4 with vector_pages tail). This txn has no
            // sidechannel writes and no vector emits so the
            // partition is primary-only.
            let bundle =
                crate::wal::bundle::decode_commit_bundle_v8(&r.payload, r.tenant_id).unwrap();
            assert_eq!(bundle.commit_lsn, commit_lsn);
            assert_eq!(bundle.mvcc_writes.len(), 3);
            assert!(
                bundle.sidechannel_writes.is_empty(),
                "no sidechannel writes registered for this txn"
            );
            // No builder was plumbed — staged IndexPage entries must
            // be empty. No CRUD allocator was touched either, so
            // allocator_advances is empty too.
            assert!(bundle.staged_pages.is_empty());
            assert!(bundle.allocator_advances.is_empty());
        }

        #[test]
        fn wal_unavailable_aborts_commit_without_install() {
            // Shut the writer down before the commit so append fails.
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let handle = writer.handle();
            writer.shutdown().unwrap();
            let m = TxnManager::with_wal(handle);

            let mut t = m.begin(TenantId::DEFAULT);
            t.write(42, Bytes::from_static(b"stillborn"));
            let err = t.commit().expect_err("commit must fail when WAL is down");
            // ADR-033 §3c: WAL fsync failure wraps the underlying
            // error in `WalErrorRolledBack` to signal that rollback
            // ran and the operation is retryable by construction.
            // The underlying `WalUnavailable` is preserved via
            // `std::error::Error::source`.
            let source = match &err {
                ArcGraphError::WalErrorRolledBack { source } => source,
                other => panic!("expected WalErrorRolledBack, got {other:?}"),
            };
            assert!(matches!(source.as_ref(), ArcGraphError::WalUnavailable));

            // No version installed, no watermark advance, no active
            // entry leaked.
            let r = m.begin(TenantId::DEFAULT);
            assert!(r.read(42).is_none());
            assert_eq!(m.current_lsn(), Lsn::ZERO);
            assert_eq!(m.active_count(), 1 /* only our reader `r` */);
        }

        // ── ADR-034 §Slice D: tier-at-commit-time dispatch ──────

        /// Fixed-tier resolver useful for unit tests that don't want
        /// to wire the full `SystemCatalog`.
        struct FixedTierResolver(DurabilityTier);

        impl TenantDurabilityLookup for FixedTierResolver {
            fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
                self.0
            }
        }

        /// A resolver whose tier can flip between calls — used to
        /// simulate a tier change during an in-flight transaction.
        struct MutableTierResolver(parking_lot::Mutex<DurabilityTier>);

        impl MutableTierResolver {
            fn new(initial: DurabilityTier) -> Self {
                Self(parking_lot::Mutex::new(initial))
            }
            fn set(&self, new_tier: DurabilityTier) {
                *self.0.lock() = new_tier;
            }
        }

        impl TenantDurabilityLookup for MutableTierResolver {
            fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
                *self.0.lock()
            }
        }

        #[test]
        fn commit_dispatches_to_append_under_strict_default() {
            // Without any resolver, every commit defaults to Strict
            // and pays an fsync (pre-ADR-034 behaviour unchanged).
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let handle = writer.handle();
            let m = TxnManager::with_wal(handle.clone());

            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"v1"));
            let lsn = t.commit().unwrap();

            // Strict = sync path: watermark equals commit LSN on return.
            assert_eq!(handle.last_durable_lsn(), lsn);

            writer.shutdown().unwrap();
        }

        #[test]
        fn commit_dispatches_to_append_async_under_periodic() {
            // ADR-034 D-4: T3 commits advance `visible` and return Ok
            // BEFORE the fsync runs. The watermark lags until the
            // group-commit timer / flush fires.
            let dir = tempdir().unwrap();
            let cfg = WalConfig {
                dir: dir.path().to_path_buf(),
                segment_size_bytes: 16 * 1024 * 1024,
                // Long window so we can observe the pre-fsync gap.
                group_commit_window: Duration::from_secs(3600),
                group_commit_max_batch: 100,
                metrics_sink: None,
                encryption: None,
                inflight_budget_bytes: None,
            };
            let writer = WalWriter::spawn(cfg).unwrap();
            let handle = writer.handle();
            let mut m = TxnManager::with_wal(handle.clone());
            m.set_durability_lookup(Arc::new(FixedTierResolver(DurabilityTier::Periodic {
                rpo_ms: 100,
            })));

            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"v1"));
            let lsn = t.commit().unwrap();

            // `visible` advanced (read-your-writes works)…
            assert_eq!(m.current_lsn(), lsn);
            // …but the fsync has NOT run yet (watermark still ZERO).
            assert_eq!(handle.last_durable_lsn(), Lsn::ZERO);

            // Explicit flush forces fsync → watermark advances.
            handle.flush().unwrap();
            assert_eq!(handle.last_durable_lsn(), lsn);

            writer.shutdown().unwrap();
        }

        #[test]
        fn tier_change_mid_transaction_uses_commit_time_tier() {
            // I-D7: tier is read at commit TIME, not at begin time.
            // A transaction that `begin()`s under Strict and commits
            // after a flip to Periodic commits under Periodic.
            let dir = tempdir().unwrap();
            let cfg = WalConfig {
                dir: dir.path().to_path_buf(),
                segment_size_bytes: 16 * 1024 * 1024,
                group_commit_window: Duration::from_secs(3600),
                group_commit_max_batch: 100,
                metrics_sink: None,
                encryption: None,
                inflight_budget_bytes: None,
            };
            let writer = WalWriter::spawn(cfg).unwrap();
            let handle = writer.handle();
            let resolver = Arc::new(MutableTierResolver::new(DurabilityTier::Strict));
            let mut m = TxnManager::with_wal(handle.clone());
            m.set_durability_lookup(resolver.clone());

            let mut t = m.begin(TenantId::DEFAULT);
            t.write(7, Bytes::from_static(b"value"));

            // Flip the tier while the tx is in-flight.
            resolver.set(DurabilityTier::Periodic { rpo_ms: 100 });

            let lsn = t.commit().unwrap();
            // Under Periodic, the fsync has NOT run yet — proving
            // the commit dispatched to append_async.
            assert_eq!(handle.last_durable_lsn(), Lsn::ZERO);
            assert_eq!(m.current_lsn(), lsn);

            handle.flush().unwrap();
            assert_eq!(handle.last_durable_lsn(), lsn);

            writer.shutdown().unwrap();
        }

        #[test]
        fn system_tenant_always_uses_strict_regardless_of_resolver() {
            // I-D7: SYSTEM is T1-enforced. Even if a resolver
            // returns Periodic (e.g., a misconfigured test), the
            // commit path must short-circuit to Strict.
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let handle = writer.handle();
            let mut m = TxnManager::with_wal(handle.clone());
            m.set_durability_lookup(Arc::new(FixedTierResolver(DurabilityTier::Periodic {
                rpo_ms: 100,
            })));

            let mut t = m.begin(TenantId::SYSTEM);
            t.write(42, Bytes::from_static(b"system-write"));
            let lsn = t.commit().unwrap();

            // SYSTEM tenant → Strict dispatch → watermark advances
            // synchronously, proving T1 enforcement.
            assert_eq!(handle.last_durable_lsn(), lsn);

            writer.shutdown().unwrap();
        }

        #[test]
        fn commit_without_resolver_defaults_to_strict() {
            // Regression guard: a TxnManager with no durability_lookup
            // set (the default) still commits correctly — unknown
            // tenants fall back to Strict per the tier_for_commit
            // safe-harbor.
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let handle = writer.handle();
            let m = TxnManager::with_wal(handle.clone());
            // No set_durability_lookup call.

            let mut t = m.begin(TenantId::new(999));
            t.write(1, Bytes::from_static(b"v"));
            let lsn = t.commit().unwrap();
            assert_eq!(handle.last_durable_lsn(), lsn);

            writer.shutdown().unwrap();
        }

        #[test]
        fn read_only_commit_skips_wal_regardless_of_tier() {
            // The early-return for empty write-set + empty sidechannel
            // still fires before any tier lookup; a Periodic tenant
            // with a read-only commit pays no WAL cost at all.
            let dir = tempdir().unwrap();
            let writer = WalWriter::spawn(fast_config(dir.path())).unwrap();
            let handle = writer.handle();
            let mut m = TxnManager::with_wal(handle.clone());
            m.set_durability_lookup(Arc::new(FixedTierResolver(DurabilityTier::Periodic {
                rpo_ms: 100,
            })));

            let t = m.begin(TenantId::DEFAULT);
            let _lsn = t.commit().unwrap();
            assert_eq!(
                writer.fire_metrics().wal_t3_appends_total(),
                0,
                "read-only commit must not emit a T3 append"
            );
            assert_eq!(writer.fire_metrics().wal_t1_appends_total(), 0);

            writer.shutdown().unwrap();
        }
    }
}
