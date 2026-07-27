//! WAL replay executor (ADR-032 §3, §R1–R7).
//!
//! The executor iterates every WAL record in WAL-LSN order via an
//! existing [`WalRecoveryReader`], decodes `CommitBundle = 12`
//! payloads according to the owning segment's `format_version`
//! (ADR-031 + ADR-032 Slice 1 codec dispatch), buffers decoded
//! bundles in a [`BTreeMap`] sorted by their logical `commit_lsn`,
//! and drains the buffer in ascending `commit_lsn` order by
//! applying each bundle to MVCC chains + page stores. The buffer
//! protects against the commit-order ≠ WAL-order slack that
//! ADR-031 §R3 documents as a fundamental property of the three-
//! phase commit pipeline.
//!
//! The design is specified in ADR-032 §3 ("Replay executor
//! architecture — streaming with bounded out-of-order buffer");
//! the replay contract is §R1–R7; the memory bound is §5; the
//! torn-tail / corruption boundary is §6; and the 13 counters +
//! 4 gauges + 5 tracing events are §7.
//!
//! # Observability (ADR-032 §7; Slice 4 + PR #79 Y-3 fold-in)
//!
//! 15 counters + 4 gauges exposed as [`ReplayMetrics`] atomics
//! (snapshot via [`ReplayMetrics::snapshot`]). The `out_of_order_apply_rejected`
//! counter was added by the Y-3 review fold-in to separate the
//! legitimate Lemma I1 idempotent-skip case (good, common on
//! double-replay) from the OOO-rejection case (bad, always an executor
//! bug); `interns_recovered` was added by the P0 #776 fix to count
//! recovered label / rel-type name↔id bindings.
//!
//! 5 structured tracing events at `info!` / `warn!` / `error!`
//! severities:
//!
//! | Event | Level | When |
//! |-------|-------|------|
//! | `wal_replay_started` | `info` | at [`ReplayExecutor::run`] entry |
//! | `wal_replay_overflow_flush_fired` | `warn` | on each OVERFLOW_FLUSH trigger |
//! | `wal_replay_spill_engaged` | `info` | on each spill file write |
//! | `wal_replay_orphan_detected` | `warn` | when legacy IndexPage orphans exist post-drain |
//! | `wal_replay_completed` | `info` | at run success, with elapsed + totals |
//!
//! Corruption and format-mismatch halts emit an additional
//! `error!` line before propagating the `Err` to the caller.
//!
//! The `wal_replay_bundles_buffered` gauge (§7 tightening) is
//! the X-1 pathology detector — operators watching this gauge in
//! steady state should see a ceiling ≤ 64; sustained growth
//! towards `max_buffer_bundles` signals a slow-Phase-2 writer.
//!
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

use arcgraph_core::{ArcGraphError, Lsn, PAGE_SIZE, PageId, Result};
use tracing::{debug, error, info, info_span, warn};

use crate::extent::{EXTENT_PAGES, ExtentDataPageStore, ExtentDirectory};
use crate::idempotency::IdempotencyStore;
use crate::intern::{InternTable, decode_intern_payload};
use crate::permissions::PermissionIndex;
use crate::redo::{DeltaPageStore, DirtyPageTable, apply_recovery_delta};
use crate::transaction::{MvccKey, ReplayApplyOutcome, TxnManager};
use crate::wal::bundle::{
    AllocatorAdvance, DecodedCommitBundle, DecodedIndexPage, decode_commit_bundle_for_version,
};
use crate::wal::record::{WalRecord, WalRecordType};
use crate::wal::recovery::WalRecoveryReader;
use crate::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_core::TenantId;

// ─── Configuration ───────────────────────────────────────────────

/// Replay executor configuration knobs. Defaults match ADR-032 §5.
///
/// All four knobs may be overridden via environment variables read
/// by [`ReplayConfig::from_env`]:
///
/// - `ARCGRAPH_REPLAY_MAX_BUFFER_BUNDLES` (default 8192)
/// - `ARCGRAPH_REPLAY_MAX_BUFFER_BYTES` (default 1 GiB)
/// - `ARCGRAPH_REPLAY_SPILL` = `on` | `off` (default `on` per §5
///   X-1 tightening)
/// - `ARCGRAPH_REPLAY_SPILL_DIR` (default `${wal_dir}/replay-spill`
///   when the executor is constructed with an explicit WAL dir via
///   `from_wal_dir`; otherwise `<tempdir>/arcgraph-replay-spill-<pid>`).
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Upper bound on buffered bundles. ADR-032 §5 default = 8192.
    pub max_buffer_bundles: usize,
    /// Upper bound on buffered-bundle payload sum. ADR-032 §5
    /// default = 1 GiB.
    pub max_buffer_bytes: usize,
    /// Whether spill-to-disk is enabled (§5 X-1: default `true` /
    /// safety-by-default). Disable only for air-gapped tests via
    /// `ARCGRAPH_REPLAY_SPILL=off`.
    pub spill_enabled: bool,
    /// Directory where spill files live. See §5 ("Spill file
    /// format"). Deleted on successful replay completion.
    pub spill_dir: PathBuf,
}

impl ReplayConfig {
    /// ADR-032 §5 default values with a spill dir under
    /// `std::env::temp_dir()`.
    #[must_use]
    pub fn default_with_temp_spill() -> Self {
        let mut dir = std::env::temp_dir();
        dir.push(format!("arcgraph-replay-spill-{}", std::process::id()));
        Self {
            max_buffer_bundles: 8192,
            max_buffer_bytes: 1_073_741_824, // 1 GiB
            spill_enabled: true,
            spill_dir: dir,
        }
    }

    /// Build a config overlaying environment-variable overrides
    /// over [`Self::default_with_temp_spill`]. Missing / malformed
    /// env values fall through to defaults.
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default_with_temp_spill();
        if let Ok(v) = std::env::var("ARCGRAPH_REPLAY_MAX_BUFFER_BUNDLES") {
            if let Ok(n) = v.parse() {
                cfg.max_buffer_bundles = n;
            }
        }
        if let Ok(v) = std::env::var("ARCGRAPH_REPLAY_MAX_BUFFER_BYTES") {
            if let Ok(n) = v.parse() {
                cfg.max_buffer_bytes = n;
            }
        }
        if let Ok(v) = std::env::var("ARCGRAPH_REPLAY_SPILL") {
            // ADR-032 §5: default ON. Any recognised falsy spelling
            // turns spill OFF for air-gapped tests. Unknown values
            // leave the default in place.
            let v = v.to_ascii_lowercase();
            if matches!(v.as_str(), "off" | "false" | "0" | "no") {
                cfg.spill_enabled = false;
            } else if matches!(v.as_str(), "on" | "true" | "1" | "yes") {
                cfg.spill_enabled = true;
            }
        }
        if let Ok(v) = std::env::var("ARCGRAPH_REPLAY_SPILL_DIR") {
            cfg.spill_dir = PathBuf::from(v);
        }
        cfg
    }

    /// Convenience: scope the spill directory under a caller-
    /// supplied WAL dir (`{wal_dir}/replay-spill`). Env var
    /// `ARCGRAPH_REPLAY_SPILL_DIR` still takes precedence if set.
    #[must_use]
    pub fn with_wal_dir(wal_dir: &std::path::Path) -> Self {
        let mut cfg = Self::from_env();
        if std::env::var_os("ARCGRAPH_REPLAY_SPILL_DIR").is_none() {
            cfg.spill_dir = wal_dir.join("replay-spill");
        }
        cfg
    }
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

// ─── Metrics + tracing surface (ADR-032 §7; Slice 4) ────────────────

/// Executor phase for the `wal_replay_phase` gauge.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayPhase {
    /// Reading records from the WAL reader; applying legacy records
    /// immediately and buffering `CommitBundle` records.
    Reading = 0,
    /// WAL exhausted; merging buffer + spill and applying in
    /// commit_lsn order.
    Draining = 1,
    /// All bundles applied; counters / visible / install_order
    /// seeded. Terminal state.
    Completed = 2,
}

/// ADR-032 §7 observability surface.
///
/// 15 counters + 4 gauges + 5 tracing events. All counters /
/// gauges are `AtomicU64` so [`ReplayMetrics::snapshot`] is
/// lock-free. Tracing events are emitted at `info!` / `warn!` /
/// `error!` level via the `tracing` crate (see ADR-032 §7 and
/// code-quality policy "Logging: tracing with structured fields").
#[derive(Debug, Default)]
pub struct ReplayMetrics {
    // Counters
    /// `wal_replay_records_total` — total WAL records iterated.
    pub records_total: AtomicU64,
    /// `wal_replay_bundles_applied` — `CommitBundle` records fully
    /// applied (both MVCC + IndexPage steps succeeded).
    pub bundles_applied: AtomicU64,
    /// `wal_replay_bundles_skipped_idempotent` — bundles skipped
    /// because their `commit_lsn` was already on the chain (Lemma
    /// I1) or because they were below `applied_high_water`.
    pub bundles_skipped_idempotent: AtomicU64,
    /// `wal_replay_mvcc_versions_installed` — `Version`s pushed
    /// onto MVCC chains (primary + side-channel combined).
    pub mvcc_versions_installed: AtomicU64,
    /// `wal_replay_index_pages_applied` — `IndexPage` entries
    /// installed via `install_or_replace`.
    pub index_pages_applied: AtomicU64,
    /// `wal_replay_bundles_spilled` — bundles written to spill
    /// files during OVERFLOW_FLUSH.
    pub bundles_spilled: AtomicU64,
    /// `wal_replay_spill_files_created` — spill file count.
    pub spill_files_created: AtomicU64,
    /// `wal_replay_spill_files_reloaded` — spill files loaded from
    /// disk on replay start (from a crashed prior replay).
    pub spill_files_reloaded: AtomicU64,
    /// `wal_replay_orphan_pages_detected` — legacy `IndexPage = 11`
    /// records observed without a matching `CommitBundle`.
    pub orphan_pages_detected: AtomicU64,
    /// `wal_replay_bootstrap_from_mvcc_invoked` — `0` or `1`
    /// (sticky): whether post-replay `bootstrap_from_mvcc` ran.
    pub bootstrap_from_mvcc_invoked: AtomicU64,
    /// `wal_replay_wal_errors_total` — non-fatal WAL reader errors
    /// (currently unused; always 0 at v1.0).
    pub wal_errors_total: AtomicU64,
    /// `wal_replay_corruption_halts` — `0` or `1` (sticky):
    /// whether replay halted on `WalCorruption`.
    pub corruption_halts: AtomicU64,
    /// `wal_replay_overflow_flush_fired` — how many times
    /// OVERFLOW_FLUSH ran.
    pub overflow_flush_fired: AtomicU64,
    /// `wal_replay_out_of_order_apply_rejected` — PR #79 Y-3
    /// fold-in. Counts
    /// [`crate::transaction::ReplayApplyOutcome::OutOfOrder`]
    /// returns from `apply_replay_mvcc_write`. Always 0 on
    /// healthy replays; non-zero indicates an upstream executor
    /// bug (buffer didn't sort before apply, or a late-arriving
    /// bundle bypassed the skip path). `tracing::error!` fires on
    /// every increment.
    pub out_of_order_apply_rejected: AtomicU64,
    /// `wal_replay_blob_pages_applied` — N-2 (issue #81) fold-in.
    /// Counts staged_pages entries with
    /// [`crate::wal::bundle::BundlePageKind::Blob`] routed through
    /// [`crate::blob::BlobStoreHandle::install_or_replace`] during
    /// replay. Separate from `index_pages_applied` so operators
    /// can distinguish blob-chain reconstruction load from B-tree
    /// page reinstall load; the two scale independently.
    pub blob_pages_applied: AtomicU64,
    /// `wal_replay_allocator_advances_applied` — issue #129 P0 fix.
    /// Counts [`AllocatorAdvance`] entries applied to the live
    /// allocator counters during replay (across all bundles). On
    /// a healthy v4 replay this is non-zero whenever any commit
    /// allocated a NodeId / RelId / PageId; double-replay is
    /// idempotent so the counter accumulates linearly with the
    /// number of advance entries seen, regardless of monotonic
    /// no-op application.
    pub allocator_advances_applied: AtomicU64,
    /// `wal_replay_interns_recovered` — P0 #776. Counts
    /// [`WalRecordType::InternString`] records decoded + installed into
    /// the wired [`InternTable`] during replay. Non-zero after any
    /// durable restart whose workload created ≥1 distinct label /
    /// rel-type name; stays `0` when no intern table is wired (the
    /// arm is a no-op then). The durable bootstrap logs this so a
    /// name-recovery regression surfaces in stderr.
    pub interns_recovered: AtomicU64,
    /// `wal_replay_idempotency_bindings_recovered` — #352 Part 2
    /// (ADR-199). Counts v6 `CommitBundle` `idempotency_bindings` entries
    /// installed into the wired [`IdempotencyStore`] during replay.
    /// Non-zero after any durable restart whose workload bound ≥1
    /// `external_id`; stays `0` when no store is wired (the arm is a
    /// no-op then). This counter being ≥1 after a restart is the
    /// observable proof that idempotency survived the bounce.
    pub idempotency_bindings_recovered: AtomicU64,

    /// `wal_replay_acl_grants_recovered` — #1221 (ADR-218). Counts v8
    /// `CommitBundle` `acl_grants` entries re-driven into the wired
    /// [`PermissionIndex`] during replay. Non-zero after any durable
    /// restart whose workload applied ≥1 ACL grant/revoke; stays `0`
    /// when no index is wired (the arm is a no-op then). This counter
    /// being ≥1 after a restart is the observable proof that document
    /// ACLs survived the bounce (the #1221 fix).
    pub acl_grants_recovered: AtomicU64,

    // Gauges
    /// `wal_replay_bundles_buffered` — current depth of the sorted
    /// buffer. Per §7-tightening, this is the X-1 pathology
    /// detection signal.
    pub bundles_buffered: AtomicU64,
    /// `wal_replay_buffer_memory_bytes` — current bundle-payload
    /// bytes in the buffer.
    pub buffer_memory_bytes: AtomicU64,
    /// `wal_replay_current_commit_lsn` — high-water of
    /// successfully-applied `commit_lsn`.
    pub current_commit_lsn: AtomicU64,
    /// `wal_replay_phase` — see [`ReplayPhase`].
    pub phase: AtomicU8,
}

impl ReplayMetrics {
    /// Snapshot the counter/gauge state into a plain struct,
    /// suitable for test assertions.
    #[must_use]
    pub fn snapshot(&self) -> ReplayMetricsSnapshot {
        ReplayMetricsSnapshot {
            records_total: self.records_total.load(Ordering::Relaxed),
            bundles_applied: self.bundles_applied.load(Ordering::Relaxed),
            bundles_skipped_idempotent: self.bundles_skipped_idempotent.load(Ordering::Relaxed),
            mvcc_versions_installed: self.mvcc_versions_installed.load(Ordering::Relaxed),
            index_pages_applied: self.index_pages_applied.load(Ordering::Relaxed),
            bundles_spilled: self.bundles_spilled.load(Ordering::Relaxed),
            spill_files_created: self.spill_files_created.load(Ordering::Relaxed),
            spill_files_reloaded: self.spill_files_reloaded.load(Ordering::Relaxed),
            orphan_pages_detected: self.orphan_pages_detected.load(Ordering::Relaxed),
            bootstrap_from_mvcc_invoked: self.bootstrap_from_mvcc_invoked.load(Ordering::Relaxed),
            wal_errors_total: self.wal_errors_total.load(Ordering::Relaxed),
            corruption_halts: self.corruption_halts.load(Ordering::Relaxed),
            overflow_flush_fired: self.overflow_flush_fired.load(Ordering::Relaxed),
            out_of_order_apply_rejected: self.out_of_order_apply_rejected.load(Ordering::Relaxed),
            blob_pages_applied: self.blob_pages_applied.load(Ordering::Relaxed),
            allocator_advances_applied: self.allocator_advances_applied.load(Ordering::Relaxed),
            interns_recovered: self.interns_recovered.load(Ordering::Relaxed),
            idempotency_bindings_recovered: self
                .idempotency_bindings_recovered
                .load(Ordering::Relaxed),
            acl_grants_recovered: self.acl_grants_recovered.load(Ordering::Relaxed),
            bundles_buffered: self.bundles_buffered.load(Ordering::Relaxed),
            buffer_memory_bytes: self.buffer_memory_bytes.load(Ordering::Relaxed),
            current_commit_lsn: self.current_commit_lsn.load(Ordering::Relaxed),
            phase: self.phase.load(Ordering::Relaxed),
        }
    }

    fn set_phase(&self, phase: ReplayPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }
}

/// Plain-struct snapshot of [`ReplayMetrics`] for test assertions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayMetricsSnapshot {
    pub records_total: u64,
    pub bundles_applied: u64,
    pub bundles_skipped_idempotent: u64,
    pub mvcc_versions_installed: u64,
    pub index_pages_applied: u64,
    pub bundles_spilled: u64,
    pub spill_files_created: u64,
    pub spill_files_reloaded: u64,
    pub orphan_pages_detected: u64,
    pub bootstrap_from_mvcc_invoked: u64,
    pub wal_errors_total: u64,
    pub corruption_halts: u64,
    pub overflow_flush_fired: u64,
    pub out_of_order_apply_rejected: u64,
    pub blob_pages_applied: u64,
    pub allocator_advances_applied: u64,
    pub interns_recovered: u64,
    pub idempotency_bindings_recovered: u64,
    pub acl_grants_recovered: u64,
    pub bundles_buffered: u64,
    pub buffer_memory_bytes: u64,
    pub current_commit_lsn: u64,
    pub phase: u8,
}

// ─── Page-store routing trait (ADR-032 §R2 Step 3c) ──────────────

/// Routing target for a decoded `IndexPage` entry during replay.
///
/// ADR-032 §R2 Step 3c applies each `IndexPage` via
/// `install_or_replace(page_id, bytes)` on the owning store. At
/// v1.0 the runtime has two page stores that accept index bundle
/// pages:
///
/// - [`PrimaryPageStore`](crate::primary_index::PrimaryPageStore)
///   for the primary B-tree's internal / leaf / overflow pages.
/// - `SecondaryPageStore` (in `arcgraph-index`) for the secondary
///   B-tree — also internal / leaf / overflow pages.
///
/// Both stores use disjoint `PageId` allocators (§`PageAllocator`
/// in `page_alloc.rs`), so a bundle's page_id is sufficient to
/// route once the executor knows which allocator produced it. At
/// v1.0 the replay executor operates under the invariant that all
/// `IndexPage` entries in a `CommitBundle` belong to the same
/// owner (§ADR-031 CommitBundle semantics); callers that drive
/// secondary-index replay provide a [`PageStoreTarget`] that also
/// routes into the secondary store via the trait object
/// [`SecondaryPageStoreHandle`] below.
///
/// If your deployment only has a primary index, pass `None` for
/// `secondary` on [`PageStoreTarget::new`]; the executor emits a
/// [`tracing::warn!`] on any primary-unrecognised page_id (§O3
/// orphan path) and, if the page type discriminator in the page
/// bytes indicates a secondary page, halts with
/// [`ArcGraphError::WalCorruption`] (no handle registered).
pub trait SecondaryPageStoreHandle: Send + Sync {
    /// Idempotent install: overwrite if present, install if not
    /// (Lemma I2). The bytes are guaranteed byte-identical on
    /// successive calls because the staged emit was captured
    /// under the owning index's `write_gate` + per-page latch.
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> Result<()>;

    /// Whether the store already holds this `page_id`. Used by the
    /// executor for Lemma I2 byte-integrity verification.
    fn contains(&self, page_id: PageId) -> bool;
}

/// Primary page store handle mirroring [`SecondaryPageStoreHandle`].
///
/// This trait is generic over the concrete page-store type so the
/// executor can be unit-tested without pulling in the real
/// `PrimaryPageStore` + `PageAllocator` graph.
pub trait PrimaryPageStoreHandle: Send + Sync {
    /// Idempotent install: overwrite if present, install if not
    /// (Lemma I2). See [`SecondaryPageStoreHandle::install_or_replace`].
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> Result<()>;

    /// Whether the store already holds this `page_id`. Used by the
    /// executor for Lemma I2 byte-integrity verification.
    fn contains(&self, page_id: PageId) -> bool;
}

/// Handle for `RecordPageStore` during replay (X-2 review fold-in).
///
/// Slotted record pages (Node + Rel records) travel in the v3
/// `CommitBundle` under `PageStoreKind::RecordPage`; the replay
/// executor routes them through this handle.
pub trait RecordPageStoreHandle: Send + Sync {
    /// Idempotent install: overwrite if present, install if not
    /// (Lemma I2 — bundle-level idempotence; later bundles
    /// legitimately supersede earlier ones for the same page_id).
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> Result<()>;
}

/// N-2 (issue #81) re-export alias: the blob store's replay handle
/// lives in `arcgraph_storage::blob` (the trait cannot be defined
/// here without introducing a circular `wal` → `blob` dep), but
/// callers that wire the replay target read it from this module.
///
/// Behaviour and contract are defined on
/// [`crate::blob::BlobStoreHandle`].
pub use crate::blob::BlobStoreHandle;

/// M3.a Slice G.1 re-export alias mirroring [`BlobStoreHandle`]:
/// the vector arena store's replay handle lives in
/// [`crate::vector_store`] so the trait can be implemented by
/// `arcgraph-vector` without that crate taking a circular dep on
/// `arcgraph-storage::wal`. Callers that wire the replay target
/// read it from this module for symmetry with `BlobStoreHandle`.
///
/// Behaviour and contract are defined on
/// [`crate::vector_store::VectorPageStoreHandle`].
pub use crate::vector_store::VectorPageStoreHandle;

/// Issue #129 P0 fix: dispatch handle for replaying
/// [`AllocatorAdvance`] entries from a v4 `CommitBundle`.
///
/// Implementations seed the live allocator counters such that, for
/// every `(tenant, kind)` pair, the counter is at least
/// `advance.new_high_water + 1`. The seed is monotonic-max
/// (Lemma I3 — applying the same advance twice or applying an
/// older advance after a newer one is a no-op).
///
/// The CRUD-layer wiring impl (`CrudAllocatorSeedHandle`)
/// dispatches `AllocatorKind::Node` / `Rel` into
/// [`crate::crud::CrudStore`] and `Page*` variants into
/// [`crate::page_alloc::PageAllocator`].
pub trait AllocatorSeedHandle: Send + Sync {
    /// Seed the allocator named by `advance.kind` for tenant
    /// `advance.tenant` so its next allocation returns at least
    /// `advance.new_high_water + 1`. Must be idempotent under
    /// double-replay.
    fn seed_from_advance(&self, advance: AllocatorAdvance);
}

/// Apply-target for a `CommitBundle`'s staged_pages section. Holds
/// the primary + optional secondary + optional record page-store
/// handles plus a callback for post-replay orphan recovery
/// (§Slice 3c).
///
/// The executor borrows this target for the duration of `run`.
pub struct PageStoreTarget {
    primary: Arc<dyn PrimaryPageStoreHandle>,
    secondary: Option<Arc<dyn SecondaryPageStoreHandle>>,
    /// Record page store handle. When `Some`, v3 bundles with
    /// `PageStoreKind::RecordPage` entries route here. `None`
    /// preserves the v2 (index-only) executor shape.
    record_store: Option<Arc<dyn RecordPageStoreHandle>>,
    /// Blob store handle (N-2 / issue #81). When `Some`, v3 bundle
    /// entries with `BundlePageKind::Blob` route here, closing the
    /// post-replay `BlobStoreError::MissingHead` gap that PR #79's
    /// X-2 left open. `None` preserves the pre-N-2 behaviour of
    /// rejecting any Blob entry as a wiring bug.
    blob_store: Option<Arc<dyn BlobStoreHandle>>,
    /// Vector arena store handle (M3.a Slice G.1, ADR-035 §7.5).
    /// When `Some`, v3 bundle entries with
    /// `BundlePageKind::Vector` route here. When `None`, the
    /// dispatch arm logs a `tracing::warn!` and continues — this
    /// preserves replay forward-progress for pre-M3.a deployments
    /// whose WALs cannot contain Vector entries by construction
    /// (the staging side, Slice G.4, lives downstream of this
    /// stub). The stricter "no handle ⇒ reject" pattern that
    /// `Blob` and `Record` use will adopt at the same time the
    /// Slice G.2 implementor lands and Vector entries become
    /// reachable in production WALs.
    vector_store: Option<Arc<dyn VectorPageStoreHandle>>,
    /// Issue #129 P0 fix: optional allocator seed handle. When
    /// `Some`, v4 `CommitBundle` `allocator_advances` entries
    /// are routed here in commit_lsn order. `None` preserves
    /// pre-fix forward-progress (replay completes; allocator
    /// state stays at zero — equivalent to pre-#129 behaviour and
    /// a known data-loss vector for T1 strict-tier commits).
    /// Callers that recover production state MUST wire this.
    allocator_seed: Option<Arc<dyn AllocatorSeedHandle>>,
    /// P0 #776: the served intern table. When `Some`, each
    /// [`WalRecordType::InternString`] record is decoded and installed
    /// (via [`InternTable::intern_install`]) so the label / rel-type
    /// name↔id mapping survives a durable restart. When `None`, the
    /// `InternString` arm stays a no-op — preserving the prior behaviour
    /// for unit tests + callers that don't recover a name table.
    /// Production durable bootstrap MUST wire this (the SAME `Arc` the
    /// served `StorageBackend` holds) so recovered names reach
    /// `graph.schema` + the query binder.
    intern_table: Option<Arc<InternTable>>,
    /// #352 Part 2 (ADR-199): the served idempotency store. When `Some`,
    /// each v6 `CommitBundle`'s `idempotency_bindings` entries are
    /// installed (via [`IdempotencyStore::install`]) so the
    /// `external_id → internal_id` mapping survives a durable restart.
    /// When `None`, the apply arm is a no-op — preserving prior behaviour
    /// for replay-shape unit tests + callers that don't recover the map.
    /// Production durable bootstrap MUST wire this (the SAME `Arc` the
    /// served `StorageBackend` holds) so a post-restart re-ingest resolves
    /// idempotently instead of minting a duplicate (the #352 bug).
    idempotency_store: Option<Arc<IdempotencyStore>>,
    /// #1221 (ADR-218): the served per-tenant [`PermissionIndex`]. When
    /// `Some`, each v8 `CommitBundle`'s `acl_grants` entries re-drive
    /// `apply_doc_acl_replayed` (Apply) / `revoke_doc_replayed` (Revoke)
    /// against this index in ascending `commit_lsn` order — so document
    /// ACLs survive a bare `serve --data` restart instead of coming up
    /// deny-all (the #1221 defect). The replay entry points bypass the
    /// WAL sink (the op is already durable). When `None`, the apply arm
    /// is a no-op — preserving prior behaviour for replay-shape unit
    /// tests + callers that don't recover ACLs. Production durable
    /// bootstrap MUST wire this (the SAME `Arc` the served router's
    /// `TenantHandle::permissions()` returns) so enforcement is intact
    /// before serving. At v1.0 there is one user tenant (DEFAULT); a
    /// mismatched tenant entry is skipped (cross-tenant isolation,
    /// ADR-212 §5 Q3).
    permission_index: Option<Arc<PermissionIndex>>,
    /// M3 physical redo stores. Both must be wired together; store 5 is
    /// intentionally absent because it remains page-image at M3.
    delta_props: Option<Arc<dyn DeltaPageStore>>,
    delta_records: Option<Arc<dyn DeltaPageStore>>,
    delta_dpt: Option<Arc<DirtyPageTable>>,
    /// Bootstrap-addressable durable directories. This registry scales with
    /// tenant/store owners, never with extents; entries themselves remain
    /// page-backed in each directory's bounded buffer pool.
    extent_directories: BTreeMap<(TenantId, u16), Arc<ExtentDirectory>>,
    /// Production data stores paired with [`Self::extent_directories`].
    /// Legacy M3 pages remain on `delta_props` / `delta_records`; a physical
    /// op switches to this registry only after its logical extent has a
    /// durable directory mapping.
    extent_data_stores: BTreeMap<(TenantId, u16), Arc<ExtentDataPageStore>>,
    /// Post-replay bootstrap hook. Invoked with the number of
    /// orphan `IndexPage` records observed when the executor
    /// classifies the WAL as having orphans (§O3). Returning `Err`
    /// escalates to [`ArcGraphError::UnrecoverableOrphans`].
    bootstrap_from_mvcc: Option<Box<dyn Fn(u64) -> Result<()> + Send + Sync>>,
}

impl PageStoreTarget {
    /// Construct a target routing only to the primary page store.
    #[must_use]
    pub fn primary_only(primary: Arc<dyn PrimaryPageStoreHandle>) -> Self {
        Self {
            primary,
            secondary: None,
            record_store: None,
            blob_store: None,
            vector_store: None,
            allocator_seed: None,
            intern_table: None,
            idempotency_store: None,
            permission_index: None,
            delta_props: None,
            delta_records: None,
            delta_dpt: None,
            extent_directories: BTreeMap::new(),
            extent_data_stores: BTreeMap::new(),
            bootstrap_from_mvcc: None,
        }
    }

    /// Construct a target routing to both primary and secondary
    /// page stores.
    #[must_use]
    pub fn new(
        primary: Arc<dyn PrimaryPageStoreHandle>,
        secondary: Arc<dyn SecondaryPageStoreHandle>,
    ) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            record_store: None,
            blob_store: None,
            vector_store: None,
            allocator_seed: None,
            intern_table: None,
            idempotency_store: None,
            permission_index: None,
            delta_props: None,
            delta_records: None,
            delta_dpt: None,
            extent_directories: BTreeMap::new(),
            extent_data_stores: BTreeMap::new(),
            bootstrap_from_mvcc: None,
        }
    }

    /// Attach a post-replay orphan recovery hook (§Slice 3c). The
    /// executor calls this when it has observed one or more legacy
    /// `IndexPage = 11` records without a matching `CommitBundle`.
    /// `Ok(())` = orphans handled; `Err(_)` escalates to
    /// [`ArcGraphError::UnrecoverableOrphans`].
    #[must_use]
    pub fn with_bootstrap<F>(mut self, hook: F) -> Self
    where
        F: Fn(u64) -> Result<()> + Send + Sync + 'static,
    {
        self.bootstrap_from_mvcc = Some(Box::new(hook));
        self
    }

    /// Attach a record page store handle (X-2 review fold-in). The
    /// executor routes v3 bundle entries with
    /// `PageStoreKind::RecordPage` into this handle. Without it,
    /// such entries are rejected as a wiring bug.
    #[must_use]
    pub fn with_record_store(mut self, records: Arc<dyn RecordPageStoreHandle>) -> Self {
        self.record_store = Some(records);
        self
    }

    /// Attach the only two page-LSN delta stores and the recovery DPT.
    #[must_use]
    pub fn with_delta_stores(
        mut self,
        props: Arc<dyn DeltaPageStore>,
        records: Arc<dyn DeltaPageStore>,
        dpt: Arc<DirtyPageTable>,
    ) -> Self {
        self.delta_props = Some(props);
        self.delta_records = Some(records);
        self.delta_dpt = Some(dpt);
        self
    }

    /// Register one durable extent directory for ordinary v9 redo.
    #[must_use]
    pub fn with_extent_directory(mut self, directory: Arc<ExtentDirectory>) -> Self {
        self.extent_directories
            .insert((directory.tenant(), directory.store_id()), directory);
        self
    }

    /// Register the production data-page store belonging to one durable
    /// extent directory. Replay selects it only once `ExtentAlloc` has made
    /// the logical extent addressable; unmapped legacy M3 pages keep their
    /// existing stores.
    #[must_use]
    pub fn with_extent_data_store(mut self, store: Arc<ExtentDataPageStore>) -> Self {
        self.extent_data_stores.insert(
            (store.directory().tenant(), store.directory().store_id()),
            store,
        );
        self
    }

    fn apply_extent_alloc(&self, op: &crate::wal::DeltaOp) -> Result<()> {
        let Some(dpt) = self.delta_dpt.as_ref() else {
            return Err(ArcGraphError::WalCorruption {
                lsn: op.op_lsn,
                reason: "ExtentAlloc redo encountered without a recovery DPT".to_owned(),
            });
        };
        let directory = self
            .extent_directories
            .get(&(op.tenant_id, op.store_id))
            .ok_or_else(|| ArcGraphError::WalCorruption {
                lsn: op.op_lsn,
                reason: format!(
                    "ExtentAlloc redo has no directory for tenant {:?} store {}",
                    op.tenant_id, op.store_id
                ),
            })?;
        directory.apply_extent_alloc(op, dpt.as_ref())?;
        Ok(())
    }

    fn apply_physical_delta(&self, op: &crate::wal::DeltaOp, commit_lsn: Lsn) -> Result<()> {
        let Some(dpt) = self.delta_dpt.as_ref() else {
            return Err(ArcGraphError::WalCorruption {
                lsn: op.op_lsn,
                reason: "v9 physical delta encountered without DPT recovery wiring".to_owned(),
            });
        };
        let mapped_extent = match self.extent_data_stores.get(&(op.tenant_id, op.store_id)) {
            Some(store)
                if store
                    .directory()
                    .mapping(op.page_no / EXTENT_PAGES)?
                    .is_some() =>
            {
                Some(store)
            }
            _ => None,
        };
        let (Some(props), Some(records)) = (self.delta_props.as_ref(), self.delta_records.as_ref())
        else {
            return Err(ArcGraphError::WalCorruption {
                lsn: op.op_lsn,
                reason: "v9 physical delta encountered without props/record recovery wiring"
                    .to_owned(),
            });
        };

        if op.store_id == crate::wal::STORE_PROPS {
            // M4's served property read path still dereferences BlobStore.
            // Keep that owner replay-complete even when the migrated logical
            // page already has an extent mapping. Also update the mapped
            // extent cache so affinity-created property pages retain their
            // M4 home; the shared DPT coalesces the identical dirty key.
            apply_recovery_delta(
                props.as_ref(),
                records.as_ref(),
                dpt.as_ref(),
                op,
                commit_lsn,
            )?;
            if let Some(store) = mapped_extent {
                apply_recovery_delta(store.as_ref(), store.as_ref(), dpt.as_ref(), op, commit_lsn)?;
            }
            return Ok(());
        }
        if let Some(store) = mapped_extent {
            return apply_recovery_delta(
                store.as_ref(),
                store.as_ref(),
                dpt.as_ref(),
                op,
                commit_lsn,
            )
            .map(|_| ());
        }
        apply_recovery_delta(
            props.as_ref(),
            records.as_ref(),
            dpt.as_ref(),
            op,
            commit_lsn,
        )
        .map(|_| ())
    }

    /// Attach a blob store handle (N-2 / issue #81). The executor
    /// routes v3 bundle entries with `BundlePageKind::Blob` into
    /// this handle. Without it, such entries are rejected as a
    /// wiring bug. Matches [`Self::with_record_store`] for the
    /// post-PR-#79 "CommitBundle is the complete commit snapshot"
    /// invariant.
    #[must_use]
    pub fn with_blob_store(mut self, blob: Arc<dyn BlobStoreHandle>) -> Self {
        self.blob_store = Some(blob);
        self
    }

    /// Attach a vector arena store handle (M3.a Slice G.1, ADR-035
    /// §7.5). The executor routes v3 bundle entries with
    /// `BundlePageKind::Vector` into this handle. Mirrors
    /// [`Self::with_blob_store`] in carrying tenant on the per-
    /// entry call (vector arenas are physically per-tenant just
    /// like blobs — see ADR-035 §7.5).
    ///
    /// Without it, the dispatch arm `tracing::warn!`s and continues
    /// — a deliberately permissive pre-M3.a posture. Slice G.4
    /// closes the staging side; once vector entries are reachable
    /// in production WALs, the dispatch will tighten to "no handle
    /// ⇒ reject" symmetrically with `with_blob_store` /
    /// `with_record_store`.
    #[must_use]
    pub fn with_vector_store(mut self, vector: Arc<dyn VectorPageStoreHandle>) -> Self {
        self.vector_store = Some(vector);
        self
    }

    /// Issue #129 P0 fix: attach an [`AllocatorSeedHandle`]. The
    /// executor routes every v4 `CommitBundle` `allocator_advances`
    /// entry through this handle in commit_lsn order so post-replay
    /// `alloc_node` / `alloc_rel` / fresh-page allocations cannot
    /// reuse an id a pre-fault commit consumed.
    ///
    /// Without it, replay still forward-progresses (advances are
    /// silently dropped — equivalent to pre-fix behaviour). This is
    /// the right posture for unit tests that exercise replay shape
    /// without a CRUD stack; production callers MUST wire it. The
    /// `recover_from_wal` helper attaches the CRUD-layer wiring
    /// when invoked via `recover_stack`.
    #[must_use]
    pub fn with_allocator_seed(mut self, seed: Arc<dyn AllocatorSeedHandle>) -> Self {
        self.allocator_seed = Some(seed);
        self
    }

    /// P0 #776: attach the served [`InternTable`] so WAL replay
    /// reconstructs the label / rel-type name↔id mapping. The executor
    /// decodes every [`WalRecordType::InternString`] record and installs
    /// it via [`InternTable::intern_install`]. Pass the SAME `Arc` the
    /// served `StorageBackend` holds so recovered names reach
    /// `graph.schema` + the query binder.
    ///
    /// Without it, the `InternString` arm stays a no-op (the pre-fix
    /// behaviour) — correct for replay-shape unit tests and callers that
    /// don't recover a name table; a data-loss vector for durable serve,
    /// which the bootstrap closes by wiring this.
    #[must_use]
    pub fn with_intern_table(mut self, intern_table: Arc<InternTable>) -> Self {
        self.intern_table = Some(intern_table);
        self
    }

    /// Borrow the wired intern table, if any (P0 #776). The executor's
    /// `InternString` arm consults this; `None` keeps the arm a no-op.
    fn intern_table(&self) -> Option<&Arc<InternTable>> {
        self.intern_table.as_ref()
    }

    /// #352 Part 2 (ADR-199): attach the served [`IdempotencyStore`] so
    /// WAL replay reconstructs the `external_id → internal_id` map. The
    /// executor installs every v6 `CommitBundle`'s `idempotency_bindings`
    /// entry via [`IdempotencyStore::install`]. Pass the SAME `Arc` the
    /// served `StorageBackend` holds so a post-restart re-ingest resolves
    /// idempotently instead of minting a duplicate.
    ///
    /// Without it, the apply arm stays a no-op — correct for replay-shape
    /// unit tests and callers that don't recover the map; a correctness
    /// gap for durable serve, which the bootstrap closes by wiring this.
    #[must_use]
    pub fn with_idempotency_store(mut self, idempotency_store: Arc<IdempotencyStore>) -> Self {
        self.idempotency_store = Some(idempotency_store);
        self
    }

    /// Borrow the wired idempotency store, if any (#352 Part 2). The
    /// executor's `idempotency_bindings` apply arm consults this; `None`
    /// keeps the arm a no-op.
    fn idempotency_store(&self) -> Option<&Arc<IdempotencyStore>> {
        self.idempotency_store.as_ref()
    }

    /// #1221 (ADR-218): attach the served [`PermissionIndex`] so WAL
    /// replay re-drives document ACL grant/revoke ops. The executor
    /// re-drives every v8 `CommitBundle`'s `acl_grants` entry via
    /// `apply_doc_acl_replayed` / `revoke_doc_replayed` (which bypass the
    /// WAL sink — the op is already durable) in ascending `commit_lsn`
    /// order. Pass the SAME `Arc` the served router's
    /// `TenantHandle::permissions()` returns so a post-restart
    /// principal-scoped `graph.search` enforces against the recovered
    /// grants instead of denying all (the #1221 defect).
    ///
    /// Without it, the apply arm stays a no-op — correct for replay-shape
    /// unit tests; a deny-all enforcement gap for durable serve, which the
    /// bootstrap closes by wiring this.
    #[must_use]
    pub fn with_permission_index(mut self, permission_index: Arc<PermissionIndex>) -> Self {
        self.permission_index = Some(permission_index);
        self
    }

    /// Borrow the wired permission index, if any (#1221). The executor's
    /// `acl_grants` apply arm consults this; `None` keeps the arm a no-op.
    fn permission_index(&self) -> Option<&Arc<PermissionIndex>> {
        self.permission_index.as_ref()
    }

    /// Apply one [`AllocatorAdvance`] entry through the registered
    /// seed handle, if any. No-op when no handle is wired.
    fn apply_allocator_advance(&self, advance: AllocatorAdvance) {
        if let Some(seed) = &self.allocator_seed {
            seed.seed_from_advance(advance);
        }
    }

    /// Install an `IndexPage` entry into the correct store.
    ///
    /// Routing strategy (v1.0): try the primary first. If it
    /// Install a staged page from a v3 CommitBundle. Dispatches by
    /// [`BundlePageKind`]:
    ///
    /// - `PrimaryIndex` → primary page store.
    /// - `SecondaryIndex` → secondary page store (error if no
    ///   handle registered).
    /// - `Record` → record page store (error if no handle
    ///   registered).
    /// - `Blob` → blob store (error if no handle registered).
    ///   N-2 (issue #81) wires this leg; pre-N-2 this was the
    ///   "not yet wired" stub that halted replay for any
    ///   `PropertyData::Blob` workload.
    /// - `Vector` → vector arena store (M3.a Slice G.1, ADR-035
    ///   §7.5). Slice G.1 lands a permissive "warn-and-continue"
    ///   posture when no handle is registered; G.2 wires real
    ///   persistence and tightens to reject-as-wiring-bug.
    ///
    /// v1/v2 bundles synthesize `kind = PrimaryIndex` (see
    /// `decode_commit_bundle_v1`/`_v2`) so legacy WALs route
    /// through the same path.
    ///
    /// Returns the routed [`BundlePageKind`] so the executor can
    /// bump the right counter. Factors the dispatch out so the
    /// caller owns metric bookkeeping.
    fn install_index_page(
        &self,
        entry: &DecodedIndexPage,
    ) -> Result<crate::wal::bundle::BundlePageKind> {
        use crate::wal::bundle::BundlePageKind;
        // Clone bytes so the trait object can take ownership per
        // `install_or_replace(page, Box<[u8; PAGE_SIZE]>)`.
        let bytes_for_store = entry.bytes.clone();
        match entry.kind {
            BundlePageKind::PrimaryIndex => {
                self.primary
                    .install_or_replace(entry.page_id, bytes_for_store)?;
                Ok(BundlePageKind::PrimaryIndex)
            }
            BundlePageKind::SecondaryIndex => match &self.secondary {
                Some(sec) => {
                    sec.install_or_replace(entry.page_id, bytes_for_store)?;
                    Ok(BundlePageKind::SecondaryIndex)
                }
                None => Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!(
                        "replay: staged_page kind = SecondaryIndex (page {:?}) but no \
                         SecondaryPageStoreHandle registered on PageStoreTarget",
                        entry.page_id
                    ),
                }),
            },
            BundlePageKind::Record => match &self.record_store {
                Some(rec) => {
                    rec.install_or_replace(entry.page_id, bytes_for_store)?;
                    Ok(BundlePageKind::Record)
                }
                None => Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!(
                        "replay: staged_page kind = Record (page {:?}) but no \
                         RecordPageStoreHandle registered on PageStoreTarget",
                        entry.page_id
                    ),
                }),
            },
            BundlePageKind::Blob => match &self.blob_store {
                Some(blob) => {
                    // Blob handle is multi-tenant at the physical
                    // layer; route the per-entry tenant_id along
                    // with the page_id + bytes. See
                    // `crate::blob::BlobStoreHandle`.
                    blob.install_or_replace(entry.tenant_id, entry.page_id, bytes_for_store)?;
                    Ok(BundlePageKind::Blob)
                }
                None => Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!(
                        "replay: staged_page kind = Blob (page {:?}) but no \
                         BlobStoreHandle registered on PageStoreTarget",
                        entry.page_id
                    ),
                }),
            },
            BundlePageKind::PropSlotted => match &self.blob_store {
                Some(blob) => {
                    // v2 M1 (ADR-230): shared slotted property-bag heap
                    // pages route to the SAME kind-aware blob handle as
                    // `Blob` — `install_or_replace` classifies the page
                    // bytes (PropSlotted header vs chain-chunk header)
                    // and installs the matching resident entry. Multi-
                    // tenant at the physical layer exactly like Blob.
                    blob.install_or_replace(entry.tenant_id, entry.page_id, bytes_for_store)?;
                    Ok(BundlePageKind::PropSlotted)
                }
                None => Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!(
                        "replay: staged_page kind = PropSlotted (page {:?}) but no \
                         BlobStoreHandle registered on PageStoreTarget",
                        entry.page_id
                    ),
                }),
            },
            BundlePageKind::Vector => match &self.vector_store {
                Some(vector) => {
                    // Vector handle is multi-tenant at the physical
                    // layer (ADR-035 §7.5); route the per-entry
                    // tenant_id along with the page_id + bytes,
                    // mirroring the Blob arm above.
                    vector
                        .install_or_replace(
                            entry.tenant_id,
                            entry.page_id,
                            bytes_for_store.as_ref(),
                        )
                        .map_err(|e| ArcGraphError::WalCorruption {
                            lsn: Lsn::ZERO,
                            reason: format!(
                                "VectorPageStore install_or_replace failed for \
                                 (tenant={:?}, page={:?}): {e}",
                                entry.tenant_id, entry.page_id
                            ),
                        })?;
                    Ok(BundlePageKind::Vector)
                }
                None => {
                    // M3.a Slice G.1 stub posture: warn-and-continue.
                    // Pre-M3.a WALs cannot contain Vector entries by
                    // construction (G.4 owns staging), so this branch
                    // is a "newer-replay-on-older-deployment" defence
                    // that keeps replay forward-progress instead of
                    // halting on an unwired handle. Tightens to
                    // reject-as-wiring-bug once Slice G.2 lands and
                    // production WALs can carry Vector entries.
                    tracing::warn!(
                        page_id = ?entry.page_id,
                        tenant_id = ?entry.tenant_id,
                        "Vector page in bundle but no VectorPageStoreHandle wired \
                         (M3.a Slice G.1 stub posture; see ADR-035 §7.5)"
                    );
                    Ok(BundlePageKind::Vector)
                }
            },
        }
    }

    /// Legacy v1/v2 routing fallback used by tests + pre-amendment
    /// code paths that don't set `entry.kind` explicitly. Kept
    /// private so new code goes through `install_index_page`.
    #[allow(dead_code)]
    fn install_primary_page_legacy(&self, entry: &DecodedIndexPage) -> Result<()> {
        let bytes_for_store = entry.bytes.clone();
        if let Some(secondary) = &self.secondary {
            if secondary.contains(entry.page_id) {
                return secondary.install_or_replace(entry.page_id, bytes_for_store);
            }
            if self.primary.contains(entry.page_id) {
                return self
                    .primary
                    .install_or_replace(entry.page_id, bytes_for_store);
            }
            self.primary
                .install_or_replace(entry.page_id, bytes_for_store)
        } else {
            self.primary
                .install_or_replace(entry.page_id, bytes_for_store)
        }
    }
}

// ─── The executor ────────────────────────────────────────────────

/// One in-flight bundle sitting in the executor's sorted buffer.
#[derive(Debug)]
struct BufferedBundle {
    bundle: DecodedCommitBundle,
    /// Rough payload bytes — the MVCC write-set sum plus
    /// `PAGE_SIZE * n_index_pages`. Drives the byte-budget knob.
    bytes_budget: usize,
}

/// The replay executor.
///
/// Construct with [`ReplayExecutor::new`]; feed a reader via
/// [`ReplayExecutor::run`]. Metrics live on the executor and are
/// visible via [`ReplayExecutor::metrics`]. On successful drain,
/// seeds the [`TxnManager`]'s counter / visible / install_order via
/// [`TxnManager::seed_after_replay`] per §R2 Step 3d.
pub struct ReplayExecutor {
    // Core state.
    cfg: ReplayConfig,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    /// Sorted by `(redo_range.base, redo_range.end, arrival_seq)`. The final
    /// component preserves exact duplicate frames so idempotence is observed
    /// instead of silently deduping them in the map.
    buffer: BTreeMap<(u64, u64, u64), BufferedBundle>,
    arrival_seq: u64,
    /// Monotone high-water of `commit_lsn` successfully applied.
    applied_high_water: Lsn,
    /// Tracks orphan `IndexPage = 11` records for §Slice 3c
    /// escalation. An `IndexPage` is "orphan" iff no subsequent
    /// `CommitBundle` with a matching `page_id` was observed.
    /// v1.0 heuristic: every legacy standalone IndexPage is
    /// tentatively an orphan; if a later bundle installs the same
    /// page id via `install_or_replace`, the orphan effect is
    /// benign (bundle supersedes). For M2.e we log the count and
    /// leave bootstrap invocation to the caller's hook.
    orphan_pages: Vec<PageId>,
    metrics: Arc<ReplayMetrics>,
    /// SVC-1 / #849 / ADR-229 — checkpoint-anchored recovery floor. When
    /// `> Lsn::ZERO`, `run` raises `applied_high_water` to
    /// `max(txn_mgr.current_lsn(), checkpoint_floor)` at start, so every
    /// bundle with `commit_lsn <= checkpoint_floor` is skipped
    /// (its effects are already durable in the restored checkpoint
    /// snapshot) and ONLY records with `commit_lsn > checkpoint_floor`
    /// are applied. This is THE bound that makes 10M-restart recovery
    /// `O(WAL-since-checkpoint)`. `Lsn::ZERO` (the default) = replay from
    /// the very beginning (pre-ADR-229 / no-checkpoint / back-compat).
    checkpoint_floor: Lsn,
    /// M3 ARIES redo anchor. When set, the checkpoint's logical owners are
    /// already restored at `checkpoint_floor`, but physical deltas at or
    /// above this recLSN must still replay through page-LSN idempotence.
    incremental_redo_floor: Option<Lsn>,
}

impl ReplayExecutor {
    /// Construct a fresh executor.
    ///
    #[must_use]
    pub fn new(cfg: ReplayConfig, txn_mgr: Arc<TxnManager>, target: PageStoreTarget) -> Self {
        Self {
            cfg,
            txn_mgr,
            target,
            buffer: BTreeMap::new(),
            arrival_seq: 0,
            applied_high_water: Lsn::ZERO,
            orphan_pages: Vec::new(),
            metrics: Arc::new(ReplayMetrics::default()),
            checkpoint_floor: Lsn::ZERO,
            incremental_redo_floor: None,
        }
    }

    /// SVC-1 / #849 / ADR-229 — anchor this replay at a checkpoint
    /// frontier. Records with `commit_lsn <= floor` are skipped (already
    /// durable in the restored checkpoint snapshot); only `> floor` are
    /// applied. `Lsn::ZERO` is a no-op (replay from the beginning).
    #[must_use]
    pub fn with_checkpoint_floor(mut self, floor: Lsn) -> Self {
        self.checkpoint_floor = floor;
        self
    }

    /// Configure a v9 incremental checkpoint: logical/metadata effects at or
    /// below `checkpoint_floor` are already present, while physical redo must
    /// begin at `redo_floor` (the minimum DPT recLSN).
    #[must_use]
    pub fn with_incremental_checkpoint(mut self, checkpoint_floor: Lsn, redo_floor: Lsn) -> Self {
        self.checkpoint_floor = checkpoint_floor;
        self.incremental_redo_floor = Some(redo_floor);
        self
    }

    /// Shared handle to the executor's metrics surface. Cloneable
    /// so operators / tests can watch gauges during a long replay.
    #[must_use]
    pub fn metrics(&self) -> Arc<ReplayMetrics> {
        Arc::clone(&self.metrics)
    }

    /// High-water of successfully applied `commit_lsn`. Equals
    /// `TxnManager::current_lsn()` after [`Self::run`] returns.
    #[inline]
    #[must_use]
    pub fn applied_high_water(&self) -> Lsn {
        self.applied_high_water
    }

    // ─── Main run loop ────────────────────────────────────────

    /// Execute one full replay pass on `reader`.
    ///
    /// Reads records in WAL-LSN order; buffers each decoded
    /// [`DecodedCommitBundle`] by its `commit_lsn`; applies
    /// legacy records (`Commit = 2`, `IndexPage = 11`, and other
    /// pre-ADR-031 payloads) immediately; drains the buffer on
    /// (a) end-of-WAL via `Self::final_drain` or (b) buffer
    /// overflow via `Self::overflow_flush`; and finally seeds
    /// the [`TxnManager`] post-replay.
    ///
    /// Returns the max `commit_lsn` applied (or `Lsn::ZERO` if the
    /// WAL was empty). Errors:
    ///
    /// - [`ArcGraphError::WalCorruption`] — halt; §R5 / §6.
    /// - [`ArcGraphError::WalFormatMismatch`] — segment header
    ///   version unsupported; §6 "Note on WalFormatMismatch".
    /// - [`ArcGraphError::UnrecoverableOrphans`] — orphan pages
    ///   observed and `bootstrap_from_mvcc` hook failed (§Slice 3c).
    pub fn run(&mut self, mut reader: WalRecoveryReader) -> Result<Lsn> {
        let start = Instant::now();
        let span = info_span!("wal_replay", spill_enabled = self.cfg.spill_enabled);
        let _span_guard = span.enter();
        info!(
            max_buffer_bundles = self.cfg.max_buffer_bundles,
            max_buffer_bytes = self.cfg.max_buffer_bytes,
            "wal_replay_started (ADR-032 §3)",
        );
        self.metrics.set_phase(ReplayPhase::Reading);

        // Seed the applied-high-water from the TxnManager's visible
        // watermark. On a FIRST, clean durable-bootstrap replay this is
        // `Lsn::ZERO` (0): the bootstrap passes a fresh `TxnManager::new()`
        // whose `visible` initialises to `Lsn::ZERO`, and recovery runs
        // BEFORE any writer / catalog / `PrimaryIndex` attaches (so nothing
        // has advanced the counter yet — see `bootstrap.rs` §5 ordering).
        // Since `LsnCounter::INITIAL == 1`, the LOWEST real commit_lsn any
        // bundle can carry is 1, and the skip-if-applied guard in
        // `buffer_insert` skips only `commit_lsn <= applied_high_water`;
        // with the baseline at 0, `1 <= 0` is false, so even a bundle at
        // the lowest possible LSN — including a stage-1 #1221 (ADR-218)
        // `acl_grants`-only commit — is always applied, never skipped.
        //
        // The baseline is NON-zero in two legitimate cases: (a) a
        // re-replay (M7 double-replay, or a crash mid-replay that left
        // partial state) where it is the max commit_lsn already durable
        // and bundles ≤ it skip-if-applied (Lemma I1 outer guard); and
        // (b) replay-shape tests that build a `PrimaryIndex` on the SAME
        // manager before recovery (the root-page alloc advances the
        // counter). So this is NOT unconditionally `Lsn::ZERO` and we do
        // NOT assert it here — the production safety lives at the
        // `recover_from_wal` entry contract instead (see the
        // `debug_assert` there: #1221 forward-bind against a future
        // refactor that pre-seeds the recovery manager).
        // SVC-1 / #849 / ADR-229 — checkpoint-anchored recovery: raise
        // the baseline to the checkpoint frontier so bundles at/below it
        // (already durable in the restored checkpoint snapshot) skip the
        // apply, and ONLY WAL records with `commit_lsn > checkpoint_floor`
        // are replayed. `max(...)` preserves the existing re-replay /
        // pre-seeded-manager cases; `checkpoint_floor == Lsn::ZERO` (no
        // checkpoint / back-compat) leaves the pre-ADR-229 baseline.
        let baseline = self.txn_mgr.current_lsn();
        self.applied_high_water = if self.incremental_redo_floor.is_some() {
            Lsn::ZERO
        } else if self.checkpoint_floor.raw() > baseline.raw() {
            self.checkpoint_floor
        } else {
            baseline
        };

        // Reload any pre-existing spill files from a crashed prior
        // replay (§5 last paragraph: "a crash during spill produces
        // a file whose trailing CRC does not match; the next replay
        // detects this and discards the spill"). Slice 3b loads on
        // open; here we delegate to the spill module.
        self.reload_spill_from_disk()?;

        // Need the WAL dir + per-segment format_version to
        // dispatch the v1/v2 decoder. The reader exposes the
        // record's LSN but not the segment; we pre-scan the dir
        // and cache the (segment_no → format_version) map. The
        // cache is O(N_segments) entries; cheap at realistic
        // WAL sizes.
        let format_map = build_format_map(reader.dir())?;

        #[allow(clippy::while_let_on_iterator)] // we need explicit reader.next() to preserve
        //    access to reader.torn_tail() after the loop
        while let Some(item) = reader.next() {
            let record = match item {
                Ok(r) => r,
                Err(e) => {
                    // §R5 / §6: corruption errors halt; the torn-
                    // tail case is classified by the reader as
                    // `None` (with `torn_tail().is_some()`),
                    // NOT as an error. Any error reaching here is
                    // `WalCorruption` / `WalFormatMismatch` /
                    // `WalBadMagic`.
                    if matches!(
                        e,
                        ArcGraphError::WalCorruption { .. }
                            | ArcGraphError::WalFormatMismatch { .. }
                            | ArcGraphError::WalBadMagic { .. }
                    ) {
                        self.metrics.corruption_halts.store(1, Ordering::Release);
                        error!(error = ?e, "wal_replay halted on corruption (§R5 / §6)");
                    }
                    return Err(e);
                }
            };
            self.metrics.records_total.fetch_add(1, Ordering::Relaxed);
            // Route per-record format_version through the segment
            // the reader is currently positioned on. This is the
            // per-segment dispatch the ADR-032 §2 codec contract
            // demands (v1 segment → v1 decoder; v2 segment → v2
            // decoder).
            let seg_no = reader.current_seg_no();
            self.handle_record(record, seg_no, &format_map)?;

            // Post-record: check buffer pressure (§3 OVERFLOW_FLUSH
            // trigger).
            if self.buffer.len() >= self.cfg.max_buffer_bundles
                || self.current_buffer_bytes() >= self.cfg.max_buffer_bytes
            {
                self.overflow_flush()?;
            }
        }

        // Reader exhausted (clean end-of-WAL OR terminal torn
        // tail — both are §R5 recoverable). Log + drain.
        if let Some(tt) = reader.torn_tail() {
            info!(
                segment = tt.segment,
                offset = tt.offset,
                "torn tail observed (§R5)"
            );
        }
        self.final_drain()?;

        if self.incremental_redo_floor.is_some()
            && self.applied_high_water.raw() < self.checkpoint_floor.raw()
        {
            self.applied_high_water = self.checkpoint_floor;
        }

        // §R2 Step 3d + §R2 Step 5: seed TxnManager counters and
        // advance the reader's last_lsn up to the WAL writer
        // high-water.
        self.txn_mgr.seed_after_replay(self.applied_high_water);
        self.metrics
            .current_commit_lsn
            .store(self.applied_high_water.raw(), Ordering::Release);
        self.metrics.set_phase(ReplayPhase::Completed);

        // §Slice 3c orphan escalation: if we saw legacy IndexPage
        // records without a matching bundle AND the hook exists,
        // invoke bootstrap_from_mvcc. Failure => UnrecoverableOrphans.
        if !self.orphan_pages.is_empty() {
            let count = self.orphan_pages.len() as u64;
            warn!(
                orphan_count = count,
                "wal_replay_orphan_detected (§Slice 3c)"
            );
            if let Some(hook) = self.target.bootstrap_from_mvcc.as_ref() {
                self.metrics
                    .bootstrap_from_mvcc_invoked
                    .store(1, Ordering::Release);
                match hook(count) {
                    Ok(()) => {
                        info!("bootstrap_from_mvcc completed after orphan-page observation");
                    }
                    Err(e) => {
                        error!(error = ?e, "bootstrap_from_mvcc failed; halting with UnrecoverableOrphans");
                        return Err(ArcGraphError::UnrecoverableOrphans {
                            orphan_count: count,
                            reason: format!("{e}"),
                        });
                    }
                }
            } else {
                // No hook registered — log but don't escalate. The
                // ADR's contract (§Slice 3c) is that an orphan with
                // no hook is a warning, not a halt. Operators may
                // trigger bootstrap manually via the v1.0-GA
                // `arcgraph verify` tooling (§O3).
                debug!(
                    orphan_count = count,
                    "orphan pages observed but no bootstrap hook registered; leaving to operator",
                );
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;
        info!(
            elapsed_ms,
            applied = self.applied_high_water.raw(),
            records_observed = self.metrics.records_total.load(Ordering::Relaxed),
            bundles_applied = self.metrics.bundles_applied.load(Ordering::Relaxed),
            mvcc_installed = self.metrics.mvcc_versions_installed.load(Ordering::Relaxed),
            index_pages = self.metrics.index_pages_applied.load(Ordering::Relaxed),
            "wal_replay_completed (ADR-032 §3)",
        );

        // On success, discard any spill files we created during
        // this replay run. Slice 3b.
        self.discard_spill_files();

        Ok(self.applied_high_water)
    }

    // ─── Per-record dispatch ──────────────────────────────────

    fn handle_record(
        &mut self,
        record: WalRecord,
        seg_no: Option<u64>,
        format_map: &std::collections::HashMap<u64, u16>,
    ) -> Result<()> {
        match record.record_type {
            WalRecordType::CommitBundle => {
                // Decode under segment-header format_version
                // dispatch (ADR-032 Slice 1 / §2 & §R1). The
                // current-segment look-up is authoritative; we
                // fall back to the newest scanned version only if
                // the reader's internal state was cleared mid-
                // iteration (shouldn't happen — defensive only).
                let fmt_version = seg_no
                    .and_then(|n| format_map.get(&n).copied())
                    .unwrap_or_else(|| {
                        format_map
                            .values()
                            .copied()
                            .max()
                            .unwrap_or(crate::wal::segment::CURRENT_WAL_FORMAT_VERSION)
                    });
                let bundle = decode_commit_bundle_for_version(
                    &record.payload,
                    fmt_version,
                    record.tenant_id,
                )?;
                self.buffer_insert(bundle);
            }
            WalRecordType::IndexPage => {
                // Legacy pre-ADR-031 record. Post-ADR-031 the hot
                // path no longer emits these, but replay must
                // tolerate pre-fix WAL fixtures. Classify as an
                // orphan and install into the page store: ADR-032
                // Invariant 13 ("orphan-page tolerance") + §R1
                // (record-type classification) are the basis for
                // treating a legacy, non-bundle IndexPage record as an
                // orphan (#769 R1 NIT #4 corrected this from §Slice 3c).
                // The page is then tracked for the Slice-3 post-replay
                // `bootstrap_from_mvcc` escalation rung. (§R2 Step 3c is
                // a *different* rule — applying IndexPage entries carried
                // inside a CommitBundle.)
                self.apply_legacy_index_page(&record)?;
            }
            WalRecordType::Commit => {
                // Legacy pre-ADR-031 single-txn commit record. The
                // CRUD layer fully transitioned to CommitBundle at
                // PR #67; pre-PR-67 fixtures may still contain
                // these. We accept them as no-op on replay (the
                // MVCC chains have nothing to replay from this
                // shape — legacy Commit records pre-date the
                // CommitBundle mvcc_writes section). A legacy WAL
                // needing full replay must run
                // bootstrap_from_mvcc post-replay.
                // Non-fatal: the CommitBundle path supersedes.
                debug!(
                    lsn = record.lsn.raw(),
                    "legacy Commit = 2 record observed; no-op on replay (§R1)"
                );
            }
            WalRecordType::Begin | WalRecordType::Abort | WalRecordType::Checkpoint => {
                // Metadata markers: accept + no-op.
                debug!(ty = ?record.record_type, "metadata record observed (§R1)");
            }
            WalRecordType::InternString => {
                // P0 #776 — reconstruct the label / rel-type name↔id
                // mapping so `graph.schema` shows real names and typed
                // queries (`MATCH (a:Account)`) resolve after a durable
                // restart. Before this fix the arm was a no-op (grouped
                // with the legacy types below) and the production write
                // path never logged interns, so names came back as
                // synthetic `label:N` and typed queries failed -32005.
                //
                // No-op when no intern table is wired (replay-shape unit
                // tests + callers that don't recover a name table) —
                // preserving the prior behaviour for those paths.
                match self.target.intern_table() {
                    Some(table) => {
                        // The framing layer already CRC-validated the
                        // payload; a decode failure here is a genuine
                        // format violation, so halt loud + consistent
                        // with the CommitBundle corruption posture
                        // (re-stamped with the record's real LSN).
                        let (id, name) = decode_intern_payload(&record.payload).map_err(|e| {
                            ArcGraphError::WalCorruption {
                                lsn: record.lsn,
                                reason: format!("InternString payload decode failed: {e}"),
                            }
                        })?;
                        table.intern_install(record.tenant_id, id, &name);
                        self.metrics
                            .interns_recovered
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    None => {
                        debug!(
                            lsn = record.lsn.raw(),
                            "InternString record observed but no intern table wired; \
                             name recovery skipped (replay-shape caller)"
                        );
                    }
                }
            }
            WalRecordType::PutBlob
            | WalRecordType::PutNode
            | WalRecordType::PutRel
            | WalRecordType::DeleteNode
            | WalRecordType::DeleteRel => {
                // Pre-M2.c legacy record types. Apply-on-replay
                // for these is handled by the dedicated stores'
                // recovery hooks (not in scope for ADR-032 §R1 —
                // those paths pre-existed M2.e). At v1.0 these
                // are dead code in the commit path, so on fresh
                // v1.0 WALs they never appear.
                debug!(
                    ty = ?record.record_type,
                    "legacy pre-M2.c record observed; apply path not in ADR-032 scope"
                );
            }
        }
        Ok(())
    }

    fn apply_legacy_index_page(&mut self, record: &WalRecord) -> Result<()> {
        use crate::primary_index::decode_index_page_payload;
        let (page_id, _tenant, page_bytes) =
            decode_index_page_payload(&record.payload).map_err(|e| {
                ArcGraphError::WalCorruption {
                    lsn: record.lsn,
                    reason: format!("legacy IndexPage decode failed: {e}"),
                }
            })?;
        // Install via the normal route; legacy pages ride the
        // primary store by default (kind = PrimaryIndex).
        let entry = DecodedIndexPage {
            kind: crate::wal::bundle::BundlePageKind::PrimaryIndex,
            page_id,
            tenant_id: _tenant,
            bytes: page_bytes,
        };
        // Legacy IndexPage records are always PrimaryIndex-kind
        // (synthesized by `apply_legacy_index_page`); counting them
        // as index_pages_applied is unconditional here.
        let routed = self.target.install_index_page(&entry)?;
        self.bump_staged_page_counter(routed);
        // Classified as an orphan per ADR-032 Invariant 13 + §R1, then
        // tracked for the Slice-3 post-replay `bootstrap_from_mvcc`
        // escalation rung. Orphans may later be installed by a
        // subsequent bundle; that's benign (Lemma I2).
        self.orphan_pages.push(page_id);
        self.metrics
            .orphan_pages_detected
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // ─── Buffer operations ────────────────────────────────────

    fn buffer_insert(&mut self, bundle: DecodedCommitBundle) {
        let commit_lsn = bundle.commit_lsn.raw();
        // §R2 Step 1 skip-if-applied: bundles <= applied_high_water
        // have already been absorbed by a prior OVERFLOW_FLUSH.
        // Skip entirely (idempotent — Lemma I1 guards chain-level
        // correctness if we try anyway, but the bookkeeping saves
        // buffer space).
        if commit_lsn <= self.applied_high_water.raw() {
            self.metrics
                .bundles_skipped_idempotent
                .fetch_add(1, Ordering::Relaxed);
            // SVC-1 / #849 / ADR-229 BLOCK-1 belt-and-suspenders: even for
            // a bundle skipped below the checkpoint floor (its MVCC / page
            // effects are already durable in the restored snapshot), STILL
            // re-seed its allocator high-waters. `seed_from_advance` is
            // monotonic-max + idempotent (Lemma I3), so re-applying an
            // advance the checkpoint already captured is a harmless no-op —
            // but if the checkpoint's allocator capture ever under-counted
            // (it does not, given the commit-freeze fix, but defense in
            // depth), this guarantees the next post-recovery alloc_node /
            // alloc_rel / page alloc still lands STRICTLY above every id a
            // skipped-below-floor committed bundle consumed. Skips only the
            // MVCC/page apply, never the monotonic-max advance re-seed.
            for advance in bundle.allocator_advances.iter().copied() {
                self.target.apply_allocator_advance(advance);
                self.metrics
                    .allocator_advances_applied
                    .fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
        let bytes_budget = estimate_bundle_bytes(&bundle);
        let range = bundle.redo_range();
        let key = (range.base().raw(), range.end().raw(), self.arrival_seq);
        self.arrival_seq = self.arrival_seq.wrapping_add(1);
        self.buffer.insert(
            key,
            BufferedBundle {
                bundle,
                bytes_budget,
            },
        );
        self.metrics
            .bundles_buffered
            .store(self.buffer.len() as u64, Ordering::Relaxed);
        self.metrics
            .buffer_memory_bytes
            .fetch_add(bytes_budget as u64, Ordering::Relaxed);
    }

    fn current_buffer_bytes(&self) -> usize {
        self.metrics.buffer_memory_bytes.load(Ordering::Relaxed) as usize
    }

    /// OVERFLOW_FLUSH (ADR-032 §3). Default path: spill-enabled.
    /// Drain the lowest-`commit_lsn` bundles to the spill file
    /// until buffer pressure drops below the configured thresholds.
    /// Under spill-disabled, pop the MIN bundle and apply it
    /// immediately — correct iff in-flight slack ≤ buffer bound.
    fn overflow_flush(&mut self) -> Result<()> {
        self.metrics
            .overflow_flush_fired
            .fetch_add(1, Ordering::Relaxed);
        warn!(
            buffered = self.buffer.len(),
            memory_bytes = self.current_buffer_bytes(),
            spill_enabled = self.cfg.spill_enabled,
            "wal_replay_overflow_flush_fired (ADR-032 §3 / §5)",
        );
        if self.cfg.spill_enabled {
            self.spill_engage()
        } else {
            self.stream_apply_min()
        }
    }

    fn stream_apply_min(&mut self) -> Result<()> {
        // Under spill=off, pop+apply until we drop below the
        // lower hysteresis watermark. The design document §3
        // calls for a single pop-min-and-apply per overflow; to
        // make progress when the overflow was caused by a large
        // single bundle, we pop one. The caller's while-loop in
        // run() will re-enter if still over threshold.
        if let Some((_, buffered)) = self.buffer.pop_first() {
            let bytes_budget = buffered.bytes_budget;
            self.apply_bundle(buffered.bundle)?;
            self.metrics
                .bundles_buffered
                .store(self.buffer.len() as u64, Ordering::Relaxed);
            self.metrics
                .buffer_memory_bytes
                .fetch_sub(bytes_budget as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn spill_engage(&mut self) -> Result<()> {
        // Delegate to the spill module (Slice 3b). The spill
        // module creates a new spill file with a sorted batch
        // drawn from the low end of the buffer.
        use crate::wal::spill::write_spill_batch;
        let target_memory = self.cfg.max_buffer_bytes / 2; // hysteresis
        let target_count = self.cfg.max_buffer_bundles / 2;
        let mut to_spill: Vec<DecodedCommitBundle> = Vec::new();
        let mut freed_bytes: usize = 0;
        while !self.buffer.is_empty()
            && (self.buffer.len() > target_count
                || self.current_buffer_bytes().saturating_sub(freed_bytes) > target_memory)
        {
            let Some((_, buffered)) = self.buffer.pop_first() else {
                break;
            };
            freed_bytes += buffered.bytes_budget;
            to_spill.push(buffered.bundle);
        }
        if to_spill.is_empty() {
            // Nothing to drain — caller was triggered by the
            // buffer's own headroom threshold. Apply one bundle
            // to make progress.
            return self.stream_apply_min();
        }
        let count = to_spill.len() as u64;
        // Ensure spill dir exists.
        std::fs::create_dir_all(&self.cfg.spill_dir)?;
        let path = write_spill_batch(&self.cfg.spill_dir, &to_spill)?;
        self.metrics
            .bundles_spilled
            .fetch_add(count, Ordering::Relaxed);
        self.metrics
            .spill_files_created
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .bundles_buffered
            .store(self.buffer.len() as u64, Ordering::Relaxed);
        self.metrics
            .buffer_memory_bytes
            .fetch_sub(freed_bytes as u64, Ordering::Relaxed);
        info!(
            path = ?path,
            count,
            freed_bytes,
            "wal_replay_spill_engaged (ADR-032 §5 X-1)",
        );
        Ok(())
    }

    /// FINAL_DRAIN (§3). Merge in-memory buffer + all on-disk
    /// spill files in ascending commit_lsn order; apply each
    /// bundle via [`Self::apply_bundle`].
    fn final_drain(&mut self) -> Result<()> {
        self.metrics.set_phase(ReplayPhase::Draining);
        use crate::wal::spill::load_all_spill_bundles;
        let spill_bundles = load_all_spill_bundles(&self.cfg.spill_dir)?;
        if !spill_bundles.is_empty() {
            debug!(
                n_spill_bundles = spill_bundles.len(),
                "loaded spill bundles for final drain"
            );
        }
        // Merge: push spill bundles into the sorted BTreeMap
        // (dedupes by commit_lsn; buffer wins on collision which
        // is fine because the payload is byte-identical under
        // Lemma I1+I2).
        for bundle in spill_bundles {
            let commit_lsn = bundle.commit_lsn.raw();
            if commit_lsn <= self.applied_high_water.raw() {
                // Already applied — skip.
                self.metrics
                    .bundles_skipped_idempotent
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let range = bundle.redo_range();
            let key = (range.base().raw(), range.end().raw(), self.arrival_seq);
            self.arrival_seq = self.arrival_seq.wrapping_add(1);
            self.buffer.insert(
                key,
                BufferedBundle {
                    bytes_budget: estimate_bundle_bytes(&bundle),
                    bundle,
                },
            );
        }
        self.metrics
            .bundles_buffered
            .store(self.buffer.len() as u64, Ordering::Relaxed);

        // Apply in commit_lsn-ascending order (BTreeMap iteration
        // is sorted).
        let mut previous_range = None;
        while let Some((_, buffered)) = self.buffer.pop_first() {
            let bytes_budget = buffered.bytes_budget;
            let range = buffered.bundle.redo_range();
            if let Some(previous) = previous_range {
                if range != previous && range.base().raw() <= previous.end().raw() {
                    return Err(ArcGraphError::WalCorruption {
                        lsn: range.base(),
                        reason: format!(
                            "v9 redo ranges overlap: previous {previous:?}, current {range:?}"
                        ),
                    });
                }
                if range.base().raw() > previous.end().raw().saturating_add(1) {
                    debug!(previous = ?previous, current = ?range, "legal v9 redo LSN gap");
                }
            }
            if previous_range != Some(range) {
                previous_range = Some(range);
            }
            self.apply_bundle(buffered.bundle)?;
            self.metrics
                .buffer_memory_bytes
                .fetch_sub(bytes_budget as u64, Ordering::Relaxed);
        }
        self.metrics.bundles_buffered.store(0, Ordering::Relaxed);
        Ok(())
    }

    // ─── Per-bundle apply (§R2) ───────────────────────────────

    /// N-2 (issue #81) fold-in: bump the right staged_page counter
    /// based on the [`BundlePageKind`] the routing trait accepted.
    ///
    /// Primary / secondary / record / vector pages share
    /// `index_pages_applied` (back-compat with ADR-032 §7 naming —
    /// all four are "page reinstall" work). Blob pages land on
    /// `blob_pages_applied` so operators can distinguish chain-walk
    /// reconstruction from page-reinstall load. The counter name
    /// remains `index_pages_applied` to avoid breaking pre-N-2
    /// dashboards.
    ///
    /// **M3.a Slice G.1 note.** Vector pages lump into
    /// `index_pages_applied` for now to keep the observability
    /// surface stable across the stub-only G.1 landing. Slice G.2
    /// (snapshot) may split out a dedicated `vector_pages_applied`
    /// counter once vector load becomes operationally interesting;
    /// that addition extends `ReplayMetricsSnapshot` and is a
    /// deliberate observability change, not an accidental one.
    fn bump_staged_page_counter(&self, kind: crate::wal::bundle::BundlePageKind) {
        use crate::wal::bundle::BundlePageKind;
        match kind {
            BundlePageKind::PrimaryIndex
            | BundlePageKind::SecondaryIndex
            | BundlePageKind::Record
            | BundlePageKind::Vector => {
                self.metrics
                    .index_pages_applied
                    .fetch_add(1, Ordering::Relaxed);
            }
            // v2 M1: PropSlotted pages count with Blob pages — both
            // route to the blob store and share its observability
            // surface (a dedicated counter is a deliberate future
            // observability change, mirroring the Vector note above).
            BundlePageKind::Blob | BundlePageKind::PropSlotted => {
                self.metrics
                    .blob_pages_applied
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// PR #79 Y-3 fold-in: route a per-write
    /// [`ReplayApplyOutcome`] from
    /// [`TxnManager::apply_replay_mvcc_write`] into the matching
    /// counter. OOO outcomes ALSO emit `tracing::error!` so the
    /// regression is loud in logs.
    fn route_mvcc_outcome(
        &self,
        outcome: ReplayApplyOutcome,
        tenant: TenantId,
        key: MvccKey,
        commit_lsn: Lsn,
    ) {
        match outcome {
            ReplayApplyOutcome::Applied => {
                self.metrics
                    .mvcc_versions_installed
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReplayApplyOutcome::Idempotent => {
                self.metrics
                    .bundles_skipped_idempotent
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReplayApplyOutcome::OutOfOrder => {
                self.metrics
                    .out_of_order_apply_rejected
                    .fetch_add(1, Ordering::Relaxed);
                error!(
                    tenant_id = tenant.raw(),
                    key,
                    commit_lsn = commit_lsn.raw(),
                    "wal_replay: OOO apply rejected — executor bug (bundle not sorted before \
                     apply). See ReplayApplyOutcome::OutOfOrder + ADR-032 §R2 step 2.",
                );
            }
        }
    }

    fn apply_bundle(&mut self, bundle: DecodedCommitBundle) -> Result<()> {
        // Skip-if-applied at apply time too (double-check invariant).
        if bundle.commit_lsn.raw() <= self.applied_high_water.raw() {
            self.metrics
                .bundles_skipped_idempotent
                .fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let commit_lsn = bundle.commit_lsn;

        // Incremental-checkpoint recovery has two frontiers. The logical
        // owners and retained page images are absolute state captured at the
        // checkpoint, so bundles at/below it must not re-drive them. Physical
        // store-0/1 deltas are different: dirty home pages may lag the
        // checkpoint, and ARIES requires replay from min-recLSN.
        if let Some(redo_floor) = self.incremental_redo_floor
            && commit_lsn.raw() <= self.checkpoint_floor.raw()
        {
            // A retained WAL segment can contain commit ranges below the
            // incremental redo floor when disk/segment order differs from
            // commit-LSN order. The checkpointed home page is authoritative
            // for such a range. Replaying only its MVCC projection would
            // detach logical state from the (correctly skipped) physical
            // delta and can resurrect a record tombstoned in the home base.
            // `commit_lsn` is the inclusive range end, so a range wholly
            // below `redo_floor` is skipped in full.
            if commit_lsn.raw() < redo_floor.raw() {
                self.metrics
                    .bundles_skipped_idempotent
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            // Incremental metadata deliberately omits MVCC. The base loader
            // reconstructs it from home record pages, which may lag as far
            // back as redo_floor. Re-drive the bundle's logical record rows
            // alongside physical redo so creates/updates/deletes at or below
            // checkpoint_lsn cannot disappear or resurrect after restart.
            for (key, value) in bundle.mvcc_writes.iter() {
                let outcome = self.txn_mgr.apply_incremental_replay_mvcc_write(
                    commit_lsn,
                    bundle.primary_tenant,
                    *key,
                    value.clone(),
                );
                self.route_mvcc_outcome(outcome, bundle.primary_tenant, *key, commit_lsn);
            }
            for sc in bundle.sidechannel_writes.iter() {
                let outcome = self.txn_mgr.apply_incremental_replay_mvcc_write(
                    commit_lsn,
                    sc.tenant_id,
                    sc.key,
                    sc.value.clone(),
                );
                self.route_mvcc_outcome(outcome, sc.tenant_id, sc.key, commit_lsn);
            }
            let mut delta_versions = std::collections::BTreeMap::new();
            for op in &bundle.deltas {
                if let Some((key, value)) = crate::wal::delta::put_record_mvcc_write(op)? {
                    delta_versions.insert((op.tenant_id, key), value);
                }
            }
            for ((tenant, key), value) in delta_versions {
                let outcome = self.txn_mgr.apply_incremental_replay_mvcc_write(
                    commit_lsn,
                    tenant,
                    key,
                    Some(value),
                );
                self.route_mvcc_outcome(outcome, tenant, key, commit_lsn);
            }
            for op in bundle
                .deltas
                .iter()
                .filter(|op| op.kind.is_physical() && op.op_lsn.raw() >= redo_floor.raw())
            {
                if op.kind == crate::wal::DeltaOpKind::ExtentAlloc {
                    self.target.apply_extent_alloc(op)?;
                    continue;
                }
                self.target.apply_physical_delta(op, commit_lsn)?;
            }
            self.applied_high_water = commit_lsn;
            self.metrics
                .current_commit_lsn
                .store(commit_lsn.raw(), Ordering::Release);
            self.metrics.bundles_applied.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // (a) Primary MVCC writes.
        for (key, value) in bundle.mvcc_writes.iter() {
            let outcome = self.txn_mgr.apply_replay_mvcc_write(
                commit_lsn,
                bundle.primary_tenant,
                *key,
                value.clone(),
            );
            self.route_mvcc_outcome(outcome, bundle.primary_tenant, *key, commit_lsn);
        }

        // (b) Side-channel MVCC writes (v2 only).
        for sc in bundle.sidechannel_writes.iter() {
            let outcome = self.txn_mgr.apply_replay_mvcc_write(
                commit_lsn,
                sc.tenant_id,
                sc.key,
                sc.value.clone(),
            );
            self.route_mvcc_outcome(outcome, sc.tenant_id, sc.key, commit_lsn);
        }

        // IMPL-DEC-3: v9 PutRecord bytes are the MVCC version bytes. The v9
        // encoder removes the redundant section-2 copy after byte-checking
        // equality, so replay reconstructs one final version per tenant/key
        // from the delta stream before applying physical page mutations.
        let mut delta_versions = std::collections::BTreeMap::new();
        for op in &bundle.deltas {
            if let Some((key, value)) = crate::wal::delta::put_record_mvcc_write(op)? {
                delta_versions.insert((op.tenant_id, key), value);
            }
        }
        for ((tenant, key), value) in delta_versions {
            let outcome =
                self.txn_mgr
                    .apply_replay_mvcc_write(commit_lsn, tenant, key, Some(value));
            self.route_mvcc_outcome(outcome, tenant, key, commit_lsn);
        }

        // (c) staged_pages entries — Lemma I2 guarantees idempotent
        //     byte-copy install. v3 bundles dispatch by
        //     BundlePageKind; v1/v2 decoders synthesize
        //     PrimaryIndex on every entry. N-2 (issue #81) splits
        //     the counter by kind so operators can distinguish
        //     blob-chain reconstruction load from B-tree page
        //     reinstall load.
        for entry in bundle.staged_pages.iter() {
            let routed = self.target.install_index_page(entry)?;
            self.bump_staged_page_counter(routed);
        }

        let has_physical_owner_delta = bundle.deltas.iter().any(|op| {
            matches!(
                op.kind,
                crate::wal::DeltaOpKind::InternBind | crate::wal::DeltaOpKind::AclGrant
            )
        });
        if has_physical_owner_delta
            && (!bundle.idempotency_bindings.is_empty() || !bundle.acl_grants.is_empty())
        {
            return Err(ArcGraphError::WalCorruption {
                lsn: commit_lsn,
                reason: "owner bundle carries both physical rows and legacy logical owner sections"
                    .to_owned(),
            });
        }

        // (c1) v9 physiological/logical deltas. Retained index images above
        // preserve v8 SMO semantics; physical record/props ops then traverse
        // the explicit Missing→Formatted→Live state machine and page-LSN
        // idempotence. Store 5 cannot route here (decoder + apply gate).
        for op in &bundle.deltas {
            match op.kind {
                crate::wal::DeltaOpKind::ExtentAlloc => {
                    self.target.apply_extent_alloc(op)?;
                }
                crate::wal::DeltaOpKind::PutRecord
                | crate::wal::DeltaOpKind::TombstoneRecord
                | crate::wal::DeltaOpKind::PutPropBlock
                | crate::wal::DeltaOpKind::InternBind
                | crate::wal::DeltaOpKind::AclGrant
                | crate::wal::DeltaOpKind::PageAlloc => {
                    self.target.apply_physical_delta(op, commit_lsn)?;
                }
                crate::wal::DeltaOpKind::AllocAdvance => {
                    let kind = crate::wal::AllocatorKind::from_byte(op.payload[0])?;
                    let high_water = u64::from_le_bytes(
                        op.payload[1..9]
                            .try_into()
                            .expect("AllocAdvance validated at decode"),
                    );
                    self.target.apply_allocator_advance(AllocatorAdvance {
                        tenant: op.tenant_id,
                        kind,
                        new_high_water: high_water,
                    });
                    self.metrics
                        .allocator_advances_applied
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => unreachable!("reserved v9 delta kind rejected by decoder"),
            }
        }

        // (c2) vector_pages entries — M3.a Slice G.4 (commit-bundle
        //      vector page staging). Apply AFTER `staged_pages` and
        //      BEFORE `allocator_advances` (Lemma I3 — monotonic
        //      idempotent replay; double-replay is a no-op). v5
        //      bundles carry one entry per vector arena page
        //      mutation; v1-v4 decoders synthesize empty
        //      `vector_pages` so this loop is a no-op for legacy
        //      segments. Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
        //
        //      Routes through the registered
        //      `VectorPageStoreHandle` mirroring the
        //      `BundlePageKind::Vector` arm in
        //      `install_index_page` (which remains for v3/v4
        //      back-compat in case any historical bundle ever
        //      carried a Vector entry — pre-v5 it should have been
        //      empty by construction, but the dispatch arm keeps a
        //      `tracing::warn!` posture for forward-progress).
        if !bundle.vector_pages.is_empty()
            && let Some(vector) = self.target.vector_store.as_ref()
        {
            for entry in bundle.vector_pages.iter() {
                vector
                    .install_or_replace(entry.tenant, entry.page_id, entry.bytes.as_ref())
                    .map_err(|e| ArcGraphError::WalCorruption {
                        lsn: commit_lsn,
                        reason: format!(
                            "VectorPageStore install_or_replace failed for v5 vector_pages \
                             entry (tenant={:?}, page={:?}, commit_lsn={:?}): {e}",
                            entry.tenant, entry.page_id, entry.commit_lsn,
                        ),
                    })?;
            }
        } else if !bundle.vector_pages.is_empty() {
            // No handle wired; mirror the staged_pages Vector arm's
            // "warn-and-continue" stub posture (M3.a Slice G.1 ⇒ G.4
            // contract) so replay forward-progress is preserved on
            // newer-WAL-on-older-deployment scenarios. Tightens to
            // reject-as-wiring-bug when downstream slices land.
            tracing::warn!(
                n_vector_pages = bundle.vector_pages.len(),
                commit_lsn = ?commit_lsn,
                "v5 vector_pages section non-empty but no \
                 VectorPageStoreHandle wired (M3.a Slice G.4 stub \
                 posture; see ADR-035 §4.5/§4.6)"
            );
        }

        // (d) allocator advances — Issue #129 P0 fix. Apply AFTER
        //     staged_pages so the post-recovery allocator state
        //     reflects the highest high-water observed across the
        //     whole replay (Lemma I3 — monotonic-max seeding is
        //     idempotent under double-replay). Without this leg,
        //     post-fault `create_node` re-issues NodeIds that
        //     pre-fault commits consumed and orphans earlier T1
        //     commits through the primary index (ADR-034 D-1).
        //     `apply_allocator_advance` is a no-op when no
        //     `AllocatorSeedHandle` is wired (e.g., legacy unit
        //     tests).
        for advance in bundle.allocator_advances.iter().copied() {
            self.target.apply_allocator_advance(advance);
            self.metrics
                .allocator_advances_applied
                .fetch_add(1, Ordering::Relaxed);
        }

        // (e) idempotency bindings — #352 Part 2 (ADR-199 v6 fold).
        //     Apply AFTER MVCC writes so the node/rel the binding
        //     references is recovered before its `external_id →
        //     internal_id` mapping installs (Lemma I3 ordering, mirroring
        //     allocator_advances). v6 bundles carry one entry per fresh
        //     external_id; v1-v5 decoders synthesize empty
        //     `idempotency_bindings`, so this is a no-op for legacy
        //     segments. `install` is an unconditional last-write-wins map
        //     insert (Lemma I2 — double-replay is a no-op). A no-op when
        //     no `IdempotencyStore` is wired (replay-shape unit tests +
        //     callers that don't recover the idempotency map); the durable
        //     serve path MUST wire it (the bootstrap does), else a
        //     post-restart re-ingest mints a duplicate — the #352 bug.
        if !bundle.idempotency_bindings.is_empty() {
            match self.target.idempotency_store() {
                Some(store) => {
                    for entry in bundle.idempotency_bindings.iter() {
                        match entry.op {
                            crate::wal::bundle::IdempotencyBindingOp::Install => {
                                store.install(
                                    entry.tenant,
                                    entry.kind,
                                    &entry.external_id,
                                    entry.internal_id,
                                );
                            }
                            crate::wal::bundle::IdempotencyBindingOp::Release => {
                                store.release(entry.tenant, entry.kind, &entry.external_id);
                            }
                        }
                        self.metrics
                            .idempotency_bindings_recovered
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    debug!(
                        n_idempotency_bindings = bundle.idempotency_bindings.len(),
                        commit_lsn = ?commit_lsn,
                        "v6 idempotency_bindings section non-empty but no \
                         IdempotencyStore wired; idempotency recovery skipped \
                         (replay-shape caller)"
                    );
                }
            }
        }

        // (f) acl_grants — #1221 (ADR-218 v8 fold). Re-drive each
        //     document ACL grant/revoke op into the wired
        //     `PermissionIndex` so a bare `serve --data` restart enforces
        //     against the recovered grants instead of denying all.
        //     **Replayed in WIRE (= staging/append) order WITHIN a
        //     bundle**, and bundles drain in ascending `commit_lsn` order,
        //     so last-writer-wins per doc is well-defined (ADR-218). Uses
        //     the `*_replayed` entry points, which DO NOT re-enter the WAL
        //     sink (the op is already durable — re-staging would
        //     double-log it on the next restart). v8 bundles carry one
        //     entry per `apply_doc_acl`/`revoke_doc`; v1-v7 decoders
        //     synthesize empty `acl_grants`, so this is a no-op for legacy
        //     segments. A no-op when no `PermissionIndex` is wired
        //     (replay-shape unit tests); the durable serve path MUST wire
        //     it (the bootstrap does), else enforcement comes up deny-all
        //     — the #1221 defect.
        if !bundle.acl_grants.is_empty() {
            match self.target.permission_index() {
                Some(index) => {
                    for entry in bundle.acl_grants.iter() {
                        // Cross-tenant isolation (ADR-212 §5 Q3): the index
                        // is per-tenant. At v1.0 the bundle's primary
                        // tenant == the entry tenant for the user tenant;
                        // skip a mismatched entry rather than apply it to
                        // the wrong index. (Defense-in-depth — the live
                        // path always stages the routed tenant.)
                        if entry.tenant != bundle.primary_tenant {
                            debug!(
                                entry_tenant = ?entry.tenant,
                                bundle_tenant = ?bundle.primary_tenant,
                                commit_lsn = ?commit_lsn,
                                "v8 acl_grant entry tenant != bundle tenant; skipped \
                                 (cross-tenant isolation, ADR-212 §5 Q3)"
                            );
                            continue;
                        }
                        match entry.op {
                            crate::wal::bundle::AclGrantOp::Apply => {
                                index.apply_doc_acl_replayed(entry.doc, entry.grants.clone());
                            }
                            crate::wal::bundle::AclGrantOp::Revoke => {
                                index.revoke_doc_replayed(entry.doc);
                            }
                        }
                        self.metrics
                            .acl_grants_recovered
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    debug!(
                        n_acl_grants = bundle.acl_grants.len(),
                        commit_lsn = ?commit_lsn,
                        "v8 acl_grants section non-empty but no PermissionIndex \
                         wired; ACL recovery skipped (replay-shape caller)"
                    );
                }
            }
        }

        self.applied_high_water = commit_lsn;
        self.metrics
            .current_commit_lsn
            .store(commit_lsn.raw(), Ordering::Release);
        self.metrics.bundles_applied.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // ─── Spill lifecycle helpers (Slice 3b glue) ─────────────

    fn reload_spill_from_disk(&mut self) -> Result<()> {
        use crate::wal::spill::count_spill_files;
        let reloaded = count_spill_files(&self.cfg.spill_dir)?;
        if reloaded > 0 {
            self.metrics
                .spill_files_reloaded
                .fetch_add(reloaded as u64, Ordering::Relaxed);
            debug!(
                n = reloaded,
                "pre-existing spill files observed (will merge in final_drain)"
            );
        }
        Ok(())
    }

    fn discard_spill_files(&self) {
        if !self.cfg.spill_enabled {
            return;
        }
        use crate::wal::spill::discard_spill_dir;
        if let Err(e) = discard_spill_dir(&self.cfg.spill_dir) {
            warn!(error = ?e, "failed to discard spill dir on replay completion");
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────

/// Rough memory footprint for a buffered bundle. MVCC writes
/// contribute their value-byte sum; IndexPage entries contribute
/// `PAGE_SIZE` each (plus 16 B framing, ignored). Used by the
/// byte-budget overflow trigger.
fn estimate_bundle_bytes(bundle: &DecodedCommitBundle) -> usize {
    let mvcc_bytes: usize = bundle
        .mvcc_writes
        .values()
        .map(|v| v.as_ref().map_or(0, bytes::Bytes::len))
        .sum();
    let sc_bytes: usize = bundle
        .sidechannel_writes
        .iter()
        .map(|sc| sc.value.as_ref().map_or(0, bytes::Bytes::len))
        .sum();
    let page_bytes = bundle.staged_pages.len() * PAGE_SIZE;
    let delta_bytes: usize = bundle.deltas.iter().map(|op| op.encoded_len()).sum();
    mvcc_bytes + sc_bytes + page_bytes + delta_bytes + 64 /* fixed overhead */
}

/// Build a (segment_no → format_version) map so the executor can
/// decode each `CommitBundle` payload with the correct codec
/// without re-opening segment files.
fn build_format_map(dir: &std::path::Path) -> Result<std::collections::HashMap<u64, u16>> {
    let mut out = std::collections::HashMap::new();
    let segments = list_segments(dir)?;
    for seg in segments {
        let path = dir.join(segment_filename(seg));
        let mut buf = [0u8; SegmentHeader::SIZE];
        // Best-effort read: an inaccessible segment is surfaced
        // by the reader anyway.
        match std::fs::File::open(&path) {
            Ok(f) => {
                use std::os::unix::fs::FileExt;
                if f.read_exact_at(&mut buf, 0).is_ok() {
                    if let Ok(header) = SegmentHeader::decode(&buf) {
                        out.insert(seg, header.format_version);
                    }
                }
            }
            Err(_) => {
                // Skip — reader will surface.
            }
        }
    }
    Ok(out)
}

// Expose `dir` accessor so `ReplayExecutor::run` can scan segments
// without re-opening the reader. Accessor added in recovery.rs —
// see `WalRecoveryReader::dir`.

// ─── Handle impls for the in-tree page stores ──────────────────

/// [`PrimaryPageStoreHandle`] impl for the real
/// [`crate::primary_index::PrimaryPageStore`].
///
/// **PR #79 X-1 review fold-in**: unconditional byte-copy install
/// (supersede-if-present, install-if-not). Lemma I2 is
/// **bundle-level** — a later bundle's entry for the same page_id
/// is a legitimate supersession. Bundle-level idempotence is
/// enforced upstream by the executor's
/// `applied_high_water >= bundle.commit_lsn` skip and the
/// `apply_replay_mvcc_write` Lemma I1 check; per-entry byte
/// equality is over-strict and was the X-1 bug.
impl PrimaryPageStoreHandle for crate::primary_index::PrimaryPageStore {
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> Result<()> {
        self.install_or_replace(page_id, page)
            .map_err(|e| ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!(
                    "primary page_store.install_or_replace({:?}) on replay failed: {}",
                    page_id, e
                ),
            })
    }

    fn contains(&self, page_id: PageId) -> bool {
        self.latch(page_id).is_ok()
    }
}

/// X-2 review fold-in: [`RecordPageStoreHandle`] impl for the real
/// [`crate::record_store::RecordPageStore`].
///
/// Uses the store's `install_or_replace` hook (added in the same
/// X-2 slice) for unconditional overwrite per Lemma I2
/// (bundle-level idempotence; later bundles supersede earlier ones
/// for the same page_id).
impl RecordPageStoreHandle for crate::record_store::RecordPageStore {
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> Result<()> {
        self.install_or_replace(page_id, page)
            .map_err(|e| ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!("record_store.install_or_replace({page_id:?}) on replay: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::{PageId, TenantId};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::idempotency::IdempotencyStore;
    use crate::primary_index::PrimaryPageStore;
    use crate::wal::bundle::{
        BundlePageKind, IdempotencyBindingEntry, IdempotencyBindingOp, StagedEmit,
        encode_commit_bundle_v8,
    };
    use crate::wal::writer::{WalConfig, WalWriter};

    fn mk_page(fill: u8) -> Box<[u8; PAGE_SIZE]> {
        Box::new([fill; PAGE_SIZE])
    }

    fn mk_emit(page_id: u64, fill: u8) -> StagedEmit {
        StagedEmit {
            kind: BundlePageKind::PrimaryIndex,
            page_id: PageId::new(page_id),
            bytes: mk_page(fill),
        }
    }

    fn write_bundle_to_wal(
        dir: &std::path::Path,
        commit_lsn: Lsn,
        mvcc: &HashMap<u64, Option<bytes::Bytes>>,
        staged: &[StagedEmit],
    ) {
        let cfg = WalConfig {
            dir: dir.to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: std::time::Duration::from_millis(2),
            group_commit_max_batch: 4,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        };
        let writer = WalWriter::spawn(cfg).unwrap();
        let handle = writer.handle();
        // v3 staged_pages: translate each StagedEmit to a
        // (kind, page_id, tenant, bytes) 4-tuple — tests use
        // PrimaryIndex kind by default.
        let staged_v4: Vec<(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> = staged
            .iter()
            .map(|e| {
                (
                    BundlePageKind::PrimaryIndex,
                    e.page_id,
                    TenantId::DEFAULT,
                    e.bytes.clone(),
                )
            })
            .collect();
        let payload = encode_commit_bundle_v8(
            commit_lsn,
            TenantId::DEFAULT,
            mvcc,
            &[],
            &staged_v4,
            &[],
            &[],
            &[], // #352 Part 2: no idempotency bindings in this replay fixture
            &[], // #1221 (ADR-218): no acl_grants in this replay fixture
        );
        handle
            .append(
                WalRecordType::CommitBundle,
                /*txn_id*/ 1,
                /*ts*/ 0,
                TenantId::DEFAULT,
                payload,
            )
            .unwrap();
        writer.shutdown().unwrap();
    }

    fn write_v7_idempotency_bundle_to_wal(
        dir: &std::path::Path,
        commit_lsn: Lsn,
        external_id: &str,
        internal_id: u64,
    ) {
        let cfg = WalConfig {
            dir: dir.to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: std::time::Duration::from_millis(2),
            group_commit_max_batch: 4,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        };
        let writer = WalWriter::spawn(cfg).unwrap();
        let handle = writer.handle();
        let bindings = vec![IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant: TenantId::DEFAULT,
            kind: 0,
            internal_id,
            external_id: external_id.to_owned(),
        }];
        let payload = encode_commit_bundle_v8(
            commit_lsn,
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &[],
            &[],
            &[],
            &bindings,
            &[], // #1221 (ADR-218): no acl_grants in this idempotency fixture
        );
        handle
            .append(
                WalRecordType::CommitBundle,
                /*txn_id*/ 1,
                /*ts*/ 0,
                TenantId::DEFAULT,
                payload,
            )
            .unwrap();
        writer.shutdown().unwrap();
    }

    #[test]
    fn replay_empty_wal_is_noop() {
        let dir = tempdir().unwrap();
        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let high = exec.run(reader).unwrap();
        assert_eq!(high, Lsn::ZERO);
        assert_eq!(exec.metrics().snapshot().bundles_applied, 0);
        assert_eq!(
            exec.metrics().snapshot().phase,
            ReplayPhase::Completed as u8
        );
    }

    #[test]
    fn replay_single_bundle_applies_cleanly() {
        let dir = tempdir().unwrap();
        let mut mvcc = HashMap::new();
        mvcc.insert(42u64, Some(bytes::Bytes::from_static(b"hello")));
        let staged = vec![mk_emit(100, 0xAA)];
        write_bundle_to_wal(dir.path(), Lsn::new(1), &mvcc, &staged);

        let txn_mgr = Arc::new(TxnManager::new());
        let primary_store = Arc::new(PrimaryPageStore::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let high = exec.run(reader).unwrap();
        assert_eq!(high, Lsn::new(1));
        let snap = exec.metrics().snapshot();
        assert_eq!(snap.bundles_applied, 1);
        assert_eq!(snap.mvcc_versions_installed, 1);
        assert_eq!(snap.index_pages_applied, 1);
        // Post-replay the TxnManager sees the version at snapshot = 1.
        let v = txn_mgr.read_at(TenantId::DEFAULT, 42, Lsn::new(1));
        assert_eq!(v.as_deref(), Some(&b"hello"[..]));
        // Page 100 is in the primary store.
        assert!(primary_store.latch(PageId::new(100)).is_ok());
    }

    // ─── Slice 4 observability surface ─────────────────────────

    #[test]
    fn metrics_snapshot_has_16_counters_and_4_gauges() {
        // ADR-032 §7: the observability surface is 13 counters
        // + 4 gauges. **PR #79 Y-3 fold-in** extends to 14
        // counters (+ `out_of_order_apply_rejected`). **N-2 /
        // issue #81** extends to 15 counters
        // (+ `blob_pages_applied`). **Issue #129 P0 fix**
        // extends to 16 counters (+
        // `allocator_advances_applied`). Any future change to
        // the metrics set MUST update `ReplayMetricsSnapshot`
        // and this test in the same PR, or operators'
        // dashboards break silently.
        let m = ReplayMetrics::default();
        let snap = m.snapshot();

        // Counters (16): accessed by name — if a field is
        // removed, this fails to compile, which is the desired
        // observability-regression alarm.
        let counters: [u64; 16] = [
            snap.records_total,
            snap.bundles_applied,
            snap.bundles_skipped_idempotent,
            snap.mvcc_versions_installed,
            snap.index_pages_applied,
            snap.bundles_spilled,
            snap.spill_files_created,
            snap.spill_files_reloaded,
            snap.orphan_pages_detected,
            snap.bootstrap_from_mvcc_invoked,
            snap.wal_errors_total,
            snap.corruption_halts,
            snap.overflow_flush_fired,
            snap.out_of_order_apply_rejected,
            snap.blob_pages_applied,
            snap.allocator_advances_applied,
        ];
        assert_eq!(counters.len(), 16, "issue #129: 16 counters");
        assert!(counters.iter().all(|c| *c == 0), "pristine = zero");

        // Gauges (4).
        let gauges_u64: [u64; 3] = [
            snap.bundles_buffered,
            snap.buffer_memory_bytes,
            snap.current_commit_lsn,
        ];
        let _phase_u8 = snap.phase;
        assert_eq!(gauges_u64.len(), 3, "3 u64 gauges + 1 u8 phase");
        assert!(gauges_u64.iter().all(|g| *g == 0), "pristine = zero");
        assert_eq!(snap.phase, 0, "pristine phase = Reading");
    }

    #[test]
    fn metrics_phase_transitions_reading_draining_completed() {
        // ADR-032 §7: `wal_replay_phase` gauge transitions.
        // Drive a replay end-to-end and assert Completed at
        // the end. The intermediate states are observable by a
        // concurrent reader of the gauge; we don't try to race
        // them in a unit test.
        let wal_dir = tempdir().unwrap();
        let mut mvcc = HashMap::new();
        mvcc.insert(1u64, Some(bytes::Bytes::from_static(b"x")));
        write_bundle_to_wal(wal_dir.path(), Lsn::new(1), &mvcc, &[]);

        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        assert_eq!(exec.metrics().snapshot().phase, 0, "pristine = Reading (0)");
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let _ = exec.run(reader).unwrap();
        assert_eq!(
            exec.metrics().snapshot().phase,
            ReplayPhase::Completed as u8,
            "post-run = Completed"
        );
    }

    #[test]
    fn overflow_flush_gauge_records_fire_count() {
        // Drives overflow ≥ 2 times and asserts the counter
        // reflects the fire count.
        let wal_dir = tempdir().unwrap();
        let spill_dir = tempdir().unwrap();
        for i in 1u64..=10 {
            let mut mvcc = HashMap::new();
            mvcc.insert(i, Some(bytes::Bytes::from(format!("v{i}"))));
            write_bundle_to_wal(wal_dir.path(), Lsn::new(i), &mvcc, &[]);
        }
        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let cfg = ReplayConfig {
            max_buffer_bundles: 2,
            max_buffer_bytes: usize::MAX,
            spill_enabled: true,
            spill_dir: spill_dir.path().to_path_buf(),
        };
        let mut exec = ReplayExecutor::new(cfg, Arc::clone(&txn_mgr), target);
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let _ = exec.run(reader).unwrap();
        let snap = exec.metrics().snapshot();
        assert!(
            snap.overflow_flush_fired >= 2,
            "expected ≥2 overflow_flush fires, got {}",
            snap.overflow_flush_fired,
        );
        assert_eq!(snap.bundles_applied, 10);
    }

    #[test]
    fn replay_buffer_overflow_spills_to_disk() {
        // ADR-032 §5 + §3 OVERFLOW_FLUSH under spill=on. We force
        // a small buffer cap so every few bundles triggers a
        // spill-engage, then verify final_drain merges spill +
        // in-memory buffer and applies in commit_lsn order.
        let wal_dir = tempdir().unwrap();
        let spill_dir = tempdir().unwrap();
        // Write 20 bundles with distinct MVCC keys + commit_lsns.
        // We use `ordered_commit_lsn` via sequential appends so the
        // WAL LSN order matches commit_lsn order for this test.
        for i in 1u64..=20 {
            let mut mvcc = HashMap::new();
            mvcc.insert(i, Some(bytes::Bytes::from(format!("v{i}"))));
            write_bundle_to_wal(wal_dir.path(), Lsn::new(i), &mvcc, &[]);
        }

        // Build an executor with a tiny buffer cap (3 bundles) so
        // overflow fires repeatedly.
        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let cfg = ReplayConfig {
            max_buffer_bundles: 3,
            max_buffer_bytes: usize::MAX,
            spill_enabled: true,
            spill_dir: spill_dir.path().to_path_buf(),
        };
        let mut exec = ReplayExecutor::new(cfg, Arc::clone(&txn_mgr), target);
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let high = exec.run(reader).unwrap();
        assert_eq!(high, Lsn::new(20));

        let snap = exec.metrics().snapshot();
        assert_eq!(snap.bundles_applied, 20);
        assert!(
            snap.spill_files_created > 0,
            "expected at least one spill file to be created"
        );
        assert!(
            snap.bundles_spilled > 0,
            "expected some bundles to be spilled"
        );
        // Post-replay every MVCC write is visible.
        for i in 1u64..=20 {
            let expected = format!("v{i}");
            let got = txn_mgr.read_at(TenantId::DEFAULT, i, Lsn::new(20));
            assert_eq!(got.as_deref(), Some(expected.as_bytes()), "key {i}");
        }
    }

    #[test]
    fn replay_spill_preserves_idempotency_bindings_1032() {
        // #1032 / #849 scale repro: at large WAL sizes replay crosses the
        // spill path. Re-encoding spilled bundles as a pre-v6 format drops
        // the idempotency binding tail, so the recovered nodes exist but
        // post-restart re-ingest misses the durable external_id binding and
        // mints duplicates. Force spill with a tiny byte budget so this stays
        // a small deterministic unit test.
        let wal_dir = tempdir().unwrap();
        let spill_dir = tempdir().unwrap();
        write_v7_idempotency_bundle_to_wal(wal_dir.path(), Lsn::new(1), "oldest", 10);
        write_v7_idempotency_bundle_to_wal(wal_dir.path(), Lsn::new(2), "newest", 20);

        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let idempotency = Arc::new(IdempotencyStore::new());
        let target =
            PageStoreTarget::primary_only(primary).with_idempotency_store(Arc::clone(&idempotency));
        let cfg = ReplayConfig {
            max_buffer_bundles: 1024,
            max_buffer_bytes: 1,
            spill_enabled: true,
            spill_dir: spill_dir.path().to_path_buf(),
        };
        let mut exec = ReplayExecutor::new(cfg, Arc::clone(&txn_mgr), target);
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let high = exec.run(reader).unwrap();

        assert_eq!(high, Lsn::new(2));
        let snap = exec.metrics().snapshot();
        assert!(
            snap.bundles_spilled > 0,
            "test must exercise the spill path"
        );
        assert_eq!(
            snap.idempotency_bindings_recovered, 2,
            "spilled bundles must retain their idempotency binding tail"
        );
        assert_eq!(idempotency.total_len(), 2);
        assert_eq!(
            idempotency
                .get(TenantId::DEFAULT, 0, "oldest")
                .expect("oldest binding recovered")
                .internal_id,
            10,
        );
        assert_eq!(
            idempotency
                .get(TenantId::DEFAULT, 0, "newest")
                .expect("newest binding recovered")
                .internal_id,
            20,
        );
    }

    // ─── Slice 3c edge cases ────────────────────────────────

    #[test]
    fn replay_torn_tail_halts_gracefully() {
        // ADR-032 §R5: a terminal torn-tail is recoverable — the
        // pre-tear prefix is applied; no error surfaces.
        let wal_dir = tempdir().unwrap();
        for i in 1u64..=3 {
            let mut mvcc = HashMap::new();
            mvcc.insert(i, Some(bytes::Bytes::from(format!("v{i}"))));
            write_bundle_to_wal(wal_dir.path(), Lsn::new(i), &mvcc, &[]);
        }
        // Truncate the last segment by 20 bytes so the tail record
        // is torn.
        let segs = crate::wal::segment::list_segments(wal_dir.path()).unwrap();
        let last_seg = *segs.last().unwrap();
        let path = wal_dir
            .path()
            .join(crate::wal::segment::segment_filename(last_seg));
        let len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len.saturating_sub(20))
            .unwrap();

        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        // Torn tail is NOT an error — replay completes.
        let _high = exec.run(reader).unwrap();
        let snap = exec.metrics().snapshot();
        // At least the first 2 bundles durable; third is torn
        // (or may survive partial tear — at minimum the 2-bundle
        // prefix must apply).
        assert!(
            snap.bundles_applied >= 2,
            "expected ≥2 bundles applied, got {}",
            snap.bundles_applied,
        );
        assert!(
            snap.bundles_applied <= 3,
            "bundles_applied={}",
            snap.bundles_applied,
        );
    }

    #[test]
    fn replay_gap_tolerance_preserves_expired_lsn() {
        // ADR-032 §R7: a gap in the commit_lsn sequence (torn-
        // dropped middle commit) MUST NOT corrupt the chain's
        // expired_lsn semantics. Synthesize a WAL with commits
        // {1, 3} — as if commit 2 was torn-dropped.
        let wal_dir = tempdir().unwrap();
        let mut mvcc1 = HashMap::new();
        mvcc1.insert(42u64, Some(bytes::Bytes::from_static(b"v1")));
        write_bundle_to_wal(wal_dir.path(), Lsn::new(1), &mvcc1, &[]);
        let mut mvcc3 = HashMap::new();
        mvcc3.insert(42u64, Some(bytes::Bytes::from_static(b"v3")));
        write_bundle_to_wal(wal_dir.path(), Lsn::new(3), &mvcc3, &[]);

        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let high = exec.run(reader).unwrap();
        assert_eq!(high, Lsn::new(3));
        // At snapshot=1 the value is v1 (not yet overwritten).
        let v_at_1 = txn_mgr.read_at(TenantId::DEFAULT, 42, Lsn::new(1));
        assert_eq!(v_at_1.as_deref(), Some(&b"v1"[..]));
        // At snapshot=2 the value is STILL v1 — the gap does not
        // magically overwrite at L+1.
        let v_at_2 = txn_mgr.read_at(TenantId::DEFAULT, 42, Lsn::new(2));
        assert_eq!(v_at_2.as_deref(), Some(&b"v1"[..]));
        // At snapshot=3 the value is v3.
        let v_at_3 = txn_mgr.read_at(TenantId::DEFAULT, 42, Lsn::new(3));
        assert_eq!(v_at_3.as_deref(), Some(&b"v3"[..]));
    }

    #[test]
    fn replay_orphan_page_invokes_bootstrap_from_mvcc() {
        // ADR-032 §Slice 3c: a legacy `IndexPage = 11` record
        // without a matching CommitBundle is an orphan. If a
        // bootstrap hook is registered, it's invoked with the
        // orphan count. Success ⇒ replay completes.
        use crate::primary_index::encode_index_page_payload;
        let wal_dir = tempdir().unwrap();
        // Emit one legacy IndexPage record directly via the WAL
        // writer (no CommitBundle following).
        {
            let cfg = WalConfig {
                dir: wal_dir.path().to_path_buf(),
                segment_size_bytes: 64 * 1024 * 1024,
                group_commit_window: std::time::Duration::from_millis(2),
                group_commit_max_batch: 4,
                metrics_sink: None,
                encryption: None,

                inflight_budget_bytes: None,
            };
            let writer = WalWriter::spawn(cfg).unwrap();
            let h = writer.handle();
            let page_bytes: [u8; PAGE_SIZE] = [0xCC; PAGE_SIZE];
            let payload =
                encode_index_page_payload(PageId::new(999), TenantId::DEFAULT, &page_bytes);
            h.append(
                WalRecordType::IndexPage,
                /*txn_id*/ 1,
                /*ts*/ 0,
                TenantId::DEFAULT,
                payload,
            )
            .unwrap();
            writer.shutdown().unwrap();
        }

        let invoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let invoked_clone = Arc::clone(&invoked);
        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary).with_bootstrap(move |n| {
            assert!(n >= 1, "expected ≥1 orphan, got {n}");
            invoked_clone.store(true, std::sync::atomic::Ordering::Release);
            Ok(())
        });
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let _ = exec.run(reader).unwrap();
        assert!(
            invoked.load(std::sync::atomic::Ordering::Acquire),
            "bootstrap_from_mvcc hook was not invoked"
        );
        let snap = exec.metrics().snapshot();
        assert_eq!(snap.orphan_pages_detected, 1);
        assert_eq!(snap.bootstrap_from_mvcc_invoked, 1);
    }

    #[test]
    fn replay_orphan_double_failure_halts_with_error() {
        // ADR-032 §Slice 3c: if the bootstrap hook returns an
        // error, the executor surfaces
        // `ArcGraphError::UnrecoverableOrphans` and halts.
        use crate::primary_index::encode_index_page_payload;
        let wal_dir = tempdir().unwrap();
        {
            let cfg = WalConfig {
                dir: wal_dir.path().to_path_buf(),
                segment_size_bytes: 64 * 1024 * 1024,
                group_commit_window: std::time::Duration::from_millis(2),
                group_commit_max_batch: 4,
                metrics_sink: None,
                encryption: None,

                inflight_budget_bytes: None,
            };
            let writer = WalWriter::spawn(cfg).unwrap();
            let h = writer.handle();
            let page_bytes: [u8; PAGE_SIZE] = [0xDD; PAGE_SIZE];
            let payload =
                encode_index_page_payload(PageId::new(777), TenantId::DEFAULT, &page_bytes);
            h.append(WalRecordType::IndexPage, 1, 0, TenantId::DEFAULT, payload)
                .unwrap();
            writer.shutdown().unwrap();
        }

        let txn_mgr = Arc::new(TxnManager::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary).with_bootstrap(|_| {
            Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: "stub bootstrap failure for test".to_owned(),
            })
        });
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let err = exec.run(reader).unwrap_err();
        match err {
            ArcGraphError::UnrecoverableOrphans {
                orphan_count,
                reason,
            } => {
                assert!(orphan_count >= 1);
                assert!(
                    reason.contains("stub bootstrap failure"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected UnrecoverableOrphans, got {other:?}"),
        }
    }

    #[test]
    fn replay_format_mismatch_halts_with_error() {
        // ADR-032 §6: an unknown `format_version` in the segment
        // header surfaces `WalFormatMismatch` before any record
        // decode. The existing `SegmentHeader::decode` already
        // does this; replay's job is to propagate the error.
        let wal_dir = tempdir().unwrap();
        // Write a segment with a bogus format_version (99) in
        // its header. The `WalRecoveryReader` will trip on the
        // header decode.
        let seg_path = wal_dir
            .path()
            .join(crate::wal::segment::segment_filename(0));
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::wal::WAL_SEGMENT_MAGIC);
        buf.extend_from_slice(&99u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        std::fs::write(&seg_path, &buf).unwrap();

        // The reader refuses to open mid-segment with unknown
        // version; verify the error surfaces.
        let reader_res = WalRecoveryReader::open(wal_dir.path());
        // `open` may succeed and the error surfaces at first
        // `next()`, OR the error surfaces during the initial
        // `advance_segment`. Either path is acceptable — the
        // contract is "no silent accept".
        match reader_res {
            Ok(mut reader) => {
                let first = reader.next();
                match first {
                    Some(Err(ArcGraphError::WalFormatMismatch { found_version, .. })) => {
                        assert_eq!(found_version, 99);
                    }
                    other => panic!("expected WalFormatMismatch on first next(), got {other:?}"),
                }
            }
            Err(ArcGraphError::WalFormatMismatch { found_version, .. }) => {
                assert_eq!(found_version, 99);
            }
            Err(other) => panic!("expected WalFormatMismatch on open, got {other:?}"),
        }
    }

    #[test]
    fn replay_idempotent_on_double_run() {
        // ADR-032 §9 M7. A second replay pass over the same WAL
        // produces identical state with zero new applies (all
        // bundles skipped as idempotent via Lemma I1).
        let wal_dir = tempdir().unwrap();
        for i in 1u64..=5 {
            let mut mvcc = HashMap::new();
            mvcc.insert(i, Some(bytes::Bytes::from(format!("v{i}"))));
            write_bundle_to_wal(wal_dir.path(), Lsn::new(i), &mvcc, &[]);
        }

        // First run: fresh TxnManager.
        let txn_mgr = Arc::new(TxnManager::new());
        let primary_store = Arc::new(PrimaryPageStore::new());
        let primary: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
        let target = PageStoreTarget::primary_only(primary);
        let mut exec = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target,
        );
        let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let first = exec.run(reader).unwrap();
        assert_eq!(first, Lsn::new(5));
        let first_snap = exec.metrics().snapshot();
        assert_eq!(first_snap.bundles_applied, 5);

        // Second run on the SAME TxnManager + page store.
        let primary2: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
        let target2 = PageStoreTarget::primary_only(primary2);
        let mut exec2 = ReplayExecutor::new(
            ReplayConfig::default_with_temp_spill(),
            Arc::clone(&txn_mgr),
            target2,
        );
        let reader2 = WalRecoveryReader::open(wal_dir.path()).unwrap();
        let second = exec2.run(reader2).unwrap();
        assert_eq!(second, Lsn::new(5));
        // Every bundle is skipped as idempotent.
        let second_snap = exec2.metrics().snapshot();
        assert_eq!(second_snap.bundles_applied, 0);
        // O-M (W28-S3): deterministic fixture — exactly 5 bundles
        // (LSN 1..=5) were applied on the first run, so the idempotent
        // second run must skip exactly 5. Was `>= 5`, which could not
        // catch a replay that double-counted skips or under-skipped.
        assert_eq!(second_snap.bundles_skipped_idempotent, 5);

        // State is byte-identical: every key visible with the
        // same value.
        for i in 1u64..=5 {
            let expected = format!("v{i}");
            let got = txn_mgr.read_at(TenantId::DEFAULT, i, Lsn::new(5));
            assert_eq!(got.as_deref(), Some(expected.as_bytes()));
        }
    }

    // ─── M3.a Slice G.1: Vector dispatch wiring ─────────────────────

    /// Test-only [`VectorPageStoreHandle`] that records every call
    /// it receives so the dispatch test can assert routing without
    /// touching real arena machinery (Slice G.2 owns that).
    struct MockVectorStore {
        installs: std::sync::Mutex<Vec<(TenantId, PageId, Vec<u8>)>>,
    }

    impl MockVectorStore {
        fn new() -> Self {
            Self {
                installs: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn install_count(&self) -> usize {
            self.installs.lock().unwrap().len()
        }

        fn last_install(&self) -> Option<(TenantId, PageId, Vec<u8>)> {
            self.installs.lock().unwrap().last().cloned()
        }
    }

    impl VectorPageStoreHandle for MockVectorStore {
        fn install_or_replace(
            &self,
            tenant: TenantId,
            page_id: PageId,
            bytes: &[u8],
        ) -> std::result::Result<(), crate::vector_store::VectorStoreError> {
            self.installs
                .lock()
                .unwrap()
                .push((tenant, page_id, bytes.to_vec()));
            Ok(())
        }

        fn restore_page_bytes(
            &self,
            _tenant: TenantId,
            _page_id: PageId,
            _bytes: &[u8],
        ) -> std::result::Result<(), crate::vector_store::VectorStoreError> {
            // Slice G.5 owns the rollback path; this mock only
            // records the replay-side install path.
            Ok(())
        }
    }

    /// Dispatching a `BundlePageKind::Vector` entry routes the bytes
    /// into the wired `VectorPageStoreHandle`. Pins the dispatch arm
    /// added in Slice G.1 — without this test, a future refactor of
    /// `install_index_page` could silently drop the Vector arm and
    /// no integration test would catch it (G.2/G.3 don't land until
    /// later).
    #[test]
    fn replay_dispatches_vector_kind_when_store_wired() {
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let mock = Arc::new(MockVectorStore::new());
        let mock_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&mock) as _;
        let target = PageStoreTarget::primary_only(primary).with_vector_store(mock_handle);

        let entry = crate::wal::bundle::DecodedStagedPage {
            kind: BundlePageKind::Vector,
            page_id: PageId::new(7),
            tenant_id: TenantId::DEFAULT,
            bytes: mk_page(0xEE),
        };

        let routed = target.install_index_page(&entry).unwrap();
        assert_eq!(routed, BundlePageKind::Vector);
        assert_eq!(mock.install_count(), 1);
        let (tenant, page_id, bytes) = mock.last_install().expect("one install recorded");
        assert_eq!(tenant, TenantId::DEFAULT);
        assert_eq!(page_id, PageId::new(7));
        assert_eq!(bytes.len(), PAGE_SIZE);
        assert!(bytes.iter().all(|b| *b == 0xEE));
    }

    /// Without a wired vector store the dispatch arm stays
    /// permissive (warn-and-continue) so pre-M3.a deployments do
    /// not regress. This pins the deliberate G.1 stub posture; the
    /// reject-as-wiring-bug behaviour of `Blob` / `Record` is what
    /// G.2 will adopt once production WALs can carry Vector
    /// entries.
    #[test]
    fn replay_dispatch_vector_kind_warns_and_continues_when_no_store_wired() {
        let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
        let target = PageStoreTarget::primary_only(primary);

        let entry = crate::wal::bundle::DecodedStagedPage {
            kind: BundlePageKind::Vector,
            page_id: PageId::new(13),
            tenant_id: TenantId::DEFAULT,
            bytes: mk_page(0x42),
        };

        let routed = target.install_index_page(&entry).unwrap();
        assert_eq!(routed, BundlePageKind::Vector);
        // No panic, no error: the arm logs and returns Ok so replay
        // forward-progresses through the entry.
    }
}
