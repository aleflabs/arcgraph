//! WAL writer thread with group commit (roadmap M1-31 + M1-32;
//! extended by ADR-034 §Slice B with per-tier dispatch).
//!
//! One dedicated OS thread owns the open segment, a monotonic LSN
//! counter, and a batch of pending records. Producers choose one of
//! two entrypoints:
//!
//! - [`WalHandle::append`] — **T1 / Strict** — blocks until the record
//!   is durable (fsynced). Returns the assigned LSN.
//! - [`WalHandle::append_async`] — **T3 / Periodic** (ADR-034 D-4) —
//!   returns as soon as the bytes enter the pending batch. Durability
//!   is provided by the group-commit timer, by a piggybacking T1
//!   commit, or by an explicit [`WalHandle::flush`] call from
//!   `BackgroundFsyncScheduler`. Returns `(Lsn, wal_offset)`.
//!
//! **Group commit semantics** (design-v2 §4.2):
//!
//! The writer fires the batch when any of the following hits first:
//! (a) the pending batch reaches `group_commit_max_batch` records
//! (default 16); (b) `group_commit_window` has elapsed since the
//! first record of the current batch landed (default 1 ms); (c) a
//! [`WalHandle::flush`] call arrives.
//!
//! A "fire" writes the concatenated batch to the active segment,
//! fdatasyncs, and then signals every sync-ack channel. At that
//! point `append()` returns `Ok(lsn)` to each caller — the durability
//! contract is that the record is on disk when `append` returns.
//! Async callers were already notified at enqueue time (ADR-034 D-4);
//! they observe durability through `committed_fsync_watermark`.
//!
//! **ADR-034 D-5 piggyback.** A T1 commit's fire flushes the entire
//! pending batch. Any T3 commits enqueued since the last fire are
//! also on disk after the T1 caller's ack returns, regardless of
//! their own `rpo_ms`. This is a correctness-preserving property
//! (invariant I-D3), not an implementation detail.
//!
//! **ADR-034 §6.2 escalation.** If a fire's fsync fails AND the
//! pending batch contains any T3 (async-ackd) records, the writer
//! **calls `std::process::abort()` directly**. Rationale: those T3
//! commits were already ack'd; a rollback is incoherent post-ack;
//! continuing with corrupt durability is worse than abort. See
//! ADR-034 §6.2 and §I-D4.
//!
//! **Failure contract (non-§6.2 cases):** a fatal I/O error on write
//! or fsync fails every SYNC-ack pending with the underlying
//! `std::io::Error` and exits the writer thread. Subsequent `append`
//! / `flush` calls see [`ArcGraphError::WalUnavailable`] because the
//! channel is disconnected.
//!
//! WAL recovery on startup (M1-34) will populate the LSN counter so
//! newly appended records continue the sequence. For now the writer
//! starts each fresh instance at LSN 1.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arcgraph_core::{ArcGraphError, Lsn, Result, TenantId};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use dashmap::DashSet;
use parking_lot::{Condvar, Mutex};
use tracing::error;

use crate::encryption::{WAL_PAYLOAD_HEADER_LEN, WalEncryption};
use crate::metrics::{MetricsSink, WalWriteOutcome};
use crate::wal::record::{WalRecord, WalRecordType};
use crate::wal::segment::SegmentWriter;

// ---------- configuration ---------------------------------------------------

/// Configuration for a WAL writer.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory containing segment files. Created if it does not exist.
    pub dir: PathBuf,
    /// Target maximum bytes per segment. Rotates before exceeding.
    pub segment_size_bytes: u64,
    /// Group-commit window. Default 1 ms per design-v2 §4.2.
    pub group_commit_window: Duration,
    /// Group-commit batch size. Default 16 per design-v2 §4.2.
    pub group_commit_max_batch: usize,
    /// W16γ M6-07 — optional observability sink (ADR-045 draft).
    /// When `Some`, the writer thread emits per-append counters
    /// (`wal_writes_total{outcome}`) and per-fire fsync duration
    /// observations (`wal_fsync_duration_ms`) into the sink. When
    /// `None`, the legacy zero-overhead path runs — the producer
    /// pays only one nullable-ptr check per event.
    ///
    /// `Option<Arc<dyn MetricsSink>>` rather than a generic so that
    /// `WalConfig` stays `Clone` + `Debug` without parameterizing
    /// every downstream constructor; trait-object dispatch is the
    /// right shape per ADR-045 §"Alternatives considered (b)".
    pub metrics_sink: Option<Arc<dyn MetricsSink>>,
    /// W20β-3 / ADR-052: optional WAL payload encryption. When
    /// `Some`, every record's payload is wrapped in a 36-byte AEAD
    /// header + AES-256-GCM ciphertext at encode time. When `None`
    /// (the v0.1.0-alpha.0 default), the writer emits clear payloads
    /// as today. Mixed clear + encrypted WAL records are supported
    /// across recovery — see `WalEncryption::decrypt`'s magic-peek.
    pub encryption: Option<WalEncryption>,
    /// M6.1 / `docs/design/storage-architecture-v2.md` §6.1 — bounded
    /// WAL admission. `None` preserves the pre-M6 unbounded-channel
    /// behavior (the writer channel is a `crossbeam::unbounded()`,
    /// measured queue depth 0 under WAL-enabled ingest — NOT the OOM,
    /// per §6.1). `Some(bytes)` installs a byte-budget gate: a producer's
    /// `append`/`append_async` blocks in the CALLER's thread (never the
    /// writer thread) while `in_flight_bytes > wal_inflight_budget_bytes`,
    /// admitting as soon as an earlier append's ack releases its budget.
    /// Consistency-neutral by construction (§6.1): this only delays a
    /// producer's *enqueue*, one stage before Phase 2 → Phase 3; ordering,
    /// fsync-before-ack, and OCC are untouched.
    pub inflight_budget_bytes: Option<u64>,
}

impl WalConfig {
    /// Fresh config rooted at `dir` with design-v2 defaults.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_millis(1),
            group_commit_max_batch: 16,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        }
    }

    /// Builder-pattern: attach a [`MetricsSink`] for W16γ M6-07
    /// per-WAL observability (`wal_writes_total{outcome}` counter +
    /// `wal_fsync_duration_ms` histogram).
    ///
    /// Per ADR-045: storage producers stay in `arcgraph-storage`;
    /// the sink's concrete impl lives upstream (`arcgraph-mcp`'s
    /// `MetricsRegistry`). Legacy callers leave this `None`.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// Builder-pattern: enable WAL payload encryption (W20β-3 /
    /// ADR-052). The provided [`WalEncryption`] encapsulates the
    /// keyring + current key version; the writer encrypts every
    /// record's payload at encode time. Recovery + tests reuse the
    /// same encryption config so historical key versions remain
    /// readable.
    #[must_use]
    pub fn with_encryption(mut self, encryption: WalEncryption) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Builder-pattern: install the M6.1 bounded-admission byte budget
    /// (design-v2 §6.1). `budget_bytes` must be > 0; a `debug_assert`
    /// catches a misconfigured 0 budget in tests without adding a
    /// release-mode branch to the hot append path.
    #[must_use]
    pub fn with_inflight_budget_bytes(mut self, budget_bytes: u64) -> Self {
        debug_assert!(budget_bytes > 0, "WAL inflight budget must be > 0");
        self.inflight_budget_bytes = Some(budget_bytes);
        self
    }
}

// ---------- bounded WAL admission (M6.1 / design-v2 §6.1) -------------------

/// Shared in-flight-bytes gate. `admit` blocks the CALLER's thread (never
/// the writer thread) until `in_flight + len <= budget` (or the budget is
/// `None`, the pre-M6 unbounded posture), then reserves `len` bytes;
/// `release` gives them back. Consistency-neutral by construction: this
/// only delays when a producer's bytes are considered "sent", one stage
/// before the WAL command reaches the writer's channel — fsync-before-ack,
/// ordering, and OCC are all downstream of this gate and untouched.
///
/// A single append whose OWN encoded length exceeds the configured budget
/// is still admitted (blocking forever on an unsatisfiable request would
/// be a liveness bug, MECH-E8's back-pressure-never-deadlock lesson
/// applied to the WAL channel) — it admits alone once `in_flight` drains
/// to 0, i.e. the budget becomes "at least one in-flight append" rather
/// than a hard byte ceiling in that pathological case.
///
/// **#1521 M6.1 P1-4 — poison-on-writer-death.** `admit`'s reservation is
/// released by the WRITER THREAD inside `fire()` once the corresponding
/// append is durable (see `append_inner`'s doc comment). If the writer
/// thread dies (panics, or exits via `Shutdown`) WHILE a command it
/// already dequeued is still in flight — i.e. AFTER a caller's `admit()`
/// succeeded and the command was sent (so the caller-side "send failed,
/// release here" path in `append_inner`/`append_async_inner` does NOT
/// fire), NOBODY ever calls `release` for those bytes: they are stranded
/// in the budget forever, and any OTHER caller concurrently blocked in
/// `admit`'s `Condvar::wait` for room would wait FOREVER (no `notify_all`
/// is coming). `poison` (called once, from `shutdown`/`Drop`, on every
/// writer-thread exit path — normal AND panicking) sets `dead`, releases
/// every currently-stranded reservation back to 0, and wakes every
/// waiter; `admit`'s wait loop checks `dead` on every wake and returns
/// `Err(())` (surfaced as [`ArcGraphError::WalUnavailable`]) instead of
/// re-blocking, so a waiter never spins on a budget that can no longer
/// ever be released by anyone.
#[derive(Debug)]
struct WalByteBudget {
    budget: u64,
    state: Mutex<u64>,
    room: Condvar,
    dead: AtomicBool,
}

impl WalByteBudget {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            state: Mutex::new(0),
            room: Condvar::new(),
            dead: AtomicBool::new(false),
        }
    }

    /// Block until `len` bytes are admitted, then reserve them. Returns
    /// `Err(())` if the budget was poisoned (writer thread gone) either
    /// before this call started or while it was waiting — the caller
    /// must not treat a "woke up" event as "room became available"
    /// without re-checking `dead` first (poison and room-release both
    /// go through the same `notify_all`).
    fn admit(&self, len: u64) -> std::result::Result<(), ()> {
        let mut in_flight = self.state.lock();
        loop {
            if self.dead.load(Ordering::Acquire) {
                return Err(());
            }
            // Admit if there is room, OR nothing else is in flight (the
            // single-oversized-append escape hatch above).
            if in_flight.saturating_add(len) <= self.budget || *in_flight == 0 {
                *in_flight += len;
                return Ok(());
            }
            self.room.wait(&mut in_flight);
        }
    }

    /// Poison the budget: mark it dead, drain every stranded reservation
    /// to 0 (nothing will ever call `release` for bytes a dead writer
    /// thread was holding), and wake every waiter so they observe `dead`
    /// and return `Err(())` instead of blocking forever. Idempotent —
    /// safe to call from both `shutdown`'s normal-exit path and `Drop`'s
    /// fallback path without double-poisoning causing harm.
    fn poison(&self) {
        let mut in_flight = self.state.lock();
        self.dead.store(true, Ordering::Release);
        *in_flight = 0;
        drop(in_flight);
        self.room.notify_all();
    }

    /// Release `len` previously-admitted bytes, waking any waiters.
    fn release(&self, len: u64) {
        let mut in_flight = self.state.lock();
        *in_flight = in_flight.saturating_sub(len);
        drop(in_flight);
        self.room.notify_all();
    }

    /// Observability: bytes currently admitted but not yet released.
    fn in_flight(&self) -> u64 {
        *self.state.lock()
    }
}

// ---------- public API ------------------------------------------------------

/// Cheap-cloneable handle for producers. `append` and `flush` both
/// block until the writer thread acknowledges the request is durable.
///
/// [`append_async`](WalHandle::append_async) (ADR-034 §Slice B) does
/// NOT block on fsync — see the method rustdoc for the T3 contract.
#[derive(Clone, Debug)]
pub struct WalHandle {
    sender: Sender<WalCmd>,
    committed_fsync_watermark: Arc<AtomicU64>,
    format_version: u16,
    exact_durable_lsns: Arc<DashSet<u64>>,
    /// SVC-1 P2 / #849 / ADR-229 — monotonic total of record-data bytes
    /// durably fired across this writer's lifetime. Shared with the writer
    /// thread (bumped per successful fire alongside `bytes_fsynced_total`).
    /// The checkpoint byte-trigger reads this to compute
    /// "WAL bytes since last checkpoint" WITHOUT touching the writer thread's
    /// owned `SegmentWriter`. Resets to 0 on each process (fresh writer), so
    /// it is a within-process gauge; the trigger only ever compares it
    /// against a within-process baseline, so the per-process reset is
    /// correct (see `bootstrap::DurableCheckpointer`).
    wal_bytes_appended: Arc<AtomicU64>,
    /// M6.1 — `Some` iff `WalConfig::inflight_budget_bytes` was set.
    inflight_budget: Option<Arc<WalByteBudget>>,
}

impl WalHandle {
    /// Bundle format required by the active WAL generation.
    #[must_use]
    pub fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Consume proof that this exact assigned LSN completed fsync. Unlike the
    /// max watermark, this remains sound when a higher-LSN bundle reaches disk
    /// before a lower-LSN bundle.
    pub fn take_exact_durable(&self, lsn: Lsn) -> bool {
        self.exact_durable_lsns.remove(&lsn.raw()).is_some()
    }

    /// Observe without consuming the exact durability proof. Deferred v9
    /// apply drains use this to keep the proof intact until page apply has
    /// succeeded; a failed apply must remain retryable and checkpoint-blocking.
    pub(crate) fn has_exact_durable(&self, lsn: Lsn) -> bool {
        self.exact_durable_lsns.contains(&lsn.raw())
    }

    #[cfg(test)]
    pub(crate) fn __test_mark_exact_durable(&self, lsn: Lsn) {
        self.exact_durable_lsns.insert(lsn.raw());
        self.committed_fsync_watermark
            .fetch_max(lsn.raw(), Ordering::AcqRel);
    }

    /// **T1 / Strict** — append a record and BLOCK until the record is
    /// fsynced. Returns the LSN assigned by the writer.
    ///
    /// This is the pre-ADR-034 path; unchanged. Used for every commit
    /// on a [`arcgraph_core::DurabilityTier::Strict`] tenant.
    pub fn append(
        &self,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<Lsn> {
        self.append_inner(None, record_type, txn_id, timestamp_ms, tenant_id, payload)
    }

    /// Append using an LSN range-end already allocated by the transaction
    /// manager. v9 bundles use this so the record header, bundle commit LSN,
    /// and durability watermark share one monotone redo clock.
    #[allow(clippy::too_many_arguments)] // existing append fields plus assigned LSN.
    pub fn append_at(
        &self,
        lsn: Lsn,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<Lsn> {
        if lsn == Lsn::ZERO {
            return Err(ArcGraphError::WalCorruption {
                lsn,
                reason: "assigned WAL LSN must be non-zero".to_owned(),
            });
        }
        self.append_inner(
            Some(lsn),
            record_type,
            txn_id,
            timestamp_ms,
            tenant_id,
            payload,
        )
    }

    #[allow(clippy::too_many_arguments)] // mirrors WalCmd::Append wire metadata.
    fn append_inner(
        &self,
        assigned_lsn: Option<Lsn>,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<Lsn> {
        // M6.1 §6.1: admit BEFORE the send — blocks the CALLER's thread
        // only, never the writer thread. `None` (no budget configured)
        // is the legacy zero-overhead unbounded posture. Released by the
        // WRITER THREAD inside `fire()` once the bytes are durable (or
        // definitively abandoned on a hard error) — see `fire`'s budget
        // release — so the budget bounds "bytes not yet durable", the
        // exact quantity design-v2 §6.1 names, uniformly for T1 and T3.
        let budget_len = payload.len() as u64;
        if let Some(budget) = self.inflight_budget.as_ref() {
            // #1521 M6.1 P1-4 — the budget may already be poisoned (an
            // earlier writer-thread death) by the time this caller reaches
            // `admit`; surface `WalUnavailable` immediately rather than
            // ever entering the wait loop for a budget nobody can release.
            budget
                .admit(budget_len)
                .map_err(|()| ArcGraphError::WalUnavailable)?;
        }
        let (ack_tx, ack_rx) = bounded(1);
        let send_res = self.sender.send(WalCmd::Append {
            assigned_lsn,
            record_type,
            txn_id,
            timestamp_ms,
            tenant_id,
            payload,
            budget_len,
            ack: ack_tx,
        });
        if send_res.is_err() {
            // Writer thread is gone; nobody will ever release this
            // reservation from `fire()` — release it here so the budget
            // does not leak on writer-thread death.
            if let Some(budget) = self.inflight_budget.as_ref() {
                budget.release(budget_len);
            }
            return Err(ArcGraphError::WalUnavailable);
        }
        ack_rx.recv().map_err(|_| ArcGraphError::WalUnavailable)?
    }

    /// **T3 / Periodic** (ADR-034 §Slice B) — append a record and
    /// return as soon as the bytes are in the writer's pending batch.
    /// Durability is provided asynchronously by the group-commit timer,
    /// a piggybacking T1 commit, or an explicit
    /// [`BackgroundFsyncScheduler`](crate::wal::BackgroundFsyncScheduler)
    /// [`flush`](WalHandle::flush) call.
    ///
    /// Returns `(lsn, wal_offset)` — the assigned WAL LSN and the
    /// post-enqueue pending-buffer offset. `wal_offset` is the
    /// running total of encoded bytes accepted by the writer across
    /// all appends (sync + async) since writer start; it is monotonic
    /// and useful to the scheduler for "how much work is pending"
    /// observability.
    ///
    /// **Durability contract** (ADR-034 I-D2): the caller MAY see
    /// `Ok((lsn, offset))` before the bytes are on durable disk.
    /// After the configured `rpo_ms`, either the bytes ARE durable
    /// OR the process has aborted.
    ///
    /// **Visibility contract** (ADR-034 I-D3): if any subsequent
    /// `append` (sync) call returns `Ok(_)` while this record is
    /// still in the pending batch, this record is also durable at
    /// that point (piggyback).
    ///
    /// **Failure contract**: this call returns
    /// [`ArcGraphError::WalUnavailable`] iff the writer thread is
    /// gone before the bytes were accepted. A fire-time fsync
    /// failure is NOT surfaced to async callers — it escalates to
    /// `std::process::abort()` per ADR-034 §6.2.
    pub fn append_async(
        &self,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<(Lsn, u64)> {
        self.append_async_inner(None, record_type, txn_id, timestamp_ms, tenant_id, payload)
    }

    /// Async counterpart to [`Self::append_at`]. The assigned LSN is
    /// acknowledged at enqueue time; durability still follows the T3
    /// contract.
    #[allow(clippy::too_many_arguments)] // existing async fields plus assigned LSN.
    pub fn append_async_at(
        &self,
        lsn: Lsn,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<(Lsn, u64)> {
        if lsn == Lsn::ZERO {
            return Err(ArcGraphError::WalCorruption {
                lsn,
                reason: "assigned WAL LSN must be non-zero".to_owned(),
            });
        }
        self.append_async_inner(
            Some(lsn),
            record_type,
            txn_id,
            timestamp_ms,
            tenant_id,
            payload,
        )
    }

    #[allow(clippy::too_many_arguments)] // mirrors WalCmd::AppendAsync metadata.
    fn append_async_inner(
        &self,
        assigned_lsn: Option<Lsn>,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> Result<(Lsn, u64)> {
        // See `append_inner`'s comment: admitted here (caller thread),
        // released in `fire()` once durable — bounds "bytes not yet
        // durable" uniformly across T1/T3.
        let budget_len = payload.len() as u64;
        if let Some(budget) = self.inflight_budget.as_ref() {
            // #1521 M6.1 P1-4 — see `append_inner`'s comment on the same
            // poison-check.
            budget
                .admit(budget_len)
                .map_err(|()| ArcGraphError::WalUnavailable)?;
        }
        let (ack_tx, ack_rx) = bounded(1);
        let send_res = self.sender.send(WalCmd::AppendAsync {
            assigned_lsn,
            record_type,
            txn_id,
            timestamp_ms,
            tenant_id,
            payload,
            budget_len,
            ack: ack_tx,
        });
        if send_res.is_err() {
            if let Some(budget) = self.inflight_budget.as_ref() {
                budget.release(budget_len);
            }
            return Err(ArcGraphError::WalUnavailable);
        }
        ack_rx.recv().map_err(|_| ArcGraphError::WalUnavailable)?
    }

    /// Force a group-commit fire now. Blocks until fsync completes.
    ///
    /// ADR-034 §Slice C: the background fsync scheduler calls this
    /// once per tick to durify any pending async appends.
    pub fn flush(&self) -> Result<()> {
        let (ack_tx, ack_rx) = bounded(1);
        self.sender
            .send(WalCmd::Flush { ack: ack_tx })
            .map_err(|_| ArcGraphError::WalUnavailable)?;
        ack_rx.recv().map_err(|_| ArcGraphError::WalUnavailable)?
    }

    /// ADR-034 §Slice B — committed-fsync watermark.
    ///
    /// Returns the highest WAL LSN known to be durable on disk. The
    /// watermark advances monotonically after each successful
    /// group-commit fire. Async (T3) producers observe their commit's
    /// durability by polling this against their `Lsn`.
    ///
    /// **Not a producer-blocking API.** Callers who need to WAIT for
    /// durability call [`Self::flush`] instead.
    ///
    /// Cheap to clone (a `Arc<AtomicU64>`); shared with the
    /// [`BackgroundFsyncScheduler`](crate::wal::BackgroundFsyncScheduler).
    #[must_use]
    pub fn committed_fsync_watermark(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.committed_fsync_watermark)
    }

    /// Read the current value of the committed-fsync watermark.
    /// Convenience wrapper around [`Self::committed_fsync_watermark`]
    /// for callers that don't want to hold the `Arc`.
    #[inline]
    #[must_use]
    pub fn last_durable_lsn(&self) -> Lsn {
        Lsn::new(self.committed_fsync_watermark.load(Ordering::Acquire))
    }

    /// SVC-1 P2 / #849 / ADR-229 — total record-data bytes durably fired
    /// across this writer's lifetime (this process). The checkpoint
    /// byte-trigger reads this to decide when the WAL-since-checkpoint has
    /// grown past [`WalCheckpointConfig::interval_bytes`](crate::config::WalCheckpointConfig).
    ///
    /// Monotonic within a process; resets to 0 on a fresh writer (post
    /// restart). NOT a cross-restart total — the byte-trigger compares it
    /// only against a within-process baseline (bytes-at-last-checkpoint), so
    /// the per-process reset is correct.
    #[inline]
    #[must_use]
    pub fn wal_bytes_appended(&self) -> u64 {
        self.wal_bytes_appended.load(Ordering::Acquire)
    }
}

/// Lightweight, Send/Sync snapshot of the WAL writer's group-commit
/// fire telemetry. Intended for test and bench instrumentation —
/// `total_fires` and `total_records_fired` together let a caller
/// compute the mean batch-size-at-fire, which is the load-bearing
/// evidence that multiple concurrent `wal.append` callers are
/// pipelining into a single fsync (vs. serializing through a gate
/// held-across-fsync).
///
/// Cheap to clone (now five `Arc<AtomicU64>`; ADR-034 §Slice B added
/// the tier-specific append counters and the async-batch-abort
/// counter). Values are monotonic over the lifetime of the owning
/// [`WalWriter`]. Zero overhead on the hot path (one relaxed atomic
/// add per fire / per append).
#[derive(Clone, Debug, Default)]
pub struct WalFireMetrics {
    total_fires: Arc<AtomicU64>,
    total_records_fired: Arc<AtomicU64>,
    /// ADR-034 §Slice B — count of [`WalHandle::append`] calls
    /// accepted by the writer. Corresponds 1:1 with T1 commit
    /// record emissions (one append per T1 commit).
    wal_t1_appends_total: Arc<AtomicU64>,
    /// ADR-034 §Slice B — count of [`WalHandle::append_async`] calls
    /// accepted by the writer. Corresponds 1:1 with T3 commit
    /// record emissions (one async append per T3 commit).
    wal_t3_appends_total: Arc<AtomicU64>,
    /// ADR-034 §Slice B — total bytes fsynced across all successful
    /// fires. `wal_offset` monotonic; useful to derive mean
    /// bytes-per-fire and total-durable-bytes gauges.
    bytes_fsynced_total: Arc<AtomicU64>,
}

impl WalFireMetrics {
    /// Number of `fire()` invocations since writer spawn, including
    /// empty fires (window-timeout with empty batch, or
    /// flush-on-empty-batch).
    #[must_use]
    pub fn total_fires(&self) -> u64 {
        self.total_fires.load(Ordering::Acquire)
    }

    /// Number of append records durable across all fires. Divide by
    /// `total_fires()` for the mean batch-size-at-fire. Empty fires
    /// contribute zero to this counter.
    #[must_use]
    pub fn total_records_fired(&self) -> u64 {
        self.total_records_fired.load(Ordering::Acquire)
    }

    /// ADR-034 §Slice B — T1 (sync) append count.
    #[must_use]
    pub fn wal_t1_appends_total(&self) -> u64 {
        self.wal_t1_appends_total.load(Ordering::Acquire)
    }

    /// ADR-034 §Slice B — T3 (async) append count.
    #[must_use]
    pub fn wal_t3_appends_total(&self) -> u64 {
        self.wal_t3_appends_total.load(Ordering::Acquire)
    }

    /// ADR-034 §Slice B — total bytes across all successful fsyncs.
    #[must_use]
    pub fn bytes_fsynced_total(&self) -> u64 {
        self.bytes_fsynced_total.load(Ordering::Acquire)
    }
}

/// Owns the WAL writer thread. Dropping without explicit `shutdown`
/// still signals the thread and joins on drop (best-effort).
pub struct WalWriter {
    sender: Sender<WalCmd>,
    thread: Option<JoinHandle<Result<()>>>,
    fire_metrics: WalFireMetrics,
    committed_fsync_watermark: Arc<AtomicU64>,
    /// SVC-1 P2 / #849 / ADR-229 — see [`WalHandle::wal_bytes_appended`].
    wal_bytes_appended: Arc<AtomicU64>,
    format_version: u16,
    exact_durable_lsns: Arc<DashSet<u64>>,
    /// M6.1 — `Some` iff `WalConfig::inflight_budget_bytes` was set.
    inflight_budget: Option<Arc<WalByteBudget>>,
}

impl WalWriter {
    /// Spawn a writer thread on the given config. The first assigned
    /// LSN is 1. For replay-aware startup after crash recovery, use
    /// [`Self::spawn_from`] with the highest LSN returned by
    /// [`crate::wal::WalRecoveryReader::last_lsn`].
    pub fn spawn(config: WalConfig) -> Result<Self> {
        Self::spawn_from(config, Lsn::ZERO)
    }

    /// Spawn a writer thread continuing the LSN sequence from
    /// `initial_lsn`. The first assigned LSN is `initial_lsn + 1`.
    ///
    /// ADR-034 §Slice B: `committed_fsync_watermark` is seeded to
    /// `initial_lsn` so the post-replay watermark reflects the
    /// recovered durable prefix. Fresh spawn (initial_lsn = ZERO)
    /// starts the watermark at 0.
    pub fn spawn_from(config: WalConfig, initial_lsn: Lsn) -> Result<Self> {
        Self::spawn_from_inner(config, initial_lsn, false)
    }

    /// #1521 M6.1 P1-4 — test-only seam: spawn a writer thread that
    /// deliberately PANICS immediately after the first `Append` or
    /// `AppendAsync` command is dequeued and pushed onto its internal
    /// `pending` batch (i.e. AFTER a caller's `admit()` reservation was
    /// consumed and the command left the channel, but BEFORE any `fire()`
    /// call — `fire()` is the sole budget-release point, per
    /// `fire()`'s "release the whole batch's budget reservation up
    /// front" comment — so this deterministically reproduces "writer
    /// thread died with a reservation stranded" without relying on OS
    /// scheduling luck or an unrelated I/O fault. `#[doc(hidden)]`
    /// exactly like `page_store.rs`'s `try_evict_page_pinned_with_hook_for_gate`
    /// test seam — never reachable from the production `spawn`/
    /// `spawn_from` entry points.
    #[doc(hidden)]
    pub fn spawn_from_with_panic_after_first_pending_for_gate(
        config: WalConfig,
        initial_lsn: Lsn,
    ) -> Result<Self> {
        Self::spawn_from_inner(config, initial_lsn, true)
    }

    fn spawn_from_inner(
        config: WalConfig,
        initial_lsn: Lsn,
        panic_after_first_pending_for_gate: bool,
    ) -> Result<Self> {
        // Pre-open the segment on the caller's thread so `spawn`
        // surfaces any dir-creation / permissions error synchronously.
        let segment = SegmentWriter::open(&config.dir, config.segment_size_bytes)?;
        let format_version = segment.format_version();
        let (sender, receiver) = unbounded();
        let start_counter = initial_lsn.raw();
        let fire_metrics = WalFireMetrics::default();
        let fire_metrics_for_thread = fire_metrics.clone();
        let committed_fsync_watermark = Arc::new(AtomicU64::new(initial_lsn.raw()));
        let watermark_for_thread = Arc::clone(&committed_fsync_watermark);
        // SVC-1 P2: fresh at 0 — a within-process byte gauge (see the field
        // rustdoc). Never seeded from `initial_lsn` (that's an LSN, not a
        // byte count).
        let wal_bytes_appended = Arc::new(AtomicU64::new(0));
        let wal_bytes_for_thread = Arc::clone(&wal_bytes_appended);
        let metrics_sink_for_thread = config.metrics_sink.clone();
        let encryption_for_thread = config.encryption.clone();
        let exact_durable_lsns = Arc::new(DashSet::new());
        let exact_durable_for_thread = Arc::clone(&exact_durable_lsns);
        let inflight_budget = config
            .inflight_budget_bytes
            .map(|bytes| Arc::new(WalByteBudget::new(bytes)));
        let inflight_budget_for_thread = inflight_budget.clone();
        let thread = thread::Builder::new()
            .name("arcgraph-wal-writer".to_owned())
            .spawn(move || {
                // #1521 M6.1 P1-4 — poison-on-death RAII guard. Rust runs
                // local `Drop` impls during a panicking unwind (this
                // process does not build with `panic = "abort"`), so this
                // guard's `Drop::drop` fires on EVERY exit from the
                // closure below — a normal `Ok(())`/`Err(_)` return AND a
                // panic — making poison-on-death unconditional and
                // immediate (not deferred until some later, arbitrary
                // `WalWriter::drop`/`shutdown` call on the OWNING side,
                // which could run long after other callers are already
                // blocked in `admit`). `shutdown`/`Drop`'s OWN `poison()`
                // calls remain as a defense-in-depth belt (idempotent —
                // poisoning an already-poisoned or already-drained budget
                // is a no-op) for the case where the process holding a
                // `WalHandle` never joins the writer at all.
                struct PoisonOnDrop(Option<Arc<WalByteBudget>>);
                impl Drop for PoisonOnDrop {
                    fn drop(&mut self) {
                        if let Some(budget) = self.0.as_ref() {
                            budget.poison();
                        }
                    }
                }
                let _poison_guard = PoisonOnDrop(inflight_budget_for_thread.clone());
                run(
                    segment,
                    receiver,
                    config,
                    start_counter,
                    fire_metrics_for_thread,
                    watermark_for_thread,
                    wal_bytes_for_thread,
                    exact_durable_for_thread,
                    metrics_sink_for_thread,
                    encryption_for_thread,
                    inflight_budget_for_thread,
                    panic_after_first_pending_for_gate,
                )
            })
            .map_err(ArcGraphError::Io)?;
        Ok(Self {
            sender,
            thread: Some(thread),
            fire_metrics,
            committed_fsync_watermark,
            wal_bytes_appended,
            format_version,
            exact_durable_lsns,
            inflight_budget,
        })
    }

    /// A cloneable producer handle.
    #[must_use]
    pub fn handle(&self) -> WalHandle {
        WalHandle {
            sender: self.sender.clone(),
            committed_fsync_watermark: Arc::clone(&self.committed_fsync_watermark),
            format_version: self.format_version,
            exact_durable_lsns: Arc::clone(&self.exact_durable_lsns),
            wal_bytes_appended: Arc::clone(&self.wal_bytes_appended),
            inflight_budget: self.inflight_budget.clone(),
        }
    }

    /// M6.1 observability: current in-flight (admitted, not-yet-durable)
    /// WAL bytes under the configured budget gate. `0` when no budget is
    /// configured. Always compiled (not `cfg(test)`-gated): a bounded
    /// admission gate whose observability only exists in test builds
    /// would be unauditable in production — this is a cheap `AtomicU64`
    /// (well, `Mutex<u64>`) load, not a debug assertion.
    #[must_use]
    pub fn inflight_budget_bytes_in_use(&self) -> u64 {
        self.inflight_budget
            .as_ref()
            .map_or(0, |budget| budget.in_flight())
    }

    /// Handle to the fire-telemetry counters. Cheap to clone.
    /// Instrumentation only — see [`WalFireMetrics`].
    #[must_use]
    pub fn fire_metrics(&self) -> WalFireMetrics {
        self.fire_metrics.clone()
    }

    /// ADR-034 §Slice B — committed-fsync watermark.
    /// Equivalent to [`WalHandle::committed_fsync_watermark`] on any
    /// handle produced by this writer; exposed here for callers that
    /// have the writer but not yet a handle.
    #[must_use]
    pub fn committed_fsync_watermark(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.committed_fsync_watermark)
    }

    /// Tell the writer to drain, fsync, and exit. Propagates any I/O
    /// error from the last batch.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort send — if the thread already died, the recv
        // side is gone and this errors; we still try to join.
        let _ = self.sender.send(WalCmd::Shutdown);
        if let Some(handle) = self.thread.take() {
            let join_result = handle.join();
            // #1521 M6.1 P1-4 — the writer thread is DEFINITELY gone the
            // instant `join()` returns, on EITHER path (clean Shutdown
            // exit or a panic mid-command). Poison the budget here,
            // unconditionally: any reservation `admit`-ed by a caller
            // whose command was already dequeued by this thread before
            // it died has no other release path (the caller-side
            // "send failed" release in `append_inner` only covers
            // commands that never reached the channel's receiver at
            // all). Idempotent — a normal drain-to-completion shutdown
            // poisons an already-empty (0 in-flight) budget, a no-op.
            if let Some(budget) = self.inflight_budget.as_ref() {
                budget.poison();
            }
            return match join_result {
                Ok(res) => res,
                Err(panic_payload) => {
                    error!("wal writer thread panicked: {panic_payload:?}");
                    Err(ArcGraphError::WalUnavailable)
                }
            };
        }
        Ok(())
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = self.sender.send(WalCmd::Shutdown);
            let _ = thread.join();
            // #1521 M6.1 P1-4 — same poison-on-death rationale as
            // `shutdown`: a `WalWriter` dropped without an explicit
            // `shutdown()` call (e.g. an early return, a panic unwind in
            // the OWNING thread) must still release any budget
            // reservation stranded by the writer thread's death, or a
            // caller blocked in a concurrent `admit()` on another
            // `WalHandle` clone would wait forever.
            if let Some(budget) = self.inflight_budget.as_ref() {
                budget.poison();
            }
        }
    }
}

// ---------- internals -------------------------------------------------------

enum WalCmd {
    /// T1 / Strict — ack after fsync.
    Append {
        assigned_lsn: Option<Lsn>,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
        /// M6.1 — the exact byte count admitted against the caller's
        /// `WalByteBudget` reservation for this append (payload length
        /// before encoding); released once this record's fire settles.
        budget_len: u64,
        ack: Sender<Result<Lsn>>,
    },
    /// ADR-034 §Slice B — T3 / Periodic — ack pre-fsync; writer
    /// notifies the caller as soon as the record is accepted into
    /// the pending batch. Durability is provided by subsequent
    /// group-commit fires.
    AppendAsync {
        assigned_lsn: Option<Lsn>,
        record_type: WalRecordType,
        txn_id: u64,
        timestamp_ms: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
        /// M6.1 — see `Append::budget_len`.
        budget_len: u64,
        ack: Sender<Result<(Lsn, u64)>>,
    },
    Flush {
        ack: Sender<Result<()>>,
    },
    Shutdown,
}

/// Kind of acknowledgement a pending append is waiting for.
///
/// ADR-034 §Slice B: T3 (async) records are acked immediately on
/// enqueue; their `AckKind::Async` carries no sender because the
/// caller has already moved on. T1 (sync) records hold the sender
/// until the fire completes (success or failure).
enum AckKind {
    /// T1 — reply once with `Ok(lsn)` on successful fsync, or
    /// `Err(io_error)` on fsync failure.
    Sync(Sender<Result<Lsn>>),
    /// T3 — caller was already notified. The fire does not send on
    /// any channel for this entry; fsync failure triggers §6.2 abort
    /// at the fire site.
    Async,
}

struct PendingAppend {
    record: WalRecord,
    encoded_len: usize,
    ack: AckKind,
    tracks_exact_durability: bool,
    /// M6.1 — the caller's admitted budget reservation, released when
    /// this entry's fire settles (success or failure alike — the bytes
    /// stop being "in flight" either way).
    budget_len: u64,
}

#[allow(clippy::too_many_arguments)]
fn run(
    mut segment: SegmentWriter,
    rx: Receiver<WalCmd>,
    config: WalConfig,
    initial_lsn_counter: u64,
    fire_metrics: WalFireMetrics,
    committed_fsync_watermark: Arc<AtomicU64>,
    wal_bytes_appended: Arc<AtomicU64>,
    exact_durable_lsns: Arc<DashSet<u64>>,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    encryption: Option<WalEncryption>,
    inflight_budget: Option<Arc<WalByteBudget>>,
    // #1521 M6.1 P1-4 — test-only (see
    // `WalWriter::spawn_from_with_panic_after_first_pending_for_gate`):
    // when `true`, panics deliberately right after the FIRST command is
    // pushed onto `pending` (post-admit, pre-`fire()`) to deterministically
    // reproduce a writer-thread death with a stranded budget reservation.
    // Always `false` from the production `spawn`/`spawn_from` entry points.
    panic_after_first_pending_for_gate: bool,
) -> Result<()> {
    let mut pending: Vec<PendingAppend> = Vec::with_capacity(config.group_commit_max_batch);
    let mut pending_flush_acks: Vec<Sender<Result<()>>> = Vec::new();
    let mut lsn_counter: u64 = initial_lsn_counter;
    let mut wal_offset: u64 = 0;
    let mut window_started: Option<Instant> = None;

    loop {
        let timeout = match window_started {
            Some(started) => config.group_commit_window.saturating_sub(started.elapsed()),
            None => Duration::from_secs(3600),
        };
        match rx.recv_timeout(timeout) {
            Ok(WalCmd::Append {
                assigned_lsn,
                record_type,
                txn_id,
                timestamp_ms,
                tenant_id,
                payload,
                budget_len,
                ack,
            }) => {
                let tracks_exact_durability = assigned_lsn.is_some();
                let lsn = match assigned_lsn {
                    Some(lsn) => {
                        lsn_counter = lsn_counter.max(lsn.raw());
                        lsn
                    }
                    None => {
                        lsn_counter += 1;
                        Lsn::new(lsn_counter)
                    }
                };
                let record = WalRecord {
                    record_type,
                    txn_id,
                    lsn,
                    timestamp_ms,
                    tenant_id,
                    payload,
                };
                match encoded_len_for_fire(&record, encryption.as_ref()) {
                    Ok(encoded_len) => {
                        wal_offset = wal_offset.saturating_add(encoded_len as u64);
                        pending.push(PendingAppend {
                            record,
                            encoded_len,
                            ack: AckKind::Sync(ack),
                            tracks_exact_durability,
                            budget_len,
                        });
                        // #1521 M6.1 P1-4 gate seam — see `run`'s doc on
                        // this parameter.
                        if panic_after_first_pending_for_gate {
                            panic!(
                                "skeptic_wal_budget_leak_on_writer_death: injected \
                                 writer-thread death with a command in `pending` \
                                 (budget_len={budget_len}), pre-fire — deliberately \
                                 simulating a stranded reservation"
                            );
                        }
                        fire_metrics
                            .wal_t1_appends_total
                            .fetch_add(1, Ordering::Relaxed);
                        // ADR-045: emit wal_writes_total{outcome="t1_sync"}
                        // at accept-time (matches the WalFireMetrics
                        // counter increment immediately above). The
                        // sink call is no-op when metrics aren't wired.
                        if let Some(sink) = metrics_sink.as_ref() {
                            sink.record_wal_write(WalWriteOutcome::T1Sync);
                        }
                        if window_started.is_none() {
                            window_started = Some(Instant::now());
                        }
                        if pending.len() >= config.group_commit_max_batch {
                            fire(
                                &mut segment,
                                &mut pending,
                                &mut pending_flush_acks,
                                &fire_metrics,
                                &committed_fsync_watermark,
                                &wal_bytes_appended,
                                &exact_durable_lsns,
                                metrics_sink.as_ref(),
                                encryption.as_ref(),
                                inflight_budget.as_ref(),
                            )?;
                            window_started = None;
                        }
                    }
                    Err(e) => {
                        // Encode failures are caller-side — don't enter
                        // the durable batch; return to the caller. This
                        // entry never reaches `fire()`, so its budget
                        // reservation must be released HERE or it leaks.
                        if let Some(budget) = inflight_budget.as_ref() {
                            budget.release(budget_len);
                        }
                        let _ = ack.send(Err(e));
                    }
                }
            }
            Ok(WalCmd::AppendAsync {
                assigned_lsn,
                record_type,
                txn_id,
                timestamp_ms,
                tenant_id,
                payload,
                budget_len,
                ack,
            }) => {
                let tracks_exact_durability = assigned_lsn.is_some();
                let lsn = match assigned_lsn {
                    Some(lsn) => {
                        lsn_counter = lsn_counter.max(lsn.raw());
                        lsn
                    }
                    None => {
                        lsn_counter += 1;
                        Lsn::new(lsn_counter)
                    }
                };
                let record = WalRecord {
                    record_type,
                    txn_id,
                    lsn,
                    timestamp_ms,
                    tenant_id,
                    payload,
                };
                match encoded_len_for_fire(&record, encryption.as_ref()) {
                    Ok(encoded_len) => {
                        wal_offset = wal_offset.saturating_add(encoded_len as u64);
                        // ADR-034 D-4: ack the caller BEFORE fsync.
                        // The bytes are now on the writer's pending
                        // batch; durability follows per I-D2.
                        let _ = ack.send(Ok((lsn, wal_offset)));
                        pending.push(PendingAppend {
                            record,
                            encoded_len,
                            ack: AckKind::Async,
                            tracks_exact_durability,
                            budget_len,
                        });
                        // #1521 M6.1 P1-4 gate seam — see `run`'s doc on
                        // this parameter.
                        if panic_after_first_pending_for_gate {
                            panic!(
                                "skeptic_wal_budget_leak_on_writer_death: injected \
                                 writer-thread death with a T3 command in `pending` \
                                 (budget_len={budget_len}), pre-fire — deliberately \
                                 simulating a stranded reservation"
                            );
                        }
                        fire_metrics
                            .wal_t3_appends_total
                            .fetch_add(1, Ordering::Relaxed);
                        // ADR-045: emit wal_writes_total{outcome="t3_async"}
                        // at accept-time. Matches the T3 accept tier
                        // (ack-pre-fsync); durability is reported via
                        // a subsequent fire's success.
                        if let Some(sink) = metrics_sink.as_ref() {
                            sink.record_wal_write(WalWriteOutcome::T3Async);
                        }
                        if window_started.is_none() {
                            window_started = Some(Instant::now());
                        }
                        if pending.len() >= config.group_commit_max_batch {
                            fire(
                                &mut segment,
                                &mut pending,
                                &mut pending_flush_acks,
                                &fire_metrics,
                                &committed_fsync_watermark,
                                &wal_bytes_appended,
                                &exact_durable_lsns,
                                metrics_sink.as_ref(),
                                encryption.as_ref(),
                                inflight_budget.as_ref(),
                            )?;
                            window_started = None;
                        }
                    }
                    Err(e) => {
                        // Encode failures are caller-side — don't enter
                        // the durable batch; return to the caller. This
                        // entry never reaches `fire()`; release its
                        // budget reservation here (T3 already ack'd its
                        // caller, but the failed-encode path re-sends an
                        // error — the caller's `append_async_inner`
                        // still admitted budget_len up front, so it must
                        // be released regardless of what the ack carries).
                        if let Some(budget) = inflight_budget.as_ref() {
                            budget.release(budget_len);
                        }
                        let _ = ack.send(Err(e));
                    }
                }
            }
            Ok(WalCmd::Flush { ack }) => {
                pending_flush_acks.push(ack);
                fire(
                    &mut segment,
                    &mut pending,
                    &mut pending_flush_acks,
                    &fire_metrics,
                    &committed_fsync_watermark,
                    &wal_bytes_appended,
                    &exact_durable_lsns,
                    metrics_sink.as_ref(),
                    encryption.as_ref(),
                    inflight_budget.as_ref(),
                )?;
                window_started = None;
            }
            Ok(WalCmd::Shutdown) => {
                fire(
                    &mut segment,
                    &mut pending,
                    &mut pending_flush_acks,
                    &fire_metrics,
                    &committed_fsync_watermark,
                    &wal_bytes_appended,
                    &exact_durable_lsns,
                    metrics_sink.as_ref(),
                    encryption.as_ref(),
                    inflight_budget.as_ref(),
                )?;
                return Ok(());
            }
            Err(RecvTimeoutError::Timeout) => {
                if !pending.is_empty() || !pending_flush_acks.is_empty() {
                    fire(
                        &mut segment,
                        &mut pending,
                        &mut pending_flush_acks,
                        &fire_metrics,
                        &committed_fsync_watermark,
                        &wal_bytes_appended,
                        &exact_durable_lsns,
                        metrics_sink.as_ref(),
                        encryption.as_ref(),
                        inflight_budget.as_ref(),
                    )?;
                }
                window_started = None;
            }
            Err(RecvTimeoutError::Disconnected) => {
                // All producers gone. Drain and exit.
                fire(
                    &mut segment,
                    &mut pending,
                    &mut pending_flush_acks,
                    &fire_metrics,
                    &committed_fsync_watermark,
                    &wal_bytes_appended,
                    &exact_durable_lsns,
                    metrics_sink.as_ref(),
                    encryption.as_ref(),
                    inflight_budget.as_ref(),
                )?;
                return Ok(());
            }
        }
    }
}

fn encoded_len_for_fire(record: &WalRecord, encryption: Option<&WalEncryption>) -> Result<usize> {
    let mut record = record.clone();
    if encryption.is_some() {
        record
            .payload
            .resize(record.payload.len() + WAL_PAYLOAD_HEADER_LEN, 0);
    }
    record.encode_to_vec().map(|bytes| bytes.len())
}

/// W20β-3 / ADR-052: if `encryption` is `Some`, wrap the payload in
/// AEAD at fire time using the segment the batch will actually land
/// in. The outer WalRecord header stays in clear so recovery can
/// route records without the key.
fn encode_for_fire(
    record: &WalRecord,
    encryption: Option<&WalEncryption>,
    landing_segment: u64,
) -> Result<Vec<u8>> {
    let mut record = record.clone();
    if let Some(enc) = encryption {
        record.payload = enc.encrypt(landing_segment, record.lsn, &record.payload)?;
    }
    record.encode_to_vec()
}

#[allow(clippy::too_many_arguments)]
fn fire(
    segment: &mut SegmentWriter,
    pending: &mut Vec<PendingAppend>,
    flush_acks: &mut Vec<Sender<Result<()>>>,
    fire_metrics: &WalFireMetrics,
    committed_fsync_watermark: &Arc<AtomicU64>,
    wal_bytes_appended: &Arc<AtomicU64>,
    exact_durable_lsns: &Arc<DashSet<u64>>,
    metrics_sink: Option<&Arc<dyn MetricsSink>>,
    encryption: Option<&WalEncryption>,
    inflight_budget: Option<&Arc<WalByteBudget>>,
) -> Result<()> {
    // M6.1 §6.1: every entry that ENTERS this fire (regardless of the
    // outcome below) stops being "in flight" once this call returns —
    // release the whole batch's budget reservation up front so every
    // exit path (early `?` propagation included) is covered by one
    // release, matching `batch_bytes`'s own whole-batch accounting.
    if let Some(budget) = inflight_budget {
        let batch_budget_len: u64 = pending.iter().map(|p| p.budget_len).sum();
        if batch_budget_len > 0 {
            budget.release(batch_budget_len);
        }
    }
    // Concatenate all pending encoded records into one batch.
    let batch_bytes: usize = pending.iter().map(|p| p.encoded_len).sum();
    let batch_record_count = pending.len() as u64;
    let highest_lsn_in_batch = pending
        .iter()
        .map(|p| p.record.lsn.raw())
        .max()
        .unwrap_or(0);
    // ADR-034 §6.2: mixed-batch fsync-fail must escalate to abort
    // if any async-ackd record is in the pending batch (its caller
    // already believed the commit succeeded).
    let batch_contains_async = pending.iter().any(|p| matches!(p.ack, AckKind::Async));
    let landing_segment = segment.landing_segment_for(batch_bytes);
    let mut batch = Vec::with_capacity(batch_bytes);
    for p in pending.iter() {
        match encode_for_fire(&p.record, encryption, landing_segment) {
            Ok(bytes) => batch.extend_from_slice(&bytes),
            Err(err) => {
                if batch_contains_async {
                    error!(
                        "ADR-034 §6.2 escalation: WAL fire-time encryption failed with T3 \
                         records in pending batch; aborting process. error: {}",
                        err,
                    );
                    std::process::abort();
                }
                let desc = err.to_string();
                for p in pending.drain(..) {
                    if let AckKind::Sync(s) = p.ack {
                        let _ = s.send(Err(cloned_io_error(&desc)));
                    }
                }
                for ack in flush_acks.drain(..) {
                    let _ = ack.send(Err(cloned_io_error(&desc)));
                }
                return Err(err);
            }
        }
    }

    // Instrumentation (invariant 8 proof): bump fire counters BEFORE
    // the fsync so a crashing fsync still registers the fire. Mean
    // `total_records_fired / total_fires` is the M2-E2 evidence that
    // callers are pipelining rather than serializing via a
    // gate-across-fsync — a pre-fix baseline sees ≈ 1 record per
    // fire, a pipelined post-fix run sees ≥ 2 (with 8 writers
    // batching into a single group-commit cycle it approaches 8).
    fire_metrics.total_fires.fetch_add(1, Ordering::Relaxed);
    fire_metrics
        .total_records_fired
        .fetch_add(batch_record_count, Ordering::Relaxed);

    // Single append + single fsync for the whole batch.
    let write_res = if !batch.is_empty() {
        segment.append(&batch)
    } else {
        Ok(())
    };
    // ADR-045: capture fsync wall-clock for wal_fsync_duration_ms.
    // We observe the fsync portion specifically (not the surrounding
    // append + reply distribution) because design-v2 §10.2 line 704
    // pins fsync duration as the operator alerting signal (P99 ≤ 5 ms).
    // For empty batches the fsync still runs but rarely costs > 1 µs;
    // we still emit the observation so the histogram reflects the
    // full fire-rate distribution.
    let fsync_started = Instant::now();
    let sync_res = write_res.and_then(|()| fsync_with_debug_fault_gate(segment));
    let fsync_duration_ms = fsync_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(sink) = metrics_sink {
        sink.observe_wal_fsync_ms(fsync_duration_ms);
    }

    match sync_res {
        Ok(()) => {
            // ADR-034 §Slice B: advance the committed-fsync watermark
            // to the highest LSN in the batch. Readers (T3 pollers +
            // BackgroundFsyncScheduler) observe durability through
            // this monotonic counter.
            if highest_lsn_in_batch > 0 {
                committed_fsync_watermark.fetch_max(highest_lsn_in_batch, Ordering::AcqRel);
            }
            fire_metrics
                .bytes_fsynced_total
                .fetch_add(batch_bytes as u64, Ordering::Relaxed);
            // SVC-1 P2 / #849 / ADR-229: advance the WAL-bytes gauge the
            // checkpoint byte-trigger reads. Same value as the fire-metric
            // above, but exposed on the cheap `WalHandle` so the checkpointer
            // (which holds a handle, not the WalWriter) can poll it.
            wal_bytes_appended.fetch_add(batch_bytes as u64, Ordering::Relaxed);
            for pending in pending.iter().filter(|p| p.tracks_exact_durability) {
                exact_durable_lsns.insert(pending.record.lsn.raw());
            }
            for p in pending.drain(..) {
                match p.ack {
                    AckKind::Sync(s) => {
                        // Receiver may have gone away; that's fine.
                        let _ = s.send(Ok(p.record.lsn));
                    }
                    AckKind::Async => {
                        // Already ack'd at enqueue; nothing to send.
                    }
                }
            }
            for ack in flush_acks.drain(..) {
                let _ = ack.send(Ok(()));
            }
            Ok(())
        }
        Err(err) => {
            // ADR-045: emit wal_writes_total{outcome="fsync_fail"}
            // BEFORE the §6.2 abort check so the in-process counter
            // increment lands before the kill. Note: the increment is
            // NOT persistent across the abort — atomic state lives in
            // the dying process's heap (prometheus `IntCounter` is an
            // `AtomicI64` inside an `Arc` inside the registered
            // `prometheus::Registry`); `std::process::abort()` raises
            // SIGABRT without unwinding, so all heap state vanishes
            // and the next process starts with a fresh counter at 0.
            //
            // The FsyncFail metric is therefore an observable record
            // of fsync failures ONLY for the non-abort path:
            // - mixed batches with T3 records will abort below before
            //   the next scrape window in any realistic operator setup
            //   (the increment-to-abort window is ~sub-millisecond:
            //   atomic incr + tracing::error! + std::process::abort()),
            //   so the counter is effectively unobservable for the
            //   crash-causing case in production;
            // - sync-only batches with fsync failure return the error
            //   through the channel and the writer loop exits, leaving
            //   the counter scrapable by the live process for the
            //   remainder of its lifetime.
            //
            // Operator implication: a `wal_writes_total{outcome="fsync_fail"}
            // > 0` alert reliably fires for the non-abort path only.
            // The abort path is best surfaced via a panic-handler flush
            // + cross-restart log signal (forward-pinned to M6-08).
            if let Some(sink) = metrics_sink {
                sink.record_wal_write(WalWriteOutcome::FsyncFail);
            }
            // ADR-034 §6.2: if the batch contains any async-ackd
            // records, those commits were already claimed durable.
            // A rollback is incoherent post-ack; the only correct
            // response is process abort. This aligns with D-7 (I-D4)
            // for the background-fsync-fail path, extended here to
            // the foreground-fsync-with-T3-pending path.
            if batch_contains_async {
                tracing::error!(
                    "ADR-034 §6.2 escalation: fsync failed with {} T3 records in pending batch; \
                     aborting process. error: {}",
                    pending
                        .iter()
                        .filter(|p| matches!(p.ack, AckKind::Async))
                        .count(),
                    err,
                );
                std::process::abort();
            }

            // No async records — fall back to the pre-ADR-034 error
            // propagation: every sync waiter gets a cloned error,
            // then the writer loop exits.
            let desc = err.to_string();
            for p in pending.drain(..) {
                match p.ack {
                    AckKind::Sync(s) => {
                        let _ = s.send(Err(cloned_io_error(&desc)));
                    }
                    AckKind::Async => {
                        // Unreachable given the `batch_contains_async`
                        // early-abort above, but defensively leave
                        // no Async waiters unhandled.
                    }
                }
            }
            for ack in flush_acks.drain(..) {
                let _ = ack.send(Err(cloned_io_error(&desc)));
            }
            Err(err)
        }
    }
}

/// Debug/fault-injection-only deterministic crash/fsync-failure seam.
///
/// When `ARCGRAPH_M3_TEST_PAUSE_BEFORE_FSYNC` names a marker path and a
/// sibling `<marker>.arm` file exists, the writer removes the arm, durably
/// creates the marker after the WAL append but before `fdatasync`, then parks.
/// The default action lets a subprocess inspect no-steal state and deliver
/// kernel SIGKILL. `ARCGRAPH_M3_TEST_FSYNC_FAILURE=1` instead waits for a
/// sibling `<marker>.release` file and returns an injected fsync error. Release
/// binaries built without `fault-injection` contain no environment lookup or
/// fault branch.
#[cfg(any(debug_assertions, feature = "fault-injection"))]
fn fsync_with_debug_fault_gate(segment: &SegmentWriter) -> Result<()> {
    let Some(marker) = std::env::var_os("ARCGRAPH_M3_TEST_PAUSE_BEFORE_FSYNC") else {
        return segment.fsync();
    };
    let marker = PathBuf::from(marker);
    let mut arm = marker.as_os_str().to_os_string();
    arm.push(".arm");
    if std::fs::remove_file(PathBuf::from(arm)).is_err() {
        return segment.fsync();
    }
    let file = std::fs::File::create(&marker).expect("create M3 pre-fsync marker");
    file.sync_all().expect("durify M3 pre-fsync marker");
    if std::env::var_os("ARCGRAPH_M3_TEST_FSYNC_FAILURE").is_some() {
        let mut release = marker.as_os_str().to_os_string();
        release.push(".release");
        let release = PathBuf::from(release);
        while !release.exists() {
            std::thread::park_timeout(Duration::from_millis(1));
        }
        return Err(ArcGraphError::Io(std::io::Error::other(
            "injected M3 fsync failure",
        )));
    }
    loop {
        std::thread::park();
    }
}

#[cfg(not(any(debug_assertions, feature = "fault-injection")))]
#[inline]
fn fsync_with_debug_fault_gate(segment: &SegmentWriter) -> Result<()> {
    segment.fsync()
}

fn cloned_io_error(msg: &str) -> ArcGraphError {
    ArcGraphError::Io(std::io::Error::other(msg.to_owned()))
}

// ---------- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use arcgraph_core::Lsn;
    use tempfile::tempdir;

    use super::*;
    use crate::wal::record::{WalRecord, WalRecordType};
    use crate::wal::segment::{SegmentHeader, list_segments, segment_filename};

    fn read_all_records(dir: &std::path::Path) -> Vec<WalRecord> {
        let mut out = Vec::new();
        for seg in list_segments(dir).unwrap() {
            let path = dir.join(segment_filename(seg));
            let bytes = std::fs::read(&path).unwrap();
            // Skip the 8-byte segment header (ADR-issue-#39 format
            // versioning); records start at offset SegmentHeader::SIZE.
            let mut cursor = SegmentHeader::SIZE;
            while cursor < bytes.len() {
                let (r, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
                out.push(r);
                cursor += consumed;
            }
        }
        out
    }

    fn test_config(dir: impl Into<PathBuf>) -> WalConfig {
        // Short window + small batch to exercise group-commit timing
        // without inflating test wall-clock.
        WalConfig {
            dir: dir.into(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_millis(2),
            group_commit_max_batch: 4,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        }
    }

    #[test]
    fn spawn_append_shutdown_roundtrip() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        for i in 1..=10u64 {
            let lsn = handle
                .append(
                    WalRecordType::PutNode,
                    i,
                    i as i64,
                    TenantId::DEFAULT,
                    vec![i as u8],
                )
                .unwrap();
            assert_eq!(lsn, Lsn::new(i));
        }
        writer.shutdown().unwrap();

        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 10);
        for (i, r) in records.iter().enumerate() {
            let expected = (i as u64) + 1;
            assert_eq!(r.lsn, Lsn::new(expected));
            assert_eq!(r.txn_id, expected);
            assert_eq!(r.payload, vec![expected as u8]);
        }
    }

    #[test]
    fn flush_forces_fire_without_batch_fill() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        // Only one append; batch size = 4, so we would wait for the timer
        // otherwise. Flush fires it immediately.
        let lsn = handle
            .append(WalRecordType::Begin, 1, 0, TenantId::DEFAULT, vec![])
            .unwrap();
        handle.flush().unwrap();
        assert_eq!(lsn, Lsn::new(1));
        // File must already reflect the record.
        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 1);
        writer.shutdown().unwrap();
    }

    #[test]
    fn batch_full_triggers_fire() {
        let dir = tempdir().unwrap();
        let config = WalConfig {
            group_commit_window: Duration::from_secs(3600), // effectively never
            group_commit_max_batch: 4,
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        // 4 appends fired purely by batch-size cap (timer never
        // fires at 3600s). Each append blocks on ack, so we spawn
        // one thread per appender: their cmds accumulate in `pending`
        // on the writer until the 4th triggers fire.
        let mut producers = Vec::new();
        for i in 1..=4u64 {
            let h = handle.clone();
            producers.push(thread::spawn(move || {
                h.append(WalRecordType::Commit, i, 0, TenantId::DEFAULT, vec![])
            }));
        }
        for p in producers {
            p.join().unwrap().unwrap();
        }
        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 4);
        writer.shutdown().unwrap();
    }

    #[test]
    fn concurrent_producers_assign_unique_lsns() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = Arc::new(writer.handle());
        let mut assigned: Vec<Lsn> = Vec::new();
        let results: std::sync::Mutex<Vec<Lsn>> = std::sync::Mutex::new(Vec::new());
        thread::scope(|s| {
            for t in 0..8usize {
                let handle = handle.clone();
                let results = &results;
                s.spawn(move || {
                    for i in 0..10u64 {
                        let lsn = handle
                            .append(
                                WalRecordType::PutNode,
                                u64::try_from(t).unwrap(),
                                i as i64,
                                TenantId::DEFAULT,
                                vec![t as u8, i as u8],
                            )
                            .unwrap();
                        results.lock().unwrap().push(lsn);
                    }
                });
            }
        });
        assigned.extend(results.into_inner().unwrap());
        writer.shutdown().unwrap();

        // Distinct LSNs, and they cover the 1..=80 range.
        let mut sorted = assigned.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 80);
        assert_eq!(sorted.first().copied(), Some(Lsn::new(1)));
        assert_eq!(sorted.last().copied(), Some(Lsn::new(80)));

        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 80);
        // Records on disk are in LSN order (group-commit preserves
        // the order in which the writer thread received them, which
        // is a single serialization point).
        for w in records.windows(2) {
            assert!(w[0].lsn < w[1].lsn);
        }
    }

    #[test]
    fn segment_rotates_under_size_pressure() {
        let dir = tempdir().unwrap();
        let config = WalConfig {
            // Tiny segments: header ~36 B × few records then rotate.
            segment_size_bytes: 128,
            group_commit_window: Duration::from_millis(1),
            group_commit_max_batch: 1, // fire per record for deterministic bytes-per-segment
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        for i in 1..=20u64 {
            handle
                .append(
                    WalRecordType::PutRel,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 32],
                )
                .unwrap();
        }
        writer.shutdown().unwrap();

        let segs = list_segments(dir.path()).unwrap();
        assert!(
            segs.len() >= 2,
            "expected rotation to produce multiple segments, got {segs:?}"
        );
        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 20);
    }

    #[test]
    fn shutdown_flushes_pending_batch() {
        let dir = tempdir().unwrap();
        let config = WalConfig {
            group_commit_window: Duration::from_secs(3600),
            group_commit_max_batch: 100,
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        // 5 records; neither batch-full nor timer will fire — only
        // shutdown will. Producers are detached (not scoped), because
        // each `append` blocks on ack and shutdown is what releases
        // those acks.
        let mut producers = Vec::new();
        for i in 1..=5u64 {
            let h = handle.clone();
            producers.push(thread::spawn(move || {
                h.append(WalRecordType::PutNode, i, 0, TenantId::DEFAULT, vec![])
            }));
        }
        // Let the producers enqueue before we trigger shutdown.
        thread::sleep(Duration::from_millis(30));
        drop(handle);

        writer.shutdown().unwrap();
        for p in producers {
            p.join().unwrap().unwrap();
        }

        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn after_shutdown_handle_returns_wal_unavailable() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        writer.shutdown().unwrap();
        let err = handle
            .append(WalRecordType::PutNode, 1, 0, TenantId::DEFAULT, vec![])
            .unwrap_err();
        assert!(matches!(err, ArcGraphError::WalUnavailable));
    }

    // ─── ADR-034 §Slice B: append_async + committed_fsync_watermark ──

    #[test]
    fn append_async_returns_immediately_with_lsn_and_offset() {
        let dir = tempdir().unwrap();
        let config = WalConfig {
            // Long window so the timer never fires during this test;
            // we verify the caller returned before any fsync.
            group_commit_window: Duration::from_secs(3600),
            group_commit_max_batch: 100,
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();

        let t0 = Instant::now();
        let (lsn, offset) = handle
            .append_async(
                WalRecordType::CommitBundle,
                7,
                0,
                TenantId::DEFAULT,
                vec![0u8; 32],
            )
            .unwrap();
        let elapsed = t0.elapsed();

        assert_eq!(lsn, Lsn::new(1));
        assert!(offset > 0);
        assert!(
            elapsed < Duration::from_millis(50),
            "append_async blocked for {elapsed:?} — should return pre-fsync"
        );

        // Watermark has NOT advanced — fsync hasn't run yet.
        assert_eq!(
            handle.last_durable_lsn(),
            Lsn::ZERO,
            "watermark must not advance before fsync"
        );

        // Flush forces fire; watermark advances.
        handle.flush().unwrap();
        assert_eq!(handle.last_durable_lsn(), lsn);

        writer.shutdown().unwrap();
    }

    #[test]
    fn committed_fsync_watermark_advances_on_fire() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        assert_eq!(handle.last_durable_lsn(), Lsn::ZERO);

        let lsn1 = handle
            .append(WalRecordType::Begin, 1, 0, TenantId::DEFAULT, vec![])
            .unwrap();
        assert_eq!(
            handle.last_durable_lsn(),
            lsn1,
            "sync append fsyncs before return"
        );

        let lsn2 = handle
            .append(WalRecordType::Commit, 2, 0, TenantId::DEFAULT, vec![])
            .unwrap();
        assert_eq!(handle.last_durable_lsn(), lsn2);

        writer.shutdown().unwrap();
    }

    #[test]
    fn t1_piggybacks_t3_durability() {
        // ADR-034 I-D3: a T1 (sync) commit's fsync durifies any T3
        // (async) records already in the pending batch.
        //
        // To make this deterministic, size `group_commit_max_batch`
        // so the T1 append is the record that fills the batch and
        // triggers the fire. Timer is long but finite — belt-and-
        // braces if the test arrives under load.
        let dir = tempdir().unwrap();
        let config = WalConfig {
            group_commit_window: Duration::from_millis(100),
            group_commit_max_batch: 3,
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();

        // Enqueue 2 T3 records — neither fsync'd yet.
        let (lsn_t3_a, _) = handle
            .append_async(
                WalRecordType::CommitBundle,
                100,
                0,
                TenantId::DEFAULT,
                vec![0u8; 16],
            )
            .unwrap();
        let (lsn_t3_b, _) = handle
            .append_async(
                WalRecordType::CommitBundle,
                101,
                0,
                TenantId::DEFAULT,
                vec![0u8; 16],
            )
            .unwrap();
        assert_eq!(lsn_t3_a, Lsn::new(1));
        assert_eq!(lsn_t3_b, Lsn::new(2));
        assert_eq!(
            handle.last_durable_lsn(),
            Lsn::ZERO,
            "T3 bytes not yet durable pre-piggyback"
        );

        // One T1 append — must block on fsync. The fire includes both
        // T3 records + the T1 record.
        let lsn_t1 = handle
            .append(
                WalRecordType::CommitBundle,
                200,
                0,
                TenantId::DEFAULT,
                vec![0u8; 16],
            )
            .unwrap();

        // Piggyback: watermark advanced past all three.
        assert!(handle.last_durable_lsn() >= lsn_t1);
        assert!(handle.last_durable_lsn() >= lsn_t3_b);

        // Disk check: all three records on disk.
        writer.shutdown().unwrap();
        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn append_async_fires_on_batch_full() {
        // Without an explicit flush, batch-full also fires. Confirm
        // the watermark advances correctly.
        let dir = tempdir().unwrap();
        let config = WalConfig {
            group_commit_window: Duration::from_secs(3600),
            group_commit_max_batch: 3,
            ..test_config(dir.path())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();

        for i in 1..=3u64 {
            handle
                .append_async(
                    WalRecordType::CommitBundle,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 8],
                )
                .unwrap();
        }
        // Batch-full triggered fire. Wait a tick for the writer to
        // complete fsync; in-test polling loop bounded at 1 s.
        let deadline = Instant::now() + Duration::from_millis(1000);
        while handle.last_durable_lsn().raw() < 3 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(handle.last_durable_lsn(), Lsn::new(3));

        writer.shutdown().unwrap();
    }

    #[test]
    fn metrics_count_t1_and_t3_appends_separately() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        let metrics = writer.fire_metrics();

        // 3 T1 + 2 T3.
        for i in 1..=3u64 {
            handle
                .append(WalRecordType::Begin, i, 0, TenantId::DEFAULT, vec![])
                .unwrap();
        }
        for i in 1..=2u64 {
            handle
                .append_async(
                    WalRecordType::CommitBundle,
                    i + 100,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 4],
                )
                .unwrap();
        }
        handle.flush().unwrap();
        writer.shutdown().unwrap();

        assert_eq!(metrics.wal_t1_appends_total(), 3);
        assert_eq!(metrics.wal_t3_appends_total(), 2);
        assert!(metrics.bytes_fsynced_total() > 0);
    }

    /// SVC-1 P2 / #849 / ADR-229: the WAL-bytes-appended gauge advances by
    /// the fired batch bytes and is readable off a cheap `WalHandle`. This is
    /// the counter the checkpoint byte-trigger reads.
    #[test]
    fn wal_bytes_appended_advances_on_fire() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        assert_eq!(handle.wal_bytes_appended(), 0, "fresh writer at 0 bytes");

        // A sync append fsyncs (fires) before returning → bytes advance.
        handle
            .append(
                WalRecordType::PutNode,
                1,
                0,
                TenantId::DEFAULT,
                vec![0u8; 64],
            )
            .unwrap();
        let after_one = handle.wal_bytes_appended();
        assert!(
            after_one >= (WalRecord::HEADER_SIZE + 64) as u64,
            "bytes must include header + payload, got {after_one}",
        );

        handle
            .append(
                WalRecordType::PutNode,
                2,
                0,
                TenantId::DEFAULT,
                vec![0u8; 128],
            )
            .unwrap();
        assert!(
            handle.wal_bytes_appended() > after_one,
            "bytes must be monotonic across fires",
        );

        writer.shutdown().unwrap();
    }

    #[test]
    fn writer_starts_watermark_at_initial_lsn() {
        // ADR-034 §Slice B: spawn_from seeds the watermark, so
        // post-replay state is consistent.
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn_from(test_config(dir.path()), Lsn::new(42)).unwrap();
        let handle = writer.handle();
        assert_eq!(handle.last_durable_lsn(), Lsn::new(42));

        // Next append gets LSN 43 and advances the watermark.
        let lsn = handle
            .append(WalRecordType::PutNode, 1, 0, TenantId::DEFAULT, vec![0u8])
            .unwrap();
        assert_eq!(lsn, Lsn::new(43));
        assert_eq!(handle.last_durable_lsn(), Lsn::new(43));

        writer.shutdown().unwrap();
    }

    #[test]
    fn assigned_bundle_lsns_share_the_redo_clock_and_allow_disk_reordering() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();

        assert_eq!(
            handle
                .append_at(
                    Lsn::new(10),
                    WalRecordType::CommitBundle,
                    10,
                    0,
                    TenantId::DEFAULT,
                    Vec::new(),
                )
                .unwrap(),
            Lsn::new(10)
        );
        assert!(handle.take_exact_durable(Lsn::new(10)));
        assert!(!handle.take_exact_durable(Lsn::new(10)));
        assert_eq!(
            handle
                .append_at(
                    Lsn::new(5),
                    WalRecordType::CommitBundle,
                    5,
                    0,
                    TenantId::DEFAULT,
                    Vec::new(),
                )
                .unwrap(),
            Lsn::new(5),
            "Phase 2 may reach disk out of range-base order"
        );
        assert_eq!(handle.last_durable_lsn(), Lsn::new(10));
        assert!(handle.take_exact_durable(Lsn::new(5)));
        let automatic = handle
            .append(WalRecordType::Begin, 11, 0, TenantId::DEFAULT, Vec::new())
            .unwrap();
        assert_eq!(automatic, Lsn::new(11));
        writer.shutdown().unwrap();

        let lsns: Vec<_> = read_all_records(dir.path())
            .into_iter()
            .map(|record| record.lsn)
            .collect();
        assert_eq!(lsns, vec![Lsn::new(10), Lsn::new(5), Lsn::new(11)]);
    }

    #[test]
    fn handle_reports_migrated_v9_generation_format() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join(segment_filename(0)),
            SegmentHeader { format_version: 9 }.encode(),
        )
        .unwrap();

        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        assert_eq!(writer.handle().format_version(), 9);
        writer.shutdown().unwrap();
    }

    #[test]
    fn mixed_t1_t3_records_on_disk_are_indistinguishable() {
        // ADR-034 D-3 / I-D6: on-disk bytes carry NO tier marker.
        // A mixed-tier workload produces records that are
        // indistinguishable by their record-level encoding.
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();

        handle
            .append_async(
                WalRecordType::CommitBundle,
                1,
                0,
                TenantId::DEFAULT,
                vec![1u8; 8],
            )
            .unwrap();
        handle
            .append(
                WalRecordType::CommitBundle,
                2,
                0,
                TenantId::DEFAULT,
                vec![2u8; 8],
            )
            .unwrap();
        writer.shutdown().unwrap();

        let records = read_all_records(dir.path());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].record_type, WalRecordType::CommitBundle);
        assert_eq!(records[1].record_type, WalRecordType::CommitBundle);
        // No tier field in WalRecord; no field to inspect.
    }

    // -----------------------------------------------------------------
    // W16γ M6-07 — MetricsSink wire pins (ADR-045)
    // -----------------------------------------------------------------

    /// Pin: when `WalConfig::metrics_sink` is `None`, the writer thread
    /// runs the legacy zero-overhead path without panicking or
    /// degrading. Verifies the `if let Some(sink) = metrics_sink.as_ref()`
    /// branch is `None`-safe.
    #[test]
    fn metrics_sink_none_path_is_inert() {
        let dir = tempdir().unwrap();
        // test_config() leaves metrics_sink: None.
        let writer = WalWriter::spawn(test_config(dir.path())).unwrap();
        let handle = writer.handle();
        for i in 1..=5u64 {
            handle
                .append(WalRecordType::Begin, i, 0, TenantId::DEFAULT, vec![])
                .unwrap();
        }
        for i in 1..=3u64 {
            handle
                .append_async(
                    WalRecordType::CommitBundle,
                    100 + i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 4],
                )
                .unwrap();
        }
        handle.flush().unwrap();
        writer.shutdown().unwrap();
    }

    /// Pin: `WalConfig::with_metrics_sink` plumbs a sink into the
    /// writer thread, and the writer emits `WalWriteOutcome::T1Sync` /
    /// `T3Async` exactly once per accepted append.
    #[test]
    fn metrics_sink_records_t1_t3_per_append() {
        use crate::metrics::{CountingMetricsSink, MetricsSink, WalWriteOutcome};
        let dir = tempdir().unwrap();
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_arc: Arc<dyn MetricsSink> = sink.clone();
        let cfg = test_config(dir.path()).with_metrics_sink(sink_arc);
        let writer = WalWriter::spawn(cfg).unwrap();
        let handle = writer.handle();
        // 4 T1 + 2 T3 = 6 total accepted appends.
        for i in 1..=4u64 {
            handle
                .append(WalRecordType::Begin, i, 0, TenantId::DEFAULT, vec![])
                .unwrap();
        }
        for i in 1..=2u64 {
            handle
                .append_async(
                    WalRecordType::CommitBundle,
                    100 + i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 4],
                )
                .unwrap();
        }
        handle.flush().unwrap();
        writer.shutdown().unwrap();

        // The sink saw exactly 4 T1Sync + 2 T3Async + 0 FsyncFail.
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::T1Sync), 4);
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::T3Async), 2);
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::FsyncFail), 0);
    }

    /// Pin: `observe_wal_fsync_ms` fires at least once per `fire()`
    /// invocation. With a small batch + manual flush, we expect at
    /// least 2 observations (one for the batch-full or flush fire,
    /// one for shutdown drain).
    #[test]
    fn metrics_sink_observes_wal_fsync_duration() {
        use crate::metrics::{CountingMetricsSink, MetricsSink};
        let dir = tempdir().unwrap();
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_arc: Arc<dyn MetricsSink> = sink.clone();
        let cfg = test_config(dir.path()).with_metrics_sink(sink_arc);
        let writer = WalWriter::spawn(cfg).unwrap();
        let handle = writer.handle();
        for i in 1..=2u64 {
            handle
                .append(
                    WalRecordType::PutNode,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![i as u8],
                )
                .unwrap();
        }
        handle.flush().unwrap();
        writer.shutdown().unwrap();

        // At minimum: 1 fire for batch + 1 fire for shutdown drain
        // (the drain may be a no-op if the prior fire emptied
        // pending, but the fire() function still observes the
        // duration regardless of batch contents — ADR-045 §"observation
        // includes empty fires"). We assert ≥ 2 to be tolerant of
        // any extra group-commit-window fires.
        assert!(
            sink.wal_fsync_observation_count() >= 2,
            "expected ≥ 2 fsync observations, got {}",
            sink.wal_fsync_observation_count()
        );
    }

    /// Pin: `with_metrics_sink` is the builder-style entrypoint;
    /// chaining preserves the prior config fields.
    #[test]
    fn with_metrics_sink_is_additive_to_existing_config() {
        use crate::metrics::CountingMetricsSink;
        let dir = tempdir().unwrap();
        let sink: Arc<dyn MetricsSink> = Arc::new(CountingMetricsSink::new());
        let cfg = test_config(dir.path()).with_metrics_sink(sink.clone());
        assert_eq!(cfg.group_commit_max_batch, 4); // from test_config
        assert_eq!(cfg.group_commit_window, Duration::from_millis(2));
        assert!(cfg.metrics_sink.is_some());
    }
}
