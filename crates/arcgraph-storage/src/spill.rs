//! M6.2 OOC-1 bounded executor spill-run infrastructure.
//!
//! This module owns disk I/O only. Higher executor slices serialize their
//! existing `Batch` representation to bytes and use the same byte frames on
//! restore; OOC-2/3/4 deliberately remain outside this slice.
//!
//! A run is an 84-byte identity header followed by bounded frames:
//!
//! ```text
//! run header: magic/version/flags, tenant, query epoch, run number,
//!             frame count, random per-run nonce base, header CRC32C
//! frame:      stored_len:u32, plaintext_len:u32, chunk:u32,
//!             clear_crc_or_zero:u32, payload[stored_len]
//! ```
//!
//! Encrypted payloads are `AES-256-GCM(ciphertext || tag)`. The GCM AAD binds
//! the complete run identity, encryption flag, chunk counter, and plaintext
//! length. Clear frames carry CRC32C. `finish` flushes, seals the frame count,
//! and `sync_data`s before a reader can be constructed, so a same-process
//! reader is never handed a partially published frame. Scratch is not WAL and
//! has no crash-recovery contract.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
#[cfg(feature = "fault-injection")]
use std::sync::Condvar;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arcgraph_core::TenantId;
use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use zeroize::{Zeroize, Zeroizing};

use crate::encryption::{AEAD_KEY_LEN, AES_GCM_IV_LEN, AES_GCM_TAG_LEN, Aes256GcmCipher};

const RUN_MAGIC: [u8; 8] = *b"AGSPL001";
const RUN_VERSION: u16 = 1;
const FLAG_ENCRYPTED: u16 = 1;
const RUN_HEADER_BYTES: u64 = 84;
const FRAME_HEADER_BYTES: u64 = 16;
const FRAME_AAD_BYTES: usize = 68;
const SPILL_IO_BUFFER_BYTES: u64 = 8 * 1024;

/// Defensive reader/writer allocation ceiling for one serialized executor
/// batch. The executor's 2,048-row batches are expected to be much smaller.
pub const MAX_SPILL_BATCH_BYTES: usize = 64 * 1024 * 1024;
/// A run may contain at most one million chunks. This also leaves ample room
/// above a random nonce-base suffix for checked monotonic nonce derivation.
pub const MAX_SPILL_FRAMES_PER_RUN: u32 = 1_000_000;
/// Default per-tenant on-disk quota relative to its executor memory budget.
pub const DEFAULT_SPILL_QUOTA_MULTIPLIER: u64 = 4;
/// Default free-space floor on the data/WAL volume.
pub const DEFAULT_VOLUME_HEADROOM_PERCENT: u8 = 10;
/// Opportunistic production orphan sweep cadence. Startup always sweeps;
/// active engines sweep again after this many query attempts.
pub const DEFAULT_ORPHAN_SWEEP_QUERY_INTERVAL: u64 = 64;
/// Default process-wide RAM line for spill read/write staging. Operators may
/// tune this from the process envelope without charging disk bytes to the page
/// pool. Two maximum encrypted restore buffers fit concurrently.
pub const DEFAULT_SPILL_STAGING_MEMORY_BYTES: u64 =
    (MAX_SPILL_BATCH_BYTES as u64 + AES_GCM_TAG_LEN as u64) * 2 + 1024 * 1024;

#[cfg(feature = "fault-injection")]
const MAX_RETAINED_TEST_RUNS: usize = 8;
#[cfg(feature = "fault-injection")]
const MAX_RETAINED_TEST_RUN_BYTES: u64 = 1024 * 1024;

#[cfg(feature = "fault-injection")]
struct SpillFaultOptions {
    retained_run_limit: usize,
    disable_encryption: bool,
}

#[cfg(not(feature = "fault-injection"))]
struct SpillFaultOptions;

#[cfg(feature = "fault-injection")]
#[derive(Default)]
struct SpillSweepCreateBarrierState {
    tenant_inspected: bool,
    sweep_started: bool,
    sweep_finished: bool,
}

#[cfg(feature = "fault-injection")]
struct SpillSweepCreateBarrierInner {
    state: Mutex<SpillSweepCreateBarrierState>,
    changed: Condvar,
}

/// One-shot rendezvous for the scratch-directory vs orphan-sweep regression
/// gate. This surface exists only in fault-injection builds.
#[cfg(feature = "fault-injection")]
#[derive(Clone)]
pub struct SpillSweepCreateBarrier {
    inner: Arc<SpillSweepCreateBarrierInner>,
}

#[cfg(feature = "fault-injection")]
impl SpillSweepCreateBarrier {
    fn new() -> Self {
        Self {
            inner: Arc::new(SpillSweepCreateBarrierInner {
                state: Mutex::new(SpillSweepCreateBarrierState::default()),
                changed: Condvar::new(),
            }),
        }
    }

    /// Waits until `create_run` has inspected/materialized the tenant
    /// directory and reached the deterministic race rendezvous.
    pub fn wait_until_tenant_inspected(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("spill sweep/create barrier mutex poisoned");
        while !state.tenant_inspected {
            state = self
                .inner
                .changed
                .wait(state)
                .expect("spill sweep/create barrier mutex poisoned");
        }
    }

    fn wait_after_tenant_inspection(&self, lifecycle_held: bool) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("spill sweep/create barrier mutex poisoned");
        state.tenant_inspected = true;
        self.inner.changed.notify_all();
        while if lifecycle_held {
            !state.sweep_started
        } else {
            !state.sweep_finished
        } {
            state = self
                .inner
                .changed
                .wait(state)
                .expect("spill sweep/create barrier mutex poisoned");
        }
    }

    fn mark_sweep_started(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("spill sweep/create barrier mutex poisoned");
        state.sweep_started = true;
        self.inner.changed.notify_all();
    }

    fn mark_sweep_finished(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("spill sweep/create barrier mutex poisoned");
        state.sweep_finished = true;
        self.inner.changed.notify_all();
    }
}

/// Full attempt identity embedded in the path and every run header.
/// `generation` is allocated monotonically by one [`SpillManager`] boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueryEpoch {
    engine_boot_id: u128,
    generation: u64,
    query_id: u64,
    attempt: u32,
}

impl QueryEpoch {
    fn new(engine_boot_id: u128, generation: u64, query_id: u64, attempt: u32) -> Self {
        Self {
            engine_boot_id,
            generation,
            query_id,
            attempt,
        }
    }

    #[must_use]
    pub const fn engine_boot_id(self) -> u128 {
        self.engine_boot_id
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn query_id(self) -> u64 {
        self.query_id
    }

    #[must_use]
    pub const fn attempt(self) -> u32 {
        self.attempt
    }

    fn directory_name(self) -> String {
        format!(
            "{:032x}-{:016x}-{:016x}-{:08x}",
            self.engine_boot_id, self.generation, self.query_id, self.attempt
        )
    }

    fn parse_directory_name(value: &str) -> Option<Self> {
        let mut parts = value.split('-');
        let boot = parts.next()?;
        let generation = parts.next()?;
        let query_id = parts.next()?;
        let attempt = parts.next()?;
        if boot.len() != 32
            || generation.len() != 16
            || query_id.len() != 16
            || attempt.len() != 8
            || parts.next().is_some()
        {
            return None;
        }
        let epoch = Self {
            engine_boot_id: u128::from_str_radix(boot, 16).ok()?,
            generation: u64::from_str_radix(generation, 16).ok()?,
            query_id: u64::from_str_radix(query_id, 16).ok()?,
            attempt: u32::from_str_radix(attempt, 16).ok()?,
        };
        (epoch.directory_name() == value).then_some(epoch)
    }
}

impl fmt::Display for QueryEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.directory_name())
    }
}

/// Run identity stamped both in-memory and on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillRunIdentity {
    pub tenant_id: TenantId,
    pub query_epoch: QueryEpoch,
    pub run_number: u32,
}

/// Whether spill encryption is mandatory for this query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillEncryptionPolicy {
    /// The tenant's page or WAL encryption is enabled. This always mandates
    /// spill encryption; there is no production disable switch.
    pub tenant_encryption_enabled: bool,
    /// Defense-in-depth/configuration knob to encrypt scratch for a tenant
    /// whose durable pages/WAL are otherwise clear.
    pub force_encryption: bool,
}

impl SpillEncryptionPolicy {
    #[must_use]
    pub const fn encryption_required(self) -> bool {
        self.tenant_encryption_enabled || self.force_encryption
    }
}

/// Configurable global free-space floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeHeadroom {
    Percent(u8),
    Bytes(u64),
}

impl Default for VolumeHeadroom {
    fn default() -> Self {
        Self::Percent(DEFAULT_VOLUME_HEADROOM_PERCENT)
    }
}

impl VolumeHeadroom {
    fn validate(self) -> Result<Self, SpillError> {
        if matches!(self, Self::Percent(percent) if percent > 100) {
            return Err(SpillError::InvalidConfig(
                "spill volume headroom percent must be <= 100".to_owned(),
            ));
        }
        Ok(self)
    }

    fn floor_bytes(self, total_bytes: u64) -> u64 {
        match self {
            Self::Percent(percent) => {
                ((u128::from(total_bytes) * u128::from(percent)) / 100) as u64
            }
            Self::Bytes(bytes) => bytes,
        }
    }
}

/// Actual capacity readings for the filesystem holding the data directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeSpace {
    pub available_bytes: u64,
    pub total_bytes: u64,
    /// Filesystem allocation granularity used to turn encoded bytes into a
    /// conservative real-volume reservation delta.
    pub allocation_unit_bytes: u64,
}

/// Process-wide manager configuration.
#[derive(Clone, Debug)]
pub struct SpillManagerConfig {
    pub data_dir: PathBuf,
    pub volume_headroom: VolumeHeadroom,
    pub staging_memory_limit_bytes: u64,
}

impl SpillManagerConfig {
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            volume_headroom: VolumeHeadroom::default(),
            staging_memory_limit_bytes: DEFAULT_SPILL_STAGING_MEMORY_BYTES,
        }
    }
}

/// Per-query scratch configuration. `spill_quota_bytes=None` selects the
/// required `4 * executor_memory_budget_bytes` default.
#[derive(Clone, Copy, Debug)]
pub struct SpillQueryConfig {
    pub tenant_id: TenantId,
    pub query_id: u64,
    pub attempt: u32,
    pub executor_memory_budget_bytes: u64,
    pub spill_quota_bytes: Option<u64>,
    pub encryption: SpillEncryptionPolicy,
}

impl SpillQueryConfig {
    #[must_use]
    pub fn new(
        tenant_id: TenantId,
        query_id: u64,
        attempt: u32,
        executor_memory_budget_bytes: u64,
    ) -> Self {
        Self {
            tenant_id,
            query_id,
            attempt,
            executor_memory_budget_bytes,
            spill_quota_bytes: None,
            encryption: SpillEncryptionPolicy::default(),
        }
    }

    fn quota_bytes(self) -> u64 {
        self.spill_quota_bytes.unwrap_or_else(|| {
            self.executor_memory_budget_bytes
                .saturating_mul(DEFAULT_SPILL_QUOTA_MULTIPLIER)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillRejectReason {
    TenantQuota,
    VolumeHeadroom,
    SpillStagingMemory,
}

/// Typed failure surface consumed by later executor OOC slices.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SpillError {
    #[error("invalid spill configuration: {0}")]
    InvalidConfig(String),
    #[error("spill query/run counter exhausted; refusing identity reuse")]
    IdentityExhausted,
    #[error(
        "stale spill epoch: active query epoch {active_epoch}, run epoch {run_epoch}; prior-attempt scratch is never adopted"
    )]
    StaleEpoch {
        active_epoch: QueryEpoch,
        run_epoch: QueryEpoch,
    },
    #[error("spill query epoch {epoch} has ended")]
    QueryEnded { epoch: QueryEpoch },
    #[error(
        "spill resource exhausted ({reason:?}): tenant={tenant_id:?}, requested_bytes={requested_bytes}, spilled_bytes={spilled_bytes}, limit_bytes={limit_bytes}, available_bytes={available_bytes:?}"
    )]
    ResourceExhausted {
        reason: SpillRejectReason,
        tenant_id: TenantId,
        requested_bytes: u64,
        spilled_bytes: u64,
        limit_bytes: u64,
        available_bytes: Option<u64>,
    },
    #[error("spill batch is {len} bytes, maximum is {max}")]
    BatchTooLarge { len: usize, max: usize },
    #[error("spill run chunk counter exhausted; refusing AES-GCM nonce reuse")]
    NonceExhausted,
    #[error("spill run header is invalid: {0}")]
    InvalidHeader(String),
    #[error("spill frame {chunk} is corrupt: {reason}")]
    CorruptFrame { chunk: u32, reason: String },
    #[error("spill frame {chunk} authentication failed")]
    AuthenticationFailed { chunk: u32 },
    #[error("spill I/O during {operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("OS random source failed while generating {purpose}: {reason}")]
    Random {
        purpose: &'static str,
        reason: String,
    },
    #[error("spill encryption initialization failed: {0}")]
    Encryption(String),
}

impl SpillError {
    #[must_use]
    pub const fn spilled_bytes(&self) -> Option<u64> {
        match self {
            Self::ResourceExhausted { spilled_bytes, .. } => Some(*spilled_bytes),
            _ => None,
        }
    }
}

fn io_error(operation: &'static str, source: io::Error) -> SpillError {
    SpillError::Io { operation, source }
}

#[derive(Default)]
struct AccountingState {
    tenant_bytes: HashMap<TenantId, u64>,
    /// Bytes reserved but not yet reflected in `statvfs`. This closes the
    /// concurrent reserve-before-buffered-write race across tenants.
    pending_volume_bytes: u64,
    staging_memory_bytes: u64,
}

struct SpillManagerInner {
    data_dir: PathBuf,
    spill_root: PathBuf,
    headroom: VolumeHeadroom,
    staging_memory_limit_bytes: u64,
    engine_boot_id: u128,
    next_generation: AtomicU64,
    queries_since_sweep: AtomicU64,
    /// Serializes epoch registration/removal with fallback-tree sweeping.
    lifecycle: Mutex<()>,
    live_epochs: Mutex<HashSet<QueryEpoch>>,
    accounting: Mutex<AccountingState>,
    #[cfg(feature = "fault-injection")]
    retained_run_limit: usize,
    #[cfg(feature = "fault-injection")]
    retained_runs_claimed: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "fault-injection")]
    disable_encryption_for_test: bool,
    #[cfg(feature = "fault-injection")]
    sweep_create_barrier: Mutex<Option<SpillSweepCreateBarrier>>,
}

/// Shared scratch identity, quota, headroom, and orphan-lifecycle owner.
#[derive(Clone)]
pub struct SpillManager {
    inner: Arc<SpillManagerInner>,
}

impl fmt::Debug for SpillManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillManager")
            .field("spill_root", &self.inner.spill_root)
            .field("headroom", &self.inner.headroom)
            .finish_non_exhaustive()
    }
}

impl SpillManager {
    pub fn new(config: SpillManagerConfig) -> Result<Self, SpillError> {
        #[cfg(feature = "fault-injection")]
        let faults = SpillFaultOptions {
            retained_run_limit: 0,
            disable_encryption: false,
        };
        #[cfg(not(feature = "fault-injection"))]
        let faults = SpillFaultOptions;
        Self::build(config, faults)
    }

    #[cfg(feature = "fault-injection")]
    pub fn new_with_fault_injection(
        config: SpillManagerConfig,
        retained_run_limit: usize,
        disable_encryption_for_test: bool,
    ) -> Result<Self, SpillError> {
        if retained_run_limit > MAX_RETAINED_TEST_RUNS {
            return Err(SpillError::InvalidConfig(format!(
                "fault-injection spill retention is bounded at {MAX_RETAINED_TEST_RUNS} runs"
            )));
        }
        Self::build(
            config,
            SpillFaultOptions {
                retained_run_limit,
                disable_encryption: disable_encryption_for_test,
            },
        )
    }

    fn build(config: SpillManagerConfig, faults: SpillFaultOptions) -> Result<Self, SpillError> {
        #[cfg(not(feature = "fault-injection"))]
        let _ = faults;
        let headroom = config.volume_headroom.validate()?;
        if config.staging_memory_limit_bytes < SPILL_IO_BUFFER_BYTES {
            return Err(SpillError::InvalidConfig(format!(
                "spill staging memory must be at least {SPILL_IO_BUFFER_BYTES} bytes"
            )));
        }
        ensure_private_dir(&config.data_dir)?;
        let spill_root = config.data_dir.join("spill");
        ensure_private_dir(&spill_root)?;
        let mut boot_bytes = [0_u8; 16];
        fill_random(&mut boot_bytes, "spill engine boot id")?;
        let manager = Self {
            inner: Arc::new(SpillManagerInner {
                data_dir: config.data_dir,
                spill_root,
                headroom,
                staging_memory_limit_bytes: config.staging_memory_limit_bytes,
                engine_boot_id: u128::from_be_bytes(boot_bytes),
                next_generation: AtomicU64::new(1),
                queries_since_sweep: AtomicU64::new(0),
                lifecycle: Mutex::new(()),
                live_epochs: Mutex::new(HashSet::new()),
                accounting: Mutex::new(AccountingState::default()),
                #[cfg(feature = "fault-injection")]
                retained_run_limit: faults.retained_run_limit,
                #[cfg(feature = "fault-injection")]
                retained_runs_claimed: std::sync::atomic::AtomicUsize::new(0),
                #[cfg(feature = "fault-injection")]
                disable_encryption_for_test: faults.disable_encryption,
                #[cfg(feature = "fault-injection")]
                sweep_create_barrier: Mutex::new(None),
            }),
        };
        // Startup sweep: a new engine boot has no live epochs and never adopts
        // a prior boot's named fallback/test-retained scratch.
        manager.periodic_sweep()?;
        Ok(manager)
    }

    #[must_use]
    pub fn spill_root(&self) -> &Path {
        &self.inner.spill_root
    }

    pub fn volume_space(&self) -> Result<VolumeSpace, SpillError> {
        volume_space(&self.inner.data_dir)
            .map_err(|error| io_error("measure spill volume headroom", error))
    }

    /// Arms the one-shot scratch-directory vs orphan-sweep rendezvous used by
    /// the M6.2 liveness regression gate.
    #[cfg(feature = "fault-injection")]
    pub fn arm_sweep_create_barrier_for_test(&self) -> SpillSweepCreateBarrier {
        let barrier = SpillSweepCreateBarrier::new();
        *self
            .inner
            .sweep_create_barrier
            .lock()
            .expect("spill sweep/create barrier slot mutex poisoned") = Some(barrier.clone());
        barrier
    }

    #[must_use]
    pub fn spilled_bytes(&self, tenant_id: TenantId) -> u64 {
        self.inner
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned")
            .tenant_bytes
            .get(&tenant_id)
            .copied()
            .unwrap_or(0)
    }

    pub fn begin_query(&self, config: SpillQueryConfig) -> Result<SpillQuery, SpillError> {
        self.maybe_periodic_sweep()?;
        let _lifecycle = self
            .inner
            .lifecycle
            .lock()
            .expect("spill lifecycle mutex poisoned");
        // fetch_update refuses wrap rather than saturating/reusing an epoch.
        let generation = self
            .inner
            .next_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| SpillError::IdentityExhausted)?;
        let epoch = QueryEpoch::new(
            self.inner.engine_boot_id,
            generation,
            config.query_id,
            config.attempt,
        );
        let tenant_dir = self
            .inner
            .spill_root
            .join(config.tenant_id.raw().to_string());
        let query_dir = tenant_dir.join(epoch.directory_name());

        let required = config.encryption.encryption_required();
        #[cfg(feature = "fault-injection")]
        let encryption_enabled = required && !self.inner.disable_encryption_for_test;
        #[cfg(not(feature = "fault-injection"))]
        let encryption_enabled = required;

        let state = Arc::new(QueryState::new(epoch, encryption_enabled)?);
        self.inner
            .live_epochs
            .lock()
            .expect("spill live-epoch mutex poisoned")
            .insert(epoch);
        Ok(SpillQuery {
            manager: self.clone(),
            identity: QueryIdentity {
                tenant_id: config.tenant_id,
                epoch,
                query_dir,
            },
            quota_bytes: config.quota_bytes(),
            next_run: AtomicU32::new(0),
            directory_charge: Mutex::new(None),
            state,
        })
    }

    /// Periodic-sweep entry point for the engine scheduler. Only epochs in the
    /// manager's live registry survive. No stale run is ever opened/adopted.
    pub fn periodic_sweep(&self) -> Result<SpillSweepReport, SpillError> {
        #[cfg(feature = "fault-injection")]
        let race_barrier = self
            .inner
            .sweep_create_barrier
            .lock()
            .expect("spill sweep/create barrier slot mutex poisoned")
            .clone();
        #[cfg(feature = "fault-injection")]
        if let Some(barrier) = &race_barrier {
            barrier.mark_sweep_started();
        }
        let result = {
            let _lifecycle = self
                .inner
                .lifecycle
                .lock()
                .expect("spill lifecycle mutex poisoned");
            let live = self
                .inner
                .live_epochs
                .lock()
                .expect("spill live-epoch mutex poisoned")
                .clone();
            self.sweep_orphans_unlocked(&live)
        };
        #[cfg(feature = "fault-injection")]
        if let Some(barrier) = &race_barrier {
            barrier.mark_sweep_finished();
        }
        result
    }

    fn sweep_orphans_unlocked(
        &self,
        live_epochs: &HashSet<QueryEpoch>,
    ) -> Result<SpillSweepReport, SpillError> {
        let mut report = SpillSweepReport::default();
        for tenant_entry in fs::read_dir(&self.inner.spill_root)
            .map_err(|error| io_error("scan spill tenant directories", error))?
        {
            let tenant_entry =
                tenant_entry.map_err(|error| io_error("read spill tenant entry", error))?;
            let tenant_path = tenant_entry.path();
            let tenant_type = tenant_entry
                .file_type()
                .map_err(|error| io_error("inspect spill tenant entry", error))?;
            if !tenant_type.is_dir() {
                fs::remove_file(&tenant_path)
                    .map_err(|error| io_error("remove unexpected spill entry", error))?;
                report.removed_files += 1;
                continue;
            }
            for epoch_entry in fs::read_dir(&tenant_path)
                .map_err(|error| io_error("scan spill epoch directories", error))?
            {
                let epoch_entry =
                    epoch_entry.map_err(|error| io_error("read spill epoch entry", error))?;
                let epoch_path = epoch_entry.path();
                let epoch_type = epoch_entry
                    .file_type()
                    .map_err(|error| io_error("inspect spill epoch entry", error))?;
                let parsed = epoch_entry
                    .file_name()
                    .to_str()
                    .and_then(QueryEpoch::parse_directory_name);
                if epoch_type.is_dir() && parsed.is_some_and(|epoch| live_epochs.contains(&epoch)) {
                    continue;
                }
                if epoch_type.is_dir() {
                    report.removed_files += count_tree_files(&epoch_path)?;
                    fs::remove_dir_all(&epoch_path)
                        .map_err(|error| io_error("remove orphan spill epoch", error))?;
                    report.removed_directories += 1;
                } else {
                    fs::remove_file(&epoch_path)
                        .map_err(|error| io_error("remove malformed spill epoch entry", error))?;
                    report.removed_files += 1;
                }
            }
            if fs::read_dir(&tenant_path)
                .map_err(|error| io_error("inspect empty spill tenant", error))?
                .next()
                .is_none()
            {
                fs::remove_dir(&tenant_path)
                    .map_err(|error| io_error("remove empty spill tenant", error))?;
                report.removed_directories += 1;
            }
        }
        Ok(report)
    }

    fn maybe_periodic_sweep(&self) -> Result<(), SpillError> {
        let previous = self
            .inner
            .queries_since_sweep
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(if current + 1 >= DEFAULT_ORPHAN_SWEEP_QUERY_INTERVAL {
                    0
                } else {
                    current + 1
                })
            })
            .expect("periodic spill sweep counter update cannot fail");
        if previous + 1 >= DEFAULT_ORPHAN_SWEEP_QUERY_INTERVAL {
            let _ = self.periodic_sweep()?;
        }
        Ok(())
    }

    fn reserve(
        &self,
        tenant_id: TenantId,
        encoded_delta: u64,
        quota_bytes: u64,
        file_bytes_before: u64,
        metadata_units: u64,
    ) -> Result<SpillCharge, SpillError> {
        let mut accounting = self
            .inner
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        let spilled_bytes = accounting
            .tenant_bytes
            .get(&tenant_id)
            .copied()
            .unwrap_or(0);
        let observed = volume_space(&self.inner.data_dir)
            .map_err(|error| io_error("measure spill reservation delta", error))?;
        let allocation_unit = observed.allocation_unit_bytes.max(1);
        let before_allocated = round_up(file_bytes_before, allocation_unit)?;
        let after_file_bytes = file_bytes_before
            .checked_add(encoded_delta)
            .ok_or(SpillError::IdentityExhausted)?;
        let after_allocated = round_up(after_file_bytes, allocation_unit)?;
        let data_delta = after_allocated.saturating_sub(before_allocated);
        let metadata_delta = allocation_unit
            .checked_mul(metadata_units)
            .ok_or(SpillError::IdentityExhausted)?;
        // Quota and headroom both charge the conservative physical volume
        // delta, not merely payload length. This prevents tiny-run/inode and
        // allocation-block rounding from bypassing either cap.
        let volume_delta = data_delta
            .checked_add(metadata_delta)
            .ok_or(SpillError::IdentityExhausted)?;
        let projected =
            spilled_bytes
                .checked_add(volume_delta)
                .ok_or(SpillError::ResourceExhausted {
                    reason: SpillRejectReason::TenantQuota,
                    tenant_id,
                    requested_bytes: volume_delta,
                    spilled_bytes,
                    limit_bytes: quota_bytes,
                    available_bytes: None,
                })?;
        if projected > quota_bytes {
            return Err(SpillError::ResourceExhausted {
                reason: SpillRejectReason::TenantQuota,
                tenant_id,
                requested_bytes: volume_delta,
                spilled_bytes,
                limit_bytes: quota_bytes,
                available_bytes: None,
            });
        }
        let effective_available = observed
            .available_bytes
            .saturating_sub(accounting.pending_volume_bytes);
        let floor = self.inner.headroom.floor_bytes(observed.total_bytes);
        if effective_available.saturating_sub(volume_delta) < floor
            || volume_delta > effective_available
        {
            return Err(SpillError::ResourceExhausted {
                reason: SpillRejectReason::VolumeHeadroom,
                tenant_id,
                requested_bytes: volume_delta,
                spilled_bytes,
                limit_bytes: floor,
                available_bytes: Some(effective_available),
            });
        }

        accounting.tenant_bytes.insert(tenant_id, projected);
        accounting.pending_volume_bytes =
            accounting.pending_volume_bytes.saturating_add(volume_delta);
        Ok(SpillCharge {
            manager: Arc::clone(&self.inner),
            tenant_id,
            bytes: volume_delta,
            pending_volume_bytes: volume_delta,
            pending: true,
        })
    }

    fn is_live(&self, epoch: QueryEpoch) -> bool {
        self.inner
            .live_epochs
            .lock()
            .expect("spill live-epoch mutex poisoned")
            .contains(&epoch)
    }

    fn reserve_staging(
        &self,
        tenant_id: TenantId,
        bytes: u64,
    ) -> Result<StagingCharge, SpillError> {
        let mut accounting = self
            .inner
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        let projected = accounting.staging_memory_bytes.checked_add(bytes).ok_or(
            SpillError::ResourceExhausted {
                reason: SpillRejectReason::SpillStagingMemory,
                tenant_id,
                requested_bytes: bytes,
                spilled_bytes: accounting
                    .tenant_bytes
                    .get(&tenant_id)
                    .copied()
                    .unwrap_or(0),
                limit_bytes: self.inner.staging_memory_limit_bytes,
                available_bytes: Some(
                    self.inner
                        .staging_memory_limit_bytes
                        .saturating_sub(accounting.staging_memory_bytes),
                ),
            },
        )?;
        if projected > self.inner.staging_memory_limit_bytes {
            return Err(SpillError::ResourceExhausted {
                reason: SpillRejectReason::SpillStagingMemory,
                tenant_id,
                requested_bytes: bytes,
                spilled_bytes: accounting
                    .tenant_bytes
                    .get(&tenant_id)
                    .copied()
                    .unwrap_or(0),
                limit_bytes: self.inner.staging_memory_limit_bytes,
                available_bytes: Some(
                    self.inner
                        .staging_memory_limit_bytes
                        .saturating_sub(accounting.staging_memory_bytes),
                ),
            });
        }
        accounting.staging_memory_bytes = projected;
        Ok(StagingCharge {
            manager: Arc::clone(&self.inner),
            bytes,
        })
    }

    fn verify_current_headroom(&self, tenant_id: TenantId) -> Result<(), SpillError> {
        // Use the same accounting-lock -> statvfs order as reserve(). If the
        // sample happened first, a concurrent writer could sync and clear its
        // pending charge before this lock, leaving that allocation absent
        // from both the stale sample and pending_volume_bytes.
        let accounting = self
            .inner
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        let observed = volume_space(&self.inner.data_dir)
            .map_err(|error| io_error("verify spill volume headroom after sync", error))?;
        let effective_available = observed
            .available_bytes
            .saturating_sub(accounting.pending_volume_bytes);
        let floor = self.inner.headroom.floor_bytes(observed.total_bytes);
        if effective_available < floor {
            return Err(SpillError::ResourceExhausted {
                reason: SpillRejectReason::VolumeHeadroom,
                tenant_id,
                requested_bytes: 0,
                spilled_bytes: accounting
                    .tenant_bytes
                    .get(&tenant_id)
                    .copied()
                    .unwrap_or(0),
                limit_bytes: floor,
                available_bytes: Some(effective_available),
            });
        }
        Ok(())
    }

    #[cfg(feature = "fault-injection")]
    fn claim_retained_run(&self) -> bool {
        self.inner
            .retained_runs_claimed
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.inner.retained_run_limit).then_some(current + 1)
            })
            .is_ok()
    }

    #[cfg(feature = "fault-injection")]
    fn wait_at_create_run_tenant_inspection(&self) {
        let barrier = self
            .inner
            .sweep_create_barrier
            .lock()
            .expect("spill sweep/create barrier slot mutex poisoned")
            .clone();
        if let Some(barrier) = barrier {
            let lifecycle_held = match self.inner.lifecycle.try_lock() {
                Ok(lifecycle) => {
                    drop(lifecycle);
                    false
                }
                Err(std::sync::TryLockError::WouldBlock) => true,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    panic!("spill lifecycle mutex poisoned")
                }
            };
            barrier.wait_after_tenant_inspection(lifecycle_held);
        }
    }
}

#[derive(Debug)]
struct QueryIdentity {
    tenant_id: TenantId,
    epoch: QueryEpoch,
    query_dir: PathBuf,
}

struct QueryState {
    epoch: QueryEpoch,
    ended: AtomicBool,
    key: Option<Arc<Mutex<Zeroizing<[u8; AEAD_KEY_LEN]>>>>,
}

impl QueryState {
    fn new(epoch: QueryEpoch, encrypted: bool) -> Result<Self, SpillError> {
        let key = if encrypted {
            // Fill the final zeroizing allocation directly. No non-zero raw
            // stack array is left behind on success or a partial RNG error.
            let mut bytes = Zeroizing::new([0_u8; AEAD_KEY_LEN]);
            fill_random(bytes.as_mut(), "spill query key")?;
            Some(Arc::new(Mutex::new(bytes)))
        } else {
            None
        };
        Ok(Self {
            epoch,
            ended: AtomicBool::new(false),
            key,
        })
    }

    fn ensure_live(&self) -> Result<(), SpillError> {
        if self.ended.load(Ordering::Acquire) {
            Err(SpillError::QueryEnded { epoch: self.epoch })
        } else {
            Ok(())
        }
    }

    fn end(&self) {
        if self.ended.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(key) = &self.key {
            // A panic while a cipher held the key lock must not create a
            // poison escape that leaves key bytes live after query end.
            key.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .zeroize();
        }
    }

    fn allocate_nonce_base(&self, run_number: u32) -> Result<[u8; AES_GCM_IV_LEN], SpillError> {
        self.ensure_live()?;
        for _ in 0..128 {
            let mut base = [0_u8; AES_GCM_IV_LEN];
            fill_random(&mut base, "spill run nonce base")?;
            // Bytes 4..8 make the per-query run domain injective while the
            // other eight bytes remain OS-random. This is stronger than a
            // collision-detected random prefix and needs no query-lifetime
            // HashSet that could grow after sequential runs are reclaimed.
            base[4..8].copy_from_slice(&run_number.to_be_bytes());
            let start = u32::from_be_bytes(base[8..12].try_into().expect("fixed nonce suffix"));
            if start > u32::MAX - MAX_SPILL_FRAMES_PER_RUN {
                continue;
            }
            return Ok(base);
        }
        Err(SpillError::Random {
            purpose: "spill run nonce base with safe counter range",
            reason: "counter-range exhaustion after 128 OS-RNG draws".to_owned(),
        })
    }

    fn encrypt(
        &self,
        nonce: &[u8; AES_GCM_IV_LEN],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SpillError> {
        self.ensure_live()?;
        let key = self.key.as_ref().ok_or_else(|| {
            SpillError::Encryption("encrypted run has no query-scoped key".to_owned())
        })?;
        let key = key
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_live()?;
        let cipher = Aes256GcmCipher::from_key(&key)
            .map_err(|error| SpillError::Encryption(error.to_string()))?;
        cipher
            .encrypt(nonce, aad, plaintext)
            .map_err(|error| SpillError::Encryption(error.to_string()))
    }

    fn decrypt(
        &self,
        nonce: &[u8; AES_GCM_IV_LEN],
        aad: &[u8],
        ciphertext: Vec<u8>,
        chunk: u32,
    ) -> Result<Vec<u8>, SpillError> {
        self.ensure_live()?;
        let key = self.key.as_ref().ok_or_else(|| {
            SpillError::Encryption("encrypted run has no query-scoped key".to_owned())
        })?;
        let key = key
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_live()?;
        let cipher = Aes256GcmCipher::from_key(&key)
            .map_err(|error| SpillError::Encryption(error.to_string()))?;
        cipher
            .decrypt_owned(nonce, aad, ciphertext)
            .map_err(|_| SpillError::AuthenticationFailed { chunk })
    }
}

impl Drop for QueryState {
    fn drop(&mut self) {
        self.end();
    }
}

struct SpillCharge {
    manager: Arc<SpillManagerInner>,
    tenant_id: TenantId,
    bytes: u64,
    pending_volume_bytes: u64,
    pending: bool,
}

impl SpillCharge {
    fn mark_committed(&mut self) {
        if !self.pending {
            return;
        }
        let mut accounting = self
            .manager
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        accounting.pending_volume_bytes = accounting
            .pending_volume_bytes
            .saturating_sub(self.pending_volume_bytes);
        self.pending = false;
    }

    fn absorb(&mut self, mut other: Self) {
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.pending_volume_bytes = self
            .pending_volume_bytes
            .saturating_add(other.pending_volume_bytes);
        other.bytes = 0;
        other.pending_volume_bytes = 0;
        other.pending = false;
    }
}

impl Drop for SpillCharge {
    fn drop(&mut self) {
        let mut accounting = self
            .manager
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        if self.pending {
            accounting.pending_volume_bytes = accounting
                .pending_volume_bytes
                .saturating_sub(self.pending_volume_bytes);
        }
        if let Some(value) = accounting.tenant_bytes.get_mut(&self.tenant_id) {
            *value = value.saturating_sub(self.bytes);
            if *value == 0 {
                accounting.tenant_bytes.remove(&self.tenant_id);
            }
        }
    }
}

struct StagingCharge {
    manager: Arc<SpillManagerInner>,
    bytes: u64,
}

impl Drop for StagingCharge {
    fn drop(&mut self) {
        let mut accounting = self
            .manager
            .accounting
            .lock()
            .expect("spill accounting mutex poisoned");
        accounting.staging_memory_bytes =
            accounting.staging_memory_bytes.saturating_sub(self.bytes);
    }
}

/// One active query attempt. Dropping it invalidates all handles immediately,
/// zeroizes the query key, and removes the epoch from the live registry.
pub struct SpillQuery {
    manager: SpillManager,
    identity: QueryIdentity,
    quota_bytes: u64,
    next_run: AtomicU32,
    /// Two allocation units remain charged for this query's tenant/epoch
    /// directories until query end, even after every run has been consumed.
    directory_charge: Mutex<Option<SpillCharge>>,
    state: Arc<QueryState>,
}

impl fmt::Debug for SpillQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillQuery")
            .field("tenant_id", &self.identity.tenant_id)
            .field("epoch", &self.identity.epoch)
            .field("quota_bytes", &self.quota_bytes)
            .finish_non_exhaustive()
    }
}

impl SpillQuery {
    #[must_use]
    pub const fn epoch(&self) -> QueryEpoch {
        self.identity.epoch
    }

    #[must_use]
    pub const fn quota_bytes(&self) -> u64 {
        self.quota_bytes
    }

    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.manager.spilled_bytes(self.identity.tenant_id)
    }

    pub fn create_run(&self) -> Result<SpillRunWriter, SpillError> {
        self.state.ensure_live()?;
        if !self.manager.is_live(self.identity.epoch) {
            return Err(SpillError::QueryEnded {
                epoch: self.identity.epoch,
            });
        }
        let run_number = self
            .next_run
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| SpillError::IdentityExhausted)?;
        let identity = SpillRunIdentity {
            tenant_id: self.identity.tenant_id,
            query_epoch: self.identity.epoch,
            run_number,
        };
        let encrypted = self.state.key.is_some();
        let nonce_base = if encrypted {
            self.state.allocate_nonce_base(run_number)?
        } else {
            [0_u8; AES_GCM_IV_LEN]
        };
        let mut directory_charge = self
            .directory_charge
            .lock()
            .expect("spill directory-charge mutex poisoned");
        // Keep a new directory reservation local until every fallible step
        // needed to materialize the directory has succeeded. On any early
        // return its guard rolls both tenant and pending-volume accounting
        // back, so a retry cannot inherit a phantom reservation.
        let mut pending_directory_charge = if directory_charge.is_none() {
            Some(
                self.manager
                    .reserve(identity.tenant_id, 0, self.quota_bytes, 0, 2)?,
            )
        } else {
            None
        };
        // One metadata unit covers this run's inode/directory entry. File
        // data is charged from its encoded-length block delta separately.
        let mut charges = vec![self.manager.reserve(
            identity.tenant_id,
            RUN_HEADER_BYTES,
            self.quota_bytes,
            0,
            1,
        )?];
        let tenant_dir = self
            .identity
            .query_dir
            .parent()
            .expect("spill query directory has tenant parent");
        {
            // Serialize only directory materialization against orphan removal.
            // No shared accounting lock is held across this lifecycle edge.
            let _lifecycle = self
                .manager
                .inner
                .lifecycle
                .lock()
                .expect("spill lifecycle mutex poisoned");
            ensure_private_dir_with_inspection_hook(tenant_dir, || {
                #[cfg(feature = "fault-injection")]
                self.manager.wait_at_create_run_tenant_inspection();
            })?;
            ensure_private_dir(&self.identity.query_dir)?;
        }
        if let Some(mut charge) = pending_directory_charge.take() {
            charge.mark_committed();
            *directory_charge = Some(charge);
        }
        drop(directory_charge);
        let path = self
            .identity
            .query_dir
            .join(format!("run-{run_number}.spill"));
        let io_staging_charge = self
            .manager
            .reserve_staging(identity.tenant_id, SPILL_IO_BUFFER_BYTES)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(&path)
            .map_err(|error| io_error("create O_EXCL spill run", error))?;

        #[cfg(feature = "fault-injection")]
        let retain_named = self.manager.claim_retained_run();
        #[cfg(not(feature = "fault-injection"))]
        let retain_named = false;

        #[cfg(unix)]
        let named_path = if retain_named {
            Some(path.clone())
        } else {
            fs::remove_file(&path).map_err(|error| io_error("eagerly unlink spill run", error))?;
            None
        };
        #[cfg(not(unix))]
        let named_path = Some(path.clone());

        let header = encode_run_header(identity, encrypted, 0, nonce_base);
        let mut writer = BufWriter::with_capacity(SPILL_IO_BUFFER_BYTES as usize, file);
        if let Err(error) = writer.write_all(&header) {
            drop(writer);
            if !retain_named {
                if let Some(path) = &named_path {
                    let _ = fs::remove_file(path);
                }
            }
            charges.clear();
            return Err(io_error("write spill run header", error));
        }
        Ok(SpillRunWriter {
            file: Some(writer),
            named_path,
            retain_named,
            identity,
            encrypted,
            nonce_base,
            next_chunk: 0,
            encoded_bytes: RUN_HEADER_BYTES,
            io_staging_charge: Some(io_staging_charge),
            manager: self.manager.clone(),
            quota_bytes: self.quota_bytes,
            state: Arc::clone(&self.state),
            charges,
            poisoned: false,
        })
    }

    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub fn key_zeroize_probe_for_test(&self) -> Option<SpillKeyZeroizeProbe> {
        self.state.key.as_ref().map(|bytes| SpillKeyZeroizeProbe {
            bytes: Arc::clone(bytes),
        })
    }
}

impl Drop for SpillQuery {
    fn drop(&mut self) {
        self.state.end();
        let _lifecycle = self
            .manager
            .inner
            .lifecycle
            .lock()
            .expect("spill lifecycle mutex poisoned");
        self.manager
            .inner
            .live_epochs
            .lock()
            .expect("spill live-epoch mutex poisoned")
            .remove(&self.identity.epoch);
        // Eager-unlinked query directories are empty. Retained/fallback runs
        // remain named until their bounded test scan or periodic sweep.
        let _ = fs::remove_dir(&self.identity.query_dir);
        if let Some(tenant_dir) = self.identity.query_dir.parent() {
            let _ = fs::remove_dir(tenant_dir);
        }
    }
}

/// Bounded cfg-only observer used by the security gate. It references the
/// actual guarded key allocation (not a mirror) and observes query-end zeros.
#[cfg(feature = "fault-injection")]
#[derive(Clone)]
pub struct SpillKeyZeroizeProbe {
    bytes: Arc<Mutex<Zeroizing<[u8; AEAD_KEY_LEN]>>>,
}

#[cfg(feature = "fault-injection")]
impl SpillKeyZeroizeProbe {
    #[must_use]
    pub fn snapshot(&self) -> [u8; AEAD_KEY_LEN] {
        **self
            .bytes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[must_use]
    pub fn is_zeroized(&self) -> bool {
        self.snapshot() == [0_u8; AEAD_KEY_LEN]
    }
}

/// Append-only framed run writer.
pub struct SpillRunWriter {
    file: Option<BufWriter<File>>,
    named_path: Option<PathBuf>,
    retain_named: bool,
    identity: SpillRunIdentity,
    encrypted: bool,
    nonce_base: [u8; AES_GCM_IV_LEN],
    next_chunk: u32,
    encoded_bytes: u64,
    io_staging_charge: Option<StagingCharge>,
    manager: SpillManager,
    quota_bytes: u64,
    state: Arc<QueryState>,
    charges: Vec<SpillCharge>,
    poisoned: bool,
}

impl fmt::Debug for SpillRunWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillRunWriter")
            .field("identity", &self.identity)
            .field("encrypted", &self.encrypted)
            .field("frames", &self.next_chunk)
            .finish_non_exhaustive()
    }
}

impl SpillRunWriter {
    #[must_use]
    pub const fn identity(&self) -> SpillRunIdentity {
        self.identity
    }

    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.encrypted
    }

    pub fn append_batch(&mut self, batch: &[u8]) -> Result<(), SpillError> {
        self.state.ensure_live()?;
        if self.poisoned {
            return Err(SpillError::CorruptFrame {
                chunk: self.next_chunk,
                reason: "writer was poisoned by an earlier failed write".to_owned(),
            });
        }
        if batch.len() > MAX_SPILL_BATCH_BYTES {
            return Err(SpillError::BatchTooLarge {
                len: batch.len(),
                max: MAX_SPILL_BATCH_BYTES,
            });
        }
        #[cfg(feature = "fault-injection")]
        if self.retain_named
            && self
                .encoded_bytes
                .saturating_add(FRAME_HEADER_BYTES)
                .saturating_add(batch.len() as u64)
                .saturating_add(u64::from(self.encrypted) * AES_GCM_TAG_LEN as u64)
                > MAX_RETAINED_TEST_RUN_BYTES
        {
            return Err(SpillError::BatchTooLarge {
                len: batch.len(),
                max: MAX_RETAINED_TEST_RUN_BYTES as usize,
            });
        }
        if self.next_chunk >= MAX_SPILL_FRAMES_PER_RUN {
            return Err(SpillError::NonceExhausted);
        }
        let plaintext_len = u32::try_from(batch.len()).map_err(|_| SpillError::BatchTooLarge {
            len: batch.len(),
            max: MAX_SPILL_BATCH_BYTES,
        })?;
        let chunk = self.next_chunk;
        let stored_len = if self.encrypted {
            plaintext_len
                .checked_add(AES_GCM_TAG_LEN as u32)
                .ok_or(SpillError::IdentityExhausted)?
        } else {
            plaintext_len
        };
        let frame_bytes = FRAME_HEADER_BYTES
            .checked_add(u64::from(stored_len))
            .ok_or(SpillError::IdentityExhausted)?;
        let charge = self.manager.reserve(
            self.identity.tenant_id,
            frame_bytes,
            self.quota_bytes,
            self.encoded_bytes,
            0,
        )?;
        let _staging = self
            .manager
            .reserve_staging(self.identity.tenant_id, u64::from(stored_len))?;
        let payload = if self.encrypted {
            let nonce = chunk_nonce(self.nonce_base, chunk)?;
            let aad = frame_aad(self.identity, true, chunk, plaintext_len);
            match self.state.encrypt(&nonce, &aad, batch) {
                Ok(payload) => payload,
                Err(error) => {
                    // Once passed to AES-GCM, this nonce is treated as
                    // consumed even if the provider reports failure.
                    self.poisoned = true;
                    return Err(error);
                }
            }
        } else {
            batch.to_vec()
        };
        if payload.len() != stored_len as usize {
            self.poisoned = true;
            return Err(SpillError::Encryption(
                "spill cipher returned an unexpected payload length".to_owned(),
            ));
        }
        // Aggregate into one run guard: frame count cannot turn into an
        // unbounded in-RAM vector of accounting guards. A partial I/O error
        // still keeps the whole charge until the poisoned run closes.
        self.charges
            .first_mut()
            .expect("spill header charge present")
            .absorb(charge);

        let mut prefix = [0_u8; FRAME_HEADER_BYTES as usize];
        prefix[..4].copy_from_slice(&stored_len.to_le_bytes());
        prefix[4..8].copy_from_slice(&plaintext_len.to_le_bytes());
        prefix[8..12].copy_from_slice(&chunk.to_le_bytes());
        if !self.encrypted {
            let mut crc = crc32c::crc32c(&prefix[..12]);
            crc = crc32c::crc32c_append(crc, &payload);
            prefix[12..16].copy_from_slice(&crc.to_le_bytes());
        }
        let file = self.file.as_mut().expect("spill writer file present");
        if let Err(error) = file
            .write_all(&prefix)
            .and_then(|()| file.write_all(&payload))
        {
            self.poisoned = true;
            return Err(io_error("append spill frame", error));
        }
        self.next_chunk = self
            .next_chunk
            .checked_add(1)
            .ok_or(SpillError::NonceExhausted)?;
        self.encoded_bytes = self
            .encoded_bytes
            .checked_add(frame_bytes)
            .ok_or(SpillError::IdentityExhausted)?;
        Ok(())
    }

    #[cfg(feature = "fault-injection")]
    pub fn exhaust_chunk_counter_for_test(&mut self) {
        self.next_chunk = MAX_SPILL_FRAMES_PER_RUN;
    }

    pub fn finish(mut self) -> Result<SpillRun, SpillError> {
        self.state.ensure_live()?;
        if self.poisoned {
            return Err(SpillError::CorruptFrame {
                chunk: self.next_chunk,
                reason: "cannot publish a poisoned spill writer".to_owned(),
            });
        }
        let mut buffered = self.file.take().expect("spill writer file present");
        buffered
            .flush()
            .map_err(|error| io_error("flush spill frames", error))?;
        let mut file = buffered
            .into_inner()
            .map_err(|error| io_error("extract flushed spill file", error.into_error()))?;
        drop(self.io_staging_charge.take());
        file.seek(SeekFrom::Start(0))
            .map_err(|error| io_error("seek spill header", error))?;
        let header = encode_run_header(
            self.identity,
            self.encrypted,
            u64::from(self.next_chunk),
            self.nonce_base,
        );
        file.write_all(&header)
            .map_err(|error| io_error("seal spill run header", error))?;
        file.sync_data()
            .map_err(|error| io_error("sync spill run", error))?;
        for charge in &mut self.charges {
            charge.mark_committed();
        }
        if let Err(error) = self
            .manager
            .verify_current_headroom(self.identity.tenant_id)
        {
            // A WAL/other-writer race can consume space after preflight. Close
            // and reclaim this run, including a cfg-retained name, then fail
            // only the query rather than leaving the volume below its floor.
            drop(file);
            if let Some(path) = self.named_path.take() {
                let _ = fs::remove_file(path);
            }
            self.retain_named = false;
            return Err(error);
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| io_error("rewind spill run", error))?;
        Ok(SpillRun {
            file: Some(file),
            named_path: self.named_path.take(),
            retain_named: self.retain_named,
            identity: self.identity,
            encrypted: self.encrypted,
            nonce_base: self.nonce_base,
            frame_count: self.next_chunk,
            manager: self.manager.clone(),
            state: Arc::clone(&self.state),
            charges: std::mem::take(&mut self.charges),
        })
    }
}

impl Drop for SpillRunWriter {
    fn drop(&mut self) {
        self.file.take();
        if !self.retain_named {
            if let Some(path) = self.named_path.take() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// Sealed fd-owned spill run. It is single-consumer: `into_reader` transfers
/// the open descriptor and quota charge to its streaming reader.
pub struct SpillRun {
    file: Option<File>,
    named_path: Option<PathBuf>,
    retain_named: bool,
    identity: SpillRunIdentity,
    encrypted: bool,
    nonce_base: [u8; AES_GCM_IV_LEN],
    frame_count: u32,
    manager: SpillManager,
    state: Arc<QueryState>,
    charges: Vec<SpillCharge>,
}

impl fmt::Debug for SpillRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillRun")
            .field("identity", &self.identity)
            .field("encrypted", &self.encrypted)
            .field("frame_count", &self.frame_count)
            .finish_non_exhaustive()
    }
}

impl SpillRun {
    #[must_use]
    pub const fn identity(&self) -> SpillRunIdentity {
        self.identity
    }

    #[must_use]
    pub const fn frame_count(&self) -> u32 {
        self.frame_count
    }

    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.encrypted
    }

    pub fn into_reader(mut self, active_epoch: QueryEpoch) -> Result<SpillRunReader, SpillError> {
        if active_epoch != self.identity.query_epoch {
            return Err(SpillError::StaleEpoch {
                active_epoch,
                run_epoch: self.identity.query_epoch,
            });
        }
        self.state.ensure_live()?;
        if !self.manager.is_live(active_epoch) {
            return Err(SpillError::QueryEnded {
                epoch: active_epoch,
            });
        }
        let io_staging_charge = self
            .manager
            .reserve_staging(self.identity.tenant_id, SPILL_IO_BUFFER_BYTES)?;
        let file = self.file.take().expect("sealed spill file present");
        let mut reader = SpillRunReader {
            file: Some(BufReader::with_capacity(
                SPILL_IO_BUFFER_BYTES as usize,
                file,
            )),
            io_staging_charge: Some(io_staging_charge),
            named_path: self.named_path.take(),
            retain_named: self.retain_named,
            identity: self.identity,
            encrypted: self.encrypted,
            nonce_base: self.nonce_base,
            remaining: self.frame_count,
            next_chunk: 0,
            eof_checked: false,
            poisoned: false,
            manager: self.manager.clone(),
            state: Arc::clone(&self.state),
            charges: std::mem::take(&mut self.charges),
        };
        reader.validate_header()?;
        Ok(reader)
    }

    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub fn retained_path_for_test(&self) -> Option<&Path> {
        self.named_path.as_deref()
    }

    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub const fn nonce_base_for_test(&self) -> [u8; AES_GCM_IV_LEN] {
        self.nonce_base
    }

    #[cfg(feature = "fault-injection")]
    pub fn nonce_for_chunk_for_test(&self, chunk: u32) -> Result<[u8; AES_GCM_IV_LEN], SpillError> {
        chunk_nonce(self.nonce_base, chunk)
    }

    #[cfg(feature = "fault-injection")]
    pub fn corrupt_first_payload_byte_for_test(&mut self) -> Result<(), SpillError> {
        let file = self.file.as_mut().expect("sealed spill file present");
        file.seek(SeekFrom::Start(RUN_HEADER_BYTES + FRAME_HEADER_BYTES))
            .map_err(|error| io_error("seek spill ciphertext fault", error))?;
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte)
            .map_err(|error| io_error("read spill ciphertext fault byte", error))?;
        byte[0] ^= 0x80;
        file.seek(SeekFrom::Current(-1))
            .map_err(|error| io_error("rewind spill ciphertext fault byte", error))?;
        file.write_all(&byte)
            .map_err(|error| io_error("write spill ciphertext fault byte", error))?;
        file.sync_data()
            .map_err(|error| io_error("sync spill ciphertext fault", error))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| io_error("rewind corrupted spill run", error))?;
        Ok(())
    }
}

impl Drop for SpillRun {
    fn drop(&mut self) {
        self.file.take();
        if !self.retain_named {
            if let Some(path) = self.named_path.take() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// Streaming framed run reader.
pub struct SpillRunReader {
    file: Option<BufReader<File>>,
    io_staging_charge: Option<StagingCharge>,
    named_path: Option<PathBuf>,
    retain_named: bool,
    identity: SpillRunIdentity,
    encrypted: bool,
    nonce_base: [u8; AES_GCM_IV_LEN],
    remaining: u32,
    next_chunk: u32,
    eof_checked: bool,
    poisoned: bool,
    manager: SpillManager,
    state: Arc<QueryState>,
    charges: Vec<SpillCharge>,
}

/// One restored record batch with its spill-staging reservation attached.
///
/// Consumers deserialize from the borrowed byte slice and then drop this
/// guard. Holding several restored batches therefore remains bounded by the
/// process spill-staging line; there is deliberately no uncharged `into_vec`
/// escape hatch. The executor's serialized input passed to the writer remains
/// owned and budgeted by the caller.
pub struct SpillBatch {
    bytes: Vec<u8>,
    _staging_charge: StagingCharge,
}

impl fmt::Debug for SpillBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillBatch")
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl AsRef<[u8]> for SpillBatch {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Deref for SpillBatch {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl fmt::Debug for SpillRunReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillRunReader")
            .field("identity", &self.identity)
            .field("remaining", &self.remaining)
            .finish_non_exhaustive()
    }
}

impl SpillRunReader {
    fn validate_header(&mut self) -> Result<(), SpillError> {
        let file = self.file.as_mut().expect("spill reader file present");
        let mut bytes = [0_u8; RUN_HEADER_BYTES as usize];
        file.read_exact(&mut bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                SpillError::InvalidHeader("truncated run header".to_owned())
            } else {
                io_error("read spill run header", error)
            }
        })?;
        let header = decode_run_header(&bytes)?;
        if header.identity != self.identity
            || header.encrypted != self.encrypted
            || header.nonce_base != self.nonce_base
            || header.frame_count != u64::from(self.remaining)
        {
            return Err(SpillError::InvalidHeader(
                "fd header identity/metadata differs from its run handle".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn next_batch(&mut self) -> Result<Option<SpillBatch>, SpillError> {
        if self.poisoned {
            return Err(SpillError::CorruptFrame {
                chunk: self.next_chunk,
                reason: "spill reader is terminal after an earlier restore failure".to_owned(),
            });
        }
        let result = self.next_batch_inner();
        if result.is_err() {
            // Reads are sequential. Any error may have consumed a partial
            // header/payload (or the trailing-byte probe), so retrying from
            // the fd's new offset could turn corruption into apparent EOF.
            self.poisoned = true;
        }
        result
    }

    fn next_batch_inner(&mut self) -> Result<Option<SpillBatch>, SpillError> {
        self.state.ensure_live()?;
        if !self.manager.is_live(self.identity.query_epoch) {
            return Err(SpillError::QueryEnded {
                epoch: self.identity.query_epoch,
            });
        }
        if self.remaining == 0 {
            self.check_eof()?;
            return Ok(None);
        }
        let expected_chunk = self.next_chunk;
        let file = self.file.as_mut().expect("spill reader file present");
        let mut prefix = [0_u8; FRAME_HEADER_BYTES as usize];
        file.read_exact(&mut prefix).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                SpillError::CorruptFrame {
                    chunk: expected_chunk,
                    reason: "torn frame header".to_owned(),
                }
            } else {
                io_error("read spill frame header", error)
            }
        })?;
        let stored_len = u32::from_le_bytes(prefix[..4].try_into().expect("fixed stored length"));
        let plaintext_len =
            u32::from_le_bytes(prefix[4..8].try_into().expect("fixed plaintext length"));
        let chunk = u32::from_le_bytes(prefix[8..12].try_into().expect("fixed chunk counter"));
        let integrity = u32::from_le_bytes(prefix[12..16].try_into().expect("fixed integrity"));
        if chunk != expected_chunk {
            return Err(SpillError::CorruptFrame {
                chunk: expected_chunk,
                reason: format!("non-monotonic chunk counter {chunk}"),
            });
        }
        if plaintext_len as usize > MAX_SPILL_BATCH_BYTES {
            return Err(SpillError::BatchTooLarge {
                len: plaintext_len as usize,
                max: MAX_SPILL_BATCH_BYTES,
            });
        }
        let expected_stored = if self.encrypted {
            plaintext_len
                .checked_add(AES_GCM_TAG_LEN as u32)
                .ok_or(SpillError::CorruptFrame {
                    chunk,
                    reason: "encrypted stored length overflow".to_owned(),
                })?
        } else {
            plaintext_len
        };
        if stored_len != expected_stored {
            return Err(SpillError::CorruptFrame {
                chunk,
                reason: format!(
                    "stored length {stored_len} does not match expected {expected_stored}"
                ),
            });
        }
        if self.encrypted && integrity != 0 {
            return Err(SpillError::CorruptFrame {
                chunk,
                reason: "encrypted frame integrity-reserved field is non-zero".to_owned(),
            });
        }
        // AES-GCM decrypts this owned allocation in place. The charge moves
        // into SpillBatch and lives until the consumer releases the bytes.
        let staging = self
            .manager
            .reserve_staging(self.identity.tenant_id, u64::from(stored_len))?;
        let mut payload = vec![0_u8; stored_len as usize];
        file.read_exact(&mut payload).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                SpillError::CorruptFrame {
                    chunk,
                    reason: "torn frame payload".to_owned(),
                }
            } else {
                io_error("read spill frame payload", error)
            }
        })?;
        let plaintext = if self.encrypted {
            let nonce = chunk_nonce(self.nonce_base, chunk)?;
            let aad = frame_aad(self.identity, true, chunk, plaintext_len);
            self.state.decrypt(&nonce, &aad, payload, chunk)?
        } else {
            let mut actual_crc = crc32c::crc32c(&prefix[..12]);
            actual_crc = crc32c::crc32c_append(actual_crc, &payload);
            if integrity != actual_crc {
                return Err(SpillError::CorruptFrame {
                    chunk,
                    reason: "clear-frame checksum mismatch".to_owned(),
                });
            }
            payload
        };
        if plaintext.len() != plaintext_len as usize {
            return Err(SpillError::CorruptFrame {
                chunk,
                reason: "restored plaintext length changed".to_owned(),
            });
        }
        self.remaining -= 1;
        self.next_chunk = self
            .next_chunk
            .checked_add(1)
            .ok_or(SpillError::NonceExhausted)?;
        Ok(Some(SpillBatch {
            bytes: plaintext,
            _staging_charge: staging,
        }))
    }

    fn check_eof(&mut self) -> Result<(), SpillError> {
        if self.eof_checked {
            return Ok(());
        }
        let mut trailing = [0_u8; 1];
        if self
            .file
            .as_mut()
            .expect("spill reader file present")
            .read(&mut trailing)
            .map_err(|error| io_error("check spill run trailing bytes", error))?
            != 0
        {
            return Err(SpillError::CorruptFrame {
                chunk: self.next_chunk,
                reason: "trailing bytes after declared frame count".to_owned(),
            });
        }
        self.eof_checked = true;
        Ok(())
    }
}

impl Drop for SpillRunReader {
    fn drop(&mut self) {
        self.file.take();
        drop(self.io_staging_charge.take());
        if !self.retain_named {
            if let Some(path) = self.named_path.take() {
                let _ = fs::remove_file(path);
            }
        }
        // Read explicitly so the charge field is clearly intentional: its
        // guards release quota only after this reader's fd has closed.
        let _ = self.charges.len();
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillSweepReport {
    pub removed_files: u64,
    pub removed_directories: u64,
}

#[derive(Clone, Copy)]
struct DecodedRunHeader {
    identity: SpillRunIdentity,
    encrypted: bool,
    frame_count: u64,
    nonce_base: [u8; AES_GCM_IV_LEN],
}

fn encode_run_header(
    identity: SpillRunIdentity,
    encrypted: bool,
    frame_count: u64,
    nonce_base: [u8; AES_GCM_IV_LEN],
) -> [u8; RUN_HEADER_BYTES as usize] {
    let mut bytes = [0_u8; RUN_HEADER_BYTES as usize];
    bytes[..8].copy_from_slice(&RUN_MAGIC);
    bytes[8..10].copy_from_slice(&RUN_VERSION.to_le_bytes());
    bytes[10..12].copy_from_slice(&(u16::from(encrypted) * FLAG_ENCRYPTED).to_le_bytes());
    bytes[12..20].copy_from_slice(&identity.tenant_id.raw().to_le_bytes());
    bytes[20..36].copy_from_slice(&identity.query_epoch.engine_boot_id.to_le_bytes());
    bytes[36..44].copy_from_slice(&identity.query_epoch.generation.to_le_bytes());
    bytes[44..52].copy_from_slice(&identity.query_epoch.query_id.to_le_bytes());
    bytes[52..56].copy_from_slice(&identity.query_epoch.attempt.to_le_bytes());
    bytes[56..60].copy_from_slice(&identity.run_number.to_le_bytes());
    bytes[60..68].copy_from_slice(&frame_count.to_le_bytes());
    bytes[68..80].copy_from_slice(&nonce_base);
    let crc = crc32c::crc32c(&bytes[..80]);
    bytes[80..84].copy_from_slice(&crc.to_le_bytes());
    bytes
}

fn decode_run_header(
    bytes: &[u8; RUN_HEADER_BYTES as usize],
) -> Result<DecodedRunHeader, SpillError> {
    if bytes[..8] != RUN_MAGIC {
        return Err(SpillError::InvalidHeader("bad spill magic".to_owned()));
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().expect("fixed version"));
    if version != RUN_VERSION {
        return Err(SpillError::InvalidHeader(format!(
            "unsupported version {version}"
        )));
    }
    let flags = u16::from_le_bytes(bytes[10..12].try_into().expect("fixed flags"));
    if flags & !FLAG_ENCRYPTED != 0 {
        return Err(SpillError::InvalidHeader(format!(
            "unknown flags 0x{flags:04x}"
        )));
    }
    let expected_crc = u32::from_le_bytes(bytes[80..84].try_into().expect("fixed header crc"));
    if crc32c::crc32c(&bytes[..80]) != expected_crc {
        return Err(SpillError::InvalidHeader(
            "run header checksum mismatch".to_owned(),
        ));
    }
    let frame_count = u64::from_le_bytes(bytes[60..68].try_into().expect("fixed frame count"));
    if frame_count > u64::from(MAX_SPILL_FRAMES_PER_RUN) {
        return Err(SpillError::InvalidHeader(format!(
            "frame count {frame_count} exceeds bound {MAX_SPILL_FRAMES_PER_RUN}"
        )));
    }
    let epoch = QueryEpoch::new(
        u128::from_le_bytes(bytes[20..36].try_into().expect("fixed boot id")),
        u64::from_le_bytes(bytes[36..44].try_into().expect("fixed generation")),
        u64::from_le_bytes(bytes[44..52].try_into().expect("fixed query id")),
        u32::from_le_bytes(bytes[52..56].try_into().expect("fixed attempt")),
    );
    Ok(DecodedRunHeader {
        identity: SpillRunIdentity {
            tenant_id: TenantId::new(u64::from_le_bytes(
                bytes[12..20].try_into().expect("fixed tenant id"),
            )),
            query_epoch: epoch,
            run_number: u32::from_le_bytes(bytes[56..60].try_into().expect("fixed run number")),
        },
        encrypted: flags & FLAG_ENCRYPTED != 0,
        frame_count,
        nonce_base: bytes[68..80].try_into().expect("fixed nonce base"),
    })
}

fn frame_aad(
    identity: SpillRunIdentity,
    encrypted: bool,
    chunk: u32,
    plaintext_len: u32,
) -> [u8; FRAME_AAD_BYTES] {
    let mut aad = [0_u8; FRAME_AAD_BYTES];
    aad[..8].copy_from_slice(&RUN_MAGIC);
    aad[8..10].copy_from_slice(&RUN_VERSION.to_le_bytes());
    aad[10..12].copy_from_slice(&(u16::from(encrypted) * FLAG_ENCRYPTED).to_le_bytes());
    aad[12..20].copy_from_slice(&identity.tenant_id.raw().to_le_bytes());
    aad[20..36].copy_from_slice(&identity.query_epoch.engine_boot_id.to_le_bytes());
    aad[36..44].copy_from_slice(&identity.query_epoch.generation.to_le_bytes());
    aad[44..52].copy_from_slice(&identity.query_epoch.query_id.to_le_bytes());
    aad[52..56].copy_from_slice(&identity.query_epoch.attempt.to_le_bytes());
    aad[56..60].copy_from_slice(&identity.run_number.to_le_bytes());
    aad[60..64].copy_from_slice(&chunk.to_le_bytes());
    aad[64..68].copy_from_slice(&plaintext_len.to_le_bytes());
    aad
}

fn chunk_nonce(base: [u8; AES_GCM_IV_LEN], chunk: u32) -> Result<[u8; AES_GCM_IV_LEN], SpillError> {
    let start = u32::from_be_bytes(base[8..12].try_into().expect("fixed nonce suffix"));
    let value = start.checked_add(chunk).ok_or(SpillError::NonceExhausted)?;
    let mut nonce = base;
    nonce[8..12].copy_from_slice(&value.to_be_bytes());
    Ok(nonce)
}

fn fill_random(bytes: &mut [u8], purpose: &'static str) -> Result<(), SpillError> {
    SystemRandom::new()
        .fill(bytes)
        .map_err(|error| SpillError::Random {
            purpose,
            reason: format!("{error:?}"),
        })
}

fn round_up(bytes: u64, unit: u64) -> Result<u64, SpillError> {
    if bytes == 0 {
        return Ok(0);
    }
    bytes
        .checked_add(unit.saturating_sub(1))
        .map(|value| value / unit * unit)
        .ok_or(SpillError::IdentityExhausted)
}

fn ensure_private_dir(path: &Path) -> Result<(), SpillError> {
    ensure_private_dir_with_inspection_hook(path, || {})
}

fn ensure_private_dir_with_inspection_hook(
    path: &Path,
    after_inspection: impl FnOnce(),
) -> Result<(), SpillError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(SpillError::InvalidConfig(format!(
                "spill directory {} is not a real directory",
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|error| io_error("create spill directory", error))?;
        }
        Err(error) => return Err(io_error("inspect spill directory", error)),
    }
    after_inspection();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| io_error("protect spill directory", error))?;
    }
    Ok(())
}

fn count_tree_files(path: &Path) -> Result<u64, SpillError> {
    let mut count = 0_u64;
    for entry in fs::read_dir(path).map_err(|error| io_error("count orphan spill files", error))? {
        let entry = entry.map_err(|error| io_error("read orphan spill entry", error))?;
        let file_type = entry
            .file_type()
            .map_err(|error| io_error("inspect orphan spill entry", error))?;
        if file_type.is_dir() {
            count = count.saturating_add(count_tree_files(&entry.path())?);
        } else {
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

#[cfg(unix)]
fn volume_space(path: &Path) -> io::Result<VolumeSpace> {
    use std::os::unix::ffi::OsStrExt;

    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "spill data path contains an interior NUL",
        )
    })?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a live NUL-terminated CString and `stats` points to
    // writable aligned storage. `statvfs` retains neither pointer; the output
    // is read only after a successful return.
    let rc = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: rc == 0 means statvfs initialized the output structure.
    let stats = unsafe { stats.assume_init() };
    #[allow(clippy::unnecessary_cast)]
    let fragment = stats.f_frsize as u64;
    #[allow(clippy::unnecessary_cast)]
    let available = (stats.f_bavail as u64).saturating_mul(fragment);
    #[allow(clippy::unnecessary_cast)]
    let total = (stats.f_blocks as u64).saturating_mul(fragment);
    Ok(VolumeSpace {
        available_bytes: available,
        total_bytes: total,
        allocation_unit_bytes: fragment.max(1),
    })
}

#[cfg(not(unix))]
fn volume_space(_path: &Path) -> io::Result<VolumeSpace> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "spill volume headroom measurement is not implemented on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_run_round_trips_and_releases_quota() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 7, 0, 1024 * 1024))
            .unwrap();
        let mut writer = query.create_run().unwrap();
        #[cfg(unix)]
        assert_eq!(
            count_tree_files(manager.spill_root()).unwrap(),
            0,
            "ordinary POSIX scratch must be eagerly unlinked while its fd stays usable"
        );
        writer.append_batch(b"first").unwrap();
        writer.append_batch(b"second").unwrap();
        let run = writer.finish().unwrap();
        assert_eq!(run.frame_count(), 2);
        let mut reader = run.into_reader(query.epoch()).unwrap();
        assert_eq!(reader.next_batch().unwrap().unwrap().as_ref(), b"first");
        assert_eq!(reader.next_batch().unwrap().unwrap().as_ref(), b"second");
        assert!(reader.next_batch().unwrap().is_none());
        drop(reader);
        assert!(query.spilled_bytes() > 0, "query directories stay charged");
        drop(query);
        assert_eq!(manager.spilled_bytes(TenantId::DEFAULT), 0);
    }

    #[test]
    fn default_quota_is_four_times_executor_budget() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 8, 0, 1234))
            .unwrap();
        assert_eq!(query.quota_bytes(), 4936);
    }

    #[test]
    fn quota_reject_carries_spilled_bytes_before_write() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let unit = manager.volume_space().unwrap().allocation_unit_bytes;
        let header_charge = unit * 4; // three metadata units + one data block
        let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 9, 0, header_charge);
        config.spill_quota_bytes = Some(header_charge + unit);
        let query = manager.begin_query(config).unwrap();
        let mut writer = query.create_run().unwrap();
        let batch = vec![0xA5; unit as usize];
        writer.append_batch(&batch).unwrap();
        let before = query.spilled_bytes();
        let error = writer.append_batch(&batch).unwrap_err();
        assert!(matches!(
            error,
            SpillError::ResourceExhausted {
                reason: SpillRejectReason::TenantQuota,
                spilled_bytes,
                ..
            } if spilled_bytes == before
        ));
    }

    #[test]
    fn failed_run_setup_rolls_back_pending_directory_reservation() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let unit = manager.volume_space().unwrap().allocation_unit_bytes;
        let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 14, 0, unit * 2);
        // Exactly the two directory metadata units fit, but the run inode +
        // header block do not. A failed setup must not publish that local
        // pending charge into the query or poison a retry.
        config.spill_quota_bytes = Some(unit * 2);
        let query = manager.begin_query(config).unwrap();
        for _ in 0..2 {
            let error = query.create_run().expect_err("run header exceeds quota");
            assert!(matches!(
                error,
                SpillError::ResourceExhausted {
                    reason: SpillRejectReason::TenantQuota,
                    ..
                }
            ));
            assert_eq!(query.spilled_bytes(), 0);
        }
    }

    #[test]
    fn epoch_directory_encoding_round_trips() {
        let epoch = QueryEpoch::new(u128::MAX - 1, 42, 99, 3);
        assert_eq!(
            QueryEpoch::parse_directory_name(&epoch.directory_name()),
            Some(epoch)
        );
    }

    #[test]
    fn startup_sweep_removes_prior_boot_scratch() {
        let root = tempfile::tempdir().unwrap();
        let orphan = root.path().join("spill/1/not-an-active-epoch");
        fs::create_dir_all(&orphan).unwrap();
        fs::write(orphan.join("run-0.spill"), b"prior boot").unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        assert_eq!(count_tree_files(manager.spill_root()).unwrap(), 0);
    }

    #[test]
    fn run_creation_is_exclusive_and_never_overwrites() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 10, 0, 1024 * 1024))
            .unwrap();
        let epoch_dir = manager
            .spill_root()
            .join(TenantId::DEFAULT.raw().to_string())
            .join(query.epoch().directory_name());
        ensure_private_dir(&epoch_dir).unwrap();
        let collision = epoch_dir.join("run-0.spill");
        fs::write(&collision, b"must survive").unwrap();
        assert!(matches!(query.create_run(), Err(SpillError::Io { .. })));
        assert_eq!(fs::read(collision).unwrap(), b"must survive");
    }

    #[test]
    fn spill_staging_memory_is_process_bounded_before_payload_allocation() {
        let root = tempfile::tempdir().unwrap();
        let mut manager_config = SpillManagerConfig::new(root.path());
        manager_config.staging_memory_limit_bytes = SPILL_IO_BUFFER_BYTES;
        let manager = SpillManager::new(manager_config).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 11, 0, 1024 * 1024))
            .unwrap();
        let mut writer = query.create_run().unwrap();
        assert!(matches!(
            writer.append_batch(b"staging must bite"),
            Err(SpillError::ResourceExhausted {
                reason: SpillRejectReason::SpillStagingMemory,
                ..
            })
        ));
    }

    #[test]
    fn restored_batch_holds_its_staging_charge_until_consumed() {
        let root = tempfile::tempdir().unwrap();
        let payload = b"restored bytes stay charged";
        let mut manager_config = SpillManagerConfig::new(root.path());
        manager_config.staging_memory_limit_bytes = SPILL_IO_BUFFER_BYTES + payload.len() as u64;
        let manager = SpillManager::new(manager_config).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 12, 0, 1024 * 1024))
            .unwrap();
        let mut writer = query.create_run().unwrap();
        writer.append_batch(payload).unwrap();
        let run = writer.finish().unwrap();
        let mut reader = run.into_reader(query.epoch()).unwrap();
        let restored = reader.next_batch().unwrap().unwrap();
        assert_eq!(restored.as_ref(), payload);
        assert_eq!(
            manager
                .inner
                .accounting
                .lock()
                .unwrap()
                .staging_memory_bytes,
            SPILL_IO_BUFFER_BYTES + payload.len() as u64
        );
        drop(restored);
        assert_eq!(
            manager
                .inner
                .accounting
                .lock()
                .unwrap()
                .staging_memory_bytes,
            SPILL_IO_BUFFER_BYTES
        );
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn corruption_is_terminal_and_cannot_be_retried_into_eof() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let query = manager
            .begin_query(SpillQueryConfig::new(TenantId::DEFAULT, 13, 0, 1024 * 1024))
            .unwrap();
        let mut writer = query.create_run().unwrap();
        writer.append_batch(b"one frame").unwrap();
        let mut run = writer.finish().unwrap();
        let file = run.file.as_mut().expect("sealed run fd");
        file.seek(SeekFrom::End(0)).unwrap();
        file.write_all(&[0xA5]).unwrap();
        file.sync_data().unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut reader = run.into_reader(query.epoch()).unwrap();
        assert_eq!(reader.next_batch().unwrap().unwrap().as_ref(), b"one frame");
        assert!(matches!(
            reader.next_batch(),
            Err(SpillError::CorruptFrame { .. })
        ));
        assert!(matches!(
            reader.next_batch(),
            Err(SpillError::CorruptFrame { .. })
        ));
    }

    #[test]
    fn production_query_cadence_invokes_periodic_orphan_sweep() {
        let root = tempfile::tempdir().unwrap();
        let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
        let orphan = manager.spill_root().join("1/stale-cadence-epoch");
        fs::create_dir_all(&orphan).unwrap();
        fs::write(orphan.join("run-0.spill"), b"stale").unwrap();
        for query_id in 0..DEFAULT_ORPHAN_SWEEP_QUERY_INTERVAL {
            drop(
                manager
                    .begin_query(SpillQueryConfig::new(
                        TenantId::DEFAULT,
                        query_id,
                        0,
                        1024 * 1024,
                    ))
                    .unwrap(),
            );
        }
        assert!(
            !orphan.exists(),
            "production cadence never swept the orphan"
        );
    }
}
