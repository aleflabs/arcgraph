//! ADR-230 offline generation upgrades (v4 -> v5 and v5 -> v6).
//!
//! The source v4 directory is immutable after its final full checkpoint. A
//! complete v9 generation is built in `gen-v9.building`, synced, renamed, and
//! selected by one atomic `CURRENT` rewrite. The generation-local `VERSION=5`
//! stamp is deliberately the last durable act. A crash before `CURRENT` keeps
//! v8 selected; a crash after it resumes the final stamp without rebuilding.
//!
//! Budget: this is an offline O(data) sequential copy and needs source-size
//! transient headroom. Peak userspace memory is one 1 MiB copy buffer.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
#[cfg(feature = "fault-injection")]
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use arcgraph_core::Lsn;
use arcgraph_storage::manifest::{
    DATA_DIR_VERSION_M2, DATA_DIR_VERSION_M3, DATA_DIR_VERSION_M4, DataDirManifest,
    RECORD_FORMAT_DIRECT_M4, WAL_FORMAT_DELTA_V9, WAL_FORMAT_DELTA_V10, WAL_FORMAT_PAGE_IMAGE,
    now_rfc3339_utc,
};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, BUNDLE_FORMAT_V10, SegmentHeader, fsync_dir, segment_filename,
};
use sha2::{Digest, Sha256};

use crate::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use crate::data_lock::DataDirLock;
use crate::generation_namespace::GenerationTool;

pub(crate) const CURRENT_FILE: &str = "CURRENT";
pub(crate) const CURRENT_TMP: &str = ".CURRENT.tmp";
// INV-M5.22: every generation directory name resolves through the
// `generation_namespace` registry; this module owns the two migration legs.
const BUILDING_GENERATION: &str = GenerationTool::M3Migration.building_dir();
const FINAL_GENERATION: &str = GenerationTool::M3Migration.final_dir();
const M4_BUILDING_GENERATION: &str = GenerationTool::M4Migration.building_dir();
const M4_FINAL_GENERATION: &str = GenerationTool::M4Migration.final_dir();
/// The leg-(c) fresh-load generation (owner: `m5_load.rs`). This module only
/// ever *recognizes* the name (CURRENT resolution + commit-identity table);
/// it never creates or sweeps it — INV-M5.22 rule 2.
const M5_LOAD_FINAL: &str = GenerationTool::M5Load.final_dir();
const M5_LOAD_BUILDING: &str = GenerationTool::M5Load.building_dir();
pub(crate) const LSN_SEED_FILE: &str = "LSN_SEED";
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const RETIRED_V5_CLEANUP: &str = match GenerationTool::M4Migration.cleanup_dir() {
    Some(name) => name,
    None => unreachable!(),
};
pub const INDEX_VECTOR_CENSUS_FILE: &str = "INDEX_VECTOR_CENSUS";

/// Deterministic RE-4b crash points. Production uses [`None`](Self::None).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum MigrationFault {
    None,
    AfterScratchCreate,
    #[cfg(any(test, feature = "fault-injection"))]
    BeforeGenerationSync,
    #[cfg(any(test, feature = "fault-injection"))]
    MissingGenerationLedgerProof,
    AfterGenerationSync,
    AfterGenerationRename,
    AfterCurrentSwap,
    VersionParentDirFsync,
    AfterVersionStamp,
}

/// Deterministic fault point for the resumable old-generation reaper.
/// Production contains no mid-cleanup branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum GenerationCleanupFault {
    None,
    #[cfg(any(test, feature = "fault-injection"))]
    AfterFirstUnlink,
}

#[derive(Debug, Default)]
struct GenerationPinState {
    pins: std::collections::HashMap<PathBuf, usize>,
    retired: std::collections::HashSet<PathBuf>,
    cleanup_waiters: std::collections::HashSet<PathBuf>,
}

/// Process-local ownership for readers of immutable data-dir generations.
///
/// Pin acquisition and the cleanup-side retirement claim share one mutex. A
/// cleanup claim therefore either observes the pin and waits, or closes the
/// generation to new readers before unlinking it; there is no check/unlink
/// window in which a late reader can acquire the retiring generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationPinRegistry {
    inner: Arc<(Mutex<GenerationPinState>, Condvar)>,
}

/// A reader whose generation identity cannot change across a CURRENT swap.
#[derive(Debug)]
pub struct PinnedGenerationReader {
    generation: PathBuf,
    registry: GenerationPinRegistry,
}

static PRODUCTION_GENERATION_PINS: OnceLock<GenerationPinRegistry> = OnceLock::new();

#[cfg(feature = "fault-injection")]
#[derive(Debug, Default)]
struct BuildRendezvousFlags {
    reached: bool,
    released: bool,
}

/// One armed leak the INV-M5.10 gate can inject at the build rendezvous.
/// Each pollutes exactly ONE live checkpoint census structure, and only when
/// its env var is set (a RED-control child process arms it); an unarmed run
/// injects nothing.
#[cfg(feature = "fault-injection")]
pub struct BuildLeakInjector {
    /// Env var that arms this leak in the RED-control child process.
    pub env: &'static str,
    /// Pollutes one live structure with a build-page artifact.
    pub inject: Box<dyn Fn() + Send + Sync>,
}

#[cfg(feature = "fault-injection")]
struct BuildRendezvousState {
    flags: Mutex<BuildRendezvousFlags>,
    changed: Condvar,
    leaks: Vec<BuildLeakInjector>,
}

#[cfg(feature = "fault-injection")]
static BUILD_RENDEZVOUS: OnceLock<
    Mutex<std::collections::HashMap<PathBuf, Arc<BuildRendezvousState>>>,
> = OnceLock::new();

/// Test owner for the deterministic invisible-build rendezvous. This type and
/// its registry do not exist without the `fault-injection` feature.
#[cfg(feature = "fault-injection")]
pub struct MigrationBuildRendezvous {
    root: PathBuf,
    state: Arc<BuildRendezvousState>,
}

#[cfg(feature = "fault-injection")]
impl MigrationBuildRendezvous {
    pub fn install(root: &Path, leaks: Vec<BuildLeakInjector>) -> Result<Self> {
        let root = root.to_path_buf();
        let state = Arc::new(BuildRendezvousState {
            flags: Mutex::new(BuildRendezvousFlags::default()),
            changed: Condvar::new(),
            leaks,
        });
        let registry =
            BUILD_RENDEZVOUS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut registry = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ensure!(
            registry.insert(root.clone(), Arc::clone(&state)).is_none(),
            "build rendezvous already installed for this data dir"
        );
        Ok(Self { root, state })
    }

    pub fn wait_until_reached(&self) -> Result<()> {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut flags = self
            .state
            .flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !flags.reached {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            ensure!(
                !remaining.is_zero(),
                "builder did not reach isolation rendezvous"
            );
            let (next, timeout) = self
                .state
                .changed
                .wait_timeout(flags, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            flags = next;
            ensure!(
                !timeout.timed_out() || flags.reached,
                "builder did not reach isolation rendezvous"
            );
        }
        Ok(())
    }

    pub fn release(&self) {
        let mut flags = self
            .state
            .flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        flags.released = true;
        self.state.changed.notify_all();
    }
}

#[cfg(feature = "fault-injection")]
impl Drop for MigrationBuildRendezvous {
    fn drop(&mut self) {
        self.release();
        if let Some(registry) = BUILD_RENDEZVOUS.get() {
            registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&self.root);
        }
    }
}

#[cfg(feature = "fault-injection")]
fn rendezvous_invisible_build(root: &Path) {
    let Some(registry) = BUILD_RENDEZVOUS.get() else {
        return;
    };
    let state = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(root)
        .cloned();
    let Some(state) = state else {
        return;
    };
    for leak in &state.leaks {
        if std::env::var_os(leak.env).is_some() {
            (leak.inject)();
        }
    }
    let mut flags = state
        .flags
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    flags.reached = true;
    state.changed.notify_all();
    while !flags.released {
        flags = state
            .changed
            .wait(flags)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
}

#[cfg(feature = "fault-injection")]
#[derive(Debug, Default)]
struct PinWindowFlags {
    latched: Option<PathBuf>,
    released: bool,
    attempts: u64,
}

#[cfg(feature = "fault-injection")]
#[derive(Debug, Default)]
struct PinWindowState {
    flags: Mutex<PinWindowFlags>,
    changed: Condvar,
}

#[cfg(feature = "fault-injection")]
static PIN_WINDOW_RENDEZVOUS: OnceLock<
    Mutex<std::collections::HashMap<PathBuf, Arc<PinWindowState>>>,
> = OnceLock::new();

/// Test owner for the deterministic INV-M5.4 pin-acquisition rendezvous.
///
/// After `install`, the first `pin_current_generation` acquisition on `root`
/// parks INSIDE the resolve→pin→revalidate window: it has latched a
/// generation from `CURRENT` but has not revalidated yet. The gate advances
/// `CURRENT` while that acquisition is provably mid-window, then releases it,
/// so the revalidate-retry loop runs on every gate execution instead of
/// hoping an unsynchronized schedule happens to interleave.
#[cfg(feature = "fault-injection")]
pub struct PinAcquisitionRendezvous {
    root: PathBuf,
    state: Arc<PinWindowState>,
}

#[cfg(feature = "fault-injection")]
impl PinAcquisitionRendezvous {
    pub fn install(root: &Path) -> Result<Self> {
        let root = root.to_path_buf();
        let state = Arc::new(PinWindowState::default());
        let registry =
            PIN_WINDOW_RENDEZVOUS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut registry = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ensure!(
            registry.insert(root.clone(), Arc::clone(&state)).is_none(),
            "pin-window rendezvous already installed for this data dir"
        );
        Ok(Self { root, state })
    }

    /// Block until one acquisition parked mid-window; returns the generation
    /// that acquisition latched from `CURRENT` before parking.
    pub fn wait_until_pin_latched(&self) -> Result<PathBuf> {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut flags = self
            .state
            .flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(latched) = &flags.latched {
                return Ok(latched.clone());
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            ensure!(
                !remaining.is_zero(),
                "no pin acquisition reached the pin window"
            );
            let (next, timeout) = self
                .state
                .changed
                .wait_timeout(flags, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            flags = next;
            ensure!(
                !timeout.timed_out() || flags.latched.is_some(),
                "no pin acquisition reached the pin window"
            );
        }
    }

    pub fn release(&self) {
        let mut flags = self
            .state
            .flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        flags.released = true;
        self.state.changed.notify_all();
    }

    /// Resolve→pin acquisition attempts observed since `install`. The parked
    /// first attempt counts once; a post-release revalidation retry is
    /// attempt two. The naive no-revalidation pin can never exceed one.
    #[must_use]
    pub fn acquisition_attempts(&self) -> u64 {
        self.state
            .flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .attempts
    }
}

#[cfg(feature = "fault-injection")]
impl Drop for PinAcquisitionRendezvous {
    fn drop(&mut self) {
        self.release();
        if let Some(registry) = PIN_WINDOW_RENDEZVOUS.get() {
            registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&self.root);
        }
    }
}

#[cfg(feature = "fault-injection")]
fn rendezvous_pin_window(root: &Path, latched: &Path) {
    let Some(registry) = PIN_WINDOW_RENDEZVOUS.get() else {
        return;
    };
    let state = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(root)
        .cloned();
    let Some(state) = state else {
        return;
    };
    let mut flags = state
        .flags
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    flags.attempts += 1;
    if flags.released || flags.latched.is_some() {
        return;
    }
    flags.latched = Some(latched.to_path_buf());
    state.changed.notify_all();
    while !flags.released {
        flags = state
            .changed
            .wait(flags)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
}

/// Process-wide read-epoch registry shared by durable bootstrap and the
/// post-checkpoint generation reaper. Keeping one owner here is load-bearing:
/// constructing a fresh registry in cleanup would make every live reader pin
/// invisible and permit unlink-before-drain.
#[must_use]
pub fn production_generation_pins() -> &'static GenerationPinRegistry {
    PRODUCTION_GENERATION_PINS.get_or_init(GenerationPinRegistry::new)
}

impl GenerationPinRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin exactly the generation selected by the caller's read epoch.
    pub fn pin(&self, generation: &Path) -> Result<PinnedGenerationReader> {
        ensure!(generation.is_dir(), "generation to pin is not a directory");
        let generation = generation.to_path_buf();
        let (state, _) = &*self.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ensure!(
            !state.retired.contains(&generation),
            "generation is already retired"
        );
        *state.pins.entry(generation.clone()).or_default() += 1;
        drop(state);
        Ok(PinnedGenerationReader {
            generation,
            registry: self.clone(),
        })
    }

    fn wait_for_drain_and_retire(&self, generation: &Path) {
        let (state, drained) = &*self.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.pins.get(generation).copied().unwrap_or(0) != 0 {
            state.cleanup_waiters.insert(generation.to_path_buf());
        }
        while state.pins.get(generation).copied().unwrap_or(0) != 0 {
            state = drained
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state.cleanup_waiters.remove(generation);
        state.retired.insert(generation.to_path_buf());
    }

    /// Whether cleanup is currently held behind a live reader pin.
    #[must_use]
    pub fn cleanup_waiting(&self, generation: &Path) -> bool {
        let (state, _) = &*self.inner;
        state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cleanup_waiters
            .contains(generation)
    }
}

/// Resolve `CURRENT`, pin that exact immutable generation, then revalidate the
/// pointer. A swap that overlaps acquisition therefore yields either a wholly
/// old epoch (when the pin won before publication) or a wholly new epoch; it
/// can never return a reader assembled from both generations.
pub fn pin_current_generation(
    root: &Path,
    pins: &GenerationPinRegistry,
) -> Result<PinnedGenerationReader> {
    loop {
        let selected = current_generation(root)?.unwrap_or_else(|| root.to_path_buf());
        // INV-M5.4 gate seam: parks THIS acquisition between the CURRENT
        // resolution above and the pin/revalidate below, so the gate can
        // advance the generation strictly inside the window. Inert unless a
        // `PinAcquisitionRendezvous` is installed for `root`.
        #[cfg(feature = "fault-injection")]
        rendezvous_pin_window(root, &selected);
        match pins.pin(&selected) {
            Ok(reader) => {
                let revalidated = current_generation(root)?.unwrap_or_else(|| root.to_path_buf());
                if revalidated == selected {
                    return Ok(reader);
                }
                drop(reader);
            }
            Err(error) => {
                let revalidated = current_generation(root)?.unwrap_or_else(|| root.to_path_buf());
                if revalidated == selected {
                    return Err(error).context("pin CURRENT-selected generation");
                }
            }
        }
    }
}

/// The EXACT no-revalidation pin that INV-M5.4 forbids, kept compiled so the
/// attach gate can prove its forced schedule bites. It latches `CURRENT`
/// once, crosses the same gate seam, and pins WITHOUT revalidating — under a
/// swap forced into the window it returns a reader on the superseded
/// generation. Reverting [`pin_current_generation`] to this shape turns
/// `attach_under_concurrent_reader` RED at its wholly-new assertions.
#[cfg(feature = "fault-injection")]
pub fn naive_unrevalidated_pin_for_red_control(
    root: &Path,
    pins: &GenerationPinRegistry,
) -> Result<PinnedGenerationReader> {
    let selected = current_generation(root)?.unwrap_or_else(|| root.to_path_buf());
    rendezvous_pin_window(root, &selected);
    pins.pin(&selected)
        .context("pin CURRENT-selected generation")
}

impl PinnedGenerationReader {
    #[must_use]
    pub fn generation(&self) -> &Path {
        &self.generation
    }

    /// Read a generation-local file without re-resolving CURRENT.
    pub fn read(&self, relative: &Path) -> Result<Vec<u8>> {
        ensure!(
            relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
            "generation read path must be relative and normalized"
        );
        fs::read(self.generation.join(relative)).with_context(|| {
            format!(
                "read pinned generation file {}",
                self.generation.join(relative).display()
            )
        })
    }

    /// Read one bounded range from the pinned generation. The file path is
    /// resolved beneath the captured epoch exactly once; `CURRENT` is never
    /// consulted during the read.
    pub fn read_exact_at(&self, relative: &Path, offset: u64, len: usize) -> Result<Vec<u8>> {
        ensure!(
            relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
            "generation read path must be relative and normalized"
        );
        let path = self.generation.join(relative);
        let mut file = File::open(&path)
            .with_context(|| format!("open pinned generation file {}", path.display()))?;
        file.seek(SeekFrom::Start(offset))
            .with_context(|| format!("seek pinned generation file {}", path.display()))?;
        let mut bytes = vec![0_u8; len];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read pinned generation file {}", path.display()))?;
        Ok(bytes)
    }
}

impl Drop for PinnedGenerationReader {
    fn drop(&mut self) {
        let (state, drained) = &*self.registry.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let remove = if let Some(count) = state.pins.get_mut(&self.generation) {
            debug_assert!(*count > 0, "generation pin count underflow");
            *count -= 1;
            *count == 0
        } else {
            debug_assert!(false, "generation pin disappeared before reader drop");
            false
        };
        if remove {
            state.pins.remove(&self.generation);
            drained.notify_all();
        }
    }
}

/// Release-enforced evidence that the successor's CURRENT and VERSION commit
/// coordinates are both present and have been synchronized. Cleanup accepts
/// this proof by value, so callers cannot unlink merely because build work is
/// complete.
#[derive(Debug)]
pub struct DurableGenerationSwap {
    root: PathBuf,
    predecessor: PathBuf,
    successor: PathBuf,
}

impl DurableGenerationSwap {
    pub fn verify(root: &Path, predecessor: &Path, successor: &Path) -> Result<Self> {
        ensure!(
            predecessor.parent() == Some(root) && successor.parent() == Some(root),
            "generation swap paths must be direct children of the data-dir root"
        );
        ensure!(
            predecessor.file_name() == Some(OsStr::new(FINAL_GENERATION))
                && successor.file_name() == Some(OsStr::new(M4_FINAL_GENERATION)),
            "old-generation cleanup only accepts the v5-to-v6 generation handoff"
        );
        ensure!(
            current_generation(root)?.as_deref() == Some(successor),
            "successor is not the committed CURRENT generation"
        );
        ensure!(
            arcgraph_storage::version_file_path(successor).is_file(),
            "successor VERSION commit marker is absent"
        );

        File::open(root.join(CURRENT_FILE))
            .context("open CURRENT for durable-swap proof")?
            .sync_all()
            .context("sync CURRENT for durable-swap proof")?;
        File::open(arcgraph_storage::version_file_path(successor))
            .context("open successor VERSION for durable-swap proof")?
            .sync_all()
            .context("sync successor VERSION for durable-swap proof")?;
        File::open(successor)
            .context("open successor generation for durable-swap proof")?
            .sync_all()
            .context("sync successor generation for durable-swap proof")?;
        fsync_dir(root).context("sync data-dir root for durable-swap proof")?;

        Ok(Self {
            root: root.to_path_buf(),
            predecessor: predecessor.to_path_buf(),
            successor: successor.to_path_buf(),
        })
    }

    /// The migration pins its checkpoint at metadata generation 1 before
    /// publishing CURRENT (`m4_migration::normalize_first_v6_checkpoint_generation`
    /// — the copied v5 sidecar would otherwise carry the source's counter). A
    /// larger generation can only be selected by a later crash-atomic
    /// checkpoint of the now-live successor.
    fn successor_has_post_swap_checkpoint(&self) -> Result<bool> {
        let manifest = arcgraph_storage::read_data_dir_manifest(&self.successor)
            .context("read successor MANIFEST for cleanup proof")?
            .context("successor MANIFEST is absent")?;
        ensure!(
            manifest.data_dir_version == DATA_DIR_VERSION_M4
                && manifest.wal_format == WAL_FORMAT_DELTA_V10,
            "old-generation cleanup requires the committed v6 successor"
        );
        let migration_lsn = Lsn::new(
            manifest
                .migration_lsn
                .context("successor MANIFEST is missing its migration frontier")?,
        );
        let checkpoint = arcgraph_storage::read_latest_sidecar(&self.successor)
            .context("read successor checkpoint for cleanup proof")?
            .context("successor checkpoint is absent")?;
        if checkpoint.metadata_generation <= 1 {
            return Ok(false);
        }
        ensure!(
            checkpoint.incremental_metadata
                && !checkpoint.full_state_snapshot
                && checkpoint.checkpoint_lsn >= migration_lsn,
            "post-swap successor checkpoint does not cover the migration frontier"
        );
        Ok(true)
    }
}

/// Release-enforced evidence that every file and directory in one generation
/// has crossed the complete durability ledger. The only constructor performs
/// the recursive fsync; publishing consumes the proof, so `CURRENT` cannot be
/// swapped by a caller that has merely finished writing the build tree.
#[derive(Debug)]
pub(crate) struct CompleteGenerationLedger {
    building: PathBuf,
    complete: bool,
    index_vector_complete: bool,
}

#[derive(Debug)]
pub(crate) struct IndexVectorPassProof {
    building: PathBuf,
    files: BTreeSet<PathBuf>,
    synced: bool,
}

/// The sole publication object for one offline migration leg.  Its identity
/// binds the durability proof, final directory, `CURRENT` value, and terminal
/// VERSION stamp so none of those commit coordinates can travel separately.
#[derive(Debug)]
pub(crate) struct GenerationCommit {
    ledger: CompleteGenerationLedger,
    final_generation: PathBuf,
    generation_name: &'static str,
    data_dir_version: u16,
}

/// Result of one explicit offline upgrade invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationOutcome {
    Upgraded { migration_lsn: Lsn },
    AlreadyUpgraded { migration_lsn: Lsn },
}

/// WAL ownership for the existing-tenant re-cluster leg implemented by this
/// migration slice. It builds unpublished bytes beside the selected generation
/// and must emit into the building generation's private WAL until the atomic
/// generation switch.
///
/// Fresh-tenant attach is intentionally absent here: its live catalog-root WAL
/// path belongs to the M5-D loader slice described in
/// `docs/design/m1-m2-m4-m5-impl-designs.md` §M5.2(a) (Refs #1457; Refs #1404).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantLoadLeg<'a> {
    ExistingRecluster { building_generation: &'a Path },
}

/// Resolve the only WAL directory a tenant-load leg may see.
///
/// This is an ordinary release-mode contract, not a debug assertion. The
/// existing-tenant arm refuses the selected generation and accepts only one of
/// the two unpublished generation identities owned by this migration module.
pub fn tenant_load_wal_dir(root: &Path, leg: TenantLoadLeg<'_>) -> Result<PathBuf> {
    let live_generation = current_generation(root)?.unwrap_or_else(|| root.to_path_buf());
    let live_wal = live_generation.join("wal");
    match leg {
        TenantLoadLeg::ExistingRecluster {
            building_generation,
        } => {
            ensure!(
                building_generation.parent() == Some(root),
                "re-cluster generation must be a direct child of the data-dir root"
            );
            ensure!(
                matches!(
                    building_generation.file_name().and_then(OsStr::to_str),
                    Some(BUILDING_GENERATION | M4_BUILDING_GENERATION)
                ),
                "re-cluster WAL requires an unpublished building generation"
            );
            ensure!(
                building_generation != live_generation,
                "re-cluster WAL must not expose the live generation"
            );
            let isolated_wal = building_generation.join("wal");
            ensure!(
                isolated_wal != live_wal,
                "fresh and existing tenant legs must not share WAL exposure"
            );
            Ok(isolated_wal)
        }
    }
}

/// Operator entry point: hold the root lock, quiesce, establish the final full
/// checkpoint, create a fresh WAL inside the new generation, then perform the
/// beside-build and atomic switch. The selected source WAL is never truncated.
pub fn upgrade_data_dir(root: &Path) -> Result<MigrationOutcome> {
    if let Some(selected) = current_generation(root)? {
        // INV-M5.21/.22: a fresh-load generation is already v6 and is owned
        // by `arcgraph load`. The migrate tool must neither rebuild nor
        // complete another tool's commit — refuse with the owning tool named.
        ensure!(
            selected.file_name() != Some(OsStr::new(M5_LOAD_FINAL)),
            "data dir {} is a committed `arcgraph load` generation ({M5_LOAD_FINAL}); \
             it is already data_dir_version 6 and owned by the fresh-load tool — \
             there is nothing for `arcgraph migrate upgrade-data-dir` to do",
            root.display()
        );
        if selected.file_name() == Some(OsStr::new(M4_FINAL_GENERATION)) {
            let _lock = DataDirLock::acquire(root)?;
            return resume_after_m4_swap(&selected, MigrationFault::None);
        }
        let mode = BootstrapMode::Durable {
            data_dir: root.to_path_buf(),
        };
        let (backend, guard) = bootstrap_storage_backend(&mode)
            .with_context(|| format!("open v5 source generation at {}", root.display()))?;
        drop(backend);
        let (lock, migration_lsn) = guard.quiesce_for_migration()?;
        let outcome = upgrade_quiesced_v5_to_v6(root, migration_lsn, production_migration_fault())?;
        // M4 Slice-3a requires gen-v9 to survive the swap as the
        // crash-before-successor-checkpoint recovery fallback. INV-M5.5
        // therefore reaps it only after gen-v10 establishes a later durable
        // checkpoint and every old-generation reader pin drains.
        drop(lock);
        return Ok(outcome);
    }
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode)
        .with_context(|| format!("open v4 source generation at {}", root.display()))?;
    drop(backend);
    let (_lock, migration_lsn) = guard.quiesce_for_migration()?;
    upgrade_quiesced_data_dir(root, migration_lsn, MigrationFault::None)
}

pub(crate) fn production_cleanup_fault() -> GenerationCleanupFault {
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_PRODUCTION_CLEANUP_CRASH").is_some() {
        return GenerationCleanupFault::AfterFirstUnlink;
    }
    GenerationCleanupFault::None
}

fn production_migration_fault() -> MigrationFault {
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_PRODUCTION_MISSING_LEDGER").is_some() {
        return MigrationFault::MissingGenerationLedgerProof;
    }
    MigrationFault::None
}

pub(crate) fn inject(selected: MigrationFault, point: MigrationFault) -> Result<()> {
    if selected == point {
        bail!("injected migration crash at {point:?}");
    }
    Ok(())
}

/// Resolve the committed generation named by `CURRENT`, or `None` for a
/// legacy flat v8 directory. VERSION is the generation-local commit marker:
/// an interrupted later leg falls back to its stamped predecessor, while an
/// interrupted first leg falls back to the legacy root.
pub fn current_generation(root: &Path) -> Result<Option<PathBuf>> {
    let bytes = match fs::read(root.join(CURRENT_FILE)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("read CURRENT generation pointer"),
    };
    let name = std::str::from_utf8(&bytes)
        .context("CURRENT is not UTF-8")?
        .trim_end_matches(['\r', '\n']);
    let mut components = Path::new(name).components();
    let component = components.next();
    ensure!(
        matches!(component, Some(Component::Normal(_))) && components.next().is_none(),
        "CURRENT must name exactly one generation directory"
    );
    ensure!(
        matches!(name, FINAL_GENERATION | M4_FINAL_GENERATION | M5_LOAD_FINAL),
        "CURRENT names unsupported generation {name:?}"
    );
    let selected = root.join(name);
    if arcgraph_storage::version_file_path(&selected).is_file() {
        return Ok(Some(selected));
    }
    if name == M4_FINAL_GENERATION {
        let predecessor = root.join(FINAL_GENERATION);
        if arcgraph_storage::version_file_path(&predecessor).is_file() {
            return Ok(Some(predecessor));
        }
    }
    // INV-M5.3 for leg (c): an unstamped fresh-load generation has no
    // predecessor to fall back to — visibility rolls back to "no committed
    // generation" (the virgin state) until the loader rerun completes the
    // VERSION-last stamp per the §2.5 restart matrix.
    Ok(None)
}

/// Build/switch a v9 generation after the caller has quiesced the store and
/// established the final full checkpoint at `migration_lsn`.
pub fn upgrade_quiesced_data_dir(
    root: &Path,
    migration_lsn: Lsn,
    fault: MigrationFault,
) -> Result<MigrationOutcome> {
    ensure!(
        migration_lsn != Lsn::MAX,
        "migration LSN cannot be u64::MAX"
    );
    if let Some(selected) = current_generation(root)? {
        if selected.file_name() == Some(OsStr::new(M4_FINAL_GENERATION)) {
            return resume_after_m4_swap(&selected, fault);
        }
        return resume_after_swap(&selected, fault);
    }

    let version = arcgraph_storage::check_or_stamp_data_dir(root, true, false)
        .context("source data-dir VERSION guard")?;
    ensure!(
        version == DATA_DIR_VERSION_M2,
        "upgrade-data-dir requires data_dir_version 4, found {version}"
    );
    let prior = arcgraph_storage::read_data_dir_manifest(root)
        .context("read source MANIFEST")?
        .context("v4 source has no MANIFEST")?;
    ensure!(
        prior.data_dir_version == DATA_DIR_VERSION_M2
            && prior.wal_format == WAL_FORMAT_PAGE_IMAGE
            && prior.props_fully_typed(),
        "source MANIFEST is not the final v4 typed/page-image format"
    );
    ensure!(
        root.join(arcgraph_storage::CHECKPOINT_SNAPSHOT_FILE)
            .is_file(),
        "final full checkpoint snapshot is missing"
    );
    ensure!(
        root.join(arcgraph_storage::CHECKPOINT_SIDECAR_FILE)
            .is_file(),
        "final checkpoint sidecar is missing"
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(root)
        .context("read final checkpoint sidecar")?
        .context("final checkpoint sidecar is absent")?;
    ensure!(
        checkpoint.full_state_snapshot
            && !checkpoint.incremental_metadata
            && checkpoint.checkpoint_lsn == migration_lsn,
        "final full checkpoint frontier {} does not match migration_lsn {}",
        checkpoint.checkpoint_lsn.raw(),
        migration_lsn.raw()
    );

    let required = copied_tree_bytes(root)?;
    let available = available_bytes(root)?;
    ensure!(
        available >= required,
        "offline v4->v5 upgrade needs {required} bytes of transient headroom, only {available} bytes available"
    );

    let building = root.join(BUILDING_GENERATION);
    let final_generation = root.join(FINAL_GENERATION);
    remove_tree_if_exists(&building)?;
    remove_tree_if_exists(&final_generation)?;
    fs::create_dir(&building).with_context(|| format!("create {BUILDING_GENERATION} scratch"))?;
    fsync_dir(root).context("sync data-dir after scratch creation")?;
    inject(fault, MigrationFault::AfterScratchCreate)?;

    copy_source_generation(root, &building)?;
    let translated = arcgraph_storage::m3_migration::translate_v4_checkpoint(root, &building)
        .context("translate final v4 checkpoint into the v9 physical base")?;
    ensure!(
        translated.migration_lsn == migration_lsn,
        "translated checkpoint frontier {} differs from migration_lsn {}",
        translated.migration_lsn.raw(),
        migration_lsn.raw()
    );
    let wal_dir = tenant_load_wal_dir(
        root,
        TenantLoadLeg::ExistingRecluster {
            building_generation: &building,
        },
    )?;
    fs::create_dir(&wal_dir).context("create v9 WAL directory")?;
    write_synced(
        &wal_dir.join(segment_filename(0)),
        &SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )?;
    fsync_dir(&wal_dir).context("sync v9 WAL directory")?;
    let next_lsn = migration_lsn.raw() + 1;
    write_synced(&building.join(LSN_SEED_FILE), &next_lsn.to_le_bytes())?;
    arcgraph_storage::write_data_dir_manifest(
        &building,
        &DataDirManifest::m3_delta_from(&prior, now_rfc3339_utc(), migration_lsn),
    )
    .context("write v9 MANIFEST")?;
    #[cfg(any(test, feature = "fault-injection"))]
    inject_before_generation_sync(fault)?;
    let ledger = complete_generation_ledger(&building)?;
    #[cfg(any(test, feature = "fault-injection"))]
    let ledger = inject_generation_ledger_fault(ledger, fault);
    inject(fault, MigrationFault::AfterGenerationSync)?;
    GenerationCommit::new(
        root,
        &final_generation,
        FINAL_GENERATION,
        DATA_DIR_VERSION_M3,
        ledger,
    )?
    .commit(fault)
    .with_context(|| format!("commit complete {FINAL_GENERATION} generation"))?;
    Ok(MigrationOutcome::Upgraded { migration_lsn })
}

/// Build/switch a complete v6 generation after the caller has quiesced the
/// selected v5 generation and established its final full-flush checkpoint.
/// The selected v5 generation and its WAL remain byte-for-byte immutable.
pub fn upgrade_quiesced_v5_to_v6(
    root: &Path,
    migration_lsn: Lsn,
    fault: MigrationFault,
) -> Result<MigrationOutcome> {
    ensure!(
        migration_lsn != Lsn::ZERO && migration_lsn != Lsn::MAX,
        "v5->v6 migration LSN must be non-zero and below u64::MAX"
    );
    let selected = current_generation(root)?.context("v5->v6 requires a CURRENT generation")?;
    if selected.file_name() == Some(OsStr::new(M4_FINAL_GENERATION)) {
        return resume_after_m4_swap(&selected, fault);
    }
    ensure!(
        selected.file_name() == Some(OsStr::new(FINAL_GENERATION)),
        "v5->v6 source must be the committed {FINAL_GENERATION} generation"
    );
    let version = arcgraph_storage::check_or_stamp_data_dir(&selected, true, false)
        .context("v5 source generation VERSION guard")?;
    ensure!(
        version == DATA_DIR_VERSION_M3,
        "upgrade-data-dir requires data_dir_version 5, found {version}"
    );
    let prior = arcgraph_storage::read_data_dir_manifest(&selected)
        .context("read v5 source MANIFEST")?
        .context("v5 source has no MANIFEST")?;
    ensure!(
        prior.data_dir_version == DATA_DIR_VERSION_M3 && prior.wal_format == WAL_FORMAT_DELTA_V9,
        "source MANIFEST is not a complete v5 delta-v9 generation"
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(&selected)
        .context("read final v5 checkpoint sidecar")?
        .context("final v5 checkpoint sidecar is absent")?;
    ensure!(
        checkpoint.incremental_metadata
            && !checkpoint.full_state_snapshot
            && checkpoint.checkpoint_lsn == migration_lsn,
        "final full-flush v5 checkpoint frontier {} does not match migration_lsn {}",
        checkpoint.checkpoint_lsn.raw(),
        migration_lsn.raw()
    );
    verify_empty_v5_dpt(&selected, &checkpoint, migration_lsn)?;

    let required = copied_tree_bytes(&selected)?;
    let available = available_bytes(root)?;
    ensure!(
        available >= required,
        "offline v5->v6 upgrade needs {required} bytes of transient headroom, only {available} bytes available"
    );

    let building = root.join(M4_BUILDING_GENERATION);
    let final_generation = root.join(M4_FINAL_GENERATION);
    remove_tree_if_exists(&building)?;
    remove_tree_if_exists(&final_generation)?;
    fs::create_dir(&building)
        .with_context(|| format!("create {M4_BUILDING_GENERATION} scratch"))?;
    fsync_dir(root).context("sync data-dir after v6 scratch creation")?;
    inject(fault, MigrationFault::AfterScratchCreate)?;

    let loader_frontier =
        arcgraph_storage::m4_migration::LoaderMigrationFrontier::new(migration_lsn)
            .context("bind loader page and post-attach LSN frontier")?;
    let rewritten = arcgraph_storage::m4_migration::load_v5_generation(
        &selected,
        &building,
        loader_frontier,
        arcgraph_storage::m4_migration::LoaderTarget::ExistingRecluster,
    )
    .context("rewrite immutable v5 base into extent-backed v6 stores")?;
    #[cfg(feature = "fault-injection")]
    rendezvous_invisible_build(root);
    ensure!(
        !rewritten.tenants.is_empty(),
        "v6 rewrite produced no tenants"
    );
    let wal_dir = tenant_load_wal_dir(
        root,
        TenantLoadLeg::ExistingRecluster {
            building_generation: &building,
        },
    )?;
    fs::create_dir(&wal_dir).context("create fresh empty v6 WAL directory")?;
    fsync_dir(&wal_dir).context("sync fresh empty v6 WAL directory")?;
    let next_lsn = loader_frontier.next_lsn();
    write_synced(&building.join(LSN_SEED_FILE), &next_lsn.to_le_bytes())?;
    let first_checkpoint = arcgraph_storage::read_latest_sidecar(&building)
        .context("read first v6 checkpoint for MANIFEST proof")?
        .context("first v6 checkpoint is absent before MANIFEST write")?;
    let first_metadata = arcgraph_storage::checkpoint::incremental_metadata_path(
        &building,
        first_checkpoint.checkpoint_lsn,
        first_checkpoint.metadata_generation,
    );
    let tenant_census = rewritten
        .tenants
        .iter()
        .map(|tenant| tenant.raw())
        .collect();
    let metadata_sha256 = file_sha256(&first_metadata)
        .with_context(|| format!("checksum first v6 metadata {}", first_metadata.display()))?;
    arcgraph_storage::write_data_dir_manifest(
        &building,
        &DataDirManifest::m4_direct_from(
            &prior,
            now_rfc3339_utc(),
            migration_lsn,
            tenant_census,
            metadata_sha256,
        ),
    )
    .context("write v6 MANIFEST")?;
    verify_v6_generation(&building, &rewritten.tenants, migration_lsn, true, true)?;
    let index_vector = complete_index_vector_passes(
        Some(&selected),
        &building,
        &rewritten.tenants,
        production_index_vector_fault(),
    )?;
    #[cfg(any(test, feature = "fault-injection"))]
    inject_before_generation_sync(fault)?;
    let ledger = complete_generation_ledger(&building)?.with_index_vector_proof(index_vector)?;
    #[cfg(any(test, feature = "fault-injection"))]
    let ledger = inject_generation_ledger_fault(ledger, fault);
    inject(fault, MigrationFault::AfterGenerationSync)?;
    GenerationCommit::new(
        root,
        &final_generation,
        M4_FINAL_GENERATION,
        DATA_DIR_VERSION_M4,
        ledger,
    )?
    .commit(fault)
    .with_context(|| format!("commit complete {M4_FINAL_GENERATION} generation"))?;
    Ok(MigrationOutcome::Upgraded { migration_lsn })
}

fn verify_empty_v5_dpt(
    generation: &Path,
    checkpoint: &arcgraph_storage::checkpoint::CheckpointSidecar,
    migration_lsn: Lsn,
) -> Result<()> {
    let path = arcgraph_storage::checkpoint::incremental_metadata_path(
        generation,
        checkpoint.checkpoint_lsn,
        checkpoint.metadata_generation,
    );
    let mut header = [0_u8; 48];
    File::open(&path)
        .with_context(|| format!("open final v5 checkpoint metadata {}", path.display()))?
        .read_exact(&mut header)
        .context("read final v5 checkpoint metadata header")?;
    ensure!(
        &header[..4] == b"AGCM",
        "final v5 metadata magic is invalid"
    );
    ensure!(
        u16::from_le_bytes(header[4..6].try_into().unwrap())
            == arcgraph_storage::checkpoint::INCREMENTAL_METADATA_FORMAT_VERSION
            && u16::from_le_bytes(header[6..8].try_into().unwrap()) == 0,
        "final v5 metadata format/flags are invalid"
    );
    let checkpoint_lsn = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let redo_lsn = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let capture_lsn = u64::from_le_bytes(header[24..32].try_into().unwrap());
    let dpt_count = u64::from_le_bytes(header[32..40].try_into().unwrap());
    ensure!(
        checkpoint_lsn == migration_lsn.raw()
            && redo_lsn == migration_lsn.raw()
            && capture_lsn >= migration_lsn.raw()
            && dpt_count == 0,
        "M4 activation requires a final full-flush v5 checkpoint with DPT = empty"
    );
    Ok(())
}

fn resume_after_swap(generation: &Path, fault: MigrationFault) -> Result<MigrationOutcome> {
    let manifest = arcgraph_storage::read_data_dir_manifest(generation)
        .context("read selected generation MANIFEST")?
        .context("selected generation has no MANIFEST")?;
    ensure!(
        manifest.data_dir_version == DATA_DIR_VERSION_M3
            && manifest.wal_format == WAL_FORMAT_DELTA_V9,
        "CURRENT-selected generation is not a complete delta-v9 generation"
    );
    let next_lsn = read_lsn_seed(generation)?;
    let migration_lsn = Lsn::new(next_lsn - 1);
    let version_path = arcgraph_storage::version_file_path(generation);
    if version_path.exists() {
        let version = arcgraph_storage::check_or_stamp_data_dir(generation, true, false)
            .context("selected generation VERSION guard")?;
        ensure!(
            version == DATA_DIR_VERSION_M3,
            "selected generation is not version 5"
        );
        return Ok(MigrationOutcome::AlreadyUpgraded { migration_lsn });
    }
    arcgraph_storage::stamp_data_dir(generation, DATA_DIR_VERSION_M3)
        .context("resume version-last generation stamp")?;
    inject(fault, MigrationFault::AfterVersionStamp)?;
    Ok(MigrationOutcome::Upgraded { migration_lsn })
}

pub(crate) fn resume_after_m4_swap(
    generation: &Path,
    fault: MigrationFault,
) -> Result<MigrationOutcome> {
    let manifest = arcgraph_storage::read_data_dir_manifest(generation)
        .context("read selected v6 generation MANIFEST")?
        .context("selected v6 generation has no MANIFEST")?;
    ensure!(
        manifest.data_dir_version == DATA_DIR_VERSION_M4
            && manifest.wal_format == WAL_FORMAT_DELTA_V10
            && manifest.record_store_format == RECORD_FORMAT_DIRECT_M4,
        "CURRENT-selected generation is not a complete direct-addressed v6 generation"
    );
    let next_lsn = read_lsn_seed(generation)?;
    let migration_lsn = Lsn::new(next_lsn - 1);
    ensure!(
        manifest.migration_lsn == Some(migration_lsn.raw()),
        "v6 MANIFEST migration frontier disagrees with LSN_SEED"
    );
    let version_path = arcgraph_storage::version_file_path(generation);
    let version_exists = version_path.exists();
    let tenants = v6_generation_tenants(generation)?;
    verify_v6_generation(
        generation,
        &tenants,
        migration_lsn,
        !version_exists,
        !version_exists,
    )
    .context("validate CURRENT-selected v6 generation before VERSION resume")?;
    if version_exists {
        let version = arcgraph_storage::check_or_stamp_data_dir(generation, true, false)
            .context("selected v6 generation VERSION guard")?;
        ensure!(
            version == DATA_DIR_VERSION_M4,
            "selected generation is not version 6"
        );
        return Ok(MigrationOutcome::AlreadyUpgraded { migration_lsn });
    }
    stamp_m4_data_dir(generation, fault).context("resume version-last v6 generation stamp")?;
    inject(fault, MigrationFault::AfterVersionStamp)?;
    Ok(MigrationOutcome::Upgraded { migration_lsn })
}

fn stamp_m4_data_dir(generation: &Path, fault: MigrationFault) -> Result<()> {
    stamp_generation_version(generation, DATA_DIR_VERSION_M4, fault)
}

/// Read and validate the next logical/redo LSN persisted at the generation
/// boundary. Durable bootstrap uses this as a continuity assertion.
pub fn read_lsn_seed(generation: &Path) -> Result<u64> {
    let bytes = fs::read(generation.join(LSN_SEED_FILE)).context("read v9 LSN_SEED")?;
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("LSN_SEED must be exactly 8 bytes"))?;
    let next = u64::from_le_bytes(bytes);
    ensure!(next > 0, "LSN_SEED must be non-zero");
    Ok(next)
}

pub(crate) fn write_current_atomic(root: &Path, generation: &str) -> Result<()> {
    let tmp = root.join(CURRENT_TMP);
    match fs::remove_file(&tmp) {
        Ok(()) => fsync_dir(root).context("sync stale CURRENT temp removal")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("remove stale CURRENT temp"),
    }
    write_synced(&tmp, format!("{generation}\n").as_bytes())?;
    fs::rename(&tmp, root.join(CURRENT_FILE)).context("atomic CURRENT rename")?;
    fsync_dir(root).context("sync CURRENT parent directory")
}

/// Retire and reap the predecessor named by a proven durable generation
/// swap. The successor must first establish a later crash-atomic checkpoint;
/// only then does retirement wait for every pre-swap reader pin, atomically
/// rename the old directory to a cleanup name, and unlink it. A crash after
/// that rename is resumed by [`resume_generation_cleanup`].
pub fn cleanup_old_generation_after_drain(
    swap: DurableGenerationSwap,
    pins: &GenerationPinRegistry,
    fault: GenerationCleanupFault,
) -> Result<()> {
    // Reconcile the two durability invariants explicitly: gen-v9 survives
    // CURRENT+VERSION as M4 Slice-3a's recovery fallback; M5.5 may reap it
    // only after gen-v10 has checkpointed as the live generation and pins
    // prove that no old-generation reader is still serving from it.
    ensure!(
        swap.successor_has_post_swap_checkpoint()?,
        "old generation remains the recovery fallback until a post-swap successor checkpoint"
    );
    pins.wait_for_drain_and_retire(&swap.predecessor);
    let cleanup = swap.root.join(RETIRED_V5_CLEANUP);

    if swap.predecessor.exists() {
        ensure!(
            !cleanup.exists(),
            "old generation and its cleanup tombstone both exist"
        );
        fs::rename(&swap.predecessor, &cleanup)
            .context("retire drained v5 generation for cleanup")?;
        fsync_dir(&swap.root).context("sync old-generation retirement rename")?;
    }
    if cleanup.exists() {
        remove_cleanup_tree(&cleanup, fault)?;
        fsync_dir(&swap.root).context("sync completed old-generation cleanup")?;
    }
    Ok(())
}

/// Startup reaper for an interrupted old-generation cleanup. Removal is
/// idempotent: missing leaves/directories are success, and the cleanup root is
/// removed only after all surviving children have been visited.
pub fn resume_generation_cleanup(root: &Path, fault: GenerationCleanupFault) -> Result<()> {
    resume_generation_cleanup_with_pins(root, production_generation_pins(), fault)
}

fn resume_generation_cleanup_with_pins(
    root: &Path,
    pins: &GenerationPinRegistry,
    fault: GenerationCleanupFault,
) -> Result<()> {
    let cleanup = root.join(RETIRED_V5_CLEANUP);
    if cleanup.exists() {
        let predecessor = root.join(FINAL_GENERATION);
        let successor = root.join(M4_FINAL_GENERATION);
        let durable_swap = DurableGenerationSwap::verify(root, &predecessor, &successor)
            .context("prove durable successor before resuming old-generation cleanup")?;
        if !durable_swap.successor_has_post_swap_checkpoint()? {
            return Ok(());
        }
        remove_cleanup_tree(&cleanup, fault)?;
        fsync_dir(root).context("sync resumed old-generation cleanup")?;
        return Ok(());
    }

    // A crash may land after the successor's later durable checkpoint but
    // before the retirement rename. No process-local reader can survive
    // restart, so startup may construct an empty registry and finish the
    // already-proven handoff. An initial migration checkpoint is deliberately
    // insufficient: in that state gen-v9 is still the M4 recovery fallback.
    let predecessor = root.join(FINAL_GENERATION);
    let successor = root.join(M4_FINAL_GENERATION);
    if predecessor.exists() && current_generation(root)?.as_deref() == Some(successor.as_path()) {
        let swap = DurableGenerationSwap::verify(root, &predecessor, &successor)?;
        if swap.successor_has_post_swap_checkpoint()? {
            cleanup_old_generation_after_drain(swap, pins, fault)?;
        }
    }
    Ok(())
}

fn remove_cleanup_tree(path: &Path, fault: GenerationCleanupFault) -> Result<()> {
    let mut injected = false;
    remove_cleanup_entry(path, fault, &mut injected)
}

fn remove_cleanup_entry(
    path: &Path,
    fault: GenerationCleanupFault,
    injected: &mut bool,
) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("stat cleanup entry {}", path.display()));
        }
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        let mut children = fs::read_dir(path)
            .with_context(|| format!("read cleanup directory {}", path.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        children.sort_by_key(std::fs::DirEntry::file_name);
        for child in children {
            remove_cleanup_entry(&child.path(), fault, injected)?;
        }
        match fs::remove_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove cleanup directory {}", path.display()));
            }
        }
    } else {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("unlink cleanup file {}", path.display()));
            }
        }
    }

    if !*injected {
        *injected = true;
        if let Some(parent) = path.parent() {
            fsync_dir(parent).context("sync first old-generation cleanup unlink")?;
        }
        #[cfg(any(test, feature = "fault-injection"))]
        if fault == GenerationCleanupFault::AfterFirstUnlink {
            bail!("injected old-generation cleanup crash after first unlink");
        }
        #[cfg(not(any(test, feature = "fault-injection")))]
        let _ = fault;
    }
    Ok(())
}

#[cfg(any(test, feature = "fault-injection"))]
fn inject_before_generation_sync(fault: MigrationFault) -> Result<()> {
    inject(fault, MigrationFault::BeforeGenerationSync)
}

pub(crate) fn complete_generation_ledger(building: &Path) -> Result<CompleteGenerationLedger> {
    fsync_tree(building).context("fsync complete generation ledger")?;
    Ok(CompleteGenerationLedger {
        building: building.to_path_buf(),
        complete: true,
        index_vector_complete: false,
    })
}

#[cfg(any(test, feature = "fault-injection"))]
pub(crate) fn inject_generation_ledger_fault(
    mut ledger: CompleteGenerationLedger,
    fault: MigrationFault,
) -> CompleteGenerationLedger {
    if fault == MigrationFault::MissingGenerationLedgerProof {
        ledger.complete = false;
    }
    ledger
}

impl CompleteGenerationLedger {
    pub(crate) fn with_index_vector_proof(mut self, proof: IndexVectorPassProof) -> Result<Self> {
        ensure!(
            proof.building == self.building,
            "index/vector pass proof belongs to a different generation"
        );
        ensure!(
            proof.synced,
            "generation commit refused: index/vector pass fsync proof is absent"
        );
        ensure!(
            !proof.files.is_empty(),
            "generation commit refused: index/vector complete-file census is empty"
        );
        self.index_vector_complete = true;
        Ok(self)
    }

    /// Consume the ledger exactly at the publication boundary. This is an
    /// ordinary runtime branch by design: INV-M5.6 requires a missing proof to
    /// refuse publication even when debug assertions are compiled out.
    fn consume(self) -> Result<PathBuf> {
        if !self.complete {
            bail!("generation commit refused: complete durability ledger proof is absent");
        }
        Ok(self.building)
    }
}

impl GenerationCommit {
    pub(crate) fn new(
        root: &Path,
        final_generation: &Path,
        generation_name: &'static str,
        data_dir_version: u16,
        ledger: CompleteGenerationLedger,
    ) -> Result<Self> {
        ensure!(
            ledger.building.parent() == Some(root),
            "generation ledger proof belongs to a different data-dir root"
        );
        ensure!(
            final_generation == root.join(generation_name),
            "generation commit point does not match its CURRENT identity"
        );
        ensure!(
            matches!(
                (generation_name, data_dir_version),
                (FINAL_GENERATION, DATA_DIR_VERSION_M3)
                    | (M4_FINAL_GENERATION, DATA_DIR_VERSION_M4)
                    | (M5_LOAD_FINAL, DATA_DIR_VERSION_M4)
            ),
            "generation identity disagrees with its data-dir VERSION"
        );
        if data_dir_version == DATA_DIR_VERSION_M4 {
            ensure!(
                ledger.index_vector_complete,
                "v6 generation commit requires completed, synced index/vector passes"
            );
        }
        Ok(Self {
            ledger,
            final_generation: final_generation.to_path_buf(),
            generation_name,
            data_dir_version,
        })
    }

    pub(crate) fn commit(self, fault: MigrationFault) -> Result<()> {
        let Self {
            ledger,
            final_generation,
            generation_name,
            data_dir_version,
        } = self;
        let root = final_generation
            .parent()
            .context("generation commit target has no data-dir root")?;
        let building = ledger.consume()?;
        fs::rename(&building, &final_generation)
            .with_context(|| format!("publish complete {generation_name} directory"))?;
        fsync_dir(root).with_context(|| format!("sync {generation_name} directory rename"))?;
        inject(fault, MigrationFault::AfterGenerationRename)?;
        write_current_atomic(root, generation_name)?;
        inject(fault, MigrationFault::AfterCurrentSwap)?;

        // INV-M5.3: this is deliberately the final durable act.  Until the
        // generation-local marker lands, current_generation() rolls visibility
        // back to the preceding committed generation.
        stamp_generation_version(&final_generation, data_dir_version, fault)
            .with_context(|| format!("stamp generation data_dir_version {}", data_dir_version))?;
        inject(fault, MigrationFault::AfterVersionStamp)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexVectorFault {
    None,
    #[cfg(feature = "fault-injection")]
    SkipFsync,
}

pub(crate) fn production_index_vector_fault() -> IndexVectorFault {
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_SKIP_INDEX_FSYNC").is_some() {
        return IndexVectorFault::SkipFsync;
    }
    IndexVectorFault::None
}

/// `source = Some(..)` for the migration legs (every preserved source
/// index/vector artifact must survive byte-identical); `None` for the leg-(c)
/// fresh load, whose generation has no predecessor and therefore no preserved
/// artifacts — its census contains only the generated passes. ONE shared
/// body (the M5-D1 reuse-not-fork rule), not a second protocol.
pub(crate) fn complete_index_vector_passes(
    source: Option<&Path>,
    building: &Path,
    tenants: &BTreeSet<arcgraph_core::TenantId>,
    fault: IndexVectorFault,
) -> Result<IndexVectorPassProof> {
    let mut files = index_vector_candidate_files(building)?;

    // Every source index/vector artifact is a required, byte-identical member
    // of the successor census. The loader may add new direct-layout indexes,
    // but it may never silently omit a preserved pass output.
    if let Some(source) = source {
        for source_file in index_vector_candidate_files(source)? {
            let relative = source_file
                .strip_prefix(source)
                .expect("candidate belongs to source");
            let destination = building.join(relative);
            ensure!(
                destination.is_file(),
                "index/vector artifact missing from successor: {}",
                relative.display()
            );
            ensure!(
                fs::metadata(&source_file)?.len() == fs::metadata(&destination)?.len()
                    && file_sha256(&source_file)? == file_sha256(&destination)?,
                "index/vector artifact changed during preservation: {}",
                relative.display()
            );
            files.insert(destination);
        }
    }

    // The direct secondary-index store and every forward owner index are
    // generated passes rather than preserved files; make them mandatory even
    // when their path names change and would evade the lexical classifier.
    for tenant in tenants {
        let secondary = arcgraph_storage::extent::production_extent_store_path(
            building,
            *tenant,
            arcgraph_storage::wal::STORE_SECONDARY_INDEX,
        )
        .context("secondary index store has no production path")?;
        ensure!(
            secondary.is_file(),
            "secondary index pass output is missing"
        );
        files.insert(secondary);
        for class in arcgraph_storage::OwnerRowClass::ALL {
            if let Some(index) =
                arcgraph_storage::owner_row::owner_forward_index_path(building, *tenant, class)
            {
                ensure!(index.is_dir(), "owner forward-index pass output is missing");
                collect_regular_files(&index, &mut files)?;
            }
        }
    }

    let mut census = String::new();
    for file in &files {
        let relative = file
            .strip_prefix(building)
            .context("index/vector census file escaped building generation")?;
        use std::fmt::Write as _;
        writeln!(
            &mut census,
            "{}\t{}\t{}",
            relative.display(),
            fs::metadata(file)?.len(),
            file_sha256(file)?
        )
        .expect("writing to String is infallible");
    }
    ensure!(
        !census.is_empty(),
        "index/vector complete-file census is empty"
    );
    let census_path = building.join(INDEX_VECTOR_CENSUS_FILE);
    write_synced(&census_path, census.as_bytes())?;
    files.insert(census_path);

    #[cfg(feature = "fault-injection")]
    if fault == IndexVectorFault::SkipFsync {
        return Ok(IndexVectorPassProof {
            building: building.to_path_buf(),
            files,
            synced: false,
        });
    }
    #[cfg(not(feature = "fault-injection"))]
    let _ = fault;

    for file in &files {
        File::open(file)
            .with_context(|| format!("open index/vector pass output {}", file.display()))?
            .sync_all()
            .with_context(|| format!("fsync index/vector pass output {}", file.display()))?;
    }
    let mut directories: BTreeSet<_> = files
        .iter()
        .filter_map(|file| file.parent().map(Path::to_path_buf))
        .collect();
    while let Some(directory) = directories.pop_last() {
        File::open(&directory)
            .with_context(|| format!("open index/vector directory {}", directory.display()))?
            .sync_all()
            .with_context(|| format!("fsync index/vector directory {}", directory.display()))?;
        if directory != building
            && let Some(parent) = directory.parent()
            && parent.starts_with(building)
        {
            directories.insert(parent.to_path_buf());
        }
    }
    Ok(IndexVectorPassProof {
        building: building.to_path_buf(),
        files,
        synced: true,
    })
}

fn index_vector_candidate_files(root: &Path) -> Result<BTreeSet<PathBuf>> {
    let mut all = BTreeSet::new();
    collect_regular_files(root, &mut all)?;
    Ok(all
        .into_iter()
        .filter(|path| {
            path.strip_prefix(root).is_ok_and(|relative| {
                relative.components().any(|component| {
                    let name = component.as_os_str().to_string_lossy().to_ascii_lowercase();
                    ["index", "vector", "bm25", "diskann", "hnsw"]
                        .iter()
                        .any(|marker| name.contains(marker))
                })
            })
        })
        .collect())
}

fn collect_regular_files(root: &Path, files: &mut BTreeSet<PathBuf>) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "index/vector census rejects symlinks"
    );
    if metadata.is_file() {
        files.insert(root.to_path_buf());
        return Ok(());
    }
    ensure!(
        metadata.is_dir(),
        "index/vector census found a special file"
    );
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        collect_regular_files(&entry.path(), files)?;
    }
    Ok(())
}

pub(crate) fn stamp_generation_version(
    generation: &Path,
    data_dir_version: u16,
    fault: MigrationFault,
) -> Result<()> {
    if fault == MigrationFault::VersionParentDirFsync {
        arcgraph_storage::stamp_data_dir_with_parent_sync_error_for_test(
            generation,
            data_dir_version,
        )?;
    } else {
        arcgraph_storage::stamp_data_dir(generation, data_dir_version)?;
    }
    Ok(())
}

pub(crate) fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn excluded_root_entry(name: &OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(
            "LOCK"
                | "VERSION"
                | "MANIFEST"
                | "wal"
                | CURRENT_FILE
                | CURRENT_TMP
                | BUILDING_GENERATION
                | FINAL_GENERATION
                | M4_BUILDING_GENERATION
                | M4_FINAL_GENERATION
                // INV-M5.22: the migrate legs never copy (= consume) the
                // fresh-load tool's namespace into their own generations.
                | M5_LOAD_BUILDING
                | M5_LOAD_FINAL
                | "blob-spill.db"
                | "idempotency-spill.db"
                | "CHECKPOINT"
                | "CHECKPOINT.snap"
        )
    )
}

pub(crate) fn verify_v6_generation(
    generation: &Path,
    tenants: &std::collections::BTreeSet<arcgraph_core::TenantId>,
    migration_lsn: Lsn,
    require_empty_wal: bool,
    require_unstamped: bool,
) -> Result<()> {
    arcgraph_storage::m4_migration::verify_complete_store_set(generation, tenants)?;
    ensure!(
        generation.join("pages.db").is_file(),
        "v6 catalog root is missing"
    );
    ensure!(
        generation
            .join(arcgraph_storage::CHECKPOINT_SIDECAR_FILE)
            .is_file(),
        "first v6 checkpoint sidecar is missing"
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(generation)
        .context("read first v6 checkpoint sidecar during generation verification")?
        .context("first v6 checkpoint sidecar is absent")?;
    ensure!(
        checkpoint.incremental_metadata && !checkpoint.full_state_snapshot,
        "selected v6 checkpoint is not incremental metadata"
    );
    if require_unstamped {
        ensure!(
            checkpoint.checkpoint_lsn == migration_lsn,
            "first v6 checkpoint does not match the migration frontier"
        );
    }
    let checkpoint_metadata = arcgraph_storage::checkpoint::incremental_metadata_path(
        generation,
        checkpoint.checkpoint_lsn,
        checkpoint.metadata_generation,
    );
    ensure!(
        checkpoint_metadata.is_file(),
        "first v6 checkpoint metadata is missing"
    );
    ensure!(
        generation.join("wal").is_dir(),
        "fresh v6 WAL directory is missing"
    );
    if require_empty_wal {
        verify_semantically_empty_v6_wal(&generation.join("wal"))?;
    }
    ensure!(
        generation.join("MANIFEST").is_file(),
        "v6 MANIFEST is missing"
    );
    ensure!(
        generation.join(LSN_SEED_FILE).is_file(),
        "v6 LSN_SEED is missing"
    );
    let manifest = arcgraph_storage::read_data_dir_manifest(generation)
        .context("read v6 MANIFEST during generation verification")?
        .context("v6 MANIFEST is absent during generation verification")?;
    ensure!(
        manifest.data_dir_version == DATA_DIR_VERSION_M4
            && manifest.wal_format == WAL_FORMAT_DELTA_V10
            && manifest.record_store_format == RECORD_FORMAT_DIRECT_M4
            && manifest.migration_lsn == Some(migration_lsn.raw()),
        "v6 MANIFEST identity/frontier is inconsistent"
    );
    let expected_tenants = manifest
        .tenant_census
        .as_ref()
        .context("v6 MANIFEST is missing the tenant census")?;
    let actual_tenants: Vec<_> = tenants.iter().map(|tenant| tenant.raw()).collect();
    ensure!(
        expected_tenants == &actual_tenants,
        "v6 MANIFEST tenant census differs from the selected generation"
    );
    let expected_metadata_sha256 = manifest
        .checkpoint_metadata_sha256
        .as_deref()
        .context("v6 MANIFEST is missing the checkpoint metadata checksum")?;
    ensure!(
        expected_metadata_sha256.len() == 64
            && expected_metadata_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "v6 MANIFEST checkpoint metadata checksum is malformed"
    );
    let is_migration_checkpoint = checkpoint.checkpoint_lsn == migration_lsn;
    if require_unstamped || is_migration_checkpoint {
        ensure!(
            file_sha256(&checkpoint_metadata)? == expected_metadata_sha256,
            "v6 checkpoint metadata checksum differs from the MANIFEST"
        );
    }
    ensure!(
        read_lsn_seed(generation)? == migration_lsn.raw() + 1,
        "v6 LSN_SEED does not continue the migration frontier"
    );
    if require_unstamped || is_migration_checkpoint {
        verify_empty_v5_dpt(generation, &checkpoint, migration_lsn)
            .context("verify first v6 checkpoint DPT is empty")?;
    }
    if require_unstamped {
        ensure!(
            !generation.join("VERSION").exists(),
            "v6 VERSION must not exist before CURRENT is durable"
        );
    }
    Ok(())
}

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String is infallible");
    }
    Ok(encoded)
}

fn verify_semantically_empty_v6_wal(wal_dir: &Path) -> Result<()> {
    let segment_zero = segment_filename(0);
    let mut saw_segment_zero = false;
    let mut saw_dek = false;
    for entry in fs::read_dir(wal_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        ensure!(
            entry.file_type()?.is_file(),
            "v6 WAL contains a non-file entry {:?}",
            name
        );
        if name == OsStr::new(arcgraph_storage::WAL_DEK_SIDECAR_FILE) {
            ensure!(!saw_dek, "v6 WAL contains duplicate wal.dek entries");
            saw_dek = true;
            continue;
        }
        if name == OsStr::new(&segment_zero) {
            ensure!(
                !saw_segment_zero,
                "v6 WAL contains duplicate segment-0 entries"
            );
            let bytes = fs::read(entry.path()).context("read v6 WAL segment-0 header")?;
            ensure!(
                bytes.len() == SegmentHeader::SIZE,
                "v6 WAL segment-0 contains records or a torn header"
            );
            let header = SegmentHeader::decode(&bytes).context("decode v6 WAL segment-0 header")?;
            ensure!(
                header.format_version == BUNDLE_FORMAT_V10,
                "v6 WAL segment-0 is not delta-v10"
            );
            saw_segment_zero = true;
            continue;
        }
        bail!("v6 WAL contains unexpected entry {:?}", name);
    }
    Ok(())
}

pub(crate) fn v6_generation_tenants(
    generation: &Path,
) -> Result<std::collections::BTreeSet<arcgraph_core::TenantId>> {
    let tenants_root = generation.join(arcgraph_storage::m3_migration::M3_TENANTS_DIR);
    let mut tenants = std::collections::BTreeSet::new();
    for entry in fs::read_dir(&tenants_root)
        .with_context(|| format!("enumerate v6 tenants at {}", tenants_root.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let raw = entry
            .file_name()
            .to_str()
            .context("v6 tenant directory is not UTF-8")?
            .parse::<u64>()
            .context("v6 tenant directory is not a numeric tenant id")?;
        tenants.insert(arcgraph_core::TenantId::new(raw));
    }
    ensure!(
        tenants.contains(&arcgraph_core::TenantId::DEFAULT),
        "v6 generation is missing the DEFAULT tenant artifact set"
    );
    Ok(tenants)
}

fn copy_source_generation(root: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(root).context("enumerate v4 source generation")? {
        let entry = entry?;
        if excluded_root_entry(&entry.file_name()) {
            continue;
        }
        copy_entry(&entry.path(), &destination.join(entry.file_name()))?;
    }
    Ok(())
}

fn copy_entry(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing migration source symlink {}",
        source.display()
    );
    if metadata.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            copy_entry(&entry.path(), &destination.join(entry.file_name()))?;
        }
        fsync_dir(destination)?;
    } else if metadata.is_file() {
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
        }
        output.sync_all()?;
    } else {
        bail!("unsupported migration source entry {}", source.display());
    }
    Ok(())
}

fn fsync_tree(path: &Path) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            fsync_tree(&entry.path())?;
        } else if metadata.is_file() {
            File::open(entry.path())?.sync_all()?;
        }
    }
    fsync_dir(path).map_err(Into::into)
}

pub(crate) fn remove_tree_if_exists(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => fsync_dir(path.parent().unwrap_or_else(|| Path::new("."))).map_err(Into::into),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove stale {}", path.display())),
    }
}

fn copied_tree_bytes(root: &Path) -> Result<u64> {
    fn walk(path: &Path) -> Result<u64> {
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "refusing migration source symlink {}",
            path.display()
        );
        if metadata.is_file() {
            return Ok(metadata.len());
        }
        let mut total = 0u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            total = total
                .checked_add(walk(&entry.path())?)
                .context("migration source size overflow")?;
        }
        Ok(total)
    }
    let mut total = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !excluded_root_entry(&entry.file_name()) {
            total = total
                .checked_add(walk(&entry.path())?)
                .context("migration source size overflow")?;
        }
    }
    Ok(total)
}

#[cfg(unix)]
fn available_bytes(root: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(root.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a NUL-terminated CString valid for this call and
    // `stats` points at writable, correctly aligned storage. `statvfs` does
    // not retain either pointer. We inspect `stats` only after rc == 0.
    let rc = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("statvfs migration headroom");
    }
    // SAFETY: successful `statvfs` initialized the entire output struct.
    let stats = unsafe { stats.assume_init() };
    // `f_bavail`/`f_frsize` widths differ by platform: `c_ulong` (u64) on Linux,
    // narrower on macOS. The `as u64` casts are no-ops on Linux (clippy flags
    // them) but load-bearing on macOS. Suppress the Linux-only lints rather than
    // fork the build.
    #[allow(clippy::unnecessary_cast, clippy::useless_conversion)]
    Ok((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
}

#[cfg(not(unix))]
fn available_bytes(_root: &Path) -> Result<u64> {
    Ok(u64::MAX)
}
