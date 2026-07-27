//! Bounded-resident forward candidate B+tree for M4 owner rows.
//!
//! The direct owner row is authoritative. This index stores only
//! `(StrHash56, dense_id)` candidates in immutable, page-backed B+tree runs.
//! A lookup is therefore always candidate-then-verify: callers must fault the
//! direct row and compare the complete external key before accepting an id.
//!
//! Runs are merged by binary level (one live run per level). Both migration
//! input and run-to-run merges are chunked/streamed: no operation collects the
//! complete key set. Each run uses the same 8 KiB `IndexLeaf` /
//! `IndexInternal` page vocabulary as the secondary-index machinery. Leaves
//! stay on disk; only O(log N) run descriptors are resident.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
// Only the (cfg-gated) fault seam holds an `Arc`; a production build of this
// module has no use for it — which is the point: the seam compiles out.
#[cfg(any(test, feature = "fault-injection"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{PageHeader, PageId, PageType, TenantId};
use thiserror::Error;

const RUN_MAGIC: &[u8; 8] = b"AGOIBT02";
const RUN_VERSION: u32 = 2;
const RUN_HEADER_BYTES: u64 = PAGE_SIZE as u64;
const RUN_HEADER_CRC_OFFSET: usize = 64;
const RUN_HEADER_USED_BYTES: usize = 72;
const ENTRY_BYTES: usize = 16;

// Page layouts intentionally follow the secondary B+tree scaffold: a normal
// PageHeader followed by fixed-width, sorted entries. Owner leaves additionally
// carry a next-leaf pointer so an adversarial StrHash collision class can be
// verified without materializing it.
const LEAF_NEXT_OFFSET: usize = PageHeader::SIZE;
const LEAF_ENTRY_OFFSET: usize = PageHeader::SIZE + 8;
const LEAF_CAPACITY: usize = (PAGE_SIZE - LEAF_ENTRY_OFFSET) / ENTRY_BYTES;

const INTERNAL_FIRST_CHILD_OFFSET: usize = PageHeader::SIZE;
const INTERNAL_MIN_KEY_OFFSET: usize = PageHeader::SIZE + 8;
const INTERNAL_ENTRY_OFFSET: usize = PageHeader::SIZE + 8 + ENTRY_BYTES;
const INTERNAL_ENTRY_BYTES: usize = ENTRY_BYTES + 8;
const INTERNAL_CAPACITY: usize = (PAGE_SIZE - INTERNAL_ENTRY_OFFSET) / INTERNAL_ENTRY_BYTES;
const INTERNAL_FANOUT: usize = INTERNAL_CAPACITY + 1;

const _: () = assert!(LEAF_CAPACITY == 509);
const _: () = assert!(INTERNAL_CAPACITY == 338);

/// Bound on re-snapshot retries when a concurrent writer retires a run out
/// from under a lock-free reader (see [`OwnerForwardIndex::for_each_candidate`]).
///
/// Each retry observes a strictly newer published run set, and `insert_chunk`
/// serializes on the writer lock, so exhausting this bound means the run set is
/// churning pathologically — we fail closed rather than report a false miss.
const OWNER_INDEX_RUN_RETRY_LIMIT: u32 = 16;

const MANIFEST_NAME: &str = "MANIFEST";
const MANIFEST_TMP: &str = "MANIFEST.tmp";
const MANIFEST_VERSION: &str = "AGOIM2";

/// Maximum number of candidates sorted in RAM in one forward-index pass.
/// Larger iterators are split into independently published chunks.
pub const OWNER_INDEX_BATCH_ENTRIES: usize = 65_536;

/// Per-forward-index disk ceiling. The largest merge temporarily holds the
/// old runs, bounded intermediate runs, and one replacement run, so the
/// ceiling applies to every file in the index directory.
pub const OWNER_INDEX_DISK_CAP_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Stable 56-bit string hash used by owner forward indices.
#[must_use]
pub fn str_hash_56(value: &str) -> u64 {
    use std::hash::{Hash, Hasher};

    // Byte-identical to `arcgraph_index::hash_str_56`, the canonical
    // `SecondaryIndexValue::StrHash` contract. The canary below bites if a
    // toolchain changes the output and an explicit key migration is required.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    value.as_bytes().hash(&mut hash);
    hash.finish() & ((1_u64 << 56) - 1)
}

/// Typed owner-forward-index failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OwnerIndexError {
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// On-disk bytes or the manifest are malformed.
    #[error("owner forward index is corrupt: {0}")]
    Corrupt(String),
    /// A run named by a lock-free reader's snapshot was retired and unlinked
    /// by a concurrent writer before the reader could open it.
    ///
    /// Internal retry signal only: [`OwnerForwardIndex::for_each_candidate`]
    /// re-snapshots and rescans, and never returns this to a caller. It exists
    /// as a typed variant (rather than a bare `ErrorKind::NotFound`) so the
    /// retry cannot be confused with a genuine missing-file I/O fault, and so
    /// it can never be silently downgraded to a candidate miss.
    #[error("owner forward index run was retired mid-scan (internal retry signal)")]
    RunRetired,
    /// A write would exceed the configured bounded disk budget.
    #[error(
        "owner forward index disk budget exceeded: current={current} additional={additional} cap={cap}"
    )]
    DiskBudgetExceeded {
        /// Bytes already present in the index directory.
        current: u64,
        /// Worst-case bytes about to be added.
        additional: u64,
        /// Hard ceiling.
        cap: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    hash: u64,
    id: u64,
}

#[derive(Debug, Clone, Copy)]
struct RunHeader {
    count: u64,
    min_hash: u64,
    max_hash: u64,
    root_page: u64,
    page_count: u64,
    leaf_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunMeta {
    level: u32,
    generation: u64,
    count: u64,
    min_hash: u64,
    max_hash: u64,
    file_name: String,
}

/// Page-backed immutable-run owner candidate index.
pub struct OwnerForwardIndex {
    dir: PathBuf,
    runs: parking_lot::RwLock<Vec<RunMeta>>,
    writer: parking_lot::Mutex<()>,
    next_generation: AtomicU64,
    disk_cap_bytes: u64,
    /// Times a lock-free reader lost the race to a concurrent run retirement
    /// and re-snapshotted. Instrumentation for the concurrency gate: a
    /// zero count means the race never fired, so any test claiming to cover
    /// it is vacuous.
    retired_run_retries: AtomicU64,
    /// Fault-injection seam: invoked after the lock-free run-set snapshot is
    /// taken and BEFORE a run file is opened — i.e. exactly inside the window
    /// where a concurrent writer's `unlink` can strand the reader.
    ///
    /// It exists because the unlink race is far too narrow to hit reliably by
    /// thread racing — 700k concurrent lookups did not reproduce it once — and a
    /// RED-on-revert that fires only probabilistically is not a gate. Same
    /// discipline as the WAL/SIGKILL fault injection.
    ///
    /// **Compiled out of production.** Gated exactly like its sibling seam in
    /// `owner_row.rs`: a test/fault hook must never ship. Ungated it would pay
    /// an `RwLock` read + `Arc` clone on the HOT candidate-lookup path, and the
    /// `pub` installer would let any linking code park or panic a live reader.
    #[cfg(any(test, feature = "fault-injection"))]
    #[allow(clippy::type_complexity)]
    pre_open_hook: parking_lot::RwLock<Option<Arc<dyn Fn() + Send + Sync>>>,
    /// Fault-injection seam: number of upcoming retired-run unlinks to fail with
    /// a simulated non-`NotFound` I/O error. Compiled out of production.
    #[cfg(any(test, feature = "fault-injection"))]
    fail_next_unlinks: AtomicU64,
}

impl std::fmt::Debug for OwnerForwardIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerForwardIndex")
            .field("dir", &self.dir)
            .field("runs", &self.runs)
            .field("next_generation", &self.next_generation)
            .field("disk_cap_bytes", &self.disk_cap_bytes)
            .field("retired_run_retries", &self.retired_run_retries)
            .finish_non_exhaustive()
    }
}

impl OwnerForwardIndex {
    /// Create an empty index directory for an invisible migration build.
    pub fn create(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerIndexError> {
        fs::create_dir_all(path)?;
        let index = Self {
            dir: path.to_path_buf(),
            runs: parking_lot::RwLock::new(Vec::new()),
            retired_run_retries: AtomicU64::new(0),
            #[cfg(any(test, feature = "fault-injection"))]
            pre_open_hook: parking_lot::RwLock::new(None),
            #[cfg(any(test, feature = "fault-injection"))]
            fail_next_unlinks: AtomicU64::new(0),
            writer: parking_lot::Mutex::new(()),
            next_generation: AtomicU64::new(1),
            disk_cap_bytes,
        };
        index.publish_manifest(&[])?;
        Ok(index)
    }

    /// Open a fully-published index for the SERVE / incremental path.
    /// Missing/corrupt manifests fail loudly.
    ///
    /// `disk_cap_bytes` is the ABSOLUTE hard ceiling enforced on the write
    /// path (pre-M5-D3, unchanged semantics — every recovery/boot/serve
    /// caller goes through this path via [`crate::owner_row::OwnerRowRegistry::open_logical`]).
    /// Use [`Self::open_bulk`] for the bulk-load/migration build+verify
    /// path where a census-derived budget legitimately exceeds the
    /// incremental default (M5-D3 FIX 2 / #1518 skeptic review — the
    /// growth-above-published × 2.2 ratchet must NOT leak into the serve
    /// path).
    pub fn open(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerIndexError> {
        Self::open_with_cap(path, disk_cap_bytes, false)
    }

    /// Open a fully-published index for the BULK-LOAD / migration
    /// build+verify path only (M5-D3 amendment §5).
    ///
    /// `disk_cap_bytes` bounds GROWTH above the durably-published bytes: the
    /// effective ceiling is `existing_run_bytes × 2.2 + disk_cap_bytes` —
    /// the 2.2 factor is this index's own documented largest-merge
    /// transient (old runs + bounded intermediates + one replacement run,
    /// see [`OWNER_INDEX_DISK_CAP_BYTES`]), so a level merge that rewrites
    /// the published set stays admissible. A bulk-built index (census-
    /// derived budget) may legally exceed the incremental default;
    /// refusing its first incremental insert (whose cascading merge
    /// transiently doubles the largest run) would strand a legitimately-
    /// built store. The incremental constant keeps governing exactly what
    /// it was sized for — churn-bounded growth above a ratified build-time
    /// budget (Director ruling D-5).
    ///
    /// Do NOT call this from the serve/recovery/boot path — use
    /// [`Self::open`], whose ceiling is absolute.
    pub fn open_bulk(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerIndexError> {
        Self::open_with_cap(path, disk_cap_bytes, true)
    }

    fn open_with_cap(
        path: &Path,
        disk_cap_bytes: u64,
        growth_above_published: bool,
    ) -> Result<Self, OwnerIndexError> {
        let runs = read_manifest(path)?;
        let mut max_generation = 0_u64;
        // Baseline = manifest-live run bytes only: crash orphans are swept
        // below and must not inflate the growth budget.
        let mut existing = 0_u64;
        for run in &runs {
            validate_run(path, run)?;
            max_generation = max_generation.max(run.generation);
            existing = existing.saturating_add(fs::metadata(path.join(&run.file_name))?.len());
        }
        let effective_cap = if growth_above_published {
            (existing.saturating_mul(22) / 10).saturating_add(disk_cap_bytes)
        } else {
            disk_cap_bytes
        };
        let index = Self {
            dir: path.to_path_buf(),
            runs: parking_lot::RwLock::new(runs),
            writer: parking_lot::Mutex::new(()),
            next_generation: AtomicU64::new(max_generation.saturating_add(1)),
            disk_cap_bytes: effective_cap,
            retired_run_retries: AtomicU64::new(0),
            #[cfg(any(test, feature = "fault-injection"))]
            pre_open_hook: parking_lot::RwLock::new(None),
            #[cfg(any(test, feature = "fault-injection"))]
            fail_next_unlinks: AtomicU64::new(0),
        };
        index.sweep_orphan_runs()?;
        Ok(index)
    }

    /// Hard byte ceiling enforced for this index.
    #[must_use]
    pub const fn disk_cap_bytes(&self) -> u64 {
        self.disk_cap_bytes
    }

    /// Number of immutable run descriptors resident (at most one per binary
    /// level, O(log N)).
    #[must_use]
    pub fn resident_run_descriptors(&self) -> usize {
        self.runs.read().len()
    }

    /// Install the pre-open fault-injection hook. See `pre_open_hook`.
    ///
    /// Compiled out of production builds.
    #[cfg(any(test, feature = "fault-injection"))]
    #[doc(hidden)]
    pub fn set_pre_open_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        *self.pre_open_hook.write() = Some(hook);
    }

    /// Fail the next `count` retired-run unlinks with a simulated non-`NotFound`
    /// I/O error. Compiled out of production builds.
    #[cfg(any(test, feature = "fault-injection"))]
    #[doc(hidden)]
    pub fn fail_next_unlinks_for_gate(&self, count: u64) {
        self.fail_next_unlinks.store(count, Ordering::Release);
    }

    /// Times a lock-free reader re-snapshotted because a concurrent writer
    /// retired the run it was about to open.
    ///
    /// Exposed so the concurrency gate can assert the race ACTUALLY fired —
    /// a gate that never observes a retry is not testing anything.
    #[must_use]
    pub fn retired_run_retries(&self) -> u64 {
        self.retired_run_retries.load(Ordering::Acquire)
    }

    /// Number of candidate entries in the published run set.
    #[must_use]
    pub fn candidate_count(&self) -> u64 {
        self.runs.read().iter().map(|run| run.count).sum()
    }

    /// Add candidates with a hard in-memory sort-chunk bound. Each chunk is
    /// folded into same-level page-backed runs through streaming two-way
    /// merges; the iterator may be arbitrarily large without becoming a
    /// collect-all operation.
    pub fn insert_batch(
        &self,
        entries: impl IntoIterator<Item = (u64, u64)>,
    ) -> Result<(), OwnerIndexError> {
        let mut chunk = Vec::with_capacity(OWNER_INDEX_BATCH_ENTRIES);
        for (hash, id) in entries {
            chunk.push(Entry { hash, id });
            if chunk.len() == OWNER_INDEX_BATCH_ENTRIES {
                self.insert_chunk(&mut chunk)?;
            }
        }
        self.insert_chunk(&mut chunk)
    }

    /// Walk every candidate with the requested hash. The callback returns
    /// `true` only after the caller's full-key verification succeeds. A
    /// collision class is streamed across linked leaf pages, never collected.
    ///
    /// # Concurrency (the retired-run unlink race)
    ///
    /// The run set is read lock-free: `Self::scan_runs` clones the published
    /// `Vec<RunMeta>` under a short read lock and then releases it before
    /// touching the filesystem. A concurrent `Self::insert_chunk` may retire
    /// and `unlink` a run in that window, so `File::open` on a stale
    /// descriptor can lose the race and return `ENOENT`.
    ///
    /// That `ENOENT` is NOT "absent" — the entries still exist, in the merged
    /// replacement run. Reporting it as a miss would fail OPEN: the intern
    /// table would allocate a second id for an already-interned string, and
    /// the idempotency store would re-apply a committed request.
    ///
    /// The writer publishes the merged replacement (manifest fsync, then the
    /// `runs` swap) **before** it unlinks anything, so any snapshot taken
    /// after an `ENOENT` is guaranteed to contain a superset of the retired
    /// run's entries. Retrying against a *fresh* snapshot is therefore
    /// correct, and it terminates: each retry observes a strictly newer
    /// published generation, and the writer makes progress. If the run set
    /// churns more than `OWNER_INDEX_RUN_RETRY_LIMIT` times under one
    /// lookup we fail **closed** with a hard error rather than guess.
    pub fn for_each_candidate(
        &self,
        hash: u64,
        mut verify: impl FnMut(u64) -> Result<bool, OwnerIndexError>,
    ) -> Result<Option<u64>, OwnerIndexError> {
        for _ in 0..OWNER_INDEX_RUN_RETRY_LIMIT {
            match self.scan_runs(hash, &mut verify) {
                // A retired run was unlinked between our snapshot and the
                // open. Re-snapshot and rescan; never downgrade to `Ok(None)`.
                Err(OwnerIndexError::RunRetired) => {
                    self.retired_run_retries.fetch_add(1, Ordering::AcqRel);
                    continue;
                }
                other => return other,
            }
        }
        Err(OwnerIndexError::Corrupt(format!(
            "owner forward index run set churned more than {OWNER_INDEX_RUN_RETRY_LIMIT} times \
             during a single lookup of hash {hash}; refusing to report a candidate miss"
        )))
    }

    /// One lock-free pass over a freshly-cloned run snapshot.
    ///
    /// Returns [`OwnerIndexError::RunRetired`] iff a run named by this
    /// snapshot was unlinked before we could open it — the caller retries.
    fn scan_runs(
        &self,
        hash: u64,
        verify: &mut impl FnMut(u64) -> Result<bool, OwnerIndexError>,
    ) -> Result<Option<u64>, OwnerIndexError> {
        let runs = self.runs.read().clone();
        'runs: for run in runs.iter().rev() {
            if run.count == 0 || hash < run.min_hash || hash > run.max_hash {
                continue;
            }
            // The ONLY tolerated ENOENT in the read path, and only here: the
            // run named by our snapshot was retired+unlinked by a concurrent
            // writer. Every later read in this loop uses the already-open
            // descriptor, which survives the unlink (POSIX keeps the inode
            // alive for open fds), so an ENOENT cannot arise mid-scan.
            // Fault-injection point (compiled out of production): the reader has
            // a stale snapshot and has not yet opened the file. A writer that
            // retires this run right here is precisely the production race.
            #[cfg(any(test, feature = "fault-injection"))]
            {
                let hook = self.pre_open_hook.read().clone();
                if let Some(hook) = hook {
                    hook();
                }
            }
            let mut file = match File::open(self.dir.join(&run.file_name)) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(OwnerIndexError::RunRetired);
                }
                Err(error) => return Err(error.into()),
            };
            let header = read_run_header(&mut file)?;
            let needle = Entry { hash, id: 0 };
            let mut leaf_id = descend_to_leaf(&mut file, header, needle)?;
            let mut first_leaf = true;
            loop {
                let page = read_index_page(&mut file, leaf_id, header.page_count)?;
                let page_header = validate_index_page(&page, leaf_id, PageType::IndexLeaf)?;
                let count = usize::from(page_header.slot_count);
                if count == 0 || count > LEAF_CAPACITY {
                    return Err(OwnerIndexError::Corrupt(format!(
                        "leaf page {leaf_id} has invalid entry count {count}"
                    )));
                }
                let mut position = if first_leaf {
                    lower_bound_leaf(&page, count, needle)?
                } else {
                    0
                };
                first_leaf = false;
                while position < count {
                    let entry = decode_leaf_entry(&page, position)?;
                    if entry.hash > hash {
                        continue 'runs;
                    }
                    if entry.hash == hash && verify(entry.id)? {
                        return Ok(Some(entry.id));
                    }
                    position += 1;
                }
                let next = read_u64(&page[LEAF_NEXT_OFFSET..LEAF_ENTRY_OFFSET], "leaf next")?;
                if next == 0 {
                    continue 'runs;
                }
                if next != leaf_id.saturating_add(1) || next > header.leaf_count {
                    return Err(OwnerIndexError::Corrupt(format!(
                        "leaf page {leaf_id} has invalid next pointer {next}"
                    )));
                }
                leaf_id = next;
            }
        }
        Ok(None)
    }

    fn insert_chunk(&self, chunk: &mut Vec<Entry>) -> Result<(), OwnerIndexError> {
        if chunk.is_empty() {
            return Ok(());
        }
        chunk.sort_unstable();
        chunk.dedup();
        let _writer = self.writer.lock();
        let mut live = self.runs.read().clone();
        let mut replacement = self.write_entries_run(0, chunk)?;
        let mut retired = Vec::new();
        loop {
            let Some(position) = live.iter().position(|run| run.level == replacement.level) else {
                live.push(replacement);
                break;
            };
            let prior = live.remove(position);
            let merged = self.merge_runs(&prior, &replacement, replacement.level + 1)?;
            retired.push(prior);
            retired.push(replacement);
            replacement = merged;
        }
        live.sort_by_key(|run| run.level);
        self.publish_manifest(&live)?;
        *self.runs.write() = live;
        // Reclaim the retired runs. A transient unlink failure (EIO / EPERM /
        // EBUSY) must NOT abandon the rest of the list: the old code returned on
        // the first such error, skipping every remaining `remove_file` AND the
        // `sync_dir` below, so one blip leaked every other retired run's bytes
        // against `enforce_budget` for the process lifetime.
        //
        // Correctness never depended on this: the runs are already unpublished
        // (manifest + `runs` swap happened above), so no reader can reach them,
        // and `sweep_orphan_runs` reclaims any straggler on the next `open()`.
        // So log and keep going — a leaked file is a bounded disk cost, not a
        // reason to stop reclaiming the others.
        for run in retired {
            match self.remove_retired_run(&run.file_name) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(
                        run = %run.file_name,
                        %error,
                        "owner forward index: could not unlink a retired run; \
                         continuing with the rest (reclaimed by sweep_orphan_runs \
                         on next open)"
                    );
                }
            }
        }
        sync_dir(&self.dir)?;
        chunk.clear();
        Ok(())
    }

    /// `fs::remove_file` for a retired run, with a fault seam that is compiled
    /// out of production builds.
    fn remove_retired_run(&self, file_name: &str) -> std::io::Result<()> {
        #[cfg(any(test, feature = "fault-injection"))]
        {
            if self
                .fail_next_unlinks
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "injected transient unlink failure",
                ));
            }
        }
        fs::remove_file(self.dir.join(file_name))
    }

    fn write_entries_run(&self, level: u32, entries: &[Entry]) -> Result<RunMeta, OwnerIndexError> {
        self.enforce_budget(run_bytes_upper_bound(entries.len() as u64)?)?;
        let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
        let file_name = format!("run-{level:02}-{generation:020}.idx");
        let tmp = self.dir.join(format!("{file_name}.tmp"));
        let final_path = self.dir.join(&file_name);
        let mut builder = RunBuilder::create(&tmp)?;
        for entry in entries {
            builder.push(*entry)?;
        }
        let header = builder.finish()?;
        fs::rename(&tmp, &final_path)?;
        sync_dir(&self.dir)?;
        Ok(run_meta(level, generation, file_name, header))
    }

    fn merge_runs(
        &self,
        left: &RunMeta,
        right: &RunMeta,
        level: u32,
    ) -> Result<RunMeta, OwnerIndexError> {
        let max_count = left.count.saturating_add(right.count);
        self.enforce_budget(run_bytes_upper_bound(max_count)?)?;
        let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
        let file_name = format!("run-{level:02}-{generation:020}.idx");
        let tmp = self.dir.join(format!("{file_name}.tmp"));
        let final_path = self.dir.join(&file_name);
        let mut left_reader = RunEntryReader::open(&self.dir.join(&left.file_name))?;
        let mut right_reader = RunEntryReader::open(&self.dir.join(&right.file_name))?;
        let mut left_entry = left_reader.next_entry()?;
        let mut right_entry = right_reader.next_entry()?;
        let mut builder = RunBuilder::create(&tmp)?;
        while left_entry.is_some() || right_entry.is_some() {
            let entry = match (left_entry, right_entry) {
                (Some(left_value), Some(right_value)) if left_value <= right_value => {
                    left_entry = left_reader.next_entry()?;
                    left_value
                }
                (Some(_), Some(right_value)) => {
                    right_entry = right_reader.next_entry()?;
                    right_value
                }
                (Some(left_value), None) => {
                    left_entry = left_reader.next_entry()?;
                    left_value
                }
                (None, Some(right_value)) => {
                    right_entry = right_reader.next_entry()?;
                    right_value
                }
                (None, None) => break,
            };
            builder.push(entry)?;
        }
        let header = builder.finish()?;
        fs::rename(&tmp, &final_path)?;
        sync_dir(&self.dir)?;
        Ok(run_meta(level, generation, file_name, header))
    }

    fn publish_manifest(&self, runs: &[RunMeta]) -> Result<(), OwnerIndexError> {
        let tmp = self.dir.join(MANIFEST_TMP);
        let mut file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?,
        );
        writeln!(file, "{MANIFEST_VERSION}")?;
        for run in runs {
            writeln!(
                file,
                "{} {} {} {} {} {}",
                run.level, run.generation, run.count, run.min_hash, run.max_hash, run.file_name
            )?;
        }
        let raw = file.into_inner().map_err(|error| error.into_error())?;
        raw.sync_all()?;
        fs::rename(tmp, self.dir.join(MANIFEST_NAME))?;
        sync_dir(&self.dir)?;
        Ok(())
    }

    fn enforce_budget(&self, additional: u64) -> Result<(), OwnerIndexError> {
        let current = directory_bytes(&self.dir)?;
        if current.saturating_add(additional) > self.disk_cap_bytes {
            return Err(OwnerIndexError::DiskBudgetExceeded {
                current,
                additional,
                cap: self.disk_cap_bytes,
            });
        }
        Ok(())
    }

    fn sweep_orphan_runs(&self) -> Result<(), OwnerIndexError> {
        let live: std::collections::BTreeSet<String> = self
            .runs
            .read()
            .iter()
            .map(|run| run.file_name.clone())
            .collect();
        let mut removed = false;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if (name.starts_with("run-") && (name.ends_with(".idx") || name.ends_with(".tmp")))
                && !live.contains(name.as_ref())
            {
                fs::remove_file(entry.path())?;
                removed = true;
            }
        }
        if removed {
            sync_dir(&self.dir)?;
        }
        Ok(())
    }
}

fn run_meta(level: u32, generation: u64, file_name: String, header: RunHeader) -> RunMeta {
    RunMeta {
        level,
        generation,
        count: header.count,
        min_hash: header.min_hash,
        max_hash: header.max_hash,
        file_name,
    }
}

struct RunBuilder {
    path: PathBuf,
    writer: BufWriter<File>,
    current_leaf: Vec<Entry>,
    previous: Option<Entry>,
    count: u64,
    min_hash: u64,
    max_hash: u64,
    leaf_count: u64,
}

impl RunBuilder {
    fn create(path: &Path) -> Result<Self, OwnerIndexError> {
        let mut writer =
            BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
        writer.write_all(&[0_u8; PAGE_SIZE])?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            current_leaf: Vec::with_capacity(LEAF_CAPACITY),
            previous: None,
            count: 0,
            min_hash: 0,
            max_hash: 0,
            leaf_count: 0,
        })
    }

    fn push(&mut self, entry: Entry) -> Result<(), OwnerIndexError> {
        if self.previous == Some(entry) {
            return Ok(());
        }
        if self.previous.is_some_and(|previous| previous > entry) {
            return Err(OwnerIndexError::Corrupt(
                "run builder input is not sorted".to_owned(),
            ));
        }
        if self.current_leaf.len() == LEAF_CAPACITY {
            self.flush_leaf(true)?;
        }
        if self.count == 0 {
            self.min_hash = entry.hash;
        }
        self.max_hash = entry.hash;
        self.count = self.count.saturating_add(1);
        self.previous = Some(entry);
        self.current_leaf.push(entry);
        Ok(())
    }

    fn flush_leaf(&mut self, has_next: bool) -> Result<(), OwnerIndexError> {
        if self.current_leaf.is_empty() {
            return Ok(());
        }
        let page_id = self.leaf_count.saturating_add(1);
        let next = if has_next {
            page_id.saturating_add(1)
        } else {
            0
        };
        let page = encode_leaf_page(page_id, next, &self.current_leaf)?;
        self.writer.write_all(page.as_ref())?;
        self.leaf_count = page_id;
        self.current_leaf.clear();
        Ok(())
    }

    fn finish(mut self) -> Result<RunHeader, OwnerIndexError> {
        self.flush_leaf(false)?;
        if self.count == 0 || self.leaf_count == 0 {
            return Err(OwnerIndexError::Corrupt(
                "cannot publish an empty owner-index run".to_owned(),
            ));
        }
        let raw = self
            .writer
            .into_inner()
            .map_err(|error| error.into_error())?;
        raw.sync_all()?;

        let mut page_count = self.leaf_count;
        let mut child_start = 1_u64;
        let mut child_count = self.leaf_count;
        while child_count > 1 {
            let parent_start = page_count.saturating_add(1);
            let parent_count =
                append_internal_level(&self.path, child_start, child_count, parent_start)?;
            page_count = page_count.saturating_add(parent_count);
            child_start = parent_start;
            child_count = parent_count;
        }
        let header = RunHeader {
            count: self.count,
            min_hash: self.min_hash,
            max_hash: self.max_hash,
            root_page: child_start,
            page_count,
            leaf_count: self.leaf_count,
        };
        write_run_header(&self.path, header)?;
        Ok(header)
    }
}

fn append_internal_level(
    path: &Path,
    child_start: u64,
    child_count: u64,
    parent_start: u64,
) -> Result<u64, OwnerIndexError> {
    let mut reader = File::open(path)?;
    let writer = OpenOptions::new().append(true).open(path)?;
    let mut writer = BufWriter::new(writer);
    let mut consumed = 0_u64;
    let mut parent_count = 0_u64;
    while consumed < child_count {
        let group_len = (child_count - consumed).min(INTERNAL_FANOUT as u64);
        let first_child = child_start.saturating_add(consumed);
        let mut children = Vec::with_capacity(group_len as usize);
        for offset in 0..group_len {
            let child = first_child.saturating_add(offset);
            children.push((page_min_key(&mut reader, child)?, child));
        }
        let page_id = parent_start.saturating_add(parent_count);
        let page = encode_internal_page(page_id, &children)?;
        writer.write_all(page.as_ref())?;
        consumed = consumed.saturating_add(group_len);
        parent_count = parent_count.saturating_add(1);
    }
    let raw = writer.into_inner().map_err(|error| error.into_error())?;
    raw.sync_all()?;
    Ok(parent_count)
}

fn encode_leaf_page(
    page_id: u64,
    next: u64,
    entries: &[Entry],
) -> Result<Box<[u8; PAGE_SIZE]>, OwnerIndexError> {
    if entries.is_empty() || entries.len() > LEAF_CAPACITY {
        return Err(OwnerIndexError::Corrupt(format!(
            "cannot encode leaf with {} entries",
            entries.len()
        )));
    }
    let mut page = Box::new([0_u8; PAGE_SIZE]);
    page[LEAF_NEXT_OFFSET..LEAF_ENTRY_OFFSET].copy_from_slice(&next.to_le_bytes());
    for (position, entry) in entries.iter().enumerate() {
        let offset = LEAF_ENTRY_OFFSET + position * ENTRY_BYTES;
        page[offset..offset + ENTRY_BYTES].copy_from_slice(&encode_entry(*entry));
    }
    finish_page_header(&mut page, page_id, PageType::IndexLeaf, entries.len())?;
    Ok(page)
}

fn encode_internal_page(
    page_id: u64,
    children: &[(Entry, u64)],
) -> Result<Box<[u8; PAGE_SIZE]>, OwnerIndexError> {
    if children.is_empty() || children.len() > INTERNAL_FANOUT {
        return Err(OwnerIndexError::Corrupt(format!(
            "cannot encode internal page with {} children",
            children.len()
        )));
    }
    let mut page = Box::new([0_u8; PAGE_SIZE]);
    page[INTERNAL_FIRST_CHILD_OFFSET..INTERNAL_MIN_KEY_OFFSET]
        .copy_from_slice(&children[0].1.to_le_bytes());
    page[INTERNAL_MIN_KEY_OFFSET..INTERNAL_ENTRY_OFFSET]
        .copy_from_slice(&encode_entry(children[0].0));
    for (position, (separator, child)) in children.iter().skip(1).enumerate() {
        let offset = INTERNAL_ENTRY_OFFSET + position * INTERNAL_ENTRY_BYTES;
        page[offset..offset + ENTRY_BYTES].copy_from_slice(&encode_entry(*separator));
        page[offset + ENTRY_BYTES..offset + INTERNAL_ENTRY_BYTES]
            .copy_from_slice(&child.to_le_bytes());
    }
    finish_page_header(
        &mut page,
        page_id,
        PageType::IndexInternal,
        children.len() - 1,
    )?;
    Ok(page)
}

fn finish_page_header(
    page: &mut [u8; PAGE_SIZE],
    page_id: u64,
    page_type: PageType,
    slots: usize,
) -> Result<(), OwnerIndexError> {
    let slot_count = u16::try_from(slots)
        .map_err(|_| OwnerIndexError::Corrupt("index page slot count overflows u16".to_owned()))?;
    let mut header = PageHeader::new(PageId::new(page_id), page_type, TenantId::SYSTEM);
    header.slot_count = slot_count;
    header.checksum = crc32c::crc32c(&page[PageHeader::SIZE..]);
    page[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
    Ok(())
}

fn write_run_header(path: &Path, header: RunHeader) -> Result<(), OwnerIndexError> {
    let mut bytes = [0_u8; PAGE_SIZE];
    bytes[..8].copy_from_slice(RUN_MAGIC);
    bytes[8..12].copy_from_slice(&RUN_VERSION.to_le_bytes());
    bytes[16..24].copy_from_slice(&header.count.to_le_bytes());
    bytes[24..32].copy_from_slice(&header.min_hash.to_le_bytes());
    bytes[32..40].copy_from_slice(&header.max_hash.to_le_bytes());
    bytes[40..48].copy_from_slice(&header.root_page.to_le_bytes());
    bytes[48..56].copy_from_slice(&header.page_count.to_le_bytes());
    bytes[56..64].copy_from_slice(&header.leaf_count.to_le_bytes());
    let crc = crc32c::crc32c(&bytes[..RUN_HEADER_CRC_OFFSET]);
    bytes[RUN_HEADER_CRC_OFFSET..RUN_HEADER_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_run_header(file: &mut File) -> Result<RunHeader, OwnerIndexError> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = [0_u8; PAGE_SIZE];
    file.read_exact(&mut bytes)?;
    if &bytes[..8] != RUN_MAGIC {
        return Err(OwnerIndexError::Corrupt("invalid run magic".to_owned()));
    }
    if read_u32(&bytes[8..12], "run version")? != RUN_VERSION {
        return Err(OwnerIndexError::Corrupt(
            "unsupported owner-index run version".to_owned(),
        ));
    }
    if bytes[12..16] != [0; 4]
        || bytes[RUN_HEADER_CRC_OFFSET + 4..RUN_HEADER_USED_BYTES] != [0; 4]
        || bytes[RUN_HEADER_USED_BYTES..].iter().any(|byte| *byte != 0)
    {
        return Err(OwnerIndexError::Corrupt(
            "run header reserved bytes are nonzero".to_owned(),
        ));
    }
    let stored_crc = read_u32(
        &bytes[RUN_HEADER_CRC_OFFSET..RUN_HEADER_CRC_OFFSET + 4],
        "run header crc",
    )?;
    if crc32c::crc32c(&bytes[..RUN_HEADER_CRC_OFFSET]) != stored_crc {
        return Err(OwnerIndexError::Corrupt(
            "run header checksum mismatch".to_owned(),
        ));
    }
    let header = RunHeader {
        count: read_u64(&bytes[16..24], "run count")?,
        min_hash: read_u64(&bytes[24..32], "run min hash")?,
        max_hash: read_u64(&bytes[32..40], "run max hash")?,
        root_page: read_u64(&bytes[40..48], "run root page")?,
        page_count: read_u64(&bytes[48..56], "run page count")?,
        leaf_count: read_u64(&bytes[56..64], "run leaf count")?,
    };
    if header.count == 0
        || header.min_hash > header.max_hash
        || header.leaf_count == 0
        || header.leaf_count > header.page_count
        || header.root_page == 0
        || header.root_page > header.page_count
    {
        return Err(OwnerIndexError::Corrupt(
            "run header carries impossible bounds".to_owned(),
        ));
    }
    Ok(header)
}

fn read_manifest(path: &Path) -> Result<Vec<RunMeta>, OwnerIndexError> {
    let file = File::open(path.join(MANIFEST_NAME))?;
    let mut lines = BufReader::new(file).lines();
    let version = lines
        .next()
        .transpose()?
        .ok_or_else(|| OwnerIndexError::Corrupt("empty manifest".to_owned()))?;
    if version != MANIFEST_VERSION {
        return Err(OwnerIndexError::Corrupt(format!(
            "manifest version {version:?} is unsupported"
        )));
    }
    let mut runs = Vec::new();
    for line in lines {
        let line = line?;
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() != 6 {
            return Err(OwnerIndexError::Corrupt(format!(
                "malformed manifest line {line:?}"
            )));
        }
        let parse = |field: &str, name: &str| {
            field.parse::<u64>().map_err(|_| {
                OwnerIndexError::Corrupt(format!("invalid {name} in manifest line {line:?}"))
            })
        };
        let level = u32::try_from(parse(fields[0], "level")?)
            .map_err(|_| OwnerIndexError::Corrupt("run level overflows u32".to_owned()))?;
        let run = RunMeta {
            level,
            generation: parse(fields[1], "generation")?,
            count: parse(fields[2], "count")?,
            min_hash: parse(fields[3], "min hash")?,
            max_hash: parse(fields[4], "max hash")?,
            file_name: fields[5].to_owned(),
        };
        if run.file_name.contains('/') || run.file_name.contains("..") {
            return Err(OwnerIndexError::Corrupt(
                "manifest run name escapes index directory".to_owned(),
            ));
        }
        runs.push(run);
    }
    runs.sort_by_key(|run| run.level);
    if runs.windows(2).any(|pair| pair[0].level == pair[1].level) {
        return Err(OwnerIndexError::Corrupt(
            "manifest contains two runs at one level".to_owned(),
        ));
    }
    Ok(runs)
}

fn validate_run(path: &Path, expected: &RunMeta) -> Result<(), OwnerIndexError> {
    let mut file = File::open(path.join(&expected.file_name))?;
    let header = read_run_header(&mut file)?;
    if (header.count, header.min_hash, header.max_hash)
        != (expected.count, expected.min_hash, expected.max_hash)
    {
        return Err(OwnerIndexError::Corrupt(format!(
            "run {} header disagrees with manifest",
            expected.file_name
        )));
    }
    let expected_len = RUN_HEADER_BYTES
        .checked_add(
            header
                .page_count
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| OwnerIndexError::Corrupt("run page length wraps".to_owned()))?,
        )
        .ok_or_else(|| OwnerIndexError::Corrupt("run byte length wraps".to_owned()))?;
    if file.metadata()?.len() != expected_len {
        return Err(OwnerIndexError::Corrupt(format!(
            "run {} has wrong byte length",
            expected.file_name
        )));
    }

    let mut reader = RunEntryReader::from_open_file(file, header)?;
    let mut count = 0_u64;
    let mut first = None;
    let mut previous = None;
    while let Some(entry) = reader.next_entry()? {
        if previous.is_some_and(|prior| prior >= entry) {
            return Err(OwnerIndexError::Corrupt(format!(
                "run {} is not strictly sorted",
                expected.file_name
            )));
        }
        first.get_or_insert(entry);
        previous = Some(entry);
        count = count.saturating_add(1);
    }
    if count != header.count
        || first.map(|entry| entry.hash) != Some(header.min_hash)
        || previous.map(|entry| entry.hash) != Some(header.max_hash)
    {
        return Err(OwnerIndexError::Corrupt(format!(
            "run {} entry census disagrees with header",
            expected.file_name
        )));
    }

    // Validate every internal page and its child fences without retaining an
    // O(number-of-pages) in-memory census.
    let mut file = reader.into_file();
    for page_id in header.leaf_count.saturating_add(1)..=header.page_count {
        let page = read_index_page(&mut file, page_id, header.page_count)?;
        let page_header = validate_index_page(&page, page_id, PageType::IndexInternal)?;
        let entries = usize::from(page_header.slot_count);
        if entries > INTERNAL_CAPACITY {
            return Err(OwnerIndexError::Corrupt(format!(
                "internal page {page_id} exceeds capacity"
            )));
        }
        let first_child = read_u64(
            &page[INTERNAL_FIRST_CHILD_OFFSET..INTERNAL_MIN_KEY_OFFSET],
            "internal first child",
        )?;
        if first_child == 0 || first_child >= page_id {
            return Err(OwnerIndexError::Corrupt(format!(
                "internal page {page_id} has invalid first child {first_child}"
            )));
        }
        let stored_min = decode_entry(&page[INTERNAL_MIN_KEY_OFFSET..INTERNAL_ENTRY_OFFSET])?;
        if page_min_key(&mut file, first_child)? != stored_min {
            return Err(OwnerIndexError::Corrupt(format!(
                "internal page {page_id} has a wrong minimum fence"
            )));
        }
        let mut prior = stored_min;
        for position in 0..entries {
            let (separator, child) = decode_internal_entry(&page, position)?;
            if separator <= prior || child == 0 || child >= page_id {
                return Err(OwnerIndexError::Corrupt(format!(
                    "internal page {page_id} has an invalid child fence"
                )));
            }
            if page_min_key(&mut file, child)? != separator {
                return Err(OwnerIndexError::Corrupt(format!(
                    "internal page {page_id} separator differs from child minimum"
                )));
            }
            prior = separator;
        }
    }
    if header.leaf_count == 1 {
        if header.root_page != 1 || header.page_count != 1 {
            return Err(OwnerIndexError::Corrupt(
                "single-leaf run has an invalid root".to_owned(),
            ));
        }
    } else if header.root_page != header.page_count {
        return Err(OwnerIndexError::Corrupt(
            "multi-page run root is not the final page".to_owned(),
        ));
    }
    Ok(())
}

struct RunEntryReader {
    file: File,
    header: RunHeader,
    leaf_id: u64,
    page: Option<Box<[u8; PAGE_SIZE]>>,
    position: usize,
}

impl RunEntryReader {
    fn open(path: &Path) -> Result<Self, OwnerIndexError> {
        let mut file = File::open(path)?;
        let header = read_run_header(&mut file)?;
        Self::from_open_file(file, header)
    }

    fn from_open_file(file: File, header: RunHeader) -> Result<Self, OwnerIndexError> {
        if header.leaf_count == 0 {
            return Err(OwnerIndexError::Corrupt("run has no leaves".to_owned()));
        }
        Ok(Self {
            file,
            header,
            leaf_id: 1,
            page: None,
            position: 0,
        })
    }

    fn next_entry(&mut self) -> Result<Option<Entry>, OwnerIndexError> {
        loop {
            if self.page.is_none() {
                let page = read_index_page(&mut self.file, self.leaf_id, self.header.page_count)?;
                let header = validate_index_page(&page, self.leaf_id, PageType::IndexLeaf)?;
                let count = usize::from(header.slot_count);
                if count == 0 || count > LEAF_CAPACITY {
                    return Err(OwnerIndexError::Corrupt(format!(
                        "leaf page {} has invalid count {count}",
                        self.leaf_id
                    )));
                }
                self.page = Some(page);
                self.position = 0;
            }
            let page = self.page.as_ref().ok_or_else(|| {
                OwnerIndexError::Corrupt("run reader lost its leaf page".to_owned())
            })?;
            let count = usize::from(read_page_header(page)?.slot_count);
            if self.position < count {
                let entry = decode_leaf_entry(page, self.position)?;
                self.position += 1;
                return Ok(Some(entry));
            }
            let next = read_u64(&page[LEAF_NEXT_OFFSET..LEAF_ENTRY_OFFSET], "leaf next")?;
            if next == 0 {
                if self.leaf_id != self.header.leaf_count {
                    return Err(OwnerIndexError::Corrupt(
                        "leaf chain terminated before leaf_count".to_owned(),
                    ));
                }
                return Ok(None);
            }
            if next != self.leaf_id.saturating_add(1) || next > self.header.leaf_count {
                return Err(OwnerIndexError::Corrupt(format!(
                    "leaf {} has invalid next pointer {next}",
                    self.leaf_id
                )));
            }
            self.leaf_id = next;
            self.page = None;
        }
    }

    fn into_file(self) -> File {
        self.file
    }
}

fn descend_to_leaf(
    file: &mut File,
    header: RunHeader,
    needle: Entry,
) -> Result<u64, OwnerIndexError> {
    let mut page_id = header.root_page;
    for _ in 0..=header.page_count {
        let page = read_index_page(file, page_id, header.page_count)?;
        let page_header = read_page_header(&page)?;
        match PageType::from_byte(page_header.page_type).map_err(|error| {
            OwnerIndexError::Corrupt(format!("invalid index page type: {error}"))
        })? {
            PageType::IndexLeaf => {
                validate_index_page(&page, page_id, PageType::IndexLeaf)?;
                if page_id > header.leaf_count {
                    return Err(OwnerIndexError::Corrupt(
                        "B+tree descent reached a non-leaf-range leaf".to_owned(),
                    ));
                }
                return Ok(page_id);
            }
            PageType::IndexInternal => {
                let page_header = validate_index_page(&page, page_id, PageType::IndexInternal)?;
                let entries = usize::from(page_header.slot_count);
                if entries > INTERNAL_CAPACITY {
                    return Err(OwnerIndexError::Corrupt(format!(
                        "internal page {page_id} exceeds capacity"
                    )));
                }
                let mut child = read_u64(
                    &page[INTERNAL_FIRST_CHILD_OFFSET..INTERNAL_MIN_KEY_OFFSET],
                    "internal first child",
                )?;
                for position in 0..entries {
                    let (separator, right_child) = decode_internal_entry(&page, position)?;
                    if needle < separator {
                        break;
                    }
                    child = right_child;
                }
                if child == 0 || child >= page_id {
                    return Err(OwnerIndexError::Corrupt(format!(
                        "internal page {page_id} selected invalid child {child}"
                    )));
                }
                page_id = child;
            }
            other => {
                return Err(OwnerIndexError::Corrupt(format!(
                    "owner B+tree contains unexpected page type {other:?}"
                )));
            }
        }
    }
    Err(OwnerIndexError::Corrupt(
        "owner B+tree descent exceeded page-count bound".to_owned(),
    ))
}

fn page_min_key(file: &mut File, page_id: u64) -> Result<Entry, OwnerIndexError> {
    let page_count = file
        .metadata()?
        .len()
        .checked_div(PAGE_SIZE as u64)
        .unwrap_or(0)
        .saturating_sub(1);
    let page = read_index_page(file, page_id, page_count)?;
    let header = read_page_header(&page)?;
    match PageType::from_byte(header.page_type)
        .map_err(|error| OwnerIndexError::Corrupt(format!("invalid page type: {error}")))?
    {
        PageType::IndexLeaf => {
            validate_index_page(&page, page_id, PageType::IndexLeaf)?;
            if header.slot_count == 0 {
                return Err(OwnerIndexError::Corrupt(format!(
                    "leaf page {page_id} has no minimum"
                )));
            }
            decode_leaf_entry(&page, 0)
        }
        PageType::IndexInternal => {
            validate_index_page(&page, page_id, PageType::IndexInternal)?;
            decode_entry(&page[INTERNAL_MIN_KEY_OFFSET..INTERNAL_ENTRY_OFFSET])
        }
        other => Err(OwnerIndexError::Corrupt(format!(
            "owner B+tree minimum read saw unexpected type {other:?}"
        ))),
    }
}

fn read_index_page(
    file: &mut File,
    page_id: u64,
    page_count: u64,
) -> Result<Box<[u8; PAGE_SIZE]>, OwnerIndexError> {
    if page_id == 0 || page_id > page_count {
        return Err(OwnerIndexError::Corrupt(format!(
            "index page id {page_id} is outside 1..={page_count}"
        )));
    }
    let offset = page_id
        .checked_mul(PAGE_SIZE as u64)
        .ok_or_else(|| OwnerIndexError::Corrupt("index page offset wraps".to_owned()))?;
    file.seek(SeekFrom::Start(offset))?;
    let mut page = Box::new([0_u8; PAGE_SIZE]);
    file.read_exact(page.as_mut())?;
    Ok(page)
}

fn read_page_header(page: &[u8; PAGE_SIZE]) -> Result<PageHeader, OwnerIndexError> {
    let bytes: &[u8; PageHeader::SIZE] = page[..PageHeader::SIZE]
        .try_into()
        .map_err(|_| OwnerIndexError::Corrupt("index page header width changed".to_owned()))?;
    PageHeader::from_bytes(bytes)
        .map_err(|error| OwnerIndexError::Corrupt(format!("invalid index page header: {error}")))
}

fn validate_index_page(
    page: &[u8; PAGE_SIZE],
    page_id: u64,
    expected_type: PageType,
) -> Result<PageHeader, OwnerIndexError> {
    let header = read_page_header(page)?;
    if header.page_id != page_id
        || header.page_type != expected_type.as_byte()
        || header.tenant_id != TenantId::SYSTEM.raw()
        || header.lsn != 0
        || header.flags != 0
        || crc32c::crc32c(&page[PageHeader::SIZE..]) != header.checksum
    {
        return Err(OwnerIndexError::Corrupt(format!(
            "index page {page_id} identity/checksum mismatch"
        )));
    }
    Ok(header)
}

fn lower_bound_leaf(
    page: &[u8; PAGE_SIZE],
    count: usize,
    needle: Entry,
) -> Result<usize, OwnerIndexError> {
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = low + (high - low) / 2;
        if decode_leaf_entry(page, middle)? < needle {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

fn decode_leaf_entry(page: &[u8; PAGE_SIZE], position: usize) -> Result<Entry, OwnerIndexError> {
    if position >= LEAF_CAPACITY {
        return Err(OwnerIndexError::Corrupt(format!(
            "leaf entry position {position} exceeds capacity"
        )));
    }
    let offset = LEAF_ENTRY_OFFSET + position * ENTRY_BYTES;
    decode_entry(&page[offset..offset + ENTRY_BYTES])
}

fn decode_internal_entry(
    page: &[u8; PAGE_SIZE],
    position: usize,
) -> Result<(Entry, u64), OwnerIndexError> {
    if position >= INTERNAL_CAPACITY {
        return Err(OwnerIndexError::Corrupt(format!(
            "internal entry position {position} exceeds capacity"
        )));
    }
    let offset = INTERNAL_ENTRY_OFFSET + position * INTERNAL_ENTRY_BYTES;
    Ok((
        decode_entry(&page[offset..offset + ENTRY_BYTES])?,
        read_u64(
            &page[offset + ENTRY_BYTES..offset + INTERNAL_ENTRY_BYTES],
            "internal child",
        )?,
    ))
}

fn encode_entry(entry: Entry) -> [u8; ENTRY_BYTES] {
    let mut bytes = [0_u8; ENTRY_BYTES];
    bytes[..8].copy_from_slice(&entry.hash.to_le_bytes());
    bytes[8..].copy_from_slice(&entry.id.to_le_bytes());
    bytes
}

fn decode_entry(bytes: &[u8]) -> Result<Entry, OwnerIndexError> {
    if bytes.len() != ENTRY_BYTES {
        return Err(OwnerIndexError::Corrupt(
            "owner index entry has wrong width".to_owned(),
        ));
    }
    Ok(Entry {
        hash: read_u64(&bytes[..8], "entry hash")?,
        id: read_u64(&bytes[8..], "entry id")?,
    })
}

fn read_u32(bytes: &[u8], field: &str) -> Result<u32, OwnerIndexError> {
    let array: [u8; 4] = bytes
        .try_into()
        .map_err(|_| OwnerIndexError::Corrupt(format!("invalid {field} width")))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], field: &str) -> Result<u64, OwnerIndexError> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| OwnerIndexError::Corrupt(format!("invalid {field} width")))?;
    Ok(u64::from_le_bytes(array))
}

fn run_bytes_upper_bound(count: u64) -> Result<u64, OwnerIndexError> {
    if count == 0 {
        return Ok(RUN_HEADER_BYTES);
    }
    let leaf_capacity = LEAF_CAPACITY as u64;
    let fanout = INTERNAL_FANOUT as u64;
    let mut level = count
        .checked_add(leaf_capacity - 1)
        .ok_or_else(|| OwnerIndexError::Corrupt("leaf count arithmetic wraps".to_owned()))?
        / leaf_capacity;
    let mut pages = level;
    while level > 1 {
        level = level
            .checked_add(fanout - 1)
            .ok_or_else(|| OwnerIndexError::Corrupt("tree level arithmetic wraps".to_owned()))?
            / fanout;
        pages = pages
            .checked_add(level)
            .ok_or_else(|| OwnerIndexError::Corrupt("tree page count wraps".to_owned()))?;
    }
    RUN_HEADER_BYTES
        .checked_add(
            pages
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| OwnerIndexError::Corrupt("tree byte count wraps".to_owned()))?,
        )
        .ok_or_else(|| OwnerIndexError::Corrupt("run byte count wraps".to_owned()))
}

fn directory_bytes(path: &Path) -> Result<u64, std::io::Error> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn sync_dir(path: &Path) -> Result<(), std::io::Error> {
    File::open(path)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_CHUNK: u64 = 400;

    fn churn(index: &OwnerForwardIndex, chunk: u64) {
        let base = chunk * GATE_CHUNK;
        index
            .insert_batch((0..GATE_CHUNK).map(|i| {
                let n = base + i;
                (str_hash_56(&format!("regate-churn-{n}")), 1_000_000 + n)
            }))
            .expect("insert_batch");
    }

    fn idx_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".idx"))
            .collect();
        names.sort();
        names
    }

    /// ROOT CAUSE A gate (moved in-crate so the fault seam stays `cfg(test)`):
    /// a run is retired + unlinked in exactly the window between the reader's
    /// lock-free snapshot and its `File::open`.
    ///
    /// The window is a few instructions wide — an 8-reader / 700k-lookup thread
    /// race did not reproduce it once, and a probabilistic RED-on-revert is not
    /// a gate — so the fault is injected deterministically.
    ///
    /// RED-on-revert: drop the `RunRetired` retry arm in `for_each_candidate`
    /// (let the `File::open` ENOENT escape) → "lookup FAILED under a retired run".
    #[test]
    fn gate_reader_survives_run_retired_under_it() {
        let dir = tempfile::tempdir().unwrap();
        let index =
            Arc::new(OwnerForwardIndex::create(dir.path(), OWNER_INDEX_DISK_CAP_BYTES).unwrap());
        let key = "regate-resident-key";
        let expected = 9_000_001_u64;
        index.insert_batch([(str_hash_56(key), expected)]).unwrap();
        for chunk in 0..4 {
            churn(&index, chunk);
        }

        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let hook_index = Arc::clone(&index);
            let fired = Arc::clone(&fired);
            index.set_pre_open_hook(Arc::new(move || {
                if fired.swap(true, Ordering::AcqRel) {
                    return; // once only, else we churn forever and exhaust the retry bound
                }
                for chunk in 100..104 {
                    churn(&hook_index, chunk);
                }
            }));
        }

        let found = index
            .for_each_candidate(str_hash_56(key), |candidate| Ok(candidate == expected))
            .unwrap_or_else(|error| panic!("lookup FAILED under a retired run: {error}"));
        assert_eq!(
            found,
            Some(expected),
            "a durably published key vanished when its run was retired mid-lookup \
             (fail-open miss: intern would allocate a SECOND id for the same string)"
        );

        assert!(fired.load(Ordering::Acquire), "fault never fired — vacuous");
        assert!(
            index.retired_run_retries() > 0,
            "VACUOUS GATE: no reader ever hit a retired run"
        );
    }

    /// RE-GATE r2 (2) — a transient unlink failure must not ABANDON the rest of
    /// the retired list.
    ///
    /// `insert_chunk` used to `return Err` on the first non-`NotFound`
    /// `remove_file` error, skipping every remaining unlink and the `sync_dir`.
    /// One EIO/EPERM blip therefore leaked all the other retired runs' bytes
    /// against `enforce_budget` for the process lifetime.
    ///
    /// RED-on-revert: restore `Err(error) => return Err(error.into())` in the
    /// retire loop — the survivors are abandoned and this gate fails.
    #[test]
    fn gate_transient_unlink_failure_still_reclaims_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), OWNER_INDEX_DISK_CAP_BYTES).unwrap();

        // Build up levels so the next insert retires MORE THAN ONE run.
        for chunk in 0..3 {
            churn(&index, chunk);
        }
        let before = idx_files(dir.path());

        // Fail exactly the FIRST unlink of the next merge.
        index.fail_next_unlinks_for_gate(1);
        churn(&index, 3);

        let after = idx_files(dir.path());
        let leaked: Vec<_> = after.iter().filter(|n| before.contains(n)).collect();

        // Exactly ONE retired file survives (the injected failure). Everything
        // else the merge retired was still reclaimed.
        assert_eq!(
            leaked.len(),
            1,
            "a transient unlink failure abandoned the remaining retired runs \
             (leaked {leaked:?}); before={before:?} after={after:?}"
        );

        // And the index is still fully functional: the publish succeeded and the
        // straggler is inert (it is unpublished; `sweep_orphan_runs` reclaims it).
        let probe = index
            .for_each_candidate(str_hash_56("regate-churn-0"), |c| Ok(c == 1_000_000))
            .expect("index still readable after a failed unlink");
        assert_eq!(probe, Some(1_000_000));

        // The straggler really is reclaimed on reopen, so the leak is bounded by
        // the process lifetime, not permanent.
        drop(index);
        let reopened = OwnerForwardIndex::open(dir.path(), OWNER_INDEX_DISK_CAP_BYTES).unwrap();
        let swept = idx_files(dir.path());
        assert!(
            !swept.iter().any(|n| leaked.contains(&n)),
            "sweep_orphan_runs did not reclaim the leaked run on reopen: {swept:?}"
        );
        drop(reopened);
    }

    #[test]
    fn sorted_run_merge_and_collision_walk() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 1024 * 1024).unwrap();
        for batch in [
            vec![(7, 2), (3, 1)],
            vec![(7, 4)],
            vec![(5, 3)],
            vec![(7, 6)],
        ] {
            index.insert_batch(batch).unwrap();
        }
        let mut visited = Vec::new();
        let found = index
            .for_each_candidate(7, |id| {
                visited.push(id);
                Ok(id == 4)
            })
            .unwrap();
        assert_eq!(found, Some(4));
        assert!(visited.contains(&2));
        assert!(visited.contains(&4));
        assert!(index.resident_run_descriptors() <= 3);
        drop(index);
        let reopened = OwnerForwardIndex::open(dir.path(), 1024 * 1024).unwrap();
        assert_eq!(reopened.candidate_count(), 5);
    }

    #[test]
    fn lookup_continues_across_overlapping_immutable_runs() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 1024 * 1024).unwrap();

        // Fold two chunks into the older level-1 run. Its min/max fence spans
        // hash 50, but it deliberately has no exact candidate for that hash.
        index.insert_batch([(1, 1), (100, 2)]).unwrap();
        index.insert_batch([(2, 3)]).unwrap();
        // Leave the wanted candidate in a newer level-0 run. Lookup visits the
        // older, higher-level run first, so a per-run miss must continue rather
        // than terminate the complete run-set search.
        index.insert_batch([(50, 4)]).unwrap();

        assert_eq!(
            index.for_each_candidate(50, |id| Ok(id == 4)).unwrap(),
            Some(4)
        );
        assert_eq!(index.candidate_count(), 4);
    }

    #[test]
    fn collision_walk_crosses_page_backed_btree_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 8 * 1024 * 1024).unwrap();
        index
            .insert_batch((1..=LEAF_CAPACITY as u64 + 37).map(|id| (11, id)))
            .unwrap();
        let mut visited = 0_u64;
        let wanted = LEAF_CAPACITY as u64 + 37;
        assert_eq!(
            index
                .for_each_candidate(11, |id| {
                    visited += 1;
                    Ok(id == wanted)
                })
                .unwrap(),
            Some(wanted)
        );
        assert_eq!(visited, wanted);

        let runs = index.runs.read();
        let mut file = File::open(dir.path().join(&runs[0].file_name)).unwrap();
        let header = read_run_header(&mut file).unwrap();
        assert!(header.leaf_count >= 2);
        assert!(header.root_page > header.leaf_count);
        let root = read_index_page(&mut file, header.root_page, header.page_count).unwrap();
        assert_eq!(
            read_page_header(&root).unwrap().page_type,
            PageType::IndexInternal.as_byte()
        );
    }

    #[test]
    fn arbitrarily_large_input_is_chunked_before_sorting() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 16 * 1024 * 1024).unwrap();
        let count = OWNER_INDEX_BATCH_ENTRIES as u64 + 5;
        index
            .insert_batch((1..=count).rev().map(|id| (id, id)))
            .unwrap();
        assert_eq!(index.candidate_count(), count);
        assert!(index.resident_run_descriptors() <= 2);
    }

    #[test]
    fn page_crc_corruption_is_typed() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 1024 * 1024).unwrap();
        index.insert_batch([(1, 1), (2, 2)]).unwrap();
        let file_name = index.runs.read()[0].file_name.clone();
        drop(index);
        let path = dir.path().join(file_name);
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(RUN_HEADER_BYTES + LEAF_ENTRY_OFFSET as u64))
            .unwrap();
        file.write_all(&[0xff]).unwrap();
        file.sync_all().unwrap();
        let error = OwnerForwardIndex::open(dir.path(), 1024 * 1024).unwrap_err();
        assert!(matches!(error, OwnerIndexError::Corrupt(_)));
    }

    #[test]
    fn disk_budget_is_hard() {
        let dir = tempfile::tempdir().unwrap();
        let index = OwnerForwardIndex::create(dir.path(), 64).unwrap();
        let error = index.insert_batch([(1, 1), (2, 2)]).unwrap_err();
        assert!(matches!(error, OwnerIndexError::DiskBudgetExceeded { .. }));
    }

    /// D-5 / M5-D3: a bulk-built index whose published bytes exceed the
    /// incremental default must OPEN (via `open_bulk`) and accept
    /// churn-bounded growth — the configured cap governs growth ABOVE the
    /// published bytes, never the legitimacy of the build-time
    /// (census-derived) size.
    #[test]
    fn open_bulk_treats_cap_as_growth_budget_above_published_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let published = {
            let index = OwnerForwardIndex::create(dir.path(), u64::MAX).unwrap();
            index.insert_batch((0_u64..20_000).map(|i| (i, i))).unwrap();
            directory_bytes(dir.path()).unwrap()
        };
        // Reopen with an incremental cap SMALLER than the published bytes:
        // must open, and must accept a small incremental insert.
        let cap = published / 2;
        let reopened = OwnerForwardIndex::open_bulk(dir.path(), cap).unwrap();
        assert!(
            reopened.disk_cap_bytes() > published,
            "effective cap must sit above the published bytes"
        );
        reopened.insert_batch([(1_000_000, 7)]).unwrap();
    }

    /// FIX 2 (M5-D3 / #1518 skeptic review): the SERVE path (`open`, used by
    /// `OwnerRowRegistry::open_logical` on every boot/recovery) must keep
    /// its PRE-M5-D3 absolute-ceiling semantics — the cap passed in is the
    /// hard write-path ceiling, NOT `existing × 2.2 + cap`. RED-on-revert:
    /// if `open` regresses to the bulk growth-above-published ratchet, the
    /// `disk_cap_bytes()` assertion below fails (it would report an
    /// inflated effective cap instead of the exact incremental cap).
    #[test]
    fn open_serve_path_ceiling_is_absolute_and_unaffected_by_bulk_growth_cap() {
        let dir = tempfile::tempdir().unwrap();
        let published = {
            let index = OwnerForwardIndex::create(dir.path(), u64::MAX).unwrap();
            index.insert_batch((0_u64..20_000).map(|i| (i, i))).unwrap();
            directory_bytes(dir.path()).unwrap()
        };
        let cap = published / 2;
        let reopened = OwnerForwardIndex::open(dir.path(), cap).unwrap();
        assert_eq!(
            reopened.disk_cap_bytes(),
            cap,
            "serve-path open must report the EXACT incremental cap, not a growth-ratcheted one"
        );
        // A small incremental insert whose write-time check adds against an
        // already-over-cap baseline must fail closed exactly as pre-M5-D3.
        let error = reopened.insert_batch([(1_000_000, 7)]).unwrap_err();
        assert!(matches!(error, OwnerIndexError::DiskBudgetExceeded { .. }));
    }

    #[test]
    fn strhash_canary_matches_secondary_index_contract() {
        assert_eq!(str_hash_56("arcgraph_canary"), 29_198_083_841_200_401);
    }
}
