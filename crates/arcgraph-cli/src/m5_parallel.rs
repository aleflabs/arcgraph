//! M5-D3 — parallel decomposition of the bulk-load pipeline.
//!
//! Design authority: `docs/design/M5D-REDESIGN-AMENDMENT.md` §4 (the four
//! mechanisms) + §5 (census-derived budgets). The four mechanisms:
//!
//! 1. **Input partitioning** — native framed (newline-delimited) byte-range
//!    splits resynced to frame boundaries (`partition_input`); each worker
//!    parses its partition with no shared state.
//! 2. **Per-worker run generation** — every worker fills a private sort
//!    buffer and spills sorted runs into its own namespaced scratch subdir
//!    (`<stage>/w<k>/run-<n>`), each with a sparse key/offset index sidecar
//!    and min/max fences, fsynced under a per-stage run manifest.
//! 3. **Range-partitioned merge** (load-bearing) — sample the run fences and
//!    sparse indexes, choose `W−1` splitters → `W` half-open, disjoint,
//!    key-space-covering ranges; worker `i` k-way-merges ALL runs restricted
//!    to range `i` (`RangeMerge`). The concatenation of the `W` output
//!    segments is the exact serial sort order.
//!
//!    *Dup-detection-under-parallel-runs proof (INV-M5.19):* duplicate
//!    detection is sort-adjacency of equal keys. (i) Range assignment is a
//!    function of the KEY alone and ranges are disjoint half-open intervals
//!    covering the key space, so ALL occurrences of a key land in exactly
//!    one range. (ii) Within a range one worker performs a total merge, so
//!    equal keys are adjacent exactly as in the serial merge. (iii) A
//!    splitter equal to a duplicated key cannot split its occurrences
//!    (`k < s_{i+1}` puts every `k` in `[s_i, s_{i+1})`). Hence the
//!    sort-adjacent hard-error check runs verbatim per merge worker and
//!    misses nothing.
//! 4. **Two-phase dense-id assignment** (INV-M5.24) — phase 1 counts unique
//!    keys per range (no ids); a sequential prefix-sum over the `W` counts
//!    gives each range its id base; phase 2 re-streams each merged segment
//!    stamping `base_i + ordinal`. Dense ids are therefore BYTE-IDENTICAL
//!    to the serial assignment for ANY worker count: same global sort
//!    order, same numbering. Worker scheduling affects only timing, never
//!    content — every stage output is a pure function of the input
//!    multiset, and the final (serial) materialization consumes streams
//!    whose concatenated content is worker-count-invariant.
//!
//! **Stage-level restartability:** every stage publishes a durable JSON
//! manifest (`scratch/manifests/<stage>.json`, tmp+rename+dir-fsync) naming
//! its output files with length + head/tail CRC. A rerun after a mid-build
//! crash resumes from the last durable stage instead of from scratch.
//! Consumed intermediates are garbage-collected once their LAST consumer's
//! manifest is durable; resume validation treats a group as satisfied when
//! its consumer stage is itself valid (see `StagePlan`).
//!
//! Budget (performance-budget discipline), 100M+500M rung at W=16 on ≥2.5 GB/s scratch
//! (amendment §4.3): ~820 GB of scratch traffic ≈ 330 s of I/O with parse
//! (~55 s across 16 workers) and merge CPU under the I/O curve → ≈270–300K
//! nodes/s. Resident set: `W × sort_budget` (4 GiB at W=16 × 256 MiB) plus
//! merge fan-in buffers (≤ 64 readers × 1 MiB per merge worker) plus the
//! serial materializer's O(1) pages — inside the §M5.1 8 GiB sort plateau
//! and the 40 GiB rung cap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use arcgraph_core::TenantId;
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::m4_migration::{
    FRESH_TEL_ENTRIES_PER_PAGE, FreshNode, FreshRel, FreshTelDirection, FreshV6Builder,
    LoaderMigrationFrontier,
};
use arcgraph_storage::owner_budget::OwnerBulkBudgets;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::m5_load::{
    LoadFormat, LoadLimits, LoadRecord, LoadRecordSource, LoadReport, MAX_MERGE_FAN_IN,
    MAX_NATIVE_RECORD_BYTES, NativeRecordSource, RssSample, canonical_sort_key, create_new,
    decode_binding, decode_canonical_record, decode_endpoint_request, decode_node_artifact,
    decode_resolved_rel, decode_tel_run, encode_canonical_record, encode_endpoint_request,
    encode_tel_run, materialized_bag, plan_owner_budgets, project_disk_or_refuse, put_bytes,
    read_optional_u32, read_required_u32, ship_empty_tel, sync_writer, take_u64, tel_run_key,
    write_framed,
};

/// One sparse-index sample every this many run items.
const RUN_INDEX_SAMPLE_EVERY: u64 = 1024;
/// Merge-side read buffer per open run (fan-in ≤ [`MAX_MERGE_FAN_IN`]).
const MERGE_READ_BUFFER: usize = 1 << 20;
/// Ceiling for writer-side buffers on runs and segments.
const WRITE_BUFFER: usize = 4 << 20;

/// Writer buffers scale with the sort budget: full-size at production
/// budgets, small at CI fixture budgets so `workers x writers` never
/// dominates the resident envelope (INV-M5.15 measures this).
fn adaptive_write_buffer(sort_budget: usize) -> usize {
    sort_budget.clamp(64 * 1024, WRITE_BUFFER)
}
/// Fingerprint prefix bytes hashed from the input file.
const FINGERPRINT_HEAD_BYTES: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Fault-injection seams (cfg-gated + bounded per the standing test-hook rule)
// ---------------------------------------------------------------------------

/// RED-on-revert seam for INV-M5.24: assign per-range id bases by (modeled,
/// deterministic) worker ARRIVAL order — i.e. drop the phase-1 prefix-sum
/// ordering — instead of global-sort range order. W>1 then diverges from
/// W=1 byte-for-byte. Deterministic by construction: a timing-hopeful RED
/// is not a gate (memory: determinism-oracle concurrency tests).
pub(crate) fn arrival_order_ids() -> bool {
    seam("ARCGRAPH_M5_ARRIVAL_ORDER_IDS")
}

/// RED-on-revert seam for INV-M5.19: neuter the cross-run half of duplicate
/// detection — adjacency is only recognized between items of the SAME source
/// run (i.e. "assign [dup] ranges by RUN, not by KEY"). A planted duplicate
/// whose occurrences come from different workers' runs is then missed.
pub(crate) fn range_by_run() -> bool {
    seam("ARCGRAPH_M5_RANGE_BY_RUN")
}

/// RED-on-revert seam for INV-M5.15: never spill on the sort budget — the
/// run buffer collects the whole input in RAM. The continuous-RSS gate's
/// cap assertion MUST go red under this control.
pub(crate) fn collect_all() -> bool {
    seam("ARCGRAPH_M5_COLLECT_ALL")
}

/// Deterministic adversarial scheduler: stage worker `k` of `W` sleeps
/// `(W-1-k) × ms` before starting, forcing reverse completion order. With
/// the real two-phase assignment this changes NOTHING (the byte-identical
/// gate runs under it — that is the deterministic barrier); combined with
/// [`arrival_order_ids`] it makes the arrival-order defect deterministic.
fn stagger(worker: usize, workers: usize) {
    #[cfg(feature = "fault-injection")]
    if let Some(ms) = std::env::var("ARCGRAPH_M5_WORKER_STAGGER_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
    {
        let slots = workers.saturating_sub(1).saturating_sub(worker) as u64;
        std::thread::sleep(Duration::from_millis(slots.saturating_mul(ms).min(2_000)));
    }
    #[cfg(not(feature = "fault-injection"))]
    let _ = (worker, workers);
}

/// Restartability seam: crash (typed error) right after the named stage's
/// manifest became durable.
fn crash_after_stage(name: &str) -> Result<()> {
    #[cfg(feature = "fault-injection")]
    if std::env::var("ARCGRAPH_M5_CRASH_AFTER_STAGE").as_deref() == Ok(name) {
        bail!("injected crash after stage {name}");
    }
    let _ = name;
    Ok(())
}

fn seam(name: &str) -> bool {
    #[cfg(feature = "fault-injection")]
    {
        std::env::var_os(name).is_some()
    }
    #[cfg(not(feature = "fault-injection"))]
    {
        let _ = name;
        false
    }
}

// ---------------------------------------------------------------------------
// Worker topology
// ---------------------------------------------------------------------------

/// Production worker default: `min(physical_cores, 32)` (amendment §4.2).
#[must_use]
pub fn default_workers() -> usize {
    physical_cores().clamp(1, 32)
}

fn physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Unique (physical id, core id) pairs; SMT siblings share one.
        if let Ok(info) = fs::read_to_string("/proc/cpuinfo") {
            let mut cores = std::collections::BTreeSet::new();
            let (mut package, mut core) = (None::<u64>, None::<u64>);
            for line in info.lines().chain(Some("")) {
                if line.is_empty() {
                    if let (Some(p), Some(c)) = (package.take(), core.take()) {
                        cores.insert((p, c));
                    }
                    continue;
                }
                let mut parts = line.splitn(2, ':');
                let key = parts.next().unwrap_or("").trim();
                let value = parts.next().unwrap_or("").trim();
                match key {
                    "physical id" => package = value.parse().ok(),
                    "core id" => core = value.parse().ok(),
                    _ => {}
                }
            }
            if !cores.is_empty() {
                return cores.len();
            }
        }
    }
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// Best-effort NOFILE headroom for `workers × fan-in` concurrent run
/// readers. Failure is non-fatal: the merge would then surface a plain
/// open error.
fn raise_fd_limit(minimum: u64) {
    #[cfg(unix)]
    {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit/setrlimit write/read only the rlimit struct we
        // own on the stack; no other memory is touched.
        unsafe {
            if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0
                && limit.rlim_cur < minimum
                && limit.rlim_max >= minimum
            {
                let raised = libc::rlimit {
                    rlim_cur: minimum,
                    rlim_max: limit.rlim_max,
                };
                let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &raised);
            }
        }
    }
    #[cfg(not(unix))]
    let _ = minimum;
}

/// Run `tasks` closures over at most `workers` threads, preserving task
/// order in the returned vector. Content determinism never depends on
/// scheduling: each task's output is a pure function of its inputs.
fn run_tasks<T, F>(workers: usize, tasks: Vec<F>) -> Result<Vec<T>>
where
    T: Send,
    F: FnOnce() -> Result<T> + Send,
{
    let task_count = tasks.len();
    let lanes = workers.clamp(1, task_count.max(1));
    let mut slots: Vec<Option<Result<T>>> = Vec::with_capacity(task_count);
    slots.resize_with(task_count, || None);
    let queue = parking_lot::Mutex::new(tasks.into_iter().enumerate().collect::<Vec<_>>());
    let results = parking_lot::Mutex::new(&mut slots);
    std::thread::scope(|scope| {
        for _ in 0..lanes {
            scope.spawn(|| {
                loop {
                    let Some((index, task)) = ({
                        let mut queue = queue.lock();
                        queue.pop()
                    }) else {
                        return;
                    };
                    let outcome = task();
                    results.lock()[index] = Some(outcome);
                }
            });
        }
    });
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.with_context(|| format!("stage task {index} produced no result"))?
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Continuous RSS sampling (salvaged from PR #1504 per amendment §9)
// ---------------------------------------------------------------------------

struct SamplerShared {
    samples: parking_lot::Mutex<Vec<RssSample>>,
    stage: parking_lot::Mutex<&'static str>,
    stop: AtomicBool,
    exceeded: AtomicBool,
    cap_bytes: u64,
    started: Instant,
}

/// Time-based continuous resident-set sampler. One dedicated thread samples
/// every `sample_every_ms`; the vector grows with wall time, never row
/// count. The cap is enforced continuously: the first over-cap sample trips
/// [`Self::finish`] into a typed error (INV-M5.15).
pub(crate) struct RssSampler {
    shared: Arc<SamplerShared>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    pub(crate) fn start(cap_bytes: u64, sample_every_ms: u64) -> Self {
        let shared = Arc::new(SamplerShared {
            samples: parking_lot::Mutex::new(Vec::new()),
            stage: parking_lot::Mutex::new("start"),
            stop: AtomicBool::new(false),
            exceeded: AtomicBool::new(false),
            cap_bytes,
            started: Instant::now(),
        });
        let worker = Arc::clone(&shared);
        let every = Duration::from_millis(sample_every_ms.max(1));
        let handle = std::thread::Builder::new()
            .name("m5-rss-sampler".to_owned())
            .spawn(move || {
                loop {
                    let rss = rss_bytes().unwrap_or(0);
                    let sample = RssSample {
                        at_ms: u64::try_from(worker.started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                        rss_bytes: rss,
                        stage: *worker.stage.lock(),
                    };
                    if rss > worker.cap_bytes {
                        worker.exceeded.store(true, Ordering::SeqCst);
                    }
                    worker.samples.lock().push(sample);
                    if worker.stop.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::park_timeout(every);
                }
            })
            .expect("spawn RSS sampler thread");
        Self {
            shared,
            handle: Some(handle),
        }
    }

    pub(crate) fn set_stage(&self, stage: &'static str) {
        *self.shared.stage.lock() = stage;
    }

    /// True once any sample exceeded the cap; stages poll this at run-flush
    /// grain so an over-cap build aborts instead of running to completion.
    pub(crate) fn exceeded(&self) -> bool {
        self.shared.exceeded.load(Ordering::SeqCst)
    }

    pub(crate) fn finish(mut self) -> Result<Vec<RssSample>> {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
        let samples = std::mem::take(&mut *self.shared.samples.lock());
        if self.shared.exceeded.load(Ordering::SeqCst) {
            let worst = samples.iter().map(|sample| sample.rss_bytes).max();
            bail!(
                "continuous RSS cap exceeded: max sample {:?} > cap {} bytes",
                worst,
                self.shared.cap_bytes
            );
        }
        Ok(samples)
    }
}

impl Drop for RssSampler {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

/// Current process resident bytes — the INV-M5.15 gates use it to place
/// their cap above the harness baseline.
pub fn current_rss_bytes() -> Result<u64> {
    rss_bytes()
}

#[cfg(target_os = "linux")]
pub(crate) fn rss_bytes() -> Result<u64> {
    let statm = fs::read_to_string("/proc/self/statm").context("read /proc/self/statm")?;
    let resident_pages = statm
        .split_ascii_whitespace()
        .nth(1)
        .context("/proc/self/statm has no resident field")?
        .parse::<u64>()
        .context("parse resident pages")?;
    // SAFETY: sysconf is thread-safe and takes no pointer arguments.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(page_size > 0, "sysconf(_SC_PAGESIZE) failed");
    resident_pages
        .checked_mul(page_size as u64)
        .context("RSS byte count overflow")
}

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) fn rss_bytes() -> Result<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    ensure!(
        // SAFETY: usage points to writable rusage storage owned by this frame.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
        "getrusage failed"
    );
    // macOS reports peak bytes; the CI/rung lane is the Linux branch above.
    // SAFETY: getrusage returned 0, so the struct is fully initialized.
    Ok(unsafe { usage.assume_init() }.ru_maxrss as u64)
}

#[cfg(not(unix))]
pub(crate) fn rss_bytes() -> Result<u64> {
    Ok(0)
}

// ---------------------------------------------------------------------------
// Manifest vocabulary
// ---------------------------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0xf)] as char);
    }
    out
}

fn hex_decode(value: &str) -> Result<Vec<u8>> {
    ensure!(value.len() % 2 == 0, "hex length must be even");
    let nibble = |byte: u8| -> Result<u8> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => bail!("not lowercase hexadecimal"),
        }
    };
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

/// Cheap durable-file fingerprint: length + CRC32C of the first and last
/// 4 KiB. Files are fsynced BEFORE their manifest is published (the
/// manifest is the durability barrier); the fingerprint guards truncation
/// and cross-run mixups, not adversarial tampering.
fn file_fingerprint(path: &Path) -> Result<(u64, u32)> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let mut head = vec![0_u8; usize::try_from(len.min(4096)).expect("bounded")];
    file.read_exact(&mut head)?;
    let mut crc = crc32c::crc32c(&head);
    if len > 4096 {
        file.seek(SeekFrom::End(-4096))?;
        let mut tail = [0_u8; 4096];
        file.read_exact(&mut tail)?;
        crc = crc32c::crc32c_append(crc, &tail);
    }
    Ok((len, crc))
}

/// One durable file named by a stage manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FileEntry {
    /// Path relative to the pipeline scratch root.
    pub rel_path: String,
    pub len: u64,
    pub crc: u32,
}

impl FileEntry {
    fn record(scratch: &Path, rel_path: String) -> Result<Self> {
        let (len, crc) = file_fingerprint(&scratch.join(&rel_path))?;
        Ok(Self { rel_path, len, crc })
    }

    fn validate(&self, scratch: &Path) -> bool {
        file_fingerprint(&scratch.join(&self.rel_path))
            .is_ok_and(|(len, crc)| len == self.len && crc == self.crc)
    }
}

/// One sorted run: the framed `(key, payload)` file plus its sparse
/// `(key, offset)` index sidecar and key fences.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RunEntry {
    pub file: FileEntry,
    pub index: FileEntry,
    pub count: u64,
    pub min_key_hex: String,
    pub max_key_hex: String,
}

/// One framed output segment (payload stream) with optional key fences.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SegEntry {
    pub file: FileEntry,
    pub count: u64,
    pub min_key_hex: Option<String>,
    pub max_key_hex: Option<String>,
}

/// Full-input identity (FIX M5-D3/#1518 skeptic review): resume MUST bind
/// to the exact input bytes, not merely `(len, head-1MiB-CRC)`. A rerun
/// whose input was swapped between a crash and the rerun — SAME length,
/// SAME first `FINGERPRINT_HEAD_BYTES` — but a byte changed further into
/// the file (e.g. a planted duplicate in the tail) would previously pass
/// the head-only check and resume stale s1–s3 manifests built from the OLD
/// input, so the phase-1 duplicate check never sees the new bytes and the
/// loader reports success while serving old content. `full_crc` is a
/// CRC32C over every byte of the file, computed here as ONE dedicated
/// sequential pass (negligible next to the parse+sort+merge I/O the
/// pipeline already performs on the same file — s1 itself streams every
/// byte too, just split across worker partitions rather than one linear
/// pass) and is REQUIRED to match at resume (see `run_pipeline`); on any
/// mismatch the resume plan is discarded and the pipeline rebuilds from
/// scratch. Never serve stages computed from bytes that are not exactly
/// the current input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct InputFingerprint {
    pub len: u64,
    pub head_crc: u32,
    /// CRC32C over the ENTIRE input stream — the load-bearing identity
    /// check. `head_crc` is retained only as a fast rejection path.
    pub full_crc: u32,
}

impl InputFingerprint {
    fn of(input: &Path) -> Result<Self> {
        let mut file = File::open(input).with_context(|| format!("open {}", input.display()))?;
        let len = file.metadata()?.len();
        let mut head =
            vec![0_u8; usize::try_from(len.min(FINGERPRINT_HEAD_BYTES as u64)).expect("bounded")];
        file.read_exact(&mut head)?;
        let head_crc = crc32c::crc32c(&head);
        // Full-stream CRC: continue reading from where the head ended
        // (avoids re-reading the head bytes) sequentially to EOF.
        let mut full_crc = head_crc;
        let mut buf = vec![0_u8; 1 << 20];
        loop {
            let read = file.read(&mut buf)?;
            if read == 0 {
                break;
            }
            full_crc = crc32c::crc32c_append(full_crc, &buf[..read]);
        }
        Ok(Self {
            len,
            head_crc,
            full_crc,
        })
    }
}

/// Pass-1 census (amendment §5): exact entry counts + exact
/// `Σ|external_id|`, known before any substrate write.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadCensus {
    pub records: u64,
    pub nodes: u64,
    pub relationships: u64,
    pub node_external_id_bytes: u64,
    pub rel_external_id_bytes: u64,
    /// Total canonical payload bytes (projection input).
    pub payload_bytes: u64,
}

impl LoadCensus {
    fn absorb(&mut self, other: &Self) {
        self.records += other.records;
        self.nodes += other.nodes;
        self.relationships += other.relationships;
        self.node_external_id_bytes += other.node_external_id_bytes;
        self.rel_external_id_bytes += other.rel_external_id_bytes;
        self.payload_bytes += other.payload_bytes;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S1Manifest {
    fingerprint: InputFingerprint,
    census: LoadCensus,
    runs: Vec<RunEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S2Manifest {
    splitters_hex: Vec<String>,
    /// `(node_count, rel_count)` per range, range order.
    range_counts: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S3Range {
    bindings: SegEntry,
    nodes: SegEntry,
    rels: SegEntry,
    node_base: u64,
    rel_base: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S3Manifest {
    ranges: Vec<S3Range>,
    endpoint_runs: Vec<RunEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S4Manifest {
    resolved_runs: Vec<RunEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S5Manifest {
    rel_segs: Vec<SegEntry>,
    out_runs: Vec<RunEntry>,
    in_runs: Vec<RunEntry>,
    relationships: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct S6Manifest {
    out_segs: Vec<SegEntry>,
    in_segs: Vec<SegEntry>,
    out_entries: u64,
    in_entries: u64,
    /// EXACT STORE_TEL page (block) counts the materializer will write:
    /// one `PageType::Tel` page per (owner, type) chain block, chunked at
    /// [`FRESH_TEL_ENTRIES_PER_PAGE`] — counted during the s6 merges (the
    /// first global adjacency view), within `W−1` pages of exact (a group
    /// straddling a range boundary is counted once per side). Feeds the
    /// post-s6 disk projection: the plan-time census cannot see distinct
    /// (owner, type) group counts, and the page-per-block layout costs
    /// ≥ 8 KiB per group per direction — the class that took the 100M
    /// rung to ~3 TB of STORE_TEL (see the PR STOP-report).
    #[serde(default)]
    out_pages: u64,
    #[serde(default)]
    in_pages: u64,
}

const STAGE_S1: &str = "s1-canonical-runs";
const STAGE_S2: &str = "s2-phase1-counts";
const STAGE_S3: &str = "s3-phase2-segments";
const STAGE_S4: &str = "s4-resolved-runs";
const STAGE_S5: &str = "s5-rel-tel-runs";
const STAGE_S6: &str = "s6-tel-segments";

fn manifest_path(scratch: &Path, stage: &str) -> PathBuf {
    scratch.join("manifests").join(format!("{stage}.json"))
}

fn write_stage_manifest<M: Serialize>(scratch: &Path, stage: &str, manifest: &M) -> Result<()> {
    let dir = scratch.join("manifests");
    fs::create_dir_all(&dir)?;
    let path = manifest_path(scratch, stage);
    let tmp = dir.join(format!("{stage}.json.tmp"));
    let bytes = serde_json::to_vec_pretty(manifest).context("encode stage manifest")?;
    let mut file = File::create(&tmp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, &path)?;
    arcgraph_storage::wal::fsync_dir(&dir).context("sync manifests dir")?;
    crash_after_stage(stage)
}

fn read_stage_manifest<M: DeserializeOwned>(scratch: &Path, stage: &str) -> Option<M> {
    let bytes = fs::read(manifest_path(scratch, stage)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

// ---------------------------------------------------------------------------
// Run generation (mechanism 2)
// ---------------------------------------------------------------------------

struct SortItem {
    key: Vec<u8>,
    payload: Vec<u8>,
}

impl SortItem {
    fn resident_bytes(&self) -> usize {
        self.key.capacity() + self.payload.capacity() + std::mem::size_of::<Self>()
    }
}

/// Per-worker sorted-run generator: private sort buffer, namespaced scratch
/// subdir, sparse index sidecars, fences. `finish` compacts the worker's
/// own runs so that `Σ_workers(runs)` stays within one merge fan-in.
pub(crate) struct RunSetWriter {
    scratch_root: PathBuf,
    dir_rel: String,
    budget: usize,
    /// Adaptive writer buffer: full-size at rung budgets, budget-sized at
    /// CI fixture budgets so `workers x segments` writers stay cheap.
    write_buffer: usize,
    resident: usize,
    buffer: Vec<SortItem>,
    runs: Vec<RunEntry>,
    next_run: u64,
    max_final_runs: usize,
}

impl RunSetWriter {
    /// `dir_rel` is the worker-owned subdir relative to the scratch root
    /// (e.g. `s1/w3`); `max_final_runs` bounds this writer's contribution
    /// to a downstream merge's fan-in.
    pub(crate) fn new(
        scratch_root: &Path,
        dir_rel: &str,
        budget: usize,
        max_final_runs: usize,
    ) -> Result<Self> {
        let dir = scratch_root.join(dir_rel);
        fs::create_dir_all(&dir).with_context(|| format!("create run dir {}", dir.display()))?;
        Ok(Self {
            scratch_root: scratch_root.to_path_buf(),
            dir_rel: dir_rel.to_owned(),
            budget: budget.max(64 * 1024),
            write_buffer: adaptive_write_buffer(budget),
            resident: 0,
            buffer: Vec::new(),
            runs: Vec::new(),
            next_run: 0,
            max_final_runs: max_final_runs.max(1),
        })
    }

    pub(crate) fn push(&mut self, key: Vec<u8>, payload: Vec<u8>) -> Result<()> {
        let item = SortItem { key, payload };
        let item_bytes = item.resident_bytes();
        if !collect_all()
            && !self.buffer.is_empty()
            && self.resident.saturating_add(item_bytes) > self.budget
        {
            self.flush_run()?;
        }
        self.resident = self.resident.saturating_add(item_bytes);
        self.buffer.push(item);
        if !collect_all() && self.resident > self.budget {
            // One capped record can exceed a deliberately tiny test budget;
            // spill immediately instead of retaining it.
            self.flush_run()?;
        }
        Ok(())
    }

    fn flush_run(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.buffer.sort_unstable_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then_with(|| left.payload.cmp(&right.payload))
        });
        let rel_path = format!("{}/run-{:08}.bin", self.dir_rel, self.next_run);
        self.next_run += 1;
        let run = write_run_files(
            &self.scratch_root,
            &rel_path,
            self.write_buffer,
            self.buffer
                .drain(..)
                .map(|item| Ok((item.key, item.payload))),
        )?
        .context("non-empty buffer must produce a run")?;
        self.resident = 0;
        self.runs.push(run);
        Ok(())
    }

    /// Spill the tail buffer and compact this worker's runs down to
    /// `max_final_runs` via intra-worker merges, returning the durable set.
    pub(crate) fn finish(mut self) -> Result<Vec<RunEntry>> {
        self.flush_run()?;
        while self.runs.len() > self.max_final_runs {
            let group_size = self
                .runs
                .len()
                .div_ceil(self.max_final_runs)
                .clamp(2, MAX_MERGE_FAN_IN);
            let old_runs = std::mem::take(&mut self.runs);
            for group in old_runs.chunks(group_size) {
                if group.len() == 1 {
                    self.runs.push(group[0].clone());
                    continue;
                }
                let rel_path = format!("{}/run-{:08}.bin", self.dir_rel, self.next_run);
                self.next_run += 1;
                let mut merge =
                    RangeMerge::open(&self.scratch_root, group, None, None, self.budget)?;
                let merged = write_run_files(
                    &self.scratch_root,
                    &rel_path,
                    self.write_buffer,
                    std::iter::from_fn(|| merge.next_item().transpose()),
                )?
                .context("merged group must be non-empty")?;
                for run in group {
                    remove_run_files(&self.scratch_root, run)?;
                }
                self.runs.push(merged);
            }
        }
        Ok(self.runs)
    }
}

/// Stream sorted `(key, payload)` items into a run file + index sidecar.
/// Returns `None` for an empty stream.
fn write_run_files(
    scratch_root: &Path,
    rel_path: &str,
    write_buffer: usize,
    items: impl Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>,
) -> Result<Option<RunEntry>> {
    let path = scratch_root.join(rel_path);
    let index_rel = format!("{rel_path}.idx");
    let index_path = scratch_root.join(&index_rel);
    let mut writer = BufWriter::with_capacity(write_buffer, create_new(&path)?);
    let mut index = BufWriter::new(create_new(&index_path)?);
    let mut count = 0_u64;
    let mut offset = 0_u64;
    let mut min_key: Option<Vec<u8>> = None;
    let mut max_key: Vec<u8> = Vec::new();
    for item in items {
        let (key, payload) = item?;
        if count % RUN_INDEX_SAMPLE_EVERY == 0 {
            write_index_entry(&mut index, &key, offset)?;
        }
        let item_bytes = 8 + key.len() + payload.len();
        write_run_item(&mut writer, &key, &payload)?;
        offset += item_bytes as u64;
        count += 1;
        if min_key.is_none() {
            min_key = Some(key.clone());
        }
        max_key = key;
    }
    let Some(min_key) = min_key else {
        drop(writer);
        drop(index);
        fs::remove_file(&path)?;
        fs::remove_file(&index_path)?;
        return Ok(None);
    };
    sync_writer(&mut writer)?;
    sync_writer(&mut index)?;
    Ok(Some(RunEntry {
        file: FileEntry::record(scratch_root, rel_path.to_owned())?,
        index: FileEntry::record(scratch_root, index_rel)?,
        count,
        min_key_hex: hex_encode(&min_key),
        max_key_hex: hex_encode(&max_key),
    }))
}

fn remove_run_files(scratch_root: &Path, run: &RunEntry) -> Result<()> {
    fs::remove_file(scratch_root.join(&run.file.rel_path))?;
    fs::remove_file(scratch_root.join(&run.index.rel_path))?;
    Ok(())
}

pub(crate) fn write_run_item(mut writer: impl Write, key: &[u8], payload: &[u8]) -> Result<()> {
    let key_len = u32::try_from(key.len()).context("run key exceeds u32")?;
    let payload_len = u32::try_from(payload.len()).context("run payload exceeds u32")?;
    writer.write_all(&key_len.to_le_bytes())?;
    writer.write_all(&payload_len.to_le_bytes())?;
    writer.write_all(key)?;
    writer.write_all(payload)?;
    Ok(())
}

fn write_index_entry(writer: &mut impl Write, key: &[u8], offset: u64) -> Result<()> {
    let key_len = u32::try_from(key.len()).context("index key exceeds u32")?;
    writer.write_all(&key_len.to_le_bytes())?;
    writer.write_all(&offset.to_le_bytes())?;
    writer.write_all(key)?;
    Ok(())
}

fn read_index_entries(path: &Path) -> Result<Vec<(Vec<u8>, u64)>> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open run index {}", path.display()))?,
    );
    let mut entries = Vec::new();
    while let Some(key_len) = read_optional_u32(&mut reader)? {
        ensure!(
            key_len as usize <= MAX_NATIVE_RECORD_BYTES,
            "run index key exceeds cap"
        );
        let mut offset_bytes = [0_u8; 8];
        reader.read_exact(&mut offset_bytes)?;
        let mut key = vec![0_u8; key_len as usize];
        reader.read_exact(&mut key)?;
        entries.push((key, u64::from_le_bytes(offset_bytes)));
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Range-partitioned merge (mechanism 3)
// ---------------------------------------------------------------------------

struct RunCursor {
    reader: BufReader<File>,
}

impl RunCursor {
    fn next(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let Some(key_len) = read_optional_u32(&mut self.reader)? else {
            return Ok(None);
        };
        let payload_len = read_required_u32(&mut self.reader, "run payload length")?;
        ensure!(
            key_len as usize <= MAX_NATIVE_RECORD_BYTES
                && payload_len as usize <= MAX_NATIVE_RECORD_BYTES,
            "run item exceeds cap"
        );
        let mut key = vec![0_u8; key_len as usize];
        let mut payload = vec![0_u8; payload_len as usize];
        self.reader.read_exact(&mut key)?;
        self.reader.read_exact(&mut payload)?;
        Ok(Some((key, payload)))
    }
}

type HeapItem = Reverse<(Vec<u8>, Vec<u8>, usize)>;
/// One merged `(key, payload, source_run)` entry.
type MergeEntry = (Vec<u8>, Vec<u8>, usize);
/// One half-open key range `[start, end)` (`None` = unbounded side).
type KeyBounds = (Option<Vec<u8>>, Option<Vec<u8>>);

/// K-way merge of sorted runs restricted to the half-open key range
/// `[start, end)`. Runs are entered at the greatest indexed offset below
/// `start` (binary search over the sparse sidecar), so a worker reads only
/// its slice plus at most [`RUN_INDEX_SAMPLE_EVERY`] items of lead-in.
/// Ties order by `(key, payload)` — identical items are interchangeable
/// bytes, so output content is a pure function of the run multiset.
pub(crate) struct RangeMerge {
    heap: BinaryHeap<HeapItem>,
    cursors: Vec<Option<RunCursor>>,
    end: Option<Vec<u8>>,
}

impl RangeMerge {
    /// `read_budget` bounds this merge's TOTAL reader-buffer bytes; the
    /// per-run buffer adapts down for CI-scale runs so `workers x fan-in`
    /// readers stay inside the INV-M5.15 envelope.
    pub(crate) fn open(
        scratch_root: &Path,
        runs: &[RunEntry],
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        read_budget: usize,
    ) -> Result<Self> {
        let per_reader = (read_budget / runs.len().max(1)).clamp(8 * 1024, MERGE_READ_BUFFER);
        ensure!(
            runs.len() <= MAX_MERGE_FAN_IN,
            "merge fan-in {} exceeds cap {MAX_MERGE_FAN_IN}",
            runs.len()
        );
        let mut cursors = Vec::with_capacity(runs.len());
        let mut heap = BinaryHeap::with_capacity(runs.len());
        for (source, run) in runs.iter().enumerate() {
            // Whole-run range exclusion by fences.
            if let (Some(end), Ok(min)) = (end, hex_decode(&run.min_key_hex))
                && min.as_slice() >= end
            {
                cursors.push(None);
                continue;
            }
            if let (Some(start), Ok(max)) = (start, hex_decode(&run.max_key_hex))
                && max.as_slice() < start
            {
                cursors.push(None);
                continue;
            }
            let path = scratch_root.join(&run.file.rel_path);
            let mut file =
                File::open(&path).with_context(|| format!("open run {}", path.display()))?;
            if let Some(start) = start {
                let index = read_index_entries(&scratch_root.join(&run.index.rel_path))?;
                let position = index.partition_point(|(key, _)| key.as_slice() < start);
                let offset = position.checked_sub(1).map_or(0, |before| index[before].1);
                file.seek(SeekFrom::Start(offset))?;
            }
            // Adaptive buffer: never larger than the run itself, never
            // larger than this merge's fair share of the read budget.
            let capacity = usize::try_from(run.file.len.clamp(8 * 1024, per_reader as u64))
                .expect("bounded capacity");
            let mut cursor = RunCursor {
                reader: BufReader::with_capacity(capacity, file),
            };
            // Lead-in skip to the range start.
            let mut first = None;
            while let Some((key, payload)) = cursor.next()? {
                if start.is_none_or(|start| key.as_slice() >= start) {
                    first = Some((key, payload));
                    break;
                }
            }
            if let Some((key, payload)) = first {
                if end.is_none_or(|end| key.as_slice() < end) {
                    heap.push(Reverse((key, payload, source)));
                    cursors.push(Some(cursor));
                    continue;
                }
            }
            cursors.push(None);
        }
        Ok(Self {
            heap,
            cursors,
            end: end.map(<[u8]>::to_vec),
        })
    }

    /// Next `(key, payload, source_run)` inside the range.
    pub(crate) fn next_entry(&mut self) -> Result<Option<MergeEntry>> {
        let Some(Reverse((key, payload, source))) = self.heap.pop() else {
            return Ok(None);
        };
        if let Some(cursor) = self.cursors[source].as_mut() {
            match cursor.next()? {
                Some((next_key, next_payload))
                    if self
                        .end
                        .as_ref()
                        .is_none_or(|end| next_key.as_slice() < end.as_slice()) =>
                {
                    self.heap.push(Reverse((next_key, next_payload, source)));
                }
                _ => self.cursors[source] = None,
            }
        }
        Ok(Some((key, payload, source)))
    }

    pub(crate) fn next_item(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self.next_entry()?.map(|(key, payload, _)| (key, payload)))
    }
}

/// Deterministic splitter selection (mechanism 3): merge every run's sparse
/// index samples plus fences, sort, and take `workers − 1` evenly spaced
/// quantiles. Skew degrades balance, never correctness — range membership
/// is a pure function of the key, and INV-M5.24 holds for any split.
pub(crate) fn choose_splitters(
    scratch_root: &Path,
    runs: &[RunEntry],
    workers: usize,
) -> Result<Vec<Vec<u8>>> {
    if workers <= 1 || runs.is_empty() {
        return Ok(Vec::new());
    }
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for run in runs {
        samples.push(hex_decode(&run.min_key_hex)?);
        samples.push(hex_decode(&run.max_key_hex)?);
        for (key, _) in read_index_entries(&scratch_root.join(&run.index.rel_path))? {
            samples.push(key);
        }
    }
    samples.sort_unstable();
    let mut splitters = Vec::with_capacity(workers - 1);
    for lane in 1..workers {
        let pick = samples[(lane * samples.len()) / workers].clone();
        if splitters.last() != Some(&pick) {
            splitters.push(pick);
        }
    }
    Ok(splitters)
}

/// The `W` half-open, disjoint, key-space-covering ranges induced by the
/// splitters: `(None, s1), [s1, s2), …, [s_{W-1}, None)`.
fn splitter_bounds(splitters: &[Vec<u8>]) -> Vec<KeyBounds> {
    let mut bounds: Vec<KeyBounds> = Vec::with_capacity(splitters.len() + 1);
    let mut lower: Option<Vec<u8>> = None;
    for splitter in splitters {
        bounds.push((lower.clone(), Some(splitter.clone())));
        lower = Some(splitter.clone());
    }
    bounds.push((lower, None));
    bounds
}

// ---------------------------------------------------------------------------
// Framed segments
// ---------------------------------------------------------------------------

/// Write a framed payload segment, recording count + optional fences.
struct SegWriter {
    scratch_root: PathBuf,
    rel_path: String,
    writer: BufWriter<File>,
    count: u64,
    min_key: Option<Vec<u8>>,
    max_key: Option<Vec<u8>>,
}

impl SegWriter {
    fn create(scratch_root: &Path, rel_path: String, write_buffer: usize) -> Result<Self> {
        let path = scratch_root.join(&rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self {
            scratch_root: scratch_root.to_path_buf(),
            rel_path,
            writer: BufWriter::with_capacity(write_buffer, create_new(&path)?),
            count: 0,
            min_key: None,
            max_key: None,
        })
    }

    fn push(&mut self, payload: &[u8], fence_key: Option<&[u8]>) -> Result<()> {
        write_framed(&mut self.writer, payload)?;
        self.count += 1;
        if let Some(key) = fence_key {
            if self.min_key.is_none() {
                self.min_key = Some(key.to_vec());
            }
            self.max_key = Some(key.to_vec());
        }
        Ok(())
    }

    fn finish(mut self) -> Result<SegEntry> {
        sync_writer(&mut self.writer)?;
        drop(self.writer);
        Ok(SegEntry {
            file: FileEntry::record(&self.scratch_root, self.rel_path)?,
            count: self.count,
            min_key_hex: self.min_key.map(|key| hex_encode(&key)),
            max_key_hex: self.max_key.map(|key| hex_encode(&key)),
        })
    }
}

/// Sequential reader over the concatenation of framed segments — the
/// worker-count-invariant stream the serial materializer consumes.
pub(crate) struct SegmentedReader {
    scratch_root: PathBuf,
    segs: std::collections::VecDeque<String>,
    current: Option<crate::m5_load::FramedReader>,
}

impl SegmentedReader {
    pub(crate) fn open(scratch_root: &Path, segs: &[SegEntry]) -> Self {
        Self {
            scratch_root: scratch_root.to_path_buf(),
            segs: segs.iter().map(|seg| seg.file.rel_path.clone()).collect(),
            current: None,
        }
    }

    pub(crate) fn next(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            if let Some(reader) = self.current.as_mut() {
                if let Some(payload) = reader.next()? {
                    return Ok(Some(payload));
                }
                self.current = None;
            }
            let Some(rel) = self.segs.pop_front() else {
                return Ok(None);
            };
            self.current = Some(crate::m5_load::FramedReader::open(
                &self.scratch_root.join(rel),
            )?);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage resume plan
// ---------------------------------------------------------------------------

/// Resume plan under the staged-GC rule.
///
/// The resume point `R` means: skip stages `1..=R`, re-execute `R+1..`.
/// `R` is FEASIBLE iff manifests `1..=R` all parse AND every file group
/// produced by a stage `<= R` whose LAST consumer is a stage `> R` (or
/// the always-re-run materializer) is present and fingerprint-valid.
/// Groups whose last consumer is `<= R` may legally be absent — they were
/// garbage-collected after that consumer's manifest became durable.
/// `plan_stages` picks the maximum feasible `R`.
struct StagePlan {
    s1: Option<S1Manifest>,
    s2: Option<S2Manifest>,
    s3: Option<S3Manifest>,
    s4: Option<S4Manifest>,
    s5: Option<S5Manifest>,
    s6: Option<S6Manifest>,
    /// 0 = nothing valid (full build); N = stages 1..=N valid, resume at N+1.
    resume_from: usize,
}

fn files_ok(scratch: &Path, entries: &[FileEntry]) -> bool {
    entries.iter().all(|entry| entry.validate(scratch))
}

fn run_files(runs: &[RunEntry]) -> Vec<FileEntry> {
    runs.iter()
        .flat_map(|run| [run.file.clone(), run.index.clone()])
        .collect()
}

fn seg_files(segs: &[SegEntry]) -> Vec<FileEntry> {
    segs.iter().map(|seg| seg.file.clone()).collect()
}

/// The materializer (always re-run) is modeled as stage 7.
const MATERIALIZER_STAGE: usize = 7;

fn plan_stages(scratch: &Path) -> StagePlan {
    let s1: Option<S1Manifest> = read_stage_manifest(scratch, STAGE_S1);
    let s2: Option<S2Manifest> = read_stage_manifest(scratch, STAGE_S2);
    let s3: Option<S3Manifest> = read_stage_manifest(scratch, STAGE_S3);
    let s4: Option<S4Manifest> = read_stage_manifest(scratch, STAGE_S4);
    let s5: Option<S5Manifest> = read_stage_manifest(scratch, STAGE_S5);
    let s6: Option<S6Manifest> = read_stage_manifest(scratch, STAGE_S6);
    let parsed = [
        s1.is_some(),
        s2.is_some(),
        s3.is_some(),
        s4.is_some(),
        s5.is_some(),
        s6.is_some(),
    ];

    // (producer_stage, last_consumer_stage, files) — the staged-GC table.
    let mut groups: Vec<(usize, usize, Vec<FileEntry>)> = Vec::new();
    if let Some(s1) = &s1 {
        groups.push((1, 3, run_files(&s1.runs)));
    }
    if let Some(s3) = &s3 {
        let bindings: Vec<SegEntry> = s3.ranges.iter().map(|r| r.bindings.clone()).collect();
        let nodes: Vec<SegEntry> = s3.ranges.iter().map(|r| r.nodes.clone()).collect();
        let rels: Vec<SegEntry> = s3.ranges.iter().map(|r| r.rels.clone()).collect();
        groups.push((3, 4, seg_files(&bindings)));
        groups.push((3, 4, run_files(&s3.endpoint_runs)));
        groups.push((3, 5, seg_files(&rels)));
        groups.push((3, MATERIALIZER_STAGE, seg_files(&nodes)));
    }
    if let Some(s4) = &s4 {
        groups.push((4, 5, run_files(&s4.resolved_runs)));
    }
    if let Some(s5) = &s5 {
        groups.push((5, 6, run_files(&s5.out_runs)));
        groups.push((5, 6, run_files(&s5.in_runs)));
        groups.push((5, MATERIALIZER_STAGE, seg_files(&s5.rel_segs)));
    }
    if let Some(s6) = &s6 {
        groups.push((6, MATERIALIZER_STAGE, seg_files(&s6.out_segs)));
        groups.push((6, MATERIALIZER_STAGE, seg_files(&s6.in_segs)));
    }

    let mut resume_from = 0;
    'candidates: for candidate in (1..=6_usize).rev() {
        if !parsed[..candidate].iter().all(|ok| *ok) {
            continue;
        }
        for (producer, consumer, files) in &groups {
            if *producer <= candidate && *consumer > candidate && !files_ok(scratch, files) {
                continue 'candidates;
            }
        }
        resume_from = candidate;
        break;
    }
    StagePlan {
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        resume_from,
    }
}

/// Remove a stage's directory tree (before re-running it).
fn reset_stage_dir(scratch: &Path, stage_dir: &str, stage: &str) -> Result<()> {
    let dir = scratch.join(stage_dir);
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .with_context(|| format!("reset stale stage dir {}", dir.display()))?;
    }
    let manifest = manifest_path(scratch, stage);
    if manifest.exists() {
        fs::remove_file(&manifest)?;
    }
    Ok(())
}

/// GC a consumed file group once its last consumer's manifest is durable.
/// Missing files are fine (partial GC after a crash).
fn collect_group(scratch: &Path, entries: &[FileEntry]) {
    for entry in entries {
        let _ = fs::remove_file(scratch.join(&entry.rel_path));
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Input byte-range partitions resynced to frame boundaries (mechanism 1).
fn partition_input(input: &Path, workers: usize) -> Result<Vec<(u64, u64)>> {
    let len = fs::metadata(input)
        .with_context(|| format!("stat load input {}", input.display()))?
        .len();
    let lanes = workers.max(1) as u64;
    let chunk = len.div_ceil(lanes).max(1);
    let mut parts = Vec::new();
    let mut start = 0_u64;
    while start < len {
        let end = (start + chunk).min(len);
        parts.push((start, end));
        start = end;
    }
    if parts.is_empty() {
        parts.push((0, 0));
    }
    Ok(parts)
}

/// Drive the full §4 pipeline: stages s1–s6 (parallel, manifest-checkpointed)
/// then the serial materialization, returning the census-shaped report.
/// Byte-identical output for any worker count (INV-M5.24) — see module docs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pipeline(
    input: &Path,
    format: LoadFormat,
    building: &Path,
    root: &Path,
    tenant: TenantId,
    frontier: LoaderMigrationFrontier,
    blob: &Arc<BlobStore>,
    limits: &LoadLimits,
) -> Result<LoadReport> {
    let started = Instant::now();
    let scratch = building.join("scratch");
    let workers = limits.effective_workers();
    raise_fd_limit((workers * MAX_MERGE_FAN_IN + 512) as u64);
    let sampler = RssSampler::start(limits.rss_cap_bytes, limits.rss_sample_every_ms);
    let mut report = LoadReport {
        workers: workers as u64,
        ..LoadReport::default()
    };

    // ---- Resume plan ------------------------------------------------------
    let mut plan = plan_stages(&scratch);
    // Input identity: durable stages belong to THIS input or none at all.
    if plan.resume_from >= 1 {
        let fingerprint = InputFingerprint::of(input)?;
        if plan.s1.as_ref().map(|s1| &s1.fingerprint) != Some(&fingerprint) {
            plan = StagePlan {
                s1: None,
                s2: None,
                s3: None,
                s4: None,
                s5: None,
                s6: None,
                resume_from: 0,
            };
        }
    }
    let resume_from = plan.resume_from;
    for (index, stage) in [STAGE_S1, STAGE_S2, STAGE_S3, STAGE_S4, STAGE_S5, STAGE_S6]
        .iter()
        .enumerate()
    {
        let stage_dir = format!("s{}", index + 1);
        if index < resume_from {
            report.resumed_stages.push((*stage).to_owned());
        } else {
            reset_stage_dir(&scratch, &stage_dir, stage)?;
        }
    }
    report.resumed = report.resumed || resume_from > 0;

    // ---- s1: parse + per-worker canonical run generation -------------------
    sampler.set_stage("s1-parse");
    let s1 = if resume_from >= 1 {
        plan.s1.take().context("resume plan lost s1")?
    } else {
        let s1 = build_canonical_runs(input, format, &scratch, workers, limits, &sampler)?;
        write_stage_manifest(&scratch, STAGE_S1, &s1)?;
        s1
    };
    report.records = s1.census.records;

    // ---- census-derived budgets + plan-time projection (amendment §5) ------
    let budgets = plan_owner_budgets(&s1.census);
    project_disk_or_refuse(root, &s1.census, &budgets, limits)?;

    // ---- s2: phase-1 counts + splitters (mechanism 4, phase 1) -------------
    sampler.set_stage("s2-phase1");
    let s2 = if resume_from >= 2 {
        plan.s2.take().context("resume plan lost s2")?
    } else {
        let s2 = phase1_counts(&scratch, &s1, workers, limits)?;
        write_stage_manifest(&scratch, STAGE_S2, &s2)?;
        s2
    };

    // ---- s3: phase-2 segment emission (mechanism 4, phase 2) ---------------
    sampler.set_stage("s3-phase2");
    let s3 = if resume_from >= 3 {
        plan.s3.take().context("resume plan lost s3")?
    } else {
        let s3 = phase2_segments(&scratch, &s1, &s2, workers, limits, &sampler)?;
        write_stage_manifest(&scratch, STAGE_S3, &s3)?;
        // s1 runs' last consumer is s3.
        collect_group(&scratch, &run_files(&s1.runs));
        s3
    };
    report.nodes = s3.ranges.iter().map(|range| range.nodes.count).sum();

    // ---- s4: endpoint resolution (INV-M5.18) --------------------------------
    sampler.set_stage("s4-resolve");
    let s4 = if resume_from >= 4 {
        plan.s4.take().context("resume plan lost s4")?
    } else {
        let s4 = resolve_endpoints(&scratch, &s3, workers, limits, &sampler)?;
        write_stage_manifest(&scratch, STAGE_S4, &s4)?;
        collect_group(&scratch, &run_files(&s3.endpoint_runs));
        collect_group(
            &scratch,
            &s3.ranges
                .iter()
                .map(|range| range.bindings.file.clone())
                .collect::<Vec<_>>(),
        );
        s4
    };

    // ---- s5: resolved-rel assembly + TEL run generation ---------------------
    sampler.set_stage("s5-assemble");
    let s5 = if resume_from >= 5 {
        plan.s5.take().context("resume plan lost s5")?
    } else {
        let s5 = assemble_relationships(&scratch, &s3, &s4, workers, limits, &sampler)?;
        write_stage_manifest(&scratch, STAGE_S5, &s5)?;
        collect_group(&scratch, &run_files(&s4.resolved_runs));
        collect_group(
            &scratch,
            &s3.ranges
                .iter()
                .map(|range| range.rels.file.clone())
                .collect::<Vec<_>>(),
        );
        s5
    };
    report.relationships = s5.relationships;

    // ---- s6: TEL range merges ------------------------------------------------
    sampler.set_stage("s6-tel-merge");
    let s6 = if resume_from >= 6 {
        plan.s6.take().context("resume plan lost s6")?
    } else {
        let s6 = merge_tel_runs(&scratch, &s5, workers, limits)?;
        write_stage_manifest(&scratch, STAGE_S6, &s6)?;
        collect_group(&scratch, &run_files(&s5.out_runs));
        collect_group(&scratch, &run_files(&s5.in_runs));
        s6
    };

    // ---- post-s6 EXACT TEL disk projection (fail-fast, INV-M5.25) ----------
    // The pass-1 census cannot see distinct (owner, type) group counts, so
    // the plan-time projection only lower-bounds STORE_TEL; the s6 merges
    // count the real densified-packing cost (#1519 BLOCK_FIX FIX 3: priced
    // via `project_dense_tel_bytes_for_blocks`, not page-per-block), and a
    // store that cannot fit is refused HERE — before the materializer
    // writes a byte of it.
    if s6.out_pages + s6.in_pages > 0 {
        crate::m5_load::project_tel_or_refuse(
            root,
            &s1.census,
            s6.out_pages + s6.in_pages,
            s6.out_entries + s6.in_entries,
            limits,
        )?;
    }

    // ---- s7: serial materialization (D2 surface, untouched) -----------------
    sampler.set_stage("s7-materialize");
    materialize(
        building,
        &scratch,
        tenant,
        frontier,
        blob,
        &budgets,
        &s3,
        &s5,
        &s6,
        &mut report,
    )?;

    ensure!(
        report.nodes + report.relationships == report.records,
        "pipeline census mismatch: {} nodes + {} rels != {} records",
        report.nodes,
        report.relationships,
        report.records
    );
    report.rss_samples = sampler.finish()?;
    report.elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    Ok(report)
}

/// s1 — mechanism 1 + 2: partition the input on frame boundaries; each
/// worker parses its partition into its own sorted, fenced, indexed runs
/// while accumulating the pass-1 census.
fn build_canonical_runs(
    input: &Path,
    format: LoadFormat,
    scratch: &Path,
    workers: usize,
    limits: &LoadLimits,
    sampler: &RssSampler,
) -> Result<S1Manifest> {
    ensure!(
        format == LoadFormat::Native,
        "M5-D3 partitions native input"
    );
    let fingerprint = InputFingerprint::of(input)?;
    let partitions = partition_input(input, workers)?;
    let per_worker_runs = (MAX_MERGE_FAN_IN / partitions.len().max(1)).max(1);
    let lanes = partitions.len();
    let tasks: Vec<_> = partitions
        .into_iter()
        .enumerate()
        .map(|(worker, (start, end))| {
            let input = input.to_path_buf();
            let scratch = scratch.to_path_buf();
            move || -> Result<(LoadCensus, Vec<RunEntry>)> {
                stagger(worker, lanes);
                let mut source = NativeRecordSource::open_range(&input, start, end)?;
                let mut writer = RunSetWriter::new(
                    &scratch,
                    &format!("s1/w{worker}"),
                    limits.sort_memory_bytes,
                    per_worker_runs,
                )?;
                let mut census = LoadCensus::default();
                let mut since_check = 0_u64;
                while let Some(record) = source.next_record()? {
                    let key = canonical_sort_key(&record);
                    let payload = encode_canonical_record(&record)?;
                    census.records += 1;
                    census.payload_bytes += payload.len() as u64;
                    match &record {
                        LoadRecord::Node { external_id, .. } => {
                            census.nodes += 1;
                            census.node_external_id_bytes += external_id.len() as u64;
                        }
                        LoadRecord::Relationship { external_id, .. } => {
                            census.relationships += 1;
                            census.rel_external_id_bytes += external_id.len() as u64;
                        }
                    }
                    writer.push(key, payload)?;
                    since_check += 1;
                    if since_check % 65_536 == 0 {
                        ensure!(!sampler.exceeded(), "continuous RSS cap exceeded during s1");
                    }
                }
                Ok((census, writer.finish()?))
            }
        })
        .collect();
    let outputs = run_tasks(workers, tasks)?;
    let mut census = LoadCensus::default();
    let mut runs = Vec::new();
    for (worker_census, worker_runs) in outputs {
        census.absorb(&worker_census);
        runs.extend(worker_runs);
    }
    Ok(S1Manifest {
        fingerprint,
        census,
        runs,
    })
}

/// s2 — mechanism 4 phase 1: choose splitters, then per-range streaming
/// counts of unique node/relationship keys with the sort-adjacent duplicate
/// hard error (bulk load is not upsert — INV-M5.19).
fn phase1_counts(
    scratch: &Path,
    s1: &S1Manifest,
    workers: usize,
    limits: &LoadLimits,
) -> Result<S2Manifest> {
    fs::create_dir_all(scratch.join("s2"))?;
    let splitters = choose_splitters(scratch, &s1.runs, workers)?;
    let bounds = splitter_bounds(&splitters);
    let lanes = bounds.len();
    let dup_by_run_only = range_by_run();
    let tasks: Vec<_> = bounds
        .iter()
        .cloned()
        .enumerate()
        .map(|(range_index, (start, end))| {
            let runs = s1.runs.clone();
            let scratch = scratch.to_path_buf();
            move || -> Result<(u64, u64)> {
                stagger(range_index, lanes);
                let mut merge = RangeMerge::open(
                    &scratch,
                    &runs,
                    start.as_deref(),
                    end.as_deref(),
                    limits.sort_memory_bytes,
                )?;
                let mut nodes = 0_u64;
                let mut rels = 0_u64;
                let mut last: Option<(Vec<u8>, usize)> = None;
                while let Some((key, _payload, source)) = merge.next_entry()? {
                    if let Some((last_key, last_source)) = &last
                        && *last_key == key
                        && (!dup_by_run_only || *last_source == source)
                    {
                        let kind = if key.first() == Some(&0) {
                            "node"
                        } else {
                            "relationship"
                        };
                        bail!(
                            "duplicate {kind} external_id {:?}",
                            String::from_utf8_lossy(key.get(1..).unwrap_or_default())
                        );
                    }
                    match key.first() {
                        Some(0) => nodes += 1,
                        Some(1) => rels += 1,
                        other => bail!("unknown canonical key kind {other:?}"),
                    }
                    last = Some((key, source));
                }
                Ok((nodes, rels))
            }
        })
        .collect();
    let range_counts = run_tasks(workers, tasks)?;
    let (nodes, rels) = range_counts
        .iter()
        .fold((0_u64, 0_u64), |acc, (n, r)| (acc.0 + n, acc.1 + r));
    ensure!(
        nodes == s1.census.nodes && rels == s1.census.relationships,
        "phase-1 range counts ({nodes} nodes, {rels} rels) disagree with the parse census \
         ({} nodes, {} rels)",
        s1.census.nodes,
        s1.census.relationships
    );
    Ok(S2Manifest {
        splitters_hex: splitters.iter().map(|key| hex_encode(key)).collect(),
        range_counts,
    })
}

/// Per-range dense-id bases from the phase-1 prefix sum. The
/// [`arrival_order_ids`] seam models the reverted defect (drop phase 1;
/// bases follow worker arrival, not global sort order) deterministically
/// by reversing the range order for W>1.
fn id_bases(range_counts: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut order: Vec<usize> = (0..range_counts.len()).collect();
    if arrival_order_ids() {
        order.reverse();
    }
    let mut bases = vec![(0_u64, 0_u64); range_counts.len()];
    let (mut node_base, mut rel_base) = (0_u64, 0_u64);
    for range_index in order {
        bases[range_index] = (node_base, rel_base);
        node_base += range_counts[range_index].0;
        rel_base += range_counts[range_index].1;
    }
    bases
}

/// s3 — mechanism 4 phase 2: re-stream each range stamping
/// `base + ordinal` ids; emit per-range binding/node-artifact/rel segments
/// and the endpoint-request runs (global relationship ordinals).
fn phase2_segments(
    scratch: &Path,
    s1: &S1Manifest,
    s2: &S2Manifest,
    workers: usize,
    limits: &LoadLimits,
    sampler: &RssSampler,
) -> Result<S3Manifest> {
    let splitters = s2
        .splitters_hex
        .iter()
        .map(|hex| hex_decode(hex))
        .collect::<Result<Vec<_>>>()?;
    let bounds = splitter_bounds(&splitters);
    ensure!(
        bounds.len() == s2.range_counts.len(),
        "phase-2 range shape disagrees with phase 1"
    );
    let bases = id_bases(&s2.range_counts);
    let per_worker_runs = (MAX_MERGE_FAN_IN / bounds.len().max(1)).max(1);
    let lanes = bounds.len();
    let tasks: Vec<_> = bounds
        .iter()
        .cloned()
        .enumerate()
        .map(|(range_index, (start, end))| {
            let runs = s1.runs.clone();
            let scratch = scratch.to_path_buf();
            let (node_base, rel_base) = bases[range_index];
            let (expected_nodes, expected_rels) = s2.range_counts[range_index];
            move || -> Result<(S3Range, Vec<RunEntry>)> {
                stagger(range_index, lanes);
                let buffer = adaptive_write_buffer(limits.sort_memory_bytes);
                let mut merge = RangeMerge::open(
                    &scratch,
                    &runs,
                    start.as_deref(),
                    end.as_deref(),
                    limits.sort_memory_bytes,
                )?;
                let mut bindings = SegWriter::create(
                    &scratch,
                    format!("s3/bindings-{range_index:04}.seg"),
                    buffer,
                )?;
                let mut nodes =
                    SegWriter::create(&scratch, format!("s3/nodes-{range_index:04}.seg"), buffer)?;
                let mut rels =
                    SegWriter::create(&scratch, format!("s3/rels-{range_index:04}.seg"), buffer)?;
                let mut endpoints = RunSetWriter::new(
                    &scratch,
                    &format!("s3/w{range_index}"),
                    limits.sort_memory_bytes,
                    per_worker_runs,
                )?;
                let (mut node_ordinal, mut rel_ordinal) = (0_u64, 0_u64);
                while let Some((_key, payload, _source)) = merge.next_entry()? {
                    match decode_canonical_record(&payload)? {
                        LoadRecord::Node {
                            external_id,
                            label,
                            float_bits,
                            opaque,
                        } => {
                            let internal_id = node_base
                                .checked_add(node_ordinal)
                                .and_then(|base| base.checked_add(1))
                                .context("node id overflow")?;
                            node_ordinal += 1;
                            let mut binding = Vec::new();
                            put_bytes(&mut binding, &external_id)?;
                            binding.extend_from_slice(&internal_id.to_le_bytes());
                            bindings.push(&binding, Some(&external_id))?;
                            let mut encoded = Vec::new();
                            encoded.extend_from_slice(&internal_id.to_le_bytes());
                            put_bytes(&mut encoded, &external_id)?;
                            encoded.extend_from_slice(&label.to_le_bytes());
                            encoded.extend_from_slice(&float_bits.to_le_bytes());
                            put_bytes(&mut encoded, &opaque)?;
                            nodes.push(&encoded, None)?;
                        }
                        relationship @ LoadRecord::Relationship { .. } => {
                            let LoadRecord::Relationship {
                                external_id,
                                source_external_id,
                                target_external_id,
                                ..
                            } = &relationship
                            else {
                                unreachable!("matched relationship");
                            };
                            let ordinal = rel_base
                                .checked_add(rel_ordinal)
                                .context("relationship ordinal overflow")?;
                            rel_ordinal += 1;
                            rels.push(&encode_canonical_record(&relationship)?, None)?;
                            endpoints.push(
                                source_external_id.clone(),
                                encode_endpoint_request(
                                    source_external_id,
                                    external_id,
                                    ordinal,
                                    0,
                                )?,
                            )?;
                            endpoints.push(
                                target_external_id.clone(),
                                encode_endpoint_request(
                                    target_external_id,
                                    external_id,
                                    ordinal,
                                    1,
                                )?,
                            )?;
                        }
                    }
                    if (node_ordinal + rel_ordinal) % 65_536 == 0 {
                        ensure!(!sampler.exceeded(), "continuous RSS cap exceeded during s3");
                    }
                }
                if !arrival_order_ids() {
                    ensure!(
                        node_ordinal == expected_nodes && rel_ordinal == expected_rels,
                        "phase-2 emission ({node_ordinal} nodes, {rel_ordinal} rels) disagrees \
                         with phase-1 counts ({expected_nodes}, {expected_rels})"
                    );
                }
                Ok((
                    S3Range {
                        bindings: bindings.finish()?,
                        nodes: nodes.finish()?,
                        rels: rels.finish()?,
                        node_base,
                        rel_base,
                    },
                    endpoints.finish()?,
                ))
            }
        })
        .collect();
    let outputs = run_tasks(workers, tasks)?;
    let mut ranges = Vec::new();
    let mut endpoint_runs = Vec::new();
    for (range, runs) in outputs {
        ranges.push(range);
        endpoint_runs.extend(runs);
    }
    Ok(S3Manifest {
        ranges,
        endpoint_runs,
    })
}

/// Binding-side cursor for the s4 merge-join: walks the ordered binding
/// segments, skipping whole segments whose fences fall below the range.
struct BindingScan {
    scratch: PathBuf,
    segs: std::collections::VecDeque<SegEntry>,
    current: Option<crate::m5_load::FramedReader>,
    pending: Option<(Vec<u8>, u64)>,
}

impl BindingScan {
    fn open(scratch: &Path, ranges: &[S3Range], start: Option<&[u8]>) -> Self {
        let segs = ranges
            .iter()
            .map(|range| range.bindings.clone())
            .filter(|seg| {
                // Keep segments whose max fence can still reach the range.
                match (start, &seg.max_key_hex) {
                    (Some(start), Some(max_hex)) => {
                        hex_decode(max_hex).is_ok_and(|max| max.as_slice() >= start)
                    }
                    _ => true,
                }
            })
            .collect();
        Self {
            scratch: scratch.to_path_buf(),
            segs,
            current: None,
            pending: None,
        }
    }

    /// Advance until the current binding's external id is `>= target`,
    /// returning the binding when it equals `target`.
    fn lookup(&mut self, target: &[u8]) -> Result<Option<u64>> {
        loop {
            if self.pending.is_none() {
                let Some(payload) = self.next_payload()? else {
                    return Ok(None);
                };
                self.pending = Some(decode_binding(&payload)?);
            }
            match &self.pending {
                Some((external, internal)) if external.as_slice() == target => {
                    return Ok(Some(*internal));
                }
                Some((external, _)) if external.as_slice() < target => {
                    self.pending = None;
                }
                _ => return Ok(None),
            }
        }
    }

    fn next_payload(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            if let Some(reader) = self.current.as_mut() {
                if let Some(payload) = reader.next()? {
                    return Ok(Some(payload));
                }
                self.current = None;
            }
            let Some(seg) = self.segs.pop_front() else {
                return Ok(None);
            };
            self.current = Some(crate::m5_load::FramedReader::open(
                &self.scratch.join(&seg.file.rel_path),
            )?);
        }
    }
}

/// s4 — endpoint resolution: range-partitioned merge of the endpoint
/// requests joined against the ordered node bindings. A missing endpoint is
/// a deterministic hard error, never a skipped relationship (INV-M5.18).
fn resolve_endpoints(
    scratch: &Path,
    s3: &S3Manifest,
    workers: usize,
    limits: &LoadLimits,
    sampler: &RssSampler,
) -> Result<S4Manifest> {
    fs::create_dir_all(scratch.join("s4"))?;
    let splitters = choose_splitters(scratch, &s3.endpoint_runs, workers)?;
    let bounds = splitter_bounds(&splitters);
    let per_worker_runs = (MAX_MERGE_FAN_IN / bounds.len().max(1)).max(1);
    let lanes = bounds.len();
    let tasks: Vec<_> = bounds
        .iter()
        .cloned()
        .enumerate()
        .map(|(range_index, (start, end))| {
            let runs = s3.endpoint_runs.clone();
            let ranges = s3.ranges.clone();
            let scratch = scratch.to_path_buf();
            move || -> Result<Vec<RunEntry>> {
                stagger(range_index, lanes);
                let mut merge = RangeMerge::open(
                    &scratch,
                    &runs,
                    start.as_deref(),
                    end.as_deref(),
                    limits.sort_memory_bytes,
                )?;
                let mut bindings = BindingScan::open(&scratch, &ranges, start.as_deref());
                let mut resolved = RunSetWriter::new(
                    &scratch,
                    &format!("s4/w{range_index}"),
                    limits.sort_memory_bytes,
                    per_worker_runs,
                )?;
                let mut processed = 0_u64;
                while let Some((_key, payload, _source)) = merge.next_entry()? {
                    let (endpoint, relation, ordinal, side) = decode_endpoint_request(&payload)?;
                    let internal = bindings.lookup(&endpoint)?.with_context(|| {
                        format!(
                            "relationship {:?} references missing endpoint {:?}",
                            String::from_utf8_lossy(&relation),
                            String::from_utf8_lossy(&endpoint)
                        )
                    })?;
                    // Key: (canonical relationship ORDINAL, side), big-endian
                    // fixed width — total order aligned with the canonical
                    // relationship stream for ANY external-id shape.
                    let mut key = Vec::with_capacity(9);
                    key.extend_from_slice(&ordinal.to_be_bytes());
                    key.push(side);
                    let mut value = Vec::new();
                    put_bytes(&mut value, &relation)?;
                    value.push(side);
                    value.extend_from_slice(&internal.to_le_bytes());
                    resolved.push(key, value)?;
                    processed += 1;
                    if processed % 65_536 == 0 {
                        ensure!(!sampler.exceeded(), "continuous RSS cap exceeded during s4");
                    }
                }
                resolved.finish()
            }
        })
        .collect();
    let outputs = run_tasks(workers, tasks)?;
    Ok(S4Manifest {
        resolved_runs: outputs.into_iter().flatten().collect(),
    })
}

fn ordinal_side_key(ordinal: u64, side: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.extend_from_slice(&ordinal.to_be_bytes());
    key.push(side);
    key
}

/// s5 — pass-fused resolved merge + endpoint join + TEL run generation:
/// task `k` lock-step joins canonical relationship segment `k` against the
/// resolved-endpoint merge restricted to `[rel_base_k, rel_base_k+count)`,
/// stamps `rel_id = ordinal + 1`, and emits both TEL orderings' runs.
fn assemble_relationships(
    scratch: &Path,
    s3: &S3Manifest,
    s4: &S4Manifest,
    workers: usize,
    limits: &LoadLimits,
    sampler: &RssSampler,
) -> Result<S5Manifest> {
    fs::create_dir_all(scratch.join("s5"))?;
    let per_worker_runs = (MAX_MERGE_FAN_IN / (s3.ranges.len().max(1) * 2)).max(1);
    let lanes = s3.ranges.len();
    let tasks: Vec<_> = s3
        .ranges
        .iter()
        .cloned()
        .enumerate()
        .map(|(range_index, range)| {
            let resolved_runs = s4.resolved_runs.clone();
            let scratch = scratch.to_path_buf();
            move || -> Result<(SegEntry, Vec<RunEntry>, Vec<RunEntry>, u64)> {
                stagger(range_index, lanes);
                let buffer = adaptive_write_buffer(limits.sort_memory_bytes);
                let start = ordinal_side_key(range.rel_base, 0);
                let end = ordinal_side_key(range.rel_base + range.rels.count, 0);
                let mut resolved = RangeMerge::open(
                    &scratch,
                    &resolved_runs,
                    Some(&start),
                    Some(&end),
                    limits.sort_memory_bytes,
                )?;
                let mut rel_reader =
                    crate::m5_load::FramedReader::open(&scratch.join(&range.rels.file.rel_path))?;
                let mut rel_seg =
                    SegWriter::create(&scratch, format!("s5/rels-{range_index:04}.seg"), buffer)?;
                let mut out_runs = RunSetWriter::new(
                    &scratch,
                    &format!("s5/w{range_index}/out"),
                    limits.sort_memory_bytes / 2,
                    per_worker_runs,
                )?;
                let mut in_runs = RunSetWriter::new(
                    &scratch,
                    &format!("s5/w{range_index}/in"),
                    limits.sort_memory_bytes / 2,
                    per_worker_runs,
                )?;
                let mut local_ordinal = 0_u64;
                while let Some(payload) = rel_reader.next()? {
                    let LoadRecord::Relationship {
                        external_id,
                        type_id,
                        float_bits,
                        opaque,
                        ..
                    } = decode_canonical_record(&payload)?
                    else {
                        bail!("node record in relationship segment");
                    };
                    let ordinal = range.rel_base + local_ordinal;
                    local_ordinal += 1;
                    let (source_id, target_id) =
                        next_resolved_pair(&mut resolved, &external_id, ordinal)?;
                    let internal_id = ordinal.checked_add(1).context("rel id overflow")?;
                    let mut encoded = Vec::new();
                    encoded.extend_from_slice(&internal_id.to_le_bytes());
                    put_bytes(&mut encoded, &external_id)?;
                    encoded.extend_from_slice(&type_id.to_le_bytes());
                    encoded.extend_from_slice(&source_id.to_le_bytes());
                    encoded.extend_from_slice(&target_id.to_le_bytes());
                    encoded.extend_from_slice(&float_bits.to_le_bytes());
                    put_bytes(&mut encoded, &opaque)?;
                    rel_seg.push(&encoded, None)?;
                    out_runs.push(
                        tel_run_key(source_id, type_id, internal_id),
                        encode_tel_run(source_id, type_id, target_id, internal_id),
                    )?;
                    in_runs.push(
                        tel_run_key(target_id, type_id, internal_id),
                        encode_tel_run(target_id, type_id, source_id, internal_id),
                    )?;
                    if local_ordinal % 65_536 == 0 {
                        ensure!(!sampler.exceeded(), "continuous RSS cap exceeded during s5");
                    }
                }
                ensure!(
                    resolved.next_item()?.is_none(),
                    "resolved endpoint stream has entries for no relationship"
                );
                Ok((
                    rel_seg.finish()?,
                    out_runs.finish()?,
                    in_runs.finish()?,
                    local_ordinal,
                ))
            }
        })
        .collect();
    let outputs = run_tasks(workers, tasks)?;
    let mut manifest = S5Manifest {
        rel_segs: Vec::new(),
        out_runs: Vec::new(),
        in_runs: Vec::new(),
        relationships: 0,
    };
    for (seg, out, r#in, count) in outputs {
        manifest.rel_segs.push(seg);
        manifest.out_runs.extend(out);
        manifest.in_runs.extend(r#in);
        manifest.relationships += count;
    }
    Ok(manifest)
}

/// Exactly one source (side 0) and one target (side 1) entry per
/// relationship ordinal, adjacent in the resolved merge.
fn next_resolved_pair(
    resolved: &mut RangeMerge,
    external_id: &[u8],
    ordinal: u64,
) -> Result<(u64, u64)> {
    let mut source = None;
    let mut target = None;
    for _ in 0..2 {
        let (key, payload, _) = resolved
            .next_entry()?
            .context("resolved endpoint stream ended early")?;
        ensure!(
            key.len() == 9 && key[..8] == ordinal.to_be_bytes(),
            "resolved endpoint stream is misaligned at ordinal {ordinal}"
        );
        let (relation, side, internal) = crate::m5_load::decode_resolved_endpoint(&payload)?;
        ensure!(
            relation == external_id,
            "resolved endpoint stream names relationship {:?}, expected {:?}",
            String::from_utf8_lossy(&relation),
            String::from_utf8_lossy(external_id)
        );
        match side {
            0 => source = Some(internal),
            _ => target = Some(internal),
        }
    }
    Ok((
        source.context("relationship is missing its resolved source")?,
        target.context("relationship is missing its resolved target")?,
    ))
}

/// s6 — range-partitioned merges of both TEL orderings into segments whose
/// concatenation is the exact serial TEL stream.
fn merge_tel_runs(
    scratch: &Path,
    s5: &S5Manifest,
    workers: usize,
    limits: &LoadLimits,
) -> Result<S6Manifest> {
    fs::create_dir_all(scratch.join("s6"))?;
    let write_buffer = adaptive_write_buffer(limits.sort_memory_bytes);
    let read_budget = limits.sort_memory_bytes;
    let merge_direction = |runs: &[RunEntry], label: &str| -> Result<(Vec<SegEntry>, u64, u64)> {
        let splitters = choose_splitters(scratch, runs, workers)?;
        let bounds = splitter_bounds(&splitters);
        let lanes = bounds.len();
        let tasks: Vec<_> = bounds
            .iter()
            .cloned()
            .enumerate()
            .map(|(range_index, (start, end))| {
                let runs = runs.to_vec();
                let scratch = scratch.to_path_buf();
                let label = label.to_owned();
                move || -> Result<(SegEntry, u64)> {
                    stagger(range_index, lanes);
                    let mut merge = RangeMerge::open(
                        &scratch,
                        &runs,
                        start.as_deref(),
                        end.as_deref(),
                        read_budget,
                    )?;
                    let mut seg = SegWriter::create(
                        &scratch,
                        format!("s6/tel-{label}-{range_index:04}.seg"),
                        write_buffer,
                    )?;
                    // EXACT TEL block/page census: mirror the materializer's
                    // block boundaries — new page on (owner, type) change or
                    // a full chunk (FRESH_TEL_ENTRIES_PER_PAGE).
                    let mut pages = 0_u64;
                    let mut group: Option<[u8; 12]> = None;
                    let mut chunk = 0_u64;
                    while let Some((key, payload, _source)) = merge.next_entry()? {
                        ensure!(key.len() == 20, "TEL run key must be 20 bytes");
                        let mut owner_type = [0_u8; 12];
                        owner_type.copy_from_slice(&key[..12]);
                        if group != Some(owner_type) || chunk >= FRESH_TEL_ENTRIES_PER_PAGE {
                            pages += 1;
                            chunk = 0;
                            group = Some(owner_type);
                        }
                        chunk += 1;
                        seg.push(&payload, None)?;
                    }
                    Ok((seg.finish()?, pages))
                }
            })
            .collect();
        let outputs = run_tasks(workers, tasks)?;
        let mut segs = Vec::with_capacity(outputs.len());
        let (mut entries, mut pages) = (0_u64, 0_u64);
        for (seg, seg_pages) in outputs {
            entries += seg.count;
            pages += seg_pages;
            segs.push(seg);
        }
        Ok((segs, entries, pages))
    };
    let (out_segs, out_entries, out_pages) = merge_direction(&s5.out_runs, "out")?;
    let (in_segs, in_entries, in_pages) = merge_direction(&s5.in_runs, "in")?;
    Ok(S6Manifest {
        out_segs,
        in_segs,
        out_entries,
        in_entries,
        out_pages,
        in_pages,
    })
}

/// s7 — the serial D2 materialization, unchanged in behavior: it consumes
/// concatenated streams whose content is worker-count-invariant, so the
/// produced generation bytes are too. Always re-run in full on resume
/// (partial store files are wiped by the caller).
#[allow(clippy::too_many_arguments)]
fn materialize(
    building: &Path,
    scratch: &Path,
    tenant: TenantId,
    frontier: LoaderMigrationFrontier,
    blob: &Arc<BlobStore>,
    budgets: &OwnerBulkBudgets,
    s3: &S3Manifest,
    s5: &S5Manifest,
    s6: &S6Manifest,
    report: &mut LoadReport,
) -> Result<()> {
    // Durable bootstrap always opens the DEFAULT extent set as the
    // catalog's production substrate, so the fresh generation carries an
    // explicit empty DEFAULT set alongside the loaded tenant.
    FreshV6Builder::create(
        building,
        TenantId::DEFAULT,
        frontier.migration_lsn(),
        Arc::clone(blob),
    )?
    .finish()?;
    let mut builder = FreshV6Builder::create_with_budgets(
        building,
        tenant,
        frontier.migration_lsn(),
        Arc::clone(blob),
        Some(budgets),
    )?;

    // TEL first: both directions land in STORE_TEL before record placement
    // so each node's chain-head refs are known at its record write.
    let refs_dir = scratch.join("s7");
    if refs_dir.exists() {
        fs::remove_dir_all(&refs_dir)?;
    }
    fs::create_dir_all(&refs_dir)?;
    let out_refs = refs_dir.join("tel.out.refs");
    let in_refs = refs_dir.join("tel.in.refs");
    if ship_empty_tel() {
        // RED-on-revert seam (INV-M5.20/.17), unchanged from D2.
        sync_writer(&mut BufWriter::new(create_new(&out_refs)?))?;
        sync_writer(&mut BufWriter::new(create_new(&in_refs)?))?;
    } else {
        let mut out_stream = SegmentedReader::open(scratch, &s6.out_segs);
        build_tel_direction(
            &mut builder,
            FreshTelDirection::Out,
            &mut out_stream,
            &out_refs,
        )?;
        let mut in_stream = SegmentedReader::open(scratch, &s6.in_segs);
        build_tel_direction(
            &mut builder,
            FreshTelDirection::In,
            &mut in_stream,
            &in_refs,
        )?;
    }

    // Id-ordered node placement: concatenated node-artifact segments merged
    // with the two id-ordered TEL head-ref spools.
    {
        let node_segs: Vec<SegEntry> = s3.ranges.iter().map(|range| range.nodes.clone()).collect();
        let mut nodes = SegmentedReader::open(scratch, &node_segs);
        let mut out_ref_reader = TelRefReader::open(&out_refs)?;
        let mut in_ref_reader = TelRefReader::open(&in_refs)?;
        while let Some(payload) = nodes.next()? {
            let artifact = decode_node_artifact(&payload)?;
            let external = std::str::from_utf8(&artifact.external_id)
                .context("production binding requires UTF-8 external node id")?;
            let bag = materialized_bag(artifact.float_bits, &artifact.opaque);
            let out_tel_ref = out_ref_reader.take(artifact.internal_id)?;
            let in_tel_ref = in_ref_reader.take(artifact.internal_id)?;
            builder.push_node(FreshNode {
                id: artifact.internal_id,
                label: artifact.label,
                external_id: external,
                bag: &bag,
                out_tel_ref,
                in_tel_ref,
            })?;
        }
        out_ref_reader.finish()?;
        in_ref_reader.finish()?;
    }

    // Id-ordered relationship placement with properties.
    {
        let mut rels = SegmentedReader::open(scratch, &s5.rel_segs);
        while let Some(payload) = rels.next()? {
            let resolved = decode_resolved_rel(&payload)?;
            let external = std::str::from_utf8(&resolved.external_id)
                .context("production binding requires UTF-8 external relationship id")?;
            let bag = materialized_bag(resolved.float_bits, &resolved.opaque);
            builder.push_relationship(FreshRel {
                id: resolved.internal_id,
                type_id: resolved.type_id,
                source_id: resolved.source_id,
                target_id: resolved.target_id,
                external_id: external,
                bag: &bag,
            })?;
        }
    }
    report.out_tel_entries = builder.out_tel_entries;
    report.in_tel_entries = builder.in_tel_entries;
    report.chained_bags = builder.chained_bags;
    let base = builder.finish()?;
    report.prop_pages = base.prop_pages;
    ensure!(
        base.nodes == report.nodes && base.rels == report.relationships,
        "production v6 base census differs from loader artifacts"
    );
    ensure!(
        ship_empty_tel()
            || report.out_tel_entries == report.relationships
                && report.in_tel_entries == report.relationships,
        "TEL entry census differs from materialized relationships"
    );
    Ok(())
}

/// Stream one owner-sorted TEL entry stream into the production builder,
/// spooling `(owner, head_page)` pairs in owner order for the id-ordered
/// record placement merge.
fn build_tel_direction(
    builder: &mut FreshV6Builder,
    direction: FreshTelDirection,
    stream: &mut SegmentedReader,
    refs_out: &Path,
) -> Result<()> {
    let mut writer = BufWriter::new(create_new(refs_out)?);
    let mut current: Option<u64> = None;
    while let Some(payload) = stream.next()? {
        let (owner, type_id, neighbor, rel_id) = decode_tel_run(&payload)?;
        if current != Some(owner) {
            if let Some(finished) = current {
                let head = builder.finish_tel_chain()?;
                write_tel_ref(&mut writer, finished, head)?;
            }
            builder.begin_tel_chain(direction, owner)?;
            current = Some(owner);
        }
        builder.append_tel_entry(type_id, neighbor, rel_id)?;
    }
    if let Some(finished) = current {
        let head = builder.finish_tel_chain()?;
        write_tel_ref(&mut writer, finished, head)?;
    }
    sync_writer(&mut writer)?;
    Ok(())
}

fn write_tel_ref(writer: &mut impl Write, owner: u64, head: u64) -> Result<()> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(&owner.to_le_bytes());
    payload.extend_from_slice(&head.to_le_bytes());
    write_framed(writer, &payload)
}

/// Bounded id-ordered reader over one `(owner, head_page)` ref spool.
struct TelRefReader {
    reader: crate::m5_load::FramedReader,
    pending: Option<(u64, u64)>,
}

impl TelRefReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: crate::m5_load::FramedReader::open(path)?,
            pending: None,
        })
    }

    /// The head-page ref for `id`, or 0 when the node has no chain in this
    /// direction. Callers consume ids in ascending order.
    fn take(&mut self, id: u64) -> Result<u64> {
        if self.pending.is_none() {
            self.pending = self
                .reader
                .next()?
                .map(|payload| {
                    let mut cursor = 0;
                    let owner = take_u64(&payload, &mut cursor, "TEL ref owner")?;
                    let head = take_u64(&payload, &mut cursor, "TEL ref head")?;
                    ensure!(cursor == payload.len(), "TEL ref has trailing bytes");
                    Ok((owner, head))
                })
                .transpose()?;
        }
        match self.pending {
            Some((owner, head)) if owner == id => {
                self.pending = None;
                Ok(head)
            }
            Some((owner, _)) if owner < id => bail!(
                "TEL chain owner {owner} has no node record (id-ordered merge desynchronized)"
            ),
            _ => Ok(0),
        }
    }

    /// Every spooled ref must have been consumed by a node record.
    fn finish(mut self) -> Result<()> {
        ensure!(
            self.pending.is_none() && self.reader.next()?.is_none(),
            "TEL ref spool has chains for nonexistent nodes"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_set_writer_spills_and_range_merge_reproduces_key_order() {
        let scratch = tempfile::tempdir().expect("scratch");
        // 4 KiB budget forces multiple spilled runs from ~300 records.
        let mut writer = RunSetWriter::new(scratch.path(), "s1/w0", 4 * 1024, 4).expect("writer");
        for value in (0..300_u32).rev() {
            writer
                .push(
                    value.to_be_bytes().to_vec(),
                    format!("payload-{value}").into_bytes(),
                )
                .expect("push");
        }
        let runs = writer.finish().expect("finish");
        assert!(runs.len() <= 4, "worker compacts to its fan-in share");
        // Full-range merge reproduces the exact serial order.
        let mut merge =
            RangeMerge::open(scratch.path(), &runs, None, None, 1 << 20).expect("merge");
        let mut seen = Vec::new();
        while let Some((key, _payload)) = merge.next_item().expect("next") {
            seen.push(u32::from_be_bytes(key.try_into().expect("4-byte key")));
        }
        assert_eq!(seen, (0..300_u32).collect::<Vec<_>>());
    }

    #[test]
    fn range_merge_respects_half_open_bounds_and_covers_key_space() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut writer = RunSetWriter::new(scratch.path(), "s1/w0", 1024, 8).expect("writer");
        for value in 0..500_u32 {
            writer
                .push(value.to_be_bytes().to_vec(), vec![0])
                .expect("push");
        }
        let runs = writer.finish().expect("finish");
        let splitters = choose_splitters(scratch.path(), &runs, 4).expect("splitters");
        let bounds = splitter_bounds(&splitters);
        let mut total = Vec::new();
        for (start, end) in &bounds {
            let mut merge = RangeMerge::open(
                scratch.path(),
                &runs,
                start.as_deref(),
                end.as_deref(),
                1 << 20,
            )
            .expect("merge");
            while let Some((key, _)) = merge.next_item().expect("next") {
                if let Some(start) = start {
                    assert!(key.as_slice() >= start.as_slice(), "below range start");
                }
                if let Some(end) = end {
                    assert!(key.as_slice() < end.as_slice(), "at/after range end");
                }
                total.push(u32::from_be_bytes(key.try_into().expect("4-byte key")));
            }
        }
        // Disjoint + covering: the concatenation is the exact serial sort.
        assert_eq!(total, (0..500_u32).collect::<Vec<_>>());
    }

    #[test]
    fn id_bases_are_prefix_sums_in_range_order() {
        let bases = id_bases(&[(3, 1), (0, 4), (2, 2)]);
        assert_eq!(bases, vec![(0, 0), (3, 1), (3, 5)]);
    }
}
