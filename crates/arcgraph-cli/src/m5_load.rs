//! M5 leg-(c) — offline bootstrap-load into a virgin data dir.
//!
//! Design authority: `docs/design/M5D-REDESIGN-AMENDMENT.md` §2 (the M5-D1
//! attach spine), amending `docs/design/m1-m2-m4-m5-impl-designs.md`
//! §M5.1/§M5.2/§M5.4. The leg implements the ADR-230 M5 sentence — "loader
//! writes outside the visible catalog; attach = one root swap after
//! checkpoint — crash-before-attach is a no-op" — literally:
//!
//! 1. **`DataDirLock` first, held through commit.** Durable servers hold the
//!    same lock (`bootstrap.rs` §1), so load-vs-serve is mutually exclusive
//!    by construction. The populated-dir refusal happens with ZERO mutation
//!    (checked read-only BEFORE lock-file creation, re-checked authoritative
//!    under the lock — TOCTOU-closed).
//! 2. **Virgin-or-resumable-owned precondition (INV-M5.21).** Any entry
//!    outside the loader's own generation namespace refuses with the typed
//!    `LoadRefusal::PopulatedDataDir`, naming the supported alternatives.
//! 3. **Generation namespace (INV-M5.22):** the loader owns exactly
//!    `gen-load-v6[.building]` per [`crate::generation_namespace`]; scratch
//!    lives INSIDE the building generation so a pre-commit crash orphan is
//!    one sweepable directory and the data-dir root is never littered.
//! 4. **ONE commit object:** the landed offline `CURRENT`
//!    temp-write/fsync/rename/parent-fsync ritual, REUSED from
//!    `crate::data_dir_migration::GenerationCommit` — not forked. The
//!    fabricated `M5_TENANTS` marker, the `AGM5CAT1` record typed
//!    `WalRecordType::Checkpoint`, and the fake `m2_typed` manifest prior of
//!    the superseded PR #1504 do not exist here: the census travels in
//!    `MANIFEST.tenant_census` + the ADR-207 catalog root page
//!    ([`arcgraph_storage::catalog::CATALOG_PAGE_ID`]) inside the built
//!    generation, so a cold open treats the loaded store exactly like any
//!    other durable v6 store (amendment §2.6; the risk-2 fallback of census
//!    registration THROUGH a bespoke bootstrap hook is deliberately absent).
//! 5. **`LoadFault` (INV-M5.23):** the `MigrationFault`-analog five-point
//!    kill-9 table; every fault point reruns to completion or an
//!    `LoadOutcome::AlreadyLoaded` no-op — never `EEXIST`.
//!
//! 6. **Served-store completeness (M5-D2, amendment §3, INV-M5.20):** the
//!    served generation carries EVERYTHING the input did — records with
//!    populated `property_ref`, STORE_PROPS bag pages, BOTH TEL directions
//!    in STORE_TEL with `out_tel_ref`/`in_tel_ref` stamped into node
//!    records, and oversized bags chained through the production DEC-4
//!    blob path into the first checkpoint's page-images (the landed blob
//!    layout). Float bits and opaque payloads are NEVER discarded nor
//!    transcoded (`canonical_property_bag`; INV-M5.12's oracle
//!    terminates at the SERVED store). All pipeline intermediates are
//!    deleted before the durability ledger — nothing outside the store
//!    formats survives into the served generation.
//!
//! 7. **Parallel decomposition (M5-D3, amendment §4):** the pipeline
//!    stages run through [`crate::m5_parallel`] — byte-range input
//!    partitioning, per-worker run generation, range-partitioned merge,
//!    and two-phase dense-id assignment. Output is BYTE-IDENTICAL for any
//!    worker count (INV-M5.24). Owner-substrate disk caps on this bulk
//!    path derive from the pass-1 census (INV-M5.25, `plan_owner_budgets`)
//!    with a plan-time disk projection that refuses BEFORE building
//!    (`project_disk_or_refuse`); stage manifests make a crashed build
//!    resume from its last durable stage.
//!
//! Budget (performance-budget discipline): per-worker 256 MiB sort buffers ×
//! `min(physical_cores, 32)` workers, ≤64-way total merge fan-in with
//! 1 MiB read buffers, O(1) resident per stage; materialization holds one
//! open bag page + one page-grain TEL block; oversized-bag chain pages
//! stay resident until the first checkpoint captures them — bounded by
//! the input's oversized-payload volume, the same residency class as the
//! live engine's blob tier. Amendment §4.3 closes ≥250K nodes/s @ 100M at
//! W=16 on ≥2.5 GB/s scratch. Input formats: native only here; the
//! Parquet boundary lands with its own fuzz target per §M5.1.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use arcgraph_core::{DurabilityTier, Lsn, TenantId};
use arcgraph_storage::owner_budget::{BulkClassCensus, OwnerBulkBudgets, OwnerSubstrateBudget};
use serde_json::Value;

use crate::m5_parallel::{LoadCensus, default_workers};

use crate::data_dir_migration::{
    CURRENT_FILE, CURRENT_TMP, GenerationCommit, LSN_SEED_FILE, MigrationFault, MigrationOutcome,
    complete_generation_ledger, complete_index_vector_passes, file_sha256, inject,
    production_index_vector_fault, resume_after_m4_swap, stamp_generation_version,
    v6_generation_tenants, verify_v6_generation, write_current_atomic, write_synced,
};
use crate::data_lock::DataDirLock;
use crate::generation_namespace::GenerationTool;
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::catalog::{
    CATALOG_PAGE_ID, TenantRecord, decode_catalog_page, encode_catalog_page,
};
use arcgraph_storage::io::{PageBuf, PageIo, PosixPageIo};
use arcgraph_storage::m4_migration::{LoaderMigrationFrontier, establish_fresh_v6_checkpoint};
use arcgraph_storage::manifest::{DataDirManifest, now_rfc3339_utc};
use arcgraph_storage::wal::fsync_dir;
use arcgraph_storage::{DATA_DIR_VERSION_DIRECT_M4, write_data_dir_manifest};

/// Maximum encoded native record. Applied before JSON allocation.
pub const MAX_NATIVE_RECORD_BYTES: usize = 8 * 1024 * 1024;
/// Maximum native JSON container depth. Production records are flat; the
/// margin permits compatible envelopes without accepting recursive bombs.
pub const MAX_NATIVE_RECURSION: usize = 16;
/// Maximum external identifier size in bytes.
pub const MAX_EXTERNAL_ID_BYTES: usize = 16 * 1024;
/// Maximum opaque property payload in one input record.
pub const MAX_OPAQUE_BYTES: usize = 4 * 1024 * 1024;
/// Default PER-WORKER in-memory run buffer (amendment §4.2 mechanism 2).
/// Independent of input cardinality; `W × 256 MiB` stays inside the §M5.1
/// sort plateau at every rung.
pub const DEFAULT_SORT_MEMORY_BYTES: usize = 256 * 1024 * 1024;
/// Maximum merge readers open at once (total across one merge).
pub const MAX_MERGE_FAN_IN: usize = 64;
/// Hard M5 100M-rung continuous-RSS ceiling (§M5-D3 gate table).
pub const M5_RSS_CAP_BYTES: u64 = 40 * 1024 * 1024 * 1024;

/// INV-M5.2 — the loader's LSN frontier convention for a store born by leg
/// (c). Every loader-built page stamps `page_lsn = migration_lsn`, and
/// `LSN_SEED = migration_lsn + 1` is derived through the SAME
/// [`LoaderMigrationFrontier`] type the migration legs use (never a magic
/// constant — amendment §2.6). `Lsn(1)` is the smallest frontier the type
/// admits; a virgin dir has no earlier history, so the engine clock is never
/// re-based and redo can never be suppressed.
pub const FRESH_LOAD_MIGRATION_LSN: Lsn = Lsn::new(1);

/// Production input formats accepted by `arcgraph load`. Parquet is
/// deliberately absent until its boundary lands WITH its fuzz target and
/// caps (design §M5.1; amendment §9 keeps it with the INV-M5.16 gates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadFormat {
    Native,
}

/// Resource envelope for one bulk load (M5-D3). The defaults are the
/// production rung shape; gates shrink the sort budget to force spills at
/// CI fixture sizes.
#[derive(Debug, Clone, Copy)]
pub struct LoadLimits {
    /// Worker count; `None` = `min(physical_cores, 32)` (amendment §4.2).
    pub workers: Option<usize>,
    /// Per-worker in-memory sort buffer.
    pub sort_memory_bytes: usize,
    /// Continuous resident-set ceiling — every sample must sit below it.
    pub rss_cap_bytes: u64,
    /// Sampling cadence of the dedicated RSS thread.
    pub rss_sample_every_ms: u64,
    /// Operator disk override: caps the plan-time projection's available
    /// bytes below the filesystem's free space (`--max-disk` analog).
    pub max_disk_bytes: Option<u64>,
}

impl LoadLimits {
    #[must_use]
    pub const fn production() -> Self {
        Self {
            workers: None,
            sort_memory_bytes: DEFAULT_SORT_MEMORY_BYTES,
            rss_cap_bytes: M5_RSS_CAP_BYTES,
            rss_sample_every_ms: 100,
            max_disk_bytes: None,
        }
    }

    /// Worker count, clamped to [`MAX_MERGE_FAN_IN`]: every worker
    /// contributes at least one run to the shared range merges, so W is
    /// structurally bounded by the total merge fan-in (amendment §4.2:
    /// `min(cores, 64)` is the 1B-rung ceiling for the same reason).
    #[must_use]
    pub fn effective_workers(&self) -> usize {
        self.workers
            .unwrap_or_else(default_workers)
            .clamp(1, MAX_MERGE_FAN_IN)
    }

    fn validate(self) -> Result<Self> {
        ensure!(
            self.sort_memory_bytes >= 64 * 1024,
            "sort memory budget is below 64 KiB"
        );
        ensure!(self.rss_cap_bytes > 0, "RSS cap must be non-zero");
        ensure!(
            self.rss_sample_every_ms > 0,
            "RSS sample cadence must be non-zero"
        );
        Ok(self)
    }
}

impl Default for LoadLimits {
    fn default() -> Self {
        Self::production()
    }
}

/// One continuously-enforced resident-set sample (INV-M5.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RssSample {
    /// Milliseconds since load start.
    pub at_ms: u64,
    /// Resident bytes at the sample instant.
    pub rss_bytes: u64,
    /// Pipeline stage active when sampled.
    pub stage: &'static str,
}

/// Deterministic leg-(c) crash points (amendment §2.5) — the
/// [`MigrationFault`]-analog table. Production uses [`Self::None`]; the
/// commit-phase points inject inside the SHARED [`GenerationCommit`] via
/// [`migration_fault_analog`], so the loader and the migrate tool exercise
/// literally the same machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum LoadFault {
    None,
    AfterScratchCreate,
    AfterBuildSync,
    AfterGenerationRename,
    AfterCurrentSwap,
    AfterVersionStamp,
    /// INV-M5.6 negative control: hand the shared publication object a false
    /// durability proof. Release-lane gate only.
    #[cfg(any(test, feature = "fault-injection"))]
    MissingLoadLedgerProof,
}

/// Map the leg-(c) fault points that live inside the shared commit object
/// onto the [`MigrationFault`] injection machinery (reuse, not fork).
fn migration_fault_analog(fault: LoadFault) -> MigrationFault {
    match fault {
        LoadFault::AfterGenerationRename => MigrationFault::AfterGenerationRename,
        LoadFault::AfterCurrentSwap => MigrationFault::AfterCurrentSwap,
        LoadFault::AfterVersionStamp => MigrationFault::AfterVersionStamp,
        _ => MigrationFault::None,
    }
}

fn inject_load(selected: LoadFault, point: LoadFault) -> Result<()> {
    if selected == point {
        bail!("injected load crash at {point:?}");
    }
    Ok(())
}

/// Release-lane production fault selector (the leg-(b)
/// `production_migration_fault` analog). Unset env ⇒ no injection.
fn production_load_fault() -> LoadFault {
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_LOAD_MISSING_LEDGER").is_some() {
        return LoadFault::MissingLoadLedgerProof;
    }
    LoadFault::None
}

/// INV-M5.21 — typed refusal for the attach-leg identity precondition.
///
/// The virgin-or-resumable-owned check refuses with ZERO directory mutation.
/// `#[non_exhaustive]` per the code-quality policy error-enum convention.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoadRefusal {
    /// The target dir carries state outside the loader's own namespace.
    #[error(
        "data dir {data_dir} is populated (found: {offending:?}); `arcgraph load` only \
         bootstraps a VIRGIN data dir (leg (c), M5-D1). Supported alternatives: \
         `arcgraph migrate upgrade-data-dir` rewrites an existing store in place (leg (b)); \
         live fresh-tenant attach into a running server is leg (a), deferred to slice M5-F \
         and refused until it exists. No file in {data_dir} was created, modified, or removed."
    )]
    PopulatedDataDir {
        /// The refused target directory.
        data_dir: PathBuf,
        /// Root entries outside the loader namespace (diagnostic).
        offending: Vec<String>,
    },
    /// The plan-time disk projection (M5-D3, amendment §5, INV-M5.25)
    /// refused BEFORE building: the census-derived substrate, generation,
    /// and scratch need exceeds the available bytes. Fail-fast, not
    /// fail-at-hour-3 — a mid-build `DiskBudgetExceeded` on well-formed
    /// input is a projection bug, never an accepted outcome.
    #[error(
        "projected disk need {required_bytes} B exceeds available {available_bytes} B \
         for data dir {data_dir}; refusing before build (fail-fast).\n{table}"
    )]
    ProjectedDiskExceeded {
        /// The refused target directory.
        data_dir: PathBuf,
        /// Projected bytes the load would write.
        required_bytes: u64,
        /// Free bytes on the target filesystem (after any operator cap).
        available_bytes: u64,
        /// Human-readable projection table (per-component rows).
        table: String,
    },
}

/// Result of one `arcgraph load` invocation (INV-M5.23: rerun after any
/// crash point completes the load or lands here as a no-op — never `EEXIST`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadOutcome {
    /// The generation was built (or a crashed run was resumed) and committed.
    Loaded(LoadReport),
    /// The dir already carries a committed fresh-load generation; the rerun
    /// validated it and changed nothing (idempotent; exit 0 with a census).
    AlreadyLoaded {
        /// Sorted tenant census of the committed generation.
        tenant_census: Vec<u64>,
    },
}

/// Census of one completed load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Input records parsed (nodes + relationships).
    pub records: u64,
    /// Nodes materialized.
    pub nodes: u64,
    /// Relationships materialized.
    pub relationships: u64,
    /// Shared slotted property-bag pages written into STORE_PROPS
    /// extents (INV-M5.20).
    pub prop_pages: u64,
    /// Oversized property bags chained through the production DEC-4
    /// blob path (first-checkpoint page-images).
    pub chained_bags: u64,
    /// Outgoing TEL entries written into STORE_TEL extents.
    pub out_tel_entries: u64,
    /// Incoming TEL entries written into STORE_TEL extents.
    pub in_tel_entries: u64,
    /// True when this invocation resumed a crashed predecessor's committed
    /// generation instead of building from input.
    pub resumed: bool,
    /// Continuous RSS samples (grows with wall time, never row count).
    pub rss_samples: Vec<RssSample>,
    /// Worker count the pipeline ran with.
    pub workers: u64,
    /// Wall-clock milliseconds for the pipeline (parse → materialize).
    pub elapsed_ms: u64,
    /// Stages skipped by resuming from durable run manifests (M5-D3).
    pub resumed_stages: Vec<String>,
}

/// Canonical crud-grain property payload for one loaded record
/// (M5-D2, amendment §3.2 / INV-M5.12): the raw IEEE-754 bit pattern
/// (8 bytes LE) followed by the opaque payload VERBATIM — never
/// transcoded, so the served store can return floats bit-exactly and
/// opaque embedder payloads byte-exactly (memory: serde_json's default
/// float parse is ULP-lossy; the loader therefore never routes these
/// bytes through a decimal representation). This is the exact payload
/// an equivalent incremental ingest persists as
/// `PropertyData::Blob(canonical_property_bag(..))` — the disk
/// differential's "same logical content" definition.
#[must_use]
pub fn canonical_property_bag(float_bits: u64, opaque: &[u8]) -> Vec<u8> {
    let mut bag = Vec::with_capacity(8 + opaque.len());
    bag.extend_from_slice(&float_bits.to_le_bytes());
    bag.extend_from_slice(opaque);
    bag
}

/// One parser output at the storage grain. Float values remain raw IEEE bits;
/// opaque bytes and external identifiers are never transcoded. `float_bits`/
/// `opaque` travel end-to-end into the served generation as the record's
/// [`canonical_property_bag`] (M5-D2, INV-M5.20); INV-M5.12's fidelity
/// oracle terminates at the SERVED store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadRecord {
    Node {
        external_id: Vec<u8>,
        label: u32,
        float_bits: u64,
        opaque: Vec<u8>,
    },
    Relationship {
        external_id: Vec<u8>,
        source_external_id: Vec<u8>,
        target_external_id: Vec<u8>,
        type_id: u32,
        float_bits: u64,
        opaque: Vec<u8>,
    },
}

impl LoadRecord {
    #[must_use]
    pub fn external_id(&self) -> &[u8] {
        match self {
            Self::Node { external_id, .. } | Self::Relationship { external_id, .. } => external_id,
        }
    }
}

/// Streaming parser contract shared by the production input boundaries.
pub trait LoadRecordSource {
    fn next_record(&mut self) -> Result<Option<LoadRecord>>;
}

pub fn open_record_source(path: &Path, format: LoadFormat) -> Result<Box<dyn LoadRecordSource>> {
    match format {
        LoadFormat::Native => Ok(Box::new(NativeRecordSource::open(path)?)),
    }
}

pub(crate) struct NativeRecordSource {
    reader: BufReader<File>,
    line: u64,
    /// Absolute byte offset of the next unread input byte.
    pos: u64,
    /// Frame-boundary partition end (mechanism 1): a record is owned by
    /// this source iff its FIRST byte lies in `[start, end)`; the last
    /// owned record may extend past `end`.
    end: u64,
}

impl NativeRecordSource {
    fn open(path: &Path) -> Result<Self> {
        Self::open_range(path, 0, u64::MAX)
    }

    /// Byte-range partition resynced to the newline frame boundary
    /// (amendment §4.2 mechanism 1). serde_json escapes newlines inside
    /// strings, so `\n` is an unambiguous frame delimiter.
    pub(crate) fn open_range(path: &Path, start: u64, end: u64) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("open native input {}", path.display()))?;
        let len = file.metadata()?.len();
        let end = end.min(len);
        let mut pos = start.min(len);
        if start > 0 && start < len {
            // Seek to start-1 and discard through the next newline: if the
            // byte at start-1 IS the newline, the record starting exactly
            // at `start` is preserved; otherwise the partial record that
            // began before `start` (owned by the previous partition) is
            // consumed.
            file.seek(SeekFrom::Start(start - 1))?;
            let mut reader = BufReader::new(file);
            let skipped = read_bounded_line(&mut reader, MAX_NATIVE_RECORD_BYTES + 1)?
                .map_or(0, |line| line.len() as u64);
            pos = start - 1 + skipped;
            return Ok(Self {
                reader,
                line: 0,
                pos,
                end,
            });
        }
        file.seek(SeekFrom::Start(pos))?;
        Ok(Self {
            reader: BufReader::new(file),
            line: 0,
            pos,
            end,
        })
    }
}

/// In-memory entrypoint used by the native libFuzzer target. The production
/// size and recursion caps run before serde_json can allocate a value tree.
pub fn fuzz_native_record_boundary(bytes: &[u8]) -> Result<Option<LoadRecord>> {
    ensure!(
        bytes.len() <= MAX_NATIVE_RECORD_BYTES,
        "native record exceeds {MAX_NATIVE_RECORD_BYTES} bytes"
    );
    let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    ensure_json_depth(bytes, MAX_NATIVE_RECURSION)?;
    let value: Value = serde_json::from_slice(bytes).context("parse native fuzz record")?;
    parse_native_value(&value).map(Some)
}

impl LoadRecordSource for NativeRecordSource {
    fn next_record(&mut self) -> Result<Option<LoadRecord>> {
        loop {
            if self.pos >= self.end {
                return Ok(None);
            }
            let Some(line) = read_bounded_line(&mut self.reader, MAX_NATIVE_RECORD_BYTES)? else {
                return Ok(None);
            };
            self.pos += line.len() as u64;
            self.line += 1;
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            ensure_json_depth(line, MAX_NATIVE_RECURSION)
                .with_context(|| format!("native line {} recursion cap", self.line))?;
            let value: Value = serde_json::from_slice(line)
                .with_context(|| format!("parse native line {}", self.line))?;
            return parse_native_value(&value)
                .with_context(|| format!("validate native line {}", self.line))
                .map(Some);
        }
    }
}

fn read_bounded_line(reader: &mut impl BufRead, cap: usize) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().context("read native input")?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |idx| idx + 1);
        ensure!(
            line.len().saturating_add(take) <= cap,
            "native record exceeds {cap} bytes"
        );
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            return Ok(Some(line));
        }
    }
}

fn ensure_json_depth(bytes: &[u8], cap: usize) -> Result<()> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.checked_add(1).context("native recursion overflow")?;
                ensure!(depth <= cap, "native JSON nesting exceeds {cap}");
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

fn parse_native_value(value: &Value) -> Result<LoadRecord> {
    let object = value.as_object().context("record must be a JSON object")?;
    let kind = required_str(object.get("kind"), "kind")?;
    let external_id = decode_hex_capped(
        required_str(object.get("external_id"), "external_id")?,
        MAX_EXTERNAL_ID_BYTES,
        "external_id",
    )?;
    ensure!(!external_id.is_empty(), "external_id must not be empty");
    let label_or_type = required_u32(object.get("label_or_type"), "label_or_type")?;
    let float_bits = parse_float_bits(required_str(object.get("float_bits"), "float_bits")?)?;
    let opaque = decode_hex_capped(
        required_str(object.get("opaque"), "opaque")?,
        MAX_OPAQUE_BYTES,
        "opaque",
    )?;
    match kind {
        "node" => Ok(LoadRecord::Node {
            external_id,
            label: label_or_type,
            float_bits,
            opaque,
        }),
        "relationship" => {
            let source_external_id = decode_hex_capped(
                required_str(object.get("source_id"), "source_id")?,
                MAX_EXTERNAL_ID_BYTES,
                "source_id",
            )?;
            let target_external_id = decode_hex_capped(
                required_str(object.get("target_id"), "target_id")?,
                MAX_EXTERNAL_ID_BYTES,
                "target_id",
            )?;
            ensure!(
                !source_external_id.is_empty() && !target_external_id.is_empty(),
                "relationship endpoints must not be empty"
            );
            Ok(LoadRecord::Relationship {
                external_id,
                source_external_id,
                target_external_id,
                type_id: label_or_type,
                float_bits,
                opaque,
            })
        }
        other => bail!("unsupported record kind {other:?}"),
    }
}

fn required_str<'a>(value: Option<&'a Value>, name: &str) -> Result<&'a str> {
    value
        .and_then(Value::as_str)
        .with_context(|| format!("{name} must be a string"))
}

fn required_u32(value: Option<&Value>, name: &str) -> Result<u32> {
    let raw = value
        .and_then(Value::as_u64)
        .with_context(|| format!("{name} must be an unsigned integer"))?;
    u32::try_from(raw).with_context(|| format!("{name} exceeds u32"))
}

fn parse_float_bits(value: &str) -> Result<u64> {
    ensure!(
        value.len() == 16,
        "float_bits must contain exactly 16 hex digits"
    );
    u64::from_str_radix(value, 16).context("float_bits is not hexadecimal")
}

fn decode_hex_capped(value: &str, cap: usize, field: &str) -> Result<Vec<u8>> {
    ensure!(value.len() % 2 == 0, "{field} hex length must be even");
    ensure!(value.len() / 2 <= cap, "{field} exceeds {cap} bytes");
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0]).with_context(|| format!("{field} has non-hex byte"))?;
        let low = hex_nibble(pair[1]).with_context(|| format!("{field} has non-hex byte"))?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

fn hex_nibble(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => bail!("not hexadecimal"),
    }
}

// ---------------------------------------------------------------------------
// Leg-(c) attach protocol
// ---------------------------------------------------------------------------

/// Classification of the target dir under the §2.3 virgin-or-resumable-owned
/// precondition. Derivable read-only; every variant maps to exactly one row
/// of the §2.5 restart matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LoadDirState {
    /// Dir absent or empty of everything but the protocol-neutral `LOCK`.
    Virgin,
    /// Only a stale `gen-load-v6.building` orphan (any pre-commit crash):
    /// sweep own prefix, rebuild from input.
    StaleBuilding,
    /// Complete `gen-load-v6`, no `CURRENT` (crash at `AfterGenerationRename`):
    /// validate, then complete `CURRENT` → `VERSION`.
    CommittedUnselected,
    /// `CURRENT` names the load generation, `VERSION` absent (crash at
    /// `AfterCurrentSwap`): validate, stamp `VERSION` LAST.
    SelectedUnstamped,
    /// Fully committed (`CURRENT` + `VERSION`): idempotent no-op.
    AlreadyLoaded,
}

/// Read-only §2.3 precondition. Every root entry must be loader-owned or
/// protocol-neutral; anything else is a typed refusal that provably mutated
/// nothing (the gate compares the dir tree byte-for-byte after refusal).
fn classify_load_dir(root: &Path) -> Result<LoadDirState> {
    let building_name = GenerationTool::M5Load.building_dir();
    let final_name = GenerationTool::M5Load.final_dir();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(LoadDirState::Virgin);
        }
        Err(error) => {
            return Err(error).with_context(|| format!("enumerate data dir {}", root.display()));
        }
    };
    let mut offending = Vec::new();
    let mut has_building = false;
    let mut has_final = false;
    let mut has_current = false;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            offending.push(entry.file_name().to_string_lossy().into_owned());
            continue;
        };
        match name {
            // The advisory lockfile is protocol-neutral: the loader itself
            // creates it on its first run, so every resume scenario carries
            // it, and it holds no store state.
            crate::data_lock::LOCK_FILE => {}
            // A stale `.CURRENT.tmp` is the shared ritual's own mid-swap
            // artifact; `write_current_atomic` removes it before use.
            CURRENT_TMP => {}
            CURRENT_FILE => {
                let bytes = fs::read(root.join(CURRENT_FILE)).context("read CURRENT")?;
                let value = String::from_utf8_lossy(&bytes);
                let value = value.trim_end_matches(['\r', '\n']);
                if value == final_name {
                    has_current = true;
                } else {
                    offending.push(format!("{CURRENT_FILE} -> {value}"));
                }
            }
            name if name == building_name => has_building = true,
            name if name == final_name => has_final = true,
            other => offending.push(other.to_owned()),
        }
    }
    if !offending.is_empty() {
        offending.sort();
        return Err(LoadRefusal::PopulatedDataDir {
            data_dir: root.to_path_buf(),
            offending,
        }
        .into());
    }
    if has_current {
        ensure!(
            has_final,
            "CURRENT names the fresh-load generation but {final_name} is absent — \
             refusing a corrupt commit state (fail closed)"
        );
        let stamped = arcgraph_storage::version_file_path(&root.join(final_name)).is_file();
        return Ok(if stamped {
            LoadDirState::AlreadyLoaded
        } else {
            LoadDirState::SelectedUnstamped
        });
    }
    if has_final {
        return Ok(LoadDirState::CommittedUnselected);
    }
    if has_building {
        return Ok(LoadDirState::StaleBuilding);
    }
    Ok(LoadDirState::Virgin)
}

/// Production entry point for `arcgraph load` (leg (c)).
pub fn load_data_dir(
    input: &Path,
    format: LoadFormat,
    root: &Path,
    tenant: TenantId,
) -> Result<LoadOutcome> {
    load_data_dir_with_fault(input, format, root, tenant, production_load_fault())
}

/// [`load_data_dir`] with an explicit resource envelope (M5-D3): worker
/// count, per-worker sort budget, continuous RSS cap, disk override.
pub fn load_data_dir_with_limits(
    input: &Path,
    format: LoadFormat,
    root: &Path,
    tenant: TenantId,
    limits: LoadLimits,
) -> Result<LoadOutcome> {
    load_data_dir_with_limits_and_fault(
        input,
        format,
        root,
        tenant,
        limits,
        production_load_fault(),
    )
}

/// [`load_data_dir`] with a deterministic §2.5 crash point selected. The
/// fault parameter is test machinery (`MigrationFault`-analog); production
/// callers pass [`LoadFault::None`] via [`load_data_dir`].
pub fn load_data_dir_with_fault(
    input: &Path,
    format: LoadFormat,
    root: &Path,
    tenant: TenantId,
    fault: LoadFault,
) -> Result<LoadOutcome> {
    load_data_dir_with_limits_and_fault(input, format, root, tenant, LoadLimits::default(), fault)
}

/// Full-parameter entry: explicit limits + deterministic crash point.
pub fn load_data_dir_with_limits_and_fault(
    input: &Path,
    format: LoadFormat,
    root: &Path,
    tenant: TenantId,
    limits: LoadLimits,
    fault: LoadFault,
) -> Result<LoadOutcome> {
    let limits = limits.validate()?;
    ensure!(
        tenant != TenantId::DEFAULT && tenant != TenantId::SYSTEM,
        "M5 production load requires a non-default, non-system tenant"
    );

    // §2.3 — read-only precondition BEFORE any mutation (the lockfile is
    // itself a root entry, so a populated dir must refuse before
    // `DataDirLock::acquire` can create it).
    classify_load_dir(root)?;

    // The lock-acquisition preamble mirrors durable bootstrap exactly:
    // create-if-absent, then take the exclusive advisory lock BEFORE any
    // store mutation, and hold it through the commit (function scope).
    fs::create_dir_all(root).with_context(|| format!("create load data dir {}", root.display()))?;
    let _lock = DataDirLock::acquire(root)?;

    // Authoritative re-classification under the lock (TOCTOU-closed): a
    // concurrent process may have populated the dir between the read-only
    // check and lock acquisition.
    let state = classify_load_dir(root)?;

    let final_generation = root.join(GenerationTool::M5Load.final_dir());
    match state {
        LoadDirState::AlreadyLoaded | LoadDirState::SelectedUnstamped => {
            // §2.5 rows 3–4: reuse the landed post-swap resume (validate the
            // complete generation, stamp VERSION LAST or no-op) — the same
            // code path a migrated generation resumes through.
            let outcome = resume_after_m4_swap(&final_generation, migration_fault_analog(fault))
                .context("resume CURRENT-selected fresh-load generation")?;
            let census = committed_tenant_census(&final_generation, tenant)?;
            match outcome {
                MigrationOutcome::AlreadyUpgraded { .. } => Ok(LoadOutcome::AlreadyLoaded {
                    tenant_census: census,
                }),
                MigrationOutcome::Upgraded { .. } => Ok(LoadOutcome::Loaded(LoadReport {
                    resumed: true,
                    ..LoadReport::default()
                })),
            }
        }
        LoadDirState::CommittedUnselected => {
            // §2.5 row 2: the generation is complete and durable but
            // unpublished. Sweep any own-prefix orphan, validate, then
            // complete the commit tail (CURRENT then VERSION).
            sweep_own_building(root)?;
            resume_unselected_commit(root, &final_generation, tenant, fault)?;
            Ok(LoadOutcome::Loaded(LoadReport {
                resumed: true,
                ..LoadReport::default()
            }))
        }
        LoadDirState::Virgin | LoadDirState::StaleBuilding => {
            // §2.5 row 1, upgraded by M5-D3: a stale own `.building` is no
            // longer swept wholesale — the pipeline resumes from its last
            // durable stage manifest (stage-level restartability); absent
            // or non-matching manifests degrade to the D1 rebuild.
            let report = build_and_commit(input, format, root, tenant, limits, fault)?;
            Ok(LoadOutcome::Loaded(report))
        }
    }
}

/// INV-M5.22 rule 2 — the loader sweeps ONLY its own `.building` prefix.
/// Foreign generations (`gen-v9*`, `gen-v10*`) are unreachable here by
/// construction: the §2.3 precondition already refused any dir containing
/// them, and this helper addresses exactly one registry name.
fn sweep_own_building(root: &Path) -> Result<()> {
    let building = root.join(GenerationTool::M5Load.building_dir());
    if building.exists() {
        fs::remove_dir_all(&building)
            .with_context(|| format!("sweep stale own scratch {}", building.display()))?;
        fsync_dir(root).context("sync own-scratch sweep")?;
    }
    Ok(())
}

/// Validate the committed generation and return its census, checking the
/// requested tenant is a member (a rerun pointed at a different tenant is an
/// operator error, not a no-op).
fn committed_tenant_census(generation: &Path, tenant: TenantId) -> Result<Vec<u64>> {
    let manifest = arcgraph_storage::read_data_dir_manifest(generation)
        .context("read committed fresh-load MANIFEST")?
        .context("committed fresh-load generation has no MANIFEST")?;
    let census = manifest
        .tenant_census
        .context("committed fresh-load MANIFEST is missing its tenant census")?;
    ensure!(
        census.contains(&tenant.raw()),
        "committed fresh-load generation does not contain tenant {} (census: {census:?}); \
         rerun with a census tenant or load into a different virgin dir",
        tenant.raw()
    );
    Ok(census)
}

/// §2.5 row 2 — `AfterGenerationRename` resume: the complete, durable, but
/// unpublished generation finishes its commit through the SAME coordinates
/// the shared ritual uses (CURRENT temp-write/fsync/rename/parent-fsync,
/// then VERSION LAST).
fn resume_unselected_commit(
    root: &Path,
    generation: &Path,
    tenant: TenantId,
    fault: LoadFault,
) -> Result<()> {
    let manifest = arcgraph_storage::read_data_dir_manifest(generation)
        .context("read unselected fresh-load MANIFEST")?
        .context("unselected fresh-load generation has no MANIFEST")?;
    let migration_lsn = Lsn::new(
        manifest
            .migration_lsn
            .context("fresh-load MANIFEST is missing its migration frontier")?,
    );
    let tenants = v6_generation_tenants(generation)
        .context("enumerate unselected fresh-load generation tenants")?;
    // require_unstamped: VERSION before CURRENT is an INV-M5.3 protocol
    // violation, never a resumable state.
    verify_v6_generation(generation, &tenants, migration_lsn, true, true)
        .context("validate unselected fresh-load generation before publish")?;
    committed_tenant_census(generation, tenant)?;
    write_current_atomic(root, GenerationTool::M5Load.final_dir())?;
    inject(
        migration_fault_analog(fault),
        MigrationFault::AfterCurrentSwap,
    )?;
    stamp_generation_version(
        generation,
        DATA_DIR_VERSION_DIRECT_M4,
        migration_fault_analog(fault),
    )
    .context("resume version-last fresh-load stamp")?;
    inject(
        migration_fault_analog(fault),
        MigrationFault::AfterVersionStamp,
    )?;
    Ok(())
}

/// The §2.2 leg-(c) durable preparation + ONE-commit-object publish.
fn build_and_commit(
    input: &Path,
    format: LoadFormat,
    root: &Path,
    tenant: TenantId,
    limits: LoadLimits,
    fault: LoadFault,
) -> Result<LoadReport> {
    let building = root.join(GenerationTool::M5Load.building_dir());
    let final_generation = root.join(GenerationTool::M5Load.final_dir());
    ensure!(
        !final_generation.exists(),
        "fresh build dispatched over a committed load generation (classification bug)"
    );
    if building.exists() {
        // M5-D3 stage-level resume: keep `scratch/` (durable run manifests
        // decide what survives); wipe every partial store artifact so the
        // always-re-run materialization starts clean.
        for entry in fs::read_dir(&building).context("enumerate stale building generation")? {
            let entry = entry?;
            if entry.file_name() != std::ffi::OsStr::new("scratch") {
                let path = entry.path();
                if entry.file_type()?.is_dir() {
                    fs::remove_dir_all(&path)?;
                } else {
                    fs::remove_file(&path)?;
                }
            }
        }
        fsync_dir(&building).context("sync building generation after resume sweep")?;
    } else {
        fs::create_dir(&building)
            .with_context(|| format!("create load scratch generation {}", building.display()))?;
        fsync_dir(root).context("sync data-dir after load scratch creation")?;
    }
    inject_load(fault, LoadFault::AfterScratchCreate)?;

    // §2.4 — every pipeline intermediate lives INSIDE the building
    // generation, and none of it survives into the committed store: the
    // whole subtree is deleted before the durability ledger runs.
    let scratch = building.join("scratch");
    fs::create_dir_all(&scratch).context("create load pipeline scratch")?;

    let frontier = LoaderMigrationFrontier::new(FRESH_LOAD_MIGRATION_LSN)
        .context("bind fresh-load LSN frontier")?;
    // Shared property-bag store (M5-D2): slotted bag pages draw ids from
    // it and stream into STORE_PROPS extents; oversized bags chain
    // through the production DEC-4 path and ride the first checkpoint's
    // metadata as page-images (the landed blob layout).
    let blob = std::sync::Arc::new(BlobStore::new());
    let report = crate::m5_parallel::run_pipeline(
        input, format, &building, root, tenant, frontier, &blob, &limits,
    )?;

    // First checkpoint seed + catalog root page + fresh WAL + LSN_SEED +
    // MANIFEST — the complete §2.2 file set.
    let sidecar = establish_fresh_v6_checkpoint(&building, tenant, frontier.migration_lsn(), &blob)
        .context("establish first fresh-load checkpoint")?;
    let census_records = census_tenant_records(tenant, frontier.migration_lsn());
    write_catalog_root_census(&building, &census_records)?;
    let wal_dir = building.join("wal");
    fs::create_dir(&wal_dir).context("create fresh empty load WAL directory")?;
    fsync_dir(&wal_dir).context("sync fresh empty load WAL directory")?;
    write_synced(
        &building.join(LSN_SEED_FILE),
        &frontier.next_lsn().to_le_bytes(),
    )?;
    let metadata = arcgraph_storage::checkpoint::incremental_metadata_path(
        &building,
        sidecar.checkpoint_lsn,
        sidecar.metadata_generation,
    );
    let metadata_sha256 = file_sha256(&metadata)
        .with_context(|| format!("checksum first load metadata {}", metadata.display()))?;
    let tenant_census: Vec<u64> = census_records
        .iter()
        .map(|record| record.tenant_id.raw())
        .collect();
    write_data_dir_manifest(
        &building,
        &DataDirManifest::fresh_load(
            manifest_timestamp(),
            frontier.migration_lsn(),
            tenant_census,
            metadata_sha256,
        ),
    )
    .context("write fresh-load MANIFEST")?;

    // Nothing outside the store formats survives into the served generation.
    fs::remove_dir_all(&scratch).context("delete pipeline intermediates before ledger")?;

    let tenants = BTreeSet::from([TenantId::DEFAULT, tenant]);
    verify_v6_generation(&building, &tenants, frontier.migration_lsn(), true, true)
        .context("validate complete fresh-load generation before ledger")?;
    let index_vector =
        complete_index_vector_passes(None, &building, &tenants, production_index_vector_fault())?;
    let ledger = complete_generation_ledger(&building)?;
    #[cfg(any(test, feature = "fault-injection"))]
    let ledger = {
        let missing = matches!(fault, LoadFault::MissingLoadLedgerProof);
        crate::data_dir_migration::inject_generation_ledger_fault(
            ledger,
            if missing {
                MigrationFault::MissingGenerationLedgerProof
            } else {
                MigrationFault::None
            },
        )
    };
    inject_load(fault, LoadFault::AfterBuildSync)?;
    GenerationCommit::new(
        root,
        &final_generation,
        GenerationTool::M5Load.final_dir(),
        DATA_DIR_VERSION_DIRECT_M4,
        ledger.with_index_vector_proof(index_vector)?,
    )?
    .commit(migration_fault_analog(fault))
    .context("commit complete fresh-load generation")?;
    Ok(report)
}

/// Provenance timestamp for the fresh-load MANIFEST. The (cfg-gated,
/// bounded) override exists ONLY for the INV-M5.24 byte-identical gate: the
/// timestamp is provenance metadata, not store bytes, and pinning it lets
/// the determinism oracle compare WHOLE generations byte-for-byte with no
/// exclusion list. Production always stamps real time.
fn manifest_timestamp() -> String {
    #[cfg(feature = "fault-injection")]
    if let Ok(pinned) = std::env::var("ARCGRAPH_M5_MANIFEST_TIMESTAMP") {
        return pinned;
    }
    now_rfc3339_utc()
}

// ---------------------------------------------------------------------------
// Census-derived budgets + plan-time disk projection (M5-D3, amendment §5)
// ---------------------------------------------------------------------------

/// Derive the bulk owner-substrate caps from the pass-1 census
/// (INV-M5.25). The fixed `OWNER_*_DISK_CAP_BYTES` constants stop
/// governing the bulk path here; they remain the incremental-path
/// defaults (Director ruling D-5).
///
/// RED-on-revert seams (cfg-gated + bounded):
/// - `ARCGRAPH_M5_FIXED_BULK_CAPS` reinstates the landed fixed constants —
///   the 1B-census plan gate MUST go red under it (the V-2 regression pin).
/// - `ARCGRAPH_M5_ZERO_BULK_BUDGET` collapses the derived caps to 1 byte —
///   proving the derived value is what the substrate ENFORCES (a build
///   under it must fail with `DiskBudgetExceeded`), i.e. the gate
///   exercises the arm production dispatches to, not a parallel constant.
/// - `ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET` (M5-D3 FIX 4, #1518 skeptic
///   review) zeroes ONLY the rel-bindings `payload_cap_bytes` field —
///   node caps and the rel INDEX cap are untouched. This is a
///   field/class-asymmetric probe: a bug that transposes
///   `index_cap_bytes`/`payload_cap_bytes` at the `OwnerWriters::create`
///   consumption site (`m4_migration.rs` — passes `budgets.rel_bindings.*`
///   to `OwnerPayloadStore::create`/`OwnerForwardIndex::create`) would
///   leave every SUM-shaped or floor-shaped gate green (both caps are
///   still individually nonzero), but this seam makes the class+field
///   selection observable: a node-only load must still succeed (node caps
///   untouched) while an overflow-id-bearing rel load must trip
///   `DiskBudgetExceeded` specifically on the rel PAYLOAD companion.
#[must_use]
pub fn plan_owner_budgets(census: &LoadCensus) -> OwnerBulkBudgets {
    #[cfg(feature = "fault-injection")]
    {
        if std::env::var_os("ARCGRAPH_M5_FIXED_BULK_CAPS").is_some() {
            let fixed = OwnerSubstrateBudget {
                index_cap_bytes: arcgraph_storage::OWNER_INDEX_DISK_CAP_BYTES,
                payload_cap_bytes: arcgraph_storage::OWNER_PAYLOAD_DISK_CAP_BYTES,
            };
            return OwnerBulkBudgets {
                node_bindings: fixed,
                rel_bindings: fixed,
            };
        }
        if std::env::var_os("ARCGRAPH_M5_ZERO_BULK_BUDGET").is_some() {
            let zero = OwnerSubstrateBudget {
                index_cap_bytes: 1,
                payload_cap_bytes: 1,
            };
            return OwnerBulkBudgets {
                node_bindings: zero,
                rel_bindings: zero,
            };
        }
    }
    #[cfg_attr(not(feature = "fault-injection"), allow(unused_mut))]
    let mut budgets = OwnerBulkBudgets::derive(
        BulkClassCensus {
            entries: census.nodes,
            external_id_bytes: census.node_external_id_bytes,
        },
        BulkClassCensus {
            entries: census.relationships,
            external_id_bytes: census.rel_external_id_bytes,
        },
    );
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET").is_some() {
        budgets.rel_bindings.payload_cap_bytes = 1;
    }
    budgets
}

/// Plan-time disk projection (amendment §5 rule 2): computed AFTER the
/// pass-1 census, BEFORE any substrate or extent write.
#[derive(Debug, Clone)]
pub struct LoadProjection {
    /// Owner-substrate bytes (both binding classes, un-inflated need).
    pub substrate_bytes: u64,
    /// Served-generation bytes (records + props + owner rows + TEL).
    pub generation_bytes: u64,
    /// Remaining pipeline scratch bytes (runs already on disk excluded).
    pub scratch_bytes: u64,
    /// Total projected write need.
    pub required_bytes: u64,
    /// Human-readable projection table.
    pub table: String,
}

/// Calibrated upper-bound generation cost per binding entry: one 256 B
/// owner row + record slot + prop/extent framing. The 100M-rung gate
/// records actuals next to the headline so drift is attributable (D-3).
const PROJECTED_GENERATION_BYTES_PER_ENTRY: u64 = 352;
/// Plan-time LOWER BOUND for both TEL directions: 32 B per entry per
/// direction. The real STORE_TEL cost is one 8 KiB page per
/// (owner, type) chain block, which the census cannot see — the s6 merge
/// counts it exactly and [`project_tel_or_refuse`] re-checks before
/// materialization.
const PROJECTED_TEL_BYTES_PER_REL: u64 = 64;

/// Pure projection arithmetic (unit-testable; INV-M5.25 gate surface).
#[must_use]
pub fn project_load_disk(census: &LoadCensus, budgets: &OwnerBulkBudgets) -> LoadProjection {
    let substrate_bytes = OwnerSubstrateBudget::projected_need_bytes(BulkClassCensus {
        entries: census.nodes,
        external_id_bytes: census.node_external_id_bytes,
    })
    .saturating_add(OwnerSubstrateBudget::projected_need_bytes(
        BulkClassCensus {
            entries: census.relationships,
            external_id_bytes: census.rel_external_id_bytes,
        },
    ));
    let entries = census.nodes.saturating_add(census.relationships);
    let generation_bytes = (census.payload_bytes.saturating_mul(3) / 2)
        .saturating_add(entries.saturating_mul(PROJECTED_GENERATION_BYTES_PER_ENTRY))
        .saturating_add(
            census
                .relationships
                .saturating_mul(PROJECTED_TEL_BYTES_PER_REL),
        );
    // Remaining scratch beyond the (already-written) canonical runs:
    // segments + endpoint/resolved runs + TEL runs/segments, with staged GC.
    let run_bytes = census
        .payload_bytes
        .saturating_add(census.records.saturating_mul(24));
    let scratch_bytes = run_bytes.saturating_mul(3) / 2;
    let required_bytes = substrate_bytes
        .saturating_add(generation_bytes)
        .saturating_add(scratch_bytes);
    let table = format!(
        "projection (bytes):\n  substrate (owner index+payload need): {substrate_bytes}\n  \
         served generation:                    {generation_bytes}\n  \
         remaining pipeline scratch:           {scratch_bytes}\n  \
         TOTAL required:                       {required_bytes}\n  \
         derived caps: node(index={}, payload={}) rel(index={}, payload={})",
        budgets.node_bindings.index_cap_bytes,
        budgets.node_bindings.payload_cap_bytes,
        budgets.rel_bindings.index_cap_bytes,
        budgets.rel_bindings.payload_cap_bytes,
    );
    LoadProjection {
        substrate_bytes,
        generation_bytes,
        scratch_bytes,
        required_bytes,
        table,
    }
}

/// Refuse (typed, with the full projection table) BEFORE building if the
/// projected need exceeds the target filesystem's free bytes or the
/// operator's `max_disk_bytes` override.
pub(crate) fn project_disk_or_refuse(
    root: &Path,
    census: &LoadCensus,
    budgets: &OwnerBulkBudgets,
    limits: &LoadLimits,
) -> Result<()> {
    let projection = project_load_disk(census, budgets);
    let free = fs_available_bytes(root)?;
    let available = limits.max_disk_bytes.map_or(free, |cap| cap.min(free));
    if projection.required_bytes > available {
        return Err(LoadRefusal::ProjectedDiskExceeded {
            data_dir: root.to_path_buf(),
            required_bytes: projection.required_bytes,
            available_bytes: available,
            table: projection.table,
        }
        .into());
    }
    Ok(())
}

/// Post-s6 EXACT STORE_TEL projection (fail-fast, INV-M5.25): `tel_pages`
/// is the s6-counted number of TEL BLOCKS the materializer will write —
/// one (owner, type) chain block per group, `tel_entries` the exact total
/// entry count across those blocks (both directions; s6's `out_entries +
/// in_entries`). #1519 BLOCK_FIX FIX 3: this now prices the DENSIFIED
/// packed layout (via [`project_dense_tel_bytes_for_blocks`]) instead of
/// the pre-#1519 page-per-block bound (`tel_pages * 8 KiB`), which
/// deterministically refused low-degree loads that now FIT the densified
/// layout — defeating #1519's own 100M/1B headline (a refusal gate that
/// never fires on production-shaped data is vacuous). Before #1519,
/// STORE_TEL cost `≥ 8 KiB × Σ distinct (owner, type) groups × 2
/// directions`, which the pass-1 census cannot see (plan time only
/// lower-bounds it at [`PROJECTED_TEL_BYTES_PER_REL`]); at low average
/// degree this term DOMINATED the generation (the 100M-rung STOP-report
/// measured a ~3 TB out-TEL trajectory at avg degree 5 × 7 types).
/// Refusing here — before the materializer writes a byte of the store —
/// is still the difference between a typed plan refusal and an hour-3
/// `ENOSPC`; it is just no longer priced 87x over the actual densified
/// need.
pub(crate) fn project_tel_or_refuse(
    root: &Path,
    census: &LoadCensus,
    tel_pages: u64,
    tel_entries: u64,
    limits: &LoadLimits,
) -> Result<()> {
    let tel_bytes = project_dense_tel_bytes_for_blocks(tel_pages, tel_entries);
    // Remaining non-TEL generation need (records + props + owner rows).
    let entries = census.nodes.saturating_add(census.relationships);
    let rest_bytes = (census.payload_bytes.saturating_mul(3) / 2)
        .saturating_add(entries.saturating_mul(PROJECTED_GENERATION_BYTES_PER_ENTRY));
    let required = tel_bytes.saturating_add(rest_bytes);
    let free = fs_available_bytes(root)?;
    let available = limits.max_disk_bytes.map_or(free, |cap| cap.min(free));
    if required > available {
        return Err(LoadRefusal::ProjectedDiskExceeded {
            data_dir: root.to_path_buf(),
            required_bytes: required,
            available_bytes: available,
            table: format!(
                "post-s6 exact projection (bytes):
  STORE_TEL (densified packed layout, #1519,                  {tel_pages} blocks / {tel_entries} entries, BOTH directions): {tel_bytes}
                   records+props+owner rows:             {rest_bytes}
                   TOTAL required:                       {required}
                   NOTE: STORE_TEL is priced at the #1519 densified packed-page                  layout (see docs/adr for the densify ruling), not the pre-#1519                  page-per-(owner,type)-block layout the M5-D3 STOP-report measured."
            ),
        }
        .into());
    }
    Ok(())
}

/// Shared dense-packing arithmetic core: worst-case densified STORE_TEL
/// bytes for `blocks` (owner, type) chain blocks totalling `entries`
/// adjacency entries. Both [`project_dense_tel_bytes`] (the pure
/// worst-case-content projection, no real data) and
/// `project_tel_or_refuse` (the real post-s6 exact-block-count
/// refusal) go through this ONE arithmetic core so the production
/// refusal path and the budget-headline projection can never silently
/// diverge (#1519 BLOCK_FIX FIX 3 — `project_dense_tel_bytes` previously
/// had zero production callers, so its own regression gate was vacuous;
/// wiring both callers through the same core makes that impossible by
/// construction).
///
/// `content_bytes = blocks * HEADER_SIZE + entries * ENTRY_SIZE` is
/// EXACT (every block pays exactly one 32 B TEL header regardless of
/// size; every entry is exactly 32 B) — not a bound. `directory_bytes =
/// blocks * TEL_PACKED_DIR_SLOT_BYTES` assumes EVERY block pays a full
/// packed-directory slot, which over-counts a real supernode/chain block
/// (flags=0, zero directory overhead) — but only in the SAFE (more
/// conservative, never under-refuses) direction, since a supernode
/// consolidating many entries into one dedicated page is always at
/// least as dense as the packed-worst-case model charges for the same
/// entries. The final `* 2` page-count safety factor is the greedy
/// bin-packing bound for [`crate::m5_parallel`]'s sequential packer:
/// every PACKED item is, by construction, `< TEL_SUPERNODE_THRESHOLD_ENTRIES`
/// (at most half a page's entry capacity), so a page the packer decides
/// is "full" (the next item didn't fit) is always MORE than half-full —
/// i.e. the packer never wastes more than half of any page, so `pages ≤
/// ceil(2 * total_bytes / usable_body_bytes)` is a provable upper bound,
/// never an under-count.
#[must_use]
pub fn project_dense_tel_bytes_for_blocks(blocks: u64, entries: u64) -> u64 {
    use arcgraph_core::PAGE_SIZE;
    use arcgraph_storage::m4_migration::{TEL_PACKED_DIR_HEADER_BYTES, TEL_PACKED_DIR_SLOT_BYTES};
    use arcgraph_storage::tel::{ENTRY_SIZE, HEADER_SIZE};

    let content_bytes = blocks
        .saturating_mul(HEADER_SIZE as u64)
        .saturating_add(entries.saturating_mul(ENTRY_SIZE as u64));
    let directory_bytes = blocks.saturating_mul(TEL_PACKED_DIR_SLOT_BYTES as u64);
    let total_bytes = content_bytes.saturating_add(directory_bytes);
    let usable_body_bytes =
        (PAGE_SIZE - arcgraph_core::PageHeader::SIZE - TEL_PACKED_DIR_HEADER_BYTES) as u64;
    // >=50%-utilization greedy-packing safety factor (see doc comment):
    // pages = ceil(2 * total_bytes / usable_body_bytes).
    let pages = total_bytes
        .saturating_mul(2)
        .div_ceil(usable_body_bytes.max(1));
    pages.saturating_mul(PAGE_SIZE as u64)
}

/// #1519 pure arithmetic: worst-case densified STORE_TEL bytes for
/// `relationships` total edges, BOTH directions (`2 * relationships`
/// (owner, type) blocks, `2 * relationships` entries — the same
/// degenerate "every block distinct-type, 1 entry" worst case
/// `project_tel_or_refuse`'s conservative bound assumed pre-#1519).
/// Assumes NO block reaches
/// [`arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES`]
/// (the worst case for packing density: every block pays a full directory
/// slot for the fewest possible entry bytes). This is the "would the
/// densified layout fit the 100M/1B budget" projection the #1519 charter
/// calls for (mirrors the INV-M5.25 budget-projection arithmetic style —
/// no actual load, no I/O, pure numbers pinned by a unit test).
#[must_use]
pub fn project_dense_tel_bytes(relationships: u64) -> u64 {
    let blocks = relationships.saturating_mul(2);
    let entries = relationships.saturating_mul(2); // 1 entry per block, both directions
    project_dense_tel_bytes_for_blocks(blocks, entries)
}

#[cfg(unix)]
fn fs_available_bytes(path: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let raw = std::ffi::CString::new(path.as_os_str().as_bytes())
        .context("data dir path contains NUL")?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    ensure!(
        // SAFETY: raw is a valid NUL-terminated path and stats points at
        // writable statvfs storage owned by this frame.
        unsafe { libc::statvfs(raw.as_ptr(), stats.as_mut_ptr()) } == 0,
        "statvfs({}) failed: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    // SAFETY: statvfs returned 0, so the struct is fully initialized.
    let stats = unsafe { stats.assume_init() };
    // `f_bavail` is `fsblkcnt_t`, which is `u32` on macOS/BSD (conversion
    // required) but already `u64` on Linux gnu/x86_64 (conversion is a
    // no-op there, hence the lint only fires on the Linux CI target).
    #[cfg_attr(target_os = "linux", allow(clippy::useless_conversion))]
    Ok(u64::from(stats.f_bavail).saturating_mul(stats.f_frsize))
}

#[cfg(not(unix))]
fn fs_available_bytes(_path: &Path) -> Result<u64> {
    Ok(u64::MAX)
}

/// The generation's durable tenant census as production catalog rows
/// (ADR-207 object). `created_lsn` is the INV-M5.2 frontier — the LSN at
/// which the loaded content became durable.
fn census_tenant_records(tenant: TenantId, migration_lsn: Lsn) -> Vec<TenantRecord> {
    vec![
        TenantRecord {
            tenant_id: TenantId::DEFAULT,
            name: "default".to_owned(),
            created_lsn: migration_lsn,
            tier: DurabilityTier::default(),
        },
        TenantRecord {
            tenant_id: tenant,
            name: format!("loaded-{}", tenant.raw()),
            created_lsn: migration_lsn,
            tier: DurabilityTier::default(),
        },
    ]
}

/// Write the tenant census onto the catalog root page of the built
/// generation's `pages.db` through the production ADR-207 encoder, then
/// decode-verify the round trip (the PROVEN doctrine, same as
/// `SystemCatalog::attach_page_store` §3). A cold open of the loaded store
/// then attaches this page through the SAME path as any other durable
/// store — no loader-specific bootstrap hook exists.
fn write_catalog_root_census(generation: &Path, records: &[TenantRecord]) -> Result<()> {
    let encoded = encode_catalog_page(records)
        .map_err(|error| anyhow::anyhow!("encode fresh-load catalog root page: {error}"))?;
    let io = PosixPageIo::open_or_create(generation.join("pages.db"))
        .context("open fresh-load catalog root pages.db")?;
    io.write_page(CATALOG_PAGE_ID, &encoded)
        .context("write fresh-load catalog root page")?;
    io.flush().context("flush fresh-load catalog root page")?;
    let mut back = Box::new([0u8; arcgraph_core::PAGE_SIZE]);
    let buf: &mut PageBuf = back.as_mut();
    io.read_page(CATALOG_PAGE_ID, buf)
        .context("read back fresh-load catalog root page")?;
    let decoded = decode_catalog_page(buf)
        .map_err(|error| anyhow::anyhow!("fresh-load catalog page verify failed: {error}"))?;
    ensure!(
        decoded == records,
        "fresh-load catalog root page read-back does not match the census just written"
    );
    Ok(())
}

/// RED-on-revert seam for INV-M5.20/.17 (cfg-gated + bounded per the
/// standing test-hook rule): ship the served generation with EMPTY
/// STORE_TEL. Production builds compile the check out entirely.
pub(crate) fn ship_empty_tel() -> bool {
    #[cfg(feature = "fault-injection")]
    {
        std::env::var_os("ARCGRAPH_M5_SHIP_EMPTY_TEL").is_some()
    }
    #[cfg(not(feature = "fault-injection"))]
    false
}

/// The record's property payload at materialization. The RED-on-revert
/// seams (cfg-gated + bounded) model the V-3 regression classes:
/// `ARCGRAPH_M5_SHIP_EMPTY_PROPS` discards the payload entirely (the
/// dbf13a5a `let _float_bits`/`let _opaque` drop — STORE_PROPS ships
/// empty), and `ARCGRAPH_M5_LOSSY_FLOAT_BITS` clears the low mantissa
/// bit (a 1-ULP-lossy materialization the INV-M5.12 served-terminus
/// oracle must catch).
pub(crate) fn materialized_bag(float_bits: u64, opaque: &[u8]) -> Vec<u8> {
    #[cfg(feature = "fault-injection")]
    {
        if std::env::var_os("ARCGRAPH_M5_SHIP_EMPTY_PROPS").is_some() {
            return Vec::new();
        }
        if std::env::var_os("ARCGRAPH_M5_LOSSY_FLOAT_BITS").is_some() {
            return canonical_property_bag(float_bits & !1, opaque);
        }
    }
    canonical_property_bag(float_bits, opaque)
}

/// Sort key for one TEL run entry: big-endian `(owner, type, rel)` so
/// byte order equals numeric order (§M5.1(2) run table, extended with
/// the type component so loader blocks stay single-channel like the
/// production `TelBlock` grain).
pub(crate) fn tel_run_key(owner: u64, type_id: u32, rel_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(20);
    key.extend_from_slice(&owner.to_be_bytes());
    key.extend_from_slice(&type_id.to_be_bytes());
    key.extend_from_slice(&rel_id.to_be_bytes());
    key
}

pub(crate) fn encode_tel_run(owner: u64, type_id: u32, neighbor: u64, rel_id: u64) -> Vec<u8> {
    let mut payload = Vec::with_capacity(28);
    payload.extend_from_slice(&owner.to_le_bytes());
    payload.extend_from_slice(&type_id.to_le_bytes());
    payload.extend_from_slice(&neighbor.to_le_bytes());
    payload.extend_from_slice(&rel_id.to_le_bytes());
    payload
}

pub(crate) fn decode_tel_run(payload: &[u8]) -> Result<(u64, u32, u64, u64)> {
    let mut cursor = 0;
    let owner = take_u64(payload, &mut cursor, "TEL run owner")?;
    let type_id = take_u32(payload, &mut cursor, "TEL run type")?;
    let neighbor = take_u64(payload, &mut cursor, "TEL run neighbor")?;
    let rel_id = take_u64(payload, &mut cursor, "TEL run rel id")?;
    ensure!(cursor == payload.len(), "TEL run entry has trailing bytes");
    Ok((owner, type_id, neighbor, rel_id))
}

pub(crate) fn canonical_sort_key(record: &LoadRecord) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + record.external_id().len());
    key.push(u8::from(matches!(record, LoadRecord::Relationship { .. })));
    key.extend_from_slice(record.external_id());
    key
}

pub(crate) fn encode_canonical_record(record: &LoadRecord) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    match record {
        LoadRecord::Node {
            external_id,
            label,
            float_bits,
            opaque,
        } => {
            payload.push(0);
            put_bytes(&mut payload, external_id)?;
            payload.extend_from_slice(&label.to_le_bytes());
            payload.extend_from_slice(&float_bits.to_le_bytes());
            put_bytes(&mut payload, opaque)?;
        }
        LoadRecord::Relationship {
            external_id,
            source_external_id,
            target_external_id,
            type_id,
            float_bits,
            opaque,
        } => {
            payload.push(1);
            put_bytes(&mut payload, external_id)?;
            put_bytes(&mut payload, source_external_id)?;
            put_bytes(&mut payload, target_external_id)?;
            payload.extend_from_slice(&type_id.to_le_bytes());
            payload.extend_from_slice(&float_bits.to_le_bytes());
            put_bytes(&mut payload, opaque)?;
        }
    }
    Ok(payload)
}

pub(crate) fn decode_canonical_record(payload: &[u8]) -> Result<LoadRecord> {
    let mut cursor = 0;
    let kind = take_u8(payload, &mut cursor, "record kind")?;
    let external_id = take_bytes(payload, &mut cursor, MAX_EXTERNAL_ID_BYTES, "external id")?;
    match kind {
        0 => {
            let label = take_u32(payload, &mut cursor, "node label")?;
            let float_bits = take_u64(payload, &mut cursor, "node float bits")?;
            let opaque = take_bytes(payload, &mut cursor, MAX_OPAQUE_BYTES, "node opaque")?;
            ensure!(cursor == payload.len(), "node record has trailing bytes");
            Ok(LoadRecord::Node {
                external_id,
                label,
                float_bits,
                opaque,
            })
        }
        1 => {
            let source_external_id = take_bytes(
                payload,
                &mut cursor,
                MAX_EXTERNAL_ID_BYTES,
                "relationship source id",
            )?;
            let target_external_id = take_bytes(
                payload,
                &mut cursor,
                MAX_EXTERNAL_ID_BYTES,
                "relationship target id",
            )?;
            let type_id = take_u32(payload, &mut cursor, "relationship type")?;
            let float_bits = take_u64(payload, &mut cursor, "relationship float bits")?;
            let opaque = take_bytes(
                payload,
                &mut cursor,
                MAX_OPAQUE_BYTES,
                "relationship opaque",
            )?;
            ensure!(
                cursor == payload.len(),
                "relationship record has trailing bytes"
            );
            Ok(LoadRecord::Relationship {
                external_id,
                source_external_id,
                target_external_id,
                type_id,
                float_bits,
                opaque,
            })
        }
        _ => bail!("unknown canonical record kind {kind}"),
    }
}

/// Decoded canonical node artifact (id + identity + property payload).
pub(crate) struct NodeArtifact {
    pub(crate) internal_id: u64,
    pub(crate) external_id: Vec<u8>,
    pub(crate) label: u32,
    pub(crate) float_bits: u64,
    pub(crate) opaque: Vec<u8>,
}

pub(crate) fn decode_node_artifact(payload: &[u8]) -> Result<NodeArtifact> {
    let mut cursor = 0;
    let internal_id = take_u64(payload, &mut cursor, "node internal id")?;
    let external_id = take_bytes(
        payload,
        &mut cursor,
        MAX_EXTERNAL_ID_BYTES,
        "node external id",
    )?;
    let label = take_u32(payload, &mut cursor, "node label")?;
    // float_bits/opaque are materialized into the served generation
    // (M5-D2, INV-M5.20/.12): never discarded past this point.
    let float_bits = take_u64(payload, &mut cursor, "node float")?;
    let opaque = take_bytes(payload, &mut cursor, MAX_OPAQUE_BYTES, "node opaque")?;
    ensure!(cursor == payload.len(), "node artifact has trailing bytes");
    Ok(NodeArtifact {
        internal_id,
        external_id,
        label,
        float_bits,
        opaque,
    })
}

/// Decoded fully-resolved relationship spool entry.
pub(crate) struct ResolvedRel {
    pub(crate) internal_id: u64,
    pub(crate) external_id: Vec<u8>,
    pub(crate) type_id: u32,
    pub(crate) source_id: u64,
    pub(crate) target_id: u64,
    pub(crate) float_bits: u64,
    pub(crate) opaque: Vec<u8>,
}

pub(crate) fn decode_resolved_rel(payload: &[u8]) -> Result<ResolvedRel> {
    let mut cursor = 0;
    let internal_id = take_u64(payload, &mut cursor, "resolved rel internal id")?;
    let external_id = take_bytes(
        payload,
        &mut cursor,
        MAX_EXTERNAL_ID_BYTES,
        "resolved rel external id",
    )?;
    let type_id = take_u32(payload, &mut cursor, "resolved rel type")?;
    let source_id = take_u64(payload, &mut cursor, "resolved rel source")?;
    let target_id = take_u64(payload, &mut cursor, "resolved rel target")?;
    let float_bits = take_u64(payload, &mut cursor, "resolved rel float")?;
    let opaque = take_bytes(
        payload,
        &mut cursor,
        MAX_OPAQUE_BYTES,
        "resolved rel opaque",
    )?;
    ensure!(
        cursor == payload.len(),
        "resolved relationship has trailing bytes"
    );
    Ok(ResolvedRel {
        internal_id,
        external_id,
        type_id,
        source_id,
        target_id,
        float_bits,
        opaque,
    })
}

pub(crate) fn encode_endpoint_request(
    endpoint: &[u8],
    relation: &[u8],
    ordinal: u64,
    side: u8,
) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    put_bytes(&mut encoded, endpoint)?;
    put_bytes(&mut encoded, relation)?;
    encoded.extend_from_slice(&ordinal.to_le_bytes());
    encoded.push(side);
    Ok(encoded)
}

pub(crate) fn decode_endpoint_request(payload: &[u8]) -> Result<(Vec<u8>, Vec<u8>, u64, u8)> {
    let mut cursor = 0;
    let endpoint = take_bytes(payload, &mut cursor, MAX_EXTERNAL_ID_BYTES, "endpoint")?;
    let relation = take_bytes(
        payload,
        &mut cursor,
        MAX_EXTERNAL_ID_BYTES,
        "relationship id",
    )?;
    let ordinal = take_u64(payload, &mut cursor, "relationship ordinal")?;
    let side = take_u8(payload, &mut cursor, "endpoint side")?;
    ensure!(
        side <= 1 && cursor == payload.len(),
        "invalid endpoint request"
    );
    Ok((endpoint, relation, ordinal, side))
}

pub(crate) fn decode_binding(payload: &[u8]) -> Result<(Vec<u8>, u64)> {
    let mut cursor = 0;
    let external = take_bytes(
        payload,
        &mut cursor,
        MAX_EXTERNAL_ID_BYTES,
        "binding external id",
    )?;
    let internal = take_u64(payload, &mut cursor, "binding internal id")?;
    ensure!(cursor == payload.len(), "binding has trailing bytes");
    Ok((external, internal))
}

pub(crate) fn decode_resolved_endpoint(payload: &[u8]) -> Result<(Vec<u8>, u8, u64)> {
    let mut cursor = 0;
    let relation = take_bytes(
        payload,
        &mut cursor,
        MAX_EXTERNAL_ID_BYTES,
        "resolved relationship id",
    )?;
    let side = take_u8(payload, &mut cursor, "resolved endpoint side")?;
    let internal = take_u64(payload, &mut cursor, "resolved endpoint id")?;
    ensure!(
        side <= 1 && cursor == payload.len(),
        "invalid resolved endpoint"
    );
    Ok((relation, side, internal))
}

// ---------------------------------------------------------------------------
// Bounded external sort (serial core; D3 rebuilds it per-worker)
// ---------------------------------------------------------------------------

pub(crate) fn take_u8(bytes: &[u8], cursor: &mut usize, field: &str) -> Result<u8> {
    let value = *bytes
        .get(*cursor)
        .with_context(|| format!("missing {field}"))?;
    *cursor += 1;
    Ok(value)
}

pub(crate) fn take_u32(bytes: &[u8], cursor: &mut usize, field: &str) -> Result<u32> {
    let end = cursor.checked_add(4).context("u32 cursor overflow")?;
    let raw = bytes
        .get(*cursor..end)
        .with_context(|| format!("missing {field}"))?;
    *cursor = end;
    Ok(u32::from_le_bytes(raw.try_into().expect("4-byte slice")))
}

pub(crate) fn take_u64(bytes: &[u8], cursor: &mut usize, field: &str) -> Result<u64> {
    let end = cursor.checked_add(8).context("u64 cursor overflow")?;
    let raw = bytes
        .get(*cursor..end)
        .with_context(|| format!("missing {field}"))?;
    *cursor = end;
    Ok(u64::from_le_bytes(raw.try_into().expect("8-byte slice")))
}

pub(crate) fn take_bytes(
    bytes: &[u8],
    cursor: &mut usize,
    cap: usize,
    field: &str,
) -> Result<Vec<u8>> {
    let length = take_u32(bytes, cursor, &format!("{field} length"))? as usize;
    ensure!(length <= cap, "{field} exceeds {cap} bytes");
    let end = cursor.checked_add(length).context("byte cursor overflow")?;
    let value = bytes
        .get(*cursor..end)
        .with_context(|| format!("missing {field}"))?;
    *cursor = end;
    Ok(value.to_vec())
}

pub(crate) fn put_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    let length = u32::try_from(value.len()).context("field length exceeds u32")?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value);
    Ok(())
}

pub(crate) fn write_framed(writer: &mut impl Write, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len()).context("framed payload exceeds u32")?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(payload)?;
    Ok(())
}

pub(crate) fn sync_writer(writer: &mut BufWriter<File>) -> Result<()> {
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

pub(crate) fn create_new(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))
}

pub(crate) struct FramedReader {
    reader: BufReader<File>,
}

impl FramedReader {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: BufReader::new(
                File::open(path).with_context(|| format!("open framed file {}", path.display()))?,
            ),
        })
    }

    pub(crate) fn next(&mut self) -> Result<Option<Vec<u8>>> {
        let Some(length) = read_optional_u32(&mut self.reader)? else {
            return Ok(None);
        };
        ensure!(
            length as usize <= MAX_NATIVE_RECORD_BYTES,
            "framed record exceeds cap"
        );
        let mut payload = vec![0; length as usize];
        self.reader.read_exact(&mut payload)?;
        Ok(Some(payload))
    }
}

pub(crate) fn read_optional_u32(reader: &mut impl Read) -> Result<Option<u32>> {
    let mut bytes = [0; 4];
    let mut read = 0;
    while read < bytes.len() {
        match reader.read(&mut bytes[read..])? {
            0 if read == 0 => return Ok(None),
            0 => bail!("truncated external-sort length"),
            count => read += count,
        }
    }
    Ok(Some(u32::from_le_bytes(bytes)))
}

pub(crate) fn read_required_u32(reader: &mut impl Read, field: &str) -> Result<u32> {
    read_optional_u32(reader)?.with_context(|| format!("missing {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn native_depth_and_size_caps_precede_json_allocation() {
        let mut reader = BufReader::new(Cursor::new(vec![b'x'; MAX_NATIVE_RECORD_BYTES + 1]));
        assert!(read_bounded_line(&mut reader, MAX_NATIVE_RECORD_BYTES).is_err());
        let nested = format!(
            "{}0{}",
            "[".repeat(MAX_NATIVE_RECURSION + 1),
            "]".repeat(MAX_NATIVE_RECURSION + 1)
        );
        assert!(ensure_json_depth(nested.as_bytes(), MAX_NATIVE_RECURSION).is_err());
    }

    /// Mechanism-1 soundness: byte-range partitions resynced to frame
    /// boundaries assign every record to exactly one partition, for any
    /// split point (including splits landing exactly on a line start).
    #[test]
    fn open_range_partitions_cover_every_record_exactly_once() {
        let dir = tempfile::tempdir().expect("dir");
        let path = dir.path().join("input.jsonl");
        let mut body = String::new();
        for value in 0..97_u32 {
            body.push_str(&format!(
                "{{\"kind\":\"node\",\"external_id\":\"{:08x}\",\"label_or_type\":1,\
                 \"float_bits\":\"0000000000000000\",\"opaque\":\"\"}}\n",
                value
            ));
        }
        fs::write(&path, &body).expect("write");
        let len = body.len() as u64;
        for splits in [1_u64, 2, 3, 7, 13] {
            let mut seen = Vec::new();
            let chunk = len.div_ceil(splits);
            let mut start = 0;
            while start < len {
                let end = (start + chunk).min(len);
                let mut source =
                    NativeRecordSource::open_range(&path, start, end).expect("open range");
                while let Some(record) = source.next_record().expect("record") {
                    seen.push(record.external_id().to_vec());
                }
                start = end;
            }
            assert_eq!(seen.len(), 97, "splits={splits}");
            let unique: std::collections::BTreeSet<_> = seen.iter().cloned().collect();
            assert_eq!(unique.len(), 97, "no duplicates at splits={splits}");
        }
    }

    /// INV-M5.25 wiring shape: the plan-time projection refuses under a
    /// tiny operator disk cap and passes with a generous one.
    #[test]
    fn projection_refuses_under_operator_disk_cap() {
        let census = LoadCensus {
            records: 1_000,
            nodes: 500,
            relationships: 500,
            node_external_id_bytes: 8_000,
            rel_external_id_bytes: 8_000,
            payload_bytes: 64_000,
        };
        let budgets = plan_owner_budgets(&census);
        let projection = project_load_disk(&census, &budgets);
        assert!(projection.required_bytes > 0);
        let dir = tempfile::tempdir().expect("dir");
        let tight = LoadLimits {
            max_disk_bytes: Some(1),
            ..LoadLimits::production()
        };
        let error = project_disk_or_refuse(dir.path(), &census, &budgets, &tight)
            .expect_err("1-byte cap must refuse");
        assert!(
            error.downcast_ref::<LoadRefusal>().is_some_and(|refusal| {
                matches!(refusal, LoadRefusal::ProjectedDiskExceeded { .. })
            }),
            "typed refusal expected, got: {error:?}"
        );
        let generous = LoadLimits {
            max_disk_bytes: Some(u64::MAX),
            ..LoadLimits::production()
        };
        project_disk_or_refuse(dir.path(), &census, &budgets, &generous)
            .expect("generous cap admits the plan (bounded by real free space)");
    }
}
