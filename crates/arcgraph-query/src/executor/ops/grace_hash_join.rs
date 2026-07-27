//! Disk-backed Grace hash-join runtime (M6.2 OOC-3).
//!
//! This module is the opt-in spill path for [`super::HashJoinOp`]. Both
//! children are partitioned through OOC-1 runs before any partition is
//! joined. A configured [`MemoryBudget`] cap is a hard precondition for the
//! bigger-than-RAM guarantee: without a cap there is no finite quantity from
//! which to derive the fan-out, and `HashJoinOp` retains its legacy in-memory
//! behavior (refs #1524).
//!
//! `P = ceil(estimated_build_bytes / (available_budget / 2))`, rounded up to
//! a power of two and capped at [`MAX_GRACE_PARTITIONS`] and the next power of
//! two that can usefully hold the estimated key cardinality `K`. Half the
//! available cap is reserved for the build table; the other half leaves room
//! for the upstream/probe batch and executor bookkeeping. When `K` is too
//! small for the byte-derived fan-out, depth-capped reseeding (rather than a
//! dishonest `B / P` assumption) handles the resulting skew.
//!
//! A partition that remains oversized is recursively repartitioned with a
//! new seed. At [`DEFAULT_GRACE_MAX_REPARTITION_DEPTH`] it becomes bounded
//! build blocks. OOC-1 runs are intentionally single-consumer, so the probe
//! rows for that leaf are copied once into one run per block. Each resulting
//! pair is consumed exactly once; output stays streaming and recursion always
//! terminates, including for a single hot key that no seed can split.

use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "fault-injection")]
use std::sync::Mutex;

use arcgraph_core::TenantId;
use arcgraph_storage::{QueryEpoch, SpillQuery, SpillRun, SpillRunReader, SpillRunWriter};

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::budget::{MemoryBudget, MemoryReservation, estimate_value_bytes};
use crate::executor::context::ExecutionContext;
use crate::executor::error::{ExecutionError, ExecutorSpillError, ExecutorSpillFailureKind};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::join::join_key_fingerprint;
use crate::executor::ops::sort_spill_codec::{
    SortSpillCodecError, decode_join_rows, encode_join_rows,
};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;

/// Default number of reseeded levels before the block fallback.
///
/// With recursive fan-out `R >= 2`, three levels expose at least `R^3`
/// buckets. A uniformly distributed partition therefore shrinks by >=8x;
/// an unsplittable hot key reaches the fallback after exactly three levels.
pub const DEFAULT_GRACE_MAX_REPARTITION_DEPTH: u8 = 3;

/// Hard fan-out ceiling. Besides bounding metadata, this caps each side's
/// simultaneous writer staging at 64 x OOC-1's 8 KiB = 512 KiB.
pub const MAX_GRACE_PARTITIONS: usize = 64;

const MAX_CONFIGURED_REPARTITION_DEPTH: u8 = 8;
const INITIAL_PARTITION_SEED: u64 = 0x6a09_e667_f3bc_c909;
// Per-partition staging stays deliberately smaller than OOC-1's 64 MiB frame
// ceiling. Combined with the live-handle cap below, buffered plaintext is
// bounded even when a recursively partitioned join has many active writers.
const GRACE_PARTITION_FRAME_TARGET_BYTES: u64 = 16 * 1024;
// `HashMap` and each duplicate bucket grow geometrically. A one-row table is
// the worst amortization case: its minimum table allocation and four-element
// bucket leave roughly 211 bytes beyond the row/key/slot payload. Charge 256
// bytes per row so the same estimate can safely drive partition sizing and the
// pre-allocation MemoryBudget reservation.
const GRACE_BUILD_TABLE_ROW_SLACK_BYTES: u64 = 256;
// Initial 64-way partitioning plus one 8-way recursive level can retain 512
// build leaves and then 512 probe writers. Leave headroom for block-fallback
// runs while placing a deterministic ceiling on fd-owned OOC-1 handles; deeper
// shapes fail with a typed query-level resource error rather than growing
// without bound.
const MAX_GRACE_LIVE_RUN_HANDLES: usize = 2_048;

type BuildTable = HashMap<String, Vec<Vec<Value>>>;

/// One live OOC-1 target owned by an external [`super::HashJoinOp`].
pub struct GraceHashJoinTarget {
    query: SpillQuery,
    estimated_build_bytes: u64,
    estimated_key_cardinality: u64,
    max_repartition_depth: u8,
    telemetry: GraceTelemetry,
}

impl std::fmt::Debug for GraceHashJoinTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraceHashJoinTarget")
            .field("query", &self.query)
            .field("estimated_build_bytes", &self.estimated_build_bytes)
            .field("estimated_key_cardinality", &self.estimated_key_cardinality)
            .field("max_repartition_depth", &self.max_repartition_depth)
            .finish_non_exhaustive()
    }
}

impl GraceHashJoinTarget {
    /// Bind a spill query and the planner's build-side estimates.
    ///
    /// As in the legacy `HashJoinOp`, LEFT is the logical build input. The
    /// planner is responsible for placing the smaller estimated input there;
    /// retaining that designation preserves the in-memory operator's schema
    /// ownership and output semantics on the external path.
    #[must_use]
    pub fn new(
        query: SpillQuery,
        estimated_build_bytes: u64,
        estimated_key_cardinality: u64,
    ) -> Self {
        Self {
            query,
            estimated_build_bytes,
            estimated_key_cardinality,
            max_repartition_depth: DEFAULT_GRACE_MAX_REPARTITION_DEPTH,
            telemetry: GraceTelemetry::default(),
        }
    }

    /// Override the recursion cap. Values above eight are rejected so a
    /// configuration mistake cannot create an exponential liveness hazard.
    pub fn with_max_repartition_depth(mut self, depth: u8) -> Result<Self, ExecutionError> {
        if depth > MAX_CONFIGURED_REPARTITION_DEPTH {
            return Err(invalid_config(format!(
                "Grace hash-join repartition depth must be <= {MAX_CONFIGURED_REPARTITION_DEPTH}, got {depth}"
            )));
        }
        self.max_repartition_depth = depth;
        Ok(self)
    }

    /// Attach the cfg-only real-occupancy observer used by the release gates.
    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub fn with_probe(mut self, probe: GraceHashJoinProbe) -> Self {
        self.telemetry.probe = Some(probe);
        self
    }
}

/// Measurements taken from live buffers/tables, never predicted constants.
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraceHashJoinStats {
    pub chosen_partitions: usize,
    pub estimated_key_cardinality: u64,
    pub initial_build_runs_created: u64,
    /// Actual probe leaf/block runs sealed for join tasks.
    pub initial_probe_runs_created: u64,
    /// Distinct initial probe roots that routed at least one row.
    pub initial_probe_partitions_spilled: u64,
    pub recursive_runs_created: u64,
    pub max_recursion_depth: u8,
    pub block_fallback_partitions: u64,
    pub peak_partition_batch_bytes: u64,
    pub peak_build_table_bytes: u64,
    pub peak_probe_row_bytes: u64,
    pub peak_probe_batch_bytes: u64,
    pub peak_join_resident_bytes: u64,
}

/// Shared fault-build probe for the four OOC-3 release gates.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Default)]
pub struct GraceHashJoinProbe {
    inner: Arc<Mutex<GraceHashJoinStats>>,
}

#[cfg(feature = "fault-injection")]
impl GraceHashJoinProbe {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn snapshot(&self) -> GraceHashJoinStats {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, update: impl FnOnce(&mut GraceHashJoinStats)) {
        update(
            &mut self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
}

#[derive(Clone, Default)]
struct GraceTelemetry {
    #[cfg(feature = "fault-injection")]
    probe: Option<GraceHashJoinProbe>,
}

impl GraceTelemetry {
    fn configured(&self, partitions: usize, cardinality: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.chosen_partitions = partitions;
                stats.estimated_key_cardinality = cardinality;
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = (partitions, cardinality);
    }

    fn initial_build_run(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.initial_build_runs_created =
                    stats.initial_build_runs_created.saturating_add(1);
            });
        }
    }

    fn initial_probe_run(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.initial_probe_runs_created =
                    stats.initial_probe_runs_created.saturating_add(1);
            });
        }
    }

    fn initial_probe_partition(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.initial_probe_partitions_spilled =
                    stats.initial_probe_partitions_spilled.saturating_add(1);
            });
        }
    }

    fn recursive_run(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.recursive_runs_created = stats.recursive_runs_created.saturating_add(1);
            });
        }
    }

    fn recursion_depth(&self, depth: u8) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| stats.max_recursion_depth = stats.max_recursion_depth.max(depth));
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = depth;
    }

    fn block_fallback(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.block_fallback_partitions = stats.block_fallback_partitions.saturating_add(1);
            });
        }
    }

    fn partition_batch(&self, bytes: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.peak_partition_batch_bytes = stats.peak_partition_batch_bytes.max(bytes);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = bytes;
    }

    fn build_table(&self, bytes: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.peak_build_table_bytes = stats.peak_build_table_bytes.max(bytes);
                stats.peak_join_resident_bytes = stats.peak_join_resident_bytes.max(bytes);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = bytes;
    }

    fn active_probe(&self, table_bytes: u64, probe_row_bytes: u64, probe_batch_bytes: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.peak_probe_row_bytes = stats.peak_probe_row_bytes.max(probe_row_bytes);
                stats.peak_probe_batch_bytes = stats.peak_probe_batch_bytes.max(probe_batch_bytes);
                stats.peak_join_resident_bytes = stats
                    .peak_join_resident_bytes
                    .max(table_bytes.saturating_add(probe_batch_bytes));
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = (table_bytes, probe_row_bytes, probe_batch_bytes);
    }
}

pub(super) struct GraceHashJoinRuntime {
    target: GraceHashJoinTarget,
    prepared: bool,
    tasks: VecDeque<JoinTask>,
    active: Option<ActiveJoin>,
    done: bool,
}

impl GraceHashJoinRuntime {
    pub(super) fn new(target: GraceHashJoinTarget) -> Self {
        Self {
            target,
            prepared: false,
            tasks: VecDeque::new(),
            active: None,
            done: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        left: &mut PhysicalOperator,
        right: &mut PhysicalOperator,
        left_shared_indices: &[usize],
        right_shared_indices: &[usize],
        right_fresh_indices: &[usize],
        schema_len: usize,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        if self.done {
            return Ok(Batch::empty(schema_len));
        }
        if !self.prepared {
            self.prepare(
                left,
                right,
                left_shared_indices,
                right_shared_indices,
                ctx,
                substrate,
            )?;
            self.prepared = true;
        }

        let mut out = Batch::with_capacity(schema_len);
        while !out.is_full() {
            ctx.cancellation().check()?;
            if self.active.is_none() {
                let Some(task) = self.tasks.pop_front() else {
                    self.done = true;
                    break;
                };
                self.active = Some(ActiveJoin::open(
                    task,
                    self.target.query.epoch(),
                    left_shared_indices,
                    ctx.budget(),
                    ctx.tenant(),
                    self.target.telemetry.clone(),
                )?);
            }

            let next = match self.active.as_mut() {
                Some(active) => active.next_joined(right_shared_indices, right_fresh_indices)?,
                None => None,
            };
            match next {
                Some(row) => {
                    if !out.push_row(row) {
                        return Err(invalid_state(
                            "Grace hash-join output batch overflow despite fullness guard",
                        ));
                    }
                }
                None => {
                    self.active = None;
                }
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<S: ExecutorSubstrate>(
        &mut self,
        left: &mut PhysicalOperator,
        right: &mut PhysicalOperator,
        left_shared_indices: &[usize],
        right_shared_indices: &[usize],
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let budget = ctx.budget();
        let tenant = ctx.tenant();
        let cap = budget.cap_bytes(tenant).ok_or_else(|| {
            invalid_config(
                "Grace hash join requires a configured MemoryBudget cap (refs #1524)".to_owned(),
            )
        })?;
        let available = cap.saturating_sub(budget.current_bytes(tenant)).max(1);
        let build_limit = (available / 2).max(1);
        let partitions = choose_partition_count(
            self.target.estimated_build_bytes,
            build_limit,
            self.target.estimated_key_cardinality,
        );
        self.target
            .telemetry
            .configured(partitions, self.target.estimated_key_cardinality);
        let handles = RunHandleBudget::default();

        let initial_runs = partition_build_input(
            left,
            left_shared_indices,
            partitions,
            &self.target.query,
            &handles,
            &self.target.telemetry,
            ctx,
            substrate,
        )?;
        let recursive_fanout = partitions.clamp(2, 8);
        let mut roots = Vec::with_capacity(partitions);
        let mut leaves = Vec::new();
        for run in initial_runs {
            let node = match run {
                Some(run) => plan_build_partition(
                    run,
                    0,
                    build_limit,
                    recursive_fanout,
                    self.target.max_repartition_depth,
                    left_shared_indices,
                    &self.target.query,
                    &handles,
                    &self.target.telemetry,
                    &mut leaves,
                    ctx,
                )?,
                None => BuildNode::Empty,
            };
            roots.push(node);
        }
        let plan = BuildPlan { roots, leaves };
        self.tasks = partition_probe_input(
            right,
            right_shared_indices,
            partitions,
            plan,
            &self.target.query,
            &handles,
            &self.target.telemetry,
            ctx,
            substrate,
        )?;
        Ok(())
    }

    pub(super) const fn is_done(&self) -> bool {
        self.done
    }
}

fn choose_partition_count(
    estimated_build_bytes: u64,
    build_limit: u64,
    estimated_key_cardinality: u64,
) -> usize {
    let required = estimated_build_bytes.max(1).div_ceil(build_limit.max(1));
    let required = usize::try_from(required).unwrap_or(usize::MAX);
    let rounded = required
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    let useful_from_cardinality = usize::try_from(estimated_key_cardinality.max(1))
        .unwrap_or(usize::MAX)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    rounded
        .min(useful_from_cardinality)
        .min(MAX_GRACE_PARTITIONS)
}

#[derive(Clone, Default)]
struct RunHandleBudget {
    live: Arc<AtomicUsize>,
}

impl RunHandleBudget {
    fn acquire(&self) -> Result<RunHandlePermit, ExecutionError> {
        loop {
            let live = self.live.load(Ordering::Acquire);
            if live >= MAX_GRACE_LIVE_RUN_HANDLES {
                return Err(run_handle_limit_error(live));
            }
            if self
                .live
                .compare_exchange_weak(live, live + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(RunHandlePermit {
                    live: Arc::clone(&self.live),
                });
            }
        }
    }
}

struct RunHandlePermit {
    live: Arc<AtomicUsize>,
}

impl Drop for RunHandlePermit {
    fn drop(&mut self) {
        // A permit is unique and moves with its writer/run/reader. Keep the
        // release defensive so an internal double-drop cannot underflow the
        // counter and disable subsequent admission checks.
        let _ = self
            .live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_sub(1)
            });
    }
}

struct PendingRun {
    writer: SpillRunWriter,
    _handle: RunHandlePermit,
    rows: u64,
    planning_bytes: u64,
    buffered_rows: Vec<Vec<Value>>,
    buffered_planning_bytes: u64,
}

struct RowRun {
    run: SpillRun,
    handle: RunHandlePermit,
    rows: u64,
    planning_bytes: u64,
}

fn empty_writer_slots(count: usize) -> Vec<Option<PendingRun>> {
    std::iter::repeat_with(|| None).take(count).collect()
}

fn append_to_slot(
    slots: &mut [Option<PendingRun>],
    index: usize,
    query: &SpillQuery,
    handles: &RunHandleBudget,
    row: Vec<Value>,
    planning_bytes: u64,
) -> Result<(), ExecutionError> {
    let slot = slots
        .get_mut(index)
        .ok_or_else(|| invalid_state("Grace partition index outside writer set"))?;
    if slot.is_none() {
        *slot = Some(new_pending_run(query, handles)?);
    }
    let pending = slot
        .as_mut()
        .ok_or_else(|| invalid_state("Grace partition writer was not initialized"))?;
    append_to_pending(pending, row, planning_bytes)
}

fn append_to_pending(
    pending: &mut PendingRun,
    row: Vec<Value>,
    planning_bytes: u64,
) -> Result<(), ExecutionError> {
    let next_rows = pending
        .rows
        .checked_add(1)
        .ok_or_else(|| identity_error("partition row count"))?;
    pending.buffered_rows.try_reserve(1).map_err(|_| {
        codec_error(SortSpillCodecError::AllocationFailed {
            kind: "Grace partition frame row buffer",
            count: 1,
        })
    })?;
    pending.buffered_rows.push(row);
    pending.rows = next_rows;
    pending.planning_bytes = pending.planning_bytes.saturating_add(planning_bytes);
    pending.buffered_planning_bytes = pending
        .buffered_planning_bytes
        .saturating_add(planning_bytes);
    if pending.buffered_planning_bytes >= GRACE_PARTITION_FRAME_TARGET_BYTES
        || pending.buffered_rows.len() >= BATCH_ROWS
    {
        flush_pending(pending)?;
    }
    Ok(())
}

fn new_pending_run(
    query: &SpillQuery,
    handles: &RunHandleBudget,
) -> Result<PendingRun, ExecutionError> {
    // Admit before opening the fd. On create failure the local permit drops,
    // so no phantom handle charge survives an OOC-1 error.
    let handle = handles.acquire()?;
    let writer = query.create_run()?;
    Ok(PendingRun {
        writer,
        _handle: handle,
        rows: 0,
        planning_bytes: 0,
        buffered_rows: Vec::new(),
        buffered_planning_bytes: 0,
    })
}

fn flush_pending(pending: &mut PendingRun) -> Result<(), ExecutionError> {
    if pending.buffered_rows.is_empty() {
        return Ok(());
    }
    let rows = std::mem::take(&mut pending.buffered_rows);
    pending.buffered_planning_bytes = 0;
    let encoded = encode_join_rows(&rows).map_err(codec_error)?;
    pending.writer.append_batch(&encoded)?;
    Ok(())
}

fn finish_pending(mut pending: PendingRun) -> Result<RowRun, ExecutionError> {
    flush_pending(&mut pending)?;
    let PendingRun {
        writer,
        _handle: handle,
        rows,
        planning_bytes,
        buffered_rows: _,
        buffered_planning_bytes: _,
    } = pending;
    Ok(RowRun {
        run: writer.finish()?,
        handle,
        rows,
        planning_bytes,
    })
}

fn finish_slots(slots: Vec<Option<PendingRun>>) -> Result<Vec<Option<RowRun>>, ExecutionError> {
    let mut runs = Vec::with_capacity(slots.len());
    for slot in slots {
        runs.push(match slot {
            Some(pending) => Some(finish_pending(pending)?),
            None => None,
        });
    }
    Ok(runs)
}

fn flush_slots(slots: &mut [Option<PendingRun>]) -> Result<(), ExecutionError> {
    for pending in slots.iter_mut().flatten() {
        flush_pending(pending)?;
    }
    Ok(())
}

fn flush_probe_writers(writers: &mut [Vec<Option<PendingRun>>]) -> Result<(), ExecutionError> {
    for leaf_writers in writers {
        flush_slots(leaf_writers)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn partition_build_input<S: ExecutorSubstrate>(
    left: &mut PhysicalOperator,
    shared_indices: &[usize],
    partitions: usize,
    query: &SpillQuery,
    handles: &RunHandleBudget,
    telemetry: &GraceTelemetry,
    ctx: &ExecutionContext,
    substrate: &S,
) -> Result<Vec<Option<RowRun>>, ExecutionError> {
    let mut slots = empty_writer_slots(partitions);
    let mut buffered_planning_bytes = 0_u64;
    loop {
        ctx.cancellation().check()?;
        let batch = left.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        let rows = batch.into_rows();
        telemetry.partition_batch(measure_rows_buffer(&rows));
        for row in rows {
            let Some(fingerprint) = join_key_fingerprint(&row, shared_indices) else {
                continue;
            };
            let index = partition_index(&fingerprint, INITIAL_PARTITION_SEED, partitions);
            let bytes = planning_build_row_bytes(&row, &fingerprint);
            append_to_slot(&mut slots, index, query, handles, row, bytes)?;
            buffered_planning_bytes = buffered_planning_bytes.saturating_add(bytes);
            if buffered_planning_bytes >= GRACE_PARTITION_FRAME_TARGET_BYTES {
                flush_slots(&mut slots)?;
                buffered_planning_bytes = 0;
            }
        }
    }
    let runs = finish_slots(slots)?;
    for run in &runs {
        if run.is_some() {
            telemetry.initial_build_run();
        }
    }
    Ok(runs)
}

enum BuildNode {
    Empty,
    Branch { seed: u64, children: Vec<BuildNode> },
    Leaf(usize),
}

impl BuildNode {
    fn route(&self, fingerprint: &str) -> Result<Option<usize>, ExecutionError> {
        match self {
            Self::Empty => Ok(None),
            Self::Leaf(index) => Ok(Some(*index)),
            Self::Branch { seed, children } => {
                let index = partition_index(fingerprint, *seed, children.len());
                let child = children
                    .get(index)
                    .ok_or_else(|| invalid_state("Grace recursive partition index missing"))?;
                child.route(fingerprint)
            }
        }
    }
}

struct BuildLeaf {
    blocks: Vec<RowRun>,
}

struct BuildPlan {
    roots: Vec<BuildNode>,
    leaves: Vec<BuildLeaf>,
}

#[allow(clippy::too_many_arguments)]
fn plan_build_partition(
    run: RowRun,
    depth: u8,
    build_limit: u64,
    fanout: usize,
    max_depth: u8,
    shared_indices: &[usize],
    query: &SpillQuery,
    handles: &RunHandleBudget,
    telemetry: &GraceTelemetry,
    leaves: &mut Vec<BuildLeaf>,
    ctx: &ExecutionContext,
) -> Result<BuildNode, ExecutionError> {
    if run.planning_bytes <= build_limit {
        let index = leaves.len();
        leaves.push(BuildLeaf { blocks: vec![run] });
        return Ok(BuildNode::Leaf(index));
    }
    if depth >= max_depth {
        telemetry.block_fallback();
        let blocks = split_build_blocks(
            run,
            build_limit,
            shared_indices,
            query,
            handles,
            telemetry,
            ctx,
        )?;
        let index = leaves.len();
        leaves.push(BuildLeaf { blocks });
        return Ok(BuildNode::Leaf(index));
    }

    let next_depth = depth.saturating_add(1);
    telemetry.recursion_depth(next_depth);
    let seed = recursive_seed(next_depth);
    let child_runs = repartition_build_run(
        run,
        seed,
        fanout,
        shared_indices,
        query,
        handles,
        telemetry,
        ctx,
    )?;
    let mut children = Vec::with_capacity(fanout);
    for child in child_runs {
        children.push(match child {
            Some(child) => plan_build_partition(
                child,
                next_depth,
                build_limit,
                fanout,
                max_depth,
                shared_indices,
                query,
                handles,
                telemetry,
                leaves,
                ctx,
            )?,
            None => BuildNode::Empty,
        });
    }
    Ok(BuildNode::Branch { seed, children })
}

#[allow(clippy::too_many_arguments)]
fn repartition_build_run(
    run: RowRun,
    seed: u64,
    fanout: usize,
    shared_indices: &[usize],
    query: &SpillQuery,
    handles: &RunHandleBudget,
    telemetry: &GraceTelemetry,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<RowRun>>, ExecutionError> {
    let mut cursor = RowRunCursor::open(run, query.epoch())?;
    let mut slots = empty_writer_slots(fanout);
    let mut buffered_planning_bytes = 0_u64;
    while let Some(row) = cursor.next_row()? {
        ctx.cancellation().check()?;
        let fingerprint = join_key_fingerprint(&row, shared_indices).ok_or_else(|| {
            invalid_state("Grace build partition restored a suppressed NULL/NaN key")
        })?;
        let index = partition_index(&fingerprint, seed, fanout);
        let bytes = planning_build_row_bytes(&row, &fingerprint);
        append_to_slot(&mut slots, index, query, handles, row, bytes)?;
        buffered_planning_bytes = buffered_planning_bytes.saturating_add(bytes);
        if buffered_planning_bytes >= GRACE_PARTITION_FRAME_TARGET_BYTES {
            flush_slots(&mut slots)?;
            buffered_planning_bytes = 0;
        }
    }
    drop(cursor);
    let runs = finish_slots(slots)?;
    for run in &runs {
        if run.is_some() {
            telemetry.recursive_run();
        }
    }
    Ok(runs)
}

#[allow(clippy::too_many_arguments)]
fn split_build_blocks(
    run: RowRun,
    build_limit: u64,
    shared_indices: &[usize],
    query: &SpillQuery,
    handles: &RunHandleBudget,
    telemetry: &GraceTelemetry,
    ctx: &ExecutionContext,
) -> Result<Vec<RowRun>, ExecutionError> {
    let mut cursor = RowRunCursor::open(run, query.epoch())?;
    let mut active: Option<PendingRun> = None;
    let mut blocks = Vec::new();
    while let Some(row) = cursor.next_row()? {
        ctx.cancellation().check()?;
        let fingerprint = join_key_fingerprint(&row, shared_indices).ok_or_else(|| {
            invalid_state("Grace block fallback restored a suppressed NULL/NaN key")
        })?;
        let bytes = planning_build_row_bytes(&row, &fingerprint);
        let should_finish = active.as_ref().is_some_and(|pending| {
            pending.rows > 0 && pending.planning_bytes.saturating_add(bytes) > build_limit
        });
        if should_finish {
            if let Some(pending) = active.take() {
                blocks.push(finish_pending(pending)?);
                telemetry.recursive_run();
            }
        }
        if active.is_none() {
            active = Some(new_pending_run(query, handles)?);
        }
        let pending = active
            .as_mut()
            .ok_or_else(|| invalid_state("Grace build block writer missing"))?;
        append_to_pending(pending, row, bytes)?;
    }
    drop(cursor);
    if let Some(pending) = active {
        blocks.push(finish_pending(pending)?);
        telemetry.recursive_run();
    }
    if blocks.is_empty() {
        return Err(invalid_state(
            "oversized Grace build partition produced no fallback blocks",
        ));
    }
    Ok(blocks)
}

#[allow(clippy::too_many_arguments)]
fn partition_probe_input<S: ExecutorSubstrate>(
    right: &mut PhysicalOperator,
    shared_indices: &[usize],
    partitions: usize,
    mut plan: BuildPlan,
    query: &SpillQuery,
    handles: &RunHandleBudget,
    telemetry: &GraceTelemetry,
    ctx: &ExecutionContext,
    substrate: &S,
) -> Result<VecDeque<JoinTask>, ExecutionError> {
    let mut writers: Vec<Vec<Option<PendingRun>>> = plan
        .leaves
        .iter()
        .map(|leaf| empty_writer_slots(leaf.blocks.len()))
        .collect();
    let mut root_spilled = vec![false; partitions];
    let mut buffered_planning_bytes = 0_u64;
    loop {
        ctx.cancellation().check()?;
        let batch = right.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        let rows = batch.into_rows();
        telemetry.partition_batch(measure_rows_buffer(&rows));
        for row in rows {
            let Some(fingerprint) = join_key_fingerprint(&row, shared_indices) else {
                continue;
            };
            let root_index = partition_index(&fingerprint, INITIAL_PARTITION_SEED, partitions);
            let root = plan
                .roots
                .get(root_index)
                .ok_or_else(|| invalid_state("Grace initial probe partition missing"))?;
            let Some(leaf_index) = root.route(&fingerprint)? else {
                continue;
            };
            if let Some(spilled) = root_spilled.get_mut(root_index) {
                *spilled = true;
            } else {
                return Err(invalid_state(
                    "Grace initial probe partition observer index missing",
                ));
            }
            let bytes = allocated_row_bytes(&row);
            let block_count = writers
                .get(leaf_index)
                .ok_or_else(|| invalid_state("Grace probe leaf writer set missing"))?
                .len();
            for block_index in 0..block_count {
                let leaf_writers = writers
                    .get_mut(leaf_index)
                    .ok_or_else(|| invalid_state("Grace probe leaf writer set missing"))?;
                append_to_slot(
                    leaf_writers,
                    block_index,
                    query,
                    handles,
                    row.clone(),
                    bytes,
                )?;
                buffered_planning_bytes = buffered_planning_bytes.saturating_add(bytes);
                if buffered_planning_bytes >= GRACE_PARTITION_FRAME_TARGET_BYTES {
                    flush_probe_writers(&mut writers)?;
                    buffered_planning_bytes = 0;
                }
            }
        }
    }
    for spilled in root_spilled {
        if spilled {
            telemetry.initial_probe_partition();
        }
    }

    let mut tasks = VecDeque::new();
    if writers.len() != plan.leaves.len() {
        return Err(invalid_state(
            "Grace probe writer/leaf cardinality diverged",
        ));
    }
    for (leaf, leaf_writers) in plan.leaves.drain(..).zip(writers) {
        if leaf.blocks.len() != leaf_writers.len() {
            return Err(invalid_state(
                "Grace build-block/probe-writer cardinality diverged",
            ));
        }
        for (build, pending_probe) in leaf.blocks.into_iter().zip(leaf_writers) {
            if let Some(pending) = pending_probe {
                let probe = finish_pending(pending)?;
                telemetry.initial_probe_run();
                tasks.push_back(JoinTask { build, probe });
            }
        }
    }
    Ok(tasks)
}

struct JoinTask {
    build: RowRun,
    probe: RowRun,
}

struct PendingProbe {
    row: Vec<Value>,
    fingerprint: String,
    next_match: usize,
}

struct ActiveJoin {
    table: BuildTable,
    probe: RowRunCursor,
    current_probe: Option<PendingProbe>,
    _reservation: MemoryReservation,
    table_bytes: u64,
    telemetry: GraceTelemetry,
}

impl ActiveJoin {
    fn open(
        task: JoinTask,
        epoch: QueryEpoch,
        left_shared_indices: &[usize],
        budget: &MemoryBudget,
        tenant: TenantId,
        telemetry: GraceTelemetry,
    ) -> Result<Self, ExecutionError> {
        let JoinTask { build, probe } = task;
        // `planning_build_row_bytes` charges a table slot and key allocation
        // per row (including duplicates), so the sealed run's accumulated
        // planning bytes conservatively cover the live HashMap. Reserve that
        // amount before decoding or allocating the table, never afterwards.
        let planned_table_bytes = build.planning_bytes;
        let reservation =
            budget.try_reserve(tenant, planned_table_bytes, "HashJoinOp Grace build table")?;
        let table = {
            let mut cursor = RowRunCursor::open(build, epoch)?;
            let mut table = BuildTable::new();
            while let Some(row) = cursor.next_row()? {
                let fingerprint = join_key_fingerprint(&row, left_shared_indices)
                    .ok_or_else(|| invalid_state("Grace join restored a suppressed build key"))?;
                table.entry(fingerprint).or_default().push(row);
            }
            table
        };
        let table_bytes = measure_build_table_bytes(&table);
        if table_bytes > planned_table_bytes {
            return Err(invalid_state(format!(
                "Grace build-table planning estimate undercharged live allocation: planned={planned_table_bytes}, measured={table_bytes}"
            )));
        }
        telemetry.build_table(table_bytes);
        let probe = RowRunCursor::open(probe, epoch)?;
        Ok(Self {
            table,
            probe,
            current_probe: None,
            _reservation: reservation,
            table_bytes,
            telemetry,
        })
    }

    fn next_joined(
        &mut self,
        right_shared_indices: &[usize],
        right_fresh_indices: &[usize],
    ) -> Result<Option<Vec<Value>>, ExecutionError> {
        loop {
            if self.current_probe.is_some() {
                let (joined, finished) =
                    {
                        let pending = self
                            .current_probe
                            .as_ref()
                            .ok_or_else(|| invalid_state("Grace pending probe disappeared"))?;
                        let bucket = self.table.get(&pending.fingerprint).ok_or_else(|| {
                            invalid_state("Grace pending probe bucket disappeared")
                        })?;
                        let left = bucket.get(pending.next_match).ok_or_else(|| {
                            invalid_state("Grace pending probe match cursor exceeded bucket")
                        })?;
                        let mut joined = left.clone();
                        for &index in right_fresh_indices {
                            let value = pending.row.get(index).ok_or_else(|| {
                                invalid_state("Grace probe row missing fresh column")
                            })?;
                            joined.push(value.clone());
                        }
                        (joined, pending.next_match.saturating_add(1) >= bucket.len())
                    };
                if finished {
                    self.current_probe = None;
                } else if let Some(pending) = self.current_probe.as_mut() {
                    pending.next_match = pending.next_match.saturating_add(1);
                }
                return Ok(Some(joined));
            }

            let Some(row) = self.probe.next_row()? else {
                return Ok(None);
            };
            let probe_row_bytes = allocated_row_bytes(&row);
            let probe_batch_bytes = self.probe.decoded_frame_bytes().max(probe_row_bytes);
            self.telemetry
                .active_probe(self.table_bytes, probe_row_bytes, probe_batch_bytes);
            let Some(fingerprint) = join_key_fingerprint(&row, right_shared_indices) else {
                continue;
            };
            if self.table.get(&fingerprint).is_none_or(Vec::is_empty) {
                continue;
            }
            self.current_probe = Some(PendingProbe {
                row,
                fingerprint,
                next_match: 0,
            });
        }
    }
}

struct RowRunCursor {
    reader: SpillRunReader,
    _handle: RunHandlePermit,
    pending: std::vec::IntoIter<Vec<Value>>,
    decoded_frame_bytes: u64,
    remaining_rows: u64,
}

impl RowRunCursor {
    fn open(run: RowRun, epoch: QueryEpoch) -> Result<Self, ExecutionError> {
        let RowRun {
            run,
            handle,
            rows,
            planning_bytes: _,
        } = run;
        Ok(Self {
            reader: run.into_reader(epoch)?,
            _handle: handle,
            pending: Vec::new().into_iter(),
            decoded_frame_bytes: 0,
            remaining_rows: rows,
        })
    }

    const fn decoded_frame_bytes(&self) -> u64 {
        self.decoded_frame_bytes
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value>>, ExecutionError> {
        loop {
            if let Some(row) = self.pending.next() {
                self.remaining_rows = self
                    .remaining_rows
                    .checked_sub(1)
                    .ok_or_else(|| invalid_state("Grace run decoded too many rows"))?;
                return Ok(Some(row));
            }
            let Some(batch) = self.reader.next_batch()? else {
                if self.remaining_rows != 0 {
                    return Err(invalid_state(format!(
                        "Grace run ended with {} declared rows missing",
                        self.remaining_rows
                    )));
                }
                return Ok(None);
            };
            let decoded = decode_join_rows(batch.as_ref()).map_err(codec_error)?;
            self.decoded_frame_bytes = measure_rows_buffer(&decoded);
            let decoded_len = u64::try_from(decoded.len())
                .map_err(|_| identity_error("decoded frame row count"))?;
            if decoded_len > self.remaining_rows {
                return Err(invalid_state(format!(
                    "Grace frame decoded {decoded_len} rows with only {} remaining",
                    self.remaining_rows
                )));
            }
            self.pending = decoded.into_iter();
        }
    }
}

fn recursive_seed(depth: u8) -> u64 {
    INITIAL_PARTITION_SEED
        ^ u64::from(depth)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(u32::from(depth) * 7)
}

fn partition_index(fingerprint: &str, seed: u64, partitions: usize) -> usize {
    if partitions <= 1 {
        return 0;
    }
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for byte in fingerprint.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        hash ^= hash >> 32;
    }
    (hash % partitions as u64) as usize
}

fn nested_value_heap_bytes(value: &Value) -> usize {
    estimate_value_bytes(value).saturating_sub(size_of::<Value>())
}

fn row_heap_bytes(row: &[Value], capacity: usize) -> u64 {
    let values = capacity.saturating_mul(size_of::<Value>());
    let nested = row.iter().map(nested_value_heap_bytes).sum::<usize>();
    u64::try_from(values.saturating_add(nested)).unwrap_or(u64::MAX)
}

fn allocated_row_bytes(row: &Vec<Value>) -> u64 {
    u64::try_from(size_of::<Vec<Value>>())
        .unwrap_or(u64::MAX)
        .saturating_add(row_heap_bytes(row, row.capacity()))
}

fn measure_rows_buffer(rows: &Vec<Vec<Value>>) -> u64 {
    let outer = size_of::<Vec<Vec<Value>>>()
        .saturating_add(rows.capacity().saturating_mul(size_of::<Vec<Value>>()));
    rows.iter()
        .fold(u64::try_from(outer).unwrap_or(u64::MAX), |total, row| {
            total.saturating_add(row_heap_bytes(row, row.capacity()))
        })
}

fn planning_build_row_bytes(row: &Vec<Value>, fingerprint: &str) -> u64 {
    // Conservatively charge a key/table slot for every row, including
    // duplicates. This makes the plan bound no smaller than the measured
    // table even when bucket/key sharing would reduce real occupancy.
    allocated_row_bytes(row)
        // String capacity can approach 2x its live length after geometric
        // growth. The fixed row slack covers the small-string minimum.
        .saturating_add(
            u64::try_from(fingerprint.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(2),
        )
        .saturating_add(
            u64::try_from(size_of::<(String, Vec<Vec<Value>>)>() + size_of::<usize>())
                .unwrap_or(u64::MAX),
        )
        .saturating_add(GRACE_BUILD_TABLE_ROW_SLACK_BYTES)
}

fn measure_build_table_bytes(table: &BuildTable) -> u64 {
    let entry_slots = table
        .capacity()
        .saturating_mul(size_of::<(String, Vec<Vec<Value>>)>() + size_of::<u8>());
    let mut total =
        u64::try_from(size_of::<BuildTable>().saturating_add(entry_slots)).unwrap_or(u64::MAX);
    for (key, bucket) in table {
        total = total
            .saturating_add(u64::try_from(key.capacity()).unwrap_or(u64::MAX))
            .saturating_add(
                u64::try_from(bucket.capacity().saturating_mul(size_of::<Vec<Value>>()))
                    .unwrap_or(u64::MAX),
            );
        for row in bucket {
            total = total.saturating_add(row_heap_bytes(row, row.capacity()));
        }
    }
    total
}

fn codec_error(error: SortSpillCodecError) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail: format!("Grace hash-join spill codec: {error}"),
    })
}

fn invalid_config(detail: String) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::InvalidConfig,
        detail,
    })
}

fn invalid_state(detail: impl Into<String>) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail: detail.into(),
    })
}

fn identity_error(subject: &'static str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Identity,
        detail: format!("Grace hash-join {subject} exhausted"),
    })
}

fn run_handle_limit_error(live: usize) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::ResourceLimit,
        detail: format!(
            "Grace hash-join live spill-run handle limit reached: live={live}, limit={MAX_GRACE_LIVE_RUN_HANDLES}"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_choice_is_budget_derived_and_bounded() {
        assert_eq!(choose_partition_count(1_024, 256, 100), 4);
        assert_eq!(choose_partition_count(1, 256, 1), 1);
        assert_eq!(choose_partition_count(1_024, 256, 1), 1);
        assert_eq!(choose_partition_count(1_024, 256, 2), 2);
        assert_eq!(
            choose_partition_count(u64::MAX, 1, u64::MAX),
            MAX_GRACE_PARTITIONS
        );
    }

    #[test]
    fn reseeding_is_deterministic_and_depth_specific() {
        let key = "I:42";
        assert_eq!(
            partition_index(key, INITIAL_PARTITION_SEED, 16),
            partition_index(key, INITIAL_PARTITION_SEED, 16)
        );
        assert_ne!(recursive_seed(1), recursive_seed(2));
    }
}
