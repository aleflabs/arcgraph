//! OOC-4 FIFO spill queue shared by expand and optional-expand.
//!
//! The queue is deliberately row-shaped: OOC-1 owns opaque encrypted frames,
//! while the query crate owns the complete [`Value`] codec. A resident prefix
//! is followed by one sequential spill run. While that run is read, newly
//! enqueued rows are appended to a new tail writer; after the reader drains,
//! the tail is sealed and promoted. This ping-pong is the FIFO invariant:
//! newer rows can never jump ahead of either the resident prefix or an older
//! run, with quarter-batch frames leaving room for codec and OOC-1 copies.

use std::collections::VecDeque;
use std::mem::size_of;
#[cfg(feature = "fault-injection")]
use std::sync::Arc;
#[cfg(feature = "fault-injection")]
use std::sync::Mutex;

use arcgraph_core::TenantId;
use arcgraph_storage::{SpillQuery, SpillRunReader, SpillRunWriter};

use crate::executor::BATCH_ROWS;
use crate::executor::budget::MemoryBudget;
use crate::executor::context::ExecutionContext;
use crate::executor::error::{ExecutionError, ExecutorSpillError, ExecutorSpillFailureKind};
use crate::executor::value::{NodeView, PathSegment, RelView, Value};
use crate::semantic::error::ArcQLError;

use super::sort_spill_codec::{SortSpillCodecError, decode_expand_rows, encode_expand_rows};

// At the write peak, one decoded reader frame, the pending writer frame, its
// encoded representation, and OOC-1's encrypted/raw payload copy can coexist.
// Quarter-batch frames therefore keep those four row populations within the
// one-batch slack contract. The byte target is likewise one quarter of the
// 256-KiB executor spill-batch target; a single larger row is admitted as the
// unavoidable one-row slack and is reflected in `batch_slack_limit_bytes`.
const EXPAND_SPILL_FRAME_ROWS: usize = if BATCH_ROWS < 4 { 1 } else { BATCH_ROWS / 4 };
const EXPAND_SPILL_FRAME_BYTES: u64 = 64 * 1024;
#[cfg(feature = "fault-injection")]
const EXPAND_SPILL_LIVE_FRAME_COPIES: u64 = 4;
const EXPAND_SPILL_CODEC_FIXED_BYTES: u64 = 64;

/// OOC-1 target owned by one expand/optional-expand execution.
pub struct ExpandSpillTarget {
    query: SpillQuery,
    telemetry: ExpandTelemetry,
}

impl std::fmt::Debug for ExpandSpillTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpandSpillTarget")
            .field("query", &self.query)
            .finish_non_exhaustive()
    }
}

impl ExpandSpillTarget {
    /// Bind a live OOC-1 query to the FIFO expand spillover queue.
    #[must_use]
    pub fn new(query: SpillQuery) -> Self {
        Self {
            query,
            telemetry: ExpandTelemetry::default(),
        }
    }

    /// Attach the release-gate real-occupancy observer.
    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub fn with_probe(mut self, probe: ExpandSpillProbe) -> Self {
        self.telemetry.probe = Some(probe);
        self
    }
}

/// Measurements from the live FIFO buffers, available only in fault builds.
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpandSpillStats {
    /// Peak measured resident-prefix allocation.
    pub peak_resident_frontier_bytes: u64,
    /// Peak measured resident prefix + read/write staging allocation.
    pub peak_frontier_bytes: u64,
    /// Peak measured decoded/writer frames plus codec/OOC-1 transient copies.
    pub peak_batch_slack_bytes: u64,
    /// Peak encoded/raw transient bytes included in `peak_batch_slack_bytes`.
    pub peak_transient_frame_bytes: u64,
    /// Largest measured row payload admitted to the queue.
    pub max_frontier_row_bytes: u64,
    /// Conservative hard slack bound derived from the frame targets and the
    /// largest admitted row. `peak_frontier_bytes` must fit within the
    /// configured budget plus this value.
    pub batch_slack_limit_bytes: u64,
    pub rows_spilled: u64,
    pub rows_rehydrated: u64,
    pub runs_created: u64,
}

/// Shared cfg-only probe for the OOC-4 release gates.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Default)]
pub struct ExpandSpillProbe {
    inner: Arc<Mutex<ExpandSpillStats>>,
}

#[cfg(feature = "fault-injection")]
impl ExpandSpillProbe {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn snapshot(&self) -> ExpandSpillStats {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, update: impl FnOnce(&mut ExpandSpillStats)) {
        update(
            &mut self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
}

#[derive(Clone, Default)]
struct ExpandTelemetry {
    #[cfg(feature = "fault-injection")]
    probe: Option<ExpandSpillProbe>,
}

impl ExpandTelemetry {
    fn occupancy(&self, resident: u64, staging: u64, transient: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                let live_slack = staging.saturating_add(transient);
                stats.peak_resident_frontier_bytes =
                    stats.peak_resident_frontier_bytes.max(resident);
                stats.peak_batch_slack_bytes = stats.peak_batch_slack_bytes.max(live_slack);
                stats.peak_transient_frame_bytes = stats.peak_transient_frame_bytes.max(transient);
                stats.batch_slack_limit_bytes = stats.batch_slack_limit_bytes.max(live_slack);
                stats.peak_frontier_bytes = stats
                    .peak_frontier_bytes
                    .max(resident.saturating_add(live_slack));
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = (resident, staging, transient);
    }

    fn observed_row(&self, payload: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.max_frontier_row_bytes = stats.max_frontier_row_bytes.max(payload);
                let frame_slots =
                    u64::try_from(EXPAND_SPILL_FRAME_ROWS.saturating_mul(size_of::<Vec<Value>>()))
                        .unwrap_or(u64::MAX);
                let one_frame = EXPAND_SPILL_FRAME_BYTES
                    .saturating_add(stats.max_frontier_row_bytes)
                    .saturating_add(frame_slots)
                    .saturating_add(EXPAND_SPILL_CODEC_FIXED_BYTES);
                stats.batch_slack_limit_bytes = stats
                    .batch_slack_limit_bytes
                    .max(one_frame.saturating_mul(EXPAND_SPILL_LIVE_FRAME_COPIES));
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = payload;
    }

    fn spilled(&self, rows: usize) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.rows_spilled = stats.rows_spilled.saturating_add(rows as u64);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = rows;
    }

    fn rehydrated(&self, rows: usize) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.rows_rehydrated = stats.rows_rehydrated.saturating_add(rows as u64);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = rows;
    }

    fn run_created(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.runs_created = stats.runs_created.saturating_add(1);
            });
        }
    }
}

struct ResidentRow {
    row: Vec<Value>,
    payload_charge: u64,
}

struct TailWriter {
    writer: SpillRunWriter,
    pending: Vec<Vec<Value>>,
    pending_payload_bytes: u64,
}

struct HeadReader {
    reader: SpillRunReader,
    decoded: VecDeque<Vec<Value>>,
    decoded_payload_bytes: u64,
}

/// Exact FIFO with a budget-charged resident prefix and OOC-1 tail.
pub(super) struct ExpandSpillQueue {
    resident: VecDeque<ResidentRow>,
    resident_payload_bytes: u64,
    /// Budget charge for the resident VecDeque's measured slot capacity.
    resident_slot_charge: u64,
    writer: Option<TailWriter>,
    reader: Option<HeadReader>,
    len: usize,
    budget: Option<MemoryBudget>,
    tenant: Option<TenantId>,
    /// Declared last so writer/reader handles drop before SpillQuery ends the
    /// epoch and releases/zeroizes its final key owner.
    target: ExpandSpillTarget,
}

impl std::fmt::Debug for ExpandSpillQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpandSpillQueue")
            .field("resident_rows", &self.resident.len())
            .field("has_writer", &self.writer.is_some())
            .field("has_reader", &self.reader.is_some())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl ExpandSpillQueue {
    pub(super) fn new(target: ExpandSpillTarget) -> Self {
        Self {
            resident: VecDeque::new(),
            resident_payload_bytes: 0,
            resident_slot_charge: 0,
            writer: None,
            reader: None,
            len: 0,
            budget: None,
            tenant: None,
            target,
        }
    }

    /// Enforce #1524 before the operator pulls input or creates scratch.
    pub(super) fn prepare(&mut self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
        if self.budget.is_some() {
            if self.tenant != Some(ctx.tenant()) {
                return Err(invalid_config(
                    "expand spill queue cannot be reused across tenants".to_owned(),
                ));
            }
            return Ok(());
        }
        if ctx.budget().cap_bytes(ctx.tenant()).is_none() {
            return Err(invalid_config(
                "spillable expand requires a configured MemoryBudget cap (refs #1524)".to_owned(),
            ));
        }
        self.budget = Some(ctx.budget().clone());
        self.tenant = Some(ctx.tenant());
        self.observe_occupancy();
        Ok(())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn push(
        &mut self,
        ctx: &ExecutionContext,
        row: Vec<Value>,
    ) -> Result<(), ExecutionError> {
        self.prepare(ctx)?;
        if self.len == usize::MAX {
            return Err(resource_limit(
                "expand FIFO row-count capacity exhausted".to_owned(),
            ));
        }
        let payload = row_payload_bytes(&row, row.capacity());
        self.target.telemetry.observed_row(payload);

        // Once a disk tail exists, every newer row stays behind it even if
        // resident pops free budget. This branch is the arrival-order pin.
        if self.writer.is_some() || self.reader.is_some() {
            self.push_disk(row, payload)?;
            self.len = self.len.saturating_add(1);
            return Ok(());
        }

        let budget = self.budget_ref()?.clone();
        let tenant = self.tenant_value()?;
        match budget.try_reserve_unscoped(
            tenant,
            payload,
            "ExpandOp OOC-4 resident frontier payload",
        ) {
            Ok(()) => {
                match self.ensure_resident_slot() {
                    Ok(true) => {}
                    Ok(false) => {
                        budget.release(tenant, payload);
                        self.push_disk(row, payload)?;
                        self.len = self.len.saturating_add(1);
                        return Ok(());
                    }
                    Err(error) => {
                        budget.release(tenant, payload);
                        return Err(error);
                    }
                }
                // `ensure_resident_slot` has made `len < capacity`, so this
                // push cannot invoke the allocator.
                self.resident.push_back(ResidentRow {
                    row,
                    payload_charge: payload,
                });
                self.resident_payload_bytes = self.resident_payload_bytes.saturating_add(payload);
            }
            Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })) => {
                self.push_disk(row, payload)?;
            }
            Err(other) => return Err(other),
        }
        self.len = self.len.saturating_add(1);
        self.observe_occupancy();
        Ok(())
    }

    pub(super) fn pop(&mut self) -> Result<Option<Vec<Value>>, ExecutionError> {
        if self.len == 0 && !self.resident.is_empty() {
            return Err(invalid_state(
                "expand FIFO resident rows exist with a zero declared length",
            ));
        }
        if let Some(resident) = self.resident.pop_front() {
            self.release_charge(resident.payload_charge);
            self.resident_payload_bytes = self
                .resident_payload_bytes
                .saturating_sub(resident.payload_charge);
            self.len = self.len.saturating_sub(1);
            if self.resident.is_empty() {
                self.release_empty_resident_buffer();
            }
            self.observe_occupancy();
            return Ok(Some(resident.row));
        }

        loop {
            if let Some(head) = self.reader.as_mut() {
                if self.len == 0 && !head.decoded.is_empty() {
                    return Err(invalid_state(
                        "expand FIFO decoded rows exist with a zero declared length",
                    ));
                }
                if let Some(row) = head.decoded.pop_front() {
                    head.decoded_payload_bytes = head
                        .decoded_payload_bytes
                        .saturating_sub(row_payload_bytes(&row, row.capacity()));
                    self.len = self.len.saturating_sub(1);
                    self.observe_occupancy();
                    return Ok(Some(row));
                }
                match head.reader.next_batch()? {
                    Some(batch) => {
                        let raw_frame_bytes = u64::try_from(batch.len()).unwrap_or(u64::MAX);
                        let rows = decode_expand_rows(batch.as_ref()).map_err(codec_error)?;
                        if rows.is_empty() {
                            return Err(invalid_state(
                                "expand spill run contained an empty FIFO frame",
                            ));
                        }
                        if rows.len() > self.len {
                            return Err(invalid_state(
                                "expand spill frame exceeds the FIFO's declared remaining rows",
                            ));
                        }
                        head.decoded_payload_bytes = rows.iter().fold(0_u64, |total, row| {
                            total.saturating_add(row_payload_bytes(row, row.capacity()))
                        });
                        self.target.telemetry.rehydrated(rows.len());
                        head.decoded = rows.into();
                        // Conservatively include both the returned plaintext
                        // and OOC-1's ciphertext/plaintext handoff copy. The
                        // decoded frame itself is counted by `reader_bytes`.
                        self.observe_occupancy_with_transient(raw_frame_bytes.saturating_mul(2));
                        continue;
                    }
                    None => {
                        self.reader = None;
                    }
                }
            }

            if self.writer.is_some() {
                self.promote_writer()?;
                continue;
            }
            if self.len != 0 {
                return Err(invalid_state(
                    "expand FIFO declared rows remain without resident, writer, or reader state",
                ));
            }
            self.observe_occupancy();
            return Ok(None);
        }
    }

    fn push_disk(&mut self, row: Vec<Value>, payload: u64) -> Result<(), ExecutionError> {
        if self.writer.is_none() {
            self.writer = Some(TailWriter {
                writer: self.target.query.create_run()?,
                pending: Vec::new(),
                pending_payload_bytes: 0,
            });
        }
        let should_flush = {
            let tail = self
                .writer
                .as_mut()
                .ok_or_else(|| invalid_state("expand spill tail disappeared after run creation"))?;
            tail.pending.try_reserve(1).map_err(|_| {
                ExecutionError::Spill(ExecutorSpillError::Failure {
                    kind: ExecutorSpillFailureKind::ResourceLimit,
                    detail: "could not allocate expand spill staging row".to_owned(),
                })
            })?;
            tail.pending.push(row);
            tail.pending_payload_bytes = tail.pending_payload_bytes.saturating_add(payload);
            tail.pending.len() >= EXPAND_SPILL_FRAME_ROWS
                || tail.pending_payload_bytes >= EXPAND_SPILL_FRAME_BYTES
        };
        self.target.telemetry.spilled(1);
        self.observe_occupancy();
        if should_flush {
            self.flush_writer()?;
        }
        Ok(())
    }

    fn flush_writer(&mut self) -> Result<(), ExecutionError> {
        let Some(tail) = self.writer.as_ref() else {
            return Ok(());
        };
        if tail.pending.is_empty() {
            return Ok(());
        }
        let encoded = encode_expand_rows(&tail.pending).map_err(codec_error)?;
        let encoded_allocation = u64::try_from(encoded.capacity()).unwrap_or(u64::MAX);
        let raw_copy = u64::try_from(encoded.len())
            .unwrap_or(u64::MAX)
            .saturating_add(EXPAND_SPILL_CODEC_FIXED_BYTES);
        self.observe_occupancy_with_transient(encoded_allocation.saturating_add(raw_copy));
        let tail = self
            .writer
            .as_mut()
            .ok_or_else(|| invalid_state("expand spill tail disappeared during flush"))?;
        tail.writer.append_batch(&encoded)?;
        let pending = std::mem::take(&mut tail.pending);
        tail.pending_payload_bytes = 0;
        drop(pending);
        self.observe_occupancy();
        Ok(())
    }

    fn promote_writer(&mut self) -> Result<(), ExecutionError> {
        self.flush_writer()?;
        let tail = self
            .writer
            .take()
            .ok_or_else(|| invalid_state("expand spill promotion had no writer"))?;
        let run = tail.writer.finish()?;
        let reader = run.into_reader(self.target.query.epoch())?;
        self.target.telemetry.run_created();
        self.reader = Some(HeadReader {
            reader,
            decoded: VecDeque::new(),
            decoded_payload_bytes: 0,
        });
        self.observe_occupancy();
        Ok(())
    }

    fn budget_ref(&self) -> Result<&MemoryBudget, ExecutionError> {
        self.budget
            .as_ref()
            .ok_or_else(|| invalid_state("expand spill queue used before preparation"))
    }

    fn tenant_value(&self) -> Result<TenantId, ExecutionError> {
        self.tenant
            .ok_or_else(|| invalid_state("expand spill queue tenant missing after preparation"))
    }

    fn release_charge(&self, bytes: u64) {
        if let (Some(budget), Some(tenant)) = (&self.budget, self.tenant) {
            budget.release(tenant, bytes);
        }
    }

    /// `Ok(false)` means the configured budget cannot admit the predicted
    /// slot growth and the caller should place the row on disk instead.
    fn ensure_resident_slot(&mut self) -> Result<bool, ExecutionError> {
        if self.resident.len() < self.resident.capacity() {
            return Ok(true);
        }
        let old_capacity = self.resident.capacity();
        let target_capacity = if old_capacity == 0 {
            // `VecDeque` uses a small non-zero allocation floor. Predict it
            // before touching the allocator so a tight budget routes the row
            // to disk instead of admitting one slot and failing on allocator
            // slack after the allocation is already live.
            4
        } else {
            old_capacity.checked_mul(2).ok_or_else(|| {
                resource_limit("expand resident frontier capacity overflow".to_owned())
            })?
        };
        let predicted_slots = target_capacity.checked_sub(old_capacity).ok_or_else(|| {
            resource_limit("expand resident slot prediction underflow".to_owned())
        })?;
        let predicted_bytes = slot_bytes::<ResidentRow>(predicted_slots)?;
        let budget = self.budget_ref()?.clone();
        let tenant = self.tenant_value()?;
        match budget.try_reserve_unscoped(
            tenant,
            predicted_bytes,
            "ExpandOp OOC-4 resident frontier slots",
        ) {
            Ok(()) => {}
            Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })) => {
                return Ok(false);
            }
            Err(other) => return Err(other),
        }

        let additional = target_capacity
            .checked_sub(self.resident.len())
            .ok_or_else(|| resource_limit("expand resident reserve underflow".to_owned()))?;
        if let Err(error) = self.resident.try_reserve_exact(additional) {
            budget.release(tenant, predicted_bytes);
            return Err(resource_limit(format!(
                "could not allocate expand resident frontier slots: {error}"
            )));
        }

        let actual_slots = self.resident.capacity().checked_sub(old_capacity);
        // From this point the allocation is live. Attach the successful
        // predicted reservation before any further fallible arithmetic so a
        // typed error cannot leak budget accounting.
        self.resident_slot_charge = self.resident_slot_charge.saturating_add(predicted_bytes);
        let actual_slots = actual_slots
            .ok_or_else(|| resource_limit("expand resident capacity regressed".to_owned()))?;
        let actual_bytes = slot_bytes::<ResidentRow>(actual_slots)?;
        if actual_bytes > predicted_bytes {
            let extra = actual_bytes - predicted_bytes;
            // The predicted charge is already attached to the live
            // allocation. If this extra reservation fails, `?` reaches the
            // caller's terminal-error arm and queue Drop releases it.
            budget.try_reserve_unscoped(
                tenant,
                extra,
                "ExpandOp OOC-4 resident frontier allocator slack",
            )?;
            self.resident_slot_charge = self.resident_slot_charge.saturating_add(extra);
        } else if predicted_bytes > actual_bytes {
            let refund = predicted_bytes - actual_bytes;
            budget.release(tenant, refund);
            self.resident_slot_charge = self.resident_slot_charge.saturating_sub(refund);
        }
        Ok(true)
    }

    fn release_empty_resident_buffer(&mut self) {
        if !self.resident.is_empty() {
            return;
        }
        let resident = std::mem::take(&mut self.resident);
        drop(resident);
        let slot_charge = std::mem::take(&mut self.resident_slot_charge);
        self.release_charge(slot_charge);
    }

    fn observe_occupancy(&self) {
        self.observe_occupancy_with_transient(0);
    }

    fn observe_occupancy_with_transient(&self, transient: u64) {
        let resident = self
            .resident_payload_bytes
            .saturating_add(slot_bytes_saturating::<ResidentRow>(
                self.resident.capacity(),
            ));
        let writer = self.writer.as_ref().map_or(0, |tail| {
            tail.pending_payload_bytes
                .saturating_add(slot_bytes_saturating::<Vec<Value>>(tail.pending.capacity()))
        });
        let reader = self.reader.as_ref().map_or(0, |head| {
            head.decoded_payload_bytes
                .saturating_add(slot_bytes_saturating::<Vec<Value>>(head.decoded.capacity()))
        });
        self.target
            .telemetry
            .occupancy(resident, writer.saturating_add(reader), transient);
    }
}

impl Drop for ExpandSpillQueue {
    fn drop(&mut self) {
        let resident = std::mem::take(&mut self.resident);
        let remaining_payload_charge = resident
            .iter()
            .fold(0_u64, |total, row| total.saturating_add(row.payload_charge));
        drop(resident);
        let remaining_charge =
            remaining_payload_charge.saturating_add(std::mem::take(&mut self.resident_slot_charge));
        self.release_charge(remaining_charge);
        // Fields then drop in declaration order: writer/reader close their
        // runs, followed by SpillQuery ending the epoch and zeroizing its key.
    }
}

fn row_payload_bytes(row: &[Value], capacity: usize) -> u64 {
    slot_bytes_saturating::<Value>(capacity).saturating_add(
        row.iter().fold(0_u64, |total, value| {
            total.saturating_add(value_heap_bytes(value))
        }),
    )
}

fn value_heap_bytes(value: &Value) -> u64 {
    match value {
        Value::Null | Value::Boolean(_) | Value::Integer(_) | Value::Float(_) => 0,
        Value::String(value) => usize_bytes(value.capacity()),
        Value::Node(node) => node_heap_bytes(node),
        Value::Relationship(rel) => rel_heap_bytes(rel),
        Value::List(values) => slot_bytes_saturating::<Value>(values.capacity()).saturating_add(
            values.iter().fold(0_u64, |total, value| {
                total.saturating_add(value_heap_bytes(value))
            }),
        ),
        Value::Map(map) => map_heap_bytes(map),
        Value::Path(path) => node_heap_bytes(&path.start)
            .saturating_add(slot_bytes_saturating::<PathSegment>(
                path.segments.capacity(),
            ))
            .saturating_add(path.segments.iter().fold(0_u64, |total, segment| {
                total
                    .saturating_add(rel_heap_bytes(&segment.rel))
                    .saturating_add(node_heap_bytes(&segment.end))
            })),
        Value::Temporal(_)
        | Value::LocalDateTime(_)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::Decimal(_) => 0,
    }
}

fn node_heap_bytes(node: &NodeView) -> u64 {
    node.label_name
        .as_ref()
        .map_or(0, |name| usize_bytes(name.capacity()))
        .saturating_add(map_heap_bytes(&node.properties))
}

fn rel_heap_bytes(rel: &RelView) -> u64 {
    rel.rel_type_name
        .as_ref()
        .map_or(0, |name| usize_bytes(name.capacity()))
        .saturating_add(map_heap_bytes(&rel.properties))
}

fn map_heap_bytes(map: &std::collections::BTreeMap<String, Value>) -> u64 {
    map.iter().fold(0_u64, |total, (key, value)| {
        total
            .saturating_add(usize_bytes(key.capacity()))
            // Match the executor budget's conservative BTreeMap-node
            // allowance while measuring nested values by reference.
            .saturating_add(48)
            .saturating_add(usize_bytes(size_of::<Value>()))
            .saturating_add(value_heap_bytes(value))
    })
}

fn slot_bytes<T>(slots: usize) -> Result<u64, ExecutionError> {
    slots
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| resource_limit("expand frontier allocation size overflow".to_owned()))
}

fn slot_bytes_saturating<T>(slots: usize) -> u64 {
    slots
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX)
}

fn usize_bytes(bytes: usize) -> u64 {
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn codec_error(error: SortSpillCodecError) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail: format!("expand spill codec failure: {error}"),
    })
}

fn invalid_config(detail: String) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::InvalidConfig,
        detail,
    })
}

fn invalid_state(detail: &str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::FrontierState,
        detail: detail.to_owned(),
    })
}

fn resource_limit(detail: String) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::ResourceLimit,
        detail,
    })
}
