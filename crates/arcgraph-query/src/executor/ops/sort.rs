//! [`SortOp`] — ORDER BY operator (M4-63).
//!
//! Lowers from [`crate::logical_plan::LogicalSort`]. Without a spill target it
//! retains the legacy stable in-memory path. With
//! [`SortOp::with_spillover_target`], it builds memory-bounded sorted runs in
//! OOC-1 scratch and streams a stable k-way merge.
//!
//! # Stability
//!
//! Uses `slice::sort_by` (which is stable per the Rust std-lib
//! contract). Stability matters here because Cypher 9 §6.6 leaves the
//! relative order of equal-key rows implementation-defined; the
//! executor's choice is "preserve insertion order for ties" so that
//! a future SIMD / parallel-sort re-shape (M4-64b+) is constrained
//! to a stable algorithm — preventing the "deterministic-but-wrong"
//! regression class.
//!
//! # Memory budget enforcement
//!
//! Sort is a blocking operator: it MUST see every upstream row before
//! emitting the first sorted row. For tenants with a configured
//! [`crate::executor::MemoryBudget`] cap, each buffered row is
//! debited at insertion time; exceeding the cap surfaces
//! [`crate::semantic::error::ArcQLError::ResourceExhausted`]. For
//! unbudgeted tenants (uncapped budget = no memory limit), the buffer
//! grows with the actual cardinality, guarded against a true runaway by
//! [`crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS`] (#994 /
//! #980 lifted the old 131 072-row `SPILLOVER_MAX_ROWS` valve that
//! failed legitimate large `ORDER BY` above ~100 K rows). Exceeding the
//! guard surfaces the same `ResourceExhausted` variant (W12α fix-up
//! LOW-4 promoted it from `ExecutionError::Eval` so the byte-cap and
//! row-cap surfaces share an error class).
//!
//! # External-merge spillover (M6.2 OOC-2)
//!
//! On a budget reservation failure, the resident buffer is sorted by
//! `(ORDER BY keys, input ordinal)`, written as one OOC-1 run, and released
//! before input consumption continues. Runs are compacted online, so the
//! eager-unlinked OOC-1 run handles cannot become an unbounded fd list.
//! Drain uses a min-heap and opens at most the configured fan-in readers.
//!
//! # Forward-pin: TopK heap fusion (M4-72 / M4-64b)
//!
//! W12α fix-up NIT-4 (PR #277 retro): a `Sort → Limit(K)` upstream
//! of this operator is the canonical "top-K" idiom. The current
//! shape is `O(N log N)` sort + truncate; a future cost-walker pass
//! at M4-72 may rewrite to a `TopK(K)` operator that maintains a
//! min-heap of size K — `O(N log K)` time and `O(K)` memory.
//! Back-of-envelope: for K = 10, N = 1M the
//! speedup is ~6× (log₂(1M) ≈ 20; log₂(10) ≈ 3.3). The
//! cost-rewrite gate is the M4-51 cost walker's enumeration of
//! Sort+Limit pairs; the actual `TopK` operator is a separate
//! M4-64b inflection-point consumer. Pre-bound here so the future
//! M4-72 / M4-64b slice doesn't need to discover the optimization
//! from scratch.
//!
//! # 3VL NULL ordering
//!
//! Cypher 9 §6.6 specifies NULL sorts LAST in `ASC` and FIRST in
//! `DESC` (the "NULL is the largest value" convention). The
//! comparator implementation honors this.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 §2 D-28** — sort operator contract.
//! - **Cypher 9 §6.6** — ORDER BY semantics + NULL ordering.
//! - **W11Z #272 retro MED-3** — `SPILLOVER_MAX_ROWS` row-count cap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

#[cfg(feature = "fault-injection")]
use std::sync::Mutex;

use arcgraph_core::TenantId;
use arcgraph_storage::{SpillQuery, SpillRun, SpillRunReader, SpillRunWriter};

use crate::executor::batch::Batch;
use crate::executor::budget::{MemoryBudget, estimate_row_bytes};
use crate::executor::context::ExecutionContext;
use crate::executor::error::{ExecutionError, ExecutorSpillError, ExecutorSpillFailureKind};
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::SortDirection;
use crate::semantic::bound_ast::{BindingId, BoundExpression};
use crate::semantic::error::ArcQLError;

use super::sort_spill_codec::{
    SortSpillCodecError, SortSpillRecord, decode_records, encode_records,
};

/// Default merge fan-in.
///
/// Sixteen readers consume 16 fds and 16 × OOC-1's 8 KiB `BufReader` =
/// 128 KiB of base staging. OOC-2 targets ~256 KiB restored frames, so the
/// usual decoded-head footprint is about 4 MiB; one output writer adds one fd
/// and 8 KiB. This sits well below a conservative 64-fd executor allowance
/// while leaving room for WAL/data descriptors. Oversized single records are
/// still governed by OOC-1's staging-memory admission.
pub const DEFAULT_EXTERNAL_SORT_FAN_IN: usize = 16;

/// Hard ceiling preventing a caller from turning a configured fan-in into an
/// unbounded reader/fd allocation.
pub const MAX_EXTERNAL_SORT_FAN_IN: usize = 128;

const MIN_EXTERNAL_SORT_FAN_IN: usize = 2;
const SORT_SPILL_TARGET_FRAME_BYTES: usize = 256 * 1024;

/// OOC-1 target owned by one [`SortOp`] execution.
pub struct SortSpillTarget {
    query: SpillQuery,
    fan_in: usize,
    telemetry: SortTelemetry,
}

impl std::fmt::Debug for SortSpillTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortSpillTarget")
            .field("query", &self.query)
            .field("fan_in", &self.fan_in)
            .finish_non_exhaustive()
    }
}

impl SortSpillTarget {
    /// Bind a live OOC-1 query to an external sort.
    #[must_use]
    pub fn new(query: SpillQuery) -> Self {
        Self {
            query,
            fan_in: DEFAULT_EXTERNAL_SORT_FAN_IN,
            telemetry: SortTelemetry::default(),
        }
    }

    /// Override the reader fan-in. Values outside `2..=128` are rejected
    /// before any scratch run is created.
    pub fn with_merge_fan_in(mut self, fan_in: usize) -> Result<Self, ExecutionError> {
        if !(MIN_EXTERNAL_SORT_FAN_IN..=MAX_EXTERNAL_SORT_FAN_IN).contains(&fan_in) {
            return Err(ExecutionError::Spill(ExecutorSpillError::Failure {
                kind: ExecutorSpillFailureKind::InvalidConfig,
                detail: format!(
                    "external-sort fan-in must be in {MIN_EXTERNAL_SORT_FAN_IN}..={MAX_EXTERNAL_SORT_FAN_IN}, got {fan_in}"
                ),
            }));
        }
        self.fan_in = fan_in;
        Ok(self)
    }

    /// Attach the cfg-only real-occupancy/reader observer used by the M6.2
    /// release fault-injection gates.
    #[cfg(feature = "fault-injection")]
    #[must_use]
    pub fn with_probe(mut self, probe: ExternalSortProbe) -> Self {
        self.telemetry.probe = Some(probe);
        self
    }
}

/// Snapshot of actual external-sort events, available only in fault builds.
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExternalSortStats {
    pub peak_buffer_bytes: u64,
    pub initial_runs_created: u64,
    pub intermediate_runs_created: u64,
    pub merge_passes: u32,
    pub max_concurrent_readers: usize,
    pub max_live_runs: usize,
}

/// Shared cfg-only probe. It observes the counters used by production
/// control flow; it does not mirror or predict them.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Default)]
pub struct ExternalSortProbe {
    inner: Arc<Mutex<ExternalSortStats>>,
}

#[cfg(feature = "fault-injection")]
impl ExternalSortProbe {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn snapshot(&self) -> ExternalSortStats {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, update: impl FnOnce(&mut ExternalSortStats)) {
        update(
            &mut self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
}

#[derive(Clone, Default)]
struct SortTelemetry {
    #[cfg(feature = "fault-injection")]
    probe: Option<ExternalSortProbe>,
}

impl SortTelemetry {
    fn observe_buffer(&self, bytes: u64) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| stats.peak_buffer_bytes = stats.peak_buffer_bytes.max(bytes));
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = bytes;
    }

    fn initial_run(&self) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.initial_runs_created = stats.initial_runs_created.saturating_add(1);
            });
        }
    }

    fn intermediate_run(&self, pass: u32) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.intermediate_runs_created = stats.intermediate_runs_created.saturating_add(1);
                stats.merge_passes = stats.merge_passes.max(pass);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = pass;
    }

    fn observe_readers(&self, readers: usize, pass: u32) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| {
                stats.max_concurrent_readers = stats.max_concurrent_readers.max(readers);
                stats.merge_passes = stats.merge_passes.max(pass);
            });
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = (readers, pass);
    }

    fn observe_live_runs(&self, runs: usize) {
        #[cfg(feature = "fault-injection")]
        if let Some(probe) = &self.probe {
            probe.update(|stats| stats.max_live_runs = stats.max_live_runs.max(runs));
        }
        #[cfg(not(feature = "fault-injection"))]
        let _ = runs;
    }
}

/// One ORDER BY key — expression + direction.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Sort-key expression evaluated against each row.
    pub expr: BoundExpression,
    /// `Asc` (default) or `Desc`.
    pub direction: SortDirection,
}

/// ORDER BY operator (stable in-memory sort with budget tracking).
pub struct SortOp {
    child: Box<PhysicalOperator>,
    /// Sort keys in declared precedence (first key = primary).
    keys: Vec<SortKey>,
    /// Per-query parameter bag for expression evaluation.
    parameters: Parameters,
    /// Output schema (= input schema; sort preserves column shape).
    schema: Vec<BindingId>,
    /// Buffered rows. v1.0-alpha all-in-memory; M4-64b+ may spill to a
    /// tmp-file substrate when the budget cap is hit.
    buffer: Vec<Vec<Value>>,
    /// Have we drained the upstream + sorted the buffer?
    sorted: bool,
    /// Output cursor.
    cursor: usize,
    /// W12α fix-up MED-1 (PR #277 retro): total bytes reserved against
    /// the per-tenant memory budget by this operator. Released in
    /// [`Drop`] to prevent the long-running-tenant counter-drift class
    /// (a sequence of N successful sort queries left N row reservations
    /// in the tenant counter, eventually saturating the cap with false
    /// `ResourceExhausted` rejections).
    reserved_total: u64,
    /// Tenant captured on the first reservation. `None` until then;
    /// used by [`Drop`] to release [`Self::reserved_total`] against
    /// the right tenant slot.
    tenant_for_release: Option<TenantId>,
    /// Budget snapshot captured on the first reservation. `Arc`-shared
    /// with the [`ExecutionContext`] so the operator can release on
    /// drop without holding an `&ExecutionContext` borrow.
    budget_for_release: Option<MemoryBudget>,
    /// Present only when the caller lights the OOC-2 seam. The runtime owns
    /// the live SpillQuery, sealed runs, and final streaming merge.
    spill_runtime: Option<Box<ExternalSortRuntime>>,
    /// A partially consumed external sort is terminal after any spill/codec
    /// failure. Store the typed error so an accidental retry cannot resume
    /// from a shifted reader offset.
    terminal_error: Option<ExecutionError>,
}

impl std::fmt::Debug for SortOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortOp")
            .field("child", &self.child)
            .field("keys_count", &self.keys.len())
            .field("schema", &self.schema)
            .field("buffered_rows", &self.buffer.len())
            .field("sorted", &self.sorted)
            .field("external", &self.spill_runtime.is_some())
            .finish()
    }
}

impl SortOp {
    /// Construct a [`SortOp`] from a child + sort-key list.
    #[must_use]
    pub fn new(child: PhysicalOperator, keys: Vec<SortKey>) -> Self {
        let schema = child.schema().to_vec();
        Self {
            child: Box::new(child),
            keys,
            parameters: Parameters::new(),
            schema,
            buffer: Vec::new(),
            sorted: false,
            cursor: 0,
            reserved_total: 0,
            tenant_for_release: None,
            budget_for_release: None,
            spill_runtime: None,
            terminal_error: None,
        }
    }

    /// Record a successful reservation of `bytes` against `tenant`'s
    /// budget so [`Drop`] can release the running total. Snapshots the
    /// tenant + budget on first call.
    fn record_reservation(&mut self, ctx: &ExecutionContext, budget: &MemoryBudget, bytes: u64) {
        if self.tenant_for_release.is_none() {
            self.tenant_for_release = Some(ctx.tenant());
            self.budget_for_release = Some(budget.clone());
        }
        self.reserved_total = self.reserved_total.saturating_add(bytes);
    }

    /// Inject a per-query parameter bag.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Enable OOC-2 external merge sort with a live OOC-1 query target.
    /// Passing `None` preserves the legacy in-memory operator exactly.
    pub fn with_spillover_target(
        mut self,
        target: Option<SortSpillTarget>,
    ) -> Result<Self, ExecutionError> {
        self.spill_runtime = target.map(|target| Box::new(ExternalSortRuntime::new(target)));
        Ok(self)
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if !self.sorted {
            if self.spill_runtime.is_some() {
                self.materialize_external(ctx, substrate)?;
            } else {
                self.materialize_and_sort(ctx, substrate)?;
            }
        }
        if self
            .spill_runtime
            .as_ref()
            .is_some_and(|runtime| runtime.output.is_some())
        {
            return self.next_external_batch();
        }
        if self.cursor >= self.buffer.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut out = Batch::with_capacity(self.schema.len());
        let take = (self.buffer.len() - self.cursor).min(crate::executor::BATCH_ROWS);
        for row in &self.buffer[self.cursor..self.cursor + take] {
            if !out.push_row(row.clone()) {
                return Err(ExecutionError::Eval(
                    "SortOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(out)
    }

    fn materialize_external<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let mut runtime = self
            .spill_runtime
            .take()
            .expect("external runtime checked by caller");
        let result = self.materialize_external_inner(ctx, substrate, &mut runtime);
        match result {
            Ok(ExternalMaterialization::Resident) => {
                // No run was needed. Ending the unused spill query now keeps
                // its ephemeral key lifetime equal to the operator work.
                drop(runtime);
                self.sorted = true;
                Ok(())
            }
            Ok(ExternalMaterialization::Spilled) => {
                self.spill_runtime = Some(runtime);
                self.sorted = true;
                Ok(())
            }
            Err(error) => {
                // Dropping runtime closes every reader/run, then ends the
                // query epoch and zeroizes the OOC-1 key. Named cfg-retained
                // files become sweepable orphans; normal POSIX runs were
                // eager-unlinked already.
                drop(runtime);
                self.terminal_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn materialize_external_inner<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        runtime: &mut ExternalSortRuntime,
    ) -> Result<ExternalMaterialization, ExecutionError> {
        let budget = ctx.budget().clone();
        let tenant = ctx.tenant();
        let has_cap = budget.has_cap(tenant);
        let mut charge = ResidentBufferCharge::new(budget.clone(), tenant);
        let mut records = Vec::<SortSpillRecord>::new();
        let mut buffer_bytes = 0_u64;
        let mut next_ordinal = 0_u64;
        let directions: Arc<[SortDirection]> = self
            .keys
            .iter()
            .map(|key| key.direction)
            .collect::<Vec<_>>()
            .into();

        let lookup_schema = self.schema.clone();
        let lookup = move |binding: BindingId| schema_index(&lookup_schema, binding);

        loop {
            ctx.cancellation().check()?;
            let batch = self.child.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            for row in batch.into_rows() {
                let mut row_keys = Vec::with_capacity(self.keys.len());
                for key in &self.keys {
                    row_keys.push(evaluate(&key.expr, &row, &lookup, &self.parameters)?);
                }
                let bytes =
                    external_record_bytes(&row_keys, row_keys.capacity(), &row, row.capacity());
                let record = SortSpillRecord {
                    ordinal: next_ordinal,
                    keys: row_keys,
                    row,
                };
                next_ordinal = next_ordinal.checked_add(1).ok_or_else(|| {
                    ExecutionError::Spill(ExecutorSpillError::Failure {
                        kind: ExecutorSpillFailureKind::Identity,
                        detail: "external-sort input ordinal exhausted".to_owned(),
                    })
                })?;

                if has_cap {
                    match budget.try_reserve_unscoped(tenant, bytes, "SortOp external buffer") {
                        Ok(()) => charge.add(bytes),
                        Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })) => {
                            if !records.is_empty() {
                                runtime.spill_initial_run(
                                    &mut records,
                                    Arc::clone(&directions),
                                    ctx,
                                )?;
                                charge.release_all();
                                buffer_bytes = 0;
                            }
                            match budget.try_reserve_unscoped(
                                tenant,
                                bytes,
                                "SortOp external buffer",
                            ) {
                                Ok(()) => charge.add(bytes),
                                Err(ExecutionError::Plan(ArcQLError::ResourceExhausted {
                                    ..
                                })) => {
                                    // A single record (or other operators'
                                    // live reservations) can exceed the
                                    // remaining cap. Admit exactly this one
                                    // input-batch slack record, spill it
                                    // immediately, and never accumulate a
                                    // second uncharged record.
                                    records.push(record);
                                    buffer_bytes = bytes;
                                    runtime
                                        .target
                                        .telemetry
                                        .observe_buffer(measured_buffer_bytes(
                                            buffer_bytes,
                                            records.len(),
                                            records.capacity(),
                                        ));
                                    runtime.spill_initial_run(
                                        &mut records,
                                        Arc::clone(&directions),
                                        ctx,
                                    )?;
                                    buffer_bytes = 0;
                                    continue;
                                }
                                Err(other) => return Err(other),
                            }
                        }
                        Err(other) => return Err(other),
                    }
                } else if records.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                    return Err(row_count_fallback_err(records.len()));
                }

                records.push(record);
                buffer_bytes = buffer_bytes.saturating_add(bytes);
                runtime
                    .target
                    .telemetry
                    .observe_buffer(measured_buffer_bytes(
                        buffer_bytes,
                        records.len(),
                        records.capacity(),
                    ));
            }
        }

        if runtime.runs.is_empty() {
            sort_records(&mut records, &directions);
            self.buffer = records.into_iter().map(|record| record.row).collect();
            let reserved = charge.disarm();
            if reserved > 0 {
                self.record_reservation(ctx, &budget, reserved);
            }
            return Ok(ExternalMaterialization::Resident);
        }

        if !records.is_empty() {
            runtime.spill_initial_run(&mut records, Arc::clone(&directions), ctx)?;
            charge.release_all();
        }
        runtime.finish(Arc::clone(&directions))?;
        Ok(ExternalMaterialization::Spilled)
    }

    fn next_external_batch(&mut self) -> Result<Batch, ExecutionError> {
        // Keep every fallible reader/codec step inside a closure. A plain
        // block would let `?` return from `next_external_batch` directly and
        // bypass the cleanup/terminal-error arm below.
        let result = (|| {
            let runtime = self
                .spill_runtime
                .as_mut()
                .expect("external runtime present while draining");
            let output = runtime
                .output
                .as_mut()
                .expect("external merge present while draining");
            let mut out = Batch::with_capacity(self.schema.len());
            let mut exhausted = false;
            while !out.is_full() {
                match output.next_record()? {
                    Some(record) => {
                        if !out.push_row(record.row) {
                            return Err(ExecutionError::Eval(
                                "SortOp external merge batch overflow".to_owned(),
                            ));
                        }
                    }
                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }
            Ok::<_, ExecutionError>((out, exhausted))
        })();

        match result {
            Ok((out, exhausted)) => {
                if exhausted {
                    // Close final readers and zeroize the query key as soon
                    // as the last output row has moved into its Batch.
                    drop(self.spill_runtime.take());
                }
                Ok(out)
            }
            Err(error) => {
                drop(self.spill_runtime.take());
                self.terminal_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn materialize_and_sort<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        // Drain upstream into the buffer with budget tracking. Per-row
        // reservations recorded against `reserved_total` so [`Drop`]
        // releases them when the operator is dropped (per W12α fix-up
        // MED-1).
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());
        loop {
            ctx.cancellation().check()?;
            let batch = self.child.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            for row in batch.into_rows() {
                if has_cap {
                    let bytes = estimate_row_bytes(&row) as u64;
                    budget.try_reserve_unscoped(ctx.tenant(), bytes, "SortOp buffer")?;
                    self.record_reservation(ctx, &budget, bytes);
                } else if self.buffer.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                    return Err(row_count_fallback_err(self.buffer.len()));
                }
                self.buffer.push(row);
            }
        }
        // Sort. `sort_by` is stable per std-lib contract.
        let lookup_schema = self.schema.clone();
        let lookup = move |b: BindingId| schema_index(&lookup_schema, b);
        let keys = self.keys.clone();
        let parameters = self.parameters.clone();
        // Pre-compute sort keys per row (avoids re-evaluating during
        // O(n log n) comparisons). Each row's key is a `Vec<Value>`
        // mirroring `keys.len()`.
        let mut keyed: Vec<(Vec<Value>, Vec<Value>)> = Vec::with_capacity(self.buffer.len());
        for row in std::mem::take(&mut self.buffer) {
            let mut row_keys: Vec<Value> = Vec::with_capacity(keys.len());
            for k in &keys {
                row_keys.push(evaluate(&k.expr, &row, &lookup, &parameters)?);
            }
            keyed.push((row_keys, row));
        }
        keyed.sort_by(|(a_keys, _), (b_keys, _)| compare_key_tuples(a_keys, b_keys, &keys));
        self.buffer = keyed.into_iter().map(|(_k, r)| r).collect();
        self.sorted = true;
        Ok(())
    }
}

enum ExternalMaterialization {
    Resident,
    Spilled,
}

/// RAII accounting for the current run-generation buffer. It is disarmed
/// only when an all-resident result transfers ownership to `SortOp`; every
/// spill and every error path releases immediately.
struct ResidentBufferCharge {
    budget: MemoryBudget,
    tenant: TenantId,
    bytes: u64,
}

impl ResidentBufferCharge {
    fn new(budget: MemoryBudget, tenant: TenantId) -> Self {
        Self {
            budget,
            tenant,
            bytes: 0,
        }
    }

    fn add(&mut self, bytes: u64) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn release_all(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);
        if bytes > 0 {
            self.budget.release(self.tenant, bytes);
        }
    }

    fn disarm(&mut self) -> u64 {
        std::mem::take(&mut self.bytes)
    }
}

impl Drop for ResidentBufferCharge {
    fn drop(&mut self) {
        self.release_all();
    }
}

struct SortedRun {
    run: SpillRun,
    rows: u64,
    pass: u32,
}

struct ExternalSortRuntime {
    target: SortSpillTarget,
    runs: Vec<SortedRun>,
    output: Option<KWayMerge>,
}

impl ExternalSortRuntime {
    fn new(target: SortSpillTarget) -> Self {
        Self {
            target,
            runs: Vec::new(),
            output: None,
        }
    }

    fn spill_initial_run(
        &mut self,
        records: &mut Vec<SortSpillRecord>,
        directions: Arc<[SortDirection]>,
        ctx: &ExecutionContext,
    ) -> Result<(), ExecutionError> {
        if records.is_empty() {
            return Ok(());
        }
        sort_records(records, &directions);
        self.make_room(Arc::clone(&directions), ctx)?;

        let rows = u64::try_from(records.len()).map_err(|_| identity_error("run row count"))?;
        let mut writer = self.target.query.create_run()?;
        write_record_frames(&mut writer, records)?;
        let run = writer.finish()?;
        records.clear();
        self.runs.push(SortedRun { run, rows, pass: 0 });
        self.target.telemetry.initial_run();
        self.target.telemetry.observe_live_runs(self.runs.len());
        Ok(())
    }

    /// OOC-1 sealed runs retain their eager-unlinked fd. Before another run
    /// is created at capacity, merge the two smallest live runs. This keeps
    /// sealed handles <= F and produces balanced (Huffman-like) compaction
    /// instead of repeatedly rewriting one ever-growing run.
    fn make_room(
        &mut self,
        directions: Arc<[SortDirection]>,
        ctx: &ExecutionContext,
    ) -> Result<(), ExecutionError> {
        while self.runs.len() >= self.target.fan_in {
            let group = take_two_smallest(&mut self.runs)?;
            let merged = merge_runs_to_run(
                &self.target.query,
                group,
                Arc::clone(&directions),
                &self.target.telemetry,
                ctx,
            )?;
            self.runs.push(merged);
            self.target.telemetry.observe_live_runs(self.runs.len());
        }
        Ok(())
    }

    fn finish(&mut self, directions: Arc<[SortDirection]>) -> Result<(), ExecutionError> {
        if self.runs.len() > self.target.fan_in {
            return Err(invalid_spill_state(format!(
                "{} live runs exceed fan-in {}",
                self.runs.len(),
                self.target.fan_in
            )));
        }
        let runs = std::mem::take(&mut self.runs);
        let pass = runs
            .iter()
            .map(|run| run.pass)
            .max()
            .unwrap_or(0)
            .saturating_add(u32::from(runs.len() > 1));
        self.output = Some(KWayMerge::new(
            runs,
            self.target.query.epoch(),
            directions,
            &self.target.telemetry,
            pass,
        )?);
        Ok(())
    }
}

fn take_two_smallest(runs: &mut Vec<SortedRun>) -> Result<Vec<SortedRun>, ExecutionError> {
    if runs.len() < 2 {
        return Err(invalid_spill_state(
            "online compaction requires at least two runs".to_owned(),
        ));
    }
    let mut indices: Vec<usize> = (0..runs.len()).collect();
    indices.sort_by_key(|&index| (runs[index].rows, runs[index].pass));
    let first = indices[0];
    let second = indices[1];
    let high = first.max(second);
    let low = first.min(second);
    let high_run = runs.remove(high);
    let low_run = runs.remove(low);
    Ok(vec![low_run, high_run])
}

fn merge_runs_to_run(
    query: &SpillQuery,
    runs: Vec<SortedRun>,
    directions: Arc<[SortDirection]>,
    telemetry: &SortTelemetry,
    ctx: &ExecutionContext,
) -> Result<SortedRun, ExecutionError> {
    let rows = runs.iter().try_fold(0_u64, |total, run| {
        total
            .checked_add(run.rows)
            .ok_or_else(|| identity_error("merged row count"))
    })?;
    let pass = runs
        .iter()
        .map(|run| run.pass)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| identity_error("merge pass"))?;
    let mut writer = query.create_run()?;
    let mut merge = KWayMerge::new(runs, query.epoch(), directions, telemetry, pass)?;
    let mut output_records = Vec::with_capacity(crate::executor::BATCH_ROWS);
    while let Some(record) = merge.next_record()? {
        output_records.push(record);
        if output_records.len() == crate::executor::BATCH_ROWS {
            ctx.cancellation().check()?;
            write_record_frames(&mut writer, &output_records)?;
            output_records.clear();
        }
    }
    if !output_records.is_empty() {
        ctx.cancellation().check()?;
        write_record_frames(&mut writer, &output_records)?;
    }
    drop(merge);
    let run = writer.finish()?;
    telemetry.intermediate_run(pass);
    Ok(SortedRun { run, rows, pass })
}

fn write_record_frames(
    writer: &mut SpillRunWriter,
    records: &[SortSpillRecord],
) -> Result<(), ExecutionError> {
    let mut start = 0;
    while start < records.len() {
        let mut count = (records.len() - start).min(crate::executor::BATCH_ROWS);
        let encoded = loop {
            let encoded =
                encode_records(&records[start..start + count]).map_err(codec_execution_error)?;
            if encoded.len() <= SORT_SPILL_TARGET_FRAME_BYTES || count == 1 {
                break encoded;
            }
            count = count.div_ceil(2);
        };
        writer.append_batch(&encoded)?;
        start += count;
    }
    Ok(())
}

struct RunCursor {
    reader: SpillRunReader,
    pending: std::vec::IntoIter<SortSpillRecord>,
    remaining_rows: u64,
}

impl RunCursor {
    fn new(reader: SpillRunReader, rows: u64) -> Self {
        Self {
            reader,
            pending: Vec::new().into_iter(),
            remaining_rows: rows,
        }
    }

    fn next_record(&mut self) -> Result<Option<SortSpillRecord>, ExecutionError> {
        loop {
            if let Some(record) = self.pending.next() {
                self.remaining_rows = self.remaining_rows.checked_sub(1).ok_or_else(|| {
                    invalid_spill_state("spill run decoded more rows than declared".to_owned())
                })?;
                return Ok(Some(record));
            }
            let Some(batch) = self.reader.next_batch()? else {
                if self.remaining_rows != 0 {
                    return Err(invalid_spill_state(format!(
                        "spill run ended with {} declared rows missing",
                        self.remaining_rows
                    )));
                }
                return Ok(None);
            };
            let decoded = decode_records(batch.as_ref()).map_err(codec_execution_error)?;
            let decoded_len = u64::try_from(decoded.len())
                .map_err(|_| identity_error("decoded frame row count"))?;
            if decoded_len > self.remaining_rows {
                return Err(invalid_spill_state(format!(
                    "spill frame decoded {decoded_len} rows with only {} remaining",
                    self.remaining_rows
                )));
            }
            self.pending = decoded.into_iter();
        }
    }
}

struct KWayMerge {
    cursors: Vec<RunCursor>,
    heap: BinaryHeap<HeapEntry>,
    directions: Arc<[SortDirection]>,
}

impl KWayMerge {
    fn new(
        runs: Vec<SortedRun>,
        epoch: arcgraph_storage::QueryEpoch,
        directions: Arc<[SortDirection]>,
        telemetry: &SortTelemetry,
        pass: u32,
    ) -> Result<Self, ExecutionError> {
        let mut cursors = Vec::with_capacity(runs.len());
        let mut heap = BinaryHeap::with_capacity(runs.len());
        for sorted_run in runs {
            let reader = sorted_run.run.into_reader(epoch)?;
            // Observe the actual successfully-opened reader count, not the
            // requested group size (a staging reject can stop this loop).
            telemetry.observe_readers(cursors.len() + 1, pass);
            let mut cursor = RunCursor::new(reader, sorted_run.rows);
            let index = cursors.len();
            let first = cursor.next_record()?;
            cursors.push(cursor);
            if let Some(record) = first {
                heap.push(HeapEntry {
                    record,
                    reader_index: index,
                    directions: Arc::clone(&directions),
                });
            }
        }
        Ok(Self {
            cursors,
            heap,
            directions,
        })
    }

    fn next_record(&mut self) -> Result<Option<SortSpillRecord>, ExecutionError> {
        let Some(entry) = self.heap.pop() else {
            return Ok(None);
        };
        let reader_index = entry.reader_index;
        if let Some(record) = self.cursors[reader_index].next_record()? {
            self.heap.push(HeapEntry {
                record,
                reader_index,
                directions: Arc::clone(&self.directions),
            });
        }
        Ok(Some(entry.record))
    }
}

struct HeapEntry {
    record: SortSpillRecord,
    reader_index: usize,
    directions: Arc<[SortDirection]>,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse the semantic order to expose the
        // smallest `(keys, ordinal)` record. `reader_index` is a defensive
        // total-order tiebreak for malformed duplicate ordinals.
        compare_spill_records(&self.record, &other.record, &self.directions)
            .then_with(|| self.reader_index.cmp(&other.reader_index))
            .reverse()
    }
}

fn sort_records(records: &mut [SortSpillRecord], directions: &[SortDirection]) {
    records.sort_by(|left, right| compare_spill_records(left, right, directions));
}

fn compare_spill_records(
    left: &SortSpillRecord,
    right: &SortSpillRecord,
    directions: &[SortDirection],
) -> Ordering {
    compare_key_directions(&left.keys, &right.keys, directions)
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn compare_key_directions(
    left: &[Value],
    right: &[Value],
    directions: &[SortDirection],
) -> Ordering {
    for (index, direction) in directions.iter().copied().enumerate() {
        let Some(left_value) = left.get(index) else {
            return left.len().cmp(&right.len());
        };
        let Some(right_value) = right.get(index) else {
            return left.len().cmp(&right.len());
        };
        let ordering = compare_with_null_policy(left_value, right_value, direction);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn external_record_bytes(
    keys: &[Value],
    key_capacity: usize,
    row: &[Value],
    row_capacity: usize,
) -> u64 {
    let bytes = estimate_row_bytes(keys)
        .saturating_add(estimate_row_bytes(row))
        .saturating_add(
            key_capacity
                .saturating_sub(keys.len())
                .saturating_mul(std::mem::size_of::<Value>()),
        )
        .saturating_add(
            row_capacity
                .saturating_sub(row.len())
                .saturating_mul(std::mem::size_of::<Value>()),
        )
        .saturating_add(std::mem::size_of::<u64>());
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn measured_buffer_bytes(logical_bytes: u64, len: usize, capacity: usize) -> u64 {
    let spare_slots = capacity.saturating_sub(len);
    let spare_bytes = spare_slots.saturating_mul(std::mem::size_of::<SortSpillRecord>());
    logical_bytes.saturating_add(u64::try_from(spare_bytes).unwrap_or(u64::MAX))
}

fn codec_execution_error(error: SortSpillCodecError) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail: format!("external-sort spill codec: {error}"),
    })
}

fn invalid_spill_state(detail: String) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail,
    })
}

fn identity_error(subject: &'static str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Identity,
        detail: format!("external-sort {subject} exhausted"),
    })
}

impl Drop for SortOp {
    /// W12α fix-up MED-1 (PR #277 retro): release the operator's
    /// running budget reservation so the per-tenant counter does not
    /// drift upward across queries (a long-running tenant configured
    /// with a per-tenant byte cap would otherwise see false
    /// `ResourceExhausted` rejections after enough successful queries).
    /// The actual row bytes are freed by the field destructors; the
    /// budget release here decrements the bookkeeping to match.
    fn drop(&mut self) {
        if let (Some(tenant), Some(budget)) =
            (self.tenant_for_release, self.budget_for_release.take())
        {
            if self.reserved_total > 0 {
                budget.release(tenant, self.reserved_total);
            }
        }
    }
}

/// Compare two sort-key tuples in declared precedence with declared
/// direction. NULLs sort LAST in ASC, FIRST in DESC per Cypher 9 §6.6.
fn compare_key_tuples(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for (i, k) in keys.iter().enumerate() {
        let av = a.get(i).expect("pre-computed keys mirror keys length");
        let bv = b.get(i).expect("pre-computed keys mirror keys length");
        let ord = compare_with_null_policy(av, bv, k.direction);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Compare two values honoring the Cypher 9 §6.6 NULL ordering:
/// NULL sorts LAST in ASC, FIRST in DESC.
///
/// Implementation note: rather than special-case the NULL handling
/// in each direction, the comparator returns the order WITHOUT
/// the direction reversal applied to NULL. The non-NULL natural order
/// is reversed for DESC; the NULL slot is hard-wired so a NULL value
/// always sorts to the "outside" of the natural order in the
/// direction's perspective.
fn compare_with_null_policy(a: &Value, b: &Value, direction: SortDirection) -> Ordering {
    let null_a = matches!(a, Value::Null);
    let null_b = matches!(b, Value::Null);
    match (null_a, null_b) {
        (true, true) => Ordering::Equal,
        // ASC: NULL > non-NULL ⇒ NULL sorts LAST.
        // DESC: NULL < non-NULL ⇒ NULL sorts FIRST.
        (true, false) => match direction {
            SortDirection::Asc => Ordering::Greater,
            SortDirection::Desc => Ordering::Less,
        },
        (false, true) => match direction {
            SortDirection::Asc => Ordering::Less,
            SortDirection::Desc => Ordering::Greater,
        },
        (false, false) => {
            let nat = compare_non_null_values(a, b);
            match direction {
                SortDirection::Asc => nat,
                SortDirection::Desc => nat.reverse(),
            }
        }
    }
}

/// W12α fix-up LOW-4 (PR #277 retro): row-count-fallback diagnostic
/// surfaces as `ArcQLError::ResourceExhausted` (mirrors the byte-cap
/// path) rather than the prior `ExecutionError::Eval` string-error.
/// The shared variant lets the M5-07 / M5-11 / M5-13 transport-layer
/// renderers map row-count-fallback faults to the same HTTP-429 /
/// equivalent rate-limit class as byte-cap exhaustion.
fn row_count_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "SortOp runaway-guard".to_owned(),
        requested_bytes: 0,
        // #980 / #994 — lifted runaway-protection ceiling, not the old
        // 131 072-row valve that failed legitimate large ORDER BY.
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

/// Compare two non-NULL values. Returns `Ordering::Equal` for
/// incomparable types — this preserves the input order via the stable
/// sort.
fn compare_non_null_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Integer(x), Value::Float(y)) => {
            (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)
        }
        (Value::Float(x), Value::Integer(y)) => {
            x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)
        }
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Node(x), Value::Node(y)) => x.id.raw().cmp(&y.id.raw()),
        (Value::Relationship(x), Value::Relationship(y)) => x.id.raw().cmp(&y.id.raw()),
        // ADR-193 D-11 — paths ARE orderable (openCypher orderability is
        // a TOTAL order that never errors). A path sorts FIRST in the
        // global type-order (`Path → Rel → Node → Map → List → scalars`),
        // so a Path is `Less` than any other in-use type; two paths order
        // by node-id sequence then rel-id sequence (DETERMINISTIC, NEVER
        // `Equal`-collide for distinct paths — they must not merge under
        // sort/DISTINCT). The arm yields a real `Ordering` and never
        // errors (the function has no error channel). NO `Ord` derive on
        // `PathView` (see value.rs). Placed BEFORE the Map arm so a
        // (Path, Map) pair resolves Path-first here.
        (Value::Path(x), Value::Path(y)) => x.cmp_paths(y),
        (Value::Path(_), _) => Ordering::Less,
        (_, Value::Path(_)) => Ordering::Greater,
        // ADR-191 D-5 — any map operand routes through the openCypher
        // orderability total order: two maps tiebreak deterministically
        // (sorted-key sequence, then pairwise value orderability — so
        // distinct maps NEVER collide), and a map sorts AFTER
        // nodes/relationships, BEFORE lists/scalars. `ORDER BY` over a
        // map column is total and never errors.
        (Value::Map(_), _) | (_, Value::Map(_)) => {
            crate::executor::value::compare_orderability(a, b)
        }
        // Lists are orderable element-wise, with global orderability
        // ranking for heterogeneous elements and a prefix length tiebreak.
        (Value::List(_), _) | (_, Value::List(_)) => {
            crate::executor::value::compare_orderability(a, b)
        }
        // Heterogeneous types: fall back to "equal" — stable sort preserves order.
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::error::Span;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn make_persons_with_age(ages: &[Option<i64>]) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in ages.iter().enumerate() {
            let v = match age {
                Some(n) => Value::Integer(*n),
                None => Value::Null,
            };
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", v),
            );
        }
        s
    }

    fn person_scan() -> ScanOp {
        ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    fn prop_age() -> BoundExpression {
        BoundExpression::PropertyAccess {
            base: Box::new(BoundExpression::VariableRef {
                name: "n".into(),
                binding_id: BindingId::new(0),
                span: Span::point(1, 1),
                type_info: None,
            }),
            path: vec![crate::semantic::bound_ast::BoundPropertyRef {
                name: "age".into(),
                property_id: None,
                span: Span::point(1, 1),
            }],
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn extract_age(v: &Value) -> Option<i64> {
        match v {
            Value::Node(n) => match n.properties.get("age") {
                Some(Value::Integer(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        }
    }

    // -------------------------------------------------------------
    // SortOp stability + direction
    // -------------------------------------------------------------

    #[test]
    fn sort_asc_reorders_rows_by_key() {
        let s = make_persons_with_age(&[Some(30), Some(10), Some(20)]);
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        let ages: Vec<_> = b.rows().iter().map(|r| extract_age(&r[0])).collect();
        assert_eq!(ages, vec![Some(10), Some(20), Some(30)]);
    }

    #[test]
    fn sort_desc_reverses_natural_order() {
        let s = make_persons_with_age(&[Some(10), Some(20), Some(30)]);
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Desc,
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        let ages: Vec<_> = b.rows().iter().map(|r| extract_age(&r[0])).collect();
        assert_eq!(ages, vec![Some(30), Some(20), Some(10)]);
    }

    #[test]
    fn sort_is_stable_for_equal_keys() {
        // Two rows with the SAME age sorted ASC must preserve insertion
        // order (stable). Insertion order = NodeId-ascending in the
        // Stub substrate.
        let s = make_persons_with_age(&[Some(10), Some(10), Some(10)]);
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        let ids: Vec<u64> = b
            .rows()
            .iter()
            .map(|r| match &r[0] {
                Value::Node(n) => n.id.raw(),
                _ => panic!("Node"),
            })
            .collect();
        // Stable: 1, 2, 3 in insertion order.
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn sort_null_sorts_last_in_asc() {
        // Cypher 9 §6.6: NULL > any in ASC ⇒ sorts last.
        let s = make_persons_with_age(&[Some(20), None, Some(10)]);
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        let ages: Vec<_> = b.rows().iter().map(|r| extract_age(&r[0])).collect();
        // Two non-NULL ages first (10, 20), NULL last.
        assert_eq!(ages[0], Some(10));
        assert_eq!(ages[1], Some(20));
        assert_eq!(ages[2], None);
    }

    #[test]
    fn sort_propagates_cancel() {
        let s = make_persons_with_age(&[Some(10), Some(20)]);
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn sort_eos_after_emit() {
        let s = make_persons_with_age(&[Some(10)]);
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty());
    }

    #[test]
    fn compare_non_null_values_map_routes_through_orderability() {
        // ADR-191 D-5 — a map operand routes to the openCypher
        // orderability total order: a map sorts AFTER nodes/rels, BEFORE
        // lists/scalars, and distinct maps never collide (no Equal that
        // would merge them in ORDER BY).
        use crate::executor::value::NodeView;
        let m1 = Value::Map([("a".to_string(), Value::Integer(1))].into_iter().collect());
        let m2 = Value::Map([("a".to_string(), Value::Integer(2))].into_iter().collect());
        let node = Value::Node(NodeView::new(NodeId::new(1), None));
        assert_eq!(compare_non_null_values(&m1, &node), Ordering::Greater);
        assert_eq!(
            compare_non_null_values(&m1, &Value::Integer(0)),
            Ordering::Less
        );
        assert_eq!(
            compare_non_null_values(&m1, &Value::List(vec![Value::Integer(1)])),
            Ordering::Less
        );
        assert_eq!(compare_non_null_values(&m1, &m2), Ordering::Less);
        assert_ne!(compare_non_null_values(&m1, &m2), Ordering::Equal);
    }

    // -----------------------------------------------------------------
    // ADR-193 D-11 / test 8 — ORDERABILITY at the compare site (the ADR
    // designates `compare_non_null_values` the "real orderability
    // oracle"). Paths ARE orderable: they sort FIRST in the global
    // type-order, order deterministically by node-seq then rel-seq, and
    // NEVER collide for distinct paths (so they don't merge under sort /
    // DISTINCT). NB: full-pipeline `ORDER BY p` execution is gated on a
    // PRE-EXISTING executor limitation (projection over Sort/Aggregate
    // does not execute end-to-end for ANY graph/scalar key on main — e.g.
    // `RETURN count(n)` / `ORDER BY n.name` fail identically), so the
    // conformant ORDER-BY-path behavior is validated HERE at the
    // comparator the ADR points to, plus the DISTINCT non-collision e2e.
    // -----------------------------------------------------------------
    fn path_val(start: u64, segs: &[(u64, u64, u64)]) -> Value {
        use crate::executor::value::{NodeView, PathView, RelView};
        use arcgraph_core::{RelId, TypeId};
        let mut p = PathView::new(NodeView::new(NodeId::new(start), Some(LabelId::new(1))));
        for &(rid, from, to) in segs {
            p = p.with_segment(
                RelView::new(
                    RelId::new(rid),
                    NodeId::new(from),
                    NodeId::new(to),
                    Some(TypeId::new(1)),
                ),
                NodeView::new(NodeId::new(to), None),
            );
        }
        Value::Path(p)
    }

    #[test]
    fn adr193_paths_orderable_compare_arm() {
        let p12 = path_val(1, &[(10, 1, 2)]); // node-seq [1,2]
        let p13 = path_val(1, &[(11, 1, 3)]); // node-seq [1,3]

        // Deterministic node-seq ordering.
        assert_eq!(compare_non_null_values(&p12, &p13), Ordering::Less);
        assert_eq!(compare_non_null_values(&p13, &p12), Ordering::Greater);

        // Paths sort FIRST — a path is `Less` than any other in-use type.
        assert_eq!(
            compare_non_null_values(&p12, &Value::Integer(0)),
            Ordering::Less
        );
        assert_eq!(
            compare_non_null_values(&Value::String("z".into()), &p12),
            Ordering::Greater
        );

        // Distinct paths NEVER collide (no `_ => Equal` merge): a full
        // sort of a scrambled vec is deterministic AND retains all 3.
        let p1 = path_val(1, &[]); // zero-length, node-seq [1]
        let mut v = vec![p13.clone(), p1.clone(), p12.clone()];
        v.sort_by(compare_non_null_values);
        // node-seq order: [1] < [1,2] < [1,3].
        assert_eq!(v, vec![p1, p12, p13], "deterministic node-seq order");
    }

    /// #994 GA-blocker regression (sibling of #980) — an `ORDER BY` over
    /// more rows than the OLD fixed [`UNCAPPED_RUNAWAY_GUARD_ROWS`]
    /// predecessor `SPILLOVER_MAX_ROWS` (131 072) MUST succeed on the
    /// uncapped (no per-tenant byte cap) budget path and emit every row
    /// in sorted order. The default budget is uncapped — an explicit "no
    /// memory limit" choice — so the sort buffer is bounded only by the
    /// actual input cardinality.
    ///
    /// Pre-fix this errored the instant the buffer crossed 131 072 rows
    /// (`SortOp row-count fallback` → `ResourceExhausted`). RED-on-revert:
    /// restore the `BUDGET_FALLBACK_ROWS` ceiling on the uncapped buffer
    /// path and this fails. N = 150 000 > 131 072.
    #[test]
    fn sort_buffer_past_old_ceiling_succeeds_uncapped() {
        use crate::executor::ops::expand::SPILLOVER_MAX_ROWS;
        let n: usize = 150_000; // > old 131 072 ceiling
        assert!(n > SPILLOVER_MAX_ROWS, "must exceed the old fixed ceiling");
        // age = node id, so ASC order is strictly increasing 1..=n.
        let ages: Vec<Option<i64>> = (1..=n as i64).map(Some).collect();
        let s = make_persons_with_age(&ages);
        let c = ctx();
        assert!(
            !c.budget().has_cap(c.tenant()),
            "this test pins the UNCAPPED path (#994)"
        );
        let mut op = SortOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![SortKey {
                expr: prop_age(),
                direction: SortDirection::Asc,
            }],
        );
        let mut emitted: usize = 0;
        let mut last: i64 = i64::MIN;
        loop {
            let batch = op
                .next_batch(&c, &s)
                .expect("uncapped large sort must not error");
            if batch.is_empty() {
                break;
            }
            for row in batch.rows() {
                let age = extract_age(&row[0]).expect("age present");
                assert!(age >= last, "ascending order violated: {age} < {last}");
                last = age;
                emitted += 1;
            }
        }
        assert_eq!(emitted, n, "every row must survive past the old ceiling");
    }
}
