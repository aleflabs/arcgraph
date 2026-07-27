//! [`HashJoinOp`] — multi-pattern equi-join executor (W17α M4-08+).
//!
//! Lowers from [`crate::logical_plan::LogicalJoin`]. Implements an
//! in-memory hash join: drains the LEFT child eagerly into a per-
//! shared-binding bucket map, then streams the RIGHT child probing
//! each row against the bucket map and emitting one joined row per
//! match. Cartesian (empty `SharedBindings`) collapses to a single
//! bucket whose key is the empty fingerprint — every left row pairs
//! with every right row.
//!
//! # Why eager-left + streaming-right
//!
//! Per ADR-038 amendment-02 §M4.f the executor model is monomorphic
//! pull-based batches; a hash-join's BUILD phase is the standard
//! shape (Postgres, DuckDB, LiveGraph all materialize the BUILD
//! side). Streaming the PROBE side is what keeps the cursor-shaped
//! pipeline alive end-to-end — the operator emits an output batch as
//! soon as the right-side batch is consumed, matching the LIMIT-
//! aware backpressure semantics M4-72 forward expects.
//!
//! # Schema
//!
//! Output schema = `left_schema ++ right_fresh_bindings`. "Fresh" =
//! present in `right_schema` but NOT in `left_schema`. The shared
//! bindings stay non-duplicated (the LEFT row owns the canonical
//! column position). Mirror of the OPTIONAL MATCH op's left-outer
//! shape per [`super::optional_expand::OptionalExpandOp`].
//!
//! # Equality semantics — Cypher 3VL
//!
//! Per ADR-006 amendment-01 + ADR-038 §2 D-20, equality is 3-valued:
//! NULL ≠ NULL (Cypher's "unknown" outcome). The fingerprint helper
//! [`join_key_fingerprint`] returns `None` when any shared-column
//! value is `Value::Null`; rows with `None` fingerprints are
//! suppressed on BOTH sides (they cannot match anything).
//!
//! Per-variant fingerprint rules:
//! - `Value::Null` → fingerprint = `None` (row suppressed).
//! - `Value::Node(n)` → canonicalize on `n.id.raw()` (Cypher node
//!   equality is by id; label / property bag are derived).
//! - `Value::Relationship(r)` → canonicalize on `r.id.raw()` (same
//!   rationale).
//! - `Value::Float(f)` → `NaN` joins NEVER match (IEEE-754 NaN ≠ NaN
//!   per the [`Value`] doc comment); represented as `None`.
//! - `Value::Boolean` / `Value::Integer` / `Value::String` →
//!   straightforward byte-level fingerprint.
//! - `Value::List` → recursive fingerprint; lists join only when
//!   element-wise equal (Cypher list-equality semantics).
//!
//! # Memory budget (M4-64a integration)
//!
//! The BUILD-side hash table reserves bytes against
//! [`crate::executor::MemoryBudget`] per left row inserted. When
//! the per-tenant cap is configured, exceeding it surfaces
//! [`crate::semantic::error::ArcQLError::ResourceExhausted`]
//! routed via [`ExecutionError::Plan`]. When NO cap is configured
//! (uncapped budget = an explicit "no memory limit" choice), the
//! LEFT-side row count is bounded only by the actual left
//! cardinality, guarded against a true runaway by
//! [`super::expand::UNCAPPED_RUNAWAY_GUARD_ROWS`] (≈ 4.29 B rows).
//! #980: the old `SPILLOVER_MAX_ROWS` (131 072) valve was mis-tuned
//! as a workload limit and failed legitimate large traversals /
//! joins above ~150 K edges (the SNAP web-Google 5.1 M-edge repro);
//! the runaway guard is now far above any single-node graph.
//!
//! # PROBE-side spillover
//!
//! A single right row may match many left rows (skewed join shape).
//! When the output batch fills mid-match, the surplus rows queue
//! into `HashJoinOp::spillover`, which the next `next_batch` call
//! drains FIFO. Same byte-budget discipline as
//! `super::expand::ExpandOp::spillover`.
//!
//! # Grace spill path (M6.2 OOC-3)
//!
//! [`HashJoinOp::with_spillover_target`] enables partitioned execution over
//! OOC-1 runs. The path requires a configured `MemoryBudget` cap (refs #1524),
//! partitions both children without retaining either whole input, recursively
//! re-partitions oversized build buckets with depth-specific seeds, and uses a
//! bounded block fallback for unsplittable hot keys. Passing no target keeps
//! the legacy in-memory state machine byte-for-byte in behavior.
//!
//! # ADR provenance
//! - **ADR-038 §2 D-24** — `LogicalJoin` lowering surface.
//! - **ADR-038 amendment-02 §M4.f** — executor pull-based batch
//!   discipline.
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — multi-pattern MATCH
//!   semantics.
//! - **ADR-006 amendment-01** — Cypher 3VL equality semantics.

use std::collections::{HashMap, VecDeque};

use crate::executor::batch::Batch;
use crate::executor::budget::estimate_row_bytes;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS;
use crate::executor::ops::grace_hash_join::{GraceHashJoinRuntime, GraceHashJoinTarget};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// Build-side hash table.
///
/// Keyed by `join_key_fingerprint(shared_columns)` — `String`
/// chosen over a hand-rolled enum so `HashMap` keys are owned +
/// `Eq + Hash`. Bucket values are the full LEFT row (`Vec<Value>`),
/// so PROBE-side row materialization is a single allocation.
type LeftBucket = HashMap<String, Vec<Vec<Value>>>;

/// `LogicalJoin` executor — in-memory hash join (BUILD-side: LEFT,
/// PROBE-side: RIGHT).
pub struct HashJoinOp {
    left: Box<PhysicalOperator>,
    right: Box<PhysicalOperator>,
    /// Indices into `left.schema()` for the shared bindings (BUILD
    /// key columns).
    left_shared_indices: Vec<usize>,
    /// Indices into `right.schema()` for the shared bindings (PROBE
    /// key columns).
    right_shared_indices: Vec<usize>,
    /// Indices into `right.schema()` for bindings NOT in
    /// `left.schema()` (the "fresh" right-side columns appended to
    /// each output row).
    right_fresh_indices: Vec<usize>,
    /// Cached output schema = `left_schema ++ right_fresh_bindings`.
    schema: Vec<BindingId>,
    /// Build-side hash table. `None` until the first `next_batch`
    /// call exhausts the LEFT child + populates it; subsequent
    /// calls re-use it.
    build_table: Option<LeftBucket>,
    /// Have we drained the LEFT child?
    build_done: bool,
    /// Cumulative bytes the BUILD table reserved against the
    /// per-tenant memory budget. Released on `Drop` /
    /// completion-after-EOS via [`Self::release_build_reservation`].
    build_reserved_bytes: u64,
    /// PROBE-side spillover (overflow from a partially-emitted
    /// right row that matched many left rows). Each `SpilledRow`
    /// carries its byte reservation so the byte-budget release is
    /// paired with each pop.
    spillover: VecDeque<SpilledRow>,
    /// Have we drained the RIGHT child?
    right_done: bool,
    /// Opt-in OOC-3 runtime. It owns the live SpillQuery and every partition
    /// run, so dropping this field is the abort/zeroization boundary.
    grace_runtime: Option<Box<GraceHashJoinRuntime>>,
    /// A final non-empty Grace batch can exhaust the last task. The runtime is
    /// retained until the following EOS pull, then this bit prevents falling
    /// through into the legacy (already-consumed-child) state machine.
    grace_done: bool,
    /// Spill/codec/read faults are terminal because OOC-1 readers are
    /// sequential. Retrying returns the same typed error without shifted I/O.
    grace_terminal_error: Option<ExecutionError>,
}

/// One spilled output row + its budget reservation. Mirror of the
/// sibling spillover shape in
/// [`super::expand::ExpandOp`] / [`super::optional_expand::OptionalExpandOp`].
#[derive(Debug)]
struct SpilledRow {
    row: Vec<Value>,
    /// Bytes reserved against the per-tenant budget for this row.
    /// `0` when no cap was set at push time (the row-count fallback
    /// applied).
    reserved_bytes: u64,
}

impl std::fmt::Debug for HashJoinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HashJoinOp")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("left_shared_indices", &self.left_shared_indices)
            .field("right_shared_indices", &self.right_shared_indices)
            .field("right_fresh_indices", &self.right_fresh_indices)
            .field("schema", &self.schema)
            .field("build_done", &self.build_done)
            .field("build_reserved_bytes", &self.build_reserved_bytes)
            .field("right_done", &self.right_done)
            .field("grace", &self.grace_runtime.is_some())
            .field("grace_done", &self.grace_done)
            .field(
                "build_buckets",
                &self.build_table.as_ref().map(HashMap::len),
            )
            .finish()
    }
}

impl HashJoinOp {
    /// Construct a `HashJoinOp` from left + right children + the
    /// list of shared bindings. Both inputs MUST produce each shared
    /// binding in their schemas; constructor returns
    /// [`ExecutionError::Eval`] when a shared binding cannot be
    /// resolved (the planner contract is violated).
    pub fn new(
        left: PhysicalOperator,
        right: PhysicalOperator,
        shared: Vec<BindingId>,
    ) -> Result<Self, ExecutionError> {
        let left_schema: Vec<BindingId> = left.schema().to_vec();
        let right_schema: Vec<BindingId> = right.schema().to_vec();

        let mut left_shared_indices: Vec<usize> = Vec::with_capacity(shared.len());
        let mut right_shared_indices: Vec<usize> = Vec::with_capacity(shared.len());
        for b in &shared {
            let li = left_schema
                .iter()
                .position(|s| s == b)
                .ok_or_else(|| missing_binding_err("left", *b))?;
            let ri = right_schema
                .iter()
                .position(|s| s == b)
                .ok_or_else(|| missing_binding_err("right", *b))?;
            left_shared_indices.push(li);
            right_shared_indices.push(ri);
        }

        let right_fresh_indices: Vec<usize> = right_schema
            .iter()
            .enumerate()
            .filter(|(_, b)| !left_schema.contains(b))
            .map(|(i, _)| i)
            .collect();

        let mut schema = left_schema;
        for &idx in &right_fresh_indices {
            schema.push(right_schema[idx]);
        }

        Ok(Self {
            left: Box::new(left),
            right: Box::new(right),
            left_shared_indices,
            right_shared_indices,
            right_fresh_indices,
            schema,
            build_table: None,
            build_done: false,
            build_reserved_bytes: 0,
            spillover: VecDeque::new(),
            right_done: false,
            grace_runtime: None,
            grace_done: false,
            grace_terminal_error: None,
        })
    }

    /// Enable the OOC-3 Grace path with a live OOC-1 spill query.
    ///
    /// The bigger-than-RAM guarantee is conditional on a configured
    /// [`crate::executor::MemoryBudget`] cap. The runtime validates that
    /// precondition before creating a run. `None` preserves the in-memory
    /// operator for callers that explicitly choose an uncapped budget.
    pub fn with_spillover_target(
        mut self,
        target: Option<GraceHashJoinTarget>,
    ) -> Result<Self, ExecutionError> {
        self.grace_runtime = target.map(|target| Box::new(GraceHashJoinRuntime::new(target)));
        Ok(self)
    }

    /// Output schema (= `left_schema ++ right_fresh_bindings`).
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// # State machine
    ///
    /// 1. **BUILD-phase**: drain the LEFT child fully, partitioning
    ///    its rows by `join_key_fingerprint(shared_columns)`. Skips
    ///    rows whose fingerprint is `None` (Cypher 3VL NULL
    ///    suppression). Reserves byte budget per row; surfaces
    ///    `ResourceExhausted` when the per-tenant cap is exceeded.
    /// 2. **PROBE-phase**: stream the RIGHT child. For each right
    ///    row whose fingerprint matches a build bucket, emit one
    ///    output row per left row in that bucket. Output batch
    ///    capacity = [`crate::executor::batch::BATCH_ROWS`]; overflow
    ///    queues into `spillover`. Right rows with `None`
    ///    fingerprints are dropped.
    /// 3. **Drain spillover** first on each call so a partially-
    ///    consumed PROBE row's surplus emits before the next probe.
    /// 4. **EOS**: emit empty batch when LEFT + RIGHT are both
    ///    drained AND `spillover.is_empty()`.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;

        if let Some(error) = &self.grace_terminal_error {
            return Err(error.clone());
        }
        if self.grace_done {
            return Ok(Batch::empty(self.schema.len()));
        }
        if self.grace_runtime.is_some() {
            return self.next_grace_batch(ctx, substrate);
        }

        // Lazy BUILD on the first call.
        if !self.build_done {
            self.build_left_side(ctx, substrate)?;
            self.build_done = true;
        }

        let mut out = Batch::with_capacity(self.schema.len());
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());

        // Drain spillover first so partially-emitted matches finish
        // before we pull the next probe batch.
        while !out.is_full() {
            match self.spillover.pop_front() {
                Some(spilled) => {
                    if spilled.reserved_bytes > 0 {
                        budget.release(ctx.tenant(), spilled.reserved_bytes);
                    }
                    let _ = out.push_row(spilled.row);
                }
                None => break,
            }
        }
        if out.is_full() {
            return Ok(out);
        }

        // PROBE loop — drain the right child until the output is
        // full or the right side is exhausted AND spillover is empty.
        //
        // # Why a whole right batch is drained per iteration (NOT a
        // mid-batch return)
        //
        // The earlier shape `return Ok(out)` mid-`for right_row` DROPPED
        // every still-unprocessed row of the consumed `right_batch`:
        // `into_rows()` is a by-value iterator, so the surviving tail
        // was lost when the function returned, and the next call pulled
        // a FRESH right batch — silently skipping rows. For a Cartesian
        // (every right row matches the whole build bucket) a single
        // 50-row right batch overflowed `BATCH_ROWS` after ~41 rows, so
        // the trailing ~9 right rows vanished → `|left| × 41` instead of
        // `|left| × |right|` (issue #814: `MATCH (a),(b) RETURN count(*)`
        // returned `2000 + N` not `N²`). The keyed-join path lost rows
        // the same way once any right batch overflowed.
        //
        // The fix mirrors the sibling [`super::expand::ExpandOp`]
        // discipline EXACTLY: process the ENTIRE right batch, routing
        // each joined row through `push_row`-or-`push_spilled` (overflow
        // is preserved in `spillover`, never dropped); break the outer
        // loop only at its TOP when the output is full. A right batch is
        // therefore fully consumed before the next `self.right.next_batch`
        // call, so no right row can be skipped across batch boundaries.
        loop {
            ctx.cancellation().check()?;
            if out.is_full() {
                break;
            }
            if self.right_done && self.spillover.is_empty() {
                break;
            }
            if !self.right_done {
                let right_batch = self.right.next_batch(ctx, substrate)?;
                if right_batch.is_empty() {
                    self.right_done = true;
                    continue;
                }
                for right_row in right_batch.into_rows() {
                    let fingerprint =
                        match join_key_fingerprint(&right_row, &self.right_shared_indices) {
                            Some(fp) => fp,
                            None => continue, // 3VL NULL / NaN suppression.
                        };
                    // Take a borrow snapshot of the matching bucket
                    // (length only) so we can re-index it after the
                    // mutable `push_spilled` call without violating
                    // the borrow checker. The build table is owned
                    // by `self` for the duration of this operator;
                    // cloning each left row at emit-time is the
                    // canonical pattern (matches `OptionalExpandOp`).
                    let match_count =
                        match self.build_table.as_ref().and_then(|t| t.get(&fingerprint)) {
                            Some(b) => b.len(),
                            None => 0,
                        };
                    if match_count == 0 {
                        continue;
                    }
                    for li in 0..match_count {
                        let left_row = self
                            .build_table
                            .as_ref()
                            .and_then(|t| t.get(&fingerprint))
                            .and_then(|b| b.get(li))
                            .cloned()
                            .expect("build table bucket invariant");
                        let mut joined: Vec<Value> = left_row;
                        for &idx in &self.right_fresh_indices {
                            joined.push(right_row[idx].clone());
                        }
                        // Emit-or-spill every joined row. Overflow goes
                        // to `spillover` (drained first on the next
                        // call) — NEVER dropped. We do NOT return
                        // mid-batch: the remaining `right_row`s in this
                        // consumed `right_batch` MUST be processed in
                        // THIS call, or they are lost (issue #814).
                        if !out.push_row(joined.clone()) {
                            self.push_spilled(ctx, &budget, has_cap, joined)?;
                        }
                    }
                }
            } else {
                // right_done but spillover non-empty: looped back via
                // the drain at the top — break to flush.
                break;
            }
        }

        // EOS path: release the build-side reservation since no more
        // probes will happen.
        if out.is_empty() {
            self.release_build_reservation(ctx, &budget);
        }
        Ok(out)
    }

    fn next_grace_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        let Some(mut runtime) = self.grace_runtime.take() else {
            let error =
                ExecutionError::Spill(crate::executor::error::ExecutorSpillError::Failure {
                    kind: crate::executor::error::ExecutorSpillFailureKind::Corruption,
                    detail: "HashJoinOp Grace runtime missing during external drain".to_owned(),
                });
            self.grace_terminal_error = Some(error.clone());
            return Err(error);
        };
        let result = runtime.next_batch(
            &mut self.left,
            &mut self.right,
            &self.left_shared_indices,
            &self.right_shared_indices,
            &self.right_fresh_indices,
            self.schema.len(),
            ctx,
            substrate,
        );
        match result {
            Ok(batch) => {
                if runtime.is_done() && batch.is_empty() {
                    // Dropping the runtime closes every remaining handle,
                    // ends the epoch, and zeroizes the OOC-1 key now.
                    drop(runtime);
                    self.grace_done = true;
                } else {
                    self.grace_runtime = Some(runtime);
                }
                Ok(batch)
            }
            Err(error) => {
                // Abort is all-or-nothing. No `?` can bypass this drop arm.
                drop(runtime);
                self.grace_terminal_error = Some(error.clone());
                Err(error)
            }
        }
    }

    /// Drain the LEFT child fully + bucket rows by shared-key
    /// fingerprint.
    fn build_left_side<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let mut table: LeftBucket = HashMap::new();
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());
        let mut total_rows: usize = 0;
        loop {
            ctx.cancellation().check()?;
            let batch = self.left.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            for row in batch.into_rows() {
                let fingerprint = match join_key_fingerprint(&row, &self.left_shared_indices) {
                    Some(fp) => fp,
                    None => continue, // 3VL NULL / NaN suppression.
                };
                if has_cap {
                    let row_bytes = estimate_row_bytes(&row) as u64;
                    budget.try_reserve_unscoped(
                        ctx.tenant(),
                        row_bytes,
                        "HashJoinOp build-side",
                    )?;
                    self.build_reserved_bytes = self.build_reserved_bytes.saturating_add(row_bytes);
                } else if total_rows >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                    return Err(build_fallback_err(total_rows));
                }
                table.entry(fingerprint).or_default().push(row);
                total_rows += 1;
            }
        }
        self.build_table = Some(table);
        Ok(())
    }

    /// Push a row to PROBE-side spillover, reserving budget when
    /// configured or applying the row-count fallback when not.
    fn push_spilled(
        &mut self,
        ctx: &ExecutionContext,
        budget: &crate::executor::MemoryBudget,
        has_cap: bool,
        row: Vec<Value>,
    ) -> Result<(), ExecutionError> {
        let reserved_bytes = if has_cap {
            let bytes = estimate_row_bytes(&row) as u64;
            budget.try_reserve_unscoped(ctx.tenant(), bytes, "HashJoinOp probe-side spillover")?;
            bytes
        } else {
            if self.spillover.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(spillover_fallback_err(self.spillover.len()));
            }
            0
        };
        self.spillover.push_back(SpilledRow {
            row,
            reserved_bytes,
        });
        Ok(())
    }

    /// Release the build-side byte reservation if any. Called at
    /// EOS (clean exit) and on drop (defensive — in case the
    /// operator is destroyed mid-stream).
    fn release_build_reservation(
        &mut self,
        ctx: &ExecutionContext,
        budget: &crate::executor::MemoryBudget,
    ) {
        if self.build_reserved_bytes > 0 {
            budget.release(ctx.tenant(), self.build_reserved_bytes);
            self.build_reserved_bytes = 0;
        }
    }
}

/// Drop impl: intentionally empty.
///
/// Budget cleanup runs at the executor-session boundary — the
/// per-tenant `MemoryBudget` counter resets when the surrounding
/// [`ExecutionContext`] drops,
/// which is the canonical release point for any outstanding
/// reservation. The explicit `release_build_reservation` call on
/// the EOS path (line ~421) handles graceful BUILD cleanup; this
/// Drop body relies on session-level release for the failure /
/// cancellation path. Drop receives `&mut self` without an
/// `ExecutionContext` so it cannot call `release()` directly even
/// if we wanted to — the session reset is the structural
/// alternative.
impl Drop for HashJoinOp {
    fn drop(&mut self) {
        // No body: see rustdoc.
    }
}

/// Compute a deterministic string fingerprint of the join-key
/// columns at `indices`. Returns `None` when ANY shared column is
/// `Value::Null` (Cypher 3VL: NULL ≠ NULL) or `Value::Float(NaN)`
/// (IEEE-754: NaN ≠ NaN) — the row CANNOT match anything and is
/// suppressed.
///
/// # Format
///
/// Each cell encodes as `<kind_tag>:<payload>`; cells join with
/// `\u{1f}` (ASCII Unit Separator) so payload separators in user
/// strings don't collide. The result is `String` (not bytes) so
/// `HashMap` keys are easy + diagnosable; the format is internal
/// and not exposed on the wire.
///
/// # Variant rules
/// - `Boolean(b)` → `B:0` / `B:1`.
/// - `Integer(n)` → `I:<n>`.
/// - `Float(f)` if !NaN → `F:<f.to_bits()>` (lossless).
/// - `Float(NaN)` → `None` (no NaN match).
/// - `String(s)` → `S:<s>`.
/// - `Node(n)` → `N:<n.id.raw()>` (Cypher node equality is by id).
/// - `Relationship(r)` → `R:<r.id.raw()>` (Cypher rel equality is
///   by id).
/// - `List(xs)` → `L:[<recursive...>]` (Cypher list equality is
///   element-wise; a list containing NULL anywhere makes the WHOLE
///   list non-comparable per the 3VL propagation rule, so we
///   suppress the entire row).
/// - `Null` → `None`.
#[must_use]
pub fn join_key_fingerprint(row: &[Value], indices: &[usize]) -> Option<String> {
    if indices.is_empty() {
        // Cartesian shape: every row hashes to the same sentinel
        // bucket so PROBE-side rows match every BUILD row.
        return Some(String::from("@CARTESIAN"));
    }
    let mut out = String::new();
    for (pos, &idx) in indices.iter().enumerate() {
        if pos > 0 {
            out.push('\u{1f}');
        }
        let cell = row.get(idx)?;
        append_value_fingerprint(&mut out, cell)?;
    }
    Some(out)
}

/// Append a single value's fingerprint to `out`. Returns `None` to
/// signal "row not joinable" — caller propagates the suppression.
fn append_value_fingerprint(out: &mut String, v: &Value) -> Option<()> {
    use std::fmt::Write;
    match v {
        Value::Null => None,
        Value::Boolean(b) => {
            out.push_str(if *b { "B:1" } else { "B:0" });
            Some(())
        }
        Value::Integer(n) => {
            // Lossless integer repr.
            let _ = write!(out, "I:{n}");
            Some(())
        }
        Value::Float(f) => {
            if f.is_nan() {
                return None;
            }
            // Use bit pattern so +0.0 / -0.0 fingerprint distinctly
            // (matches IEEE-754 distinctness; Cypher's == treats
            // them as equal but the join-key contract prefers the
            // bit-exact match for determinism).
            let _ = write!(out, "F:{}", f.to_bits());
            Some(())
        }
        Value::String(s) => {
            // Mark the start AND the byte length so embedded `\u{1f}`
            // can't blur cell boundaries (defense-in-depth; the
            // outer separator already uses ASCII US, but a user
            // string containing US would otherwise collide).
            let _ = write!(out, "S:{}:{}", s.len(), s);
            Some(())
        }
        Value::Node(n) => {
            let _ = write!(out, "N:{}", n.id.raw());
            Some(())
        }
        Value::Relationship(r) => {
            let _ = write!(out, "R:{}", r.id.raw());
            Some(())
        }
        Value::List(xs) => {
            out.push_str("L:[");
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                append_value_fingerprint(out, x)?;
            }
            out.push(']');
            Some(())
        }
        // ADR-191 — a map JOIN key uses openCypher EQUALITY (the `=`
        // operator), NOT equivalence: a map with a `null` value anywhere
        // compares Unknown (`{a:null} = {a:null}` → null, D-3), so it is
        // NON-joinable and the row is suppressed — exactly as the `List`
        // arm above suppresses a list containing `null` (the inner
        // `append_value_fingerprint(out, v)?` propagates `None`).
        // `BTreeMap` sorted-key order makes a definite-valued map's
        // fingerprint deterministic. (Contrast GROUP BY / DISTINCT, which
        // use EQUIVALENCE via `canonical_row_key` and DO group `{a:null}`
        // together — a deliberately different code path.)
        Value::Map(m) => {
            out.push_str("M:{");
            for (k, v) in m {
                let _ = write!(out, "{}:{}=", k.len(), k);
                append_value_fingerprint(out, v)?;
                out.push(';');
            }
            out.push('}');
            Some(())
        }
        // ADR-193 — a path is never a legitimate equi-join key (the
        // named-path var is fresh, not shared across patterns; the
        // PlainPathOp appends it AFTER the join subtree, so a path
        // cannot flow into a join's key columns). The arm exists for
        // exhaustiveness; we fingerprint the node/rel ID sequence so the
        // (unexpected) reach is deterministic AND consistent with the
        // D-10 by-ID path equality, rather than silently suppressing the
        // row (`None`).
        Value::Path(p) => {
            let _ = write!(out, "P:{}", p.start.id.raw());
            for s in &p.segments {
                let _ = write!(out, ",{}>{}", s.rel.id.raw(), s.end.id.raw());
            }
            Some(())
        }
        // W23-V11-T-01 / ADR-090 — temporal + decimal join keys use
        // their Display form (canonical ISO-8601 / decimal). The
        // ZonedDateTime Display form is UTC-projection-then-offset so
        // two zoned values that share a UTC instant fingerprint
        // identically (matching Value's PartialEq by-UTC-instant
        // semantics).
        Value::Temporal(t) => {
            let _ = write!(out, "T:{}", t.utc_nanos());
            Some(())
        }
        Value::LocalDateTime(ldt) => {
            let _ = write!(out, "LT:{ldt}");
            Some(())
        }
        Value::Date(d) => {
            let _ = write!(out, "D:{d}");
            Some(())
        }
        Value::Duration(d) => {
            let _ = write!(out, "DU:{d}");
            Some(())
        }
        Value::Decimal(d) => {
            let _ = write!(out, "DC:{d}");
            Some(())
        }
    }
}

/// Render the build-side runaway-guard fallback error.
///
/// #980 — the cap reported is the lifted [`UNCAPPED_RUNAWAY_GUARD_ROWS`]
/// runaway-protection ceiling, NOT the old 131 072-row
/// [`SPILLOVER_MAX_ROWS`] valve that broke legitimate large joins. This
/// is `ResourceExhausted` (a resource/runtime fault), so the wire
/// surfaces classify it as transient/resource-exhausted, never
/// `Neo.ClientError.Statement.SyntaxError`.
fn build_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "HashJoinOp build-side runaway-guard".to_owned(),
        requested_bytes: 0,
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

/// Render the probe-side runaway-guard fallback error (#980 — see
/// [`build_fallback_err`]).
fn spillover_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "HashJoinOp probe-side spillover runaway-guard".to_owned(),
        requested_bytes: 0,
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

/// Build a "shared binding missing from schema" planner-contract
/// violation error.
fn missing_binding_err(side: &str, b: BindingId) -> ExecutionError {
    ExecutionError::Eval(format!(
        "HashJoinOp: shared binding {b:?} not present in {side} schema (planner-contract violation)"
    ))
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};

    use super::*;
    // The old fixed ceiling — tests reference it to prove they exceed it.
    use crate::executor::ops::expand::SPILLOVER_MAX_ROWS;
    use crate::executor::ops::{ExpandOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, RelView};
    use crate::logical_plan::Direction;

    fn fixture() -> StubExecutorSubstrate {
        // Three persons, edges Alice -[KNOWS]-> Bob; Alice -[KNOWS]->
        // Carol. Carol is unconnected from anyone else.
        StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(3), Some(LabelId::new(1))),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(10),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(11),
                    NodeId::new(1),
                    NodeId::new(3),
                    Some(TypeId::new(1)),
                ),
            )
    }

    #[test]
    fn equi_join_matches_on_shared_binding() {
        // LEFT pattern: MATCH (a:Person) → 3 rows.
        // RIGHT pattern: MATCH (a:Person)-[r:KNOWS]->(b) → 2 rows
        //   (Alice→Bob, Alice→Carol).
        // Shared = [a] → equi-join on `a`: Alice's LEFT row joins
        // BOTH right rows; Bob + Carol's LEFT rows have NO right
        // matches.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let r = BindingId::new(1);
        let b = BindingId::new(2);

        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));

        let right_scan = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right_exp = ExpandOp::new(
            right_scan,
            a,
            Some(r),
            b,
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .expect("expand build");
        let right = PhysicalOperator::Expand(right_exp);

        let mut op = HashJoinOp::new(left, right, vec![a]).expect("join construction");
        assert_eq!(op.schema(), &[a, r, b]);
        let out = op.next_batch(&ctx, &s).expect("join batch");
        // 2 joined rows (Alice's two outbound KNOWS edges).
        assert_eq!(out.row_count(), 2);
        for row in out.rows() {
            // schema = [a, r, b]. `a` must be Alice (id=1).
            assert!(matches!(&row[0], Value::Node(n) if n.id == NodeId::new(1)));
            // `r` must be a KNOWS rel.
            assert!(matches!(&row[1], Value::Relationship(_)));
            // `b` must be Bob or Carol (id 2 or 3).
            match &row[2] {
                Value::Node(n) => assert!(n.id == NodeId::new(2) || n.id == NodeId::new(3)),
                other => panic!("unexpected RIGHT row[2]: {other:?}"),
            }
        }
        // Second call returns EOS.
        let eos = op.next_batch(&ctx, &s).expect("eos");
        assert!(eos.is_empty());
    }

    #[test]
    fn cartesian_join_emits_left_times_right() {
        // LEFT: MATCH (a) → 3 rows (Alice, Bob, Carol).
        // RIGHT: MATCH (b) → 3 rows.
        // Shared = [] → Cartesian: 3 × 3 = 9 rows.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(1)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, Vec::new()).expect("cartesian construction");
        // Schema is [a, b] — both bindings are fresh.
        assert_eq!(op.schema(), &[a, b]);
        let out = op.next_batch(&ctx, &s).expect("cartesian batch");
        assert_eq!(out.row_count(), 9);
        // Every row has distinct (a, b) by NodeId pair.
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for row in out.rows() {
            let aa = match &row[0] {
                Value::Node(n) => n.id.raw(),
                _ => panic!("a not Node"),
            };
            let bb = match &row[1] {
                Value::Node(n) => n.id.raw(),
                _ => panic!("b not Node"),
            };
            pairs.push((aa, bb));
        }
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ]
        );
    }

    /// Build a stub substrate with `n` `LabelId(1)` nodes,
    /// `NodeId(1..=n)`.
    fn n_node_fixture(n: u64) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for k in 1..=n {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(k), Some(LabelId::new(1))),
            );
        }
        s
    }

    /// #814 regression — a Cartesian whose product overflows
    /// [`crate::executor::batch::BATCH_ROWS`] (2048) MUST drain across
    /// multiple `next_batch` calls and recover EVERY (a, b) pair, not
    /// just the first batch's worth.
    ///
    /// Before the fix, the PROBE loop did `return Ok(out)` the instant
    /// the output batch filled mid-`right_batch` — silently dropping the
    /// still-unprocessed tail of the consumed right batch. With N=50 the
    /// right scan returns all 50 rows in ONE batch; the probe overflowed
    /// after ~41 rows (41 × 50 = 2050 > 2048) and the trailing 9 right
    /// rows vanished, yielding `|left| × 41 = 2050` instead of `2500`
    /// (the `2000 + N` signature in issue #814). This test pulls
    /// `next_batch` to EOS and asserts the FULL `N²` set with no missing
    /// `b`. It is RED against the pre-fix mid-batch-return shape.
    #[test]
    fn cartesian_overflowing_batch_rows_recovers_full_product() {
        use std::collections::BTreeSet;
        let n: u64 = 50; // 50 × 50 = 2500 > BATCH_ROWS (2048)
        let s = n_node_fixture(n);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(1)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, Vec::new()).expect("cartesian construction");

        // Drain to EOS, collecting every (a, b) pair.
        let mut pairs: BTreeSet<(u64, u64)> = BTreeSet::new();
        let mut total_rows = 0usize;
        loop {
            let batch = op.next_batch(&ctx, &s).expect("batch");
            if batch.is_empty() {
                break;
            }
            // No single batch ever exceeds BATCH_ROWS.
            assert!(
                batch.row_count() <= crate::executor::batch::BATCH_ROWS,
                "batch overshoots BATCH_ROWS"
            );
            for row in batch.rows() {
                total_rows += 1;
                let aa = match &row[0] {
                    Value::Node(node) => node.id.raw(),
                    other => panic!("a not Node: {other:?}"),
                };
                let bb = match &row[1] {
                    Value::Node(node) => node.id.raw(),
                    other => panic!("b not Node: {other:?}"),
                };
                assert!(pairs.insert((aa, bb)), "duplicate pair ({aa}, {bb})");
            }
        }

        // Exact-cardinality oracle: N² distinct pairs, no dupes, every
        // (a, b) ∈ {1..=n}² present.
        let n2 = (n * n) as usize;
        assert_eq!(total_rows, n2, "total rows must equal N² (no dropped rows)");
        assert_eq!(pairs.len(), n2, "distinct pairs must equal N² (no dupes)");
        let distinct_a: BTreeSet<u64> = pairs.iter().map(|(x, _)| *x).collect();
        let distinct_b: BTreeSet<u64> = pairs.iter().map(|(_, y)| *y).collect();
        assert_eq!(distinct_a.len() as u64, n, "every a must survive");
        assert_eq!(
            distinct_b.len() as u64,
            n,
            "every b must survive (the dropped-tail bug truncated b to ~41)"
        );
        for x in 1..=n {
            for y in 1..=n {
                assert!(pairs.contains(&(x, y)), "missing pair ({x}, {y})");
            }
        }
    }

    /// #814 regression (keyed sibling) — an equi-join on a shared
    /// binding `k` over `n` distinct nodes returns exactly `n` matched
    /// rows (one per shared key), draining to EOS. With `n` chosen so
    /// the BUILD bucket scan and the PROBE scan are both `n` rows, this
    /// pins that the keyed PROBE loop loses no right rows (the issue's
    /// `WHERE a.uid = b.uid` returned 41 of 50 via the SAME dropped-tail
    /// bug — once any probe batch overflowed, its tail was lost; here we
    /// keep `n` under `BATCH_ROWS` so the exact oracle is `n`, and the
    /// over-`BATCH_ROWS` overflow-resume is proven by the cartesian test
    /// above which shares this loop + fix).
    #[test]
    fn keyed_join_recovers_every_shared_key_match() {
        let n: u64 = 200; // distinct keys; 200 < BATCH_ROWS so oracle = n
        let s = n_node_fixture(n);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let k = BindingId::new(0);
        // Both sides scan LabelId(1) binding `k`; shared = [k]. Each of
        // the n nodes matches itself → exactly n joined rows.
        let left = PhysicalOperator::Scan(ScanOp::new(k, Some(LabelId::new(1)), Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(k, Some(LabelId::new(1)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, vec![k]).expect("keyed construction");

        use std::collections::BTreeSet;
        let mut matched: BTreeSet<u64> = BTreeSet::new();
        loop {
            let batch = op.next_batch(&ctx, &s).expect("batch");
            if batch.is_empty() {
                break;
            }
            for row in batch.rows() {
                match &row[0] {
                    Value::Node(node) => {
                        assert!(matched.insert(node.id.raw()), "duplicate key match");
                    }
                    other => panic!("k not Node: {other:?}"),
                }
            }
        }
        assert_eq!(
            matched.len() as u64,
            n,
            "every shared key must match exactly once"
        );
        for x in 1..=n {
            assert!(matched.contains(&x), "missing key match {x}");
        }
    }

    /// #980 GA-blocker regression — a keyed equi-join whose BUILD side
    /// exceeds the OLD fixed [`SPILLOVER_MAX_ROWS`] ceiling (131 072)
    /// MUST succeed on the uncapped (no per-tenant byte cap) budget path
    /// and return EVERY shared-key match. The default budget is uncapped
    /// (`has_cap == false`), which is an EXPLICIT "no memory limit"
    /// choice — so the in-memory build table is bounded only by the
    /// actual left cardinality, NOT by the tiny 131 072-row safety valve.
    ///
    /// Pre-#980 this errored the instant `build_left_side` crossed
    /// 131 072 rows: `ResourceExhausted { requested_bytes: 0, .. }` (the
    /// "would reserve 0" symptom — the SNAP web-Google 5.1 M-edge repro
    /// failed here). RED-on-revert: re-impose the `SPILLOVER_MAX_ROWS`
    /// ceiling on the uncapped build path and this test fails with that
    /// ResourceExhausted.
    ///
    /// N = 200 000 > 131 072 (= 64 × 2048) so the BUILD side is well past
    /// the old ceiling. Each of the N nodes matches itself → exactly N
    /// joined rows, drained across many `next_batch` calls to EOS.
    #[test]
    fn keyed_join_build_side_past_old_ceiling_succeeds_uncapped() {
        let n: u64 = 200_000; // > SPILLOVER_MAX_ROWS (131 072)
        assert!(
            n as usize > SPILLOVER_MAX_ROWS,
            "test must exceed the old fixed ceiling"
        );
        let s = n_node_fixture(n);
        // DEFAULT context => uncapped budget (has_cap == false) — the
        // >150 K GA-blocker path.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert!(
            !ctx.budget().has_cap(ctx.tenant()),
            "this test pins the UNCAPPED path (the #980 blocker)"
        );
        let k = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(k, Some(LabelId::new(1)), Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(k, Some(LabelId::new(1)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, vec![k]).expect("keyed construction");

        let mut matched: u64 = 0;
        loop {
            let batch = op
                .next_batch(&ctx, &s)
                .expect("uncapped large join must not error");
            if batch.is_empty() {
                break;
            }
            matched += batch.row_count() as u64;
        }
        assert_eq!(
            matched, n,
            "every shared key must match exactly once past the old ceiling"
        );
    }

    /// #980 — the PROBE-side spillover path shares the same fixed-ceiling
    /// fallback. A single LEFT key matched by MANY right rows pushes the
    /// surplus joined rows into `spillover`; pre-#980 the uncapped path
    /// failed the moment `spillover.len()` crossed
    /// [`SPILLOVER_MAX_ROWS`]. Here a Cartesian (empty shared bindings)
    /// over a modest N yields N² rows; with N chosen so N² > the OLD
    /// ceiling, the probe overflow must drain to EOS without erroring.
    ///
    /// N = 400 → N² = 160 000 > 131 072. RED-on-revert: re-impose the
    /// `SPILLOVER_MAX_ROWS` ceiling on the uncapped `push_spilled` path.
    #[test]
    fn probe_side_spillover_past_old_ceiling_succeeds_uncapped() {
        let n: u64 = 400; // 400 × 400 = 160 000 > SPILLOVER_MAX_ROWS
        assert!(
            (n * n) as usize > SPILLOVER_MAX_ROWS,
            "N² must exceed the old fixed ceiling to exercise probe spillover"
        );
        let s = n_node_fixture(n);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(1)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, Vec::new()).expect("cartesian construction");

        let mut total: u64 = 0;
        loop {
            let batch = op
                .next_batch(&ctx, &s)
                .expect("uncapped large cartesian must not error");
            if batch.is_empty() {
                break;
            }
            total += batch.row_count() as u64;
        }
        assert_eq!(
            total,
            n * n,
            "full N² product must survive (no dropped rows)"
        );
    }

    /// #980 Part 2 (error class) — the row-count fallback errors carry the
    /// `ResourceExhausted` ArcQL variant, NOT a parse / syntax class. A
    /// genuine resource-exhaustion on a VALID query must classify as a
    /// resource/runtime error so the wire surfaces never report
    /// `Neo.ClientError.Statement.SyntaxError`. The `build_fallback_err` /
    /// `spillover_fallback_err` helpers are the executor-side source of
    /// that classification.
    #[test]
    fn fallback_errors_are_resource_exhausted_not_syntax() {
        use crate::semantic::error::ArcQLError;
        let build = build_fallback_err(SPILLOVER_MAX_ROWS);
        assert!(
            matches!(
                build,
                ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
            ),
            "build fallback must be ResourceExhausted, got {build:?}"
        );
        let probe = spillover_fallback_err(SPILLOVER_MAX_ROWS);
        assert!(
            matches!(
                probe,
                ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
            ),
            "probe fallback must be ResourceExhausted, got {probe:?}"
        );
    }

    #[test]
    fn empty_left_yields_empty_output() {
        // LEFT scans an empty tenant → 0 rows. BUILD table is empty,
        // so every PROBE row finds no bucket and we emit zero rows.
        // Both scans bind `a` so the shared-binding contract is
        // honored.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, vec![a]).expect("ok");
        let out = op.next_batch(&ctx, &s).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn empty_right_yields_empty_output() {
        // The fixture has 3 LEFT-scannable rows; we wire a RIGHT
        // scan over an empty tenant. PROBE returns immediately → 0
        // joined rows.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        // Empty label → no matches.
        let right = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(99)), Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, Vec::new()).expect("ok");
        let out = op.next_batch(&ctx, &s).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn null_join_key_suppresses_row() {
        // 3VL: a row containing NULL in a shared column matches
        // nothing. We can't construct one through the standard pipe
        // for nodes (ScanOp never emits NULL), so we sanity-check
        // the fingerprint helper directly.
        let row = vec![Value::Null, Value::Integer(7)];
        let fp = join_key_fingerprint(&row, &[0]);
        assert!(fp.is_none(), "NULL key must be suppressed");
        // Non-null shared column is fine.
        let fp_ok = join_key_fingerprint(&row, &[1]);
        assert!(fp_ok.is_some());
    }

    #[test]
    fn nan_float_key_suppresses_row() {
        let row = vec![Value::Float(f64::NAN)];
        let fp = join_key_fingerprint(&row, &[0]);
        assert!(fp.is_none(), "NaN key must be suppressed (IEEE-754)");
    }

    #[test]
    fn cartesian_empty_shared_admits_every_row() {
        // Empty shared bindings → all rows hash to one sentinel
        // bucket regardless of their value content (even NULL).
        let row_a = vec![Value::Null];
        let row_b = vec![Value::Integer(7)];
        let fp_a = join_key_fingerprint(&row_a, &[]);
        let fp_b = join_key_fingerprint(&row_b, &[]);
        assert_eq!(fp_a, fp_b);
        assert_eq!(fp_a, Some("@CARTESIAN".into()));
    }

    #[test]
    fn fingerprint_node_equality_is_by_id() {
        // Two Node values with the SAME id but DIFFERENT labels
        // share a fingerprint (Cypher node equality is by id).
        let n_a = NodeView::new(NodeId::new(7), Some(LabelId::new(1)));
        let n_b = NodeView::new(NodeId::new(7), Some(LabelId::new(2)));
        let fp_a = join_key_fingerprint(&[Value::Node(n_a)], &[0]);
        let fp_b = join_key_fingerprint(&[Value::Node(n_b)], &[0]);
        assert_eq!(fp_a, fp_b);
    }

    #[test]
    fn fingerprint_list_recurses_and_suppresses_inner_null() {
        // List containing a NULL anywhere makes the whole list
        // non-comparable per the 3VL propagation rule.
        let list_with_null = vec![Value::List(vec![Value::Integer(1), Value::Null])];
        let fp = join_key_fingerprint(&list_with_null, &[0]);
        assert!(fp.is_none(), "list with NULL element must suppress");
        // A pure list without null is fingerprintable.
        let list_ok = vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])];
        let fp_ok = join_key_fingerprint(&list_ok, &[0]);
        assert!(fp_ok.is_some());
    }

    #[test]
    fn missing_shared_binding_in_schema_is_planner_contract_violation() {
        // The shared binding `99` is in neither child schema. We
        // expect ExecutionError::Eval at construction time.
        let s = fixture();
        let _ = s;
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let bogus = BindingId::new(99);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, None, Lsn::MAX));
        let err =
            HashJoinOp::new(left, right, vec![bogus]).expect_err("contract violation must surface");
        assert!(matches!(err, ExecutionError::Eval(_)));
    }

    #[test]
    fn cancel_during_build_phase_short_circuits() {
        // Cancel BEFORE the first call. BUILD must surface
        // ExecutionError::Cancelled at the cancellation-check. Both
        // sides bind `a` so the shared-binding contract is honored.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let a = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let mut op = HashJoinOp::new(left, right, vec![a]).expect("ok");
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn multi_pattern_join_three_persons_emits_correct_rows() {
        // Build a fixture where Alice -[KNOWS]-> Bob -[KNOWS]-> Carol.
        // Two patterns: MATCH (a)-[r1]->(b) AND (b)-[r2]->(c).
        // Implemented at the executor level as a join on `b`.
        let s = StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(3), Some(LabelId::new(1))),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(100),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(101),
                    NodeId::new(2),
                    NodeId::new(3),
                    Some(TypeId::new(1)),
                ),
            );

        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let r1 = BindingId::new(1);
        let b = BindingId::new(2);
        let r2 = BindingId::new(3);
        let c = BindingId::new(4);

        // Left = a-[r1]->b
        let scan_a = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let left_pattern = PhysicalOperator::Expand(
            ExpandOp::new(
                scan_a,
                a,
                Some(r1),
                b,
                Some(TypeId::new(1)),
                Direction::LeftToRight,
                None,
                Lsn::MAX,
            )
            .expect("left expand"),
        );

        // Right = b-[r2]->c (root scan over b)
        let scan_b = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(1)), Lsn::MAX));
        let right_pattern = PhysicalOperator::Expand(
            ExpandOp::new(
                scan_b,
                b,
                Some(r2),
                c,
                Some(TypeId::new(1)),
                Direction::LeftToRight,
                None,
                Lsn::MAX,
            )
            .expect("right expand"),
        );

        let mut op = HashJoinOp::new(left_pattern, right_pattern, vec![b]).expect("ok");
        // Schema = [a, r1, b, r2, c] (b is shared, not duplicated).
        assert_eq!(op.schema(), &[a, r1, b, r2, c]);
        let out = op.next_batch(&ctx, &s).expect("batch");
        // Only one full 2-pattern match: Alice -> Bob -> Carol.
        assert_eq!(out.row_count(), 1);
        let row = &out.rows()[0];
        let a_id = match &row[0] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        let b_id = match &row[2] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        let c_id = match &row[4] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        assert_eq!(a_id, NodeId::new(1));
        assert_eq!(b_id, NodeId::new(2));
        assert_eq!(c_id, NodeId::new(3));
    }
}
