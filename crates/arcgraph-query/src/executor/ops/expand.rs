//! [`ExpandOp`] — one-hop relationship traversal (M4-61).
//!
//! Lowers from [`crate::logical_plan::LogicalExpand`]. Pulls upstream
//! rows from a child operator (typically a [`super::ScanOp`] or
//! another [`ExpandOp`]); for each row, looks up neighbors via
//! [`crate::executor::ExecutorSubstrate::expand`] and emits one
//! output row per neighbor.
//!
//! # Schema extension
//!
//! Output schema = `child_schema ++ [rel_var (if Some)] ++ [to_var]`.
//! Reuses the same row layout for the inherited columns; appends one
//! cell for the rel binding (if present) + one cell for the
//! destination node.
//!
//! # Variable-length patterns (`*N..M`) — ADR-186 / #650-C
//!
//! [`ExpandOp`] executes openCypher v9 §3 bounded variable-length
//! traversal `(a)-[*N..M]->(b)` (and the unbounded `*N..` / bare `*`
//! forms). The traversal is a breadth-first expansion that reuses the
//! single-hop [`ExecutorSubstrate::expand`] substrate per frontier
//! step, emitting one output row per **distinct path** whose hop-count
//! lands in `[min, max]`.
//!
//! Three openCypher §3 semantics are load-bearing (ADR-186 FROZEN
//! CONTRACT RC-1/RC-2/RC-3):
//!
//! - **Edge-uniqueness (RC-3):** no relationship is traversed twice
//!   within a single path; each live path-state carries an exact per-PATH
//!   relationship set and prunes frontier expansions that re-use a `RelId`
//!   already on the path. The legacy/deeper stream uses a `HashSet<RelId>`;
//!   the spillable k<=2 stream scans its at-most-two relationship IDs, which
//!   has the same zero-false-positive/zero-false-negative contract without a
//!   separately growing visited allocation. **Node repeats ARE allowed** (we
//!   do NOT dedup on node id). This guarantees finite termination on cyclic
//!   graphs while preserving the existing openCypher result multiplicity.
//! - **Unbounded cap (RC-1):** the unbounded forms (`*N..` / `*`) are
//!   bounded by an explicit depth cap [`VARLENGTH_UNBOUNDED_MAX_DEPTH`]
//!   (aligned with the planner's `DEFAULT_UNBOUNDED_MAX_HOPS = 5` at
//!   `planner/cost/operator.rs`) AND a frontier-size budget
//!   [`VARLENGTH_MAX_FRONTIER`]. Reaching either cap surfaces a
//!   structured [`ExecutionError`] — **never a silently-truncated
//!   result set** (a truncated traversal is a wrong answer presented
//!   as complete).
//! - **rel_var = LIST (RC-2):** when `length_range` is a var-length
//!   form, the rel binding is `Value::List(Vec<Value::Relationship>)`
//!   in traversal order (the single-hop case keeps the scalar
//!   `Value::Relationship` shape). `*0` binds an empty list.
//!
//! The GQL `{N,M}` form (`LengthRange::Quantified`) remains reserved
//! to v1.1 (rejected upstream at `semantic/type_check.rs` per ADR-038
//! §2 D-9 + D-16); the executor defensively rejects it with
//! `NotImplemented`.

use std::collections::{HashSet, VecDeque};

use arcgraph_core::{Lsn, RelId};

use crate::ast::LengthRange;
use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::budget::{MemoryBudget, estimate_row_bytes};
use crate::executor::context::ExecutionContext;
use crate::executor::error::{ExecutionError, ExecutorSpillError, ExecutorSpillFailureKind};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::expand_spill::{ExpandSpillQueue, ExpandSpillTarget};
use crate::executor::simd::expand::simd_neighbor_match_mask;
use crate::executor::substrate::{BoundEdgeCursor, ExecutorSubstrate};
use crate::executor::value::{NodeView, RelView, Value};
use crate::logical_plan::Direction;
use crate::semantic::bound_ast::BindingId;
use arcgraph_core::{LabelId, TypeId};

/// W11Z fix-up MED-3 (PR #268 retro): hard cap on the number of rows
/// the spillover queue may hold before [`ExpandOp::next_batch`] surfaces
/// a "spillover bound exceeded" fault. W12α fix-up LOW-4 (PR #277
/// retro) promoted that fault from `ExecutionError::Eval` (string) to
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] so it
/// shares the variant with the byte-cap path.
///
/// # W12α (M4-64a) supersession
///
/// The W12α slice (M4-64a per amendment-03 §Structural-1) replaces this
/// row-count cap with a proper per-tenant byte budget via
/// [`crate::executor::MemoryBudget`]. The constant is preserved as a
/// **fallback** for tenants without a configured byte cap (the v1.0-
/// alpha default). When [`crate::executor::MemoryBudget::has_cap`] is
/// `true` for the tenant, the byte budget takes precedence and surfaces
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] when
/// exceeded; when `false`, this row-count cap applies as before.
///
/// # Pre-W12α rationale (preserved for reviewer context)
///
/// The pre-fix-up implementation could grow the spillover Vec to
/// arbitrary size: a single upstream batch (2048 rows) × 100 neighbors
/// per row = 204K rows held in spillover. The cap below is
/// `64 × BATCH_ROWS` (≈ 131072 rows) — large enough that no realistic
/// v1.0-alpha workload exercises the bound, small enough that an
/// unbounded pathological fanout produces a clean diagnostic instead
/// of an OOM. The bound applies symmetrically in
/// [`super::optional_expand::OptionalExpandOp`] (which has the same
/// spillover shape).
pub const SPILLOVER_MAX_BATCHES: usize = 64;

/// Maximum rows held in spillover before
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] fires AS A
/// FALLBACK when no per-tenant memory budget is configured. =
/// `SPILLOVER_MAX_BATCHES * BATCH_ROWS`.
///
/// The W12α (M4-64a) memory budget surface
/// ([`crate::executor::MemoryBudget`]) takes precedence over this cap
/// when configured; this constant remains the v1.0-alpha fallback for
/// tenants without a configured byte cap.
///
/// W12α fix-up LOW-2 (PR #277 retro):
/// [`crate::executor::BUDGET_FALLBACK_ROWS`] is now a true `pub const`
/// alias for THIS constant rather than a duplicate `64 * BATCH_ROWS`
/// definition — a single source of truth so a future tune of
/// [`SPILLOVER_MAX_BATCHES`] flows everywhere automatically.
pub const SPILLOVER_MAX_ROWS: usize = SPILLOVER_MAX_BATCHES * BATCH_ROWS;

/// Runaway-protection ceiling for the **uncapped** budget path (no
/// per-tenant byte cap configured — the v1.0-α default for an embedded /
/// single-tenant server).
///
/// # Why this replaces `SPILLOVER_MAX_ROWS` on the uncapped path (#980)
///
/// An *uncapped* budget is an EXPLICIT operator choice meaning "no memory
/// limit." The pre-#980 code contradicted that choice: on the uncapped
/// path the spillover / build / buffer accumulators failed the moment
/// they crossed [`SPILLOVER_MAX_ROWS`] (= 131 072 rows), even though the
/// operator was told it had no budget to respect. That tiny ceiling was
/// originally sized as a *byte-tracking-off OOM safety valve*, NOT as a
/// workload limit — but at 131 072 rows it broke every legitimate large
/// traversal / multi-pattern join (issues #980 relationship-pattern
/// joins, #994 ORDER BY > 100 K, transitively #1008 GROUP BY whose
/// upstream scan/expand/join feeds the aggregate). The SNAP web-Google
/// graph (5.1 M edges) failed entirely: `MATCH ()-[r]->() RETURN
/// count(r)` over 5.1 M edges errored with
/// `ResourceExhausted { requested_bytes: 0 }` — the misleading
/// "would reserve 0" symptom.
///
/// # The fix: lift the valve, do NOT remove it
///
/// We keep a guard (genuine OOM-runaway protection for a pathological
/// fan-out when byte tracking is off) but raise it to a value FAR above
/// any realistic single-machine graph: `1 << 32` = 4 294 967 296 rows
/// (~4.29 billion). That is ~840× the web-Google repro and well past the
/// largest practical in-memory graph target. A query that genuinely
/// materializes > 4.29 B rows in an in-memory accumulator
/// on one node is the runaway class the valve exists to catch — and it
/// still surfaces a clean structured `ResourceExhausted` diagnostic
/// rather than an OOM kill.
///
/// # Back-of-envelope (PD#5)
///
/// At the minimum [`crate::executor::budget::estimate_row_bytes`] floor
/// of 24 bytes/row, `1 << 32` rows ≈ 103 GB of accumulator — past the
/// RAM of any v1.0-α single-node deployment, so the valve fires before a
/// true OOM on realistic hardware while never tripping on a 5.1 M-edge
/// (or even a 100 M-edge) graph. A tenant that wants a TIGHTER, byte-
/// accurate bound configures a per-tenant
/// [`crate::executor::MemoryBudget`] cap, which takes precedence over
/// this guard entirely (the `has_cap` branch).
///
/// # Configurability
///
/// At v1.0-α no `pub struct *Config` is user-deserialized, so this is a
/// `const`. The
/// per-tenant byte cap ([`crate::executor::MemoryBudget::set_per_tenant_cap`])
/// is the configurable surface that overrides it; the M5 / M6 server
/// config landing is the forward consumer that will expose a tune knob.
pub const UNCAPPED_RUNAWAY_GUARD_ROWS: usize = 1usize << 32;

/// Depth cap for UNBOUNDED variable-length expansion (`*N..` / bare
/// `*`, i.e. `max: None`). ADR-186 RC-1 (the load-bearing honesty pin).
///
/// Aligned with the planner's cost-model assumption
/// `DEFAULT_UNBOUNDED_MAX_HOPS = 5` (`planner/cost/operator.rs`) so the
/// executor never silently diverges from the cardinality the planner
/// costed against. An unbounded traversal whose frontier is still
/// non-empty at this depth surfaces a structured [`ExecutionError`]
/// (`Plan(ArcQLError::ResourceExhausted)`, surface label
/// `"var-length unbounded depth cap (*N..)"`) rather than truncating —
/// a truncated traversal is a wrong answer presented as complete
/// (openCypher §3 honesty pin).
pub const VARLENGTH_UNBOUNDED_MAX_DEPTH: u32 = 5;

/// Frontier-size budget for the materialized k<=2 variable-length
/// expansion path. ADR-186 RC-1.
///
/// # Back-of-envelope (PD#5)
///
/// A var-length BFS frontier grows as `b^d` (branching factor `b`,
/// depth `d`). At a realistic dense-graph `b = 100`, `d = 5`, the
/// worst-case live frontier is `100^5 = 10^10` states — an OOM long
/// before the depth cap bites. The depth cap alone is therefore NOT
/// sufficient protection for the old materialized BFS shape; this
/// frontier-size budget remains the memory guard for k<=2, where ADR-025
/// still permits materialization. The k>=3 path uses `VarLengthStream`
/// instead, so it does not allocate a BFS frontier or a full `results`
/// vector. We cap the number of in-flight path-states + emitted rows per
/// child row at this constant; exceeding it surfaces a structured
/// [`ExecutionError`] (surface label `"var-length frontier-size budget"`)
/// — never a silent truncation.
///
/// `100_000` path-states × O(depth ≤ 5) `RelId`s each ≈ a few MB per
/// child row — bounded, orders of magnitude below the per-tenant
/// memory budget, while still admitting every realistic v1.0-alpha
/// traversal.
pub const VARLENGTH_MAX_FRONTIER: usize = 100_000;

/// One-hop relationship-traversal operator.
#[derive(Debug)]
pub struct ExpandOp {
    /// Upstream child operator producing rows that contain `from_var`.
    child: Box<PhysicalOperator>,
    /// The "from" binding (must be in the child's schema).
    from_var: BindingId,
    /// Optional rel-binding (when `(a)-[r:KNOWS]->(b)` introduces
    /// `r`).
    rel_var: Option<BindingId>,
    /// The "to" binding — fresh in this operator. Mirrored in the
    /// last slot of `schema`; the field is preserved for diagnostic
    /// + future M4-71 row-count-observer attribution.
    #[allow(dead_code)]
    to_var: BindingId,
    /// Optional rel-type filter.
    rel_type: Option<TypeId>,
    /// **F2 (PE-1 §F2)** — optional per-edge FAR-END label filter. When
    /// `Some(label)`, single-hop expansion keeps only edges whose `dst`
    /// node carries `label` (`edge.dst.label == Some(label)`). Set by the
    /// pipelined-expand fast path
    /// ([`crate::executor::Pipeline::build`]) when it folds an outer
    /// `Join(_, Scan(to, Some(label)), [to])` tail-node label into the
    /// Expand instead of a third scan + hash join. It is multiset-
    /// identical to that folded semi-join: `Scan(to, label)` yields each
    /// labeled node exactly once (unique-key semi-join == filter) and
    /// `edge.dst.label` is materialized from the SAME `read_node` label the
    /// scan filters on. `None` = no fold (every dst passes). Applies only
    /// to the single-hop path; the fast path guards `length_range.is_none()`
    /// so a var-length Expand never carries a `to_label`.
    to_label: Option<LabelId>,
    /// Direction.
    direction: Direction,
    /// MVCC visibility key.
    plan_read_lsn: Lsn,
    /// Output schema (cached): `child_schema ++ rel_var? ++ to_var`.
    schema: Vec<BindingId>,
    /// Cached child column-count for slicing the inherited prefix.
    child_columns: usize,
    /// Pre-buffered output rows from a partially-consumed inner batch
    /// of the child (a single child row may produce many output rows
    /// — we may have to spill into the next batch).
    ///
    /// W11Z fix-up MED-3 (PR #268 retro): switched from `Vec<...>` to
    /// `VecDeque<...>` so per-batch drain is O(1) instead of O(n)
    /// (`Vec::remove(0)` shifts every remaining row).
    ///
    /// W12α / M4-64a: each spilled row carries the byte cost it
    /// reserved against the per-tenant memory budget; on pop the
    /// reservation is released. When no per-tenant cap is configured,
    /// the [`UNCAPPED_RUNAWAY_GUARD_ROWS`] runaway-protection guard
    /// applies (#980 lifted the old `SPILLOVER_MAX_ROWS` valve).
    spillover: VecDeque<SpilledRow>,
    /// OOC-4 FIFO runtime. `None` preserves the legacy in-memory path.
    spill_queue: Option<Box<ExpandSpillQueue>>,
    /// When true, the OOC queue contains encoded variable-length BFS states,
    /// not completed output rows. Keeping the role explicit prevents a
    /// frontier state from ever being emitted as a query result.
    spill_queue_is_frontier: bool,
    /// A spill/read/codec fault is terminal: retrying from a shifted FIFO
    /// reader would risk duplicate or missing traversal rows.
    terminal_error: Option<ExecutionError>,
    /// Have we observed an EOS batch from the child yet?
    child_done: bool,
    /// W13α / M4-64b — optional dst-NodeId allow-set. When `Some`, the
    /// substrate-returned edges are post-filtered via
    /// [`simd_neighbor_match_mask`]: only edges whose `dst.id.raw()`
    /// appears in the allow-set survive. The list is small (typically
    /// K ≤ 4 per the SIMD helper's amortization profile); the planner-
    /// side pushdown surface (`WHERE b.id IN [...]`) lands at M4-72.
    dst_allow_set: Option<Vec<u64>>,
    /// ADR-186 / #650-C — variable-length traversal spec when the
    /// pattern is `(a)-[*N..M]->(b)`. `None` = single-hop (the M4-61
    /// scalar-rel-binding path); `Some(Cypher | Unbounded)` = BFS
    /// var-length path with the `Value::List` rel binding + the RC-1
    /// caps. `Some(Quantified)` is rejected at construction (GQL
    /// `{N,M}` is reserved to v1.1).
    length_range: Option<LengthRange>,
    /// Suspendable k>=3 variable-length stream for the current child
    /// row. Drained before pulling more upstream rows.
    active_vl: Option<VarLengthStream>,
    /// OOC-4's level-order stream for the shipped k<=2 BFS path. Unlike the
    /// legacy eager implementation, its frontier is the spill queue and its
    /// current adjacency is an owned cursor, so neither fan-out nor the
    /// frontier is materialized wholesale.
    active_spill_vl: Option<Box<SpillVarLengthStream>>,
    /// Unconsumed child rows from an upstream batch that was suspended
    /// because a k>=3 variable-length stream filled the output batch.
    ///
    /// The current row is owned by `active_vl`; this queue preserves
    /// the remaining sibling rows from the same upstream batch so a
    /// later `next_batch` does not pull a fresh child batch and drop
    /// the tail (#814-class mid-batch return bug).
    pending_child_rows: VecDeque<Vec<Value>>,
}

/// One spilled row + its budget reservation. The byte count is
/// captured at push time so it matches the size estimated when the
/// budget was reserved.
#[derive(Debug)]
struct SpilledRow {
    row: Vec<Value>,
    /// Bytes reserved against the per-tenant budget for this row.
    /// `0` when no cap was set at push time (the row-count fallback
    /// applied).
    reserved_bytes: u64,
}

impl ExpandOp {
    /// Construct an `ExpandOp` from a [`crate::logical_plan::LogicalExpand`]
    /// + the upstream child operator.
    ///
    /// Argument count reflects the LogicalExpand's surface (8 fields
    /// per ADR-038 §2 D-24); a builder helper would add indirection
    /// without clarity (every call-site sets every field). Allow
    /// `clippy::too_many_arguments`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child: PhysicalOperator,
        from_var: BindingId,
        rel_var: Option<BindingId>,
        to_var: BindingId,
        rel_type: Option<TypeId>,
        direction: Direction,
        length_range: Option<LengthRange>,
        plan_read_lsn: Lsn,
    ) -> Result<Self, ExecutionError> {
        // ADR-186 / #650-C: openCypher `*N..M` (`Cypher`) + `*N..`/`*`
        // (`Unbounded`) are EXECUTED via the BFS path below. The GQL
        // `{N,M}` form (`Quantified`) stays reserved to v1.1 (rejected
        // upstream at `semantic/type_check.rs`); reject defensively
        // here too so a mis-lowered plan cannot silently mis-execute.
        if matches!(length_range, Some(LengthRange::Quantified { .. })) {
            return Err(ExecutionError::NotImplemented {
                feature: "ExpandOp GQL quantified length range (`{N,M}`)".into(),
                target_slice: "v1.1 (GQL length range)".into(),
                section: "ADR-038 §2 D-9 + D-16".into(),
            });
        }
        let child_columns = child.schema().len();
        let mut schema: Vec<BindingId> = child.schema().to_vec();
        if let Some(rv) = rel_var {
            schema.push(rv);
        }
        schema.push(to_var);
        Ok(Self {
            child: Box::new(child),
            from_var,
            rel_var,
            to_var,
            rel_type,
            to_label: None,
            direction,
            plan_read_lsn,
            schema,
            child_columns,
            spillover: VecDeque::new(),
            spill_queue: None,
            spill_queue_is_frontier: false,
            terminal_error: None,
            child_done: false,
            dst_allow_set: None,
            length_range,
            active_vl: None,
            active_spill_vl: None,
            pending_child_rows: VecDeque::new(),
        })
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// W13α / M4-64b — attach a dst-NodeId allow-set. Only edges whose
    /// destination node-id appears in `allow` survive the
    /// substrate-side scan. Internally routed through
    /// [`simd_neighbor_match_mask`] for SIMD-batched membership.
    ///
    /// # Forward-pin (M4-72)
    ///
    /// At v1.0-alpha the planner does not push down `WHERE b.id IN [...]`
    /// predicates into ExpandOp; the wiring lands at M4-72 alongside
    /// the cost-model pushdown rules. The constructor exists today so
    /// the SIMD helper has a real consumer surface (the unit + bench
    /// invoke this path directly).
    #[must_use]
    pub fn with_dst_allow_set(mut self, allow: Vec<u64>) -> Self {
        self.dst_allow_set = Some(allow);
        self
    }

    /// Whether the dst-NodeId allow-set is active for this operator.
    /// Tests + EXPLAIN annotation use this to assert the SIMD route.
    #[must_use]
    pub fn uses_simd_dst_allow_set(&self) -> bool {
        self.dst_allow_set.as_ref().is_some_and(|s| !s.is_empty())
    }

    /// **F2 (PE-1 §F2)** — attach a per-edge FAR-END label filter. Only
    /// single-hop edges whose `dst` node carries `label` survive. Chainable.
    /// Set by the pipelined-expand fast path when it folds a `(to:Label)`
    /// tail-node label into the Expand (see the `to_label` field docs for
    /// the multiset-identity argument).
    #[must_use]
    pub fn with_to_label(mut self, label: LabelId) -> Self {
        self.to_label = Some(label);
        self
    }

    /// The far-end label filter, if the F2 pipelined-expand fast path
    /// folded a `(to:Label)` tail-node label into this Expand. Tests assert
    /// the fold via this accessor.
    #[must_use]
    pub fn to_label(&self) -> Option<LabelId> {
        self.to_label
    }

    /// Enable the OOC-4 FIFO spillover queue. Passing `None` preserves the
    /// shipped in-memory behavior. A configured MemoryBudget cap is checked
    /// before the first child batch is pulled (#1524).
    pub fn with_spillover_target(
        mut self,
        target: Option<ExpandSpillTarget>,
    ) -> Result<Self, ExecutionError> {
        self.spill_queue = target.map(|target| Box::new(ExpandSpillQueue::new(target)));
        Ok(self)
    }

    /// Pull the next batch.
    ///
    /// # Spillover bound (W12α / M4-64a integration)
    ///
    /// At high-fanout (e.g., 2048 rows × 100 neighbors), a single
    /// upstream batch can produce tens of thousands of output rows.
    /// They overflow into the per-operator spillover queue for the
    /// next call to drain. The spillover is bounded by:
    ///
    /// 1. **OOC-4 target:** a configured [`crate::executor::MemoryBudget`]
    ///    cap is mandatory. The resident FIFO prefix is charged to that cap;
    ///    overflow is streamed into OOC-1 runs and only scratch quota,
    ///    headroom, or I/O failure aborts the traversal.
    /// 2. **Legacy capped path:** without an OOC target, each overflow row
    ///    reserves [`estimate_row_bytes`]; exceeding the cap surfaces
    ///    [`crate::semantic::error::ArcQLError::ResourceExhausted`] via
    ///    [`ExecutionError::Plan`].
    /// 3. **Runaway-protection guard** ([`UNCAPPED_RUNAWAY_GUARD_ROWS`])
    ///    when no per-tenant cap is configured (v1.0-alpha default). An
    ///    uncapped budget means "no memory limit," so the spillover grows
    ///    with the actual fan-out; only a true runaway (#980 lifted the
    ///    old 131 072-row `SPILLOVER_MAX_ROWS` valve far above any
    ///    single-node graph) surfaces
    ///    [`crate::semantic::error::ArcQLError::ResourceExhausted`]
    ///    via [`ExecutionError::Plan`] (W12α fix-up LOW-4 promoted
    ///    from the prior `ExecutionError::Eval` to share the variant
    ///    with the byte-cap path).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        let result = (|| {
            if let Some(queue) = self.spill_queue.as_mut() {
                queue.prepare(ctx)?;
            }
            self.next_batch_inner(ctx, substrate)
        })();
        match result {
            Ok(batch) => {
                if batch.is_empty() {
                    // EOS: end the OOC-1 epoch and zeroize its ephemeral key
                    // immediately instead of waiting for operator drop.
                    self.spill_queue = None;
                }
                Ok(batch)
            }
            Err(error) => {
                if self.spill_queue.is_some() {
                    self.spill_queue = None;
                    self.terminal_error = Some(error.clone());
                }
                Err(error)
            }
        }
    }

    fn next_batch_inner<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        let mut out = Batch::with_capacity(self.schema.len());
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());

        // Drain spillover first. W11Z MED-3: VecDeque::pop_front is
        // O(1), replacing the prior O(n) Vec::remove(0). Each pop
        // releases the row's budget reservation.
        while !out.is_full() && !self.spill_queue_is_frontier {
            if let Some(queue) = self.spill_queue.as_mut() {
                match queue.pop()? {
                    Some(row) => {
                        if !out.push_row(row) {
                            return Err(ExecutionError::Eval(
                                "ExpandOp: OOC-4 drain overflow despite fullness guard".into(),
                            ));
                        }
                    }
                    None => break,
                }
            } else {
                match self.spillover.pop_front() {
                    Some(spilled) => {
                        if spilled.reserved_bytes > 0 {
                            budget.release(ctx.tenant(), spilled.reserved_bytes);
                        }
                        if !out.push_row(spilled.row) {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        if out.is_full() {
            return Ok(out);
        }

        self.drain_active_var_length(ctx, substrate, &mut out)?;
        if out.is_full() {
            return Ok(out);
        }
        self.drain_pending_child_rows(ctx, substrate, &mut out, has_cap, &budget)?;
        if out.is_full() {
            return Ok(out);
        }

        // Pull upstream batches until we either fill the output or
        // observe child EOS.
        loop {
            if out.is_full() {
                break;
            }
            if self.child_done && self.spillover_is_empty() {
                break;
            }
            if !has_cap && self.spillover_len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(spillover_fallback_err(self.spillover_len()));
            }
            if !self.child_done {
                let child_batch = self.child.next_batch(ctx, substrate)?;
                if child_batch.is_empty() {
                    self.child_done = true;
                    if self.spillover_is_empty() {
                        break;
                    }
                    continue;
                }
                let from_idx = self.find_child_index(self.from_var)?;
                let mut rows: VecDeque<Vec<Value>> = child_batch.into_rows().into_iter().collect();
                self.drain_child_rows(
                    ctx, substrate, &mut out, from_idx, &mut rows, has_cap, &budget,
                )?;
                if out.is_full() {
                    self.pending_child_rows = rows;
                    break;
                }
            } else {
                break;
            }
        }
        Ok(out)
    }

    fn spillover_is_empty(&self) -> bool {
        self.spill_queue
            .as_ref()
            .map_or_else(|| self.spillover.is_empty(), |queue| queue.is_empty())
    }

    fn spillover_len(&self) -> usize {
        self.spill_queue
            .as_ref()
            .map_or_else(|| self.spillover.len(), |queue| queue.len())
    }

    fn drain_active_var_length<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        out: &mut Batch,
    ) -> Result<(), ExecutionError> {
        if self.active_spill_vl.is_some() {
            let mut stream = self.active_spill_vl.take().ok_or_else(|| {
                frontier_state_error("spillable variable-length stream disappeared")
            })?;
            let queue = self.spill_queue.as_mut().ok_or_else(|| {
                frontier_state_error("spillable variable-length stream has no frontier queue")
            })?;
            let mut finished = false;
            while !out.is_full() {
                match stream.next_row(ctx, substrate, queue)? {
                    Some(row) => {
                        if !out.push_row(row) {
                            return Err(frontier_state_error(
                                "spillable BFS row did not fit a non-full batch",
                            ));
                        }
                    }
                    None => {
                        finished = true;
                        break;
                    }
                }
            }
            if finished {
                self.spill_queue_is_frontier = false;
            } else {
                self.active_spill_vl = Some(stream);
            }
            return Ok(());
        }

        while !out.is_full() {
            let Some(stream) = self.active_vl.as_mut() else {
                return Ok(());
            };
            match stream.next_row(ctx, substrate)? {
                Some(row) => {
                    if !out.push_row(row) {
                        break;
                    }
                }
                None => {
                    self.active_vl = None;
                }
            }
        }
        Ok(())
    }

    fn drain_pending_child_rows<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        out: &mut Batch,
        has_cap: bool,
        budget: &MemoryBudget,
    ) -> Result<(), ExecutionError> {
        if self.pending_child_rows.is_empty() {
            return Ok(());
        }
        let from_idx = self.find_child_index(self.from_var)?;
        let mut rows = std::mem::take(&mut self.pending_child_rows);
        self.drain_child_rows(ctx, substrate, out, from_idx, &mut rows, has_cap, budget)?;
        self.pending_child_rows = rows;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn drain_child_rows<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        out: &mut Batch,
        from_idx: usize,
        rows: &mut VecDeque<Vec<Value>>,
        has_cap: bool,
        budget: &MemoryBudget,
    ) -> Result<(), ExecutionError> {
        while !out.is_full() {
            let Some(row) = rows.pop_front() else {
                return Ok(());
            };
            self.process_child_row(ctx, substrate, out, from_idx, row, has_cap, budget)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_child_row<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        out: &mut Batch,
        from_idx: usize,
        row: Vec<Value>,
        has_cap: bool,
        budget: &MemoryBudget,
    ) -> Result<(), ExecutionError> {
        let from_value = row.get(from_idx).ok_or_else(|| {
            ExecutionError::Eval("ExpandOp: child row missing `from` binding column".into())
        })?;
        let from_node = match from_value {
            Value::Node(n) => n.clone(),
            Value::Null => return Ok(()), // optional-MATCH passthrough
            _ => {
                return Err(ExecutionError::Eval(
                    "ExpandOp: `from` binding is not a Node".into(),
                ));
            }
        };

        // ADR-186 / #650-C — variable-length BFS branch.
        // `length_range` is `Some(Cypher | Unbounded)` here
        // (`Quantified` is rejected at construction). Each child row
        // fans out into one output row per distinct path. The dst-
        // allow-set SIMD pushdown is single-hop-only at v1.0-alpha.
        if let Some(lr) = self.length_range.clone() {
            let (_min, max) = var_length_bounds(&lr);
            if self.spill_queue.is_some() && max.is_some_and(|m| m < 3) {
                let queue = self.spill_queue.as_mut().ok_or_else(|| {
                    frontier_state_error("spill target disappeared before BFS initialization")
                })?;
                if !queue.is_empty() {
                    return Err(frontier_state_error(
                        "output spillover was not drained before BFS frontier reuse",
                    ));
                }
                self.spill_queue_is_frontier = true;
                self.active_spill_vl = Some(Box::new(SpillVarLengthStream::new(
                    ctx,
                    queue,
                    row.clone(),
                    from_node.clone(),
                    self.rel_var.is_some(),
                    self.rel_type,
                    self.direction,
                    self.plan_read_lsn,
                    &lr,
                )?));
                self.drain_active_var_length(ctx, substrate, out)?;
            } else if max.is_none_or(|m| m >= 3) {
                self.active_vl = Some(VarLengthStream::new(
                    ctx,
                    substrate,
                    row.clone(),
                    from_node.clone(),
                    self.rel_var.is_some(),
                    self.rel_type,
                    self.direction,
                    self.plan_read_lsn,
                    &lr,
                )?);
                self.drain_active_var_length(ctx, substrate, out)?;
            } else {
                let path_rows = self.var_length_paths(ctx, substrate, &row, &from_node, &lr)?;
                for new_row in path_rows {
                    self.emit_or_spill(out, new_row, has_cap, budget, ctx)?;
                }
            }
            return Ok(());
        }

        // An attached OOC-4 target consumes the adjacency cursor directly,
        // so a supernode does not first materialize its entire adjacency Vec
        // before rows reach the bounded spill queue. Keep allow-set filtering
        // incremental on this path for the same reason.
        if self.spill_queue.is_some() {
            let edges = substrate.expand_cursor_with_context(
                ctx,
                from_node.id,
                self.rel_type,
                self.direction,
                self.plan_read_lsn,
            )?;
            for edge in edges {
                ctx.cancellation().check()?;
                let edge = edge?;
                if self
                    .to_label
                    .is_some_and(|want| edge.dst.label != Some(want))
                {
                    continue;
                }
                if let Some(allow) = self.dst_allow_set.as_ref().filter(|s| !s.is_empty())
                    && !allow.contains(&edge.dst.id.raw())
                {
                    continue;
                }
                let mut new_row = row.clone();
                if self.rel_var.is_some() {
                    new_row.push(Value::Relationship(edge.rel));
                }
                new_row.push(Value::Node(edge.dst));
                self.emit_or_spill(out, new_row, has_cap, budget, ctx)?;
            }
            return Ok(());
        }

        // Single-hop (M4-61) path.
        let mut edges = substrate.expand_with_context(
            ctx,
            from_node.id,
            self.rel_type,
            self.direction,
            self.plan_read_lsn,
        )?;
        // F2 (PE-1 §F2) — per-edge FAR-END label filter. When the
        // pipelined-expand fast path folded a `(to:Label)` tail-node label
        // into this Expand, keep only edges whose `dst` node carries that
        // label. Multiset-identical to the folded `Join(_, Scan(to,
        // Some(label)), [to])` semi-join (each labeled node appears once in
        // the scan → semi-join == filter; `edge.dst.label` and the scan's
        // label both derive from the same `read_node` label). Runs before
        // the dst-allow-set SIMD filter (both are commuting post-substrate
        // retains).
        if let Some(want) = self.to_label {
            edges.retain(|edge| edge.dst.label == Some(want));
        }
        // W13α / M4-64b — SIMD post-substrate dst filter.
        if let Some(allow) = self.dst_allow_set.as_ref().filter(|s| !s.is_empty()) {
            let candidates: Vec<u64> = edges.iter().map(|e| e.dst.id.raw()).collect();
            let mask = simd_neighbor_match_mask(&candidates, allow);
            let mut keep_idx = 0;
            edges.retain(|_| {
                let k = mask[keep_idx];
                keep_idx += 1;
                k
            });
        }
        for edge in edges {
            let mut new_row = row.clone();
            if self.rel_var.is_some() {
                new_row.push(Value::Relationship(edge.rel.clone()));
            }
            new_row.push(Value::Node(edge.dst.clone()));
            self.emit_or_spill(out, new_row, has_cap, budget, ctx)?;
        }
        Ok(())
    }

    /// Look up `binding`'s column index in the child schema's
    /// (= the inherited prefix of our own schema).
    fn find_child_index(&self, binding: BindingId) -> Result<usize, ExecutionError> {
        self.schema[..self.child_columns]
            .iter()
            .position(|&b| b == binding)
            .ok_or_else(|| {
                ExecutionError::Eval(format!(
                    "ExpandOp: binding {:?} not found in child schema",
                    binding
                ))
            })
    }

    /// Enumerate every distinct variable-length path from `from_node`
    /// whose hop-count lands in the `length_range`-derived `[min, max]`
    /// window, returning one output row per path. ADR-186 / #650-C.
    ///
    /// openCypher v9 §3 semantics:
    /// - **RC-3 edge-uniqueness:** a relationship is never traversed
    ///   twice within one path (per-PATH `HashSet<RelId>`); node
    ///   repeats ARE allowed. Guarantees termination on cyclic graphs.
    /// - **RC-2 rel binding:** the rel column (when `rel_var` is
    ///   `Some`) is a `Value::List` of the path's relationships in
    ///   traversal order.
    /// - **`*0`:** depth-0 emits `from_node` bound to `to_var` with an
    ///   empty rel-list, before the first frontier step.
    /// - **RC-1 caps:** an unbounded `max` is bounded by
    ///   [`VARLENGTH_UNBOUNDED_MAX_DEPTH`]; the live frontier + emitted
    ///   rows are bounded by [`VARLENGTH_MAX_FRONTIER`]. Reaching
    ///   either surfaces a structured error — never a silent
    ///   truncation.
    fn var_length_paths<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        child_row: &[Value],
        from_node: &NodeView,
        length_range: &LengthRange,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let (min, max) = var_length_bounds(length_range);
        let bind_rel = self.rel_var.is_some();
        let mut results: Vec<Vec<Value>> = Vec::new();

        // RC-2 / `*0`: the depth-0 identity path = the start node
        // itself with an empty rel-list. Emitted only when `min == 0`.
        if min == 0 {
            results.push(build_path_row(child_row, &[], from_node, bind_rel));
        }

        // Effective depth ceiling: an explicit `max` for the bounded
        // forms, else the unbounded cap (RC-1). `unbounded` records
        // whether a non-empty frontier at the ceiling is an honesty
        // error (vs. a legitimate bounded stop).
        let unbounded = max.is_none();
        let ceiling = max.unwrap_or(VARLENGTH_UNBOUNDED_MAX_DEPTH);

        // BFS frontier. Each state = (current node, rels so far, the
        // per-PATH visited-RelId set). `depth` is the hop count of the
        // states currently in `frontier`.
        let mut frontier: Vec<VarLengthState> = vec![VarLengthState {
            node: from_node.clone(),
            rels: Vec::new(),
            visited: HashSet::new(),
        }];
        let mut depth: u32 = 0;

        while !frontier.is_empty() && depth < ceiling {
            ctx.cancellation().check()?;
            depth += 1;
            let mut next: Vec<VarLengthState> = Vec::with_capacity(frontier.len());
            for state in &frontier {
                let edges = substrate.expand_with_context(
                    ctx,
                    state.node.id,
                    self.rel_type,
                    self.direction,
                    self.plan_read_lsn,
                )?;
                for edge in edges {
                    // RC-3: prune edges already used on THIS path. Node
                    // repeats are allowed (we never dedup on node id).
                    if state.visited.contains(&edge.rel.id) {
                        continue;
                    }
                    let mut rels = state.rels.clone();
                    rels.push(edge.rel.clone());
                    let mut visited = state.visited.clone();
                    visited.insert(edge.rel.id);

                    // Emit when the hop-count is within the window.
                    if depth >= min {
                        results.push(build_path_row(child_row, &rels, &edge.dst, bind_rel));
                    }
                    next.push(VarLengthState {
                        node: edge.dst.clone(),
                        rels,
                        visited,
                    });

                    // RC-1 frontier-size budget: bound in-flight states
                    // + emitted rows (b^d blowup guard). Error, never
                    // truncate.
                    if next.len() + results.len() > VARLENGTH_MAX_FRONTIER {
                        return Err(var_length_frontier_cap_err(next.len() + results.len()));
                    }
                }
            }
            frontier = next;
        }

        // RC-1 depth cap (the honesty pin): an UNBOUNDED traversal that
        // could continue PAST the ceiling cannot be answered completely
        // within the cap — error rather than return a silently-truncated
        // (wrong) result set. We probe one level: a depth-`ceiling`
        // state with an un-pruned outgoing edge proves a length-
        // `(ceiling+1)` path exists. A non-empty frontier of pure
        // dead-ends (or states whose only edges re-use the path) is
        // COMPLETE despite being non-empty — no error.
        if unbounded {
            for state in &frontier {
                let edges = substrate.expand_with_context(
                    ctx,
                    state.node.id,
                    self.rel_type,
                    self.direction,
                    self.plan_read_lsn,
                )?;
                if edges.iter().any(|e| !state.visited.contains(&e.rel.id)) {
                    return Err(var_length_depth_cap_err(ceiling));
                }
            }
        }

        Ok(results)
    }

    /// Push `new_row` into `out`, spilling the overflow into the
    /// per-operator spillover queue under the same per-tenant byte
    /// budget / row-count fallback as the single-hop path. Factored
    /// (ADR-186) so the var-length BFS emission and the single-hop
    /// emission share one bounded-spillover code path.
    fn emit_or_spill(
        &mut self,
        out: &mut Batch,
        new_row: Vec<Value>,
        has_cap: bool,
        budget: &MemoryBudget,
        ctx: &ExecutionContext,
    ) -> Result<(), ExecutionError> {
        if !out.push_row(new_row.clone()) {
            if let Some(queue) = self.spill_queue.as_mut() {
                return queue.push(ctx, new_row);
            }
            // Spillover. Reserve budget if a cap is set; else apply the
            // row-count fallback.
            let reserved_bytes = if has_cap {
                let bytes = estimate_row_bytes(&new_row) as u64;
                budget.try_reserve_unscoped(ctx.tenant(), bytes, "ExpandOp spillover")?;
                bytes
            } else {
                if self.spillover.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                    return Err(spillover_fallback_err(self.spillover.len()));
                }
                0
            };
            self.spillover.push_back(SpilledRow {
                row: new_row,
                reserved_bytes,
            });
        }
        Ok(())
    }
}

// Mark `BATCH_ROWS` used so its export is exercised even when the
// expand operator's batch fills via spillover instead of bulk push.
#[allow(dead_code)]
const _: usize = BATCH_ROWS;

/// Render the W11Z #272 row-count-fallback error consistently for both
/// the pre-loop check and the inner per-edge check.
///
/// W12α fix-up LOW-4 (PR #277 retro): promoted from
/// [`ExecutionError::Eval`] (string-error) to
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] so the
/// row-count-fallback fault carries the same variant as the byte-cap
/// path — the M5-07 / M5-11 / M5-13 transport-layer renderers can map
/// both surfaces to the same HTTP-429 / equivalent rate-limit class
/// via a single match arm. The byte-vs-row unit mismatch is acceptable
/// because the `feature` discriminator carries the surface label.
fn spillover_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "ExpandOp runaway-guard".to_owned(),
        requested_bytes: 0,
        // #980 — report the lifted runaway-protection ceiling, not the
        // old 131 072-row valve that broke legitimate large traversals.
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

fn build_path_row(
    child_row: &[Value],
    rels: &[RelView],
    to_node: &NodeView,
    bind_rel: bool,
) -> Vec<Value> {
    let mut row = child_row.to_vec();
    if bind_rel {
        row.push(Value::List(
            rels.iter().cloned().map(Value::Relationship).collect(),
        ));
    }
    row.push(Value::Node(to_node.clone()));
    row
}

/// Suspendable level-order stream for the legacy k<=2 BFS shape.
///
/// Each queue row is one exact path state `(terminal node, relationships,
/// depth)`. Relationship IDs are derived from the path itself, so the
/// cycle-safety set has no false positives and cannot under-visit: membership
/// is an O(depth) scan and depth is at most two on this path. The queue owns
/// every cold state, while only one adjacency cursor and one decoded queue
/// frame are resident.
struct SpillVarLengthStream {
    child_row: Vec<Value>,
    bind_rel: bool,
    rel_type: Option<TypeId>,
    direction: Direction,
    plan_read_lsn: Lsn,
    min: u32,
    ceiling: u32,
    current: Option<(SpillVarLengthState, BoundEdgeCursor)>,
    identity: Option<NodeView>,
}

struct SpillVarLengthState {
    node: NodeView,
    rels: Vec<RelView>,
    depth: u32,
}

impl std::fmt::Debug for SpillVarLengthStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillVarLengthStream")
            .field("child_row_len", &self.child_row.len())
            .field("bind_rel", &self.bind_rel)
            .field("rel_type", &self.rel_type)
            .field("direction", &self.direction)
            .field("min", &self.min)
            .field("ceiling", &self.ceiling)
            .field("has_current_cursor", &self.current.is_some())
            .field("identity_pending", &self.identity.is_some())
            .finish_non_exhaustive()
    }
}

impl SpillVarLengthStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &ExecutionContext,
        queue: &mut ExpandSpillQueue,
        child_row: Vec<Value>,
        from_node: NodeView,
        bind_rel: bool,
        rel_type: Option<TypeId>,
        direction: Direction,
        plan_read_lsn: Lsn,
        length_range: &LengthRange,
    ) -> Result<Self, ExecutionError> {
        let (min, max) = var_length_bounds(length_range);
        let ceiling = max.ok_or_else(|| {
            frontier_state_error("unbounded variable-length traversal routed to the k<=2 FIFO")
        })?;
        let identity = (min == 0).then(|| from_node.clone());
        queue.push(
            ctx,
            encode_spill_vl_state(SpillVarLengthState {
                node: from_node,
                rels: Vec::new(),
                depth: 0,
            })?,
        )?;
        Ok(Self {
            child_row,
            bind_rel,
            rel_type,
            direction,
            plan_read_lsn,
            min,
            ceiling,
            current: None,
            identity,
        })
    }

    fn next_row<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        queue: &mut ExpandSpillQueue,
    ) -> Result<Option<Vec<Value>>, ExecutionError> {
        if let Some(root) = self.identity.take() {
            ctx.cancellation().check()?;
            return Ok(Some(build_path_row(
                &self.child_row,
                &[],
                &root,
                self.bind_rel,
            )));
        }

        loop {
            ctx.cancellation().check()?;
            if let Some((state, cursor)) = self.current.as_mut() {
                match cursor.next() {
                    Some(Ok(edge)) => {
                        if state.rels.iter().any(|rel| rel.id == edge.rel.id) {
                            continue;
                        }
                        let next_depth = state.depth.saturating_add(1);
                        let mut rels = Vec::new();
                        rels.try_reserve_exact(state.rels.len().saturating_add(1))
                            .map_err(|_| frontier_resource_limit("variable-length path state"))?;
                        rels.extend(state.rels.iter().cloned());
                        rels.push(edge.rel);
                        if next_depth < self.ceiling {
                            let mut queued_rels = Vec::new();
                            queued_rels.try_reserve_exact(rels.len()).map_err(|_| {
                                frontier_resource_limit("queued variable-length path state")
                            })?;
                            queued_rels.extend(rels.iter().cloned());
                            queue.push(
                                ctx,
                                encode_spill_vl_state(SpillVarLengthState {
                                    node: edge.dst.clone(),
                                    rels: queued_rels,
                                    depth: next_depth,
                                })?,
                            )?;
                        }
                        if next_depth >= self.min {
                            return Ok(Some(build_path_row(
                                &self.child_row,
                                &rels,
                                &edge.dst,
                                self.bind_rel,
                            )));
                        }
                    }
                    Some(Err(error)) => return Err(error.into()),
                    None => self.current = None,
                }
                continue;
            }

            let Some(row) = queue.pop()? else {
                return Ok(None);
            };
            let state = decode_spill_vl_state(row)?;
            if state.depth > self.ceiling {
                return Err(frontier_corruption(
                    "restored frontier depth exceeds the traversal ceiling",
                ));
            }
            if state.depth >= self.ceiling {
                continue;
            }
            let cursor = substrate.expand_cursor_with_context(
                ctx,
                state.node.id,
                self.rel_type,
                self.direction,
                self.plan_read_lsn,
            )?;
            self.current = Some((state, cursor));
        }
    }
}

fn encode_spill_vl_state(state: SpillVarLengthState) -> Result<Vec<Value>, ExecutionError> {
    let mut relationships = Vec::new();
    relationships
        .try_reserve_exact(state.rels.len())
        .map_err(|_| frontier_resource_limit("encoded frontier relationship list"))?;
    relationships.extend(state.rels.into_iter().map(Value::Relationship));

    let mut row = Vec::new();
    row.try_reserve_exact(3)
        .map_err(|_| frontier_resource_limit("encoded frontier row"))?;
    row.push(Value::Node(state.node));
    row.push(Value::List(relationships));
    row.push(Value::Integer(i64::from(state.depth)));
    Ok(row)
}

fn decode_spill_vl_state(row: Vec<Value>) -> Result<SpillVarLengthState, ExecutionError> {
    let mut values = row.into_iter();
    let node = match values.next() {
        Some(Value::Node(node)) => node,
        _ => return Err(frontier_corruption("frontier state has no terminal Node")),
    };
    let rels = match values.next() {
        Some(Value::List(encoded_rels)) => {
            let mut rels = Vec::new();
            rels.try_reserve_exact(encoded_rels.len())
                .map_err(|_| frontier_resource_limit("decoded frontier relationship list"))?;
            for value in encoded_rels {
                match value {
                    Value::Relationship(rel) => rels.push(rel),
                    _ => {
                        return Err(frontier_corruption(
                            "frontier relationship list contains a non-relationship",
                        ));
                    }
                }
            }
            rels
        }
        _ => {
            return Err(frontier_corruption(
                "frontier state has no relationship list",
            ));
        }
    };
    let depth = match values.next() {
        Some(Value::Integer(depth)) => u32::try_from(depth)
            .map_err(|_| frontier_corruption("frontier depth is outside u32 range"))?,
        _ => return Err(frontier_corruption("frontier state has no integer depth")),
    };
    if values.next().is_some() || rels.len() != depth as usize {
        return Err(frontier_corruption(
            "frontier state depth does not match its relationship path",
        ));
    }
    if rels
        .iter()
        .enumerate()
        .any(|(index, rel)| rels[..index].iter().any(|prior| prior.id == rel.id))
    {
        return Err(frontier_corruption(
            "frontier state repeats a relationship within one path",
        ));
    }
    Ok(SpillVarLengthState { node, rels, depth })
}

fn frontier_corruption(detail: &str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::Corruption,
        detail: format!("expand frontier restore failed: {detail}"),
    })
}

fn frontier_state_error(detail: &str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::FrontierState,
        detail: detail.to_owned(),
    })
}

fn frontier_resource_limit(allocation: &str) -> ExecutionError {
    ExecutionError::Spill(ExecutorSpillError::Failure {
        kind: ExecutorSpillFailureKind::ResourceLimit,
        detail: format!("could not allocate {allocation}"),
    })
}

/// Suspendable DFS path stream for k>=3 variable-length expansion.
///
/// # Performance budget (PD #5)
///
/// The stream keeps at most one cursor per depth level plus one shared
/// path (`rels` + `visited`). With the v1.0 unbounded ceiling of
/// `VARLENGTH_UNBOUNDED_MAX_DEPTH = 5`, in-flight executor memory is
/// `O(5)` cursors and one path instead of the old `O(paths * depth)`
/// cloned BFS frontier and `results` vector. Each descend opens one
/// `expand_cursor` and each emitted row is produced on demand.
struct VarLengthStream {
    child_row: Vec<Value>,
    bind_rel: bool,
    rel_type: Option<TypeId>,
    direction: Direction,
    plan_read_lsn: Lsn,
    min: u32,
    ceiling: u32,
    unbounded: bool,
    stack: Vec<VarLengthLevel>,
    rels: Vec<RelView>,
    visited: HashSet<RelId>,
    emitted_identity: bool,
}

impl std::fmt::Debug for VarLengthStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarLengthStream")
            .field("child_row_len", &self.child_row.len())
            .field("bind_rel", &self.bind_rel)
            .field("rel_type", &self.rel_type)
            .field("direction", &self.direction)
            .field("plan_read_lsn", &self.plan_read_lsn)
            .field("min", &self.min)
            .field("ceiling", &self.ceiling)
            .field("unbounded", &self.unbounded)
            .field("stack_depth", &self.stack.len())
            .field("rels_len", &self.rels.len())
            .field("visited_len", &self.visited.len())
            .field("emitted_identity", &self.emitted_identity)
            .finish()
    }
}

struct VarLengthLevel {
    node: NodeView,
    cursor: BoundEdgeCursor,
    entered_rel: Option<RelId>,
}

impl VarLengthStream {
    #[allow(clippy::too_many_arguments)]
    fn new<S: ExecutorSubstrate>(
        ctx: &ExecutionContext,
        substrate: &S,
        child_row: Vec<Value>,
        from_node: NodeView,
        bind_rel: bool,
        rel_type: Option<TypeId>,
        direction: Direction,
        plan_read_lsn: Lsn,
        length_range: &LengthRange,
    ) -> Result<Self, ExecutionError> {
        let (min, max) = var_length_bounds(length_range);
        let ceiling = max.unwrap_or(VARLENGTH_UNBOUNDED_MAX_DEPTH);
        let cursor = substrate.expand_cursor_with_context(
            ctx,
            from_node.id,
            rel_type,
            direction,
            plan_read_lsn,
        )?;
        Ok(Self {
            child_row,
            bind_rel,
            rel_type,
            direction,
            plan_read_lsn,
            min,
            ceiling,
            unbounded: max.is_none(),
            stack: vec![VarLengthLevel {
                node: from_node,
                cursor,
                entered_rel: None,
            }],
            rels: Vec::new(),
            visited: HashSet::new(),
            emitted_identity: false,
        })
    }

    fn next_row<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Option<Vec<Value>>, ExecutionError> {
        if !self.emitted_identity {
            self.emitted_identity = true;
            if self.min == 0 {
                let root = self
                    .stack
                    .first()
                    .expect("VarLengthStream always starts with a root level");
                ctx.cancellation().check()?;
                return Ok(Some(build_path_row(
                    &self.child_row,
                    &[],
                    &root.node,
                    self.bind_rel,
                )));
            }
        }

        loop {
            ctx.cancellation().check()?;
            let depth = match self.stack.len() {
                0 => return Ok(None),
                len => (len - 1) as u32,
            };
            let Some(level) = self.stack.last_mut() else {
                return Ok(None);
            };

            if depth == self.ceiling && !self.unbounded {
                self.pop_level();
                continue;
            }

            match level.cursor.next() {
                Some(Ok(edge)) => {
                    if self.visited.contains(&edge.rel.id) {
                        continue;
                    }
                    if depth == self.ceiling {
                        return Err(var_length_depth_cap_err(self.ceiling));
                    }

                    self.visited.insert(edge.rel.id);
                    self.rels.push(edge.rel.clone());
                    let next_depth = depth + 1;
                    let row = if next_depth >= self.min {
                        Some(build_path_row(
                            &self.child_row,
                            &self.rels,
                            &edge.dst,
                            self.bind_rel,
                        ))
                    } else {
                        None
                    };
                    let cursor = substrate.expand_cursor_with_context(
                        ctx,
                        edge.dst.id,
                        self.rel_type,
                        self.direction,
                        self.plan_read_lsn,
                    )?;
                    self.stack.push(VarLengthLevel {
                        node: edge.dst,
                        cursor,
                        entered_rel: Some(edge.rel.id),
                    });
                    ctx.cancellation().check()?;
                    if let Some(row) = row {
                        return Ok(Some(row));
                    }
                }
                Some(Err(err)) => return Err(err.into()),
                None => self.pop_level(),
            }
        }
    }

    fn pop_level(&mut self) {
        if let Some(level) = self.stack.pop()
            && let Some(rel_id) = level.entered_rel
        {
            self.visited.remove(&rel_id);
            let _ = self.rels.pop();
        }
    }
}

/// One in-flight BFS path-state for variable-length traversal
/// (ADR-186 / #650-C). Carries the per-PATH visited-`RelId` set (RC-3)
/// so edge-uniqueness is enforced INDEPENDENTLY per path — a global
/// visited-set would wrongly prune legal diamond paths (two distinct
/// paths sharing no edge but converging on a node).
struct VarLengthState {
    /// The path's current terminal node.
    node: NodeView,
    /// Relationships traversed so far, in order (becomes the
    /// `Value::List` rel binding on emit).
    rels: Vec<RelView>,
    /// `RelId`s already on this path; the RC-3 edge-uniqueness key.
    visited: HashSet<RelId>,
}

/// Resolve a [`LengthRange`] to `(min, max)` hop bounds. The bare `*`
/// form ([`LengthRange::Unbounded`]) is openCypher `*1..` (min 1, no
/// max). [`LengthRange::Quantified`] is unreachable here (rejected at
/// [`ExpandOp::new`]); GQL `{N,M}` and openCypher `*N..M` carry
/// identical bounds, so mapping it through is bounds-correct and avoids
/// a panic on a hypothetically mis-lowered plan.
fn var_length_bounds(lr: &LengthRange) -> (u32, Option<u32>) {
    match lr {
        LengthRange::Cypher { min, max } => (*min, *max),
        LengthRange::Unbounded => (1, None),
        LengthRange::Quantified { min, max } => (*min, *max),
    }
}

/// RC-1 unbounded-depth-cap error.
///
/// # Why reuse `ResourceExhausted` rather than a new `ExecutionError` variant
///
/// [`ExecutionError`] is deliberately exempt from `#[non_exhaustive]`
/// (it is the frozen M5↔M4 contract surface; see its type doc): adding
/// a variant would be a coordinated breaking change forcing synchronized
/// M5-07 / M5-11 / M5-13 renderer amendments. We therefore route the
/// var-length caps through the existing
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] surface —
/// exactly as [`spillover_fallback_err`] does — with the `feature`
/// string carrying the surface label. This keeps the slice file-disjoint
/// from the concurrent set-ops work (no `semantic/error.rs` edit) AND
/// honours code-quality policy. The contract's `VarLengthDepthCapExceeded`
/// naming (RC-1) is illustrative ("e.g."); the load-bearing requirement
/// is a STRUCTURED error, never a silent truncation.
fn var_length_depth_cap_err(cap: u32) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "var-length unbounded depth cap (*N..)".to_owned(),
        requested_bytes: 0,
        cap_bytes: cap as u64,
        projected_bytes: cap as u64 + 1,
        span: crate::error::Span::point(0, 0),
    })
}

/// RC-1 frontier-size-budget error (the b^d blowup guard). See
/// [`var_length_depth_cap_err`] for the `ResourceExhausted`-reuse
/// rationale.
fn var_length_frontier_cap_err(live: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "var-length frontier-size budget".to_owned(),
        requested_bytes: live as u64,
        cap_bytes: VARLENGTH_MAX_FRONTIER as u64,
        projected_bytes: live as u64,
        span: crate::error::Span::point(0, 0),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arcgraph_core::{LabelId, NodeId, PartitionId, RelId, TenantId};

    use super::*;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::{
        BoundEdge, BoundEdgeCursor, BoundNode, RankedHit, StubExecutorSubstrate,
        SubstrateAccessError,
    };
    use crate::executor::value::{NodeView, RelView};

    fn alice() -> NodeView {
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
    }
    fn bob() -> NodeView {
        NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
    }
    fn carol() -> NodeView {
        NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
    }

    fn fixture() -> StubExecutorSubstrate {
        StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, alice())
            .with_node(TenantId::DEFAULT, bob())
            .with_node(TenantId::DEFAULT, carol())
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
    fn expand_emits_one_row_per_neighbor() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap();
        let b = exp.next_batch(&ctx, &s).unwrap();
        // 3 nodes scanned; only Alice has 2 outbound KNOWS edges.
        assert_eq!(b.row_count(), 2);
        // Each row is [Alice (Node), Neighbor (Node)].
        for row in b.rows() {
            assert_eq!(row.len(), 2);
            let from = match &row[0] {
                Value::Node(n) => n.id,
                _ => panic!("expected Node"),
            };
            assert_eq!(from, NodeId::new(1));
        }
        let b2 = exp.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after exhausting upstream");
    }

    #[test]
    fn expand_with_rel_var_extends_schema() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            Some(BindingId::new(2)),
            BindingId::new(1),
            None,
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap();
        assert_eq!(
            exp.schema(),
            &[BindingId::new(0), BindingId::new(2), BindingId::new(1)]
        );
        let b = exp.next_batch(&ctx, &s).unwrap();
        // Each row: [from (Node), rel (Relationship), to (Node)]
        for row in b.rows() {
            assert_eq!(row.len(), 3);
            assert!(matches!(&row[0], Value::Node(_)));
            assert!(matches!(&row[1], Value::Relationship(_)));
            assert!(matches!(&row[2], Value::Node(_)));
        }
    }

    #[test]
    fn expand_with_to_label_filters_far_end_by_label() {
        // F2 (PE-1 §F2): the per-edge to-label filter keeps only edges
        // whose `dst` node carries the folded label. Alice(1, L1) knows
        // Bob(2, L1) AND Acme(4, L9); with `to_label = L1` only Bob
        // survives — matching the semi-join `(b:L1)` would produce.
        let s = fixture()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(4), Some(LabelId::new(9))),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(12),
                    NodeId::new(1),
                    NodeId::new(4),
                    Some(TypeId::new(1)),
                ),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        // Root at Alice only (unique label 1 is shared by Bob/Carol, so
        // scan Alice via a singleton to isolate her fan-out).
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap()
        .with_to_label(LabelId::new(1));
        assert_eq!(exp.to_label(), Some(LabelId::new(1)));
        let b = exp.next_batch(&ctx, &s).unwrap();
        // Alice → Bob(L1), Carol(L1) survive; Acme(L9) is dropped. Bob
        // and Carol have no outbound edges. → 2 rows, both L1 dst.
        assert_eq!(b.row_count(), 2, "Acme (L9) dropped by to_label = L1");
        for row in b.rows() {
            match row.last().unwrap() {
                Value::Node(n) => assert_eq!(n.label, Some(LabelId::new(1))),
                other => panic!("expected Node dst, got {other:?}"),
            }
        }
    }

    #[test]
    fn expand_without_to_label_keeps_all_far_ends() {
        // Same fixture, NO to-label fold — the mismatched-label neighbour
        // (Acme, L9) is retained (RED-on-revert companion of the filter
        // test above).
        let s = fixture()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(4), Some(LabelId::new(9))),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(12),
                    NodeId::new(1),
                    NodeId::new(4),
                    Some(TypeId::new(1)),
                ),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap();
        assert_eq!(exp.to_label(), None);
        let b = exp.next_batch(&ctx, &s).unwrap();
        // Alice → Bob, Carol, Acme (all kept). → 3 rows.
        assert_eq!(b.row_count(), 3, "no to_label: Acme (L9) retained");
    }

    // ===== ADR-186 / #650-C variable-length path execution =====
    //
    // These fixtures give node 1 a UNIQUE label (`START_LABEL`) so a
    // `ScanOp` on that label yields exactly the single start row; the
    // BFS then runs from node 1 only. Traversed/destination nodes carry
    // `LabelId 1`. The KNOWS rel-type is `TypeId 1`.

    const START_LABEL: u32 = 2;

    fn vl_node(id: u64, label: u32) -> NodeView {
        NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
    }
    fn knows(rel: u64, src: u64, dst: u64) -> RelView {
        RelView::new(
            RelId::new(rel),
            NodeId::new(src),
            NodeId::new(dst),
            Some(TypeId::new(1)),
        )
    }

    /// 1 →10→ 2 →20→ 3 →30→ 4 (KNOWS chain; node 1 = Start label).
    fn chain_fixture() -> StubExecutorSubstrate {
        StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, vl_node(1, START_LABEL))
            .with_node(TenantId::DEFAULT, vl_node(2, 1))
            .with_node(TenantId::DEFAULT, vl_node(3, 1))
            .with_node(TenantId::DEFAULT, vl_node(4, 1))
            .with_edge(TenantId::DEFAULT, knows(10, 1, 2))
            .with_edge(TenantId::DEFAULT, knows(20, 2, 3))
            .with_edge(TenantId::DEFAULT, knows(30, 3, 4))
    }

    /// Build + drive a var-length `ExpandOp` over `fixture`, scanning
    /// the Start label. Returns ALL output rows.
    fn run_var_length(
        fixture: &StubExecutorSubstrate,
        lr: LengthRange,
        with_rel_var: bool,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(START_LABEL)), Lsn::MAX);
        let rel_var = if with_rel_var {
            Some(BindingId::new(2))
        } else {
            None
        };
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            rel_var,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            Some(lr),
            Lsn::MAX,
        )?;
        let mut rows = Vec::new();
        loop {
            let b = exp.next_batch(&ctx, fixture)?;
            if b.is_empty() {
                break;
            }
            for r in b.into_rows() {
                rows.push(r);
            }
        }
        Ok(rows)
    }

    /// Sorted to-node ids (the last column of each row is the dst).
    fn to_ids(rows: &[Vec<Value>]) -> Vec<u64> {
        let mut v: Vec<u64> = rows
            .iter()
            .map(|r| match r.last().expect("non-empty row") {
                Value::Node(n) => n.id.raw(),
                other => panic!("to-col not a Node: {other:?}"),
            })
            .collect();
        v.sort_unstable();
        v
    }

    /// Rel-ids of a row's `Value::List` rel binding (column index 1).
    fn rel_ids(row: &[Value]) -> Vec<u64> {
        match &row[1] {
            Value::List(items) => items
                .iter()
                .map(|v| match v {
                    Value::Relationship(rel) => rel.id.raw(),
                    other => panic!("rel-list item not a Relationship: {other:?}"),
                })
                .collect(),
            other => panic!("rel col not a List (RC-2): {other:?}"),
        }
    }

    #[test]
    fn vl_accepts_cypher_var_length_at_construction() {
        // ADR-186: openCypher `*1..3` (Cypher) now CONSTRUCTS (the
        // M4-61 `NotImplemented` gate is lit for execution).
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let r = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            None,
            Direction::LeftToRight,
            Some(LengthRange::Cypher {
                min: 1,
                max: Some(3),
            }),
            Lsn::MAX,
        );
        assert!(r.is_ok(), "openCypher *1..3 must construct (ADR-186)");
    }

    #[test]
    fn vl_rejects_gql_quantified_at_construction() {
        // GQL `{N,M}` (Quantified) stays reserved to v1.1 — the
        // executor defensively rejects it.
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let r = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            None,
            Direction::LeftToRight,
            Some(LengthRange::Quantified {
                min: 1,
                max: Some(3),
            }),
            Lsn::MAX,
        );
        assert!(matches!(r, Err(ExecutionError::NotImplemented { .. })));
    }

    #[test]
    fn vl_bounded_1_2_enumerates_one_and_two_hop_paths() {
        // `(start)-[*1..2]->(b)` on 1→2→3→4 ⇒ to ∈ {2 (1-hop), 3 (2-hop)}.
        let rows = run_var_length(
            &chain_fixture(),
            LengthRange::Cypher {
                min: 1,
                max: Some(2),
            },
            false,
        )
        .unwrap();
        assert_eq!(to_ids(&rows), vec![2, 3]);
    }

    #[derive(Debug)]
    struct CursorCountingSubstrate {
        inner: StubExecutorSubstrate,
        cursor_calls: Arc<AtomicU64>,
    }

    impl CursorCountingSubstrate {
        fn new(inner: StubExecutorSubstrate) -> Self {
            Self {
                inner,
                cursor_calls: Arc::new(AtomicU64::new(0)),
            }
        }

        fn cursor_calls(&self) -> u64 {
            self.cursor_calls.load(Ordering::Relaxed)
        }
    }

    impl ExecutorSubstrate for CursorCountingSubstrate {
        fn scan_nodes(
            &self,
            tenant: TenantId,
            label: Option<LabelId>,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.scan_nodes(tenant, label, read_lsn)
        }

        fn expand(
            &self,
            tenant: TenantId,
            from: NodeId,
            rel_type: Option<TypeId>,
            direction: Direction,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
            self.inner
                .expand(tenant, from, rel_type, direction, read_lsn)
        }

        fn expand_cursor(
            &self,
            tenant: TenantId,
            from: NodeId,
            rel_type: Option<TypeId>,
            direction: Direction,
            read_lsn: Lsn,
        ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
            self.cursor_calls.fetch_add(1, Ordering::Relaxed);
            self.inner
                .expand_cursor(tenant, from, rel_type, direction, read_lsn)
        }

        fn vector_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_vec: &[f32],
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner
                .vector_search(tenant, property, query_vec, k, read_lsn)
        }

        fn bm25_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_text: &str,
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner
                .bm25_search(tenant, property, query_text, k, read_lsn)
        }

        fn community_members(
            &self,
            tenant: TenantId,
            community_id: i64,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.community_members(tenant, community_id, read_lsn)
        }
    }

    #[test]
    fn vl_k2_stays_on_materialized_expand_path() {
        let s = CursorCountingSubstrate::new(chain_fixture());
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(START_LABEL)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            Some(LengthRange::Cypher {
                min: 1,
                max: Some(2),
            }),
            Lsn::MAX,
        )
        .unwrap();
        let first = exp.next_batch(&ctx, &s).unwrap();
        assert_eq!(to_ids(&first.into_rows()), vec![2, 3]);
        assert_eq!(
            s.cursor_calls(),
            0,
            "k<=2 must keep the byte-identical materialized path"
        );
    }

    #[test]
    fn vl_rel_var_binds_to_relationship_list_in_order() {
        // RC-2 pinning: `MATCH (a)-[rs*1..2]->(b) RETURN rs` ⇒ rs is a
        // LIST of 1 or 2 relationships (not scalar, not flattened), in
        // traversal order.
        let rows = run_var_length(
            &chain_fixture(),
            LengthRange::Cypher {
                min: 1,
                max: Some(2),
            },
            true,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        for r in &rows {
            let n = rel_ids(r).len();
            assert!((1..=2).contains(&n), "rel-list length 1..2, got {n}");
        }
        // The 2-hop path's rel-list is [10, 20] in traversal order.
        let two_hop = rows.iter().find(|r| rel_ids(r).len() == 2).unwrap();
        assert_eq!(
            rel_ids(two_hop),
            vec![10, 20],
            "rel-list must be in traversal order"
        );
    }

    #[test]
    fn vl_zero_length_emits_start_node_with_empty_rel_list() {
        // `*0..2` includes the depth-0 identity path: to = start (1),
        // rel-list = []. RC-2 / Q3 (`*0` in-scope, never dropped).
        let rows = run_var_length(
            &chain_fixture(),
            LengthRange::Cypher {
                min: 0,
                max: Some(2),
            },
            true,
        )
        .unwrap();
        assert_eq!(to_ids(&rows), vec![1, 2, 3]);
        let zero = rows
            .iter()
            .find(|r| matches!(r.last().unwrap(), Value::Node(n) if n.id.raw() == 1))
            .expect("depth-0 identity row present");
        assert!(
            rel_ids(zero).is_empty(),
            "*0 identity path must bind an EMPTY rel-list"
        );
    }

    #[test]
    fn vl_cycle_terminates_with_finite_correct_paths() {
        // RC-3 cycle-fixture pin: 1→2→3→1 with `*1..` (unbounded)
        // returns finite correct paths and does NOT hang. Edge-
        // uniqueness caps the simple-edge walk at the cycle length;
        // node 1 IS re-visited (node repeats are allowed).
        let cyc = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, vl_node(1, START_LABEL))
            .with_node(TenantId::DEFAULT, vl_node(2, 1))
            .with_node(TenantId::DEFAULT, vl_node(3, 1))
            .with_edge(TenantId::DEFAULT, knows(10, 1, 2))
            .with_edge(TenantId::DEFAULT, knows(20, 2, 3))
            .with_edge(TenantId::DEFAULT, knows(30, 3, 1));
        let rows = run_var_length(&cyc, LengthRange::Unbounded, false).unwrap();
        // depths 1,2,3 ⇒ to = 2, 3, 1. Edge 10 re-use at depth 4 is
        // pruned ⇒ frontier empties ⇒ finite (no error, no hang).
        assert_eq!(to_ids(&rows), vec![1, 2, 3]);
    }

    #[test]
    fn vl_per_path_edge_uniqueness_keeps_both_converging_paths() {
        // RC-3 (the load-bearing per-path-vs-global distinction): two
        // paths CONVERGE at node 4 and SHARE the exit edge 30 (4→5).
        // A GLOBAL visited-set would prune the second path's use of
        // edge 30; the per-PATH set keeps BOTH.
        // Fixture: 1→2→4→5 and 1→3→4→5.
        let s = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, vl_node(1, START_LABEL))
            .with_node(TenantId::DEFAULT, vl_node(2, 1))
            .with_node(TenantId::DEFAULT, vl_node(3, 1))
            .with_node(TenantId::DEFAULT, vl_node(4, 1))
            .with_node(TenantId::DEFAULT, vl_node(5, 1))
            .with_edge(TenantId::DEFAULT, knows(10, 1, 2))
            .with_edge(TenantId::DEFAULT, knows(11, 1, 3))
            .with_edge(TenantId::DEFAULT, knows(20, 2, 4))
            .with_edge(TenantId::DEFAULT, knows(21, 3, 4))
            .with_edge(TenantId::DEFAULT, knows(30, 4, 5));
        let rows = run_var_length(
            &s,
            LengthRange::Cypher {
                min: 3,
                max: Some(3),
            },
            true,
        )
        .unwrap();
        assert_eq!(
            rows.len(),
            2,
            "both converging paths must survive (per-path RC-3)"
        );
        assert_eq!(to_ids(&rows), vec![5, 5]);
        // Both 3-hop paths share the exit edge 30; prefixes differ.
        let mut prefixes: Vec<Vec<u64>> = rows.iter().map(|r| rel_ids(r)).collect();
        prefixes.sort();
        assert_eq!(prefixes, vec![vec![10, 20, 30], vec![11, 21, 30]]);
    }

    #[test]
    fn vl_unbounded_depth_cap_errors_not_truncates() {
        // RC-1 honesty pin: an unbounded `*1..` whose paths exceed the
        // depth cap (5) ERRORS (structured `ResourceExhausted`) rather
        // than silently truncating. Chain of 6 edges (node 6 at depth 5
        // still has an outgoing edge to node 7).
        let mut s =
            StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, vl_node(1, START_LABEL));
        for i in 2..=7u64 {
            s = s.with_node(TenantId::DEFAULT, vl_node(i, 1));
        }
        for i in 1..=6u64 {
            s = s.with_edge(TenantId::DEFAULT, knows(i * 10, i, i + 1));
        }
        let r = run_var_length(&s, LengthRange::Unbounded, false);
        match r {
            Err(ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
                feature,
                ..
            })) => {
                assert!(
                    feature.contains("depth cap"),
                    "must be the depth-cap surface: {feature}"
                );
            }
            other => panic!("expected structured depth-cap error, got {other:?}"),
        }
    }

    #[test]
    fn vl_unbounded_terminates_when_within_cap() {
        // Counterpart: an unbounded `*1..` whose longest path is == cap
        // (5 hops, node 6 is a dead-end) returns COMPLETE results, no
        // error — a dead-end frontier AT the cap is not a truncation.
        let mut s =
            StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, vl_node(1, START_LABEL));
        for i in 2..=6u64 {
            s = s.with_node(TenantId::DEFAULT, vl_node(i, 1));
        }
        for i in 1..=5u64 {
            s = s.with_edge(TenantId::DEFAULT, knows(i * 10, i, i + 1));
        }
        let rows = run_var_length(&s, LengthRange::Unbounded, false).unwrap();
        assert_eq!(to_ids(&rows), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn vl_dense_fanout_frontier_cap_errors_not_truncates() {
        // RC-1 frontier-size-budget pin (the b^d OOM guard — the REAL
        // memory protection, DISTINCT from the depth cap). A DENSE
        // fan-out drives the live frontier past `VARLENGTH_MAX_FRONTIER`
        // (100_000) at depth 1 — BEFORE any depth limit could bite — and
        // MUST surface the structured frontier-budget error, never a
        // silently-truncated result.
        //
        // This is the second of the two RC-1 caps; the existing
        // `vl_unbounded_depth_cap_errors_not_truncates` (single chain,
        // frontier always size 1) only ever trips the DEPTH cap. Both
        // failure modes are now pinned (per
        // `feedback_load_bearing_pr_requires_fault_injection_tests`).
        //
        // Back-of-envelope: node 1 fans out to `FANOUT` distinct edges
        // (all → node 2; distinct `RelId`s so RC-3 per-path edge-
        // uniqueness keeps EVERY one alive at depth 1). With `*1..2`,
        // each surviving edge pushes one state to BOTH `results` and the
        // next `frontier`, so `next.len() + results.len()` reaches `2*i`
        // after `i` edges and crosses the 100_000 cap at `i = 50_001`.
        // `FANOUT = 60_000` guarantees the cap trips mid-frontier
        // (≈10_000 edges of margin) while exercising the REAL production
        // constant, not a test-substituted value. `*1..2` is BOUNDED, so
        // the depth-cap branch (`if unbounded`) is structurally
        // unreachable — any error here is UNAMBIGUOUSLY the frontier cap.
        const FANOUT: u64 = 60_000;
        let mut s = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, vl_node(1, START_LABEL))
            .with_node(TenantId::DEFAULT, vl_node(2, 1));
        for rel in 1..=FANOUT {
            s = s.with_edge(TenantId::DEFAULT, knows(rel, 1, 2));
        }
        let r = run_var_length(
            &s,
            LengthRange::Cypher {
                min: 1,
                max: Some(2),
            },
            false,
        );
        match r {
            Err(ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
                feature,
                ..
            })) => {
                assert!(
                    feature.contains("frontier"),
                    "must be the frontier-size-budget surface: {feature}"
                );
                assert!(
                    !feature.contains("depth cap"),
                    "frontier cap must be DISTINCT from the depth cap: {feature}"
                );
            }
            Ok(rows) => panic!(
                "frontier cap must ERROR, never truncate — got {} rows",
                rows.len()
            ),
            other => panic!("expected structured frontier-cap error, got {other:?}"),
        }
    }

    #[test]
    fn vl_k3_layered_fanout_streams_first_batch_without_frontier_cap() {
        // V11-S-03 Q1 flip test. Pre-S-03, this shape materializes all
        // 216_000 three-hop paths for the single start row and trips the
        // 100_000 var-length frontier-size budget before returning any
        // output. The streaming k>=3 path must return the first batch
        // without full-result materialization; consumers can stop after
        // a small prefix.
        const FANOUT: u64 = 60;
        let mut s =
            StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, vl_node(1, START_LABEL));
        for id in 10_000..10_000 + FANOUT {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }
        for id in 20_000..20_000 + FANOUT {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }
        for id in 30_000..30_000 + FANOUT {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }

        let mut rel_id = 1;
        for dst in 10_000..10_000 + FANOUT {
            s = s.with_edge(TenantId::DEFAULT, knows(rel_id, 1, dst));
            rel_id += 1;
        }
        for src in 10_000..10_000 + FANOUT {
            for dst in 20_000..20_000 + FANOUT {
                s = s.with_edge(TenantId::DEFAULT, knows(rel_id, src, dst));
                rel_id += 1;
            }
        }
        for src in 20_000..20_000 + FANOUT {
            for dst in 30_000..30_000 + FANOUT {
                s = s.with_edge(TenantId::DEFAULT, knows(rel_id, src, dst));
                rel_id += 1;
            }
        }

        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(START_LABEL)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            Some(LengthRange::Cypher {
                min: 3,
                max: Some(3),
            }),
            Lsn::MAX,
        )
        .unwrap();

        let first = exp.next_batch(&ctx, &s);
        match first {
            Ok(batch) => {
                assert_eq!(batch.row_count(), BATCH_ROWS);
                for row in batch.rows().iter().take(10) {
                    assert!(matches!(row.last(), Some(Value::Node(n)) if n.id.raw() >= 30_000));
                }
            }
            Err(ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
                feature,
                ..
            })) => panic!("k=3 streaming path must not trip old frontier cap: {feature}"),
            Err(other) => panic!("unexpected k=3 streaming error: {other:?}"),
        }
    }

    fn add_three_hop_pyramid(
        mut s: StubExecutorSubstrate,
        root: u64,
        layer_base: u64,
        rel_start: u64,
        fanout: u64,
    ) -> StubExecutorSubstrate {
        s = s.with_node(TenantId::DEFAULT, vl_node(root, START_LABEL));
        for id in layer_base..layer_base + fanout {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }
        for id in layer_base + 1_000..layer_base + 1_000 + fanout {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }
        for id in layer_base + 2_000..layer_base + 2_000 + fanout {
            s = s.with_node(TenantId::DEFAULT, vl_node(id, 1));
        }

        let mut rel_id = rel_start;
        for dst in layer_base..layer_base + fanout {
            s = s.with_edge(TenantId::DEFAULT, knows(rel_id, root, dst));
            rel_id += 1;
        }
        for src in layer_base..layer_base + fanout {
            for dst in layer_base + 1_000..layer_base + 1_000 + fanout {
                s = s.with_edge(TenantId::DEFAULT, knows(rel_id, src, dst));
                rel_id += 1;
            }
        }
        for src in layer_base + 1_000..layer_base + 1_000 + fanout {
            for dst in layer_base + 2_000..layer_base + 2_000 + fanout {
                s = s.with_edge(TenantId::DEFAULT, knows(rel_id, src, dst));
                rel_id += 1;
            }
        }
        s
    }

    #[test]
    fn vl_k3_batch_suspension_preserves_remaining_child_rows() {
        // R-1 / #814-class regression: the first start row emits more
        // than one output batch (15^3 = 3375 paths > BATCH_ROWS). When
        // that stream suspends mid-child-batch, the second start row
        // must remain owned by ExpandOp and emit its full 3375 paths.
        const FANOUT: u64 = 15;
        let s = add_three_hop_pyramid(StubExecutorSubstrate::new(), 1, 10_000, 1, FANOUT);
        let s = add_three_hop_pyramid(s, 2, 20_000, 10_000, FANOUT);
        let rows = run_var_length(
            &s,
            LengthRange::Cypher {
                min: 3,
                max: Some(3),
            },
            false,
        )
        .unwrap();
        assert_eq!(
            rows.len() as u64,
            2 * FANOUT * FANOUT * FANOUT,
            "both start rows' three-hop pyramids must fully emit"
        );
    }

    #[test]
    fn expand_propagates_cancel_at_batch_boundary() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            None,
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap();
        ctx.cancellation().cancel();
        let r = exp.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    // --------- W13α / M4-64b SIMD dst-allow-set pins ----------

    #[test]
    fn dst_allow_set_filters_neighbors_via_simd_helper() {
        // Pin: when dst_allow_set is configured, only edges whose
        // dst.id is in the allow-set survive. The SIMD helper is the
        // load-bearing routing point.
        let s = fixture(); // Alice → Bob (id=2), Alice → Carol (id=3)
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap()
        .with_dst_allow_set(vec![2_u64]); // only Bob
        assert!(exp.uses_simd_dst_allow_set());
        let b = exp.next_batch(&ctx, &s).unwrap();
        // Only Alice → Bob edge (Carol filtered out by allow-set).
        assert_eq!(b.row_count(), 1);
        let dst = match &b.row(0)[1] {
            Value::Node(n) => n.id,
            _ => panic!("expected dst Node"),
        };
        assert_eq!(dst, NodeId::new(2));
    }

    #[test]
    fn dst_allow_set_empty_short_circuits_to_full_passthrough() {
        // Pin: an empty allow-set is treated as "no filter" (the
        // operator's no-overhead path), NOT as "drop everything".
        // Justification: the SIMD helper's strict semantic returns
        // all-false for empty targets, but the operator-side guard
        // skips the helper entirely when the set is empty so the
        // pre-W13α behavior is preserved bit-for-bit.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap()
        .with_dst_allow_set(vec![]); // empty
        assert!(!exp.uses_simd_dst_allow_set());
        let b = exp.next_batch(&ctx, &s).unwrap();
        // Both edges pass (empty allow-set → no filter).
        assert_eq!(b.row_count(), 2);
    }

    #[test]
    fn dst_allow_set_default_is_no_filter_no_overhead() {
        // Pin: ExpandOp without with_dst_allow_set MUST behave
        // identically to its pre-W13α self (no SIMD path engaged).
        // The default constructor leaves dst_allow_set = None.
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let exp = ExpandOp::new(
            PhysicalOperator::Scan(scan),
            BindingId::new(0),
            None,
            BindingId::new(1),
            None,
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .unwrap();
        assert!(!exp.uses_simd_dst_allow_set());
    }
}
