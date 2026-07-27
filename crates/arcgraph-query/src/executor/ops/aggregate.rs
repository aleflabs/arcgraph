//! [`AggregateOp`] — GROUP BY + aggregation functions (M4-63).
//!
//! Lowers from [`crate::logical_plan::LogicalAggregate`]. Materializes
//! ALL upstream rows, partitions them by the group-by-key tuple, and
//! folds each partition's aggregations into a single output row.
//!
//! # Aggregate semantics (amendment-03 §TIER-2-b)
//!
//! Per openCypher 9 §6.4 + amendment-03 §TIER-2-b 3VL aggregate
//! semantics:
//!
//! - `COUNT(expr)` — counts rows where `expr` evaluates to non-NULL.
//! - `COUNT(*)` — counts input ROWS, including rows where every value is
//!   NULL (#773 G4; the `star` flag on [`AggregateCall`]). The materialize
//!   loop folds a non-NULL sentinel per row so the count never skips a
//!   row — distinct from `COUNT(expr)`, which excludes NULL `expr`.
//! - `SUM(expr)` — sums non-NULL numeric values; the result is NULL
//!   for an entirely-NULL or empty input.
//! - `AVG(expr)` — arithmetic mean of non-NULL numeric values; the
//!   denominator counts only non-NULL entries.
//! - `MIN(expr)` / `MAX(expr)` — ignore NULL operands.
//! - `COLLECT(expr)` — accumulates non-NULL values into a list (NULLs
//!   are dropped per Cypher 9 §3.2.7).
//! - `<agg>(DISTINCT expr)` — deduplicates the per-group non-NULL values
//!   before the fold (#773 G5; the `distinct` flag on [`AggregateCall`]).
//!   `count(DISTINCT x)` counts distinct non-NULL `x`; `collect(DISTINCT
//!   x)` collects the distinct non-NULL `x` (in first-seen order).
//!
//! # Why a blocking operator
//!
//! `AggregateOp` materializes the entire upstream batch stream before
//! emitting its first output row — it CANNOT yield until it has seen
//! every group's last row. For tenants with a configured memory budget
//! ([`crate::executor::MemoryBudget`]), the COLLECT fold + each emitted
//! group row are debited at insertion time and released when the
//! group's row is emitted (surfacing
//! [`crate::semantic::error::ArcQLError::ResourceExhausted`] on
//! overflow). For unbudgeted tenants (uncapped budget = no memory
//! limit), the group hash table grows with the actual distinct-group
//! cardinality with NO fixed row-count ceiling on the group count
//! (#1008: a stale doc here previously claimed a 131 072-group cap that
//! the drain loop never enforced — a `MATCH … RETURN g, count(*)` over
//! more than 200 K groups failed only because its UPSTREAM scan / expand
//! / join hit the old
//! [`crate::executor::ops::expand::SPILLOVER_MAX_ROWS`] valve, which #980
//! lifted to [`crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS`]).
//!
//! # Forward-pin
//!
//! Streaming pre-sorted aggregations (where the input is already sorted
//! by the group key) is M4-72+ scope. The `AggregateOp` here is the
//! "blocking hash aggregate" shape; the cost-walker can hoist a Sort →
//! Aggregate to a Sort → MergeAggregate at M4-72.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL aggregate semantics
//!   (NULL exclusion for COUNT/SUM/AVG; COUNT(*) v1.0 surrogate).
//! - **ADR-038 §2 D-28** — aggregation operator contract.
//! - **openCypher 9 §6.4** — aggregate function semantics.
//! - **`feedback_seqlock_panic_safety_primitive.md`** — the budget
//!   tracker is `Mutex`-backed (NOT SeqLock); the per-tenant fault-
//!   isolation discipline that file describes does NOT apply here.

use std::collections::{HashMap, HashSet};

use arcgraph_core::TenantId;

use crate::executor::batch::Batch;
use crate::executor::budget::{MemoryBudget, estimate_row_bytes, estimate_value_bytes};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::AggregationKind;
use crate::semantic::bound_ast::{
    BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind,
};

/// One aggregation function call to evaluate within a group.
#[derive(Debug, Clone)]
pub struct AggregateCall {
    /// Aggregation kind (count/sum/avg/min/max/collect).
    pub kind: AggregationKind,
    /// The argument expression evaluated against each group row. For a
    /// `star` call (`count(*)`) this is a placeholder that is NEVER
    /// evaluated (the materialize loop folds a non-NULL sentinel).
    pub arg: BoundExpression,
    /// Output binding-id (#746) — the [`BindingId`] this aggregation's
    /// result column is emitted under, sourced from the lowering's
    /// [`crate::logical_plan::AggregationSpec::output_id`]. The
    /// layered-over `ProjectOp` references this id (the lowering
    /// rewrote `count(n)` → `VariableRef(output_id)`), so it MUST match
    /// what that Project resolves — closing the binder↔executor
    /// mismatch that blocked end-to-end aggregate execution.
    pub output_id: BindingId,
    /// `count(DISTINCT x)` / `collect(DISTINCT x)` (#773 G5) — deduplicate
    /// the per-group non-NULL values before folding. Applies to ANY kind
    /// (openCypher v9 §3 admits `sum(DISTINCT …)` etc.; dedup before the
    /// fold is correct for all — `min`/`max` are dedup-invariant, the
    /// rest change). Mutually exclusive with `star`.
    pub distinct: bool,
    /// `count(*)` (#773 G4) — count input ROWS rather than non-NULL `arg`
    /// values. The materialize loop folds a non-NULL sentinel per row, so
    /// the count includes rows whose properties are all NULL (unlike
    /// `count(expr)`, which excludes NULL `expr`). Only ever set with
    /// `kind == Count` (the sole star aggregate per openCypher v9 §3).
    pub star: bool,
}

/// Per-group accumulator state.
#[derive(Debug, Clone)]
struct GroupState {
    /// One accumulator slot per [`AggregateCall`].
    accumulators: Vec<Accumulator>,
    /// The group-by-key cells (cached so we can emit them as the
    /// leading columns of the output row).
    group_key_cells: Vec<Value>,
    /// #773 G5 — per-aggregation DISTINCT dedup set. `Some(set)` for a
    /// `distinct` call (the set holds the [`canonical_row_key`] rendering
    /// of each already-seen non-NULL value, so a duplicate is skipped
    /// before its fold); `None` for a non-distinct call (no dedup). One
    /// slot per [`AggregateCall`], index-aligned with `accumulators`.
    distinct_seen: Vec<Option<HashSet<String>>>,
}

/// Per-aggregation accumulator.
///
/// Stores the running state needed to fold one more row into the
/// aggregation. Per amendment-03 §TIER-2-b NULLs are excluded from
/// COUNT/SUM/AVG/MIN/MAX/COLLECT.
///
/// `pub(crate)` (with `empty`/`fold`/`finalize`/`merge`) so the S5
/// morsel-driven parallel aggregate ([`super::parallel_aggregate`],
/// ADR-226 §4 CONC-D) reuses the EXACT serial fold + a matching merge —
/// the parallel partial aggregate ≡ serial by construction, since both
/// paths fold through this one accumulator. See [`Self::merge`] for the
/// mergeable-decomposition per kind (esp. AVG = carry `{sum, n}`).
#[derive(Debug, Clone)]
pub(crate) enum Accumulator {
    /// COUNT(expr): increment the counter for every non-NULL `expr`.
    Count(u64),
    /// SUM(expr): running total. `None` while the input is entirely
    /// NULL / empty (Cypher 9 §6.4 returns NULL on an empty input);
    /// the typed inner is integer or float to honor Cypher's numeric
    /// promotion rules.
    Sum(Option<NumericRunning>),
    /// AVG(expr): running total + count for non-NULL operands.
    Avg { sum: Option<NumericRunning>, n: u64 },
    /// MIN(expr): running minimum across non-NULL operands.
    Min(Option<Value>),
    /// MAX(expr): running maximum across non-NULL operands.
    Max(Option<Value>),
    /// COLLECT(expr): accumulator list (NULLs dropped per Cypher 9
    /// §3.2.7).
    Collect(Vec<Value>),
}

/// `pub(crate)` to match [`Accumulator`]'s visibility (it appears in the
/// `pub(crate)` `Sum` / `Avg` variants). NOT part of the public API — the
/// running numeric total stays an internal detail of the aggregate ops.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NumericRunning {
    Int(i64),
    Float(f64),
}

impl Accumulator {
    /// A fresh zero-state accumulator for `kind`. `pub(crate)` for S5
    /// parallel reuse (each morsel starts from `empty`).
    pub(crate) fn empty(kind: AggregationKind) -> Self {
        match kind {
            AggregationKind::Count => Self::Count(0),
            AggregationKind::Sum => Self::Sum(None),
            AggregationKind::Avg => Self::Avg { sum: None, n: 0 },
            AggregationKind::Min => Self::Min(None),
            AggregationKind::Max => Self::Max(None),
            AggregationKind::Collect => Self::Collect(Vec::new()),
        }
    }

    /// Fold one more `value` into the accumulator. NULL operands are
    /// excluded per amendment-03 §TIER-2-b. `pub(crate)` for S5 parallel
    /// reuse (each morsel folds its rows through this EXACT logic).
    pub(crate) fn fold(&mut self, value: Value) -> Result<(), ExecutionError> {
        // amendment-03 §TIER-2-b: NULL exclusion across all aggregates
        // except COLLECT (which also drops NULL per Cypher 9 §3.2.7).
        if matches!(value, Value::Null) {
            return Ok(());
        }
        match self {
            Self::Count(n) => {
                *n = n.saturating_add(1);
            }
            Self::Sum(running) => {
                *running = Some(numeric_add(*running, &value)?);
            }
            Self::Avg { sum, n } => {
                *sum = Some(numeric_add(*sum, &value)?);
                *n = n.saturating_add(1);
            }
            Self::Min(running) => match running {
                None => *running = Some(value),
                Some(curr) => {
                    if compare_values(&value, curr).is_lt() {
                        *curr = value;
                    }
                }
            },
            Self::Max(running) => match running {
                None => *running = Some(value),
                Some(curr) => {
                    if compare_values(&value, curr).is_gt() {
                        *curr = value;
                    }
                }
            },
            Self::Collect(list) => {
                list.push(value);
            }
        }
        Ok(())
    }

    /// Materialize the accumulator's final [`Value`]. `pub(crate)` for
    /// S5 parallel reuse (the merged accumulator finalizes through this
    /// EXACT logic, so parallel finalization ≡ serial).
    pub(crate) fn finalize(self) -> Value {
        match self {
            Self::Count(n) => Value::Integer(n as i64),
            Self::Sum(None) => Value::Null,
            Self::Sum(Some(n)) => match n {
                NumericRunning::Int(i) => Value::Integer(i),
                NumericRunning::Float(f) => Value::Float(f),
            },
            Self::Avg { sum: None, .. } | Self::Avg { n: 0, .. } => Value::Null,
            Self::Avg { sum: Some(s), n } => {
                let total: f64 = match s {
                    NumericRunning::Int(i) => i as f64,
                    NumericRunning::Float(f) => f,
                };
                Value::Float(total / n as f64)
            }
            Self::Min(v) => v.unwrap_or(Value::Null),
            Self::Max(v) => v.unwrap_or(Value::Null),
            Self::Collect(list) => Value::List(list),
        }
    }

    /// Merge a partial accumulator `other` (produced by folding one
    /// morsel) INTO `self` (the running merge), for the S5 parallel
    /// partial-aggregate (ADR-226 §4 CONC-D). Both `self` and `other`
    /// MUST hold the same variant (built from the same
    /// [`AggregationKind`]); the merge is the mergeable-decomposition of
    /// each aggregate:
    ///
    /// - **COUNT** — partial counts ADD (`Σ` of per-morsel row/non-NULL
    ///   counts = the total). Uses the same `saturating_add` as
    ///   [`Self::fold`].
    /// - **SUM** — partial sums ADD through [`numeric_add`] (the EXACT
    ///   numeric-promotion + overflow-checked path a serial fold uses).
    ///   A `None` partial (all-NULL / empty morsel) contributes nothing;
    ///   the merged SUM is `None` (→ NULL) IFF every morsel was `None`.
    /// - **AVG** — **carry `(sum, n)` partials, NOT partial means**:
    ///   merged `sum = Σ sumᵢ` (via [`numeric_add`]) and `n = Σ nᵢ`, so
    ///   the final [`Self::finalize`] computes `Σsum / Σn` — identical to
    ///   the serial single-pass mean. Averaging per-morsel means would be
    ///   WRONG (unequal morsel sizes). This is the classic mergeable AVG.
    /// - **MIN / MAX** — merged extreme = extreme of the partials, via
    ///   the SAME [`compare_values`] oracle a serial fold uses; a `None`
    ///   partial (all-NULL morsel) is skipped.
    /// - **COLLECT** — appends `other`'s list after `self`'s. Present for
    ///   completeness; S5 does NOT route COLLECT through the parallel
    ///   path at rc (order/dup + budget semantics) — it falls back to the
    ///   serial op (see [`super::parallel_aggregate`]).
    ///
    /// Equivalence: because MIN/MAX/COUNT/SUM are associative +
    /// commutative under these merges and AVG carries the associative
    /// `(sum, n)` pair, folding a partition then merging the partials
    /// yields the SAME accumulator state as a single serial fold over the
    /// concatenation (the S5 equivalence proptest pins this on random
    /// data incl. NULL / empty / all-NULL).
    pub(crate) fn merge(&mut self, other: Self) -> Result<(), ExecutionError> {
        match (self, other) {
            (Self::Count(a), Self::Count(b)) => {
                *a = a.saturating_add(b);
            }
            (Self::Sum(a), Self::Sum(b)) => {
                if let Some(NumericRunning::Int(_) | NumericRunning::Float(_)) = b {
                    let bv = running_to_value(b.expect("matched Some above"));
                    *a = Some(numeric_add(*a, &bv)?);
                }
            }
            (Self::Avg { sum: sa, n: na }, Self::Avg { sum: sb, n: nb }) => {
                if let Some(inner) = sb {
                    let bv = running_to_value(inner);
                    *sa = Some(numeric_add(*sa, &bv)?);
                }
                *na = na.saturating_add(nb);
            }
            (Self::Min(a), Self::Min(b)) => {
                if let Some(bv) = b {
                    match a {
                        None => *a = Some(bv),
                        Some(curr) => {
                            if compare_values(&bv, curr).is_lt() {
                                *curr = bv;
                            }
                        }
                    }
                }
            }
            (Self::Max(a), Self::Max(b)) => {
                if let Some(bv) = b {
                    match a {
                        None => *a = Some(bv),
                        Some(curr) => {
                            if compare_values(&bv, curr).is_gt() {
                                *curr = bv;
                            }
                        }
                    }
                }
            }
            (Self::Collect(a), Self::Collect(b)) => {
                a.extend(b);
            }
            // Mismatched variants are an internal invariant violation
            // (the parallel op builds every morsel's accumulators from
            // the SAME kinds), surfaced loudly rather than mis-merging.
            _ => {
                return Err(ExecutionError::Eval(
                    "AggregateOp: merge of mismatched accumulator variants".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Re-materialize a [`NumericRunning`] as a [`Value`] so a partial SUM /
/// AVG sum can be re-added through [`numeric_add`] (reusing the serial
/// numeric-promotion + overflow-checked path — no duplicate arithmetic).
fn running_to_value(r: NumericRunning) -> Value {
    match r {
        NumericRunning::Int(i) => Value::Integer(i),
        NumericRunning::Float(f) => Value::Float(f),
    }
}

/// Add a (possibly typed) numeric `value` to a running total. Returns
/// the updated running, propagating numeric type per Cypher 9 §3.4
/// (Int + Int = Int; any Float promotes to Float).
fn numeric_add(
    running: Option<NumericRunning>,
    value: &Value,
) -> Result<NumericRunning, ExecutionError> {
    let added = match value {
        Value::Integer(n) => NumericRunning::Int(*n),
        Value::Float(f) => NumericRunning::Float(*f),
        _ => {
            return Err(ExecutionError::Eval(
                "AggregateOp: SUM/AVG arg must be numeric".into(),
            ));
        }
    };
    match (running, added) {
        (None, x) => Ok(x),
        (Some(NumericRunning::Int(a)), NumericRunning::Int(b)) => {
            // checked_add to surface overflow loudly under the code-quality policy
            // (no silent wraparound in production code).
            let r = a
                .checked_add(b)
                .ok_or_else(|| ExecutionError::Eval("AggregateOp: SUM integer overflow".into()))?;
            Ok(NumericRunning::Int(r))
        }
        (Some(NumericRunning::Int(a)), NumericRunning::Float(b)) => {
            Ok(NumericRunning::Float(a as f64 + b))
        }
        (Some(NumericRunning::Float(a)), NumericRunning::Int(b)) => {
            Ok(NumericRunning::Float(a + b as f64))
        }
        (Some(NumericRunning::Float(a)), NumericRunning::Float(b)) => {
            Ok(NumericRunning::Float(a + b))
        }
    }
}

/// Cypher MIN/MAX comparison across [`Value`] variants. Returns
/// `Ordering::Equal` for incomparable scalar mixes — the running
/// accumulator keeps its current value (Cypher 9 §3.4 leaves
/// heterogeneous-MIN implementation-defined; the executor's choice is
/// "first wins" via the `Equal` branch never replacing the running).
///
/// ADR-193 D-11 — paths ARE orderable (openCypher orderability is a
/// total order that never errors). A path sorts FIRST in the global
/// type-order, so `min(p)`/`max(p)` return the orderability-extreme path
/// (they do NOT error); two paths order by node-id then rel-id sequence
/// ([`PathView::cmp_paths`]), deterministically, never colliding.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Equal),
        (Value::Integer(x), Value::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Equal),
        (Value::Float(x), Value::Integer(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Equal),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        // ADR-193 D-11 — paths sort FIRST (global type-order); two paths
        // order deterministically by node-id then rel-id sequence. Placed
        // BEFORE the Map arm so a (Path, Map) pair resolves Path-first.
        (Value::Path(x), Value::Path(y)) => x.cmp_paths(y),
        (Value::Path(_), _) => Less,
        (_, Value::Path(_)) => Greater,
        // ADR-191 D-5 — a map operand in MIN/MAX routes through the
        // openCypher orderability total order (same oracle as `sort`),
        // so MIN/MAX over a map column is deterministic and never
        // collapses distinct maps.
        (Value::Map(_), _) | (_, Value::Map(_)) => {
            crate::executor::value::compare_orderability(a, b)
        }
        // Lists share the ORDER BY total order: element-wise comparison,
        // heterogeneous element ranking, then prefix length.
        (Value::List(_), _) | (_, Value::List(_)) => {
            crate::executor::value::compare_orderability(a, b)
        }
        _ => Equal,
    }
}

/// GROUP BY + aggregate operator.
pub struct AggregateOp {
    child: Box<PhysicalOperator>,
    /// Group-by projection items — evaluated against each upstream row
    /// to derive the group-key tuple.
    group_by: Vec<BoundProjectionItem>,
    /// Aggregate function calls evaluated per group.
    aggregations: Vec<AggregateCall>,
    /// Per-query parameter bag (forwarded to expression evaluation).
    parameters: Parameters,
    /// Accumulated per-group state; key is the canonical group-key
    /// rendering. v1.0-alpha uses the canonical Display rendering for
    /// keying — sufficient for `Value::Integer/Float/String/Boolean/Null`.
    /// Heterogeneous lists / nodes / relationships fall back to a
    /// debug-rendered key that's stable within a single query but
    /// not portable across queries.
    groups: HashMap<String, GroupState>,
    /// Insertion-order tracking for deterministic output across runs:
    /// HashMap iteration order is randomized, so we record the order
    /// in which groups were first seen and emit in that order.
    group_order: Vec<String>,
    /// Output schema. Leading slots = group-by-fresh-bindings;
    /// trailing slots = aggregate-fresh-bindings.
    schema: Vec<BindingId>,
    /// Cached child schema for upstream-row evaluation.
    child_schema: Vec<BindingId>,
    /// Whether we've drained all upstream rows.
    upstream_drained: bool,
    /// Whether we've emitted output.
    emitted: bool,
    /// Cached output rows after `materialize()` runs once.
    output_rows: Option<Vec<Vec<Value>>>,
    /// Output cursor.
    cursor: usize,
    /// W12α fix-up MED-1 (PR #277 retro): total bytes reserved against
    /// the per-tenant memory budget by this operator. Released in
    /// [`Drop`] to prevent the long-running-tenant counter-drift class
    /// (a sequence of N successful aggregate queries left N output-row
    /// reservations in the tenant counter, eventually saturating the
    /// cap with false `ResourceExhausted` rejections).
    reserved_total: u64,
    /// Tenant captured on the first reservation. `None` until then;
    /// used by [`Drop`] to release [`Self::reserved_total`] against
    /// the right tenant slot.
    tenant_for_release: Option<TenantId>,
    /// Budget snapshot captured on the first reservation. `Arc`-shared
    /// with the [`ExecutionContext`] so the operator can release on
    /// drop without holding an `&ExecutionContext` borrow.
    budget_for_release: Option<MemoryBudget>,
}

impl std::fmt::Debug for AggregateOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AggregateOp")
            .field("child", &self.child)
            .field("group_by_count", &self.group_by.len())
            .field("aggregations_count", &self.aggregations.len())
            .field("schema", &self.schema)
            .field("groups_seen", &self.groups.len())
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl AggregateOp {
    /// Construct an [`AggregateOp`].
    ///
    /// # Output schema (#746 binder↔executor contract)
    ///
    /// `schema = [group-by output ids…] ++ [aggregation output ids…]`,
    /// matching the row layout `materialize` emits (group-key cells
    /// then accumulator finalizations). Each column carries the
    /// BINDER-ASSIGNED output id (group-by items' `output_id`,
    /// aggregations' `output_id`) so the layered-over [`super::ProjectOp`]
    /// — whose items the lowering rewrote to `VariableRef(output_id)` —
    /// resolves every column. The legacy synthetic
    /// `0xFFFF_FFFF_8000_0000` base survives ONLY as a fallback for a
    /// group-by `Wildcard` (which carries no single output id); the real
    /// lowering path always supplies ids.
    pub fn new(
        child: PhysicalOperator,
        group_by: Vec<BoundProjectionItem>,
        aggregations: Vec<AggregateCall>,
    ) -> Self {
        let child_schema = child.schema().to_vec();
        // Schema = group_by columns (in declared order) + aggregation
        // columns (in declared order). #746: use the binder-assigned
        // output ids; a Wildcard group item (no output id) gets a
        // synthetic high-half fallback.
        let mut schema: Vec<BindingId> = Vec::with_capacity(group_by.len() + aggregations.len());
        let mut next_fallback: u64 = 0xFFFF_FFFF_8000_0000;
        for item in &group_by {
            let id = item.output_id.unwrap_or_else(|| {
                let synthetic = BindingId::new(next_fallback);
                next_fallback += 1;
                synthetic
            });
            schema.push(id);
        }
        for call in &aggregations {
            schema.push(call.output_id);
        }
        Self {
            child: Box::new(child),
            group_by,
            aggregations,
            parameters: Parameters::new(),
            groups: HashMap::new(),
            group_order: Vec::new(),
            schema,
            child_schema,
            upstream_drained: false,
            emitted: false,
            output_rows: None,
            cursor: 0,
            reserved_total: 0,
            tenant_for_release: None,
            budget_for_release: None,
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

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// On first call, drains all upstream batches into the group-by
    /// hashmap, then emits one output row per distinct group. On
    /// subsequent calls returns paginated batches of the cached
    /// outputs.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if !self.upstream_drained {
            self.materialize(ctx, substrate)?;
        }
        let rows = self.output_rows.as_ref().expect("materialized above");
        if self.cursor >= rows.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut out = Batch::with_capacity(self.schema.len());
        let take = (rows.len() - self.cursor).min(crate::executor::BATCH_ROWS);
        for row in &rows[self.cursor..self.cursor + take] {
            if !out.push_row(row.clone()) {
                return Err(ExecutionError::Eval(
                    "AggregateOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        self.emitted = true;
        Ok(out)
    }

    /// Drain all upstream batches, fold per-group, and stash the final
    /// output rows into [`Self::output_rows`].
    fn materialize<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let lookup_schema = self.child_schema.clone();
        let lookup = move |b: BindingId| schema_index(&lookup_schema, b);
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());
        // Drain upstream.
        loop {
            ctx.cancellation().check()?;
            let batch = self.child.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            for row in batch.into_rows() {
                // Compute the group-key tuple by evaluating each
                // group_by expression against the row.
                let mut key_cells: Vec<Value> = Vec::with_capacity(self.group_by.len());
                for item in &self.group_by {
                    let cell = match &item.kind {
                        BoundProjectionKind::Wildcard { .. } => {
                            // Wildcard at the group-by site is unusual
                            // but admissible — fold the entire row into
                            // a list value as the key. Cell order within
                            // the key is irrelevant to grouping (a key is
                            // compared whole + consistently across rows),
                            // so the verbatim row order is fine here; the
                            // alphabetical `RETURN *` reorder happens in
                            // the sibling `Project` wildcard item.
                            Value::List(row.clone())
                        }
                        BoundProjectionKind::Expr(e) => {
                            evaluate(e, &row, &lookup, &self.parameters)?
                        }
                    };
                    key_cells.push(cell);
                }
                let key = crate::executor::ops::canonical_row_key(&key_cells);
                // W12α fix-up LOW-3 (PR #277 retro): pre-evaluate each
                // aggregation arg + reserve per-fold COLLECT bytes
                // BEFORE the entry borrow so the unbounded
                // `Accumulator::Collect(Vec<Value>)` growth surfaces
                // `ResourceExhausted` mid-fold (not OOM-then-emit).
                // Other accumulators (Count/Sum/Avg/Min/Max) hold O(1)
                // intermediate state per group so don't need per-fold
                // tracking. Indexed iteration (not `for call in &...`)
                // so the immutable borrow of `self.aggregations` ends
                // at each `evaluate(...)?` and `record_reservation` can
                // borrow `self` mutably mid-loop.
                let agg_count = self.aggregations.len();
                let mut arg_values: Vec<Value> = Vec::with_capacity(agg_count);
                for i in 0..agg_count {
                    let call_kind = self.aggregations[i].kind;
                    // #773 G4 — `count(*)` (star): count EVERY row, so fold
                    // a non-NULL sentinel rather than evaluating the
                    // (placeholder) arg. Because the sentinel is never NULL,
                    // the Count accumulator increments for every row —
                    // including rows whose properties are all NULL — which
                    // is precisely count(*)'s ROW semantics (vs count(expr)
                    // which excludes NULL expr).
                    let v = if self.aggregations[i].star {
                        Value::Integer(1)
                    } else {
                        evaluate(&self.aggregations[i].arg, &row, &lookup, &self.parameters)?
                    };
                    // COLLECT budget reservation (W12α LOW-3). For
                    // collect(DISTINCT …) this reserves duplicates too
                    // (the dedup happens at fold time, below, where the
                    // per-group seen-set lives — it is not available here,
                    // before the entry borrow); the over-reservation is
                    // conservative (never under-counts → never OOMs) and is
                    // released on Drop.
                    if has_cap
                        && matches!(call_kind, AggregationKind::Collect)
                        && !matches!(v, Value::Null)
                    {
                        let bytes = estimate_value_bytes(&v) as u64;
                        budget.try_reserve_unscoped(
                            ctx.tenant(),
                            bytes,
                            "AggregateOp COLLECT fold",
                        )?;
                        self.record_reservation(ctx, &budget, bytes);
                    }
                    arg_values.push(v);
                }
                // #1008 — O(N) group accumulation. The insertion-order
                // push into `group_order` is gated on the HashMap's own O(1)
                // first-insert vacancy (the `Vacant` arm runs EXACTLY once
                // per distinct key, on first sight, preserving first-seen
                // order), NOT on a per-row O(G) `group_order.contains(&key)`
                // linear scan. At high cardinality (G ≈ N) the old
                // `.contains()` made GROUP BY O(N²) — per #1008's
                // reproduce-first, per-row cost grew 14.7→33.3→76.7 µs/row
                // as N doubled 25K→50K→100K, hitting the 30s query timeout
                // at ~200K distinct groups. The Entry match folds the
                // vacancy check into the single hash probe (no double-probe)
                // and is semantically identical: same groups, same
                // first-seen output order, same accumulator results.
                let entry = match self.groups.entry(key.clone()) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        // First sight of this key — record insertion order.
                        self.group_order.push(key);
                        slot.insert(GroupState {
                            accumulators: self
                                .aggregations
                                .iter()
                                .map(|a| Accumulator::empty(a.kind))
                                .collect(),
                            group_key_cells: key_cells.clone(),
                            // #773 G5 — one dedup set per DISTINCT call;
                            // `None` for non-distinct calls (no per-value
                            // tracking).
                            distinct_seen: self
                                .aggregations
                                .iter()
                                .map(|a| a.distinct.then(HashSet::new))
                                .collect(),
                        })
                    }
                    std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
                };
                for (i, v) in arg_values.into_iter().enumerate() {
                    // #773 G5 — DISTINCT dedup. A non-NULL value already
                    // seen in this group is skipped before its fold; NULL
                    // is excluded by `Accumulator::fold` regardless, so it
                    // never enters the seen-set. `min`/`max` are
                    // dedup-invariant; count/collect/sum/avg observe the
                    // dedup. (`self.aggregations[i]` and `entry` borrow
                    // DISJOINT fields of `self`, so both borrows coexist.)
                    if self.aggregations[i].distinct {
                        if matches!(v, Value::Null) {
                            continue;
                        }
                        let vkey =
                            crate::executor::ops::canonical_row_key(std::slice::from_ref(&v));
                        let seen = entry.distinct_seen[i]
                            .as_mut()
                            .expect("distinct call has a seen-set");
                        if !seen.insert(vkey) {
                            continue; // duplicate — exclude from the fold.
                        }
                    }
                    entry.accumulators[i].fold(v)?;
                }
            }
        }
        self.upstream_drained = true;
        // amendment-03 §TIER-2-b empty-input rule: a single-row
        // aggregate with no input rows still emits ONE row (count=0,
        // sum=NULL, etc.) iff there are no group_by columns. With
        // group_by, an empty input emits zero rows.
        if self.groups.is_empty() && self.group_by.is_empty() {
            let single_acc: Vec<Accumulator> = self
                .aggregations
                .iter()
                .map(|a| Accumulator::empty(a.kind))
                .collect();
            let single_row: Vec<Value> =
                single_acc.into_iter().map(Accumulator::finalize).collect();
            self.output_rows = Some(vec![single_row]);
            return Ok(());
        }
        // Emit one row per group, in INSERTION order, with optional
        // budget tracking. Per-row reservations recorded against
        // `reserved_total` so [`Drop`] releases them when the operator
        // is dropped (per W12α fix-up MED-1). `std::mem::take` of
        // `group_order` avoids the immutable borrow that would block
        // `record_reservation`'s `&mut self`.
        let group_order = std::mem::take(&mut self.group_order);
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(group_order.len());
        for key in &group_order {
            let group = self
                .groups
                .remove(key)
                .expect("group_order is a subset of groups keys");
            let mut row: Vec<Value> = Vec::with_capacity(self.schema.len());
            for cell in group.group_key_cells {
                row.push(cell);
            }
            for acc in group.accumulators {
                row.push(acc.finalize());
            }
            // Per-row budget reservation when a per-tenant cap is set
            // (the materialized result holds in memory until the
            // upstream pipe drains it).
            if has_cap {
                let bytes = estimate_row_bytes(&row) as u64;
                budget.try_reserve_unscoped(ctx.tenant(), bytes, "AggregateOp output")?;
                self.record_reservation(ctx, &budget, bytes);
            }
            rows.push(row);
        }
        self.output_rows = Some(rows);
        Ok(())
    }
}

impl Drop for AggregateOp {
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

// The group-by key canonicalization was hoisted to
// [`crate::executor::ops::canonical_row_key`] (ADR-185, #649-A1) so
// GROUP BY (here) and `RETURN DISTINCT` / `UNION` ([`DistinctOp`])
// share ONE value-equality oracle. The former private
// `canonical_group_key` was a byte-identical precursor; the shared
// helper preserves its encoding exactly (verified by the GROUP BY
// tests below + the DistinctOp tests).

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::error::Span;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn make_n_persons(n: u64) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=n {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Integer((i % 5) as i64 * 10))
                    .with_property("city", Value::String(format!("city{}", i % 3))),
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

    fn var_ref(b: BindingId) -> BoundExpression {
        BoundExpression::VariableRef {
            name: "n".into(),
            binding_id: b,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn prop_access(base: BoundExpression, name: &str) -> BoundExpression {
        BoundExpression::PropertyAccess {
            base: Box::new(base),
            path: vec![crate::semantic::bound_ast::BoundPropertyRef {
                name: name.into(),
                property_id: None,
                span: Span::point(1, 1),
            }],
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    // -------------------------------------------------------------
    // 6 aggregate-kind unit tests (COUNT/SUM/AVG/MIN/MAX/COLLECT)
    // -------------------------------------------------------------

    #[test]
    fn aggregate_count_excludes_null_per_amendment_03_tier_2_b() {
        // amendment-03 §TIER-2-b: COUNT(expr) excludes NULL.
        // Build a substrate where some rows have NULL `age`.
        let mut s = StubExecutorSubstrate::new();
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(30)),
        );
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))).with_property("age", Value::Null),
        );
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(40)),
        );
        let scan = person_scan();
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(scan),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        // 2 non-NULL ages out of 3 rows.
        assert_eq!(b.row(0)[0], Value::Integer(2));
    }

    #[test]
    fn aggregate_sum_excludes_null_and_propagates_numeric_type() {
        // SUM ignores NULL; integer + integer = integer; integer + float
        // promotes to float.
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in [Value::Integer(10), Value::Null, Value::Integer(20)]
            .into_iter()
            .enumerate()
        {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", age),
            );
        }
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Sum,
                distinct: false,
                star: false,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row(0)[0], Value::Integer(30));
    }

    #[test]
    fn aggregate_avg_excludes_null_from_numerator_and_denominator() {
        // amendment-03 §TIER-2-b: AVG denominator counts only non-NULL
        // rows (NULL excluded from BOTH numerator and denominator).
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in [
            Value::Integer(10),
            Value::Null,
            Value::Integer(30),
            Value::Null,
        ]
        .into_iter()
        .enumerate()
        {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", age),
            );
        }
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Avg,
                distinct: false,
                star: false,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        // (10 + 30) / 2 = 20.0 (NOT 10.0 — would fail if denom were 4).
        assert_eq!(b.row(0)[0], Value::Float(20.0));
    }

    #[test]
    fn aggregate_min_max_ignore_null() {
        // MIN / MAX ignore NULL; result reflects non-NULL operands only.
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in [Value::Integer(50), Value::Null, Value::Integer(20)]
            .into_iter()
            .enumerate()
        {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", age),
            );
        }
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![
                AggregateCall {
                    kind: AggregationKind::Min,
                    distinct: false,
                    star: false,
                    arg: prop_access(var_ref(BindingId::new(0)), "age"),
                    output_id: BindingId::new(0),
                },
                AggregateCall {
                    kind: AggregationKind::Max,
                    distinct: false,
                    star: false,
                    arg: prop_access(var_ref(BindingId::new(0)), "age"),
                    output_id: BindingId::new(0),
                },
            ],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row(0)[0], Value::Integer(20));
        assert_eq!(b.row(0)[1], Value::Integer(50));
    }

    #[test]
    fn aggregate_collect_drops_null_per_cypher_9_3_2_7() {
        // COLLECT drops NULL per Cypher 9 §3.2.7.
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in [
            Value::Integer(10),
            Value::Null,
            Value::Integer(20),
            Value::Null,
            Value::Integer(30),
        ]
        .into_iter()
        .enumerate()
        {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", age),
            );
        }
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Collect,
                distinct: false,
                star: false,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        match &b.row(0)[0] {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Integer(10));
                assert_eq!(items[1], Value::Integer(20));
                assert_eq!(items[2], Value::Integer(30));
            }
            other => panic!("expected List; got {other:?}"),
        }
    }

    #[test]
    fn aggregate_count_n_returns_total_row_count_when_n_is_non_null_for_every_row() {
        // amendment-03 §TIER-2-b: count(n) where n is non-NULL for every
        // row returns total row count. The "count(n) is the v1.0
        // surrogate for COUNT(*)" claim ONLY holds in this scope —
        // under OPTIONAL MATCH (TIER-1 GAP D + ADR-006 amendment-01
        // §A-2), n itself can be NULL in the outer scope, in which
        // case count(n) excludes those rows but COUNT(*) would include
        // them. The grammar at v1.0 does NOT admit COUNT(*); future
        // grammar lights add a variant via amendment.
        // W12α fix-up NIT-5 (PR #277 retro) renamed from
        // `aggregate_count_star_surrogate_via_count_n_returns_total_rows`
        // to remove the surrogate-equivalence overclaim.
        let s = make_n_persons(7);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        assert_eq!(b.row(0)[0], Value::Integer(7));
    }

    #[test]
    fn aggregate_with_group_by_emits_one_row_per_group() {
        // GROUP BY n.city; count(n) per city. 9 persons across 3 cities
        // (mod 3 keys) — 3 groups.
        let s = make_n_persons(9);
        let group_item = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(prop_access(var_ref(BindingId::new(0)), "city")),
            alias: None,
            // #746: group-by output id; this test reads columns
            // positionally so `Some(1)` is an arbitrary stable id (the
            // AggregateOp emits the group column under it).
            output_id: Some(BindingId::new(1)),
            // #353: hand-built bound item; not exercised for column-name
            // display in this positional-read test.
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![group_item],
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 3, "3 groups (city0/city1/city2)");
        // Each row: [city, count]. Sum of counts == total rows.
        let total: i64 = b
            .rows()
            .iter()
            .map(|r| match r[1] {
                Value::Integer(n) => n,
                _ => panic!("count column must be Integer"),
            })
            .sum();
        assert_eq!(total, 9);
    }

    #[test]
    fn aggregate_empty_input_no_groupby_emits_one_row_with_empty_aggregates() {
        // amendment-03 §TIER-2-b empty-input rule: a single-row aggregate
        // with no group_by + no input rows emits ONE row (count=0,
        // sum/avg/min/max/collect all NULL or empty list).
        let s = StubExecutorSubstrate::new();
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![
                AggregateCall {
                    kind: AggregationKind::Count,
                    distinct: false,
                    star: false,
                    arg: var_ref(BindingId::new(0)),
                    output_id: BindingId::new(0),
                },
                AggregateCall {
                    kind: AggregationKind::Sum,
                    distinct: false,
                    star: false,
                    arg: prop_access(var_ref(BindingId::new(0)), "age"),
                    output_id: BindingId::new(0),
                },
                AggregateCall {
                    kind: AggregationKind::Collect,
                    distinct: false,
                    star: false,
                    arg: var_ref(BindingId::new(0)),
                    output_id: BindingId::new(0),
                },
            ],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        assert_eq!(b.row(0)[0], Value::Integer(0));
        assert_eq!(b.row(0)[1], Value::Null);
        match &b.row(0)[2] {
            Value::List(items) => assert!(items.is_empty()),
            other => panic!("expected List; got {other:?}"),
        }
    }

    #[test]
    fn aggregate_propagates_cancel() {
        let s = make_n_persons(5);
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn aggregate_eos_after_emit_then_empty_batch() {
        // Single-group aggregate emits one batch then EOS.
        let s = make_n_persons(3);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "second batch is EOS");
    }

    #[test]
    fn aggregate_returns_null_for_sum_of_pure_null_input() {
        // SUM with all-NULL input → NULL (not zero).
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=3 {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Null),
            );
        }
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Sum,
                distinct: false,
                star: false,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row(0)[0], Value::Null);
    }

    #[test]
    fn accumulator_count_via_fold_only_increments_for_non_null() {
        let mut a = Accumulator::empty(AggregationKind::Count);
        a.fold(Value::Integer(1)).unwrap();
        a.fold(Value::Null).unwrap(); // Should NOT count.
        a.fold(Value::Integer(2)).unwrap();
        a.fold(Value::Null).unwrap();
        assert_eq!(a.finalize(), Value::Integer(2));
    }

    // -------------------------------------------------------------
    // Aggregate-associativity invariant — load-bearing for proptest
    // -------------------------------------------------------------

    #[test]
    fn aggregate_count_returns_total_row_count_across_concatenated_groups() {
        // Pin: count(n) over a 10-person substrate returns 10 (n is
        // non-NULL for every row). The actual associativity invariant
        // (`f(A∪B) == f(A) + f(B)`) lives in
        // `tests/m4_63_aggregate_proptest.rs::aggregate_count_is_associative`
        // — this unit test documents the simpler "count on a single
        // input" contract; W12α fix-up NIT-1 (PR #277 retro) renamed
        // from `aggregate_count_is_associative_under_concatenation_oracle`
        // to remove the overclaim (the test does NOT exercise
        // partition + merge — see the proptest for that invariant).
        let s_full = make_n_persons(10);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: false,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s_full).unwrap();
        match b.row(0)[0] {
            Value::Integer(n) => assert_eq!(n, 10),
            _ => panic!("count must be Integer"),
        }
    }

    #[test]
    fn compare_values_map_routes_through_orderability() {
        // ADR-191 D-5 — MIN/MAX over a map column is deterministic and
        // never collapses distinct maps (shares the `sort` oracle).
        use std::cmp::Ordering;
        let m1 = Value::Map([("a".to_string(), Value::Integer(1))].into_iter().collect());
        let m2 = Value::Map([("a".to_string(), Value::Integer(2))].into_iter().collect());
        assert_eq!(compare_values(&m1, &m2), Ordering::Less);
        assert_ne!(compare_values(&m1, &m2), Ordering::Equal);
        let node = Value::Node(NodeView::new(NodeId::new(1), None));
        assert_eq!(compare_values(&m1, &node), Ordering::Greater);
    }

    // -----------------------------------------------------------------
    // ADR-193 D-11 / test 8 — `min(p)` / `max(p)` over paths return the
    // orderability-extreme path (they do NOT error). Validated at the
    // accumulator (the real min/max semantics); `compare_values` gains
    // the deterministic Path arm (paths sort FIRST). The full-pipeline
    // `RETURN min(p)` is gated on the SAME pre-existing executor gap as
    // `RETURN count(n)` (projection over Aggregate does not execute
    // end-to-end on main), so the conformant min/max-path semantics are
    // pinned HERE.
    // -----------------------------------------------------------------
    fn agg_path(start: u64, segs: &[(u64, u64, u64)]) -> Value {
        use crate::executor::value::{NodeView, PathView, RelView};
        use arcgraph_core::{NodeId, RelId, TypeId};
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
    fn adr193_min_max_over_paths_return_extreme_path() {
        let p12 = agg_path(1, &[(10, 1, 2)]); // node-seq [1,2]
        let p13 = agg_path(1, &[(11, 1, 3)]); // node-seq [1,3]

        // compare_values: paths sort FIRST + deterministic node-seq order.
        assert_eq!(compare_values(&p12, &p13), std::cmp::Ordering::Less);
        assert_eq!(
            compare_values(&p12, &Value::Integer(0)),
            std::cmp::Ordering::Less
        );

        // MIN over {p13, p12} → p12 (smaller node-seq). NOT an error.
        let mut min_acc = Accumulator::empty(AggregationKind::Min);
        min_acc.fold(p13.clone()).expect("min fold p13");
        min_acc.fold(p12.clone()).expect("min fold p12");
        assert_eq!(min_acc.finalize(), p12, "min(p) returns the extreme path");

        // MAX over {p12, p13} → p13.
        let mut max_acc = Accumulator::empty(AggregationKind::Max);
        max_acc.fold(p12.clone()).expect("max fold p12");
        max_acc.fold(p13.clone()).expect("max fold p13");
        assert_eq!(max_acc.finalize(), p13, "max(p) returns the extreme path");
    }

    // -----------------------------------------------------------------
    // #773 G4/G5 — count(*) (star) + count/collect(DISTINCT) executor
    // semantics (openCypher v9 §3).
    // -----------------------------------------------------------------

    /// A `count(*)` aggregation (star). The arg is a placeholder the
    /// materialize loop never evaluates.
    fn count_star_call() -> AggregateCall {
        AggregateCall {
            kind: AggregationKind::Count,
            arg: var_ref(BindingId::new(0)), // placeholder; ignored for star
            output_id: BindingId::new(0),
            distinct: false,
            star: true,
        }
    }

    #[test]
    fn cz773_count_star_counts_every_row_including_all_null() {
        // 3 nodes; node 2 has a NULL `age`. count(*) counts ROWS (= 3),
        // while count(age) excludes the NULL (= 2). This is the
        // load-bearing count(*) ≠ count(expr) null distinction.
        let mut s = StubExecutorSubstrate::new();
        for (i, age) in [Value::Integer(30), Value::Null, Value::Integer(40)]
            .into_iter()
            .enumerate()
        {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("age", age),
            );
        }
        // count(*) → 3 (every row).
        let mut star_op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![count_star_call()],
        );
        let b = star_op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row(0)[0], Value::Integer(3), "count(*) counts all 3 rows");
        // count(age) → 2 (NULL excluded) — the contrast.
        let mut expr_op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Count,
                arg: prop_access(var_ref(BindingId::new(0)), "age"),
                output_id: BindingId::new(0),
                distinct: false,
                star: false,
            }],
        );
        let b2 = expr_op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(
            b2.row(0)[0],
            Value::Integer(2),
            "count(age) excludes the NULL row"
        );
    }

    #[test]
    fn cz773_count_star_with_group_by_counts_rows_per_group() {
        // 9 persons across 3 cities (mod-3 key). count(*) per city — each
        // group's count is its ROW count; the three sum to 9.
        let s = make_n_persons(9);
        let group_item = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(prop_access(var_ref(BindingId::new(0)), "city")),
            alias: None,
            output_id: Some(BindingId::new(1)),
            // #353: hand-built bound item; positional-read test.
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![group_item],
            vec![count_star_call()],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row_count(), 3, "3 city groups");
        let total: i64 = b
            .rows()
            .iter()
            .map(|r| match r[1] {
                Value::Integer(n) => n,
                _ => panic!("count(*) column must be Integer"),
            })
            .sum();
        assert_eq!(total, 9, "count(*) per group sums to the total row count");
    }

    #[test]
    fn cz773_count_star_empty_input_is_zero() {
        // amendment-03 §TIER-2-b empty-input rule: count(*) over zero rows
        // (no group-by) emits ONE row of 0.
        let s = StubExecutorSubstrate::new();
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![count_star_call()],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row_count(), 1);
        assert_eq!(b.row(0)[0], Value::Integer(0), "count(*) of nothing is 0");
    }

    /// Build a substrate of accounts each carrying a `country` property.
    fn accounts_with_countries(countries: &[Value]) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for (i, c) in countries.iter().enumerate() {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                    .with_property("country", c.clone()),
            );
        }
        s
    }

    fn distinct_count_country_call() -> AggregateCall {
        AggregateCall {
            kind: AggregationKind::Count,
            arg: prop_access(var_ref(BindingId::new(0)), "country"),
            output_id: BindingId::new(0),
            distinct: true,
            star: false,
        }
    }

    #[test]
    fn cz773_count_distinct_dedups_values() {
        // countries [US, US, UK] → count(DISTINCT country) = 2.
        let s = accounts_with_countries(&[
            Value::String("US".into()),
            Value::String("US".into()),
            Value::String("UK".into()),
        ]);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![distinct_count_country_call()],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row(0)[0], Value::Integer(2), "2 distinct countries");
    }

    #[test]
    fn cz773_count_distinct_excludes_null() {
        // [US, US, NULL, UK] → count(DISTINCT country) = 2 (NULL excluded,
        // like count(expr)).
        let s = accounts_with_countries(&[
            Value::String("US".into()),
            Value::String("US".into()),
            Value::Null,
            Value::String("UK".into()),
        ]);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![distinct_count_country_call()],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(
            b.row(0)[0],
            Value::Integer(2),
            "count(DISTINCT) excludes NULL"
        );
    }

    #[test]
    fn cz773_collect_distinct_dedups_in_first_seen_order() {
        // collect(DISTINCT country) over [US, US, UK, US] → [US, UK]
        // (deduped; first-seen accumulation order). NULL excluded.
        let s = accounts_with_countries(&[
            Value::String("US".into()),
            Value::String("US".into()),
            Value::String("UK".into()),
            Value::Null,
            Value::String("US".into()),
        ]);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Collect,
                arg: prop_access(var_ref(BindingId::new(0)), "country"),
                output_id: BindingId::new(0),
                distinct: true,
                star: false,
            }],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        match &b.row(0)[0] {
            Value::List(items) => {
                assert_eq!(
                    items,
                    &vec![Value::String("US".into()), Value::String("UK".into())],
                    "collect(DISTINCT) dedups + drops NULL, first-seen order"
                );
            }
            other => panic!("expected List; got {other:?}"),
        }
    }

    #[test]
    fn cz773_collect_non_distinct_keeps_duplicates() {
        // Regression: collect(country) (NON-distinct) keeps duplicates.
        let s = accounts_with_countries(&[
            Value::String("US".into()),
            Value::String("US".into()),
            Value::String("UK".into()),
        ]);
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            Vec::new(),
            vec![AggregateCall {
                kind: AggregationKind::Collect,
                arg: prop_access(var_ref(BindingId::new(0)), "country"),
                output_id: BindingId::new(0),
                distinct: false,
                star: false,
            }],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        match &b.row(0)[0] {
            Value::List(items) => assert_eq!(items.len(), 3, "non-distinct keeps all 3"),
            other => panic!("expected List; got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // #1008 — high-cardinality GROUP BY complexity fix (O(N²)→O(N))
    // -------------------------------------------------------------

    /// `make_n_persons(n)` keys `city` as `format!("city{}", i % 3)` over
    /// `i = 1..=n`. The scan visits nodes in ascending `NodeId` order, so
    /// the FIRST-SEEN order of distinct cities is:
    ///   i=1 → city1, i=2 → city2, i=3 → city0, i=4 → city1 (repeat) …
    /// i.e. **city1, city2, city0**. This fixes the deterministic output
    /// order under test so the #1008 fix (gate the `group_order` push on
    /// the HashMap's first-insert vacancy instead of a per-row `.contains()`
    /// scan) is proven to preserve EXACT first-seen ordering, not just the
    /// group SET. A fix that pushed on every row, or used HashMap iteration
    /// order, or reordered groups would fail this assertion.
    #[test]
    fn aggregate_group_by_preserves_exact_first_seen_order_1008() {
        let s = make_n_persons(9);
        let group_item = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(prop_access(var_ref(BindingId::new(0)), "city")),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![group_item],
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: true,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row_count(), 3, "3 distinct cities");
        // EXACT first-seen order: city1, city2, city0 (NOT sorted, NOT
        // HashMap-iteration order). Each city appears for 3 of 9 persons.
        let observed: Vec<(String, i64)> = b
            .rows()
            .iter()
            .map(|r| {
                let city = match &r[0] {
                    Value::String(s) => s.clone(),
                    other => panic!("group column must be String; got {other:?}"),
                };
                let count = match r[1] {
                    Value::Integer(n) => n,
                    ref other => panic!("count column must be Integer; got {other:?}"),
                };
                (city, count)
            })
            .collect();
        assert_eq!(
            observed,
            vec![
                ("city1".to_string(), 3),
                ("city2".to_string(), 3),
                ("city0".to_string(), 3),
            ],
            "GROUP BY must emit groups in EXACT first-seen order (#1008 fix \
             must not change ordering when gating the group_order push on \
             first-insert vacancy)"
        );
    }

    /// Perf-shape guard for #1008: GROUP BY over a high-cardinality key
    /// (one distinct group PER ROW). The pre-fix code ran an O(G) linear
    /// `Vec::contains` scan on `group_order` PER ROW → O(N²) total, which
    /// blew the 30s query timeout at ~200K distinct groups. The fix gates
    /// the push on the HashMap's first-insert vacancy → O(N) total.
    ///
    /// `#[ignore]` (not a default-CI gate): this is a corroborating
    /// timing-shape check, NOT the primary correctness guard (that is
    /// `aggregate_group_by_preserves_exact_first_seen_order_1008`). It is
    /// deliberately kept off the flaky-wall-clock CI path per the
    /// flaky-threshold lesson; run with `--ignored` to corroborate the
    /// complexity. 200 000 distinct groups completes in well under a
    /// second on the O(N) path; the O(N²) path (which scales quadratically
    /// — per #1008's reproduce-first: 14.7→33.3→76.7 µs/row as N doubles
    /// 25K→50K→100K) takes tens of seconds at 200K and blows the budget.
    #[test]
    #[ignore = "perf-shape corroboration for #1008; run with --ignored (O(N²) regression blows the budget)"]
    fn aggregate_group_by_high_cardinality_completes_1008() {
        // Each person's id is its own group key (uid == id over n=200_000
        // distinct ids ⇒ one row per group ⇒ G == N == 200_000). At 200K the
        // O(N²) `.contains()` dedup runs tens of seconds (well past the 10s
        // budget); the O(N) first-insert-gated path stays sub-second.
        const N: u64 = 200_000;
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=N {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("uid", Value::Integer(i as i64)),
            );
        }
        let group_item = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(prop_access(var_ref(BindingId::new(0)), "uid")),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut op = AggregateOp::new(
            PhysicalOperator::Scan(person_scan()),
            vec![group_item],
            vec![AggregateCall {
                kind: AggregationKind::Count,
                distinct: false,
                star: true,
                arg: var_ref(BindingId::new(0)),
                output_id: BindingId::new(0),
            }],
        );
        let ctx = ctx();
        let start = std::time::Instant::now();
        // `materialize` (where the O(N²) `group_order` dedup lives) runs on
        // the first `next_batch`; output is then paginated at BATCH_ROWS, so
        // drain every batch to count all distinct groups.
        let mut total_groups: u64 = 0;
        loop {
            let b = op.next_batch(&ctx, &s).unwrap();
            if b.row_count() == 0 {
                break;
            }
            total_groups += b.row_count() as u64;
        }
        let elapsed = start.elapsed();
        assert_eq!(total_groups, N, "one group per distinct uid");
        // Generous threshold: the O(N) path is sub-second; the O(N²) path
        // (per-row Vec::contains over up to 100K entries) takes tens of
        // seconds and blows this. NOT on the default CI path (#[ignore]).
        assert!(
            elapsed.as_secs() < 10,
            "200K-group GROUP BY took {elapsed:?} — O(N²) regression suspected (#1008)"
        );
    }
}
