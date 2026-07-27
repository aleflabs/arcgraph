//! Per-operator implementations for the M4-61 / M4-62 executor.
//!
//! [`PhysicalOperator`] is a concrete enum dispatching to per-operator
//! state machines. Per the 7-slice 3-strike pattern the operator layer
//! is NOT a `Box<dyn Op>` trait abstraction — it's an enum so the
//! whole pipeline is monomorphic + alloc-free at the operator layer.
//! M4-72 is the inflection point that re-evaluates trait extraction.
//!
//! # Per-operator state
//!
//! Each variant carries its own state struct (e.g.,
//! [`scan::ScanOp`] holds the buffered scan-result vec + cursor).
//! State is initialized at [`crate::executor::Pipeline::build`] time
//! (mostly cheap clones from the LogicalPlan); per-operator hot-loop
//! state lives in mutable fields on the variant.
//!
//! # Cancellation
//!
//! Every variant's `next_batch` calls
//! [`crate::executor::CancellationToken::check`] BEFORE pulling rows
//! from a child / substrate. This is the per-operator batch-boundary
//! check that ADR-038 amendment-02 §M4.f mandates.

pub mod aggregate;
pub mod call;
pub mod count_store;
pub mod create_node;
pub mod create_property_index;
pub mod create_rel;
pub mod create_spine;
pub mod create_vector_index;
pub mod delete;
pub mod distinct;
pub mod expand;
mod expand_spill;
pub mod filter;
mod grace_hash_join;
pub mod join;
pub mod limit;
/// ADR-152-amendment-02 (W28) — shared write-op composite-literal lift
/// (`List`-of-scalars persistence). Internal to `ops`; consumed by
/// `create_node` / `create_rel` / `set` via `super::literal_lift`.
mod literal_lift;
pub mod merge;
pub mod merge_join;
pub mod optional_expand;
/// ADR-226 §4 S4 (CONC-D) — morsel-driven parallel node-scan operator.
pub mod parallel_aggregate;
pub mod parallel_scan;
pub mod path;
pub mod plain_path;
/// ADR-197 (#802) — `CALL <proc>(…) [YIELD …]` / `SHOW …` generating op.
pub mod procedure_call;
pub mod project;
pub mod property_index_scan;
pub mod rank_by_hybrid;
pub mod remove;
pub mod scan;
pub mod set;
pub mod skip;
pub mod sort;
mod sort_spill_codec;
pub mod union;
pub mod unwind;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::substrate::ExecutorSubstrate;
use crate::observer::OperatorKind;
use crate::observer::dispatcher::record_dispatch;
use crate::semantic::bound_ast::BindingId;

// ADR-147-amendment-03 §B1: re-export the result-level list cap so the
// eval-layer per-op concat cap (`eval::MAX_CONCAT_LIST_LEN`) can be
// compiler-checked equal to it (the two MUST agree; see the const docs).
pub(crate) use literal_lift::MAX_CREATE_PROP_LIST_LEN;

pub use aggregate::{AggregateCall, AggregateOp};
pub use call::{CallBodyFactory, CallOp, CorrelationSeedOp};
pub use count_store::CountStoreOp;
pub use create_node::CreateNodeOp;
pub use create_property_index::CreatePropertyIndexOp;
pub use create_rel::CreateRelOp;
pub use create_spine::{CreateSpineItem, CreateSpineNode, CreateSpineOp, CreateSpineRel};
pub use create_vector_index::CreateVectorIndexOp;
pub use delete::{DeleteItemSpec, DeleteOp};
pub use distinct::DistinctOp;
pub use expand::ExpandOp;
pub use expand_spill::ExpandSpillTarget;
#[cfg(feature = "fault-injection")]
pub use expand_spill::{ExpandSpillProbe, ExpandSpillStats};
pub use filter::FilterOp;
pub use grace_hash_join::{
    DEFAULT_GRACE_MAX_REPARTITION_DEPTH, GraceHashJoinTarget, MAX_GRACE_PARTITIONS,
};
#[cfg(feature = "fault-injection")]
pub use grace_hash_join::{GraceHashJoinProbe, GraceHashJoinStats};
pub use join::HashJoinOp;
pub use limit::LimitOp;
pub use merge::{MergeActionSpec, MergeOp};
// NN-4 (#1384) re-spin, Fix 1 — the driver-level guard-acquisition hook.
pub(crate) use merge::acquire_merge_guards;
pub use merge_join::MergeJoinOp;
pub use optional_expand::{OptionalExpandOp, SingletonScanOp};
pub use parallel_aggregate::ParallelAggregateOp;
pub use parallel_scan::ParallelScanOp;
pub use path::{NamedShortestPathOp, PathSpec};
pub use plain_path::PlainPathOp;
pub use procedure_call::ProcedureCallOp;
pub use project::ProjectOp;
pub use property_index_scan::PropertyIndexScanOp;
pub use rank_by_hybrid::{FusionOp, RankByHybridOp};
pub use remove::{RemoveItemSpec, RemoveOp};
pub use scan::{EmptyOp, ScanOp};
pub use set::{SetItemSpec, SetOp};
pub use skip::SkipOp;
pub use sort::{
    DEFAULT_EXTERNAL_SORT_FAN_IN, MAX_EXTERNAL_SORT_FAN_IN, SortKey, SortOp, SortSpillTarget,
};
#[cfg(feature = "fault-injection")]
pub use sort::{ExternalSortProbe, ExternalSortStats};
pub use union::UnionOp;
pub use unwind::UnwindOp;

/// Concrete physical-operator dispatch enum.
///
/// Each variant wraps the per-operator state. The
/// [`Self::next_batch`] dispatch is exhaustive — adding a new variant
/// requires updating the match arm.
#[derive(Debug)]
pub enum PhysicalOperator {
    /// Sequential scan over a tenant's nodes (label-filtered).
    Scan(ScanOp),
    /// **#1366 (Phase 2).** Indexed point-lookup — candidate B+tree
    /// lookup → MVCC-verify → residual filter → page (`O(matches)`).
    /// Replaces the anchor `Scan + Filter` for an equality on an
    /// **Online**-indexed labelled property.
    PropertyIndexScan(PropertyIndexScanOp),
    /// **ADR-226 §4 S4 (gate CONC-D).** Morsel-driven parallel scan —
    /// same result as [`Self::Scan`] (same multiset + id-order) but the
    /// buffer is split into ~64K-record morsels filtered in parallel on
    /// a dedicated rayon pool (`cores − 4`). Built by the pipeline only
    /// when `ARCGRAPH_PARALLEL_SCAN` is set (revert path: flag off →
    /// serial [`Self::Scan`]). Carries the optional WHERE predicate for
    /// per-morsel filter pushdown.
    ParallelScan(ParallelScanOp),
    /// O(1) tenant-wide count from the counts store.
    CountStore(CountStoreOp),
    /// Single-row scan keyed on a specific NodeId — used by
    /// [`OptionalExpandOp`]'s right-side factory to root a per-LEFT-
    /// row sub-pipeline. v1.0-alpha bridge; M4-32 / M4-72 forward
    /// will re-shape OPTIONAL MATCH right-side parameterization.
    Singleton(SingletonScanOp),
    /// One-hop relationship traversal.
    Expand(ExpandOp),
    /// WHERE / WITH WHERE predicate filter.
    Filter(FilterOp),
    /// RETURN / WITH projection.
    Project(ProjectOp),
    /// Sentinel for the degenerate empty-clauses case.
    Empty(EmptyOp),
    /// `RANK BY HYBRID(VECTOR(...), TEXT(...))` orchestration.
    RankByHybrid(RankByHybridOp),
    /// `WITH FUSION = RRF(k = N)` fusion node.
    Fusion(FusionOp),
    /// OPTIONAL MATCH lowered to a left-outer-expand (M4-62).
    OptionalExpand(OptionalExpandOp),
    /// GROUP BY + aggregate functions (M4-63).
    Aggregate(AggregateOp),
    /// **ADR-226 §4 S5 (gate CONC-D).** Morsel-driven parallel partial
    /// aggregate — same result as [`Self::Aggregate`] on the mergeable
    /// scan-aggregate shape (no GROUP BY, no DISTINCT, no COLLECT) but the
    /// buffered rows are split into morsels, folded to partial
    /// accumulators in parallel on S4's dedicated pool, then merged
    /// single-threaded (COUNT/SUM add, MIN/MAX extreme, AVG carries
    /// `(sum,count)`). Built by the pipeline only when
    /// `ARCGRAPH_PARALLEL_SCAN` is set AND
    /// [`ParallelAggregateOp::is_mergeable`] holds (revert path: flag off
    /// or not mergeable → serial [`Self::Aggregate`]).
    ParallelAggregate(ParallelAggregateOp),
    /// ORDER BY (M4-63). Stable sort with optional spillover when the
    /// per-tenant memory budget would be exceeded.
    Sort(SortOp),
    /// `MATCH p = SHORTEST_PATH(...)` named-shortest-path (M4-63 BFS
    /// only at v1.0; DFS / A* deferred per ADR-038 §2 D-7).
    NamedShortestPath(NamedShortestPathOp),
    /// **ADR-193 D-4.** `MATCH p = (a)-[..]->(b)` plain named-path —
    /// materializes a `Value::Path` from the MATCH-bound rows (appends
    /// `path_var` as a new column). Sibling of [`Self::NamedShortestPath`]
    /// (both originate from [`crate::logical_plan::LogicalNamedPath`]);
    /// the Plain variant does NOT re-traverse the substrate.
    PlainPath(PlainPathOp),
    /// `LIMIT N` (M4-63).
    Limit(LimitOp),
    /// `SKIP N` offset-pagination (#842 part A). Companion of
    /// [`Self::Limit`]; `SKIP n LIMIT m` composes `Limit(Skip(child))`.
    Skip(SkipOp),
    /// `LogicalJoin` (multi-pattern equi-join + Cartesian) executor
    /// — W17α / M4-08+ per ADR-038 §2 D-24.
    HashJoin(HashJoinOp),
    /// `LogicalJoin` sort-merge equi-join executor — W25-M4-61b /
    /// ADR-097. Picked when the cost-based picker
    /// ([`crate::planner::pick_join_algorithms`]) selects
    /// [`crate::logical_plan::JoinAlgorithm::MergeJoin`] over hash;
    /// Cartesian always routes to [`PhysicalOperator::HashJoin`]
    /// regardless of cost.
    MergeJoin(MergeJoinOp),
    /// **ADR-147 W26-θ Phase 1.** Write-op operator — `CREATE
    /// (var?:Label? {props})` — emits one row binding the new
    /// `NodeId` (or an empty row for anonymous CREATEs).
    CreateNode(CreateNodeOp),
    /// **ADR-148 W26-θ Phase 2.** Write-op operator — `CREATE
    /// (a)-[r:LABEL {props}]->(b)` — pulls source + target NodeIds
    /// from upstream sub-pipelines, writes the rel via the substrate,
    /// emits one row binding the new `RelId` (or an empty row for
    /// anonymous CREATE-rels).
    CreateRel(CreateRelOp),
    /// Composite CREATE-item spine. Executes contiguous CREATE items in
    /// an iterative loop per input row so pull stack depth is bounded by
    /// clause nesting, not CREATE item count (#1123 R2).
    CreateSpine(CreateSpineOp),
    /// **#830 / ADR-198 §OQ-7 / ADR-200.** Accept-and-register write-op
    /// — `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
    /// var.prop [OPTIONS {…}]` — resolves the `$name` / OPTIONS params,
    /// registers a metadata entry in the per-tenant vector-index catalog
    /// via the substrate (NO heavyweight build; the served HNSW
    /// auto-builds on ingest per #765 PART-1). Emits ZERO rows.
    CreateVectorIndex(CreateVectorIndexOp),
    /// **#1366 (task #248, Phase 1).** Write-op operator — `CREATE INDEX
    /// <name> [IF NOT EXISTS] FOR (var:Label) ON (var.prop)` (property
    /// index). Registers `Building`, backfills the MVCC-visible nodes,
    /// flips `Online` via the substrate. Emits ZERO rows.
    CreatePropertyIndex(CreatePropertyIndexOp),
    /// **ADR-149 W26-θ Phase 3.** Write-op operator — `DELETE var
    /// (, var)*` / `DETACH DELETE var (, var)*` — consumes the
    /// upstream MATCH-bound rows, tombstones each item's resolved
    /// NodeId / RelId via the substrate's `delete_node` /
    /// `delete_rel`. Output schema is empty (DELETE is a terminal
    /// clause at Phase 3).
    Delete(DeleteOp),
    /// **ADR-150 W26-θ Phase 4 (#709 fix, R1-narrowed).** Write-op
    /// operator — `SET <item> (, <item>)*` — consumes the upstream
    /// MATCH-bound rows, dispatches each item's mutation (property
    /// assign / merge / replace / label-add) to the substrate's
    /// `set_node` / `set_rel`. Emission is **terminal-vs-stacked**
    /// (output schema = input schema either way): a **stacked** SET (the
    /// inner clause of `SET … SET …` / `SET … REMOVE …`, marked at
    /// [`crate::executor::Pipeline::build`] time) PASSES its mutated rows
    /// THROUGH so the outer write-op composes (#709 last-writer-wins); a
    /// **terminal** SET (pipeline root / no write-op consumer above)
    /// DRAINS the upstream and emits **0 rows** — the RETURN-less
    /// terminal-write contract (openCypher v9 / ADR-149/150 §D /
    /// ADR-182). See [`SetOp`] for the flag mechanics.
    Set(SetOp),
    /// **ADR-150 W26-θ Phase 4 (#709 fix, R1-narrowed).** Write-op
    /// operator — `REMOVE <item> (, <item>)*` — consumes the upstream
    /// MATCH-bound rows, dispatches each item's removal (property /
    /// label) to the substrate's `remove_node` / `remove_rel`. Emission
    /// is **terminal-vs-stacked** like [`SetOp`]: a **stacked** REMOVE
    /// passes its mutated rows through so a stacked outer write-op
    /// composes (#709); a **terminal** REMOVE drains the upstream and
    /// emits **0 rows** (the RETURN-less terminal-write contract).
    Remove(RemoveOp),
    /// **ADR-151 W26-θ Phase 5.** Write-op operator — `MERGE
    /// <pattern> [ON CREATE SET …]* [ON MATCH SET …]*` — probes the
    /// match-branch sub-pipeline; if non-empty, fires `on_match`
    /// actions per matched row; if empty, fires the create-branch
    /// sub-pipeline + the `on_create` actions per created row.
    /// Output schema is empty (MERGE is terminal at Phase 5).
    Merge(MergeOp),
    /// **ADR-185 (#649-A1, W28).** Row-dedup operator — closes the
    /// prior `RETURN DISTINCT` `NotImplemented` (`pipeline.rs`).
    /// Materializes its child, emits each row whose canonical key is
    /// seen for the first time (hash-set; O(distinct-cardinality)
    /// memory). A bare `UNION` (distinct) composes this over a
    /// [`UnionOp`] in #649-A2.
    Distinct(DistinctOp),
    /// **ADR-185 (#649-A1, W28).** `UNION ALL` concat — streams each
    /// arm's rows in order (O(1) extra memory), realigning columns to
    /// arm 0's order per openCypher v9 §8.
    Union(UnionOp),
    /// **ADR-038 D-28 §7 (#618).** `UNWIND <list> AS <var>` — streaming
    /// 1-to-N op: one output row per list element, extending the child
    /// schema with `var`. Closes the prior `LogicalPlan::Unwind`
    /// `NotImplemented` (`pipeline.rs`). openCypher v9 §6.7.
    Unwind(UnwindOp),
    /// **ADR-197 (#802).** `CALL <proc>(…) [YIELD …]` / `SHOW …`
    /// schema-introspection generating op (the langchain-neo4j
    /// `refresh_schema` surface). Emits the procedure / SHOW result
    /// rows from a single driving (unit) row, then EOS.
    ProcedureCall(ProcedureCallOp),
    /// **ADR-192 (#623).** `CALL { <subquery> }` correlated brace-subquery
    /// (Cypher 25, beyond openCypher v9). Per driving row: pushes a
    /// correlation frame, (re-)runs the subquery body, emits
    /// `driving_row ++ body_output` (UNION-ALL across driving rows). The
    /// [`UnwindOp`] analogue with a `body` sub-plan + a `pending`
    /// cross-batch cursor.
    Call(CallOp),
    /// **ADR-192 (#623).** The one-row correlation seed feeding a
    /// `CALL { … }` body — emits the current driving row's imported
    /// bindings (read from the [`ExecutionContext`] correlation frame
    /// [`CallOp`] pushes). Appears ONLY inside a [`CallOp`]'s body
    /// sub-pipeline.
    CorrelationSeed(CorrelationSeedOp),
}

impl PhysicalOperator {
    /// Pull the next batch of rows from this operator.
    ///
    /// Returns an empty [`Batch`] (`row_count() == 0`) when the
    /// operator is exhausted (the EOS sentinel per
    /// [`crate::executor::execute_with_context`]).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Cancel-check at every operator boundary — defense in depth.
        // Individual operators also call this; the redundant check is
        // cheap and prevents a missed-trip from a future op edit.
        ctx.cancellation().check()?;
        // Bump the per-query batch sequence counter for slow-query
        // observability (forward to M4-71). Done at the dispatch
        // boundary so EVERY operator's batch tick is counted, not
        // just the root.
        let _seq = ctx.next_batch_seq();
        // M4-71 row-count observer hook. We capture the operator kind +
        // start-time BEFORE the inner match so we can record the per-
        // batch row count + wall-time when the inner match returns.
        // The hook is gated by `ctx.observer().is_some()` inside
        // `record_dispatch`; the option-presence check is the only
        // overhead on the no-observer hot path.
        let op_kind = self.op_kind();
        let observer_start = std::time::Instant::now();
        let result = match self {
            PhysicalOperator::Scan(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::PropertyIndexScan(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::ParallelScan(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CountStore(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Singleton(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Expand(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Filter(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Project(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Empty(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::RankByHybrid(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Fusion(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::OptionalExpand(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Aggregate(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::ParallelAggregate(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Sort(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::NamedShortestPath(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::PlainPath(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Limit(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Skip(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::HashJoin(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::MergeJoin(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CreateNode(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CreateVectorIndex(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CreatePropertyIndex(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CreateRel(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CreateSpine(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Delete(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Set(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Remove(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Merge(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Distinct(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Union(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Unwind(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::ProcedureCall(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::Call(op) => op.next_batch(ctx, substrate),
            PhysicalOperator::CorrelationSeed(op) => op.next_batch(ctx, substrate),
        };
        record_dispatch(ctx, op_kind, &result, observer_start);
        result
    }

    /// The output schema (the per-row column layout this operator
    /// produces). The schema is a list of [`BindingId`]s in column
    /// order; the i-th row cell corresponds to the i-th schema slot.
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        match self {
            PhysicalOperator::Scan(op) => op.schema(),
            PhysicalOperator::PropertyIndexScan(op) => op.schema(),
            PhysicalOperator::ParallelScan(op) => op.schema(),
            PhysicalOperator::CountStore(op) => op.schema(),
            PhysicalOperator::Singleton(op) => op.schema(),
            PhysicalOperator::Expand(op) => op.schema(),
            PhysicalOperator::Filter(op) => op.schema(),
            PhysicalOperator::Project(op) => op.schema(),
            PhysicalOperator::Empty(op) => op.schema(),
            PhysicalOperator::RankByHybrid(op) => op.schema(),
            PhysicalOperator::Fusion(op) => op.schema(),
            PhysicalOperator::OptionalExpand(op) => op.schema(),
            PhysicalOperator::Aggregate(op) => op.schema(),
            PhysicalOperator::ParallelAggregate(op) => op.schema(),
            PhysicalOperator::Sort(op) => op.schema(),
            PhysicalOperator::NamedShortestPath(op) => op.schema(),
            PhysicalOperator::PlainPath(op) => op.schema(),
            PhysicalOperator::Limit(op) => op.schema(),
            PhysicalOperator::Skip(op) => op.schema(),
            PhysicalOperator::HashJoin(op) => op.schema(),
            PhysicalOperator::MergeJoin(op) => op.schema(),
            PhysicalOperator::CreateNode(op) => op.schema(),
            PhysicalOperator::CreateVectorIndex(op) => op.schema(),
            PhysicalOperator::CreatePropertyIndex(op) => op.schema(),
            PhysicalOperator::CreateRel(op) => op.schema(),
            PhysicalOperator::CreateSpine(op) => op.schema(),
            PhysicalOperator::Delete(op) => op.schema(),
            PhysicalOperator::Set(op) => op.schema(),
            PhysicalOperator::Remove(op) => op.schema(),
            PhysicalOperator::Merge(op) => op.schema(),
            PhysicalOperator::Distinct(op) => op.schema(),
            PhysicalOperator::Union(op) => op.schema(),
            PhysicalOperator::Unwind(op) => op.schema(),
            PhysicalOperator::ProcedureCall(op) => op.schema(),
            PhysicalOperator::Call(op) => op.schema(),
            PhysicalOperator::CorrelationSeed(op) => op.schema(),
        }
    }

    pub(crate) fn rearm_create_endpoint_leaf(&mut self) {
        match self {
            PhysicalOperator::CreateNode(op) => op.rearm_leaf_endpoint(),
            PhysicalOperator::CreateSpine(op) => op.rearm_leaf_endpoint(),
            _ => {}
        }
    }

    /// Return the [`OperatorKind`] for this physical operator variant.
    ///
    /// Mirror of the variant set; used by the M4-71 observer dispatcher
    /// hook (see [`crate::observer::dispatcher::record_dispatch`]) and
    /// future M4-72 replan-from-position attribution. Per the
    /// `feedback_avoid_speculative_scaffolding.md` 7-slice 3-strike
    /// pattern, this is a CONCRETE method on the enum (not a trait), so
    /// the dispatch is monomorphic + zero-cost.
    ///
    /// # W13β precondition fix — W12α/W12β rebase miss
    ///
    /// PR #278 (W12β observer) added this method but omitted the four
    /// W12α-added variants ([`PhysicalOperator::Aggregate`],
    /// [`PhysicalOperator::Sort`], [`PhysicalOperator::NamedShortestPath`],
    /// [`PhysicalOperator::Limit`]) that PR #277 had landed two commits
    /// earlier. The result was a compile break on `main` (E0004
    /// non-exhaustive match). The four arms below dispatch to the
    /// dedicated [`OperatorKind`] variants that the W12α follow-up
    /// (see `crate::observer::row_count::OperatorKind`'s `Aggregate /
    /// Sort / NamedShortestPath / Limit` doc-comments) introduced in
    /// lockstep with this method. A future slice that adds further
    /// physical-operator variants MUST extend both this method and
    /// `walk_inner` in [`crate::observer::row_count`] together —
    /// the exhaustive matches on both sides will fail to compile
    /// until they are synchronized.
    #[inline]
    #[must_use]
    pub fn op_kind(&self) -> OperatorKind {
        match self {
            PhysicalOperator::Scan(_) => OperatorKind::Scan,
            // #1366 (Phase 2): a node SOURCE (produces node rows), same
            // observer taxonomy class as `Scan` — the operator NAME
            // distinguishes the index path in EXPLAIN; the row-count
            // observer groups both as `Scan`.
            PhysicalOperator::PropertyIndexScan(_) => OperatorKind::Scan,
            // The parallel scan is the same LOGICAL operator (a node
            // scan) as [`Self::Scan`], just morsel-parallelized — the
            // observer groups both under `Scan` (no new taxonomy).
            PhysicalOperator::ParallelScan(_) => OperatorKind::Scan,
            PhysicalOperator::CountStore(_) => OperatorKind::Aggregate,
            PhysicalOperator::Singleton(_) => OperatorKind::Singleton,
            PhysicalOperator::Expand(_) => OperatorKind::Expand,
            PhysicalOperator::Filter(_) => OperatorKind::Filter,
            PhysicalOperator::Project(_) => OperatorKind::Project,
            PhysicalOperator::Empty(_) => OperatorKind::Empty,
            PhysicalOperator::RankByHybrid(_) => OperatorKind::RankByHybrid,
            PhysicalOperator::Fusion(_) => OperatorKind::Fusion,
            PhysicalOperator::OptionalExpand(_) => OperatorKind::OptionalExpand,
            PhysicalOperator::Aggregate(_) => OperatorKind::Aggregate,
            // The parallel partial aggregate is the same LOGICAL operator
            // (an aggregate) as [`Self::Aggregate`], just morsel-
            // parallelized — the observer groups both under `Aggregate`.
            PhysicalOperator::ParallelAggregate(_) => OperatorKind::Aggregate,
            PhysicalOperator::Sort(_) => OperatorKind::Sort,
            PhysicalOperator::NamedShortestPath(_) => OperatorKind::NamedShortestPath,
            // ADR-193 — the plain named-path op shares the NamedPath
            // logical-operator family with NamedShortestPath; the
            // observer groups both under `NamedShortestPath` (the
            // `LogicalPlan::NamedPath` → `OperatorKind::NamedShortestPath`
            // mapping in `observer::row_count::walk_inner` already covers
            // both kinds, so no new `OperatorKind` variant is required).
            PhysicalOperator::PlainPath(_) => OperatorKind::NamedShortestPath,
            PhysicalOperator::Limit(_) => OperatorKind::Limit,
            PhysicalOperator::Skip(_) => OperatorKind::Skip,
            PhysicalOperator::HashJoin(_) => OperatorKind::HashJoin,
            PhysicalOperator::MergeJoin(_) => OperatorKind::MergeJoin,
            PhysicalOperator::CreateNode(_) => OperatorKind::CreateNode,
            PhysicalOperator::CreateVectorIndex(_) => OperatorKind::CreateVectorIndex,
            PhysicalOperator::CreatePropertyIndex(_) => OperatorKind::CreatePropertyIndex,
            PhysicalOperator::CreateRel(_) => OperatorKind::CreateRel,
            PhysicalOperator::CreateSpine(_) => OperatorKind::CreateRel,
            PhysicalOperator::Delete(_) => OperatorKind::Delete,
            PhysicalOperator::Set(_) => OperatorKind::Set,
            PhysicalOperator::Remove(_) => OperatorKind::Remove,
            PhysicalOperator::Merge(_) => OperatorKind::Merge,
            PhysicalOperator::Distinct(_) => OperatorKind::Distinct,
            PhysicalOperator::Union(_) => OperatorKind::Union,
            PhysicalOperator::Unwind(_) => OperatorKind::Unwind,
            PhysicalOperator::ProcedureCall(_) => OperatorKind::ProcedureCall,
            PhysicalOperator::Call(_) => OperatorKind::Call,
            PhysicalOperator::CorrelationSeed(_) => OperatorKind::CorrelationSeed,
        }
    }
}

/// Look up a binding's column index in a schema. Linear scan since
/// per-operator schemas are small (typically 1-5 columns).
#[inline]
#[must_use]
pub fn schema_index(schema: &[BindingId], target: BindingId) -> Option<usize> {
    schema.iter().position(|&b| b == target)
}

/// Default per-query parameter bag — empty.
///
/// v1.0-alpha tests pass an explicit parameters bag through
/// [`crate::executor::Pipeline::build_with_parameters`]; the default
/// for queries that don't reference parameters is the empty bag.
#[must_use]
pub fn empty_parameters() -> Parameters {
    Parameters::new()
}

/// Canonical, hashable rendering of a row's value tuple — the dedup /
/// grouping key. Stable across the life of ONE query (used as a
/// `HashMap`/`HashSet` key); NOT portable across queries or processes.
///
/// Shared by [`crate::executor::ops::DistinctOp`] (ADR-185 row dedup)
/// and [`crate::executor::ops::aggregate::AggregateOp`] (GROUP BY) so
/// `RETURN DISTINCT` / `UNION` and `GROUP BY` use IDENTICAL value-
/// equality semantics (openCypher v9 §3.1: two rows are duplicates iff
/// every corresponding cell is equal, with NULLs treated as equal to
/// each other for dedup/grouping). Hoisted from `aggregate.rs`'s
/// former private `canonical_group_key` (DRY — one equality oracle).
///
/// Encoding notes:
/// - `Float` uses `to_bits` so the key is hashable (`f64: !Eq`); this
///   makes a canonical-bit NaN equal to itself for grouping (matching
///   Cypher's grouping treatment of NaN) and keeps `-0.0` / `0.0`
///   distinct (their bit patterns differ — a deliberate, documented
///   edge).
/// - `Null` collapses to a single key (NULLs are equal for dedup).
/// - `String` is length-prefixed (`S:<len>:<text>`) so user text
///   containing the cell separator OR the map framing delimiters
///   cannot blur a boundary (#735 R1; mirrors `join.rs`).
/// - `Map` (ADR-191 D-12) renders via a typed, recursive
///   `M{<klen>:k=<cell>;…}` form in `BTreeMap` sorted-key order, with
///   LENGTH-PREFIXED keys — equal maps key identically (so `{a:null}`
///   and `{a:null}` collapse to ONE group under EQUIVALENCE), distinct
///   maps never collide even under delimiter injection.
/// - Other composite / graph / temporal values fall back to their
///   `Debug` rendering, which is stable + self-delimiting within a
///   process run.
/// - A `\u{1f}` (ASCII unit separator) delimits top-level cells; the
///   per-cell length prefixes (above) make the key unambiguous even
///   when a user string itself contains `\u{1f}`.
#[must_use]
pub fn canonical_row_key(cells: &[crate::executor::value::Value]) -> String {
    let mut buf = String::new();
    for (i, c) in cells.iter().enumerate() {
        if i > 0 {
            buf.push('\u{1f}');
        }
        push_canonical_cell(&mut buf, c);
    }
    buf
}

/// Append the canonical key rendering of ONE cell to `buf`. Shared by
/// [`canonical_row_key`] and recursed into by the `Map` arm so a map's
/// values render with the SAME per-type encoding the top level uses.
/// Per ADR-191 D-12: keying is EQUIVALENCE (`null ≡ null`), distinct
/// from the `=`-operator 3VL equality — two `{a:null}` maps produce a
/// byte-identical key and group together.
fn push_canonical_cell(buf: &mut String, c: &crate::executor::value::Value) {
    use crate::executor::value::Value;
    match c {
        Value::Null => buf.push_str("NULL"),
        Value::Boolean(b) => buf.push_str(if *b { "T" } else { "F" }),
        Value::Integer(n) => {
            buf.push_str("I:");
            buf.push_str(&n.to_string());
        }
        Value::Float(f) => {
            buf.push_str("F:");
            buf.push_str(&f.to_bits().to_string());
        }
        // Length-prefixed (`S:<len>:<text>`, mirroring
        // `join.rs::append_value_fingerprint`) so user text containing
        // the cell separator (`\u{1f}`) or the map framing delimiters
        // (`=` / `;` / `}`) cannot forge a boundary — #735 R1.
        Value::String(s) => {
            buf.push_str("S:");
            buf.push_str(&s.len().to_string());
            buf.push(':');
            buf.push_str(s);
        }
        // ADR-191 D-12 — typed, recursive map rendering (NOT the `Debug`
        // catch-all). `M{<klen>:key=<cell>;…}` in `BTreeMap` sorted-key
        // order is deterministic, so DISTINCT / UNION dedup + GROUP BY
        // equivalence collapse equal maps and keep distinct maps apart.
        // Keys are LENGTH-PREFIXED (`<klen>:`, mirroring
        // `join.rs::append_value_fingerprint`) and the recursed `String`
        // value is length-prefixed too, so an injected `=` / `;` / `}`
        // in a key OR a string value cannot forge the framing (#735 R1):
        // `{a:"x",b:1}` and `{a:"x;b=I:1"}` MUST key differently. The
        // `M{`…`}` framing + per-type prefixes also keep a map key
        // disjoint from any scalar key.
        Value::Map(m) => {
            buf.push_str("M{");
            for (k, v) in m {
                buf.push_str(&k.len().to_string());
                buf.push(':');
                buf.push_str(k);
                buf.push('=');
                push_canonical_cell(buf, v);
                buf.push(';');
            }
            buf.push('}');
        }
        other => buf.push_str(&format!("{other:?}")),
    }
}

#[cfg(test)]
mod canonical_row_key_map_tests {
    use super::canonical_row_key;
    use crate::executor::value::Value;
    use std::collections::BTreeMap;

    fn vmap(entries: &[(&str, Value)]) -> Value {
        Value::Map(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn map_keys_deterministic_distinct_and_equivalence_collapses() {
        // ADR-191 D-12 — DISTINCT / UNION / GROUP BY share this single
        // oracle. Equal maps (ANY author order) key identically; distinct
        // maps key differently; and under EQUIVALENCE (`null ≡ null`) two
        // `{a:null}` maps collapse to ONE group — distinct from the
        // `=`-operator's 3VL null (D-3).
        let a = vmap(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        let a_reordered = vmap(&[("b", Value::Integer(2)), ("a", Value::Integer(1))]);
        let diff = vmap(&[("a", Value::Integer(1)), ("b", Value::Integer(3))]);
        assert_eq!(
            canonical_row_key(std::slice::from_ref(&a)),
            canonical_row_key(&[a_reordered]),
            "equal maps must key identically (order-independent)"
        );
        assert_ne!(
            canonical_row_key(&[a]),
            canonical_row_key(&[diff]),
            "distinct maps must key differently"
        );
        // EQUIVALENCE collision — one group.
        let n1 = vmap(&[("a", Value::Null)]);
        let n2 = vmap(&[("a", Value::Null)]);
        assert_eq!(
            canonical_row_key(std::slice::from_ref(&n1)),
            canonical_row_key(&[n2]),
            "{{a:null}} and {{a:null}} must collapse to ONE group (equivalence)"
        );
        // {a:null} stays distinct from {} and {a:1}.
        assert_ne!(
            canonical_row_key(std::slice::from_ref(&n1)),
            canonical_row_key(&[Value::Map(BTreeMap::new())])
        );
        assert_ne!(
            canonical_row_key(&[n1]),
            canonical_row_key(&[vmap(&[("a", Value::Integer(1))])])
        );
    }

    #[test]
    fn map_key_disjoint_from_scalar_keys() {
        // The `M{`/`=`/`;`/`}` framing + per-type prefixes keep a map key
        // from colliding with a string whose text mimics the encoding.
        let map_key = canonical_row_key(&[vmap(&[("a", Value::Integer(1))])]);
        let str_key = canonical_row_key(&[Value::String("M{a=I:1;}".into())]);
        assert_ne!(map_key, str_key);
        // Nested map vs flat map don't collide.
        let nested = vmap(&[("a", vmap(&[("b", Value::Integer(1))]))]);
        let flat = vmap(&[("a", Value::Integer(1))]);
        assert_ne!(canonical_row_key(&[nested]), canonical_row_key(&[flat]));
    }

    #[test]
    fn distinct_maps_never_collide_under_delimiter_injection() {
        // ADR-191 D-12 strong oracle (#735 R1) — "distinct maps never
        // collide" must hold even when a user injects the canonical-key
        // framing delimiters (`=` / `;` / `}`) into a STRING VALUE or a
        // KEY. The D-14 suite previously only checked map-vs-SCALAR
        // (`map_key_disjoint_from_scalar_keys`); it MISSED map-vs-map
        // delimiter injection. Before the length-prefix fix BOTH pairs
        // below produced byte-identical canonical keys → silently merged
        // into ONE DISTINCT / GROUP BY / UNION group (a silent-wrong-
        // answer correctness violation). This test FAILS on the pre-fix
        // code (both `assert_ne!`s trip) — strong-oracle discipline: it
        // can fail on its bug.

        // Value-side injection: `{a:"x", b:1}` vs `{a:"x;b=I:1"}`.
        // Pre-fix BOTH rendered `M{a=S:x;b=I:1;}` → COLLISION.
        let v_two = vmap(&[("a", Value::String("x".into())), ("b", Value::Integer(1))]);
        let v_inject = vmap(&[("a", Value::String("x;b=I:1".into()))]);
        assert_ne!(
            canonical_row_key(std::slice::from_ref(&v_two)),
            canonical_row_key(std::slice::from_ref(&v_inject)),
            "value-side delimiter injection must NOT collide: \
             {{a:\"x\",b:1}} vs {{a:\"x;b=I:1\"}}"
        );

        // Key-side injection: `{a:1, b:2}` vs `{\"a=I:1;b\": 2}`.
        // Pre-fix BOTH rendered `M{a=I:1;b=I:2;}` → COLLISION.
        let k_two = vmap(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        let k_inject = vmap(&[("a=I:1;b", Value::Integer(2))]);
        assert_ne!(
            canonical_row_key(std::slice::from_ref(&k_two)),
            canonical_row_key(std::slice::from_ref(&k_inject)),
            "key-side delimiter injection must NOT collide: \
             {{a:1,b:2}} vs {{\"a=I:1;b\":2}}"
        );

        // EQUIVALENCE preserved — the fix changes ENCODING (injection
        // safety), NOT equivalence. Structurally-identical maps still
        // collapse to ONE group (order-independent), incl. the `{a:null}`
        // D-3 equivalence case.
        let e1 = vmap(&[("a", Value::String("x".into())), ("b", Value::Integer(1))]);
        let e2 = vmap(&[("b", Value::Integer(1)), ("a", Value::String("x".into()))]);
        assert_eq!(
            canonical_row_key(std::slice::from_ref(&e1)),
            canonical_row_key(std::slice::from_ref(&e2)),
            "structurally-identical maps must collapse to ONE group"
        );
        let null1 = vmap(&[("a", Value::Null)]);
        let null2 = vmap(&[("a", Value::Null)]);
        assert_eq!(
            canonical_row_key(std::slice::from_ref(&null1)),
            canonical_row_key(std::slice::from_ref(&null2)),
            "{{a:null}} and {{a:null}} must still collapse to ONE group (equivalence)"
        );
    }
}
