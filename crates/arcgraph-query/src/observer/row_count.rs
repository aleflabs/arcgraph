//! `RowCountObserver` — per-operator row-count + wall-time + memory
//! observer for the M4-61/62 vectorized executor.
//!
//! Per ADR-038 amendment-02 §M4.g + amendment-03 §TIER-2-c.
//!
//! # Observation model
//!
//! The observer is registered on [`crate::executor::ExecutionContext`]
//! via [`crate::executor::ExecutionContext::with_observer`]. The
//! executor's dispatcher (`crate::executor::ops::PhysicalOperator::next_batch`)
//! calls [`crate::observer::dispatcher::record_dispatch`] after every
//! per-operator `next_batch` invocation; the dispatcher's hook delegates
//! here to [`RowCountObserver::record_batch`], which accumulates:
//!
//! - **Row count** — observed rows per [`OperatorKind`] (`Scan`,
//!   `Expand`, `Filter`, `Project`, ...). Aggregation is by operator
//!   KIND across every operator instance of that kind in the plan tree.
//!   Per-instance attribution is forward-deferred (M4-72 + future); the
//!   v1.0-alpha model is per-kind aggregate, which is sufficient for
//!   per-label / per-rel-type catalog feedback (a single MATCH pattern
//!   has one Scan + zero-or-one Expand at v1.0).
//! - **Wall-time** — accumulated nanoseconds spent inside `next_batch`
//!   per kind (excluding cancellation-token check overhead, which is
//!   pre-dispatcher-hook).
//! - **Memory high-water** — peak per-batch memory estimate per kind,
//!   computed from `batch.row_count() × batch.column_count() ×
//!   sizeof(Value)` (32-byte conservative pin per the row-major Value
//!   layout in `executor::value::Value`).
//!
//! # Plan-walk anchor (estimated cardinalities)
//!
//! Per amendment-02 §M4.g, the 10× threshold compares observed to
//! estimated cardinality. The estimated cardinality comes from the
//! M4-51 cost walker's [`crate::planner::cost::CostedTree`]. The
//! observer is constructed via [`RowCountObserver::from_plan_and_costs`]
//! which walks the plan + cost tree in lockstep (pre-order) and pins
//! the per-kind estimated-cardinality SUM at construction time. This
//! sum is the denominator the threshold check uses; per-kind aggregation
//! sidesteps the per-instance attribution problem cited above.
//!
//! For the per-label / per-rel-type observed-stats feedback (M4-04
//! channel per amendment-03 §"Implicit dependency edges" item 4), the
//! plan walk also captures each Scan's `Option<LabelId>` and each
//! Expand's `Option<TypeId>`. The observer apportions observed rows
//! to labels / rel-types proportionally to estimated-card weights —
//! this is best-effort attribution at v1.0; future M4-72 introduces
//! per-instance position-tagged attribution.
//!
//! # Concurrency
//!
//! Observer is `Send + Sync` (held behind `Arc<RowCountObserver>` on
//! the `ExecutionContext`). Internal state is guarded by `parking_lot::Mutex`
//! per the same poisoning-free discipline as `crate::planner::PlanCache`
//! — observer state is soft (diagnostic) and a panic during one record
//! must not taint subsequent records.
//!
//! # Tracing
//!
//! Per amendment-03 §TIER-2-c, every threshold breach emits a
//! `tracing::warn!` event with structured fields. Per-record updates
//! emit `tracing::trace!` events at `target =
//! "arcgraph_query::observer::row_count"`.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.g — primary M4-71 cite.
//! - ADR-038 amendment-03 §TIER-2-c — observability / per-query metrics.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 4 — M4-04
//!   feedback channel.

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{LabelId, TypeId};
use parking_lot::Mutex;

use crate::executor::batch::Batch;
use crate::explain::ExecutionMetrics;
use crate::logical_plan::LogicalPlan;
use crate::observer::feedback::ObservedStatsOverrides;
use crate::observer::threshold::ThresholdBreach;
use crate::planner::cost::CostedTree;

/// 10× threshold factor per ADR-038 amendment-02 §M4.g.
///
/// Configurable per-observer via [`RowCountObserver::with_threshold_factor`].
/// The 10× literal is empirically chosen — large enough that random
/// selectivity drift doesn't trigger, small enough to catch genuine
/// catalog-stat staleness (typical post-bulk-load drift is 10⁴–10⁶× per
/// the W11Z M4-04d empirical fixtures).
pub const DEFAULT_THRESHOLD_FACTOR: f64 = 10.0;

/// Per-Value size pin for the per-batch memory estimate.
///
/// 32 bytes is a conservative cap on the actual `Value` enum size
/// (`Value::Node(NodeView)` is the largest at 24 bytes payload + 8 byte
/// discriminant tag on most architectures; smaller variants fit). The
/// estimate is high-water-pinned so memory_bytes_high_water never
/// under-reports.
const VALUE_SIZE_BYTES: u64 = 32;

/// Operator kind. Stable mirror of
/// [`crate::executor::PhysicalOperator`]'s variants — additive sync via
/// `crate::executor::ops::PhysicalOperator::op_kind`.
///
/// # Why a separate enum (not re-using `PhysicalOperator` directly)
///
/// `PhysicalOperator` carries per-instance state; this enum is
/// payload-free for use as a HashMap key. The two stay in sync via the
/// `op_kind()` method on `PhysicalOperator`.
///
/// # Why exempt from `#[non_exhaustive]`
///
/// Under the code-quality policy exemption rule, the variant set IS the public
/// contract for downstream pattern-matching consumption. Adding a new
/// kind requires synchronized changes in (a) `PhysicalOperator::op_kind`
/// dispatch, (b) replan logic in [`crate::observer::ReplanController`],
/// (c) every `match` site in tracing-field rendering. The exhaustive
/// match guarantees compile-time coverage of these consumers — if a
/// future operator variant lands without updating the observer, the
/// compile fails loudly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Scan,
    Singleton,
    Expand,
    Filter,
    Project,
    Empty,
    RankByHybrid,
    Fusion,
    OptionalExpand,
    /// W12α follow-up — PR #277 lit `PhysicalOperator::Aggregate` after
    /// the OperatorKind enum's introduction in #278; pinning the
    /// 1-to-1 PhysicalOperator↔OperatorKind correspondence here.
    Aggregate,
    /// W12α follow-up — see Aggregate.
    Sort,
    /// W12α follow-up — see Aggregate. The variant is named after the
    /// physical operator (`NamedShortestPathOp`) rather than the
    /// logical (`NamedPath`) for parity with the rest of the enum.
    NamedShortestPath,
    /// W12α follow-up — see Aggregate.
    Limit,
    /// **#842 part A.** `SKIP N` offset operator
    /// ([`crate::executor::ops::SkipOp`]) — lit by this slice (closes the
    /// prior `LogicalPlan::Skip` `NotImplemented`). Promoted out of the
    /// `Empty` attribution bucket because the executor now genuinely
    /// dispatches it (per the module-level no-catch-all contract: a lit
    /// op carries its own kind to avoid Empty-bucket contamination).
    Skip,
    /// W17α / M4-08+ — `LogicalJoin` executor lit
    /// ([`crate::executor::ops::join::HashJoinOp`]). Surfaces multi-
    /// pattern equi-join + Cartesian shapes per ADR-038 §2 D-24.
    HashJoin,
    /// W25-M4-61b / ADR-097 — sort-merge join variant
    /// ([`crate::executor::ops::merge_join::MergeJoinOp`]). Picked by
    /// the cost-based planner ([`crate::planner::pick_join_algorithms`])
    /// when the merge cost (sort(L) + sort(R) + merge) is below the
    /// hash-join cost. Cartesian always routes to `HashJoin`.
    MergeJoin,
    /// **ADR-147 W26-θ Phase 1.** Write-op operator — CREATE node.
    CreateNode,
    /// **#830 / ADR-200.** Accept-and-register write-op — CREATE VECTOR
    /// INDEX (metadata-only catalog registration; no build).
    CreateVectorIndex,
    /// **#1366 (task #248, Phase 1).** Write-op — CREATE INDEX
    /// (property index: register + backfill + Online flip).
    CreatePropertyIndex,
    /// **ADR-148 W26-θ Phase 2.** Write-op operator — CREATE rel.
    CreateRel,
    /// **ADR-149 W26-θ Phase 3.** Write-op operator — DELETE
    /// (covers both bare DELETE and DETACH DELETE; the substrate
    /// dispatch handles the node-vs-rel + detach-vs-bare distinction).
    Delete,
    /// **ADR-150 W26-θ Phase 4.** Write-op operator — SET (covers
    /// all four set_item shapes; the substrate dispatch handles the
    /// node-vs-rel + property-vs-label distinction).
    Set,
    /// **ADR-150 W26-θ Phase 4.** Write-op operator — REMOVE (covers
    /// both remove_item shapes).
    Remove,
    /// **ADR-151 W26-θ Phase 5.** Write-op operator — MERGE (covers
    /// the match-or-create pattern with optional ON CREATE SET /
    /// ON MATCH SET action clauses).
    Merge,
    /// **ADR-185 (#649-A1, W28).** Row-dedup operator
    /// ([`crate::executor::ops::DistinctOp`]) — lit by this slice
    /// (closes the prior `RETURN DISTINCT` `NotImplemented`). Promoted
    /// out of the `Empty` attribution bucket because `record_batch`
    /// now genuinely fires for it.
    Distinct,
    /// **ADR-185 (#649-A1, W28).** UNION ALL concat operator
    /// ([`crate::executor::ops::UnionOp`]).
    Union,
    /// **ADR-038 D-28 §7 (#618).** `UNWIND <list> AS <var>` operator
    /// ([`crate::executor::ops::UnwindOp`]) — lit by this slice (closes
    /// the prior `LogicalPlan::Unwind` `NotImplemented`). Promoted out
    /// of the `Empty` attribution bucket because `record_batch` now
    /// genuinely fires for it.
    Unwind,
    /// **ADR-192 (#623).** `CALL { <subquery> }` correlated subquery
    /// operator ([`crate::executor::ops::CallOp`]) — Cypher 25, beyond
    /// openCypher v9.
    Call,
    /// **ADR-192 (#623).** The one-row correlation seed feeding a
    /// `CALL { … }` body ([`crate::executor::ops::CorrelationSeedOp`]).
    CorrelationSeed,
    /// **ADR-197 (#802).** `CALL <proc>(…) [YIELD …]` / `SHOW …`
    /// schema-introspection generating operator
    /// ([`crate::executor::ops::ProcedureCallOp`]).
    ProcedureCall,
}

impl OperatorKind {
    /// Stable string slug for tracing-field emission.
    #[inline]
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Singleton => "singleton",
            Self::Expand => "expand",
            Self::Filter => "filter",
            Self::Project => "project",
            Self::Empty => "empty",
            Self::RankByHybrid => "rank_by_hybrid",
            Self::Fusion => "fusion",
            Self::OptionalExpand => "optional_expand",
            Self::Aggregate => "aggregate",
            Self::Sort => "sort",
            Self::NamedShortestPath => "named_shortest_path",
            Self::Limit => "limit",
            Self::Skip => "skip",
            Self::HashJoin => "hash_join",
            Self::MergeJoin => "merge_join",
            Self::CreateNode => "create_node",
            Self::CreateVectorIndex => "create_vector_index",
            Self::CreatePropertyIndex => "create_property_index",
            Self::CreateRel => "create_rel",
            Self::Delete => "delete",
            Self::Set => "set",
            Self::Remove => "remove",
            Self::Merge => "merge",
            Self::Distinct => "distinct",
            Self::Union => "union",
            Self::Unwind => "unwind",
            Self::Call => "call",
            Self::CorrelationSeed => "correlation_seed",
            Self::ProcedureCall => "procedure_call",
        }
    }

    /// All variants in canonical order (used by tests + for-each
    /// rendering of the observer's full per-kind state).
    pub const ALL: [Self; 29] = [
        Self::Scan,
        Self::Singleton,
        Self::Expand,
        Self::Filter,
        Self::Project,
        Self::Empty,
        Self::RankByHybrid,
        Self::Fusion,
        Self::OptionalExpand,
        Self::Aggregate,
        Self::Sort,
        Self::NamedShortestPath,
        Self::Limit,
        Self::Skip,
        Self::HashJoin,
        Self::MergeJoin,
        Self::CreateNode,
        Self::CreateVectorIndex,
        Self::CreatePropertyIndex,
        Self::CreateRel,
        Self::Delete,
        Self::Set,
        Self::Remove,
        Self::Merge,
        Self::Distinct,
        Self::Union,
        Self::Unwind,
        Self::Call,
        Self::CorrelationSeed,
    ];
}

/// Per-kind aggregated metrics extracted from the observer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OperatorMetrics {
    /// The operator kind these metrics aggregate over.
    pub op_kind: Option<OperatorKind>,
    /// Total observed rows across all operator instances of this kind.
    pub observed_rows: u64,
    /// Sum of estimated cardinalities across all operator instances of
    /// this kind in the plan (from the cost tree at observer
    /// construction).
    pub estimated_card: f64,
    /// Total wall-time nanoseconds across all next_batch calls of this
    /// kind.
    pub wall_time_ns: u64,
    /// Convenience getter — wall_time_ns / 1_000_000 (with rounding-up
    /// for sub-millisecond accumulations).
    pub wall_time_ms: u64,
    /// High-water memory bytes observed at any single batch of this kind.
    pub memory_bytes_high_water: u64,
    /// Total batches recorded for this kind.
    pub batches: u64,
}

/// One pre-order entry from the plan-walk anchor.
///
/// Constructed by [`walk_plan_and_costs`]; consumed by
/// [`RowCountObserver::from_plan_and_costs`] to populate per-kind
/// estimated-card sums + per-Scan label / per-Expand rel-type
/// attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanWalkEntry {
    /// Operator kind for this plan node.
    pub op_kind: OperatorKind,
    /// Estimated output cardinality (from the cost tree).
    pub estimated_card: f64,
    /// Scan label, if this is a Scan.
    pub scan_label: Option<LabelId>,
    /// Expand rel-type, if this is an Expand.
    pub expand_rel_type: Option<TypeId>,
}

/// Walk a [`LogicalPlan`] + parallel [`CostedTree`] in lockstep,
/// producing one [`PlanWalkEntry`] per operator in pre-order.
///
/// Used by [`RowCountObserver::from_plan_and_costs`] at observer
/// construction time AND by [`crate::observer::ReplanController`] for
/// post-replan plan-shape comparison.
///
/// # Pre-order discipline
///
/// The walker emits parents before children so the entry sequence
/// mirrors the executor's dispatch order under linear plans (Project →
/// Filter → Scan). Non-linear plans (Fusion / OptionalExpand) emit the
/// fan-out parent first, then each child in source-declared order.
#[must_use]
pub fn walk_plan_and_costs(plan: &LogicalPlan, costs: &CostedTree) -> Vec<PlanWalkEntry> {
    let mut entries = Vec::new();
    walk_inner(plan, costs, &mut entries);
    entries
}

fn walk_inner(plan: &LogicalPlan, costs: &CostedTree, out: &mut Vec<PlanWalkEntry>) {
    let estimated_card = costs.cost.output_card.rows();
    // Exhaustive match — per W12β fix-up LOW-3, no catch-all `_ =>`.
    // The executor's `op_kind()` (in `executor/ops/mod.rs`) is also
    // exhaustive against `PhysicalOperator`; both must be updated in
    // lockstep when LogicalPlan or PhysicalOperator gain variants. The
    // exhaustive walker match here forces compile failure when a new
    // LogicalPlan variant lands without a corresponding observer arm,
    // closing the silent-corruption gap the catch-all previously left
    // open: an unrecognized LogicalPlan variant now surfaces as a
    // compile error in this match, NOT as silent contamination of the
    // `Empty` bucket's per-kind accounting.
    //
    // # Forward-deferred operators (M4-63 / M4-33 / M4-08+)
    //
    // Variants without a dedicated `OperatorKind` (Aggregate, Sort,
    // Limit, Skip, DynamicLimit, Distinct, Unwind, NamedPath, Join,
    // CommunityLookup, VectorNear, TextMatch) currently map to
    // `OperatorKind::Empty` because the v1.0-alpha executor surfaces
    // `NotImplemented` before reaching them — no `record_batch` ever
    // fires for these kinds, so the `Empty` mapping is harmless at the
    // current wave. When PR #277 (W12α) merges and lights
    // `PhysicalOperator::{Aggregate, Sort, Limit}`, this match MUST be
    // updated alongside extending the `OperatorKind` enum, the `ALL`
    // table, and `as_str()`. The exhaustive match ensures the compile
    // breaks when LogicalPlan variants are added without these
    // synchronized changes.
    let (op_kind, scan_label, expand_rel_type) = match plan {
        LogicalPlan::Scan(s) => (OperatorKind::Scan, s.label, None),
        // #1366 (Phase 2): a node SOURCE, same observer class as `Scan`;
        // it carries the (always-present) index label for attribution.
        LogicalPlan::PropertyIndexScan(p) => (OperatorKind::Scan, Some(p.label), None),
        LogicalPlan::CountStore(_) => (OperatorKind::Aggregate, None, None),
        LogicalPlan::Expand(e) => (OperatorKind::Expand, None, e.rel_type),
        LogicalPlan::Filter(_) => (OperatorKind::Filter, None, None),
        LogicalPlan::Project(_) => (OperatorKind::Project, None, None),
        LogicalPlan::Empty(_) => (OperatorKind::Empty, None, None),
        LogicalPlan::RankByHybrid(_) => (OperatorKind::RankByHybrid, None, None),
        LogicalPlan::Fusion(_) => (OperatorKind::Fusion, None, None),
        LogicalPlan::LeftOuterJoin(_) => (OperatorKind::OptionalExpand, None, None),
        // W12α follow-up — PR #277 lit `PhysicalOperator::{Aggregate,
        // Sort, Limit, NamedShortestPath}`; OperatorKind dedicated
        // variants land in lockstep here so PROFILE / observer
        // attribution carries the right slug.
        LogicalPlan::Aggregate(_) => (OperatorKind::Aggregate, None, None),
        LogicalPlan::Sort(_) => (OperatorKind::Sort, None, None),
        LogicalPlan::Limit(_) => (OperatorKind::Limit, None, None),
        // #842 part A: SKIP is now lit at the executor (SkipOp), so it
        // gets dedicated OperatorKind attribution rather than the Empty
        // bucket — the executor genuinely dispatches it (same promotion
        // the Distinct / Union / Unwind lit-op slices made).
        LogicalPlan::Skip(_) => (OperatorKind::Skip, None, None),
        LogicalPlan::NamedPath(_) => (OperatorKind::NamedShortestPath, None, None),
        // W17α / M4-08+: `LogicalJoin` is now lit at the executor.
        // W25-M4-61b / ADR-097 adds the sort-merge variant; the
        // observer attributes per-batch row counts to the dedicated
        // kind that matches the resolved algorithm. `Auto` (= picker
        // has not yet run) defaults to `HashJoin` for parity with
        // pipeline build's defensive fallback (see
        // `crate::executor::pipeline::Pipeline::build_with_parameters`).
        LogicalPlan::Join(j) => match j.algorithm {
            crate::logical_plan::JoinAlgorithm::MergeJoin => (OperatorKind::MergeJoin, None, None),
            crate::logical_plan::JoinAlgorithm::HashJoin
            | crate::logical_plan::JoinAlgorithm::Auto => (OperatorKind::HashJoin, None, None),
        },
        // Forward-deferred — see module-level rustdoc above. These
        // variants exist in EXPLAIN plan trees but the v1.0-alpha
        // executor never dispatches them (NotImplemented gate at the
        // pipeline build). Mapped to Empty for diagnostic continuity;
        // future M4-33 / M4-08+ slices add dedicated OperatorKind
        // variants in lockstep.
        LogicalPlan::DynamicLimit(_)
        | LogicalPlan::CommunityLookup(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_) => (OperatorKind::Empty, None, None),
        // ADR-038 D-28 §7 (#618): UNWIND is now lit at the executor
        // (UnwindOp), so it gets dedicated OperatorKind attribution
        // rather than the Empty bucket — `record_batch` genuinely fires.
        LogicalPlan::Unwind(_) => (OperatorKind::Unwind, None, None),
        LogicalPlan::ProcedureCall(_) => (OperatorKind::ProcedureCall, None, None),
        // ADR-185 (#649-A1, W28): Distinct + Union are now lit at the
        // executor (DistinctOp / UnionOp), so they get dedicated
        // OperatorKind attribution rather than the Empty bucket — the
        // executor genuinely fires `record_batch` for them, and the
        // module-level no-catch-all contract requires lit ops carry
        // their own kind to avoid Empty-bucket contamination.
        LogicalPlan::Distinct(_) => (OperatorKind::Distinct, None, None),
        LogicalPlan::Union(_) => (OperatorKind::Union, None, None),
        // ADR-147 W26-θ Phase 1: write-op operator surfaces the
        // dedicated CreateNode kind.
        LogicalPlan::CreateNode(_) => (OperatorKind::CreateNode, None, None),
        // #830 / ADR-200: CREATE VECTOR INDEX accept-and-register write
        // op surfaces its dedicated kind (emits 0 rows).
        LogicalPlan::CreateVectorIndex(_) => (OperatorKind::CreateVectorIndex, None, None),
        // #1366 (task #248): CREATE INDEX (property index) surfaces its
        // dedicated kind (emits 0 rows).
        LogicalPlan::CreatePropertyIndex(_) => (OperatorKind::CreatePropertyIndex, None, None),
        // ADR-148 W26-θ Phase 2: CreateRel mirrors the CreateNode
        // attribution shape.
        LogicalPlan::CreateRel(_) => (OperatorKind::CreateRel, None, None),
        // ADR-149 W26-θ Phase 3: Delete is the dedicated DELETE
        // op surfaced through OperatorKind::Delete.
        LogicalPlan::Delete(_) => (OperatorKind::Delete, None, None),
        // ADR-150 W26-θ Phase 4: Set / Remove are the dedicated SET /
        // REMOVE ops surfaced through OperatorKind::{Set, Remove}.
        LogicalPlan::Set(_) => (OperatorKind::Set, None, None),
        LogicalPlan::Remove(_) => (OperatorKind::Remove, None, None),
        // ADR-151 W26-θ Phase 5: Merge is the dedicated MERGE op.
        LogicalPlan::Merge(_) => (OperatorKind::Merge, None, None),
        // ADR-192 (#623): CALL{} correlated subquery + its body seed.
        LogicalPlan::Call(_) => (OperatorKind::Call, None, None),
        LogicalPlan::CorrelationSeed(_) => (OperatorKind::CorrelationSeed, None, None),
    };
    out.push(PlanWalkEntry {
        op_kind,
        estimated_card,
        scan_label,
        expand_rel_type,
    });
    // Recurse into children. The plan's child set + cost tree's children
    // must be in lockstep (post-order isomorphism), which the M4-51 cost
    // walker guarantees.
    let plan_children = plan_children(plan);
    let cost_children = &costs.children;
    for (child_plan, child_cost) in plan_children.iter().zip(cost_children.iter()) {
        walk_inner(child_plan, child_cost, out);
    }
}

fn plan_children(plan: &LogicalPlan) -> Vec<&LogicalPlan> {
    match plan {
        LogicalPlan::Scan(_)
        // #1366 (Phase 2): the indexed point-lookup is a LEAF (no
        // children) — the anchor scan it replaces is gone.
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_) => Vec::new(),
        LogicalPlan::Expand(_) => Vec::new(),
        LogicalPlan::Filter(f) => vec![&f.input],
        LogicalPlan::Project(p) => vec![&p.input],
        LogicalPlan::Join(j) => vec![&j.left, &j.right],
        LogicalPlan::LeftOuterJoin(j) => vec![&j.left, &j.right],
        LogicalPlan::Limit(l) => vec![&l.input],
        LogicalPlan::Skip(s) => vec![&s.input],
        LogicalPlan::DynamicLimit(d) => vec![&d.input],
        LogicalPlan::Fusion(f) => f.inputs.iter().map(|b| b.as_ref()).collect(),
        // ADR-185 (#649-A1, W28): UNION ALL children are the arms.
        LogicalPlan::Union(u) => u.arms.iter().collect(),
        LogicalPlan::RankByHybrid(_) => Vec::new(),
        LogicalPlan::CommunityLookup(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_) => Vec::new(),
        LogicalPlan::Aggregate(a) => vec![&a.input],
        LogicalPlan::Sort(s) => vec![&s.input],
        LogicalPlan::Distinct(d) => vec![&d.input],
        LogicalPlan::Unwind(u) => vec![&u.input],
        LogicalPlan::ProcedureCall(p) => vec![&p.input],
        LogicalPlan::NamedPath(_) => Vec::new(),
        // #832: CreateNode is a LEAF only at the chain bottom; a multi-
        // item `CREATE (a),(b),(c)` lowers to a left-deep chain via the
        // optional `input` child. Mirror the cost + EXPLAIN plan-tree
        // walkers (input at child index 0) so the row-count attribution
        // `zip` against the cost tree stays in LOCKSTEP — else it stops
        // short and the chained creates' rows are silently dropped
        // (mis-attributed) on multi-CREATE PROFILE.
        LogicalPlan::CreateNode(c) => c.input.as_deref().map(|i| vec![i]).unwrap_or_default(),
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf DDL — zero
        // children (no input child). LEAF in lockstep with the cost +
        // EXPLAIN plan-tree walkers so the row-count `zip` stays aligned.
        LogicalPlan::CreateVectorIndex(_) => Vec::new(),
        // #1366 (task #248): CREATE INDEX (property index) is a leaf DDL
        // — zero children (backfill is internal, not a plan child).
        LogicalPlan::CreatePropertyIndex(_) => Vec::new(),
        // ADR-148 W26-θ Phase 2: CreateRel walks source + target sub-
        // plans (each typically a CreateNode); #832 appends the optional
        // chain `input` as the THIRD child (after source + target),
        // mirroring the executor pipeline + cost + EXPLAIN plan-tree
        // walkers so the lockstep `zip` covers the whole chain.
        LogicalPlan::CreateRel(c) => {
            let mut kids: Vec<&LogicalPlan> = vec![&c.source_plan, &c.target_plan];
            if let Some(input) = c.input.as_deref() {
                kids.push(input);
            }
            kids
        }
        // ADR-150 W26-θ Phase 4: Set / Remove walk the single input
        // sub-plan (typically the prior MATCH's lowered plan).
        LogicalPlan::Set(s) => vec![&s.input],
        LogicalPlan::Remove(r) => vec![&r.input],
        // ADR-149 W26-θ Phase 3: Delete walks the single input
        // sub-plan (typically the prior MATCH's lowered plan).
        LogicalPlan::Delete(d) => vec![&d.input],
        // ADR-151 W26-θ Phase 5: Merge walks BOTH the match-branch
        // and create-branch sub-plans (in that source order — match
        // probe runs first; create fires on probe miss).
        LogicalPlan::Merge(m) => vec![&m.match_branch, &m.create_branch],
        // ADR-192 (#623): CALL{} walks the driving `input` then the
        // subquery `body` (in that source order — the body runs per
        // input row). The cost walker mirrors this child order
        // (lockstep). CorrelationSeed is a LEAF.
        LogicalPlan::Call(c) => vec![&c.input, &c.body],
        LogicalPlan::CorrelationSeed(_) => Vec::new(),
    }
}

/// Per-operator row-count + wall-time + memory observer.
///
/// `Send + Sync` via internal `Mutex<ObserverState>`. Cheap to clone —
/// shared via `Arc<RowCountObserver>` between the
/// [`crate::executor::ExecutionContext`] (writer) and the post-execute
/// reader (test / [`crate::observer::ReplanController`]).
pub struct RowCountObserver {
    state: Mutex<ObserverState>,
    threshold_factor: f64,
}

impl std::fmt::Debug for RowCountObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("RowCountObserver")
            .field("threshold_factor", &self.threshold_factor)
            .field("kinds_seen", &state.per_kind.len())
            .field("walk_entries", &state.plan_walk.len())
            .finish()
    }
}

#[derive(Debug, Default)]
struct ObserverState {
    /// Per-kind observed-rows + wall-time + memory accumulators.
    per_kind: HashMap<OperatorKind, PerKindState>,
    /// Pre-built plan walk anchoring estimated cardinalities.
    plan_walk: Vec<PlanWalkEntry>,
    /// Per-kind sum of estimated cardinalities (cached from plan_walk
    /// at construction; immutable after).
    estimated_per_kind: HashMap<OperatorKind, f64>,
}

#[derive(Debug, Default, Clone)]
struct PerKindState {
    observed_rows: u64,
    wall_time_ns: u64,
    memory_bytes_high_water: u64,
    batches: u64,
}

impl RowCountObserver {
    /// Construct an empty observer with no plan-walk anchor.
    ///
    /// Useful for tests that drive the observer directly without an
    /// EXPLAIN pass first. Threshold detection against an empty
    /// plan-walk produces no breaches (every kind's estimated_per_kind
    /// is zero; the under-estimate branch fires on observed > 0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ObserverState::default()),
            threshold_factor: DEFAULT_THRESHOLD_FACTOR,
        }
    }

    /// Construct an observer anchored to a [`LogicalPlan`] + parallel
    /// [`CostedTree`]. Pre-walks the plan to populate per-kind estimated-
    /// cardinality sums + per-Scan labels.
    #[must_use]
    pub fn from_plan_and_costs(plan: &LogicalPlan, costs: &CostedTree) -> Self {
        let walk = walk_plan_and_costs(plan, costs);
        let mut estimated_per_kind: HashMap<OperatorKind, f64> = HashMap::new();
        for entry in &walk {
            *estimated_per_kind.entry(entry.op_kind).or_default() += entry.estimated_card;
        }
        Self {
            state: Mutex::new(ObserverState {
                per_kind: HashMap::new(),
                plan_walk: walk,
                estimated_per_kind,
            }),
            threshold_factor: DEFAULT_THRESHOLD_FACTOR,
        }
    }

    /// Override the 10× threshold factor. Tests use a smaller factor
    /// (e.g., 2.0) to exercise the breach path with smaller fixtures;
    /// production callers stay at [`DEFAULT_THRESHOLD_FACTOR`].
    #[must_use]
    pub fn with_threshold_factor(mut self, factor: f64) -> Self {
        debug_assert!(factor > 1.0, "threshold factor must be strictly > 1.0");
        self.threshold_factor = factor;
        self
    }

    /// Wrap in `Arc` for sharing with the
    /// [`crate::executor::ExecutionContext`].
    #[must_use]
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Read the configured threshold factor.
    #[inline]
    #[must_use]
    pub fn threshold_factor(&self) -> f64 {
        self.threshold_factor
    }

    /// Record a single batch event for one operator kind.
    ///
    /// Called by the dispatcher hook in
    /// [`crate::observer::dispatcher::record_dispatch`] AFTER each
    /// per-operator `next_batch` invocation. Accumulates rows, wall-
    /// time, and high-water memory.
    ///
    /// # Tracing
    ///
    /// Emits a `tracing::trace!` event at
    /// `target = "arcgraph_query::observer::row_count"` per recorded batch
    /// (gated by trace level so production deployments at `info`
    /// level pay zero serialization cost).
    pub fn record_batch(
        &self,
        op_kind: OperatorKind,
        rows: u64,
        wall_time_ns: u64,
        memory_bytes: u64,
    ) {
        let mut state = self.state.lock();
        let slot = state.per_kind.entry(op_kind).or_default();
        slot.observed_rows = slot.observed_rows.saturating_add(rows);
        slot.wall_time_ns = slot.wall_time_ns.saturating_add(wall_time_ns);
        slot.memory_bytes_high_water = slot.memory_bytes_high_water.max(memory_bytes);
        slot.batches = slot.batches.saturating_add(1);
        let observed_rows = slot.observed_rows;
        let batches = slot.batches;
        drop(state);
        tracing::trace!(
            target: "arcgraph_query::observer::row_count",
            op_kind = op_kind.as_str(),
            rows,
            wall_time_ns,
            memory_bytes,
            observed_total = observed_rows,
            batches,
            "row_count_observed",
        );
    }

    /// Convenience: record a batch via a [`Batch`] reference; computes
    /// the high-water memory estimate from `row_count × column_count ×
    /// VALUE_SIZE_BYTES`.
    pub fn record_dispatched_batch(&self, op_kind: OperatorKind, batch: &Batch, wall_time_ns: u64) {
        let mem = (batch.row_count() as u64)
            .saturating_mul(batch.column_count() as u64)
            .saturating_mul(VALUE_SIZE_BYTES);
        self.record_batch(op_kind, batch.row_count() as u64, wall_time_ns, mem);
    }

    /// Snapshot per-kind metrics. Returns one [`OperatorMetrics`] per
    /// kind seen at least once OR per kind present in the plan walk
    /// (zero-observed kinds report `observed_rows=0` for the consumer's
    /// "every plan operator was seen" check).
    #[must_use]
    pub fn metrics(&self) -> Vec<OperatorMetrics> {
        let state = self.state.lock();
        let mut keys: Vec<OperatorKind> = state.per_kind.keys().copied().collect();
        for kind in state.estimated_per_kind.keys() {
            if !keys.contains(kind) {
                keys.push(*kind);
            }
        }
        // Stable order for downstream rendering: by `OperatorKind::ALL`.
        keys.sort_by_key(|k| {
            OperatorKind::ALL
                .iter()
                .position(|o| o == k)
                .unwrap_or(usize::MAX)
        });
        keys.iter()
            .map(|k| {
                let s = state.per_kind.get(k).cloned().unwrap_or_default();
                let estimated = state.estimated_per_kind.get(k).copied().unwrap_or(0.0);
                OperatorMetrics {
                    op_kind: Some(*k),
                    observed_rows: s.observed_rows,
                    estimated_card: estimated,
                    wall_time_ns: s.wall_time_ns,
                    wall_time_ms: ns_to_ms_ceil(s.wall_time_ns),
                    memory_bytes_high_water: s.memory_bytes_high_water,
                    batches: s.batches,
                }
            })
            .collect()
    }

    /// Detect 10× threshold breaches by comparing per-kind observed
    /// sums to per-kind estimated sums.
    ///
    /// Returns one [`ThresholdBreach`] per kind whose ratio crosses the
    /// configured factor in either direction.
    ///
    /// # Tracing
    ///
    /// Each detected breach emits a `tracing::warn!` event with
    /// structured fields per amendment-03 §TIER-2-c.
    #[must_use]
    pub fn threshold_breaches(&self) -> Vec<ThresholdBreach> {
        let state = self.state.lock();
        let mut breaches = Vec::new();
        for (kind, est) in &state.estimated_per_kind {
            let observed = state
                .per_kind
                .get(kind)
                .map(|s| s.observed_rows)
                .unwrap_or(0);
            if let Some(b) = self.compute_breach(*kind, *est, observed) {
                breaches.push(b);
            }
        }
        // Stable order: by OperatorKind::ALL.
        breaches.sort_by_key(|b| {
            OperatorKind::ALL
                .iter()
                .position(|o| o == &b.op_kind)
                .unwrap_or(usize::MAX)
        });
        drop(state);
        for b in &breaches {
            tracing::warn!(
                target: "arcgraph_query::observer::row_count",
                op_kind = b.op_kind.as_str(),
                direction = b.direction.as_str(),
                estimated_card = b.estimated_card_sum,
                observed_rows = b.observed_rows_sum,
                ratio = b.ratio,
                "threshold_breach",
            );
        }
        breaches
    }

    /// Compute a single breach if the observed/estimated ratio crosses
    /// the threshold factor. Returns `None` for in-bounds ratios.
    fn compute_breach(
        &self,
        op_kind: OperatorKind,
        estimated: f64,
        observed: u64,
    ) -> Option<ThresholdBreach> {
        let factor = self.threshold_factor;
        let observed_f = observed as f64;
        // Special case: estimated == 0.
        if estimated <= 0.0 {
            // Any positive observed against zero-estimated is unbounded
            // under-estimate (any factor exceeded vacuously). We require
            // SOME observed activity to fire — a fully-skipped operator
            // (no batches) does not breach.
            if observed > 0 {
                return Some(ThresholdBreach::under_estimate(
                    op_kind, estimated, observed,
                ));
            }
            return None;
        }
        // Normal case.
        let ratio = observed_f / estimated;
        if ratio >= factor {
            return Some(ThresholdBreach::under_estimate(
                op_kind, estimated, observed,
            ));
        }
        // Over-estimate path. observed == 0 with estimated > 0 fires only
        // if estimated >= factor (the planner asserted "≥factor rows" but
        // we got nothing). For estimated < factor we don't fire — the
        // operator may legitimately produce no rows at low estimates.
        if observed == 0 {
            if estimated >= factor {
                return Some(ThresholdBreach::over_estimate(op_kind, estimated, observed));
            }
            return None;
        }
        let inv_ratio = estimated / observed_f;
        if inv_ratio >= factor {
            return Some(ThresholdBreach::over_estimate(op_kind, estimated, observed));
        }
        None
    }

    /// Aggregate the observer state into the M4-91 PROFILE [`ExecutionMetrics`]
    /// shape per amendment-03 §TIER-2-c.
    #[must_use]
    pub fn execution_metrics(&self) -> ExecutionMetrics {
        let state = self.state.lock();
        let mut wall_total: u64 = 0;
        let mut mem_high: u64 = 0;
        // The "rows emitted" at the root operator is a proxy for the
        // user-visible result-row count. We use the Project kind's
        // observed rows when present (RETURN clause); falling back to
        // the Filter kind, then RankByHybrid (for hybrid retrieval),
        // then any kind with observed rows. This matches the M4-31
        // root-of-plan invariant for v1.0 read queries.
        let mut rows_emitted: u64 = 0;
        for (kind, s) in &state.per_kind {
            wall_total = wall_total.saturating_add(s.wall_time_ns);
            mem_high = mem_high.max(s.memory_bytes_high_water);
            if matches!(kind, OperatorKind::Project) {
                rows_emitted = s.observed_rows;
            }
        }
        if rows_emitted == 0 {
            // Fallback: project-less plans (raw RETURN) — use the topmost
            // kind that observed any rows. Plan walk's first entry is
            // the root by pre-order discipline.
            if let Some(root) = state.plan_walk.first() {
                if let Some(s) = state.per_kind.get(&root.op_kind) {
                    rows_emitted = s.observed_rows;
                }
            }
        }
        ExecutionMetrics {
            wall_time_ms: ns_to_ms_ceil(wall_total),
            memory_bytes_high_water: mem_high,
            rows_emitted,
        }
    }

    /// Project the observer's state into per-tenant observed-stats
    /// overrides for M4-04 catalog feedback.
    ///
    /// Apportionment rules:
    /// - **Per-label observed (Scan)** — for each `(label, est_card)`
    ///   pair from Scan plan-walk entries, attribute the observed Scan
    ///   row count proportionally to that label's share of the
    ///   estimated card. Single-Scan plans (typical at v1.0) attribute
    ///   100% to the single label.
    /// - **Per-rel-type observed (Expand)** — same pattern for Expand
    ///   plan-walk entries.
    /// - **Total nodes / rels** — derived from Scan + Expand observed
    ///   sums respectively. These are NOT exact (the Scan may be label-
    ///   filtered); the observer reports them only when the scan was
    ///   label-free OR when there's exactly one labelled Scan and the
    ///   caller confirms it covered the entire tenant.
    #[must_use]
    pub fn observed_overrides(&self) -> ObservedStatsOverrides {
        let state = self.state.lock();
        let scan_observed = state
            .per_kind
            .get(&OperatorKind::Scan)
            .map(|s| s.observed_rows)
            .unwrap_or(0);
        let expand_observed = state
            .per_kind
            .get(&OperatorKind::Expand)
            .map(|s| s.observed_rows)
            .unwrap_or(0);
        let mut label_observed: HashMap<LabelId, u64> = HashMap::new();
        let mut rel_type_observed: HashMap<TypeId, u64> = HashMap::new();
        // Build per-Scan estimated weights for label apportionment.
        let scan_entries: Vec<&PlanWalkEntry> = state
            .plan_walk
            .iter()
            .filter(|e| e.op_kind == OperatorKind::Scan && e.scan_label.is_some())
            .collect();
        let scan_weight_sum: f64 = scan_entries.iter().map(|e| e.estimated_card.max(1.0)).sum();
        for entry in &scan_entries {
            if let Some(label) = entry.scan_label {
                let weight = entry.estimated_card.max(1.0) / scan_weight_sum;
                let attributed = (scan_observed as f64 * weight).round() as u64;
                *label_observed.entry(label).or_default() += attributed;
            }
        }
        // Same for Expand → rel_type.
        let expand_entries: Vec<&PlanWalkEntry> = state
            .plan_walk
            .iter()
            .filter(|e| e.op_kind == OperatorKind::Expand && e.expand_rel_type.is_some())
            .collect();
        let expand_weight_sum: f64 = expand_entries
            .iter()
            .map(|e| e.estimated_card.max(1.0))
            .sum();
        for entry in &expand_entries {
            if let Some(rt) = entry.expand_rel_type {
                let weight = entry.estimated_card.max(1.0) / expand_weight_sum;
                let attributed = (expand_observed as f64 * weight).round() as u64;
                *rel_type_observed.entry(rt).or_default() += attributed;
            }
        }
        // Totals: only emit when all Scans / Expands are unlabelled OR
        // when there's exactly one labelled Scan with full coverage.
        // v1.0-alpha: report scan_observed as total_nodes when there are
        // no label-filtered Scans (single SCAN over all nodes).
        let any_labelled_scan = state
            .plan_walk
            .iter()
            .any(|e| e.op_kind == OperatorKind::Scan && e.scan_label.is_some());
        let total_nodes_observed = if any_labelled_scan {
            None
        } else if state
            .plan_walk
            .iter()
            .any(|e| e.op_kind == OperatorKind::Scan)
        {
            Some(scan_observed)
        } else {
            None
        };
        let any_typed_expand = state
            .plan_walk
            .iter()
            .any(|e| e.op_kind == OperatorKind::Expand && e.expand_rel_type.is_some());
        let total_rels_observed = if any_typed_expand {
            None
        } else if state
            .plan_walk
            .iter()
            .any(|e| e.op_kind == OperatorKind::Expand)
        {
            Some(expand_observed)
        } else {
            None
        };
        ObservedStatsOverrides {
            label_observed,
            rel_type_observed,
            total_nodes_observed,
            total_rels_observed,
        }
    }

    /// Read the plan-walk anchor (immutable after construction).
    #[must_use]
    pub fn plan_walk(&self) -> Vec<PlanWalkEntry> {
        self.state.lock().plan_walk.clone()
    }

    /// Reset the observer state. Used by replan to clear observations
    /// before re-executing under a new plan.
    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.per_kind.clear();
    }
}

impl Default for RowCountObserver {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn ns_to_ms_ceil(ns: u64) -> u64 {
    if ns == 0 {
        0
    } else {
        // Round up sub-millisecond accumulations to 1ms so wall_time_ms
        // never reports 0 for non-zero work.
        ns.saturating_add(999_999) / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::{LogicalEmpty, LogicalPlan};
    use crate::observer::threshold::BreachDirection;
    use crate::planner::cost::{Cardinality, Cost, CostNode, CostedTree};

    /// M4-71 unit test #1: a fresh observer reports zero across the
    /// board. Nothing recorded, no breaches.
    #[test]
    fn fresh_observer_reports_no_breaches_and_zero_metrics() {
        let obs = RowCountObserver::new();
        assert!(obs.threshold_breaches().is_empty());
        let metrics = obs.metrics();
        assert!(metrics.is_empty(), "no plan walk + no recordings → empty");
        let exec = obs.execution_metrics();
        assert_eq!(exec.rows_emitted, 0);
        assert_eq!(exec.wall_time_ms, 0);
        assert_eq!(exec.memory_bytes_high_water, 0);
    }

    /// M4-71 unit test #2: record_batch accumulates rows + wall-time;
    /// memory is high-water (max), not summed.
    #[test]
    fn record_batch_accumulates_rows_walltime_and_memory_high_water() {
        let obs = RowCountObserver::new();
        obs.record_batch(OperatorKind::Scan, 100, 1_500_000, 2_048);
        obs.record_batch(OperatorKind::Scan, 50, 500_000, 1_024);
        obs.record_batch(OperatorKind::Scan, 75, 1_000_000, 4_096);
        let metrics = obs.metrics();
        assert_eq!(metrics.len(), 1);
        let m = &metrics[0];
        assert_eq!(m.op_kind, Some(OperatorKind::Scan));
        assert_eq!(m.observed_rows, 225, "rows are summed");
        assert_eq!(m.wall_time_ns, 3_000_000);
        assert_eq!(m.memory_bytes_high_water, 4_096, "memory is HIGH-water");
        assert_eq!(m.batches, 3);
    }

    /// M4-71 unit test #3: per-kind aggregation isolates kinds — Scan
    /// recordings don't bleed into Filter.
    #[test]
    fn per_kind_aggregation_isolates_kinds() {
        let obs = RowCountObserver::new();
        obs.record_batch(OperatorKind::Scan, 100, 1_000_000, 2_000);
        obs.record_batch(OperatorKind::Filter, 75, 500_000, 1_500);
        obs.record_batch(OperatorKind::Project, 75, 250_000, 1_500);
        let metrics = obs.metrics();
        assert_eq!(metrics.len(), 3);
        let kinds: Vec<OperatorKind> = metrics.iter().filter_map(|m| m.op_kind).collect();
        // Stable order via OperatorKind::ALL.
        assert_eq!(
            kinds,
            vec![
                OperatorKind::Scan,
                OperatorKind::Filter,
                OperatorKind::Project
            ]
        );
    }

    /// M4-71 unit test #4: 10× threshold detection in the under-estimate
    /// direction (observed ≥ 10× estimated).
    #[test]
    fn under_estimate_breach_fires_when_observed_exceeds_factor_times_estimated() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::new(10.0)));
        let obs = RowCountObserver::from_plan_and_costs(&plan, &costs);
        // Observed 100 against estimated 10 → 10× — breach (≥ factor).
        obs.record_batch(OperatorKind::Empty, 100, 0, 0);
        let breaches = obs.threshold_breaches();
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].direction, BreachDirection::UnderEstimate);
        assert_eq!(breaches[0].observed_rows_sum, 100);
        assert_eq!(breaches[0].estimated_card_sum, 10.0);
        assert_eq!(breaches[0].ratio, 10.0);
    }

    /// M4-71 unit test #5: 10× threshold detection in the over-estimate
    /// direction (estimated ≥ 10× observed).
    #[test]
    fn over_estimate_breach_fires_when_estimated_exceeds_factor_times_observed() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::new(1000.0)));
        let obs = RowCountObserver::from_plan_and_costs(&plan, &costs);
        // Observed 50 against estimated 1000 → inv-ratio 20× → breach.
        obs.record_batch(OperatorKind::Empty, 50, 0, 0);
        let breaches = obs.threshold_breaches();
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].direction, BreachDirection::OverEstimate);
    }

    /// M4-71 unit test #6: in-bounds ratios produce NO breaches.
    /// Observed 50 against estimated 100 → 0.5× — well within 10×.
    #[test]
    fn in_bounds_ratio_produces_no_breach() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::new(100.0)));
        let obs = RowCountObserver::from_plan_and_costs(&plan, &costs);
        obs.record_batch(OperatorKind::Empty, 50, 0, 0);
        assert!(obs.threshold_breaches().is_empty());
    }

    /// M4-71 unit test (extra): structured-field threshold_breach event
    /// fires per amendment-03 §TIER-2-c "tracing::warn! per breach".
    #[test]
    #[tracing_test::traced_test]
    fn threshold_breach_emits_structured_tracing_event() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::new(10.0)));
        let obs = RowCountObserver::from_plan_and_costs(&plan, &costs);
        obs.record_batch(OperatorKind::Empty, 100, 0, 0);
        let _ = obs.threshold_breaches();
        assert!(
            logs_contain("threshold_breach"),
            "tracing::warn! threshold_breach event must fire on detected breach",
        );
        assert!(
            logs_contain(r#"direction="under_estimate""#),
            "structured field `direction` must carry the slug",
        );
    }

    /// M4-71 unit test (extra): row_count_observed event fires on
    /// record_batch (trace-level for production cost gating).
    #[test]
    #[tracing_test::traced_test]
    fn record_batch_emits_row_count_observed_tracing_event() {
        let obs = RowCountObserver::new();
        obs.record_batch(OperatorKind::Scan, 5, 100, 200);
        assert!(
            logs_contain("row_count_observed"),
            "tracing event `row_count_observed` must fire on record_batch",
        );
    }

    /// Threshold factor override pin — tests at v1.0 use 2.0 to shrink
    /// fixture sizes; production stays at 10.0 default.
    #[test]
    fn threshold_factor_is_configurable() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::new(10.0)));
        let obs = RowCountObserver::from_plan_and_costs(&plan, &costs).with_threshold_factor(2.0);
        // Observed 21 against estimated 10 → 2.1× — breach at 2.0
        // factor; would NOT breach at 10.0 default.
        obs.record_batch(OperatorKind::Empty, 21, 0, 0);
        assert_eq!(obs.threshold_breaches().len(), 1);
    }

    /// `OperatorKind::ALL` covers every variant — pinned so future
    /// additions don't drift.
    #[test]
    fn operator_kind_all_covers_every_variant() {
        // Round-trip: every variant in ALL has a unique as_str.
        let mut slugs: Vec<&str> = OperatorKind::ALL.iter().map(|k| k.as_str()).collect();
        slugs.sort();
        slugs.dedup();
        assert_eq!(slugs.len(), OperatorKind::ALL.len(), "no duplicate slugs");
    }
}
