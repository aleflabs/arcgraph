//! Plan enumeration + DP-based binary-join ordering (M4-52 / M4-05b).
//!
//! # Slice scope
//!
//! M4-52 (M4-05b) ships **System-R-style DP join ordering** over
//! [`LogicalPlan`] sub-trees per ADR-038 amendment-02 §M4.e. The
//! enumerator consumes:
//!
//! - The **M4-51 cost walker** ([`crate::planner::cost::estimate_costs`])
//!   for full-plan cost evaluation of leaf-relation candidates.
//! - The **M4-51 per-operator cost functions**
//!   ([`crate::planner::cost::operator::cost_join`]) for incremental
//!   costing inside the DP loop (avoids the `O(2^N)` walk-the-whole-
//!   tree-per-candidate complexity blow-up; see [`dp`] module budget).
//! - The **M4-04d-anchored selectivity constants** through the
//!   walker — M4-52 reads them transitively, never directly.
//!
//! Outputs a re-rooted [`LogicalPlan`] with a cost-optimal left-deep
//! join ordering for star + linear shapes (per amendment-02 §M4.e
//! "binary joins at v1.0; bushy deferred to v1.1"). Costs are
//! re-derived by downstream consumers (M4-91 EXPLAIN, M4-71 row-count
//! observer); M4-52 returns plain `LogicalPlan`, NOT `CostedPlan`,
//! because re-rooting changes the cost-tree shape and the cost
//! consumer should re-derive from the post-enumeration plan.
//!
//! # Snapshot-once contract (cross-PR coherence with PR #220)
//!
//! Per ADR-038 §2 D-25 + amendment-03 §M4-04e (issue #210) + PR #220
//! (CatalogStats two-marker SeqLock cross-key consistency), the
//! M4-51 walker calls
//! [`crate::semantic::CatalogProvider::snapshot()`] EXACTLY ONCE per
//! `estimate_costs` invocation. M4-52 enumerates many candidate
//! plans; without precaution each candidate would acquire its own
//! snapshot, violating the across-candidates apples-to-apples
//! comparison that join-ordering DP requires.
//!
//! M4-52 closes this for the **cost-cardinality flow** with the
//! `FrozenCatalog` wrapper:
//!
//! 1. The DP captures `catalog.snapshot()` ONCE at enumeration entry.
//! 2. Every per-candidate cost evaluation reads CARDINALITY data via
//!    `FrozenCatalog::snapshot`, whose return is a clone of the
//!    captured value. The walker's `cost_*` functions in
//!    [`crate::planner::cost::operator`] all read from the snapshot
//!    (see `cost::operator::cost_scan` / `cost_expand` / friends —
//!    all consume `&CatalogSnapshot`, not the live catalog).
//! 3. Across candidates the walker observes identical snapshots,
//!    producing apples-to-apples cost comparisons inside the DP loop
//!    (`cost_join` reads only cached `output_card` values + frozen
//!    snapshot totals).
//!
//! **Scope limit (M4-04d selectivity flow).** The M4-04d
//! [`crate::semantic::SelectivityEstimator`] (per ADR-038 §D-27)
//! reads the LIVE catalog for predicate selectivity (filter
//! coefficients), inheriting M4-51's design. The N initial leaf-cost
//! walks (one per leaf in `dp::enumerate`) traverse any Filter
//! wrappers above the bare leaves and therefore re-read live
//! selectivity per walk; concurrent commits during that window CAN
//! drift selectivity coefficients. The DP loop proper (per-candidate
//! `cost_join`) operates only on cached `output_card` values + the
//! frozen snapshot's cardinality totals, so post-leaf-walk
//! apples-to-apples comparison IS preserved. v1.0 inputs to M4-52
//! are dominantly bare-leaf (Scan / Expand / hybrid retrieval); the
//! Filter-wrapper case where this drift matters is rare in practice
//! and bounded by per-commit selectivity deltas. v1.1 may layer
//! adaptive replan-driven re-snapshot, but that's M4-72's concern.
//!
//! # Local-only checklist (Q1/Q2/Q3 per ADR-024 amendment-02)
//!
//! - **Q1 (shared memory)** — every per-call DP table (the
//!   `Vec<Option<DpEntry>>` allocated inside `dp::enumerate`) is
//!   owned by the enumeration invocation; it is dropped at function
//!   return. NO `static mut`, no `OnceCell`-backed cross-tenant
//!   cache. Per-tenant by construction. Evidence: the table is
//!   heap-local at `dp::enumerate_inner`'s `let mut dp:
//!   Vec<Option<DpEntry>> = vec![None; dp_size]` declaration.
//! - **Q2 (multi-machine)** — the DP runs entirely within the calling
//!   thread; no cross-node coordination, no RPC, no message passing.
//!   Trivially partition-aware (no cross-tenant state to partition).
//! - **Q3 (consistent recovery LSN)** — the DP is stateless; no WAL
//!   writes, no on-disk persistence, no `Lsn`-tagged side tables.
//!   The `read_lsn` carried on each [`LogicalPlan::Scan`] /
//!   [`LogicalPlan::Expand`] / hybrid leaf flows through unchanged
//!   from input to output (re-rooting preserves leaf identity by
//!   move). Recovery semantics are unaffected.
//!
//! # Budget (Prime Directive 5)
//!
//! Per ADR-036 §D-25 the M4-05 plan-build budget is **5 ms** end-to-
//! end. The M4-51 cost walker consumes ~5–50 µs at v1.0 plan sizes.
//! M4-52 layered on top must stay well within the remaining budget.
//!
//! Left-deep DP complexity:
//! - State space: `O(N × 2^N)` subsets of N relations × N split
//!   positions per subset.
//! - Per-candidate work: O(1) — incremental costing via
//!   [`crate::planner::cost::operator::cost_join`] + a binding-set
//!   intersection (≤ 8 BindingIds per leaf at v1.0 plan sizes).
//!
//! For N ∈ {2..8} (the v1.0 LDBC SNB envelope per design-v2 §10.5),
//! total work is `8 × 256 × O(1) ≈ 2K` operations + N initial
//! `estimate_costs` walks. Wall-clock ≤ 1 ms, well inside budget.
//! For N > 8 the enumerator bails out and returns the input plan
//! unchanged (see [`MAX_DP_RELATIONS`]); the v1.1 sketch-aware
//! refinement may lift the cap.
//!
//! # Exhaustive-match discipline (ADR-038 §2 D-24)
//!
//! [`LogicalPlan`] is NOT `#[non_exhaustive]`. The
//! [`enumerate_join_order`] entry recursively rewrites the plan tree
//! and MUST exhaustively match all 20 variants — adding a new
//! variant to [`LogicalPlan`] forces a compile-error here, by
//! design. The walker carrying the recursion is in `rewrite`.
//!
//! Mirrors the M4-31 / M4-32 / M4-33 / M4-51 walker discipline. All
//! six current consumers of [`LogicalPlan`] honor the contract; M4-52
//! is the seventh.
//!
//! # ADR provenance
//!
//! Note: ADR-038 §D-25 and ADR-036 §D-25 are different load-bearing
//! sections — disambiguated below to avoid skim-misreading. Each cite
//! within the source body of this module spells out the full ADR-NNN
//! prefix; the two §D-25s are NEVER co-located on adjacent lines per
//! W9d M4-52b F-7 NIT closure.
//!
//! - ADR-038 §2 D-24 — `LogicalPlan` exhaustive-match contract.
//! - ADR-038 §2 D-25 — **catalog stats schema** (M4-41 input). NOT
//!   the plan-build budget; that lives at ADR-036 §D-25 below.
//! - ADR-036 §D-25 — **5 ms M4-05 plan-build budget**. NOT the
//!   catalog stats schema; that lives at ADR-038 §2 D-25 above.
//! - ADR-038 amendment-02 §M4.e — M4-52 (M4-05b) DP join ordering
//!   slice scope.
//! - ADR-038 amendment-03 §M4-04e — cross-key snapshot contract
//!   (PR #220 producer; M4-51 + M4-52 consumers).
//! - ADR-038 amendment-07 — M4-04d empirical selectivity tuning
//!   (read transitively via M4-51).
//! - ADR-024 amendment-02 §5 — local-only posture (Q1/Q2/Q3
//!   above).

pub mod dp;
pub mod reroot;
pub mod shape_detect;

use crate::logical_plan::{
    HybridOperand, JoinAlgorithm, LogicalAggregate, LogicalCommunityLookup, LogicalDelete,
    LogicalDistinct, LogicalDynamicLimit, LogicalEmpty, LogicalFilter, LogicalFusion, LogicalJoin,
    LogicalLeftOuterJoin, LogicalLimit, LogicalNamedPath, LogicalPlan, LogicalProcedureCall,
    LogicalProject, LogicalRemove, LogicalSet, LogicalSkip, LogicalSort, LogicalUnion,
    LogicalUnwind,
};
use crate::semantic::bound_ast::{BindingId, BoundProjectionKind};
use crate::semantic::{CatalogProvider, CatalogSnapshot};
use arcgraph_core::{LabelId, PartitionId, PropertyId, TenantId, TypeId};

pub use dp::{DpFallbackReason, DpStats};
pub use shape_detect::JoinShape;

/// Maximum number of leaf relations the DP enumerates over.
///
/// At `N ≤ MAX_DP_RELATIONS` the DP runs to completion. Above the
/// cap the enumerator returns the input plan unchanged — the cost
/// of full DP exceeds the ADR-036 §D-25 5 ms plan-build budget for
/// `N > 8` even with incremental costing (`8 × 256 = 2K` candidate
/// evaluations is the v1.0 ceiling). v1.1 may lift this when sketch-
/// aware pruning makes wider plans tractable.
pub const MAX_DP_RELATIONS: usize = 8;

/// **Test-only** sibling of [`enumerate_join_order`] that picks the
/// MAXIMUM-cost candidate at every DP transition instead of the
/// minimum. Used by the Phase 4.2 controlled-mutation probe in the
/// M4-52 proptest harness to demonstrate the cost-optimality oracle
/// is non-vacuous (per the spawn-prompt acceptance criterion + PR
/// #232 review §"controlled-mutation probe").
///
/// # Discipline
///
/// **Production code paths MUST NEVER call this.** The function is
/// named to be loud about its test-only role (`_for_proptest` suffix);
/// a future reviewer who finds it cited at a non-test call site has
/// discovered a regression and SHOULD reject the change. The hook is
/// `#[doc(hidden)] pub` because the only legitimate consumer
/// (`tests/m4_52_join_enumeration_proptest.rs`) is an integration test
/// crate that cannot reach `pub(crate)` symbols. The contract is
/// enforced by:
///
/// 1. `#[doc(hidden)]` — keeps the function out of generated rustdoc.
/// 2. The `_for_proptest` naming convention — visually obvious at
///    every call site.
/// 3. The proptest is the ONLY current consumer (verified by the
///    pre-fix `grep -rn "enumerate_join_order_pick_max_for_proptest"`
///    sweep; W9d M4-52b F-1 LOW closure).
///
/// # Forward-recommendation
///
/// W9b cross-PR review F-1 (LOW) recommends migrating to a
/// `test-hooks` Cargo feature gate
/// (`#[cfg(feature = "test-hooks")]` + `[[test]] required-features =
/// ["test-hooks"]`) for compile-time enforcement. Deferred to a
/// follow-up slice — the doc-strengthening + naming convention here
/// closes the W9b LOW finding at the discipline level; the structural
/// hardening is purely additive when it lands.
#[doc(hidden)]
#[must_use]
pub fn enumerate_join_order_pick_max_for_proptest(
    plan: LogicalPlan,
    catalog: &dyn CatalogProvider,
) -> LogicalPlan {
    let snapshot = catalog.snapshot();
    let frozen = FrozenCatalog::new(catalog, snapshot);
    rewrite(plan, &frozen, true)
}

/// Re-root a [`LogicalPlan`] with cost-optimal join ordering.
///
/// Walks the input plan top-down. For each contiguous chain of
/// [`LogicalPlan::Join`] nodes (inner equi-joins per
/// `JoinCondition::SharedBindings`), extracts the leaf relations,
/// runs DP enumeration, and replants the cost-optimal left-deep tree
/// at the original join's position. Recurses into all other variants
/// (including [`LogicalPlan::LeftOuterJoin`] — outer-join ordering is
/// preserved per Cypher 9 §6.5 OPTIONAL MATCH semantics; the DP does
/// NOT cross outer-join boundaries at v1.0).
///
/// # Snapshot-once
///
/// Captures one [`CatalogSnapshot`] from `catalog` at function entry
/// and threads it through the DP via `FrozenCatalog`. Every
/// candidate plan in the enumeration sees the SAME snapshot, so the
/// cost comparison is apples-to-apples even under concurrent commit
/// activity.
///
/// # Determinism
///
/// The DP is fully deterministic — same plan + same catalog
/// snapshot → same output plan. Tie-breaks fall back to the
/// original input order (stable).
///
/// # Errors
///
/// This function never errors. Pathological inputs (N >
/// [`MAX_DP_RELATIONS`] leaves; disconnected join graphs;
/// unsupported operator mixes) all degrade gracefully to "return
/// input plan unchanged". Degradation paths are observable via the
/// structured [`DpStats::fallback_reason`] enum returned from
/// `dp::enumerate_with_stats` (test-only entry; M4-91 EXPLAIN forward-
/// consumer per W9b F-6 closure). The reason is also emitted as a
/// `tracing::debug!` event at each fallback site under target
/// `arcgraph_query::planner::dp` (W9b F-4 closure; co-packed in W9d
/// M4-52b after code-quality policy `tracing` dep landed via PR #244).
#[must_use]
pub fn enumerate_join_order(plan: LogicalPlan, catalog: &dyn CatalogProvider) -> LogicalPlan {
    let snapshot = catalog.snapshot();
    let frozen = FrozenCatalog::new(catalog, snapshot);
    enumerate_join_order_with_frozen(plan, &frozen)
}

/// [`enumerate_join_order`] sibling that reuses an externally-captured
/// [`FrozenCatalog`] instead of taking its own snapshot.
///
/// # Why
///
/// Per issue #261 (W9d retro Agent A §A-LOW-1), the EXPLAIN pipeline
/// previously captured TWO independent [`crate::semantic::CatalogSnapshot`]s
/// inside `crate::explain::plan_tree_for`: one for join-order DP (via
/// [`enumerate_join_order`]) and one for cost-keying (via
/// [`crate::planner::cost::estimate_costs`]). Under v1.1+ multi-tenant
/// concurrent writers the two snapshots could drift between calls,
/// producing apples-to-oranges cost annotations within a single
/// EXPLAIN. The fix threads a single captured [`FrozenCatalog`] through
/// both stages — this shim is the DP-side entry; the cost-side entry
/// is [`crate::planner::cost::estimate_costs_with_frozen`].
///
/// # Snapshot-once
///
/// The DP's snapshot-once contract is preserved by construction —
/// every per-candidate cost evaluation reads from `frozen.snapshot()`
/// which returns the captured value. Apples-to-apples comparison
/// inside the DP loop is unchanged.
#[must_use]
pub(crate) fn enumerate_join_order_with_frozen(
    plan: LogicalPlan,
    frozen: &FrozenCatalog<'_>,
) -> LogicalPlan {
    rewrite(plan, frozen, false)
}

/// Recursively rewrite a [`LogicalPlan`] sub-tree, picking the
/// cost-optimal join order for any contiguous chain of
/// [`LogicalPlan::Join`] nodes encountered along the way.
///
/// # Exhaustive-match contract
///
/// Every [`LogicalPlan`] variant is matched explicitly per ADR-038
/// §2 D-24. Adding a new variant requires updating this match.
fn rewrite(plan: LogicalPlan, frozen: &FrozenCatalog<'_>, pick_max: bool) -> LogicalPlan {
    match plan {
        // -----------------------------------------------------------
        // Inner-join cluster: extract contiguous Join chain leaves and
        // DP-enumerate. The leaves themselves are re-walked first so
        // any nested Join / LeftOuterJoin sub-trees inside the leaves
        // get their own optimization pass (depth-first re-rooting).
        // -----------------------------------------------------------
        LogicalPlan::Join(join)
            if matches!(join.algorithm, JoinAlgorithm::HashJoin)
                && matches!(join.right.as_ref(), LogicalPlan::RankByHybrid(_)) =>
        {
            // `RANK BY HYBRID` emits relevance order from the right/probe
            // side. Swapping this join makes the candidate scan the probe
            // and silently replaces relevance order with scan order.
            // Recurse into the candidate subtree, but retain this semantic
            // boundary exactly as lowered.
            LogicalPlan::Join(LogicalJoin {
                left: Box::new(rewrite(*join.left, frozen, pick_max)),
                right: join.right,
                on: join.on,
                algorithm: join.algorithm,
                span: join.span,
            })
        }
        LogicalPlan::Join(join) => {
            let original_span = join.span.clone();
            let leaves = reroot::extract_inner_join_leaves(LogicalPlan::Join(join));
            let leaves: Vec<LogicalPlan> = leaves
                .into_iter()
                .map(|leaf| rewrite(leaf, frozen, pick_max))
                .collect();

            if leaves.len() < 2 {
                // Defensive — extraction always returns ≥ 2 for a
                // Join input, but if a future refactor breaks that
                // we degrade cleanly. The single leaf becomes the
                // returned plan.
                return leaves
                    .into_iter()
                    .next()
                    .unwrap_or(LogicalPlan::Empty(LogicalEmpty {
                        span: original_span,
                    }));
            }

            if pick_max {
                dp::enumerate_pick_max_for_test(
                    leaves,
                    frozen as &dyn CatalogProvider,
                    &original_span,
                )
            } else {
                dp::enumerate(leaves, frozen, &original_span)
            }
        }

        // -----------------------------------------------------------
        // Outer-join boundary: do NOT enumerate across (left-outer
        // semantics constrain ordering per Cypher 9 §6.5). Recurse
        // into each side independently.
        // -----------------------------------------------------------
        LogicalPlan::LeftOuterJoin(j) => LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
            left: Box::new(rewrite(*j.left, frozen, pick_max)),
            right: Box::new(rewrite(*j.right, frozen, pick_max)),
            on: j.on,
            span: j.span,
        }),

        // -----------------------------------------------------------
        // Unary wrappers: recurse into the single input.
        // -----------------------------------------------------------
        LogicalPlan::Filter(f) => LogicalPlan::Filter(LogicalFilter {
            input: Box::new(rewrite(*f.input, frozen, pick_max)),
            predicate: f.predicate,
            span: f.span,
        }),
        LogicalPlan::Project(p) => LogicalPlan::Project(LogicalProject {
            input: Box::new(rewrite(*p.input, frozen, pick_max)),
            items: p.items,
            span: p.span,
        }),
        LogicalPlan::Limit(l) => LogicalPlan::Limit(LogicalLimit {
            input: Box::new(rewrite(*l.input, frozen, pick_max)),
            count: l.count,
            span: l.span,
        }),
        LogicalPlan::Skip(s) => LogicalPlan::Skip(LogicalSkip {
            input: Box::new(rewrite(*s.input, frozen, pick_max)),
            count: s.count,
            span: s.span,
        }),
        LogicalPlan::DynamicLimit(l) => LogicalPlan::DynamicLimit(LogicalDynamicLimit {
            input: Box::new(rewrite(*l.input, frozen, pick_max)),
            kind: l.kind,
            count_expr: l.count_expr,
            span: l.span,
        }),
        LogicalPlan::Sort(s) => LogicalPlan::Sort(LogicalSort {
            input: Box::new(rewrite(*s.input, frozen, pick_max)),
            order_by: s.order_by,
            span: s.span,
        }),
        LogicalPlan::Distinct(d) => LogicalPlan::Distinct(LogicalDistinct {
            input: Box::new(rewrite(*d.input, frozen, pick_max)),
            on: d.on,
            span: d.span,
        }),
        LogicalPlan::Unwind(u) => LogicalPlan::Unwind(LogicalUnwind {
            input: Box::new(rewrite(*u.input, frozen, pick_max)),
            list_expr: u.list_expr,
            var: u.var,
            span: u.span,
        }),
        // ADR-197 (#802): no joins inside a procedure call; rewrite the
        // (unit-row) input for uniformity.
        LogicalPlan::ProcedureCall(p) => LogicalPlan::ProcedureCall(LogicalProcedureCall {
            input: Box::new(rewrite(*p.input, frozen, pick_max)),
            source: p.source,
            args: p.args,
            columns: p.columns,
            span: p.span,
        }),
        LogicalPlan::Aggregate(a) => LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(rewrite(*a.input, frozen, pick_max)),
            group_by: a.group_by,
            aggregations: a.aggregations,
            span: a.span,
        }),
        LogicalPlan::CommunityLookup(c) => LogicalPlan::CommunityLookup(LogicalCommunityLookup {
            input: Box::new(rewrite(*c.input, frozen, pick_max)),
            node_var: c.node_var,
            community_id: c.community_id,
            read_lsn: c.read_lsn,
            span: c.span,
        }),
        LogicalPlan::NamedPath(n) => LogicalPlan::NamedPath(LogicalNamedPath {
            input: Box::new(rewrite(*n.input, frozen, pick_max)),
            path_var: n.path_var,
            algorithm: n.algorithm,
            // ADR-193 — preserve the Plain-path element-binding shape
            // across the rewrite (the subtree's bindings are unchanged by
            // cost-based join rewrites — only join algorithms are picked).
            plain_shape: n.plain_shape,
            // ADR-194 D-3a — preserve the captured head (source) +
            // tail-endpoint (target) bindings across the rewrite (likewise
            // unchanged: cost-based rewrites pick join algorithms, not
            // bindings).
            source: n.source,
            target: n.target,
            span: n.span,
        }),

        // -----------------------------------------------------------
        // n-ary Fusion: recurse into each input.
        // -----------------------------------------------------------
        LogicalPlan::Fusion(f) => LogicalPlan::Fusion(LogicalFusion {
            spec: f.spec,
            inputs: f
                .inputs
                .into_iter()
                .map(|input| Box::new(rewrite(*input, frozen, pick_max)))
                .collect(),
            span: f.span,
        }),

        // ADR-185 (#649-A1, W28) — UNION ALL: DP-optimize each arm
        // independently (a Join chain inside one arm enumerates within
        // that arm only; arms do not share a join space).
        LogicalPlan::Union(u) => LogicalPlan::Union(LogicalUnion {
            arms: u
                .arms
                .into_iter()
                .map(|arm| rewrite(arm, frozen, pick_max))
                .collect(),
            column_orders: u.column_orders,
            span: u.span,
        }),

        LogicalPlan::CreateNode(_) | LogicalPlan::CreateRel(_) => {
            rewrite_create_spine(plan, frozen, pick_max)
        }

        // -----------------------------------------------------------
        // Delete (ADR-149 W26-θ Phase 3) — recurse into the input
        // sub-plan; the per-item Delete metadata (items + detach
        // flag) is structural and passes through unchanged.
        // -----------------------------------------------------------
        LogicalPlan::Delete(d) => LogicalPlan::Delete(LogicalDelete {
            input: Box::new(rewrite(*d.input, frozen, pick_max)),
            items: d.items,
            detach: d.detach,
            span: d.span,
        }),

        // -----------------------------------------------------------
        // Set / Remove (ADR-150 W26-θ Phase 4) — recurse into the
        // input sub-plan; the per-item mutation metadata passes
        // through unchanged.
        // -----------------------------------------------------------
        LogicalPlan::Set(s) => LogicalPlan::Set(LogicalSet {
            input: Box::new(rewrite(*s.input, frozen, pick_max)),
            items: s.items,
            span: s.span,
        }),
        LogicalPlan::Remove(r) => LogicalPlan::Remove(LogicalRemove {
            input: Box::new(rewrite(*r.input, frozen, pick_max)),
            items: r.items,
            span: r.span,
        }),

        // -----------------------------------------------------------
        // Merge (ADR-151 W26-θ Phase 5) — recurse into BOTH the match
        // and create sub-plans; the per-branch action item metadata
        // passes through unchanged.
        // -----------------------------------------------------------
        LogicalPlan::Merge(m) => LogicalPlan::Merge(crate::logical_plan::types::LogicalMerge {
            match_branch: Box::new(rewrite(*m.match_branch, frozen, pick_max)),
            create_branch: Box::new(rewrite(*m.create_branch, frozen, pick_max)),
            on_create: m.on_create,
            on_match: m.on_match,
            // ADR-151-amendment-01 §D-1 — the RETURN-after-MERGE
            // emission discriminator is shape-invariant under plan
            // rewrite (the binding set does not change); pass through.
            output_binding: m.output_binding,
            // NN-4 (#1384) — the get-or-create serialization keys are
            // shape-invariant under join-order rewrite (they name the
            // pattern's label + property literals, not a plan shape);
            // pass through unchanged.
            merge_keys: m.merge_keys,
            span: m.span,
        }),

        // -----------------------------------------------------------
        // CALL { … } (ADR-192 #623) — recurse into BOTH the driving
        // input and the subquery body so a Join chain inside either
        // gets DP-optimized (the body is an independent sub-plan
        // re-executed per driving row).
        // -----------------------------------------------------------
        LogicalPlan::Call(c) => LogicalPlan::Call(crate::logical_plan::types::LogicalCall {
            input: Box::new(rewrite(*c.input, frozen, pick_max)),
            body: Box::new(rewrite(*c.body, frozen, pick_max)),
            imported: c.imported,
            returned: c.returned,
            span: c.span,
        }),

        // -----------------------------------------------------------
        // Hybrid retrieval leaves carry no LogicalPlan children — they
        // are sub-tree leaves from the planner's perspective. Pass
        // through unchanged.
        // -----------------------------------------------------------
        LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::Scan(_)
        // #1366 (Phase 2): the indexed point-lookup is a read leaf.
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf write-op.
        | LogicalPlan::CreateVectorIndex(_)
        // #1366: CREATE INDEX (property index) is a leaf write-op.
        | LogicalPlan::CreatePropertyIndex(_)
        // ADR-192 (#623): the correlation seed is a leaf — pass through.
        | LogicalPlan::CorrelationSeed(_)
        | LogicalPlan::Empty(_) => plan,
    }
}

fn rewrite_create_spine(
    plan: LogicalPlan,
    frozen: &FrozenCatalog<'_>,
    pick_max: bool,
) -> LogicalPlan {
    let mut spine = Vec::new();
    let mut cursor = plan;
    let tail = loop {
        match cursor {
            LogicalPlan::CreateNode(mut c) => {
                let input = c.input.take();
                spine.push(LogicalPlan::CreateNode(c));
                match input {
                    Some(input) => cursor = *input,
                    None => break None,
                }
            }
            LogicalPlan::CreateRel(mut c) => {
                let input = c.input.take();
                spine.push(LogicalPlan::CreateRel(c));
                match input {
                    Some(input) => cursor = *input,
                    None => break None,
                }
            }
            other => break Some(other),
        }
    };

    let mut chain = tail.map(|tail| rewrite(tail, frozen, pick_max));
    for step in spine.into_iter().rev() {
        let rebuilt = match step {
            LogicalPlan::CreateNode(mut c) => {
                c.input = chain.take().map(Box::new);
                LogicalPlan::CreateNode(c)
            }
            LogicalPlan::CreateRel(mut c) => {
                c.source_plan = Box::new(rewrite(*c.source_plan, frozen, pick_max));
                c.target_plan = Box::new(rewrite(*c.target_plan, frozen, pick_max));
                c.input = chain.take().map(Box::new);
                LogicalPlan::CreateRel(c)
            }
            _ => unreachable!("create spine contains only CREATE write ops"),
        };
        chain = Some(rebuilt);
    }

    chain.expect("create spine rewrite starts with a CREATE plan")
}

/// Snapshot-locked [`CatalogProvider`] adapter (snapshot-once
/// contract; see module docs §"Snapshot-once contract").
///
/// Wraps an underlying [`CatalogProvider`] but overrides
/// [`CatalogProvider::snapshot`] to return a captured value clone
/// regardless of what the inner provider would currently produce.
/// All other methods delegate to the inner provider unchanged.
///
/// Constructed at the top of [`enumerate_join_order`] and dropped at
/// function return. Per-call lifetime; never shared across queries.
pub(crate) struct FrozenCatalog<'a> {
    inner: &'a dyn CatalogProvider,
    snapshot: CatalogSnapshot,
}

impl<'a> FrozenCatalog<'a> {
    /// Wrap an underlying [`CatalogProvider`] with a captured
    /// snapshot.
    pub(crate) fn new(inner: &'a dyn CatalogProvider, snapshot: CatalogSnapshot) -> Self {
        Self { inner, snapshot }
    }
}

impl<'a> CatalogProvider for FrozenCatalog<'a> {
    fn lookup_label(&self, name: &str) -> Option<LabelId> {
        self.inner.lookup_label(name)
    }
    fn lookup_rel_type(&self, name: &str) -> Option<TypeId> {
        self.inner.lookup_rel_type(name)
    }
    fn lookup_property(&self, name: &str) -> Option<PropertyId> {
        self.inner.lookup_property(name)
    }
    fn tenant(&self) -> TenantId {
        self.inner.tenant()
    }
    fn partition(&self) -> PartitionId {
        self.inner.partition()
    }
    fn has_vector_index(&self) -> bool {
        self.inner.has_vector_index()
    }
    fn has_bm25_index(&self) -> bool {
        self.inner.has_bm25_index()
    }
    fn has_community_index(&self) -> bool {
        self.inner.has_community_index()
    }
    fn online_property_index(&self, label: LabelId, property: &str) -> bool {
        // #1366 (Phase 2): delegate the RC-6 planner-visible gate to the
        // inner provider. Index eligibility is NOT part of the frozen
        // stats snapshot (it is a catalog-state read, not a cardinality),
        // so it delegates live like the other `*_index` capability
        // methods.
        self.inner.online_property_index(label, property)
    }
    fn label_cardinality(&self, label: LabelId) -> Option<u64> {
        self.inner.label_cardinality(label)
    }
    fn rel_type_cardinality(&self, rel_type: TypeId) -> Option<u64> {
        self.inner.rel_type_cardinality(rel_type)
    }
    fn total_node_count(&self) -> Option<u64> {
        self.inner.total_node_count()
    }
    fn total_rel_count(&self) -> Option<u64> {
        self.inner.total_rel_count()
    }

    /// Returns a clone of the captured snapshot — the snapshot-once
    /// load-bearing override. Identical across every call within a
    /// single [`enumerate_join_order`] invocation.
    fn snapshot(&self) -> CatalogSnapshot {
        self.snapshot.clone()
    }
}

/// Collect the bindings introduced by a [`LogicalPlan`] subtree.
///
/// Used by the DP to derive per-candidate [`JoinCondition`]
/// constraints from binding-set overlap. Matches exhaustively over
/// all 20 [`LogicalPlan`] variants per ADR-038 §2 D-24.
///
/// # Semantics
///
/// - **Producing variants** ([`LogicalPlan::Scan`],
///   [`LogicalPlan::Expand`], [`LogicalPlan::Unwind`],
///   [`LogicalPlan::NamedPath`]) introduce fresh bindings — those are
///   inserted into the result set.
/// - **Pass-through variants** (Filter, Limit, Skip, Sort, Distinct,
///   CommunityLookup, DynamicLimit) propagate their input's binding set
///   unchanged.
/// - **Renaming variants** (Project, Aggregate) report their OUTPUT
///   schema — the projected / group-by / aggregation `output_id`s (#841,
///   #746), NOT their input. A `WITH a` renames `a`, so the input id is
///   absent from the rows a downstream join sees; reporting the input
///   would yield an empty join key (a silent Cartesian). Mirrors the
///   executor's `ProjectOp` / `AggregateOp` schema and
///   `lowering::collect_bindings`.
/// - **n-ary variants** (Join, LeftOuterJoin, Fusion) union their
///   children's binding sets.
/// - **Hybrid leaves** (RankByHybrid, VectorNear, TextMatch) carry
///   binding REFERENCES (not introductions) — they read from
///   pre-existing bindings, so they contribute the referenced
///   binding (via [`HybridOperand::var`] / `var` field) so a
///   downstream join can see "this leaf depends on binding X".
/// - **`LogicalPlan::Empty`** introduces nothing.
///
/// Used as a read-only helper from [`dp`] and tests; not part of the
/// public API.
pub(crate) fn bindings_in(plan: &LogicalPlan) -> std::collections::BTreeSet<BindingId> {
    let mut out = std::collections::BTreeSet::new();
    visit_bindings(plan, &mut out);
    out
}

fn visit_bindings(plan: &LogicalPlan, out: &mut std::collections::BTreeSet<BindingId>) {
    match plan {
        LogicalPlan::Scan(s) => {
            out.insert(s.var);
        }
        // #1366 (Phase 2): the indexed point-lookup binds `var`, same as
        // the `Scan` it replaces.
        LogicalPlan::PropertyIndexScan(p) => {
            out.insert(p.var);
        }
        LogicalPlan::CountStore(c) => {
            out.insert(c.output_id);
        }
        LogicalPlan::Expand(e) => {
            out.insert(e.from);
            out.insert(e.to);
            if let Some(rv) = e.rel_var {
                out.insert(rv);
            }
        }
        LogicalPlan::Filter(f) => visit_bindings(&f.input, out),
        LogicalPlan::Project(p) => {
            // #841: report the Project's OUTPUT schema (its projected
            // `output_id`s), NOT its input. A renaming `WITH a` / `RETURN
            // a` (#746) mints a FRESH output id, so the input ids are
            // ABSENT from the output rows the DP joins on. The prior
            // input-recursion gave an EMPTY join key → a silent CARTESIAN
            // for correlated `CALL { WITH a … }` and the plain
            // `MATCH (a) WITH a MATCH (a)…` re-reference. TWIN of
            // `lowering::collect_bindings` (both derive join keys from
            // binding-set overlap — keep in sync); mirrors
            // `ProjectOp::derive_schema` (Wildcard passes input through; an
            // Expr item contributes its `output_id`).
            for item in &p.items {
                match &item.kind {
                    BoundProjectionKind::Wildcard { .. } => visit_bindings(&p.input, out),
                    BoundProjectionKind::Expr(_) => {
                        if let Some(id) = item.output_id {
                            out.insert(id);
                        }
                    }
                }
            }
        }
        LogicalPlan::Limit(l) => visit_bindings(&l.input, out),
        LogicalPlan::Skip(s) => visit_bindings(&s.input, out),
        LogicalPlan::DynamicLimit(l) => visit_bindings(&l.input, out),
        LogicalPlan::Sort(s) => visit_bindings(&s.input, out),
        LogicalPlan::Distinct(d) => visit_bindings(&d.input, out),
        LogicalPlan::Unwind(u) => {
            visit_bindings(&u.input, out);
            out.insert(u.var);
        }
        LogicalPlan::ProcedureCall(p) => {
            visit_bindings(&p.input, out);
            for (_, bid) in &p.columns {
                out.insert(*bid);
            }
        }
        LogicalPlan::Aggregate(a) => {
            // #841 (twin of the Project arm): an Aggregate's OUTPUT schema
            // is its group-by + aggregation `output_id`s (mirror
            // `AggregateOp::new`), NOT its input — `WITH a, count(b) AS n
            // MATCH (a)…` re-references the group-key `a` by its projected
            // output id. See the Project arm + `lowering::collect_bindings`.
            for item in &a.group_by {
                if let Some(id) = item.output_id {
                    out.insert(id);
                }
            }
            for call in &a.aggregations {
                out.insert(call.output_id);
            }
        }
        LogicalPlan::CommunityLookup(c) => {
            visit_bindings(&c.input, out);
            out.insert(c.node_var);
        }
        LogicalPlan::NamedPath(n) => {
            visit_bindings(&n.input, out);
            out.insert(n.path_var);
        }
        LogicalPlan::Join(j) => {
            visit_bindings(&j.left, out);
            visit_bindings(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            visit_bindings(&j.left, out);
            visit_bindings(&j.right, out);
        }
        LogicalPlan::Fusion(f) => {
            for input in &f.inputs {
                visit_bindings(input, out);
            }
        }
        // ADR-185 (#649-A1, W28) — UNION ALL: union the bindings of
        // every arm (each arm contributes its own binding ids).
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                visit_bindings(arm, out);
            }
        }
        LogicalPlan::RankByHybrid(r) => {
            visit_hybrid_operands(&r.operands, out);
            if let Some(score) = r.score_binding {
                out.insert(score);
            }
        }
        LogicalPlan::VectorNear(v) => {
            out.insert(v.var);
        }
        LogicalPlan::TextMatch(t) => {
            out.insert(t.var);
        }
        // ADR-147 W26-θ Phase 1.
        LogicalPlan::CreateNode(c) => {
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf DDL — it declares
        // no query bindings (it returns 0 rows / 0 columns).
        LogicalPlan::CreateVectorIndex(_) => {}
        // #1366: CREATE INDEX (property index) is a leaf DDL — no
        // query bindings (0 rows / 0 columns).
        LogicalPlan::CreatePropertyIndex(_) => {}
        // ADR-148 W26-θ Phase 2: union source + target sub-plan
        // bindings (the rel-binding, if present, is fresh and
        // contributed separately).
        LogicalPlan::CreateRel(c) => {
            visit_bindings(&c.source_plan, out);
            visit_bindings(&c.target_plan, out);
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        // ADR-149 W26-θ Phase 3: Delete passes through the input
        // sub-plan's bindings (parallel to Limit / Skip / Project).
        // The DELETE items REFERENCE existing bindings (no fresh
        // declaration); the upstream's binding set is the authoritative
        // contribution.
        LogicalPlan::Delete(d) => {
            visit_bindings(&d.input, out);
        }
        // ADR-150 W26-θ Phase 4: Set / Remove pass through the input
        // sub-plan's bindings — items REFERENCE existing bindings (no
        // fresh declaration; the upstream's binding set is the
        // authoritative contribution).
        LogicalPlan::Set(s) => {
            visit_bindings(&s.input, out);
        }
        LogicalPlan::Remove(r) => {
            visit_bindings(&r.input, out);
        }
        // ADR-151 W26-θ Phase 5: Merge introduces FRESH bindings via
        // the pattern (parallel to CREATE — match_branch + create_branch
        // both declare the pattern's variables). The union of both
        // sub-plans' binding sets covers the pattern's binding set
        // (match emits Scan/Expand bindings; create emits CreateNode/
        // CreateRel bindings — same binding-ids resolve in both).
        LogicalPlan::Merge(m) => {
            visit_bindings(&m.match_branch, out);
            visit_bindings(&m.create_branch, out);
        }
        // ADR-192 (#623): a CALL{} node's OUTPUT bindings = the driving
        // input's bindings + the body's returned columns (the body's
        // INTERNAL bindings do NOT escape the scoping fence). Mirrors
        // `lowering::collect_bindings`.
        LogicalPlan::Call(c) => {
            visit_bindings(&c.input, out);
            for b in &c.returned {
                out.insert(*b);
            }
        }
        LogicalPlan::CorrelationSeed(s) => {
            for b in &s.imported {
                out.insert(*b);
            }
        }
        LogicalPlan::Empty(_) => {}
    }
}

fn visit_hybrid_operands(
    operands: &[HybridOperand],
    out: &mut std::collections::BTreeSet<BindingId>,
) {
    for op in operands {
        out.insert(op.var);
    }
}

/// Compute the [`JoinCondition`] for a join between two binding sets:
/// the set of bindings present on BOTH sides (i.e., the natural-join
/// keys per the M4-31 lowering convention). Returns `None` for
/// disjoint sets — the DP rejects Cartesian candidates during
/// enumeration except when the input plan was originally Cartesian
/// (handled separately in [`dp`]).
///
/// Used by [`dp::enumerate`] and re-exported `pub(crate)` so the
/// integration tests can pin the convention.
pub(crate) fn join_condition_for(
    left: &std::collections::BTreeSet<BindingId>,
    right: &std::collections::BTreeSet<BindingId>,
) -> Vec<BindingId> {
    let mut shared: Vec<BindingId> = left.intersection(right).copied().collect();
    shared.sort_by_key(|b| b.raw());
    shared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::{BoundExpression, BoundProjectionItem, BoundProjectionKind};
    use arcgraph_core::Lsn;

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn scan(label_raw: u32, var_raw: u64) -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(label_raw)),
            var: BindingId::new(var_raw),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    }

    fn join(left: LogicalPlan, right: LogicalPlan, on: Vec<BindingId>) -> LogicalPlan {
        LogicalPlan::Join(LogicalJoin {
            left: Box::new(left),
            right: Box::new(right),
            on: JoinCondition::SharedBindings(on),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        })
    }

    /// Sanity check: `enumerate_join_order` is identity on a plan
    /// with no joins.
    #[test]
    fn enumerate_no_joins_returns_input_unchanged() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let plan = LogicalPlan::Project(LogicalProject {
            input: Box::new(scan(1, 0)),
            items: Vec::new(),
            span: span(),
        });
        let original = plan.clone();
        let out = enumerate_join_order(plan, &cat);
        assert_eq!(out, original);
    }

    /// 2-way join: enumeration MAY swap left/right but MUST preserve
    /// SharedBindings semantically.
    #[test]
    fn enumerate_two_way_join_preserves_join_keys() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 50);
        let plan = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let out = enumerate_join_order(plan, &cat);
        match out {
            LogicalPlan::Join(j) => match j.on {
                JoinCondition::SharedBindings(ids) => {
                    assert_eq!(ids, vec![BindingId::new(0)]);
                }
            },
            _ => panic!("expected Join at root"),
        }
    }

    /// Frozen catalog returns the captured snapshot regardless of
    /// underlying mutations. Pins the snapshot-once contract.
    #[test]
    fn frozen_catalog_returns_captured_snapshot() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100);
        let snap_before = cat.snapshot();
        let frozen = FrozenCatalog::new(&cat, snap_before.clone());
        // Two snapshots through the frozen wrapper are byte-identical
        // by construction (Clone of the same captured value).
        let s1 = frozen.snapshot();
        let s2 = frozen.snapshot();
        assert_eq!(s1.total_nodes(), s2.total_nodes());
        assert_eq!(s1.total_nodes(), Some(1_000));
        assert_eq!(s1.label_card(LabelId::new(1)), Some(100));
    }

    /// `bindings_in` recursively collects all binding IDs introduced
    /// in a sub-tree.
    #[test]
    fn bindings_in_collects_scan_and_expand_endpoints() {
        let plan = LogicalPlan::Filter(LogicalFilter {
            input: Box::new(LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(1),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: Some(BindingId::new(3)),
                span: span(),
            })),
            predicate: BoundExpression::Literal {
                value: crate::ast::Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        });
        let s = bindings_in(&plan);
        assert!(s.contains(&BindingId::new(1)));
        assert!(s.contains(&BindingId::new(2)));
        assert!(s.contains(&BindingId::new(3)));
        assert_eq!(s.len(), 3);
    }

    /// A passthrough projection item `VariableRef(input_id)` whose
    /// binder-assigned output column is `output_id` — the `WITH a` shape
    /// (`a` read under its pre-projection id, re-emitted under a fresh id,
    /// #746).
    fn with_item(input_id: u64, output_id: u64) -> BoundProjectionItem {
        BoundProjectionItem {
            kind: BoundProjectionKind::Expr(BoundExpression::VariableRef {
                name: "a".into(),
                binding_id: BindingId::new(input_id),
                span: span(),
                type_info: None,
            }),
            alias: None,
            output_id: Some(BindingId::new(output_id)),
            source_text: None,
            span: span(),
        }
    }

    /// #841: a renaming `Project` (`WITH a`: input id 0 → output id 1)
    /// reports its OUTPUT id, NOT its input — so a downstream re-reference
    /// `MATCH (a)` (bound to the projected id 1) shares a non-empty join
    /// key with it. The prior input-recursion reported {0}, giving an
    /// EMPTY `join_condition_for` with the pattern's {1} → a silent
    /// Cartesian (the #841 ×|outer| inflation).
    #[test]
    fn bindings_in_project_reports_output_id_not_input() {
        let project = LogicalPlan::Project(LogicalProject {
            input: Box::new(scan(1, 0)),  // Scan binds `a` = id 0
            items: vec![with_item(0, 1)], // WITH a: reads 0, emits 1
            span: span(),
        });
        let s = bindings_in(&project);
        assert!(
            s.contains(&BindingId::new(1)),
            "projected output id 1 present"
        );
        assert!(
            !s.contains(&BindingId::new(0)),
            "pre-projection input id 0 is NOT in the output schema"
        );
        assert_eq!(s.len(), 1);

        // The anti-Cartesian: a downstream `Scan` re-referencing the
        // renamed `a` (id 1) now shares the join key {1} — pre-fix this
        // intersection was empty ({0} ∩ {1}).
        let downstream = bindings_in(&scan(1, 1));
        assert_eq!(
            join_condition_for(&s, &downstream),
            vec![BindingId::new(1)],
            "WITH-renamed Project joins the re-reference on id 1, NOT Cartesian"
        );
    }

    /// #841 twin: a renaming `Aggregate` (`WITH a, count(b) AS n`) reports
    /// its group-by + aggregation OUTPUT ids (the `AggregateOp` schema),
    /// NOT its input — same anti-Cartesian property for the group key.
    #[test]
    fn bindings_in_aggregate_reports_output_ids_not_input() {
        let agg = LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(scan(1, 0)),     // input `a` = id 0
            group_by: vec![with_item(0, 1)], // group key `a`: 0 → 1
            aggregations: vec![AggregationSpec {
                function: AggregationKind::Count,
                arg: BoundExpression::VariableRef {
                    name: "b".into(),
                    binding_id: BindingId::new(2),
                    span: span(),
                    type_info: None,
                },
                output_id: BindingId::new(5),
                alias: Some("n".into()),
                distinct: false,
                star: false,
                span: span(),
            }],
            span: span(),
        });
        let s = bindings_in(&agg);
        assert!(
            s.contains(&BindingId::new(1)),
            "group-key output id 1 present"
        );
        assert!(
            s.contains(&BindingId::new(5)),
            "aggregation output id 5 present"
        );
        assert!(
            !s.contains(&BindingId::new(0)),
            "pre-aggregation input id 0 is NOT in the output schema"
        );
        assert!(
            !s.contains(&BindingId::new(2)),
            "the aggregated arg id 2 is consumed, NOT an output column"
        );
        assert_eq!(s.len(), 2);
    }

    /// `join_condition_for` returns the sorted intersection of two
    /// binding sets.
    #[test]
    fn join_condition_for_returns_sorted_intersection() {
        let l: std::collections::BTreeSet<_> =
            [BindingId::new(7), BindingId::new(3), BindingId::new(1)]
                .into_iter()
                .collect();
        let r: std::collections::BTreeSet<_> =
            [BindingId::new(3), BindingId::new(7), BindingId::new(9)]
                .into_iter()
                .collect();
        let shared = join_condition_for(&l, &r);
        assert_eq!(shared, vec![BindingId::new(3), BindingId::new(7)]);
    }

    /// Disjoint binding sets produce empty intersection.
    #[test]
    fn join_condition_for_disjoint_returns_empty() {
        let l: std::collections::BTreeSet<_> =
            [BindingId::new(1), BindingId::new(2)].into_iter().collect();
        let r: std::collections::BTreeSet<_> =
            [BindingId::new(3), BindingId::new(4)].into_iter().collect();
        assert!(join_condition_for(&l, &r).is_empty());
    }

    /// **Bushy-deferral pin (per ADR-038 amendment-02 §M4.e v1.1
    /// deferral).** A bushy input plan `(A⨝B)⨝(C⨝D)` is enumerated
    /// to a left-deep output. The DP never produces a bushy
    /// candidate — the right side of every Join is a singleton
    /// leaf. v1.1 lifts this restriction.
    #[test]
    fn enumerate_emits_left_deep_not_bushy() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 100)
            .with_label_cardinality(LabelId::new(3), 100)
            .with_label_cardinality(LabelId::new(4), 100);
        // Bushy input: (A⨝B) ⨝ (C⨝D), all sharing var=0.
        let lhs = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let rhs = join(scan(3, 0), scan(4, 0), vec![BindingId::new(0)]);
        let plan = join(lhs, rhs, vec![BindingId::new(0)]);
        let out = enumerate_join_order(plan, &cat);
        // Walk the output and verify NO Join node has a Join on its
        // right side (left-deep invariant).
        fn assert_left_deep(plan: &LogicalPlan) {
            if let LogicalPlan::Join(j) = plan {
                assert!(
                    !matches!(*j.right, LogicalPlan::Join(_)),
                    "DP must produce a left-deep tree (right side must be a singleton leaf, not a Join); v1.1 lifts this"
                );
                assert_left_deep(&j.left);
            }
        }
        assert_left_deep(&out);
    }

    /// **Wrapper pass-through pin (exhaustive-match contract).** A
    /// `Filter` wrapping a 3-way Join sub-tree is preserved at the
    /// root after enumeration; the inner Join is reordered. Mirrors
    /// the rewriter's exhaustive match — a regression that drops
    /// the wrapper would surface here.
    #[test]
    fn enumerate_preserves_filter_wrapper() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 200)
            .with_label_cardinality(LabelId::new(3), 50);
        let inner_join = join(
            join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]),
            scan(3, 0),
            vec![BindingId::new(0)],
        );
        let plan = LogicalPlan::Filter(LogicalFilter {
            input: Box::new(inner_join),
            predicate: BoundExpression::Literal {
                value: crate::ast::Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        });
        let out = enumerate_join_order(plan, &cat);
        // Filter at the root preserved; its input is a (possibly
        // re-ordered) Join-rooted sub-tree.
        match out {
            LogicalPlan::Filter(f) => {
                assert!(matches!(*f.input, LogicalPlan::Join(_)));
            }
            _ => panic!("expected Filter at root"),
        }
    }

    /// **Wrapper pass-through pin: Project + Sort + Limit nested
    /// wrappers.** Tests that the rewriter's exhaustive match
    /// recurses through multiple unary wrappers.
    #[test]
    fn enumerate_preserves_nested_unary_wrappers() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 50);
        let inner_join = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let plan = LogicalPlan::Project(LogicalProject {
            input: Box::new(LogicalPlan::Limit(LogicalLimit {
                input: Box::new(LogicalPlan::Sort(LogicalSort {
                    input: Box::new(inner_join),
                    order_by: Vec::new(),
                    span: span(),
                })),
                count: 10,
                span: span(),
            })),
            items: Vec::new(),
            span: span(),
        });
        let out = enumerate_join_order(plan, &cat);
        // Verify the wrapper chain Project→Limit→Sort→Join is
        // preserved.
        match out {
            LogicalPlan::Project(p) => match *p.input {
                LogicalPlan::Limit(l) => match *l.input {
                    LogicalPlan::Sort(s) => assert!(matches!(*s.input, LogicalPlan::Join(_))),
                    _ => panic!("expected Sort"),
                },
                _ => panic!("expected Limit"),
            },
            _ => panic!("expected Project at root"),
        }
    }

    /// **OPTIONAL MATCH boundary pin (Cypher 9 §6.5).** A
    /// `LeftOuterJoin` is preserved at its position; the DP does NOT
    /// reorder across the outer-join boundary. Inner-join sub-trees
    /// inside each side ARE optimized.
    #[test]
    fn enumerate_preserves_left_outer_join_boundary() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 50)
            .with_label_cardinality(LabelId::new(3), 25);
        // (Inner Join on (A,B)) LEFT OUTER (C)
        let inner = join(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]);
        let plan = LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
            left: Box::new(inner),
            right: Box::new(scan(3, 0)),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            span: span(),
        });
        let out = enumerate_join_order(plan, &cat);
        // The root remains LeftOuterJoin (boundary preserved).
        match out {
            LogicalPlan::LeftOuterJoin(j) => {
                // The left side was an inner join cluster; it gets
                // re-rooted (still a Join, possibly with reordered
                // operands).
                assert!(matches!(*j.left, LogicalPlan::Join(_)));
                // Right side is the scan(3) leaf — preserved.
                assert!(matches!(*j.right, LogicalPlan::Scan(_)));
            }
            _ => panic!("LeftOuterJoin must be preserved at the boundary"),
        }
    }
}
