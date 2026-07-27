//! M4-31 + M4-32 + M4-33 LogicalPlan lowering pass.
//!
//! [`LogicalPlanLoweringVisitor::lower`] consumes a `BoundQuery`
//! (post-M4-23 cross-substrate validation) and produces a
//! [`LogicalPlan`] tree per ADR-038 §2 D-24 (M4-31 baseline), §2 D-26
//! (M4-32 hybrid retrieval + OPTIONAL MATCH), and §2 D-28 (M4-33
//! aggregation + sort + path operators).
//!
//! # Walker shape — CUSTOM, not a trait (7-slice 3-strike compliance)
//!
//! `LogicalPlanLoweringVisitor` walks `BoundQuery` through dedicated
//! `lower_*` methods. **There is no trait abstraction.** This is the
//! 7th slice in the M4-2x / M4-3x family that ships (or extends) a
//! custom walker over the query tree:
//!
//! - M4-21 [`crate::semantic::BindingVisitor`] (custom);
//! - M4-22 [`crate::semantic::TypeCheckVisitor`] (custom; the
//!   speculative `BoundAstVisitor` trait shipped briefly was deleted
//!   in PR #165 reviewer Finding 1 fix-up);
//! - M4-22b binding-time `may_be_null` refinement (reused
//!   `BindingVisitor`; no trait);
//! - M4-23 [`crate::semantic::CrossSubstrateValidator`] (custom);
//! - M4-31 [`LogicalPlanLoweringVisitor`] (custom — original);
//! - M4-32 EXTENDS [`LogicalPlanLoweringVisitor`] additively (custom;
//!   NO trait abstraction);
//! - M4-33 EXTENDS [`LogicalPlanLoweringVisitor`] additively (custom;
//!   NO trait abstraction — this file).
//!
//! Per ADR-038 §2 D-23 + D-24 + D-26 + D-28, M4-05 inherits the same
//! constraint: any future walker over `BoundQuery` or
//! [`LogicalPlan`] ships as a custom struct unless ≥2 real consumers
//! within the same slice justify the trait abstraction. See
//! `feedback_avoid_speculative_scaffolding.md`.
//!
//! # Operator scope (post-M4-33)
//!
//! - SIMPLE operators (M4-31): [`LogicalPlan::Scan`],
//!   [`LogicalPlan::Expand`], [`LogicalPlan::Filter`],
//!   [`LogicalPlan::Project`], [`LogicalPlan::Join`],
//!   [`LogicalPlan::Limit`], [`LogicalPlan::Skip`],
//!   [`LogicalPlan::Empty`].
//! - HYBRID retrieval (M4-32): [`LogicalPlan::RankByHybrid`],
//!   [`LogicalPlan::Fusion`], [`LogicalPlan::CommunityLookup`],
//!   [`LogicalPlan::VectorNear`], [`LogicalPlan::TextMatch`].
//! - OPTIONAL MATCH (M4-32): [`LogicalPlan::LeftOuterJoin`].
//! - Aggregation + sort + path (M4-33): [`LogicalPlan::Aggregate`],
//!   [`LogicalPlan::Sort`], [`LogicalPlan::Distinct`],
//!   [`LogicalPlan::Unwind`], [`LogicalPlan::NamedPath`],
//!   [`LogicalPlan::DynamicLimit`].
//!
//! Surfaces still deferred emit
//! [`LogicalPlanError::NotImplementedAtM4_31`] with the
//! `target_slice` slot naming the future slice (v1.1 / v1.2 surfaces
//! only after M4-33).
//!
//! # Anonymous-binding handling
//!
//! Node patterns and rel patterns may omit the variable
//! (`(p:Person)-[:R]->()` — the trailing `()` is anonymous). The M4-21
//! binding pass does NOT allocate a `BindingId` for anonymous patterns
//! — `BoundNodePattern::var = None` / `BoundRelPattern::var = None`.
//! M4-31 needs an identity for plan-tree shape, so it allocates
//! synthetic `BindingId`s starting at `max(observed binding IDs) + 1`.
//! These IDs are namespace-disjoint from the binding-pass IDs and are
//! local to a single `LogicalPlan` tree.
//!
//! # Error accumulation
//!
//! The pass does NOT short-circuit on the first error: it surfaces
//! every lowering fault in a single walk, matching the M4-21 / M4-22
//! / M4-23 discipline. Multiple unsupported surfaces in one query are
//! all reported together.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 — logical-plan-lowering contract (this file's
//!   primary spec).
//! - ADR-038 §2 D-23 — visitor-trait discipline lock (M4-31 inherits;
//!   this file ships a CUSTOM walker).
//! - ADR-038 amendment-03 §M4-31 row — test-artifact pin (8 unit + 4
//!   integration + 1 proptest).

use arcgraph_core::{LabelId, Lsn};

use crate::ast::{BinOp, LengthRange, Literal, OrderDirection, UnaryOp};
use crate::error::Span;
use crate::logical_plan::error::LogicalPlanError;
use crate::logical_plan::types::{
    AggregationKind, AggregationSpec, CountStoreSource, DeleteKind, Direction, DynamicLimitKind,
    FusionKind, FusionSpec, HybridOperand, HybridOperandKind, JoinAlgorithm, JoinCondition,
    LogicalAggregate, LogicalCall, LogicalCommunityLookup, LogicalCorrelationSeed,
    LogicalCountStore, LogicalCreateEndpoint, LogicalCreateNode, LogicalCreatePropertyIndex,
    LogicalCreateRel, LogicalCreateVectorIndex, LogicalDelete, LogicalDeleteItem, LogicalDistinct,
    LogicalDynamicLimit, LogicalEmpty, LogicalExpand, LogicalFilter, LogicalFusion, LogicalJoin,
    LogicalLeftOuterJoin, LogicalLimit, LogicalMerge, LogicalNamedPath, LogicalPlan,
    LogicalProcedureCall, LogicalProject, LogicalPropertyIndexScan, LogicalRankByHybrid,
    LogicalRemove, LogicalRemoveItem, LogicalRemoveMutation, LogicalScan, LogicalSet,
    LogicalSetItem, LogicalSetMutation, LogicalSkip, LogicalSort, LogicalTextMatch, LogicalUnion,
    LogicalUnwind, LogicalVectorNear, MergeKeySpec, OrderByItem, PathAlgorithm,
    PlainPathSegmentShape, PlainPathShape, ProcedureSource, SetTargetKind, SortDirection,
};
use crate::semantic::bound_ast::{
    BindingId, BoundCallClause, BoundCallProcedureClause, BoundClause, BoundCreateClause,
    BoundCreateItem, BoundCreateNodeSpec, BoundCreatePathSpec, BoundDeleteClause, BoundExpression,
    BoundFieldRef, BoundFusion, BoundMapProjectionItem, BoundMatchBody, BoundMatchClause,
    BoundMergeClause, BoundMergePattern, BoundNamedPath, BoundNamedPathKind, BoundNodePattern,
    BoundOrderItem, BoundPathPattern, BoundProjectionItem, BoundProjectionKind, BoundQuery,
    BoundRankArg, BoundRankByClause, BoundRanker, BoundRelPattern, BoundRemoveClause,
    BoundRemoveMutation, BoundReturnClause, BoundSetClause, BoundSetItem, BoundSetMutation,
    BoundShowClause, BoundStatement, BoundUnionQuery, BoundUnionTail, BoundUnwindClause,
    BoundWithClause, BoundWithFusionClause, CreateEndpointBinding, TypeInfo,
};
use crate::semantic::error::ArcQLError;

// Aggregation function detection lives at the type level via
// [`AggregationKind::from_function_name`] (resolved from the v1.0
// [`crate::semantic::functions`] aggregation registry). M4-33 uses
// the resolver directly; the M4-31-era aggregation-name constant
// has been removed in favor of the resolver.

/// M4-31 LogicalPlan lowering pass.
///
/// Construct via [`Self::lower`] (defaults to `Lsn::MAX` —
/// read-latest semantic) or [`Self::lower_at_snapshot`] for a
/// specific `read_lsn`; the internal struct is not part of the
/// public API.
///
/// ## MVCC visibility (per ADR-041 §D-4)
///
/// Every substrate-touching operator the visitor emits
/// (`LogicalScan`, `LogicalVectorNear`, `LogicalTextMatch`,
/// `LogicalCommunityLookup`, `HybridOperand`) carries a
/// `read_lsn: Lsn` field. The visitor populates each of those
/// fields with the value supplied at construction time. The M4
/// executor sources that LSN from the active transaction's
/// `current_lsn()` value at v1.0; the M5 parser surface
/// (`BEGIN AT SNAPSHOT lsn=N`) supplies a fixed historical LSN.
pub struct LogicalPlanLoweringVisitor {
    errors: Vec<LogicalPlanError>,
    /// Counter for synthetic anonymous-pattern binding IDs (seeded
    /// from `max(observed binding ID) + 1`).
    next_anon_id: u64,
    /// MVCC visibility key per ADR-041 §D-4. Stamped on every
    /// substrate-touching `LogicalPlan` operator the visitor
    /// emits. Defaults to `Lsn::MAX` for callers using
    /// [`Self::lower`] (read-latest); pass a concrete LSN via
    /// [`Self::lower_at_snapshot`] to pin a snapshot.
    read_lsn: Lsn,
}

impl LogicalPlanLoweringVisitor {
    /// Lower a [`BoundStatement`] (post-M4-23 cross-substrate
    /// validation) to a [`LogicalPlan`] tree. Returns `Ok(plan)` on
    /// success or `Err(Vec<ArcQLError>)` carrying
    /// `ArcQLError::LogicalPlan(_)` variants for accumulated lowering
    /// faults.
    ///
    /// Stamps every substrate-touching operator with `read_lsn =
    /// Lsn::MAX` (read-latest semantic per ADR-041 §D-2). Callers
    /// with snapshot context use [`Self::lower_at_snapshot`].
    pub fn lower(stmt: &BoundStatement) -> Result<LogicalPlan, Vec<ArcQLError>> {
        Self::lower_at_snapshot(stmt, Lsn::MAX)
    }

    /// Lower with an explicit `read_lsn` per ADR-041 §D-4. Every
    /// substrate-touching operator the visitor emits gets stamped
    /// with this value. The executor passes it through to vector
    /// / BM25 / community substrate calls so each substrate's
    /// visibility filter resolves to the same snapshot.
    ///
    /// Production callers source `read_lsn` from
    /// `TransactionManager::current_lsn()` at the executor's
    /// transaction-begin boundary; tests construct an explicit
    /// `Lsn::new(N)` to pin specific snapshots.
    pub fn lower_at_snapshot(
        stmt: &BoundStatement,
        read_lsn: Lsn,
    ) -> Result<LogicalPlan, Vec<ArcQLError>> {
        let mut v = Self {
            errors: Vec::new(),
            next_anon_id: 0,
            read_lsn,
        };
        let plan = v.lower_statement(stmt);
        if v.errors.is_empty() {
            Ok(plan)
        } else {
            Err(v.errors.into_iter().map(ArcQLError::from).collect())
        }
    }

    // ---------- Statement / query dispatch ----------

    fn lower_statement(&mut self, stmt: &BoundStatement) -> LogicalPlan {
        match stmt {
            BoundStatement::Read(q) => self.lower_query(q),
            // ADR-185 (#649-A1, W28) — UNION / UNION ALL.
            BoundStatement::Union(u) => self.lower_union(u),
            // #830 (ADR-198 §OQ-7 / ADR-200) — Neo4j-compat index DDL.
            // CREATE VECTOR INDEX lowers to the accept-and-register
            // write op (the OPTIONS map is carried verbatim; the op
            // resolves dims / similarity / `$name` against the parameter
            // bag at execute-time). DROP INDEX remains a typed
            // NotImplemented (lifecycle is a vector-track follow-up; the
            // #830 langchain happy path never DROPs) — the type-check
            // pass already rejects it, so this is the defensive twin.
            BoundStatement::IndexDdl(d) => {
                use crate::ast::IndexDdlStatement;
                match d {
                    IndexDdlStatement::CreateVector(c) => {
                        LogicalPlan::CreateVectorIndex(LogicalCreateVectorIndex {
                            name: c.name.clone(),
                            if_not_exists: c.if_not_exists,
                            label: c.label.clone(),
                            property: c.property.clone(),
                            options: c.options.clone(),
                            span: Span::point(1, 1),
                        })
                    }
                    IndexDdlStatement::CreateProperty(c) => {
                        // #1366 (task #248, Phase 1) — lower to the
                        // property-index CREATE op. Registration + backfill
                        // + Online-flip happen at execute-time.
                        LogicalPlan::CreatePropertyIndex(LogicalCreatePropertyIndex {
                            name: c.name.clone(),
                            if_not_exists: c.if_not_exists,
                            label: c.label.clone(),
                            property: c.property.clone(),
                            span: Span::point(1, 1),
                        })
                    }
                    IndexDdlStatement::Drop(_) => {
                        self.errors.push(LogicalPlanError::NotImplementedAtM4_31 {
                            surface: "DROP INDEX statement (index lifecycle)",
                            target_slice:
                                "#830 vector grammar surface; lifecycle owned by vector track (ADR-198 §OQ-7)",
                            span: Span::point(1, 1),
                        });
                        LogicalPlan::Empty(LogicalEmpty {
                            span: Span::point(1, 1),
                        })
                    }
                }
            }
        }
    }

    fn lower_query(&mut self, q: &BoundQuery) -> LogicalPlan {
        // Seed the anonymous-binding counter from `max(observed) + 1`
        // so synthetic IDs never collide with binding-pass IDs.
        let max_observed = collect_max_binding_id(q);
        self.next_anon_id = max_observed.saturating_add(1);

        if q.clauses.is_empty() {
            return LogicalPlan::Empty(LogicalEmpty {
                span: q.span.clone(),
            });
        }

        let mut current: Option<LogicalPlan> = None;
        for c in &q.clauses {
            current = Some(self.lower_clause(c, current));
        }
        current.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: q.span.clone(),
            })
        })
    }

    // ---------- UNION / UNION ALL (ADR-185 §8) ----------

    /// Lower a bound `UNION` / `UNION ALL` per ADR-185 (#649, W28 —
    /// openCypher v9 §8).
    ///
    /// Both kinds lower to a [`LogicalUnion`] concat of the lowered arms
    /// (streaming, O(1); the per-arm `column_orders` realign differently-
    /// ordered arms to arm-0's canonical column order). They differ only
    /// in dedup:
    ///
    /// - **`UNION ALL`** (every boundary `ALL`) — the concat IS the
    ///   result (keep duplicates).
    /// - **bare `UNION`** (every boundary distinct) — wrap the concat in
    ///   a [`LogicalDistinct`] (full-row dedup over the union output
    ///   columns) per the PE FROZEN CONTRACT item 2: dedup is the
    ///   standalone `DistinctOp` composed OVER `UnionOp`, NOT buried in
    ///   it. This is the **#649-A2** scope (#649-A1 shipped the concat +
    ///   the standalone `DistinctOp` and emitted a structured deferral
    ///   here; A2 lifts the deferral by composing the two ops). See
    ///   ADR-185 §Consequences ("#649-A2 is unblocked and minimal: wrap
    ///   `LogicalUnion` in `LogicalDistinct` …").
    ///
    /// The no-MIXING rule (openCypher §8 — `UNION` and `UNION ALL` cannot
    /// be combined; TCK `Union3` → `InvalidClauseComposition`) is enforced
    /// at bind time ([`crate::semantic::BindingVisitor`]'s
    /// `check_union_no_mixing` → `BindingError::UnionMixedSetOps`), so by
    /// the time a [`BoundUnionQuery`] reaches here `all` is uniform — the
    /// only two cases are all-`ALL` and all-distinct. There is no
    /// per-boundary dedup grouping. (We read `is_union_all` from `all[0]`
    /// defensively via `iter().all`; an empty `all` — a degenerate
    /// single-arm union the grammar never produces — falls into the
    /// `UNION ALL` branch as a pass-through concat.)
    ///
    /// In ALL cases the post-union tail (ORDER BY → SKIP → LIMIT) binds
    /// the WHOLE result (the RC-2 fix) — for bare `UNION` it wraps the
    /// `Distinct`, so the tail applies AFTER dedup per §8 (`… UNION … LIMIT n`
    /// limits the deduped row set, not the pre-dedup concat).
    fn lower_union(&mut self, u: &BoundUnionQuery) -> LogicalPlan {
        let is_union_all = u.all.iter().all(|&a| a);

        // Lower each arm + concat (identical for both kinds).
        let arms: Vec<LogicalPlan> = u.arms.iter().map(|arm| self.lower_query(arm)).collect();
        let union = LogicalPlan::Union(LogicalUnion {
            arms,
            column_orders: u.column_orders.clone(),
            span: u.span.clone(),
        });

        // Bare UNION (distinct): wrap the concat in a LogicalDistinct
        // (#649-A2). The DistinctOp executor dedups on the FULL output
        // row (`canonical_row_key`), which is exactly §8 row-equality
        // over every union output column; the `on` slot is the forward
        // cost-planner hint (which bindings identify a duplicate),
        // populated — mirroring `lower_distinct` — from arm-0's terminal
        // RETURN projection bindings (arm-0 is the canonical column
        // source per the binder's `check_union_column_compat`).
        let combined = if is_union_all {
            union
        } else {
            LogicalPlan::Distinct(LogicalDistinct {
                input: Box::new(union),
                on: self.union_output_dedup_key(u),
                span: u.span.clone(),
            })
        };

        // Post-union tail, bound to the WHOLE combined result (the RC-2
        // fix). For bare UNION this is OVER the Distinct, so the tail
        // applies post-dedup per §8.
        self.lower_union_tail(&u.tail, &u.span, combined)
    }

    /// The `LogicalDistinct::on` dedup-key hint for a bare `UNION` —
    /// the bindings referenced by arm-0's terminal RETURN projection
    /// (arm-0 is the union's canonical column source per the binder's
    /// `check_union_column_compat`). Mirrors [`Self::lower_distinct`]'s
    /// `on` derivation so UNION-distinct and `RETURN DISTINCT` agree on
    /// the hint's shape. The executor's `DistinctOp` dedups on the full
    /// row regardless of `on` (it is a forward planner hint, per
    /// `LogicalDistinct::on` docs), so this does not change semantics —
    /// it keeps the logical plan honest for the M4-05 cost planner.
    fn union_output_dedup_key(&self, u: &BoundUnionQuery) -> Vec<BindingId> {
        let mut on: std::collections::BTreeSet<BindingId> = std::collections::BTreeSet::new();
        if let Some(first_arm) = u.arms.first() {
            if let Some(BoundClause::Return(r)) = first_arm
                .clauses
                .iter()
                .rev()
                .find(|c| matches!(c, BoundClause::Return(_)))
            {
                for it in &r.items {
                    if let BoundProjectionKind::Expr(e) = &it.kind {
                        collect_referenced_bindings(e, &mut on);
                    }
                }
            }
        }
        on.into_iter().collect()
    }

    /// Apply the post-union ORDER BY / SKIP / LIMIT tail over `input`.
    /// Reuses the read-query tail lowering primitives
    /// ([`Self::lower_order_by`] / [`Self::lower_skip_or_limit`]) so the
    /// produced Sort / Skip / Limit / DynamicLimit nodes are IDENTICAL
    /// in shape to the standalone-query tail — only the POSITION (over
    /// the whole union) differs. Order mirrors the read-query tail:
    /// Sort → Skip → Limit.
    fn lower_union_tail(
        &mut self,
        tail: &BoundUnionTail,
        span: &Span,
        input: LogicalPlan,
    ) -> LogicalPlan {
        let mut current = input;
        if !tail.order_by.is_empty() {
            current = self.lower_order_by(&tail.order_by, span, current);
        }
        if let Some(skip) = &tail.skip {
            current = self.lower_skip_or_limit(skip, DynamicLimitKind::Skip, current);
        }
        if let Some(limit) = &tail.limit {
            current = self.lower_skip_or_limit(limit, DynamicLimitKind::Limit, current);
        }
        current
    }

    // ---------- Clause dispatch ----------

    fn lower_clause(&mut self, c: &BoundClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        match c {
            BoundClause::Match(m) => self.lower_match(m, prev),
            BoundClause::With(w) => self.lower_with(w, prev),
            BoundClause::Return(r) => self.lower_return(r, prev),
            BoundClause::TailSkip(e, span) => self.lower_tail_skip(e, span, prev),
            BoundClause::TailLimit(e, span) => self.lower_tail_limit(e, span, prev),

            // ---- M4-32: hybrid retrieval lowering ----
            BoundClause::RankBy(r) => self.lower_rank_by(r, prev),
            BoundClause::WithFusion(f) => self.lower_with_fusion(f, prev),

            // ---- M4-33: ORDER BY (tail clause) ----
            BoundClause::TailOrderBy(items, span) => self.lower_tail_order_by(items, span, prev),

            // ---- M4-33: UNWIND ----
            BoundClause::Unwind(u) => self.lower_unwind(u, prev),

            // ---- ADR-192 (#623): CALL { <subquery> } ----
            BoundClause::Call(c) => self.lower_call(c, prev),

            // ---- ADR-197 (#802): CALL <proc>(…) [YIELD …] / SHOW … ----
            BoundClause::CallProcedure(c) => self.lower_call_procedure(c, prev),
            BoundClause::Show(s) => self.lower_show(s, prev),

            // ---- ADR-147 W26-θ Phase 1: CREATE node ----
            BoundClause::Create(c) => self.lower_create(c, prev),

            // ---- ADR-149 W26-θ Phase 3: DELETE ----
            BoundClause::Delete(d) => self.lower_delete(d, prev),

            // ---- ADR-150 W26-θ Phase 4: SET / REMOVE ----
            BoundClause::Set(s) => self.lower_set(s, prev),
            BoundClause::Remove(r) => self.lower_remove(r, prev),

            // ---- ADR-151 W26-θ Phase 5: MERGE ----
            BoundClause::Merge(m) => self.lower_merge(m, prev),
        }
    }

    // ADR-147 W26-θ Phase 1 — CREATE node lowering.
    //
    // Each CreateItem::Node lowers to a `LogicalPlan::CreateNode`.
    // Multi-item `CREATE (a), (b)` builds a left-deep chain:
    //
    //     (prev?) → CreateNode(a) → CreateNode(b) → ...
    //
    // The chain's leaf is `prev` when present (e.g., a future
    // `MATCH ... CREATE ...` combo at Phase 2/3+) or `LogicalEmpty`
    // when CREATE is the leading clause of the query. The executor
    // walks the chain bottom-up: the LogicalEmpty / prev emits zero
    // or more "trigger rows", then each CreateNode op consumes the
    // input row, performs the write, and emits one row carrying the
    // new binding(s).
    //
    // At Phase 1 the typical entry shape is `CREATE (n:User {...})
    // RETURN n` with no prior MATCH — so `prev = None` and the
    // chain's leaf is a `LogicalEmpty` whose `next_batch` returns
    // exactly one empty row (the trigger). The CreateNode op then
    // produces one row. This matches the openCypher v9 §6
    // "single-trigger" semantic for leading CREATEs.
    fn lower_create(&mut self, c: &BoundCreateClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        // `chain` accumulates the left-deep CREATE-item chain: each
        // item's `input` is the sub-plan for the items already
        // processed, so EVERY item executes. This is the issue #832
        // fix — the prior code did `current = Some(op)` each
        // iteration, OVERWRITING the accumulator and keeping ONLY the
        // last item, so `CREATE (:T{n:1}),(:T{n:2}),(:T{n:3})`
        // silently persisted only `{n:3}` (count=1, collect=[3]).
        //
        // Phase-5 forward-pin (#1123): seed the CREATE chain from the
        // prior row stream when present. Each write op is already a
        // streaming transform, so this composes CREATE per upstream row
        // while standalone CREATE still self-seeds as a single shot.
        let mut chain: Option<LogicalPlan> = prev;
        for item in &c.items {
            match item {
                BoundCreateItem::Node(spec) => {
                    let var = spec.var.as_ref().map(|v| v.binding_id);
                    let label = spec.label.clone();
                    let properties: Vec<(String, BoundExpression)> = spec
                        .properties
                        .as_ref()
                        .map(|m| {
                            m.entries
                                .iter()
                                .map(|e| (e.key.clone(), e.value.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    // #832: thread the chain-so-far as this item's
                    // `input`, then make this item the new chain top.
                    // The executor streams one create per upstream row
                    // and emits the row extended with the new binding.
                    let op = LogicalPlan::CreateNode(LogicalCreateNode {
                        var,
                        label,
                        properties,
                        input: chain.take().map(Box::new),
                        span: spec.span.clone(),
                    });
                    chain = Some(op);
                }
                // ADR-148 W26-θ Phase 2 — CREATE-path lowering.
                //
                // Builds the sub-plan tree:
                //
                //     CreateRel { source_plan, target_plan, ... }
                //         ├─ source_plan: CreateNode(source)
                //         └─ target_plan: CreateNode(target)
                //
                // The executor (Pipeline::build) materializes both
                // sub-plans as concrete CreateNodeOps; the CreateRelOp
                // pulls one row from each to resolve the source +
                // target NodeIds, then writes the rel via the
                // substrate's create_rel.
                //
                // Phase-5 composition (parallel to Phase 1): the
                // chain's top is the CreateRel. The item-chain input
                // carries `prev` when this CREATE follows a row stream,
                // so the write executes once per upstream row.
                //
                // We synthesize a fresh binding for source / target if
                // the source AST didn't bind a variable (CreateRel
                // requires concrete `source` + `target` BindingIds to
                // route the upstream row's NodeIds).
                BoundCreateItem::Path(path) => {
                    let source_visible = path.source.var.is_some();
                    let source_var = match path.source.var.as_ref() {
                        Some(v) => v.binding_id,
                        None => self.fresh_anon_binding_id(),
                    };
                    let source_label = path.source.label.clone();
                    let source_props = bound_props_to_vec(path.source.properties.as_ref());
                    let source_plan = LogicalPlan::CreateNode(LogicalCreateNode {
                        var: Some(source_var),
                        label: source_label,
                        properties: source_props,
                        // Endpoint producer — driven by CreateRelOp's
                        // internal pull, NOT part of the item-chain.
                        input: None,
                        span: path.source.span.clone(),
                    });
                    let target_visible = path.target.var.is_some();
                    let target_var = match path.target.var.as_ref() {
                        Some(v) => v.binding_id,
                        None => self.fresh_anon_binding_id(),
                    };
                    let target_label = path.target.label.clone();
                    let target_props = bound_props_to_vec(path.target.properties.as_ref());
                    let target_plan = LogicalPlan::CreateNode(LogicalCreateNode {
                        var: Some(target_var),
                        label: target_label,
                        properties: target_props,
                        input: None,
                        span: path.target.span.clone(),
                    });
                    let rel_var = path.rel.var.as_ref().map(|v| v.binding_id);
                    let rel_label = path.rel.label.clone();
                    let rel_props = bound_props_to_vec(path.rel.properties.as_ref());
                    // #832: thread the chain-so-far as this path's
                    // `input` so a prior CREATE item still executes.
                    let rel_op = LogicalPlan::CreateRel(LogicalCreateRel {
                        var: rel_var,
                        label: rel_label,
                        properties: rel_props,
                        source_plan: Box::new(source_plan),
                        source: source_var,
                        source_visible,
                        source_endpoint: lower_create_endpoint(path.source.endpoint_binding),
                        target_plan: Box::new(target_plan),
                        target: target_var,
                        target_visible,
                        target_endpoint: lower_create_endpoint(path.target.endpoint_binding),
                        direction: path.rel.direction.clone(),
                        input: chain.take().map(Box::new),
                        span: path.rel.span.clone(),
                    });
                    chain = Some(rel_op);
                }
            }
        }
        chain.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: c.span.clone(),
            })
        })
    }

    // ADR-149 W26-θ Phase 3 — DELETE lowering.
    //
    // Wraps the prior MATCH-produced row stream in a
    // `LogicalPlan::Delete`. Each `BoundDeleteItem` lowers to a
    // `LogicalDeleteItem` whose `kind` is derived from the resolved
    // `BoundVariable::type_info` populated by the M4-22 type-check
    // pass (Node → DeleteKind::Node; Relationship → DeleteKind::Rel).
    //
    // If `prev` is `None` (DELETE without prior MATCH — the
    // type-checker has already rejected this case because every
    // item must resolve to a prior-bound variable, but the lowering
    // defends in depth), we wrap `LogicalEmpty` as the input.
    //
    // If an item's type_info is missing / mismatched (e.g., the
    // M4-22 pass has not run, or the resolution failed), we
    // conservatively default to DeleteKind::Node + accumulate a
    // lowering error (the executor would surface a runtime fault
    // otherwise; the error here is defense-in-depth — the canonical
    // diagnostic is the type-check pass's DeleteNonGraphValue).
    fn lower_delete(&mut self, d: &BoundDeleteClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: d.span.clone(),
            })
        });
        let items: Vec<LogicalDeleteItem> = d
            .items
            .iter()
            .map(|item| {
                let kind = match item.var.type_info.as_ref() {
                    Some(TypeInfo::Node { .. }) => DeleteKind::Node,
                    Some(TypeInfo::Relationship { .. }) => DeleteKind::Rel,
                    // The type-check pass already rejected non-graph
                    // bindings; the binding pass also rejected
                    // unresolved references. We conservatively
                    // default to Node so the lowering produces a
                    // structurally complete plan — the upstream
                    // diagnostics carry the canonical fault.
                    _ => DeleteKind::Node,
                };
                LogicalDeleteItem {
                    binding: item.var.binding_id,
                    kind,
                    span: item.span.clone(),
                }
            })
            .collect();
        LogicalPlan::Delete(LogicalDelete {
            input: Box::new(input),
            items,
            detach: d.detach,
            span: d.span.clone(),
        })
    }

    // ADR-150 W26-θ Phase 4 — SET lowering.
    //
    // Wraps the prior MATCH-produced row stream in a
    // `LogicalPlan::Set`. Each `BoundSetItem` lowers to a
    // `LogicalSetItem` whose `kind` derives from the resolved
    // `BoundVariable::type_info` populated by the M4-22 type-check
    // pass (Node → SetTargetKind::Node; Relationship →
    // SetTargetKind::Rel; other → conservatively Node + the type-
    // check already emitted SetRemoveNonGraphValue per ADR-150 §D-4).
    //
    // If `prev` is `None` (SET without prior MATCH), wrap
    // `LogicalEmpty` as the input — the type-check pass has ALREADY
    // rejected this case because every item must resolve to a prior-
    // bound variable, but the lowering defends in depth.
    fn lower_set(&mut self, s: &BoundSetClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: s.span.clone(),
            })
        });
        let items: Vec<LogicalSetItem> = s
            .items
            .iter()
            .map(|item| {
                let kind = match item.var.type_info.as_ref() {
                    Some(TypeInfo::Node { .. }) => SetTargetKind::Node,
                    Some(TypeInfo::Relationship { .. }) => SetTargetKind::Rel,
                    _ => SetTargetKind::Node,
                };
                let mutation = match &item.mutation {
                    BoundSetMutation::PropertyAssign { name, value } => {
                        LogicalSetMutation::PropertyAssign {
                            name: name.clone(),
                            value: value.clone(),
                        }
                    }
                    BoundSetMutation::PropertyReplace(map) => LogicalSetMutation::PropertyReplace(
                        map.entries
                            .iter()
                            .map(|e| (e.key.clone(), e.value.clone()))
                            .collect(),
                    ),
                    BoundSetMutation::PropertyMerge(map) => LogicalSetMutation::PropertyMerge(
                        map.entries
                            .iter()
                            .map(|e| (e.key.clone(), e.value.clone()))
                            .collect(),
                    ),
                    BoundSetMutation::LabelAdd(labels) => {
                        LogicalSetMutation::LabelAdd(labels.clone())
                    }
                };
                LogicalSetItem {
                    binding: item.var.binding_id,
                    kind,
                    mutation,
                    span: item.span.clone(),
                }
            })
            .collect();
        LogicalPlan::Set(LogicalSet {
            input: Box::new(input),
            items,
            span: s.span.clone(),
        })
    }

    // ADR-151 W26-θ Phase 5 — MERGE lowering.
    //
    // Build both branches of the match-or-create operator:
    //
    // - **match_branch:** synthesize the LogicalPlan that scans /
    //   expands the merge pattern in the current snapshot. For
    //   Node-shape: a `LogicalScan` (label-filtered if the pattern
    //   carried a label). For Path-shape: a `Scan + Expand` chain
    //   (parallel to `lower_path_pattern`'s machinery).
    // - **create_branch:** synthesize the LogicalPlan that creates
    //   the merge pattern atomically. For Node-shape: a
    //   `LogicalCreateNode` (parallel to `lower_create`'s Node
    //   branch). For Path-shape: a `LogicalCreateRel` wrapping
    //   `LogicalCreateNode` source + target (parallel to
    //   `lower_create`'s Path branch).
    //
    // Action items (on_create / on_match) lower to `LogicalSetItem`
    // (parallel to `lower_set`'s per-item lowering).
    //
    // The lowering DISCARDS `prev` (parallel to ADR-147 + ADR-148
    // CREATE narrowing) — MATCH→MERGE per-row composition is
    // forward-pinned to v1.1 per ADR-151 §"Forward-deferred".
    fn lower_merge(&mut self, m: &BoundMergeClause, _prev: Option<LogicalPlan>) -> LogicalPlan {
        let match_branch = self.lower_merge_match_branch(&m.pattern, &m.span);
        let create_branch = self.lower_merge_create_branch(&m.pattern, &m.span);
        let on_create = m
            .on_create
            .iter()
            .map(|it| self.lower_merge_action_item(it))
            .collect();
        let on_match = m
            .on_match
            .iter()
            .map(|it| self.lower_merge_action_item(it))
            .collect();
        // ADR-151-amendment-01 §D-1/§D-3 — RETURN-after-MERGE emission
        // discriminator. `Some(binding)` ONLY for a node-shape NAMED
        // merge (the binding the binder minted via `bind_create_node_spec`,
        // which BOTH the match-scan and the create-branch thread); `None`
        // for anonymous node merges (no binding to project) and ALL
        // path-shape merges (the match `[source, rel, target]` and create
        // `[rel]` schemas are un-unionable — RC-3 keeps path terminal).
        // The executor emits the binding row(s) iff this is `Some`.
        let output_binding = match &m.pattern {
            BoundMergePattern::Node(spec) => spec.var.as_ref().map(|v| v.binding_id),
            BoundMergePattern::Path(_) => None,
        };
        // NN-4 (#1384) — build the get-or-create serialization key(s) from
        // the merge pattern. The property VALUES are threaded as
        // `BoundExpression` (not resolved literals) because a MERGE
        // property may be a parameter (`MERGE (u:User {email:$e})`),
        // evaluated only at execute time.
        //
        // - Node-shape → ONE key (the node's label + property set).
        // - Path-shape → TWO keys (source + target endpoints — NN-4
        //   re-spin Fix 3). The executor acquires them in canonical total
        //   order (sorted) so two path-MERGEs naming the same endpoints in
        //   opposite pattern order cannot inter-path deadlock. Anonymous
        //   endpoints (no label AND no properties) contribute no useful
        //   identity so are skipped; an anonymous node-shape merge
        //   (`MERGE (:Label)`) still lowers to no output binding but DOES
        //   carry a label key so a labelled anonymous merge serializes.
        let merge_keys = match &m.pattern {
            BoundMergePattern::Node(spec) => vec![MergeKeySpec {
                label: spec.label.clone(),
                properties: bound_props_to_vec(spec.properties.as_ref()),
            }],
            BoundMergePattern::Path(path) => {
                // Endpoint identity = label + inline property set (the same
                // shape the match branch filters on). An endpoint with
                // NEITHER a label NOR properties carries no distinguishing
                // identity to lock on (a bare `(a)`), so it is omitted; a
                // path with two such endpoints yields an empty key vec and
                // runs unserialized (there is nothing to serialize on).
                let mut keys = Vec::new();
                for ep in [&path.source, &path.target] {
                    let props = bound_props_to_vec(ep.properties.as_ref());
                    if ep.label.is_some() || !props.is_empty() {
                        keys.push(MergeKeySpec {
                            label: ep.label.clone(),
                            properties: props,
                        });
                    }
                }
                keys
            }
        };
        LogicalPlan::Merge(LogicalMerge {
            match_branch: Box::new(match_branch),
            create_branch: Box::new(create_branch),
            on_create,
            on_match,
            output_binding,
            merge_keys,
            span: m.span.clone(),
        })
    }

    // Build the match-branch sub-plan for the merge pattern. Reuses
    // the existing Phase 0 read-side machinery (`LogicalScan` /
    // `LogicalExpand` / `LogicalEmpty`). Post-ADR-152 the property-bag
    // round-trip persists, so the property-filter narrows the
    // production match-branch by property (ADR-152 §D-4); post-ADR-152-
    // amendment-01 the match-branch also enforces the node / path-source
    // LABEL via the binder-resolved `match_label_id` (§D-2/§D-3), so
    // `MERGE (n:User …)` no longer cross-matches a different label. The
    // residual v1.2 strict-schema property TYPING (typed-tuple encoding)
    // is tracked at issue #356 — distinct from this label-enforcement
    // fix, which needs no catalog index.
    fn lower_merge_match_branch(
        &mut self,
        pattern: &BoundMergePattern,
        span: &Span,
    ) -> LogicalPlan {
        match pattern {
            BoundMergePattern::Node(spec) => self.lower_merge_node_scan(spec),
            BoundMergePattern::Path(path) => self.lower_merge_path_scan(path, span),
        }
    }

    fn lower_merge_node_scan(&mut self, spec: &BoundCreateNodeSpec) -> LogicalPlan {
        // ADR-152-amendment-01 §D-2/§D-3 — MERGE match-branch LABEL
        // enforcement, pulled forward to v1.0-α (lifts ADR-152
        // §"Forward-deferred" #8). The binder resolves the match-side
        // label NAME → `Option<LabelId>` at bind time (None-tolerant —
        // see `BoundCreateNodeSpec::match_label_id`), so the lowering
        // visitor does NOT need a CatalogProvider handle. Three cases:
        let var = match spec.var.as_ref() {
            Some(v) => v.binding_id,
            None => self.fresh_anon_binding_id(),
        };
        if let Some(label_id) = spec.match_label_id {
            // (1) Label present AND interned → reuse MATCH's proven
            // `Scan{label: Some(id)}` path (the production substrate's
            // `scan_nodes(tenant, Some(id), lsn)` filters
            // `rec.label_id != id`). When the pattern also carries
            // inline properties (`MERGE (n:User {id:42})`) the existing
            // property-filter wrap (ADR-152 §D-4) narrows further; a
            // bare `MERGE (n:User)` lowers to `Scan{Some(id)}` alone
            // (the wrap is a no-op on an empty property bag) and now
            // correctly enforces the label.
            let scan = LogicalPlan::Scan(LogicalScan {
                label: Some(label_id),
                var,
                read_lsn: self.read_lsn,
                span: spec.span.clone(),
            });
            wrap_create_node_spec_with_property_filter(scan, spec, var)
        } else if spec.label.is_some() {
            // (2) Label present but NOT interned ⇒ no live node can
            // carry it ⇒ the match-branch is PROVABLY EMPTY. Lower to
            // `LogicalPlan::Empty` (O(1) EOS — `EmptyOp` never touches
            // the substrate; NOT a `Scan{None}` + const-false filter,
            // which would be an O(node_high_water) scan-and-discard —
            // scalability-day-zero). `MergeOp` pulls the match-branch
            // to exhaustion → empty → fires the create-branch, which
            // mints the label by name at execute-time.
            LogicalPlan::Empty(LogicalEmpty {
                span: spec.span.clone(),
            })
        } else {
            // (3) No label at all (`MERGE (n {id:42})`) — label-agnostic
            // by design: match ANY node satisfying the property
            // predicate, regardless of label. Unchanged v1.0-α behavior:
            // `Scan{None}` + the property-filter wrap (ADR-152 §D-4).
            let scan = LogicalPlan::Scan(LogicalScan {
                label: None,
                var,
                read_lsn: self.read_lsn,
                span: spec.span.clone(),
            });
            wrap_create_node_spec_with_property_filter(scan, spec, var)
        }
    }

    fn lower_merge_path_scan(&mut self, path: &BoundCreatePathSpec, _span: &Span) -> LogicalPlan {
        // ADR-152-amendment-01 §D-2/§D-3/§D-5 — enforce the path SOURCE
        // node label (same binder-resolved `match_label_id` mechanism
        // as `lower_merge_node_scan`).
        //
        // SCOPE BOUNDARY (amendment-01 §D-5): the path TARGET label +
        // the rel-TYPE are NOT enforced at v1.0-α. The target arrives
        // via `Expand`+`Join` (not a `Scan`, so `Scan{label}` cannot
        // reach it), the `Expand` carries `rel_type: None`, and the
        // property-filter (`properties_to_filter_predicate`) synthesizes
        // only `n.<key> = <lit>` predicates — it cannot express a
        // label or rel-type predicate. Path-target label + rel-type
        // enforcement is forward-pinned to v1.1+ (needs a label-aware
        // `Expand` or a new label/rel-type predicate kind).
        //
        // ADR-152 §D-4 — source + target node patterns may carry inline
        // property literals; wrap each sub-plan with the property-filter.
        let source_var = match path.source.var.as_ref() {
            Some(v) => v.binding_id,
            None => self.fresh_anon_binding_id(),
        };
        if path.source.match_label_id.is_none() && path.source.label.is_some() {
            // Source label present but NOT interned ⇒ no source node
            // can exist ⇒ the entire path match-branch is provably
            // empty ⇒ the create-branch fires (mints the labels by
            // name). O(1) `Empty` — see `lower_merge_node_scan` case 2.
            return LogicalPlan::Empty(LogicalEmpty {
                span: path.span.clone(),
            });
        }
        let mut source_scan: LogicalPlan = LogicalPlan::Scan(LogicalScan {
            // `Some(id)` when the source label is interned; `None` when
            // the source pattern carried no label (label-agnostic).
            label: path.source.match_label_id,
            var: source_var,
            read_lsn: self.read_lsn,
            span: path.source.span.clone(),
        });
        source_scan =
            wrap_create_node_spec_with_property_filter(source_scan, &path.source, source_var);
        let target_var = match path.target.var.as_ref() {
            Some(v) => v.binding_id,
            None => self.fresh_anon_binding_id(),
        };
        let rel_var = path.rel.var.as_ref().map(|v| v.binding_id);
        // Map AST `CreateRelDirection` to logical-plan `Direction`
        // for the match-branch Expand. Phase 2 CREATE narrowing
        // (per ADR-148 §D-1) admits only LeftToRight / RightToLeft;
        // Undirected forward-pinned per ADR-148 §"Forward-deferred".
        let direction = match path.rel.direction {
            crate::ast::CreateRelDirection::LeftToRight => Direction::LeftToRight,
            crate::ast::CreateRelDirection::RightToLeft => Direction::RightToLeft,
        };
        let expand = LogicalPlan::Expand(LogicalExpand {
            from: source_var,
            to: target_var,
            direction,
            rel_type: None,
            length_range: None,
            rel_var,
            span: path.rel.span.clone(),
        });
        let join = LogicalPlan::Join(LogicalJoin {
            left: Box::new(source_scan),
            right: Box::new(expand),
            on: JoinCondition::SharedBindings(vec![source_var]),
            algorithm: JoinAlgorithm::Auto,
            span: path.span.clone(),
        });
        // ADR-152 §D-4 — wrap the joined plan with a target-side
        // property-filter so the match-branch's joined-side rows are
        // narrowed by the target node pattern's literals.
        wrap_create_node_spec_with_property_filter(join, &path.target, target_var)
    }

    // Build the create-branch sub-plan for the merge pattern.
    // Reuses the existing Phase 1 / Phase 2 CREATE lowering shape
    // (`LogicalCreateNode` / `LogicalCreateRel`).
    fn lower_merge_create_branch(
        &mut self,
        pattern: &BoundMergePattern,
        _span: &Span,
    ) -> LogicalPlan {
        match pattern {
            BoundMergePattern::Node(spec) => {
                let var = spec.var.as_ref().map(|v| v.binding_id);
                let label = spec.label.clone();
                let properties = bound_props_to_vec(spec.properties.as_ref());
                LogicalPlan::CreateNode(LogicalCreateNode {
                    var,
                    label,
                    properties,
                    // MERGE create-branch is driven by MergeOp, not the
                    // CREATE-item chain (#832).
                    input: None,
                    span: spec.span.clone(),
                })
            }
            BoundMergePattern::Path(path) => {
                // Build source + target sub-plans (each a CreateNode
                // mirror of the merge pattern's source / target
                // CreateNodeSpec — parallel to `lower_create`'s Path
                // branch).
                let source_visible = path.source.var.is_some();
                let source_var = match path.source.var.as_ref() {
                    Some(v) => v.binding_id,
                    None => self.fresh_anon_binding_id(),
                };
                let source_props = bound_props_to_vec(path.source.properties.as_ref());
                let source_plan = LogicalPlan::CreateNode(LogicalCreateNode {
                    var: Some(source_var),
                    label: path.source.label.clone(),
                    properties: source_props,
                    input: None,
                    span: path.source.span.clone(),
                });
                let target_visible = path.target.var.is_some();
                let target_var = match path.target.var.as_ref() {
                    Some(v) => v.binding_id,
                    None => self.fresh_anon_binding_id(),
                };
                let target_props = bound_props_to_vec(path.target.properties.as_ref());
                let target_plan = LogicalPlan::CreateNode(LogicalCreateNode {
                    var: Some(target_var),
                    label: path.target.label.clone(),
                    properties: target_props,
                    input: None,
                    span: path.target.span.clone(),
                });
                let rel_var = path.rel.var.as_ref().map(|v| v.binding_id);
                let rel_label = path.rel.label.clone();
                let rel_props = bound_props_to_vec(path.rel.properties.as_ref());
                LogicalPlan::CreateRel(LogicalCreateRel {
                    var: rel_var,
                    label: rel_label,
                    properties: rel_props,
                    source_plan: Box::new(source_plan),
                    source: source_var,
                    source_visible,
                    source_endpoint: LogicalCreateEndpoint::Fresh,
                    target_plan: Box::new(target_plan),
                    target: target_var,
                    target_visible,
                    target_endpoint: LogicalCreateEndpoint::Fresh,
                    direction: path.rel.direction.clone(),
                    // MERGE create-branch — not part of an item-chain.
                    input: None,
                    span: path.rel.span.clone(),
                })
            }
        }
    }

    // Lower a Phase 4 BoundSetItem to a Phase 4 LogicalSetItem
    // (parallel to `lower_set`'s per-item lowering machinery).
    fn lower_merge_action_item(&mut self, item: &BoundSetItem) -> LogicalSetItem {
        let kind = match item.var.type_info.as_ref() {
            Some(TypeInfo::Node { .. }) => SetTargetKind::Node,
            Some(TypeInfo::Relationship { .. }) => SetTargetKind::Rel,
            _ => SetTargetKind::Node,
        };
        let mutation = match &item.mutation {
            BoundSetMutation::PropertyAssign { name, value } => {
                LogicalSetMutation::PropertyAssign {
                    name: name.clone(),
                    value: value.clone(),
                }
            }
            BoundSetMutation::PropertyReplace(map) => LogicalSetMutation::PropertyReplace(
                map.entries
                    .iter()
                    .map(|e| (e.key.clone(), e.value.clone()))
                    .collect(),
            ),
            BoundSetMutation::PropertyMerge(map) => LogicalSetMutation::PropertyMerge(
                map.entries
                    .iter()
                    .map(|e| (e.key.clone(), e.value.clone()))
                    .collect(),
            ),
            BoundSetMutation::LabelAdd(labels) => LogicalSetMutation::LabelAdd(labels.clone()),
        };
        LogicalSetItem {
            binding: item.var.binding_id,
            kind,
            mutation,
            span: item.span.clone(),
        }
    }

    // ADR-150 W26-θ Phase 4 — REMOVE lowering. Symmetric to
    // `lower_set` — see that fn for the kind-derivation rationale.
    fn lower_remove(&mut self, r: &BoundRemoveClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: r.span.clone(),
            })
        });
        let items: Vec<LogicalRemoveItem> = r
            .items
            .iter()
            .map(|item| {
                let kind = match item.var.type_info.as_ref() {
                    Some(TypeInfo::Node { .. }) => SetTargetKind::Node,
                    Some(TypeInfo::Relationship { .. }) => SetTargetKind::Rel,
                    _ => SetTargetKind::Node,
                };
                let mutation = match &item.mutation {
                    BoundRemoveMutation::Property(name) => {
                        LogicalRemoveMutation::Property(name.clone())
                    }
                    BoundRemoveMutation::LabelRemove(labels) => {
                        LogicalRemoveMutation::LabelRemove(labels.clone())
                    }
                };
                LogicalRemoveItem {
                    binding: item.var.binding_id,
                    kind,
                    mutation,
                    span: item.span.clone(),
                }
            })
            .collect();
        LogicalPlan::Remove(LogicalRemove {
            input: Box::new(input),
            items,
            span: r.span.clone(),
        })
    }

    // ---------- MATCH lowering ----------

    fn lower_match(&mut self, m: &BoundMatchClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let pattern_subtree = match &m.body {
            BoundMatchBody::Patterns(ps) => self.lower_patterns(ps, &m.span),
            BoundMatchBody::NamedPath(np) => self.lower_named_path(np, &m.span),
        };

        // Apply pattern-local WHERE conjuncts before joining with prev —
        // keeps the filter close to its input for M4-05 push-down. Any
        // conjunct that references carried pipeline bindings must wait
        // until the join has restored those bindings to the row schema.
        let (pushdown_filters, post_join_filters) = match &m.where_clause {
            Some(pred) => {
                let mut pattern_bindings = std::collections::BTreeSet::new();
                collect_bindings(&pattern_subtree, &mut pattern_bindings);
                split_match_where_filters(pred, &pattern_bindings)
            }
            None => (Vec::new(), Vec::new()),
        };
        let filtered = self.apply_where_conjuncts(pattern_subtree, &pushdown_filters);

        // OPTIONAL MATCH lowers to LeftOuterJoin per ADR-006
        // amendment-01 §A-2. The first clause in a query that is an
        // OPTIONAL MATCH (`prev = None`) is a left-outer-join over the
        // IMPLICIT SINGLE-ROW (unit) driving table — openCypher 9 §6.5:
        // a leading OPTIONAL MATCH over a graph with no match still emits
        // exactly ONE row with the optional variables bound to null. We
        // root it on the leading-clause [`LogicalEmpty`] unit row (the
        // SAME idiom leading UNWIND / CALL{} / RETURN-1 use; see
        // [`Self::lower_unwind`] and the `EmptyOp::unit()` default in
        // `pipeline.rs`). The shared-binding set is empty (the unit row
        // shares no bindings with the pattern), so the join degenerates to
        // "pass the right rows through, or emit one null-extended row if
        // the right side is empty" — exactly the openCypher leading-OPTIONAL
        // semantics. (#996-followup: the prior `None => filtered` shortcut
        // dropped the unit row and returned 0 rows over an empty graph.)
        if m.is_optional {
            let joined = match prev {
                None => LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
                    left: Box::new(LogicalPlan::Empty(LogicalEmpty {
                        span: m.span.clone(),
                    })),
                    right: Box::new(filtered),
                    // No shared bindings: the unit-row left and the pattern
                    // share nothing — a Cartesian unit join that null-extends
                    // when the right side is empty.
                    on: JoinCondition::SharedBindings(Vec::new()),
                    span: m.span.clone(),
                }),
                Some(p) => {
                    let shared = shared_bindings(&p, &filtered);
                    LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
                        left: Box::new(p),
                        right: Box::new(filtered),
                        on: JoinCondition::SharedBindings(shared),
                        span: m.span.clone(),
                    })
                }
            };
            return self.apply_where_conjuncts(joined, &post_join_filters);
        }

        // Combine with previous plan (multi-MATCH chain). Inner join
        // on shared bindings (variables that appear in both subtrees).
        let joined = match prev {
            None => filtered,
            Some(p) => {
                let shared = shared_bindings(&p, &filtered);
                LogicalPlan::Join(LogicalJoin {
                    left: Box::new(p),
                    right: Box::new(filtered),
                    on: JoinCondition::SharedBindings(shared),
                    algorithm: JoinAlgorithm::Auto,
                    span: m.span.clone(),
                })
            }
        };
        self.apply_where_conjuncts(joined, &post_join_filters)
    }

    /// Apply a WHERE predicate to a pattern subtree, recognizing
    /// substrate-bearing shapes that should lower to dedicated
    /// retrieval operators rather than a generic Filter.
    ///
    /// Recognized shapes (M4-32):
    /// - `n IN COMMUNITY($cid)` → [`LogicalCommunityLookup`].
    /// - `community(n) = $cid` / `$cid = community(n)` →
    ///   [`LogicalCommunityLookup`] (IDENTICAL tree to predicate
    ///   form modulo span coordinates per ADR-038 §2 D-26 — closes
    ///   PR #154 reviewer Finding 5).
    /// - `<expr> NEAR <expr>` → [`LogicalVectorNear`] when LHS is a
    ///   property access on a bound variable.
    /// - `<expr> MATCH <expr>` → [`LogicalTextMatch`] when LHS is a
    ///   property access on a bound variable.
    /// - everything else → generic [`LogicalFilter`].
    fn apply_where(&mut self, input: LogicalPlan, pred: &BoundExpression) -> LogicalPlan {
        if let Some(community) = self.try_lower_community_predicate(&input, pred) {
            return community;
        }
        if let Some(near) = self.try_lower_near_predicate(&input, pred) {
            return near;
        }
        if let Some(text) = self.try_lower_text_match_predicate(&input, pred) {
            return text;
        }
        LogicalPlan::Filter(LogicalFilter {
            input: Box::new(input),
            predicate: pred.clone(),
            span: pred.span().clone(),
        })
    }

    /// **#1290** — apply a split WHERE conjunct set to `input`, BOUNDING
    /// the resulting plan-tree depth.
    ///
    /// Up to [`MAX_FILTER_CHAIN_NODES`] generic conjuncts keep the
    /// per-conjunct [`LogicalFilter`] wrapping (byte-identical plan
    /// shape to the pre-#1290 fold: per-predicate SIMD detection in the
    /// executor, one EXPLAIN row per conjunct). BEYOND that, the
    /// generic conjuncts fold into ONE `LogicalFilter` carrying their
    /// left-nested `AND` spine, which every post-#1290 expression walk
    /// (bind / type-check / cost / eval / Display) handles iteratively.
    ///
    /// Why: the per-conjunct fold made PLAN depth equal the conjunct
    /// count, and the plan-tree passes recurse per node (join-order
    /// `rewrite`, algorithm picking, cost walk, pipeline compile, the
    /// executor's per-batch operator pull, plan `Drop` glue) — a
    /// 200-predicate flat WHERE overflowed the debug-profile stack in
    /// `enumeration::rewrite` even with every EXPRESSION walk
    /// iterative. Bounding the emitted Filter-chain length kills that
    /// whole class at the source instead of despining six independent
    /// plan walkers plus the runtime pull chain.
    ///
    /// Semantics: `Filter(Filter(x, p1), p2)` and
    /// `Filter(x, p1 AND p2)` admit exactly the same rows (a row
    /// passes iff every conjunct is `true` — Cypher 3VL `AND` is
    /// `true` iff both operands are `true`). The only observable
    /// difference is evaluation strictness on DROPPED rows (the fused
    /// spine evaluates all conjuncts; chained filters stop at the
    /// first non-pass) — openCypher does not guarantee conjunct
    /// evaluation order, and the strict behavior already matches
    /// RETURN-position `AND` evaluation.
    ///
    /// Hybrid-shaped conjuncts (`NEAR` / text `MATCH` / community
    /// membership — see [`is_hybrid_predicate_shape`]) NEVER fold:
    /// they must reach [`Self::apply_where`]'s dedicated-operator
    /// recognizers individually (a `Near` inside a generic Filter
    /// predicate is un-executable). They are applied first (closest to
    /// the pattern); relative order among them is preserved.
    /// Conjunction commutes, so the reorder cannot change the row set.
    fn apply_where_conjuncts(
        &mut self,
        input: LogicalPlan,
        preds: &[BoundExpression],
    ) -> LogicalPlan {
        let generic_count = preds
            .iter()
            .filter(|p| !is_hybrid_predicate_shape(p))
            .count();
        if generic_count <= MAX_FILTER_CHAIN_NODES {
            return preds
                .iter()
                .fold(input, |plan, pred| self.apply_where(plan, pred));
        }
        let mut plan = input;
        let mut generic: Vec<&BoundExpression> = Vec::new();
        for pred in preds {
            if is_hybrid_predicate_shape(pred) {
                plan = self.apply_where(plan, pred);
            } else {
                generic.push(pred);
            }
        }
        let predicate = fold_and_spine(&generic)
            .expect("generic_count > MAX_FILTER_CHAIN_NODES >= 1, so `generic` is non-empty");
        let span = predicate.span().clone();
        LogicalPlan::Filter(LogicalFilter {
            input: Box::new(plan),
            predicate,
            span,
        })
    }

    /// Try to recognize a community-membership predicate.
    ///
    /// Both surfaces yield IDENTICAL trees rooted at
    /// [`LogicalPlan::CommunityLookup`] (modulo span coordinates of
    /// the wrapper node — the carried `node_var` BindingId and
    /// `community_id` expression are byte-equal between the two
    /// surfaces). Per ADR-038 §2 D-26 + amendment-01 §A-2.
    fn try_lower_community_predicate(
        &mut self,
        input: &LogicalPlan,
        pred: &BoundExpression,
    ) -> Option<LogicalPlan> {
        // Predicate shape: `n IN COMMUNITY($cid)`.
        if let BoundExpression::InCommunity {
            node,
            community,
            span,
            ..
        } = pred
        {
            let node_var = match node.as_ref() {
                BoundExpression::VariableRef { binding_id, .. } => *binding_id,
                _ => return None,
            };
            return Some(LogicalPlan::CommunityLookup(LogicalCommunityLookup {
                input: Box::new(input.clone()),
                node_var,
                community_id: (**community).clone(),
                read_lsn: self.read_lsn,
                span: span.clone(),
            }));
        }
        // Canonical shape: `community(n) = $cid` or `$cid = community(n)`.
        if let BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs,
            rhs,
            span,
            ..
        } = pred
        {
            if let Some((node_var, community_id)) = match_community_equality(lhs, rhs) {
                return Some(LogicalPlan::CommunityLookup(LogicalCommunityLookup {
                    input: Box::new(input.clone()),
                    node_var,
                    community_id,
                    read_lsn: self.read_lsn,
                    span: span.clone(),
                }));
            }
        }
        None
    }

    /// Try to recognize a `<var>.<prop> NEAR <expr>` predicate and
    /// lower it to [`LogicalPlan::VectorNear`] composed onto the
    /// pattern subtree as an outer Filter-like wrapper. Bare-shape
    /// (no explicit K) sets `k = 0` per [`LogicalVectorNear::k`]
    /// docs.
    fn try_lower_near_predicate(
        &mut self,
        input: &LogicalPlan,
        pred: &BoundExpression,
    ) -> Option<LogicalPlan> {
        if let BoundExpression::Near {
            lhs, target, span, ..
        } = pred
        {
            if let Some((var, property)) = match_property_access(lhs) {
                let near = LogicalPlan::VectorNear(LogicalVectorNear {
                    var,
                    property,
                    query_vector: (**target).clone(),
                    k: 0,
                    read_lsn: self.read_lsn,
                    span: span.clone(),
                });
                // Wrap the input by joining on the bound variable.
                return Some(LogicalPlan::Join(LogicalJoin {
                    left: Box::new(input.clone()),
                    right: Box::new(near),
                    on: JoinCondition::SharedBindings(vec![var]),
                    algorithm: JoinAlgorithm::Auto,
                    span: span.clone(),
                }));
            }
        }
        // `vector_distance(field, query)` function-call shape.
        if let BoundExpression::FunctionCall {
            name, args, span, ..
        } = pred
        {
            if name.eq_ignore_ascii_case("vector_distance") && args.len() == 2 {
                if let Some((var, property)) = match_property_access(&args[0]) {
                    let near = LogicalPlan::VectorNear(LogicalVectorNear {
                        var,
                        property,
                        query_vector: args[1].clone(),
                        k: 0,
                        read_lsn: self.read_lsn,
                        span: span.clone(),
                    });
                    return Some(LogicalPlan::Join(LogicalJoin {
                        left: Box::new(input.clone()),
                        right: Box::new(near),
                        on: JoinCondition::SharedBindings(vec![var]),
                        algorithm: JoinAlgorithm::Auto,
                        span: span.clone(),
                    }));
                }
            }
        }
        None
    }

    /// Try to recognize a `<var>.<prop> MATCH <expr>` predicate and
    /// lower it to [`LogicalPlan::TextMatch`] composed onto the
    /// pattern subtree as an outer Join wrapper.
    fn try_lower_text_match_predicate(
        &mut self,
        input: &LogicalPlan,
        pred: &BoundExpression,
    ) -> Option<LogicalPlan> {
        if let BoundExpression::TextMatch {
            lhs, query, span, ..
        } = pred
        {
            if let Some((var, property)) = match_property_access(lhs) {
                let text = LogicalPlan::TextMatch(LogicalTextMatch {
                    var,
                    property,
                    query_text: (**query).clone(),
                    k: None,
                    read_lsn: self.read_lsn,
                    span: span.clone(),
                });
                return Some(LogicalPlan::Join(LogicalJoin {
                    left: Box::new(input.clone()),
                    right: Box::new(text),
                    on: JoinCondition::SharedBindings(vec![var]),
                    algorithm: JoinAlgorithm::Auto,
                    span: span.clone(),
                }));
            }
        }
        // `text_match(field, query)` function-call shape.
        if let BoundExpression::FunctionCall {
            name, args, span, ..
        } = pred
        {
            if name.eq_ignore_ascii_case("text_match") && args.len() == 2 {
                if let Some((var, property)) = match_property_access(&args[0]) {
                    let text = LogicalPlan::TextMatch(LogicalTextMatch {
                        var,
                        property,
                        query_text: args[1].clone(),
                        k: None,
                        read_lsn: self.read_lsn,
                        span: span.clone(),
                    });
                    return Some(LogicalPlan::Join(LogicalJoin {
                        left: Box::new(input.clone()),
                        right: Box::new(text),
                        on: JoinCondition::SharedBindings(vec![var]),
                        algorithm: JoinAlgorithm::Auto,
                        span: span.clone(),
                    }));
                }
            }
        }
        None
    }

    /// Lower a list of path patterns (the multi-pattern MATCH form
    /// `MATCH (a), (b), ...`). Single-pattern MATCH degenerates to
    /// the head case.
    fn lower_patterns(&mut self, patterns: &[BoundPathPattern], match_span: &Span) -> LogicalPlan {
        let mut iter = patterns.iter();
        let first = match iter.next() {
            Some(p) => self.lower_path_pattern(p),
            None => {
                // Defensive: a MATCH with zero patterns shouldn't
                // reach lowering (the parser rejects it).
                self.errors.push(LogicalPlanError::InvalidPlanStructure {
                    reason: "MATCH clause has zero patterns".into(),
                    span: match_span.clone(),
                });
                return LogicalPlan::Empty(LogicalEmpty {
                    span: match_span.clone(),
                });
            }
        };
        iter.fold(first, |acc, p| {
            let right = self.lower_path_pattern(p);
            let shared = shared_bindings(&acc, &right);
            LogicalPlan::Join(LogicalJoin {
                left: Box::new(acc),
                right: Box::new(right),
                on: JoinCondition::SharedBindings(shared),
                algorithm: JoinAlgorithm::Auto,
                span: match_span.clone(),
            })
        })
    }

    /// Lower a single path pattern `(a)-[r:R]->(b)-[s:S]->(c)`.
    ///
    /// - The head node always becomes a [`LogicalScan`]
    ///   (label-filtered if labeled).
    /// - Each `(rel, next_node)` tail step becomes a [`LogicalExpand`]
    ///   joined onto the running subtree on the shared `from`
    ///   binding.
    fn lower_path_pattern(&mut self, p: &BoundPathPattern) -> LogicalPlan {
        self.lower_path_pattern_inner(p, None)
    }

    /// Shared path-pattern lowering. When `path_shape` is `Some`, this is
    /// a `Plain` NAMED path (ADR-193 D-4): every relationship is
    /// FORCE-BOUND a binding (synthesizing one for anonymous rels, which
    /// a plain MATCH would leave unbound) so the executor can materialize
    /// the path, and the ordered element-binding sequence is recorded
    /// into the shape. When `None`, this is a plain MATCH pattern and the
    /// historical behavior is preserved EXACTLY (anonymous rels stay
    /// unbound; no extra schema columns).
    fn lower_path_pattern_inner(
        &mut self,
        p: &BoundPathPattern,
        mut path_shape: Option<&mut PlainPathShape>,
    ) -> LogicalPlan {
        let head_var = self.binding_for_node(&p.head);
        if let Some(sh) = path_shape.as_deref_mut() {
            sh.start = head_var;
        }
        let head_label = single_label(&p.head, &mut self.errors);

        let mut current = LogicalPlan::Scan(LogicalScan {
            label: head_label,
            var: head_var,
            read_lsn: self.read_lsn,
            span: p.head.span.clone(),
        });
        // ADR-152 §D-4 — wrap head Scan with property-filter when the
        // head node pattern carries inline literals like
        // `(n {id:42})`. Closes the MATCH-by-property narrowing.
        current = wrap_node_with_property_filter(current, &p.head, head_var);
        let mut prev_to = head_var;

        for (rel, node) in &p.tail {
            // Reject reserved length-range forms (GQL `{N,M}` is
            // rejected at M4-22; defensive emit here too).
            if let Some(lr) = &rel.length {
                if matches!(lr, LengthRange::Quantified { .. }) {
                    self.errors.push(LogicalPlanError::NotImplementedAtM4_31 {
                        surface: "GQL length-range `{N,M}`",
                        target_slice: "v1.1 (per ADR-038 D-9)",
                        span: rel.span.clone(),
                    });
                }
            }

            let to_var = self.binding_for_node(node);
            let rel_type = single_rel_type(rel, &mut self.errors);
            // ADR-193 D-4 — a Plain NAMED path FORCE-BINDS every rel
            // (synthesizing a binding for anonymous rels) so the path can
            // materialize it. A plain MATCH (`path_shape == None`) keeps
            // bare anonymous rels unbound, but property-bearing anonymous
            // rels still need an internal binding so the inline predicate
            // can inspect the expanded relationship.
            let named_path = path_shape.is_some();
            let anonymous_rel_needs_filter_binding = rel.var.is_none()
                && rel
                    .properties
                    .as_ref()
                    .map(|props| !props.entries.is_empty())
                    .unwrap_or(false);
            let rel_var = match rel.var.as_ref() {
                Some(v) => Some(v.binding_id),
                None if named_path || anonymous_rel_needs_filter_binding => {
                    Some(self.fresh_anon_binding_id())
                }
                None => None,
            };
            let direction = Direction::from(&rel.direction);
            let length_range = rel.length.clone();
            if let Some(sh) = path_shape.as_deref_mut() {
                sh.segments.push(PlainPathSegmentShape {
                    rel: rel_var.expect("named-path rels are force-bound"),
                    end: to_var,
                    var_length: rel.length.is_some(),
                });
            }

            let mut expand: LogicalPlan = LogicalPlan::Expand(LogicalExpand {
                from: prev_to,
                to: to_var,
                direction,
                rel_type,
                length_range,
                rel_var,
                span: rel.span.clone(),
            });
            // ADR-152 §D-4 — wrap Expand with rel-property filter when
            // the rel pattern carries inline literals like
            // `(a)-[r:R {since: 2020}]->(b)` or
            // `(a)-[:R {since: 2020}]->(b)`. Anonymous property-bearing
            // rels use the internal binding allocated above.
            expand = wrap_rel_with_property_filter(expand, rel, rel_var);

            // Combine via Join on the shared `from` binding (the
            // previous-step output produces `prev_to`; the new Expand
            // also references `prev_to` as its `from` endpoint).
            current = LogicalPlan::Join(LogicalJoin {
                left: Box::new(current),
                right: Box::new(expand),
                on: JoinCondition::SharedBindings(vec![prev_to]),
                algorithm: JoinAlgorithm::Auto,
                span: node.span.clone(),
            });

            // If the tail node has a label, intersect with a Scan
            // (Join on the to_var binding) so downstream operators see
            // the label-filtered set.
            if let Some(label) = single_label(node, &mut self.errors) {
                let mut scan: LogicalPlan = LogicalPlan::Scan(LogicalScan {
                    label: Some(label),
                    var: to_var,
                    read_lsn: self.read_lsn,
                    span: node.span.clone(),
                });
                // ADR-152 §D-4 — wrap with property-filter when tail
                // node has property literals like `(b:User {id:42})`.
                scan = wrap_node_with_property_filter(scan, node, to_var);
                current = LogicalPlan::Join(LogicalJoin {
                    left: Box::new(current),
                    right: Box::new(scan),
                    on: JoinCondition::SharedBindings(vec![to_var]),
                    algorithm: JoinAlgorithm::Auto,
                    span: node.span.clone(),
                });
            } else if node
                .properties
                .as_ref()
                .map(|m| !m.entries.is_empty())
                .unwrap_or(false)
            {
                // Tail node has properties but no label — wrap the
                // entire `current` Join with a property-filter
                // predicate. Without a Scan to wrap (no label means
                // no intersect-Scan emit), we filter on `current`
                // directly.
                current = wrap_node_with_property_filter(current, node, to_var);
            }

            prev_to = to_var;
        }
        current
    }

    /// Return the binding id for a node pattern, allocating a synthetic
    /// id for anonymous nodes.
    fn binding_for_node(&mut self, n: &BoundNodePattern) -> BindingId {
        match &n.var {
            Some(v) => v.binding_id,
            None => self.fresh_anon_binding_id(),
        }
    }

    fn fresh_anon_binding_id(&mut self) -> BindingId {
        let id = BindingId::new(self.next_anon_id);
        self.next_anon_id = self.next_anon_id.saturating_add(1);
        id
    }

    // ---------- WITH lowering ----------

    fn lower_with(&mut self, w: &BoundWithClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let mut input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: w.span.clone(),
            })
        });

        let output_ids = with_projection_output_ids(&w.items);
        let has_wildcard = w
            .items
            .iter()
            .any(|i| matches!(i.kind, BoundProjectionKind::Wildcard { .. }));
        let (pre_projection_filters, post_projection_filters) = match &w.where_clause {
            Some(pred) => split_with_where_filters(pred, &output_ids, has_wildcard),
            None => (Vec::new(), Vec::new()),
        };

        // #1290 — bounded per-conjunct wrapping (see
        // `apply_plain_filter_conjuncts`): a wide flat `WITH … WHERE`
        // must not emit one nested Filter per conjunct.
        input = apply_plain_filter_conjuncts(input, &pre_projection_filters);

        // M4-33: aggregation detection. If any WITH item is an
        // aggregation function call, emit Aggregate beneath Project;
        // otherwise emit a bare Project. #746: the aggregate path also
        // REWRITES the Project items to pass the Aggregate's output
        // columns through (see `lower_aggregation_clause`); the
        // non-aggregate path is unchanged (items cloned verbatim).
        let (projection_input, project_items) = if items_contain_aggregation(&w.items) {
            self.lower_aggregation_clause(&w.items, &w.span, input)
        } else {
            (input, w.items.clone())
        };

        let project = LogicalPlan::Project(LogicalProject {
            input: Box::new(projection_input),
            items: project_items,
            span: w.span.clone(),
        });

        // #842 part B — `WITH DISTINCT …` dedups the projected rows. Reuse
        // the SAME `LogicalDistinct` operator `RETURN DISTINCT` / bare-UNION
        // lower to (`lower_distinct`; #622/#649 lit it). Placed DIRECTLY over
        // the Project — the post-Project schema IS the dedup-column set, which
        // is `DistinctOp`'s correctness contract (it dedups the full output
        // row) — and BELOW the post-WITH WHERE filter, so `WITH DISTINCT x
        // WHERE …` dedups before filtering (openCypher v9 §6.4 sub-clause
        // order: projection → DISTINCT → … → WHERE).
        let deduped = if w.distinct {
            self.lower_distinct(&w.items, &w.span, project)
        } else {
            project
        };

        // #1290 — bounded per-conjunct wrapping (see
        // `apply_plain_filter_conjuncts`).
        apply_plain_filter_conjuncts(deduped, &post_projection_filters)
    }

    // ---------- RETURN lowering ----------

    fn lower_return(&mut self, r: &BoundReturnClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: r.span.clone(),
            })
        });

        // M4-33: aggregation detection. If any RETURN item is an
        // aggregation function call, splice an Aggregate node between
        // the input and the Project; otherwise use the input
        // unchanged. #746: the aggregate path also REWRITES the Project
        // items to pass the Aggregate's output columns through (see
        // `lower_aggregation_clause`); the non-aggregate path is
        // unchanged (items cloned verbatim).
        let (projection_input, project_items) = if items_contain_aggregation(&r.items) {
            self.lower_aggregation_clause(&r.items, &r.span, input)
        } else {
            (input, r.items.clone())
        };

        let mut current = LogicalPlan::Project(LogicalProject {
            input: Box::new(projection_input),
            items: project_items,
            span: r.span.clone(),
        });

        // M4-33: ORDER BY (return-clause form). Applied after Project
        // so the sort sees the projected/aliased values per Cypher 9
        // §6.6.
        if !r.order_by.is_empty() {
            current = self.lower_order_by(&r.order_by, &r.span, current);
        }

        // M4-33: DISTINCT (after ORDER BY per Cypher 9 §6.4 — the
        // sort is on pre-distinct rows, but the cardinality post-
        // distinct is what's visible to SKIP / LIMIT). At v1.0 we
        // place DISTINCT after Sort; M4-05 cost-planner is free to
        // commute when safe.
        if r.distinct {
            current = self.lower_distinct(&r.items, &r.span, current);
        }

        // Apply SKIP first (semantically: skip rows, then take).
        if let Some(skip_expr) = &r.skip {
            current = self.lower_skip_or_limit(skip_expr, DynamicLimitKind::Skip, current);
        }

        if let Some(limit_expr) = &r.limit {
            current = self.lower_skip_or_limit(limit_expr, DynamicLimitKind::Limit, current);
        }

        current
    }

    // ---------- Tail SKIP / LIMIT lowering ----------

    fn lower_tail_skip(
        &mut self,
        e: &BoundExpression,
        span: &Span,
        prev: Option<LogicalPlan>,
    ) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| LogicalPlan::Empty(LogicalEmpty { span: span.clone() }));
        self.lower_skip_or_limit_with_span(e, DynamicLimitKind::Skip, input, span)
    }

    fn lower_tail_limit(
        &mut self,
        e: &BoundExpression,
        span: &Span,
        prev: Option<LogicalPlan>,
    ) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| LogicalPlan::Empty(LogicalEmpty { span: span.clone() }));
        self.lower_skip_or_limit_with_span(e, DynamicLimitKind::Limit, input, span)
    }

    /// Lower a SKIP / LIMIT expression. Literal-integer expressions
    /// produce the static [`LogicalPlan::Skip`] / [`LogicalPlan::Limit`]
    /// variants (M4-31 contract preserved); parameter / expression
    /// values produce [`LogicalPlan::DynamicLimit`] (M4-33 surface).
    /// Uses the expression's own span as the produced node's span.
    fn lower_skip_or_limit(
        &mut self,
        expr: &BoundExpression,
        kind: DynamicLimitKind,
        input: LogicalPlan,
    ) -> LogicalPlan {
        let span = expr.span().clone();
        self.lower_skip_or_limit_with_span(expr, kind, input, &span)
    }

    /// As [`Self::lower_skip_or_limit`], but with an explicit span
    /// override (the surrounding clause's span, used by tail-clause
    /// lowering to mirror the M4-31 contract).
    fn lower_skip_or_limit_with_span(
        &mut self,
        expr: &BoundExpression,
        kind: DynamicLimitKind,
        input: LogicalPlan,
        span: &Span,
    ) -> LogicalPlan {
        match literal_int(expr) {
            Some(n) => match kind {
                DynamicLimitKind::Skip => LogicalPlan::Skip(LogicalSkip {
                    input: Box::new(input),
                    count: n,
                    span: span.clone(),
                }),
                DynamicLimitKind::Limit => LogicalPlan::Limit(LogicalLimit {
                    input: Box::new(input),
                    count: n,
                    span: span.clone(),
                }),
            },
            None => LogicalPlan::DynamicLimit(LogicalDynamicLimit {
                input: Box::new(input),
                kind,
                count_expr: expr.clone(),
                span: span.clone(),
            }),
        }
    }

    // ---------- M4-33: Aggregation lowering (RETURN / WITH) ----------

    /// Lower a RETURN / WITH item list into a [`LogicalAggregate`]
    /// node when at least one item contains an aggregation function
    /// call. The caller (`lower_with` / `lower_return`) gates on
    /// [`items_contain_aggregation`] before invoking this method.
    ///
    /// **Rule (per ADR-038 §2 D-28 + openCypher 9 §6.4):** items whose
    /// expression contains an aggregation function call become
    /// [`AggregationSpec`] entries; non-aggregation items become
    /// `group_by` keys (the implicit GROUP BY). Wildcard items
    /// (`RETURN *`) are treated as group-by keys (they expand to all
    /// in-scope bindings, none of which are aggregations).
    fn lower_aggregation_clause(
        &mut self,
        items: &[BoundProjectionItem],
        clause_span: &Span,
        input: LogicalPlan,
    ) -> (LogicalPlan, Vec<BoundProjectionItem>) {
        let mut group_by: Vec<BoundProjectionItem> = Vec::new();
        let mut aggregations: Vec<AggregationSpec> = Vec::new();
        // #746: the `Project` layered over the `Aggregate` must NOT
        // re-evaluate the aggregate function — the executor's `eval`
        // has no aggregate path (`count(n)` there errors), and even its
        // argument (`n`) is absent from the Aggregate's output row. So
        // each projection item is REWRITTEN to a `VariableRef` of the
        // column's output binding-id: the `Project` then passes the
        // precomputed `Aggregate` column through (and reorders the
        // group-then-aggregation layout back into source order). Built
        // in source order.
        let mut project_items: Vec<BoundProjectionItem> = Vec::with_capacity(items.len());

        // #910 — two-phase lowering pre-pass. Pre-assign each non-aggregate
        // (implicit GROUP BY) item's output id and collect the grouping keys,
        // so a NESTED-aggregate projection (`count(n)*2`, `size(collect(x))`,
        // `me.age + count(...)`) can rewrite a grouping-key reference in its
        // OUTER expression to the key's precomputed `Aggregate` column.
        // openCypher v9 §6.4 lets the referenced key appear in ANY source
        // position relative to the aggregating item, so all keys must be known
        // before any nested item is rewritten. The id is assigned ONCE here and
        // reused by the group-by branch below, so the `Aggregate`'s emitted
        // column and the rewrite's `VariableRef` agree. In a bound query every
        // `Expr` item already carries `output_id` (the binder assigns it), so
        // this allocates no fresh ids for real queries — only the embedded
        // aggregates below get fresh HIDDEN ids.
        let group_key_ids: Vec<Option<BindingId>> = items
            .iter()
            .map(|it| match &it.kind {
                BoundProjectionKind::Expr(e) if !expr_contains_aggregation(e) => {
                    Some(it.output_id.unwrap_or_else(|| self.fresh_anon_binding_id()))
                }
                _ => None,
            })
            .collect();
        let grouping_keys: Vec<(BoundExpression, BindingId)> = items
            .iter()
            .zip(&group_key_ids)
            .filter_map(|(it, id)| match (&it.kind, id) {
                (BoundProjectionKind::Expr(e), Some(id)) => Some((e.clone(), *id)),
                _ => None,
            })
            .collect();

        for (it, group_key_id) in items.iter().zip(&group_key_ids) {
            match &it.kind {
                BoundProjectionKind::Expr(e) => {
                    if let Some(spec) = self.try_lift_aggregation(it, e) {
                        let out_id = spec.output_id;
                        let alias = spec.alias.clone();
                        let span = spec.span.clone();
                        aggregations.push(spec);
                        // #353 — carry the original RETURN item's
                        // implicit-column source text (`count(*)`,
                        // `sum(x)`) so the post-Aggregate passthrough
                        // still yields the user-meaningful column name
                        // when there is no explicit alias (the rewrite to
                        // a synthetic `VariableRef` would otherwise lose
                        // it, leaving an empty name).
                        project_items.push(Self::passthrough_item(
                            out_id,
                            alias,
                            it.source_text.clone(),
                            span,
                        ));
                    } else if expr_contains_aggregation(e) {
                        // #910 — aggregate NESTED in an expression
                        // (`count(n)*2`, `size(collect(x))`, `sum(x)+1`,
                        // `toString(count(n))`, `100.0*count(a)/count(b)`,
                        // `collect(x)[i]`, …). Lift each embedded aggregate
                        // into `aggregations` under a fresh HIDDEN id and
                        // rewrite the outer expression to read those hidden
                        // columns (+ any grouping-key reference). The Project
                        // layered over the Aggregate then EVALUATES the
                        // rewritten outer expression — composing the existing
                        // `AggregateOp` + `ProjectOp` (NO new operator),
                        // reusing the #746/#864 Aggregate→Project tunnel. The
                        // Project column keeps the item's own binder-assigned
                        // output id so a downstream `WITH count(n)*2 AS c …
                        // RETURN c` resolves `c` to the same id.
                        let out_id = it.output_id.unwrap_or_else(|| self.fresh_anon_binding_id());
                        let rewritten =
                            self.rewrite_nested_aggregations(e, &mut aggregations, &grouping_keys);
                        project_items.push(BoundProjectionItem {
                            kind: BoundProjectionKind::Expr(rewritten),
                            alias: it.alias.clone(),
                            output_id: Some(out_id),
                            source_text: it.source_text.clone(),
                            span: it.span.clone(),
                        });
                    } else {
                        // Non-aggregation item → implicit GROUP BY key.
                        // The Aggregate emits the group value under the
                        // item's own output id; the Project references
                        // it. The id was pre-assigned (#910 pre-pass) so it
                        // matches any nested-aggregate grouping-key rewrite.
                        let out_id = group_key_id.expect("pre-pass assigned a group-by key id");
                        let mut gb = it.clone();
                        gb.output_id = Some(out_id);
                        group_by.push(gb);
                        // #353 — same: a group-by key projected without an
                        // alias (`RETURN a.x, count(*)`) keeps its source
                        // text (`a.x`) through the passthrough rewrite.
                        project_items.push(Self::passthrough_item(
                            out_id,
                            it.alias.clone(),
                            it.source_text.clone(),
                            it.span.clone(),
                        ));
                    }
                }
                BoundProjectionKind::Wildcard { .. } => {
                    // `RETURN *` with an aggregate: the wildcard expands
                    // to all in-scope bindings (none aggregations). It
                    // stays a wildcard in BOTH the group-by AND the
                    // Project (the Project passes the Aggregate's output
                    // schema through unchanged).
                    group_by.push(it.clone());
                    project_items.push(it.clone());
                }
            }
        }

        let agg = LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(input),
            group_by,
            aggregations,
            span: clause_span.clone(),
        });
        (agg, project_items)
    }

    /// Build a passthrough projection item `VariableRef(id)` that also
    /// carries `output_id = id` (#746). This is the rewrite that turns a
    /// `Project`-over-`Aggregate` item into a pure column passthrough:
    /// the `Aggregate` already computed the value under `id`, so the
    /// `Project` just reads + re-emits it (reordering as needed). The
    /// rewrite is created at lowering time (after type-check), so the
    /// synthesized `VariableRef` is never re-type-checked; `name` is
    /// cosmetic (the executor resolves by `binding_id`).
    fn passthrough_item(
        id: BindingId,
        alias: Option<String>,
        source_text: Option<String>,
        span: Span,
    ) -> BoundProjectionItem {
        let name = alias.clone().unwrap_or_default();
        BoundProjectionItem {
            kind: BoundProjectionKind::Expr(BoundExpression::VariableRef {
                name,
                binding_id: id,
                span: span.clone(),
                type_info: None,
            }),
            alias,
            output_id: Some(id),
            // #353 — preserve the original RETURN item's implicit-column
            // source text through the aggregate-passthrough rewrite. The
            // synthetic `VariableRef`'s `name` is cosmetic/empty for an
            // un-aliased aggregate (`count(*)`), so `display_name` would
            // fall to an empty string without this; carrying `source_text`
            // lets it surface `count(*)` instead.
            source_text,
            span,
        }
    }

    /// If `expr` is a top-level aggregation function call, build the
    /// matching [`AggregationSpec`]; otherwise return `None`.
    ///
    /// We only lift TOP-LEVEL aggregation calls (e.g., `count(n)`);
    /// nested aggregation forms like `sum(count(n))` are rejected by
    /// M4-22 type-check (per openCypher 9 §3 — aggregation functions
    /// are not composable). The `arg` slot carries the argument
    /// expression preserved verbatim for downstream NULL-handling.
    fn try_lift_aggregation(
        &mut self,
        item: &BoundProjectionItem,
        expr: &BoundExpression,
    ) -> Option<AggregationSpec> {
        if let BoundExpression::FunctionCall {
            name,
            args,
            distinct,
            star,
            span,
            ..
        } = expr
        {
            if let Some(kind) = AggregationKind::from_function_name(name) {
                // #773 G4 — `count(*)` (star) has no expression argument;
                // the `arg` is a placeholder the executor never reads (it
                // folds a non-NULL sentinel per row). Non-star aggregates
                // preserve their argument verbatim for downstream
                // NULL-handling.
                let arg = args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| BoundExpression::Literal {
                        value: Literal::Null,
                        span: span.clone(),
                        type_info: None,
                    });
                // #746: the aggregation's result column is emitted under
                // the projection item's binder-assigned output id (so a
                // downstream `WITH count(n) AS c … RETURN c` resolves `c`
                // to the same id). Fall back to a fresh anon id only for
                // a hand-built item with no binder id.
                let output_id = item
                    .output_id
                    .unwrap_or_else(|| self.fresh_anon_binding_id());
                return Some(AggregationSpec {
                    function: kind,
                    arg,
                    output_id,
                    alias: item.alias.clone(),
                    // #773 G4/G5 — thread the source DISTINCT / star flags
                    // into the spec so the executor dedups (DISTINCT) and
                    // counts-rows (star).
                    distinct: *distinct,
                    star: *star,
                    span: span.clone(),
                });
            }
        }
        None
    }

    /// #910 — rewrite an outer projection expression that CONTAINS an
    /// aggregate (but is not itself a BARE aggregate — that is
    /// [`Self::try_lift_aggregation`]'s job) into the expression the `Project`
    /// layered over the `Aggregate` evaluates. Each embedded aggregate
    /// function call is lifted into `aggregations` under a FRESH hidden binding
    /// id and replaced by a `VariableRef` to it; each grouping-key reference (a
    /// simple variable / property access equal to one of `grouping_keys`) is
    /// replaced by a `VariableRef` to the key's precomputed `Aggregate` column.
    /// The result references ONLY hidden-aggregate ids, grouping-key output
    /// ids, literals, and parameters — all resolvable from the `Aggregate`'s
    /// output schema (so the layered `ProjectOp` never re-evaluates an
    /// aggregate row-wise, which is the pre-#910 `-32005` `NotImplemented`).
    ///
    /// Reuses the #746/#864 Aggregate→Project hidden-column tunnel; composes
    /// the existing `AggregateOp` + `ProjectOp` (NO new operator). The
    /// synthesized `VariableRef`s are created AFTER type-check, so they are
    /// never re-type-checked (the executor resolves by `binding_id`).
    ///
    /// The walk mirrors [`expr_contains_aggregation`] (which gated entry):
    /// every variant that can carry an aggregate is recursed into. A scoped-
    /// variable body (list-predicate / reduce / list-comprehension predicate +
    /// projection) cannot contain an aggregate — binding rejects it
    /// (`InvalidAggregation`) — and its scoped variable never equals a grouping
    /// key, so recursing through it is inert (no aggregate to lift, no leaf to
    /// rewrite).
    fn rewrite_nested_aggregations(
        &mut self,
        expr: &BoundExpression,
        aggregations: &mut Vec<AggregationSpec>,
        grouping_keys: &[(BoundExpression, BindingId)],
    ) -> BoundExpression {
        use BoundExpression as BE;
        match expr {
            BE::FunctionCall {
                name,
                args,
                distinct,
                star,
                span,
                type_info,
            } => {
                if let Some(kind) = AggregationKind::from_function_name(name) {
                    // Embedded aggregate → lift it. The ARGUMENT is preserved
                    // verbatim (the `AggregateOp` evaluates it against the
                    // pre-grouping input rows, where its bindings are still
                    // live), so we do NOT recurse into the args. `count(*)`
                    // (star) has no expression argument — mirror
                    // `try_lift_aggregation`'s NULL placeholder.
                    let hidden_id = self.fresh_anon_binding_id();
                    let arg = args.first().cloned().unwrap_or_else(|| BE::Literal {
                        value: Literal::Null,
                        span: span.clone(),
                        type_info: None,
                    });
                    aggregations.push(AggregationSpec {
                        function: kind,
                        arg,
                        output_id: hidden_id,
                        // An embedded aggregate is never directly aliased — the
                        // alias belongs to the WHOLE outer projection item.
                        alias: None,
                        distinct: *distinct,
                        star: *star,
                        span: span.clone(),
                    });
                    return Self::hidden_ref(hidden_id, span.clone());
                }
                // Non-aggregate function (`size`, `toString`, `head`, `abs`,
                // `round`, …) — recurse into each argument.
                BE::FunctionCall {
                    name: name.clone(),
                    args: args
                        .iter()
                        .map(|a| self.rewrite_nested_aggregations(a, aggregations, grouping_keys))
                        .collect(),
                    distinct: *distinct,
                    star: *star,
                    span: span.clone(),
                    type_info: type_info.clone(),
                }
            }
            // Grouping-key reference → read the precomputed `Aggregate` column.
            BE::VariableRef { span, .. } => match grouping_key_output(expr, grouping_keys) {
                Some(id) => Self::hidden_ref(id, span.clone()),
                None => expr.clone(),
            },
            BE::PropertyAccess {
                base,
                path,
                span,
                type_info,
            } => {
                if let Some(id) = grouping_key_output(expr, grouping_keys) {
                    return Self::hidden_ref(id, span.clone());
                }
                BE::PropertyAccess {
                    base: Box::new(self.rewrite_nested_aggregations(
                        base,
                        aggregations,
                        grouping_keys,
                    )),
                    path: path.clone(),
                    span: span.clone(),
                    type_info: type_info.clone(),
                }
            }
            // Pure leaves — nothing to lift / rewrite.
            BE::Literal { .. } | BE::Parameter { .. } | BE::UnresolvedVariable { .. } => {
                expr.clone()
            }
            BE::ListLiteral {
                elements,
                span,
                type_info,
            } => BE::ListLiteral {
                elements: elements
                    .iter()
                    .map(|e| self.rewrite_nested_aggregations(e, aggregations, grouping_keys))
                    .collect(),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::MapLiteral {
                entries,
                span,
                type_info,
            } => BE::MapLiteral {
                entries: entries
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            self.rewrite_nested_aggregations(v, aggregations, grouping_keys),
                        )
                    })
                    .collect(),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            // Compound — the four left-spine operator variants rebuild
            // through one iterative driver (#1290): a flat operator
            // chain folds into a left-nested spine up to
            // `MAX_FLAT_CHAIN_DEPTH` deep (possibly interleaving the
            // four variants), and recursing per level overflowed the
            // native stack. Walk down the left/operand edge collecting
            // one borrowed frame per level, rewrite the non-spine base,
            // then fold back up — the rewrite order (base subtree
            // first, then each level's rhs innermost→outermost) matches
            // the recursion this replaces, so lifted-aggregate
            // `aggregations` push order and hidden-id minting order are
            // unchanged. Non-spine children recurse (bracket-bounded).
            BE::BinaryOp { .. } | BE::UnaryOp { .. } | BE::In { .. } | BE::IsNull { .. } => {
                enum SpineFrame<'a> {
                    Binary {
                        op: &'a BinOp,
                        rhs: &'a BoundExpression,
                        span: &'a Span,
                        type_info: &'a Option<TypeInfo>,
                    },
                    Unary {
                        op: &'a UnaryOp,
                        span: &'a Span,
                        type_info: &'a Option<TypeInfo>,
                    },
                    In {
                        rhs: &'a BoundExpression,
                        span: &'a Span,
                        type_info: &'a Option<TypeInfo>,
                    },
                    IsNull {
                        negated: bool,
                        span: &'a Span,
                        type_info: &'a Option<TypeInfo>,
                    },
                }
                let mut frames: Vec<SpineFrame<'_>> = Vec::new();
                let mut cur = expr;
                let mut acc = loop {
                    match cur {
                        BE::BinaryOp {
                            op,
                            lhs,
                            rhs,
                            span,
                            type_info,
                        } => {
                            frames.push(SpineFrame::Binary {
                                op,
                                rhs,
                                span,
                                type_info,
                            });
                            cur = lhs;
                        }
                        BE::UnaryOp {
                            op,
                            operand,
                            span,
                            type_info,
                        } => {
                            frames.push(SpineFrame::Unary {
                                op,
                                span,
                                type_info,
                            });
                            cur = operand;
                        }
                        BE::In {
                            lhs,
                            rhs,
                            span,
                            type_info,
                        } => {
                            frames.push(SpineFrame::In {
                                rhs,
                                span,
                                type_info,
                            });
                            cur = lhs;
                        }
                        BE::IsNull {
                            lhs,
                            negated,
                            span,
                            type_info,
                        } => {
                            frames.push(SpineFrame::IsNull {
                                negated: *negated,
                                span,
                                type_info,
                            });
                            cur = lhs;
                        }
                        other => {
                            break self.rewrite_nested_aggregations(
                                other,
                                aggregations,
                                grouping_keys,
                            );
                        }
                    }
                };
                while let Some(frame) = frames.pop() {
                    acc = match frame {
                        SpineFrame::Binary {
                            op,
                            rhs,
                            span,
                            type_info,
                        } => BE::BinaryOp {
                            op: op.clone(),
                            lhs: Box::new(acc),
                            rhs: Box::new(self.rewrite_nested_aggregations(
                                rhs,
                                aggregations,
                                grouping_keys,
                            )),
                            span: span.clone(),
                            type_info: type_info.clone(),
                        },
                        SpineFrame::Unary {
                            op,
                            span,
                            type_info,
                        } => BE::UnaryOp {
                            op: op.clone(),
                            operand: Box::new(acc),
                            span: span.clone(),
                            type_info: type_info.clone(),
                        },
                        SpineFrame::In {
                            rhs,
                            span,
                            type_info,
                        } => BE::In {
                            lhs: Box::new(acc),
                            rhs: Box::new(self.rewrite_nested_aggregations(
                                rhs,
                                aggregations,
                                grouping_keys,
                            )),
                            span: span.clone(),
                            type_info: type_info.clone(),
                        },
                        SpineFrame::IsNull {
                            negated,
                            span,
                            type_info,
                        } => BE::IsNull {
                            lhs: Box::new(acc),
                            negated,
                            span: span.clone(),
                            type_info: type_info.clone(),
                        },
                    };
                }
                acc
            }
            BE::Near {
                lhs,
                target,
                vector_index,
                span,
                type_info,
            } => BE::Near {
                lhs: Box::new(self.rewrite_nested_aggregations(lhs, aggregations, grouping_keys)),
                target: Box::new(self.rewrite_nested_aggregations(
                    target,
                    aggregations,
                    grouping_keys,
                )),
                vector_index: vector_index.clone(),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::TextMatch {
                lhs,
                query,
                span,
                type_info,
            } => BE::TextMatch {
                lhs: Box::new(self.rewrite_nested_aggregations(lhs, aggregations, grouping_keys)),
                query: Box::new(self.rewrite_nested_aggregations(
                    query,
                    aggregations,
                    grouping_keys,
                )),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::InCommunity {
                node,
                community,
                span,
                type_info,
            } => BE::InCommunity {
                node: Box::new(self.rewrite_nested_aggregations(node, aggregations, grouping_keys)),
                community: Box::new(self.rewrite_nested_aggregations(
                    community,
                    aggregations,
                    grouping_keys,
                )),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::Subscript {
                base,
                index,
                span,
                type_info,
            } => BE::Subscript {
                base: Box::new(self.rewrite_nested_aggregations(base, aggregations, grouping_keys)),
                index: Box::new(self.rewrite_nested_aggregations(
                    index,
                    aggregations,
                    grouping_keys,
                )),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::Slice {
                base,
                start,
                end,
                span,
                type_info,
            } => BE::Slice {
                base: Box::new(self.rewrite_nested_aggregations(base, aggregations, grouping_keys)),
                start: start.as_ref().map(|s| {
                    Box::new(self.rewrite_nested_aggregations(s, aggregations, grouping_keys))
                }),
                end: end.as_ref().map(|e| {
                    Box::new(self.rewrite_nested_aggregations(e, aggregations, grouping_keys))
                }),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::Case {
                test,
                branches,
                default,
                span,
                type_info,
            } => BE::Case {
                test: test.as_ref().map(|t| {
                    Box::new(self.rewrite_nested_aggregations(t, aggregations, grouping_keys))
                }),
                branches: branches
                    .iter()
                    .map(|(w, t)| {
                        (
                            self.rewrite_nested_aggregations(w, aggregations, grouping_keys),
                            self.rewrite_nested_aggregations(t, aggregations, grouping_keys),
                        )
                    })
                    .collect(),
                default: default.as_ref().map(|d| {
                    Box::new(self.rewrite_nested_aggregations(d, aggregations, grouping_keys))
                }),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::ListPredicate {
                quantifier,
                var_bid,
                list,
                predicate,
                span,
                type_info,
            } => BE::ListPredicate {
                quantifier: *quantifier,
                var_bid: *var_bid,
                list: Box::new(self.rewrite_nested_aggregations(list, aggregations, grouping_keys)),
                predicate: Box::new(self.rewrite_nested_aggregations(
                    predicate,
                    aggregations,
                    grouping_keys,
                )),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::Reduce {
                acc_bid,
                init,
                var_bid,
                list,
                expr: body,
                span,
                type_info,
            } => BE::Reduce {
                acc_bid: *acc_bid,
                init: Box::new(self.rewrite_nested_aggregations(init, aggregations, grouping_keys)),
                var_bid: *var_bid,
                list: Box::new(self.rewrite_nested_aggregations(list, aggregations, grouping_keys)),
                expr: Box::new(self.rewrite_nested_aggregations(body, aggregations, grouping_keys)),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::ListComprehension {
                var_bid,
                list,
                predicate,
                projection,
                span,
                type_info,
            } => BE::ListComprehension {
                var_bid: *var_bid,
                list: Box::new(self.rewrite_nested_aggregations(list, aggregations, grouping_keys)),
                predicate: predicate.as_ref().map(|p| {
                    Box::new(self.rewrite_nested_aggregations(p, aggregations, grouping_keys))
                }),
                projection: projection.as_ref().map(|p| {
                    Box::new(self.rewrite_nested_aggregations(p, aggregations, grouping_keys))
                }),
                span: span.clone(),
                type_info: type_info.clone(),
            },
            BE::MapProjection {
                base,
                items,
                span,
                type_info,
            } => BE::MapProjection {
                base: Box::new(self.rewrite_nested_aggregations(base, aggregations, grouping_keys)),
                items: items
                    .iter()
                    .map(|item| match item {
                        BoundMapProjectionItem::Literal { alias, value } => {
                            BoundMapProjectionItem::Literal {
                                alias: alias.clone(),
                                value: Box::new(self.rewrite_nested_aggregations(
                                    value,
                                    aggregations,
                                    grouping_keys,
                                )),
                            }
                        }
                        other => other.clone(),
                    })
                    .collect(),
                span: span.clone(),
                type_info: type_info.clone(),
            },
        }
    }

    /// A synthesized `VariableRef` to a precomputed `Aggregate` column (a
    /// hidden-aggregate id or a grouping-key output id). `name` is cosmetic
    /// (the executor resolves by `binding_id`) and `type_info` is `None`
    /// (created post-type-check). #910 — shares the #864 hidden-column ref
    /// shape used by [`Self::lower_order_by_over_project`].
    fn hidden_ref(id: BindingId, span: Span) -> BoundExpression {
        BoundExpression::VariableRef {
            name: String::new(),
            binding_id: id,
            span,
            type_info: None,
        }
    }

    // ---------- M4-33: ORDER BY (tail + return) ----------

    /// Lower a tail-clause `ORDER BY` (`MATCH ... RETURN x ORDER BY x.age`
    /// — when ORDER BY appears in its own tail clause) into a
    /// [`LogicalSort`]. Used by [`Self::lower_clause`] for
    /// [`BoundClause::TailOrderBy`].
    fn lower_tail_order_by(
        &mut self,
        items: &[BoundOrderItem],
        span: &Span,
        prev: Option<LogicalPlan>,
    ) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| LogicalPlan::Empty(LogicalEmpty { span: span.clone() }));
        // #864 — ORDER BY by a NON-projected in-scope expression. When the sort
        // is directly over the RETURN/WITH `Project`, a sort key referencing a
        // binding the projection DROPPED (`RETURN e.id ORDER BY e.n` — `e` is
        // not projected) cannot be evaluated by `SortOp` against the
        // post-Project schema (#746) and fails "binding … missing from row
        // schema". #857 already handles a key that MATCHES a projected item
        // (the binder rewrote it to a `VariableRef` to that output). The
        // genuinely non-projected key is carried as a HIDDEN `Project` column
        // (its bindings are still live in the projection's input), sorted by,
        // then trimmed off — the standard openCypher hidden-sort-column form.
        match input {
            LogicalPlan::Project(proj) => self.lower_order_by_over_project(items, span, proj),
            other => self.lower_order_by(items, span, other),
        }
    }

    /// #864 — lower ORDER BY directly over a [`LogicalProject`], carrying any
    /// non-projected sort key as a hidden column trimmed back off after the
    /// Sort. A key already satisfiable from the projection output (a
    /// projected/aliased value — #857) is left unchanged.
    ///
    /// Falls back to a plain Sort (preserving the existing error for a
    /// non-projected key) when:
    /// - the Project is over an `Aggregate` — openCypher forbids ordering by a
    ///   non-grouped/non-aggregated value (the binding was collapsed); or
    /// - the projection is a wildcard (`RETURN *` passes every binding through,
    ///   so no key is ever dropped).
    ///
    /// The DISTINCT case never reaches here: `lower_return` places `Distinct`
    /// ABOVE the `Project`, so the tail-ORDER-BY input is a `Distinct`, not a
    /// `Project` — a non-projected key correctly stays an error (openCypher
    /// forbids ordering a DISTINCT result by a non-output value).
    fn lower_order_by_over_project(
        &mut self,
        items: &[BoundOrderItem],
        span: &Span,
        mut proj: LogicalProject,
    ) -> LogicalPlan {
        let over_aggregate = matches!(proj.input.as_ref(), LogicalPlan::Aggregate(_));
        let has_wildcard = proj
            .items
            .iter()
            .any(|it| matches!(it.kind, BoundProjectionKind::Wildcard { .. }));
        if over_aggregate {
            // #1053 — a sort key that contains an INLINE aggregate
            // (`ORDER BY me.age + count(you.age)`) over an aggregating
            // projection: lift the aggregate into the SAME `Aggregate` node and
            // point the `Sort` at the computed column (openCypher v9 §6.6). The
            // binder has already validated (every non-aggregated leaf is a
            // grouping key); here we compose the existing #910 hidden-aggregate
            // tunnel. A wildcard aggregating projection (`RETURN *` with an
            // aggregate) is excluded — its schema already passes every column
            // through, and an inline-aggregate sort key over it is not a
            // produced TCK shape — so it falls through to the plain Sort.
            if !has_wildcard && items.iter().any(|o| expr_contains_aggregation(&o.expr)) {
                return self.lower_order_by_over_aggregate(items, span, proj);
            }
            return self.lower_order_by(items, span, LogicalPlan::Project(proj));
        }
        if has_wildcard {
            return self.lower_order_by(items, span, LogicalPlan::Project(proj));
        }

        let user_output_ids: std::collections::BTreeSet<BindingId> =
            proj.items.iter().filter_map(|it| it.output_id).collect();
        let user_items = proj.items.clone();
        let mut order_by = items.to_vec();
        let mut hidden_added = false;
        for o in order_by.iter_mut() {
            let mut refs = std::collections::BTreeSet::new();
            collect_referenced_bindings(&o.expr, &mut refs);
            // Every referenced binding already a projection output ⇒ the key is
            // evaluable from the post-Project schema (#857) — leave it.
            if refs.iter().all(|b| user_output_ids.contains(b)) {
                continue;
            }
            // Non-projected ⇒ compute it as a hidden column (the projection's
            // input still carries `e`), and point the Sort at that output id.
            let hidden_id = self.fresh_anon_binding_id();
            proj.items.push(BoundProjectionItem {
                kind: BoundProjectionKind::Expr(o.expr.clone()),
                alias: None,
                output_id: Some(hidden_id),
                source_text: None,
                span: o.span.clone(),
            });
            o.expr = BoundExpression::VariableRef {
                name: String::new(),
                binding_id: hidden_id,
                span: o.span.clone(),
                type_info: None,
            };
            hidden_added = true;
        }

        let sorted = self.lower_order_by(&order_by, span, LogicalPlan::Project(proj));
        if !hidden_added {
            return sorted;
        }
        // Trim the hidden column(s): re-project the user items through their own
        // output ids, dropping the hidden columns before the result.
        let trim_items: Vec<BoundProjectionItem> = user_items
            .iter()
            .map(|it| match it.output_id {
                Some(id) => Self::passthrough_item(
                    id,
                    it.alias.clone(),
                    it.source_text.clone(),
                    it.span.clone(),
                ),
                None => it.clone(),
            })
            .collect();
        LogicalPlan::Project(LogicalProject {
            input: Box::new(sorted),
            items: trim_items,
            span: span.clone(),
        })
    }

    /// #1053 — lower an `ORDER BY` whose sort key contains an INLINE aggregate
    /// (`ORDER BY me.age + count(you.age)`) over a `Project`-over-`Aggregate`
    /// (openCypher v9 §6.6). The inline aggregate is computed ONCE alongside
    /// the projection's aggregates and the `Sort` references that computed
    /// column.
    ///
    /// Composes the existing #910 hidden-aggregate tunnel:
    /// 1. The sort key is rewritten by [`Self::rewrite_nested_aggregations`]
    ///    against the `Aggregate`'s GROUPING KEYS (its `group_by` items'
    ///    expr+id): each embedded aggregate is lifted into the SAME
    ///    `Aggregate`'s `aggregations` under a fresh HIDDEN id, and each
    ///    grouping-key reference (`me.age`) is mapped to the precomputed
    ///    group-by column. The rewritten key references ONLY `Aggregate`
    ///    output columns, so it is evaluable from the post-`Aggregate` schema.
    /// 2. The rewritten key is added as a HIDDEN `Project` column (its input is
    ///    the mutated `Aggregate`, whose output schema now carries the lifted
    ///    aggregate), the `Sort` points at that hidden column, and a trimming
    ///    `Project` drops it before the result — mirroring
    ///    [`Self::lower_order_by_over_project`]'s #864 hidden-column trick.
    ///
    /// A sort key WITHOUT an aggregate is handled by the same rewrite: a
    /// grouping-key reference maps to its `Aggregate` column, and a bare
    /// reference to a projection output (already in the schema) is left as-is
    /// then needs no hidden column. The binder guarantees every non-aggregated
    /// leaf is a grouping key, so the rewrite never strands a pre-projection
    /// binding.
    fn lower_order_by_over_aggregate(
        &mut self,
        items: &[BoundOrderItem],
        span: &Span,
        proj: LogicalProject,
    ) -> LogicalPlan {
        // Own the `Aggregate` so its `aggregations` can absorb the lifted
        // sort-key aggregates. `lower_order_by_over_project` only routes a
        // `Project` whose input IS an `Aggregate` here, so this destructure is
        // total; the `else` is defensive (fall back to a plain Sort).
        let LogicalProject {
            items: proj_items,
            input: proj_input,
            span: proj_span,
        } = proj;
        let LogicalPlan::Aggregate(mut agg) = *proj_input else {
            return self.lower_order_by(
                items,
                span,
                LogicalPlan::Project(LogicalProject {
                    items: proj_items,
                    input: proj_input,
                    span: proj_span,
                }),
            );
        };

        // The `Aggregate`'s GROUPING KEYS as (expr, output-id) pairs — the
        // rewrite maps a grouping-key reference in the sort key to the
        // precomputed group-by column (same construction as
        // `lower_aggregation_clause`).
        let grouping_keys: Vec<(BoundExpression, BindingId)> = agg
            .group_by
            .iter()
            .filter_map(|it| match (&it.kind, it.output_id) {
                (BoundProjectionKind::Expr(e), Some(id)) => Some((e.clone(), id)),
                _ => None,
            })
            .collect();

        // #1053 R1 — a grouping key's POST-aggregation OUTPUT id mapped back to
        // its PRE-aggregation INPUT expression (the `group_by` item's
        // `kind.Expr`, e.g. `me.age` / `x` resolved to the input binding). When
        // a grouping key is projected under its OWN name (`RETURN x, count(x)
        // … ORDER BY count(x)`), the binder resolves the sort-key aggregate's
        // ARGUMENT `x` to the grouping key's OUTPUT binding (the output-name
        // scope shadows the pre-projection input). But `AggregateOp` evaluates
        // an aggregate's argument against the PRE-grouping INPUT rows, where
        // only the input binding exists — so an arg left as the output binding
        // crashes "binding … missing from row schema" at execution. This map
        // lets `remap_aggregate_args_to_input` rewrite such an output-bound arg
        // back to its input expression, making the lifted sort-key spec's arg
        // IDENTICAL to the projection's own aggregate spec (which the binder
        // already resolved against the pre-projection input). A grouping key
        // projected under an ALIAS (`x AS age`) or an aggregate over a DISTINCT
        // variable (`count(you.age)`) never hits this (the arg already binds to
        // the input), so the remap is a no-op there.
        let output_to_input: std::collections::BTreeMap<BindingId, BoundExpression> = agg
            .group_by
            .iter()
            .filter_map(|it| match (&it.kind, it.output_id) {
                (BoundProjectionKind::Expr(e), Some(id)) => Some((id, e.clone())),
                _ => None,
            })
            .collect();

        // The columns the post-`Aggregate` `Project` already emits (the user's
        // RETURN/WITH outputs) — a sort key already evaluable from them
        // (`ORDER BY cnt` where `cnt` is the aggregate alias) needs no hidden
        // column.
        let user_output_ids: std::collections::BTreeSet<BindingId> =
            proj_items.iter().filter_map(|it| it.output_id).collect();
        let user_items = proj_items.clone();

        let mut project_items = proj_items;
        let mut order_by = items.to_vec();
        let mut hidden_added = false;
        for o in order_by.iter_mut() {
            // A sort key with NO aggregate that already references only
            // projection outputs is evaluable as-is (the #857 path) — leave it.
            if !expr_contains_aggregation(&o.expr) {
                let mut refs = std::collections::BTreeSet::new();
                collect_referenced_bindings(&o.expr, &mut refs);
                if refs.iter().all(|b| user_output_ids.contains(b)) {
                    continue;
                }
            }
            // #1053 R1 — FIRST remap any reference to a grouping-key OUTPUT
            // binding back to the key's PRE-aggregation INPUT expression. This
            // is load-bearing for an aggregate ARGUMENT that references a
            // grouping key projected under its OWN name (`ORDER BY count(x)`
            // where `x` is both the grouping key and the agg arg): the binder
            // resolved that `x` to the grouping key's OUTPUT binding, but
            // `AggregateOp` evaluates the argument against the PRE-grouping
            // INPUT rows, where only the input binding exists — leaving it
            // crashes "binding … missing from row schema". Remapping the whole
            // key is safe: a grouping-key reference OUTSIDE the aggregate is now
            // the key's INPUT expression, which `rewrite_nested_aggregations`
            // STILL maps onto the `Aggregate` output column (it matches a
            // grouping key by its input expression, not by id), so the outer
            // reference resolves to the precomputed column exactly as before.
            let arg_remapped = substitute_output_bindings(&o.expr, &output_to_input);
            // Rewrite the key: lift inline aggregates into `agg.aggregations`,
            // map grouping-key references to their `Aggregate` columns. The
            // result references ONLY `Aggregate` output columns.
            let rewritten = self.rewrite_nested_aggregations(
                &arg_remapped,
                &mut agg.aggregations,
                &grouping_keys,
            );
            // Add it as a HIDDEN `Project` column so the post-`Aggregate`
            // `Project` evaluates it; point the `Sort` at that column.
            let hidden_id = self.fresh_anon_binding_id();
            project_items.push(BoundProjectionItem {
                kind: BoundProjectionKind::Expr(rewritten),
                alias: None,
                output_id: Some(hidden_id),
                source_text: None,
                span: o.span.clone(),
            });
            o.expr = Self::hidden_ref(hidden_id, o.span.clone());
            hidden_added = true;
        }

        let project = LogicalPlan::Project(LogicalProject {
            input: Box::new(LogicalPlan::Aggregate(agg)),
            items: project_items,
            span: proj_span,
        });
        let sorted = self.lower_order_by(&order_by, span, project);
        if !hidden_added {
            return sorted;
        }
        // Trim the hidden sort column(s): re-project the user items through
        // their own output ids, dropping the hidden columns before the result.
        let trim_items: Vec<BoundProjectionItem> = user_items
            .iter()
            .map(|it| match it.output_id {
                Some(id) => Self::passthrough_item(
                    id,
                    it.alias.clone(),
                    it.source_text.clone(),
                    it.span.clone(),
                ),
                None => it.clone(),
            })
            .collect();
        LogicalPlan::Project(LogicalProject {
            input: Box::new(sorted),
            items: trim_items,
            span: span.clone(),
        })
    }

    /// Lower a list of [`BoundOrderItem`]s into a [`LogicalSort`]
    /// wrapping `input`. Shared between the return-clause embedded
    /// `ORDER BY` (see [`Self::lower_return`]) and the tail-clause
    /// `ORDER BY` (see [`Self::lower_tail_order_by`]) per ADR-038
    /// §2 D-28 — both surfaces lower to the SAME variant shape.
    fn lower_order_by(
        &mut self,
        items: &[BoundOrderItem],
        clause_span: &Span,
        input: LogicalPlan,
    ) -> LogicalPlan {
        let order_by = items
            .iter()
            .map(|o| OrderByItem {
                expr: o.expr.clone(),
                direction: sort_direction_from_ast(&o.direction),
                span: o.span.clone(),
            })
            .collect();
        LogicalPlan::Sort(LogicalSort {
            input: Box::new(input),
            order_by,
            span: clause_span.clone(),
        })
    }

    // ---------- M4-33: DISTINCT ----------

    /// Lower a `RETURN DISTINCT ...` into a [`LogicalDistinct`] node
    /// wrapping `input`. The `on` slot carries the bindings whose
    /// joint tuple identifies a row for deduplication — derived from
    /// the projection items by collecting referenced bindings.
    fn lower_distinct(
        &mut self,
        items: &[BoundProjectionItem],
        clause_span: &Span,
        input: LogicalPlan,
    ) -> LogicalPlan {
        let mut on: std::collections::BTreeSet<BindingId> = std::collections::BTreeSet::new();
        for it in items {
            if let BoundProjectionKind::Expr(e) = &it.kind {
                collect_referenced_bindings(e, &mut on);
            }
        }
        LogicalPlan::Distinct(LogicalDistinct {
            input: Box::new(input),
            on: on.into_iter().collect(),
            span: clause_span.clone(),
        })
    }

    // ---------- ADR-197 (#802): CALL <proc>(…) [YIELD …] / SHOW … ----------

    /// Lower a `CALL <proc>(args) [YIELD …]` into a
    /// [`LogicalProcedureCall`]. Roots on a leading [`LogicalEmpty`]
    /// unit row (same idiom as [`Self::lower_unwind`]) so a top-level
    /// CALL produces its rows from exactly one driving row.
    fn lower_call_procedure(
        &mut self,
        c: &BoundCallProcedureClause,
        prev: Option<LogicalPlan>,
    ) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: c.span.clone(),
            })
        });
        let columns: Vec<(String, BindingId)> = c
            .yields
            .iter()
            .map(|y| (y.column.clone(), y.var.binding_id))
            .collect();
        let proc = LogicalPlan::ProcedureCall(LogicalProcedureCall {
            input: Box::new(input),
            source: ProcedureSource::Procedure(c.kind),
            args: c.args.clone(),
            columns,
            span: c.span.clone(),
        });
        // `CALL … YIELD … WHERE <pred>` → wrap the procedure rows in a
        // Filter (the predicate references the YIELD'd columns).
        match &c.where_clause {
            Some(pred) => LogicalPlan::Filter(LogicalFilter {
                input: Box::new(proc),
                predicate: pred.clone(),
                span: c.span.clone(),
            }),
            None => proc,
        }
    }

    /// Lower a `SHOW CONSTRAINTS | INDEXES | DATABASES | VECTOR INDEXES`
    /// `[YIELD … [WHERE …]]` into a [`LogicalProcedureCall`] with a
    /// [`ProcedureSource::Show`] source, wrapping the rows in a
    /// [`LogicalFilter`] when a `WHERE` is present (#830) — mirrors
    /// [`Self::lower_call_procedure`].
    fn lower_show(&mut self, s: &BoundShowClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: s.span.clone(),
            })
        });
        let columns: Vec<(String, BindingId)> = s
            .columns
            .iter()
            .map(|v| (v.name.clone(), v.binding_id))
            .collect();
        let show = LogicalPlan::ProcedureCall(LogicalProcedureCall {
            input: Box::new(input),
            source: ProcedureSource::Show(s.kind),
            args: Vec::new(),
            columns,
            span: s.span.clone(),
        });
        // `SHOW … YIELD … WHERE <pred>` (#830) → wrap the SHOW rows in a
        // Filter (the predicate references the YIELD'd columns).
        match &s.where_clause {
            Some(pred) => LogicalPlan::Filter(LogicalFilter {
                input: Box::new(show),
                predicate: pred.clone(),
                span: s.span.clone(),
            }),
            None => show,
        }
    }

    // ---------- M4-33: UNWIND ----------

    /// Lower an `UNWIND <list> AS <var>` into a [`LogicalUnwind`].
    fn lower_unwind(&mut self, u: &BoundUnwindClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: u.span.clone(),
            })
        });
        LogicalPlan::Unwind(LogicalUnwind {
            input: Box::new(input),
            list_expr: u.expr.clone(),
            var: u.var.binding_id,
            span: u.span.clone(),
        })
    }

    // ---------- ADR-192 (#623): CALL { <subquery> } ----------

    /// Lower a `CALL { <subquery> }` correlated brace-subquery (ADR-192).
    ///
    /// Two structural pins:
    /// - **D-5a leading-CALL unit row.** When `CALL { … }` is the FIRST
    ///   clause (`prev = None`) the `input` roots on the leading-clause
    ///   [`LogicalEmpty`] unit row — the SAME idiom [`Self::lower_unwind`]
    ///   uses — so the body runs exactly once (one driving row). Without
    ///   it a leading CALL{} has zero driving rows and emits zero output
    ///   rows (the #618 UNWIND gap).
    /// - **D-5 correlation seed.** The body is lowered with a
    ///   [`LogicalCorrelationSeed`] as its leading `prev`. The seed
    ///   carries the imported bindings; the EXISTING clause lowering then
    ///   threads them through the body uniformly — a MATCH-led body
    ///   equi-joins the seed on the imported start variable
    ///   ([`Self::lower_match`]'s shared-binding join), and a WITH/RETURN-
    ///   led body reads the imports directly off the seed (the seed is the
    ///   leading-clause `prev`). This needs NO special-casing of imported
    ///   vs. fresh leading variables.
    fn lower_call(&mut self, c: &BoundCallClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let input = prev.unwrap_or_else(|| {
            LogicalPlan::Empty(LogicalEmpty {
                span: c.span.clone(),
            })
        });
        let seed = LogicalPlan::CorrelationSeed(LogicalCorrelationSeed {
            imported: c.imported.clone(),
            span: c.span.clone(),
        });
        // COST FORWARD-PIN (ADR-192 OQ-192-3 / OQ-192-4): a correlated
        // MATCH-led body equi-joins the seed against a full label-scan of
        // the (imported) start variable — `Join(seed{a}, Scan(a)→…, on=a)`
        // — so the body re-executes a scan per driving row (O(outer ×
        // body-cost), the documented v1.0-α correctness-first shape). The
        // uncorrelated-subquery execute-once cache (`imported = []`,
        // OQ-192-3) + the per-row-multiplier cost-model refinement
        // (OQ-192-4) are forward-pinned optimizations, NOT v1.0-α
        // requirements; correctness is independent of them.
        let body = self.lower_call_body(&c.body, seed);
        LogicalPlan::Call(LogicalCall {
            input: Box::new(input),
            body: Box::new(body),
            imported: c.imported.clone(),
            returned: c.returned.clone(),
            span: c.span.clone(),
        })
    }

    /// Lower a `CALL { … }` body (a [`BoundStatement::Read`] or
    /// [`BoundStatement::Union`]) with `seed` as the leading-clause
    /// `prev`. The seed flows into the body's first clause exactly like a
    /// preceding clause's plan would.
    fn lower_call_body(&mut self, body: &BoundStatement, seed: LogicalPlan) -> LogicalPlan {
        match body {
            BoundStatement::Read(q) => {
                self.lower_clauses_with_prev(&q.clauses, Some(seed), &q.span)
            }
            BoundStatement::Union(u) => self.lower_call_union(u, seed),
            // Grammar admits only Read/Union inside CALL{}.
            _ => seed,
        }
    }

    /// Fold a clause sequence into a logical plan with an explicit
    /// starting `prev` — the [`Self::lower_query`] fold, but seeded with
    /// the correlation seed instead of `None`. Does NOT re-seed
    /// `next_anon_id` (the enclosing [`Self::lower_query`] already seeded
    /// it from the WHOLE statement's max binding id — see
    /// [`collect_max_binding_id`]'s `Call` arm — so synthetic anon-node
    /// ids minted while lowering the body never collide).
    fn lower_clauses_with_prev(
        &mut self,
        clauses: &[BoundClause],
        prev: Option<LogicalPlan>,
        span: &Span,
    ) -> LogicalPlan {
        let mut current = prev;
        for c in clauses {
            current = Some(self.lower_clause(c, current));
        }
        current.unwrap_or_else(|| LogicalPlan::Empty(LogicalEmpty { span: span.clone() }))
    }

    /// Lower a UNION body inside `CALL { … }` (ADR-185 set-op shape) with
    /// each arm seeded by the correlation `seed`. Mirrors
    /// [`Self::lower_union`] (UNION-ALL concat vs bare-UNION
    /// `Distinct`-over-concat + the whole-union tail) but every arm is
    /// lowered with the seed as its leading `prev` so each arm sees the
    /// imported bindings.
    fn lower_call_union(&mut self, u: &BoundUnionQuery, seed: LogicalPlan) -> LogicalPlan {
        let is_union_all = u.all.iter().all(|&a| a);
        let arms: Vec<LogicalPlan> = u
            .arms
            .iter()
            .map(|arm| self.lower_clauses_with_prev(&arm.clauses, Some(seed.clone()), &arm.span))
            .collect();
        let union = LogicalPlan::Union(LogicalUnion {
            arms,
            column_orders: u.column_orders.clone(),
            span: u.span.clone(),
        });
        let combined = if is_union_all {
            union
        } else {
            LogicalPlan::Distinct(LogicalDistinct {
                input: Box::new(union),
                on: self.union_output_dedup_key(u),
                span: u.span.clone(),
            })
        };
        self.lower_union_tail(&u.tail, &u.span, combined)
    }

    // ---------- M4-33: Named path / SHORTEST_PATH ----------

    /// Lower a named-path `MATCH p = <pattern>` into a
    /// [`LogicalNamedPath`] wrapping the lowered pattern subtree.
    fn lower_named_path(&mut self, np: &BoundNamedPath, match_span: &Span) -> LogicalPlan {
        let (algorithm, pattern, plain) = match &np.kind {
            BoundNamedPathKind::ShortestPath(p) => (PathAlgorithm::ShortestPath, p, false),
            // ADR-194 D-2 — `allShortestPaths` lowers to the
            // `AllShortestPaths` algorithm. Like `ShortestPath` it
            // re-traverses the substrate (no `plain_shape`); the
            // source/target capture below feeds its REQUIRED src→dst BFS.
            BoundNamedPathKind::AllShortestPath(p) => (PathAlgorithm::AllShortestPaths, p, false),
            BoundNamedPathKind::Plain(p) => (PathAlgorithm::Plain, p, true),
        };
        // ADR-193 D-4 — for a Plain path, capture the ordered element
        // bindings (force-binding anon rels) so the executor can
        // materialize a `Value::Path`. ShortestPath re-traverses the
        // substrate (executor::ops::path) and needs no shape.
        let (pattern_subtree, plain_shape) = if plain {
            let mut shape = PlainPathShape {
                start: np.var.binding_id, // overwritten by the head node below
                segments: Vec::new(),
            };
            let subtree = self.lower_path_pattern_inner(pattern, Some(&mut shape));
            (subtree, Some(shape))
        } else {
            (self.lower_path_pattern(pattern), None)
        };
        // ADR-194 D-3a — capture the pattern's TAIL-endpoint node binding
        // so the `ShortestPath` executor runs bidirectional source→target
        // BFS (one path per `(source, target)` pair) rather than
        // single-source BFS (one row per reachable node — the pre-D-3a
        // bug). The tail endpoint is the LAST node-pattern in the pattern
        // (`b` in `(a)-[..]->(b)`); for the degenerate single-node pattern
        // `(a)` (empty tail) it is the head node, yielding a zero-length
        // `source == target` path. An anonymous tail endpoint `(..)-[..]->()`
        // carries no binding → `None`, falling back to the efficient
        // single-source enumeration (shortest path to every reachable
        // node). Named nodes have deterministic binding-ids
        // (`var.binding_id`) identical to the ones
        // `lower_path_pattern_inner` installs into the child subtree's
        // schema, so reading them from the pattern AST here does NOT
        // re-allocate a fresh anonymous id. Consumed only by
        // `PathAlgorithm::ShortestPath` (the pipeline `Plain` arm ignores
        // it and materializes from `plain_shape`).
        let target = pattern
            .tail
            .last()
            .map(|(_, tail_node)| tail_node)
            .unwrap_or(&pattern.head)
            .var
            .as_ref()
            .map(|v| v.binding_id);
        // ADR-194 D-3a — capture the HEAD (source) node binding for the
        // same reason. The pipeline previously derived the source from
        // `child.schema().first()`, but the tail-label `Scan` join can
        // reorder the schema so the head is NOT first; reading the head
        // binding from the pattern AST is stable. Named heads have
        // deterministic binding-ids; an anonymous head yields `None`
        // (the pipeline keeps the legacy schema-first fallback for that
        // degenerate shortest-path form).
        let source = pattern.head.var.as_ref().map(|v| v.binding_id);
        LogicalPlan::NamedPath(LogicalNamedPath {
            input: Box::new(pattern_subtree),
            path_var: np.var.binding_id,
            algorithm,
            plain_shape,
            source,
            target,
            span: match_span.clone(),
        })
    }

    // ---------- M4-32: RANK BY HYBRID lowering ----------

    /// Lower a `RANK BY HYBRID(VECTOR(...), TEXT(...), ...)` clause.
    ///
    /// Per M4-23 cross-substrate validation, the operand list is
    /// guaranteed to contain ≥1 VECTOR and ≥1 TEXT, each with an
    /// explicit `K = N` parameter.
    fn lower_rank_by(&mut self, r: &BoundRankByClause, prev: Option<LogicalPlan>) -> LogicalPlan {
        let BoundRanker::Hybrid(args) = &r.ranker;

        let mut operands: Vec<HybridOperand> = Vec::with_capacity(args.len());
        for a in args {
            if let Some(op) = self.lower_rank_arg(a) {
                operands.push(op);
            }
        }
        let hybrid = LogicalPlan::RankByHybrid(LogicalRankByHybrid {
            operands,
            score_binding: r.score.as_ref().map(|score| score.binding_id),
            fusion: None,
            span: r.span.clone(),
        });

        // Compose with prev (typical shape: `MATCH (n) RANK BY ...`
        // — `prev` is the MATCH subtree that produced the candidate
        // set; the hybrid clause re-ranks it). The candidate-set
        // semantics are M4-05's concern; M4-32 simply attaches the
        // hybrid orchestrator as a sibling joined on the operand
        // bindings shared with the candidate set.
        match prev {
            None => hybrid,
            Some(p) => {
                let shared = shared_bindings(&p, &hybrid);
                LogicalPlan::Join(LogicalJoin {
                    left: Box::new(p),
                    right: Box::new(hybrid),
                    on: JoinCondition::SharedBindings(shared),
                    // The right/probe side emits relevance order. This
                    // join is therefore a semantic ordering boundary,
                    // not a freely commutative cost-planning join.
                    algorithm: JoinAlgorithm::HashJoin,
                    span: r.span.clone(),
                })
            }
        }
    }

    /// Lower a single `RANK BY HYBRID` operand to a [`HybridOperand`].
    fn lower_rank_arg(&mut self, a: &BoundRankArg) -> Option<HybridOperand> {
        match a {
            BoundRankArg::Vector {
                field,
                query,
                k,
                span,
            } => {
                let (var, property) = self.field_to_var_and_property(field)?;
                Some(HybridOperand {
                    kind: HybridOperandKind::Vector,
                    var,
                    property,
                    query: query.clone(),
                    k: rank_arg_k(*k, span, &mut self.errors),
                    read_lsn: self.read_lsn,
                    span: span.clone(),
                })
            }
            BoundRankArg::Text {
                field,
                query,
                k,
                span,
            } => {
                let (var, property) = self.field_to_var_and_property(field)?;
                Some(HybridOperand {
                    kind: HybridOperandKind::Text,
                    var,
                    property,
                    query: query.clone(),
                    k: rank_arg_k(*k, span, &mut self.errors),
                    read_lsn: self.read_lsn,
                    span: span.clone(),
                })
            }
        }
    }

    /// Lower a `WITH FUSION = RRF(k = N)` clause to
    /// [`LogicalPlan::Fusion`] wrapping the previous plan.
    fn lower_with_fusion(
        &mut self,
        f: &BoundWithFusionClause,
        prev: Option<LogicalPlan>,
    ) -> LogicalPlan {
        let spec = match &f.fusion {
            BoundFusion::Rrf { k } => FusionSpec {
                kind: FusionKind::Rrf,
                k: if *k <= 0 { 0 } else { *k as u64 },
                span: f.span.clone(),
            },
        };
        let inputs = match prev {
            None => Vec::new(),
            Some(mut p) => {
                attach_fusion_to_rank_by_hybrid(&mut p, &spec);
                vec![Box::new(p)]
            }
        };
        LogicalPlan::Fusion(LogicalFusion {
            spec,
            inputs,
            span: f.span.clone(),
        })
    }

    /// Resolve a [`BoundFieldRef`] to `(var, property_name)`. The
    /// field's base must be a bound variable (M4-21 binding has
    /// already validated this); the path must be a single segment
    /// (M4-22 admits property paths of length-1 for VECTOR / TEXT
    /// operands at v1.0).
    fn field_to_var_and_property(&mut self, field: &BoundFieldRef) -> Option<(BindingId, String)> {
        if field.base.binding_id.raw() == u64::MAX {
            // Sentinel for an unresolved variable (set by
            // BindingVisitor::bind_field_ref when resolution fails).
            // M4-21 has already emitted UndeclaredVariable; defensive
            // skip here.
            return None;
        }
        let property = {
            let p = field.path.first()?;
            p.name.clone()
        };
        Some((field.base.binding_id, property))
    }
}

/// Attach a `WITH FUSION` specification to the immediately preceding
/// hybrid retrieval. `MATCH ... RANK BY ...` lowers to a join whose
/// right child is the retrieval leaf, while a leading `RANK BY` lowers
/// directly to that leaf.
fn attach_fusion_to_rank_by_hybrid(plan: &mut LogicalPlan, spec: &FusionSpec) -> bool {
    match plan {
        LogicalPlan::RankByHybrid(rank) => {
            rank.fusion = Some(spec.clone());
            true
        }
        LogicalPlan::Join(join) => {
            attach_fusion_to_rank_by_hybrid(&mut join.right, spec)
                || attach_fusion_to_rank_by_hybrid(&mut join.left, spec)
        }
        _ => false,
    }
}

// =====================================================================
// Helpers
// =====================================================================

/// Match the canonical community-equality shape:
/// `community(n) = $cid` or `$cid = community(n)`.
///
/// Returns `Some((node_var, community_id))` if exactly one operand is
/// a `FunctionCall("community", [VariableRef(n)])` and the other is
/// the community-id expression. Returns `None` otherwise.
///
/// Order independence: the BinaryOp's `lhs` / `rhs` may carry the
/// `community(...)` call in either position. Both encodings yield the
/// same `LogicalCommunityLookup` tree per ADR-038 §2 D-26 (closes
/// PR #154 reviewer Finding 5 at the logical-plan level).
fn match_community_equality(
    lhs: &BoundExpression,
    rhs: &BoundExpression,
) -> Option<(BindingId, BoundExpression)> {
    if let Some(node_var) = match_community_call(lhs) {
        return Some((node_var, rhs.clone()));
    }
    if let Some(node_var) = match_community_call(rhs) {
        return Some((node_var, lhs.clone()));
    }
    None
}

/// Match `community(<VariableRef>)` and return the variable's
/// binding id.
fn match_community_call(e: &BoundExpression) -> Option<BindingId> {
    match e {
        BoundExpression::FunctionCall { name, args, .. }
            if name.eq_ignore_ascii_case("community") && args.len() == 1 =>
        {
            match &args[0] {
                BoundExpression::VariableRef { binding_id, .. } => Some(*binding_id),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Match a length-1 property access on a bound variable:
/// `n.prop` where `n` is a `VariableRef`. Returns
/// `Some((binding_id, property_name))` on a match.
fn match_property_access(e: &BoundExpression) -> Option<(BindingId, String)> {
    if let BoundExpression::PropertyAccess { base, path, .. } = e {
        if let BoundExpression::VariableRef { binding_id, .. } = base.as_ref() {
            if let Some(p) = path.first() {
                return Some((*binding_id, p.name.clone()));
            }
        }
    }
    None
}

/// Translate the optional `K` parameter from a `BoundRankArg` into a
/// non-negative `u64`. Records a NotImplemented marker if K is missing
/// (defensive — M4-23 already rejects bare operands without K).
fn rank_arg_k(k: Option<i64>, span: &Span, errors: &mut Vec<LogicalPlanError>) -> u64 {
    match k {
        Some(n) if n > 0 => n as u64,
        _ => {
            errors.push(LogicalPlanError::NotImplementedAtM4_31 {
                surface: "RANK BY operand without explicit K",
                target_slice: "v1.0 (per ADR-038 D-3 — defensive)",
                span: span.clone(),
            });
            0
        }
    }
}

/// Resolve a node pattern's single label (M4-31 admits at most one
/// label per node — multi-label `(:A:B)` is rejected by M4-22 reserved-
/// variant rejection upstream of M4-31).
fn single_label(
    n: &BoundNodePattern,
    errors: &mut Vec<LogicalPlanError>,
) -> Option<arcgraph_core::LabelId> {
    match n.labels.len() {
        0 => None,
        1 => Some(n.labels[0].label_id),
        _ => {
            errors.push(LogicalPlanError::NotImplementedAtM4_31 {
                surface: "multi-label node pattern",
                target_slice: "v1.1 (per ADR-038 D-9)",
                span: n.span.clone(),
            });
            Some(n.labels[0].label_id)
        }
    }
}

/// Resolve a rel pattern's single rel-type (M4-31 admits at most one
/// rel-type — multi-type `:A|:B` is rejected upstream).
fn single_rel_type(
    r: &BoundRelPattern,
    errors: &mut Vec<LogicalPlanError>,
) -> Option<arcgraph_core::TypeId> {
    match r.rel_types.len() {
        0 => None,
        1 => Some(r.rel_types[0].type_id),
        _ => {
            errors.push(LogicalPlanError::NotImplementedAtM4_31 {
                surface: "multi-type relationship pattern",
                target_slice: "v1.1 (per ADR-038 D-9)",
                span: r.span.clone(),
            });
            Some(r.rel_types[0].type_id)
        }
    }
}

/// Extract a non-negative integer literal from a [`BoundExpression`].
/// Returns `None` for parameter-driven, computed, or negative values.
fn literal_int(e: &BoundExpression) -> Option<u64> {
    match e {
        BoundExpression::Literal {
            value: Literal::Integer(n),
            ..
        } => {
            if *n < 0 {
                None
            } else {
                Some(*n as u64)
            }
        }
        _ => None,
    }
}

/// Find the maximum BindingId observed in a BoundQuery. Returns 0 if
/// the query has no bindings.
fn collect_max_binding_id(q: &BoundQuery) -> u64 {
    let mut max = 0u64;
    for c in &q.clauses {
        max_in_clause(c, &mut max);
    }
    max
}

fn max_in_clause(c: &BoundClause, max: &mut u64) {
    match c {
        BoundClause::Match(m) => {
            match &m.body {
                BoundMatchBody::Patterns(ps) => {
                    for p in ps {
                        max_in_path(p, max);
                    }
                }
                BoundMatchBody::NamedPath(np) => {
                    bump(np.var.binding_id, max);
                    let pp = match &np.kind {
                        crate::semantic::bound_ast::BoundNamedPathKind::ShortestPath(p)
                        | crate::semantic::bound_ast::BoundNamedPathKind::AllShortestPath(p)
                        | crate::semantic::bound_ast::BoundNamedPathKind::Plain(p) => p,
                    };
                    max_in_path(pp, max);
                }
            }
            if let Some(w) = &m.where_clause {
                max_in_expr(w, max);
            }
        }
        BoundClause::With(w) => {
            for it in &w.items {
                // #864 — the projection OUTPUT id (#746) is a binding-pass id
                // too; the synthetic-id seed MUST observe it, else a lowering-
                // synthesized id (e.g. a #864 hidden sort column) can collide
                // with an output id and the post-Project schema becomes
                // ambiguous (the Sort resolves the wrong column).
                if let Some(out) = it.output_id {
                    bump(out, max);
                }
                if let BoundProjectionKind::Expr(e) = &it.kind {
                    max_in_expr(e, max);
                }
            }
            if let Some(e) = &w.where_clause {
                max_in_expr(e, max);
            }
        }
        BoundClause::Unwind(u) => {
            bump(u.var.binding_id, max);
            max_in_expr(&u.expr, max);
        }
        BoundClause::CallProcedure(c) => {
            for y in &c.yields {
                bump(y.var.binding_id, max);
            }
            for a in &c.args {
                max_in_expr(a, max);
            }
            if let Some(w) = &c.where_clause {
                max_in_expr(w, max);
            }
        }
        BoundClause::Show(s) => {
            for col in &s.columns {
                bump(col.binding_id, max);
            }
            // #830 — the optional WHERE references the YIELD'd columns.
            if let Some(w) = &s.where_clause {
                max_in_expr(w, max);
            }
        }
        BoundClause::Return(r) => {
            for it in &r.items {
                // #864 — observe the projection OUTPUT id (see the `With` arm).
                if let Some(out) = it.output_id {
                    bump(out, max);
                }
                if let BoundProjectionKind::Expr(e) = &it.kind {
                    max_in_expr(e, max);
                }
            }
            for o in &r.order_by {
                max_in_expr(&o.expr, max);
            }
            if let Some(e) = &r.skip {
                max_in_expr(e, max);
            }
            if let Some(e) = &r.limit {
                max_in_expr(e, max);
            }
        }
        BoundClause::TailOrderBy(items, _) => {
            for o in items {
                max_in_expr(&o.expr, max);
            }
        }
        BoundClause::TailSkip(e, _) | BoundClause::TailLimit(e, _) => {
            max_in_expr(e, max);
        }
        BoundClause::RankBy(_) | BoundClause::WithFusion(_) => {}
        BoundClause::Delete(d) => {
            // ADR-149 W26-θ Phase 3: DELETE items reference upstream
            // bindings (the binding pass resolved them); bump the
            // max so subsequent visitor passes treat the references
            // as live.
            for item in &d.items {
                bump(item.var.binding_id, max);
            }
        }
        BoundClause::Set(s) => {
            // ADR-150 W26-θ Phase 4: SET items reference upstream
            // bindings; PropertyAssign / Replace / Merge property
            // values are literal-only at Phase 4 (no embedded
            // bindings), but we walk them defensively.
            for item in &s.items {
                bump(item.var.binding_id, max);
                match &item.mutation {
                    BoundSetMutation::PropertyAssign { value, .. } => {
                        max_in_expr(value, max);
                    }
                    BoundSetMutation::PropertyReplace(map)
                    | BoundSetMutation::PropertyMerge(map) => {
                        for entry in &map.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                    BoundSetMutation::LabelAdd(_) => {}
                }
            }
        }
        BoundClause::Remove(r) => {
            // ADR-150 W26-θ Phase 4: REMOVE items reference upstream
            // bindings; mutations carry no expression sub-trees.
            for item in &r.items {
                bump(item.var.binding_id, max);
            }
        }
        BoundClause::Create(c) => {
            // ADR-147 W26-θ Phase 1 + ADR-148 W26-θ Phase 2: each
            // CreateNodeSpec / CreatePathSpec's optional vars
            // contribute their binding_ids; property-value expressions
            // are literal-only (no embedded bindings).
            for item in &c.items {
                match item {
                    crate::semantic::bound_ast::BoundCreateItem::Node(spec) => {
                        if let Some(v) = &spec.var {
                            bump(v.binding_id, max);
                        }
                        if let Some(props) = &spec.properties {
                            for entry in &props.entries {
                                max_in_expr(&entry.value, max);
                            }
                        }
                    }
                    crate::semantic::bound_ast::BoundCreateItem::Path(path) => {
                        if let Some(v) = &path.source.var {
                            bump(v.binding_id, max);
                        }
                        if let Some(v) = &path.target.var {
                            bump(v.binding_id, max);
                        }
                        if let Some(v) = &path.rel.var {
                            bump(v.binding_id, max);
                        }
                        if let Some(props) = &path.source.properties {
                            for entry in &props.entries {
                                max_in_expr(&entry.value, max);
                            }
                        }
                        if let Some(props) = &path.rel.properties {
                            for entry in &props.entries {
                                max_in_expr(&entry.value, max);
                            }
                        }
                        if let Some(props) = &path.target.properties {
                            for entry in &props.entries {
                                max_in_expr(&entry.value, max);
                            }
                        }
                    }
                }
            }
        }
        BoundClause::Merge(m) => {
            // ADR-151 W26-θ Phase 5: MERGE pattern declares fresh
            // bindings (like CREATE); on_create / on_match action
            // items reference those bindings (like SET).
            match &m.pattern {
                BoundMergePattern::Node(spec) => {
                    if let Some(v) = &spec.var {
                        bump(v.binding_id, max);
                    }
                    if let Some(props) = &spec.properties {
                        for entry in &props.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                }
                BoundMergePattern::Path(path) => {
                    if let Some(v) = &path.source.var {
                        bump(v.binding_id, max);
                    }
                    if let Some(v) = &path.target.var {
                        bump(v.binding_id, max);
                    }
                    if let Some(v) = &path.rel.var {
                        bump(v.binding_id, max);
                    }
                    if let Some(props) = &path.source.properties {
                        for entry in &props.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                    if let Some(props) = &path.rel.properties {
                        for entry in &props.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                    if let Some(props) = &path.target.properties {
                        for entry in &props.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                }
            }
            for item in m.on_create.iter().chain(m.on_match.iter()) {
                bump(item.var.binding_id, max);
                match &item.mutation {
                    BoundSetMutation::PropertyAssign { value, .. } => {
                        max_in_expr(value, max);
                    }
                    BoundSetMutation::PropertyReplace(map)
                    | BoundSetMutation::PropertyMerge(map) => {
                        for entry in &map.entries {
                            max_in_expr(&entry.value, max);
                        }
                    }
                    BoundSetMutation::LabelAdd(_) => {}
                }
            }
        }
        BoundClause::Call(c) => {
            // ADR-192 (#623): the subquery body's binding-ids live in the
            // SAME id space as the outer query (the bind pass mints ids
            // monotonically across the whole statement). Recurse into the
            // body so `next_anon_id` is seeded above the body's ids too —
            // otherwise a synthetic anon-node id minted while lowering the
            // body could collide with a real body binding-id. Also bump
            // the imported + returned ids.
            for b in c.imported.iter().chain(c.returned.iter()) {
                bump(*b, max);
            }
            max_in_call_body(&c.body, max);
        }
    }
}

/// Recurse [`max_in_clause`] into a `CALL { … }` subquery body so the
/// anon-id seed covers the body's binding-ids (ADR-192 #623).
fn max_in_call_body(body: &BoundStatement, max: &mut u64) {
    match body {
        BoundStatement::Read(q) => {
            for c in &q.clauses {
                max_in_clause(c, max);
            }
        }
        BoundStatement::Union(u) => {
            for arm in &u.arms {
                for c in &arm.clauses {
                    max_in_clause(c, max);
                }
            }
        }
        _ => {}
    }
}

fn max_in_path(p: &BoundPathPattern, max: &mut u64) {
    if let Some(v) = &p.head.var {
        bump(v.binding_id, max);
    }
    for (rel, node) in &p.tail {
        if let Some(v) = &rel.var {
            bump(v.binding_id, max);
        }
        if let Some(v) = &node.var {
            bump(v.binding_id, max);
        }
    }
}

fn max_in_expr(e: &BoundExpression, max: &mut u64) {
    match e {
        BoundExpression::VariableRef { binding_id, .. } => bump(*binding_id, max),
        BoundExpression::PropertyAccess { base, .. } => max_in_expr(base, max),
        BoundExpression::ListLiteral { elements, .. } => {
            for element in elements {
                max_in_expr(element, max);
            }
        }
        BoundExpression::MapLiteral { entries, .. } => {
            for (_, value) in entries {
                max_in_expr(value, max);
            }
        }
        // #1290 — left-nested operator SPINE walked iteratively (the
        // spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may interleave
        // BinaryOp / UnaryOp / In / IsNull levels; recursing per level
        // overflowed the native stack). Base subtree first, then each
        // level's rhs innermost→outermost — the same visit order as the
        // recursion this replaces. Non-spine children recurse
        // (bracket-bounded by `MAX_EXPRESSION_DEPTH`).
        BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        | BoundExpression::In { .. }
        | BoundExpression::IsNull { .. } => {
            let mut rhs_stack: Vec<&BoundExpression> = Vec::new();
            let mut cur = e;
            loop {
                match cur {
                    BoundExpression::BinaryOp { lhs, rhs, .. }
                    | BoundExpression::In { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    BoundExpression::UnaryOp { operand, .. } => cur = operand,
                    BoundExpression::IsNull { lhs, .. } => cur = lhs,
                    other => {
                        max_in_expr(other, max);
                        break;
                    }
                }
            }
            while let Some(rhs) = rhs_stack.pop() {
                max_in_expr(rhs, max);
            }
        }
        BoundExpression::FunctionCall { args, .. } => {
            for a in args {
                max_in_expr(a, max);
            }
        }
        BoundExpression::Near { lhs, target, .. } => {
            max_in_expr(lhs, max);
            max_in_expr(target, max);
        }
        BoundExpression::TextMatch { lhs, query, .. } => {
            max_in_expr(lhs, max);
            max_in_expr(query, max);
        }
        BoundExpression::InCommunity {
            node, community, ..
        } => {
            max_in_expr(node, max);
            max_in_expr(community, max);
        }
        // ADR-188 — list-predicates / reduce carry their scoped
        // binding_ids (`var_bid`, and `acc_bid` for reduce); count them
        // so the anonymous-binding seed never collides, AND descend into
        // every sub-expression (the list / predicate / init / body may
        // reference OUTER row bindings, e.g.
        // `all(x IN n.friends WHERE x > n.threshold)`).
        BoundExpression::ListPredicate {
            var_bid,
            list,
            predicate,
            ..
        } => {
            bump(*var_bid, max);
            max_in_expr(list, max);
            max_in_expr(predicate, max);
        }
        BoundExpression::Reduce {
            acc_bid,
            init,
            var_bid,
            list,
            expr,
            ..
        } => {
            bump(*acc_bid, max);
            bump(*var_bid, max);
            max_in_expr(init, max);
            max_in_expr(list, max);
            max_in_expr(expr, max);
        }
        // ADR-188 (#620 list-half) — a list-comprehension carries its
        // scoped `var_bid`; count it so the anonymous-binding seed never
        // collides, AND descend into every sub-expression (the list /
        // predicate / projection may reference OUTER row bindings, e.g.
        // `[x IN n.friends WHERE x > n.threshold | x + n.base]`).
        BoundExpression::ListComprehension {
            var_bid,
            list,
            predicate,
            projection,
            ..
        } => {
            bump(*var_bid, max);
            max_in_expr(list, max);
            if let Some(p) = predicate {
                max_in_expr(p, max);
            }
            if let Some(e) = projection {
                max_in_expr(e, max);
            }
        }
        // ADR-191 D-6 (#620 map-half) — a map projection carries NO scoped
        // var (no `bump`); descend into the base (a row VariableRef) + each
        // literal-entry value (which may reference outer row bindings, e.g.
        // `n{x: n.base + 1}`). The `.key` / `.*` selectors carry only
        // property names (no binding).
        BoundExpression::MapProjection { base, items, .. } => {
            max_in_expr(base, max);
            for item in items {
                if let BoundMapProjectionItem::Literal { value, .. } = item {
                    max_in_expr(value, max);
                }
            }
        }
        // #621 — openCypher v9 §3.4 postfix accessors carry NO scoped
        // binding; descend into base + index / bounds (which may reference
        // outer row bindings, e.g. `n.friends[n.idx]`).
        BoundExpression::Subscript { base, index, .. } => {
            max_in_expr(base, max);
            max_in_expr(index, max);
        }
        BoundExpression::Slice {
            base, start, end, ..
        } => {
            max_in_expr(base, max);
            if let Some(s) = start {
                max_in_expr(s, max);
            }
            if let Some(e) = end {
                max_in_expr(e, max);
            }
        }
        // #621 — a CASE carries NO scoped binding (unlike the
        // comprehensions); descend into the test + every WHEN / THEN + the
        // ELSE (each may reference outer row bindings, e.g.
        // `CASE WHEN n.flag THEN n.a ELSE n.b END`).
        BoundExpression::Case {
            test,
            branches,
            default,
            ..
        } => {
            if let Some(t) = test {
                max_in_expr(t, max);
            }
            for (when, then) in branches {
                max_in_expr(when, max);
                max_in_expr(then, max);
            }
            if let Some(d) = default {
                max_in_expr(d, max);
            }
        }
        BoundExpression::Literal { .. }
        | BoundExpression::Parameter { .. }
        | BoundExpression::UnresolvedVariable { .. } => {}
    }
}

fn bump(id: BindingId, max: &mut u64) {
    if id.raw() > *max {
        *max = id.raw();
    }
}

/// ADR-148 W26-θ Phase 2 helper — flatten a `Option<&BoundPropertyMap>`
/// to the `Vec<(String, BoundExpression)>` shape consumed by
/// [`LogicalCreateNode::properties`] / [`LogicalCreateRel::properties`].
/// Returns the empty vec when the input is `None`.
fn bound_props_to_vec(
    m: Option<&crate::semantic::bound_ast::BoundPropertyMap>,
) -> Vec<(String, BoundExpression)> {
    m.map(|map| {
        map.entries
            .iter()
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect()
    })
    .unwrap_or_default()
}

fn lower_create_endpoint(endpoint: CreateEndpointBinding) -> LogicalCreateEndpoint {
    match endpoint {
        CreateEndpointBinding::Fresh => LogicalCreateEndpoint::Fresh,
        CreateEndpointBinding::RowBinding(binding) => LogicalCreateEndpoint::RowBinding(binding),
    }
}

/// **#1290** — the maximum number of GENERIC WHERE conjuncts lowered as
/// individually-nested [`LogicalFilter`] nodes before the remainder
/// folds into a single Filter carrying their left-nested `AND` spine.
///
/// Budget: every plan-tree pass (join-order `rewrite`, algorithm
/// picking, cost walk, pipeline compile, per-batch operator pull, plan
/// `Drop` glue) recurses once per plan node; the heaviest measured
/// frame (`enumeration::rewrite`, a by-value `LogicalPlan` match) is
/// ~10 KiB in the debug profile — 200 nested Filters overflowed a
/// 2 MiB test thread E2E. 32 nodes × ~10 KiB ≈ 320 KiB worst-case for
/// the whole chain (≥6× margin on a 2 MiB stack), while staying far
/// above the conjunct count of any TCK scenario or hand-written query
/// (so the per-conjunct plan shape — one EXPLAIN row per predicate,
/// per-predicate SIMD detection — is byte-identical for every realistic
/// query). Wide generated filters (up to `MAX_FLAT_CHAIN_DEPTH`
/// conjuncts) execute via the fused-spine Filter, whose expression walk
/// is iterative post-#1290.
const MAX_FILTER_CHAIN_NODES: usize = 32;

/// **#1290** — `true` iff `pred` is one of the hybrid-retrieval shapes
/// [`LogicalPlanLoweringVisitor::apply_where`] lowers to a DEDICATED
/// plan operator instead of a generic [`LogicalFilter`]. Such conjuncts
/// must never fold into a fused AND-spine Filter (an un-lowered `NEAR`
/// / text `MATCH` / community predicate inside a generic Filter
/// predicate is un-executable — the evaluator returns
/// `NotImplemented`).
///
/// MUST stay in lockstep with the recognizers
/// `try_lower_community_predicate` / `try_lower_near_predicate` /
/// `try_lower_text_match_predicate` (pinned by the
/// `hybrid_shape_classifier_matches_apply_where` unit test below): a
/// shape is classified hybrid exactly when one of those would return
/// `Some`.
fn is_hybrid_predicate_shape(pred: &BoundExpression) -> bool {
    match pred {
        // `n IN COMMUNITY($cid)` — recognized when the node operand is
        // a variable reference.
        BoundExpression::InCommunity { node, .. } => {
            matches!(node.as_ref(), BoundExpression::VariableRef { .. })
        }
        // Canonical `community(n) = $cid` / `$cid = community(n)`.
        BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs,
            rhs,
            ..
        } => match_community_equality(lhs, rhs).is_some(),
        // `<var>.<prop> NEAR <expr>` / `<var>.<prop> MATCH <expr>`.
        BoundExpression::Near { lhs, .. } | BoundExpression::TextMatch { lhs, .. } => {
            match_property_access(lhs).is_some()
        }
        // `vector_distance(field, q)` / `text_match(field, q)`
        // function-call shapes.
        BoundExpression::FunctionCall { name, args, .. } => {
            (name.eq_ignore_ascii_case("vector_distance")
                || name.eq_ignore_ascii_case("text_match"))
                && args.len() == 2
                && match_property_access(&args[0]).is_some()
        }
        _ => false,
    }
}

/// **#1290** — the `WITH … WHERE` twin of
/// [`LogicalPlanLoweringVisitor::apply_where_conjuncts`]: wrap up to
/// [`MAX_FILTER_CHAIN_NODES`] conjuncts as individually-nested
/// [`LogicalFilter`] nodes (the pre-#1290 shape), folding a wider set
/// into ONE Filter carrying the left-nested `AND` spine so the plan
/// depth stays bounded. `WITH … WHERE` conjuncts never carry the
/// hybrid-retrieval shapes (`apply_where`'s recognizers only run on
/// MATCH-clause predicates), so no shape partition applies here.
fn apply_plain_filter_conjuncts(input: LogicalPlan, preds: &[BoundExpression]) -> LogicalPlan {
    if preds.len() <= MAX_FILTER_CHAIN_NODES {
        return preds.iter().fold(input, |plan, pred| {
            LogicalPlan::Filter(LogicalFilter {
                input: Box::new(plan),
                span: pred.span().clone(),
                predicate: pred.clone(),
            })
        });
    }
    let refs: Vec<&BoundExpression> = preds.iter().collect();
    let predicate = fold_and_spine(&refs)
        .expect("preds.len() > MAX_FILTER_CHAIN_NODES >= 1, so `preds` is non-empty");
    let span = predicate.span().clone();
    LogicalPlan::Filter(LogicalFilter {
        input: Box::new(input),
        predicate,
        span,
    })
}

/// **#1290** — left-fold `preds` into the `AND` spine
/// `(((p1 AND p2) AND p3) … AND pN)`, preserving conjunct order.
/// Returns `None` for an empty slice. The synthesized `And` nodes
/// carry `TypeInfo::Boolean` (what the type-check stamps for a
/// boolean-operand `AND`; they are created post-type-check, matching
/// the `hidden_ref` / `property_eq_expr` post-check synthesis
/// discipline).
fn fold_and_spine(preds: &[&BoundExpression]) -> Option<BoundExpression> {
    let mut iter = preds.iter();
    let mut acc = (*iter.next()?).clone();
    for pred in iter {
        let span = acc.span().clone();
        acc = BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs: Box::new(acc),
            rhs: Box::new((*pred).clone()),
            span,
            type_info: Some(TypeInfo::Boolean),
        };
    }
    Some(acc)
}

/// ADR-152 §D-4 helper — synthesize a Filter predicate from a
/// pattern's inline property literal map.
///
/// For a pattern `(n {k1: v1, k2: v2})`, returns the
/// `BoundExpression` representing
/// `n.k1 = v1 AND n.k2 = v2`. Returns `None` when the map is empty
/// (the caller skips wrapping the Scan in a Filter).
///
/// The synthesized expression's `type_info` slots are populated
/// best-effort:
/// - The variable reference's `type_info` is forwarded from the
///   pattern's `BoundVariable::type_info` so the M4-22 type-check
///   downstream knows it's a Node.
/// - The PropertyAccess + BinaryOp `type_info` slots are left
///   `None` — the executor's `eval` doesn't need them at v1.0-α
///   per the existing PropertyAccess + apply_binop machinery.
/// - The Literal RHS expressions are forwarded verbatim from the
///   bound-AST entries (already carrying their own type_info from
///   the binding pass).
fn properties_to_filter_predicate(
    var_binding: BindingId,
    var_name: &str,
    var_span: &Span,
    var_type_info: Option<TypeInfo>,
    properties: &crate::semantic::bound_ast::BoundPropertyMap,
) -> Option<BoundExpression> {
    let mut iter = properties.entries.iter();
    let first = iter.next()?;
    let mut acc = property_eq_expr(
        var_binding,
        var_name,
        var_span,
        var_type_info.clone(),
        first,
    );
    for entry in iter {
        let term = property_eq_expr(
            var_binding,
            var_name,
            var_span,
            var_type_info.clone(),
            entry,
        );
        let span = entry.span.clone();
        acc = BoundExpression::BinaryOp {
            op: crate::ast::BinOp::And,
            lhs: Box::new(acc),
            rhs: Box::new(term),
            span,
            type_info: None,
        };
    }
    Some(acc)
}

/// Build a single `var.key = literal` BoundExpression for one entry
/// in a property literal map. Used by [`properties_to_filter_predicate`]
/// to build the per-key AND-conjunction.
fn property_eq_expr(
    var_binding: BindingId,
    var_name: &str,
    var_span: &Span,
    var_type_info: Option<TypeInfo>,
    entry: &crate::semantic::bound_ast::BoundPropertyEntry,
) -> BoundExpression {
    let prop_ref = crate::semantic::bound_ast::BoundPropertyRef {
        name: entry.key.clone(),
        property_id: entry.property_id,
        span: entry.span.clone(),
    };
    let var_ref = BoundExpression::VariableRef {
        name: var_name.to_string(),
        binding_id: var_binding,
        span: var_span.clone(),
        type_info: var_type_info,
    };
    let prop_access = BoundExpression::PropertyAccess {
        base: Box::new(var_ref),
        path: vec![prop_ref],
        span: entry.span.clone(),
        type_info: None,
    };
    BoundExpression::BinaryOp {
        op: crate::ast::BinOp::Eq,
        lhs: Box::new(prop_access),
        rhs: Box::new(entry.value.clone()),
        span: entry.span.clone(),
        type_info: None,
    }
}

/// ADR-152 §D-4 helper — wrap a sub-plan in a [`LogicalFilter`] when
/// the node pattern carries inline property literals; otherwise
/// return the sub-plan unchanged.
fn wrap_node_with_property_filter(
    plan: LogicalPlan,
    pattern: &BoundNodePattern,
    binding: BindingId,
) -> LogicalPlan {
    let Some(props) = pattern.properties.as_ref() else {
        return plan;
    };
    if props.entries.is_empty() {
        return plan;
    }
    let name = pattern
        .var
        .as_ref()
        .map(|v| v.name.as_str())
        .unwrap_or("_pattern_anon");
    let var_type_info = pattern.var.as_ref().and_then(|v| v.type_info.clone());
    let Some(predicate) =
        properties_to_filter_predicate(binding, name, &pattern.span, var_type_info, props)
    else {
        return plan;
    };
    let span = predicate.span().clone();
    LogicalPlan::Filter(LogicalFilter {
        input: Box::new(plan),
        predicate,
        span,
    })
}

/// ADR-152 §D-4 helper — wrap a sub-plan in a [`LogicalFilter`] when
/// the rel pattern carries inline property literals.
fn wrap_rel_with_property_filter(
    plan: LogicalPlan,
    rel: &BoundRelPattern,
    binding: Option<BindingId>,
) -> LogicalPlan {
    let Some(props) = rel.properties.as_ref() else {
        return plan;
    };
    if props.entries.is_empty() {
        return plan;
    }
    let Some(rel_binding) = binding else {
        // Anonymous rel patterns cannot carry filter predicates at
        // v1.0-α — the PropertyAccess needs a bound variable. The
        // M4-22 type-check pass forbids `(a)-[{p:v}]->(b)` (no rel
        // var) from carrying property literals; this branch is
        // defensive.
        return plan;
    };
    let name = rel
        .var
        .as_ref()
        .map(|v| v.name.as_str())
        .unwrap_or("_rel_anon");
    let var_type_info = rel.var.as_ref().and_then(|v| v.type_info.clone());
    let Some(predicate) =
        properties_to_filter_predicate(rel_binding, name, &rel.span, var_type_info, props)
    else {
        return plan;
    };
    let span = predicate.span().clone();
    LogicalPlan::Filter(LogicalFilter {
        input: Box::new(plan),
        predicate,
        span,
    })
}

/// ADR-152 §D-4 helper — wrap a sub-plan in a [`LogicalFilter`] when
/// a `BoundCreateNodeSpec` (used by MERGE patterns as the pattern
/// shape) carries inline property literals.
fn wrap_create_node_spec_with_property_filter(
    plan: LogicalPlan,
    spec: &crate::semantic::bound_ast::BoundCreateNodeSpec,
    binding: BindingId,
) -> LogicalPlan {
    let Some(props) = spec.properties.as_ref() else {
        return plan;
    };
    if props.entries.is_empty() {
        return plan;
    }
    let name = spec
        .var
        .as_ref()
        .map(|v| v.name.as_str())
        .unwrap_or("_merge_anon");
    let var_type_info = spec.var.as_ref().and_then(|v| v.type_info.clone());
    let Some(predicate) =
        properties_to_filter_predicate(binding, name, &spec.span, var_type_info, props)
    else {
        return plan;
    };
    let span = predicate.span().clone();
    LogicalPlan::Filter(LogicalFilter {
        input: Box::new(plan),
        predicate,
        span,
    })
}

/// Compute the set of BindingIds that appear in BOTH plan subtrees.
/// Used by [`LogicalPlanLoweringVisitor::lower_match`] +
/// [`LogicalPlanLoweringVisitor::lower_patterns`] +
/// [`LogicalPlanLoweringVisitor::lower_path_pattern`] to derive the
/// shared-binding equi-join condition.
fn shared_bindings(left: &LogicalPlan, right: &LogicalPlan) -> Vec<BindingId> {
    let mut left_set = std::collections::BTreeSet::new();
    collect_bindings(left, &mut left_set);
    let mut right_set = std::collections::BTreeSet::new();
    collect_bindings(right, &mut right_set);
    left_set.intersection(&right_set).copied().collect()
}

fn collect_bindings(p: &LogicalPlan, out: &mut std::collections::BTreeSet<BindingId>) {
    match p {
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
        LogicalPlan::Filter(f) => collect_bindings(&f.input, out),
        LogicalPlan::Project(p) => {
            // #841: a Project's join-visible bindings are its OUTPUT
            // schema, NOT its input. A `WITH a` / `RETURN a` projection
            // RENAMES — the binder mints a FRESH `output_id` for each
            // item (#746), and `ProjectOp::derive_schema` emits the row
            // under THAT id. Recursing into `p.input` here surfaced the
            // PRE-projection ids (absent from the Project's output rows),
            // so `shared_bindings(Project, pattern)` found NO overlap with
            // a downstream `MATCH (a)` that re-references the projected
            // variable by its output id → an EMPTY equi-join key → a
            // silent CARTESIAN (`MATCH (a) … WITH a MATCH (a)-[…]->(b)`,
            // and the correlated `CALL { WITH a MATCH (a)…}` of #841).
            // Mirror `derive_schema`: a Wildcard passes the input schema
            // through; an Expr item contributes its `output_id` (a missing
            // output_id ⇒ a synthetic exec-only id nothing references,
            // contributing nothing — matching the executor).
            for item in &p.items {
                match &item.kind {
                    BoundProjectionKind::Wildcard { .. } => collect_bindings(&p.input, out),
                    BoundProjectionKind::Expr(_) => {
                        if let Some(id) = item.output_id {
                            out.insert(id);
                        }
                    }
                }
            }
        }
        LogicalPlan::Join(j) => {
            collect_bindings(&j.left, out);
            collect_bindings(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_bindings(&j.left, out);
            collect_bindings(&j.right, out);
        }
        LogicalPlan::Limit(l) => collect_bindings(&l.input, out),
        LogicalPlan::Skip(s) => collect_bindings(&s.input, out),
        LogicalPlan::RankByHybrid(r) => {
            for op in &r.operands {
                out.insert(op.var);
            }
            if let Some(score) = r.score_binding {
                out.insert(score);
            }
        }
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_bindings(inp, out);
            }
        }
        // ADR-185 (#649-A1, W28): a UNION ALL is a top-level set op,
        // not a join input, so this is not reached on the join-
        // derivation path in practice; recurse into the arms for
        // defensive exhaustiveness (consistent with `Fusion`).
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_bindings(arm, out);
            }
        }
        LogicalPlan::CommunityLookup(c) => {
            collect_bindings(&c.input, out);
            out.insert(c.node_var);
        }
        LogicalPlan::VectorNear(v) => {
            out.insert(v.var);
        }
        LogicalPlan::TextMatch(t) => {
            out.insert(t.var);
        }
        LogicalPlan::Aggregate(a) => {
            // #841 (sister of the Project arm): an Aggregate's
            // join-visible bindings are its OUTPUT schema — the group-by
            // items' + aggregations' `output_id`s (mirror
            // `AggregateOp::new`'s schema) — NOT its input. A
            // `WITH a, count(b) AS n MATCH (a)…` re-references the
            // group-key `a` by its projected output id; recursing into
            // the input would surface the pre-aggregation ids (absent from
            // the Aggregate's output rows) → an empty join key → the same
            // silent CARTESIAN. A group-by item with no `output_id` ⇒ a
            // synthetic exec-only id (contributes nothing), matching
            // `AggregateOp`.
            for item in &a.group_by {
                if let Some(id) = item.output_id {
                    out.insert(id);
                }
            }
            for call in &a.aggregations {
                out.insert(call.output_id);
            }
        }
        LogicalPlan::Sort(s) => collect_bindings(&s.input, out),
        LogicalPlan::Distinct(d) => collect_bindings(&d.input, out),
        LogicalPlan::Unwind(u) => {
            collect_bindings(&u.input, out);
            out.insert(u.var);
        }
        LogicalPlan::ProcedureCall(p) => {
            collect_bindings(&p.input, out);
            for (_, bid) in &p.columns {
                out.insert(*bid);
            }
        }
        LogicalPlan::NamedPath(np) => {
            collect_bindings(&np.input, out);
            out.insert(np.path_var);
        }
        LogicalPlan::DynamicLimit(l) => collect_bindings(&l.input, out),
        LogicalPlan::CreateNode(c) => {
            // ADR-147 W26-θ Phase 1: contributes its var binding to
            // the downstream binding set (if any).
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf DDL — declares
        // no query bindings (returns 0 rows / 0 columns).
        LogicalPlan::CreateVectorIndex(_) => {}
        // #1366: CREATE INDEX (property index) is a leaf DDL.
        LogicalPlan::CreatePropertyIndex(_) => {}
        LogicalPlan::CreateRel(c) => {
            // ADR-148 W26-θ Phase 2: walks the source + target
            // sub-plans + contributes the rel var (if any). The
            // source / target bindings are produced by the sub-plans
            // so they're surfaced by `collect_bindings` on the
            // sub-plans (no duplicate insertion here).
            collect_bindings(&c.source_plan, out);
            collect_bindings(&c.target_plan, out);
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        LogicalPlan::Delete(d) => {
            // ADR-149 W26-θ Phase 3: passes through the input plan's
            // binding set. Delete itself contributes NO new bindings
            // — items are references, not declarations.
            collect_bindings(&d.input, out);
        }
        LogicalPlan::Set(s) => {
            // ADR-150 W26-θ Phase 4: passes through the input plan's
            // binding set. SET itself contributes NO new bindings —
            // items are references, not declarations.
            collect_bindings(&s.input, out);
        }
        LogicalPlan::Remove(r) => {
            // ADR-150 W26-θ Phase 4: passes through the input plan's
            // binding set. REMOVE itself contributes NO new bindings.
            collect_bindings(&r.input, out);
        }
        LogicalPlan::Merge(m) => {
            // ADR-151 W26-θ Phase 5: walks BOTH match and create
            // sub-plans. The MERGE pattern's bindings are declared by
            // both branches (Scan or CreateNode emit the binding for
            // the pattern's variables); the union covers all pattern
            // bindings.
            collect_bindings(&m.match_branch, out);
            collect_bindings(&m.create_branch, out);
        }
        LogicalPlan::Call(c) => {
            // ADR-192 (#623): a CALL{} node's OUTPUT bindings = the
            // driving input's bindings ++ the body's returned columns.
            // The body's INTERNAL bindings do NOT escape (the scoping
            // fence — they are not output-row columns), so we do NOT
            // recurse into `body`. This keeps `shared_bindings` (used for
            // join-key derivation when a clause follows the CALL) correct:
            // a post-CALL clause can only join on the driving + returned
            // columns.
            collect_bindings(&c.input, out);
            for b in &c.returned {
                out.insert(*b);
            }
        }
        LogicalPlan::CorrelationSeed(s) => {
            // The seed's output schema IS the imported binding set; this
            // is what `lower_match`'s `shared_bindings(seed, pattern)`
            // intersects to derive the correlated equi-join key.
            for b in &s.imported {
                out.insert(*b);
            }
        }
        LogicalPlan::Empty(_) => {}
    }
}

// =====================================================================
// M4-33 helpers
// =====================================================================

/// Return `true` if any item in `items` is an aggregation function
/// call (top-level or nested) at the projection-expression level.
///
/// Used by [`LogicalPlanLoweringVisitor::lower_with`] +
/// [`LogicalPlanLoweringVisitor::lower_return`] to gate emission of
/// [`LogicalAggregate`].
fn items_contain_aggregation(items: &[BoundProjectionItem]) -> bool {
    items.iter().any(|it| match &it.kind {
        BoundProjectionKind::Expr(e) => expr_contains_aggregation(e),
        BoundProjectionKind::Wildcard { .. } => false,
    })
}

/// Rewrite the exact unfiltered-count shape to a counts-store source.
///
/// This is intentionally narrow. It accepts only a single projected
/// `count` aggregate with no grouping or DISTINCT over either:
/// - a node scan (`MATCH (n) RETURN count(n|*)`), optionally carrying a
///   single resolved label (`MATCH (n:Label) …`), or
/// - the bare relationship pattern lowering
///   `Join(Scan(label=None), Expand(rel_var?))`, the `Expand` optionally
///   carrying a single resolved rel-type (`MATCH ()-[:TYPE]->() …`).
///
/// F1 (#1356 §F1) lowers the labelled / typed forms to the EXISTING
/// per-label / per-type `CatalogStats` counters (an O(1) read) instead of
/// a full scan. A node-side label on BOTH ends of a relationship
/// (`(a:User)-[:KNOWS]->(b:User)` — the F1b `(src,type,dst)` triple) is
/// out of scope: the `scan.label.is_none()` guard on the rel path keeps a
/// labelled anchor on the scan path.
///
/// Any `Filter`, property filter, extra pattern, extra projection,
/// grouping key, DISTINCT, ORDER BY, SKIP, or LIMIT leaves the plan
/// unchanged.
#[must_use]
pub fn rewrite_unfiltered_count_to_count_store(plan: LogicalPlan) -> LogicalPlan {
    let LogicalPlan::Project(mut project) = plan else {
        return plan;
    };
    if project.items.len() != 1 {
        return LogicalPlan::Project(project);
    }

    let LogicalPlan::Aggregate(agg) = project.input.as_ref() else {
        return LogicalPlan::Project(project);
    };
    let Some((source, output_id)) = count_store_candidate(agg) else {
        return LogicalPlan::Project(project);
    };

    project.input = Box::new(LogicalPlan::CountStore(LogicalCountStore {
        source,
        output_id,
        span: agg.span.clone(),
    }));
    LogicalPlan::Project(project)
}

/// **#1366 (Phase 2) — the planner-selection rewrite.** Rewrite a
/// `Filter(pred, Scan{label: Some(l), var})` subtree into a
/// [`LogicalPropertyIndexScan`] when `pred` carries an exact-equality on
/// a property that `(l, property)` has an **Online** secondary index for
/// (RC-6 planner-visible gate via
/// [`crate::semantic::CatalogProvider::online_property_index`]).
///
/// Runs AFTER lowering (which has no catalog handle) at the same point
/// as [`rewrite_unfiltered_count_to_count_store`] — see
/// `explain::plan_for_execute_with_bound_options`. The walk is
/// clause-order-agnostic: it recurses read wrappers (`Project`, `Filter`,
/// `Limit`, joins, …) so a `MATCH (n:User {email:"x"}) RETURN n` (whose
/// inline property lowers to `Project(Filter(Scan))`) and a
/// `MATCH (n:User) WHERE n.email = "x" RETURN n` (identical lowered
/// shape) both route to the index.
///
/// # What is NOT rewritten (the full-scan fallback)
///
/// - **Unlabelled** `MATCH (n {email:"x"})` — the index is label-scoped;
///   the label-agnostic union is out of RC scope. The `Scan{label:None}`
///   keeps its full-scan path.
/// - **No Online index** on `(label, property)` — absent, `Building`, or
///   dropped. `online_property_index` returns `false` → keep the scan.
/// - **Unsupported value type** (RC-5) — only `String` / `Integer` /
///   `Boolean` literals + parameters are index-eligible; a `Float`, a
///   list, a map, a computed expression, or an inequality keeps the scan.
/// - **A write plan** — writes are never rewritten (a write sub-plan's
///   driving scan stays a scan). The walk skips write operators.
///
/// The rewrite is a pure plan-shape transform: correctness lives in the
/// executor op's candidate-then-verify (the index is a candidate source,
/// never a visibility authority).
#[must_use]
pub fn rewrite_scan_to_property_index_scan<C: crate::semantic::CatalogProvider + ?Sized>(
    plan: LogicalPlan,
    catalog: &C,
) -> LogicalPlan {
    match plan {
        // The target shape: a Filter directly over a labelled Scan.
        LogicalPlan::Filter(filter) => {
            // First recurse into the input so a nested Filter(Scan) deeper
            // in the tree is also considered (defensive; the lowered
            // MATCH shape is Filter-directly-over-Scan).
            let input = rewrite_scan_to_property_index_scan(*filter.input, catalog);
            if let LogicalPlan::Scan(scan) = &input {
                if let Some(label) = scan.label {
                    if let Some(rewritten) = try_route_filter_scan_to_index(
                        &filter.predicate,
                        scan,
                        label,
                        catalog,
                        &filter.span,
                    ) {
                        return rewritten;
                    }
                }
            }
            // No route — keep the Filter over the (possibly-rewritten)
            // input.
            LogicalPlan::Filter(LogicalFilter {
                input: Box::new(input),
                predicate: filter.predicate,
                span: filter.span,
            })
        }
        // Read wrappers — recurse into the driving input (the write ops
        // and leaves fall through unchanged below).
        LogicalPlan::Project(mut p) => {
            p.input = Box::new(rewrite_scan_to_property_index_scan(*p.input, catalog));
            LogicalPlan::Project(p)
        }
        LogicalPlan::Limit(mut l) => {
            l.input = Box::new(rewrite_scan_to_property_index_scan(*l.input, catalog));
            LogicalPlan::Limit(l)
        }
        LogicalPlan::Skip(mut s) => {
            s.input = Box::new(rewrite_scan_to_property_index_scan(*s.input, catalog));
            LogicalPlan::Skip(s)
        }
        LogicalPlan::DynamicLimit(mut l) => {
            l.input = Box::new(rewrite_scan_to_property_index_scan(*l.input, catalog));
            LogicalPlan::DynamicLimit(l)
        }
        LogicalPlan::Sort(mut s) => {
            s.input = Box::new(rewrite_scan_to_property_index_scan(*s.input, catalog));
            LogicalPlan::Sort(s)
        }
        LogicalPlan::Distinct(mut d) => {
            d.input = Box::new(rewrite_scan_to_property_index_scan(*d.input, catalog));
            LogicalPlan::Distinct(d)
        }
        LogicalPlan::Aggregate(mut a) => {
            a.input = Box::new(rewrite_scan_to_property_index_scan(*a.input, catalog));
            LogicalPlan::Aggregate(a)
        }
        LogicalPlan::Unwind(mut u) => {
            u.input = Box::new(rewrite_scan_to_property_index_scan(*u.input, catalog));
            LogicalPlan::Unwind(u)
        }
        LogicalPlan::Join(mut j) => {
            j.left = Box::new(rewrite_scan_to_property_index_scan(*j.left, catalog));
            j.right = Box::new(rewrite_scan_to_property_index_scan(*j.right, catalog));
            LogicalPlan::Join(j)
        }
        LogicalPlan::LeftOuterJoin(mut j) => {
            j.left = Box::new(rewrite_scan_to_property_index_scan(*j.left, catalog));
            j.right = Box::new(rewrite_scan_to_property_index_scan(*j.right, catalog));
            LogicalPlan::LeftOuterJoin(j)
        }
        // Everything else — bare Scan, Expand, hybrid leaves, write ops,
        // Call/Union/NamedPath/ProcedureCall etc. — pass through
        // unchanged. A BARE labelled `Scan` (no Filter) is not a point
        // lookup (no equality predicate), so it correctly stays a scan.
        // Write plans + their sub-scans are never rewritten.
        other => other,
    }
}

/// Try to route `Filter(pred, Scan{label})` to a
/// [`LogicalPropertyIndexScan`]. Returns `Some(rewritten)` when the
/// predicate carries an index-eligible exact-equality on an Online index;
/// `None` (keep the scan) otherwise.
fn try_route_filter_scan_to_index<C: crate::semantic::CatalogProvider + ?Sized>(
    predicate: &BoundExpression,
    scan: &LogicalScan,
    label: LabelId,
    catalog: &C,
    filter_span: &Span,
) -> Option<LogicalPlan> {
    // Flatten the predicate's top-level AND-spine into conjuncts.
    let mut conjuncts: Vec<&BoundExpression> = Vec::new();
    flatten_and_conjuncts(predicate, &mut conjuncts);

    // Find the FIRST index-eligible equality conjunct: `var.prop = <lit
    // | param>` (either operand order) where `(label, prop)` has an
    // Online index and the value type is RC-supported.
    let mut chosen: Option<(usize, String, BoundExpression)> = None;
    for (i, conj) in conjuncts.iter().enumerate() {
        if let Some((prop, value)) = index_equality_on(conj, scan.var) {
            if is_rc_index_value(&value) && catalog.online_property_index(label, &prop) {
                chosen = Some((i, prop, value));
                break;
            }
        }
    }
    let (idx, property, value) = chosen?;

    // Remaining conjuncts (all except the chosen one) fold back into a
    // residual filter over the verified rows. `None` when the chosen
    // equality was the whole predicate.
    let residual_preds: Vec<&BoundExpression> = conjuncts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, e)| *e)
        .collect();
    let residual = fold_and_spine(&residual_preds);

    Some(LogicalPlan::PropertyIndexScan(LogicalPropertyIndexScan {
        label,
        property,
        value,
        var: scan.var,
        read_lsn: scan.read_lsn,
        residual,
        span: filter_span.clone(),
    }))
}

/// Flatten a top-level `AND` spine into its leaf conjuncts. A non-`AND`
/// expression is a single conjunct. Only the TOP-level `AND` structure is
/// flattened (a nested `OR` / other op is one opaque conjunct).
fn flatten_and_conjuncts<'a>(expr: &'a BoundExpression, out: &mut Vec<&'a BoundExpression>) {
    if let BoundExpression::BinaryOp {
        op: BinOp::And,
        lhs,
        rhs,
        ..
    } = expr
    {
        flatten_and_conjuncts(lhs, out);
        flatten_and_conjuncts(rhs, out);
    } else {
        out.push(expr);
    }
}

/// Recognize an exact-equality `var.<prop> = <literal|parameter>` (or the
/// mirror `<literal|parameter> = var.<prop>`) rooted at `var`. Returns the
/// single-segment property name + the value expression when matched.
///
/// Rejects: non-`Eq` ops, multi-segment property paths (`n.a.b`), a
/// property on a DIFFERENT binding, and a non-literal/non-parameter RHS
/// (a computed expression, a variable, a function call).
fn index_equality_on(expr: &BoundExpression, var: BindingId) -> Option<(String, BoundExpression)> {
    let BoundExpression::BinaryOp {
        op: BinOp::Eq,
        lhs,
        rhs,
        ..
    } = expr
    else {
        return None;
    };
    // Try both operand orders.
    if let Some(prop) = single_prop_on_var(lhs, var) {
        if is_literal_or_param(rhs) {
            return Some((prop, (**rhs).clone()));
        }
    }
    if let Some(prop) = single_prop_on_var(rhs, var) {
        if is_literal_or_param(lhs) {
            return Some((prop, (**lhs).clone()));
        }
    }
    None
}

/// `Some(prop)` iff `expr` is `var.<prop>` (single-segment property
/// access rooted at `VariableRef(var)`); `None` otherwise.
fn single_prop_on_var(expr: &BoundExpression, var: BindingId) -> Option<String> {
    let BoundExpression::PropertyAccess { base, path, .. } = expr else {
        return None;
    };
    if path.len() != 1 {
        return None;
    }
    let BoundExpression::VariableRef { binding_id, .. } = base.as_ref() else {
        return None;
    };
    if *binding_id != var {
        return None;
    }
    Some(path[0].name.clone())
}

/// Whether `expr` is a plain literal or a `$param` reference (the only
/// index-eligible lookup value shapes — a computed expression, a
/// variable, or a function call is NOT a point-lookup key).
fn is_literal_or_param(expr: &BoundExpression) -> bool {
    matches!(
        expr,
        BoundExpression::Literal { .. } | BoundExpression::Parameter { .. }
    )
}

/// **RC-5.** Whether a lookup value is an RC-supported index type. A
/// literal must be `String` / `Integer` / `Boolean` (NOT `Float` — the
/// int/float coercion false-split, dropped from the RC index). A
/// `$param` is admitted at plan time UNCONDITIONALLY: its runtime value
/// is unknown until it binds, so the type check is deferred to lookup.
///
/// # Runtime `None`-key values scan, they do NOT return empty (#1415)
///
/// When a `$param` binds to a value with NO canonical index key — a
/// fractional / out-of-i64-range `Float`, a NEGATIVE `Integer`, a `List`
/// / `Map` — `PropertyIndexManager::lookup_candidates` returns an EMPTY
/// candidate set (`canonical_key_for` is `None`). That empty set is NOT
/// the answer: a full scan's `Filter(prop = v)` still matches such a
/// value via `values_equal_3vl` (`10.5`, `[1,2]`, `-5` all compare). So
/// the `PropertyIndexScan` OP checks
/// [`crate::executor::ExecutorSubstrate::value_is_indexable`] BEFORE the
/// lookup and, when the resolved value is NOT keyable, falls back to a
/// Scan+Filter over the label (the same rows the un-routed path would
/// produce). Admitting `$param` here is therefore SAFE for keyable and
/// unkeyable runtime values alike — keyable ones use the index, unkeyable
/// ones scan; NEITHER silently drops a row. (The prior version of this
/// comment asserted an unsupported runtime type "degrades to an empty
/// result, never a wrong result" — that was FALSE and was the charter of
/// the #1415 silent-wrong-results bug; the op-level scan-fallback is what
/// now makes routing `$param` here correct.)
fn is_rc_index_value(value: &BoundExpression) -> bool {
    match value {
        BoundExpression::Parameter { .. } => true,
        BoundExpression::Literal { value, .. } => matches!(
            value,
            Literal::String(_) | Literal::Integer(_) | Literal::Bool(_)
        ),
        _ => false,
    }
}

fn count_store_candidate(agg: &LogicalAggregate) -> Option<(CountStoreSource, BindingId)> {
    if !agg.group_by.is_empty() || agg.aggregations.len() != 1 {
        return None;
    }
    let spec = &agg.aggregations[0];
    if spec.function != AggregationKind::Count || spec.distinct {
        return None;
    }

    match agg.input.as_ref() {
        LogicalPlan::Scan(scan) => {
            // F1 (#1356 §F1): a single resolved label lowers to the
            // per-label counter; an unlabelled scan keeps the tenant-wide
            // counter. A property / WHERE predicate wraps the `Scan` in a
            // `Filter` (falling to `_ => None` below), so only a BARE
            // labelled scan reaches here — never a filtered one.
            if spec.star || count_arg_binding(&spec.arg) == Some(scan.var) {
                let source = match scan.label {
                    Some(label) => CountStoreSource::NodesWithLabel(label),
                    None => CountStoreSource::Nodes,
                };
                Some((source, spec.output_id))
            } else {
                None
            }
        }
        LogicalPlan::Join(join) => {
            bare_relationship_count_source(join, spec).map(|source| (source, spec.output_id))
        }
        _ => None,
    }
}

fn bare_relationship_count_source(
    join: &LogicalJoin,
    spec: &AggregationSpec,
) -> Option<CountStoreSource> {
    let (scan, expand) = match (join.left.as_ref(), join.right.as_ref()) {
        (LogicalPlan::Scan(scan), LogicalPlan::Expand(expand)) => (scan, expand),
        _ => return None,
    };
    if join.on != JoinCondition::SharedBindings(vec![expand.from])
        || scan.label.is_some()
        || expand.length_range.is_some()
        || !matches!(
            expand.direction,
            Direction::LeftToRight | Direction::RightToLeft
        )
        || expand.from != scan.var
    {
        return None;
    }
    // F1 (#1356 §F1): a single resolved rel-type lowers to the per-type
    // counter; a type-free expand keeps the tenant-wide rel counter. The
    // `scan.label.is_none()` guard above still holds, so a labelled anchor
    // (the F1b `(src,type,dst)` triple) stays on the scan path — F1 only
    // reuses the single-key `rel_type_counts` the commit pipeline already
    // maintains.
    let source = match expand.rel_type {
        Some(rel_type) => CountStoreSource::RelsWithType(rel_type),
        None => CountStoreSource::Relationships,
    };
    if spec.star
        || expand
            .rel_var
            .is_some_and(|rel| count_arg_binding(&spec.arg) == Some(rel))
    {
        Some(source)
    } else {
        None
    }
}

fn count_arg_binding(arg: &BoundExpression) -> Option<BindingId> {
    match arg {
        BoundExpression::VariableRef { binding_id, .. } => Some(*binding_id),
        _ => None,
    }
}

/// Recursive scan for any aggregation function call in `e`.
fn expr_contains_aggregation(e: &BoundExpression) -> bool {
    match e {
        BoundExpression::FunctionCall { name, args, .. } => {
            if AggregationKind::from_function_name(name).is_some() {
                return true;
            }
            args.iter().any(expr_contains_aggregation)
        }
        BoundExpression::PropertyAccess { base, .. } => expr_contains_aggregation(base),
        BoundExpression::ListLiteral { elements, .. } => {
            elements.iter().any(expr_contains_aggregation)
        }
        BoundExpression::MapLiteral { entries, .. } => entries
            .iter()
            .any(|(_, value)| expr_contains_aggregation(value)),
        // #1290 — left-nested operator SPINE walked iteratively (see
        // `max_in_expr` for the pattern rationale).
        BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        | BoundExpression::In { .. }
        | BoundExpression::IsNull { .. } => {
            let mut rhs_stack: Vec<&BoundExpression> = Vec::new();
            let mut cur = e;
            let base_hit = loop {
                match cur {
                    BoundExpression::BinaryOp { lhs, rhs, .. }
                    | BoundExpression::In { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    BoundExpression::UnaryOp { operand, .. } => cur = operand,
                    BoundExpression::IsNull { lhs, .. } => cur = lhs,
                    other => break expr_contains_aggregation(other),
                }
            };
            base_hit
                || rhs_stack
                    .iter()
                    .rev()
                    .any(|rhs| expr_contains_aggregation(rhs))
        }
        BoundExpression::Near { lhs, target, .. } => {
            expr_contains_aggregation(lhs) || expr_contains_aggregation(target)
        }
        BoundExpression::TextMatch { lhs, query, .. } => {
            expr_contains_aggregation(lhs) || expr_contains_aggregation(query)
        }
        BoundExpression::InCommunity {
            node, community, ..
        } => expr_contains_aggregation(node) || expr_contains_aggregation(community),
        // ADR-188 — an aggregation may appear in a list-predicate /
        // reduce sub-expression (e.g. the list could be
        // `collect(n.v)`); recurse into every child.
        BoundExpression::ListPredicate {
            list, predicate, ..
        } => expr_contains_aggregation(list) || expr_contains_aggregation(predicate),
        BoundExpression::Reduce {
            init, list, expr, ..
        } => {
            expr_contains_aggregation(init)
                || expr_contains_aggregation(list)
                || expr_contains_aggregation(expr)
        }
        // ADR-188 (#620 list-half) — an aggregation may appear in a
        // list-comprehension sub-expression (e.g. the list could be
        // `collect(n.v)`); recurse into every child (list + optional
        // predicate + optional projection).
        BoundExpression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            expr_contains_aggregation(list)
                || predicate.as_deref().is_some_and(expr_contains_aggregation)
                || projection.as_deref().is_some_and(expr_contains_aggregation)
        }
        // ADR-191 D-6 (#620 map-half) — an aggregation may appear in a
        // map-projection base or a literal-entry value (e.g.
        // `n{cnt: count(*)}`); recurse into the base + every literal value.
        BoundExpression::MapProjection { base, items, .. } => {
            expr_contains_aggregation(base)
                || items.iter().any(|item| match item {
                    BoundMapProjectionItem::Literal { value, .. } => {
                        expr_contains_aggregation(value)
                    }
                    BoundMapProjectionItem::Property(_) | BoundMapProjectionItem::AllProperties => {
                        false
                    }
                })
        }
        // #621 — an aggregation may appear in a subscript / slice operand
        // (e.g. `collect(n.v)[0]`); recurse into each.
        BoundExpression::Subscript { base, index, .. } => {
            expr_contains_aggregation(base) || expr_contains_aggregation(index)
        }
        BoundExpression::Slice {
            base, start, end, ..
        } => {
            expr_contains_aggregation(base)
                || start.as_deref().is_some_and(expr_contains_aggregation)
                || end.as_deref().is_some_and(expr_contains_aggregation)
        }
        // #621 — an aggregation may appear in any CASE sub-expression (e.g.
        // `CASE WHEN n.x > 0 THEN count(*) ELSE 0 END`); recurse into the
        // test + every WHEN / THEN + the ELSE.
        BoundExpression::Case {
            test,
            branches,
            default,
            ..
        } => {
            test.as_deref().is_some_and(expr_contains_aggregation)
                || branches.iter().any(|(when, then)| {
                    expr_contains_aggregation(when) || expr_contains_aggregation(then)
                })
                || default.as_deref().is_some_and(expr_contains_aggregation)
        }
        BoundExpression::Literal { .. }
        | BoundExpression::Parameter { .. }
        | BoundExpression::VariableRef { .. }
        | BoundExpression::UnresolvedVariable { .. } => false,
    }
}

/// #1053 R1 — deep-substitute a `VariableRef` whose `binding_id` is a
/// grouping-key OUTPUT id with that key's PRE-aggregation INPUT expression
/// (`output_to_input`). Applied to an aggregate ARGUMENT so the lifted sort-key
/// aggregate's arg resolves against the input rows `AggregateOp` folds over,
/// matching how the projection's own aggregate spec is already bound. The walk
/// is total over `BoundExpression`; only a `VariableRef` leaf can match a
/// grouping-key output id (a `PropertyAccess` grouping key is projected under
/// its base node's id, never a bare output id, and an aggregate over it would
/// reference the base node binding which is unchanged across the group). No
/// scoped-variable capture concern: a grouping-key output id is a top-level row
/// binding, distinct from any expression-internal scoped `var_bid`.
fn substitute_output_bindings(
    expr: &BoundExpression,
    output_to_input: &std::collections::BTreeMap<BindingId, BoundExpression>,
) -> BoundExpression {
    use BoundExpression as BE;
    match expr {
        BE::VariableRef { binding_id, .. } => match output_to_input.get(binding_id) {
            Some(input_expr) => input_expr.clone(),
            None => expr.clone(),
        },
        BE::PropertyAccess {
            base,
            path,
            span,
            type_info,
        } => BE::PropertyAccess {
            base: Box::new(substitute_output_bindings(base, output_to_input)),
            path: path.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        // #1290 — left-nested operator SPINE rebuilt iteratively (the
        // spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may interleave
        // BinaryOp / UnaryOp / In / IsNull levels; recursing per level
        // overflowed the native stack): walk down the left/operand edge
        // collecting one borrowed frame per level, substitute the
        // non-spine base via the ordinary arms, then fold the frames
        // back up. Rebuild order (base subtree first, then each level's
        // rhs innermost→outermost) matches the recursion this replaces.
        // `rhs` operands substitute recursively — they are never part
        // of the LEFT spine, so their depth is bracket-bounded.
        BE::BinaryOp { .. } | BE::UnaryOp { .. } | BE::In { .. } | BE::IsNull { .. } => {
            enum SpineFrame<'a> {
                Binary {
                    op: &'a BinOp,
                    rhs: &'a BoundExpression,
                    span: &'a Span,
                    type_info: &'a Option<TypeInfo>,
                },
                Unary {
                    op: &'a UnaryOp,
                    span: &'a Span,
                    type_info: &'a Option<TypeInfo>,
                },
                In {
                    rhs: &'a BoundExpression,
                    span: &'a Span,
                    type_info: &'a Option<TypeInfo>,
                },
                IsNull {
                    negated: bool,
                    span: &'a Span,
                    type_info: &'a Option<TypeInfo>,
                },
            }
            let mut frames: Vec<SpineFrame<'_>> = Vec::new();
            let mut cur = expr;
            let mut acc = loop {
                match cur {
                    BE::BinaryOp {
                        op,
                        lhs,
                        rhs,
                        span,
                        type_info,
                    } => {
                        frames.push(SpineFrame::Binary {
                            op,
                            rhs,
                            span,
                            type_info,
                        });
                        cur = lhs;
                    }
                    BE::UnaryOp {
                        op,
                        operand,
                        span,
                        type_info,
                    } => {
                        frames.push(SpineFrame::Unary {
                            op,
                            span,
                            type_info,
                        });
                        cur = operand;
                    }
                    BE::In {
                        lhs,
                        rhs,
                        span,
                        type_info,
                    } => {
                        frames.push(SpineFrame::In {
                            rhs,
                            span,
                            type_info,
                        });
                        cur = lhs;
                    }
                    BE::IsNull {
                        lhs,
                        negated,
                        span,
                        type_info,
                    } => {
                        frames.push(SpineFrame::IsNull {
                            negated: *negated,
                            span,
                            type_info,
                        });
                        cur = lhs;
                    }
                    other => break substitute_output_bindings(other, output_to_input),
                }
            };
            while let Some(frame) = frames.pop() {
                acc = match frame {
                    SpineFrame::Binary {
                        op,
                        rhs,
                        span,
                        type_info,
                    } => BE::BinaryOp {
                        op: op.clone(),
                        lhs: Box::new(acc),
                        rhs: Box::new(substitute_output_bindings(rhs, output_to_input)),
                        span: span.clone(),
                        type_info: type_info.clone(),
                    },
                    SpineFrame::Unary {
                        op,
                        span,
                        type_info,
                    } => BE::UnaryOp {
                        op: op.clone(),
                        operand: Box::new(acc),
                        span: span.clone(),
                        type_info: type_info.clone(),
                    },
                    SpineFrame::In {
                        rhs,
                        span,
                        type_info,
                    } => BE::In {
                        lhs: Box::new(acc),
                        rhs: Box::new(substitute_output_bindings(rhs, output_to_input)),
                        span: span.clone(),
                        type_info: type_info.clone(),
                    },
                    SpineFrame::IsNull {
                        negated,
                        span,
                        type_info,
                    } => BE::IsNull {
                        lhs: Box::new(acc),
                        negated,
                        span: span.clone(),
                        type_info: type_info.clone(),
                    },
                };
            }
            acc
        }
        BE::FunctionCall {
            name,
            args,
            distinct,
            star,
            span,
            type_info,
        } => BE::FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| substitute_output_bindings(a, output_to_input))
                .collect(),
            distinct: *distinct,
            star: *star,
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::ListLiteral {
            elements,
            span,
            type_info,
        } => BE::ListLiteral {
            elements: elements
                .iter()
                .map(|e| substitute_output_bindings(e, output_to_input))
                .collect(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::MapLiteral {
            entries,
            span,
            type_info,
        } => BE::MapLiteral {
            entries: entries
                .iter()
                .map(|(k, v)| (k.clone(), substitute_output_bindings(v, output_to_input)))
                .collect(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Subscript {
            base,
            index,
            span,
            type_info,
        } => BE::Subscript {
            base: Box::new(substitute_output_bindings(base, output_to_input)),
            index: Box::new(substitute_output_bindings(index, output_to_input)),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Slice {
            base,
            start,
            end,
            span,
            type_info,
        } => BE::Slice {
            base: Box::new(substitute_output_bindings(base, output_to_input)),
            start: start
                .as_deref()
                .map(|s| Box::new(substitute_output_bindings(s, output_to_input))),
            end: end
                .as_deref()
                .map(|e| Box::new(substitute_output_bindings(e, output_to_input))),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Case {
            test,
            branches,
            default,
            span,
            type_info,
        } => BE::Case {
            test: test
                .as_deref()
                .map(|t| Box::new(substitute_output_bindings(t, output_to_input))),
            branches: branches
                .iter()
                .map(|(when, then)| {
                    (
                        substitute_output_bindings(when, output_to_input),
                        substitute_output_bindings(then, output_to_input),
                    )
                })
                .collect(),
            default: default
                .as_deref()
                .map(|d| Box::new(substitute_output_bindings(d, output_to_input))),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        // Leaves with nothing to substitute, and scoped-variable / predicate
        // bodies whose internal `var_bid` is distinct from any top-level
        // grouping-key output id — a grouping-key output is never captured by an
        // inner scope, and an aggregate argument does not legally contain these
        // forms over a grouping key, so clone verbatim.
        BE::Literal { .. }
        | BE::Parameter { .. }
        | BE::UnresolvedVariable { .. }
        | BE::Near { .. }
        | BE::TextMatch { .. }
        | BE::InCommunity { .. }
        | BE::ListPredicate { .. }
        | BE::Reduce { .. }
        | BE::ListComprehension { .. }
        | BE::MapProjection { .. } => expr.clone(),
    }
}

/// #910 — if `expr` is a simple variable / property reference equal to one of
/// the `grouping_keys`, return the key's `Aggregate` output id. Used by
/// [`LogicalPlanLoweringVisitor::rewrite_nested_aggregations`] to redirect a
/// grouping-key reference inside an aggregating projection
/// (`me.age + count(...)`) to the key's precomputed column. Per ADR-038
/// amendment-12 (#796) every non-aggregate reference in a VALID aggregating
/// projection IS a grouping key, so a simple leaf that matches no key here is
/// left untouched (it is already a hidden-aggregate ref, a literal, or a
/// parameter).
fn grouping_key_output(
    expr: &BoundExpression,
    grouping_keys: &[(BoundExpression, BindingId)],
) -> Option<BindingId> {
    grouping_keys
        .iter()
        .find(|(key, _)| same_binding_ref(expr, key))
        .map(|(_, id)| *id)
}

/// Span/type-free structural equality for the two grouping-key leaf shapes a
/// valid aggregating projection can reference (#796): a variable reference
/// (compared by `binding_id`) or a property access (compared by base binding +
/// property-name path). Any other shape never matches — a complex grouping key
/// (`a + b`) does NOT make its leaves referenceable (ADR-038 amendment-12), so
/// it is never matched here (and the binder already rejected referencing it as
/// a whole inside an aggregating expression). Strict equality ⇒ no false
/// positive can redirect a non-grouping-key leaf.
fn same_binding_ref(a: &BoundExpression, b: &BoundExpression) -> bool {
    use BoundExpression as BE;
    match (a, b) {
        (BE::VariableRef { binding_id: x, .. }, BE::VariableRef { binding_id: y, .. }) => x == y,
        (
            BE::PropertyAccess {
                base: ba, path: pa, ..
            },
            BE::PropertyAccess {
                base: bb, path: pb, ..
            },
        ) => {
            same_binding_ref(ba, bb)
                && pa.len() == pb.len()
                && pa.iter().zip(pb).all(|(x, y)| x.name == y.name)
        }
        _ => false,
    }
}

/// Translate the AST [`OrderDirection`] into a logical-plan
/// [`SortDirection`]. The AST `Default` variant collapses to `Asc`
/// per Cypher 9 §6.6 (ascending is the spec default).
fn sort_direction_from_ast(d: &OrderDirection) -> SortDirection {
    match d {
        OrderDirection::Asc | OrderDirection::Default => SortDirection::Asc,
        OrderDirection::Desc => SortDirection::Desc,
    }
}

/// Recursively collect all [`BindingId`]s referenced by a
/// `BoundExpression`. Used by [`LogicalPlanLoweringVisitor::lower_distinct`]
/// to derive the on-set for [`LogicalDistinct`].
fn collect_referenced_bindings(
    e: &BoundExpression,
    out: &mut std::collections::BTreeSet<BindingId>,
) {
    match e {
        BoundExpression::VariableRef { binding_id, .. } => {
            out.insert(*binding_id);
        }
        BoundExpression::PropertyAccess { base, .. } => collect_referenced_bindings(base, out),
        BoundExpression::ListLiteral { elements, .. } => {
            for element in elements {
                collect_referenced_bindings(element, out);
            }
        }
        BoundExpression::MapLiteral { entries, .. } => {
            for (_, value) in entries {
                collect_referenced_bindings(value, out);
            }
        }
        // #1290 — left-nested operator SPINE walked iteratively (see
        // `max_in_expr` for the pattern rationale). The spine variants
        // carry no scoped `var_bid`, so no post-visit removal applies
        // at spine levels — the ADR-188 scoped-variable arms below keep
        // their recurse-then-remove discipline unchanged.
        BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        | BoundExpression::In { .. }
        | BoundExpression::IsNull { .. } => {
            let mut rhs_stack: Vec<&BoundExpression> = Vec::new();
            let mut cur = e;
            loop {
                match cur {
                    BoundExpression::BinaryOp { lhs, rhs, .. }
                    | BoundExpression::In { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    BoundExpression::UnaryOp { operand, .. } => cur = operand,
                    BoundExpression::IsNull { lhs, .. } => cur = lhs,
                    other => {
                        collect_referenced_bindings(other, out);
                        break;
                    }
                }
            }
            while let Some(rhs) = rhs_stack.pop() {
                collect_referenced_bindings(rhs, out);
            }
        }
        BoundExpression::FunctionCall { args, .. } => {
            for a in args {
                collect_referenced_bindings(a, out);
            }
        }
        BoundExpression::Near { lhs, target, .. } => {
            collect_referenced_bindings(lhs, out);
            collect_referenced_bindings(target, out);
        }
        BoundExpression::TextMatch { lhs, query, .. } => {
            collect_referenced_bindings(lhs, out);
            collect_referenced_bindings(query, out);
        }
        BoundExpression::InCommunity {
            node, community, ..
        } => {
            collect_referenced_bindings(node, out);
            collect_referenced_bindings(community, out);
        }
        // ADR-188 — a list-predicate / reduce references the OUTER
        // bindings in its operands + any outer bindings inside its
        // predicate/body, but NOT its EXPRESSION-INTERNAL scoped vars
        // (`var_bid`, and `acc_bid` for reduce) — those are not output-
        // row columns, so a DISTINCT on-set must not key on them. We
        // recurse into every child then REMOVE the scoped ids, which
        // correctly keeps outer refs (e.g. `n` in
        // `all(x IN n.friends WHERE x > n.threshold)`) while dropping
        // the scoped `x`.
        BoundExpression::ListPredicate {
            var_bid,
            list,
            predicate,
            ..
        } => {
            collect_referenced_bindings(list, out);
            collect_referenced_bindings(predicate, out);
            out.remove(var_bid);
        }
        BoundExpression::Reduce {
            acc_bid,
            init,
            var_bid,
            list,
            expr,
            ..
        } => {
            collect_referenced_bindings(init, out);
            collect_referenced_bindings(list, out);
            collect_referenced_bindings(expr, out);
            out.remove(acc_bid);
            out.remove(var_bid);
        }
        // ADR-188 (#620 list-half) — a list-comprehension references the
        // OUTER bindings in its operands + any outer bindings inside its
        // predicate/projection, but NOT its EXPRESSION-INTERNAL scoped
        // var (`var_bid`) — that is not an output-row column, so a
        // DISTINCT on-set must not key on it. We recurse into every
        // child then REMOVE the scoped id, which correctly keeps outer
        // refs (e.g. `n` in `[x IN n.friends WHERE x > n.t | x + n.b]`)
        // while dropping the scoped `x`.
        BoundExpression::ListComprehension {
            var_bid,
            list,
            predicate,
            projection,
            ..
        } => {
            collect_referenced_bindings(list, out);
            if let Some(p) = predicate {
                collect_referenced_bindings(p, out);
            }
            if let Some(e) = projection {
                collect_referenced_bindings(e, out);
            }
            out.remove(var_bid);
        }
        // ADR-191 D-6 (#620 map-half) — a map projection references its
        // BASE (a row VariableRef — a genuine output-row dependency that
        // MUST be kept, e.g. a DISTINCT on `n{.name}` keys on `n`) plus any
        // outer bindings inside its literal-entry values. There is NO
        // expression-internal scoped var to drop (unlike the comprehensions).
        BoundExpression::MapProjection { base, items, .. } => {
            collect_referenced_bindings(base, out);
            for item in items {
                if let BoundMapProjectionItem::Literal { value, .. } = item {
                    collect_referenced_bindings(value, out);
                }
            }
        }
        // #621 — postfix accessors reference the OUTER bindings in their
        // operands; no expression-internal scoped var to drop. Recurse
        // into base + index / bounds.
        BoundExpression::Subscript { base, index, .. } => {
            collect_referenced_bindings(base, out);
            collect_referenced_bindings(index, out);
        }
        BoundExpression::Slice {
            base, start, end, ..
        } => {
            collect_referenced_bindings(base, out);
            if let Some(s) = start {
                collect_referenced_bindings(s, out);
            }
            if let Some(e) = end {
                collect_referenced_bindings(e, out);
            }
        }
        // #621 — a CASE references the OUTER bindings in its test + every
        // WHEN / THEN + ELSE; there is NO expression-internal scoped var to
        // drop (unlike the comprehensions). Recurse into every child so a
        // DISTINCT on-set correctly keys on the referenced row columns
        // (e.g. `n` in `CASE WHEN n.flag THEN n.a ELSE n.b END`).
        BoundExpression::Case {
            test,
            branches,
            default,
            ..
        } => {
            if let Some(t) = test {
                collect_referenced_bindings(t, out);
            }
            for (when, then) in branches {
                collect_referenced_bindings(when, out);
                collect_referenced_bindings(then, out);
            }
            if let Some(d) = default {
                collect_referenced_bindings(d, out);
            }
        }
        BoundExpression::Literal { .. }
        | BoundExpression::Parameter { .. }
        | BoundExpression::UnresolvedVariable { .. } => {}
    }
}

fn with_projection_output_ids(
    items: &[BoundProjectionItem],
) -> std::collections::BTreeSet<BindingId> {
    items.iter().filter_map(|item| item.output_id).collect()
}

fn split_with_where_filters(
    pred: &BoundExpression,
    output_ids: &std::collections::BTreeSet<BindingId>,
    has_wildcard: bool,
) -> (Vec<BoundExpression>, Vec<BoundExpression>) {
    let mut pre_projection = Vec::new();
    let mut post_projection = Vec::new();
    split_with_where_filters_inner(
        pred,
        output_ids,
        has_wildcard,
        &mut pre_projection,
        &mut post_projection,
    );
    (pre_projection, post_projection)
}

fn split_with_where_filters_inner(
    pred: &BoundExpression,
    output_ids: &std::collections::BTreeSet<BindingId>,
    has_wildcard: bool,
    pre_projection: &mut Vec<BoundExpression>,
    post_projection: &mut Vec<BoundExpression>,
) {
    // #1290 — the AND spine is walked with an explicit worklist (a
    // flat N-conjunct WHERE folds into an N-deep left-nested `And`
    // spine; recursing per conjunct overflowed the native stack).
    // Popping `rhs` LAST-pushed-first after pushing `[rhs, lhs]`
    // preserves the recursion's in-order conjunct sequence.
    let mut work: Vec<&BoundExpression> = vec![pred];
    while let Some(p) = work.pop() {
        if let BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs,
            rhs,
            ..
        } = p
        {
            work.push(rhs);
            work.push(lhs);
            continue;
        }

        let mut refs = std::collections::BTreeSet::new();
        collect_referenced_bindings(p, &mut refs);
        if has_wildcard || refs.iter().all(|id| output_ids.contains(id)) {
            post_projection.push(p.clone());
        } else {
            pre_projection.push(p.clone());
        }
    }
}

fn split_match_where_filters(
    pred: &BoundExpression,
    pattern_ids: &std::collections::BTreeSet<BindingId>,
) -> (Vec<BoundExpression>, Vec<BoundExpression>) {
    let mut pushdown = Vec::new();
    let mut post_join = Vec::new();
    split_match_where_filters_inner(pred, pattern_ids, &mut pushdown, &mut post_join);
    (pushdown, post_join)
}

fn split_match_where_filters_inner(
    pred: &BoundExpression,
    pattern_ids: &std::collections::BTreeSet<BindingId>,
    pushdown: &mut Vec<BoundExpression>,
    post_join: &mut Vec<BoundExpression>,
) {
    // #1290 — explicit worklist over the AND spine (see
    // `split_with_where_filters_inner`).
    let mut work: Vec<&BoundExpression> = vec![pred];
    while let Some(p) = work.pop() {
        if let BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs,
            rhs,
            ..
        } = p
        {
            work.push(rhs);
            work.push(lhs);
            continue;
        }

        let mut refs = std::collections::BTreeSet::new();
        collect_referenced_bindings(p, &mut refs);
        if refs.iter().all(|id| pattern_ids.contains(id)) {
            pushdown.push(p.clone());
        } else {
            post_join.push(p.clone());
        }
    }
}

// =====================================================================
// Tests
// =====================================================================
//
// 8 unit tests below, one per simple-operator lowering rule, per
// amendment-03 §M4-31 row pin. End-to-end pins (parse → bind →
// type-check → cross-substrate → lower) live in
// `tests/logical_plan_integration.rs`; the semantic-preservation
// proptest lives in `tests/logical_plan_proptest.rs`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use crate::semantic::{
        BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
    };

    fn cat() -> StubCatalogProvider {
        StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["age", "name", "x", "embedding", "content"])
    }

    /// Run the full pipeline: parse → bind → type-check →
    /// cross-substrate → lower. Panics on any prior-stage failure
    /// since those are out of M4-31 scope.
    fn lower_ok(input: &str) -> LogicalPlan {
        let stmt = parse(input).expect("parse");
        let cat = cat();
        let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
        CrossSubstrateValidator::validate(&bound, &cat).expect("validate");
        LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
    }

    /// #1290 — the hybrid-shape classifier stays in lockstep with
    /// `apply_where`'s recognizers: a conjunct is classified hybrid
    /// exactly when `apply_where` lowers it to a NON-Filter root. A
    /// divergence would either fold an un-executable hybrid predicate
    /// into a fused Filter (execution regression) or needlessly keep a
    /// generic conjunct out of the fold (depth-bound regression).
    #[test]
    fn hybrid_shape_classifier_matches_apply_where() {
        use crate::semantic::bound_ast::BoundPropertyRef;
        let span = || Span::point(1, 1);
        let var = |id: u64| BoundExpression::VariableRef {
            name: format!("v{id}"),
            binding_id: BindingId::new(id),
            span: span(),
            type_info: None,
        };
        let prop = |id: u64| BoundExpression::PropertyAccess {
            base: Box::new(var(id)),
            path: vec![BoundPropertyRef {
                name: "embedding".into(),
                property_id: None,
                span: span(),
            }],
            span: span(),
            type_info: None,
        };
        let param = || BoundExpression::Parameter {
            name: "q".into(),
            span: span(),
            type_info: None,
        };
        let lit_one = || BoundExpression::Literal {
            value: Literal::Integer(1),
            span: span(),
            type_info: None,
        };
        let fn_call = |name: &str, args: Vec<BoundExpression>| BoundExpression::FunctionCall {
            name: name.into(),
            args,
            distinct: false,
            star: false,
            span: span(),
            type_info: None,
        };
        let cases: Vec<BoundExpression> = vec![
            // Hybrid shapes — one per `apply_where` recognizer arm.
            BoundExpression::InCommunity {
                node: Box::new(var(1)),
                community: Box::new(param()),
                span: span(),
                type_info: None,
            },
            BoundExpression::Near {
                lhs: Box::new(prop(1)),
                target: Box::new(param()),
                vector_index: None,
                span: span(),
                type_info: None,
            },
            BoundExpression::TextMatch {
                lhs: Box::new(prop(1)),
                query: Box::new(param()),
                span: span(),
                type_info: None,
            },
            fn_call("vector_distance", vec![prop(1), param()]),
            fn_call("text_match", vec![prop(1), param()]),
            BoundExpression::BinaryOp {
                op: BinOp::Eq,
                lhs: Box::new(fn_call("community", vec![var(1)])),
                rhs: Box::new(param()),
                span: span(),
                type_info: None,
            },
            // Generic shapes — including the NEAR-misses of each
            // recognizer (non-var community node, non-property NEAR
            // lhs, wrong arity, plain comparison).
            BoundExpression::BinaryOp {
                op: BinOp::Eq,
                lhs: Box::new(prop(1)),
                rhs: Box::new(lit_one()),
                span: span(),
                type_info: None,
            },
            BoundExpression::InCommunity {
                node: Box::new(lit_one()),
                community: Box::new(param()),
                span: span(),
                type_info: None,
            },
            BoundExpression::Near {
                lhs: Box::new(var(1)),
                target: Box::new(param()),
                vector_index: None,
                span: span(),
                type_info: None,
            },
            fn_call("vector_distance", vec![prop(1)]),
            fn_call("size", vec![prop(1)]),
        ];
        for pred in &cases {
            let mut v = LogicalPlanLoweringVisitor {
                errors: Vec::new(),
                next_anon_id: 0,
                read_lsn: Lsn::MAX,
            };
            let lowered = v.apply_where(LogicalPlan::Empty(LogicalEmpty { span: span() }), pred);
            let is_generic_filter = matches!(lowered, LogicalPlan::Filter(_));
            assert_eq!(
                is_hybrid_predicate_shape(pred),
                !is_generic_filter,
                "classifier and apply_where disagree on {pred:?}"
            );
        }
    }

    /// #1290 — plan-depth bounding: at or under
    /// `MAX_FILTER_CHAIN_NODES` conjuncts the per-conjunct nested
    /// Filter shape is preserved byte-for-byte; above it the generic
    /// conjuncts fuse into ONE Filter carrying their AND spine.
    #[test]
    fn wide_where_folds_into_single_filter_above_chain_cap() {
        let wide = |n: usize| {
            let mut q = String::from("MATCH (n:Person) WHERE ");
            for i in 1..=n {
                if i > 1 {
                    q.push_str(" AND ");
                }
                q.push_str(&format!("n.age = {i}"));
            }
            q.push_str(" RETURN n");
            q
        };
        // At the cap: one Filter node per conjunct (status quo).
        let at_cap = lower_ok(&wide(MAX_FILTER_CHAIN_NODES));
        let filters_at_cap = shape(&at_cap).iter().filter(|s| **s == "Filter").count();
        assert_eq!(
            filters_at_cap, MAX_FILTER_CHAIN_NODES,
            "≤ cap conjuncts must keep the per-conjunct Filter chain"
        );
        // One over the cap: the generics fuse into a SINGLE Filter.
        let over_cap = lower_ok(&wide(MAX_FILTER_CHAIN_NODES + 1));
        let filters_over_cap = shape(&over_cap).iter().filter(|s| **s == "Filter").count();
        assert_eq!(
            filters_over_cap, 1,
            "> cap conjuncts must fuse into one AND-spine Filter"
        );
    }

    /// Walk the LogicalPlan tree and return a flat list of node-kind
    /// labels (in pre-order). Used by tests to assert the SHAPE of
    /// the produced plan without committing to exact layout.
    fn shape(plan: &LogicalPlan) -> Vec<&'static str> {
        let mut out = Vec::new();
        walk(plan, &mut out);
        out
    }

    fn walk(p: &LogicalPlan, out: &mut Vec<&'static str>) {
        match p {
            LogicalPlan::Scan(_) => out.push("Scan"),
            LogicalPlan::PropertyIndexScan(_) => out.push("PropertyIndexScan"),
            LogicalPlan::CountStore(_) => out.push("CountStore"),
            LogicalPlan::Expand(_) => out.push("Expand"),
            LogicalPlan::Filter(f) => {
                out.push("Filter");
                walk(&f.input, out);
            }
            LogicalPlan::Project(p) => {
                out.push("Project");
                walk(&p.input, out);
            }
            LogicalPlan::Join(j) => {
                out.push("Join");
                walk(&j.left, out);
                walk(&j.right, out);
            }
            LogicalPlan::LeftOuterJoin(j) => {
                out.push("LeftOuterJoin");
                walk(&j.left, out);
                walk(&j.right, out);
            }
            LogicalPlan::Limit(l) => {
                out.push("Limit");
                walk(&l.input, out);
            }
            LogicalPlan::Skip(s) => {
                out.push("Skip");
                walk(&s.input, out);
            }
            LogicalPlan::RankByHybrid(_) => out.push("RankByHybrid"),
            LogicalPlan::Fusion(f) => {
                out.push("Fusion");
                for inp in &f.inputs {
                    walk(inp, out);
                }
            }
            LogicalPlan::Union(u) => {
                out.push("Union");
                for arm in &u.arms {
                    walk(arm, out);
                }
            }
            LogicalPlan::CommunityLookup(c) => {
                out.push("CommunityLookup");
                walk(&c.input, out);
            }
            LogicalPlan::VectorNear(_) => out.push("VectorNear"),
            LogicalPlan::TextMatch(_) => out.push("TextMatch"),
            LogicalPlan::Aggregate(a) => {
                out.push("Aggregate");
                walk(&a.input, out);
            }
            LogicalPlan::Sort(s) => {
                out.push("Sort");
                walk(&s.input, out);
            }
            LogicalPlan::Distinct(d) => {
                out.push("Distinct");
                walk(&d.input, out);
            }
            LogicalPlan::Unwind(u) => {
                out.push("Unwind");
                walk(&u.input, out);
            }
            LogicalPlan::ProcedureCall(p) => {
                out.push("ProcedureCall");
                walk(&p.input, out);
            }
            LogicalPlan::NamedPath(np) => {
                out.push("NamedPath");
                walk(&np.input, out);
            }
            LogicalPlan::DynamicLimit(l) => {
                out.push("DynamicLimit");
                walk(&l.input, out);
            }
            LogicalPlan::CreateNode(_) => out.push("CreateNode"),
            LogicalPlan::CreateVectorIndex(_) => out.push("CreateVectorIndex"),
            LogicalPlan::CreatePropertyIndex(_) => out.push("CreatePropertyIndex"),
            LogicalPlan::CreateRel(c) => {
                out.push("CreateRel");
                walk(&c.source_plan, out);
                walk(&c.target_plan, out);
            }
            LogicalPlan::Delete(d) => {
                out.push("Delete");
                walk(&d.input, out);
            }
            LogicalPlan::Set(s) => {
                out.push("Set");
                walk(&s.input, out);
            }
            LogicalPlan::Remove(r) => {
                out.push("Remove");
                walk(&r.input, out);
            }
            LogicalPlan::Merge(m) => {
                out.push("Merge");
                walk(&m.match_branch, out);
                walk(&m.create_branch, out);
            }
            LogicalPlan::Call(c) => {
                out.push("Call");
                walk(&c.input, out);
                walk(&c.body, out);
            }
            LogicalPlan::CorrelationSeed(_) => out.push("CorrelationSeed"),
            LogicalPlan::Empty(_) => out.push("Empty"),
        }
    }

    // -------------------------------------------------------------
    // 1. MATCH (n) → Scan
    // -------------------------------------------------------------

    #[test]
    fn lower_single_match_to_logical_scan() {
        let plan = lower_ok("MATCH (n) RETURN n");
        let s = shape(&plan);
        assert!(s.contains(&"Scan"), "expected Scan node, got: {s:?}");
        assert!(s.contains(&"Project"), "expected Project node, got: {s:?}");
    }

    // -------------------------------------------------------------
    // 2. MATCH (n:Person) → Scan { label: Some(person_id) }
    // -------------------------------------------------------------

    #[test]
    fn lower_match_with_label_to_logical_scan() {
        let plan = lower_ok("MATCH (n:Person) RETURN n");
        // Find the Scan and assert it carries a label.
        let scan = find_scan(&plan).expect("scan present");
        assert!(
            scan.label.is_some(),
            "expected Scan to carry a resolved LabelId for :Person"
        );
    }

    // -------------------------------------------------------------
    // 3. MATCH (a)-[:KNOWS]->(b) → Scan + Expand
    // -------------------------------------------------------------

    #[test]
    fn lower_relationship_pattern_to_logical_expand() {
        let plan = lower_ok("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        let s = shape(&plan);
        assert!(s.contains(&"Scan"), "expected Scan in plan: {s:?}");
        assert!(s.contains(&"Expand"), "expected Expand in plan: {s:?}");
    }

    // -------------------------------------------------------------
    // 4. MATCH (n) WHERE n.age > 30 → Scan + Filter
    // -------------------------------------------------------------

    #[test]
    fn lower_where_clause_to_logical_filter() {
        let plan = lower_ok("MATCH (n:Person) WHERE n.age > 30 RETURN n");
        let s = shape(&plan);
        assert!(s.contains(&"Scan"), "expected Scan in plan: {s:?}");
        assert!(s.contains(&"Filter"), "expected Filter in plan: {s:?}");
        assert!(s.contains(&"Project"), "expected Project in plan: {s:?}");
    }

    // -------------------------------------------------------------
    // 5. RETURN n.name → Project
    // -------------------------------------------------------------

    #[test]
    fn lower_return_to_logical_project() {
        let plan = lower_ok("MATCH (n:Person) RETURN n.name");
        let s = shape(&plan);
        assert!(s.contains(&"Scan"), "expected Scan in plan: {s:?}");
        assert!(s.contains(&"Project"), "expected Project in plan: {s:?}");
    }

    // -------------------------------------------------------------
    // 6. RETURN n LIMIT 10 SKIP 5 → Project + Limit + Skip
    // -------------------------------------------------------------

    #[test]
    fn lower_limit_skip_to_logical_limit_skip() {
        let plan = lower_ok("MATCH (n:Person) RETURN n SKIP 5 LIMIT 10");
        let s = shape(&plan);
        assert!(s.contains(&"Project"), "expected Project: {s:?}");
        assert!(s.contains(&"Skip"), "expected Skip: {s:?}");
        assert!(s.contains(&"Limit"), "expected Limit: {s:?}");
        // Verify the literal counts surfaced correctly.
        let (skip_count, limit_count) = find_skip_limit_counts(&plan);
        assert_eq!(skip_count, Some(5));
        assert_eq!(limit_count, Some(10));
    }

    // -------------------------------------------------------------
    // 7. MATCH (a), (b) → Scan(a) + Scan(b) + Join
    // -------------------------------------------------------------

    #[test]
    fn lower_multi_pattern_match_to_logical_join() {
        let plan = lower_ok("MATCH (a:Person), (b:Doc) RETURN a, b");
        let s = shape(&plan);
        // Two Scans + a Join.
        assert!(
            s.iter().filter(|x| **x == "Scan").count() >= 2,
            "expected ≥2 Scans: {s:?}"
        );
        assert!(s.contains(&"Join"), "expected Join: {s:?}");
    }

    // -------------------------------------------------------------
    // 8. MATCH (n) WITH n WHERE n.x > 0 RETURN n
    //    → Scan + Project + Filter + Project
    // -------------------------------------------------------------

    #[test]
    fn lower_with_clause_to_project_filter() {
        let plan = lower_ok("MATCH (n:Person) WITH n WHERE n.x > 0 RETURN n");
        let s = shape(&plan);
        assert!(s.contains(&"Scan"), "expected Scan: {s:?}");
        assert!(
            s.iter().filter(|x| **x == "Project").count() >= 2,
            "expected ≥2 Projects (one for WITH, one for RETURN): {s:?}"
        );
        assert!(
            s.contains(&"Filter"),
            "expected Filter for WITH WHERE: {s:?}"
        );
    }

    // -------------------------------------------------------------
    // M4-33: 8 unit tests, one per new operator (per ADR-038
    // amendment-03 §M4-33 row pin).
    // -------------------------------------------------------------

    // 9. Aggregation: RETURN count(n) → Aggregate over Scan.
    #[test]
    fn lower_aggregation_to_logical_aggregate() {
        let plan = lower_ok("MATCH (n:Person) RETURN count(n)");
        let s = shape(&plan);
        assert!(s.contains(&"Aggregate"), "expected Aggregate: {s:?}");
        assert!(s.contains(&"Scan"), "expected Scan: {s:?}");
        let agg = find_aggregate(&plan).expect("Aggregate present");
        assert_eq!(agg.aggregations.len(), 1);
        assert_eq!(agg.aggregations[0].function, AggregationKind::Count);
        // Single-row aggregate has no group-by keys.
        assert!(agg.group_by.is_empty(), "single-row aggregate group_by");
    }

    // 10. ORDER BY ascending direction is preserved.
    #[test]
    fn lower_order_by_asc_to_logical_sort() {
        let plan = lower_ok("MATCH (n:Person) RETURN n ORDER BY n.age");
        let s = shape(&plan);
        assert!(s.contains(&"Sort"), "expected Sort: {s:?}");
        let sort = find_sort(&plan).expect("Sort present");
        assert_eq!(sort.order_by.len(), 1);
        // Default direction is Asc per Cypher 9 §6.6.
        assert_eq!(sort.order_by[0].direction, SortDirection::Asc);
    }

    // 11. ORDER BY DESC direction is preserved.
    #[test]
    fn lower_order_by_desc_to_logical_sort() {
        let plan = lower_ok("MATCH (n:Person) RETURN n ORDER BY n.age DESC");
        let sort = find_sort(&plan).expect("Sort present");
        assert_eq!(sort.order_by[0].direction, SortDirection::Desc);
    }

    // 12. RETURN DISTINCT lowers to LogicalDistinct.
    #[test]
    fn lower_distinct_to_logical_distinct() {
        let plan = lower_ok("MATCH (n:Person) RETURN DISTINCT n");
        let s = shape(&plan);
        assert!(s.contains(&"Distinct"), "expected Distinct: {s:?}");
        let d = find_distinct(&plan).expect("Distinct present");
        // The `n` binding is in the on-set.
        assert_eq!(d.on.len(), 1);
    }

    // 12b. UNION ALL lowers to a bare LogicalUnion (no Distinct).
    //      ADR-185 §8 — keep duplicates.
    #[test]
    fn lower_union_all_to_bare_union() {
        let plan = lower_ok(
            "MATCH (a:Person) RETURN a.age AS x UNION ALL MATCH (b:Doc) RETURN b.age AS x",
        );
        let s = shape(&plan);
        assert!(s.contains(&"Union"), "expected Union: {s:?}");
        assert!(
            !s.contains(&"Distinct"),
            "UNION ALL must NOT wrap a Distinct (keep dupes): {s:?}",
        );
        // The plan root IS the Union (no tail in this query).
        assert!(
            matches!(plan, LogicalPlan::Union(_)),
            "UNION ALL root is a bare Union, got {plan:?}",
        );
    }

    // 12c. Bare UNION (distinct) lowers to a LogicalDistinct WRAPPING a
    //      LogicalUnion (#649-A2 — the PE FROZEN CONTRACT item 2
    //      composition; dedup is the standalone DistinctOp OVER UnionOp,
    //      NOT buried in UnionOp). ADR-185 §8 — remove duplicate rows.
    #[test]
    fn lower_bare_union_to_distinct_over_union() {
        let plan =
            lower_ok("MATCH (a:Person) RETURN a.age AS x UNION MATCH (b:Doc) RETURN b.age AS x");
        let s = shape(&plan);
        // Distinct appears ABOVE Union in the pre-order shape.
        let di = s.iter().position(|k| *k == "Distinct");
        let ui = s.iter().position(|k| *k == "Union");
        assert!(di.is_some(), "expected Distinct: {s:?}");
        assert!(ui.is_some(), "expected Union: {s:?}");
        assert!(di < ui, "Distinct must wrap (precede) Union: {s:?}");
        // The plan root IS the Distinct (no tail in this query) wrapping
        // a Union.
        let LogicalPlan::Distinct(d) = &plan else {
            panic!("bare UNION root must be Distinct, got {plan:?}");
        };
        assert!(
            matches!(d.input.as_ref(), LogicalPlan::Union(_)),
            "Distinct must wrap a Union, got {:?}",
            d.input,
        );
        // The dedup-key hint is populated from arm-0's terminal RETURN
        // projection (the `a.age AS x` reference → 1 binding), mirroring
        // RETURN DISTINCT's `on` derivation.
        let dd = find_distinct(&plan).expect("Distinct present");
        assert_eq!(
            dd.on.len(),
            1,
            "on-set carries arm-0's terminal RETURN referenced binding(s): {:?}",
            dd.on,
        );
    }

    // 12d. Bare UNION with a post-union LIMIT tail: the Distinct wraps
    //      the Union, and the static LIMIT wraps the Distinct (tail binds
    //      the WHOLE union AFTER dedup per §8 — the RC-2 fix composed
    //      with A2).
    #[test]
    fn lower_bare_union_tail_limit_wraps_distinct() {
        let plan = lower_ok(
            "MATCH (a:Person) RETURN a.age AS x UNION MATCH (b:Doc) RETURN b.age AS x LIMIT 5",
        );
        let s = shape(&plan);
        let li = s.iter().position(|k| *k == "Limit");
        let di = s.iter().position(|k| *k == "Distinct");
        let ui = s.iter().position(|k| *k == "Union");
        assert!(li.is_some(), "expected Limit: {s:?}");
        assert!(di.is_some(), "expected Distinct: {s:?}");
        assert!(ui.is_some(), "expected Union: {s:?}");
        // Limit ⊐ Distinct ⊐ Union (outermost → innermost).
        assert!(
            li < di && di < ui,
            "expected Limit-over-Distinct-over-Union nesting: {s:?}",
        );
    }

    // 13. UNWIND lowers to LogicalUnwind.
    #[test]
    fn lower_unwind_to_logical_unwind() {
        let plan = lower_ok("UNWIND [1, 2, 3] AS x RETURN x");
        let s = shape(&plan);
        assert!(s.contains(&"Unwind"), "expected Unwind: {s:?}");
    }

    // 14. Named path with SHORTEST_PATH lowers to LogicalNamedPath
    //     carrying ShortestPath.
    #[test]
    fn lower_named_path_with_shortest_to_logical_named_path() {
        let plan = lower_ok("MATCH p = SHORTEST_PATH((a:Person)-[:KNOWS]->(b:Person)) RETURN p");
        let s = shape(&plan);
        assert!(s.contains(&"NamedPath"), "expected NamedPath: {s:?}");
        let np = find_named_path(&plan).expect("NamedPath present");
        assert_eq!(np.algorithm, PathAlgorithm::ShortestPath);
    }

    // 15. Plain named path (no SHORTEST_PATH wrapper) lowers to
    //     LogicalNamedPath carrying Plain.
    #[test]
    fn lower_named_path_plain_to_logical_named_path() {
        let plan = lower_ok("MATCH p = (a:Person)-[:KNOWS]->(b:Person) RETURN p");
        let np = find_named_path(&plan).expect("NamedPath present");
        assert_eq!(np.algorithm, PathAlgorithm::Plain);
    }

    // 16. Parameter-driven LIMIT lowers to LogicalDynamicLimit.
    #[test]
    fn lower_dynamic_limit_with_parameter_to_logical_dynamic_limit() {
        let plan = lower_ok("MATCH (n:Person) RETURN n LIMIT $n");
        let s = shape(&plan);
        assert!(s.contains(&"DynamicLimit"), "expected DynamicLimit: {s:?}");
        let dl = find_dynamic_limit(&plan).expect("DynamicLimit present");
        assert_eq!(dl.kind, DynamicLimitKind::Limit);
    }

    // -------------------------------------------------------------
    // M4-33 helper functions for the unit tests above
    // -------------------------------------------------------------

    fn find_aggregate(p: &LogicalPlan) -> Option<&LogicalAggregate> {
        match p {
            LogicalPlan::Aggregate(a) => Some(a),
            LogicalPlan::Filter(f) => find_aggregate(&f.input),
            LogicalPlan::Project(pr) => find_aggregate(&pr.input),
            LogicalPlan::Join(j) => find_aggregate(&j.left).or_else(|| find_aggregate(&j.right)),
            LogicalPlan::LeftOuterJoin(j) => {
                find_aggregate(&j.left).or_else(|| find_aggregate(&j.right))
            }
            LogicalPlan::Limit(l) => find_aggregate(&l.input),
            LogicalPlan::Skip(s) => find_aggregate(&s.input),
            LogicalPlan::CommunityLookup(c) => find_aggregate(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_aggregate(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_aggregate),
            LogicalPlan::Sort(s) => find_aggregate(&s.input),
            LogicalPlan::Distinct(d) => find_aggregate(&d.input),
            LogicalPlan::Unwind(u) => find_aggregate(&u.input),
            LogicalPlan::ProcedureCall(p) => find_aggregate(&p.input),
            LogicalPlan::NamedPath(np) => find_aggregate(&np.input),
            LogicalPlan::DynamicLimit(l) => find_aggregate(&l.input),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    fn find_sort(p: &LogicalPlan) -> Option<&LogicalSort> {
        match p {
            LogicalPlan::Sort(s) => Some(s),
            LogicalPlan::Filter(f) => find_sort(&f.input),
            LogicalPlan::Project(pr) => find_sort(&pr.input),
            LogicalPlan::Join(j) => find_sort(&j.left).or_else(|| find_sort(&j.right)),
            LogicalPlan::LeftOuterJoin(j) => find_sort(&j.left).or_else(|| find_sort(&j.right)),
            LogicalPlan::Limit(l) => find_sort(&l.input),
            LogicalPlan::Skip(s) => find_sort(&s.input),
            LogicalPlan::CommunityLookup(c) => find_sort(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_sort(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_sort),
            LogicalPlan::Aggregate(a) => find_sort(&a.input),
            LogicalPlan::Distinct(d) => find_sort(&d.input),
            LogicalPlan::Unwind(u) => find_sort(&u.input),
            LogicalPlan::ProcedureCall(p) => find_sort(&p.input),
            LogicalPlan::NamedPath(np) => find_sort(&np.input),
            LogicalPlan::DynamicLimit(l) => find_sort(&l.input),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    fn find_distinct(p: &LogicalPlan) -> Option<&LogicalDistinct> {
        match p {
            LogicalPlan::Distinct(d) => Some(d),
            LogicalPlan::Filter(f) => find_distinct(&f.input),
            LogicalPlan::Project(pr) => find_distinct(&pr.input),
            LogicalPlan::Join(j) => find_distinct(&j.left).or_else(|| find_distinct(&j.right)),
            LogicalPlan::LeftOuterJoin(j) => {
                find_distinct(&j.left).or_else(|| find_distinct(&j.right))
            }
            LogicalPlan::Limit(l) => find_distinct(&l.input),
            LogicalPlan::Skip(s) => find_distinct(&s.input),
            LogicalPlan::CommunityLookup(c) => find_distinct(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_distinct(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_distinct),
            LogicalPlan::Aggregate(a) => find_distinct(&a.input),
            LogicalPlan::Sort(s) => find_distinct(&s.input),
            LogicalPlan::Unwind(u) => find_distinct(&u.input),
            LogicalPlan::ProcedureCall(p) => find_distinct(&p.input),
            LogicalPlan::NamedPath(np) => find_distinct(&np.input),
            LogicalPlan::DynamicLimit(l) => find_distinct(&l.input),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    fn find_named_path(p: &LogicalPlan) -> Option<&LogicalNamedPath> {
        match p {
            LogicalPlan::NamedPath(np) => Some(np),
            LogicalPlan::Filter(f) => find_named_path(&f.input),
            LogicalPlan::Project(pr) => find_named_path(&pr.input),
            LogicalPlan::Join(j) => find_named_path(&j.left).or_else(|| find_named_path(&j.right)),
            LogicalPlan::LeftOuterJoin(j) => {
                find_named_path(&j.left).or_else(|| find_named_path(&j.right))
            }
            LogicalPlan::Limit(l) => find_named_path(&l.input),
            LogicalPlan::Skip(s) => find_named_path(&s.input),
            LogicalPlan::CommunityLookup(c) => find_named_path(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_named_path(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_named_path),
            LogicalPlan::Aggregate(a) => find_named_path(&a.input),
            LogicalPlan::Sort(s) => find_named_path(&s.input),
            LogicalPlan::Distinct(d) => find_named_path(&d.input),
            LogicalPlan::Unwind(u) => find_named_path(&u.input),
            LogicalPlan::ProcedureCall(p) => find_named_path(&p.input),
            LogicalPlan::DynamicLimit(l) => find_named_path(&l.input),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    fn find_dynamic_limit(p: &LogicalPlan) -> Option<&LogicalDynamicLimit> {
        match p {
            LogicalPlan::DynamicLimit(l) => Some(l),
            LogicalPlan::Filter(f) => find_dynamic_limit(&f.input),
            LogicalPlan::Project(pr) => find_dynamic_limit(&pr.input),
            LogicalPlan::Join(j) => {
                find_dynamic_limit(&j.left).or_else(|| find_dynamic_limit(&j.right))
            }
            LogicalPlan::LeftOuterJoin(j) => {
                find_dynamic_limit(&j.left).or_else(|| find_dynamic_limit(&j.right))
            }
            LogicalPlan::Limit(l) => find_dynamic_limit(&l.input),
            LogicalPlan::Skip(s) => find_dynamic_limit(&s.input),
            LogicalPlan::CommunityLookup(c) => find_dynamic_limit(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_dynamic_limit(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_dynamic_limit),
            LogicalPlan::Aggregate(a) => find_dynamic_limit(&a.input),
            LogicalPlan::Sort(s) => find_dynamic_limit(&s.input),
            LogicalPlan::Distinct(d) => find_dynamic_limit(&d.input),
            LogicalPlan::Unwind(u) => find_dynamic_limit(&u.input),
            LogicalPlan::ProcedureCall(p) => find_dynamic_limit(&p.input),
            LogicalPlan::NamedPath(np) => find_dynamic_limit(&np.input),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    // -------------------------------------------------------------
    // ADR-041 §D-4 — `read_lsn` carrier on substrate-touching operators
    // -------------------------------------------------------------

    /// PIN: ADR-041 §D-4 — `LogicalPlanLoweringVisitor::lower` (no
    /// snapshot) stamps `read_lsn = Lsn::MAX` on every substrate-
    /// touching operator. Mirror of the BM25 + vector + community
    /// "callers without snapshot context default to read-latest"
    /// posture.
    #[test]
    fn lower_default_stamps_read_lsn_max_on_scan() {
        let plan = lower_ok("MATCH (n:Person) RETURN n");
        let scan = find_scan(&plan).expect("scan");
        assert_eq!(
            scan.read_lsn,
            Lsn::MAX,
            "PIN: ADR-041 §D-4 — default lower() stamps Lsn::MAX on Scan",
        );
    }

    /// PIN: `lower_at_snapshot(stmt, lsn)` propagates the supplied
    /// LSN to every substrate-touching operator.
    #[test]
    fn lower_at_snapshot_propagates_read_lsn_to_scan() {
        let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
        let cat = cat();
        let mut bound =
            BindingVisitor::bind(&stmt, "MATCH (n:Person) RETURN n", &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
        CrossSubstrateValidator::validate(&bound, &cat).expect("validate");
        let plan = LogicalPlanLoweringVisitor::lower_at_snapshot(&bound, Lsn::new(42))
            .expect("lower at snapshot");
        let scan = find_scan(&plan).expect("scan");
        assert_eq!(
            scan.read_lsn,
            Lsn::new(42),
            "PIN: ADR-041 §D-4 — lower_at_snapshot propagates explicit LSN to Scan",
        );
    }

    /// PIN: every substrate-touching operator (`Scan`,
    /// `VectorNear`, `TextMatch`, `CommunityLookup`,
    /// `HybridOperand`) receives the snapshot LSN. The hybrid +
    /// community surface is the F-1 forcing function — without
    /// the carrier, a hybrid query at a fixed snapshot would
    /// silently mix snapshot-isolated text hits with read-latest
    /// vector hits.
    #[test]
    fn lower_at_snapshot_propagates_to_hybrid_operands() {
        let input = "MATCH (n:Person) RANK BY HYBRID(VECTOR(n.embedding, $v, K = 10), TEXT(n.content, $q, K = 5)) RETURN n";
        let stmt = parse(input).expect("parse");
        let cat = cat().with_vector_index().with_bm25_index();
        let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
        CrossSubstrateValidator::validate(&bound, &cat).expect("validate");
        let plan = LogicalPlanLoweringVisitor::lower_at_snapshot(&bound, Lsn::new(99))
            .expect("lower at snapshot");

        // Walk the tree to find the RankByHybrid node and its operands.
        fn find_rank(p: &LogicalPlan) -> Option<&LogicalRankByHybrid> {
            match p {
                LogicalPlan::RankByHybrid(r) => Some(r),
                LogicalPlan::Project(p) => find_rank(&p.input),
                LogicalPlan::Filter(f) => find_rank(&f.input),
                LogicalPlan::Limit(l) => find_rank(&l.input),
                LogicalPlan::Skip(s) => find_rank(&s.input),
                LogicalPlan::Join(j) => find_rank(&j.left).or_else(|| find_rank(&j.right)),
                LogicalPlan::LeftOuterJoin(j) => find_rank(&j.left).or_else(|| find_rank(&j.right)),
                LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|i| find_rank(i)),
                _ => None,
            }
        }
        let rank = find_rank(&plan).expect("rank-by-hybrid");
        assert!(!rank.operands.is_empty());
        for operand in &rank.operands {
            assert_eq!(
                operand.read_lsn,
                Lsn::new(99),
                "PIN: ADR-041 §D-4 — every HybridOperand receives the snapshot LSN",
            );
        }
    }

    // -------------------------------------------------------------
    // Helpers used by the unit tests above
    // -------------------------------------------------------------

    fn find_scan(p: &LogicalPlan) -> Option<&LogicalScan> {
        match p {
            LogicalPlan::Scan(s) => Some(s),
            // A PropertyIndexScan is NOT a LogicalScan (it replaced one) —
            // this helper is only used by tests that assert the pre-index
            // Scan shape, so it correctly reports "no scan here".
            LogicalPlan::PropertyIndexScan(_) => None,
            LogicalPlan::CountStore(_) => None,
            LogicalPlan::Filter(f) => find_scan(&f.input),
            LogicalPlan::Project(pr) => find_scan(&pr.input),
            LogicalPlan::Join(j) => find_scan(&j.left).or_else(|| find_scan(&j.right)),
            LogicalPlan::LeftOuterJoin(j) => find_scan(&j.left).or_else(|| find_scan(&j.right)),
            LogicalPlan::Limit(l) => find_scan(&l.input),
            LogicalPlan::Skip(s) => find_scan(&s.input),
            LogicalPlan::CommunityLookup(c) => find_scan(&c.input),
            LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_scan(inp)),
            LogicalPlan::Union(u) => u.arms.iter().find_map(find_scan),
            LogicalPlan::Aggregate(a) => find_scan(&a.input),
            LogicalPlan::Sort(s) => find_scan(&s.input),
            LogicalPlan::Distinct(d) => find_scan(&d.input),
            LogicalPlan::Unwind(u) => find_scan(&u.input),
            LogicalPlan::ProcedureCall(p) => find_scan(&p.input),
            LogicalPlan::NamedPath(np) => find_scan(&np.input),
            LogicalPlan::DynamicLimit(l) => find_scan(&l.input),
            LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed are non-recursing leaves
            // for these test-only finders (no existing test passes a CALL
            // plan to them; the per-finder oracle does the asserting).
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => None,
        }
    }

    fn find_skip_limit_counts(p: &LogicalPlan) -> (Option<u64>, Option<u64>) {
        let mut skip = None;
        let mut limit = None;
        walk_collect_counts(p, &mut skip, &mut limit);
        (skip, limit)
    }

    fn walk_collect_counts(p: &LogicalPlan, skip: &mut Option<u64>, limit: &mut Option<u64>) {
        match p {
            LogicalPlan::Skip(s) => {
                *skip = Some(s.count);
                walk_collect_counts(&s.input, skip, limit);
            }
            LogicalPlan::Limit(l) => {
                *limit = Some(l.count);
                walk_collect_counts(&l.input, skip, limit);
            }
            LogicalPlan::Filter(f) => walk_collect_counts(&f.input, skip, limit),
            LogicalPlan::Project(pr) => walk_collect_counts(&pr.input, skip, limit),
            LogicalPlan::Join(j) => {
                walk_collect_counts(&j.left, skip, limit);
                walk_collect_counts(&j.right, skip, limit);
            }
            LogicalPlan::LeftOuterJoin(j) => {
                walk_collect_counts(&j.left, skip, limit);
                walk_collect_counts(&j.right, skip, limit);
            }
            LogicalPlan::CommunityLookup(c) => walk_collect_counts(&c.input, skip, limit),
            LogicalPlan::Fusion(f) => {
                for inp in &f.inputs {
                    walk_collect_counts(inp, skip, limit);
                }
            }
            LogicalPlan::Union(u) => {
                for arm in &u.arms {
                    walk_collect_counts(arm, skip, limit);
                }
            }
            LogicalPlan::Aggregate(a) => walk_collect_counts(&a.input, skip, limit),
            LogicalPlan::Sort(s) => walk_collect_counts(&s.input, skip, limit),
            LogicalPlan::Distinct(d) => walk_collect_counts(&d.input, skip, limit),
            LogicalPlan::Unwind(u) => walk_collect_counts(&u.input, skip, limit),
            LogicalPlan::ProcedureCall(p) => walk_collect_counts(&p.input, skip, limit),
            LogicalPlan::NamedPath(np) => walk_collect_counts(&np.input, skip, limit),
            LogicalPlan::DynamicLimit(l) => walk_collect_counts(&l.input, skip, limit),
            LogicalPlan::Scan(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            // ADR-192 (#623): CALL{} + its seed carry no SKIP/LIMIT.
            | LogicalPlan::Call(_)
            | LogicalPlan::CorrelationSeed(_) => {}
        }
    }
}
