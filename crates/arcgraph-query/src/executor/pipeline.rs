//! Pipeline builder: [`LogicalPlan`] → [`PhysicalOperator`] tree.
//!
//! The builder walks the LogicalPlan recursively, instantiating per-
//! operator state. Operators that the M4-61 / M4-62 slice does NOT
//! support surface [`ExecutionError::NotImplemented`] with a
//! forward-link cite.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — M4-61 / M4-62 slice scope.
//! - **ADR-038 amendment-02 §M4.g** — M4-63 forward slice
//!   (aggregation / sort / etc.).

use std::sync::Arc;

use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::{
    AggregateCall, AggregateOp, CallBodyFactory, CallOp, CorrelationSeedOp, CountStoreOp,
    CreateSpineItem, CreateSpineNode, CreateSpineOp, CreateSpineRel, DistinctOp, EmptyOp, ExpandOp,
    FilterOp, FusionOp, HashJoinOp, LimitOp, MergeJoinOp, NamedShortestPathOp, OptionalExpandOp,
    ParallelAggregateOp, ParallelScanOp, PathSpec, PhysicalOperator, PlainPathOp, ProcedureCallOp,
    ProjectOp, RankByHybridOp, ScanOp, SingletonScanOp, SkipOp, SortKey, SortOp, UnionOp, UnwindOp,
};
use crate::executor::value::Value;
use crate::logical_plan::{
    DynamicLimitKind, HybridOperand, JoinAlgorithm, JoinCondition, LogicalCreateEndpoint,
    LogicalExpand, LogicalJoin, LogicalLeftOuterJoin, LogicalPlan, LogicalRankByHybrid,
    LogicalScan, PathAlgorithm,
};
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// Pipeline builder + driver.
///
/// Constructed once per [`crate::execute`] call; immutable after
/// construction. The thin `Pipeline::build` entry point is what
/// [`crate::execute_with_context`] calls.
pub struct Pipeline;

impl Pipeline {
    /// Build a [`PhysicalOperator`] tree from a [`LogicalPlan`].
    ///
    /// The builder is recursive: each variant either constructs a
    /// leaf op (Scan, Empty, RankByHybrid) or recurses into its
    /// children (Filter, Project, Fusion, LeftOuterJoin).
    pub fn build(plan: &LogicalPlan) -> Result<PhysicalOperator, ExecutionError> {
        Self::build_with_parameters(plan, &Parameters::new())
    }

    /// Build with a per-query parameter bag (used by M4-62 hybrid
    /// operands that resolve `$qv` / `$qt` parameters).
    pub fn build_with_parameters(
        plan: &LogicalPlan,
        parameters: &Parameters,
    ) -> Result<PhysicalOperator, ExecutionError> {
        match plan {
            // ADR-226 §4 S4 (CONC-D): a bare scan promotes to the
            // morsel-driven parallel scan ONLY when the
            // `ARCGRAPH_PARALLEL_SCAN` flag is set; otherwise the serial
            // `ScanOp` is the revert path (flag off → serial). The
            // parallel op produces an identical result (same multiset +
            // id-order); the small-scan guard keeps small scans serial
            // internally, so there is no correctness risk in promoting.
            LogicalPlan::Scan(s) => {
                if ParallelScanOp::enabled_by_env() {
                    Ok(PhysicalOperator::ParallelScan(ParallelScanOp::new(
                        s.var, s.label, s.read_lsn,
                    )))
                } else {
                    Ok(PhysicalOperator::Scan(ScanOp::new(
                        s.var, s.label, s.read_lsn,
                    )))
                }
            }
            // #1366 (Phase 2): the indexed point-lookup leaf. Threads
            // the parameter bag so a `$param` lookup value + a
            // parameterized residual resolve at execute time. Does NOT
            // participate in the `ARCGRAPH_PARALLEL_SCAN` promotion — a
            // point lookup is already `O(matches)`.
            LogicalPlan::PropertyIndexScan(p) => Ok(PhysicalOperator::PropertyIndexScan(
                crate::executor::ops::PropertyIndexScanOp::new(
                    p.var,
                    p.label,
                    p.property.clone(),
                    p.value.clone(),
                    p.residual.clone(),
                    p.read_lsn,
                )
                .with_parameters(parameters.clone()),
            )),
            LogicalPlan::CountStore(c) => Ok(PhysicalOperator::CountStore(CountStoreOp::new(
                c.source,
                c.output_id,
            ))),
            // #618: a `LogicalEmpty` on a plan spine is the openCypher
            // leading-clause DRIVING TABLE — exactly one unit row (see
            // [`EmptyOp::unit`]). Without it a leading no-MATCH clause
            // (`RETURN 1`, `WITH [..] AS l ...`, `UNWIND [..] AS x ...`)
            // emits zero rows. The sole 0-row exception (MERGE's
            // provably-empty match-branch, ADR-151) is built explicitly
            // as `EmptyOp::new()` in the `Merge` arm below.
            LogicalPlan::Empty(_) => Ok(PhysicalOperator::Empty(EmptyOp::unit())),
            LogicalPlan::Filter(f) => {
                // ADR-226 §4 S4 (CONC-D) per-morsel filter pushdown:
                // when the parallel scan is enabled AND the filter sits
                // directly over a bare `LogicalScan`, push the WHERE
                // predicate INTO the parallel scan so the predicate is
                // evaluated per morsel in parallel (parallelism covers
                // the filter cost too). This folds `Filter(Scan)` into a
                // single `ParallelScan{predicate}` — result-identical to
                // the serial `Filter(Scan)` (the equivalence proptest
                // pins this). Any other child shape keeps the serial
                // `FilterOp` wrapping (fallback path).
                if ParallelScanOp::enabled_by_env() {
                    if let LogicalPlan::Scan(s) = f.input.as_ref() {
                        return Ok(PhysicalOperator::ParallelScan(
                            ParallelScanOp::new(s.var, s.label, s.read_lsn)
                                .with_predicate(f.predicate.clone())
                                .with_parameters(parameters.clone()),
                        ));
                    }
                }
                let child = Self::build_with_parameters(&f.input, parameters)?;
                Ok(PhysicalOperator::Filter(
                    FilterOp::new(child, f.predicate.clone()).with_parameters(parameters.clone()),
                ))
            }
            LogicalPlan::Project(p) => {
                // v2 M2 (design §M2.3) — property-projection pushdown.
                // When the WHOLE `Project(Scan)` / `Project(Filter(Scan))`
                // chain provably consumes only `var.<name>` accesses of
                // the scanned variable, annotate the scan with that
                // complete name set so the typed substrate materializes
                // only those key_ids ("the cost of a read is
                // O(|projection|), not O(|bag|)"). The Project is the
                // escape barrier: nothing above it can reference the
                // scan variable. Serial-scan path only at M2 — the
                // env-gated ParallelScan promotion keeps full bags
                // (over-fetch, always correct).
                if !ParallelScanOp::enabled_by_env() {
                    if let Some((var, names)) =
                        crate::executor::projection::scan_projection_for_chain(plan)
                    {
                        let (scan, filter): (&LogicalScan, Option<&BoundExpression>) =
                            match p.input.as_ref() {
                                LogicalPlan::Scan(s) => (s, None),
                                LogicalPlan::Filter(f) => match f.input.as_ref() {
                                    LogicalPlan::Scan(s) => (s, Some(&f.predicate)),
                                    _ => unreachable!(
                                        "scan_projection_for_chain admits only \
                                         Project(Scan) / Project(Filter(Scan))"
                                    ),
                                },
                                _ => unreachable!(
                                    "scan_projection_for_chain admits only Project-rooted chains"
                                ),
                            };
                        debug_assert_eq!(var, scan.var, "pushdown var is the chain's scan var");
                        let mut child = PhysicalOperator::Scan(
                            ScanOp::new(scan.var, scan.label, scan.read_lsn).with_projection(names),
                        );
                        if let Some(pred) = filter {
                            child = PhysicalOperator::Filter(
                                FilterOp::new(child, pred.clone())
                                    .with_parameters(parameters.clone()),
                            );
                        }
                        return Ok(PhysicalOperator::Project(
                            ProjectOp::new(child, p.items.clone())
                                .with_parameters(parameters.clone()),
                        ));
                    }
                }
                let mut child = match p.input.as_ref() {
                    LogicalPlan::CreateNode(_) | LogicalPlan::CreateRel(_) => {
                        Self::build_create_spine(&p.input, parameters, true)?
                    }
                    _ => Self::build_with_parameters(&p.input, parameters)?,
                };
                // #772 (§2.3 silent-wrong → correct): `SET … RETURN …` /
                // `REMOVE … RETURN …` (and the WITH-over-write form, which
                // also lowers to a `Project`) lower to `Project(Set(…))` /
                // `Project(Remove(…))`. A row-consuming `Project` over a
                // write-op input is exactly the RETURN-after-write case: the
                // write-op must PASS its mutated rows THROUGH (stacked) rather
                // than DRAIN them (terminal), so the RETURN projects the rows
                // instead of an empty stream. Flip the child here with the
                // SAME `mark_writeop_input_stacked` the `Set`/`Remove` arms use
                // on a write-op child (the #709 chained-clause fix). A bare
                // `SET …` / `REMOVE …` with NO `Project` above it has no
                // write-op consumer, so its build arm leaves it terminal → 0
                // rows (the openCypher v9 / ADR-149/150 §D / ADR-182
                // RETURN-less terminal-write contract — preserved). Only a
                // direct `Set`/`Remove` child is flipped; a non-write-op child
                // is left untouched.
                mark_writeop_input_stacked(&mut child);
                Ok(PhysicalOperator::Project(
                    ProjectOp::new(child, p.items.clone()).with_parameters(parameters.clone()),
                ))
            }
            LogicalPlan::Expand(e) => {
                // W17α / M4-08+: the M4-31 lowering produces a bare
                // [`LogicalExpand`] (no child sub-plan) as the right
                // side of a [`LogicalJoin`] / [`LogicalLeftOuterJoin`].
                // The executor's join-builder paths can wire it
                // directly:
                //
                // - `LogicalLeftOuterJoin` → routes through
                //   [`Self::build_optional_expand`], which injects a
                //   [`SingletonScanOp`] keyed on the LEFT row's
                //   `from_binding` value.
                // - `LogicalJoin` (the W17α slice) → recurses into
                //   `build_with_parameters(j.right)`, which lands
                //   HERE for a bare Expand. We synthesize an
                //   implicit [`ScanOp`] over the `from` binding (no
                //   label filter; the Join's other side carries the
                //   label-filtered scan when one is present) so the
                //   ExpandOp has a child feeding it row values. The
                //   hash-join's shared-binding semantics filter the
                //   resulting rows down to the matching set.
                //
                // The LogicalExpand at M4-31 does NOT carry an MVCC
                // read_lsn; the executor uses [`arcgraph_core::Lsn::MAX`]
                // (read-latest sentinel — see
                // [`crate::executor::context`]).
                let read_lsn = arcgraph_core::Lsn::MAX;
                let child = PhysicalOperator::Scan(ScanOp::new(e.from, None, read_lsn));
                let exp = ExpandOp::new(
                    child,
                    e.from,
                    e.rel_var,
                    e.to,
                    e.rel_type,
                    e.direction,
                    e.length_range.clone(),
                    read_lsn,
                )?;
                Ok(PhysicalOperator::Expand(exp))
            }
            LogicalPlan::Join(j) => {
                // F2 (PE-1 §F2): pipelined anchor-seeded Expand fast-path.
                // When `j` is the tightly-guarded `(left ⋈ bare-Expand-
                // sharing-from)` inner-join traversal shape (or the label-
                // folded `(… ⋈ bare-label-Scan-on-to)` shape), lower to a
                // pipelined ExpandOp fed by the LEFT child instead of the
                // implicit-unlabeled-rescan + hash join. The rewrite is
                // multiset-identical to the hash join (proof in
                // [`Self::try_pipelined_expand`] / [`Self::build_pipelined_expand`]);
                // any shape it cannot prove identical returns `None` and
                // takes the DP hash/merge join below (bushy patterns stay).
                if let Some(op) = Self::try_pipelined_expand(j, parameters)? {
                    return Ok(op);
                }
                // W25-M4-61b / ADR-097: dispatch on `j.algorithm`.
                // `Auto` falls back to HashJoin defensively — the
                // [`crate::planner::pick_join_algorithms`] pass is
                // expected to resolve `Auto` to a concrete variant
                // before pipeline build, but tests that call
                // `Pipeline::build` directly (skipping the picker)
                // still get a working executor. Cartesian
                // (empty shared bindings) ALWAYS routes to HashJoin
                // regardless of the algorithm field — merge-join is
                // undefined without join keys. See ADR-097
                // §"Algorithm picker policy".
                let left = Self::build_with_parameters(&j.left, parameters)?;
                let right = Self::build_with_parameters(&j.right, parameters)?;
                let JoinCondition::SharedBindings(shared) = &j.on;
                let use_merge =
                    matches!(j.algorithm, JoinAlgorithm::MergeJoin) && !shared.is_empty();
                if use_merge {
                    let op = MergeJoinOp::new(left, right, shared.clone())?;
                    Ok(PhysicalOperator::MergeJoin(op))
                } else {
                    let op = HashJoinOp::new(left, right, shared.clone())?;
                    Ok(PhysicalOperator::HashJoin(op))
                }
            }
            LogicalPlan::LeftOuterJoin(j) => Self::build_optional_expand(j, parameters),
            LogicalPlan::RankByHybrid(r) => Ok(PhysicalOperator::RankByHybrid(
                Self::build_rank_by_hybrid(r, parameters),
            )),
            LogicalPlan::Fusion(f) => {
                // The fusion clause attaches a `k` to the inner
                // RankByHybrid; M4-32 lowering produces
                // `Fusion { spec, inputs: [hybrid_root] }`.
                // v1.0-alpha admits a single-input Fusion over a
                // RankByHybrid; if the inner is something else, we
                // surface NotImplemented.
                let inner = f
                    .inputs
                    .first()
                    .ok_or_else(|| ExecutionError::NotImplemented {
                        feature: "Fusion with no inputs".into(),
                        target_slice: "M4-32 lowering invariant".into(),
                        section: "ADR-038 §2 D-9".into(),
                    })?;
                let inner_op = Self::build_with_parameters(inner, parameters)?;
                Ok(PhysicalOperator::Fusion(FusionOp::new(inner_op)))
            }
            LogicalPlan::Limit(l) => {
                let child = Self::build_with_parameters(&l.input, parameters)?;
                Ok(PhysicalOperator::Limit(LimitOp::new(child, l.count)))
            }
            LogicalPlan::Skip(s) => {
                // #842 part A — literal `SKIP N` is now lit (mirrors the
                // LIMIT arm above). The M4-33 lowering only produces
                // `LogicalSkip` for a LITERAL count (see
                // `lower_skip_or_limit_with_span`); a parameter /
                // expression `SKIP $n` lowers to `LogicalDynamicLimit`
                // and is handled by that arm below.
                let child = Self::build_with_parameters(&s.input, parameters)?;
                Ok(PhysicalOperator::Skip(SkipOp::new(child, s.count)))
            }
            LogicalPlan::DynamicLimit(d) => {
                let child = Self::build_with_parameters(&d.input, parameters)?;
                let count = Self::resolve_dynamic_limit_count(
                    &d.count_expr,
                    dynamic_limit_clause_name(d.kind),
                    parameters,
                )?;
                match d.kind {
                    DynamicLimitKind::Limit => {
                        Ok(PhysicalOperator::Limit(LimitOp::new(child, count)))
                    }
                    DynamicLimitKind::Skip => Ok(PhysicalOperator::Skip(SkipOp::new(child, count))),
                }
            }
            LogicalPlan::Aggregate(a) => {
                let mut child = Self::build_with_parameters(&a.input, parameters)?;
                // #772 — `SET … RETURN <agg>` / `WITH <agg>` (e.g.
                // `RETURN count(*)` / `sum(a.x)`) lowers to
                // `Project(Aggregate(Set(…)))`: the AGGREGATE — not the
                // Project — is the write-op's direct parent, so the
                // Aggregate must flip its `Set`/`Remove` child to stacked
                // (else the terminal SET drains its rows and the aggregate
                // folds over 0 rows → `count(*)=0` / `sum=NULL`, a
                // silent-wrong). Same `mark_writeop_input_stacked` the
                // Project / Set / Remove arms use; a non-write-op child is
                // left untouched.
                mark_writeop_input_stacked(&mut child);
                let aggregations: Vec<AggregateCall> = a
                    .aggregations
                    .iter()
                    .map(|spec| AggregateCall {
                        kind: spec.function,
                        arg: spec.arg.clone(),
                        // #746: thread the lowering's output id through
                        // so the AggregateOp emits each result under the
                        // id the layered ProjectOp references.
                        output_id: spec.output_id,
                        // #773 G4/G5 — DISTINCT dedup + count(*) row count.
                        distinct: spec.distinct,
                        star: spec.star,
                    })
                    .collect();
                // ADR-226 §4 S5 (CONC-D) — route to the morsel-driven
                // PARALLEL partial aggregate when the shared
                // `ARCGRAPH_PARALLEL_SCAN` flag is set AND the aggregate is
                // provably mergeable (no GROUP BY / DISTINCT / COLLECT).
                // Otherwise the serial AggregateOp (flag off or not
                // mergeable → byte-identical serial result; the revert
                // path). The two ops produce the same result; the flag +
                // predicate only pick the execution strategy.
                if ParallelAggregateOp::enabled_by_env()
                    && ParallelAggregateOp::is_mergeable(&a.group_by, &aggregations)
                {
                    return Ok(PhysicalOperator::ParallelAggregate(
                        ParallelAggregateOp::new(child, aggregations)
                            .with_parameters(parameters.clone()),
                    ));
                }
                Ok(PhysicalOperator::Aggregate(
                    AggregateOp::new(child, a.group_by.clone(), aggregations)
                        .with_parameters(parameters.clone()),
                ))
            }
            LogicalPlan::Sort(s) => {
                let child = Self::build_with_parameters(&s.input, parameters)?;
                let keys: Vec<SortKey> = s
                    .order_by
                    .iter()
                    .map(|item| SortKey {
                        expr: item.expr.clone(),
                        direction: item.direction,
                    })
                    .collect();
                Ok(PhysicalOperator::Sort(
                    SortOp::new(child, keys).with_parameters(parameters.clone()),
                ))
            }
            LogicalPlan::NamedPath(np) => {
                let child = Self::build_with_parameters(&np.input, parameters)?;
                match np.algorithm {
                    // ADR-193 D-4 — the Plain (full path enumeration)
                    // variant is now LIT (ADR-038 §2 D-7's v1.1 deferral
                    // is REVERSED per ADR-190's GA-rescope; the reversal
                    // is recorded in ADR-038-amendment-10). Materialize a
                    // `Value::Path` from the bound MATCH rows using the
                    // ordered element bindings the lowering captured.
                    PathAlgorithm::Plain => {
                        let shape = np.plain_shape.clone().ok_or_else(|| {
                            ExecutionError::Eval(
                                "NamedPath::Plain missing plain_shape (lowering invariant — \
                                 ADR-193 D-4)"
                                    .into(),
                            )
                        })?;
                        Ok(PhysicalOperator::PlainPath(PlainPathOp::new(
                            child,
                            shape,
                            np.path_var,
                        )))
                    }
                    PathAlgorithm::ShortestPath => {
                        // The path source binding is derived from the
                        // child's schema's first slot per the v1.0
                        // lowering convention. ADR-194 D-3a — the target
                        // slot is now threaded from the lowering's
                        // captured pattern tail-endpoint binding
                        // (`np.target`): `Some(b)` ⇒ bidirectional
                        // source→target BFS (one path per `(a, b)` pair);
                        // `None` (anonymous tail endpoint) ⇒ single-source
                        // enumeration. This LIGHTS the previously-dead
                        // `bidirectional()` path in `executor::ops::path`.
                        // ADR-194 D-3a — prefer the lowering-captured head
                        // (source) binding; fall back to the legacy
                        // schema-first slot only for an anonymous head
                        // (degenerate). The schema-first slot is unstable
                        // when a tail-label join reorders the child schema.
                        let source = match np.source.or_else(|| child.schema().first().copied()) {
                            Some(b) => b,
                            None => {
                                return Err(ExecutionError::Eval(
                                    "NamedShortestPathOp: child schema is empty".into(),
                                ));
                            }
                        };
                        Ok(PhysicalOperator::NamedShortestPath(
                            NamedShortestPathOp::new(
                                child,
                                PathSpec {
                                    source,
                                    target: np.target,
                                    rel_type: None,
                                    direction: crate::logical_plan::Direction::Undirected,
                                    path_var: np.path_var,
                                    all_shortest: false,
                                },
                                arcgraph_core::Lsn::MAX,
                            ),
                        ))
                    }
                    // ADR-194 D-4 — `allShortestPaths` enumerates ALL
                    // equal-minimum-length source→target paths. It is
                    // INTRINSICALLY src→dst: an anonymous tail endpoint
                    // (`np.target = None`) is ill-formed, so we reject with
                    // a clean error rather than silently degrading to the
                    // single-source enumeration the `ShortestPath` arm uses
                    // for `target = None`.
                    PathAlgorithm::AllShortestPaths => {
                        let source = match np.source.or_else(|| child.schema().first().copied()) {
                            Some(b) => b,
                            None => {
                                return Err(ExecutionError::Eval(
                                    "allShortestPaths: child schema is empty".into(),
                                ));
                            }
                        };
                        let target = np.target.ok_or_else(|| {
                            ExecutionError::Eval(
                                "allShortestPaths requires a bound target endpoint \
                                 (e.g. `allShortestPaths((a)-[..]-(b))` with a named `b`); \
                                 an anonymous tail endpoint is ill-formed (ADR-194 D-4)"
                                    .into(),
                            )
                        })?;
                        Ok(PhysicalOperator::NamedShortestPath(
                            NamedShortestPathOp::new(
                                child,
                                PathSpec {
                                    source,
                                    target: Some(target),
                                    rel_type: None,
                                    direction: crate::logical_plan::Direction::Undirected,
                                    path_var: np.path_var,
                                    all_shortest: true,
                                },
                                arcgraph_core::Lsn::MAX,
                            ),
                        ))
                    }
                }
            }
            // ADR-185 (#649-A1, W28): DISTINCT is now LIT — closes the
            // prior M4-72 deferral. `RETURN DISTINCT` executes via the
            // standalone DistinctOp (full-row hash-set dedup).
            LogicalPlan::Distinct(d) => {
                let child = Self::build_with_parameters(&d.input, parameters)?;
                Ok(PhysicalOperator::Distinct(DistinctOp::new(child)))
            }
            // ADR-185 (#649-A1, W28): UNION ALL concat. Each arm is
            // built independently; the per-arm column permutation rides
            // from the bind pass for §8 order-independent realignment.
            LogicalPlan::Union(u) => {
                let arms: Vec<PhysicalOperator> = u
                    .arms
                    .iter()
                    .map(|arm| Self::build_with_parameters(arm, parameters))
                    .collect::<Result<_, _>>()?;
                Ok(PhysicalOperator::Union(UnionOp::new(
                    arms,
                    u.column_orders.clone(),
                )))
            }
            // ADR-038 D-28 §7 (#618): UNWIND is now lit — closes the
            // prior `NotImplemented`. `UNWIND <list> AS <var>` builds the
            // child sub-pipeline, then streams one row per list element
            // (openCypher v9 §6.7). Mirrors the `Project` arm shape.
            LogicalPlan::Unwind(u) => {
                let mut child = Self::build_with_parameters(&u.input, parameters)?;
                // #772 — `SET … UNWIND … (RETURN …)` lowers to
                // `Unwind(Set(…))`: UNWIND is the write-op's direct parent
                // and consumes its rows (one output row per list element per
                // input row), so it must flip its `Set`/`Remove` child to
                // stacked (else the terminal SET drains and UNWIND expands 0
                // rows → `[]`, a silent-wrong). Same `mark_writeop_input_stacked`
                // as the Project / Aggregate / Set / Remove arms.
                mark_writeop_input_stacked(&mut child);
                Ok(PhysicalOperator::Unwind(
                    UnwindOp::new(child, u.list_expr.clone(), u.var)
                        .with_parameters(parameters.clone()),
                ))
            }
            // ADR-197 (#802): CALL <proc>(…) [YIELD …] / SHOW …
            LogicalPlan::ProcedureCall(p) => {
                let child = Self::build_with_parameters(&p.input, parameters)?;
                // #830 D4: thread the bound args + the per-query
                // parameter bag into the op so `db.index.vector.queryNodes`
                // can evaluate its (indexName, k, queryVector) arguments —
                // including langchain's `$top_k * $effective_search_ratio`
                // k expression, which references parameters.
                Ok(PhysicalOperator::ProcedureCall(
                    ProcedureCallOp::new(
                        child,
                        p.source.clone(),
                        p.columns.clone(),
                        p.args.clone(),
                    )
                    .with_parameters(parameters.clone()),
                ))
            }
            LogicalPlan::CommunityLookup(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_) => Err(ExecutionError::NotImplemented {
                feature: format!(
                    "LogicalPlan variant {} (filter-shaped substrate op)",
                    plan_variant(plan)
                ),
                target_slice: "M4-62b (M4-32 sub-substrate filter op)".into(),
                section: "ADR-038 §2 D-26".into(),
            }),
            LogicalPlan::CreateNode(_) => Self::build_create_spine(plan, parameters, false),
            // #830 / ADR-200: CREATE VECTOR INDEX accept-and-register. A
            // leaf op (no child). Threads the per-query parameter bag so
            // the real client's `$name` / `toInteger($dimensions)` /
            // `$similarity_fn` resolve at execute time.
            LogicalPlan::CreateVectorIndex(c) => Ok(PhysicalOperator::CreateVectorIndex(
                crate::executor::ops::CreateVectorIndexOp::new(
                    c.name.clone(),
                    c.if_not_exists,
                    c.label.clone(),
                    c.property.clone(),
                    c.options.clone(),
                )
                .with_parameters(parameters.clone()),
            )),
            // #1366 (task #248, Phase 1): CREATE INDEX (property index)
            // register+backfill leaf op. Threads the parameter bag for a
            // `$name` resolution at execute time.
            LogicalPlan::CreatePropertyIndex(c) => Ok(PhysicalOperator::CreatePropertyIndex(
                crate::executor::ops::CreatePropertyIndexOp::new(
                    c.name.clone(),
                    c.if_not_exists,
                    c.label.clone(),
                    c.property.clone(),
                )
                .with_parameters(parameters.clone()),
            )),
            LogicalPlan::CreateRel(_) => Self::build_create_spine(plan, parameters, false),
            LogicalPlan::Delete(d) => {
                // ADR-149 W26-θ Phase 3: build the upstream
                // sub-pipeline (typically the prior MATCH's lowered
                // plan), then wire it into the DeleteOp. The DeleteOp
                // pulls upstream rows + per-row per-item dispatches
                // to the substrate's delete_node / delete_rel.
                let input_op = Self::build_with_parameters(&d.input, parameters)?;
                let items: Vec<crate::executor::ops::DeleteItemSpec> = d
                    .items
                    .iter()
                    .map(|it| crate::executor::ops::DeleteItemSpec {
                        binding: it.binding,
                        kind: it.kind,
                    })
                    .collect();
                Ok(PhysicalOperator::Delete(
                    crate::executor::ops::DeleteOp::new(input_op, items, d.detach),
                ))
            }
            LogicalPlan::Set(s) => {
                // ADR-150 W26-θ Phase 4 (#709 fix, R1-narrowed): build the
                // upstream sub-pipeline, then wire it into the SetOp. The
                // SetOp pulls upstream rows + per-row per-item dispatches
                // to the substrate's set_node / set_rel.
                let mut input_op = Self::build_with_parameters(&s.input, parameters)?;
                // Terminal-vs-stacked discriminator (#709): if THIS SET's
                // input is itself a write-op (SET/REMOVE), that child is
                // *stacked* — it has a write-op consumer (this SET) above
                // it, so it must PASS its mutated rows through (not drain).
                // Flip it here. The SET we build below stays terminal
                // unless ITS own parent (a grandparent write-op) flips it.
                // Only a direct write-op→write-op parent edge marks a child
                // stacked; a SET under any other parent (root / Project /
                // etc.) is terminal → 0 rows (the openCypher / ADR-149/150
                // §D / ADR-182 terminal-write contract). The lowering
                // produces exactly this `Set(Set(Scan))` nesting for
                // chained clauses (see `lower_query`'s clause fold).
                mark_writeop_input_stacked(&mut input_op);
                let items: Vec<crate::executor::ops::SetItemSpec> = s
                    .items
                    .iter()
                    .map(|it| crate::executor::ops::SetItemSpec {
                        binding: it.binding,
                        kind: it.kind,
                        mutation: it.mutation.clone(),
                    })
                    .collect();
                Ok(PhysicalOperator::Set(crate::executor::ops::SetOp::new(
                    input_op, items,
                )))
            }
            LogicalPlan::Remove(r) => {
                // ADR-150 W26-θ Phase 4 (#709 fix, R1-narrowed): build the
                // upstream sub-pipeline, then wire it into the RemoveOp.
                let mut input_op = Self::build_with_parameters(&r.input, parameters)?;
                // Same terminal-vs-stacked discriminator as the Set arm: a
                // write-op input of this REMOVE is stacked (pass-through).
                mark_writeop_input_stacked(&mut input_op);
                let items: Vec<crate::executor::ops::RemoveItemSpec> = r
                    .items
                    .iter()
                    .map(|it| crate::executor::ops::RemoveItemSpec {
                        binding: it.binding,
                        kind: it.kind,
                        mutation: it.mutation.clone(),
                    })
                    .collect();
                Ok(PhysicalOperator::Remove(
                    crate::executor::ops::RemoveOp::new(input_op, items),
                ))
            }
            LogicalPlan::Merge(m) => {
                // ADR-151 W26-θ Phase 5: build both the match and
                // create sub-pipelines, then wire them into the
                // MergeOp. The MergeOp pulls the match-branch first;
                // if empty, fires the create-branch.
                //
                // #618: MERGE's provably-empty match-branch (the ADR-151
                // uninterned-label case lowers to `LogicalEmpty`) is a
                // 0-row EOS probe, NOT the leading-clause unit driving
                // row. Build it explicitly as the EOS-sentinel
                // `EmptyOp::new()` so the generic `LogicalEmpty →
                // EmptyOp::unit()` default (one unit row) cannot leak in
                // and make `MergeOp` believe the pattern matched.
                let match_op = match m.match_branch.as_ref() {
                    LogicalPlan::Empty(_) => PhysicalOperator::Empty(EmptyOp::new()),
                    other => Self::build_with_parameters(other, parameters)?,
                };
                let create_op = Self::build_with_parameters(&m.create_branch, parameters)?;
                let on_create: Vec<crate::executor::ops::MergeActionSpec> = m
                    .on_create
                    .iter()
                    .map(|it| crate::executor::ops::MergeActionSpec {
                        binding: it.binding,
                        kind: it.kind,
                        mutation: it.mutation.clone(),
                    })
                    .collect();
                let on_match: Vec<crate::executor::ops::MergeActionSpec> = m
                    .on_match
                    .iter()
                    .map(|it| crate::executor::ops::MergeActionSpec {
                        binding: it.binding,
                        kind: it.kind,
                        mutation: it.mutation.clone(),
                    })
                    .collect();
                Ok(PhysicalOperator::Merge(crate::executor::ops::MergeOp::new(
                    match_op,
                    create_op,
                    on_create,
                    on_match,
                    // ADR-151-amendment-01 §D-1 — node-shape named merge
                    // emits its binding row(s) (RETURN-after-MERGE);
                    // path-shape / anonymous stay terminal.
                    m.output_binding,
                    // NN-4 (#1384) re-spin — the get-or-create serialization
                    // guard(s) are acquired by the query DRIVER from
                    // `m.merge_keys` BEFORE the snapshot pin (see
                    // `acquire_merge_guards`), NOT by the op — so the op no
                    // longer carries the key spec.
                )))
            }
            // ADR-192 (#623): CALL { <subquery> } correlated brace-subquery
            // (Cypher 25, beyond openCypher v9). Build the driving `input`;
            // capture the body plan + parameters into a factory that
            // rebuilds the body sub-pipeline per driving row (operators
            // carry per-run cursor state, so they are rebuilt, not reset —
            // the OptionalExpandOp precedent). The body's CorrelationSeedOp
            // reads the per-row imports from the ExecutionContext frame
            // CallOp pushes, so the factory is argument-free. The body
            // output is relabeled to `returned` (the binder-declared
            // outer-scope ids) positionally by CallOp (D-4 — the body's
            // own projection mints fresh synthetic ids).
            LogicalPlan::Call(c) => {
                let child = Self::build_with_parameters(&c.input, parameters)?;
                let body_plan = Arc::new((*c.body).clone());
                let params = parameters.clone();
                let factory: CallBodyFactory =
                    Box::new(move || Self::build_with_parameters(body_plan.as_ref(), &params));
                Ok(PhysicalOperator::Call(CallOp::new(
                    child,
                    factory,
                    c.imported.clone(),
                    c.returned.clone(),
                )))
            }
            // ADR-192 (#623): the correlation seed — a one-row table of the
            // imported bindings (read from the ExecutionContext frame at
            // exec time). Appears only inside a `LogicalCall::body`.
            LogicalPlan::CorrelationSeed(s) => Ok(PhysicalOperator::CorrelationSeed(
                CorrelationSeedOp::new(s.imported.clone()),
            )),
        }
    }

    fn build_create_spine(
        plan: &LogicalPlan,
        parameters: &Parameters,
        expose_path_endpoint_bindings: bool,
    ) -> Result<PhysicalOperator, ExecutionError> {
        let mut spine = Vec::new();
        let mut cursor = plan;
        let tail = loop {
            match cursor {
                LogicalPlan::CreateNode(c) => {
                    spine.push(cursor);
                    match c.input.as_deref() {
                        Some(input) => cursor = input,
                        None => break None,
                    }
                }
                LogicalPlan::CreateRel(c) => {
                    spine.push(cursor);
                    match c.input.as_deref() {
                        Some(input) => cursor = input,
                        None => break None,
                    }
                }
                other => break Some(other),
            }
        };

        let input = tail
            .map(|tail| Self::build_with_parameters(tail, parameters))
            .transpose()?;

        let mut items = Vec::with_capacity(spine.len());
        for step in spine.into_iter().rev() {
            let item = match step {
                LogicalPlan::CreateNode(c) => CreateSpineItem::Node(CreateSpineNode {
                    var: c.var,
                    label: c.label.clone(),
                    properties: c.properties.clone(),
                }),
                LogicalPlan::CreateRel(c) => {
                    let source_op = match c.source_endpoint {
                        LogicalCreateEndpoint::Fresh => {
                            Self::build_with_parameters(&c.source_plan, parameters)?
                        }
                        LogicalCreateEndpoint::RowBinding(_) => {
                            PhysicalOperator::Empty(EmptyOp::new())
                        }
                    };
                    let target_op = match c.target_endpoint {
                        LogicalCreateEndpoint::Fresh => {
                            Self::build_with_parameters(&c.target_plan, parameters)?
                        }
                        LogicalCreateEndpoint::RowBinding(_) => {
                            PhysicalOperator::Empty(EmptyOp::new())
                        }
                    };
                    CreateSpineItem::Rel(Box::new(CreateSpineRel {
                        var: c.var,
                        label: c.label.clone(),
                        properties: c.properties.clone(),
                        source_op,
                        source_binding: c.source,
                        source_visible: c.source_visible,
                        source_endpoint: c.source_endpoint,
                        target_op,
                        target_binding: c.target,
                        target_visible: c.target_visible,
                        target_endpoint: c.target_endpoint,
                        direction: c.direction.clone(),
                    }))
                }
                _ => unreachable!("create spine contains only CREATE write ops"),
            };
            items.push(item);
        }

        if items.is_empty() {
            return Err(ExecutionError::Eval(
                "Pipeline::build_create_spine called with empty spine".into(),
            ));
        }

        Ok(PhysicalOperator::CreateSpine(
            // ADR-147-amendment-03 (D-1): thread the per-query parameter
            // bag so CREATE property values referencing `$param` /
            // previously-bound rows resolve at materialization time.
            CreateSpineOp::new(input, items, expose_path_endpoint_bindings)
                .with_parameters(parameters.clone()),
        ))
    }

    fn resolve_dynamic_limit_count(
        expr: &BoundExpression,
        clause: &'static str,
        parameters: &Parameters,
    ) -> Result<u64, ExecutionError> {
        let lookup = |_| None;
        match evaluate(expr, &[], &lookup, parameters)? {
            Value::Integer(n) if n >= 0 => Ok(n as u64),
            Value::Integer(n) => Err(ExecutionError::Eval(format!(
                "{clause} value must be non-negative; got {n}"
            ))),
            Value::Null => Err(ExecutionError::Eval(format!(
                "{clause} value must not be null"
            ))),
            other => Err(ExecutionError::Eval(format!(
                "{clause} value must be an integer; got {}",
                dynamic_limit_value_type(&other)
            ))),
        }
    }

    fn build_rank_by_hybrid(r: &LogicalRankByHybrid, parameters: &Parameters) -> RankByHybridOp {
        let mut op = RankByHybridOp::new(r.operands.clone(), default_read_lsn(&r.operands))
            .with_parameters(parameters.clone());
        if let Some(fusion) = &r.fusion {
            op = op.with_fusion_k(fusion.k);
        }
        if let Some(score) = r.score_binding {
            op = op.with_score_binding(score);
        }
        op
    }

    fn build_optional_expand(
        j: &LogicalLeftOuterJoin,
        parameters: &Parameters,
    ) -> Result<PhysicalOperator, ExecutionError> {
        // Build the LEFT side eagerly. The RIGHT side must be
        // re-runnable per LEFT row, so we capture the right plan +
        // the join condition, building a fresh sub-pipeline per LEFT
        // row inside the factory closure.
        let left = Self::build_with_parameters(&j.left, parameters)?;
        let right_plan = Arc::new((*j.right).clone());
        let join = j.on.clone();
        let parameters_owned = parameters.clone();
        // Identify the join binding(s).
        let JoinCondition::SharedBindings(shared) = join;
        // For OPTIONAL MATCH the right-side sub-pipeline pre-binds
        // the shared variable to the LEFT row's value via a
        // SingletonScanOp that emits the LEFT row's `from` node;
        // the right plan then expands from that node.
        //
        // Empty-shared (#996-followup): a LEADING OPTIONAL MATCH
        // (`OPTIONAL MATCH (n) ...` as the first clause) lowers to a
        // LeftOuterJoin whose LEFT is the unit row ([`LogicalEmpty`] →
        // [`EmptyOp::unit`]) and whose `shared` set is EMPTY — the unit
        // row shares no bindings with the pattern. This is NOT the
        // singleton-root expand shape (there is no LEFT-row node to root
        // the right side at); it is a Cartesian unit join that emits the
        // right rows verbatim, or exactly ONE null-extended row when the
        // right side is empty (openCypher 9 §6.5). We build the right side
        // ONCE via the plain `build_with_parameters` (it is independent of
        // any LEFT-row value) and hand `OptionalExpandOp` a factory that
        // ignores the left row and rebuilds the right plan fresh per call.
        // `join_pairs` stays empty, so `OptionalExpandOp` passes every
        // right row through and null-extends only when the right is empty.
        let Some(from_binding) = shared.first().copied() else {
            let right_schema = Self::build_with_parameters(&right_plan, parameters)?
                .schema()
                .to_vec();
            let factory_plan = right_plan.clone();
            let factory_params = parameters_owned.clone();
            let factory = move |_left_row: &[Value]| -> PhysicalOperator {
                // The right plan is independent of the LEFT (unit) row, so
                // the only LEFT row is the single unit row; this rebuild is
                // hit exactly once. The right plan is a leading-clause shape
                // (Scan / Filter / Project over the optional pattern), which
                // `build_with_parameters` always builds; the EmptyOp backstop
                // preserves OPTIONAL null-extension if a future refactor ever
                // broke that invariant.
                Self::build_with_parameters(&factory_plan, &factory_params)
                    .unwrap_or_else(|_| PhysicalOperator::Empty(EmptyOp::new()))
            };
            return Ok(PhysicalOperator::OptionalExpand(OptionalExpandOp::new(
                left,
                right_schema,
                factory,
            )));
        };
        // #771 (§2.3 silent-wrong → correct-or-loud): derive the
        // right-side schema from the SINGLETON-ROOTED build — the SAME
        // path the per-row factory walks — NOT the plain
        // `build_with_parameters` build. The two diverge for the
        // labeled-node-with-rel idiom `(c:Commit)-[:FIXES]->(i)`, whose
        // right side lowers to a `Join(Scan(c), Expand(c→i))`: the
        // singleton-rooted build REVERSES the Expand (roots the shared
        // `i` at the SingletonScan), reordering an Expand's internal
        // `[from,to]` columns. Probing with the plain build would hand
        // `OptionalExpandOp` a schema whose fresh-binding indices don't
        // line up with the factory's actual rows.
        //
        // This probe ALSO converts the prior silent-all-NULL into an
        // HONEST error: previously a right-side shape
        // `build_right_with_singleton_root` could not build surfaced
        // `NotImplemented`, which the per-row factory's
        // `unwrap_or_else(|_| EmptyOp)` SWALLOWED into "every left row
        // null-extends" (a §2.3 silent-wrong left-join). Hoisting the
        // fallible build here — to pipeline-build time, BEFORE any
        // OptionalExpandOp is constructed — makes an unbuildable right
        // side fail LOUDLY (the client sees the `NotImplemented`)
        // rather than silently nulling real matches. The factory's
        // closure returns a bare `PhysicalOperator`
        // (`RightSidePipelineFactory` in `ops::optional_expand`), so the
        // error cannot be `?`-propagated per-row; the build-time probe
        // is the strictly-earlier, strictly-cleaner place to fail.
        //
        // `NodeId::ZERO` is a schema-only placeholder: it flows solely
        // into `SingletonScanOp::new`, which never resolves it at build
        // time, and the real LEFT-row node id threads in per-row inside
        // the factory. The probe's buildability + schema are therefore
        // identical to every per-row rebuild (which differs only in that
        // placeholder), so a probe that succeeds guarantees the per-row
        // builds succeed too.
        let right_probe = build_right_with_singleton_root(
            right_plan.as_ref(),
            from_binding,
            arcgraph_core::NodeId::ZERO,
            parameters,
        )?;
        let right_schema = right_probe.schema().to_vec();
        let left_schema = left.schema().to_vec();
        let join_pairs = resolve_optional_join_pairs(&left_schema, &right_schema, &shared)?;
        // The factory: for each LEFT row, peel out the `from_binding`
        // value, build a SingletonScanOp + the right_plan rooted at
        // it. The right_plan walked from the SingletonScanOp may
        // contain Scan / Expand / Filter / Project / Join — we re-walk
        // via `build_right_with_singleton_root` and graft.
        Self::build_optional_expand_with_unwrapped_expand(
            left,
            right_plan,
            right_schema,
            join_pairs,
            from_binding,
            parameters_owned,
        )
    }

    fn build_optional_expand_with_unwrapped_expand(
        left: PhysicalOperator,
        right_plan: Arc<LogicalPlan>,
        right_schema: Vec<BindingId>,
        join_pairs: Vec<(usize, usize)>,
        from_binding: BindingId,
        parameters: Parameters,
    ) -> Result<PhysicalOperator, ExecutionError> {
        // Snapshot the right-plan into a closure-friendly Arc.
        let right_plan = right_plan.clone();
        let parameters = parameters.clone();
        let factory = move |left_row: &[Value]| -> PhysicalOperator {
            // Identify the from-node from the left row. If the
            // left schema doesn't include the binding, we panic
            // (this is a planner contract violation).
            //
            // v1.0-alpha simplification: assume the LEFT row's first
            // column is the from-binding. This holds for the M4-62
            // smoke fixtures; richer schemas land at M4-63.
            let from_node = match left_row.first() {
                Some(Value::Node(n)) => n.id,
                _ => {
                    // Defensive: build an EmptyOp so the OPTIONAL
                    // MATCH null-row branch fires.
                    return PhysicalOperator::Empty(EmptyOp::new());
                }
            };
            // Build the right side ROOTED at a SingletonScanOp.
            // We re-walk the right_plan's Scan / Expand / Filter /
            // Project / Join shape and substitute the shared-binding
            // leaf with the SingletonScanOp keyed on this LEFT row's
            // node (rooting + reversing the Expand as needed so the
            // optional pattern executes constrained to this row).
            //
            // #771: the `unwrap_or_else` EmptyOp fallback is now an
            // UNREACHABLE backstop, NOT a silent-wrong path. The
            // `build_optional_expand` build-time probe already ran this
            // exact `build_right_with_singleton_root` call (with the
            // `NodeId::ZERO` placeholder) and propagated any error; the
            // per-row rebuild differs only in the node id, which never
            // affects buildability. So a right side that reaches this
            // closure is provably buildable. The EmptyOp branch survives
            // only to keep the closure total without panicking in the
            // hot path; it preserves OPTIONAL null-extension semantics
            // if a future refactor ever broke that invariant.
            build_right_with_singleton_root(&right_plan, from_binding, from_node, &parameters)
                .unwrap_or_else(|_| PhysicalOperator::Empty(EmptyOp::new()))
        };
        Ok(PhysicalOperator::OptionalExpand(
            OptionalExpandOp::new_with_join_pairs(left, right_schema, join_pairs, factory),
        ))
    }

    /// F2 (PE-1 §F2) — pipelined anchor-seeded Expand fast-path for the
    /// inner-join traversal shape. Returns `Some(op)` when `j` matches the
    /// tightly-guarded shape below, `None` (caller falls through to the DP
    /// hash/merge join, so genuinely bushy patterns are untouched)
    /// otherwise.
    ///
    /// # The shape (post-enumeration)
    ///
    /// The M4-52 join enumerator reorders the naive left-deep lowering, so
    /// F2 keys on the ENUMERATED shape (empirically confirmed via
    /// `enumerate_join_order` in the `f2_pipelined_expand` tests):
    ///
    /// - **from-seeded expand** — `Join(A, B, [k])` where ONE side is a
    ///   bare `Expand(e)` with `k == e.from` (the enumerator places it on
    ///   the left for the single-pattern anchored form) and the OTHER side
    ///   binds `k`. → `Expand(child = build(other), e)`.
    /// - **to-label-folded expand** — the expand-bearing side may instead
    ///   be a to-label semi-join `Join(Scan(to, Some(label)), Expand(e),
    ///   [e.to])` (the enumerator's shape for a `(b:Label)` tail node). The
    ///   `(to:Label)` semi-join folds into a per-edge `edge.dst.label ==
    ///   label` filter on the pipelined Expand. See
    ///   [`Self::match_bare_or_labeled_expand`].
    ///
    /// The recursion composes: `build(other)` re-enters this arm, so a
    /// two-hop `(z)-[]->(a)-[]->(b)` folds to `Expand(Expand(Scan(z)))`.
    ///
    /// # Multiset-identity contract (LOAD-BEARING — openCypher bag
    /// semantics)
    ///
    /// The DP hash join over `Join(X, Y, [k])` emits, for each edge the
    /// independent (implicitly full-scan-fed) Expand produces, one output
    /// per row of the other side sharing the join key `k`. The pipelined
    /// Expand emits the SAME multiset:
    ///
    /// - **from-seed** — for each `other`-row carrying `f`, the pipelined
    ///   ExpandOp expands `f`'s edges, emitting one row per edge; the hash
    ///   join emits `[f, r?, t]` once per `other`-row whose join value is
    ///   `f` for each edge `f→t`. Both yield exactly `(#other rows with f)
    ///   × (#edges f→t)` copies of each `[f, r?, t]` — same cardinality,
    ///   same duplicates, order-independent. Edges from `f`-values absent
    ///   from `other` are generated by the independent Expand but dropped
    ///   by the join (no matching bucket); the pipelined form never
    ///   generates them. `rel_type` / `direction` / `length_range` ride
    ///   through the SAME `ExpandOp`, so per-edge behavior is identical.
    /// - **to-label fold** — `Scan(to, Some(label))` yields each labeled
    ///   node EXACTLY once (unique key), so its equi-join on `to` is a
    ///   SEMI-join == a FILTER keeping inner rows whose `to` node carries
    ///   `label`. The per-edge `edge.dst.label == Some(label)` check drops
    ///   exactly those rows: production AND the stub materialize
    ///   `edge.dst.label` from the SAME `read_node` label the scan filters
    ///   on (`substrate.rs`: expand's `far_label` vs scan's `rec.label_id
    ///   == filter.raw()`). Single-hop only (guarded on
    ///   `length_range.is_none()`).
    ///
    /// Intermediate COLUMN ORDER may differ from the hash join (e.g. the
    /// to-label form's inner join yields `[to, from, r?]` while the
    /// pipelined Expand yields `[from, r?, to]`), but every consumer
    /// resolves bindings by id-in-schema and the terminal `Project`
    /// normalizes to RETURN order, so the final result bag is identical.
    /// The `f2_pipelined_expand` A/B tests pin this against the hash-join
    /// reference (F2 toggled off) for the full result.
    fn try_pipelined_expand(
        j: &LogicalJoin,
        parameters: &Parameters,
    ) -> Result<Option<PhysicalOperator>, ExecutionError> {
        if !pipelined_expand_enabled() {
            return Ok(None);
        }
        let JoinCondition::SharedBindings(shared) = &j.on;
        // Exactly one shared binding — the Expand's `from`. Cartesian
        // (empty) and multi-key joins fall through to the hash join.
        let [k] = shared.as_slice() else {
            return Ok(None);
        };
        let k = *k;
        // Try each side as the expand-bearing side; the other seeds `from`.
        for (exp_side, from_source) in [
            (j.left.as_ref(), j.right.as_ref()),
            (j.right.as_ref(), j.left.as_ref()),
        ] {
            if let Some((e, to_label)) = Self::match_bare_or_labeled_expand(exp_side) {
                if k == e.from {
                    if let Some(op) =
                        Self::build_pipelined_expand(from_source, e, to_label, parameters)?
                    {
                        return Ok(Some(op));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Recognize the expand-bearing side of an F2 join: either a BARE
    /// `Expand(e)` (no to-label fold), or a to-label semi-join
    /// `Join(Scan(to, Some(label)), Expand(e), [e.to])` (in either child
    /// order) whose `Expand`'s `to` is the labeled, singly-shared scan
    /// binding — folding `(to:Label)` into a per-edge filter. Returns the
    /// inner `Expand` + the optional folded to-label. The to-label form is
    /// single-hop only (`length_range.is_none()`); a var-length far-end
    /// label is NOT a single-hop `edge.dst` check, so it stays on the hash
    /// join.
    fn match_bare_or_labeled_expand(
        plan: &LogicalPlan,
    ) -> Option<(&LogicalExpand, Option<arcgraph_core::LabelId>)> {
        match plan {
            LogicalPlan::Expand(e) => Some((e, None)),
            LogicalPlan::Join(inner) => {
                let JoinCondition::SharedBindings(inner_shared) = &inner.on;
                let [to] = inner_shared.as_slice() else {
                    return None;
                };
                let to = *to;
                for (scan_side, exp_side) in [
                    (inner.left.as_ref(), inner.right.as_ref()),
                    (inner.right.as_ref(), inner.left.as_ref()),
                ] {
                    if let (LogicalPlan::Scan(s), LogicalPlan::Expand(e)) = (scan_side, exp_side) {
                        if let Some(label) = s.label {
                            if s.var == to && e.to == to && e.length_range.is_none() {
                                return Some((e, Some(label)));
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Build the pipelined [`ExpandOp`] for F2: build the `from`-source
    /// child, then verify the multiset-preserving schema invariants BEFORE
    /// committing to the fast path. Returns `None` (caller falls through to
    /// the hash join) when an invariant fails, so a shape we cannot prove
    /// multiset-identical NEVER takes the pipelined path.
    fn build_pipelined_expand(
        from_source: &LogicalPlan,
        e: &LogicalExpand,
        to_label: Option<arcgraph_core::LabelId>,
        parameters: &Parameters,
    ) -> Result<Option<PhysicalOperator>, ExecutionError> {
        let child = Self::build_with_parameters(from_source, parameters)?;
        let child_schema: Vec<BindingId> = child.schema().to_vec();
        // INVARIANT 1: the Expand's `from` MUST be bound by the child (so
        // the pipelined Expand reads each row's `from` node) — this is
        // exactly the shared binding the hash join keyed on. Guaranteed by
        // the `k == e.from` guard + a valid equi-join binding both sides,
        // but re-checked so the ExpandOp `find_child_index` never faults.
        if !child_schema.contains(&e.from) {
            return Ok(None);
        }
        // INVARIANT 2: the Expand's appended columns (`to`, and `rel_var`
        // if present) MUST be FRESH wrt the child schema. The hash join
        // appends `right_schema \ left_schema`; if `to` / `rel_var`
        // already appeared in the child, the hash join would NOT re-append
        // it (it becomes a shared/absorbed column) whereas ExpandOp always
        // appends — a schema + multiset divergence. Fall through to the
        // hash join for that degenerate shape (e.g. a self-loop
        // `(a)-[r]->(a)` where `to == from`).
        if child_schema.contains(&e.to) {
            return Ok(None);
        }
        if let Some(rv) = e.rel_var {
            if child_schema.contains(&rv) {
                return Ok(None);
            }
        }
        // The LogicalExpand at M4-31 carries no MVCC read_lsn; the executor
        // uses `Lsn::MAX` (read-latest sentinel) — mirrors the bare-Expand
        // build arm + `build_right_with_singleton_root`.
        let read_lsn = arcgraph_core::Lsn::MAX;
        let mut exp = ExpandOp::new(
            child,
            e.from,
            e.rel_var,
            e.to,
            e.rel_type,
            e.direction,
            e.length_range.clone(),
            read_lsn,
        )?;
        if let Some(label) = to_label {
            exp = exp.with_to_label(label);
        }
        Ok(Some(PhysicalOperator::Expand(exp)))
    }

    /// Test + rollback knob (per-thread): enable/disable the F2 pipelined-
    /// expand rewrite, returning the previous value. Production toggles via
    /// the `ARCGRAPH_DISABLE_PIPELINED_EXPAND` env var (read once per
    /// thread at first build); this thread-local override lets the A/B
    /// multiset tests run the hash-join reference path in-process without a
    /// global race across parallel test threads.
    #[doc(hidden)]
    pub fn set_pipelined_expand_enabled(enabled: bool) -> bool {
        PIPELINED_EXPAND_ENABLED.with(|c| c.replace(enabled))
    }
}

thread_local! {
    /// F2 (PE-1 §F2) pipelined-expand enable flag, per-thread so the A/B
    /// multiset tests can run the hash-join reference path without a global
    /// race. Defaults ON unless `ARCGRAPH_DISABLE_PIPELINED_EXPAND` is set
    /// (the operational kill-switch, mirroring the `ARCGRAPH_PARALLEL_SCAN`
    /// env-flag precedent for the S4 morsel scan).
    static PIPELINED_EXPAND_ENABLED: std::cell::Cell<bool> =
        std::cell::Cell::new(pipelined_expand_env_default());
}

/// F2 default: ON unless the kill-switch env var is set to `1`/`true`.
fn pipelined_expand_env_default() -> bool {
    !matches!(
        std::env::var("ARCGRAPH_DISABLE_PIPELINED_EXPAND").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Whether the F2 pipelined-expand rewrite is active for the current
/// thread (see [`PIPELINED_EXPAND_ENABLED`]).
fn pipelined_expand_enabled() -> bool {
    PIPELINED_EXPAND_ENABLED.with(std::cell::Cell::get)
}

fn resolve_optional_join_pairs(
    left_schema: &[BindingId],
    right_schema: &[BindingId],
    shared: &[BindingId],
) -> Result<Vec<(usize, usize)>, ExecutionError> {
    let mut pairs = Vec::with_capacity(shared.len());
    for binding in shared {
        let left_idx = left_schema
            .iter()
            .position(|b| b == binding)
            .ok_or_else(|| {
                ExecutionError::Eval(format!(
                    "OptionalExpandOp: shared binding {:?} missing from left schema",
                    binding
                ))
            })?;
        let right_idx = right_schema
            .iter()
            .position(|b| b == binding)
            .ok_or_else(|| {
                ExecutionError::Eval(format!(
                    "OptionalExpandOp: shared binding {:?} missing from right schema",
                    binding
                ))
            })?;
        pairs.push((left_idx, right_idx));
    }
    Ok(pairs)
}

/// Recursively build the right-side sub-pipeline for OPTIONAL MATCH,
/// substituting the deepest leaf `Scan` (filtered by the
/// `from_binding` shared variable) with a SingletonScanOp keyed on
/// the LEFT row's `from_node`.
fn build_right_with_singleton_root(
    plan: &LogicalPlan,
    from_binding: BindingId,
    from_node: arcgraph_core::NodeId,
    parameters: &Parameters,
) -> Result<PhysicalOperator, ExecutionError> {
    match plan {
        LogicalPlan::Scan(s) => {
            // Substitute the scan with a singleton if the binding
            // matches; otherwise keep the scan.
            if s.var == from_binding {
                Ok(PhysicalOperator::Singleton(SingletonScanOp::new(
                    s.var, from_node,
                )))
            } else {
                Ok(PhysicalOperator::Scan(ScanOp::new(
                    s.var, s.label, s.read_lsn,
                )))
            }
        }
        LogicalPlan::Expand(e) => {
            // Expand has no LogicalPlan child — but it implicitly
            // sits atop a Scan over `e.from`. Synthesize that scan
            // as a SingletonScan when the binding matches the join
            // key. LogicalExpand does NOT carry an MVCC read_lsn
            // field at M4-31; the executor uses Lsn::MAX (read-
            // latest sentinel — see `executor::context` module
            // docs §"Snapshot LSN").
            let read_lsn = arcgraph_core::Lsn::MAX;
            if e.from == from_binding {
                // Forward root: the shared binding is the traversal
                // START. Root `from` at the singleton and expand in
                // source order. This is the `MATCH (a) OPTIONAL MATCH
                // (a)-[r]-(b)` idiom (LDBC IS5/IS6 shape).
                let child = PhysicalOperator::Singleton(SingletonScanOp::new(e.from, from_node));
                let exp = ExpandOp::new(
                    child,
                    e.from,
                    e.rel_var,
                    e.to,
                    e.rel_type,
                    e.direction,
                    e.length_range.clone(),
                    read_lsn,
                )?;
                Ok(PhysicalOperator::Expand(exp))
            } else if e.to == from_binding {
                // #771 reverse root: the shared binding is the Expand's
                // `to` (e.g. `OPTIONAL MATCH (c:Commit)-[:FIXES]->(i)`
                // where `i` is the LEFT-bound seed). ExpandOp can only
                // traverse FROM its child rows, so we ROOT the known
                // `to` node at the singleton and REVERSE the edge:
                // swap from/to and flip the direction. The traversal
                // then starts at this LEFT row's `i` and walks the
                // FIXES edge backward to bind the fresh `from` endpoint
                // (`c`). Without this, the only `from`-rooted option is
                // a label-free scan of ALL nodes, whose results are not
                // constrained to this row — the §2.3 silent-wrong.
                let child = PhysicalOperator::Singleton(SingletonScanOp::new(e.to, from_node));
                let exp = ExpandOp::new(
                    child,
                    e.to,
                    e.rel_var,
                    e.from,
                    e.rel_type,
                    reverse_direction(e.direction),
                    e.length_range.clone(),
                    read_lsn,
                )?;
                Ok(PhysicalOperator::Expand(exp))
            } else {
                // Neither endpoint is the shared binding (e.g. the
                // second hop of a multi-hop optional pattern, whose
                // `from` is bound by a SIBLING expand under a shared-
                // binding Join). Scan all nodes of any label; the
                // enclosing Join constrains the endpoint via its shared
                // binding. v1.0-alpha single-hop tests rarely hit this.
                let child = PhysicalOperator::Scan(ScanOp::new(e.from, None, read_lsn));
                let exp = ExpandOp::new(
                    child,
                    e.from,
                    e.rel_var,
                    e.to,
                    e.rel_type,
                    e.direction,
                    e.length_range.clone(),
                    read_lsn,
                )?;
                Ok(PhysicalOperator::Expand(exp))
            }
        }
        LogicalPlan::Join(j) => {
            // #771: the labeled-node-with-rel idiom
            // `(c:Commit)-[:FIXES]->(i)` lowers its right side to a
            // `Join(Scan(c, Commit), Expand(c→i, FIXES))` (forward) or a
            // nested `Join(Join(Scan(i), Expand(i→c)), Scan(c, Commit))`
            // (backward `(i)<-[:FIXES]-(c:Commit)`). Recurse into BOTH
            // inputs so the shared-binding leaf gets rooted at the
            // singleton (a `Scan(i)` becomes a SingletonScan; an
            // `Expand` whose `to` is the shared binding gets reverse-
            // rooted), while the fresh-node label `Scan(c, Commit)` stays
            // a full scan. The Join then intersects on its own shared
            // binding (`c`), constraining the fresh node to its label.
            //
            // Before #771 this whole variant fell through to the
            // `NotImplemented` arm below, which the per-row factory
            // SWALLOWED into all-NULL — every left-join silently wrong.
            // Mirror the algorithm dispatch of the main `LogicalPlan::
            // Join` build arm (ADR-097): `Auto` / cartesian → HashJoin.
            let left =
                build_right_with_singleton_root(&j.left, from_binding, from_node, parameters)?;
            let right =
                build_right_with_singleton_root(&j.right, from_binding, from_node, parameters)?;
            let JoinCondition::SharedBindings(shared) = &j.on;
            let use_merge = matches!(j.algorithm, JoinAlgorithm::MergeJoin) && !shared.is_empty();
            if use_merge {
                Ok(PhysicalOperator::MergeJoin(MergeJoinOp::new(
                    left,
                    right,
                    shared.clone(),
                )?))
            } else {
                Ok(PhysicalOperator::HashJoin(HashJoinOp::new(
                    left,
                    right,
                    shared.clone(),
                )?))
            }
        }
        LogicalPlan::Filter(f) => {
            let child =
                build_right_with_singleton_root(&f.input, from_binding, from_node, parameters)?;
            Ok(PhysicalOperator::Filter(
                FilterOp::new(child, f.predicate.clone()).with_parameters(parameters.clone()),
            ))
        }
        LogicalPlan::Project(p) => {
            let child =
                build_right_with_singleton_root(&p.input, from_binding, from_node, parameters)?;
            Ok(PhysicalOperator::Project(
                ProjectOp::new(child, p.items.clone()).with_parameters(parameters.clone()),
            ))
        }
        // A NAMED PATH (`OPTIONAL MATCH p = (a)-[r]->()`) on the right side:
        // singleton-root the inner pattern subtree (so `a` roots at the
        // LEFT row's node), then re-wrap with the SAME `PlainPathOp` the
        // main build arm uses (appending `path_var` to the child schema).
        // This is the path the null-anchored shape from TCK
        // `expressions/path/Path1[1]` + `Path2[3]` takes:
        // `WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p)`.
        // For that null-anchor case the LEFT row's `a` is `Value::Null`
        // (not a node), so the per-row factory short-circuits to an
        // `EmptyOp` BEFORE this arm runs at execution time and the
        // `OptionalExpandOp` null-extends `p` to null; this arm exists so
        // the build-time probe (`NodeId::ZERO` placeholder) can derive the
        // right schema (`child ++ [p]`) and the NON-null-anchor case
        // (`MATCH (a) OPTIONAL MATCH p = (a)-[r]->()` over a real node)
        // materializes the path normally.
        //
        // Only the `Plain` (full enumeration) algorithm is supported here:
        // a `ShortestPath` / `AllShortestPaths` named path on an OPTIONAL
        // right side stays `NotImplemented` (no TCK scenario exercises it
        // and the BFS-re-traversal seam is out of scope for this slice).
        LogicalPlan::NamedPath(np) if matches!(np.algorithm, PathAlgorithm::Plain) => {
            let child =
                build_right_with_singleton_root(&np.input, from_binding, from_node, parameters)?;
            let shape = np.plain_shape.clone().ok_or_else(|| {
                ExecutionError::Eval(
                    "NamedPath::Plain missing plain_shape (lowering invariant — ADR-193 D-4)"
                        .into(),
                )
            })?;
            Ok(PhysicalOperator::PlainPath(PlainPathOp::new(
                child,
                shape,
                np.path_var,
            )))
        }
        // The M4-32 lowering produces Scan / Expand / Filter / Project /
        // Join (+ a Plain NamedPath, above) on the OPTIONAL MATCH right
        // side. Any OTHER shape surfaces `NotImplemented` LOUDLY — #771:
        // `build_optional_expand`'s build-time probe propagates this error
        // so the client sees an honest "not supported" rather than the
        // prior swallowed all-NULL (correct-or-loud, never silent-wrong).
        _ => Err(ExecutionError::NotImplemented {
            feature: format!(
                "OPTIONAL MATCH right-side has plan variant {} (only Scan/Expand/Filter/Project/Join supported)",
                plan_variant(plan)
            ),
            target_slice: "M4-63 / M4-72".into(),
            section: "ADR-006 amendment-01 §A-2".into(),
        }),
    }
}

/// Reverse a traversal [`Direction`] for #771's reverse-rooted Expand.
///
/// When an OPTIONAL MATCH pattern's shared (LEFT-bound) binding is the
/// Expand's `to` endpoint, [`build_right_with_singleton_root`] roots the
/// `to` node at the singleton and swaps `from`/`to`; the edge must then
/// be walked in the opposite sense, so `LeftToRight` ⇄ `RightToLeft`.
/// `Undirected` is its own reverse.
fn reverse_direction(d: crate::logical_plan::Direction) -> crate::logical_plan::Direction {
    use crate::logical_plan::Direction;
    match d {
        Direction::LeftToRight => Direction::RightToLeft,
        Direction::RightToLeft => Direction::LeftToRight,
        Direction::Undirected => Direction::Undirected,
    }
}

/// Render a [`LogicalPlan`] variant's name for diagnostic output.
fn plan_variant(plan: &LogicalPlan) -> &'static str {
    match plan {
        LogicalPlan::Scan(_) => "Scan",
        LogicalPlan::PropertyIndexScan(_) => "PropertyIndexScan",
        LogicalPlan::CountStore(_) => "CountStore",
        LogicalPlan::Expand(_) => "Expand",
        LogicalPlan::Filter(_) => "Filter",
        LogicalPlan::Project(_) => "Project",
        LogicalPlan::Join(_) => "Join",
        LogicalPlan::LeftOuterJoin(_) => "LeftOuterJoin",
        LogicalPlan::Limit(_) => "Limit",
        LogicalPlan::Skip(_) => "Skip",
        LogicalPlan::RankByHybrid(_) => "RankByHybrid",
        LogicalPlan::Fusion(_) => "Fusion",
        LogicalPlan::CommunityLookup(_) => "CommunityLookup",
        LogicalPlan::VectorNear(_) => "VectorNear",
        LogicalPlan::TextMatch(_) => "TextMatch",
        LogicalPlan::Aggregate(_) => "Aggregate",
        LogicalPlan::Sort(_) => "Sort",
        LogicalPlan::Distinct(_) => "Distinct",
        LogicalPlan::Union(_) => "Union",
        LogicalPlan::Unwind(_) => "Unwind",
        LogicalPlan::ProcedureCall(_) => "ProcedureCall",
        LogicalPlan::NamedPath(_) => "NamedPath",
        LogicalPlan::DynamicLimit(_) => "DynamicLimit",
        LogicalPlan::CreateNode(_) => "CreateNode",
        LogicalPlan::CreateVectorIndex(_) => "CreateVectorIndex",
        LogicalPlan::CreatePropertyIndex(_) => "CreatePropertyIndex",
        LogicalPlan::CreateRel(_) => "CreateRel",
        LogicalPlan::Delete(_) => "Delete",
        LogicalPlan::Set(_) => "Set",
        LogicalPlan::Remove(_) => "Remove",
        LogicalPlan::Merge(_) => "Merge",
        LogicalPlan::Call(_) => "Call",
        LogicalPlan::CorrelationSeed(_) => "CorrelationSeed",
        LogicalPlan::Empty(_) => "Empty",
    }
}

fn default_read_lsn(operands: &[HybridOperand]) -> arcgraph_core::Lsn {
    operands
        .first()
        .map(|o| o.read_lsn)
        .unwrap_or(arcgraph_core::Lsn::MAX)
}

fn dynamic_limit_clause_name(kind: DynamicLimitKind) -> &'static str {
    match kind {
        DynamicLimitKind::Limit => "LIMIT",
        DynamicLimitKind::Skip => "SKIP",
    }
}

fn dynamic_limit_value_type(value: &Value) -> &'static str {
    match value {
        Value::Integer(_) => "integer",
        Value::Float(_) => "float",
        Value::String(_) => "string",
        Value::Boolean(_) => "boolean",
        Value::List(_) => "list",
        Value::Map(_) => "map",
        Value::Node(_) => "node",
        Value::Relationship(_) => "relationship",
        Value::Path(_) => "path",
        Value::Temporal(_) => "datetime",
        Value::LocalDateTime(_) => "localdatetime",
        Value::Date(_) => "date",
        Value::Duration(_) => "duration",
        Value::Decimal(_) => "decimal",
        Value::Null => "null",
    }
}

/// Terminal-vs-stacked discriminator for a write-op under a ROW-CONSUMER
/// parent (#709 chained writes + #772 RETURN/WITH/aggregate/UNWIND).
///
/// A [`PhysicalOperator::Set`] / [`PhysicalOperator::Remove`] defaults to
/// **terminal**: with no consumer above it (the pipeline root, i.e. a
/// RETURN-less `SET …` / `REMOVE …`) it DRAINS its input rows and emits 0
/// result rows — the openCypher v9 / ADR-149/150 §D / ADR-182 terminal-write
/// contract. This helper flips it to **stacked** (pass its mutated rows
/// through) when, and only when, it is wired as the `input` of a ROW-CONSUMER
/// parent that needs those rows:
///
/// - Another write-op (#709) — the inner clause of `SET … SET …` /
///   `SET … REMOVE …` / `REMOVE … REMOVE …`; the chain must compose.
/// - A [`LogicalPlan::Project`] (#772) — `SET … RETURN …` / `… WITH …`
///   (the RETURN/WITH projection over the write-op).
/// - A [`LogicalPlan::Aggregate`] (#772) — `SET … RETURN count(a)` /
///   `sum(a.x)` / `… WITH <agg> …`; the lowering splices the Aggregate
///   BETWEEN the Project and the write-op (`Project(Aggregate(Set(…)))`),
///   so the Aggregate — not the Project — is the write-op's direct parent
///   and must do the flip (else the terminal SET drains and the aggregate
///   folds over 0 rows → `count(a)=0` / `sum=NULL`, a silent-wrong).
/// - A [`LogicalPlan::Unwind`] (#772) — `SET … UNWIND … RETURN …`.
///
/// `ORDER BY` / `DISTINCT` / `LIMIT` wrap a `Project`, so the Project flip
/// reaches the write-op through them — they need no flip of their own.
///
/// Called by the [`LogicalPlan::Set`] / [`LogicalPlan::Remove`] build arms
/// (on their just-built write-op `input_op`) AND by the
/// [`LogicalPlan::Project`] / [`LogicalPlan::Aggregate`] /
/// [`LogicalPlan::Unwind`] build arms (on their just-built child). A
/// non-write-op input is left untouched (so a row-consumer over a plain
/// scan/expand/etc. is unaffected). The signal is build-time + local (a
/// direct row-consumer→write-op parent edge), matching the
/// `Project(Set(Scan))` / `Aggregate(Set(Scan))` / `Set(Set(Scan))` nesting
/// the lowering produces (see `LogicalPlanLoweringVisitor`).
fn mark_writeop_input_stacked(input_op: &mut PhysicalOperator) {
    match input_op {
        PhysicalOperator::Set(op) => op.mark_stacked(),
        PhysicalOperator::Remove(op) => op.mark_stacked(),
        _ => {}
    }
}

#[cfg(test)]
mod f2_guard_tests {
    //! F2 (PE-1 §F2) decision guards. The end-to-end multiset + plan-shape
    //! pins live in `tests/f2_pipelined_expand_e2e.rs`; these unit tests
    //! pin the DEGENERATE-shape fallthroughs the e2e enumerated plans do
    //! not reach — a fallthrough here is a correctness guard (F2 declines a
    //! shape it cannot prove multiset-identical and takes the hash join).

    use super::*;
    use crate::error::Span;
    use crate::logical_plan::{Direction, LogicalScan};
    use crate::observer::OperatorKind;

    fn scan(var: u64, label: Option<u32>) -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: label.map(arcgraph_core::LabelId::new),
            var: BindingId::new(var),
            read_lsn: arcgraph_core::Lsn::MAX,
            span: Span::point(1, 1),
        })
    }

    fn expand(from: u64, to: u64) -> LogicalPlan {
        LogicalPlan::Expand(LogicalExpand {
            from: BindingId::new(from),
            to: BindingId::new(to),
            direction: Direction::LeftToRight,
            rel_type: Some(arcgraph_core::TypeId::new(1)),
            length_range: None,
            rel_var: None,
            span: Span::point(1, 1),
        })
    }

    fn join(left: LogicalPlan, right: LogicalPlan, shared: Vec<u64>) -> LogicalPlan {
        LogicalPlan::Join(LogicalJoin {
            left: Box::new(left),
            right: Box::new(right),
            on: JoinCondition::SharedBindings(shared.into_iter().map(BindingId::new).collect()),
            algorithm: JoinAlgorithm::HashJoin,
            span: Span::point(1, 1),
        })
    }

    #[test]
    fn from_seeded_join_folds_to_expand() {
        // `Join(Scan(a,User), Expand(a→b), [a])` → pipelined Expand.
        let plan = join(scan(0, Some(1)), expand(0, 1), vec![0]);
        let phys = Pipeline::build(&plan).expect("build");
        assert_eq!(phys.op_kind(), OperatorKind::Expand);
    }

    #[test]
    fn from_seeded_join_reverts_to_hashjoin_when_disabled() {
        // RED-on-revert via the kill-switch: same plan, F2 off → HashJoin.
        let prev = Pipeline::set_pipelined_expand_enabled(false);
        let plan = join(scan(0, Some(1)), expand(0, 1), vec![0]);
        let phys = Pipeline::build(&plan).expect("build");
        Pipeline::set_pipelined_expand_enabled(prev);
        assert_eq!(phys.op_kind(), OperatorKind::HashJoin);
    }

    #[test]
    fn self_loop_to_equals_from_falls_through() {
        // `(a)-[r]->(a)`: `to == from` fails INVARIANT 2 (the appended
        // `to` column is NOT fresh wrt the child schema), so F2 declines
        // and the hash join stands.
        let plan = join(scan(0, Some(1)), expand(0, 0), vec![0]);
        let phys = Pipeline::build(&plan).expect("build");
        assert_eq!(phys.op_kind(), OperatorKind::HashJoin);
    }

    #[test]
    fn cartesian_join_falls_through() {
        // Empty shared bindings (Cartesian) — not the single-`[from]`
        // shape; F2 declines.
        let plan = join(scan(0, Some(1)), scan(1, Some(1)), vec![]);
        let phys = Pipeline::build(&plan).expect("build");
        assert_eq!(phys.op_kind(), OperatorKind::HashJoin);
    }

    #[test]
    fn to_label_semi_join_folds_with_label() {
        // `Join(Scan(b,User), Expand(a→b), [b])` → to-label-folded Expand.
        // (The enumerator's inner shape for a `(b:Label)` tail node.)
        let plan = join(scan(1, Some(1)), expand(0, 1), vec![1]);
        // shared is [b]=e.to, so this side alone is NOT from-seeded — but
        // `match_bare_or_labeled_expand` recognizes it as a to-label wrap
        // only when it is the EXPAND-BEARING side of an OUTER from-join.
        // Standalone, the outer from-seed is absent, so it stays a join.
        let phys = Pipeline::build(&plan).expect("build");
        assert_eq!(
            phys.op_kind(),
            OperatorKind::HashJoin,
            "a bare to-label semi-join with no enclosing from-seed is not folded standalone"
        );
    }
}

#[cfg(test)]
mod tests {
    //! ADR-226 rc-GATE **T4** — the flag-OFF safety-switch test.
    //!
    //! Pins the planner's **byte-identical-revert** invariant, which the
    //! combined-concurrency ultracode verdict (§4 T4 + §5) named as one of
    //! the two "SOUND-resting-on-zero-tests" load-bearing facts that MUST
    //! land before the rc tag:
    //!
    //! - `ARCGRAPH_PARALLEL_SCAN` **UNSET** ⇒ the planner builds the
    //!   SERIAL [`PhysicalOperator::Scan`] / [`PhysicalOperator::Aggregate`]
    //!   (the operator's safety-switch revert path).
    //! - `ARCGRAPH_PARALLEL_SCAN=1` ⇒ the planner builds
    //!   [`PhysicalOperator::ParallelScan`] /
    //!   [`PhysicalOperator::ParallelAggregate`] (when mergeable).
    //!
    //! Before this test the claim was inspection-only (the three flag sites
    //! at pipeline.rs:63 bare-`Scan`, :96 `Filter(Scan)`-fold, and :298
    //! `Aggregate`). A pure planner-SHAPE assertion — no execution needed,
    //! fast + deterministic.

    use arcgraph_core::{LabelId, Lsn};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::ops::PhysicalOperator;
    use crate::logical_plan::{
        AggregationKind, AggregationSpec, LogicalAggregate, LogicalFilter, LogicalPlan, LogicalScan,
    };
    use crate::semantic::bound_ast::{
        BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind,
    };

    /// Env var the planner reads to choose parallel vs serial (the shared
    /// S4/S5 flag). Literal here so the test pins the exact wire name the
    /// operators + ops both read.
    const ENV_PARALLEL_SCAN: &str = "ARCGRAPH_PARALLEL_SCAN";

    /// Serialize env-mutating tests (Rust runs `#[test]`s on shared process
    /// threads; these poke the process-global `ARCGRAPH_PARALLEL_SCAN`).
    /// Mirrors the `ENV_TEST_LOCK` pattern the `parallel_scan` /
    /// `parallel_aggregate` unit tests use.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn span() -> Span {
        Span::point(1, 1)
    }

    /// A bare `MATCH (n:Label) RETURN n` scan plan.
    fn scan_plan() -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    }

    /// A `Filter(Scan)` plan — `MATCH (n:Label) WHERE <lit> RETURN n` —
    /// exactly the shape the pipeline.rs:96 fold site inspects. The
    /// predicate is a trivial `true` literal (its VALUE is irrelevant to
    /// the planner-shape decision; only the `Filter`-directly-over-`Scan`
    /// STRUCTURE gates the fold).
    fn filter_over_scan_plan() -> LogicalPlan {
        LogicalPlan::Filter(LogicalFilter {
            input: Box::new(scan_plan()),
            predicate: BoundExpression::Literal {
                value: Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        })
    }

    /// A `Filter(Aggregate)` plan — a `Filter` whose child is NOT a bare
    /// `Scan`, so the pipeline.rs:96 fold does NOT apply and the serial
    /// `FilterOp` wrapping is kept even with the flag ON.
    fn filter_over_nonscan_plan() -> LogicalPlan {
        LogicalPlan::Filter(LogicalFilter {
            input: Box::new(mergeable_aggregate_plan()),
            predicate: BoundExpression::Literal {
                value: Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        })
    }

    /// A MERGEABLE aggregate plan: `MATCH (n:Label) RETURN count(n)` — no
    /// GROUP BY, no DISTINCT, no COLLECT, so
    /// [`crate::executor::ops::ParallelAggregateOp::is_mergeable`] is true
    /// and the flag-ON path builds `ParallelAggregate`.
    fn mergeable_aggregate_plan() -> LogicalPlan {
        LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(scan_plan()),
            group_by: Vec::new(),
            aggregations: vec![AggregationSpec {
                function: AggregationKind::Count,
                arg: BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: span(),
                    type_info: None,
                },
                output_id: BindingId::new(1),
                alias: Some("c".into()),
                distinct: false,
                star: false,
                span: span(),
            }],
            span: span(),
        })
    }

    /// A NON-mergeable aggregate plan: a GROUP BY key makes
    /// `is_mergeable` false, so even with the flag ON the planner builds
    /// the SERIAL `Aggregate` (the mergeability guard — verdict §4 T4
    /// "ParallelAggregate ... (when mergeable)").
    fn grouped_aggregate_plan() -> LogicalPlan {
        let group_key = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(BoundExpression::VariableRef {
                name: "n".into(),
                binding_id: BindingId::new(0),
                span: span(),
                type_info: None,
            }),
            alias: None,
            output_id: Some(BindingId::new(2)),
            source_text: None,
            span: span(),
        };
        LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(scan_plan()),
            group_by: vec![group_key],
            aggregations: vec![AggregationSpec {
                function: AggregationKind::Count,
                arg: BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: span(),
                    type_info: None,
                },
                output_id: BindingId::new(1),
                alias: Some("c".into()),
                distinct: false,
                star: false,
                span: span(),
            }],
            span: span(),
        })
    }

    /// RAII env guard: sets `ARCGRAPH_PARALLEL_SCAN` (or clears it) and
    /// restores the prior value on drop.
    ///
    /// SAFETY (edition-2024 `set_var`/`remove_var` unsafe): every env
    /// mutation happens while the caller holds [`ENV_TEST_LOCK`], so no
    /// concurrent reader of the var runs; the prior value is restored (or
    /// the var removed) before the lock is released.
    struct FlagGuard {
        prior: Option<String>,
    }

    impl FlagGuard {
        fn set(value: Option<&str>) -> Self {
            let prior = std::env::var(ENV_PARALLEL_SCAN).ok();
            // SAFETY: see the struct doc — guarded by ENV_TEST_LOCK.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(ENV_PARALLEL_SCAN, v),
                    None => std::env::remove_var(ENV_PARALLEL_SCAN),
                }
            }
            Self { prior }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            // SAFETY: see the struct doc — guarded by ENV_TEST_LOCK.
            unsafe {
                match &self.prior {
                    Some(v) => std::env::set_var(ENV_PARALLEL_SCAN, v),
                    None => std::env::remove_var(ENV_PARALLEL_SCAN),
                }
            }
        }
    }

    /// Build a plan with the flag in the requested state (guarded).
    fn build_under_flag(plan: &LogicalPlan, flag: Option<&str>) -> PhysicalOperator {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _guard = FlagGuard::set(flag);
        Pipeline::build(plan).expect("pipeline build OK")
    }

    // ── flag UNSET ⇒ SERIAL (the revert / safety-switch path) ─────────

    #[test]
    fn flag_unset_bare_scan_builds_serial_scan() {
        let op = build_under_flag(&scan_plan(), None);
        assert!(
            matches!(op, PhysicalOperator::Scan(_)),
            "flag UNSET must build the SERIAL Scan (not ParallelScan): got {op:?}"
        );
        assert!(!matches!(op, PhysicalOperator::ParallelScan(_)));
    }

    #[test]
    fn flag_unset_filter_scan_builds_serial_filter() {
        let op = build_under_flag(&filter_over_scan_plan(), None);
        // Flag off → the fold does NOT fire; the serial FilterOp wraps a
        // serial Scan child.
        match op {
            PhysicalOperator::Filter(_) => {}
            other => panic!("flag UNSET must build the SERIAL Filter(Scan): got {other:?}"),
        }
        assert!(!matches!(
            build_under_flag(&filter_over_scan_plan(), None),
            PhysicalOperator::ParallelScan(_)
        ));
    }

    #[test]
    fn flag_unset_aggregate_builds_serial_aggregate() {
        let op = build_under_flag(&mergeable_aggregate_plan(), None);
        assert!(
            matches!(op, PhysicalOperator::Aggregate(_)),
            "flag UNSET must build the SERIAL Aggregate (not ParallelAggregate): got {op:?}"
        );
        assert!(!matches!(op, PhysicalOperator::ParallelAggregate(_)));
    }

    // ── flag SET ⇒ PARALLEL (the enabled path) ────────────────────────

    #[test]
    fn flag_set_bare_scan_builds_parallel_scan() {
        let op = build_under_flag(&scan_plan(), Some("1"));
        assert!(
            matches!(op, PhysicalOperator::ParallelScan(_)),
            "flag SET must build ParallelScan: got {op:?}"
        );
    }

    #[test]
    fn flag_set_filter_scan_folds_into_parallel_scan() {
        // pipeline.rs:96 — `Filter` directly over a bare `Scan` folds the
        // WHERE predicate INTO a single `ParallelScan{predicate}` (the
        // per-morsel filter-pushdown site), NOT a Filter(ParallelScan).
        let op = build_under_flag(&filter_over_scan_plan(), Some("1"));
        assert!(
            matches!(op, PhysicalOperator::ParallelScan(_)),
            "flag SET must fold Filter(Scan) into a single ParallelScan{{predicate}}: got {op:?}"
        );
    }

    #[test]
    fn flag_set_mergeable_aggregate_builds_parallel_aggregate() {
        let op = build_under_flag(&mergeable_aggregate_plan(), Some("1"));
        assert!(
            matches!(op, PhysicalOperator::ParallelAggregate(_)),
            "flag SET + mergeable must build ParallelAggregate: got {op:?}"
        );
    }

    // ── mergeability guard: flag SET but NOT mergeable ⇒ still SERIAL ──

    #[test]
    fn flag_set_grouped_aggregate_stays_serial() {
        // A GROUP BY makes the aggregate non-mergeable; even flag ON keeps
        // the serial `Aggregate` (verdict §4 T4 "... when mergeable").
        let op = build_under_flag(&grouped_aggregate_plan(), Some("1"));
        assert!(
            matches!(op, PhysicalOperator::Aggregate(_)),
            "flag SET but non-mergeable (GROUP BY) must stay SERIAL Aggregate: got {op:?}"
        );
        assert!(!matches!(op, PhysicalOperator::ParallelAggregate(_)));
    }

    #[test]
    fn flag_set_filter_over_nonscan_keeps_serial_filter() {
        // The fold applies ONLY when the Filter's child is a bare Scan; a
        // Filter over an Aggregate keeps the serial `FilterOp` wrapping
        // even with the flag ON (pipeline.rs:96 fallback path).
        let op = build_under_flag(&filter_over_nonscan_plan(), Some("1"));
        assert!(
            matches!(op, PhysicalOperator::Filter(_)),
            "flag SET but Filter child is NOT a bare Scan → serial Filter wrapping: got {op:?}"
        );
    }

    // ── "true"/"TRUE" accepted; other values off (the flag parser) ────

    #[test]
    fn flag_true_word_builds_parallel_but_other_values_stay_serial() {
        assert!(matches!(
            build_under_flag(&scan_plan(), Some("true")),
            PhysicalOperator::ParallelScan(_)
        ));
        assert!(matches!(
            build_under_flag(&scan_plan(), Some("TRUE")),
            PhysicalOperator::ParallelScan(_)
        ));
        // Any non-truthy value is OFF (the conservative revert posture).
        assert!(matches!(
            build_under_flag(&scan_plan(), Some("0")),
            PhysicalOperator::Scan(_)
        ));
        assert!(matches!(
            build_under_flag(&scan_plan(), Some("yes")),
            PhysicalOperator::Scan(_)
        ));
    }
}
