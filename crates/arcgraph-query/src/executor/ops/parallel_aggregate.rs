//! [`ParallelAggregateOp`] — morsel-driven parallel partial aggregation
//! (ADR-226 §4 slice S5, gate **CONC-D**; reuses S4's morsel infra).
//!
//! # What
//!
//! A parallel replacement for the serial `super::AggregateOp` on the
//! **scan-aggregate** shape (no GROUP BY): `MATCH … RETURN count(*),
//! sum(a.x), avg(a.y), min(a.z), max(a.w)`. It materializes the upstream
//! rows once (like the serial op), splits them into fixed-size
//! **morsels** (S4's split), computes a **partial** aggregate per morsel
//! in parallel on S4's **dedicated** rayon pool, then does a cheap
//! single-threaded O(morsels) **merge** of the partials into the final
//! result. The result is **identical** to the serial aggregate — by
//! construction, because both paths fold through the SAME
//! `super::aggregate::Accumulator` and the merge is that accumulator's
//! mergeable-decomposition (see `Accumulator::merge`).
//!
//! # Evidence
//!
//! Partial (a.k.a. two-phase / decomposable) aggregation is the standard
//! parallel-aggregate technique: each worker computes a partial
//! aggregate over its input partition, then a serial combiner merges the
//! partials — Graefe, *"Query Evaluation Techniques for Large
//! Databases"*, ACM Computing Surveys 1993 §8 (aggregation);
//! Leis, Boncz, Kemper, Neumann, *"Morsel-Driven Parallelism"*, SIGMOD
//! 2014 §5 (morsel-local aggregation + a final merge). The decomposition
//! per aggregate — COUNT/SUM add, MIN/MAX take the extreme, and **AVG
//! carries `(sum, count)` (NOT partial means)** — is the classic
//! "algebraic / distributive" aggregate taxonomy (Gray et al., *"Data
//! Cube"*, ICDE 1996 §3: distributive = COUNT/SUM/MIN/MAX; algebraic =
//! AVG via the `(sum, count)` pair). DISTINCT is holistic (needs a global
//! set) → NOT partial-mergeable → excluded at rc.
//!
//! # Back-of-envelope (ADR-226 §4 CONC-D target)
//!
//! Merge cost is O(morsels), NOT O(rows): a 10M-row aggregate over
//! ⌈10M / 64K⌉ ≈ 153 morsels does ~153 constant-time
//! `Accumulator::merge` calls (a few adds / compares each) — nanoseconds
//! total, ≪ the ~2 s serial scan-and-fold. So the parallel fold (the
//! O(rows) work, split `cores − 4` ways on S4's pool) dominates and the
//! merge is free: end-to-end tracks the scan-aggregate ≥ 4× target of
//! S4. Below `resolve_row_threshold` the fan-out overhead dominates →
//! the small-input guard keeps small aggregates serial.
//!
//! # Mergeable scope + serial fallback (ADR-226 §4 line 347)
//!
//! The parallel path runs ONLY when the aggregate is provably mergeable:
//!
//! - **No GROUP BY** — the mergeable scan-aggregate shape (ADR-226 line
//!   282). A grouped aggregate needs per-morsel hash tables + a keyed
//!   merge (a bigger lift, deferred) → falls back to serial.
//! - **No DISTINCT** — `count(DISTINCT x)` needs a GLOBAL dedup set; a
//!   union of per-morsel sets is not a simple partial-merge, so DISTINCT
//!   is **excluded at rc** (ADR-226 §4 line 347) → serial.
//! - **No COLLECT** — order/duplicate-sensitive + carries per-fold memory
//!   budget reservation semantics the serial op handles specially;
//!   conservatively kept serial at rc.
//!
//! [`ParallelAggregateOp::is_mergeable`] is the predicate; the pipeline
//! ([`crate::executor::pipeline`]) consults it + `Self::enabled_by_env`
//! (the SHARED `ARCGRAPH_PARALLEL_SCAN` flag, via
//! `super::ParallelScanOp::enabled_by_env`) to choose this op vs the
//! serial `super::AggregateOp`. Flag off OR not mergeable → serial,
//! byte-identical. Default off at rc (the conservative revert posture).
//!
//! # Correctness (the #1 gate)
//!
//! Every morsel folds its rows through `Accumulator::fold` — the EXACT
//! serial fold (same NULL exclusion: COUNT(expr) skips NULL, COUNT(*)
//! counts rows, SUM/AVG/MIN/MAX ignore NULL, SUM/AVG are NULL when
//! all-NULL). The merge combines partials through `Accumulator::merge`
//! (COUNT/SUM add, MIN/MAX extreme, **AVG carries `(sum, count)` →
//! `Σsum/Σcount`**). Because these folds+merges are associative and the
//! morsels are a disjoint, total cover of the input, folding a partition
//! then merging yields the SAME accumulator state as one serial fold over
//! the whole input. The headline equivalence proptest pins
//! `parallel ≡ serial` for COUNT/COUNT(*)/SUM/AVG/MIN/MAX on random data
//! (incl. NULL / empty / all-NULL), forcing tiny morsels to exercise the
//! multi-morsel merge path.
//!
//! # ADR provenance
//! - **ADR-226 §4 slice S5 / gate CONC-D** — parallel partial
//!   aggregation; per-morsel partials + single-threaded merge; reuses
//!   S4's morsel infra (same dedicated pool, same split, same flag +
//!   threshold); DISTINCT excluded at rc.
//! - **S4 (`super::parallel_scan`)** — the reused morsel infrastructure.
//! - **Gray et al. ICDE 1996 / Leis et al. SIGMOD 2014** — decomposable
//!   aggregation evidence.

use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::aggregate::Accumulator;
use crate::executor::ops::parallel_scan::{resolve_morsel_size, resolve_row_threshold, scan_pool};
use crate::executor::ops::{AggregateCall, PhysicalOperator, schema_index};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::AggregationKind;
use crate::semantic::bound_ast::{BindingId, BoundProjectionItem};

/// Morsel-driven parallel partial-aggregate operator (no GROUP BY).
///
/// Buffers all upstream rows at first-batch, folds each morsel to a
/// partial `Accumulator` set in parallel on S4's dedicated pool, merges
/// the partials single-threaded, and emits the ONE finalized result row.
/// Produces a byte-identical result to the serial [`super::AggregateOp`].
#[derive(Debug)]
pub struct ParallelAggregateOp {
    /// Upstream operator (materialized once at first-batch).
    child: Box<PhysicalOperator>,
    /// Aggregate calls (all provably mergeable — see [`Self::is_mergeable`]).
    aggregations: Vec<AggregateCall>,
    /// Per-query parameter bag (forwarded to arg evaluation).
    parameters: Parameters,
    /// Cached child schema (for arg evaluation's binding lookup).
    child_schema: Vec<BindingId>,
    /// Output schema: one slot per aggregation (no group-by columns).
    schema: Vec<BindingId>,
    /// Cached output rows after the fold+merge runs once (`Some` after
    /// first-batch). Always exactly ONE row (no GROUP BY).
    output_rows: Option<Vec<Vec<Value>>>,
    /// Output cursor.
    cursor: usize,
}

impl ParallelAggregateOp {
    /// Construct a `ParallelAggregateOp`.
    ///
    /// # Preconditions
    ///
    /// The caller (pipeline) MUST have checked [`Self::is_mergeable`] on
    /// `group_by` + `aggregations` first — this op handles ONLY the
    /// no-GROUP-BY, no-DISTINCT, no-COLLECT mergeable shape. The output
    /// schema is `[aggregation output ids…]` (no leading group-by
    /// columns), matching the serial op's schema for an empty `group_by`.
    #[must_use]
    pub fn new(child: PhysicalOperator, aggregations: Vec<AggregateCall>) -> Self {
        let child_schema = child.schema().to_vec();
        let schema: Vec<BindingId> = aggregations.iter().map(|c| c.output_id).collect();
        Self {
            child: Box::new(child),
            aggregations,
            parameters: Parameters::new(),
            child_schema,
            schema,
            output_rows: None,
            cursor: 0,
        }
    }

    /// Inject a per-query parameter bag.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema (one slot per aggregation; no group-by columns).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Whether this aggregate is provably mergeable → eligible for the
    /// parallel partial path (ADR-226 §4 line 347). Requires: (a) no
    /// GROUP BY, (b) no DISTINCT aggregate, (c) no COLLECT aggregate. Any
    /// of these → the pipeline builds the serial [`super::AggregateOp`]
    /// instead (byte-identical result, serial execution).
    #[must_use]
    pub fn is_mergeable(group_by: &[BoundProjectionItem], aggregations: &[AggregateCall]) -> bool {
        group_by.is_empty()
            && aggregations
                .iter()
                .all(|c| !c.distinct && !matches!(c.kind, AggregationKind::Collect))
    }

    /// Whether the parallel path is enabled — reuses S4's SHARED
    /// `ARCGRAPH_PARALLEL_SCAN` flag (aggregate fan-out shares the scan's
    /// dedicated pool + flag, so one revert switch governs both). Default
    /// off at rc.
    #[must_use]
    pub fn enabled_by_env() -> bool {
        super::ParallelScanOp::enabled_by_env()
    }

    /// Fold ONE morsel (a contiguous slice of buffered rows) into a fresh
    /// partial [`Accumulator`] set — the per-morsel body that runs in
    /// parallel. Evaluates each aggregation's arg (or a non-NULL sentinel
    /// for `count(*)`) per row and folds it through the EXACT serial
    /// [`Accumulator::fold`], so a morsel's partial ≡ the serial fold over
    /// just that morsel's rows.
    fn fold_morsel(
        aggregations: &[AggregateCall],
        child_schema: &[BindingId],
        params: &Parameters,
        rows: &[Vec<Value>],
    ) -> Result<Vec<Accumulator>, ExecutionError> {
        let mut accs: Vec<Accumulator> = aggregations
            .iter()
            .map(|c| Accumulator::empty(c.kind))
            .collect();
        let lookup = |b: BindingId| schema_index(child_schema, b);
        for row in rows {
            for (i, call) in aggregations.iter().enumerate() {
                // #773 G4 — `count(*)` (star): fold a non-NULL sentinel so
                // every ROW is counted (incl. all-NULL rows), matching the
                // serial materialize loop; otherwise evaluate the arg.
                let v = if call.star {
                    Value::Integer(1)
                } else {
                    evaluate(&call.arg, row, &lookup, params)?
                };
                accs[i].fold(v)?;
            }
        }
        Ok(accs)
    }

    /// Merge per-morsel partial accumulator sets into one final set
    /// (single-threaded, O(morsels)). Seeds from the first partial and
    /// folds the rest in via [`Accumulator::merge`] (COUNT/SUM add,
    /// MIN/MAX extreme, AVG carries `(sum, count)`). An empty `partials`
    /// (zero input rows) returns fresh empties → the single-row
    /// empty-aggregate contract (count=0, sum=NULL, …).
    fn merge_partials(
        &self,
        partials: Vec<Vec<Accumulator>>,
    ) -> Result<Vec<Accumulator>, ExecutionError> {
        let mut iter = partials.into_iter();
        let mut merged: Vec<Accumulator> = match iter.next() {
            Some(first) => first,
            None => {
                return Ok(self
                    .aggregations
                    .iter()
                    .map(|c| Accumulator::empty(c.kind))
                    .collect());
            }
        };
        for partial in iter {
            for (m, p) in merged.iter_mut().zip(partial) {
                m.merge(p)?;
            }
        }
        Ok(merged)
    }

    /// Drain the upstream, fold morsels (parallel), merge (serial), and
    /// stash the single finalized output row.
    fn materialize<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        // Drain all upstream rows (the aggregate is blocking — it cannot
        // emit until it has seen every row, same as the serial op).
        let mut rows: Vec<Vec<Value>> = Vec::new();
        loop {
            ctx.cancellation().check()?;
            let batch = self.child.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            rows.extend(batch.into_rows());
        }
        // Re-check cancellation after the (potentially large) drain, before
        // fanning the fold onto the pool.
        ctx.cancellation().check()?;

        let partials = self.fold_all(&rows)?;
        let merged = self.merge_partials(partials)?;
        let out_row: Vec<Value> = merged.into_iter().map(Accumulator::finalize).collect();
        self.output_rows = Some(vec![out_row]);
        Ok(())
    }

    /// Produce the per-morsel partials for `rows`. Splits into contiguous
    /// morsels + folds each in parallel on S4's dedicated pool. Falls back
    /// to a single serial fold when (a) below the small-input threshold or
    /// (b) the dedicated pool could not be built — either way the merged
    /// RESULT is identical, only the execution strategy differs.
    fn fold_all(&self, rows: &[Vec<Value>]) -> Result<Vec<Vec<Accumulator>>, ExecutionError> {
        let threshold = resolve_row_threshold();
        if rows.len() < threshold {
            // Small-input guard: one serial morsel.
            return Ok(vec![Self::fold_morsel(
                &self.aggregations,
                &self.child_schema,
                &self.parameters,
                rows,
            )?]);
        }
        let Some(pool) = scan_pool() else {
            // Pool build failed (thread exhaustion) — degrade to serial.
            return Ok(vec![Self::fold_morsel(
                &self.aggregations,
                &self.child_schema,
                &self.parameters,
                rows,
            )?]);
        };
        let morsel_size = resolve_morsel_size().max(1);
        // Bind the `Sync` inputs the closure needs as locals so the rayon
        // closure captures ONLY these (all `Send + Sync`), NOT `&self` —
        // `self` holds a `Box<PhysicalOperator>` child that is not `Sync`,
        // so a `&self`-capturing closure would not satisfy rayon's
        // `Fn: Sync + Send` bound. The child is already drained (it is not
        // touched here), so nothing is lost.
        let aggregations = &self.aggregations;
        let child_schema = &self.child_schema;
        let parameters = &self.parameters;
        // Partition into contiguous morsels + fold each in parallel on the
        // DEDICATED pool. `par_chunks` yields morsels in ascending index
        // order and `collect` preserves that order — irrelevant for the
        // commutative COUNT/SUM/AVG/MIN/MAX merges, but it keeps the merge
        // deterministic run-to-run.
        pool.install(|| {
            rows.par_chunks(morsel_size)
                .map(|morsel| Self::fold_morsel(aggregations, child_schema, parameters, morsel))
                .collect()
        })
    }

    /// Pull the next batch. Primes (drain + parallel fold + merge) lazily
    /// at first-batch, then emits the single result row (then EOS).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.output_rows.is_none() {
            self.materialize(ctx, substrate)?;
        }
        let rows = self.output_rows.as_ref().expect("materialized above");
        if self.cursor >= rows.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut out = Batch::with_capacity(self.schema.len());
        let take = (rows.len() - self.cursor).min(BATCH_ROWS);
        for row in &rows[self.cursor..self.cursor + take] {
            if !out.push_row(row.clone()) {
                return Err(ExecutionError::Eval(
                    "ParallelAggregateOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
    use proptest::prelude::*;

    use super::*;
    use crate::error::Span;
    use crate::executor::ops::{AggregateOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::semantic::bound_ast::{BoundExpression, BoundPropertyRef};

    // Serialize env-mutating tests: Rust runs #[test]s on shared process
    // threads and these poke process-global env (morsel size / threshold).
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    fn person_scan() -> ScanOp {
        ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
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
            path: vec![BoundPropertyRef {
                name: name.into(),
                property_id: None,
                span: Span::point(1, 1),
            }],
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Substrate of `(id, age)` nodes; `None` age → `Value::Null`.
    fn substrate_from(ages: &[(u64, Option<i64>)]) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for &(id, age) in ages {
            let mut view = NodeView::new(NodeId::new(id), Some(LabelId::new(1)));
            view = match age {
                Some(a) => view.with_property("age", Value::Integer(a)),
                None => view.with_property("age", Value::Null),
            };
            s = s.with_node(TenantId::DEFAULT, view);
        }
        s
    }

    fn count_star() -> AggregateCall {
        AggregateCall {
            kind: AggregationKind::Count,
            arg: var_ref(BindingId::new(0)),
            output_id: BindingId::new(10),
            distinct: false,
            star: true,
        }
    }

    fn agg_on_age(kind: AggregationKind, out: u64) -> AggregateCall {
        AggregateCall {
            kind,
            arg: prop_access(var_ref(BindingId::new(0)), "age"),
            output_id: BindingId::new(out),
            distinct: false,
            star: false,
        }
    }

    /// The FULL supported mergeable set: count(*), count(age), sum, avg,
    /// min, max — the exact calls the equivalence proptest sweeps.
    fn all_mergeable_calls() -> Vec<AggregateCall> {
        vec![
            count_star(),
            agg_on_age(AggregationKind::Count, 11),
            agg_on_age(AggregationKind::Sum, 12),
            agg_on_age(AggregationKind::Avg, 13),
            agg_on_age(AggregationKind::Min, 14),
            agg_on_age(AggregationKind::Max, 15),
        ]
    }

    /// Drive the SERIAL [`AggregateOp`] (no group-by) to its single output
    /// row — the ground truth the parallel op must match.
    fn serial_row(s: &StubExecutorSubstrate, calls: Vec<AggregateCall>) -> Vec<Value> {
        let mut op = AggregateOp::new(PhysicalOperator::Scan(person_scan()), Vec::new(), calls);
        let b = op.next_batch(&ctx(), s).unwrap();
        assert_eq!(
            b.row_count(),
            1,
            "no-group-by aggregate emits exactly one row"
        );
        b.row(0).to_vec()
    }

    /// Drive the PARALLEL [`ParallelAggregateOp`] to its single output row.
    fn parallel_row(s: &StubExecutorSubstrate, calls: Vec<AggregateCall>) -> Vec<Value> {
        let mut op = ParallelAggregateOp::new(PhysicalOperator::Scan(person_scan()), calls);
        let b = op.next_batch(&ctx(), s).unwrap();
        assert_eq!(
            b.row_count(),
            1,
            "no-group-by aggregate emits exactly one row"
        );
        b.row(0).to_vec()
    }

    /// Assert parallel ≡ serial cell-for-cell. Floats (AVG) compared via a
    /// tight epsilon; everything else via exact `Value` equality.
    fn assert_rows_equiv(serial: &[Value], parallel: &[Value]) {
        assert_eq!(serial.len(), parallel.len(), "column count mismatch");
        for (i, (sv, pv)) in serial.iter().zip(parallel.iter()).enumerate() {
            match (sv, pv) {
                (Value::Float(a), Value::Float(b)) => assert!(
                    (a - b).abs() < 1e-9,
                    "col {i}: float mismatch serial={a} parallel={b}"
                ),
                _ => assert_eq!(sv, pv, "col {i}: parallel ≢ serial"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Unit tests — NULL edges + fallback + single-morsel.
    // -----------------------------------------------------------------

    #[test]
    fn all_null_sum_is_null_avg_is_null_but_count_star_counts_rows() {
        let ages = vec![(1, None), (2, None), (3, None)];
        let s = substrate_from(&ages);
        let calls = all_mergeable_calls();
        let serial = serial_row(&s, calls.clone());
        let parallel = parallel_row(&s, calls);
        // count(*) = 3, count(age) = 0, sum = NULL, avg = NULL, min = NULL,
        // max = NULL.
        assert_eq!(parallel[0], Value::Integer(3), "count(*) counts all rows");
        assert_eq!(parallel[1], Value::Integer(0), "count(age) excludes NULL");
        assert_eq!(parallel[2], Value::Null, "sum of all-NULL is NULL");
        assert_eq!(parallel[3], Value::Null, "avg of all-NULL is NULL");
        assert_eq!(parallel[4], Value::Null, "min of all-NULL is NULL");
        assert_eq!(parallel[5], Value::Null, "max of all-NULL is NULL");
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn empty_input_emits_one_row_with_empty_aggregates() {
        let s = StubExecutorSubstrate::new();
        let calls = all_mergeable_calls();
        let serial = serial_row(&s, calls.clone());
        let parallel = parallel_row(&s, calls);
        assert_eq!(parallel[0], Value::Integer(0), "count(*) of nothing is 0");
        assert_eq!(parallel[2], Value::Null, "sum of nothing is NULL");
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn count_star_counts_all_rows_including_all_null() {
        let ages = vec![(1, Some(30)), (2, None), (3, Some(40))];
        let s = substrate_from(&ages);
        let calls = vec![count_star(), agg_on_age(AggregationKind::Count, 11)];
        let parallel = parallel_row(&s, calls.clone());
        let serial = serial_row(&s, calls);
        assert_eq!(parallel[0], Value::Integer(3), "count(*) counts 3 rows");
        assert_eq!(
            parallel[1],
            Value::Integer(2),
            "count(age) excludes the NULL"
        );
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn min_max_skip_null() {
        let ages = vec![(1, Some(50)), (2, None), (3, Some(20)), (4, Some(80))];
        let s = substrate_from(&ages);
        let calls = vec![
            agg_on_age(AggregationKind::Min, 14),
            agg_on_age(AggregationKind::Max, 15),
        ];
        let parallel = parallel_row(&s, calls.clone());
        let serial = serial_row(&s, calls);
        assert_eq!(parallel[0], Value::Integer(20), "min skips NULL");
        assert_eq!(parallel[1], Value::Integer(80), "max skips NULL");
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn avg_carries_sum_count_pair_not_partial_means() {
        // The AVG trap: unequal morsel sizes. 5 non-NULL ages spread so a
        // morsel-size of 2 yields morsels [10,20],[30,40],[50] — averaging
        // per-morsel means (15, 35, 50)/3 = 33.3 would be WRONG; the
        // correct mean is (10+20+30+40+50)/5 = 30.0.
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        // SAFETY (edition-2024 env mutation): serialized by ENV_TEST_LOCK;
        // both vars removed before the guard drops (below).
        unsafe {
            std::env::set_var("ARCGRAPH_SCAN_MORSEL_SIZE", "2");
            std::env::set_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD", "0");
        }
        let ages = vec![
            (1, Some(10)),
            (2, Some(20)),
            (3, Some(30)),
            (4, Some(40)),
            (5, Some(50)),
        ];
        let s = substrate_from(&ages);
        let calls = vec![agg_on_age(AggregationKind::Avg, 13)];
        let parallel = parallel_row(&s, calls.clone());
        let serial = serial_row(&s, calls);
        unsafe {
            std::env::remove_var("ARCGRAPH_SCAN_MORSEL_SIZE");
            std::env::remove_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD");
        }
        assert_eq!(
            parallel[0],
            Value::Float(30.0),
            "AVG = Σsum/Σcount, not mean-of-means"
        );
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn is_mergeable_rejects_group_by_distinct_and_collect() {
        // No group-by, plain calls → mergeable.
        assert!(ParallelAggregateOp::is_mergeable(
            &[],
            &all_mergeable_calls()
        ));
        // A DISTINCT call → NOT mergeable (needs a global set).
        let distinct_call = AggregateCall {
            distinct: true,
            ..count_star()
        };
        assert!(!ParallelAggregateOp::is_mergeable(&[], &[distinct_call]));
        // A COLLECT call → NOT mergeable at rc.
        let collect_call = agg_on_age(AggregationKind::Collect, 16);
        assert!(!ParallelAggregateOp::is_mergeable(&[], &[collect_call]));
        // A group-by column → NOT mergeable (grouped agg deferred).
        let group_item = BoundProjectionItem {
            kind: crate::semantic::bound_ast::BoundProjectionKind::Expr(prop_access(
                var_ref(BindingId::new(0)),
                "age",
            )),
            alias: None,
            output_id: Some(BindingId::new(1)),
            source_text: None,
            span: Span::point(1, 1),
        };
        assert!(!ParallelAggregateOp::is_mergeable(
            &[group_item],
            &all_mergeable_calls()
        ));
    }

    #[test]
    fn single_morsel_equals_serial() {
        // A small input (below the default threshold) takes the serial
        // single-morsel guard; the result must still equal serial.
        let ages: Vec<(u64, Option<i64>)> =
            (1..=20u64).map(|i| (i, Some((i * 3) as i64))).collect();
        let s = substrate_from(&ages);
        let calls = all_mergeable_calls();
        let serial = serial_row(&s, calls.clone());
        let parallel = parallel_row(&s, calls);
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn ragged_morsel_boundary_equals_serial() {
        // Force tiny morsels + zero threshold so a non-divisible row count
        // spans several morsels with a ragged final morsel (37 rows at
        // M=8 → 4 full + 1 partial). No row may be dropped or double-folded.
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        // SAFETY: serialized by ENV_TEST_LOCK; vars removed before drop.
        unsafe {
            std::env::set_var("ARCGRAPH_SCAN_MORSEL_SIZE", "8");
            std::env::set_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD", "0");
        }
        let ages: Vec<(u64, Option<i64>)> = (1..=37u64)
            .map(|i| {
                (
                    i,
                    if i % 4 == 0 {
                        None
                    } else {
                        Some((i % 13) as i64)
                    },
                )
            })
            .collect();
        let s = substrate_from(&ages);
        let calls = all_mergeable_calls();
        let parallel = parallel_row(&s, calls.clone());
        let serial = serial_row(&s, calls);
        unsafe {
            std::env::remove_var("ARCGRAPH_SCAN_MORSEL_SIZE");
            std::env::remove_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD");
        }
        assert_rows_equiv(&serial, &parallel);
    }

    #[test]
    fn propagates_cancel() {
        let s = substrate_from(&[(1, Some(10))]);
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op =
            ParallelAggregateOp::new(PhysicalOperator::Scan(person_scan()), all_mergeable_calls());
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }

    proptest! {
        // ---- THE HEADLINE TEST (ADR-226 §4 S5 correctness gate) ----
        //
        // On random datasets (incl. NULLs / empty / all-NULL), the
        // morsel-driven parallel partial aggregate produces the SAME
        // result as the serial aggregate for EVERY supported aggregate:
        // count(*), count(age), sum, avg, min, max — all in ONE sweep. We
        // force tiny morsels + zero threshold so even modest datasets
        // exercise the true multi-morsel partial-MERGE path (unequal
        // morsel sizes stress the AVG (sum,count) carry). A parallel agg
        // that differs from serial on ANY aggregate FAILS here.
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn parallel_agg_equiv_serial_agg_proptest(
            // 0..=300 nodes, each age in {Null (15%), 0..=200}.
            ages in proptest::collection::vec(
                proptest::option::weighted(0.85, 0i64..=200),
                0usize..=300,
            ),
        ) {
            // SAFETY (edition-2024 env mutation): serialized by
            // ENV_TEST_LOCK; both vars removed before the guard drops.
            let _guard = ENV_TEST_LOCK.lock().unwrap();
            unsafe {
                std::env::set_var("ARCGRAPH_SCAN_MORSEL_SIZE", "13");
                std::env::set_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD", "0");
            }
            let pairs: Vec<(u64, Option<i64>)> = ages
                .iter()
                .enumerate()
                .map(|(i, a)| ((i as u64) + 1, *a))
                .collect();
            let s = substrate_from(&pairs);
            let calls = all_mergeable_calls();
            let serial = serial_row(&s, calls.clone());
            let parallel = parallel_row(&s, calls);
            unsafe {
                std::env::remove_var("ARCGRAPH_SCAN_MORSEL_SIZE");
                std::env::remove_var("ARCGRAPH_SCAN_PARALLEL_THRESHOLD");
            }
            // Cell-for-cell equivalence across all 6 aggregates.
            prop_assert_eq!(serial.len(), parallel.len());
            for (i, (sv, pv)) in serial.iter().zip(parallel.iter()).enumerate() {
                match (sv, pv) {
                    (Value::Float(a), Value::Float(b)) => prop_assert!(
                        (a - b).abs() < 1e-9,
                        "col {}: float mismatch serial={} parallel={}", i, a, b
                    ),
                    _ => prop_assert_eq!(sv, pv, "col {}: parallel ≢ serial", i),
                }
            }
        }
    }
}
