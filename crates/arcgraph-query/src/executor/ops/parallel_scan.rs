//! [`ParallelScanOp`] — morsel-driven parallel node-scan operator
//! (ADR-226 §4 slice S4, gate **CONC-D**).
//!
//! # What
//!
//! A drop-in parallel replacement for [`super::ScanOp`] that produces
//! an **identical** result — same multiset of rows, same values, and
//! (because morsels are concatenated in id-order) the same row ORDER —
//! by splitting the scan buffer into fixed-size **morsels** (~64K
//! records) and filtering each morsel in parallel on a **dedicated**
//! rayon thread pool.
//!
//! # Evidence
//!
//! Leis, Boncz, Kemper, Neumann — *"Morsel-Driven Parallelism: A
//! NUMA-Aware Query Evaluation Framework for the Many-Core Age"*,
//! SIGMOD 2014. Morsel-driven execution splits a scan's input range
//! into small, fixed-size chunks ("morsels") dispatched to a worker
//! pool; because each morsel is independent, the filter (predicate
//! pushdown) parallelizes with near-linear scaling to core count
//! (SIGMOD 2014 §5, Fig. 9). The dedicated pool (not the global rayon
//! pool) keeps the scan's CPU fan-out off the async reactor's
//! work-stealing pool under the code-quality policy (Monoio hot path / Tokio
//! background) + the ADR-225 pinned-core migration.
//!
//! # Back-of-envelope latency budget (ADR-226 §4 CONC-D target)
//!
//! A 10M-node label scan materializes ~10M [`BoundNode`] then filters:
//! serial ≈ 2 s (single core, ~200 ns/row incl. predicate eval). With
//! `cores − 4` workers on an 8-core box (= 4 workers) over
//! ⌈10M / 64K⌉ ≈ 153 morsels, near-linear scaling targets ≤ 0.5 s
//! (≥ 4×). The morsel size (64K) is chosen so each unit is ~large
//! enough to amortize task-dispatch overhead (a rayon `spawn` is
//! ~hundreds of ns) yet ~small enough to keep the working set in L2
//! and load-balance across workers. Below
//! [`DEFAULT_PARALLEL_ROW_THRESHOLD`] the fan-out overhead dominates,
//! so the small-scan guard keeps small scans serial.
//!
//! # Correctness (the #1 gate)
//!
//! The parallel op reads the **same** materialized buffer the serial
//! op reads (`scan_nodes_with_context`), so the source multiset is
//! identical by construction. The only added logic is (a) partitioning
//! the buffer into contiguous morsels and (b) applying the optional
//! WHERE predicate per morsel. Morsel `k` covers buffer indices
//! `[k·M, (k+1)·M)`; the partition is a total, disjoint cover of
//! `[0, buffer.len())`, and each morsel's surviving rows are emitted in
//! their in-morsel order, morsels concatenated in ascending `k`. Thus
//! the parallel output equals the serial output row-for-row (the
//! headline `parallel ≡ serial` proptest pins this).
//!
//! # ORDER BY
//!
//! ORDER BY is a **separate downstream** [`super::SortOp`]
//! ([`crate::executor::ops::PhysicalOperator::Sort`]), never the scan's
//! concern — so this operator does NOT sort. It preserves the serial
//! scan's id-ordered output (morsel concatenation is order-preserving),
//! which is stronger than the substrate's "implementation-defined
//! order" contract requires and keeps a downstream Sort's input
//! byte-identical to the serial path.
//!
//! # Feature flag / fallback
//!
//! The planner decides whether to build this op vs the serial
//! [`super::ScanOp`] via [`ParallelScanOp::enabled_by_env`] (env
//! `ARCGRAPH_PARALLEL_SCAN`). Default **off** at rc: the conservative
//! posture is serial-unless-enabled, and the flag is the revert path
//! per ADR-226 §4 risk table line 346 ("feature-flag the parallel
//! operator OFF → planner falls back to serial"). The op itself is
//! always correct regardless of the flag; the flag only governs
//! whether the pipeline instantiates it.
//!
//! # ADR provenance
//! - **ADR-226 §4 slice S4 / gate CONC-D** — morsel-driven parallel
//!   scan; pool sized `cores − 4`; per-morsel filter pushdown;
//!   ordered-merge only when ORDER BY requires; small-scan guard;
//!   feature-flag revert. Env knobs for the ADR-225 pinned-core
//!   migration.
//! - **Leis et al. SIGMOD 2014** — morsel-driven parallelism evidence.
//! - **ADR-038 §2 D-18 rule 1** — snapshot LSN acquired at first batch
//!   (inherited from [`super::ScanOp`]).

use std::sync::OnceLock;

use arcgraph_core::{LabelId, Lsn};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::schema_index;
use crate::executor::substrate::{BoundNode, ExecutorSubstrate};
use crate::executor::three_vl::ThreeValued;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// Default morsel size in records (~64K per ADR-226 §4 S4). A morsel
/// is one contiguous slice of the scan buffer dispatched to one rayon
/// task; 64K amortizes task-dispatch overhead while staying L2-friendly.
/// Env-overridable via [`ENV_MORSEL_SIZE`].
pub const DEFAULT_MORSEL_SIZE: usize = 64 * 1024;

/// Row-count threshold below which the scan stays **serial** (the
/// small-scan guard, ADR-226 §4 risk line 346). Below this the
/// fan-out + pool-dispatch overhead outweighs the parallel win, so a
/// small scan is filtered in-line on the calling thread. Set to one
/// morsel: a scan that does not even fill a single morsel has no
/// parallelism to gain. Env-overridable via [`ENV_ROW_THRESHOLD`].
pub const DEFAULT_PARALLEL_ROW_THRESHOLD: usize = DEFAULT_MORSEL_SIZE;

/// Reserve this many cores for the async reactor / OS when sizing the
/// dedicated pool (`cores − 4` per ADR-226 §4 S4). On a box with
/// `≤ RESERVED_CORES` logical cores the pool falls back to a single
/// worker (clamped to `≥ 1`) — parallelism degrades gracefully rather
/// than oversubscribing.
pub const RESERVED_CORES: usize = 4;

/// Env var: enable the parallel scan operator (`"1"` / `"true"` →
/// on). Absent / any other value → off (serial fallback). ADR-226 §4
/// risk-table revert path.
pub const ENV_PARALLEL_SCAN: &str = "ARCGRAPH_PARALLEL_SCAN";

/// Env var: override the dedicated pool's worker count. Parsed as
/// `usize`; clamped to `≥ 1`. For the ADR-225 pinned-core migration.
pub const ENV_POOL_THREADS: &str = "ARCGRAPH_SCAN_POOL_THREADS";

/// Env var: override the morsel size in records. Parsed as `usize`;
/// a value of `0` is ignored (falls back to [`DEFAULT_MORSEL_SIZE`]).
pub const ENV_MORSEL_SIZE: &str = "ARCGRAPH_SCAN_MORSEL_SIZE";

/// Env var: override the small-scan row threshold. Parsed as `usize`.
pub const ENV_ROW_THRESHOLD: &str = "ARCGRAPH_SCAN_PARALLEL_THRESHOLD";

/// Process-wide dedicated rayon pool for parallel scans. Built once on
/// first use (env-sized) so we do not pay pool-construction cost per
/// query and do not spawn a fresh OS-thread set per scan. Distinct from
/// rayon's global pool by construction, per ADR-226 §4 S4 ("dedicated
/// rayon pool, not the global pool").
static SCAN_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Resolve the dedicated pool's worker count: `ARCGRAPH_SCAN_POOL_THREADS`
/// if set + parseable + `≥ 1`, else `available_parallelism − RESERVED_CORES`
/// clamped to `≥ 1`.
fn resolve_pool_threads() -> usize {
    if let Some(n) = std::env::var(ENV_POOL_THREADS)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n;
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cores.saturating_sub(RESERVED_CORES).max(1)
}

/// Borrow the process-wide dedicated scan pool, building it (env-sized)
/// on first call. A pool-build failure (extremely rare — only on OS
/// thread-spawn exhaustion) degrades to `None` so the caller runs the
/// scan serially rather than failing the query.
///
/// `pub(crate)` so the S5 morsel-driven parallel AGGREGATE
/// ([`super::parallel_aggregate`], ADR-226 §4 CONC-D) fans its
/// per-morsel partial folds onto the SAME dedicated pool — the aggregate
/// fan-out shares the scan's `cores − 4` workers and stays OFF the async
/// reactor's work-stealing pool (code-quality policy), rather than spinning up a
/// second competing pool.
pub(crate) fn scan_pool() -> Option<&'static ThreadPool> {
    // `get_or_init` cannot fail-through, so build first then store; on
    // build error we return `None` and the caller stays serial.
    if let Some(pool) = SCAN_POOL.get() {
        return Some(pool);
    }
    let threads = resolve_pool_threads();
    match ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("arcgraph-scan-{i}"))
        .build()
    {
        Ok(pool) => Some(SCAN_POOL.get_or_init(|| pool)),
        Err(_) => None,
    }
}

/// Resolve the effective morsel size (env-override or default).
/// `pub(crate)` so S5's parallel aggregate splits its input into the
/// SAME-sized morsels the scan uses ([`super::parallel_aggregate`]).
pub(crate) fn resolve_morsel_size() -> usize {
    std::env::var(ENV_MORSEL_SIZE)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MORSEL_SIZE)
}

/// Resolve the effective small-scan row threshold (env-override or
/// default). `pub(crate)` so S5's parallel aggregate applies the SAME
/// small-input guard (small aggregates stay serial —
/// [`super::parallel_aggregate`]).
pub(crate) fn resolve_row_threshold() -> usize {
    std::env::var(ENV_ROW_THRESHOLD)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PARALLEL_ROW_THRESHOLD)
}

/// Morsel-driven parallel node-scan operator.
///
/// Buffers the scan result once at first-batch (like [`super::ScanOp`]),
/// applies the optional WHERE predicate per morsel in parallel on the
/// dedicated pool, then paginates the filtered result out in
/// [`BATCH_ROWS`]-sized chunks. Produces a byte-identical result to the
/// serial scan-then-filter pipeline.
#[derive(Debug)]
pub struct ParallelScanOp {
    /// Variable bound by this scan (mirrored in `schema[0]`).
    #[allow(dead_code)]
    binding: BindingId,
    /// Optional label filter (threaded to the substrate scan).
    label: Option<LabelId>,
    /// MVCC read LSN copied from the plan (ADR-041 §D-4).
    plan_read_lsn: Lsn,
    /// Optional pushed-down WHERE predicate, applied per morsel. `None`
    /// → a bare scan (every buffered node survives).
    predicate: Option<BoundExpression>,
    /// Per-query parameter bag for predicate evaluation.
    parameters: Parameters,
    /// Cached per-batch schema (length-1: just the binding).
    schema: Vec<BindingId>,
    /// Buffered + already-filtered scan result. `None` until
    /// first-batch primes it.
    buffer: Option<Vec<BoundNode>>,
    /// Cursor into the (filtered) buffer.
    cursor: usize,
}

impl ParallelScanOp {
    /// Construct a fresh `ParallelScanOp` for a bare scan (no WHERE
    /// pushdown — every node survives). Filter pushdown is opt-in via
    /// [`Self::with_predicate`].
    #[must_use]
    pub fn new(binding: BindingId, label: Option<LabelId>, plan_read_lsn: Lsn) -> Self {
        Self {
            binding,
            label,
            plan_read_lsn,
            predicate: None,
            parameters: Parameters::new(),
            schema: vec![binding],
            buffer: None,
            cursor: 0,
        }
    }

    /// Push a WHERE predicate down into the parallel scan. The
    /// predicate is evaluated per morsel in parallel; rows whose
    /// predicate is not [`ThreeValued::True`] are dropped (Cypher 9
    /// §6.2 WHERE-boundary semantics, matching [`super::FilterOp`]).
    #[must_use]
    pub fn with_predicate(mut self, predicate: BoundExpression) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Inject a per-query parameter bag for predicate evaluation.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema. Always `[binding]`.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Whether the parallel scan operator is enabled via the
    /// `ARCGRAPH_PARALLEL_SCAN` env flag. The planner calls this to
    /// choose between building this op and the serial [`super::ScanOp`]
    /// (ADR-226 §4 risk-table revert path). Default off.
    #[must_use]
    pub fn enabled_by_env() -> bool {
        matches!(
            std::env::var(ENV_PARALLEL_SCAN).ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
    }

    /// Apply the (optional) predicate to one node, returning `true` if
    /// the node survives the WHERE boundary. A `None` predicate keeps
    /// every node. A predicate that errors on this row surfaces the
    /// error (propagated out of the parallel region).
    fn node_passes(
        predicate: &BoundExpression,
        schema: &[BindingId],
        params: &Parameters,
        node: &BoundNode,
    ) -> Result<bool, ExecutionError> {
        // Single-column row `[Value::Node(..)]` — same shape the serial
        // ScanOp → FilterOp pipeline evaluates against.
        let row = [Value::Node(node.node.clone())];
        let lookup = |b: BindingId| schema_index(schema, b);
        let v = evaluate(predicate, &row, &lookup, params)?;
        Ok(ThreeValued::from_value(&v).passes_filter())
    }

    /// Serial per-morsel filter (small-scan guard path + the parallel
    /// path's per-morsel body). Filters `nodes` in place-order, keeping
    /// survivors. Order-preserving.
    fn filter_serial(&self, nodes: &[BoundNode]) -> Result<Vec<BoundNode>, ExecutionError> {
        match &self.predicate {
            None => Ok(nodes.to_vec()),
            Some(pred) => {
                let mut out = Vec::with_capacity(nodes.len());
                for node in nodes {
                    if Self::node_passes(pred, &self.schema, &self.parameters, node)? {
                        out.push(node.clone());
                    }
                }
                Ok(out)
            }
        }
    }

    /// Morsel-parallel filter over the whole buffer on the dedicated
    /// pool. Splits `nodes` into contiguous `morsel_size` chunks,
    /// filters each in parallel, and concatenates survivors in morsel
    /// order (order-preserving, so ≡ the serial output).
    ///
    /// Falls back to [`Self::filter_serial`] when (a) the buffer is
    /// below the small-scan threshold or (b) the dedicated pool could
    /// not be built — either way the RESULT is identical, only the
    /// execution strategy differs.
    fn filter_parallel(&self, nodes: Vec<BoundNode>) -> Result<Vec<BoundNode>, ExecutionError> {
        let threshold = resolve_row_threshold();
        if nodes.len() < threshold {
            // Small-scan guard: stay serial.
            return self.filter_serial(&nodes);
        }
        let Some(pool) = scan_pool() else {
            // Pool build failed (thread exhaustion) — degrade to serial.
            return self.filter_serial(&nodes);
        };
        let morsel_size = resolve_morsel_size().max(1);

        // Partition into contiguous morsels; filter each in parallel on
        // the DEDICATED pool. `par_chunks` yields morsels in ascending
        // index order and `collect` preserves that order into the outer
        // Vec, so the flattened result is id-ordered (≡ serial).
        let per_morsel: Result<Vec<Vec<BoundNode>>, ExecutionError> = pool.install(|| {
            nodes
                .par_chunks(morsel_size)
                .map(|morsel| self.filter_serial(morsel))
                .collect()
        });
        let per_morsel = per_morsel?;

        // Flatten survivors in morsel order.
        let total: usize = per_morsel.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(total);
        for mut morsel_out in per_morsel {
            out.append(&mut morsel_out);
        }
        Ok(out)
    }

    /// Pull the next batch. Primes (scan + parallel filter) lazily at
    /// first-batch, then paginates the filtered buffer in
    /// [`BATCH_ROWS`]-sized chunks.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check (mirrors ScanOp / dispatcher).
        ctx.cancellation().check()?;

        if self.buffer.is_none() {
            // Snapshot LSN acquired at first batch, ADR-038 §2 D-18
            // rule 1 (identical to the serial ScanOp).
            let _exec_lsn = ctx.ensure_snapshot_lsn();
            let nodes = substrate.scan_nodes_with_context(ctx, self.label, self.plan_read_lsn)?;
            // Re-check cancellation after the (potentially large) scan +
            // before the parallel filter fans out onto the pool.
            ctx.cancellation().check()?;
            let filtered = self.filter_parallel(nodes)?;
            self.buffer = Some(filtered);
        }

        let buf = self.buffer.as_ref().expect("primed above");
        if self.cursor >= buf.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut batch = Batch::with_capacity(self.schema.len());
        let take = (buf.len() - self.cursor).min(BATCH_ROWS);
        for node in &buf[self.cursor..self.cursor + take] {
            if !batch.push_row(vec![Value::Node(node.node.clone())]) {
                return Err(ExecutionError::Eval(
                    "ParallelScanOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{NodeId, PartitionId, TenantId};
    use proptest::prelude::*;

    use super::*;
    use crate::ast::{BinOp, Literal};
    use crate::error::Span;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::semantic::bound_ast::BoundPropertyRef;

    /// `n.age <op> <target>` predicate over the node binding.
    fn predicate_age(node_binding: BindingId, op: BinOp, target: i64) -> BoundExpression {
        BoundExpression::BinaryOp {
            op,
            lhs: Box::new(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: node_binding,
                    span: Span::point(1, 1),
                    type_info: None,
                }),
                path: vec![BoundPropertyRef {
                    name: "age".into(),
                    property_id: None,
                    span: Span::point(1, 1),
                }],
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::Literal {
                value: Literal::Integer(target),
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Build a stub substrate from `(id, age)` pairs.
    fn substrate_from(ages: &[(u64, Option<i64>)]) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for &(id, age) in ages {
            let mut view = NodeView::new(NodeId::new(id), Some(LabelId::new(1)));
            if let Some(a) = age {
                view = view.with_property("age", Value::Integer(a));
            } else {
                view = view.with_property("age", Value::Null);
            }
            s = s.with_node(TenantId::DEFAULT, view);
        }
        s
    }

    /// Reference serial pipeline result: serial `ScanOp` → serial
    /// per-row `FilterOp` semantics, expressed as the survivor id list.
    /// This is the ground-truth the parallel scan must match.
    fn serial_filter_ids(s: &StubExecutorSubstrate, predicate: &BoundExpression) -> Vec<NodeId> {
        use crate::executor::ops::{FilterOp, PhysicalOperator};
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut op = FilterOp::new(PhysicalOperator::Scan(scan), predicate.clone());
        let mut ids = Vec::new();
        loop {
            let b = op.next_batch(&ctx, s).unwrap();
            if b.is_empty() {
                break;
            }
            for row in b.rows() {
                if let Value::Node(n) = &row[0] {
                    ids.push(n.id);
                }
            }
        }
        ids
    }

    /// Drive a single serial `ScanOp` to its id list (no predicate).
    fn serial_scan_ids(s: &StubExecutorSubstrate) -> Vec<NodeId> {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let mut ids = Vec::new();
        loop {
            let b = op.next_batch(&ctx, s).unwrap();
            if b.is_empty() {
                break;
            }
            for row in b.rows() {
                if let Value::Node(n) = &row[0] {
                    ids.push(n.id);
                }
            }
        }
        ids
    }

    /// Drive a `ParallelScanOp` to its id list.
    fn parallel_scan_ids(s: &StubExecutorSubstrate, op: ParallelScanOp) -> Vec<NodeId> {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op;
        let mut ids = Vec::new();
        loop {
            let b = op.next_batch(&ctx, s).unwrap();
            if b.is_empty() {
                break;
            }
            for row in b.rows() {
                if let Value::Node(n) = &row[0] {
                    ids.push(n.id);
                }
            }
        }
        ids
    }

    #[test]
    fn parallel_bare_scan_equals_serial_scan() {
        let ages: Vec<(u64, Option<i64>)> = (1..=1000u64).map(|i| (i, Some(i as i64))).collect();
        let s = substrate_from(&ages);
        let serial = serial_scan_ids(&s);
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX),
        );
        assert_eq!(serial, parallel, "bare scan must equal serial (same order)");
    }

    #[test]
    fn parallel_filter_pushdown_equals_serial_filter() {
        let ages: Vec<(u64, Option<i64>)> =
            (1..=500u64).map(|i| (i, Some((i % 100) as i64))).collect();
        let s = substrate_from(&ages);
        let pred = predicate_age(BindingId::new(0), BinOp::Gt, 50);
        let serial = serial_filter_ids(&s, &pred);
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
                .with_predicate(pred),
        );
        assert_eq!(serial, parallel, "filter pushdown must equal serial filter");
    }

    #[test]
    fn parallel_filter_drops_null_predicate_rows() {
        // A NULL `age` → predicate Unknown → row dropped (3VL), same as
        // the serial FilterOp.
        let ages = vec![
            (1, Some(50)),
            (2, None),
            (3, Some(10)),
            (4, None),
            (5, Some(80)),
        ];
        let s = substrate_from(&ages);
        let pred = predicate_age(BindingId::new(0), BinOp::Gt, 30);
        let serial = serial_filter_ids(&s, &pred);
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
                .with_predicate(pred),
        );
        assert_eq!(serial, vec![NodeId::new(1), NodeId::new(5)]);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn empty_scan_yields_empty() {
        let s = StubExecutorSubstrate::new();
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX),
        );
        assert!(parallel.is_empty(), "empty range → empty result");
    }

    #[test]
    fn small_scan_stays_serial_but_correct() {
        // A scan below the threshold takes the serial guard path; the
        // result must still be correct. Force a tiny threshold override
        // is unnecessary — 5 rows is far below the default threshold.
        let ages = vec![(1, Some(10)), (2, Some(20)), (3, Some(30))];
        let s = substrate_from(&ages);
        let serial = serial_scan_ids(&s);
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX),
        );
        assert_eq!(serial, parallel);
    }

    #[test]
    fn morsel_boundary_not_divisible_stays_correct() {
        // Force a tiny morsel size + threshold so 250 rows spans several
        // morsels with a ragged final morsel (250 = 4*64 - 6 at M=64).
        // SAFETY of env mutation: this test sets process-global env; it
        // is isolated by running with a locally-scoped guard that
        // restores prior values. Rust test threads share the process,
        // so we serialize env-mutating tests via a mutex.
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        // Threshold 0 so even a small buffer takes the parallel path.
        unsafe {
            std::env::set_var(ENV_MORSEL_SIZE, "64");
            std::env::set_var(ENV_ROW_THRESHOLD, "0");
        }
        let ages: Vec<(u64, Option<i64>)> = (1..=250u64).map(|i| (i, Some(i as i64))).collect();
        let s = substrate_from(&ages);
        let serial = serial_scan_ids(&s);
        let parallel = parallel_scan_ids(
            &s,
            ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX),
        );
        unsafe {
            std::env::remove_var(ENV_MORSEL_SIZE);
            std::env::remove_var(ENV_ROW_THRESHOLD);
        }
        assert_eq!(
            serial, parallel,
            "ragged final morsel must not drop/dupe rows"
        );
    }

    #[test]
    fn enabled_by_env_reads_flag() {
        // SAFETY (edition-2024 `set_var`/`remove_var` unsafe): process
        // env mutation is serialized by `ENV_TEST_LOCK` (no concurrent
        // reader of these vars runs while the guard is held) and each var
        // is restored (removed) before the guard drops.
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        unsafe { std::env::set_var(ENV_PARALLEL_SCAN, "1") };
        assert!(ParallelScanOp::enabled_by_env());
        unsafe { std::env::set_var(ENV_PARALLEL_SCAN, "0") };
        assert!(!ParallelScanOp::enabled_by_env());
        unsafe { std::env::remove_var(ENV_PARALLEL_SCAN) };
        assert!(!ParallelScanOp::enabled_by_env(), "default off");
    }

    #[test]
    fn pool_threads_resolves_cores_minus_reserved() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        // SAFETY: see `enabled_by_env_reads_flag` — `ENV_TEST_LOCK`
        // serializes env mutation + the var is removed before drop.
        unsafe { std::env::set_var(ENV_POOL_THREADS, "3") };
        assert_eq!(resolve_pool_threads(), 3, "env override wins");
        unsafe { std::env::remove_var(ENV_POOL_THREADS) };
        // Default is cores − 4, clamped ≥ 1 — must always be ≥ 1.
        assert!(resolve_pool_threads() >= 1);
    }

    // Serialize env-mutating tests: Rust runs #[test]s on shared
    // process threads, and these tests poke process-global env vars.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    proptest! {
        // ---- THE HEADLINE TEST (ADR-226 §4 S4 correctness gate) ----
        //
        // On random datasets, the morsel-driven parallel scan produces
        // the SAME rows as the serial scan (bare + with a random
        // predicate). A parallel scan that drops / dupes / reorders rows
        // FAILS here. We force a tiny morsel size + zero threshold so
        // even modest random datasets exercise the true parallel
        // (multi-morsel) path.
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn parallel_scan_equiv_serial_scan_proptest(
            // Random dataset: 0..=400 nodes, each with an age in
            // {Null, 0..=200}. Ids are 1..=n so identity is stable.
            ages in proptest::collection::vec(
                proptest::option::weighted(0.85, 0i64..=200),
                0usize..=400,
            ),
            // Random predicate threshold + comparison op.
            target in 0i64..=200,
            op_pick in 0u8..6,
        ) {
            // SAFETY (edition-2024 env mutation): serialized by
            // `ENV_TEST_LOCK`; both vars are removed before the guard
            // drops (below), so no concurrent reader observes them.
            let _guard = ENV_TEST_LOCK.lock().unwrap();
            unsafe {
                std::env::set_var(ENV_MORSEL_SIZE, "16");
                std::env::set_var(ENV_ROW_THRESHOLD, "0");
            }

            let pairs: Vec<(u64, Option<i64>)> = ages
                .iter()
                .enumerate()
                .map(|(i, a)| ((i as u64) + 1, *a))
                .collect();
            let s = substrate_from(&pairs);

            // (1) Bare scan equivalence — same multiset AND order.
            let serial_bare = serial_scan_ids(&s);
            let parallel_bare = parallel_scan_ids(
                &s,
                ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX),
            );
            prop_assert_eq!(&serial_bare, &parallel_bare);

            // (2) Filter-pushdown equivalence over a random predicate.
            let op = match op_pick {
                0 => BinOp::Eq,
                1 => BinOp::Neq,
                2 => BinOp::Lt,
                3 => BinOp::Le,
                4 => BinOp::Gt,
                _ => BinOp::Ge,
            };
            let pred = predicate_age(BindingId::new(0), op, target);
            let serial_f = serial_filter_ids(&s, &pred);
            let parallel_f = parallel_scan_ids(
                &s,
                ParallelScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
                    .with_predicate(pred),
            );

            unsafe {
                std::env::remove_var(ENV_MORSEL_SIZE);
                std::env::remove_var(ENV_ROW_THRESHOLD);
            }
            prop_assert_eq!(serial_f, parallel_f);
        }
    }
}
