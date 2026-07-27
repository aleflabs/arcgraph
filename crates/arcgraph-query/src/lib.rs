//! ArcQL — query language for ArcGraph.
//!
//! # Slice scope (M4-01)
//!
//! This crate at M4-01 ships:
//! - The `pest` PEG grammar covering the supported openCypher subset
//!   plus ArcGraph-specific vector-nearness, text-match, community-
//!   membership, shortest-path, and hybrid-ranking/fusion syntax.
//! - A typed AST (see [`ast`]).
//! - A parser entrypoint ([`parse`]).
//! - A syntactic error type ([`error::ParseError`]).
//!
//! What this slice **deliberately does not** ship (per the M4-01
//! task brief):
//! - Semantic analysis (variable binding, type checking,
//!   label-existence verification) — M4-02.
//! - Logical / physical plan generation — M4-03..M4-05.
//! - Vectorized executor — M4-06..M4-07.
//! - Reserved-clause `ArcQLError::NotImplemented` enforcement
//!   per ADR-038 §2 D-16 — M4-02.
//!
//! # ADR provenance
//! - **ADR-006 D-1** — `pest` PEG, openCypher subset baseline.
//! - **ADR-038 §2 D-1..D-10** — every grammar production traces
//!   back to one of these decisions.
//! - **ADR-038 §3.4** — locked test names; the parser-side
//!   half pinned by `tests/parser_smoke.rs` (executor-side
//!   half by M4-02).
//! - **ADR-038 §5** — v1.0 / v1.1 / v1.2 sequencing of which
//!   reserved clauses become "lit"; this slice cares only that
//!   they all PARSE.

#![recursion_limit = "256"]
// W13γ fix-up MED-4 (closes review-pr-285-final.md MED-4): the 3 doc-link
// rots in `parser.rs` / `binding.rs` referencing the non-existent
// `crate::executor::execute_multi_statement` were corrected to
// `crate::materialize::materialize_multi` +
// `crate::QueryEngine::execute_multi` in this fix-up. The structural
// gate — `#![deny(rustdoc::broken_intra_doc_links)]` — remains deferred
// because enabling it surfaces 21 existing
// ambiguous-link / unresolved-link errors across `explain/mod.rs` /
// `planner/cost/operator.rs` / `planner/enumeration/*.rs` /
// `semantic/binding.rs`. Resolve those links in a dedicated change
// before enabling the gate across `arcgraph-query`.

pub mod ast;
pub mod cancel;
pub mod cursor;
pub mod error;
pub mod executor;
pub mod explain;
pub mod logical_plan;
pub mod materialize;
pub mod observer;
pub mod parser;
pub mod planner;
pub mod semantic;
/// W26-γ-3 / ADR-136 — test-support utilities (arcql-smith generator).
/// Public so the libfuzzer harness at
/// `fuzz/fuzz_targets/arcql_smith_fuzz.rs` can consume it; carries no
/// production code path (every call site is a test, a fuzz target, or
/// a smoke-bench).
pub mod test_support;

pub use ast::{
    BinOp, Clause, CreateClause, CreateItem, CreateNodeSpec, Expression, FieldRef, Fusion,
    LengthRange, Literal, MatchBody, MatchClause, NamedPath, NamedPathKind, NodePattern,
    NumericLiteral, OrderDirection, OrderItem, PathPattern, ProjectionItem, ProjectionKind,
    PropertyMap, RankArg, RankByClause, Ranker, ReadQuery, RelDirection, RelPattern, ReturnClause,
    Statement, UnaryOp, UnwindClause, WithClause, WithFusionClause,
};
pub use cancel::{
    CancellationRegistry, DEFAULT_QUERY_TIMEOUT_MS, DeadlineHandle, spawn_deadline_timer,
    spawn_default_deadline_timer,
};
pub use cursor::StreamingCursor;
pub use error::{ParseError, Span};
pub use executor::value::ValueJsonError;
pub use executor::{
    BATCH_ROWS, BUDGET_FALLBACK_ROWS, Batch, BoundEdge, BoundNode, CancellationError,
    CancellationToken, ExecutionContext, ExecutionError, ExecutorSpillError,
    ExecutorSpillFailureKind, ExecutorSubstrate, MemoryBudget, MemoryReservation, PhysicalOperator,
    Pipeline, QueryId, RankedHit, SnapshotLsnGuard, StubExecutorSubstrate, SubstrateAccessError,
    ThreeValued, Value, estimate_row_bytes, estimate_value_bytes, execute, execute_with_context,
};
pub use explain::{
    ExecutionMetrics, ExplainError, PlanTree, PlanTreeOp, QueryEngine, explain, explain_with_cache,
    plan_tree_as_rows, profile,
};
pub use materialize::{MaterializedResult, materialize, materialize_multi, output_column_names};
pub use observer::{
    BreachDirection, DEFAULT_THRESHOLD_FACTOR, ObservedStatsOverrides, OperatorKind,
    OperatorMetrics, PlanWalkEntry, ReplanController, ReplanError, ReplanOutcome, ReplanReason,
    RowCountObserver, ThresholdBreach, apply_overrides_to_stub_catalog, walk_plan_and_costs,
};
pub use parser::{parse, parse_multi};
pub use planner::{
    CachedPlan as PlanCacheEntry, DEFAULT_MAX_ENTRIES_PER_TENANT, LookupOutcome, PlanCache,
    PlanCacheKey,
};
