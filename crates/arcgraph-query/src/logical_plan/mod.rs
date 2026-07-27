//! M4-31 logical-plan layer.
//!
//! This module is delivered in three sub-slices per ADR-038
//! amendment-03 (M4-03 decomposition):
//!
//! - **M4-31 (this slice — landed):** logical plan types + simple
//!   operator lowering. Ships [`LogicalPlan`] tree (Scan / Expand /
//!   Filter / Project / Join / Limit / Skip / Empty) +
//!   [`LogicalPlanLoweringVisitor`] (custom walker over `BoundQuery`,
//!   per ADR-038 §2 D-24 visitor-trait discipline lock).
//! - **M4-32 (this slice — landed):** hybrid retrieval lowering +
//!   OPTIONAL MATCH. Adds `RankByHybrid` / `Fusion` /
//!   `CommunityLookup` / `VectorNear` / `TextMatch` /
//!   `LeftOuterJoin` variants additively. Replaces the M4-31
//!   `NotImplementedAtM4_31` emission sites for hybrid surfaces +
//!   OPTIONAL MATCH per ADR-038 §2 D-26 + ADR-006 amendment-01 §A-2.
//! - **M4-33 (this slice — landed):** aggregation, ORDER BY, DISTINCT,
//!   UNWIND, named path, and dynamic LIMIT / SKIP. Adds `Aggregate`,
//!   `Sort`, `Distinct`, `Unwind`, `NamedPath`, and `DynamicLimit`
//!   variants additively. Replaces the M4-31 `NotImplementedAtM4_31`
//!   emission sites for aggregation, ORDER BY, DISTINCT, UNWIND,
//!   named path, and non-literal LIMIT / SKIP per ADR-038 §2 D-28.
//!   M4-03 substrate fully closed: M4-31 plus M4-32 plus M4-33 yields
//!   20 `LogicalPlan` variants.
//!
//! # Public surface
//!
//! - [`LogicalPlanLoweringVisitor::lower`] — the M4-31 entry point.
//!   Takes a `BoundStatement` (post-M4-23 cross-substrate validation;
//!   the "ValidatedQuery" of the M4-31 brief); returns a
//!   [`LogicalPlan`] or `Err(Vec<ArcQLError>)` carrying
//!   `ArcQLError::LogicalPlan` variants.
//! - [`LogicalPlan`] + supporting types
//!   ([`Direction`], [`JoinCondition`], operator-specific structs).
//! - [`LogicalPlanError`] — error taxonomy with `span_byte_range`
//!   (mirrors [`crate::semantic::error::BindingError`]).
//!
//! # Visitor-trait discipline (3-strike pattern, 6-slice consistency)
//!
//! Per ADR-038 §2 D-23 + D-24 + D-26, [`LogicalPlanLoweringVisitor`]
//! is a CUSTOM walker — NOT a trait abstraction. The 3-strike pattern
//! from M4-21 / M4-22 / M4-23 is the established discipline; M4-31 +
//! M4-32 inherit it:
//!
//! - M4-21 [`crate::semantic::BindingVisitor`] (custom; the
//!   speculative `AstVisitor` trait it briefly shipped was deleted in
//!   M4-22 review per `feedback_avoid_speculative_scaffolding.md`).
//! - M4-22 [`crate::semantic::TypeCheckVisitor`] (custom; the
//!   speculative `BoundAstVisitor` trait it briefly shipped was
//!   deleted in PR #165 reviewer Finding 1 fix-up).
//! - M4-22b binding-time `may_be_null` refinement (reused
//!   `BindingVisitor`; no trait).
//! - M4-23 [`crate::semantic::CrossSubstrateValidator`] (custom).
//! - M4-31 [`LogicalPlanLoweringVisitor`] (custom).
//! - M4-32 EXTENDS [`LogicalPlanLoweringVisitor`] additively (custom;
//!   NO trait abstraction).
//!
//! M4-33 / M4-05 inherit the same constraint: any future walker over
//! `BoundQuery` or [`LogicalPlan`] ships as a custom struct unless
//! ≥2 real consumers within the same slice justify the trait
//! abstraction.

pub mod error;
pub mod lowering;
pub mod types;

pub use error::LogicalPlanError;
pub use lowering::{
    LogicalPlanLoweringVisitor, rewrite_scan_to_property_index_scan,
    rewrite_unfiltered_count_to_count_store,
};
pub use types::{
    AggregationKind, AggregationSpec, CountStoreSource, DeleteKind, Direction, DynamicLimitKind,
    FusionKind, FusionSpec, HybridOperand, HybridOperandKind, JoinAlgorithm, JoinCondition,
    LogicalAggregate, LogicalCall, LogicalCommunityLookup, LogicalCorrelationSeed,
    LogicalCountStore, LogicalCreateEndpoint, LogicalCreateNode, LogicalCreateRel, LogicalDelete,
    LogicalDeleteItem, LogicalDistinct, LogicalDynamicLimit, LogicalEmpty, LogicalExpand,
    LogicalFilter, LogicalFusion, LogicalJoin, LogicalLeftOuterJoin, LogicalLimit,
    LogicalNamedPath, LogicalPlan, LogicalProcedureCall, LogicalProject, LogicalPropertyIndexScan,
    LogicalRankByHybrid, LogicalRemove, LogicalRemoveItem, LogicalRemoveMutation, LogicalScan,
    LogicalSet, LogicalSetItem, LogicalSetMutation, LogicalSkip, LogicalSort, LogicalTextMatch,
    LogicalUnion, LogicalUnwind, LogicalVectorNear, MergeKeySpec, OrderByItem, PathAlgorithm,
    PlainPathSegmentShape, PlainPathShape, ProcedureSource, SetTargetKind, SortDirection,
};
