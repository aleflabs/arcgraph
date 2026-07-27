//! M4-02 semantic-analysis layer.
//!
//! This module is delivered in three sub-slices per ADR-038
//! amendment-03:
//!
//! - **M4-21 (landed)**: symbol resolution + binding scopes +
//!   label/rel-type resolution. Builds a parallel
//!   [`bound_ast::BoundStatement`] from the syntactic
//!   [`crate::ast::Statement`].
//! - **M4-22 (landed)**: type checking +
//!   reserved-variant rejection (D-2 / D-7 / D-9 / D-10 / D-16) per
//!   ADR-038 §2 D-22; OPTIONAL MATCH binding rules per
//!   ADR-006 amendment-01.
//! - **M4-23 (landed)**: cross-substrate validation
//!   (per-tenant substrate-availability gating + RANK BY HYBRID
//!   semantic shape + WITH FUSION = RRF k requirement) + IN-COMMUNITY
//!   ↔ canonical `community(...)` lowering equivalence proptest per
//!   ADR-038 §2 D-23. Closes M4-02.
//!
//! Beyond the semantic-analysis core, this module also hosts:
//!
//! - **M4-42 (M4-04b — landed)**:
//!   [`selectivity::SelectivityEstimator`] — concrete struct over
//!   [`CatalogProvider`] that converts M4-41's per-tenant cardinality
//!   stats into per-predicate selectivity factors `f ∈ [0, 1]` for the
//!   future M4-05 cost-based planner. NO trait per the 6-slice
//!   3-strike pattern (single in-flight consumer at slice time). Per
//!   ADR-038 §2 D-27.
//!
//! # Public surface
//!
//! - [`BindingVisitor::bind`] — the M4-21 entry point. Takes an
//!   AST [`crate::ast::Statement`], the source string, and a
//!   [`CatalogProvider`]; returns a
//!   [`bound_ast::BoundStatement`] or a `Vec<BindingError>`.
//! - [`type_check::TypeCheckVisitor::check`] — the M4-22 entry
//!   point. Takes a `BoundStatement` (post-M4-21) and a
//!   [`CatalogProvider`]; populates `type_info` slots in-place;
//!   returns `Ok(())` on success or `Err(Vec<ArcQLError>)` on
//!   type-check / reserved-variant rejection.
//! - [`cross_substrate::CrossSubstrateValidator::validate`] — the
//!   M4-23 entry point. Takes a `BoundStatement` (post-M4-22) and a
//!   [`CatalogProvider`]; walks read-only and returns `Ok(())` on a
//!   clean pass or `Err(Vec<ArcQLError>)` with cross-substrate
//!   diagnostics (substrate-availability + RANK BY HYBRID semantic
//!   shape + RRF k requirement).
//! - [`CatalogProvider`] — schema-catalog adapter trait. Tests
//!   use [`StubCatalogProvider`]; production wires
//!   `arcgraph-storage`'s tenant catalog at executor-wiring time.
//! - [`BindingError`] / [`ArcQLError`] / [`TypeCheckError`] /
//!   [`CrossSubstrateError`] — error taxonomy with `span_byte_range`
//!   (mirrors [`crate::error::ParseError`]).
//!
//! # `AstVisitor` removal (M4-22) — M4-23 inheritance
//!
//! M4-21 shipped a default-walking `AstVisitor` over the raw AST
//! that had zero consumers (PR #164 reviewer ask + memory
//! `feedback_avoid_speculative_scaffolding.md`). M4-22 deletes that
//! trait. M4-22 also briefly introduced a `BoundAstVisitor` over the
//! BOUND ast, but PR #165 reviewer Finding 1 observed the same
//! pattern recurrence — `TypeCheckVisitor` walked the bound AST via
//! `&mut`-aware `check_*` methods directly, not through the trait —
//! so the trait + its surface-lock test (474 LOC) were deleted.
//!
//! **M4-23 honors the precedent.**
//! [`cross_substrate::CrossSubstrateValidator`] is a custom walker
//! over `BoundQuery` (read-only, no mutation), NOT a trait
//! abstraction. The 3-strike pattern (M4-21 trait deleted, M4-22
//! trait deleted, M4-23 doesn't ship a third) is the established
//! discipline. M4-31 (logical plan generator) MUST also ship its own
//! custom walker — it MUST NOT inherit any abstraction from the
//! M4-2x family.

pub mod binding;
pub mod bound_ast;
pub mod catalog;
pub mod cross_substrate;
pub mod error;
pub mod functions;
pub mod selectivity;
pub mod type_check;

pub use binding::BindingVisitor;
pub use bound_ast::{
    BindingId, BoundClause, BoundExpression, BoundFieldRef, BoundFusion, BoundLabelRef,
    BoundMapProjectionItem, BoundMatchBody, BoundMatchClause, BoundNamedPath, BoundNamedPathKind,
    BoundNodePattern, BoundOrderItem, BoundPathPattern, BoundProjectionItem, BoundProjectionKind,
    BoundPropertyEntry, BoundPropertyMap, BoundPropertyRef, BoundQuery, BoundRankArg,
    BoundRankByClause, BoundRanker, BoundRelPattern, BoundRelTypeRef, BoundReturnClause,
    BoundStatement, BoundUnwindClause, BoundVariable, BoundWithClause, BoundWithFusionClause,
    PropertyType, ScopeId, TypeInfo,
};
pub use catalog::{CatalogProvider, CatalogSnapshot, MaxOutDegreeEntry, StubCatalogProvider};

/// Production-named alias for [`StubCatalogProvider`].
///
/// The underlying struct ships as the v1.0-α catalog impl for BOTH
/// test fixtures and production paths (the MCP / Bolt raw-query
/// surfaces seed an instance per call). The "stub" name dates to
/// when the struct was test-only; per R1 review MED-2 (PR #349)
/// production callers SHOULD use this alias so the test/production
/// boundary is visible at the import site. v1.1 replaces the
/// production-side impl with a storage-backed `CatalogProvider`;
/// the alias goes away at that point.
pub type InMemoryCatalogProvider = StubCatalogProvider;
pub use cross_substrate::CrossSubstrateValidator;
pub use error::{ArcQLError, BindingError, CrossSubstrateError, SubstrateKind, TypeCheckError};
pub use selectivity::{
    DEFAULT_EQ_SELECTIVITY, DEFAULT_IN_SELECTIVITY, DEFAULT_LABEL_SELECTIVITY,
    DEFAULT_LT_SELECTIVITY, DEFAULT_REL_TYPE_SELECTIVITY, SelectivityEstimator,
};
pub use type_check::TypeCheckVisitor;
