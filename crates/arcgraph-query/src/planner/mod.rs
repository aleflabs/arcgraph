//! ArcQL query planner.
//!
//! # Slice scope (M4-51 — M4-05a per ADR-038 amendment-02 §M4.e)
//!
//! M4-51 ships the **cost model**: per-operator cost functions plus
//! the entry point that walks a [`crate::logical_plan::LogicalPlan`]
//! and produces a [`cost::CostedPlan`] (cost-annotated parallel tree).
//! Subsequent slices in the M4-05 family extend additively without
//! re-shaping this module:
//!
//! - **M4-51 (M4-05a — landed):** cost model + per-operator cost
//!   functions + compositional selectivity helper.
//! - **M4-52 (this slice — M4-05b):** plan enumeration + DP-based
//!   binary-join ordering. Consumes [`cost::estimate_costs`] for
//!   per-leaf initial costing + the per-operator
//!   [`cost::operator::cost_join`] for incremental candidate costing
//!   inside the DP loop. Picks a cost-optimal left-deep ordering for
//!   star + linear shapes (bushy deferred to v1.1) per ADR-038
//!   amendment-02 §M4.e. See [`enumeration`].
//! - **M4-53 (M4-05c — forward):** plan validation + per-tenant plan
//!   cache. Will consume [`cost::CostedPlan`] for cache-key derivation.
//!
//! # Inputs / outputs
//!
//! ```text
//!     LogicalPlan (M4-31..M4-33 producer)
//!        │
//!        ▼
//!     planner::cost::estimate_costs(&plan, &catalog)
//!        │   reads CatalogProvider::snapshot() ONCE
//!        │   reads SelectivityEstimator (M4-42 surface)
//!        │   composes per-predicate selectivity via
//!        │   planner::cost::composition::{and,or,not}
//!        ▼
//!     CostedPlan { plan, cost_tree }
//! ```
//!
//! # Visitor-trait discipline (7-slice 3-strike consistency)
//!
//! The cost walker is a CONCRETE STRUCT, NOT a `pub trait
//! LogicalPlanVisitor` / `LogicalPlanRewrite`. The 7-slice precedent
//! across M4-21 / M4-22 / M4-22b / M4-23 / M4-31 / M4-32 / M4-33 is
//! "ship the abstraction when there are ≥2 real consumers, not when
//! there is one imagined consumer". M4-51 has exactly ONE consumer
//! (this cost walker); M4-52's plan enumeration may justify the
//! trait extraction in a later slice once its consumer surface is
//! known.
//!
//! Per the M4-33 codex review forward-note 4, M4-51 is the *second
//! consumer* of the LogicalPlan tree (M4-31..M4-33 was the producer).
//! The 2-consumer bar IS satisfied for the LogicalPlan side; however,
//! the cost-model walker is itself a single consumer, so the
//! abstraction does not yet carry its weight. M4-52 can introduce
//! the trait alongside its enumeration walker.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 / D-26 / D-28 — `LogicalPlan` taxonomy (the
//!   cost walker's input shape).
//! - ADR-038 §2 D-27 — selectivity estimators (M4-42; the cost
//!   walker's per-predicate selectivity source).
//! - ADR-038 §2 D-25 — catalog stats (M4-41; the cost walker's
//!   cardinality source via [`crate::semantic::CatalogProvider::snapshot`]).
//! - ADR-036 §D-25 — multi-step pipeline budget; M4-05 plan parse +
//!   cost row pins the **5 ms plan-build budget** the M4-51 walker
//!   must fit inside.
//! - ADR-038 amendment-02 §M4.e — M4-05 decomposition into M4-51 /
//!   M4-52 / M4-53.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 1 —
//!   M4-51 → M4-42 selectivity dep.

pub mod algorithm_picker;
pub mod cache;
pub mod cost;
pub mod enumeration;

pub use algorithm_picker::pick_join_algorithms;
pub use cache::{
    CachedPlan, DEFAULT_MAX_ENTRIES_PER_TENANT, LitKind, LookupOutcome, PlanCache, PlanCacheKey,
    Slot,
};
pub use enumeration::{
    DpFallbackReason, DpStats, JoinShape, MAX_DP_RELATIONS, enumerate_join_order,
};
