//! M4-71 row-count observer + M4-72 replan + plan-cache invalidation.
//!
//! Per ADR-038 amendment-02 §M4.g + amendment-03 §"Implicit dependency
//! edges" items 3 + 4.
//!
//! # Slice scope
//!
//! - **M4-71 (M4-07a — this slice)** — [`RowCountObserver`]: per-operator
//!   instrumentation collected at every dispatch boundary; emits
//!   structured `row_count` / `wall_time_ns` / `memory_bytes_high_water`
//!   per [`OperatorKind`] aggregate per amendment-03 §TIER-2-c. 10×
//!   threshold detection produces [`ThresholdBreach`] events when
//!   observed/estimated diverges by ≥10× either direction. Observed
//!   cardinalities feed back to M4-04 catalog stats per
//!   [`feedback::ObservedStatsOverrides`] — the consumer (production CLI
//!   wiring or test harness) applies the overrides to the catalog
//!   producer per the per-tenant boundary in ADR-037 §D-1.
//!
//! - **M4-72 (M4-07b — this slice)** — [`ReplanController`]: replan-from-
//!   current-operator mechanism. Reads the observer's accumulated
//!   threshold breaches, builds a synthetic [`crate::semantic::CatalogSnapshot`]
//!   with observed-card overrides, re-runs the planner pipeline (lower →
//!   enumerate → cost) under the synthetic snapshot, and signals the
//!   M4-53 plan cache to invalidate the original-plan entry per
//!   amendment-03 §"Implicit dependency edges" item 3. Replan does NOT
//!   re-acquire the snapshot LSN — original LSN is inherited per
//!   amendment-03 §TIER-1 GAP E rule 5.
//!
//! - **Plan-cache invalidation surface.** [`ReplanController`] consumes
//!   [`crate::planner::cache::PlanCache::invalidate`] directly via an
//!   `Option<Arc<PlanCache>>` field — NOT through a trait. The W12β
//!   review (PR #278 HIGH-1) caught a speculative trait extraction
//!   (`PlanCacheBackend`) shipping with only ONE real consumer; per
//!   `feedback_avoid_speculative_scaffolding.md` §"ship the abstraction
//!   when first CONSUMED, not when first imagined", the trait was
//!   removed in the W12β fix-up. Trait extraction is forward-deferred to
//!   v1.2+ persistent-cache or M5 multi-process-cache slices, the natural
//!   inflection point with a real second consumer.
//!
//! # Why a NEW module (not folded into `executor/` or `planner/`)?
//!
//! - **Cross-PR isolation.** W12α (M4-63 + M4-64a) ships in parallel and
//!   touches `executor/ops/` adding new operator variants; the observer
//!   wiring at the dispatcher in `executor/ops/mod.rs` is constrained to
//!   a single hook addition + an `op_kind()` method. The bulk of the
//!   M4-71 / M4-72 surface lives here in the new module so rebases
//!   stay clean.
//!
//! # 10× threshold semantics
//!
//! Per amendment-02 §M4.g:
//! - `observed >= threshold_factor × estimated` → [`BreachDirection::UnderEstimate`]
//!   (the planner under-estimated cardinality; replan with higher selectivity).
//! - `observed <= estimated / threshold_factor` → [`BreachDirection::OverEstimate`]
//!   (the planner over-estimated; replan with lower selectivity).
//! - Estimated cardinality `0` is special-cased: any observed > 0 with
//!   estimated == 0 is an UnderEstimate breach (the planner thought the
//!   query would return nothing).
//!
//! The threshold is configurable via [`RowCountObserver::with_threshold_factor`];
//! default is [`DEFAULT_THRESHOLD_FACTOR`] = 10.0 per the spec.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.g — primary M4-71 / M4-72 cite.
//! - ADR-038 amendment-03 §TIER-2-c — observability contract; per-operator
//!   structured-field metrics emission.
//! - ADR-038 amendment-03 §TIER-1 GAP E rule 5 — replan does NOT re-acquire
//!   snapshot LSN.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 3 — M4-72 →
//!   M4-53 invalidation channel.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 4 — M4-71 →
//!   M4-04 observed-stats feedback channel (per-tenant per ADR-037 §D-1).
//! - `feedback_avoid_speculative_scaffolding.md` — trait-extraction-at-
//!   first-CONSUMED discipline. Honored by routing replan invalidation
//!   directly through `Option<Arc<PlanCache>>` (one consumer at v1.0).
//! - Sin #5 PROFILE-with-cache pin + issue #262 dynamic transit closure.
//! - MED-2 strict producer→consumer transit pin + Sin #4 capacity-sweep
//!   bench. The "trait-extraction
//!   inflection point" sub-note is forward-deferred to v1.2+ per the
//!   W12β fix-up (HIGH-1).
//! - Deferred trait extraction lands with the second consumer.

pub mod dispatcher;
pub mod feedback;
pub mod replan;
pub mod row_count;
pub mod threshold;

pub use feedback::{ObservedStatsOverrides, apply_overrides_to_stub_catalog};
pub use replan::{MidQueryState, ReplanController, ReplanError, ReplanOutcome, ReplanReason};
pub use row_count::{
    DEFAULT_THRESHOLD_FACTOR, OperatorKind, OperatorMetrics, PlanWalkEntry, RowCountObserver,
    walk_plan_and_costs,
};
pub use threshold::{BreachDirection, ThresholdBreach};
