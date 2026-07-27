//! Traversal error taxonomy.
//!
//! Generic over the [`crate::EdgeSource::Error`] so adapter errors cross
//! the crate boundary losslessly (the consumer translates back at its own
//! public boundary per `docs/codec-error-translation.md`).

use thiserror::Error;

/// Errors surfaced by the traversal algorithms.
///
/// `#[non_exhaustive]` per the workspace error-enum convention
/// (code-quality policy): adding a variant is not a breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TraversalError<E> {
    /// The underlying [`crate::EdgeSource`] failed. Carried losslessly so
    /// the consumer can translate at its own boundary.
    #[error("edge source error: {0}")]
    Source(#[source] E),

    /// The cost budget tripped mid-traversal (the PRIM-1 H-1 discipline:
    /// the check lives INSIDE the expansion loop, never as a post-hoc
    /// filter). `cost_consumed` includes the item whose charge tripped the
    /// budget — matching `collect_reachable`'s shipped `bytes_consumed`
    /// accounting so the PRIM-1 fault-injection tests hold across the
    /// ADR-205 §D-5 refactor.
    #[error("cost budget exceeded: consumed {cost_consumed} of budget {cost_budget}")]
    CostBudgetExceeded {
        /// The configured budget (e.g. bytes for the PRIM-1 adapter).
        cost_budget: u64,
        /// Cost consumed at the trip point (over-approximation discipline:
        /// includes the tripping item).
        cost_consumed: u64,
    },

    /// A request parameter is structurally invalid (e.g. `k == 0` for
    /// k-shortest, zero-capacity reservoir). Surfaced as a structured
    /// error per `feedback_noop_trampoline_anti_pattern.md` — never a
    /// silent empty result.
    #[error("invalid traversal request: {reason}")]
    InvalidRequest {
        /// Human-readable rejection reason.
        reason: &'static str,
    },
}
