//! Community-engine codec-local error type.
//!
//! Per the workspace pattern in `docs/codec-error-translation.md`,
//! this error never `impl From<…> for arcgraph_core::ArcGraphError`
//! directly. Translation to the workspace error happens at the
//! crate boundary that wires community + storage + planner together
//! (the M3.d-1 router accessor; see ADR-040 §D-7).
//!
//! The variant set follows ADR-040 §D-3 (API surface) and §5
//! (Consequences). Under the code-quality policy every variant is a
//! `thiserror`-derived enum entry, the enum is `#[non_exhaustive]`
//! so future variants do not break downstream pattern matches,
//! and `is_retryable` is the const callable that retry policy
//! consumes.

use thiserror::Error;

use arcgraph_core::TenantId;

use crate::ids::{CommunityId, Level};

/// Every error produced by the community-detection engine.
///
/// The variant set is intentionally small per ADR-040 §D-3 / §5:
/// most failure modes degenerate to either `Refreshing` (retryable
/// during a static refresh window), `IndexNotReady` (first refresh
/// pending), or one of the structural mismatches (`TenantMismatch`,
/// `UnknownCommunity`, `UnknownLevel`).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CommunityError {
    /// Caller's `TenantId` does not match the handle's `TenantId`.
    /// Cross-tenant query attempt; rejected unconditionally per
    /// ADR-011 / ADR-040 §D-8.
    #[error(
        "cross-tenant community access rejected: handle belongs to {handle_tenant:?}, query is {query_tenant:?}"
    )]
    TenantMismatch {
        /// Tenant the handle was scoped to.
        handle_tenant: TenantId,
        /// Tenant the caller presented.
        query_tenant: TenantId,
    },

    /// The membership index has not yet completed its first
    /// static refresh for this tenant. Surfaces from queries
    /// issued before the daily-refresh scheduler (ADR-040 §D-7)
    /// has produced any communities. Not retryable at the engine
    /// surface; the caller must wait for the refresh to land.
    #[error("community index not ready for tenant {tenant:?}: {reason}")]
    IndexNotReady {
        /// Tenant the handle was scoped to.
        tenant: TenantId,
        /// Operator-facing diagnostic.
        reason: String,
    },

    /// A static refresh is in progress; queries are temporarily
    /// blocked while the new generation is published per ADR-040
    /// §D-6. **Retryable**: callers may retry after a short
    /// backoff; the operation is not lost.
    #[error("community index refresh in progress for tenant {tenant:?}; retry after backoff")]
    Refreshing {
        /// Tenant whose index is mid-refresh.
        tenant: TenantId,
    },

    /// Caller asked for members of a `CommunityId` that does not
    /// exist at the given `Level`. Surfaces from `members()` per
    /// ADR-040 §D-3.
    #[error("unknown community {community:?} at level {level:?} for tenant {tenant:?}")]
    UnknownCommunity {
        /// Tenant the handle was scoped to.
        tenant: TenantId,
        /// The community id that was requested.
        community: CommunityId,
        /// The hierarchy level that was requested.
        level: Level,
    },

    /// Caller asked for a `Level` beyond the current Leiden
    /// hierarchy depth for this tenant. Surfaces from any of the
    /// three retrieval methods on `CommunityIndexHandle`.
    #[error("unknown level {level:?} for tenant {tenant:?}: hierarchy max is {max_level:?}")]
    UnknownLevel {
        /// Tenant the handle was scoped to.
        tenant: TenantId,
        /// The level that was requested.
        level: Level,
        /// The maximum level present in the hierarchy.
        max_level: Level,
    },

    /// `rank_by_seeds` was called with no seeds. The score function
    /// per ADR-040 §D-3 is undefined on the empty seed set
    /// (`0/N` collapses to a uniform-zero ranking, which is not
    /// what callers want); reject at the boundary instead.
    #[error("rank_by_seeds called with empty seeds set")]
    EmptySeeds,

    /// Replay / recovery exhausted both the snapshot path and the
    /// membership-index reconstruction fallback. Operator
    /// intervention required: a full refresh must be triggered
    /// to re-establish the membership index.
    #[error("community membership index irrecoverably lost for tenant {tenant:?}: {reason}")]
    IrrecoverableLoss {
        /// Tenant whose index was lost.
        tenant: TenantId,
        /// Operator-facing diagnostic.
        reason: String,
    },
}

impl CommunityError {
    /// Whether the caller may retry the operation against a fresh
    /// state. Used by higher-level retry policy.
    ///
    /// - `Refreshing` → retry after backoff (refresh completes).
    /// - everything else → do not retry (caller-side bug,
    ///   structural mismatch, or permanent loss).
    #[inline]
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Refreshing { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_tenant_mismatch_distinguishes_sides() {
        let e = CommunityError::TenantMismatch {
            handle_tenant: TenantId::new(1),
            query_tenant: TenantId::new(2),
        };
        let s = format!("{e}");
        assert!(s.contains("Cross") || s.contains("cross"), "got: {s}");
    }

    #[test]
    fn display_index_not_ready_includes_tenant_and_reason() {
        let e = CommunityError::IndexNotReady {
            tenant: TenantId::new(7),
            reason: "first refresh pending".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("not ready"), "got: {s}");
        assert!(s.contains("first refresh"), "got: {s}");
    }

    #[test]
    fn display_refreshing_includes_retry_hint() {
        let e = CommunityError::Refreshing {
            tenant: TenantId::DEFAULT,
        };
        let s = format!("{e}");
        assert!(s.contains("refresh"), "got: {s}");
        assert!(s.contains("retry"), "got: {s}");
    }

    #[test]
    fn display_unknown_community_includes_community_id() {
        let e = CommunityError::UnknownCommunity {
            tenant: TenantId::DEFAULT,
            community: CommunityId::new(42),
            level: Level::FINEST,
        };
        let s = format!("{e}");
        assert!(s.contains("42"), "got: {s}");
        assert!(s.contains("unknown community"), "got: {s}");
    }

    #[test]
    fn display_unknown_level_includes_max() {
        let e = CommunityError::UnknownLevel {
            tenant: TenantId::DEFAULT,
            level: Level::new(7),
            max_level: Level::new(3),
        };
        let s = format!("{e}");
        assert!(s.contains("unknown level"), "got: {s}");
        // Both the requested and the max should print.
        assert!(s.contains('7'), "got: {s}");
        assert!(s.contains('3'), "got: {s}");
    }

    #[test]
    fn display_empty_seeds_mentions_seeds() {
        let e = CommunityError::EmptySeeds;
        let s = format!("{e}");
        assert!(s.contains("empty"), "got: {s}");
        assert!(s.contains("seeds"), "got: {s}");
    }

    #[test]
    fn display_irrecoverable_loss_includes_reason() {
        let e = CommunityError::IrrecoverableLoss {
            tenant: TenantId::DEFAULT,
            reason: "snapshot CRC mismatch".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("irrecoverably"), "got: {s}");
        assert!(s.contains("CRC"), "got: {s}");
    }

    #[test]
    fn refreshing_is_retryable() {
        let e = CommunityError::Refreshing {
            tenant: TenantId::DEFAULT,
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn other_variants_are_not_retryable() {
        for e in [
            CommunityError::TenantMismatch {
                handle_tenant: TenantId::new(1),
                query_tenant: TenantId::new(2),
            },
            CommunityError::IndexNotReady {
                tenant: TenantId::DEFAULT,
                reason: "x".to_owned(),
            },
            CommunityError::UnknownCommunity {
                tenant: TenantId::DEFAULT,
                community: CommunityId::ZERO,
                level: Level::FINEST,
            },
            CommunityError::UnknownLevel {
                tenant: TenantId::DEFAULT,
                level: Level::new(9),
                max_level: Level::FINEST,
            },
            CommunityError::EmptySeeds,
            CommunityError::IrrecoverableLoss {
                tenant: TenantId::DEFAULT,
                reason: "x".to_owned(),
            },
        ] {
            assert!(!e.is_retryable(), "expected non-retryable: {e}");
        }
    }

    #[test]
    fn pattern_match_compiles_for_each_variant() {
        // Compile-time: every variant is reachable from a match
        // arm at the public API. Future variants must update this.
        let e = CommunityError::EmptySeeds;
        match e {
            CommunityError::EmptySeeds => {}
            CommunityError::TenantMismatch { .. }
            | CommunityError::IndexNotReady { .. }
            | CommunityError::Refreshing { .. }
            | CommunityError::UnknownCommunity { .. }
            | CommunityError::UnknownLevel { .. }
            | CommunityError::IrrecoverableLoss { .. } => {
                panic!("unexpected variant from helper")
            }
        }
    }
}
