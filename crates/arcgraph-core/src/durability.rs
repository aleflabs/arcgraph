//! Per-tenant durability tiers.
//!
//! [`DurabilityTier`] is the v1.0 per-tenant durability knob.
//!
//! Two tiers ship:
//!
//! - [`DurabilityTier::Strict`] ("T1") — fsync-per-commit before ack.
//!   Zero-data-loss contract modulo hardware precondition (see ADR-034
//!   §8.5). The pre-ADR-034 shape; default for every tenant.
//! - [`DurabilityTier::Periodic`] ("T3") — WAL append before ack,
//!   background fsync within `rpo_ms` after ack. Up to `rpo_ms` of
//!   recent commits may be lost on crash. Use for replayable batch
//!   workloads (log ingest, telemetry).
//!
//! T2 (`F_BARRIERFSYNC`) and T4 (no-fsync) are explicitly deferred to
//! v1.1; see ADR-034 §4.
//!
//! The tier is a **writer-side concept only**. ADR-031 CommitBundle
//! bytes are identical across tiers (D-3); ADR-032 replay is
//! tier-agnostic (I-D6). Tier choice shapes only **when** bytes
//! reach disk, not **what** bytes are written or **how** replay
//! interprets them.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::TenantId;

/// Per-tenant durability tier. Stored on the tenant's catalog record
/// (see `arcgraph_storage::catalog::TenantRecord`).
///
/// Serialized to JSON for diagnostics / config files as `{"tier":
/// "strict"}` or `{"tier": "periodic", "rpo_ms": 100}` via the
/// `#[serde(tag)]` form. The binary on-disk MVCC encoding (the
/// per-tenant catalog entry) is maintained separately by the catalog
/// module — serde is for non-load-bearing config paths only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case")]
pub enum DurabilityTier {
    /// T1 — fsync-per-commit before ack. Zero-data-loss modulo
    /// hardware. Default for every tenant at v1.0.
    Strict,

    /// T3 — WAL append before ack, background fsync within `rpo_ms`
    /// after ack. Up to `rpo_ms` of recent commits may be lost on
    /// crash.
    ///
    /// `rpo_ms` must be in `[MIN_T3_RPO_MS, MAX_T3_RPO_MS]`; see
    /// [`DurabilityTier::validate`].
    Periodic {
        /// Upper bound on recoverable-crash data-loss in
        /// milliseconds. Observed loss is between 0 (piggyback by
        /// a T1 commit or clean fsync-tick) and this bound.
        rpo_ms: u64,
    },
}

impl Default for DurabilityTier {
    /// Default is [`Self::Strict`]: pre-existing deployments upgraded
    /// to ADR-034 have every tenant on T1 automatically (D-1).
    #[inline]
    fn default() -> Self {
        Self::Strict
    }
}

impl DurabilityTier {
    /// Recommended RPO for T3 tenants when no explicit value is
    /// supplied: 100 ms. Matches the expected scheduler interval
    /// for a mid-volume log-ingest deployment.
    pub const DEFAULT_T3_RPO_MS: u64 = 100;

    /// Minimum accepted `rpo_ms` for T3. Below this, T3 ≈ T1 — the
    /// scheduler ticks faster than the fsync completes, so throughput
    /// degenerates to synchronous fsync with extra bookkeeping.
    /// Operators who want T1 semantics configure [`Self::Strict`].
    pub const MIN_T3_RPO_MS: u64 = 10;

    /// Maximum accepted `rpo_ms` for T3. Above this, operator risk
    /// increases non-linearly (longer windows of in-memory-only
    /// commits, larger post-crash regressions). Accepted with a
    /// `tracing::warn!` at the catalog layer.
    pub const MAX_T3_RPO_MS: u64 = 60_000;

    /// Validate the tier's invariants. Returns `Ok(())` if the tier
    /// is acceptable.
    ///
    /// - [`Self::Strict`]: always valid.
    /// - [`Self::Periodic`] with `rpo_ms` in
    ///   `[MIN_T3_RPO_MS, MAX_T3_RPO_MS]`: valid.
    /// - Out-of-range `rpo_ms`: [`DurabilityTierError`].
    #[inline]
    pub fn validate(&self) -> Result<(), DurabilityTierError> {
        match self {
            Self::Strict => Ok(()),
            Self::Periodic { rpo_ms } => {
                if *rpo_ms < Self::MIN_T3_RPO_MS {
                    Err(DurabilityTierError::RpoTooSmall {
                        got: *rpo_ms,
                        min: Self::MIN_T3_RPO_MS,
                    })
                } else if *rpo_ms > Self::MAX_T3_RPO_MS {
                    Err(DurabilityTierError::RpoTooLarge {
                        got: *rpo_ms,
                        max: Self::MAX_T3_RPO_MS,
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    /// `true` iff this is the strict / T1 tier.
    #[inline]
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// `true` iff this is the periodic / T3 tier.
    #[inline]
    #[must_use]
    pub const fn is_periodic(&self) -> bool {
        matches!(self, Self::Periodic { .. })
    }

    /// The RPO in milliseconds if this is [`Self::Periodic`], else
    /// `None`. Strict tier has no RPO bound (durability is synchronous).
    #[inline]
    #[must_use]
    pub const fn rpo_ms(&self) -> Option<u64> {
        match self {
            Self::Strict => None,
            Self::Periodic { rpo_ms } => Some(*rpo_ms),
        }
    }

    /// Short stable name (`"strict"` or `"periodic"`) for logs,
    /// metrics labels, and error messages. Does NOT encode `rpo_ms`
    /// — use [`Self::rpo_ms`] alongside for full context.
    #[inline]
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Periodic { .. } => "periodic",
        }
    }
}

/// Errors returned by [`DurabilityTier::validate`] and by the
/// catalog layer's `set_durability_tier` surface.
///
/// Kept as a narrow enum in `arcgraph-core` so it can be re-exported
/// from the catalog crate without pulling in catalog-specific
/// dependencies.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurabilityTierError {
    /// Configured `rpo_ms` is below [`DurabilityTier::MIN_T3_RPO_MS`].
    /// Operators who want T1 semantics should use
    /// [`DurabilityTier::Strict`] directly.
    #[error(
        "T3 rpo_ms {got} below MIN_T3_RPO_MS ({min}); use DurabilityTier::Strict for T1 semantics"
    )]
    RpoTooSmall {
        /// The configured value.
        got: u64,
        /// The minimum accepted value.
        min: u64,
    },

    /// Configured `rpo_ms` exceeds
    /// [`DurabilityTier::MAX_T3_RPO_MS`]. This is a hard ceiling;
    /// operators needing looser bounds should file an ADR amendment.
    #[error("T3 rpo_ms {got} exceeds MAX_T3_RPO_MS ({max})")]
    RpoTooLarge {
        /// The configured value.
        got: u64,
        /// The maximum accepted value.
        max: u64,
    },

    /// Attempted to configure the SYSTEM tenant to any tier other
    /// than [`DurabilityTier::Strict`]. SYSTEM is the catalog tenant;
    /// losing a tier-change or bootstrap commit would silently corrupt
    /// the post-crash catalog state. T1 is therefore enforced.
    ///
    /// See ADR-034 §I-D7 and §10 invariant D-4.
    #[error("SYSTEM tenant must be Strict (T1); non-configurable per ADR-034 I-D7")]
    SystemTenantMustBeStrict,

    /// Attempted to set the tier for a tenant that isn't in the
    /// catalog. Callers should register the tenant first (v1.1 DDL)
    /// or use the DEFAULT tenant at v1.0.
    #[error("tenant not found in catalog: {tenant_raw}")]
    TenantNotFound {
        /// Raw `TenantId` value that was not found.
        tenant_raw: u64,
    },
}

/// Trait for resolving the current [`DurabilityTier`] of a tenant.
///
/// Implemented by `arcgraph_storage::catalog::SystemCatalog` (the
/// only v1.0 implementer). Pulled into `arcgraph-core` so the MVCC
/// kernel (`TxnManager` in `arcgraph-storage::transaction`) can
/// query the tier at commit time without importing the catalog
/// directly — keeps the bounded-context seam clean (see
/// `docs/bounded-contexts.md`).
///
/// **Semantics (ADR-034 §I-D7):** callers invoke
/// [`Self::durability_tier`] at commit TIME (Phase 2 of the
/// three-phase commit), not at transaction begin. This matches the
/// operator intuition "flip the tier, the next commit uses it" and
/// is the simplest correctness story for in-flight transactions.
///
/// **`TenantId::SYSTEM` MUST return [`DurabilityTier::Strict`]**
/// regardless of implementation — the SYSTEM tenant is T1-enforced
/// per I-D7. v1.0 implementers short-circuit SYSTEM before any
/// lookup.
pub trait TenantDurabilityLookup: Send + Sync {
    /// Look up the current durability tier for `tenant`.
    ///
    /// Returns [`DurabilityTier::Strict`] for unknown tenants and
    /// for [`TenantId::SYSTEM`] (both cases are I-D7 safe-harbor
    /// behaviour).
    fn durability_tier(&self, tenant: TenantId) -> DurabilityTier;
}

/// A trivial implementer that returns [`DurabilityTier::Strict`]
/// for every tenant. Useful as the default for `TxnManager`
/// instances constructed without a catalog (ad-hoc tests,
/// pre-bootstrap code paths). Zero-overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlwaysStrict;

impl TenantDurabilityLookup for AlwaysStrict {
    #[inline]
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Strict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Default / name / is_* ────────────────────────────────────

    #[test]
    fn default_is_strict() {
        assert_eq!(DurabilityTier::default(), DurabilityTier::Strict);
        assert!(DurabilityTier::default().is_strict());
        assert!(!DurabilityTier::default().is_periodic());
    }

    #[test]
    fn strict_has_no_rpo() {
        assert_eq!(DurabilityTier::Strict.rpo_ms(), None);
    }

    #[test]
    fn periodic_exposes_rpo() {
        let t = DurabilityTier::Periodic { rpo_ms: 250 };
        assert_eq!(t.rpo_ms(), Some(250));
        assert!(t.is_periodic());
        assert!(!t.is_strict());
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(DurabilityTier::Strict.name(), "strict");
        assert_eq!(DurabilityTier::Periodic { rpo_ms: 100 }.name(), "periodic");
    }

    // ─── Validate: min / max / defaults ──────────────────────────

    #[test]
    fn validate_strict_always_ok() {
        assert_eq!(DurabilityTier::Strict.validate(), Ok(()));
    }

    #[test]
    fn validate_periodic_default_rpo_ok() {
        let t = DurabilityTier::Periodic {
            rpo_ms: DurabilityTier::DEFAULT_T3_RPO_MS,
        };
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn validate_periodic_min_ok() {
        let t = DurabilityTier::Periodic {
            rpo_ms: DurabilityTier::MIN_T3_RPO_MS,
        };
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn validate_periodic_max_ok() {
        let t = DurabilityTier::Periodic {
            rpo_ms: DurabilityTier::MAX_T3_RPO_MS,
        };
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn validate_periodic_below_min_rejected() {
        let t = DurabilityTier::Periodic { rpo_ms: 0 };
        assert!(matches!(
            t.validate(),
            Err(DurabilityTierError::RpoTooSmall { got: 0, min: 10 })
        ));
    }

    #[test]
    fn validate_periodic_just_below_min_rejected() {
        let t = DurabilityTier::Periodic {
            rpo_ms: DurabilityTier::MIN_T3_RPO_MS - 1,
        };
        assert!(matches!(
            t.validate(),
            Err(DurabilityTierError::RpoTooSmall { .. })
        ));
    }

    #[test]
    fn validate_periodic_above_max_rejected() {
        let t = DurabilityTier::Periodic {
            rpo_ms: DurabilityTier::MAX_T3_RPO_MS + 1,
        };
        assert!(matches!(
            t.validate(),
            Err(DurabilityTierError::RpoTooLarge { .. })
        ));
    }

    #[test]
    fn validate_periodic_way_above_max_rejected() {
        let t = DurabilityTier::Periodic { rpo_ms: u64::MAX };
        assert!(matches!(
            t.validate(),
            Err(DurabilityTierError::RpoTooLarge { .. })
        ));
    }

    // ─── Error display (operator-facing) ─────────────────────────

    #[test]
    fn error_display_rpo_too_small_is_actionable() {
        let e = DurabilityTierError::RpoTooSmall { got: 5, min: 10 };
        let s = format!("{e}");
        assert!(s.contains("5"), "got: {s}");
        assert!(s.contains("10"), "got: {s}");
        assert!(
            s.to_lowercase().contains("strict"),
            "should point operators to Strict: {s}"
        );
    }

    #[test]
    fn error_display_rpo_too_large_names_bound() {
        let e = DurabilityTierError::RpoTooLarge {
            got: 120_000,
            max: 60_000,
        };
        let s = format!("{e}");
        assert!(s.contains("120000"), "got: {s}");
        assert!(s.contains("60000"), "got: {s}");
    }

    #[test]
    fn error_display_system_must_be_strict_references_adr() {
        let e = DurabilityTierError::SystemTenantMustBeStrict;
        let s = format!("{e}");
        assert!(s.contains("SYSTEM"), "got: {s}");
        // ADR-034 invariant I-D7 — reference kept in the error for
        // operator diagnosis.
        assert!(s.contains("ADR-034"), "got: {s}");
    }

    // ─── Copy + equality semantics ───────────────────────────────

    #[test]
    fn is_copy() {
        let t = DurabilityTier::Periodic { rpo_ms: 100 };
        let u = t; // Copy
        assert_eq!(t, u);
    }

    #[test]
    fn equality_discriminates_rpo() {
        assert_ne!(
            DurabilityTier::Periodic { rpo_ms: 100 },
            DurabilityTier::Periodic { rpo_ms: 200 }
        );
        assert_ne!(
            DurabilityTier::Strict,
            DurabilityTier::Periodic { rpo_ms: 100 }
        );
    }

    #[test]
    fn hash_is_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let t1 = DurabilityTier::Periodic { rpo_ms: 100 };
        let t2 = DurabilityTier::Periodic { rpo_ms: 100 };
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        t1.hash(&mut h1);
        t2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
