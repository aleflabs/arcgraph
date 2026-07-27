//! Vector-engine codec-local error type.
//!
//! Per the workspace pattern in `docs/codec-error-translation.md`,
//! this error never `impl From<…> for arcgraph_core::ArcGraphError`
//! directly. Translation to the workspace error happens at the
//! crate boundary that wires vector + storage + planner together
//! (the M3.a CRUD-equivalent surface, not yet present).

use thiserror::Error;

use arcgraph_core::TenantId;

use crate::{Encoding, IndexId, IndexType, Metric, VectorId};

/// Every error produced by the vector engine.
///
/// The variant set is intentionally small: per ADR-035 §9 most
/// failure modes either degenerate to `ArenaNotFound` (catalog
/// race), `Rebuilding` (replay window), or `IrrecoverableLoss`
/// (corruption beyond bootstrap-from-MVCC).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VectorIndexError {
    /// Query / vector dimension does not match the index dimension.
    /// Surfaces from the public search/insert APIs and from
    /// quantizer training.
    #[error("vector dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension the index was built for (catalog DDL value).
        expected: usize,
        /// Dimension of the offending input.
        got: usize,
    },

    /// No arena registered for the
    /// `(tenant, partition, index)` tuple. Either the index does
    /// not exist, or it was dropped out from under a stale handle.
    #[error("vector arena not found for tenant {tenant:?} index {index:?}")]
    ArenaNotFound {
        /// Tenant the handle was scoped to.
        tenant: TenantId,
        /// Index the handle was scoped to.
        index: IndexId,
    },

    /// The arena exists but the index is mid-rebuild — typically
    /// during quantizer training (ADR-035 §6.3) or post-replay
    /// bootstrap-from-MVCC (ADR-035 §4.6). Callers may retry
    /// after a backoff; the operation is not lost.
    #[error(
        "vector index for tenant {tenant:?} index {index:?} is rebuilding (kind={kind:?}); retry after backoff"
    )]
    Rebuilding {
        /// Tenant the handle was scoped to.
        tenant: TenantId,
        /// Index the handle was scoped to.
        index: IndexId,
        /// Which index type is being rebuilt.
        kind: IndexType,
    },

    /// Caller's `TenantId` does not match the arena's `TenantId`.
    /// Cross-tenant query attempt; rejected unconditionally per
    /// ADR-011 / ADR-035 §9.11.
    #[error(
        "cross-tenant vector access rejected: arena belongs to {arena_tenant:?}, caller is {caller_tenant:?}"
    )]
    TenantMismatch {
        /// Tenant the arena belongs to.
        arena_tenant: TenantId,
        /// Tenant the caller presented.
        caller_tenant: TenantId,
    },

    /// Encoding × metric combination is not supported by any
    /// kernel (e.g., Hamming on F32). Caught at dispatch time;
    /// distinct from a missing kernel for a supported pair.
    #[error("unsupported vector flags: encoding={encoding:?} metric={metric:?}")]
    UnsupportedFlags {
        /// Encoding of the index / arena.
        encoding: Encoding,
        /// Metric the caller requested.
        metric: Metric,
    },

    /// Replay / recovery exhausted both the snapshot path and the
    /// bootstrap-from-MVCC fallback (ADR-035 §4.6 B-1 resolution).
    /// Operator intervention required: either the MVCC store is
    /// itself corrupt, or the arena's vectors live outside MVCC
    /// (a class that v1.0 does not support).
    #[error("vector index irrecoverably lost for index {index:?}: {reason}")]
    IrrecoverableLoss {
        /// Index that could not be recovered.
        index: IndexId,
        /// Operator-facing diagnostic.
        reason: String,
    },

    /// `rescore_factor` parameter is invalid (must be ≥ 1). Per
    /// ADR-035 D-4, `rescore_factor = 1` is the operator opt-out
    /// (SQ8-alone, AC-1b best-effort), not an error; `0` is the
    /// only invalid value at the public surface. Surfaces from
    /// `search_with_rescore` on HNSW + DiskANN per Slice E.2 / E.3.
    #[error("invalid rescore_factor: {factor} (must be ≥ 1; 1 disables rescore)")]
    InvalidRescoreFactor {
        /// Offending factor value.
        factor: usize,
    },

    /// Rescore arena is missing a full-precision vector that the
    /// primary index reported as a candidate. Per ADR-035 §3.3
    /// rescore path the arena's `rescore_vectors` view MUST cover
    /// every live `VectorId` the primary index can return; a
    /// missing entry means the rescore arena drifted out of sync
    /// with the primary index. Operator-visible inconsistency;
    /// the recovery path is to rebuild the rescore arena from
    /// the primary arena (Slice F.1 follow-up). Surfaces from
    /// `search_with_rescore` on HNSW + DiskANN per Slice E.2 /
    /// E.3.
    #[error(
        "rescore vector missing for vector_id {vector_id:?}: rescore arena out of sync with primary index"
    )]
    RescoreVectorMissing {
        /// Vector id the primary index returned but the rescore
        /// arena could not resolve.
        vector_id: VectorId,
    },

    /// A streaming-insert / merge-fold batch failed pre-flight
    /// validation (dimension mismatch or non-finite float entry).
    ///
    /// Per issue #109 part 2: prior to this variant,
    /// [`crate::diskann::DiskAnnGraph::merge_delta`] drained its
    /// delta-segment via `mem::take` BEFORE validating each entry,
    /// then validated inside the merge loop. Any mid-batch
    /// validation failure (dim mismatch / NaN / Inf in a single
    /// entry) lost every remaining entry in the delta — silently
    /// violating the I-V7 RYW invariant for a successfully ack'd
    /// T3 insert that shared the same later-merged batch as the
    /// malformed entry.
    ///
    /// `merge_delta` now pre-validates the entire batch before
    /// taking it; on any failure it leaves the delta intact and
    /// returns this error. The caller may inspect `reason`,
    /// remove the offending entry (e.g., via
    /// [`crate::diskann::DiskAnnGraph::delete`]), and retry.
    /// Operator-visible; not retryable as-is.
    #[error("vector batch failed pre-flight validation: {reason}")]
    BatchValidation {
        /// Human-readable diagnostic naming the offending entry
        /// index and the validation rule it violated.
        reason: String,
    },

    /// A backend received a [`crate::Filter`] variant it does not
    /// support at v1.0.
    ///
    /// Per ADR-035 §6 + amendment-03 (issue #127), the canonical
    /// [`crate::Filter`] enum carries seven variants but each
    /// backend's v1.0 capability is narrower: HNSW (Slice F.2)
    /// supports the full enum; DiskANN (Slice F.3) supports only
    /// [`crate::Filter::Any`] + [`crate::Filter::LabelEq`] (its
    /// per-label entry-point cache hot path).
    ///
    /// The Phase 6 F.4 selectivity dispatcher inspects this error
    /// and re-routes the query to a capable backend (i.e., HNSW
    /// for compound filters); the F.5 / G.4 follow-up adds a
    /// per-label inverted index that lets DiskANN handle
    /// [`crate::Filter::LabelIn`] and the `And` / `Or` closure
    /// directly, at which point the variants currently raising
    /// this error become supported.
    ///
    /// `reason` is a human-readable diagnostic naming the
    /// offending variant and pointing at the F.4/F.5 escalation
    /// path. Operator-visible; not retryable (the planner must
    /// reshape the query, not the runtime).
    #[error("vector backend rejected unsupported filter: {reason}")]
    UnsupportedFilter {
        /// Diagnostic: which variant was rejected, why, and what
        /// dispatcher escalation path applies.
        reason: String,
    },

    /// The SSD-resident DiskANN serving tier's RSS guard observed a
    /// process resident-set size above the configured cap
    /// (`ARCGRAPH_VECTOR_RSS_CAP_MB`, default 14000) and aborted the
    /// run CLEANLY at the next safe checkpoint.
    ///
    /// Per ADR-195 §2.2 / §4: the bounded [`arcgraph_storage`] buffer
    /// pool + the bounded-batch build PREVENT the steady-state and
    /// build-time breaches; this error is the detect-and-abort
    /// BACKSTOP for a transient spike — fail-CLEAN, no swap-thrash to
    /// an OOM-kill. `observed_mb` is the real high-water mark the
    /// sampler recorded (the honest RSS number for the §6 report).
    /// Operator-visible; the recovery path is to lower the working set
    /// (smaller buffer-pool `frame_count`, PQ-nav per ADR-195 §2.1) or
    /// raise the cap on a larger box, then re-run.
    #[error(
        "SSD DiskANN RSS guard tripped: observed {observed_mb} MB > cap {cap_mb} MB \
         (ARCGRAPH_VECTOR_RSS_CAP_MB); aborted cleanly per ADR-195 §2.2"
    )]
    RssCapExceeded {
        /// Observed process-RSS high-water mark in MB.
        observed_mb: u64,
        /// Configured cap in MB.
        cap_mb: u64,
    },
}

impl VectorIndexError {
    /// Whether the caller may retry the operation against a fresh
    /// state. Used by higher-level retry policy.
    ///
    /// - `Rebuilding` → retry after backoff (rebuild completes).
    /// - everything else → do not retry (caller-side bug or
    ///   permanent loss).
    #[inline]
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Rebuilding { .. })
    }

    /// The default unit type for a placeholder
    /// [`VectorIndexError`] used when constructing a
    /// `VectorId::ZERO`-shaped error in tests. **Not** part of
    /// the production surface.
    #[cfg(test)]
    #[must_use]
    pub fn dim_mismatch_768_vs_512() -> Self {
        Self::DimensionMismatch {
            expected: 768,
            got: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_dimension_mismatch_includes_both_sizes() {
        let e = VectorIndexError::DimensionMismatch {
            expected: 768,
            got: 384,
        };
        let s = format!("{e}");
        assert!(s.contains("768"), "got: {s}");
        assert!(s.contains("384"), "got: {s}");
    }

    #[test]
    fn display_arena_not_found_includes_index() {
        let e = VectorIndexError::ArenaNotFound {
            tenant: TenantId::new(7),
            index: IndexId::new(42),
        };
        let s = format!("{e}");
        assert!(s.contains("42"), "got: {s}");
    }

    #[test]
    fn display_rebuilding_includes_kind() {
        let e = VectorIndexError::Rebuilding {
            tenant: TenantId::new(1),
            index: IndexId::new(2),
            kind: IndexType::Hnsw,
        };
        let s = format!("{e}");
        assert!(s.contains("Hnsw"), "got: {s}");
        assert!(s.contains("retry"), "got: {s}");
    }

    #[test]
    fn display_tenant_mismatch_distinguishes_sides() {
        let e = VectorIndexError::TenantMismatch {
            arena_tenant: TenantId::new(1),
            caller_tenant: TenantId::new(2),
        };
        let s = format!("{e}");
        assert!(s.contains("Cross") || s.contains("cross"), "got: {s}");
    }

    #[test]
    fn display_unsupported_flags_lists_pair() {
        let e = VectorIndexError::UnsupportedFlags {
            encoding: Encoding::F32,
            metric: Metric::Hamming,
        };
        let s = format!("{e}");
        assert!(s.contains("F32"), "got: {s}");
        assert!(s.contains("Hamming"), "got: {s}");
    }

    #[test]
    fn display_irrecoverable_loss_includes_reason() {
        let e = VectorIndexError::IrrecoverableLoss {
            index: IndexId::new(99),
            reason: "snapshot CRC + MVCC both failed".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("99"), "got: {s}");
        assert!(s.contains("CRC"), "got: {s}");
    }

    #[test]
    fn rebuilding_is_retryable() {
        let e = VectorIndexError::Rebuilding {
            tenant: TenantId::DEFAULT,
            index: IndexId::ZERO,
            kind: IndexType::DiskAnn,
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn other_variants_are_not_retryable() {
        for e in [
            VectorIndexError::DimensionMismatch {
                expected: 768,
                got: 1,
            },
            VectorIndexError::ArenaNotFound {
                tenant: TenantId::DEFAULT,
                index: IndexId::ZERO,
            },
            VectorIndexError::TenantMismatch {
                arena_tenant: TenantId::DEFAULT,
                caller_tenant: TenantId::SYSTEM,
            },
            VectorIndexError::UnsupportedFlags {
                encoding: Encoding::Binary,
                metric: Metric::L2,
            },
            VectorIndexError::IrrecoverableLoss {
                index: IndexId::ZERO,
                reason: "x".to_owned(),
            },
            VectorIndexError::InvalidRescoreFactor { factor: 0 },
            VectorIndexError::RescoreVectorMissing {
                vector_id: VectorId::ZERO,
            },
            VectorIndexError::UnsupportedFilter {
                reason: "test".to_owned(),
            },
            VectorIndexError::BatchValidation {
                reason: "test".to_owned(),
            },
        ] {
            assert!(!e.is_retryable(), "expected non-retryable: {e}");
        }
    }

    #[test]
    fn display_unsupported_filter_includes_reason() {
        let e = VectorIndexError::UnsupportedFilter {
            reason: "DiskANN v1.0 does not support compound filters".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("compound"), "got: {s}");
        assert!(s.contains("filter"), "got: {s}");
    }

    #[test]
    fn display_invalid_rescore_factor_includes_factor() {
        let e = VectorIndexError::InvalidRescoreFactor { factor: 0 };
        let s = format!("{e}");
        assert!(s.contains("0"), "got: {s}");
        assert!(s.contains("rescore"), "got: {s}");
    }

    #[test]
    fn display_batch_validation_includes_reason() {
        let e = VectorIndexError::BatchValidation {
            reason: "entry 3 contains NaN".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("entry 3"), "got: {s}");
        assert!(s.contains("validation"), "got: {s}");
    }

    #[test]
    fn display_rescore_vector_missing_includes_vector_id() {
        let e = VectorIndexError::RescoreVectorMissing {
            vector_id: VectorId::new(7),
        };
        let s = format!("{e}");
        assert!(s.contains("7"), "got: {s}");
    }

    #[test]
    fn pattern_match_compiles_for_each_variant() {
        // Compile-time: every variant is reachable from a match
        // arm at the public API. Future variants must update this.
        let e = VectorIndexError::dim_mismatch_768_vs_512();
        match e {
            VectorIndexError::DimensionMismatch { .. } => {}
            VectorIndexError::ArenaNotFound { .. }
            | VectorIndexError::Rebuilding { .. }
            | VectorIndexError::TenantMismatch { .. }
            | VectorIndexError::UnsupportedFlags { .. }
            | VectorIndexError::IrrecoverableLoss { .. }
            | VectorIndexError::InvalidRescoreFactor { .. }
            | VectorIndexError::RescoreVectorMissing { .. }
            | VectorIndexError::UnsupportedFilter { .. }
            | VectorIndexError::RssCapExceeded { .. }
            | VectorIndexError::BatchValidation { .. } => {
                panic!("unexpected variant from helper")
            }
        }
    }
}
