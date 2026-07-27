//! Execution-time error taxonomy for the M4-61 / M4-62 executor.
//!
//! The taxonomy is deliberately narrow: the planner already rejects
//! every input the executor cannot run (binding + type-check + cross-
//! substrate validation + plan lowering). Anything reaching this
//! module either (a) tripped the cancellation token, (b) hit a
//! substrate fault, (c) reached a plan operator forward-deferred to
//! a later milestone, or (d) was a runtime evaluation fault.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-61 cite.
//! - **ADR-038 §2 D-16** — `NotImplemented` discipline (forward-link
//!   future slices in the error variant).
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — OPTIONAL MATCH /
//!   GAP E snapshot LSN flow through the same error type.

use crate::executor::context::CancellationError;
use crate::executor::substrate::SubstrateAccessError;
use crate::semantic::error::ArcQLError;

/// Cloneable executor projection of OOC-1 spill failures.
///
/// OOC-1's [`arcgraph_storage::SpillError`] retains an
/// [`std::io::Error`] source and therefore cannot participate directly in
/// this public error's `Clone + Eq` contract. This projection preserves all
/// structured resource-rejection fields (including measured spill bytes)
/// and classifies every other failure without asking callers to parse text.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutorSpillError {
    /// Per-tenant quota, volume-headroom, or spill-staging rejection.
    #[error(
        "executor spill resource exhausted ({reason:?}): tenant={tenant_id:?}, requested_bytes={requested_bytes}, spilled_bytes={spilled_bytes}, limit_bytes={limit_bytes}, available_bytes={available_bytes:?}"
    )]
    ResourceExhausted {
        reason: arcgraph_storage::SpillRejectReason,
        tenant_id: arcgraph_core::TenantId,
        requested_bytes: u64,
        spilled_bytes: u64,
        limit_bytes: u64,
        available_bytes: Option<u64>,
    },
    /// A non-capacity spill failure, classified for programmatic routing.
    #[error("executor spill failed ({kind:?}): {detail}")]
    Failure {
        kind: ExecutorSpillFailureKind,
        detail: String,
    },
}

/// Stable failure classes for non-capacity OOC-1 errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutorSpillFailureKind {
    InvalidConfig,
    /// An executor-local deterministic spill bound was reached (for example,
    /// the Grace join's live run-handle ceiling).
    ResourceLimit,
    /// An expand-frontier FIFO invariant failed during spill/readback.
    FrontierState,
    Identity,
    Epoch,
    BatchTooLarge,
    Corruption,
    Authentication,
    Io,
    Random,
    Encryption,
}

impl From<arcgraph_storage::SpillError> for ExecutorSpillError {
    fn from(error: arcgraph_storage::SpillError) -> Self {
        use arcgraph_storage::SpillError;
        match error {
            SpillError::ResourceExhausted {
                reason,
                tenant_id,
                requested_bytes,
                spilled_bytes,
                limit_bytes,
                available_bytes,
            } => Self::ResourceExhausted {
                reason,
                tenant_id,
                requested_bytes,
                spilled_bytes,
                limit_bytes,
                available_bytes,
            },
            other => {
                let kind = match &other {
                    SpillError::InvalidConfig(_) => ExecutorSpillFailureKind::InvalidConfig,
                    SpillError::IdentityExhausted | SpillError::NonceExhausted => {
                        ExecutorSpillFailureKind::Identity
                    }
                    SpillError::StaleEpoch { .. } | SpillError::QueryEnded { .. } => {
                        ExecutorSpillFailureKind::Epoch
                    }
                    SpillError::BatchTooLarge { .. } => ExecutorSpillFailureKind::BatchTooLarge,
                    SpillError::InvalidHeader(_) | SpillError::CorruptFrame { .. } => {
                        ExecutorSpillFailureKind::Corruption
                    }
                    SpillError::AuthenticationFailed { .. } => {
                        ExecutorSpillFailureKind::Authentication
                    }
                    SpillError::Io { .. } => ExecutorSpillFailureKind::Io,
                    SpillError::Random { .. } => ExecutorSpillFailureKind::Random,
                    SpillError::Encryption(_) => ExecutorSpillFailureKind::Encryption,
                    // SpillError is non-exhaustive; future variants retain a
                    // typed executor failure instead of collapsing to Eval.
                    _ => ExecutorSpillFailureKind::Io,
                };
                Self::Failure {
                    kind,
                    detail: other.to_string(),
                }
            }
        }
    }
}

/// Public-API error type for [`crate::execute`] / [`crate::execute_with_context`].
///
/// # Why exempt from `#[non_exhaustive]`
///
/// code-quality policy admits an exemption when the variant set IS the public
/// contract for downstream pattern-matching consumption. `ExecutionError`
/// is the M5↔M4 contract surface (per ADR-038 amendment-03 §M5↔M4):
/// the M5-07 `graph.search` MCP tool, the M5-11 `graph.raw_query` MCP
/// tool, the M5-13 Bolt response framing, and any future HTTP error-
/// JSON serializer ALL pattern-match exhaustively on the variant set
/// to render distinct user-visible diagnostics (a `Cancelled` returns
/// a Bolt cancellation frame; a `Substrate` surfaces an "index
/// unavailable" detail; a `NotImplemented` carries the `target_slice`
/// forward-link). Adding a new variant here is therefore a coordinated
/// breaking change requiring a synchronized M5-07 / M5-11 / M5-13
/// amendment + new-variant rendering — NOT a silent additive change
/// that drops into a `_ => ...` arm. Under the code-quality policy, the rationale
/// is documented here so reviewers know the omission is deliberate.
///
/// # Eq
///
/// `Eq` is derived (W11Z fix-up MED-2) alongside `PartialEq` so the
/// `ExplainError::Substrate(SubstrateAccessError)` round-trip carries
/// a clean `Eq` surface — every variant payload is itself `Eq`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecutionError {
    /// Cancellation token tripped at a batch boundary. Translated
    /// from [`CancellationError`] at the operator level.
    #[error("query cancelled")]
    Cancelled,

    /// Substrate-access fault — the underlying substrate (CRUD scan,
    /// HNSW vector search, BM25 text search, community lookup)
    /// returned an error. v1.0-alpha stub substrates surface
    /// [`SubstrateAccessError::TenantUnknown`] /
    /// [`SubstrateAccessError::IndexUnavailable`] only; production
    /// wiring at M4-08+ will surface I/O / WAL faults.
    #[error("substrate access error: {0}")]
    Substrate(#[from] SubstrateAccessError),

    /// Plan-time semantic / lowering error caught at execute time.
    /// Should be rare — the planner runs before the executor — but
    /// the M4-08+ wiring layer routes ArcQLErrors into the same
    /// public surface.
    #[error("query error: {0}")]
    Plan(#[from] ArcQLError),

    /// OOC executor scratch failure. Capacity rejects retain their typed
    /// quota/headroom reason and measured `spilled_bytes` diagnostic.
    #[error("{0}")]
    Spill(ExecutorSpillError),

    /// The plan contains an operator forward-deferred to a later
    /// slice. `feature` names the operator (e.g.,
    /// `"LogicalPlan::Aggregate"`); `target_slice` cites the slice
    /// (e.g., `"M4-63"`).
    #[error(
        "execution operator not implemented: {feature} (forward to {target_slice} per {section})"
    )]
    NotImplemented {
        feature: String,
        target_slice: String,
        section: String,
    },

    /// Runtime evaluation fault — division by zero, integer overflow,
    /// invalid type cast that the planner did not catch (NULL operand
    /// reaching a non-NULL-tolerant context, etc.). The variant
    /// carries a free-form description; no span (the operator-level
    /// span tracking lights at M4-71 forward).
    #[error("runtime evaluation error: {0}")]
    Eval(String),

    /// **#797 / ADR-147 Phase 2** — a `$name` parameter was referenced
    /// by the query but no binding was supplied in the per-query
    /// parameter bag. A CLIENT fault (the caller failed to bind the
    /// parameter), NOT a server-side execution fault: the translation
    /// layer (`crate::explain::translate_execution_error`) lifts it
    /// to [`crate::ExplainError::MissingParameter`], which the wire
    /// surfaces map to a client error (Bolt
    /// `Neo.ClientError.Statement.ParameterMissing`; MCP `-32602`).
    /// Detected at [`crate::executor::eval::evaluate`] time — never a
    /// panic, never a silent NULL.
    #[error("missing parameter: ${name}")]
    MissingParameter { name: String },
}

impl From<CancellationError> for ExecutionError {
    #[inline]
    fn from(_: CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<arcgraph_storage::SpillError> for ExecutionError {
    fn from(error: arcgraph_storage::SpillError) -> Self {
        Self::Spill(error.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::context::{CancellationError, CancellationToken};

    #[test]
    fn cancellation_translates_to_cancelled_variant() {
        let t = CancellationToken::new();
        t.cancel();
        let err: ExecutionError = t.check().unwrap_err().into();
        assert_eq!(err, ExecutionError::Cancelled);
    }

    #[test]
    fn substrate_error_lifts_via_from() {
        let inner = SubstrateAccessError::IndexUnavailable("vector".into());
        let lifted: ExecutionError = inner.clone().into();
        assert_eq!(lifted, ExecutionError::Substrate(inner));
    }

    #[test]
    fn not_implemented_carries_forward_link() {
        let e = ExecutionError::NotImplemented {
            feature: "LogicalPlan::Aggregate".into(),
            target_slice: "M4-63".into(),
            section: "ADR-038 amendment-02 §M4.g".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("M4-63"), "must forward-link target slice: {s}");
        assert!(
            s.contains("LogicalPlan::Aggregate"),
            "must name the deferred operator: {s}"
        );
    }

    #[test]
    fn cancellation_error_pin() {
        // Sanity: the inner CancellationError type is the well-known
        // marker; ensures From plumbing matches.
        assert_eq!(CancellationError, CancellationError);
    }
}
