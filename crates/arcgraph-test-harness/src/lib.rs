//! Reusable database workload and failure-path test harness.
//!
//! ## Public surface
//!
//! - [`Dataset`] — names a workload's input dataset.
//! - [`Workload`] — names a query pattern over a [`Dataset`].
//! - [`OracleAdapter`] — verdict generator for a [`WorkloadResult`].
//! - [`RegressionGate`] — Criterion-style perf threshold.
//! - [`workloads`] — retained database workload modules.

#![forbid(unsafe_code)]
#![recursion_limit = "256"]

pub mod dataset;
pub mod oracle;
pub mod workload;
pub mod workloads;

pub use dataset::{Dataset, DatasetHandle, DatasetScale};
pub use oracle::{OracleAdapter, OracleClass, OracleVerdict};
pub use workload::{RegressionGate, Workload, WorkloadResult};

/// Errors surfaced by the harness scaffold.
///
/// Pre-v1.0-alpha most loaders + the Bolt oracle return
/// [`HarnessError::NotImplementedAtV1`] with a `feature` tag that
/// names the milestone gating the closure (`M4-61` for executor
/// wiring, `M5-08` for `graph.ingest()`, `M5-13` for Bolt).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// The requested capability is not yet shipped at v1.0-alpha.
    /// `feature` names the gating milestone (e.g. `"M4-61"`,
    /// `"M5-08"`, `"M5-13"`) so log readers can route the stub-hit
    /// to the right roadmap entry.
    #[error("not implemented at v1.0-alpha (gated on {feature}): {reason}")]
    NotImplementedAtV1 {
        feature: &'static str,
        reason: String,
    },
    /// A loader fixture failed to materialise.
    #[error("dataset fixture build failed: {reason}")]
    FixtureFailed { reason: String },
    /// An oracle disagreed with the observed result.
    #[error("oracle disagreement: {reason}")]
    OracleDisagreement { reason: String },
}

/// Result alias used across the harness crate.
pub type HarnessResult<T> = Result<T, HarnessError>;
